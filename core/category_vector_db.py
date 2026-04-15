"""
Category Vector Database
------------------------
Builds and searches a vector index of distinct MATERIAL_CATEGORY pairs
using Azure OpenAI text-embedding-3-small.

Structure:
  - Each entry = one unique (ZZMCATG_M, ZZMCATG_S) pair
  - Embedded text = "CATE_M_NAME: {m_name} | CATE_S_NAME: {s_name}"
  - Stored as numpy arrays + metadata pickle on disk
  - Cosine similarity search for top-K candidates

Fallback trigger:
  - First GPT call returns low/error confidence, OR
  - Fuzzy matcher returns zero results above threshold
"""
import os
import json
import pickle
import numpy as np
from openai import AzureOpenAI
from config import settings

_CACHE_DIR = settings.target_cache_dir
_VECTOR_DB_FILE = os.path.join(_CACHE_DIR, "category_vectors.npz")
_VECTOR_META_FILE = os.path.join(_CACHE_DIR, "category_vectors_meta.pkl")
_VECTOR_INFO_FILE = os.path.join(_CACHE_DIR, "category_vectors_info.txt")

# Module-level in-memory cache
_embeddings_cache: np.ndarray | None = None
_metadata_cache: list[dict] | None = None

# Lazy-init sync client (embeddings API is sync)
_embed_client: AzureOpenAI | None = None


def _get_embed_client() -> AzureOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_embedding_api_version,
        )
    return _embed_client


def _build_category_text(row: dict) -> str:
    """Build the text string to embed for a category pair."""
    m_name = str(row.get("CATE_M_NAME", "") or "").strip()
    s_name = str(row.get("CATE_S_NAME", "") or "").strip()
    m_code = str(row.get("ZZMCATG_M", "") or "").strip()
    s_code = str(row.get("ZZMCATG_S", "") or "").strip()
    return f"Category: {m_name} ({m_code}) > {s_name} ({s_code})"


def _get_embeddings(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Call Azure OpenAI embedding API in batches, return numpy array of shape (N, dim)."""
    client = _get_embed_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=settings.azure_openai_embedding_deployment,
            input=batch,
        )
        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)

    return np.array(all_embeddings, dtype=np.float32)


def build_vector_db(force_refresh: bool = False) -> dict:
    """
    Build the category vector DB from distinct categories in PBI.
    Returns stats dict.
    """
    from core.data_fetcher import fetch_distinct_categories

    # Check if already built and not forcing refresh
    if not force_refresh and os.path.exists(_VECTOR_DB_FILE) and os.path.exists(_VECTOR_META_FILE):
        meta = _load_metadata()
        return {
            "status": "already_built",
            "total_categories": len(meta),
        }

    # 1. Fetch distinct categories from PBI
    cat_df = fetch_distinct_categories(force_refresh=force_refresh)
    if cat_df.empty:
        return {"status": "error", "detail": "No categories found in PBI"}

    # 2. Build text for each category
    records = cat_df.to_dict(orient="records")
    texts = [_build_category_text(r) for r in records]

    # 3. Get embeddings from Azure OpenAI
    embeddings = _get_embeddings(texts)

    # 4. Normalize for cosine similarity (dot product after normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings_norm = embeddings / norms

    # 5. Save to disk
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez_compressed(_VECTOR_DB_FILE, embeddings=embeddings_norm)

    metadata = []
    for i, r in enumerate(records):
        metadata.append({
            "index": i,
            "ZZMCATG_M": r["ZZMCATG_M"],
            "ZZMCATG_S": r["ZZMCATG_S"],
            "CATE_M_NAME": r.get("CATE_M_NAME", ""),
            "CATE_S_NAME": r.get("CATE_S_NAME", ""),
            "MATERIAL_CATEGORY": r["MATERIAL_CATEGORY"],
            "embedded_text": texts[i],
        })
    with open(_VECTOR_META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    from datetime import datetime
    ts = datetime.now().isoformat()
    with open(_VECTOR_INFO_FILE, "w") as f:
        f.write(json.dumps({
            "built_at": ts,
            "total_categories": len(metadata),
            "embedding_model": settings.azure_openai_embedding_deployment,
            "embedding_dim": int(embeddings_norm.shape[1]),
        }))

    # Refresh in-memory cache
    global _embeddings_cache, _metadata_cache
    _embeddings_cache = embeddings_norm
    _metadata_cache = metadata

    return {
        "status": "ok",
        "total_categories": len(metadata),
        "embedding_dim": int(embeddings_norm.shape[1]),
        "built_at": ts,
    }


def _load_embeddings() -> np.ndarray:
    global _embeddings_cache
    if _embeddings_cache is None:
        data = np.load(_VECTOR_DB_FILE)
        _embeddings_cache = data["embeddings"]
    return _embeddings_cache


def _load_metadata() -> list[dict]:
    global _metadata_cache
    if _metadata_cache is None:
        with open(_VECTOR_META_FILE, "rb") as f:
            _metadata_cache = pickle.load(f)
    return _metadata_cache


def is_vector_db_ready() -> bool:
    """Check if the vector DB has been built."""
    return os.path.exists(_VECTOR_DB_FILE) and os.path.exists(_VECTOR_META_FILE)


def get_vector_db_info() -> dict | None:
    """Return metadata about the vector DB."""
    if not os.path.exists(_VECTOR_INFO_FILE):
        return None
    with open(_VECTOR_INFO_FILE) as f:
        return json.loads(f.read())


def search_categories(query_text: str, top_k: int = None) -> list[dict]:
    """
    Search the vector DB for the top-K most relevant category pairs
    given a free-text query (typically Item_Desc + AI_reason).

    Returns list of dicts with keys:
      ZZMCATG_M, ZZMCATG_S, CATE_M_NAME, CATE_S_NAME,
      MATERIAL_CATEGORY, similarity, embedded_text
    """
    top_k = top_k or settings.vector_db_top_k

    if not is_vector_db_ready():
        return []

    # Embed the query
    query_emb = _get_embeddings([query_text])[0]
    query_norm = query_emb / (np.linalg.norm(query_emb) or 1.0)

    # Cosine similarity via dot product (embeddings are pre-normalized)
    embeddings = _load_embeddings()
    similarities = embeddings @ query_norm

    # Top-K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    metadata = _load_metadata()
    results = []
    for idx in top_indices:
        entry = metadata[idx].copy()
        entry["similarity"] = float(similarities[idx])
        results.append(entry)

    return results
