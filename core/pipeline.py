"""
Batch Pipeline Orchestrator
----------------------------
Runs one full batch:
  1. Fetch target items from PBI (or local cache)
  2. Fetch their MPN data from iv_plm_zagile_manufacture
  3. Load ATJ reference pool (cached per session + on disk)
  4. For each target: fuzzy-match → enrich → GPT suggest
  5. Collect results for export
"""
import asyncio
import pandas as pd
from datetime import datetime, timezone
from core.data_fetcher import (
    fetch_atj_target_batch,
    fetch_manufacture_for_items,
    fetch_items_info,
    fetch_atj_reference_pool,
)
from core.fuzzy_matcher import find_top_k_similar
from core.gpt_caller import suggest_category, suggest_category_from_candidates
from core.category_vector_db import search_categories, is_vector_db_ready
from config import settings

# Module-level in-memory cache for reference pool
_reference_pool_cache: pd.DataFrame | None = None


def get_reference_pool(force_refresh: bool = False) -> pd.DataFrame:
    global _reference_pool_cache
    if _reference_pool_cache is None or force_refresh:
        _reference_pool_cache = fetch_atj_reference_pool(force_refresh=force_refresh)
    return _reference_pool_cache


async def process_single_item(
    target_row: dict,
    mfr_df: pd.DataFrame,
    reference_pool: pd.DataFrame,
    vector_top_k: int | None = None,
) -> dict:
    """
    Process one ATJ item:
      1. Fuzzy match → 1st GPT call
      2. If low/error confidence OR zero fuzzy matches → vector DB search → 2nd GPT call
    """
    item_no = target_row["Item_Number"]

    # Get manufacturer info (take first row if 1:many)
    if mfr_df.empty or "Item_Number" not in mfr_df.columns:
        mfr_rows = pd.DataFrame()
    else:
        mfr_rows = mfr_df[mfr_df["Item_Number"] == item_no]
    if mfr_rows.empty:
        mfr_name = ""
        mpn = ""
    else:
        mfr_name = mfr_rows.iloc[0]["MANUFACTURE_NAME"]
        mpn = mfr_rows.iloc[0]["MFR_PART_NUMBER"]

    target = {**target_row, "MANUFACTURE_NAME": mfr_name, "MFR_PART_NUMBER": mpn}

    # --- Pass 1: Fuzzy match + GPT ---
    similar_df = find_top_k_similar(target, reference_pool)
    references = similar_df.to_dict(orient="records") if not similar_df.empty else []
    no_fuzzy_matches = len(references) == 0

    suggestion = await suggest_category(target, references)
    first_confidence = suggestion.get("confidence", "error")
    first_reason = suggestion.get("reason", "")

    # --- Pass 2: Vector DB fallback ---
    # Trigger when: low/error confidence OR zero fuzzy matches
    needs_fallback = first_confidence in ("low", "error") or no_fuzzy_matches
    vector_candidates = []

    if needs_fallback and is_vector_db_ready():
        # Build search query from item description + first GPT reason
        item_desc = str(target_row.get("Item_Desc", "") or "")
        search_query = f"{item_desc} {first_reason}".strip()

        if search_query:
            vector_candidates = search_categories(search_query, top_k=vector_top_k or settings.vector_db_top_k)

        if vector_candidates:
            # 2nd GPT call with vector candidates
            fallback_suggestion = await suggest_category_from_candidates(
                target, first_reason, vector_candidates
            )
            # Always prefer vector fallback when first pass was low/error,
            # because vector categories are more semantically relevant than
            # weak fuzzy-match references. Only skip if fallback itself errored.
            fallback_conf = fallback_suggestion.get("confidence", "error")
            if fallback_conf != "error":
                suggestion = fallback_suggestion

    # Build result row
    top1 = references[0] if references else {}
    source = suggestion.get("source", "fuzzy_match")
    return {
        "Item_Number":          item_no,
        "Item_Desc":            target_row.get("Item_Desc", ""),
        "LifeCycle_Phase":      target_row.get("LifeCycle_Phase", ""),
        "MANUFACTURE_NAME":     mfr_name,
        "MFR_PART_NUMBER":      mpn,
        # Top-1 similar reference (for CE review)
        "Ref_Item_Number":      top1.get("Item_Number", ""),
        "Ref_MPN":              top1.get("MFR_PART_NUMBER", ""),
        "Ref_Similarity":       round(top1.get("similarity_score", 0), 1),
        "Ref_MATERIAL_CATEGORY":top1.get("MATERIAL_CATEGORY", ""),
        "Ref_CATE_M_NAME":      top1.get("CATE_M_NAME", ""),
        "Ref_CATE_S_NAME":      top1.get("CATE_S_NAME", ""),
        # GPT suggestion (may be from pass 1 or pass 2 fallback)
        "AI_ZZMCATG_M":         suggestion.get("ZZMCATG_M", ""),
        "AI_ZZMCATG_S":         suggestion.get("ZZMCATG_S", ""),
        "AI_MATERIAL_CATEGORY": suggestion.get("MATERIAL_CATEGORY", ""),
        "AI_confidence":        suggestion.get("confidence", ""),
        "AI_reason":            suggestion.get("reason", ""),
        "AI_source":            source,
        # Vector fallback info
        "Vector_used":          "Y" if source == "vector_fallback" else "N",
        "Vector_top1_category": vector_candidates[0]["MATERIAL_CATEGORY"] if vector_candidates else "",
        "Vector_top1_score":    round(vector_candidates[0]["similarity"], 3) if vector_candidates else 0,
        # CE review columns (blank — to be filled by CE)
        "CE_MATERIAL_CATEGORY": "",
        "CE_approved":          "",
        "CE_comment":           "",
        "processed_at":         datetime.now(timezone.utc).isoformat(),
    }


async def run_batch(
    offset: int = 0,
    limit: int = None,
    lifecycle_filter: list[str] | None = None,
    force_refresh_pool: bool = False,
    vector_top_k: int | None = None,
) -> pd.DataFrame:
    """
    Main entry point. Returns DataFrame of results for this batch.
    lifecycle_filter: if provided, only process items in these phases.
    """
    limit = limit or settings.batch_size

    # 1. Fetch targets (from local cache or PBI)
    # lifecycle_filter is applied inside fetch_atj_target_batch before offset/limit
    targets_df = fetch_atj_target_batch(
        offset=offset, limit=limit, lifecycle_filter=lifecycle_filter
    )

    if targets_df.empty:
        return pd.DataFrame()

    # 2. Fetch all MPN data for this batch in one DAX call
    item_numbers = targets_df["Item_Number"].tolist()
    mfr_df = fetch_manufacture_for_items(item_numbers)

    # 3. Load reference pool (in-memory + disk cache)
    reference_pool = get_reference_pool(force_refresh=force_refresh_pool)

    # 4. Process all items concurrently (bounded concurrency)
    sem = asyncio.Semaphore(10)  # max 10 concurrent GPT calls

    async def bounded(row):
        async with sem:
            return await process_single_item(row, mfr_df, reference_pool, vector_top_k=vector_top_k)

    tasks = [bounded(row) for row in targets_df.to_dict(orient="records")]
    results = await asyncio.gather(*tasks)

    return pd.DataFrame(results)


async def run_lookup(item_numbers: list[str], vector_top_k: int | None = None) -> pd.DataFrame:
    """
    Process a user-supplied list of item numbers through the same
    fuzzy-match + GPT categorization pipeline used for batch processing.
    Fetches item info and MPN data from PBI for the given items.
    """
    if not item_numbers:
        return pd.DataFrame()

    # 1. Fetch item details from PBI
    items_df = fetch_items_info(item_numbers)
    if items_df.empty:
        return pd.DataFrame()

    # 2. Fetch MPN data
    found_items = items_df["Item_Number"].tolist()
    mfr_df = fetch_manufacture_for_items(found_items)

    # 3. Load reference pool
    reference_pool = get_reference_pool()

    # 4. Process all items concurrently
    sem = asyncio.Semaphore(10)

    async def bounded(row):
        async with sem:
            return await process_single_item(row, mfr_df, reference_pool, vector_top_k=vector_top_k)

    tasks = [bounded(row) for row in items_df.to_dict(orient="records")]
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame(results)

    # Mark items that were requested but not found in PBI
    found_set = set(items_df["Item_Number"].tolist())
    not_found = [n for n in item_numbers if n not in found_set]
    if not_found:
        missing_rows = [{
            "Item_Number": n, "Item_Desc": "", "LifeCycle_Phase": "",
            "MANUFACTURE_NAME": "", "MFR_PART_NUMBER": "",
            "Ref_Item_Number": "", "Ref_MPN": "", "Ref_Similarity": 0,
            "Ref_MATERIAL_CATEGORY": "", "Ref_CATE_M_NAME": "", "Ref_CATE_S_NAME": "",
            "AI_ZZMCATG_M": "", "AI_ZZMCATG_S": "", "AI_MATERIAL_CATEGORY": "",
            "AI_confidence": "error", "AI_reason": "Item not found in PBI",
            "AI_source": "", "Vector_used": "N",
            "Vector_top1_category": "", "Vector_top1_score": 0,
            "CE_MATERIAL_CATEGORY": "", "CE_approved": "", "CE_comment": "",
            "processed_at": datetime.now(timezone.utc).isoformat(),
        } for n in not_found]
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    return df
