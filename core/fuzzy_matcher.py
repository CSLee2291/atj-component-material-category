"""
MPN Fuzzy Matching Engine
-------------------------
Strategy (in priority order):
  1. Filter reference pool to same MANUFACTURE_NAME = "ATJ"
  2. Score each candidate MPN against target MPN using:
       - token_sort_ratio   (handles word reordering, e.g. "XC6SLX75 2FGG" vs "2FGG XC6SLX75")
       - partial_ratio      (prefix/substring match, good for long MPNs with suffix variants)
       - common_prefix_len  (bonus for shared alphanumeric prefix)
  3. Weighted composite score, return top-K
"""
import pandas as pd
from rapidfuzz import fuzz
from config import settings


def _common_prefix_score(a: str, b: str) -> float:
    """Score 0-100 based on shared leading characters (case-insensitive, stripped)."""
    a, b = a.upper().strip(), b.upper().strip()
    max_len = max(len(a), len(b), 1)
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix_len += 1
        else:
            break
    return (prefix_len / max_len) * 100


def score_mpn(target_mpn: str, candidate_mpn: str) -> float:
    """
    Composite similarity score 0–100.
    Weights:
      40% token_sort_ratio  — reorder-robust
      35% partial_ratio     — substring / prefix variant
      25% common_prefix     — exact-prefix bonus
    """
    if not target_mpn or not candidate_mpn:
        return 0.0
    t = target_mpn.upper().strip()
    c = candidate_mpn.upper().strip()
    tsr = fuzz.token_sort_ratio(t, c)
    pr  = fuzz.partial_ratio(t, c)
    cpx = _common_prefix_score(t, c)
    return 0.40 * tsr + 0.35 * pr + 0.25 * cpx


def find_top_k_similar(
    target_item: dict,
    reference_pool: pd.DataFrame,
    k: int = None,
    min_score: float = None,
) -> pd.DataFrame:
    """
    Given one target item dict (with keys Item_Number, MFR_PART_NUMBER, MANUFACTURE_NAME),
    search the reference_pool DataFrame and return the top-K most similar rows.

    reference_pool must have columns:
      Item_Number, MANUFACTURE_NAME, MFR_PART_NUMBER,
      Item_Desc, MATERIAL_CATEGORY, ZZMCATG_M, ZZMCATG_S, CATE_M_NAME, CATE_S_NAME

    Returns DataFrame sorted by similarity_score descending.
    """
    k = k or settings.top_k_similar
    min_score = min_score or settings.min_similarity_score * 100  # scale to 0-100

    target_mpn  = str(target_item.get("MFR_PART_NUMBER", "") or "")
    target_item_no = str(target_item.get("Item_Number", ""))
    target_mfr  = str(target_item.get("MANUFACTURE_NAME", "") or "")

    # Step 1: exclude the target itself from pool
    # (reference pool is already pre-filtered to MANUFACTURE_NAME="ATJ")
    pool = reference_pool[
        reference_pool["Item_Number"] != target_item_no
    ].copy()

    if pool.empty:
        return pd.DataFrame()

    # Step 2: score
    pool["similarity_score"] = pool["MFR_PART_NUMBER"].apply(
        lambda mpn: score_mpn(target_mpn, str(mpn or ""))
    )

    # Step 3: filter + sort + top-K
    pool = pool[pool["similarity_score"] >= min_score]
    pool = pool.sort_values("similarity_score", ascending=False).head(k)
    pool = pool.reset_index(drop=True)

    return pool
