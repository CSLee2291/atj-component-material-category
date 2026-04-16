"""
MPN Fuzzy Matching Engine
-------------------------
Strategy (in priority order):
  1. Pre-condition: target item MUST have MANUFACTURE_NAME from iv_plm_zagile_manufacture.
     If no manufacture data found → skip fuzzy search entirely (return empty).
  2. Filter reference pool to same MANUFACTURE_NAME (exact match).
     If no same-manufacturer references → fall back to full pool (MANUFACTURE_NAME="ATJ").
  3. Score each candidate MPN against target MPN using:
       - token_sort_ratio   (handles word reordering, e.g. "XC6SLX75 2FGG" vs "2FGG XC6SLX75")
       - partial_ratio      (prefix/substring match, good for long MPNs with suffix variants)
       - common_prefix_len  (bonus for shared alphanumeric prefix)
  4. Weighted composite score, return top-K
"""
import logging
import pandas as pd
from rapidfuzz import fuzz
from config import settings

logger = logging.getLogger(__name__)


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

    Pre-conditions:
      - Target item MUST have a non-empty MANUFACTURE_NAME (from iv_plm_zagile_manufacture).
        If MANUFACTURE_NAME is missing, fuzzy search is skipped (returns empty DataFrame).
      - Target item MUST have a non-empty MFR_PART_NUMBER for meaningful fuzzy matching.

    Search strategy:
      1. First try: filter reference pool to same MANUFACTURE_NAME (exact match).
      2. If no same-manufacturer candidates found: fall back to full pool.
      3. Score MPN similarity and return top-K.

    reference_pool must have columns:
      Item_Number, MANUFACTURE_NAME, MFR_PART_NUMBER,
      Item_Desc, MATERIAL_CATEGORY, ZZMCATG_M, ZZMCATG_S, CATE_M_NAME, CATE_S_NAME

    Returns DataFrame sorted by similarity_score descending.
    """
    k = k or settings.top_k_similar
    min_score = min_score or settings.min_similarity_score * 100  # scale to 0-100

    target_mpn     = str(target_item.get("MFR_PART_NUMBER", "") or "").strip()
    target_item_no = str(target_item.get("Item_Number", ""))
    target_mfr     = str(target_item.get("MANUFACTURE_NAME", "") or "").strip()

    # --- Pre-condition: MANUFACTURE_NAME must exist ---
    # If no manufacture data was found from iv_plm_zagile_manufacture,
    # we cannot do meaningful fuzzy matching → skip entirely.
    if not target_mfr:
        logger.debug(
            "Skipping fuzzy match for %s: no MANUFACTURE_NAME from iv_plm_zagile_manufacture",
            target_item_no,
        )
        return pd.DataFrame()

    # --- Pre-condition: MFR_PART_NUMBER must exist ---
    # Even with a manufacturer, no MPN means nothing to compare against.
    if not target_mpn:
        logger.debug(
            "Skipping fuzzy match for %s: MANUFACTURE_NAME=%s but no MFR_PART_NUMBER",
            target_item_no, target_mfr,
        )
        return pd.DataFrame()

    # Step 1: exclude the target itself from pool
    pool = reference_pool[
        reference_pool["Item_Number"] != target_item_no
    ].copy()

    if pool.empty:
        return pd.DataFrame()

    # Step 2: try same-manufacturer filter first
    if "MANUFACTURE_NAME" in pool.columns:
        same_mfr_pool = pool[
            pool["MANUFACTURE_NAME"].fillna("").str.strip().str.upper()
            == target_mfr.upper()
        ]
        if not same_mfr_pool.empty:
            logger.debug(
                "Fuzzy match %s: using %d same-manufacturer (%s) references",
                target_item_no, len(same_mfr_pool), target_mfr,
            )
            pool = same_mfr_pool
        else:
            # No same-manufacturer references → use full pool
            logger.debug(
                "Fuzzy match %s: no same-manufacturer (%s) references, using full pool (%d items)",
                target_item_no, target_mfr, len(pool),
            )

    # Step 3: score MPN similarity
    pool["similarity_score"] = pool["MFR_PART_NUMBER"].apply(
        lambda mpn: score_mpn(target_mpn, str(mpn or ""))
    )

    # Step 4: filter + sort + top-K
    pool = pool[pool["similarity_score"] >= min_score]
    pool = pool.sort_values("similarity_score", ascending=False).head(k)
    pool = pool.reset_index(drop=True)

    return pool
