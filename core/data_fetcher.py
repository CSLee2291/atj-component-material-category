"""
Data Fetcher Facade
-------------------
Exposes the same public API as pbi_fetcher but tries Denodo REST API first,
falling back to Power BI Desktop (ADOMD/DAX) when Denodo is unreachable.

All function signatures match pbi_fetcher exactly so callers only change
their import line.
"""
import os
import logging
import pandas as pd
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache paths (shared with pbi_fetcher — same files)
# ---------------------------------------------------------------------------
_CACHE_DIR = settings.target_cache_dir
_TARGET_CACHE_FILE = os.path.join(_CACHE_DIR, "atj_targets.parquet")
_TARGET_CACHE_META = os.path.join(_CACHE_DIR, "atj_targets_meta.txt")
_REFPOOL_CACHE_FILE = os.path.join(_CACHE_DIR, "atj_refpool.parquet")
_REFPOOL_CACHE_META = os.path.join(_CACHE_DIR, "atj_refpool_meta.txt")
_CATEGORY_CACHE_FILE = os.path.join(_CACHE_DIR, "distinct_categories.parquet")
_CATEGORY_CACHE_META = os.path.join(_CACHE_DIR, "distinct_categories_meta.txt")


def _denodo_available() -> bool:
    """Check if Denodo is enabled and reachable."""
    if not settings.denodo_enabled:
        return False
    if not settings.denodo_username:
        logger.debug("Denodo credentials not configured, skipping")
        return False
    try:
        from core.denodo_client import is_available
        return is_available()
    except Exception as e:
        logger.debug("Denodo availability check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# 1. Fetch all ATJ targets (for cache refresh)
# ---------------------------------------------------------------------------

def _fetch_all_atj_targets_from_denodo() -> pd.DataFrame:
    """Fetch all ATJ-Component items with blank MATERIAL_CATEGORY from Denodo."""
    from core.denodo_client import fetch_allparts

    # VQL filter: blank MATERIAL_CATEGORY + TJ prefix (not CMTJ)
    filter_expr = (
        "(MATERIAL_CATEGORY is null OR MATERIAL_CATEGORY = '') "
        "AND Item_Number like '%TJ%' "
        "AND Item_Number not like '%CMTJ%'"
    )
    df = fetch_allparts(
        filter_expr=filter_expr,
        select=["Item_Number", "Item_Desc", "LifeCycle_Phase", "MATERIAL_CATEGORY"],
        order_by="Item_Number ASC",
    )

    if not df.empty:
        # Python-side safety filter: ensure TJ is in the prefix (before first '-')
        prefix = df["Item_Number"].str.split("-").str[0]
        mask = prefix.str.contains("TJ", na=False) & ~prefix.str.contains("CMTJ", na=False)
        df = df[mask].reset_index(drop=True)

    return df


def _fetch_all_atj_targets() -> pd.DataFrame:
    """Try Denodo, fall back to PBI."""
    if _denodo_available():
        try:
            logger.info("Fetching ATJ targets from Denodo...")
            df = _fetch_all_atj_targets_from_denodo()
            logger.info("Denodo returned %d ATJ targets", len(df))
            return df
        except Exception as e:
            logger.warning("Denodo failed for ATJ targets, falling back to PBI: %s", e)

    logger.info("Fetching ATJ targets from PBI...")
    from core.pbi_fetcher import _fetch_all_atj_targets_from_pbi
    return _fetch_all_atj_targets_from_pbi()


# ---------------------------------------------------------------------------
# 2. Target cache management
# ---------------------------------------------------------------------------

def refresh_target_cache() -> dict:
    """Re-query for the full target list and save to local Parquet."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df = _fetch_all_atj_targets()
    df.to_parquet(_TARGET_CACHE_FILE, index=False)
    ts = datetime.now().isoformat()
    with open(_TARGET_CACHE_META, "w") as f:
        f.write(ts)
    return {
        "total_targets": len(df),
        "cached_at": ts,
        "cache_file": _TARGET_CACHE_FILE,
        "phases": df["LifeCycle_Phase"].value_counts().to_dict() if not df.empty else {},
    }


def get_target_cache_info() -> dict | None:
    """Return cache metadata if cache exists, else None."""
    if not os.path.exists(_TARGET_CACHE_FILE):
        return None
    cached_at = ""
    if os.path.exists(_TARGET_CACHE_META):
        with open(_TARGET_CACHE_META) as f:
            cached_at = f.read().strip()
    df = pd.read_parquet(_TARGET_CACHE_FILE)
    return {
        "total_targets": len(df),
        "cached_at": cached_at,
        "cache_file": _TARGET_CACHE_FILE,
        "phases": df["LifeCycle_Phase"].value_counts().to_dict() if not df.empty else {},
    }


def fetch_atj_target_batch(
    offset: int = 0,
    limit: int = None,
    lifecycle_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns ATJ-Component items with null MATERIAL_CATEGORY.
    Uses local Parquet cache if available; falls back to live query.
    """
    limit = limit or settings.batch_size

    if os.path.exists(_TARGET_CACHE_FILE):
        df = pd.read_parquet(_TARGET_CACHE_FILE)
        if lifecycle_filter:
            df = df[df["LifeCycle_Phase"].isin(lifecycle_filter)].reset_index(drop=True)
        return df.iloc[offset:offset + limit].reset_index(drop=True)

    # No cache — fetch live (tries Denodo first, then PBI)
    df = _fetch_all_atj_targets()
    if lifecycle_filter and not df.empty:
        df = df[df["LifeCycle_Phase"].isin(lifecycle_filter)].reset_index(drop=True)
    return df.iloc[offset:offset + limit].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Distinct categories (for vector DB)
# ---------------------------------------------------------------------------

def fetch_distinct_categories(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch ALL distinct ZZMCATG_M + ZZMCATG_S pairs.
    Uses local Parquet cache; force_refresh re-queries.
    """
    if not force_refresh and os.path.exists(_CATEGORY_CACHE_FILE):
        return pd.read_parquet(_CATEGORY_CACHE_FILE)

    df = None

    # Try Denodo first
    if _denodo_available():
        try:
            from core.denodo_client import fetch_allparts
            logger.info("Fetching distinct categories from Denodo...")
            df = fetch_allparts(
                filter_expr="MATERIAL_CATEGORY is not null AND MATERIAL_CATEGORY <> ''",
                select=["ZZMCATG_M", "ZZMCATG_S", "CATE_M_NAME", "CATE_S_NAME", "MATERIAL_CATEGORY"],
                group_by=["ZZMCATG_M", "ZZMCATG_S", "CATE_M_NAME", "CATE_S_NAME", "MATERIAL_CATEGORY"],
            )
            logger.info("Denodo returned %d category rows", len(df) if df is not None else 0)
        except Exception as e:
            logger.warning("Denodo failed for categories, falling back to PBI: %s", e)
            df = None

    # Fallback to PBI
    if df is None or df.empty:
        try:
            from core.pbi_fetcher import fetch_distinct_categories as pbi_fetch_categories
            return pbi_fetch_categories(force_refresh=force_refresh)
        except Exception as pbi_err:
            # Last resort: refpool cache
            if os.path.exists(_REFPOOL_CACHE_FILE):
                logger.warning("Both Denodo and PBI failed (%s), using refpool cache", pbi_err)
                refpool = pd.read_parquet(_REFPOOL_CACHE_FILE)
                df = refpool[["ZZMCATG_M", "ZZMCATG_S", "CATE_M_NAME", "CATE_S_NAME"]].copy()
            else:
                raise

    # Build MATERIAL_CATEGORY and deduplicate
    if "MATERIAL_CATEGORY" not in df.columns:
        df["MATERIAL_CATEGORY"] = df["ZZMCATG_M"] + "|" + df["ZZMCATG_S"]
    df = df.dropna(subset=["ZZMCATG_M", "ZZMCATG_S"])
    df = df[(df["ZZMCATG_M"].str.strip() != "") & (df["ZZMCATG_S"].str.strip() != "")]
    df = df.drop_duplicates(subset=["ZZMCATG_M", "ZZMCATG_S"]).reset_index(drop=True)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(_CATEGORY_CACHE_FILE, index=False)
    ts = datetime.now().isoformat()
    with open(_CATEGORY_CACHE_META, "w") as f:
        f.write(ts)
    return df


def get_category_cache_info() -> dict | None:
    """Return metadata about the distinct categories cache."""
    if not os.path.exists(_CATEGORY_CACHE_FILE):
        return None
    cached_at = ""
    if os.path.exists(_CATEGORY_CACHE_META):
        with open(_CATEGORY_CACHE_META) as f:
            cached_at = f.read().strip()
    df = pd.read_parquet(_CATEGORY_CACHE_FILE)
    return {
        "total_categories": len(df),
        "cached_at": cached_at,
    }


# ---------------------------------------------------------------------------
# 4. Manufacture data (per-batch, not cached)
# ---------------------------------------------------------------------------

def fetch_manufacture_for_items(item_numbers: list[str]) -> pd.DataFrame:
    """
    Returns MANUFACTURE_NAME + MFR_PART_NUMBER for the given items.
    One item may have multiple rows (1:many).
    """
    if not item_numbers:
        return pd.DataFrame()

    # Try Denodo first
    if _denodo_available():
        try:
            from core.denodo_client import fetch_manufacture_small
            # Build VQL IN clause
            items_str = ",".join(f"'{n}'" for n in item_numbers)
            filter_expr = f"ITEM_NUMBER IN ({items_str})"
            df = fetch_manufacture_small(
                filter_expr=filter_expr,
                select=["ITEM_NUMBER", "MANUFACTURE_NAME", "MFR_PART_NUMBER"],
            )
            logger.debug("Denodo returned %d manufacture rows", len(df))
            return df
        except Exception as e:
            logger.warning("Denodo failed for manufacture data, falling back to PBI: %s", e)

    from core.pbi_fetcher import fetch_manufacture_for_items as pbi_fetch_mfr
    return pbi_fetch_mfr(item_numbers)


# ---------------------------------------------------------------------------
# 5. Item info (for manual lookup)
# ---------------------------------------------------------------------------

def fetch_items_info(item_numbers: list[str]) -> pd.DataFrame:
    """
    Returns Item_Desc, LifeCycle_Phase, MATERIAL_CATEGORY for specific items.
    """
    if not item_numbers:
        return pd.DataFrame()

    # Try Denodo first
    if _denodo_available():
        try:
            from core.denodo_client import fetch_allparts
            items_str = ",".join(f"'{n}'" for n in item_numbers)
            filter_expr = f"Item_Number IN ({items_str})"
            df = fetch_allparts(
                filter_expr=filter_expr,
                select=["Item_Number", "Item_Desc", "LifeCycle_Phase", "MATERIAL_CATEGORY"],
            )
            logger.debug("Denodo returned %d items info rows", len(df))
            return df
        except Exception as e:
            logger.warning("Denodo failed for items info, falling back to PBI: %s", e)

    from core.pbi_fetcher import fetch_items_info as pbi_fetch_items
    return pbi_fetch_items(item_numbers)


# ---------------------------------------------------------------------------
# 6. Reference pool (for fuzzy matching)
# ---------------------------------------------------------------------------

def fetch_atj_reference_pool(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns ALL ATJ items with MATERIAL_CATEGORY for fuzzy matching.
    Uses local Parquet cache; force_refresh re-queries.
    """
    if not force_refresh and os.path.exists(_REFPOOL_CACHE_FILE):
        return pd.read_parquet(_REFPOOL_CACHE_FILE)

    df = None

    # Try Denodo first
    if _denodo_available():
        try:
            from core.denodo_client import fetch_manufacture, fetch_allparts
            logger.info("Fetching ATJ reference pool from Denodo...")

            # Manufacture data: ATJ items with MPN
            mfr_df = fetch_manufacture(
                filter_expr="MANUFACTURE_NAME = 'ATJ' AND MFR_PART_NUMBER is not null AND MFR_PART_NUMBER <> ''",
                select=["ITEM_NUMBER", "MANUFACTURE_NAME", "MFR_PART_NUMBER"],
            )

            # Allparts: items with MATERIAL_CATEGORY
            cat_df = fetch_allparts(
                filter_expr="MATERIAL_CATEGORY is not null AND MATERIAL_CATEGORY <> ''",
                select=[
                    "Item_Number", "Item_Desc", "MATERIAL_CATEGORY",
                    "ZZMCATG_M", "ZZMCATG_S", "CATE_M_NAME", "CATE_S_NAME",
                ],
            )

            df = mfr_df.merge(cat_df, on="Item_Number", how="left")
            df = df.dropna(subset=["MATERIAL_CATEGORY"])
            logger.info("Denodo reference pool: %d items", len(df))
        except Exception as e:
            logger.warning("Denodo failed for reference pool, falling back to PBI: %s", e)
            df = None

    # Fallback to PBI
    if df is None or df.empty:
        from core.pbi_fetcher import fetch_atj_reference_pool as pbi_fetch_refpool
        return pbi_fetch_refpool(force_refresh=force_refresh)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(_REFPOOL_CACHE_FILE, index=False)
    ts = datetime.now().isoformat()
    with open(_REFPOOL_CACHE_META, "w") as f:
        f.write(ts)
    return df


# ---------------------------------------------------------------------------
# 7. Fetch ALL ATJ components (for KPI — both filled and blank)
# ---------------------------------------------------------------------------

def fetch_all_atj_components() -> pd.DataFrame:
    """
    Fetch ALL ATJ-Component items regardless of MATERIAL_CATEGORY status.
    Returns DataFrame with: Item_Number, Item_Desc, LifeCycle_Phase,
    MATERIAL_CATEGORY, ZZMCATG_M, ZZMCATG_S.
    Used by KPI tracker to compute completion stats.
    """
    if _denodo_available():
        try:
            from core.denodo_client import fetch_allparts
            logger.info("Fetching ALL ATJ components from Denodo for KPI...")
            filter_expr = (
                "Item_Number like '%TJ%' "
                "AND Item_Number not like '%CMTJ%'"
            )
            df = fetch_allparts(
                filter_expr=filter_expr,
                select=[
                    "Item_Number", "Item_Desc", "LifeCycle_Phase",
                    "MATERIAL_CATEGORY", "ZZMCATG_M", "ZZMCATG_S",
                ],
                order_by="Item_Number ASC",
            )
            if not df.empty:
                # Python-side safety filter
                prefix = df["Item_Number"].str.split("-").str[0]
                mask = prefix.str.contains("TJ", na=False) & ~prefix.str.contains("CMTJ", na=False)
                df = df[mask].reset_index(drop=True)
            logger.info("Denodo returned %d total ATJ components", len(df))
            return df
        except Exception as e:
            logger.warning("Denodo failed for all ATJ components, falling back to cache merge: %s", e)

    # Fallback: combine target cache (blank) + refpool cache (filled)
    logger.info("Merging target + refpool caches for KPI fallback...")
    frames = []
    if os.path.exists(_TARGET_CACHE_FILE):
        frames.append(pd.read_parquet(_TARGET_CACHE_FILE))
    if os.path.exists(_REFPOOL_CACHE_FILE):
        rp = pd.read_parquet(_REFPOOL_CACHE_FILE)
        cols = ["Item_Number", "Item_Desc", "LifeCycle_Phase", "MATERIAL_CATEGORY"]
        available = [c for c in cols if c in rp.columns]
        if available:
            frames.append(rp[available])
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["Item_Number"]).reset_index(drop=True)
        return df
    return pd.DataFrame()


def fetch_manufacture_for_items_batched(item_numbers: list[str], batch_size: int = 500) -> pd.DataFrame:
    """
    Fetch MANUFACTURE_NAME + MFR_PART_NUMBER for a large list of items.
    Splits into batches to avoid query-length limits.
    """
    if not item_numbers:
        return pd.DataFrame()

    frames = []
    for i in range(0, len(item_numbers), batch_size):
        batch = item_numbers[i:i + batch_size]
        df = fetch_manufacture_for_items(batch)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# 8. Denodo health check (for API endpoint)
# ---------------------------------------------------------------------------

def check_denodo_health() -> dict:
    """Return Denodo connectivity status for diagnostics."""
    if not settings.denodo_enabled:
        return {"enabled": False, "status": "disabled"}
    if not settings.denodo_username:
        return {"enabled": True, "status": "not_configured", "message": "DENODO_USERNAME not set in .env"}
    try:
        from core.denodo_client import is_available
        available = is_available()
        return {
            "enabled": True,
            "status": "connected" if available else "unreachable",
            "allparts_url": settings.denodo_base_url_allparts,
            "manufacture_url": settings.denodo_base_url_manufacture,
        }
    except Exception as e:
        return {"enabled": True, "status": "error", "error": str(e)}
