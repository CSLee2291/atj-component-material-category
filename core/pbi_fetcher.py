"""
Fetches data from Power BI Desktop via ADOMD / DAX queries.
Uses pyadomd for Analysis Services connectivity.

Includes local Parquet caching for the full target list (~62K items)
to avoid re-querying PBI on every batch run.
"""
import os
import sys
import pandas as pd
from datetime import datetime
from config import settings

# Add ADOMD.NET DLL path before importing pyadomd
_ADOMD_PATHS = [
    r"C:\Program Files\DAX Studio\bin",
    r"C:\Program Files\Microsoft Power BI Desktop\bin",
    r"C:\Program Files (x86)\Microsoft Power BI Desktop\bin",
]
for _p in _ADOMD_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

from pyadomd import Pyadomd

# ---------------------------------------------------------------------------
# Target list cache paths
# ---------------------------------------------------------------------------
_CACHE_DIR = settings.target_cache_dir
_TARGET_CACHE_FILE = os.path.join(_CACHE_DIR, "atj_targets.parquet")
_TARGET_CACHE_META = os.path.join(_CACHE_DIR, "atj_targets_meta.txt")
_REFPOOL_CACHE_FILE = os.path.join(_CACHE_DIR, "atj_refpool.parquet")
_REFPOOL_CACHE_META = os.path.join(_CACHE_DIR, "atj_refpool_meta.txt")


def _run_dax(query: str) -> pd.DataFrame:
    conn_str = settings.pbi_connection_string
    with Pyadomd(conn_str) as conn:
        with conn.cursor().execute(query) as cur:
            cols = [c.name.split("[")[-1].rstrip("]") for c in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Target list: full fetch + local Parquet cache
# ---------------------------------------------------------------------------

def _fetch_all_atj_targets_from_pbi() -> pd.DataFrame:
    """Fetch ALL ATJ-Component items with blank MATERIAL_CATEGORY from PBI (no TOPN limit)."""
    query = """
    EVALUATE
    FILTER(
        SELECTCOLUMNS(
            'iv_plm_allparts_info_latest',
            "Item_Number",      'iv_plm_allparts_info_latest'[Item_Number],
            "Item_Desc",        'iv_plm_allparts_info_latest'[Item_Desc],
            "LifeCycle_Phase",  'iv_plm_allparts_info_latest'[LifeCycle_Phase],
            "MATERIAL_CATEGORY",'iv_plm_allparts_info_latest'[MATERIAL_CATEGORY]
        ),
        VAR beforeDash = FIND("-", [Item_Number], 1, 0)
        VAR prefix = IF(beforeDash > 0, LEFT([Item_Number], beforeDash - 1), [Item_Number])
        RETURN
            [MATERIAL_CATEGORY] = BLANK() &&
            CONTAINSSTRING(prefix, "TJ") &&
            NOT CONTAINSSTRING(prefix, "CMTJ")
    )
    """
    df = _run_dax(query)
    df = df.sort_values("Item_Number").reset_index(drop=True)
    return df


def refresh_target_cache() -> dict:
    """
    Re-query PBI for the full target list and save to local Parquet.
    Returns cache stats dict.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df = _fetch_all_atj_targets_from_pbi()
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
    Returns ATJ-Component items with null MATERIAL_CATEGORY and TJ prefix.
    Uses local Parquet cache if available; falls back to PBI DAX query.
    When lifecycle_filter is set, filters BEFORE applying offset/limit.
    """
    limit = limit or settings.batch_size

    if os.path.exists(_TARGET_CACHE_FILE):
        df = pd.read_parquet(_TARGET_CACHE_FILE)
        if lifecycle_filter:
            df = df[df["LifeCycle_Phase"].isin(lifecycle_filter)].reset_index(drop=True)
        return df.iloc[offset:offset + limit].reset_index(drop=True)

    # Fallback: query PBI directly with TOPN (original behavior)
    query = f"""
    EVALUATE
    TOPN(
        {offset + limit},
        FILTER(
            SELECTCOLUMNS(
                'iv_plm_allparts_info_latest',
                "Item_Number",      'iv_plm_allparts_info_latest'[Item_Number],
                "Item_Desc",        'iv_plm_allparts_info_latest'[Item_Desc],
                "LifeCycle_Phase",  'iv_plm_allparts_info_latest'[LifeCycle_Phase],
                "MATERIAL_CATEGORY",'iv_plm_allparts_info_latest'[MATERIAL_CATEGORY]
            ),
            VAR beforeDash = FIND("-", [Item_Number], 1, 0)
            VAR prefix = IF(beforeDash > 0, LEFT([Item_Number], beforeDash - 1), [Item_Number])
            RETURN
                [MATERIAL_CATEGORY] = BLANK() &&
                CONTAINSSTRING(prefix, "TJ") &&
                NOT CONTAINSSTRING(prefix, "CMTJ")
        ),
        [Item_Number]
    )
    """
    df = _run_dax(query)
    return df.iloc[offset:offset + limit].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distinct category pairs (for vector DB)
# ---------------------------------------------------------------------------

_CATEGORY_CACHE_FILE = os.path.join(_CACHE_DIR, "distinct_categories.parquet")
_CATEGORY_CACHE_META = os.path.join(_CACHE_DIR, "distinct_categories_meta.txt")


def fetch_distinct_categories(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch ALL distinct ZZMCATG_M + ZZMCATG_S pairs from iv_plm_allparts_info_latest
    (broader than ATJ-only). Returns DataFrame with columns:
      ZZMCATG_M, ZZMCATG_S, CATE_M_NAME, CATE_S_NAME, MATERIAL_CATEGORY
    Uses local Parquet cache; force_refresh re-queries PBI.
    """
    if not force_refresh and os.path.exists(_CATEGORY_CACHE_FILE):
        return pd.read_parquet(_CATEGORY_CACHE_FILE)

    try:
        query = """
        EVALUATE
        SUMMARIZECOLUMNS(
            'iv_plm_allparts_info_latest'[ZZMCATG_M],
            'iv_plm_allparts_info_latest'[ZZMCATG_S],
            'iv_plm_allparts_info_latest'[CATE_M_NAME],
            'iv_plm_allparts_info_latest'[CATE_S_NAME],
            FILTER(
                VALUES('iv_plm_allparts_info_latest'[MATERIAL_CATEGORY]),
                NOT ISBLANK('iv_plm_allparts_info_latest'[MATERIAL_CATEGORY])
            )
        )
        """
        df = _run_dax(query)
    except Exception as pbi_err:
        # Fallback: extract from cached reference pool if PBI is offline
        if os.path.exists(_REFPOOL_CACHE_FILE):
            print(f"[WARN] PBI offline ({pbi_err}), falling back to refpool cache for categories")
            refpool = pd.read_parquet(_REFPOOL_CACHE_FILE)
            df = refpool[["ZZMCATG_M", "ZZMCATG_S", "CATE_M_NAME", "CATE_S_NAME"]].copy()
        else:
            raise
    # Clean column names (SUMMARIZECOLUMNS may return full paths)
    df.columns = [c.split("[")[-1].rstrip("]") for c in df.columns]
    # Build MATERIAL_CATEGORY as ZZMCATG_M|ZZMCATG_S
    df["MATERIAL_CATEGORY"] = df["ZZMCATG_M"] + "|" + df["ZZMCATG_S"]
    # Drop rows with blank codes
    df = df.dropna(subset=["ZZMCATG_M", "ZZMCATG_S"])
    df = df[(df["ZZMCATG_M"].str.strip() != "") & (df["ZZMCATG_S"].str.strip() != "")]
    # Deduplicate by ZZMCATG_M + ZZMCATG_S
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
# MPN lookup (per-batch, not cached — small per-batch query)
# ---------------------------------------------------------------------------

def fetch_manufacture_for_items(item_numbers: list[str]) -> pd.DataFrame:
    """
    Returns MANUFACTURE_NAME + MFR_PART_NUMBER for the given target items.
    One item may have multiple rows (1:many).
    """
    items_literal = ", ".join(f'"{i}"' for i in item_numbers)
    query = f"""
    EVALUATE
    FILTER(
        SELECTCOLUMNS(
            'iv_plm_zagile_manufacture',
            "Item_Number",      'iv_plm_zagile_manufacture'[Item_Number],
            "MANUFACTURE_NAME", 'iv_plm_zagile_manufacture'[MANUFACTURE_NAME],
            "MFR_PART_NUMBER",  'iv_plm_zagile_manufacture'[MFR_PART_NUMBER]
        ),
        [Item_Number] IN {{ {items_literal} }}
    )
    """
    return _run_dax(query)


def fetch_items_info(item_numbers: list[str]) -> pd.DataFrame:
    """
    Returns Item_Desc, LifeCycle_Phase, MATERIAL_CATEGORY for specific items
    from iv_plm_allparts_info_latest.
    """
    items_literal = ", ".join(f'"{i}"' for i in item_numbers)
    query = f"""
    EVALUATE
    FILTER(
        SELECTCOLUMNS(
            'iv_plm_allparts_info_latest',
            "Item_Number",      'iv_plm_allparts_info_latest'[Item_Number],
            "Item_Desc",        'iv_plm_allparts_info_latest'[Item_Desc],
            "LifeCycle_Phase",  'iv_plm_allparts_info_latest'[LifeCycle_Phase],
            "MATERIAL_CATEGORY",'iv_plm_allparts_info_latest'[MATERIAL_CATEGORY]
        ),
        [Item_Number] IN {{ {items_literal} }}
    )
    """
    return _run_dax(query)


# ---------------------------------------------------------------------------
# Reference pool: cached both in-memory and on disk (Parquet)
# ---------------------------------------------------------------------------

def fetch_atj_reference_pool(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns ALL ATJ items with MATERIAL_CATEGORY for fuzzy matching.
    Uses local Parquet cache; force_refresh re-queries PBI.
    """
    if not force_refresh and os.path.exists(_REFPOOL_CACHE_FILE):
        return pd.read_parquet(_REFPOOL_CACHE_FILE)

    # Fetch mfr and category tables separately, join in Python
    # (avoids DAX NATURALLEFTOUTERJOIN lineage conflicts)
    mfr_query = """
    EVALUATE
    FILTER(
        SELECTCOLUMNS(
            'iv_plm_zagile_manufacture',
            "Item_Number",      'iv_plm_zagile_manufacture'[Item_Number],
            "MANUFACTURE_NAME", 'iv_plm_zagile_manufacture'[MANUFACTURE_NAME],
            "MFR_PART_NUMBER",  'iv_plm_zagile_manufacture'[MFR_PART_NUMBER]
        ),
        [MANUFACTURE_NAME] = "ATJ" && NOT ISBLANK([MFR_PART_NUMBER])
    )
    """
    cat_query = """
    EVALUATE
    FILTER(
        SELECTCOLUMNS(
            'iv_plm_allparts_info_latest',
            "Item_Number",       'iv_plm_allparts_info_latest'[Item_Number],
            "Item_Desc",         'iv_plm_allparts_info_latest'[Item_Desc],
            "MATERIAL_CATEGORY", 'iv_plm_allparts_info_latest'[MATERIAL_CATEGORY],
            "ZZMCATG_M",         'iv_plm_allparts_info_latest'[ZZMCATG_M],
            "ZZMCATG_S",         'iv_plm_allparts_info_latest'[ZZMCATG_S],
            "CATE_M_NAME",       'iv_plm_allparts_info_latest'[CATE_M_NAME],
            "CATE_S_NAME",       'iv_plm_allparts_info_latest'[CATE_S_NAME]
        ),
        NOT ISBLANK([MATERIAL_CATEGORY])
    )
    """
    mfr_df = _run_dax(mfr_query)
    cat_df = _run_dax(cat_query)
    df = mfr_df.merge(cat_df, on="Item_Number", how="left")
    df = df.dropna(subset=["MATERIAL_CATEGORY"])

    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(_REFPOOL_CACHE_FILE, index=False)
    ts = datetime.now().isoformat()
    with open(_REFPOOL_CACHE_META, "w") as f:
        f.write(ts)
    return df
