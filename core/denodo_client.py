"""
Denodo REST API Client
----------------------
Low-level HTTP client for Denodo Virtual DataPort REST web services.
Handles authentication, pagination, and column normalization.

Two endpoints:
  - iv_allparts_info_for_ce  (allparts view)
  - iv_plm_zagile_manufacture (manufacture view)
"""
import logging
import httpx
import pandas as pd
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Denodo REST endpoints
# ---------------------------------------------------------------------------
_ALLPARTS_URL = f"{settings.denodo_base_url_allparts}/views/iv_allparts_info_for_ce"
_MANUFACTURE_URL = f"{settings.denodo_base_url_manufacture}/views/iv_plm_zagile_manufacture"
_PAGE_SIZE = settings.denodo_page_size
_TIMEOUT = settings.denodo_timeout


def _auth() -> tuple[str, str]:
    return (settings.denodo_username, settings.denodo_password)


def _build_params(
    filter_expr: str | None = None,
    select: list[str] | None = None,
    group_by: list[str] | None = None,
    order_by: str | None = None,
    start_index: int = 0,
    count: int | None = None,
) -> dict:
    """Build Denodo REST query parameters."""
    params: dict = {"$format": "JSON"}
    if filter_expr:
        params["$filter"] = filter_expr
    if select:
        params["$select"] = ",".join(select)
    if group_by:
        params["$groupby"] = ",".join(group_by)
    if order_by:
        params["$orderby"] = order_by
    if start_index > 0:
        params["$start_index"] = start_index
    if count is not None:
        params["$count"] = count
    # Suppress RESTful self-links per row (reduces payload)
    params["$displayRESTfulReferences"] = "false"
    return params


def _paginated_fetch(
    url: str,
    filter_expr: str | None = None,
    select: list[str] | None = None,
    group_by: list[str] | None = None,
    order_by: str | None = None,
) -> pd.DataFrame:
    """
    Fetch all rows from a Denodo view with automatic pagination.
    Returns a DataFrame of all accumulated elements.
    """
    all_rows: list[dict] = []
    offset = 0

    with httpx.Client(auth=_auth(), timeout=_TIMEOUT, verify=False) as client:
        while True:
            params = _build_params(
                filter_expr=filter_expr,
                select=select,
                group_by=group_by,
                order_by=order_by,
                start_index=offset,
                count=_PAGE_SIZE,
            )
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])

            # Strip RESTful 'links' key from each row if present
            for row in elements:
                row.pop("links", None)

            all_rows.extend(elements)
            logger.debug(
                "Denodo fetch: url=%s offset=%d got=%d total_so_far=%d",
                url, offset, len(elements), len(all_rows),
            )

            if len(elements) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def _single_fetch(
    url: str,
    filter_expr: str | None = None,
    select: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch rows without pagination (for small result sets like item lookups).
    """
    with httpx.Client(auth=_auth(), timeout=_TIMEOUT, verify=False) as client:
        params = _build_params(filter_expr=filter_expr, select=select)
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])
        for row in elements:
            row.pop("links", None)
    return pd.DataFrame(elements) if elements else pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Quick connectivity check — fetch 1 row with 3s timeout."""
    try:
        with httpx.Client(auth=_auth(), timeout=3, verify=False) as client:
            params = {"$format": "JSON", "$count": 1, "$select": "Item_Number"}
            resp = client.get(_ALLPARTS_URL, params=params)
            return resp.status_code == 200
    except Exception:
        return False


def fetch_allparts(
    filter_expr: str | None = None,
    select: list[str] | None = None,
    group_by: list[str] | None = None,
    order_by: str | None = None,
) -> pd.DataFrame:
    """Query iv_allparts_info_for_ce with auto-pagination."""
    return _paginated_fetch(
        _ALLPARTS_URL,
        filter_expr=filter_expr,
        select=select,
        group_by=group_by,
        order_by=order_by,
    )


def fetch_manufacture(
    filter_expr: str | None = None,
    select: list[str] | None = None,
) -> pd.DataFrame:
    """
    Query iv_plm_zagile_manufacture with auto-pagination.
    Normalizes ITEM_NUMBER → Item_Number for consistency with allparts.
    """
    df = _paginated_fetch(
        _MANUFACTURE_URL,
        filter_expr=filter_expr,
        select=select,
    )
    if not df.empty and "ITEM_NUMBER" in df.columns:
        df = df.rename(columns={"ITEM_NUMBER": "Item_Number"})
    return df


def fetch_manufacture_small(
    filter_expr: str | None = None,
    select: list[str] | None = None,
) -> pd.DataFrame:
    """
    Query iv_plm_zagile_manufacture without pagination (for per-batch lookups).
    Normalizes ITEM_NUMBER → Item_Number.
    """
    df = _single_fetch(
        _MANUFACTURE_URL,
        filter_expr=filter_expr,
        select=select,
    )
    if not df.empty and "ITEM_NUMBER" in df.columns:
        df = df.rename(columns={"ITEM_NUMBER": "Item_Number"})
    return df
