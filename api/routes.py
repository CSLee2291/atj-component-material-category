"""
API Routes
----------
POST /api/batch/run          — trigger a new batch
GET  /api/batch/{id}/status  — poll batch status
GET  /api/batch/{id}/export  — download Excel
GET  /api/batches            — list all past batches
POST /api/cache/refresh      — refresh target list cache from PBI
GET  /api/cache/status       — check cache status
GET  /api/config             — current settings
GET  /                       — serve web UI (batch process)
GET  /lookup                 — manual lookup page
GET  /kpi                    — KPI dashboard
POST /api/kpi/snapshot       — take weekly KPI snapshot
GET  /api/kpi/snapshots      — all historical snapshots
GET  /api/kpi/latest         — latest snapshot
GET  /api/kpi/details        — per-item detail data
"""
import uuid
import os
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from core.pipeline import run_batch, run_lookup
from core.data_fetcher import refresh_target_cache, get_target_cache_info, check_denodo_health
from core.kpi_tracker import (
    take_snapshot, get_snapshots, get_latest_snapshot, get_detail_data,
    take_phase1_snapshot, get_phase1_snapshots, get_phase1_latest,
    get_phase1_detail, get_phase1_batch_status, write_phase1_results_to_excel,
)
from core.category_vector_db import build_vector_db, get_vector_db_info
from export.excel_exporter import export_to_excel
from config import settings

router = APIRouter()

# In-memory job store (replace with Redis/DB for production)
_jobs: dict[str, dict] = {}


class BatchRequest(BaseModel):
    offset: int = 0
    limit: Optional[int] = None
    lifecycle_filter: Optional[list[str]] = None
    force_refresh_pool: bool = False
    vector_top_k: Optional[int] = None  # 5 or 10, None = use config default


async def _run_batch_job(job_id: str, req: BatchRequest):
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        df = await run_batch(
            offset=req.offset,
            limit=req.limit,
            lifecycle_filter=req.lifecycle_filter,
            force_refresh_pool=req.force_refresh_pool,
            vector_top_k=req.vector_top_k,
        )
        if df.empty:
            _jobs[job_id].update({
                "status": "done", "total": 0, "high": 0, "medium": 0,
                "low": 0, "error": 0, "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            return
        filepath = export_to_excel(df, job_id, req.lifecycle_filter)
        # Store results as JSON-safe records for preview
        results_records = df.fillna("").to_dict(orient="records")
        _jobs[job_id].update({
            "status":        "done",
            "total":         len(df),
            "excel_path":    filepath,
            "excel_file":    os.path.basename(filepath),
            "finished_at":   datetime.now(timezone.utc).isoformat(),
            "high":          int((df["AI_confidence"] == "high").sum()),
            "medium":        int((df["AI_confidence"] == "medium").sum()),
            "low":           int((df["AI_confidence"] == "low").sum()),
            "error":         int((df["AI_confidence"] == "error").sum()),
            "results":       results_records,
        })
    except Exception as e:
        _jobs[job_id].update({"status": "error", "error": str(e)})


@router.post("/api/batch/run")
async def start_batch(req: BatchRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "id":           job_id,
        "status":       "queued",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "params":       req.model_dump(),
    }
    background_tasks.add_task(_run_batch_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/api/batch/{job_id}/status")
async def batch_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/api/batch/{job_id}/export")
async def download_export(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Export not ready")
    path = job["excel_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=job["excel_file"],
    )


@router.get("/api/batches")
async def list_batches():
    return list(reversed(list(_jobs.values())))


# ---------------------------------------------------------------------------
# Cache management endpoints
# ---------------------------------------------------------------------------

@router.post("/api/cache/refresh")
async def cache_refresh():
    """Re-query PBI for full target list + reference pool, save to local Parquet."""
    try:
        result = refresh_target_cache()
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/cache/status")
async def cache_status():
    """Return info about the local target cache."""
    info = get_target_cache_info()
    if info is None:
        return {"cached": False, "message": "No cache found. Use POST /api/cache/refresh to build it."}
    return {"cached": True, **info}


# ---------------------------------------------------------------------------
# Vector DB endpoints
# ---------------------------------------------------------------------------

@router.post("/api/vector-db/build")
async def vector_db_build(force: bool = False):
    """Build or refresh the category vector database from PBI distinct categories."""
    try:
        result = build_vector_db(force_refresh=force)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/vector-db/status")
async def vector_db_status():
    """Return info about the category vector database."""
    info = get_vector_db_info()
    if info is None:
        return {"built": False, "message": "Vector DB not built yet. Use POST /api/vector-db/build"}
    return {"built": True, **info}


# ---------------------------------------------------------------------------
# Denodo status
# ---------------------------------------------------------------------------

@router.get("/api/denodo/status")
async def denodo_status():
    """Return Denodo connectivity status."""
    return check_denodo_health()


# ---------------------------------------------------------------------------
# Lookup (manual item list) endpoints
# ---------------------------------------------------------------------------

class LookupRequest(BaseModel):
    item_numbers: list[str]
    vector_top_k: Optional[int] = None


async def _run_lookup_job(job_id: str, item_numbers: list[str], vector_top_k: int | None = None):
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        df = await run_lookup(item_numbers, vector_top_k=vector_top_k)
        if df.empty:
            _jobs[job_id].update({
                "status": "done", "total": 0, "high": 0, "medium": 0,
                "low": 0, "error": 0, "results": [],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            return
        # Store results as JSON-safe records for preview
        results_records = df.fillna("").to_dict(orient="records")
        _jobs[job_id].update({
            "status":        "done",
            "total":         len(df),
            "finished_at":   datetime.now(timezone.utc).isoformat(),
            "high":          int((df["AI_confidence"] == "high").sum()),
            "medium":        int((df["AI_confidence"] == "medium").sum()),
            "low":           int((df["AI_confidence"] == "low").sum()),
            "error":         int((df["AI_confidence"] == "error").sum()),
            "results":       results_records,
        })
    except Exception as e:
        _jobs[job_id].update({"status": "error", "error": str(e)})


@router.post("/api/lookup/run")
async def start_lookup(req: LookupRequest, background_tasks: BackgroundTasks):
    if not req.item_numbers:
        raise HTTPException(status_code=400, detail="No item numbers provided")
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "id":           job_id,
        "type":         "lookup",
        "status":       "queued",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "params":       {"item_count": len(req.item_numbers), "items": req.item_numbers[:20]},
    }
    background_tasks.add_task(_run_lookup_job, job_id, req.item_numbers, req.vector_top_k)
    return {"job_id": job_id, "status": "queued", "item_count": len(req.item_numbers)}


# ---------------------------------------------------------------------------
# Export selected items (for lookup preview)
# ---------------------------------------------------------------------------

class ExportSelectedRequest(BaseModel):
    job_id: str
    selected_items: list[str]  # Item_Number list


@router.post("/api/batch/{job_id}/export-selected")
async def export_selected(job_id: str, req: ExportSelectedRequest):
    """Export only selected items from a completed job to Excel."""
    import pandas as pd
    job = _jobs.get(job_id)
    if not job or job["status"] != "done" or not job.get("results"):
        raise HTTPException(status_code=404, detail="Job results not available")

    # Filter to selected items
    all_results = job["results"]
    selected = [r for r in all_results if r.get("Item_Number") in req.selected_items]
    if not selected:
        raise HTTPException(status_code=400, detail="No matching items selected")

    df = pd.DataFrame(selected)
    lifecycle_filter = ["lookup"] if job.get("type") == "lookup" else None
    filepath = export_to_excel(df, f"{job_id}_sel", lifecycle_filter=lifecycle_filter)
    filename = os.path.basename(filepath)

    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Config + UI
# ---------------------------------------------------------------------------

@router.get("/api/config")
async def get_config():
    return {
        "env":        settings.env,
        "batch_size": settings.batch_size,
        "top_k":      settings.top_k_similar,
    }


@router.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@router.get("/lookup", response_class=HTMLResponse)
async def serve_lookup_ui(request: Request):
    with open("templates/lookup.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@router.get("/kpi", response_class=HTMLResponse)
async def serve_kpi_ui(request: Request):
    with open("templates/kpi.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ---------------------------------------------------------------------------
# KPI Dashboard endpoints
# ---------------------------------------------------------------------------

@router.post("/api/kpi/snapshot")
async def kpi_take_snapshot():
    """Take a new KPI snapshot (one per ISO week, overwrites same week)."""
    try:
        snapshot = take_snapshot()
        return {"status": "ok", "snapshot": snapshot}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/kpi/snapshots")
async def kpi_list_snapshots():
    """Return all historical snapshots for trend charts."""
    return get_snapshots()


@router.get("/api/kpi/latest")
async def kpi_latest_snapshot():
    """Return the most recent snapshot."""
    snap = get_latest_snapshot()
    return {"snapshot": snap}


@router.get("/api/kpi/details")
async def kpi_details(filter: str = "all", lifecycle: str = None):
    """Return per-item detail data for the KPI table.
    filter: all | filled | blank
    lifecycle: optional lifecycle phase string
    """
    try:
        data = get_detail_data(filter_mode=filter, lifecycle=lifecycle)
        return {"total": len(data), "items": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Phase I KPI endpoints
# ---------------------------------------------------------------------------

@router.post("/api/kpi/phase1/snapshot")
async def phase1_take_snapshot():
    """Take a Phase I KPI snapshot."""
    try:
        snapshot = take_phase1_snapshot()
        return {"status": "ok", "snapshot": snapshot}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/kpi/phase1/snapshots")
async def phase1_list_snapshots():
    return get_phase1_snapshots()


@router.get("/api/kpi/phase1/latest")
async def phase1_latest_snapshot():
    snap = get_phase1_latest()
    return {"snapshot": snap}


@router.get("/api/kpi/phase1/details")
async def phase1_details(filter: str = "all", lifecycle: str = None):
    try:
        data = get_phase1_detail(filter_mode=filter, lifecycle=lifecycle)
        return {"total": len(data), "items": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/kpi/phase1/status")
async def phase1_batch_status():
    """Return Phase I item count and Excel path."""
    try:
        return get_phase1_batch_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Phase1BatchRequest(BaseModel):
    offset: int = 0
    limit: int = 100
    vector_top_k: Optional[int] = 10


async def _run_phase1_batch(job_id: str, offset: int, limit: int, vector_top_k: int):
    """Background task: run AI categorization on Phase I items, write results to Excel."""
    from core.kpi_tracker import _load_phase1_items
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        items = _load_phase1_items()
        batch_items = items[offset:offset + limit]
        if not batch_items:
            _jobs[job_id].update({
                "status": "done", "total": 0, "high": 0, "medium": 0,
                "low": 0, "error": 0, "written_to_excel": 0,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            return

        # Run through the lookup pipeline (same as manual lookup)
        df = await run_lookup(batch_items, vector_top_k=vector_top_k)

        if df.empty:
            _jobs[job_id].update({
                "status": "done", "total": 0, "high": 0, "medium": 0,
                "low": 0, "error": 0, "written_to_excel": 0,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            return

        results_records = df.fillna("").to_dict(orient="records")

        # Write high/medium results to Excel
        written = write_phase1_results_to_excel(results_records)

        _jobs[job_id].update({
            "status":          "done",
            "total":           len(df),
            "high":            int((df["AI_confidence"] == "high").sum()),
            "medium":          int((df["AI_confidence"] == "medium").sum()),
            "low":             int((df["AI_confidence"] == "low").sum()),
            "error":           int((df["AI_confidence"] == "error").sum()),
            "written_to_excel": written,
            "finished_at":     datetime.now(timezone.utc).isoformat(),
            "results":         results_records,
        })
    except Exception as e:
        _jobs[job_id].update({"status": "error", "error": str(e)})


@router.post("/api/phase1/batch/run")
async def start_phase1_batch(req: Phase1BatchRequest, background_tasks: BackgroundTasks):
    """Run AI categorization on a batch of Phase I items."""
    job_id = f"p1-{str(uuid.uuid4())[:8]}"
    _jobs[job_id] = {
        "id":           job_id,
        "type":         "phase1_batch",
        "status":       "queued",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "params":       {"offset": req.offset, "limit": req.limit, "vector_top_k": req.vector_top_k},
    }
    background_tasks.add_task(_run_phase1_batch, job_id, req.offset, req.limit, req.vector_top_k or 10)
    return {"job_id": job_id, "status": "queued", "offset": req.offset, "limit": req.limit}
