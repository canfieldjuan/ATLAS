"""
REST API for the universal web scraper.

Endpoints for submitting scrape jobs (ad-hoc or from config files),
monitoring progress, retrieving results, and managing jobs.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import settings
from ..services.scraping.universal.orchestrator import (
    get_universal_scraper,
    load_config_file,
)
from ..services.scraping.universal.schemas import ScrapeJobConfig
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.universal_scrape")

router = APIRouter(prefix="/scraper", tags=["universal-scraper"])


# ── Schemas ──────────────────────────────────────────────────────────


class FromFileRequest(BaseModel):
    path: str = Field(description="Filename inside the configured scrape_configs directory")


# ── Endpoints ────────────────────────────────────────────────────────


@router.post("/jobs")
async def create_job(config: ScrapeJobConfig):
    """Submit a scrape job. Returns immediately with job_id; scraping runs in background.

    URL validation (SSRF protection) is enforced on each target URL by the
    ScrapeTarget Pydantic model.
    """
    if not settings.universal_scrape.enabled:
        raise HTTPException(503, "Universal scraper is disabled")
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    try:
        scraper = get_universal_scraper()
        job_id = await scraper.run_job(config)
        return {"job_id": str(job_id), "status": "pending", "targets": len(config.targets)}
    except Exception as exc:
        logger.exception("Failed to create scrape job")
        raise HTTPException(500, str(exc))


@router.post("/jobs/from-file")
async def create_job_from_file(req: FromFileRequest):
    """Load a scrape job from a JSON config file and start it.

    The file path is resolved relative to the configured ``config_dir``
    (default: ``scrape_configs/``). Path traversal outside that directory
    is blocked.
    """
    if not settings.universal_scrape.enabled:
        raise HTTPException(503, "Universal scraper is disabled")
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    try:
        config = load_config_file(req.path)
    except FileNotFoundError:
        raise HTTPException(404, f"Config file not found: {req.path}")
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Invalid config file: {exc}")

    try:
        scraper = get_universal_scraper()
        job_id = await scraper.run_job(config)
        return {"job_id": str(job_id), "status": "pending", "targets": len(config.targets)}
    except Exception as exc:
        logger.exception("Failed to create scrape job from file")
        raise HTTPException(500, str(exc))


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List scrape jobs, optionally filtered by status."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    conditions: list[str] = []
    args: list = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        args.append(status)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, name, status, total_targets, completed_targets,
               failed_targets, total_records, error, started_at, finished_at, created_at
        FROM universal_scrape_jobs
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *args,
        limit,
        offset,
    )
    return [dict(r) for r in rows]


@router.get("/jobs/{job_id}")
async def get_job(job_id: UUID):
    """Get job status and progress."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    row = await pool.fetchrow(
        """
        SELECT id, name, status, config, total_targets, completed_targets,
               failed_targets, total_records, error, started_at, finished_at, created_at
        FROM universal_scrape_jobs WHERE id = $1
        """,
        job_id,
    )
    if not row:
        raise HTTPException(404, "Job not found")
    return dict(row)


@router.get("/jobs/{job_id}/results")
async def get_results(
    job_id: UUID,
    include_raw: bool = Query(
        default=False,
        description="Include raw LLM response (may contain sensitive page content)",
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Get extracted data for a job."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    cols = (
        "id, job_id, target_url, page_number, page_title, "
        "extracted_data, item_count, duration_ms, error, created_at"
    )
    if include_raw:
        cols += ", raw_llm_response"

    rows = await pool.fetch(
        f"""
        SELECT {cols}
        FROM universal_scrape_results
        WHERE job_id = $1
        ORDER BY target_url, page_number
        LIMIT $2 OFFSET $3
        """,
        job_id,
        limit,
        offset,
    )
    return [dict(r) for r in rows]


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: UUID):
    """Cancel a pending or running job.

    Signals the running background task to stop processing new targets
    and pages. Already-in-flight requests will complete but no new ones
    will be started.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    scraper = get_universal_scraper()
    cancelled = await scraper.cancel_job(job_id)
    return {"cancelled": cancelled}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: UUID):
    """Delete a job and all its results (CASCADE)."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    result = await pool.execute(
        "DELETE FROM universal_scrape_jobs WHERE id = $1", job_id
    )
    return {"deleted": "DELETE 1" in result}
