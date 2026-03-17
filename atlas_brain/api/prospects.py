"""
REST endpoints for Prospect audit: list enriched prospects and aggregate stats.
"""

import logging
import re
from typing import Literal
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_auth
from ..services.apollo_company_overrides import (
    bootstrap_company_overrides_from_settings,
    delete_company_override,
    fetch_company_override_map,
    upsert_company_override,
)
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.prospects")

router = APIRouter(prefix="/b2b/prospects", tags=["b2b-prospects"])


class ManualQueueResolveRequest(BaseModel):
    action: Literal["retry", "dismiss"]
    domain: Optional[str] = None


class CompanyOverrideUpsertRequest(BaseModel):
    company_name_raw: str
    search_names: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


def _normalize_domain(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    cleaned = re.sub(r"^https?://", "", domain.strip().lower())
    cleaned = cleaned.lstrip("www.").strip("/")
    return cleaned or None


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _row_to_dict(row) -> dict:
    d = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, UUID):
            d[key] = str(val)
        elif hasattr(val, "isoformat"):
            d[key] = val.isoformat()
        else:
            d[key] = val
    return d


@router.get("")
async def list_prospects(
    company: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    seniority: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: AuthUser = Depends(require_auth),
):
    """List enriched prospects with optional filters."""
    pool = _pool_or_503()

    conditions = []
    params: list = []
    idx = 1

    if company:
        conditions.append(f"LOWER(company_name) LIKE ${idx}")
        params.append(f"%{company.lower()}%")
        idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if seniority:
        conditions.append(f"seniority = ${idx}")
        params.append(seniority)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    params.extend([limit, offset])
    rows = await pool.fetch(
        f"""
        SELECT id, first_name, last_name, email, email_status, title,
               seniority, department, company_name, company_domain,
               linkedin_url, city, state, country, status, created_at, updated_at
        FROM prospects
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    # Get total count with same filters
    count_params = params[:-2]  # exclude limit/offset
    count_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS total FROM prospects {where}",
        *count_params,
    )

    return {
        "prospects": [_row_to_dict(r) for r in rows],
        "count": count_row["total"] if count_row else 0,
    }


@router.get("/stats")
async def prospect_stats(
    _user: AuthUser = Depends(require_auth),
):
    """Aggregate prospect counts."""
    pool = _pool_or_503()

    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE status = 'active') AS active,
            COUNT(*) FILTER (WHERE status = 'contacted') AS contacted,
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') AS this_month
        FROM prospects
        """
    )

    return {
        "total": row["total"],
        "active": row["active"],
        "contacted": row["contacted"],
        "this_month": row["this_month"],
    }


@router.get("/manual-queue")
async def list_manual_prospect_queue(
    company: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: AuthUser = Depends(require_auth),
):
    """List prospect-org entries routed to manual/domain-assisted enrichment."""
    pool = _pool_or_503()

    conditions = ["status = 'manual_review'"]
    params: list = []
    idx = 1

    if company:
        conditions.append(f"(LOWER(company_name_raw) LIKE ${idx} OR LOWER(company_name_norm) LIKE ${idx})")
        params.append(f"%{company.lower()}%")
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}"
    params.extend([limit, offset])

    rows = await pool.fetch(
        f"""
        SELECT id, company_name_raw, company_name_norm, domain, status,
               error_detail, enriched_at, created_at, updated_at
        FROM prospect_org_cache
        {where}
        ORDER BY updated_at DESC, created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    count_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS total FROM prospect_org_cache {where}",
        *params[:-2],
    )

    return {
        "queue": [_row_to_dict(r) for r in rows],
        "count": count_row["total"] if count_row else 0,
    }


@router.get("/company-overrides")
async def list_company_overrides(
    company: Optional[str] = Query(None),
    _user: AuthUser = Depends(require_auth),
):
    """List DB-backed Apollo company overrides."""
    pool = _pool_or_503()
    overrides = list((await fetch_company_override_map(pool)).values())
    if company:
        term = company.lower()
        overrides = [
            row for row in overrides
            if term in str(row.get("company_name_raw", "")).lower()
            or term in str(row.get("company_name_norm", "")).lower()
        ]
    return {"overrides": [_row_to_dict(r) for r in overrides], "count": len(overrides)}


@router.post("/manual-queue/{queue_id}/resolve")
async def resolve_manual_prospect_queue(
    queue_id: UUID,
    payload: ManualQueueResolveRequest,
    _user: AuthUser = Depends(require_auth),
):
    """Resolve a manual-review queue entry by retrying or dismissing it."""
    pool = _pool_or_503()
    row = await pool.fetchrow(
        """
        SELECT id, company_name_raw, company_name_norm, status, domain
        FROM prospect_org_cache
        WHERE id = $1
        """,
        queue_id,
    )
    if not row or row["status"] != "manual_review":
        raise HTTPException(status_code=404, detail="Manual-review queue entry not found")

    domain = _normalize_domain(payload.domain)
    if payload.action == "retry":
        updated = await pool.fetchrow(
            """
            UPDATE prospect_org_cache
            SET status = 'pending',
                domain = COALESCE($2, domain),
                updated_at = NOW()
            WHERE id = $1
            RETURNING id, company_name_raw, company_name_norm, domain, status,
                      error_detail, enriched_at, created_at, updated_at
            """,
            queue_id,
            domain,
        )
    else:
        updated = await pool.fetchrow(
            """
            UPDATE prospect_org_cache
            SET status = 'not_found',
                domain = COALESCE($2, domain),
                enriched_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
            RETURNING id, company_name_raw, company_name_norm, domain, status,
                      error_detail, enriched_at, created_at, updated_at
            """,
            queue_id,
            domain,
        )
    return {"entry": _row_to_dict(updated)}


@router.post("/company-overrides")
async def create_or_update_company_override(
    payload: CompanyOverrideUpsertRequest,
    _user: AuthUser = Depends(require_auth),
):
    """Create or update an Apollo company override."""
    pool = _pool_or_503()
    try:
        row = await upsert_company_override(pool, payload.company_name_raw, payload.search_names, payload.domains)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"override": _row_to_dict(row)}


@router.post("/company-overrides/bootstrap")
async def bootstrap_company_overrides(
    _user: AuthUser = Depends(require_auth),
):
    """Import Apollo company overrides from settings into the DB table."""
    pool = _pool_or_503()
    return await bootstrap_company_overrides_from_settings(pool)


@router.delete("/company-overrides/{override_id}")
async def remove_company_override(
    override_id: UUID,
    _user: AuthUser = Depends(require_auth),
):
    """Delete an Apollo company override."""
    pool = _pool_or_503()
    deleted = await delete_company_override(pool, override_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Company override not found")
    return {"deleted": True}
