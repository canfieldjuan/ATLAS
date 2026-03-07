"""
REST endpoints for Prospect audit: list enriched prospects and aggregate stats.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth.dependencies import AuthUser, require_auth
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.prospects")

router = APIRouter(prefix="/b2b/prospects", tags=["b2b-prospects"])


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
