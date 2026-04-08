"""
REST endpoints for Prospect audit: list enriched prospects and aggregate stats.
"""

import json
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
from ..services.campaign_reasoning_context import campaign_review_reasoning_context
from ..services.company_normalization import normalize_company_name, normalized_company_name_sql
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


def _clean_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_context_blob(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _normalized_company_key(*values: object) -> str | None:
    for value in values:
        normalized = normalize_company_name(str(value or ""))
        if normalized:
            return normalized
    return None


def _prospect_sequence_summary(row: dict) -> dict:
    company_context = _coerce_context_blob(row.get("company_context"))
    summary = campaign_review_reasoning_context(company_context)
    return {
        "churning_from": _clean_text(company_context.get("churning_from")),
        "target_persona": _clean_text(company_context.get("target_persona")),
        "related_sequence_id": str(row["id"]) if row.get("id") else None,
        "related_sequence_status": _clean_text(row.get("status")),
        "related_sequence_current_step": row.get("current_step"),
        "related_sequence_max_steps": row.get("max_steps"),
        "related_sequence_last_sent_at": (
            row["last_sent_at"].isoformat() if getattr(row.get("last_sent_at"), "isoformat", None) else None
        ),
        **summary,
    }


async def _load_prospect_sequence_summaries(
    pool,
    prospects: list[dict],
) -> dict[str, dict]:
    emails = sorted({
        str(item.get("email") or "").strip().lower()
        for item in prospects
        if str(item.get("email") or "").strip()
    })
    company_names = sorted({
        company_name
        for item in prospects
        if (company_name := _normalized_company_key(
            item.get("company_name_norm"),
            item.get("company_name"),
        ))
    })
    if not emails and not company_names:
        return {}

    normalized_company_name_expr = normalized_company_name_sql("company_name")
    normalized_context_company_expr = normalized_company_name_sql("company_context ->> 'company'")
    rows = await pool.fetch(
        f"""
        SELECT id, company_name, recipient_email, status, current_step, max_steps,
               last_sent_at, updated_at, created_at, company_context
        FROM campaign_sequences
        WHERE (
            recipient_email IS NOT NULL
            AND LOWER(recipient_email) = ANY($1::text[])
        ) OR (
            {normalized_company_name_expr} = ANY($2::text[])
            OR {normalized_context_company_expr} = ANY($2::text[])
        )
        ORDER BY updated_at DESC, created_at DESC
        """,
        emails,
        company_names,
    )

    by_email: dict[str, dict] = {}
    by_company: dict[str, dict] = {}
    for row in rows:
        record = _row_to_dict(row)
        email_key = str(record.get("recipient_email") or "").strip().lower()
        if email_key and email_key not in by_email:
            by_email[email_key] = record
        company_key = _normalized_company_key(
            _coerce_context_blob(record.get("company_context")).get("company"),
            record.get("company_name"),
        )
        if company_key and company_key not in by_company:
            by_company[company_key] = record

    summaries: dict[str, dict] = {}
    for prospect in prospects:
        prospect_id = str(prospect.get("id") or "")
        email_key = str(prospect.get("email") or "").strip().lower()
        company_key = _normalized_company_key(
            prospect.get("company_name_norm"),
            prospect.get("company_name"),
        )
        match = by_email.get(email_key) if email_key else None
        if match is None and company_key:
            match = by_company.get(company_key)
        if match is not None and prospect_id:
            summaries[prospect_id] = _prospect_sequence_summary(match)
    return summaries


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
               linkedin_url, city, state, country, company_name_norm,
               status, created_at, updated_at
        FROM prospects
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    prospect_rows = [_row_to_dict(r) for r in rows]
    sequence_summaries = await _load_prospect_sequence_summaries(pool, prospect_rows)
    for prospect in prospect_rows:
        summary = sequence_summaries.get(str(prospect.get("id") or ""))
        if summary:
            prospect.update(summary)
        prospect.pop("company_name_norm", None)

    # Get total count with same filters
    count_params = params[:-2]  # exclude limit/offset
    count_row = await pool.fetchrow(
        f"SELECT COUNT(*) AS total FROM prospects {where}",
        *count_params,
    )

    return {
        "prospects": prospect_rows,
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
