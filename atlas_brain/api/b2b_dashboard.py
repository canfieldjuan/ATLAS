"""
B2B churn intelligence API.

Action-ready intelligence feeds for the B2B displacement intelligence
network. Mirrors ``atlas_brain.mcp.b2b_churn_server`` queries over HTTP
for direct consumption by thin delivery surfaces.
"""

import asyncio
import csv
import io
import json
import logging
import re
import uuid as _uuid
from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from ..auth.dependencies import AuthUser, optional_auth, require_b2b_plan
from ..config import settings
from ..services.b2b.source_impact import (
    build_source_impact_ledger,
    get_consumer_wiring_baseline,
    summarize_source_field_baseline,
)
from ..services.b2b.report_trust import report_section_evidence_payload, report_trust_payload
from ..services.tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)
from ..services.scraping.capabilities import get_capability
from ..services.scraping.sources import ALL_SOURCES, ReviewSource, display_name as source_display_name
from ..autonomous.tasks._b2b_shared import (
    company_signal_review_unlock_snapshot,
    has_complete_core_run_marker,
    latest_complete_core_report_date,
    read_company_signal_candidates,
    read_company_signal_candidate_groups,
    read_company_signal_candidate_group_summary,
    read_company_signal_review_impact_summary,
    read_high_intent_companies,
    read_ranked_vendor_signal_rows,
    read_vendor_signal_detail,
    read_vendor_signal_rows,
    read_vendor_signal_summary,
)
from ..storage.database import get_db_pool
from ..storage.models import ScheduledTask

logger = logging.getLogger("atlas.api.b2b_dashboard")

EXPORT_ROW_LIMIT = 10_000

VALID_ENTITY_TYPES = {
    "review", "vendor", "displacement_edge", "pain_point",
    "churn_signal", "buyer_profile", "use_case", "integration",
    "source",
}
VALID_CORRECTION_TYPES = {"suppress", "flag", "override_field", "merge_vendor", "reclassify", "suppress_source"}
_KNOWN_SOURCES = {s.value for s in ReviewSource}
VALID_CORRECTION_STATUSES = {"applied", "reverted", "pending_review"}

router = APIRouter(prefix="/b2b/dashboard", tags=["b2b-dashboard"])


class VendorComparisonRequest(BaseModel):
    primary_vendor: str = Field(..., min_length=1, max_length=200)
    comparison_vendor: str = Field(..., min_length=1, max_length=200)
    window_days: int = Field(90, ge=1, le=3650)
    persist: bool = True


class AccountComparisonRequest(BaseModel):
    primary_company: str = Field(..., min_length=1, max_length=200)
    comparison_company: str = Field(..., min_length=1, max_length=200)
    window_days: int = Field(90, ge=1, le=3650)
    persist: bool = True


class AccountDeepDiveRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=200)
    window_days: int = Field(90, ge=1, le=3650)
    persist: bool = True


class CreateCorrectionBody(BaseModel):
    entity_type: str = Field(...)
    entity_id: str = Field(...)
    correction_type: str = Field(...)
    field_name: str | None = None
    old_value: str | None = None
    new_value: str | None = None
    reason: str = Field(..., min_length=1, max_length=2000)
    metadata: dict | None = None

    @field_validator("entity_type", "entity_id", "correction_type", "reason", mode="before")
    @classmethod
    def _trim_required_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value is required")
        return text

    @field_validator("field_name", "old_value", "new_value", mode="before")
    @classmethod
    def _trim_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class RevertCorrectionBody(BaseModel):
    reason: str | None = None

    @field_validator("reason", mode="before")
    @classmethod
    def _trim_reason(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class CompanySignalCandidateReviewBody(BaseModel):
    notes: str | None = Field(default=None, max_length=2000)
    trigger_rebuild: bool = True


class BulkCompanySignalCandidateGroupReviewBody(BaseModel):
    group_ids: list[str] = Field(..., min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    trigger_rebuild: bool = True


def _safe_json(val):
    """Return val if already a list/dict, else try json.loads, else as-is."""
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=None):
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_dict(val):
    if isinstance(val, dict):
        return val
    return {}


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _canonical_review_predicate(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"{prefix}duplicate_of_review_id IS NULL"


def _normalize_vendor_name(value: str | None) -> str:
    return str(value or "").strip().lower()


async def _load_reasoning_views_for_vendors(pool, vendor_names: list[str]) -> dict[str, Any]:
    requested = [
        str(vendor_name).strip()
        for vendor_name in vendor_names
        if str(vendor_name or "").strip()
    ]
    if not requested:
        return {}
    try:
        from ..autonomous.tasks._b2b_synthesis_reader import load_best_reasoning_views

        views = await load_best_reasoning_views(
            pool,
            requested,
            as_of=date.today(),
            analysis_window_days=settings.b2b_churn.intelligence_window_days,
        )
    except Exception:
        logger.debug("Dashboard reasoning view load failed", exc_info=True)
        return {}
    return {
        _normalize_vendor_name(vendor_name): view
        for vendor_name, view in views.items()
        if _normalize_vendor_name(vendor_name)
    }


def _overlay_reasoning_summary_from_view(target: dict[str, Any], view: Any) -> None:
    from ..autonomous.tasks._b2b_synthesis_reader import synthesis_view_to_reasoning_entry

    entry = synthesis_view_to_reasoning_entry(view)
    if entry.get("archetype"):
        target["archetype"] = entry["archetype"]
    if entry.get("confidence") is not None:
        target["archetype_confidence"] = float(entry["confidence"] or 0)
    if entry.get("mode"):
        target["reasoning_mode"] = entry["mode"]
    if entry.get("risk_level"):
        target["reasoning_risk_level"] = entry["risk_level"]
    primary_wedge = getattr(view, "primary_wedge", None)
    if primary_wedge:
        target["synthesis_wedge"] = primary_wedge.value
        target["synthesis_wedge_label"] = view.wedge_label
    reasoning_delta = getattr(view, "reasoning_delta", None)
    if isinstance(reasoning_delta, dict) and reasoning_delta:
        target["reasoning_delta"] = reasoning_delta
    target["reasoning_source"] = "b2b_reasoning_synthesis"


def _overlay_reasoning_detail_from_view(
    target: dict[str, Any],
    view: Any,
    *,
    requested_as_of: date | None = None,
) -> None:
    from ..autonomous.tasks._b2b_synthesis_reader import inject_synthesis_freshness
    from ..autonomous.tasks._b2b_synthesis_reader import synthesis_view_to_reasoning_entry

    _overlay_reasoning_summary_from_view(target, view)
    entry = synthesis_view_to_reasoning_entry(view)
    if entry.get("falsification_conditions"):
        target["falsification_conditions"] = entry["falsification_conditions"]
    if entry.get("executive_summary"):
        target["reasoning_executive_summary"] = entry["executive_summary"]
    if entry.get("key_signals"):
        target["reasoning_key_signals"] = entry["key_signals"]
    if entry.get("uncertainty_sources"):
        target["reasoning_uncertainty_sources"] = entry["uncertainty_sources"]

    context_getter = getattr(view, "filtered_consumer_context", None)
    if callable(context_getter):
        context = context_getter("vendor_scorecard")
    else:
        context = view.consumer_context("vendor_scorecard")
    contracts = context.get("reasoning_contracts")
    if isinstance(contracts, dict) and contracts:
        target["reasoning_contracts"] = contracts
    reference_ids = context.get("reference_ids")
    if isinstance(reference_ids, dict) and reference_ids:
        target["reasoning_reference_ids"] = reference_ids
    scope_manifest = context.get("scope_manifest")
    if isinstance(scope_manifest, dict) and scope_manifest:
        target["reasoning_scope_manifest"] = scope_manifest
    atoms = context.get("reasoning_atoms")
    if isinstance(atoms, dict) and atoms:
        target["reasoning_atoms"] = atoms
    delta = context.get("reasoning_delta")
    if isinstance(delta, dict) and delta:
        target["reasoning_delta"] = delta
    contract_gaps = context.get("reasoning_contract_gaps")
    if isinstance(contract_gaps, list) and contract_gaps:
        target["reasoning_contract_gaps"] = contract_gaps
    section_disclaimers = context.get("reasoning_section_disclaimers")
    if isinstance(section_disclaimers, dict) and section_disclaimers:
        target["reasoning_section_disclaimers"] = section_disclaimers
    if getattr(view, "meta", None):
        target["evidence_window"] = view.meta
    primary_wedge = getattr(view, "primary_wedge", None)
    if primary_wedge:
        target["synthesis_wedge"] = primary_wedge.value
        target["synthesis_wedge_label"] = view.wedge_label
    target["reasoning_source"] = "b2b_reasoning_synthesis"
    inject_synthesis_freshness(
        target,
        view,
        requested_as_of=requested_as_of,
    )


def _should_scope(user: AuthUser | None) -> bool:
    """Return True if the request should be tenant-scoped (non-admin authenticated user)."""
    return user is not None and not getattr(user, "is_admin", False)


async def _get_scoped_vendors(pool, user: AuthUser | None) -> list[str] | None:
    """Fetch tracked vendor names for a tenant-scoped user, or None if unscoped."""
    if not _should_scope(user):
        return None
    rows = await pool.fetch(
        "SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid",
        user.account_id,
    )
    return [r["vendor_name"] for r in rows]


def _candidate_reviewer(user: AuthUser | None) -> str:
    return f"api:{user.user_id}" if user else "analyst"


async def _fetch_company_signal_candidate_or_404(
    pool,
    candidate_id: str,
    user: AuthUser | None,
) -> dict[str, Any]:
    try:
        cid = _uuid.UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="candidate_id must be a valid UUID")

    row = await pool.fetchrow(
        """
        SELECT id,
               review_id,
               company_name,
               company_name_raw,
               vendor_name,
               product_category,
               source,
               urgency_score,
               signal_evidence_present,
               pain_category,
               buyer_role,
               decision_maker,
               seat_count,
               contract_end,
               buying_stage,
               confidence_score,
               canonical_gap_reason,
               candidate_bucket,
               review_status
        FROM b2b_company_signal_candidates
        WHERE id = $1
        """,
        cid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Company signal candidate not found")

    scoped_vendors = await _get_scoped_vendors(pool, user)
    scoped_vendor_keys = {
        _normalize_vendor_name(vendor_name)
        for vendor_name in (scoped_vendors or [])
        if _normalize_vendor_name(vendor_name)
    }
    if scoped_vendors is not None and _normalize_vendor_name(row.get("vendor_name")) not in scoped_vendor_keys:
        raise HTTPException(status_code=404, detail="Company signal candidate not found")

    return dict(row)


async def _fetch_company_signal_candidate_group_or_404(
    pool,
    group_id: str,
    user: AuthUser | None,
) -> dict[str, Any]:
    try:
        gid = _uuid.UUID(group_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="group_id must be a valid UUID")

    row = await pool.fetchrow(
        """
        SELECT id,
               company_name,
               display_company_name,
               vendor_name,
               product_category,
               review_count,
               distinct_source_count,
               decision_maker_count,
               signal_evidence_count,
               canonical_ready_review_count,
               max_urgency_score,
               corroborated_confidence_score,
               canonical_gap_reason,
               candidate_bucket,
               representative_review_id,
               representative_source,
               representative_pain_category,
               representative_buyer_role,
               representative_decision_maker,
               representative_seat_count,
               representative_contract_end,
               representative_buying_stage,
               review_status
        FROM b2b_company_signal_candidate_groups
        WHERE id = $1
        """,
        gid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Company signal candidate group not found")

    scoped_vendors = await _get_scoped_vendors(pool, user)
    scoped_vendor_keys = {
        _normalize_vendor_name(vendor_name)
        for vendor_name in (scoped_vendors or [])
        if _normalize_vendor_name(vendor_name)
    }
    if scoped_vendors is not None and _normalize_vendor_name(row.get("vendor_name")) not in scoped_vendor_keys:
        raise HTTPException(status_code=404, detail="Company signal candidate group not found")

    return dict(row)


async def _set_company_signal_candidate_group_status(
    pool,
    *,
    company_name: str,
    vendor_name: str,
    status: str,
    reviewed_by: str,
    review_notes: str | None,
    group_id: str | None = None,
) -> dict[str, Any] | None:
    if group_id is not None:
        return await pool.fetchrow(
            """
            UPDATE b2b_company_signal_candidate_groups
            SET review_status = $2,
                review_status_updated_at = NOW(),
                reviewed_by = $3,
                review_notes = $4
            WHERE id = $1::uuid
            RETURNING id, review_status, review_status_updated_at, reviewed_by, review_notes
            """,
            group_id,
            status,
            reviewed_by,
            review_notes,
        )
    await pool.execute(
        """
        UPDATE b2b_company_signal_candidate_groups
        SET review_status = $3,
            review_status_updated_at = NOW(),
            reviewed_by = $4,
            review_notes = $5
        WHERE company_name = $1
          AND vendor_name = $2
        """,
        company_name,
        vendor_name,
        status,
        reviewed_by,
        review_notes,
    )
    return None


async def _set_company_signal_candidate_member_status(
    pool,
    *,
    company_name: str,
    vendor_name: str,
    status: str,
    reviewed_by: str,
    review_notes: str | None,
) -> None:
    await pool.execute(
        """
        UPDATE b2b_company_signal_candidates
        SET review_status = $3,
            review_status_updated_at = NOW(),
            reviewed_by = $4,
            review_notes = $5
        WHERE company_name = $1
          AND vendor_name = $2
        """,
        company_name,
        vendor_name,
        status,
        reviewed_by,
        review_notes,
    )


async def _upsert_company_signal(
    pool,
    *,
    company_name: str,
    vendor_name: str,
    urgency_score: Any,
    pain_category: Any,
    buyer_role: Any,
    decision_maker: Any,
    seat_count: Any,
    contract_end: Any,
    buying_stage: Any,
    review_id: Any,
    source: Any,
    confidence_score: Any,
) -> dict[str, Any] | None:
    existing_id = await pool.fetchval(
        """
        SELECT id
        FROM b2b_company_signals
        WHERE company_name = $1
          AND vendor_name = $2
        """,
        company_name,
        vendor_name,
    )
    row = await pool.fetchrow(
        """
        INSERT INTO b2b_company_signals (
            company_name, vendor_name, urgency_score,
            pain_category, buyer_role, decision_maker,
            seat_count, contract_end, buying_stage,
            review_id, source, confidence_score, last_seen_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
        ON CONFLICT (company_name, vendor_name)
        DO UPDATE SET
            urgency_score = GREATEST(b2b_company_signals.urgency_score, EXCLUDED.urgency_score),
            pain_category = COALESCE(EXCLUDED.pain_category, b2b_company_signals.pain_category),
            buyer_role = COALESCE(EXCLUDED.buyer_role, b2b_company_signals.buyer_role),
            decision_maker = COALESCE(EXCLUDED.decision_maker, b2b_company_signals.decision_maker),
            seat_count = COALESCE(EXCLUDED.seat_count, b2b_company_signals.seat_count),
            contract_end = COALESCE(EXCLUDED.contract_end, b2b_company_signals.contract_end),
            buying_stage = COALESCE(EXCLUDED.buying_stage, b2b_company_signals.buying_stage),
            review_id = COALESCE(EXCLUDED.review_id, b2b_company_signals.review_id),
            source = COALESCE(EXCLUDED.source, b2b_company_signals.source),
            confidence_score = GREATEST(b2b_company_signals.confidence_score, EXCLUDED.confidence_score),
            last_seen_at = EXCLUDED.last_seen_at
        RETURNING id, company_name, vendor_name, review_id
        """,
        company_name,
        vendor_name,
        urgency_score,
        pain_category,
        buyer_role,
        decision_maker,
        seat_count,
        contract_end,
        buying_stage,
        review_id,
        source,
        confidence_score,
    )
    if not row:
        return None
    payload = dict(row)
    payload["company_signal_action"] = "updated" if existing_id else "created"
    return payload


async def _delete_company_signal(
    pool,
    *,
    company_name: str,
    vendor_name: str,
) -> dict[str, Any] | None:
    row = await pool.fetchrow(
        """
        DELETE FROM b2b_company_signals
        WHERE company_name = $1
          AND vendor_name = $2
        RETURNING id, company_name, vendor_name, review_id
        """,
        company_name,
        vendor_name,
    )
    if row:
        payload = dict(row)
        payload["company_signal_action"] = "deleted"
        return payload
    return {
        "id": None,
        "company_name": company_name,
        "vendor_name": vendor_name,
        "company_signal_action": "none",
    }


def _unique_strings_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _normalize_company_signal_review_rebuild(
    rebuild: dict[str, Any] | None,
    *,
    rebuild_requested: bool,
) -> dict[str, Any]:
    payload = rebuild or {}
    result = _safe_dict(payload.get("result"))
    return {
        "rebuild_requested": bool(rebuild_requested),
        "rebuild_triggered": bool(payload.get("triggered")),
        "rebuild_reason": payload.get("reason"),
        "rebuild_as_of": payload.get("as_of"),
        "rebuild_persisted_count": _safe_int(payload.get("persisted"), _safe_int(result.get("persisted"), 0)),
        "rebuild_total_accounts": _safe_int(payload.get("total_accounts"), _safe_int(result.get("total_accounts"), 0)),
        "rebuild_vendor_count": _safe_int(payload.get("vendors"), _safe_int(result.get("vendors"), 0)),
    }


def _company_signal_candidate_review_priority_values(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    if row.get("candidate_bucket") == "canonical_ready":
        return "promote_now", "canonical_ready"
    if bool(row.get("signal_evidence_present")) and bool(row.get("decision_maker")):
        return "high", "has_signal_evidence_and_decision_maker"
    if bool(row.get("signal_evidence_present")):
        return "medium", "has_signal_evidence"
    if bool(row.get("decision_maker")):
        return "medium", "has_decision_maker"
    return "low", str(row.get("canonical_gap_reason") or "low_signal")


def _company_signal_group_review_priority_values(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    if row.get("candidate_bucket") == "canonical_ready":
        return "promote_now", "canonical_ready"
    if _safe_int(row.get("canonical_ready_review_count"), 0) > 0:
        return "high", "has_canonical_ready_member"
    if _safe_int(row.get("signal_evidence_count"), 0) > 0 and _safe_int(row.get("decision_maker_count"), 0) > 0:
        return "high", "has_signal_evidence_and_decision_maker"
    if _safe_int(row.get("signal_evidence_count"), 0) > 0:
        return "medium", "has_signal_evidence"
    if _safe_int(row.get("decision_maker_count"), 0) > 0:
        return "medium", "has_decision_maker"
    if _safe_int(row.get("distinct_source_count"), 0) > 1:
        return "medium", "cross_source_corroboration"
    return "low", str(row.get("canonical_gap_reason") or "low_signal")


def _company_signal_review_priority_snapshot(
    row: Mapping[str, Any] | None,
    *,
    scope: str,
) -> tuple[str | None, str | None]:
    if not row:
        return None, None
    if scope == "candidate":
        return _company_signal_candidate_review_priority_values(row)
    return _company_signal_group_review_priority_values(row)


def _optional_query_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _required_query_text(value: Any, field_name: str) -> str:
    normalized = _optional_query_text(value)
    if normalized is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    return normalized


def _normalize_source_query(value: Any) -> str | None:
    value = _optional_query_text(value)
    return value.lower() if value else None


def _company_signal_review_unlock_snapshot(
    row: Mapping[str, Any] | None,
    *,
    scope: str,
) -> dict[str, Any]:
    if not row:
        return company_signal_review_unlock_snapshot()
    if scope == "candidate":
        return company_signal_review_unlock_snapshot(
            source=row.get("source"),
            canonical_gap_reason=row.get("canonical_gap_reason"),
            confidence_score=row.get("confidence_score"),
            urgency_score=row.get("urgency_score"),
        )
    return company_signal_review_unlock_snapshot(
        source=row.get("representative_source"),
        canonical_gap_reason=row.get("canonical_gap_reason"),
        confidence_score=row.get("corroborated_confidence_score"),
        urgency_score=row.get("max_urgency_score"),
    )


async def _record_company_signal_review_event(
    pool,
    *,
    review_batch_id: str,
    review_scope: str,
    review_action: str,
    company_name: str,
    vendor_name: str,
    reviewer: str,
    review_notes: str | None,
    rebuild_requested: bool,
    rebuild: dict[str, Any] | None,
    candidate_id: str | None = None,
    candidate_group_id: str | None = None,
    review_priority_band: str | None = None,
    review_priority_reason: str | None = None,
    candidate_source: str | None = None,
    canonical_gap_reason: str | None = None,
    review_unlock_path: str | None = None,
    review_unlock_reason: str | None = None,
    company_signal_id: str | None = None,
    company_signal_action: str = "none",
) -> None:
    outcome = _normalize_company_signal_review_rebuild(
        rebuild,
        rebuild_requested=rebuild_requested,
    )
    try:
        await pool.execute(
            """
            INSERT INTO b2b_company_signal_review_events (
                review_batch_id,
                review_scope,
                review_action,
                candidate_id,
                candidate_group_id,
                company_name,
                vendor_name,
                reviewer,
                review_notes,
                review_priority_band,
                review_priority_reason,
                candidate_source,
                canonical_gap_reason,
                review_unlock_path,
                review_unlock_reason,
                company_signal_id,
                company_signal_action,
                rebuild_requested,
                rebuild_triggered,
                rebuild_reason,
                rebuild_as_of,
                rebuild_persisted_count,
                rebuild_total_accounts,
                rebuild_vendor_count
            ) VALUES (
                $1::uuid,
                $2,
                $3,
                $4::uuid,
                $5::uuid,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16::uuid,
                $17,
                $18,
                $19,
                $20::date,
                $21,
                $22,
                $23
            )
            """,
            review_batch_id,
            review_scope,
            review_action,
            candidate_id,
            candidate_group_id,
            company_name,
            vendor_name,
            reviewer,
            review_notes,
            review_priority_band,
            review_priority_reason,
            candidate_source,
            canonical_gap_reason,
            review_unlock_path,
            review_unlock_reason,
            company_signal_id,
            company_signal_action,
            outcome["rebuild_requested"],
            outcome["rebuild_triggered"],
            outcome["rebuild_reason"],
            outcome["rebuild_as_of"],
            outcome["rebuild_persisted_count"],
            outcome["rebuild_total_accounts"],
            outcome["rebuild_vendor_count"],
        )
    except Exception:
        logger.exception(
            "Failed to persist company signal review event for %s/%s",
            vendor_name,
            company_name,
        )


def _build_company_signal_update_webhook_event(
    update: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    vendor_name = _optional_query_text(update.get("vendor_name"))
    company_name = _optional_query_text(update.get("company_name"))
    company_signal_id = (
        _optional_query_text(update.get("company_signal_id"))
        or _optional_query_text(update.get("retracted_company_signal_id"))
    )
    company_signal_action = _optional_query_text(update.get("company_signal_action")) or "none"
    if not vendor_name or not company_name or not company_signal_id or company_signal_action == "none":
        return None

    payload = {
        "signal_type": "company_signal",
        "company_name": company_name,
        "company_signal_id": company_signal_id,
        "company_signal_action": company_signal_action,
    }
    review_id = (
        _optional_query_text(update.get("review_id"))
        or _optional_query_text(update.get("representative_review_id"))
    )
    if review_id:
        payload["review_id"] = review_id
    review_scope = _optional_query_text(update.get("review_scope"))
    if review_scope:
        payload["review_scope"] = review_scope
    review_action = _optional_query_text(update.get("review_action"))
    if review_action:
        payload["review_action"] = review_action
    candidate_source = _optional_query_text(update.get("candidate_source"))
    if candidate_source:
        payload["source"] = candidate_source
    return vendor_name, payload


async def _dispatch_company_signal_update_webhooks(
    pool,
    updates: list[Mapping[str, Any]],
) -> int:
    events: list[tuple[str, dict[str, Any]]] = []
    for update in updates:
        event = _build_company_signal_update_webhook_event(update)
        if event is not None:
            events.append(event)

    if not events:
        return 0

    try:
        from ..services.b2b.webhook_dispatcher import dispatch_webhooks_multi

        return await dispatch_webhooks_multi(pool, "signal_update", events)
    except Exception:
        logger.exception("Failed to dispatch company signal update webhooks")
        return 0


async def _trigger_accounts_in_motion_rebuilds(
    pool,
    vendor_names: list[str],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for vendor_name in _unique_strings_preserving_order(vendor_names):
        if not enabled:
            results.append(
                {
                    "vendor_name": vendor_name,
                    "triggered": False,
                    "reason": "disabled",
                }
            )
            continue
        try:
            rebuild = await _trigger_accounts_in_motion_rebuild(pool, vendor_name)
        except Exception:
            logger.exception(
                "Company signal candidate review rebuild failed for %s",
                vendor_name,
            )
            rebuild = {"triggered": False, "reason": "rebuild_failed"}
        results.append(
            {
                "vendor_name": vendor_name,
                **rebuild,
            }
        )
    return results


async def _trigger_accounts_in_motion_rebuild(pool, vendor_name: str) -> dict[str, Any]:
    as_of = await latest_complete_core_report_date(pool)
    if as_of is None:
        return {
            "triggered": False,
            "reason": "no_complete_core_run",
        }
    from ..autonomous.tasks import b2b_accounts_in_motion as accounts_mod

    task = ScheduledTask(
        id=_uuid.uuid4(),
        name="company_signal_candidate_review_rebuild",
        task_type="builtin",
        schedule_type="once",
        metadata={
            "maintenance_run": True,
            "test_vendors": [vendor_name],
        },
    )
    result = await accounts_mod.run(task, as_of=as_of)
    return {
        "triggered": True,
        "vendor_name": vendor_name,
        "as_of": str(as_of),
        "persisted": _safe_int(_safe_dict(result).get("persisted"), 0),
        "total_accounts": _safe_int(_safe_dict(result).get("total_accounts"), 0),
        "vendors": _safe_int(_safe_dict(result).get("vendors"), 0),
        "result": result,
    }


from ..services.b2b.corrections import suppress_predicate as _suppress_predicate  # noqa: E402
from ..services.b2b.corrections import apply_field_overrides as _apply_field_overrides  # noqa: E402


# ---------------------------------------------------------------------------
# GET /signals
# ---------------------------------------------------------------------------


@router.get("/signals")
async def list_signals(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _optional_query_text(vendor_name)
    category = _optional_query_text(category)
    pool = _pool_or_503()
    tracked_account_id = user.account_id if _should_scope(user) else None
    rows = await read_vendor_signal_rows(
        pool,
        vendor_name_query=vendor_name,
        min_urgency=min_urgency,
        product_category=category,
        tracked_account_id=tracked_account_id,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
        limit=limit,
    )

    summary = await read_vendor_signal_summary(
        pool,
        vendor_name_query=vendor_name,
        min_urgency=min_urgency,
        product_category=category,
        tracked_account_id=tracked_account_id,
        exclude_suppressed=True,
    )

    signals = [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "total_reviews": r["total_reviews"],
            "churn_intent_count": r["churn_intent_count"],
            "avg_urgency_score": _safe_float(r["avg_urgency_score"], 0.0),
            "avg_rating_normalized": _safe_float(r["avg_rating_normalized"]),
            "nps_proxy": _safe_float(r["nps_proxy"]),
            "price_complaint_rate": _safe_float(r["price_complaint_rate"]),
            "decision_maker_churn_rate": _safe_float(r["decision_maker_churn_rate"]),
            "support_sentiment": _safe_float(r["support_sentiment"]),
            "legacy_support_score": _safe_float(r["legacy_support_score"]),
            "new_feature_velocity": _safe_float(r["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(r["employee_growth_rate"]),
            "archetype": None,
            "archetype_confidence": None,
            "reasoning_mode": None,
            "reasoning_risk_level": None,
            "keyword_spike_count": r["keyword_spike_count"],
            "insider_signal_count": r["insider_signal_count"],
            "last_computed_at": str(r["last_computed_at"]) if r["last_computed_at"] else None,
        }
        for r in rows
    ]
    reasoning_views = await _load_reasoning_views_for_vendors(
        pool,
        [signal.get("vendor_name", "") for signal in signals],
    )
    for signal in signals:
        view = reasoning_views.get(_normalize_vendor_name(signal.get("vendor_name")))
        if view is not None:
            _overlay_reasoning_summary_from_view(signal, view)

    return {
        "signals": signals,
        "count": len(signals),
        "total_vendors": summary["total_vendors"] if summary else 0,
        "high_urgency_count": summary["high_urgency_count"] if summary else 0,
        "total_signal_reviews": summary["total_signal_reviews"] if summary else 0,
    }


@router.get("/slow-burn-watchlist")
async def list_slow_burn_watchlist(
    vendor_name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _optional_query_text(vendor_name)
    category = _optional_query_text(category)
    pool = _pool_or_503()
    rows = await read_ranked_vendor_signal_rows(
        pool,
        vendor_name_query=vendor_name,
        product_category=category,
        tracked_account_id=user.account_id if _should_scope(user) else None,
        exclude_suppressed=True,
        require_snapshot_activity=True,
        limit=limit,
    )

    signals = [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "total_reviews": r["total_reviews"],
            "churn_intent_count": r["churn_intent_count"],
            "avg_urgency_score": _safe_float(r["avg_urgency_score"], 0.0),
            "avg_rating_normalized": _safe_float(r["avg_rating_normalized"]),
            "nps_proxy": _safe_float(r["nps_proxy"]),
            "price_complaint_rate": _safe_float(r["price_complaint_rate"]),
            "decision_maker_churn_rate": _safe_float(r["decision_maker_churn_rate"]),
            "support_sentiment": _safe_float(r["support_sentiment"]),
            "legacy_support_score": _safe_float(r["legacy_support_score"]),
            "new_feature_velocity": _safe_float(r["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(r["employee_growth_rate"]),
            "archetype": None,
            "archetype_confidence": None,
            "reasoning_mode": None,
            "reasoning_risk_level": None,
            "keyword_spike_count": r["keyword_spike_count"],
            "insider_signal_count": r["insider_signal_count"],
            "last_computed_at": str(r["last_computed_at"]) if r["last_computed_at"] else None,
        }
        for r in rows
    ]
    reasoning_views = await _load_reasoning_views_for_vendors(
        pool,
        [signal.get("vendor_name", "") for signal in signals],
    )
    for signal in signals:
        view = reasoning_views.get(_normalize_vendor_name(signal.get("vendor_name")))
        if view is not None:
            _overlay_reasoning_summary_from_view(signal, view)

    return {
        "signals": signals,
        "count": len(rows),
    }


# ---------------------------------------------------------------------------
# GET /signals/{vendor_name}
# ---------------------------------------------------------------------------


@router.get("/signals/{vendor_name}")
async def get_signal(
    vendor_name: str,
    product_category: Optional[str] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _required_query_text(vendor_name, "vendor_name")
    product_category = _optional_query_text(product_category)
    pool = _pool_or_503()
    vname = vendor_name
    row = await read_vendor_signal_detail(
        pool,
        vendor_name_query=vname,
        product_category=product_category,
        tracked_account_id=user.account_id if _should_scope(user) else None,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
    )

    if not row:
        raise HTTPException(status_code=404, detail="No churn signal found for that vendor")

    result = {
        "vendor_name": row["vendor_name"],
        "product_category": row["product_category"],
        "total_reviews": row["total_reviews"],
        "negative_reviews": row["negative_reviews"],
        "churn_intent_count": row["churn_intent_count"],
        "avg_urgency_score": _safe_float(row["avg_urgency_score"], 0.0),
        "avg_rating_normalized": _safe_float(row["avg_rating_normalized"]),
        "nps_proxy": _safe_float(row["nps_proxy"]),
        "price_complaint_rate": _safe_float(row["price_complaint_rate"]),
        "decision_maker_churn_rate": _safe_float(row["decision_maker_churn_rate"]),
        "support_sentiment": _safe_float(row.get("support_sentiment")),
        "legacy_support_score": _safe_float(row.get("legacy_support_score")),
        "new_feature_velocity": _safe_float(row.get("new_feature_velocity")),
        "employee_growth_rate": _safe_float(row.get("employee_growth_rate")),
        "top_pain_categories": _safe_json(row["top_pain_categories"]),
        "top_competitors": _safe_json(row["top_competitors"]),
        "top_feature_gaps": _safe_json(row["top_feature_gaps"]),
        "company_churn_list": _safe_json(row["company_churn_list"]),
        "quotable_evidence": _safe_json(row["quotable_evidence"]),
        "top_use_cases": _safe_json(row["top_use_cases"]),
        "top_integration_stacks": _safe_json(row["top_integration_stacks"]),
        "budget_signal_summary": _safe_json(row["budget_signal_summary"]),
        "sentiment_distribution": _safe_json(row["sentiment_distribution"]),
        "buyer_authority_summary": _safe_json(row["buyer_authority_summary"]),
        "timeline_summary": _safe_json(row["timeline_summary"]),
        "source_distribution": _safe_json(row["source_distribution"]),
        "sample_review_ids": [str(uid) for uid in (row["sample_review_ids"] or [])],
        "review_window_start": str(row["review_window_start"]) if row["review_window_start"] else None,
        "review_window_end": str(row["review_window_end"]) if row["review_window_end"] else None,
        "confidence_score": _safe_float(row["confidence_score"], 0),
        "archetype": None,
        "archetype_confidence": None,
        "reasoning_mode": None,
        "falsification_conditions": [],
        "reasoning_risk_level": None,
        "reasoning_executive_summary": None,
        "reasoning_key_signals": [],
        "reasoning_uncertainty_sources": [],
        "insider_signal_count": row.get("insider_signal_count"),
        "insider_org_health_summary": row.get("insider_org_health_summary"),
        "insider_talent_drain_rate": _safe_float(row.get("insider_talent_drain_rate")),
        "insider_quotable_evidence": _safe_json(row.get("insider_quotable_evidence")),
        "keyword_spike_count": row.get("keyword_spike_count"),
        "keyword_spike_keywords": _safe_json(row.get("keyword_spike_keywords")),
        "keyword_trend_summary": row.get("keyword_trend_summary"),
        "last_computed_at": str(row["last_computed_at"]) if row["last_computed_at"] else None,
        "created_at": str(row["created_at"]) if row["created_at"] else None,
    }
    reasoning_views = await _load_reasoning_views_for_vendors(pool, [row["vendor_name"]])
    view = reasoning_views.get(_normalize_vendor_name(row["vendor_name"]))
    if view is not None:
        _overlay_reasoning_detail_from_view(
            result,
            view,
            requested_as_of=date.today(),
        )
    result = await _apply_field_overrides(pool, "churn_signal", str(row["id"]), result)
    return result


# ---------------------------------------------------------------------------
# GET /high-intent
# ---------------------------------------------------------------------------


@router.get("/high-intent")
async def list_high_intent(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _optional_query_text(vendor_name)
    pool = _pool_or_503()
    scoped_vendors = await _get_scoped_vendors(pool, user)
    capped = min(limit, 100)

    rows = await read_high_intent_companies(
        pool,
        min_urgency=min_urgency,
        window_days=window_days,
        vendor_name=vendor_name,
        scoped_vendors=scoped_vendors,
        limit=capped,
    )

    companies = []
    for r in rows:
        companies.append({
            "company": r.get("company"),
            "vendor": r.get("vendor"),
            "category": r.get("category"),
            "role_level": r.get("role_level"),
            "decision_maker": r.get("decision_maker"),
            "urgency": _safe_float(r.get("urgency"), 0),
            "pain": r.get("pain"),
            "alternatives": r.get("alternatives"),
            "contract_signal": r.get("contract_signal"),
            "seat_count": r.get("seat_count"),
            "lock_in_level": r.get("lock_in_level"),
            "contract_end": r.get("contract_end"),
            "buying_stage": r.get("buying_stage"),
            "reviewer_title": r.get("title"),
            "company_size": r.get("company_size"),
            "industry": r.get("industry"),
            "review_id": r.get("review_id"),
            "source": r.get("source"),
            "quotes": _safe_json(r.get("quotes")),
            "intent_signals": r.get("intent_signals"),
            "relevance_score": _safe_float(r.get("relevance_score")),
            "author_churn_score": _safe_float(r.get("author_churn_score")),
            "resolution_confidence": r.get("resolution_confidence"),
            "verified_employee_count": r.get("verified_employee_count"),
            "company_domain": r.get("company_domain"),
            "company_country": r.get("company_country"),
            "revenue_range": r.get("revenue_range"),
            "founded_year": r.get("founded_year"),
            "company_description": r.get("company_description"),
        })

    return {"companies": companies, "count": len(companies)}


# ---------------------------------------------------------------------------
# GET /vendors/{vendor_name}
# ---------------------------------------------------------------------------


@router.get("/vendors/{vendor_name}")
async def get_vendor_profile(vendor_name: str, user: AuthUser | None = Depends(optional_auth)):
    vname = _required_query_text(vendor_name, "vendor_name")
    pool = _pool_or_503()

    # When authenticated, verify vendor is tracked by the user's account
    if _should_scope(user):
        is_tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND vendor_name ILIKE $2",
            user.account_id, vname,
        )
        if not is_tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    signal_row = await read_vendor_signal_detail(
        pool,
        vendor_name_query=vname,
        exclude_suppressed=True,
    )

    counts = await pool.fetchrow(
        f"""
        SELECT
            COUNT(*) AS total_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'pending') AS pending_enrichment,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND {_canonical_review_predicate()}
          AND {_suppress_predicate('review')}
        """,
        vname,
    )

    hi_rows = await read_high_intent_companies(
        pool,
        min_urgency=7.0,
        vendor_name=vname,
        limit=5,
    )

    # APPROVED-ENRICHMENT-READ: pain_category
    # Reason: inline aggregate, GROUP BY pain_category
    pain_rows = await pool.fetch(
        f"""
        SELECT enrichment->>'pain_category' AS pain, COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND enrichment_status = 'enriched'
          AND {_canonical_review_predicate()}
          AND enrichment->>'pain_category' IS NOT NULL
          AND {_suppress_predicate('review')}
        GROUP BY enrichment->>'pain_category'
        ORDER BY cnt DESC
        LIMIT 50
        """,
        vname,
    )

    profile: dict = {"vendor_name": vname}

    if signal_row:
        sig = {
            "avg_urgency_score": _safe_float(signal_row["avg_urgency_score"], 0.0),
            "churn_intent_count": signal_row["churn_intent_count"],
            "total_reviews": signal_row["total_reviews"],
            "nps_proxy": _safe_float(signal_row["nps_proxy"]),
            "price_complaint_rate": _safe_float(signal_row["price_complaint_rate"]),
            "decision_maker_churn_rate": _safe_float(signal_row["decision_maker_churn_rate"]),
            "top_pain_categories": _safe_json(signal_row["top_pain_categories"]),
            "top_competitors": _safe_json(signal_row["top_competitors"]),
            "top_feature_gaps": _safe_json(signal_row["top_feature_gaps"]),
            "quotable_evidence": _safe_json(signal_row["quotable_evidence"]),
            "top_use_cases": _safe_json(signal_row["top_use_cases"]),
            "top_integration_stacks": _safe_json(signal_row["top_integration_stacks"]),
            "budget_signal_summary": _safe_json(signal_row["budget_signal_summary"]),
            "sentiment_distribution": _safe_json(signal_row["sentiment_distribution"]),
            "buyer_authority_summary": _safe_json(signal_row["buyer_authority_summary"]),
            "timeline_summary": _safe_json(signal_row["timeline_summary"]),
            "archetype": None,
            "archetype_confidence": None,
            "reasoning_mode": None,
            "falsification_conditions": [],
            "reasoning_risk_level": None,
            "reasoning_executive_summary": None,
            "reasoning_key_signals": [],
            "reasoning_uncertainty_sources": [],
            "insider_signal_count": signal_row.get("insider_signal_count"),
            "insider_org_health_summary": signal_row.get("insider_org_health_summary"),
            "insider_talent_drain_rate": _safe_float(signal_row.get("insider_talent_drain_rate")),
            "insider_quotable_evidence": _safe_json(signal_row.get("insider_quotable_evidence")),
            "keyword_spike_count": signal_row.get("keyword_spike_count"),
            "keyword_spike_keywords": _safe_json(signal_row.get("keyword_spike_keywords")),
            "keyword_trend_summary": signal_row.get("keyword_trend_summary"),
            "last_computed_at": str(signal_row["last_computed_at"]) if signal_row["last_computed_at"] else None,
        }
        reasoning_views = await _load_reasoning_views_for_vendors(
            pool,
            [signal_row["vendor_name"]],
        )
        view = reasoning_views.get(_normalize_vendor_name(signal_row["vendor_name"]))
        if view is not None:
            _overlay_reasoning_detail_from_view(
                sig,
                view,
                requested_as_of=date.today(),
            )
        sig = await _apply_field_overrides(pool, "churn_signal", str(signal_row["id"]), sig)
        profile["churn_signal"] = sig
    else:
        profile["churn_signal"] = None

    profile["review_counts"] = {
        "total": counts["total_reviews"] if counts else 0,
        "pending_enrichment": counts["pending_enrichment"] if counts else 0,
        "enriched": counts["enriched"] if counts else 0,
    }

    profile["high_intent_companies"] = [
        {
            "company": r.get("company"),
            "urgency": _safe_float(r.get("urgency"), 0),
            "pain": r.get("pain"),
            "title": r.get("title"),
            "company_size": r.get("company_size"),
            "industry": r.get("industry"),
            "founded_year": r.get("founded_year"),
            "total_funding": r.get("total_funding"),
            "funding_stage": r.get("funding_stage"),
            "headcount_growth_6m": _safe_float(r.get("headcount_growth_6m")),
            "headcount_growth_12m": _safe_float(r.get("headcount_growth_12m")),
            "headcount_growth_24m": _safe_float(r.get("headcount_growth_24m")),
            "publicly_traded": r.get("publicly_traded"),
            "ticker": r.get("ticker"),
            "company_description": r.get("company_description"),
        }
        for r in hi_rows
    ]

    profile["pain_distribution"] = [
        {"pain_category": r["pain"], "count": r["cnt"]}
        for r in pain_rows
    ]

    return profile


# ---------------------------------------------------------------------------
# GET /reports
# ---------------------------------------------------------------------------

VALID_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
    "exploratory_overview",
    "vendor_comparison",
    "account_comparison",
    "account_deep_dive",
    "vendor_retention",
    "challenger_intel",
    "battle_card",
    "accounts_in_motion",
    "challenger_brief",
    "vendor_deep_dive",
)

# challenger_intel is the UI-facing alias for the challenger_brief report type
_REPORT_TYPE_ALIASES: dict[str, str] = {
    "challenger_intel": "challenger_brief",
}


@router.post("/reports/compare")
async def generate_comparison_report(
    body: VendorComparisonRequest,
    user: AuthUser | None = Depends(optional_auth),
):
    primary_vendor = _optional_query_text(body.primary_vendor)
    comparison_vendor = _optional_query_text(body.comparison_vendor)
    if not primary_vendor or not comparison_vendor:
        raise HTTPException(status_code=400, detail="Both vendors are required")
    if primary_vendor.lower() == comparison_vendor.lower():
        raise HTTPException(status_code=400, detail="Choose two different vendors")
    pool = _pool_or_503()
    if _should_scope(user):
        tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND vendor_name ILIKE $2 LIMIT 1",
            user.account_id,
            primary_vendor,
        )
        if not tracked:
            raise HTTPException(status_code=403, detail="Primary vendor must be in your tracked vendor list")
    from ..autonomous.tasks.b2b_churn_intelligence import generate_vendor_comparison_report

    report = await generate_vendor_comparison_report(
        pool,
        primary_vendor,
        comparison_vendor,
        window_days=body.window_days,
        persist=body.persist,
    )
    if not report:
        raise HTTPException(status_code=404, detail="Insufficient comparison data for the selected vendors")
    return report


@router.post("/reports/compare-companies")
async def generate_account_comparison_report(
    body: AccountComparisonRequest,
    user: AuthUser | None = Depends(optional_auth),
):
    primary_company = _optional_query_text(body.primary_company)
    comparison_company = _optional_query_text(body.comparison_company)
    if not primary_company or not comparison_company:
        raise HTTPException(status_code=400, detail="Both companies are required")
    if primary_company.lower() == comparison_company.lower():
        raise HTTPException(status_code=400, detail="Choose two different companies")
    pool = _pool_or_503()
    from ..autonomous.tasks.b2b_churn_intelligence import generate_company_comparison_report

    report = await generate_company_comparison_report(
        pool,
        primary_company,
        comparison_company,
        window_days=body.window_days,
        persist=body.persist,
        account_id=(user.account_id if user else None),
    )
    if not report:
        raise HTTPException(status_code=404, detail="Insufficient company comparison data for the selected accounts")
    return report


@router.post("/reports/company-deep-dive")
async def generate_account_deep_dive_report(
    body: AccountDeepDiveRequest,
    user: AuthUser | None = Depends(optional_auth),
):
    company_name = _optional_query_text(body.company_name)
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required")
    pool = _pool_or_503()
    from ..autonomous.tasks.b2b_churn_intelligence import generate_company_deep_dive_report

    report = await generate_company_deep_dive_report(
        pool,
        company_name,
        window_days=body.window_days,
        persist=body.persist,
        account_id=(user.account_id if user else None),
    )
    if not report:
        raise HTTPException(status_code=404, detail="No company-level churn data found for the selected account")
    return report


# ---------------------------------------------------------------------------
# On-demand vendor reasoning
# ---------------------------------------------------------------------------


@router.post("/vendors/{vendor_name}/reason")
async def reason_vendor(
    vendor_name: str,
    force: bool = Query(False, description="Compatibility no-op; returns the best available persisted reasoning"),
    user: AuthUser | None = Depends(optional_auth),
):
    """Return the best available persisted reasoning for a single vendor."""
    _ = force
    vendor_name = _required_query_text(vendor_name, "vendor_name")

    pool = _pool_or_503()

    from ..autonomous.tasks._b2b_synthesis_reader import (
        load_best_reasoning_view,
        synthesis_view_to_reasoning_entry,
    )
    view = await load_best_reasoning_view(
        pool,
        vendor_name,
    )
    if view is None:
        raise HTTPException(status_code=404, detail=f"No reasoning data for vendor: {vendor_name}")
    entry = synthesis_view_to_reasoning_entry(view)
    conclusion = {
        "archetype": entry.get("archetype"),
        "risk_level": entry.get("risk_level"),
        "executive_summary": entry.get("executive_summary"),
        "key_signals": entry.get("key_signals", []),
        "falsification_conditions": entry.get("falsification_conditions", []),
        "uncertainty_sources": entry.get("uncertainty_sources", []),
    }
    return {
        "vendor_name": view.vendor_name,
        "mode": entry.get("mode"),
        "cached": True,
        "confidence": entry.get("confidence"),
        "tokens_used": 0,
        "archetype": conclusion.get("archetype"),
        "risk_level": conclusion.get("risk_level"),
        "executive_summary": conclusion.get("executive_summary"),
        "key_signals": conclusion.get("key_signals", []),
        "falsification_conditions": conclusion.get("falsification_conditions", []),
        "uncertainty_sources": conclusion.get("uncertainty_sources", []),
    }


@router.post("/vendors/compare-reasoning")
async def compare_vendor_reasoning(
    body: dict,
    user: AuthUser | None = Depends(optional_auth),
):
    """Side-by-side persisted reasoning for multiple vendors (2-5)."""

    raw_vendors = body.get("vendors", [])
    _ = body.get("force", False)

    if not isinstance(raw_vendors, list):
        raise HTTPException(status_code=400, detail="vendors must be a list of 2-5 vendor names")
    vendors = [_optional_query_text(value) for value in raw_vendors]
    vendors = [value for value in vendors if value]
    if len(vendors) < 2 or len(vendors) > 5:
        raise HTTPException(status_code=400, detail="vendors must be a list of 2-5 vendor names")

    pool = _pool_or_503()

    from ..autonomous.tasks._b2b_synthesis_reader import (
        load_best_reasoning_views,
        synthesis_view_to_reasoning_entry,
    )
    views = await load_best_reasoning_views(
        pool,
        vendors,
    )
    results = []
    for requested_name in vendors:
        matched_view = None
        for current_name, view in views.items():
            if str(current_name or "").strip().lower() == str(requested_name or "").strip().lower():
                matched_view = view
                break
        if matched_view is None:
            results.append({"vendor_name": requested_name, "error": "No reasoning data"})
            continue
        entry = synthesis_view_to_reasoning_entry(matched_view)
        results.append({
            "vendor_name": matched_view.vendor_name,
            "mode": entry.get("mode"),
            "cached": True,
            "confidence": entry.get("confidence"),
            "tokens_used": 0,
            "archetype": entry.get("archetype"),
            "risk_level": entry.get("risk_level"),
            "executive_summary": entry.get("executive_summary"),
            "key_signals": entry.get("key_signals", []),
            "falsification_conditions": entry.get("falsification_conditions", []),
        })
    return {"vendors": results, "count": len(results)}


@router.get("/reports")
async def list_reports(
    report_type: Optional[str] = Query(None),
    vendor_filter: Optional[str] = Query(None),
    include_stale: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    user: AuthUser | None = Depends(optional_auth),
):
    report_type = _optional_query_text(report_type)
    vendor_filter = _optional_query_text(vendor_filter)
    if report_type and report_type not in VALID_REPORT_TYPES:
        raise HTTPException(status_code=400, detail=f"report_type must be one of {VALID_REPORT_TYPES}")

    pool = _pool_or_503()
    conditions: list[str] = []
    params: list = []
    idx = 1

    if _should_scope(user):
        conditions.append(
            f"(vendor_filter IS NULL OR LOWER(vendor_filter) IN (SELECT LOWER(vendor_name) FROM tracked_vendors WHERE account_id = ${idx}::uuid) OR account_id = ${idx}::uuid)"
        )
        params.append(user.account_id)
        idx += 1

    if report_type:
        resolved_type = _REPORT_TYPE_ALIASES.get(report_type, report_type)
        conditions.append(f"report_type = ${idx}")
        params.append(resolved_type)
        idx += 1

    if vendor_filter:
        conditions.append(f"vendor_filter ILIKE '%' || ${idx} || '%'")
        params.append(vendor_filter)
        idx += 1

    if not include_stale:
        conditions.append(
            "COALESCE((intelligence_data->>'data_stale')::boolean, false) = false"
        )

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    capped = min(limit, 500)
    params.append(capped)

    rows = await pool.fetch(
        f"""
        SELECT id, report_date, report_type, executive_summary,
             vendor_filter, category_filter, status, created_at,
             COALESCE((intelligence_data->>'data_stale')::boolean, false) AS data_stale,
             latest_failure_step, latest_error_code, latest_error_summary,
             blocker_count, warning_count,
             (
               SELECT COUNT(*)
               FROM pipeline_visibility_reviews r
               JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
               WHERE r.status = 'open'
                 AND e.entity_type = 'churn_report'
                 AND e.entity_id = b2b_intelligence.id::text
             ) AS unresolved_issue_count,
             CASE
               WHEN report_type = 'battle_card'
               THEN COALESCE(intelligence_data->>'quality_status', intelligence_data->'battle_card_quality'->>'status')
               ELSE NULL
             END AS quality_status,
             CASE
               WHEN report_type = 'battle_card'
               THEN COALESCE(
                 (intelligence_data->'battle_card_quality'->>'score')::int,
                 NULL
               )
               ELSE NULL
             END AS quality_score
        FROM b2b_intelligence
        {where}
        ORDER BY report_date DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        LIMIT ${idx}
        """,
        *params,
    )

    reports = []
    for r in rows:
        blocker_count = r["blocker_count"] or 0
        warning_count = r["warning_count"] or 0
        unresolved_issue_count = r["unresolved_issue_count"] or 0
        trust = report_trust_payload(
            report_date=r["report_date"],
            created_at=r["created_at"],
            data_stale=bool(r["data_stale"]),
            blocker_count=blocker_count,
            warning_count=warning_count,
            unresolved_issue_count=unresolved_issue_count,
            status=r["status"],
        )
        reports.append(
            {
                "id": str(r["id"]),
                "report_date": str(r["report_date"]) if r["report_date"] else None,
                "report_type": r["report_type"],
                "executive_summary": r["executive_summary"],
                "vendor_filter": r["vendor_filter"],
                "category_filter": r["category_filter"],
                "status": _normalize_crm_push_status(r["status"]),
                "latest_failure_step": r["latest_failure_step"],
                "latest_error_code": r["latest_error_code"],
                "latest_error_summary": r["latest_error_summary"],
                "blocker_count": blocker_count,
                "warning_count": warning_count,
                "unresolved_issue_count": unresolved_issue_count,
                "quality_status": r["quality_status"],
                "quality_score": r["quality_score"],
                "artifact_state": trust["artifact_state"],
                "artifact_label": trust["artifact_label"],
                "freshness_state": trust["freshness_state"],
                "freshness_label": trust["freshness_label"],
                "review_state": trust["review_state"],
                "review_label": trust["review_label"],
                "trust": trust,
                "has_pdf_export": trust["artifact_state"] == "ready",
                "created_at": str(r["created_at"]) if r["created_at"] else None,
            }
        )

    return {"reports": reports, "count": len(reports)}


# ---------------------------------------------------------------------------
# GET /reports/{report_id}
# ---------------------------------------------------------------------------


@router.get("/reports/{report_id}")
async def get_report(report_id: str, user: AuthUser | None = Depends(optional_auth)):
    try:
        rid = _uuid.UUID(report_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid report_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow("SELECT * FROM b2b_intelligence WHERE id = $1", rid)

    if not row:
        raise HTTPException(status_code=404, detail="Report not found")

    if _should_scope(user) and row["account_id"] == user.account_id:
        pass
    elif _should_scope(user) and row["vendor_filter"]:
        is_tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND LOWER(vendor_name) = LOWER($2)",
            user.account_id, row["vendor_filter"],
        )
        if not is_tracked:
            raise HTTPException(status_code=403, detail="Report vendor not in your tracked list")
    # NULL vendor_filter (global reports) visible to any authenticated user
    intelligence_data = _safe_json(row["intelligence_data"])
    quality = {}
    quality_status = None
    unresolved_issue_count = await pool.fetchval(
        """
        SELECT COUNT(*)
        FROM pipeline_visibility_reviews r
        JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
        WHERE r.status = 'open'
          AND e.entity_type = 'churn_report'
          AND e.entity_id = $1
        """,
        str(row["id"]),
    )
    if isinstance(intelligence_data, dict):
        quality = _safe_dict(intelligence_data.get("battle_card_quality"))
        quality_status = intelligence_data.get("quality_status")
    data_stale = False
    if isinstance(intelligence_data, dict):
        data_stale = bool(intelligence_data.get("data_stale") is True)
    blocker_count = row["blocker_count"] or 0
    warning_count = row["warning_count"] or 0
    open_issue_count = unresolved_issue_count or 0
    section_evidence = report_section_evidence_payload(intelligence_data)
    trust = report_trust_payload(
        report_date=row["report_date"],
        created_at=row["created_at"],
        data_stale=data_stale,
        blocker_count=blocker_count,
        warning_count=warning_count,
        unresolved_issue_count=open_issue_count,
        status=row["status"],
    )

    return {
        "id": str(row["id"]),
        "report_date": str(row["report_date"]) if row["report_date"] else None,
        "report_type": row["report_type"],
        "vendor_filter": row["vendor_filter"],
        "category_filter": row["category_filter"],
        "executive_summary": row["executive_summary"],
        "intelligence_data": intelligence_data,
        "section_evidence": section_evidence,
        "data_density": _safe_json(row["data_density"]),
        "status": row["status"],
        "latest_failure_step": row["latest_failure_step"],
        "latest_error_code": row["latest_error_code"],
        "latest_error_summary": row["latest_error_summary"],
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "unresolved_issue_count": open_issue_count,
        "quality_status": quality_status or quality.get("status"),
        "quality_score": quality.get("score"),
        "artifact_state": trust["artifact_state"],
        "artifact_label": trust["artifact_label"],
        "freshness_state": trust["freshness_state"],
        "freshness_label": trust["freshness_label"],
        "review_state": trust["review_state"],
        "review_label": trust["review_label"],
        "trust": trust,
        "has_pdf_export": trust["artifact_state"] == "ready",
        "llm_model": row["llm_model"],
        "created_at": str(row["created_at"]) if row["created_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /reports/{report_id}/pdf
# ---------------------------------------------------------------------------


@router.get("/reports/{report_id}/pdf")
async def export_report_pdf(report_id: str, user: AuthUser | None = Depends(optional_auth)):
    """Download a B2B intelligence report as PDF."""
    try:
        rid = _uuid.UUID(report_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid report_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow("SELECT * FROM b2b_intelligence WHERE id = $1", rid)

    if not row:
        raise HTTPException(status_code=404, detail="Report not found")

    # Access control (same as get_report)
    if _should_scope(user) and row["account_id"] == user.account_id:
        pass
    elif _should_scope(user) and row["vendor_filter"]:
        is_tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND LOWER(vendor_name) = LOWER($2)",
            user.account_id, row["vendor_filter"],
        )
        if not is_tracked:
            raise HTTPException(status_code=403, detail="Report vendor not in your tracked list")

    from ..services.b2b.pdf_renderer import render_report_pdf

    pdf_bytes = render_report_pdf(
        report_type=row["report_type"],
        vendor_filter=row["vendor_filter"],
        category_filter=row["category_filter"],
        report_date=row["report_date"],
        executive_summary=row["executive_summary"],
        intelligence_data=_safe_json(row["intelligence_data"]),
        data_density=_safe_json(row["data_density"]),
    )

    vendor = row["vendor_filter"] or row["report_type"]
    filename = f"atlas-report-{vendor}-{row['report_date'] or 'latest'}.pdf"
    filename = re.sub(r"[^a-z0-9._-]", "-", filename.lower())

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /reviews
# ---------------------------------------------------------------------------


@router.get("/reviews")
async def search_reviews(
    vendor_name: Optional[str] = Query(None),
    pain_category: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0, le=10),
    min_relevance: Optional[float] = Query(None, ge=0, le=1),
    company: Optional[str] = Query(None),
    has_churn_intent: Optional[bool] = Query(None),
    exclude_low_fidelity: bool = Query(False),
    window_days: int = Query(30, ge=1, le=3650),
    limit: int = Query(20, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    from ..autonomous.tasks._b2b_shared import read_review_details

    vendor_name = _optional_query_text(vendor_name)
    pain_category = _optional_query_text(pain_category)
    company = _optional_query_text(company)
    pool = _pool_or_503()

    scoped_vendors: list[str] | None = None
    if _should_scope(user):
        rows = await pool.fetch(
            "SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid",
            user.account_id,
        )
        scoped_vendors = [r["vendor_name"] for r in rows]

    capped = min(limit, 100)
    details = await read_review_details(
        pool,
        window_days=window_days,
        vendor_name=vendor_name,
        scoped_vendors=scoped_vendors,
        pain_category=pain_category,
        min_urgency=min_urgency,
        company=company,
        has_churn_intent=has_churn_intent,
        min_relevance=min_relevance,
        exclude_low_fidelity=exclude_low_fidelity,
        recency_column="enriched_at",
        limit=capped,
    )

    reviews = [
        {
            "id": d.get("id"),
            "vendor_name": d.get("vendor_name"),
            "product_category": d.get("product_category"),
            "reviewer_company": d.get("reviewer_company"),
            "rating": d.get("rating"),
            "urgency_score": d.get("urgency_score"),
            "pain_category": d.get("pain_category"),
            "intent_to_leave": d.get("intent_to_leave"),
            "decision_maker": d.get("decision_maker"),
            "source": d.get("source"),
            "reviewed_at": str(d["reviewed_at"]) if d.get("reviewed_at") else None,
            "role_level": d.get("role_level"),
            "buying_stage": d.get("buying_stage"),
            "sentiment_direction": d["sentiment_direction"] if d.get("sentiment_direction") != "unknown" else None,
            "competitors_mentioned": d.get("competitors_mentioned") or [],
            "quotable_phrases": d.get("quotable_phrases") or [],
            "positive_aspects": d.get("positive_aspects") or [],
            "specific_complaints": d.get("specific_complaints") or [],
            "enriched_at": str(d["enriched_at"]) if d.get("enriched_at") else None,
            "reviewer_title": d.get("reviewer_title"),
            "company_size": d.get("company_size"),
            "industry": d.get("industry"),
            "relevance_score": d.get("relevance_score"),
            "author_churn_score": d.get("author_churn_score"),
            "low_fidelity": bool(d["low_fidelity"]) if d.get("low_fidelity") is not None else False,
            "low_fidelity_reasons": d.get("low_fidelity_reasons") or [],
        }
        for d in details
    ]

    return {"reviews": reviews, "count": len(reviews)}


# ---------------------------------------------------------------------------
# GET /reviews/{review_id}
# ---------------------------------------------------------------------------


@router.get("/reviews/{review_id}")
async def get_review(review_id: str, user: AuthUser | None = Depends(optional_auth)):
    try:
        rid = _uuid.UUID(review_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid review_id (must be UUID)")

    pool = _pool_or_503()
    row = await pool.fetchrow("SELECT * FROM b2b_reviews WHERE id = $1", rid)

    if not row:
        raise HTTPException(status_code=404, detail="Review not found")

    suppressed = await pool.fetchval(
        """SELECT 1 FROM data_corrections
           WHERE entity_type = 'review' AND entity_id = $1
             AND correction_type = 'suppress' AND status = 'applied'""",
        row["id"],
    )
    if suppressed:
        raise HTTPException(status_code=404, detail="Review not found")

    if _should_scope(user):
        is_tracked = await pool.fetchval(
            "SELECT 1 FROM tracked_vendors WHERE account_id = $1::uuid AND vendor_name ILIKE $2",
            user.account_id, row["vendor_name"],
        )
        if not is_tracked:
            raise HTTPException(status_code=403, detail="Vendor not in your tracked list")

    result = {
        "id": str(row["id"]),
        "source": row["source"],
        "source_url": row["source_url"],
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"],
        "product_category": row["product_category"],
        "rating": _safe_float(row["rating"]),
        "summary": row["summary"],
        "review_text": row["review_text"],
        "pros": row["pros"],
        "cons": row["cons"],
        "reviewer_name": row["reviewer_name"],
        "reviewer_title": row["reviewer_title"],
        "reviewer_company": row["reviewer_company"],
        "company_size_raw": row["company_size_raw"],
        "reviewer_industry": row["reviewer_industry"],
        "reviewed_at": str(row["reviewed_at"]) if row["reviewed_at"] else None,
        "imported_at": str(row["imported_at"]) if row["imported_at"] else None,
        "enrichment": _safe_json(row["enrichment"]),
        "enrichment_status": row["enrichment_status"],
        "enriched_at": str(row["enriched_at"]) if row["enriched_at"] else None,
    }
    result = await _apply_field_overrides(pool, "review", str(row["id"]), result)
    return result


# ---------------------------------------------------------------------------
# GET /pipeline
# ---------------------------------------------------------------------------


@router.get("/pipeline")
async def get_pipeline_status(user: AuthUser | None = Depends(optional_auth)):
    pool = _pool_or_503()

    # Build vendor scope clause for authenticated users
    _rev_sup = _suppress_predicate('review')
    vendor_scope = f" WHERE {_rev_sup}"
    scrape_scope = ""
    scope_params: list = []
    if _should_scope(user):
        vendor_scope = (
            f" WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid)"
            f" AND {_canonical_review_predicate()}"
            f" AND {_rev_sup}"
        )
        scrape_scope = " WHERE vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid)"
        scope_params = [user.account_id]
    else:
        vendor_scope = f" WHERE {_canonical_review_predicate()} AND {_rev_sup}"

    status_rows = await pool.fetch(
        f"""
        SELECT enrichment_status, COUNT(*) AS cnt
        FROM b2b_reviews
        {vendor_scope}
        GROUP BY enrichment_status
        """,
        *scope_params,
    )
    enrichment_counts = {r["enrichment_status"]: r["cnt"] for r in status_rows}

    stats = await pool.fetchrow(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE imported_at > NOW() - INTERVAL '24 hours') AS recent_imports_24h,
            MAX(enriched_at) AS last_enrichment_at
        FROM b2b_reviews
        {vendor_scope}
        """,
        *scope_params,
    )

    scrape_stats = await pool.fetchrow(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE enabled) AS active_scrape_targets,
            MAX(last_scraped_at) AS last_scrape_at
        FROM b2b_scrape_targets
        {scrape_scope}
        """,
        *scope_params,
    )

    return {
        "enrichment_counts": enrichment_counts,
        "recent_imports_24h": stats["recent_imports_24h"] if stats else 0,
        "last_enrichment_at": str(stats["last_enrichment_at"]) if stats and stats["last_enrichment_at"] else None,
        "active_scrape_targets": scrape_stats["active_scrape_targets"] if scrape_stats else 0,
        "last_scrape_at": str(scrape_stats["last_scrape_at"]) if scrape_stats and scrape_stats["last_scrape_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /source-health
# ---------------------------------------------------------------------------

_SOURCE_HEALTH_SQL = """
WITH current_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'partial')          AS partial_count,
        COUNT(*) FILTER (WHERE status = 'failed')           AS failed_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found,
        AVG(reviews_inserted)                               AS avg_reviews_inserted,
        AVG(duration_ms)                                    AS avg_duration_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
        MAX(started_at) FILTER (WHERE status = 'success')   AS last_success_at,
        MAX(started_at)                                     AS last_scrape_at
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1)
    {source_filter_current}
    GROUP BY source
),
prev_window AS (
    SELECT
        source,
        COUNT(*)                                            AS total_scrapes,
        COUNT(*) FILTER (WHERE status = 'success')          AS success_count,
        COUNT(*) FILTER (WHERE status = 'blocked')          AS blocked_count,
        AVG(reviews_found)                                  AS avg_reviews_found
    FROM b2b_scrape_log
    WHERE started_at >= NOW() - make_interval(days => $1 * 2)
      AND started_at <  NOW() - make_interval(days => $1)
    {source_filter_prev}
    GROUP BY source
),
target_counts AS (
    SELECT source, COUNT(*) FILTER (WHERE enabled) AS active_targets
    FROM b2b_scrape_targets
    {target_filter}
    GROUP BY source
)
SELECT
    c.source, c.total_scrapes, c.success_count, c.partial_count,
    c.failed_count, c.blocked_count, c.avg_reviews_found,
    c.avg_reviews_inserted, c.avg_duration_ms, c.p95_duration_ms,
    c.last_success_at, c.last_scrape_at,
    COALESCE(t.active_targets, 0)  AS active_targets,
    p.total_scrapes                AS prev_total_scrapes,
    p.success_count                AS prev_success_count,
    p.blocked_count                AS prev_blocked_count,
    p.avg_reviews_found            AS prev_avg_reviews_found
FROM current_window c
LEFT JOIN prev_window p USING (source)
LEFT JOIN target_counts t USING (source)
ORDER BY c.total_scrapes DESC
"""


def _build_source_health_query(source: str | None):
    """Return (sql, params) for the source-health CTE query."""
    if source:
        sql = _SOURCE_HEALTH_SQL.format(
            source_filter_current="AND source = $2",
            source_filter_prev="AND source = $2",
            target_filter="WHERE source = $2",
        )
        return sql, [source]
    sql = _SOURCE_HEALTH_SQL.format(
        source_filter_current="",
        source_filter_prev="",
        target_filter="",
    )
    return sql, []


def _row_to_source_dict(r) -> dict:
    """Convert a DB row to a source-health dict with computed rates."""
    total = r["total_scrapes"] or 1
    success_rate = round(r["success_count"] / total, 3)
    block_rate = round(r["blocked_count"] / total, 3)

    prev_total = r["prev_total_scrapes"] or 0
    prev_success_rate = round(r["prev_success_count"] / max(prev_total, 1), 3) if prev_total else None
    prev_block_rate = round(r["prev_blocked_count"] / max(prev_total, 1), 3) if prev_total else None
    prev_avg = _safe_float(r["prev_avg_reviews_found"])

    trend = {
        "prev_window_scrapes": prev_total,
        "prev_success_rate": prev_success_rate,
        "prev_block_rate": prev_block_rate,
        "prev_avg_reviews_found": prev_avg,
        "success_rate_delta": round(success_rate - prev_success_rate, 3) if prev_success_rate is not None else None,
        "block_rate_delta": round(block_rate - prev_block_rate, 3) if prev_block_rate is not None else None,
    }

    cap = get_capability(r["source"])
    capabilities = cap.to_dict() if cap else None

    return {
        "source": r["source"],
        "display_name": source_display_name(r["source"]),
        "total_scrapes": r["total_scrapes"],
        "success_count": r["success_count"],
        "partial_count": r["partial_count"],
        "failed_count": r["failed_count"],
        "blocked_count": r["blocked_count"],
        "success_rate": success_rate,
        "block_rate": block_rate,
        "avg_reviews_found": _safe_float(r["avg_reviews_found"]),
        "avg_reviews_inserted": _safe_float(r["avg_reviews_inserted"]),
        "avg_duration_ms": _safe_float(r["avg_duration_ms"]),
        "p95_duration_ms": _safe_float(r["p95_duration_ms"]),
        "last_success_at": str(r["last_success_at"]) if r["last_success_at"] else None,
        "last_scrape_at": str(r["last_scrape_at"]) if r["last_scrape_at"] else None,
        "active_targets": r["active_targets"],
        "capabilities": capabilities,
        "trend": trend,
    }


@router.get("/source-health")
async def get_source_health(
    window_days: int = Query(7, ge=1, le=30),
    source: Optional[str] = Query(None),
):
    source = _normalize_source_query(source)
    if source is not None:
        if source not in ALL_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source. Must be one of: {sorted(s.value for s in ALL_SOURCES)}",
            )

    sql, extra_params = _build_source_health_query(source)
    pool = _pool_or_503()
    rows = await pool.fetch(sql, window_days, *extra_params)

    sources_list = [_row_to_source_dict(r) for r in rows]

    total_scrapes = sum(s["total_scrapes"] for s in sources_list)
    total_success = sum(s["success_count"] for s in sources_list)
    total_blocked = sum(s["blocked_count"] for s in sources_list)

    summary = {
        "total_sources": len(sources_list),
        "total_scrapes": total_scrapes,
        "overall_success_rate": round(total_success / max(total_scrapes, 1), 3),
        "overall_block_rate": round(total_blocked / max(total_scrapes, 1), 3),
        "worst_source": min(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
        "best_source": max(sources_list, key=lambda s: s["success_rate"])["source"] if sources_list else None,
    }

    return {
        "window_days": window_days,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources_list,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# GET /source-health/telemetry
# ---------------------------------------------------------------------------


@router.get("/source-health/telemetry")
async def get_source_telemetry(
    window_days: int = Query(7, ge=1, le=30),
    source: Optional[str] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    """CAPTCHA attempts, solve times, block type distribution, and proxy usage per source."""
    conditions = ["started_at >= NOW() - make_interval(days => $1)"]
    params: list = [window_days]
    idx = 2
    source = _normalize_source_query(source)
    if source is not None:
        if source not in ALL_SOURCES:
            raise HTTPException(400, f"Invalid source: {source}")
        conditions.append(f"source = ${idx}")
        params.append(source)
        idx += 1

    pool = _pool_or_503()
    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT
            source,
            COUNT(*)                                                AS total_scrapes,
            SUM(COALESCE(captcha_attempts, 0))                      AS total_captcha_attempts,
            COUNT(*) FILTER (WHERE captcha_attempts > 0)            AS scrapes_with_captcha,
            AVG(captcha_solve_ms) FILTER (WHERE captcha_solve_ms > 0) AS avg_captcha_solve_ms,
            MAX(captcha_solve_ms)                                    AS max_captcha_solve_ms,
            COUNT(*) FILTER (WHERE block_type IS NOT NULL)           AS total_blocks,
            COUNT(*) FILTER (WHERE block_type = 'captcha')           AS blocks_captcha,
            COUNT(*) FILTER (WHERE block_type = 'ip_ban')            AS blocks_ip_ban,
            COUNT(*) FILTER (WHERE block_type = 'rate_limit')        AS blocks_rate_limit,
            COUNT(*) FILTER (WHERE block_type = 'waf')               AS blocks_waf,
            COUNT(*) FILTER (WHERE block_type = 'unknown')           AS blocks_unknown,
            COUNT(*) FILTER (WHERE proxy_type = 'datacenter')        AS proxy_datacenter,
            COUNT(*) FILTER (WHERE proxy_type = 'residential')       AS proxy_residential,
            COUNT(*) FILTER (WHERE proxy_type = 'none')              AS proxy_none
        FROM b2b_scrape_log
        WHERE {where}
        GROUP BY source
        ORDER BY total_captcha_attempts DESC
        """,
        *params,
    )

    sources_out = []
    for r in rows:
        total = r["total_scrapes"] or 1
        sources_out.append({
            "source": r["source"],
            "total_scrapes": r["total_scrapes"],
            "captcha": {
                "total_attempts": r["total_captcha_attempts"] or 0,
                "scrapes_with_captcha": r["scrapes_with_captcha"],
                "captcha_rate": round(r["scrapes_with_captcha"] / total, 3),
                "avg_solve_ms": round(float(r["avg_captcha_solve_ms"]), 0) if r["avg_captcha_solve_ms"] else None,
                "max_solve_ms": r["max_captcha_solve_ms"],
            },
            "blocks": {
                "total": r["total_blocks"],
                "captcha": r["blocks_captcha"],
                "ip_ban": r["blocks_ip_ban"],
                "rate_limit": r["blocks_rate_limit"],
                "waf": r["blocks_waf"],
                "unknown": r["blocks_unknown"],
            },
            "proxy_usage": {
                "datacenter": r["proxy_datacenter"],
                "residential": r["proxy_residential"],
                "none": r["proxy_none"],
            },
        })

    return {
        "window_days": window_days,
        "sources": sources_out,
        "total_sources": len(sources_out),
    }


# ---------------------------------------------------------------------------
# GET /source-capabilities
# ---------------------------------------------------------------------------


@router.get("/source-capabilities")
async def list_source_capabilities(
    source: Optional[str] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    """List source capability profiles (access pattern, anti-bot, proxy tier, fallback chain)."""
    source = _normalize_source_query(source)
    if source is not None:
        cap = get_capability(source)
        if not cap:
            raise HTTPException(404, f"No capability profile for source: {source}")
        return {"source": source, "capabilities": cap.to_dict()}

    from ..services.scraping.capabilities import get_all_capabilities
    all_caps = get_all_capabilities()
    return {
        "sources": [
            {"source": name, "capabilities": cap.to_dict()}
            for name, cap in sorted(all_caps.items())
        ],
        "total": len(all_caps),
    }


# ---------------------------------------------------------------------------
# GET /source-impact-ledger
# ---------------------------------------------------------------------------


@router.get("/source-impact-ledger")
async def get_source_impact_ledger(
    source: Optional[str] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    include_field_baseline: bool = Query(True),
    include_consumer_wiring: bool = Query(True),
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
):
    """Return source-to-pool impact mappings plus live field and wiring baselines."""
    source = _normalize_source_query(source)
    if source is not None:
        if source not in _KNOWN_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source. Must be one of: {sorted(_KNOWN_SOURCES)}",
            )

    ledger = build_source_impact_ledger(source=source)
    field_baseline: dict[str, Any] | None = None
    if include_field_baseline:
        pool = get_db_pool()
        if pool.is_initialized:
            field_baseline = await summarize_source_field_baseline(
                pool,
                window_days=window_days,
                source=source,
            )
        else:
            field_baseline = {
                "available": False,
                "reason": "Database not ready",
                "window_days": window_days,
                "source_filter": source,
                "rows": [],
                "summary": {
                    "total_sources": 0,
                    "total_reviews": 0,
                    "total_enriched_reviews": 0,
                },
            }

    consumer_wiring = (
        get_consumer_wiring_baseline() if include_consumer_wiring else None
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "source_filter": source,
        "impact_summary": ledger["summary"],
        "sources": ledger["sources"],
        "field_baseline": field_baseline,
        "consumer_wiring": consumer_wiring,
    }


# ---------------------------------------------------------------------------
# GET /operational-overview
# ---------------------------------------------------------------------------


@router.get("/operational-overview")
async def get_operational_overview(
    user: AuthUser | None = Depends(optional_auth),
):
    """Single endpoint combining pipeline status, source health, telemetry, and recent events."""
    pool = _pool_or_503()

    # Run all queries concurrently
    import asyncio
    pipeline_row, health_rows, telemetry_row, event_rows, review_count_row = await asyncio.gather(
        pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE enrichment_status = 'pending')   AS pending,
                COUNT(*) FILTER (WHERE enrichment_status = 'enriched')  AS enriched,
                COUNT(*) FILTER (WHERE enrichment_status = 'failed')    AS failed,
                COUNT(*) FILTER (WHERE enrichment_status = 'no_signal') AS no_signal,
                COUNT(*)                                                 AS total
            FROM b2b_reviews
            WHERE duplicate_of_review_id IS NULL
        """),
        pool.fetch("""
            SELECT source,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE status = 'success') AS success,
                   COUNT(*) FILTER (WHERE status = 'blocked') AS blocked
            FROM b2b_scrape_log
            WHERE started_at >= NOW() - INTERVAL '7 days'
            GROUP BY source ORDER BY total DESC
        """),
        pool.fetchrow("""
            SELECT
                SUM(COALESCE(captcha_attempts, 0))       AS captcha_total,
                COUNT(*) FILTER (WHERE block_type IS NOT NULL) AS blocks_total,
                COUNT(*) FILTER (WHERE block_type = 'captcha') AS blocks_captcha,
                COUNT(*) FILTER (WHERE block_type = 'ip_ban')  AS blocks_ip_ban,
                COUNT(*) FILTER (WHERE block_type = 'waf')     AS blocks_waf
            FROM b2b_scrape_log
            WHERE started_at >= NOW() - INTERVAL '7 days'
        """),
        pool.fetch("""
            SELECT vendor_name, event_type, event_date, description
            FROM b2b_change_events
            ORDER BY event_date DESC, created_at DESC
            LIMIT 10
        """),
        pool.fetchrow("""
            SELECT
                COUNT(*) AS total_reviews,
                COUNT(DISTINCT vendor_name) AS vendors_tracked,
                MAX(imported_at) AS last_review_at
            FROM b2b_reviews
            WHERE duplicate_of_review_id IS NULL
        """),
    )

    # Pipeline
    pipeline = {
        "pending": pipeline_row["pending"],
        "enriched": pipeline_row["enriched"],
        "failed": pipeline_row["failed"],
        "no_signal": pipeline_row["no_signal"],
        "total": pipeline_row["total"],
    }

    # Source health summary (7d)
    source_health = []
    total_scrapes = 0
    total_success = 0
    total_blocked = 0
    for r in health_rows:
        t = r["total"] or 1
        source_health.append({
            "source": r["source"],
            "total": r["total"],
            "success_rate": round(r["success"] / t, 3),
            "block_rate": round(r["blocked"] / t, 3),
        })
        total_scrapes += r["total"]
        total_success += r["success"]
        total_blocked += r["blocked"]

    # Telemetry summary (7d)
    telemetry = {
        "captcha_attempts_7d": telemetry_row["captcha_total"] or 0,
        "blocks_7d": telemetry_row["blocks_total"] or 0,
        "block_breakdown": {
            "captcha": telemetry_row["blocks_captcha"] or 0,
            "ip_ban": telemetry_row["blocks_ip_ban"] or 0,
            "waf": telemetry_row["blocks_waf"] or 0,
        },
    }

    # Recent change events
    recent_events = [
        {
            "vendor_name": r["vendor_name"],
            "event_type": r["event_type"],
            "event_date": str(r["event_date"]),
            "description": r["description"],
        }
        for r in event_rows
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "total_reviews": review_count_row["total_reviews"],
            "vendors_tracked": review_count_row["vendors_tracked"],
            "last_review_at": str(review_count_row["last_review_at"]) if review_count_row["last_review_at"] else None,
        },
        "pipeline": pipeline,
        "source_health_7d": {
            "sources": source_health,
            "overall_success_rate": round(total_success / max(total_scrapes, 1), 3),
            "overall_block_rate": round(total_blocked / max(total_scrapes, 1), 3),
            "total_scrapes": total_scrapes,
        },
        "telemetry_7d": telemetry,
        "recent_change_events": recent_events,
    }


# ---------------------------------------------------------------------------
# GET /source-health/telemetry-timeline
# ---------------------------------------------------------------------------


@router.get("/source-health/telemetry-timeline")
async def get_telemetry_timeline(
    days: int = Query(14, ge=1, le=30),
    source: Optional[str] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    """Daily time-series of CAPTCHA attempts, blocks, and success rates for trending."""
    conditions = ["started_at >= NOW() - make_interval(days => $1)"]
    params: list = [days]
    idx = 2
    source = _normalize_source_query(source)
    if source is not None:
        if source not in ALL_SOURCES:
            raise HTTPException(400, f"Invalid source: {source}")
        conditions.append(f"source = ${idx}")
        params.append(source)
        idx += 1

    pool = _pool_or_503()
    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT
            started_at::date AS day,
            COUNT(*) AS total_scrapes,
            COUNT(*) FILTER (WHERE status = 'success') AS success,
            COUNT(*) FILTER (WHERE status = 'blocked') AS blocked,
            SUM(COALESCE(captcha_attempts, 0)) AS captcha_attempts,
            AVG(captcha_solve_ms) FILTER (WHERE captcha_solve_ms > 0) AS avg_captcha_ms,
            COUNT(*) FILTER (WHERE block_type = 'captcha') AS blocks_captcha,
            COUNT(*) FILTER (WHERE block_type = 'ip_ban') AS blocks_ip_ban,
            COUNT(*) FILTER (WHERE block_type = 'rate_limit') AS blocks_rate_limit,
            COUNT(*) FILTER (WHERE block_type = 'waf') AS blocks_waf
        FROM b2b_scrape_log
        WHERE {where}
        GROUP BY day
        ORDER BY day ASC
        """,
        *params,
    )

    timeline = []
    for r in rows:
        t = r["total_scrapes"] or 1
        timeline.append({
            "date": str(r["day"]),
            "total_scrapes": r["total_scrapes"],
            "success_rate": round(r["success"] / t, 3),
            "block_rate": round(r["blocked"] / t, 3),
            "captcha_attempts": r["captcha_attempts"] or 0,
            "avg_captcha_ms": round(float(r["avg_captcha_ms"]), 0) if r["avg_captcha_ms"] else None,
            "blocks": {
                "captcha": r["blocks_captcha"],
                "ip_ban": r["blocks_ip_ban"],
                "rate_limit": r["blocks_rate_limit"],
                "waf": r["blocks_waf"],
            },
        })

    return {
        "days": days,
        "source_filter": source,
        "timeline": timeline,
        "data_points": len(timeline),
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _csv_response(rows: list[dict], filename: str) -> StreamingResponse:
    """Build a StreamingResponse from a list of dicts."""
    if not rows:
        buf = io.StringIO()
        buf.write("")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /displacement-edges
# ---------------------------------------------------------------------------


@router.get("/displacement-edges")
async def list_displacement_edges(
    from_vendor: Optional[str] = Query(None),
    to_vendor: Optional[str] = Query(None),
    min_strength: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    window_days: int = Query(90, ge=1, le=3650),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    from_vendor = _optional_query_text(from_vendor)
    to_vendor = _optional_query_text(to_vendor)
    min_strength = _optional_query_text(min_strength)
    strength_order = {"strong": 3, "moderate": 2, "emerging": 1}
    if min_strength and min_strength not in strength_order:
        raise HTTPException(400, f"Invalid min_strength: {min_strength}")

    pool = _pool_or_503()
    conditions: list[str] = ["computed_date > NOW() - make_interval(days => $1)"]
    params: list = [window_days]
    idx = 2

    if _should_scope(user):
        conditions.append(
            f"(from_vendor IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
            f" OR to_vendor IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid))"
        )
        params.append(user.account_id)
        idx += 1

    if from_vendor:
        conditions.append(f"from_vendor ILIKE '%' || ${idx} || '%'")
        params.append(from_vendor)
        idx += 1

    if to_vendor:
        conditions.append(f"to_vendor ILIKE '%' || ${idx} || '%'")
        params.append(to_vendor)
        idx += 1

    if min_strength:
        min_val = strength_order[min_strength]
        allowed = [k for k, v in strength_order.items() if v >= min_val]
        conditions.append(f"signal_strength = ANY(${idx}::text[])")
        params.append(allowed)
        idx += 1

    if min_confidence is not None:
        conditions.append(f"confidence_score >= ${idx}")
        params.append(min_confidence)
        idx += 1

    conditions.append(_suppress_predicate('displacement_edge'))
    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT id, from_vendor, to_vendor, mention_count,
               primary_driver, signal_strength, key_quote,
               source_distribution, confidence_score,
               computed_date, report_id, created_at
        FROM b2b_displacement_edges
        WHERE {where}
        ORDER BY confidence_score DESC, mention_count DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    edges = []
    for r in rows:
        edges.append({
            "id": str(r["id"]),
            "from_vendor": r["from_vendor"],
            "to_vendor": r["to_vendor"],
            "mention_count": r["mention_count"],
            "primary_driver": r["primary_driver"],
            "signal_strength": r["signal_strength"],
            "key_quote": r["key_quote"],
            "source_distribution": _safe_json(r["source_distribution"]),
            "confidence_score": _safe_float(r["confidence_score"], 0),
            "computed_date": str(r["computed_date"]),
            "report_id": str(r["report_id"]) if r["report_id"] else None,
        })

    return {"edges": edges, "count": len(edges)}


# ---------------------------------------------------------------------------
# GET /displacement-history
# ---------------------------------------------------------------------------


@router.get("/displacement-history")
async def get_displacement_history(
    from_vendor: str = Query(...),
    to_vendor: str = Query(...),
    window_days: int = Query(365, ge=1, le=730),
    user: AuthUser | None = Depends(optional_auth),
):
    """Time-series of displacement edge strength for a vendor pair."""
    from_vendor = _required_query_text(from_vendor, "from_vendor")
    to_vendor = _required_query_text(to_vendor, "to_vendor")
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT computed_date, mention_count, signal_strength,
               confidence_score, primary_driver, key_quote
        FROM b2b_displacement_edges
        WHERE LOWER(from_vendor) = LOWER($1)
          AND LOWER(to_vendor) = LOWER($2)
          AND computed_date > NOW() - make_interval(days => $3)
        ORDER BY computed_date ASC
        """,
        from_vendor, to_vendor, window_days,
    )
    history = []
    for r in rows:
        history.append({
            "computed_date": str(r["computed_date"]),
            "mention_count": r["mention_count"],
            "signal_strength": _safe_float(r["signal_strength"], 0),
            "confidence_score": _safe_float(r["confidence_score"], 0),
            "primary_driver": r["primary_driver"],
            "key_quote": r["key_quote"],
        })
    return {
        "from_vendor": from_vendor,
        "to_vendor": to_vendor,
        "window_days": window_days,
        "history": history,
        "data_points": len(history),
    }


# ---------------------------------------------------------------------------
# GET /company-signals
# ---------------------------------------------------------------------------


@router.get("/company-signals")
async def list_company_signals(
    vendor_name: Optional[str] = Query(None),
    company_name: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    decision_makers_only: bool = Query(False),
    window_days: int = Query(90, ge=1, le=3650),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    conditions: list[str] = ["last_seen_at > NOW() - make_interval(days => $1)"]
    params: list = [window_days]
    idx = 2

    if _should_scope(user):
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if company_name:
        conditions.append(f"company_name ILIKE '%' || ${idx} || '%'")
        params.append(company_name)
        idx += 1

    if min_urgency > 0:
        conditions.append(f"urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    if min_confidence is not None:
        conditions.append(f"confidence_score >= ${idx}")
        params.append(min_confidence)
        idx += 1

    if decision_makers_only:
        conditions.append("decision_maker = true")

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT id, company_name, vendor_name, urgency_score,
               pain_category, buyer_role, decision_maker,
               seat_count, contract_end, buying_stage,
               source, first_seen_at, last_seen_at,
               confidence_score
        FROM b2b_company_signals
        WHERE {where}
        ORDER BY urgency_score DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    signals = []
    for r in rows:
        signals.append({
            "id": str(r["id"]),
            "company_name": r["company_name"],
            "vendor_name": r["vendor_name"],
            "urgency_score": _safe_float(r["urgency_score"], 0),
            "pain_category": r["pain_category"],
            "buyer_role": r["buyer_role"],
            "decision_maker": r["decision_maker"],
            "seat_count": r["seat_count"],
            "contract_end": r["contract_end"],
            "buying_stage": r["buying_stage"],
            "source": r["source"],
            "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
            "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            "confidence_score": _safe_float(r["confidence_score"], 0),
        })

    return {"signals": signals, "count": len(signals)}


@router.get("/company-signal-candidate-groups")
async def list_company_signal_candidate_groups(
    vendor_name: Optional[str] = Query(None),
    company_name: Optional[str] = Query(None),
    source_name: Optional[str] = Query(None),
    candidate_bucket: str = Query("analyst_review"),
    review_status: str = Query("pending"),
    canonical_gap_reason: Optional[str] = Query(None),
    review_priority_band: Optional[str] = Query(None),
    review_priority_reason: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    min_reviews: int = Query(1, ge=1, le=100),
    decision_makers_only: bool = Query(False),
    signal_evidence_present: Optional[bool] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    """List grouped company-signal candidates as the primary operator queue."""
    vendor_name = _optional_query_text(vendor_name)
    company_name = _optional_query_text(company_name)
    source_name = _optional_query_text(source_name)
    canonical_gap_reason = _optional_query_text(canonical_gap_reason)
    review_priority_band = _optional_query_text(review_priority_band)
    review_priority_reason = _optional_query_text(review_priority_reason)

    if candidate_bucket not in {"analyst_review", "canonical_ready"}:
        raise HTTPException(
            status_code=400,
            detail="candidate_bucket must be 'analyst_review' or 'canonical_ready'",
        )
    if review_status not in {"pending", "approved", "suppressed"}:
        raise HTTPException(
            status_code=400,
            detail="review_status must be 'pending', 'approved', or 'suppressed'",
        )
    if review_priority_band is not None and review_priority_band not in {"promote_now", "high", "medium", "low"}:
        raise HTTPException(
            status_code=400,
            detail="review_priority_band must be 'promote_now', 'high', 'medium', or 'low'",
        )

    pool = _pool_or_503()
    scoped_vendors = await _get_scoped_vendors(pool, user)
    groups = await read_company_signal_candidate_groups(
        pool,
        window_days=window_days,
        vendor_name=vendor_name,
        company_name=company_name,
        source_name=source_name,
        scoped_vendors=scoped_vendors,
        candidate_bucket=candidate_bucket,
        review_status=review_status,
        canonical_gap_reason=canonical_gap_reason,
        review_priority_band=review_priority_band,
        review_priority_reason=review_priority_reason,
        min_urgency=min_urgency,
        min_confidence=min_confidence,
        min_reviews=min_reviews,
        decision_makers_only=decision_makers_only,
        signal_evidence_present=signal_evidence_present,
        limit=limit,
    )
    return {
        "groups": groups,
        "count": len(groups),
        "candidate_bucket": candidate_bucket,
        "review_status": review_status,
        "review_priority_band": review_priority_band,
        "review_priority_reason": review_priority_reason,
        "source_name": source_name,
    }


@router.get("/company-signal-candidate-group-summary")
async def get_company_signal_candidate_group_summary(
    vendor_name: Optional[str] = Query(None),
    company_name: Optional[str] = Query(None),
    source_name: Optional[str] = Query(None),
    candidate_bucket: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    canonical_gap_reason: Optional[str] = Query(None),
    review_priority_band: Optional[str] = Query(None),
    review_priority_reason: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    min_reviews: int = Query(1, ge=1, le=100),
    decision_makers_only: bool = Query(False),
    signal_evidence_present: Optional[bool] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    top_n: int = Query(10, ge=1, le=25),
    user: AuthUser | None = Depends(optional_auth),
):
    """Summarize grouped company-signal queue health for analyst review."""
    vendor_name = _optional_query_text(vendor_name)
    company_name = _optional_query_text(company_name)
    source_name = _optional_query_text(source_name)
    candidate_bucket = _optional_query_text(candidate_bucket)
    review_status = _optional_query_text(review_status)
    canonical_gap_reason = _optional_query_text(canonical_gap_reason)
    review_priority_band = _optional_query_text(review_priority_band)
    review_priority_reason = _optional_query_text(review_priority_reason)

    if candidate_bucket is not None and candidate_bucket not in {"analyst_review", "canonical_ready"}:
        raise HTTPException(
            status_code=400,
            detail="candidate_bucket must be 'analyst_review' or 'canonical_ready'",
        )
    if review_status is not None and review_status not in {"pending", "approved", "suppressed"}:
        raise HTTPException(
            status_code=400,
            detail="review_status must be 'pending', 'approved', or 'suppressed'",
        )
    if review_priority_band is not None and review_priority_band not in {"promote_now", "high", "medium", "low"}:
        raise HTTPException(
            status_code=400,
            detail="review_priority_band must be 'promote_now', 'high', 'medium', or 'low'",
        )

    pool = _pool_or_503()
    scoped_vendors = await _get_scoped_vendors(pool, user)
    summary = await read_company_signal_candidate_group_summary(
        pool,
        window_days=window_days,
        vendor_name=vendor_name,
        company_name=company_name,
        source_name=source_name,
        scoped_vendors=scoped_vendors,
        candidate_bucket=candidate_bucket,
        review_status=review_status,
        canonical_gap_reason=canonical_gap_reason,
        review_priority_band=review_priority_band,
        review_priority_reason=review_priority_reason,
        min_urgency=min_urgency,
        min_confidence=min_confidence,
        min_reviews=min_reviews,
        decision_makers_only=decision_makers_only,
        signal_evidence_present=signal_evidence_present,
        top_n=top_n,
    )
    summary["candidate_bucket"] = candidate_bucket
    summary["review_status"] = review_status
    summary["review_priority_band"] = review_priority_band
    summary["review_priority_reason"] = review_priority_reason
    summary["source_name"] = source_name
    return summary


@router.get("/company-signal-review-impact-summary")
async def get_company_signal_review_impact_summary(
    vendor_name: Optional[str] = Query(None),
    review_scope: Optional[str] = Query(None),
    review_action: Optional[str] = Query(None),
    company_signal_action: Optional[str] = Query(None),
    canonical_gap_reason: Optional[str] = Query(None),
    review_priority_band: Optional[str] = Query(None),
    review_priority_reason: Optional[str] = Query(None),
    review_unlock_path: Optional[str] = Query(None),
    review_unlock_reason: Optional[str] = Query(None),
    candidate_source: Optional[str] = Query(None),
    rebuild_outcome: Optional[str] = Query(None),
    rebuild_reason: Optional[str] = Query(None),
    window_days: int = Query(30, ge=1, le=3650),
    top_n: int = Query(10, ge=1, le=25),
    user: AuthUser | None = Depends(optional_auth),
):
    """Summarize downstream yield from company-signal review actions."""
    vendor_name = _optional_query_text(vendor_name)
    review_scope = _optional_query_text(review_scope)
    review_action = _optional_query_text(review_action)
    company_signal_action = _optional_query_text(company_signal_action)
    canonical_gap_reason = _optional_query_text(canonical_gap_reason)
    review_priority_band = _optional_query_text(review_priority_band)
    review_priority_reason = _optional_query_text(review_priority_reason)
    review_unlock_path = _optional_query_text(review_unlock_path)
    review_unlock_reason = _optional_query_text(review_unlock_reason)
    candidate_source = _optional_query_text(candidate_source)
    rebuild_outcome = _optional_query_text(rebuild_outcome)
    rebuild_reason = _optional_query_text(rebuild_reason)

    if review_scope is not None and review_scope not in {"candidate", "group", "bulk_group"}:
        raise HTTPException(
            status_code=400,
            detail="review_scope must be 'candidate', 'group', or 'bulk_group'",
        )
    if review_action is not None and review_action not in {"approved", "suppressed"}:
        raise HTTPException(
            status_code=400,
            detail="review_action must be 'approved' or 'suppressed'",
        )
    if company_signal_action is not None and company_signal_action not in {"created", "updated", "deleted", "none"}:
        raise HTTPException(
            status_code=400,
            detail="company_signal_action must be 'created', 'updated', 'deleted', or 'none'",
        )
    if rebuild_outcome is not None and rebuild_outcome not in {"requested", "triggered", "blocked", "not_requested"}:
        raise HTTPException(
            status_code=400,
            detail="rebuild_outcome must be 'requested', 'triggered', 'blocked', or 'not_requested'",
        )
    if review_priority_band is not None and review_priority_band not in {"promote_now", "high", "medium", "low"}:
        raise HTTPException(
            status_code=400,
            detail="review_priority_band must be 'promote_now', 'high', 'medium', or 'low'",
        )

    pool = _pool_or_503()
    scoped_vendors = await _get_scoped_vendors(pool, user)
    summary = await read_company_signal_review_impact_summary(
        pool,
        window_days=window_days,
        vendor_name=vendor_name,
        scoped_vendors=scoped_vendors,
        review_scope=review_scope,
        review_action=review_action,
        company_signal_action=company_signal_action,
        canonical_gap_reason=canonical_gap_reason,
        review_priority_band=review_priority_band,
        review_priority_reason=review_priority_reason,
        review_unlock_path=review_unlock_path,
        review_unlock_reason=review_unlock_reason,
        candidate_source=candidate_source,
        rebuild_outcome=rebuild_outcome,
        rebuild_reason=rebuild_reason,
        top_n=top_n,
    )
    summary["review_scope"] = review_scope
    summary["review_action"] = review_action
    summary["company_signal_action"] = company_signal_action
    summary["canonical_gap_reason"] = canonical_gap_reason
    summary["review_priority_band"] = review_priority_band
    summary["review_priority_reason"] = review_priority_reason
    summary["review_unlock_path"] = review_unlock_path
    summary["review_unlock_reason"] = review_unlock_reason
    summary["candidate_source"] = candidate_source
    summary["rebuild_outcome"] = rebuild_outcome
    summary["rebuild_reason"] = rebuild_reason
    return summary


@router.get("/company-signal-candidates")
async def list_company_signal_candidates(
    vendor_name: Optional[str] = Query(None),
    company_name: Optional[str] = Query(None),
    candidate_bucket: str = Query("analyst_review"),
    review_status: str = Query("pending"),
    canonical_gap_reason: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    decision_makers_only: bool = Query(False),
    signal_evidence_present: Optional[bool] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    """List analyst-assist company-signal candidates without mixing them into canonical signals."""
    vendor_name = _optional_query_text(vendor_name)
    company_name = _optional_query_text(company_name)
    canonical_gap_reason = _optional_query_text(canonical_gap_reason)

    if candidate_bucket not in {"analyst_review", "canonical_ready"}:
        raise HTTPException(
            status_code=400,
            detail="candidate_bucket must be 'analyst_review' or 'canonical_ready'",
        )
    if review_status not in {"pending", "approved", "suppressed"}:
        raise HTTPException(
            status_code=400,
            detail="review_status must be 'pending', 'approved', or 'suppressed'",
        )

    pool = _pool_or_503()
    scoped_vendors = await _get_scoped_vendors(pool, user)
    candidates = await read_company_signal_candidates(
        pool,
        window_days=window_days,
        vendor_name=vendor_name,
        company_name=company_name,
        scoped_vendors=scoped_vendors,
        candidate_bucket=candidate_bucket,
        review_status=review_status,
        canonical_gap_reason=canonical_gap_reason,
        min_urgency=min_urgency,
        min_confidence=min_confidence,
        decision_makers_only=decision_makers_only,
        signal_evidence_present=signal_evidence_present,
        limit=limit,
    )
    return {
        "candidates": candidates,
        "count": len(candidates),
        "candidate_bucket": candidate_bucket,
        "review_status": review_status,
    }


@router.post("/company-signal-candidates/{candidate_id}/approve")
async def approve_company_signal_candidate(
    candidate_id: str,
    body: CompanySignalCandidateReviewBody,
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    candidate = await _fetch_company_signal_candidate_or_404(pool, candidate_id, user)
    reviewer = _candidate_reviewer(user)
    review_batch_id = str(_uuid.uuid4())
    review_priority_band, review_priority_reason = _company_signal_review_priority_snapshot(
        candidate,
        scope="candidate",
    )
    unlock_snapshot = _company_signal_review_unlock_snapshot(
        candidate,
        scope="candidate",
    )

    canonical_row = await _upsert_company_signal(
        pool,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
        urgency_score=candidate.get("urgency_score"),
        pain_category=candidate.get("pain_category"),
        buyer_role=candidate.get("buyer_role"),
        decision_maker=candidate.get("decision_maker"),
        seat_count=candidate.get("seat_count"),
        contract_end=candidate.get("contract_end"),
        buying_stage=candidate.get("buying_stage"),
        review_id=candidate.get("review_id"),
        source=candidate.get("source"),
        confidence_score=candidate.get("confidence_score"),
    )

    reviewed = await pool.fetchrow(
        """
        UPDATE b2b_company_signal_candidates
        SET review_status = 'approved',
            review_status_updated_at = NOW(),
            reviewed_by = $2,
            review_notes = $3
        WHERE id = $1::uuid
        RETURNING id, review_status, review_status_updated_at, reviewed_by, review_notes
        """,
        candidate_id,
        reviewer,
        body.notes,
    )
    await _set_company_signal_candidate_group_status(
        pool,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
        status="approved",
        reviewed_by=reviewer,
        review_notes=body.notes,
    )
    await _set_company_signal_candidate_member_status(
        pool,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
        status="approved",
        reviewed_by=reviewer,
        review_notes=body.notes,
    )

    rebuild = {"triggered": False, "reason": "disabled"}
    if body.trigger_rebuild:
        try:
            rebuild = await _trigger_accounts_in_motion_rebuild(pool, candidate["vendor_name"])
        except Exception:
            logger.exception(
                "Company signal candidate approval rebuild failed for %s",
                candidate["vendor_name"],
            )
            rebuild = {"triggered": False, "reason": "rebuild_failed"}

    await _record_company_signal_review_event(
        pool,
        review_batch_id=review_batch_id,
        review_scope="candidate",
        review_action="approved",
        candidate_id=candidate_id,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
        reviewer=reviewer,
        review_notes=body.notes,
        rebuild_requested=body.trigger_rebuild,
        rebuild=rebuild,
        review_priority_band=review_priority_band,
        review_priority_reason=review_priority_reason,
        candidate_source=unlock_snapshot.get("candidate_source"),
        canonical_gap_reason=unlock_snapshot.get("canonical_gap_reason"),
        review_unlock_path=unlock_snapshot.get("review_unlock_path"),
        review_unlock_reason=unlock_snapshot.get("review_unlock_reason"),
        company_signal_id=str(canonical_row["id"]) if canonical_row and canonical_row.get("id") else None,
        company_signal_action=(
            canonical_row.get("company_signal_action")
            if canonical_row
            else "none"
        ),
    )
    await _dispatch_company_signal_update_webhooks(
        pool,
        [
            {
                "vendor_name": candidate["vendor_name"],
                "company_name": canonical_row["company_name"] if canonical_row else candidate["company_name"],
                "company_signal_id": (
                    str(canonical_row["id"])
                    if canonical_row and canonical_row.get("id")
                    else None
                ),
                "company_signal_action": (
                    canonical_row.get("company_signal_action")
                    if canonical_row
                    else "none"
                ),
                "review_id": (
                    (canonical_row.get("review_id") if canonical_row else None)
                    or candidate.get("review_id")
                ),
                "review_scope": "candidate",
                "review_action": "approved",
                "candidate_source": unlock_snapshot.get("candidate_source"),
            }
        ],
    )

    return {
        "review_batch_id": review_batch_id,
        "candidate_id": str(reviewed["id"]),
        "review_status": reviewed["review_status"],
        "review_status_updated_at": (
            reviewed["review_status_updated_at"].isoformat()
            if reviewed["review_status_updated_at"]
            else None
        ),
        "reviewed_by": reviewed["reviewed_by"],
        "review_notes": reviewed["review_notes"],
        "company_signal_id": str(canonical_row["id"]) if canonical_row else None,
        "company_signal_action": (
            canonical_row.get("company_signal_action")
            if canonical_row
            else "none"
        ),
        "company_name": canonical_row["company_name"] if canonical_row else candidate["company_name"],
        "vendor_name": canonical_row["vendor_name"] if canonical_row else candidate["vendor_name"],
        "review_priority_band": review_priority_band,
        "review_priority_reason": review_priority_reason,
        "candidate_source": unlock_snapshot.get("candidate_source"),
        "canonical_gap_reason": unlock_snapshot.get("canonical_gap_reason"),
        "review_unlock_path": unlock_snapshot.get("review_unlock_path"),
        "review_unlock_reason": unlock_snapshot.get("review_unlock_reason"),
        "rebuild": rebuild,
    }


@router.post("/company-signal-candidates/{candidate_id}/suppress")
async def suppress_company_signal_candidate(
    candidate_id: str,
    body: CompanySignalCandidateReviewBody,
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    candidate = await _fetch_company_signal_candidate_or_404(pool, candidate_id, user)
    reviewer = _candidate_reviewer(user)
    review_batch_id = str(_uuid.uuid4())
    review_priority_band, review_priority_reason = _company_signal_review_priority_snapshot(
        candidate,
        scope="candidate",
    )
    unlock_snapshot = _company_signal_review_unlock_snapshot(
        candidate,
        scope="candidate",
    )
    deleted_signal = await _delete_company_signal(
        pool,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
    )
    reviewed = await pool.fetchrow(
        """
        UPDATE b2b_company_signal_candidates
        SET review_status = 'suppressed',
            review_status_updated_at = NOW(),
            reviewed_by = $2,
            review_notes = $3
        WHERE id = $1::uuid
        RETURNING id, review_status, review_status_updated_at, reviewed_by, review_notes
        """,
        candidate_id,
        reviewer,
        body.notes,
    )
    await _set_company_signal_candidate_group_status(
        pool,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
        status="suppressed",
        reviewed_by=reviewer,
        review_notes=body.notes,
    )
    await _set_company_signal_candidate_member_status(
        pool,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
        status="suppressed",
        reviewed_by=reviewer,
        review_notes=body.notes,
    )
    rebuilds = await _trigger_accounts_in_motion_rebuilds(
        pool,
        [candidate["vendor_name"]],
        enabled=body.trigger_rebuild,
    )
    rebuild = rebuilds[0] if rebuilds else {"triggered": False, "reason": "disabled"}
    await _record_company_signal_review_event(
        pool,
        review_batch_id=review_batch_id,
        review_scope="candidate",
        review_action="suppressed",
        candidate_id=candidate_id,
        company_name=candidate["company_name"],
        vendor_name=candidate["vendor_name"],
        reviewer=reviewer,
        review_notes=body.notes,
        rebuild_requested=body.trigger_rebuild,
        rebuild=rebuild,
        review_priority_band=review_priority_band,
        review_priority_reason=review_priority_reason,
        candidate_source=unlock_snapshot.get("candidate_source"),
        canonical_gap_reason=unlock_snapshot.get("canonical_gap_reason"),
        review_unlock_path=unlock_snapshot.get("review_unlock_path"),
        review_unlock_reason=unlock_snapshot.get("review_unlock_reason"),
        company_signal_id=(
            str(deleted_signal["id"])
            if deleted_signal and deleted_signal.get("id")
            else None
        ),
        company_signal_action=(
            deleted_signal.get("company_signal_action")
            if deleted_signal
            else "none"
        ),
    )
    await _dispatch_company_signal_update_webhooks(
        pool,
        [
            {
                "vendor_name": candidate["vendor_name"],
                "company_name": candidate["company_name"],
                "company_signal_id": (
                    str(deleted_signal["id"])
                    if deleted_signal and deleted_signal.get("id")
                    else None
                ),
                "company_signal_action": (
                    deleted_signal.get("company_signal_action")
                    if deleted_signal
                    else "none"
                ),
                "review_id": (
                    (deleted_signal.get("review_id") if deleted_signal else None)
                    or candidate.get("review_id")
                ),
                "review_scope": "candidate",
                "review_action": "suppressed",
                "candidate_source": unlock_snapshot.get("candidate_source"),
            }
        ],
    )
    return {
        "review_batch_id": review_batch_id,
        "candidate_id": str(reviewed["id"]),
        "review_status": reviewed["review_status"],
        "review_status_updated_at": (
            reviewed["review_status_updated_at"].isoformat()
            if reviewed["review_status_updated_at"]
            else None
        ),
        "reviewed_by": reviewed["reviewed_by"],
        "review_notes": reviewed["review_notes"],
        "retracted_company_signal_id": (
            str(deleted_signal["id"])
            if deleted_signal and deleted_signal.get("id")
            else None
        ),
        "company_name": candidate["company_name"],
        "vendor_name": candidate["vendor_name"],
        "company_signal_action": (
            deleted_signal.get("company_signal_action")
            if deleted_signal
            else "none"
        ),
        "review_priority_band": review_priority_band,
        "review_priority_reason": review_priority_reason,
        "candidate_source": unlock_snapshot.get("candidate_source"),
        "canonical_gap_reason": unlock_snapshot.get("canonical_gap_reason"),
        "review_unlock_path": unlock_snapshot.get("review_unlock_path"),
        "review_unlock_reason": unlock_snapshot.get("review_unlock_reason"),
        "rebuild": rebuild,
    }


@router.post("/company-signal-candidate-groups/{group_id}/approve")
async def approve_company_signal_candidate_group(
    group_id: str,
    body: CompanySignalCandidateReviewBody,
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    group = await _fetch_company_signal_candidate_group_or_404(pool, group_id, user)
    reviewer = _candidate_reviewer(user)
    review_batch_id = str(_uuid.uuid4())
    review_priority_band, review_priority_reason = _company_signal_review_priority_snapshot(
        group,
        scope="group",
    )
    unlock_snapshot = _company_signal_review_unlock_snapshot(
        group,
        scope="group",
    )

    canonical_row = await _upsert_company_signal(
        pool,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
        urgency_score=group.get("max_urgency_score"),
        pain_category=group.get("representative_pain_category"),
        buyer_role=group.get("representative_buyer_role"),
        decision_maker=group.get("representative_decision_maker"),
        seat_count=group.get("representative_seat_count"),
        contract_end=group.get("representative_contract_end"),
        buying_stage=group.get("representative_buying_stage"),
        review_id=group.get("representative_review_id"),
        source=group.get("representative_source"),
        confidence_score=group.get("corroborated_confidence_score"),
    )
    reviewed = await _set_company_signal_candidate_group_status(
        pool,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
        status="approved",
        reviewed_by=reviewer,
        review_notes=body.notes,
        group_id=group_id,
    )
    await _set_company_signal_candidate_member_status(
        pool,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
        status="approved",
        reviewed_by=reviewer,
        review_notes=body.notes,
    )

    rebuild = {"triggered": False, "reason": "disabled"}
    if body.trigger_rebuild:
        try:
            rebuild = await _trigger_accounts_in_motion_rebuild(pool, group["vendor_name"])
        except Exception:
            logger.exception(
                "Company signal candidate group approval rebuild failed for %s",
                group["vendor_name"],
            )
            rebuild = {"triggered": False, "reason": "rebuild_failed"}

    await _record_company_signal_review_event(
        pool,
        review_batch_id=review_batch_id,
        review_scope="group",
        review_action="approved",
        candidate_group_id=group_id,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
        reviewer=reviewer,
        review_notes=body.notes,
        rebuild_requested=body.trigger_rebuild,
        rebuild=rebuild,
        review_priority_band=review_priority_band,
        review_priority_reason=review_priority_reason,
        candidate_source=unlock_snapshot.get("candidate_source"),
        canonical_gap_reason=unlock_snapshot.get("canonical_gap_reason"),
        review_unlock_path=unlock_snapshot.get("review_unlock_path"),
        review_unlock_reason=unlock_snapshot.get("review_unlock_reason"),
        company_signal_id=str(canonical_row["id"]) if canonical_row and canonical_row.get("id") else None,
        company_signal_action=(
            canonical_row.get("company_signal_action")
            if canonical_row
            else "none"
        ),
    )
    await _dispatch_company_signal_update_webhooks(
        pool,
        [
            {
                "vendor_name": group["vendor_name"],
                "company_name": canonical_row["company_name"] if canonical_row else group["company_name"],
                "company_signal_id": (
                    str(canonical_row["id"])
                    if canonical_row and canonical_row.get("id")
                    else None
                ),
                "company_signal_action": (
                    canonical_row.get("company_signal_action")
                    if canonical_row
                    else "none"
                ),
                "review_id": (
                    (canonical_row.get("review_id") if canonical_row else None)
                    or group.get("representative_review_id")
                ),
                "review_scope": "group",
                "review_action": "approved",
                "candidate_source": unlock_snapshot.get("candidate_source"),
            }
        ],
    )

    return {
        "review_batch_id": review_batch_id,
        "group_id": str(reviewed["id"]) if reviewed else group_id,
        "review_status": reviewed["review_status"] if reviewed else "approved",
        "review_status_updated_at": (
            reviewed["review_status_updated_at"].isoformat()
            if reviewed and reviewed["review_status_updated_at"]
            else None
        ),
        "reviewed_by": reviewed["reviewed_by"] if reviewed else reviewer,
        "review_notes": reviewed["review_notes"] if reviewed else body.notes,
        "company_signal_id": str(canonical_row["id"]) if canonical_row else None,
        "company_signal_action": (
            canonical_row.get("company_signal_action")
            if canonical_row
            else "none"
        ),
        "company_name": canonical_row["company_name"] if canonical_row else group["company_name"],
        "vendor_name": canonical_row["vendor_name"] if canonical_row else group["vendor_name"],
        "review_count": group.get("review_count") or 0,
        "review_priority_band": review_priority_band,
        "review_priority_reason": review_priority_reason,
        "candidate_source": unlock_snapshot.get("candidate_source"),
        "canonical_gap_reason": unlock_snapshot.get("canonical_gap_reason"),
        "review_unlock_path": unlock_snapshot.get("review_unlock_path"),
        "review_unlock_reason": unlock_snapshot.get("review_unlock_reason"),
        "rebuild": rebuild,
    }


@router.post("/company-signal-candidate-groups/{group_id}/suppress")
async def suppress_company_signal_candidate_group(
    group_id: str,
    body: CompanySignalCandidateReviewBody,
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    group = await _fetch_company_signal_candidate_group_or_404(pool, group_id, user)
    reviewer = _candidate_reviewer(user)
    review_batch_id = str(_uuid.uuid4())
    review_priority_band, review_priority_reason = _company_signal_review_priority_snapshot(
        group,
        scope="group",
    )
    unlock_snapshot = _company_signal_review_unlock_snapshot(
        group,
        scope="group",
    )
    deleted_signal = await _delete_company_signal(
        pool,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
    )
    reviewed = await _set_company_signal_candidate_group_status(
        pool,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
        status="suppressed",
        reviewed_by=reviewer,
        review_notes=body.notes,
        group_id=group_id,
    )
    await _set_company_signal_candidate_member_status(
        pool,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
        status="suppressed",
        reviewed_by=reviewer,
        review_notes=body.notes,
    )
    rebuilds = await _trigger_accounts_in_motion_rebuilds(
        pool,
        [group["vendor_name"]],
        enabled=body.trigger_rebuild,
    )
    rebuild = rebuilds[0] if rebuilds else {"triggered": False, "reason": "disabled"}
    await _record_company_signal_review_event(
        pool,
        review_batch_id=review_batch_id,
        review_scope="group",
        review_action="suppressed",
        candidate_group_id=group_id,
        company_name=group["company_name"],
        vendor_name=group["vendor_name"],
        reviewer=reviewer,
        review_notes=body.notes,
        rebuild_requested=body.trigger_rebuild,
        rebuild=rebuild,
        review_priority_band=review_priority_band,
        review_priority_reason=review_priority_reason,
        candidate_source=unlock_snapshot.get("candidate_source"),
        canonical_gap_reason=unlock_snapshot.get("canonical_gap_reason"),
        review_unlock_path=unlock_snapshot.get("review_unlock_path"),
        review_unlock_reason=unlock_snapshot.get("review_unlock_reason"),
        company_signal_id=(
            str(deleted_signal["id"])
            if deleted_signal and deleted_signal.get("id")
            else None
        ),
        company_signal_action=(
            deleted_signal.get("company_signal_action")
            if deleted_signal
            else "none"
        ),
    )
    await _dispatch_company_signal_update_webhooks(
        pool,
        [
            {
                "vendor_name": group["vendor_name"],
                "company_name": group["company_name"],
                "company_signal_id": (
                    str(deleted_signal["id"])
                    if deleted_signal and deleted_signal.get("id")
                    else None
                ),
                "company_signal_action": (
                    deleted_signal.get("company_signal_action")
                    if deleted_signal
                    else "none"
                ),
                "review_id": (
                    (deleted_signal.get("review_id") if deleted_signal else None)
                    or group.get("representative_review_id")
                ),
                "review_scope": "group",
                "review_action": "suppressed",
                "candidate_source": unlock_snapshot.get("candidate_source"),
            }
        ],
    )
    return {
        "review_batch_id": review_batch_id,
        "group_id": str(reviewed["id"]) if reviewed else group_id,
        "review_status": reviewed["review_status"] if reviewed else "suppressed",
        "review_status_updated_at": (
            reviewed["review_status_updated_at"].isoformat()
            if reviewed and reviewed["review_status_updated_at"]
            else None
        ),
        "reviewed_by": reviewed["reviewed_by"] if reviewed else reviewer,
        "review_notes": reviewed["review_notes"] if reviewed else body.notes,
        "company_name": group["company_name"],
        "vendor_name": group["vendor_name"],
        "review_count": group.get("review_count") or 0,
        "retracted_company_signal_id": (
            str(deleted_signal["id"])
            if deleted_signal and deleted_signal.get("id")
            else None
        ),
        "company_signal_action": (
            deleted_signal.get("company_signal_action")
            if deleted_signal
            else "none"
        ),
        "review_priority_band": review_priority_band,
        "review_priority_reason": review_priority_reason,
        "candidate_source": unlock_snapshot.get("candidate_source"),
        "canonical_gap_reason": unlock_snapshot.get("canonical_gap_reason"),
        "review_unlock_path": unlock_snapshot.get("review_unlock_path"),
        "review_unlock_reason": unlock_snapshot.get("review_unlock_reason"),
        "rebuild": rebuild,
    }


@router.post("/company-signal-candidate-groups/approve")
async def approve_company_signal_candidate_groups(
    body: BulkCompanySignalCandidateGroupReviewBody,
    user: AuthUser | None = Depends(optional_auth),
):
    reviewer = _candidate_reviewer(user)
    review_batch_id = str(_uuid.uuid4())
    group_ids = _unique_strings_preserving_order(body.group_ids)
    if not group_ids:
        raise HTTPException(status_code=400, detail="group_ids must include at least one non-empty UUID")
    pool = _pool_or_503()
    groups: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    webhook_updates: list[dict[str, Any]] = []

    async with pool.transaction() as conn:
        for group_id in group_ids:
            groups.append(await _fetch_company_signal_candidate_group_or_404(conn, group_id, user))
        for group_id, group in zip(group_ids, groups):
            unlock_snapshot = _company_signal_review_unlock_snapshot(
                group,
                scope="group",
            )
            canonical_row = await _upsert_company_signal(
                conn,
                company_name=group["company_name"],
                vendor_name=group["vendor_name"],
                urgency_score=group.get("max_urgency_score"),
                pain_category=group.get("representative_pain_category"),
                buyer_role=group.get("representative_buyer_role"),
                decision_maker=group.get("representative_decision_maker"),
                seat_count=group.get("representative_seat_count"),
                contract_end=group.get("representative_contract_end"),
                buying_stage=group.get("representative_buying_stage"),
                review_id=group.get("representative_review_id"),
                source=group.get("representative_source"),
                confidence_score=group.get("corroborated_confidence_score"),
            )
            reviewed = await _set_company_signal_candidate_group_status(
                conn,
                company_name=group["company_name"],
                vendor_name=group["vendor_name"],
                status="approved",
                reviewed_by=reviewer,
                review_notes=body.notes,
                group_id=group_id,
            )
            await _set_company_signal_candidate_member_status(
                conn,
                company_name=group["company_name"],
                vendor_name=group["vendor_name"],
                status="approved",
                reviewed_by=reviewer,
                review_notes=body.notes,
            )
            results.append(
                {
                    "group_id": str(reviewed["id"]) if reviewed else group_id,
                    "review_status": reviewed["review_status"] if reviewed else "approved",
                    "review_status_updated_at": (
                        reviewed["review_status_updated_at"].isoformat()
                        if reviewed and reviewed["review_status_updated_at"]
                        else None
                    ),
                    "reviewed_by": reviewed["reviewed_by"] if reviewed else reviewer,
                    "review_notes": reviewed["review_notes"] if reviewed else body.notes,
                    "company_signal_id": str(canonical_row["id"]) if canonical_row else None,
                    "company_signal_action": (
                        canonical_row.get("company_signal_action")
                        if canonical_row
                        else "none"
                    ),
                    "company_name": canonical_row["company_name"] if canonical_row else group["company_name"],
                    "vendor_name": canonical_row["vendor_name"] if canonical_row else group["vendor_name"],
                    "review_count": group.get("review_count") or 0,
                    "review_priority_band": _company_signal_group_review_priority_values(group)[0],
                    "review_priority_reason": _company_signal_group_review_priority_values(group)[1],
                    "candidate_source": unlock_snapshot.get("candidate_source"),
                    "canonical_gap_reason": unlock_snapshot.get("canonical_gap_reason"),
                    "review_unlock_path": unlock_snapshot.get("review_unlock_path"),
                    "review_unlock_reason": unlock_snapshot.get("review_unlock_reason"),
                }
            )
            webhook_updates.append(
                {
                    "vendor_name": group["vendor_name"],
                    "company_name": canonical_row["company_name"] if canonical_row else group["company_name"],
                    "company_signal_id": str(canonical_row["id"]) if canonical_row else None,
                    "company_signal_action": (
                        canonical_row.get("company_signal_action")
                        if canonical_row
                        else "none"
                    ),
                    "review_id": (
                        (canonical_row.get("review_id") if canonical_row else None)
                        or group.get("representative_review_id")
                    ),
                    "review_scope": "bulk_group",
                    "review_action": "approved",
                    "candidate_source": unlock_snapshot.get("candidate_source"),
                }
            )

    rebuilds = await _trigger_accounts_in_motion_rebuilds(
        pool,
        [group["vendor_name"] for group in groups],
        enabled=body.trigger_rebuild,
    )
    rebuilds_by_vendor = {
        _normalize_vendor_name(item.get("vendor_name")): item
        for item in rebuilds
        if _normalize_vendor_name(item.get("vendor_name"))
    }
    for result in results:
        rebuild = rebuilds_by_vendor.get(_normalize_vendor_name(result.get("vendor_name")))
        await _record_company_signal_review_event(
            pool,
            review_batch_id=review_batch_id,
            review_scope="bulk_group",
            review_action="approved",
            candidate_group_id=result.get("group_id"),
            company_name=result.get("company_name") or "",
            vendor_name=result.get("vendor_name") or "",
            reviewer=reviewer,
            review_notes=body.notes,
            rebuild_requested=body.trigger_rebuild,
            rebuild=rebuild,
            review_priority_band=result.get("review_priority_band"),
            review_priority_reason=result.get("review_priority_reason"),
            candidate_source=result.get("candidate_source"),
            canonical_gap_reason=result.get("canonical_gap_reason"),
            review_unlock_path=result.get("review_unlock_path"),
            review_unlock_reason=result.get("review_unlock_reason"),
            company_signal_id=result.get("company_signal_id"),
            company_signal_action=result.get("company_signal_action") or "none",
        )
    await _dispatch_company_signal_update_webhooks(pool, webhook_updates)
    return {
        "review_batch_id": review_batch_id,
        "groups": results,
        "count": len(results),
        "rebuilds": rebuilds,
    }


@router.post("/company-signal-candidate-groups/suppress")
async def suppress_company_signal_candidate_groups(
    body: BulkCompanySignalCandidateGroupReviewBody,
    user: AuthUser | None = Depends(optional_auth),
):
    reviewer = _candidate_reviewer(user)
    review_batch_id = str(_uuid.uuid4())
    group_ids = _unique_strings_preserving_order(body.group_ids)
    if not group_ids:
        raise HTTPException(status_code=400, detail="group_ids must include at least one non-empty UUID")
    pool = _pool_or_503()
    groups: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    webhook_updates: list[dict[str, Any]] = []

    async with pool.transaction() as conn:
        for group_id in group_ids:
            groups.append(await _fetch_company_signal_candidate_group_or_404(conn, group_id, user))
        for group_id, group in zip(group_ids, groups):
            unlock_snapshot = _company_signal_review_unlock_snapshot(
                group,
                scope="group",
            )
            deleted_signal = await _delete_company_signal(
                conn,
                company_name=group["company_name"],
                vendor_name=group["vendor_name"],
            )
            reviewed = await _set_company_signal_candidate_group_status(
                conn,
                company_name=group["company_name"],
                vendor_name=group["vendor_name"],
                status="suppressed",
                reviewed_by=reviewer,
                review_notes=body.notes,
                group_id=group_id,
            )
            await _set_company_signal_candidate_member_status(
                conn,
                company_name=group["company_name"],
                vendor_name=group["vendor_name"],
                status="suppressed",
                reviewed_by=reviewer,
                review_notes=body.notes,
            )
            results.append(
                {
                    "group_id": str(reviewed["id"]) if reviewed else group_id,
                    "review_status": reviewed["review_status"] if reviewed else "suppressed",
                    "review_status_updated_at": (
                        reviewed["review_status_updated_at"].isoformat()
                        if reviewed and reviewed["review_status_updated_at"]
                        else None
                    ),
                    "reviewed_by": reviewed["reviewed_by"] if reviewed else reviewer,
                    "review_notes": reviewed["review_notes"] if reviewed else body.notes,
                    "company_name": group["company_name"],
                    "vendor_name": group["vendor_name"],
                    "review_count": group.get("review_count") or 0,
                    "retracted_company_signal_id": (
                        str(deleted_signal["id"])
                        if deleted_signal and deleted_signal.get("id")
                        else None
                    ),
                    "company_signal_action": (
                        deleted_signal.get("company_signal_action")
                        if deleted_signal
                        else "none"
                    ),
                    "review_priority_band": _company_signal_group_review_priority_values(group)[0],
                    "review_priority_reason": _company_signal_group_review_priority_values(group)[1],
                    "candidate_source": unlock_snapshot.get("candidate_source"),
                    "canonical_gap_reason": unlock_snapshot.get("canonical_gap_reason"),
                    "review_unlock_path": unlock_snapshot.get("review_unlock_path"),
                    "review_unlock_reason": unlock_snapshot.get("review_unlock_reason"),
                }
            )
            webhook_updates.append(
                {
                    "vendor_name": group["vendor_name"],
                    "company_name": group["company_name"],
                    "company_signal_id": (
                        str(deleted_signal["id"])
                        if deleted_signal and deleted_signal.get("id")
                        else None
                    ),
                    "company_signal_action": (
                        deleted_signal.get("company_signal_action")
                        if deleted_signal
                        else "none"
                    ),
                    "review_id": (
                        (deleted_signal.get("review_id") if deleted_signal else None)
                        or group.get("representative_review_id")
                    ),
                    "review_scope": "bulk_group",
                    "review_action": "suppressed",
                    "candidate_source": unlock_snapshot.get("candidate_source"),
                }
            )

    rebuilds = await _trigger_accounts_in_motion_rebuilds(
        pool,
        [group["vendor_name"] for group in groups],
        enabled=body.trigger_rebuild,
    )
    rebuilds_by_vendor = {
        _normalize_vendor_name(item.get("vendor_name")): item
        for item in rebuilds
        if _normalize_vendor_name(item.get("vendor_name"))
    }
    for result in results:
        rebuild = rebuilds_by_vendor.get(_normalize_vendor_name(result.get("vendor_name")))
        await _record_company_signal_review_event(
            pool,
            review_batch_id=review_batch_id,
            review_scope="bulk_group",
            review_action="suppressed",
            candidate_group_id=result.get("group_id"),
            company_name=result.get("company_name") or "",
            vendor_name=result.get("vendor_name") or "",
            reviewer=reviewer,
            review_notes=body.notes,
            rebuild_requested=body.trigger_rebuild,
            rebuild=rebuild,
            review_priority_band=result.get("review_priority_band"),
            review_priority_reason=result.get("review_priority_reason"),
            candidate_source=result.get("candidate_source"),
            canonical_gap_reason=result.get("canonical_gap_reason"),
            review_unlock_path=result.get("review_unlock_path"),
            review_unlock_reason=result.get("review_unlock_reason"),
            company_signal_id=result.get("retracted_company_signal_id"),
            company_signal_action=result.get("company_signal_action") or "none",
        )
    await _dispatch_company_signal_update_webhooks(pool, webhook_updates)
    return {
        "review_batch_id": review_batch_id,
        "groups": results,
        "count": len(results),
        "rebuilds": rebuilds,
    }


# ---------------------------------------------------------------------------
# GET /vendor-pain-points
# ---------------------------------------------------------------------------


@router.get("/vendor-pain-points")
async def list_vendor_pain_points(
    vendor_name: Optional[str] = Query(None),
    pain_category: Optional[str] = Query(None),
    min_confidence: float = Query(0, ge=0, le=1),
    min_mentions: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)
    pain_category = _optional_query_text(pain_category)
    conditions: list[str] = []
    params: list = []
    idx = 1

    if _should_scope(user):
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if pain_category:
        conditions.append(f"pain_category = ${idx}")
        params.append(pain_category)
        idx += 1

    if min_confidence > 0:
        conditions.append(f"confidence_score >= ${idx}")
        params.append(min_confidence)
        idx += 1

    if min_mentions > 0:
        conditions.append(f"mention_count >= ${idx}")
        params.append(min_mentions)
        idx += 1

    conditions.append(_suppress_predicate('pain_point'))
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, pain_category, mention_count,
               primary_count, secondary_count, minor_count,
               avg_urgency, avg_rating,
               source_distribution, sample_review_ids,
               confidence_score, first_seen_at, last_seen_at
        FROM b2b_vendor_pain_points
        {where}
        ORDER BY mention_count DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    items = []
    for r in rows:
        items.append({
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "pain_category": r["pain_category"],
            "mention_count": r["mention_count"],
            "primary_count": r["primary_count"],
            "secondary_count": r["secondary_count"],
            "minor_count": r["minor_count"],
            "avg_urgency": _safe_float(r["avg_urgency"], 0),
            "avg_rating": _safe_float(r["avg_rating"], 0),
            "source_distribution": _safe_json(r["source_distribution"]),
            "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
            "confidence_score": _safe_float(r["confidence_score"], 0),
            "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
            "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
        })

    return {"pain_points": items, "count": len(items)}


# ---------------------------------------------------------------------------
# GET /vendor-use-cases
# ---------------------------------------------------------------------------


@router.get("/vendor-use-cases")
async def list_vendor_use_cases(
    vendor_name: Optional[str] = Query(None),
    use_case_name: Optional[str] = Query(None),
    min_confidence: float = Query(0, ge=0, le=1),
    min_mentions: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)
    use_case_name = _optional_query_text(use_case_name)
    conditions: list[str] = []
    params: list = []
    idx = 1

    if _should_scope(user):
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if use_case_name:
        conditions.append(f"use_case_name ILIKE '%' || ${idx} || '%'")
        params.append(use_case_name)
        idx += 1

    if min_confidence > 0:
        conditions.append(f"confidence_score >= ${idx}")
        params.append(min_confidence)
        idx += 1

    if min_mentions > 0:
        conditions.append(f"mention_count >= ${idx}")
        params.append(min_mentions)
        idx += 1

    conditions.append(_suppress_predicate('use_case'))
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, use_case_name, mention_count,
               avg_urgency, lock_in_distribution,
               source_distribution, sample_review_ids,
               confidence_score, first_seen_at, last_seen_at
        FROM b2b_vendor_use_cases
        {where}
        ORDER BY mention_count DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    items = []
    for r in rows:
        items.append({
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "use_case_name": r["use_case_name"],
            "mention_count": r["mention_count"],
            "avg_urgency": _safe_float(r["avg_urgency"], 0),
            "lock_in_distribution": _safe_json(r["lock_in_distribution"]),
            "source_distribution": _safe_json(r["source_distribution"]),
            "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
            "confidence_score": _safe_float(r["confidence_score"], 0),
            "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
            "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
        })

    return {"use_cases": items, "count": len(items)}


# ---------------------------------------------------------------------------
# GET /vendor-integrations
# ---------------------------------------------------------------------------


@router.get("/vendor-integrations")
async def list_vendor_integrations(
    vendor_name: Optional[str] = Query(None),
    integration_name: Optional[str] = Query(None),
    min_confidence: float = Query(0, ge=0, le=1),
    min_mentions: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)
    integration_name = _optional_query_text(integration_name)
    conditions: list[str] = []
    params: list = []
    idx = 1

    if _should_scope(user):
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if integration_name:
        conditions.append(f"integration_name ILIKE '%' || ${idx} || '%'")
        params.append(integration_name)
        idx += 1

    if min_confidence > 0:
        conditions.append(f"confidence_score >= ${idx}")
        params.append(min_confidence)
        idx += 1

    if min_mentions > 0:
        conditions.append(f"mention_count >= ${idx}")
        params.append(min_mentions)
        idx += 1

    conditions.append(_suppress_predicate('integration'))
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, integration_name, mention_count,
               source_distribution, sample_review_ids,
               confidence_score, first_seen_at, last_seen_at
        FROM b2b_vendor_integrations
        {where}
        ORDER BY mention_count DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    items = []
    for r in rows:
        items.append({
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "integration_name": r["integration_name"],
            "mention_count": r["mention_count"],
            "source_distribution": _safe_json(r["source_distribution"]),
            "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
            "confidence_score": _safe_float(r["confidence_score"], 0),
            "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
            "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
        })

    return {"integrations": items, "count": len(items)}


# ---------------------------------------------------------------------------
# GET /vendor-buyer-profiles
# ---------------------------------------------------------------------------


@router.get("/vendor-buyer-profiles")
async def list_vendor_buyer_profiles(
    vendor_name: Optional[str] = Query(None),
    role_type: Optional[str] = Query(None),
    buying_stage: Optional[str] = Query(None),
    min_confidence: float = Query(0, ge=0, le=1),
    min_reviews: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)
    role_type = _optional_query_text(role_type)
    buying_stage = _optional_query_text(buying_stage)
    conditions: list[str] = []
    params: list = []
    idx = 1

    if _should_scope(user):
        conditions.append(f"vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)")
        params.append(user.account_id)
        idx += 1

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if role_type:
        conditions.append(f"role_type = ${idx}")
        params.append(role_type)
        idx += 1

    if buying_stage:
        conditions.append(f"buying_stage = ${idx}")
        params.append(buying_stage)
        idx += 1

    if min_confidence > 0:
        conditions.append(f"confidence_score >= ${idx}")
        params.append(min_confidence)
        idx += 1

    if min_reviews > 0:
        conditions.append(f"review_count >= ${idx}")
        params.append(min_reviews)
        idx += 1

    conditions.append(_suppress_predicate('buyer_profile'))
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, role_type, buying_stage,
               review_count, dm_count, avg_urgency,
               source_distribution, sample_review_ids,
               confidence_score, first_seen_at, last_seen_at
        FROM b2b_vendor_buyer_profiles
        {where}
        ORDER BY review_count DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )

    items = []
    for r in rows:
        items.append({
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "role_type": r["role_type"],
            "buying_stage": r["buying_stage"],
            "review_count": r["review_count"],
            "dm_count": r["dm_count"],
            "avg_urgency": _safe_float(r["avg_urgency"], 0),
            "source_distribution": _safe_json(r["source_distribution"]),
            "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
            "confidence_score": _safe_float(r["confidence_score"], 0),
            "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
            "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
        })

    return {"profiles": items, "count": len(items)}


# ---------------------------------------------------------------------------
# GET /vendor-history
# ---------------------------------------------------------------------------


@router.get("/vendor-history")
async def get_vendor_history(
    vendor_name: str = Query(...),
    days: int = Query(90, ge=1, le=365),
    limit: int = Query(90, ge=1, le=365),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _required_query_text(vendor_name, "vendor_name")
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
               churn_density, avg_urgency, positive_review_pct, recommend_ratio,
               support_sentiment, legacy_support_score,
               new_feature_velocity, employee_growth_rate,
               top_pain, top_competitor, pain_count, competitor_count,
               displacement_edge_count, high_intent_company_count
        FROM b2b_vendor_snapshots
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND snapshot_date >= CURRENT_DATE - $2::int
        ORDER BY snapshot_date DESC
        LIMIT $3
        """,
        vendor_name, days, limit,
    )
    resolved = rows[0]["vendor_name"] if rows else vendor_name
    snapshots = []
    for r in rows:
        snapshots.append({
            "snapshot_date": str(r["snapshot_date"]),
            "total_reviews": r["total_reviews"],
            "churn_intent": r["churn_intent"],
            "churn_density": _safe_float(r["churn_density"], 0),
            "avg_urgency": _safe_float(r["avg_urgency"], 0),
            "positive_review_pct": _safe_float(r["positive_review_pct"]),
            "recommend_ratio": _safe_float(r["recommend_ratio"]),
            "support_sentiment": _safe_float(r["support_sentiment"]),
            "legacy_support_score": _safe_float(r["legacy_support_score"]),
            "new_feature_velocity": _safe_float(r["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(r["employee_growth_rate"]),
            "top_pain": r["top_pain"],
            "top_competitor": r["top_competitor"],
            "pain_count": r["pain_count"],
            "competitor_count": r["competitor_count"],
            "displacement_edge_count": r["displacement_edge_count"],
            "high_intent_company_count": r["high_intent_company_count"],
        })
    return {"vendor_name": resolved, "snapshots": snapshots, "count": len(snapshots)}


# ---------------------------------------------------------------------------
# GET /product-profile
# ---------------------------------------------------------------------------


@router.get("/product-profile")
async def get_product_profile(
    vendor_name: str = Query(...),
    user: AuthUser | None = Depends(optional_auth),
):
    """Fetch pre-computed product profile knowledge card for a vendor."""
    vendor_name = _required_query_text(vendor_name, "vendor_name")
    pool = _pool_or_503()
    row = await pool.fetchrow(
        """
        SELECT id, vendor_name, product_category,
               strengths, weaknesses, pain_addressed,
               total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
               primary_use_cases, typical_company_size, typical_industries,
               top_integrations, commonly_compared_to, commonly_switched_from,
               profile_summary, last_computed_at, created_at
        FROM b2b_product_profiles
        WHERE vendor_name ILIKE '%' || $1 || '%'
        ORDER BY total_reviews_analyzed DESC
        LIMIT 1
        """,
        vendor_name,
    )
    if not row:
        raise HTTPException(404, f"No product profile found for '{vendor_name}'")

    return {
        "id": str(row["id"]),
        "vendor_name": row["vendor_name"],
        "product_category": row["product_category"],
        "strengths": _safe_json(row["strengths"]),
        "weaknesses": _safe_json(row["weaknesses"]),
        "pain_addressed": _safe_json(row["pain_addressed"]),
        "total_reviews_analyzed": row["total_reviews_analyzed"],
        "avg_rating": _safe_float(row["avg_rating"]),
        "recommend_rate": _safe_float(row["recommend_rate"]),
        "avg_urgency": _safe_float(row["avg_urgency"]),
        "primary_use_cases": _safe_json(row["primary_use_cases"]),
        "typical_company_size": _safe_json(row["typical_company_size"]),
        "typical_industries": _safe_json(row["typical_industries"]),
        "top_integrations": _safe_json(row["top_integrations"]),
        "commonly_compared_to": _safe_json(row["commonly_compared_to"]),
        "commonly_switched_from": _safe_json(row["commonly_switched_from"]),
        "profile_summary": row["profile_summary"],
        "last_computed_at": str(row["last_computed_at"]) if row["last_computed_at"] else None,
        "created_at": str(row["created_at"]) if row["created_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /product-profile-history
# ---------------------------------------------------------------------------


@router.get("/product-profile-history")
async def get_product_profile_history(
    vendor_name: str = Query(...),
    days: int = Query(90, ge=1, le=365),
    limit: int = Query(90, ge=1, le=365),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _required_query_text(vendor_name, "vendor_name")
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT vendor_name, snapshot_date,
               total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
               strength_count, weakness_count, top_strength, top_weakness,
               top_use_case, top_integration,
               compared_to_count, switched_from_count,
               pain_categories_covered, profile_summary_len
        FROM b2b_product_profile_snapshots
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND snapshot_date >= CURRENT_DATE - $2::int
        ORDER BY snapshot_date DESC
        LIMIT $3
        """,
        vendor_name, days, limit,
    )
    resolved = rows[0]["vendor_name"] if rows else vendor_name
    snapshots = []
    for r in rows:
        snapshots.append({
            "snapshot_date": str(r["snapshot_date"]),
            "total_reviews_analyzed": r["total_reviews_analyzed"],
            "avg_rating": _safe_float(r["avg_rating"]),
            "recommend_rate": _safe_float(r["recommend_rate"]),
            "avg_urgency": _safe_float(r["avg_urgency"]),
            "strength_count": r["strength_count"],
            "weakness_count": r["weakness_count"],
            "top_strength": r["top_strength"],
            "top_weakness": r["top_weakness"],
            "top_use_case": r["top_use_case"],
            "top_integration": r["top_integration"],
            "compared_to_count": r["compared_to_count"],
            "switched_from_count": r["switched_from_count"],
            "pain_categories_covered": r["pain_categories_covered"],
            "profile_summary_len": r["profile_summary_len"],
        })
    return {"vendor_name": resolved, "snapshots": snapshots, "count": len(snapshots)}


# ---------------------------------------------------------------------------
# GET /change-events
# ---------------------------------------------------------------------------


@router.get("/change-events")
async def list_change_events(
    vendor_name: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)
    event_type = _optional_query_text(event_type)
    conditions: list[str] = ["event_date >= CURRENT_DATE - $1::int"]
    params: list = [days]
    idx = 2

    if vendor_name:
        conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    if event_type:
        conditions.append(f"event_type = ${idx}")
        params.append(event_type)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT vendor_name, event_date, event_type, description,
               old_value, new_value, delta, metadata
        FROM b2b_change_events
        {where}
        ORDER BY event_date DESC, created_at DESC
        LIMIT ${idx}
        """,
        *params, limit,
    )

    events = []
    for r in rows:
        events.append({
            "vendor_name": r["vendor_name"],
            "event_date": str(r["event_date"]),
            "event_type": r["event_type"],
            "description": r["description"],
            "old_value": _safe_float(r["old_value"]),
            "new_value": _safe_float(r["new_value"]),
            "delta": _safe_float(r["delta"]),
            "metadata": _safe_json(r["metadata"]),
        })

    return {"events": events, "count": len(events)}


# ---------------------------------------------------------------------------
# GET /change-events/summary
# ---------------------------------------------------------------------------


@router.get("/change-events/summary")
async def change_events_summary(
    days: int = Query(7, ge=1, le=90),
    user: AuthUser | None = Depends(optional_auth),
):
    pool = _pool_or_503()

    # Count by event type
    type_rows = await pool.fetch(
        """
        SELECT event_type, count(*) AS cnt
        FROM b2b_change_events
        WHERE event_date >= CURRENT_DATE - $1::int
        GROUP BY event_type
        ORDER BY cnt DESC
        """,
        days,
    )
    by_type = {r["event_type"]: r["cnt"] for r in type_rows}
    total = sum(by_type.values())

    # Most active vendors
    vendor_rows = await pool.fetch(
        """
        SELECT vendor_name, count(*) AS cnt
        FROM b2b_change_events
        WHERE event_date >= CURRENT_DATE - $1::int
        GROUP BY vendor_name
        ORDER BY cnt DESC
        LIMIT 10
        """,
        days,
    )
    most_active = [{"vendor_name": r["vendor_name"], "event_count": r["cnt"]} for r in vendor_rows]

    return {
        "period_days": days,
        "total_events": total,
        "by_type": by_type,
        "most_active_vendors": most_active,
    }


# ---------------------------------------------------------------------------
# GET /concurrent-events -- Cross-vendor trend correlation
# ---------------------------------------------------------------------------


@router.get("/concurrent-events")
async def list_concurrent_events(
    days: int = Query(30, ge=1, le=365),
    event_type: Optional[str] = Query(None),
    min_vendors: int = Query(2, ge=2, le=50),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    """Find dates where multiple vendors experienced the same change event type.

    Surfaces cross-vendor correlations like 'urgency spiked at 4 vendors on the
    same day' which may indicate a market-level trend rather than vendor-specific.
    """
    pool = _pool_or_503()
    event_type = _optional_query_text(event_type)

    type_filter = ""
    params: list = [days, min_vendors, limit]
    if event_type:
        type_filter = "AND event_type = $4"
        params.append(event_type)

    rows = await pool.fetch(
        f"""
        SELECT event_date, event_type,
               COUNT(DISTINCT vendor_name) AS vendor_count,
               ARRAY_AGG(DISTINCT vendor_name ORDER BY vendor_name) AS vendors,
               AVG(delta) AS avg_delta,
               MIN(delta) AS min_delta,
               MAX(delta) AS max_delta
        FROM b2b_change_events
        WHERE event_date >= CURRENT_DATE - $1::int
          {type_filter}
        GROUP BY event_date, event_type
        HAVING COUNT(DISTINCT vendor_name) >= $2
        ORDER BY vendor_count DESC, event_date DESC
        LIMIT $3
        """,
        *params,
    )

    return {
        "period_days": days,
        "min_vendors": min_vendors,
        "event_type_filter": event_type,
        "concurrent_events": [
            {
                "event_date": str(r["event_date"]),
                "event_type": r["event_type"],
                "vendor_count": r["vendor_count"],
                "vendors": r["vendors"],
                "avg_delta": round(float(r["avg_delta"]), 2) if r["avg_delta"] is not None else None,
                "min_delta": round(float(r["min_delta"]), 2) if r["min_delta"] is not None else None,
                "max_delta": round(float(r["max_delta"]), 2) if r["max_delta"] is not None else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }


# ---------------------------------------------------------------------------
# GET /fuzzy-vendor-search -- Fuzzy vendor name search (pg_trgm)
# ---------------------------------------------------------------------------


@router.get("/fuzzy-vendor-search")
async def fuzzy_vendor_search(
    q: str = "",
    limit: int = 10,
    min_similarity: float = 0.3,
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q parameter is required")
    limit = max(1, min(limit, 100))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    from ..services.vendor_registry import fuzzy_search_vendors

    results = await fuzzy_search_vendors(q.strip(), limit=limit, min_similarity=min_similarity)
    return {"query": q.strip(), "results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# GET /fuzzy-company-search -- Fuzzy company name search (pg_trgm)
# ---------------------------------------------------------------------------


@router.get("/fuzzy-company-search")
async def fuzzy_company_search(
    q: str = "",
    vendor_name: str | None = None,
    limit: int = 10,
    min_similarity: float = 0.3,
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q parameter is required")
    vendor_name = _optional_query_text(vendor_name)
    limit = max(1, min(limit, 100))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    from ..services.vendor_registry import fuzzy_search_companies

    results = await fuzzy_search_companies(
        q.strip(), vendor_name=vendor_name, limit=limit, min_similarity=min_similarity,
    )
    return {"query": q.strip(), "vendor_filter": vendor_name, "results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# GET /parser-version-status -- Parser version mismatch summary
# ---------------------------------------------------------------------------


@router.get("/parser-version-status")
async def parser_version_status():
    """Show per-source parser version status and count of reviews needing re-extraction."""
    pool = _pool_or_503()

    from ..services.scraping.parsers import get_all_parsers

    parsers = get_all_parsers()
    sources = []
    for source_name, parser in sorted(parsers.items()):
        current_version = getattr(parser, "version", None)
        if not current_version:
            continue

        row = await pool.fetchrow(
            """
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE parser_version = $2) AS current_count,
                   COUNT(*) FILTER (WHERE parser_version IS NOT NULL AND parser_version != $2) AS outdated_count,
                   COUNT(*) FILTER (WHERE parser_version IS NULL) AS unknown_count
            FROM b2b_reviews
            WHERE source = $1
            """,
            source_name, current_version,
        )
        sources.append({
            "source": source_name,
            "current_version": current_version,
            "total_reviews": row["total"],
            "current_version_count": row["current_count"],
            "outdated_version_count": row["outdated_count"],
            "unknown_version_count": row["unknown_count"],
        })

    return {"sources": sources, "count": len(sources)}


# ---------------------------------------------------------------------------
# GET /vendor-correlation -- Pairwise vendor metric correlation
# ---------------------------------------------------------------------------


@router.get("/vendor-correlation")
async def get_vendor_correlation(
    vendor_a: str = Query(...),
    vendor_b: str = Query(...),
    days: int = Query(90, ge=7, le=365),
    metric: str = Query("churn_density"),
    user: AuthUser | None = Depends(optional_auth),
):
    """Compare two vendors' metric trends over time and compute correlation.

    Returns aligned time-series data and Pearson correlation coefficient.
    Supported metrics: churn_density, avg_urgency, recommend_ratio, total_reviews,
    displacement_edge_count, high_intent_company_count.
    """
    vendor_a = _required_query_text(vendor_a, "vendor_a")
    vendor_b = _required_query_text(vendor_b, "vendor_b")

    valid_metrics = {
        "churn_density", "avg_urgency", "recommend_ratio", "total_reviews",
        "displacement_edge_count", "high_intent_company_count", "pain_count",
        "competitor_count", "support_sentiment", "legacy_support_score",
        "new_feature_velocity", "employee_growth_rate",
    }
    if metric not in valid_metrics:
        raise HTTPException(status_code=400, detail=f"metric must be one of: {sorted(valid_metrics)}")

    pool = _pool_or_503()
    # Fetch aligned snapshots for both vendors
    rows = await pool.fetch(
        f"""
        SELECT a.snapshot_date,
               a.{metric} AS value_a,
               b.{metric} AS value_b
        FROM b2b_vendor_snapshots a
        JOIN b2b_vendor_snapshots b
          ON a.snapshot_date = b.snapshot_date
        WHERE a.vendor_name ILIKE '%' || $1 || '%'
          AND b.vendor_name ILIKE '%' || $2 || '%'
          AND a.snapshot_date >= CURRENT_DATE - $3::int
        ORDER BY a.snapshot_date ASC
        """,
        vendor_a, vendor_b, days,
    )

    if not rows:
        raise HTTPException(status_code=404, detail="No overlapping snapshots found for these vendors")

    series = [
        {
            "date": str(r["snapshot_date"]),
            "value_a": _safe_float(r["value_a"]),
            "value_b": _safe_float(r["value_b"]),
        }
        for r in rows
    ]

    # Compute Pearson correlation coefficient in Python
    vals_a = [_safe_float(r["value_a"], 0) for r in rows]
    vals_b = [_safe_float(r["value_b"], 0) for r in rows]
    correlation = _pearson_r(vals_a, vals_b)

    # Check displacement edges between the two vendors
    edge_rows = await pool.fetch(
        """
        SELECT from_vendor, to_vendor, mention_count, signal_strength, primary_driver
        FROM b2b_displacement_edges
        WHERE (from_vendor ILIKE '%' || $1 || '%' AND to_vendor ILIKE '%' || $2 || '%')
           OR (from_vendor ILIKE '%' || $2 || '%' AND to_vendor ILIKE '%' || $1 || '%')
        ORDER BY computed_date DESC
        LIMIT 5
        """,
        vendor_a, vendor_b,
    )
    displacement = [
        {
            "from_vendor": r["from_vendor"],
            "to_vendor": r["to_vendor"],
            "mention_count": r["mention_count"],
            "signal_strength": r["signal_strength"],
            "primary_driver": r["primary_driver"],
        }
        for r in edge_rows
    ]

    resolved_a = rows[0]["snapshot_date"]  # Just to get the vendor names from ILIKE
    return {
        "vendor_a": vendor_a,
        "vendor_b": vendor_b,
        "metric": metric,
        "period_days": days,
        "data_points": len(series),
        "correlation": correlation,
        "series": series,
        "displacement_edges": displacement,
    }


def _pearson_r(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient. Returns None if insufficient data."""
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denom_x = sum(a * a for a in dx) ** 0.5
    denom_y = sum(b * b for b in dy) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return round(numerator / (denom_x * denom_y), 4)


# ---------------------------------------------------------------------------
# GET /compare-vendor-periods
# ---------------------------------------------------------------------------


@router.get("/compare-vendor-periods")
async def compare_vendor_periods(
    vendor_name: str = Query(...),
    period_a_days_ago: int = Query(30, ge=0, le=365),
    period_b_days_ago: int = Query(0, ge=0, le=365),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _required_query_text(vendor_name, "vendor_name")
    pool = _pool_or_503()

    async def _nearest_snapshot(target_days_ago: int):
        return await pool.fetchrow(
            """
            SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
                   churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                   support_sentiment, legacy_support_score,
                   new_feature_velocity, employee_growth_rate,
                   top_pain, top_competitor, pain_count, competitor_count,
                   displacement_edge_count, high_intent_company_count
            FROM b2b_vendor_snapshots
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND snapshot_date <= CURRENT_DATE - $2::int
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            vendor_name, target_days_ago,
        )

    snap_a = await _nearest_snapshot(period_a_days_ago)
    snap_b = await _nearest_snapshot(period_b_days_ago)

    if not snap_a and not snap_b:
        raise HTTPException(status_code=404, detail=f"No snapshots found for vendor matching '{vendor_name}'")

    def _format(snap):
        if not snap:
            return None
        return {
            "snapshot_date": str(snap["snapshot_date"]),
            "total_reviews": snap["total_reviews"],
            "churn_intent": snap["churn_intent"],
            "churn_density": _safe_float(snap["churn_density"], 0),
            "avg_urgency": _safe_float(snap["avg_urgency"], 0),
            "positive_review_pct": _safe_float(snap["positive_review_pct"]),
            "recommend_ratio": _safe_float(snap["recommend_ratio"]),
            "support_sentiment": _safe_float(snap["support_sentiment"]),
            "legacy_support_score": _safe_float(snap["legacy_support_score"]),
            "new_feature_velocity": _safe_float(snap["new_feature_velocity"]),
            "employee_growth_rate": _safe_float(snap["employee_growth_rate"]),
            "top_pain": snap["top_pain"],
            "top_competitor": snap["top_competitor"],
            "pain_count": snap["pain_count"],
            "competitor_count": snap["competitor_count"],
            "displacement_edge_count": snap["displacement_edge_count"],
            "high_intent_company_count": snap["high_intent_company_count"],
        }

    a_fmt = _format(snap_a)
    b_fmt = _format(snap_b)

    deltas = {}
    if a_fmt and b_fmt:
        for key in ("churn_density", "avg_urgency", "recommend_ratio", "total_reviews",
                    "churn_intent", "pain_count", "competitor_count",
                    "displacement_edge_count", "high_intent_company_count",
                    "support_sentiment", "legacy_support_score",
                    "new_feature_velocity", "employee_growth_rate"):
            a_val = a_fmt.get(key)
            b_val = b_fmt.get(key)
            if a_val is not None and b_val is not None:
                deltas[key] = round(b_val - a_val, 2)

    resolved = (snap_a or snap_b)["vendor_name"]
    return {
        "vendor_name": resolved,
        "period_a": a_fmt,
        "period_b": b_fmt,
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# GET /signal-effectiveness
# ---------------------------------------------------------------------------

_SIGNAL_GROUP_EXPRESSIONS = {
    "buying_stage": "bc.buying_stage",
    "role_type": "bc.role_type",
    "target_mode": "bc.target_mode",
    "opportunity_score_bucket": """CASE
        WHEN bc.opportunity_score >= 90 THEN '90-100'
        WHEN bc.opportunity_score >= 70 THEN '70-89'
        WHEN bc.opportunity_score >= 50 THEN '50-69'
        ELSE 'below_50'
    END""",
    "urgency_bucket": """CASE
        WHEN bc.urgency_score >= 8 THEN 'high_8+'
        WHEN bc.urgency_score >= 5 THEN 'medium_5-7'
        ELSE 'low_0-4'
    END""",
    "pain_category": "bc.pain_categories->0->>'category'",
}


@router.get("/signal-effectiveness")
async def signal_effectiveness(
    vendor_name: Optional[str] = Query(None),
    min_sequences: int = Query(5, ge=1, le=100),
    group_by: str = Query("buying_stage"),
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
):
    """Correlate signal dimensions with campaign outcomes."""
    if group_by not in _SIGNAL_GROUP_EXPRESSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by. Must be one of: {sorted(_SIGNAL_GROUP_EXPRESSIONS.keys())}",
        )

    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)
    group_expr = _SIGNAL_GROUP_EXPRESSIONS[group_by]

    conditions: list[str] = [
        "bc.sequence_id IS NOT NULL",
        "cs.outcome != 'pending'",
        "bc.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $2::uuid)",
    ]
    params: list = [min_sequences, user.account_id]
    idx = 3

    if vendor_name:
        conditions.append(f"bc.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    where = " AND ".join(conditions)

    sql = f"""
    WITH seq_signals AS (
        SELECT DISTINCT ON (cs.id)
            cs.id, cs.outcome, cs.outcome_revenue,
            ({group_expr}) AS signal_group
        FROM campaign_sequences cs
        JOIN b2b_campaigns bc ON bc.sequence_id = cs.id
        WHERE {where}
        ORDER BY cs.id, bc.step_number ASC
    )
    SELECT
        signal_group,
        COUNT(*) AS total_sequences,
        COUNT(*) FILTER (WHERE outcome = 'meeting_booked') AS meetings,
        COUNT(*) FILTER (WHERE outcome = 'deal_opened') AS deals_opened,
        COUNT(*) FILTER (WHERE outcome = 'deal_won') AS deals_won,
        COUNT(*) FILTER (WHERE outcome = 'deal_lost') AS deals_lost,
        COUNT(*) FILTER (WHERE outcome = 'no_opportunity') AS no_opportunity,
        COUNT(*) FILTER (WHERE outcome = 'disqualified') AS disqualified,
        ROUND(
            COUNT(*) FILTER (WHERE outcome IN ('meeting_booked', 'deal_opened', 'deal_won'))::numeric
            / NULLIF(COUNT(*), 0), 3
        ) AS positive_outcome_rate,
        COALESCE(SUM(outcome_revenue) FILTER (WHERE outcome = 'deal_won'), 0) AS total_revenue
    FROM seq_signals
    WHERE signal_group IS NOT NULL
    GROUP BY signal_group
    HAVING COUNT(*) >= $1
    ORDER BY positive_outcome_rate DESC
    """

    rows = await pool.fetch(sql, *params)

    groups = []
    for r in rows:
        groups.append({
            "signal_group": r["signal_group"],
            "total_sequences": r["total_sequences"],
            "meetings": r["meetings"],
            "deals_opened": r["deals_opened"],
            "deals_won": r["deals_won"],
            "deals_lost": r["deals_lost"],
            "no_opportunity": r["no_opportunity"],
            "disqualified": r["disqualified"],
            "positive_outcome_rate": _safe_float(r["positive_outcome_rate"], 0.0),
            "total_revenue": _safe_float(r["total_revenue"], 0.0),
        })

    return {
        "group_by": group_by,
        "vendor_filter": vendor_name,
        "min_sequences": min_sequences,
        "groups": groups,
        "total_groups": len(groups),
    }


# ---------------------------------------------------------------------------
# GET /parser-health
# ---------------------------------------------------------------------------


@router.get("/parser-health")
async def get_parser_health(user: AuthUser | None = Depends(optional_auth)):
    """Parser version distribution and stale review counts per source."""
    pool = _pool_or_503()
    rows = await pool.fetch("""
        WITH version_counts AS (
            SELECT source,
                   COALESCE(parser_version, 'unknown') AS parser_version,
                   COUNT(*) AS review_count
            FROM b2b_reviews
            GROUP BY source, COALESCE(parser_version, 'unknown')
        ),
        latest AS (
            SELECT DISTINCT ON (source) source, parser_version AS latest_version
            FROM b2b_scrape_log
            WHERE parser_version IS NOT NULL
            ORDER BY source, started_at DESC
        )
        SELECT vc.source, vc.parser_version, vc.review_count,
               l.latest_version,
               (vc.parser_version != COALESCE(l.latest_version, vc.parser_version))
                   AS is_stale
        FROM version_counts vc
        LEFT JOIN latest l USING (source)
        ORDER BY vc.source, vc.review_count DESC
    """)

    sources: dict[str, dict] = {}
    for r in rows:
        src = r["source"]
        if src not in sources:
            sources[src] = {
                "source": src,
                "latest_version": r["latest_version"],
                "versions": [],
                "total_reviews": 0,
                "stale_reviews": 0,
            }
        entry = sources[src]
        entry["versions"].append({
            "parser_version": r["parser_version"],
            "review_count": r["review_count"],
            "is_stale": r["is_stale"],
        })
        entry["total_reviews"] += r["review_count"]
        if r["is_stale"]:
            entry["stale_reviews"] += r["review_count"]

    result = sorted(sources.values(), key=lambda x: x["stale_reviews"], reverse=True)
    total_stale = sum(s["stale_reviews"] for s in result)
    return {
        "sources": result,
        "total_stale_reviews": total_stale,
        "total_sources": len(result),
    }


# ---------------------------------------------------------------------------
# GET /calibration-weights
# ---------------------------------------------------------------------------


@router.get("/calibration-weights")
async def get_calibration_weights(
    dimension: Optional[str] = Query(None),
    model_version: Optional[int] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    """View score calibration weights derived from campaign outcomes."""
    dimension = _optional_query_text(dimension)
    valid_dims = {"role_type", "buying_stage", "urgency_bucket", "seat_bucket", "context_keyword"}
    if dimension and dimension not in valid_dims:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension. Must be one of: {sorted(valid_dims)}",
        )

    pool = _pool_or_503()

    if model_version is None:
        model_version = await pool.fetchval(
            "SELECT MAX(model_version) FROM score_calibration_weights"
        )
        if model_version is None:
            return {
                "weights": [],
                "count": 0,
                "model_version": None,
                "message": "No calibration data yet",
            }

    conditions: list[str] = ["model_version = $1"]
    params: list = [model_version]
    idx = 2

    if dimension:
        conditions.append(f"dimension = ${idx}")
        params.append(dimension)
        idx += 1

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT dimension, dimension_value, total_sequences, positive_outcomes,
               deals_won, total_revenue, positive_rate, baseline_rate, lift,
               weight_adjustment, static_default, calibrated_at,
               sample_window_days, model_version
        FROM score_calibration_weights
        WHERE {where}
        ORDER BY dimension, lift DESC
        """,
        *params,
    )

    weights = []
    for r in rows:
        weights.append({
            "dimension": r["dimension"],
            "dimension_value": r["dimension_value"],
            "total_sequences": r["total_sequences"],
            "positive_outcomes": r["positive_outcomes"],
            "deals_won": r["deals_won"],
            "total_revenue": _safe_float(r["total_revenue"], 0.0),
            "positive_rate": _safe_float(r["positive_rate"], 0.0),
            "baseline_rate": _safe_float(r["baseline_rate"], 0.0),
            "lift": _safe_float(r["lift"], 1.0),
            "weight_adjustment": _safe_float(r["weight_adjustment"], 0.0),
            "static_default": _safe_float(r["static_default"], 0.0),
            "calibrated_at": r["calibrated_at"].isoformat() if r["calibrated_at"] else None,
            "sample_window_days": r["sample_window_days"],
            "model_version": r["model_version"],
        })

    return {
        "model_version": model_version,
        "weights": weights,
        "count": len(weights),
    }


# ---------------------------------------------------------------------------
# GET /outcome-distribution  -- system-wide outcome funnel
# ---------------------------------------------------------------------------


@router.get("/outcome-distribution")
async def get_outcome_distribution(
    vendor_name: Optional[str] = Query(None),
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
):
    """System-wide campaign outcome distribution (funnel view)."""
    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)

    conditions: list[str] = [
        "bc.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid)"
    ]
    params: list = [str(user.account_id)]
    idx = 2

    if vendor_name:
        conditions.append(f"bc.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    rows = await pool.fetch(
        f"""
        SELECT cs.outcome,
               COUNT(*) AS count,
               COALESCE(SUM(cs.outcome_revenue), 0) AS total_revenue,
               MIN(cs.outcome_recorded_at) AS first_recorded,
               MAX(cs.outcome_recorded_at) AS last_recorded
        FROM campaign_sequences cs
        LEFT JOIN b2b_campaigns bc ON bc.sequence_id = cs.id AND bc.step_number = 1
        {where}
        GROUP BY cs.outcome
        ORDER BY count DESC
        """,
        *params,
    )

    total = sum(r["count"] for r in rows)
    buckets = []
    for r in rows:
        buckets.append({
            "outcome": r["outcome"],
            "count": r["count"],
            "pct": round(r["count"] / total * 100, 1) if total else 0,
            "total_revenue": _safe_float(r["total_revenue"], 0.0),
            "first_recorded": str(r["first_recorded"]) if r["first_recorded"] else None,
            "last_recorded": str(r["last_recorded"]) if r["last_recorded"] else None,
        })

    return {
        "total_sequences": total,
        "vendor_filter": vendor_name,
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# GET /signal-to-outcome  -- attribution view
# ---------------------------------------------------------------------------

_ATTRIBUTION_GROUP_EXPRS = {
    "buying_stage": "bc.buying_stage",
    "role_type": "bc.role_type",
    "target_mode": "bc.target_mode",
    "opportunity_score_bucket": """CASE
        WHEN bc.opportunity_score >= 90 THEN '90-100'
        WHEN bc.opportunity_score >= 70 THEN '70-89'
        WHEN bc.opportunity_score >= 50 THEN '50-69'
        ELSE 'below_50'
    END""",
    "urgency_bucket": """CASE
        WHEN bc.urgency_score >= 8 THEN 'high_8+'
        WHEN bc.urgency_score >= 5 THEN 'medium_5-7'
        ELSE 'low_0-4'
    END""",
    "pain_category": "bc.pain_categories->0->>'category'",
}


@router.get("/signal-to-outcome")
async def get_signal_to_outcome(
    vendor_name: Optional[str] = Query(None),
    min_sequences: int = Query(5, ge=1, le=100),
    group_by: str = Query("buying_stage"),
    user: AuthUser | None = Depends(optional_auth),
):
    """Signal effectiveness attribution view: which signal dimensions produce the best outcomes."""
    if group_by not in _ATTRIBUTION_GROUP_EXPRS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by. Must be one of: {sorted(_ATTRIBUTION_GROUP_EXPRS.keys())}",
        )

    pool = _pool_or_503()
    vendor_name = _optional_query_text(vendor_name)
    group_expr = _ATTRIBUTION_GROUP_EXPRS[group_by]

    conditions: list[str] = ["bc.sequence_id IS NOT NULL", "cs.outcome != 'pending'"]
    params: list = [min_sequences]
    idx = 2

    if _should_scope(user):
        conditions.append(
            f"bc.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(str(user.account_id))
        idx += 1

    if vendor_name:
        conditions.append(f"bc.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1

    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        WITH seq_signals AS (
            SELECT DISTINCT ON (cs.id)
                cs.id, cs.outcome, cs.outcome_revenue,
                ({group_expr}) AS signal_group
            FROM campaign_sequences cs
            JOIN b2b_campaigns bc ON bc.sequence_id = cs.id
            WHERE {where}
            ORDER BY cs.id, bc.step_number ASC
        )
        SELECT
            signal_group,
            COUNT(*) AS total_sequences,
            COUNT(*) FILTER (WHERE outcome = 'meeting_booked') AS meetings,
            COUNT(*) FILTER (WHERE outcome = 'deal_opened') AS deals_opened,
            COUNT(*) FILTER (WHERE outcome = 'deal_won') AS deals_won,
            COUNT(*) FILTER (WHERE outcome = 'deal_lost') AS deals_lost,
            COUNT(*) FILTER (WHERE outcome = 'no_opportunity') AS no_opportunity,
            COUNT(*) FILTER (WHERE outcome = 'disqualified') AS disqualified,
            ROUND(
                COUNT(*) FILTER (WHERE outcome IN ('meeting_booked', 'deal_opened', 'deal_won'))::numeric
                / NULLIF(COUNT(*), 0), 3
            ) AS positive_outcome_rate,
            COALESCE(SUM(outcome_revenue) FILTER (WHERE outcome = 'deal_won'), 0) AS total_revenue
        FROM seq_signals
        WHERE signal_group IS NOT NULL
        GROUP BY signal_group
        HAVING COUNT(*) >= $1
        ORDER BY positive_outcome_rate DESC
        """,
        *params,
    )

    groups = []
    for r in rows:
        groups.append({
            "signal_group": r["signal_group"],
            "total_sequences": r["total_sequences"],
            "meetings": r["meetings"],
            "deals_opened": r["deals_opened"],
            "deals_won": r["deals_won"],
            "deals_lost": r["deals_lost"],
            "no_opportunity": r["no_opportunity"],
            "disqualified": r["disqualified"],
            "positive_outcome_rate": _safe_float(r["positive_outcome_rate"], 0.0),
            "total_revenue": _safe_float(r["total_revenue"], 0.0),
        })

    return {
        "group_by": group_by,
        "vendor_filter": vendor_name,
        "min_sequences": min_sequences,
        "groups": groups,
        "total_groups": len(groups),
    }


# ---------------------------------------------------------------------------
# POST /calibration/trigger  -- kick off score calibration
# ---------------------------------------------------------------------------


@router.post("/calibration/trigger")
async def trigger_calibration(
    window_days: int = Query(180, ge=30, le=730),
    user: AuthUser | None = Depends(optional_auth),
):
    """Trigger score calibration from campaign outcomes (on-demand)."""
    pool = _pool_or_503()

    try:
        from ..autonomous.tasks.b2b_score_calibration import calibrate

        result = await calibrate(pool, window_days=window_days)
        return {
            "triggered": True,
            "window_days": window_days,
            "triggered_by": f"api:{user.email}" if user else "api",
            **result,
        }
    except Exception as exc:
        logger.exception("Calibration trigger failed")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {exc}")


# ---------------------------------------------------------------------------
# GET /export/signals  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/signals")
async def export_signals(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(0, ge=0, le=10),
    category: Optional[str] = Query(None),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _optional_query_text(vendor_name)
    category = _optional_query_text(category)
    pool = _pool_or_503()
    from ..autonomous.tasks._b2b_shared import read_vendor_signal_rows

    rows = await read_vendor_signal_rows(
        pool,
        vendor_name_query=vendor_name,
        min_urgency=min_urgency,
        product_category=category,
        tracked_account_id=user.account_id if _should_scope(user) else None,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
        limit=EXPORT_ROW_LIMIT,
    )

    data = [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"] or "",
            "total_reviews": r["total_reviews"],
            "churn_intent_count": r["churn_intent_count"],
            "avg_urgency_score": _safe_float(r["avg_urgency_score"], ""),
            "avg_rating_normalized": _safe_float(r["avg_rating_normalized"], ""),
            "nps_proxy": _safe_float(r["nps_proxy"], ""),
            "price_complaint_rate": _safe_float(r["price_complaint_rate"], ""),
            "decision_maker_churn_rate": _safe_float(r["decision_maker_churn_rate"], ""),
            "support_sentiment": _safe_float(r["support_sentiment"], ""),
            "legacy_support_score": _safe_float(r["legacy_support_score"], ""),
            "new_feature_velocity": _safe_float(r["new_feature_velocity"], ""),
            "employee_growth_rate": _safe_float(r["employee_growth_rate"], ""),
            "archetype": "",
            "archetype_confidence": "",
            "reasoning_risk_level": "",
            "keyword_spike_count": r["keyword_spike_count"] if r["keyword_spike_count"] is not None else "",
            "insider_signal_count": r["insider_signal_count"] if r["insider_signal_count"] is not None else "",
            "last_computed_at": str(r["last_computed_at"]) if r["last_computed_at"] else "",
        }
        for r in rows
    ]

    reasoning_views = await _load_reasoning_views_for_vendors(
        pool,
        [row["vendor_name"] for row in rows if row.get("vendor_name")],
    )
    for row in data:
        view = reasoning_views.get(_normalize_vendor_name(row.get("vendor_name")))
        if view is not None:
            _overlay_reasoning_summary_from_view(row, view)

    return _csv_response(data, "churn_signals.csv")


# ---------------------------------------------------------------------------
# GET /export/reviews  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/reviews")
async def export_reviews(
    vendor_name: Optional[str] = Query(None),
    pain_category: Optional[str] = Query(None),
    min_urgency: Optional[float] = Query(None, ge=0, le=10),
    company: Optional[str] = Query(None),
    has_churn_intent: Optional[bool] = Query(None),
    window_days: int = Query(90, ge=1, le=3650),
    user: AuthUser | None = Depends(optional_auth),
):
    from ..autonomous.tasks._b2b_shared import read_review_details

    vendor_name = _optional_query_text(vendor_name)
    pain_category = _optional_query_text(pain_category)
    company = _optional_query_text(company)
    pool = _pool_or_503()

    scoped_vendors: list[str] | None = None
    if _should_scope(user):
        vrows = await pool.fetch(
            "SELECT vendor_name FROM tracked_vendors WHERE account_id = $1::uuid",
            user.account_id,
        )
        scoped_vendors = [r["vendor_name"] for r in vrows]

    details = await read_review_details(
        pool,
        window_days=window_days,
        vendor_name=vendor_name,
        scoped_vendors=scoped_vendors,
        pain_category=pain_category,
        min_urgency=min_urgency,
        company=company,
        has_churn_intent=has_churn_intent,
        recency_column="enriched_at",
        limit=EXPORT_ROW_LIMIT,
    )

    data = [
        {
            "vendor_name": d.get("vendor_name", ""),
            "product_category": d.get("product_category") or "",
            "reviewer_company": d.get("reviewer_company") or "",
            "rating": d["rating"] if d.get("rating") is not None else "",
            "urgency_score": d["urgency_score"] if d.get("urgency_score") is not None else "",
            "pain_category": d.get("pain_category") or "",
            "intent_to_leave": d["intent_to_leave"] if d.get("intent_to_leave") is not None else "",
            "decision_maker": d["decision_maker"] if d.get("decision_maker") is not None else "",
            "enriched_at": str(d["enriched_at"]) if d.get("enriched_at") else "",
        }
        for d in details
    ]

    return _csv_response(data, "enriched_reviews.csv")


# ---------------------------------------------------------------------------
# GET /export/high-intent  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/high-intent")
async def export_high_intent(
    vendor_name: Optional[str] = Query(None),
    min_urgency: float = Query(7, ge=0, le=10),
    window_days: int = Query(90, ge=1, le=3650),
    user: AuthUser | None = Depends(optional_auth),
):
    vendor_name = _optional_query_text(vendor_name)
    pool = _pool_or_503()
    scoped_vendors = await _get_scoped_vendors(pool, user)

    rows = await read_high_intent_companies(
        pool,
        min_urgency=min_urgency,
        window_days=window_days,
        vendor_name=vendor_name,
        scoped_vendors=scoped_vendors,
        limit=EXPORT_ROW_LIMIT,
    )

    data = []
    for r in rows:
        alternatives = r.get("alternatives")
        if isinstance(alternatives, list):
            alt_str = "; ".join(str(a) for a in alternatives)
        else:
            alt_str = str(alternatives) if alternatives else ""

        data.append({
            "company": r.get("company") or "",
            "vendor": r.get("vendor") or "",
            "category": r.get("category") or "",
            "role_level": r.get("role_level") or "",
            "decision_maker": r.get("decision_maker") if r.get("decision_maker") is not None else "",
            "urgency": _safe_float(r.get("urgency"), ""),
            "pain": r.get("pain") or "",
            "alternatives": alt_str,
            "contract_signal": r.get("contract_signal") or "",
            "seat_count": r.get("seat_count") if r.get("seat_count") is not None else "",
            "lock_in_level": r.get("lock_in_level") or "",
            "contract_end": r.get("contract_end") or "",
            "buying_stage": r.get("buying_stage") or "",
        })

    return _csv_response(data, "high_intent_leads.csv")


# ---------------------------------------------------------------------------
# GET /export/source-health  (CSV)
# ---------------------------------------------------------------------------


@router.get("/export/source-health")
async def export_source_health(
    window_days: int = Query(7, ge=1, le=30),
    source: Optional[str] = Query(None),
):
    source = _normalize_source_query(source)
    if source is not None:
        if source not in ALL_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source. Must be one of: {sorted(s.value for s in ALL_SOURCES)}",
            )

    pool = _pool_or_503()
    sql, extra_params = _build_source_health_query(source)
    rows = await pool.fetch(sql, window_days, *extra_params)

    data = []
    for r in rows:
        total = r["total_scrapes"] or 1
        prev_total = r["prev_total_scrapes"] or 0
        data.append({
            "source": r["source"],
            "display_name": source_display_name(r["source"]),
            "total_scrapes": r["total_scrapes"],
            "success_count": r["success_count"],
            "partial_count": r["partial_count"],
            "failed_count": r["failed_count"],
            "blocked_count": r["blocked_count"],
            "success_rate": round(r["success_count"] / total, 3),
            "block_rate": round(r["blocked_count"] / total, 3),
            "avg_reviews_found": _safe_float(r["avg_reviews_found"], ""),
            "avg_reviews_inserted": _safe_float(r["avg_reviews_inserted"], ""),
            "avg_duration_ms": _safe_float(r["avg_duration_ms"], ""),
            "p95_duration_ms": _safe_float(r["p95_duration_ms"], ""),
            "last_success_at": str(r["last_success_at"]) if r["last_success_at"] else "",
            "last_scrape_at": str(r["last_scrape_at"]) if r["last_scrape_at"] else "",
            "active_targets": r["active_targets"],
            "prev_window_scrapes": prev_total,
            "prev_success_rate": round(r["prev_success_count"] / max(prev_total, 1), 3) if prev_total else "",
            "prev_block_rate": round(r["prev_blocked_count"] / max(prev_total, 1), 3) if prev_total else "",
        })

    return _csv_response(data, "source_health.csv")


# ---------------------------------------------------------------------------
# Webhook subscription management
# ---------------------------------------------------------------------------

VALID_WEBHOOK_EVENT_TYPES = {"change_event", "churn_alert", "report_generated", "signal_update", "high_intent_push"}


CRM_ONLY_WEBHOOK_EVENT_TYPES = {"high_intent_push"}


VALID_WEBHOOK_CHANNELS = {
    "generic", "slack", "teams",
    "crm_hubspot", "crm_salesforce", "crm_pipedrive",
}


class CreateWebhookBody(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    secret: str = Field(..., min_length=16, max_length=256)
    event_types: list[str] = Field(...)
    channel: str = "generic"
    auth_header: str | None = None
    description: str | None = None

    @field_validator("url", "secret", "channel", mode="before")
    @classmethod
    def _trim_required_text(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value is required")
        return text

    @field_validator("auth_header", "description", mode="before")
    @classmethod
    def _trim_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("event_types", mode="before")
    @classmethod
    def _trim_event_types(cls, value: Any) -> Any:
        if value is None or not isinstance(value, list):
            return value
        return [str(item).strip() for item in value if str(item).strip()]


class UpdateWebhookBody(BaseModel):
    url: str | None = None
    event_types: list[str] | None = None
    enabled: bool | None = None
    description: str | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _trim_url(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("description", mode="before")
    @classmethod
    def _trim_description(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("event_types", mode="before")
    @classmethod
    def _trim_event_types(cls, value: Any) -> Any:
        if value is None or not isinstance(value, list):
            return value
        return [str(item).strip() for item in value if str(item).strip()]


def _validate_webhook_event_types_for_channel(event_types: list[str], channel: str) -> None:
    invalid = set(event_types) - VALID_WEBHOOK_EVENT_TYPES
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_types: {sorted(invalid)}. Must be from: {sorted(VALID_WEBHOOK_EVENT_TYPES)}",
        )
    if not event_types:
        raise HTTPException(status_code=400, detail="event_types must not be empty")
    crm_only = set(event_types) & CRM_ONLY_WEBHOOK_EVENT_TYPES
    if crm_only and not str(channel or "").startswith("crm_"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_types for channel {channel}: {sorted(crm_only)} require a CRM channel",
        )


@router.post("/webhooks")
async def create_webhook(
    body: CreateWebhookBody,
    user: AuthUser | None = Depends(optional_auth),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Validate URL scheme (prevent SSRF)
    if not body.url.startswith(("https://", "http://")):
        raise HTTPException(status_code=400, detail="url must begin with https:// or http://")

    if body.channel not in VALID_WEBHOOK_CHANNELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid channel: {body.channel}. Must be one of: {sorted(VALID_WEBHOOK_CHANNELS)}",
        )

    _validate_webhook_event_types_for_channel(body.event_types, body.channel)

    # CRM channels require auth_header
    if body.channel.startswith("crm_") and not body.auth_header:
        raise HTTPException(
            status_code=400,
            detail="auth_header is required for CRM channels (e.g., 'Bearer pat-xxx')",
        )

    pool = _pool_or_503()
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_webhook_subscriptions
                (account_id, url, secret, event_types, channel, auth_header, description)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
            RETURNING id, account_id, url, event_types,
                      COALESCE(channel, 'generic') AS channel,
                      enabled, description, created_at
            """,
            user.account_id,
            body.url,
            body.secret,
            body.event_types,
            body.channel,
            body.auth_header,
            body.description,
        )
    except Exception as exc:
        if "uq_webhook_account_url" in str(exc):
            raise HTTPException(status_code=409, detail="Webhook URL already registered for this account")
        raise

    return {
        "id": str(row["id"]),
        "account_id": str(row["account_id"]),
        "url": row["url"],
        "event_types": row["event_types"],
        "channel": row["channel"],
        "enabled": row["enabled"],
        "description": row["description"],
        "created_at": row["created_at"].isoformat(),
    }


@router.get("/webhooks")
async def list_webhooks(
    user: AuthUser | None = Depends(optional_auth),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    pool = _pool_or_503()

    rows = await pool.fetch(
        """
        SELECT ws.id, ws.url, ws.event_types,
               COALESCE(ws.channel, 'generic') AS channel,
               ws.enabled, ws.description, ws.created_at, ws.updated_at,
               COALESCE(recent_activity.recent_deliveries, 0) AS recent_deliveries,
               COALESCE(recent_activity.recent_successes, 0) AS recent_successes,
               latest_failure.event_type AS latest_failure_event_type,
               latest_failure.status_code AS latest_failure_status_code,
               latest_failure.error AS latest_failure_error,
               latest_failure.delivered_at AS latest_failure_at,
               latest_failure.signal_id AS latest_failure_signal_id,
               latest_failure.review_id AS latest_failure_review_id,
               latest_failure.report_id AS latest_failure_report_id,
               latest_failure.vendor_name AS latest_failure_vendor_name,
               latest_failure.company_name AS latest_failure_company_name,
               latest_test.success AS latest_test_success,
               latest_test.status_code AS latest_test_status_code,
               latest_test.error AS latest_test_error,
               latest_test.delivered_at AS latest_test_at,
               latest_test.signal_id AS latest_test_signal_id,
               latest_test.review_id AS latest_test_review_id,
               latest_test.report_id AS latest_test_report_id,
               latest_test.vendor_name AS latest_test_vendor_name,
               latest_test.company_name AS latest_test_company_name,
               latest_crm.id AS latest_crm_id,
               latest_crm.signal_type AS latest_crm_signal_type,
               latest_crm.signal_id AS latest_crm_signal_id,
               latest_crm.review_id AS latest_crm_review_id,
               latest_crm.vendor_name AS latest_crm_vendor_name,
               latest_crm.company_name AS latest_crm_company_name,
               latest_crm.crm_record_id AS latest_crm_record_id,
               latest_crm.crm_record_type AS latest_crm_record_type,
               latest_crm.status AS latest_crm_status,
               latest_crm.error AS latest_crm_error,
               latest_crm.pushed_at AS latest_crm_at
        FROM b2b_webhook_subscriptions ws
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS recent_deliveries,
                   COUNT(*) FILTER (WHERE dl.success) AS recent_successes
            FROM b2b_webhook_delivery_log dl
            WHERE dl.subscription_id = ws.id
              AND dl.delivered_at > NOW() - INTERVAL '7 days'
        ) AS recent_activity ON TRUE
        LEFT JOIN LATERAL (
            SELECT event_type, status_code, error, delivered_at, signal_id, review_id, report_id, vendor_name, company_name
            FROM b2b_webhook_delivery_log dl
            WHERE dl.subscription_id = ws.id
              AND NOT dl.success
            ORDER BY dl.delivered_at DESC
            LIMIT 1
        ) AS latest_failure ON TRUE
        LEFT JOIN LATERAL (
            SELECT success, status_code, error, delivered_at, signal_id, review_id, report_id, vendor_name, company_name
            FROM b2b_webhook_delivery_log dl
            WHERE dl.subscription_id = ws.id
              AND dl.event_type = 'test'
            ORDER BY dl.delivered_at DESC
            LIMIT 1
        ) AS latest_test ON TRUE
        LEFT JOIN LATERAL (
            SELECT id, signal_type, signal_id, review_id, vendor_name, company_name,
                   crm_record_id, crm_record_type, status, error, pushed_at
            FROM b2b_crm_push_log cp
            WHERE cp.subscription_id = ws.id
            ORDER BY cp.pushed_at DESC
            LIMIT 1
        ) AS latest_crm ON TRUE
        WHERE ws.account_id = $1::uuid
        ORDER BY ws.created_at DESC
        """,
        user.account_id,
    )

    report_cache: dict[str, Any] = {}
    tracked_vendor_cache: dict[str, Any] = {}
    signal_cache: dict[str, Any] = {}
    report_activity_cache: dict[str, Any] = {}
    webhooks = []
    for r in rows:
        recent_total = r["recent_deliveries"] or 0
        latest_test_success = r["latest_test_success"] if "latest_test_success" in r else None
        latest_test_status_code = r["latest_test_status_code"] if "latest_test_status_code" in r else None
        latest_test_error = r["latest_test_error"] if "latest_test_error" in r else None
        latest_test_at = r["latest_test_at"] if "latest_test_at" in r else None
        latest_crm_push = None
        latest_crm_id = r["latest_crm_id"] if "latest_crm_id" in r else None
        if latest_crm_id:
            latest_crm_signal_type = str(r["latest_crm_signal_type"] or "").strip()
            latest_crm_signal_id = str(r["latest_crm_signal_id"]) if "latest_crm_signal_id" in r and r["latest_crm_signal_id"] else None
            latest_crm_signal_context = None
            if latest_crm_signal_id and latest_crm_signal_type != "report_generated":
                latest_crm_signal_context = signal_cache.get(latest_crm_signal_id)
                if latest_crm_signal_context is None and latest_crm_signal_id not in signal_cache:
                    latest_crm_signal_context = await _fetch_company_signal_focus_context(pool, latest_crm_signal_id)
                    signal_cache[latest_crm_signal_id] = latest_crm_signal_context
            latest_crm_report_context = None
            if latest_crm_signal_type == "report_generated" and latest_crm_signal_id:
                latest_crm_report_context = await _fetch_webhook_activity_report_context(
                    pool,
                    latest_crm_signal_id,
                    report_activity_cache,
                )
            latest_crm_review_id = _normalize_webhook_activity_uuid_text(r["latest_crm_review_id"]) if "latest_crm_review_id" in r and r["latest_crm_review_id"] else None
            latest_crm_resolved_review_id = latest_crm_review_id or (latest_crm_signal_context or {}).get("review_id")
            latest_crm_vendor_name = (
                _optional_query_text(r["latest_crm_vendor_name"])
                if "latest_crm_vendor_name" in r and r["latest_crm_vendor_name"]
                else None
            ) or (latest_crm_signal_context or {}).get("vendor_name") or (latest_crm_report_context or {}).get("vendor_name")
            latest_crm_company_name = (
                _optional_query_text(r["latest_crm_company_name"])
                if "latest_crm_company_name" in r and r["latest_crm_company_name"]
                else None
            ) or (latest_crm_signal_context or {}).get("company_name")
            latest_crm_account_review_focus = await _resolve_webhook_activity_account_focus(
                pool,
                user,
                vendor_name=latest_crm_vendor_name,
                company_name=latest_crm_company_name,
                signal_id=None if latest_crm_signal_type == "report_generated" else latest_crm_signal_id,
                review_id=latest_crm_resolved_review_id,
                report_cache=report_cache,
                tracked_vendor_cache=tracked_vendor_cache,
                signal_cache=signal_cache,
            )
            latest_crm_push = {
                "id": str(latest_crm_id),
                "signal_type": latest_crm_signal_type,
                "signal_id": latest_crm_signal_id,
                "vendor_name": latest_crm_vendor_name,
                "company_name": latest_crm_company_name,
                "review_id": latest_crm_resolved_review_id,
                "report_id": latest_crm_signal_id if latest_crm_signal_type == "report_generated" else None,
                "report_type": (latest_crm_report_context or {}).get("report_type"),
                "report_title": (latest_crm_report_context or {}).get("report_title"),
                "crm_record_id": r["latest_crm_record_id"] if "latest_crm_record_id" in r else None,
                "crm_record_type": r["latest_crm_record_type"] if "latest_crm_record_type" in r else None,
                "status": _normalize_crm_push_status(r["latest_crm_status"] if "latest_crm_status" in r else None),
                "error": r["latest_crm_error"] if "latest_crm_error" in r else None,
                "pushed_at": r["latest_crm_at"].isoformat() if "latest_crm_at" in r and r["latest_crm_at"] else None,
                "account_review_focus": latest_crm_account_review_focus,
            }
        webhooks.append({
            "id": str(r["id"]),
            "url": r["url"],
            "event_types": r["event_types"],
            "channel": r["channel"],
            "enabled": r["enabled"],
            "description": r["description"],
            "created_at": r["created_at"].isoformat(),
            "updated_at": r["updated_at"].isoformat(),
            "recent_deliveries_7d": recent_total,
            "recent_success_rate_7d": round(r["recent_successes"] / max(recent_total, 1), 3) if recent_total else None,
            "latest_failure_event_type": r["latest_failure_event_type"],
            "latest_failure_status_code": r["latest_failure_status_code"],
            "latest_failure_error": r["latest_failure_error"],
            "latest_failure_at": r["latest_failure_at"].isoformat() if r["latest_failure_at"] else None,
            "latest_failure_signal_id": str(r["latest_failure_signal_id"]) if "latest_failure_signal_id" in r and r["latest_failure_signal_id"] else None,
            "latest_failure_review_id": str(r["latest_failure_review_id"]) if "latest_failure_review_id" in r and r["latest_failure_review_id"] else None,
            "latest_failure_report_id": str(r["latest_failure_report_id"]) if "latest_failure_report_id" in r and r["latest_failure_report_id"] else None,
            "latest_failure_vendor_name": r["latest_failure_vendor_name"] if "latest_failure_vendor_name" in r and r["latest_failure_vendor_name"] else None,
            "latest_failure_company_name": r["latest_failure_company_name"] if "latest_failure_company_name" in r and r["latest_failure_company_name"] else None,
            "latest_test_success": latest_test_success,
            "latest_test_status_code": latest_test_status_code,
            "latest_test_error": latest_test_error,
            "latest_test_at": latest_test_at.isoformat() if latest_test_at else None,
            "latest_test_signal_id": str(r["latest_test_signal_id"]) if "latest_test_signal_id" in r and r["latest_test_signal_id"] else None,
            "latest_test_review_id": str(r["latest_test_review_id"]) if "latest_test_review_id" in r and r["latest_test_review_id"] else None,
            "latest_test_report_id": str(r["latest_test_report_id"]) if "latest_test_report_id" in r and r["latest_test_report_id"] else None,
            "latest_test_vendor_name": r["latest_test_vendor_name"] if "latest_test_vendor_name" in r and r["latest_test_vendor_name"] else None,
            "latest_test_company_name": r["latest_test_company_name"] if "latest_test_company_name" in r and r["latest_test_company_name"] else None,
            "latest_crm_push": latest_crm_push,
        })

    return {"webhooks": webhooks, "count": len(webhooks)}


# ---------------------------------------------------------------------------
# GET /webhooks/delivery-summary -- aggregate delivery health
# MUST be defined BEFORE /webhooks/{webhook_id} to avoid route shadowing
# ---------------------------------------------------------------------------


@router.get("/webhooks/delivery-summary")
async def webhook_delivery_summary(
    days: int = Query(7, ge=1, le=90),
    user: AuthUser | None = Depends(optional_auth),
):
    """Aggregate delivery health across all webhooks for the authenticated account."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    pool = _pool_or_503()

    row = await pool.fetchrow(
        """
        SELECT
            COUNT(DISTINCT ws.id) AS active_subscriptions,
            COUNT(dl.id) AS total_deliveries,
            COUNT(dl.id) FILTER (WHERE dl.success) AS successful,
            COUNT(dl.id) FILTER (WHERE NOT dl.success) AS failed,
            AVG(dl.duration_ms) FILTER (WHERE dl.success) AS avg_success_duration_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY dl.duration_ms)
                FILTER (WHERE dl.success) AS p95_success_duration_ms,
            MAX(dl.delivered_at) AS last_delivery_at
        FROM b2b_webhook_subscriptions ws
        LEFT JOIN b2b_webhook_delivery_log dl
            ON dl.subscription_id = ws.id
            AND dl.delivered_at > NOW() - ($2 || ' days')::interval
        WHERE ws.account_id = $1::uuid
          AND ws.enabled = true
        """,
        user.account_id, str(days),
    )

    total = row["total_deliveries"] or 0
    return {
        "window_days": days,
        "active_subscriptions": row["active_subscriptions"] or 0,
        "total_deliveries": total,
        "successful": row["successful"] or 0,
        "failed": row["failed"] or 0,
        "success_rate": round((row["successful"] or 0) / max(total, 1), 3) if total else None,
        "avg_success_duration_ms": round(row["avg_success_duration_ms"], 1) if row["avg_success_duration_ms"] else None,
        "p95_success_duration_ms": round(row["p95_success_duration_ms"], 1) if row["p95_success_duration_ms"] else None,
        "last_delivery_at": row["last_delivery_at"].isoformat() if row["last_delivery_at"] else None,
    }


@router.get("/webhooks/{webhook_id}")
async def get_webhook(
    webhook_id: str,
    user: AuthUser | None = Depends(optional_auth),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        wid = _uuid.UUID(webhook_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="webhook_id must be a valid UUID")

    pool = _pool_or_503()
    row = await pool.fetchrow(
        """
        SELECT id, url, event_types,
               COALESCE(channel, 'generic') AS channel,
               enabled, description, created_at, updated_at
        FROM b2b_webhook_subscriptions
        WHERE id = $1 AND account_id = $2::uuid
        """,
        wid, user.account_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return {
        "id": str(row["id"]),
        "url": row["url"],
        "event_types": row["event_types"],
        "channel": row["channel"],
        "enabled": row["enabled"],
        "description": row["description"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    user: AuthUser | None = Depends(optional_auth),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        wid = _uuid.UUID(webhook_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="webhook_id must be a valid UUID")

    pool = _pool_or_503()
    result = await pool.execute(
        "DELETE FROM b2b_webhook_subscriptions WHERE id = $1 AND account_id = $2::uuid",
        wid, user.account_id,
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Webhook not found")

    return {"deleted": True, "id": webhook_id}


@router.patch("/webhooks/{webhook_id}")
async def update_webhook(
    webhook_id: str,
    body: UpdateWebhookBody,
    user: AuthUser | None = Depends(optional_auth),
):
    """Update webhook subscription fields (enabled, event_types, url, description)."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        wid = _uuid.UUID(webhook_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="webhook_id must be a valid UUID")

    if body.url is not None and not body.url.startswith(("https://", "http://")):
        raise HTTPException(status_code=400, detail="url must begin with https:// or http://")
    if (
        body.enabled is None
        and body.event_types is None
        and body.url is None
        and body.description is None
    ):
        raise HTTPException(status_code=400, detail="No fields to update")

    pool = _pool_or_503()
    # Build dynamic SET clause
    sets: list[str] = []
    params: list = []
    idx = 1

    if body.enabled is not None:
        sets.append(f"enabled = ${idx}")
        params.append(body.enabled)
        idx += 1
    if body.event_types is not None:
        current_channel = await pool.fetchval(
            "SELECT COALESCE(channel, 'generic') AS channel FROM b2b_webhook_subscriptions WHERE id = $1 AND account_id = $2::uuid",
            wid,
            user.account_id,
        )
        if current_channel is None:
            raise HTTPException(status_code=404, detail="Webhook not found")
        _validate_webhook_event_types_for_channel(body.event_types, current_channel)
        sets.append(f"event_types = ${idx}")
        params.append(body.event_types)
        idx += 1
    if body.url is not None:
        sets.append(f"url = ${idx}")
        params.append(body.url)
        idx += 1
    if body.description is not None:
        sets.append(f"description = ${idx}")
        params.append(body.description)
        idx += 1

    if not sets:
        raise HTTPException(status_code=400, detail="No fields to update")

    sets.append("updated_at = NOW()")
    params.extend([wid, user.account_id])

    row = await pool.fetchrow(
        f"""
        UPDATE b2b_webhook_subscriptions
        SET {', '.join(sets)}
        WHERE id = ${idx} AND account_id = ${idx + 1}::uuid
        RETURNING id, url, event_types, COALESCE(channel, 'generic') AS channel,
                  enabled, description, created_at, updated_at
        """,
        *params,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return {
        "id": str(row["id"]),
        "url": row["url"],
        "event_types": row["event_types"],
        "channel": row["channel"],
        "enabled": row["enabled"],
        "description": row["description"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


@router.get("/webhooks/{webhook_id}/deliveries")
async def list_webhook_deliveries(
    webhook_id: str,
    success: Optional[bool] = Query(None, description="Filter by delivery success"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    start_date: Optional[str] = Query(None, description="Deliveries on or after (ISO 8601)"),
    end_date: Optional[str] = Query(None, description="Deliveries before (ISO 8601)"),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        wid = _uuid.UUID(webhook_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="webhook_id must be a valid UUID")

    event_type = _optional_query_text(event_type)
    start_date = _optional_query_text(start_date)
    end_date = _optional_query_text(end_date)

    parsed_start_date = None
    parsed_end_date = None
    if start_date:
        try:
            parsed_start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date (ISO 8601 expected)")
    if end_date:
        try:
            parsed_end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date (ISO 8601 expected)")

    pool = _pool_or_503()

    # Verify ownership
    owns = await pool.fetchval(
        "SELECT 1 FROM b2b_webhook_subscriptions WHERE id = $1 AND account_id = $2::uuid",
        wid, user.account_id,
    )
    if not owns:
        raise HTTPException(status_code=404, detail="Webhook not found")

    conditions = ["subscription_id = $1"]
    params: list = [wid]
    idx = 2

    if success is not None:
        conditions.append(f"success = ${idx}")
        params.append(success)
        idx += 1
    if event_type:
        conditions.append(f"event_type = ${idx}")
        params.append(event_type)
        idx += 1
    if parsed_start_date is not None:
        conditions.append(f"delivered_at >= ${idx}")
        params.append(parsed_start_date)
        idx += 1
    if parsed_end_date is not None:
        conditions.append(f"delivered_at < ${idx}")
        params.append(parsed_end_date)
        idx += 1

    where = " AND ".join(conditions)
    params.append(limit)

    rows = await pool.fetch(
        f"""
        SELECT id, event_type, payload, signal_id, review_id, report_id,
               vendor_name, company_name, status_code, duration_ms,
               attempt, success, error, delivered_at
        FROM b2b_webhook_delivery_log
        WHERE {where}
        ORDER BY delivered_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    report_cache: dict[str, Any] = {}
    tracked_vendor_cache: dict[str, Any] = {}
    signal_cache: dict[str, Any] = {}
    report_activity_cache: dict[str, Any] = {}
    deliveries = []
    for r in rows:
        context = _extract_webhook_activity_context(r.get("payload"))
        signal_id = _normalize_webhook_activity_uuid_text(r.get("signal_id")) or context.get("signal_id")
        signal_context = None
        if signal_id:
            signal_context = signal_cache.get(signal_id)
            if signal_context is None and signal_id not in signal_cache:
                signal_context = await _fetch_company_signal_focus_context(pool, signal_id)
                signal_cache[signal_id] = signal_context
        report_id = _normalize_webhook_activity_uuid_text(r.get("report_id")) or context.get("report_id")
        report_context = await _fetch_webhook_activity_report_context(
            pool,
            report_id,
            report_activity_cache,
        )
        resolved_review_id = (
            _normalize_webhook_activity_uuid_text(r.get("review_id"))
            or context.get("review_id")
            or (signal_context or {}).get("review_id")
        )
        vendor_name = _optional_query_text(r.get("vendor_name")) or context.get("vendor_name") or (report_context or {}).get("vendor_name")
        company_name = _optional_query_text(r.get("company_name")) or context.get("company_name") or (signal_context or {}).get("company_name")
        account_review_focus = await _resolve_webhook_activity_account_focus(
            pool,
            user,
            vendor_name=vendor_name,
            company_name=company_name,
            signal_id=signal_id,
            review_id=resolved_review_id,
            report_cache=report_cache,
            tracked_vendor_cache=tracked_vendor_cache,
            signal_cache=signal_cache,
        )
        deliveries.append({
            "id": str(r["id"]),
            "event_type": r["event_type"],
            "status_code": r["status_code"],
            "duration_ms": r["duration_ms"],
            "attempt": r["attempt"],
            "success": r["success"],
            "error": r["error"],
            "delivered_at": r["delivered_at"].isoformat(),
            "vendor_name": vendor_name,
            "company_name": company_name,
            "signal_id": signal_id,
            "signal_type": context.get("signal_type") or r["event_type"],
            "review_id": resolved_review_id,
            "report_id": report_id or (report_context or {}).get("report_id"),
            "report_type": context.get("report_type") or (report_context or {}).get("report_type"),
            "report_title": context.get("report_title") or (report_context or {}).get("report_title"),
            "account_review_focus": account_review_focus,
        })

    return {"deliveries": deliveries, "count": len(deliveries)}


@router.post("/webhooks/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    user: AuthUser | None = Depends(optional_auth),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        wid = _uuid.UUID(webhook_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="webhook_id must be a valid UUID")

    pool = _pool_or_503()

    # Verify ownership
    owns = await pool.fetchval(
        "SELECT 1 FROM b2b_webhook_subscriptions WHERE id = $1 AND account_id = $2::uuid",
        wid, user.account_id,
    )
    if not owns:
        raise HTTPException(status_code=404, detail="Webhook not found")

    from ..services.b2b.webhook_dispatcher import send_test_webhook
    result = await send_test_webhook(pool, wid)
    return result


# ---------------------------------------------------------------------------
# GET /webhooks/{webhook_id}/crm-push-log -- CRM push history
# ---------------------------------------------------------------------------


@router.get("/webhooks/{webhook_id}/crm-push-log")
async def list_crm_push_log(
    webhook_id: str,
    limit: int = 50,
    user: AuthUser | None = Depends(optional_auth),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        wid = _uuid.UUID(webhook_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="webhook_id must be a valid UUID")

    limit = max(1, min(limit, 200))
    pool = _pool_or_503()

    # Verify ownership
    owns = await pool.fetchval(
        "SELECT 1 FROM b2b_webhook_subscriptions WHERE id = $1 AND account_id = $2::uuid",
        wid, user.account_id,
    )
    if not owns:
        raise HTTPException(status_code=404, detail="Webhook not found")

    rows = await pool.fetch(
        """
        SELECT id, signal_type, signal_id, review_id, vendor_name, company_name,
               crm_record_id, crm_record_type, status, error, pushed_at
        FROM b2b_crm_push_log
        WHERE subscription_id = $1
        ORDER BY pushed_at DESC
        LIMIT $2
        """,
        wid, limit,
    )

    report_cache: dict[str, Any] = {}
    tracked_vendor_cache: dict[str, Any] = {}
    signal_cache: dict[str, Any] = {}
    report_activity_cache: dict[str, Any] = {}
    pushes = []
    for r in rows:
        signal_type = str(r["signal_type"] or "").strip()
        signal_id = str(r["signal_id"]) if r["signal_id"] else None
        signal_context = None
        if signal_id and signal_type != "report_generated":
            signal_context = signal_cache.get(signal_id)
            if signal_context is None and signal_id not in signal_cache:
                signal_context = await _fetch_company_signal_focus_context(pool, signal_id)
                signal_cache[signal_id] = signal_context
        report_context = None
        if signal_type == "report_generated" and signal_id:
            report_context = await _fetch_webhook_activity_report_context(pool, signal_id, report_activity_cache)
        resolved_review_id = _normalize_webhook_activity_uuid_text(r.get("review_id")) or (signal_context or {}).get("review_id")
        account_review_focus = await _resolve_webhook_activity_account_focus(
            pool,
            user,
            vendor_name=r["vendor_name"],
            company_name=r["company_name"],
            signal_id=None if signal_type == "report_generated" else signal_id,
            review_id=resolved_review_id,
            report_cache=report_cache,
            tracked_vendor_cache=tracked_vendor_cache,
            signal_cache=signal_cache,
        )
        pushes.append({
            "id": str(r["id"]),
            "signal_type": signal_type,
            "signal_id": signal_id,
            "vendor_name": r["vendor_name"] or (signal_context or {}).get("vendor_name") or (report_context or {}).get("vendor_name"),
            "company_name": r["company_name"] or (signal_context or {}).get("company_name"),
            "review_id": resolved_review_id,
            "report_id": signal_id if signal_type == "report_generated" else None,
            "report_type": (report_context or {}).get("report_type"),
            "report_title": (report_context or {}).get("report_title"),
            "crm_record_id": r["crm_record_id"],
            "crm_record_type": r["crm_record_type"],
            "status": _normalize_crm_push_status(r["status"]),
            "error": r["error"],
            "pushed_at": r["pushed_at"].isoformat(),
            "account_review_focus": account_review_focus,
        })

    return {"pushes": pushes, "count": len(pushes)}


# ---------------------------------------------------------------------------
# POST /corrections -- Create a data correction
# ---------------------------------------------------------------------------


@router.post("/corrections")
async def create_correction(
    body: CreateCorrectionBody,
    user: AuthUser | None = Depends(optional_auth),
):
    metadata = dict(body.metadata or {})
    metadata_source_name = _optional_query_text(metadata.get("source_name"))
    if "source_name" in metadata:
        if metadata_source_name is None:
            metadata.pop("source_name", None)
        else:
            metadata["source_name"] = metadata_source_name

    span = tracer.start_span(
        span_name="b2b.correction.create",
        operation_type="business_operation",
        session_id=str(user.account_id) if user else None,
        metadata={
            "business": build_business_trace_context(
                account_id=str(user.account_id) if user else None,
                workflow="analyst_correction",
                entity_type=body.entity_type,
                correction_type=body.correction_type,
                vendor_name=body.new_value if body.correction_type == "merge_vendor" else None,
                source_name=metadata_source_name,
            ),
        },
    )

    if body.entity_type not in VALID_ENTITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity_type. Must be one of: {sorted(VALID_ENTITY_TYPES)}",
        )
    if body.correction_type not in VALID_CORRECTION_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid correction_type. Must be one of: {sorted(VALID_CORRECTION_TYPES)}",
        )
    if body.correction_type == "override_field":
        if not body.field_name:
            raise HTTPException(status_code=400, detail="field_name required for override_field")
        if body.new_value is None:
            raise HTTPException(status_code=400, detail="new_value required for override_field")
    if body.correction_type == "merge_vendor":
        if not body.old_value or not body.new_value:
            raise HTTPException(
                status_code=400,
                detail="merge_vendor requires old_value (source vendor) and new_value (target vendor)",
            )
    if body.correction_type == "suppress_source":
        if body.entity_type != "source":
            raise HTTPException(
                status_code=400,
                detail="suppress_source corrections must use entity_type='source'",
            )
        if not metadata_source_name:
            raise HTTPException(
                status_code=400,
                detail="suppress_source requires metadata.source_name (e.g., 'reddit', 'g2')",
            )
        if metadata_source_name.lower() not in _KNOWN_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source '{metadata_source_name}'. Known sources: {sorted(_KNOWN_SOURCES)}",
            )

    try:
        entity_uuid = _uuid.UUID(body.entity_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="entity_id must be a valid UUID")

    pool = _pool_or_503()
    corrected_by = f"api:{user.user_id}" if user else "analyst"
    meta = json.dumps(metadata) if metadata else "{}"

    row = await pool.fetchrow(
        """
        INSERT INTO data_corrections
            (entity_type, entity_id, correction_type, field_name,
             old_value, new_value, reason, corrected_by, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
        RETURNING id, entity_type, entity_id, correction_type, status, created_at
        """,
        body.entity_type,
        entity_uuid,
        body.correction_type,
        body.field_name,
        body.old_value,
        body.new_value,
        body.reason,
        corrected_by,
        meta,
    )

    correction_id = row["id"]

    # Execute vendor merge if applicable
    if body.correction_type == "merge_vendor":
        from ..services.b2b.vendor_merge import execute_vendor_merge
        merge_result = await execute_vendor_merge(pool, body.old_value, body.new_value)
        await pool.execute(
            "UPDATE data_corrections SET affected_count = $1, metadata = $2::jsonb WHERE id = $3",
            merge_result["total_affected"], json.dumps(merge_result), correction_id,
        )

    response = {
        "id": str(correction_id),
        "entity_type": row["entity_type"],
        "entity_id": str(row["entity_id"]),
        "correction_type": row["correction_type"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat(),
    }
    tracer.end_span(
        span,
        status="completed",
        output_data=response,
        metadata={
            "reasoning": build_reasoning_trace_context(
                decision={
                    "entity_type": body.entity_type,
                    "correction_type": body.correction_type,
                    "field_name": body.field_name,
                },
                evidence={
                    "old_value": body.old_value,
                    "new_value": body.new_value,
                    "reason": body.reason,
                },
            ),
        },
    )
    return response


# ---------------------------------------------------------------------------
# GET /corrections -- List corrections with filters
# ---------------------------------------------------------------------------


@router.get("/corrections")
async def list_corrections(
    entity_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    correction_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    corrected_by: Optional[str] = Query(None, description="Filter by who made the correction (substring match)"),
    start_date: Optional[str] = Query(None, description="Corrections created on or after (ISO 8601)"),
    end_date: Optional[str] = Query(None, description="Corrections created before (ISO 8601)"),
    limit: int = Query(50, ge=1, le=200),
    user: AuthUser | None = Depends(optional_auth),
):
    entity_type = _optional_query_text(entity_type)
    entity_id = _optional_query_text(entity_id)
    correction_type = _optional_query_text(correction_type)
    status = _optional_query_text(status)
    corrected_by = _optional_query_text(corrected_by)
    start_date = _optional_query_text(start_date)
    end_date = _optional_query_text(end_date)
    conditions: list[str] = []
    params: list = []
    idx = 1

    if entity_type:
        if entity_type not in VALID_ENTITY_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entity_type. Must be one of: {sorted(VALID_ENTITY_TYPES)}",
            )
        conditions.append(f"entity_type = ${idx}")
        params.append(entity_type)
        idx += 1

    if entity_id:
        try:
            eid = _uuid.UUID(entity_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="entity_id must be a valid UUID")
        conditions.append(f"entity_id = ${idx}")
        params.append(eid)
        idx += 1

    if correction_type:
        if correction_type not in VALID_CORRECTION_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid correction_type. Must be one of: {sorted(VALID_CORRECTION_TYPES)}",
            )
        conditions.append(f"correction_type = ${idx}")
        params.append(correction_type)
        idx += 1

    if status:
        if status not in VALID_CORRECTION_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {sorted(VALID_CORRECTION_STATUSES)}",
            )
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if corrected_by:
        conditions.append(f"corrected_by ILIKE '%' || ${idx} || '%'")
        params.append(corrected_by)
        idx += 1

    if start_date:
        try:
            sd = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date (ISO 8601 expected)")
        conditions.append(f"created_at >= ${idx}")
        params.append(sd)
        idx += 1

    if end_date:
        try:
            ed = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date (ISO 8601 expected)")
        conditions.append(f"created_at < ${idx}")
        params.append(ed)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    pool = _pool_or_503()
    rows = await pool.fetch(
        f"""
        SELECT id, entity_type, entity_id, correction_type, field_name,
               old_value, new_value, reason, corrected_by, status,
               affected_count, metadata, created_at, reverted_at, reverted_by
        FROM data_corrections
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    corrections = []
    for r in rows:
        corrections.append({
            "id": str(r["id"]),
            "entity_type": r["entity_type"],
            "entity_id": str(r["entity_id"]),
            "correction_type": r["correction_type"],
            "field_name": r["field_name"],
            "old_value": r["old_value"],
            "new_value": r["new_value"],
            "reason": r["reason"],
            "corrected_by": r["corrected_by"],
            "status": r["status"],
            "affected_count": r["affected_count"],
            "metadata": _safe_json(r["metadata"]),
            "created_at": r["created_at"].isoformat(),
            "reverted_at": r["reverted_at"].isoformat() if r["reverted_at"] else None,
            "reverted_by": r["reverted_by"],
        })

    return {"corrections": corrections, "count": len(corrections)}


# ---------------------------------------------------------------------------
# GET /corrections/stats -- Aggregate correction activity
# MUST be defined BEFORE /corrections/{correction_id} to avoid route shadowing
# ---------------------------------------------------------------------------


@router.get("/corrections/stats")
async def get_correction_stats(
    days: int = Query(30, ge=1, le=365),
    user: AuthUser | None = Depends(optional_auth),
):
    """Aggregate correction activity: counts by type, status, and top correctors."""
    pool = _pool_or_503()

    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_corrections,
            COUNT(*) FILTER (WHERE status = 'applied') AS applied,
            COUNT(*) FILTER (WHERE status = 'reverted') AS reverted,
            COUNT(*) FILTER (WHERE status = 'pending_review') AS pending_review,
            COUNT(*) FILTER (WHERE correction_type = 'suppress') AS suppress_count,
            COUNT(*) FILTER (WHERE correction_type = 'flag') AS flag_count,
            COUNT(*) FILTER (WHERE correction_type = 'override_field') AS override_count,
            COUNT(*) FILTER (WHERE correction_type = 'merge_vendor') AS merge_count,
            COUNT(*) FILTER (WHERE correction_type = 'reclassify') AS reclassify_count,
            COUNT(*) FILTER (WHERE correction_type = 'suppress_source') AS suppress_source_count,
            COALESCE(SUM(affected_count), 0) AS total_affected_records,
            MIN(created_at) AS first_correction_at,
            MAX(created_at) AS last_correction_at
        FROM data_corrections
        WHERE created_at > NOW() - ($1 || ' days')::interval
        """,
        str(days),
    )

    # Top correctors
    correctors = await pool.fetch(
        """
        SELECT corrected_by, COUNT(*) AS count
        FROM data_corrections
        WHERE created_at > NOW() - ($1 || ' days')::interval
        GROUP BY corrected_by
        ORDER BY count DESC
        LIMIT 10
        """,
        str(days),
    )

    # Corrections by entity type
    by_entity = await pool.fetch(
        """
        SELECT entity_type, COUNT(*) AS count
        FROM data_corrections
        WHERE created_at > NOW() - ($1 || ' days')::interval
        GROUP BY entity_type
        ORDER BY count DESC
        """,
        str(days),
    )

    return {
        "window_days": days,
        "total_corrections": row["total_corrections"],
        "by_status": {
            "applied": row["applied"],
            "reverted": row["reverted"],
            "pending_review": row["pending_review"],
        },
        "by_type": {
            "suppress": row["suppress_count"],
            "flag": row["flag_count"],
            "override_field": row["override_count"],
            "merge_vendor": row["merge_count"],
            "reclassify": row["reclassify_count"],
            "suppress_source": row["suppress_source_count"],
        },
        "by_entity": [{"entity_type": r["entity_type"], "count": r["count"]} for r in by_entity],
        "total_affected_records": row["total_affected_records"],
        "top_correctors": [{"corrected_by": r["corrected_by"], "count": r["count"]} for r in correctors],
        "first_correction_at": row["first_correction_at"].isoformat() if row["first_correction_at"] else None,
        "last_correction_at": row["last_correction_at"].isoformat() if row["last_correction_at"] else None,
    }


# ---------------------------------------------------------------------------
# GET /corrections/{correction_id} -- Get single correction
# ---------------------------------------------------------------------------


@router.get("/corrections/{correction_id}")
async def get_correction(
    correction_id: str,
    user: AuthUser | None = Depends(optional_auth),
):
    try:
        cid = _uuid.UUID(correction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="correction_id must be a valid UUID")

    pool = _pool_or_503()
    row = await pool.fetchrow(
        """
        SELECT id, entity_type, entity_id, correction_type, field_name,
               old_value, new_value, reason, corrected_by, status,
               affected_count, metadata, created_at, reverted_at, reverted_by
        FROM data_corrections
        WHERE id = $1
        """,
        cid,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Correction not found")

    return {
        "id": str(row["id"]),
        "entity_type": row["entity_type"],
        "entity_id": str(row["entity_id"]),
        "correction_type": row["correction_type"],
        "field_name": row["field_name"],
        "old_value": row["old_value"],
        "new_value": row["new_value"],
        "reason": row["reason"],
        "corrected_by": row["corrected_by"],
        "status": row["status"],
        "affected_count": row["affected_count"],
        "metadata": _safe_json(row["metadata"]),
        "created_at": row["created_at"].isoformat(),
        "reverted_at": row["reverted_at"].isoformat() if row["reverted_at"] else None,
        "reverted_by": row["reverted_by"],
    }


# ---------------------------------------------------------------------------
# POST /corrections/{correction_id}/revert -- Revert a correction
# ---------------------------------------------------------------------------


@router.post("/corrections/{correction_id}/revert")
async def revert_correction(
    correction_id: str,
    body: RevertCorrectionBody,
    user: AuthUser | None = Depends(optional_auth),
):
    try:
        cid = _uuid.UUID(correction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="correction_id must be a valid UUID")

    pool = _pool_or_503()
    row = await pool.fetchrow(
        "SELECT id, status FROM data_corrections WHERE id = $1",
        cid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Correction not found")
    if row["status"] != "applied":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot revert correction with status '{row['status']}' (must be 'applied')",
        )

    reverted_by = f"api:{user.user_id}" if user else "analyst"

    updated = await pool.fetchrow(
        """
        UPDATE data_corrections
        SET status = 'reverted', reverted_at = NOW(), reverted_by = $2
        WHERE id = $1
        RETURNING id, status, reverted_at
        """,
        cid,
        reverted_by,
    )

    return {
        "id": str(updated["id"]),
        "status": updated["status"],
        "reverted_at": updated["reverted_at"].isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /source-corrections/impact -- Show impact of active source suppressions
# ---------------------------------------------------------------------------


@router.get("/source-corrections/impact")
async def get_source_correction_impact(
    user: AuthUser | None = Depends(optional_auth),
):
    """Show impact of active source suppressions: how many reviews are excluded per source."""
    pool = _pool_or_503()
    rows = await pool.fetch(
        """
        SELECT dc.metadata->>'source_name' AS source_name,
               dc.field_name AS vendor_scope,
               dc.reason,
               dc.created_at,
               (SELECT COUNT(*) FROM b2b_reviews r
                WHERE LOWER(r.source) = LOWER(dc.metadata->>'source_name')
                  AND (dc.field_name IS NULL OR LOWER(r.vendor_name) = LOWER(dc.field_name))
                  AND r.enrichment_status = 'enriched'
               ) AS affected_review_count
        FROM data_corrections dc
        WHERE dc.entity_type = 'source'
          AND dc.correction_type = 'suppress_source'
          AND dc.status = 'applied'
        ORDER BY dc.created_at DESC
        """,
    )
    return {
        "active_source_suppressions": [
            {
                "source_name": r["source_name"],
                "vendor_scope": r["vendor_scope"],
                "reason": r["reason"],
                "affected_review_count": r["affected_review_count"],
                "created_at": str(r["created_at"]),
            }
            for r in rows
        ],
        "total": len(rows),
    }


# ---------------------------------------------------------------------------
# GET /accounts-in-motion
# ---------------------------------------------------------------------------


def _company_lookup_key(company: str | None) -> str:
    return (company or "").strip().lower()


async def _fetch_latest_accounts_in_motion_report(
    pool,
    vendor_name: str,
    user: AuthUser | None,
):
    params: list[Any] = [vendor_name.strip()]
    conditions = [
        "report_type = 'accounts_in_motion'",
        "(LOWER(vendor_filter) = LOWER($1) OR vendor_filter ILIKE '%' || $1 || '%')",
    ]
    idx = 2
    if _should_scope(user):
        conditions.append(
            f"LOWER(vendor_filter) IN (SELECT LOWER(vendor_name) FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(user.account_id)
        idx += 1
    where = " AND ".join(conditions)
    return await pool.fetchrow(
        f"""
        SELECT report_date, vendor_filter, intelligence_data,
               status, latest_failure_step, latest_error_summary, created_at
        FROM b2b_intelligence
        WHERE {where}
        ORDER BY CASE WHEN LOWER(vendor_filter) = LOWER($1) THEN 0 ELSE 1 END,
                 report_date DESC, created_at DESC
        LIMIT 1
        """,
        *params,
    )


async def _fetch_accounts_in_motion_org_lookup(pool, company_keys: list[str]) -> dict[str, dict[str, Any]]:
    if not company_keys:
        return {}
    rows = await pool.fetch(
        """
        SELECT company_name_norm, employee_count, industry, annual_revenue_range, domain
        FROM prospect_org_cache
        WHERE company_name_norm = ANY($1::text[])
          AND (status = 'enriched' OR (status = 'pending' AND domain IS NOT NULL))
        """,
        company_keys,
    )
    return {
        r["company_name_norm"]: {
            "employee_count": r["employee_count"],
            "industry": r["industry"],
            "annual_revenue_range": r["annual_revenue_range"],
            "domain": r["domain"],
        }
        for r in rows
    }


async def _fetch_accounts_in_motion_contacts_lookup(
    pool,
    company_keys: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not company_keys:
        return {}
    rows = await pool.fetch(
        """
        SELECT LOWER(company_name) AS company_key,
               first_name || ' ' || last_name AS name,
               title, seniority, email, linkedin_url
        FROM prospects
        WHERE LOWER(company_name) = ANY($1::text[])
          AND seniority IN ('c_suite', 'owner', 'founder', 'vp', 'head', 'director')
          AND email IS NOT NULL
        ORDER BY LOWER(company_name),
                 CASE seniority
                     WHEN 'c_suite' THEN 1 WHEN 'founder' THEN 2
                     WHEN 'owner' THEN 2 WHEN 'vp' THEN 3
                     WHEN 'head' THEN 4 WHEN 'director' THEN 5
                 END
        """,
        company_keys,
    )
    contacts_lookup: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        company_key = row["company_key"]
        bucket = contacts_lookup.setdefault(company_key, [])
        if len(bucket) >= 5:
            continue
        bucket.append(
            {
                "name": row["name"],
                "title": row["title"],
                "seniority": row["seniority"],
                "email": row["email"],
                "linkedin_url": row["linkedin_url"],
            }
        )
    return contacts_lookup


async def _fetch_accounts_in_motion_review_lookup(
    pool,
    review_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not review_ids:
        return {}
    parsed_ids: list[_uuid.UUID] = []
    seen: set[str] = set()
    for review_id in review_ids:
        raw = str(review_id or "").strip()
        if not raw or raw in seen:
            continue
        try:
            parsed_ids.append(_uuid.UUID(raw))
            seen.add(raw)
        except (TypeError, ValueError, AttributeError):
            continue
    if not parsed_ids:
        return {}
    rows = await pool.fetch(
        """
        SELECT id, source, source_url, vendor_name, rating, summary,
               LEFT(review_text, 500) AS review_excerpt,
               reviewer_name, reviewer_title, reviewer_company, reviewed_at
        FROM b2b_reviews
        WHERE id = ANY($1::uuid[])
        ORDER BY reviewed_at DESC NULLS LAST
        """,
        parsed_ids,
    )
    return {
        str(row["id"]): {
            "id": str(row["id"]),
            "source": row["source"],
            "source_url": row["source_url"],
            "vendor_name": row["vendor_name"],
            "rating": _safe_float(row["rating"]),
            "summary": row["summary"],
            "review_excerpt": row["review_excerpt"],
            "reviewer_name": row["reviewer_name"],
            "reviewer_title": row["reviewer_title"],
            "reviewer_company": row["reviewer_company"],
            "reviewed_at": row["reviewed_at"].isoformat() if row["reviewed_at"] else None,
        }
        for row in rows
    }


def _shape_persisted_accounts_in_motion_account(account: dict[str, Any], vendor_name: str) -> dict[str, Any]:
    alternatives = [
        {"name": name, "reason": ""}
        for name in (account.get("alternatives_considering") or [])
        if isinstance(name, str) and name.strip()
    ][:5]
    pain_categories = []
    if account.get("pain_category"):
        pain_categories.append({"category": account["pain_category"], "severity": ""})
    evidence = [account["top_quote"]] if account.get("top_quote") else []
    source_review_ids = [
        str(review_id).strip()
        for review_id in (account.get("source_reviews") or [])
        if str(review_id or "").strip()
    ]
    return {
        "company": account.get("company"),
        "vendor": account.get("vendor") or vendor_name,
        "category": account.get("category"),
        "urgency": _safe_float(account.get("urgency"), 0),
        "role_type": account.get("role_level"),
        "buying_stage": account.get("buying_stage"),
        "budget_authority": bool(account.get("decision_maker")),
        "pain_categories": pain_categories,
        "evidence": evidence,
        "alternatives_considering": alternatives,
        "contract_signal": account.get("contract_end"),
        "reviewer_title": account.get("title"),
        "company_size_raw": account.get("company_size"),
        "quality_flags": account.get("quality_flags") or [],
        "opportunity_score": account.get("opportunity_score"),
        "quote_match_type": account.get("quote_match_type"),
        "confidence": account.get("confidence"),
        "source_distribution": account.get("source_distribution") or {},
        "source_review_ids": source_review_ids,
        "source_reviews": [],
        "evidence_count": int(account.get("evidence_count") or 0),
        "enriched_at": account.get("last_seen"),
    }


def _accounts_in_motion_confidence_band(account: dict[str, Any]) -> str:
    company = str(account.get("company") or "").strip()
    confidence = _safe_float(account.get("confidence"))
    quality_flags = account.get("quality_flags")
    if not company:
        return "review"
    if isinstance(quality_flags, list) and quality_flags:
        return "review"
    if confidence is not None and confidence >= 7:
        return "high"
    if confidence is not None and confidence >= 4:
        return "medium"
    return "review"


def _build_accounts_in_motion_evidence_items(
    account: dict[str, Any],
    source_reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    match_type = str(account.get("quote_match_type") or "").strip() or None
    reference_ids = account.get("reasoning_reference_ids")
    witness_ids = []
    if isinstance(reference_ids, dict):
        witness_ids = [
            str(item).strip()
            for item in (reference_ids.get("witness_ids") or [])
            if str(item or "").strip()
        ]

    for review in source_reviews:
        quote = str(review.get("review_excerpt") or review.get("summary") or "").strip()
        if not quote:
            continue
        dedupe_key = quote.lower()
        if dedupe_key in seen_quotes:
            continue
        seen_quotes.add(dedupe_key)
        items.append(
            {
                "quote": quote,
                "review_id": str(review.get("id") or "").strip() or None,
                "source": review.get("source"),
                "source_url": review.get("source_url"),
                "reviewed_at": review.get("reviewed_at"),
                "reviewer_company": review.get("reviewer_company"),
                "reviewer_title": review.get("reviewer_title"),
                "match_type": match_type,
                "witness_ids": witness_ids or None,
            }
        )

    for quote in account.get("evidence") or []:
        cleaned = str(quote or "").strip()
        if not cleaned:
            continue
        dedupe_key = cleaned.lower()
        if dedupe_key in seen_quotes:
            continue
        seen_quotes.add(dedupe_key)
        items.append(
            {
                "quote": cleaned,
                "review_id": None,
                "source": None,
                "source_url": None,
                "reviewed_at": None,
                "reviewer_company": account.get("company"),
                "reviewer_title": account.get("reviewer_title"),
                "match_type": match_type,
                "witness_ids": witness_ids or None,
            }
        )
    return items


def _accounts_in_motion_report_freshness(
    row,
    *,
    report_date: str | None,
    stale_days: int | None,
) -> tuple[str, str | None, str | None]:
    report_status = str(row.get("status") or "").strip().lower()
    failure_step = str(row.get("latest_failure_step") or "").strip() or None
    error_summary = str(row.get("latest_error_summary") or "").strip() or None
    created_at = row.get("created_at")
    created_at_text = str(created_at) if created_at else None
    if report_status and report_status not in {"completed", "succeeded", "published", "sales_ready"}:
        reason = error_summary or failure_step or f"Latest report status is {report_status}"
        return "failed", reason, created_at_text or report_date
    if not report_date:
        return "artifact_missing", "Persisted accounts-in-motion report is missing a report date", created_at_text
    if stale_days and stale_days > 0:
        return "stale", f"Persisted accounts-in-motion report is {stale_days} day(s) old", report_date
    return "fresh", None, report_date


async def _enrich_accounts_in_motion_accounts(
    pool,
    accounts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    company_keys = sorted(
        {
            _company_lookup_key(account.get("company"))
            for account in accounts
            if _company_lookup_key(account.get("company"))
        }
    )
    review_ids = sorted(
        {
            str(review_id).strip()
            for account in accounts
            for review_id in (account.get("source_review_ids") or [])
            if str(review_id or "").strip()
        }
    )
    org_lookup, contacts_lookup, review_lookup = await asyncio.gather(
        _fetch_accounts_in_motion_org_lookup(pool, company_keys),
        _fetch_accounts_in_motion_contacts_lookup(pool, company_keys),
        _fetch_accounts_in_motion_review_lookup(pool, review_ids),
    )
    enriched: list[dict[str, Any]] = []
    for account in accounts:
        company_key = _company_lookup_key(account.get("company"))
        org = org_lookup.get(company_key) or {}
        contacts = contacts_lookup.get(company_key) or []
        source_reviews = [
            review_lookup[review_id]
            for review_id in (account.get("source_review_ids") or [])
            if review_id in review_lookup
        ]
        evidence_items = _build_accounts_in_motion_evidence_items(account, source_reviews)
        enriched.append(
            {
                **account,
                "employee_count": org.get("employee_count"),
                "industry": org.get("industry") or account.get("industry"),
                "annual_revenue": org.get("annual_revenue_range"),
                "domain": org.get("domain") or account.get("domain"),
                "source_reviews": source_reviews,
                "evidence_items": evidence_items,
                "evidence": [
                    item["quote"]
                    for item in evidence_items
                    if isinstance(item, dict) and item.get("quote")
                ],
                "confidence_band": _accounts_in_motion_confidence_band(account),
                "contacts": contacts,
                "contact_count": len(contacts),
            }
        )
    return enriched


async def _list_accounts_in_motion_from_report(
    pool,
    vendor_name: str,
    min_urgency: float,
    limit: int,
    user: AuthUser | None,
):
    row = await _fetch_latest_accounts_in_motion_report(pool, vendor_name, user)
    if not row:
        return None
    report = _safe_json(row["intelligence_data"])
    if not isinstance(report, dict):
        return None
    vendor = report.get("vendor") or row["vendor_filter"] or vendor_name.strip()
    reasoning_reference_ids = report.get("reference_ids")
    if not isinstance(reasoning_reference_ids, dict):
        reasoning_reference_ids = None
    shaped = [
        {
            **_shape_persisted_accounts_in_motion_account(account, vendor),
            "reasoning_reference_ids": reasoning_reference_ids,
        }
        for account in (report.get("accounts") or [])
        if isinstance(account, dict) and _safe_float(account.get("urgency"), 0) >= min_urgency
    ]
    shaped.sort(
        key=lambda account: (
            -(account.get("opportunity_score") or 0),
            -(account.get("urgency") or 0),
        )
    )
    accounts = await _enrich_accounts_in_motion_accounts(pool, shaped[:limit])
    report_date = str(row["report_date"]) if row["report_date"] else None
    stale_days = None
    if report_date:
        try:
            stale_days = max(0, (date.today() - date.fromisoformat(report_date[:10])).days)
        except (TypeError, ValueError):
            stale_days = None
    freshness_status, freshness_reason, freshness_timestamp = _accounts_in_motion_report_freshness(
        row,
        report_date=report_date,
        stale_days=stale_days,
    )
    return {
        "vendor": vendor,
        "accounts": accounts,
        "count": len(accounts),
        "window_days": settings.b2b_churn.intelligence_window_days,
        "min_urgency": min_urgency,
        "report_date": report_date,
        "stale_days": stale_days,
        "is_stale": bool(stale_days and stale_days > 0),
        "data_source": "persisted_report",
        "freshness_status": freshness_status,
        "freshness_reason": freshness_reason,
        "freshness_timestamp": freshness_timestamp,
    }


def _normalize_webhook_activity_uuid_text(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return str(_uuid.UUID(raw))
    except (ValueError, TypeError, AttributeError):
        return None


def _normalize_crm_push_status(value: Any) -> str | None:
    status = _optional_query_text(value)
    if status is None:
        return None
    normalized = status.lower()
    if normalized == "pushed":
        return "success"
    return normalized


def _extract_webhook_activity_context(payload: Any) -> dict[str, str | None]:
    envelope = _safe_json(payload)
    if not isinstance(envelope, dict):
        envelope = {}
    data = envelope.get("data")
    if not isinstance(data, dict):
        data = {}
    vendor_name = str(envelope.get("vendor") or data.get("vendor_name") or "").strip() or None
    company_name = str(data.get("company_name") or data.get("company") or "").strip() or None
    signal_type = str(data.get("signal_type") or "").strip() or None
    signal_id_value = data.get("company_signal_id")
    if signal_id_value is None:
        signal_id_value = data.get("signal_id")
    signal_id = _normalize_webhook_activity_uuid_text(signal_id_value)
    review_id = _normalize_webhook_activity_uuid_text(data.get("review_id"))
    report_id = _normalize_webhook_activity_uuid_text(data.get("report_id"))
    report_type = str(data.get("report_type") or data.get("artifact_type") or "").strip() or None
    report_title = str(data.get("report_title") or "").strip() or None
    return {
        "vendor_name": vendor_name,
        "company_name": company_name,
        "signal_type": signal_type,
        "signal_id": signal_id,
        "review_id": review_id,
        "report_id": report_id,
        "report_type": report_type,
        "report_title": report_title,
    }


def _format_webhook_activity_report_title(
    report_type: str | None,
    vendor_name: str | None,
    category_filter: str | None = None,
) -> str | None:
    report_type_text = str(report_type or "").strip()
    vendor_text = str(vendor_name or "").strip()
    category_text = str(category_filter or "").strip()
    if not report_type_text or not vendor_text:
        return None
    report_label = report_type_text.replace("_", " ").title()
    if report_type_text == "vendor_comparison" and category_text:
        return f"{vendor_text} vs {category_text}"
    if report_type_text == "challenger_brief" and category_text:
        return f"{vendor_text} vs {category_text} Challenger Brief"
    if category_text:
        return f"{vendor_text} {report_label}: {category_text}"
    return f"{vendor_text} {report_label}"


async def _fetch_webhook_activity_report_context(
    pool,
    report_id: str | None,
    report_cache: dict[str, Any] | None = None,
) -> dict[str, str | None] | None:
    report_key = str(report_id or "").strip()
    if not report_key:
        return None
    if report_cache is not None and report_key in report_cache:
        return report_cache[report_key]
    try:
        parsed = _uuid.UUID(report_key)
    except (ValueError, TypeError, AttributeError):
        if report_cache is not None:
            report_cache[report_key] = None
        return None
    row = await pool.fetchrow(
        """
        SELECT id, report_type, vendor_filter, category_filter
        FROM b2b_intelligence
        WHERE id = $1::uuid
        """,
        parsed,
    )
    if not row:
        if report_cache is not None:
            report_cache[report_key] = None
        return None
    context = {
        "report_id": str(row["id"]),
        "report_type": str(row["report_type"] or "").strip() or None,
        "vendor_name": str(row["vendor_filter"] or "").strip() or None,
        "category_filter": str(row["category_filter"] or "").strip() or None,
    }
    context["report_title"] = _format_webhook_activity_report_title(
        context.get("report_type"),
        context.get("vendor_name"),
        context.get("category_filter"),
    )
    if report_cache is not None:
        report_cache[report_key] = context
    return context


async def _fetch_company_signal_focus_context(
    pool,
    signal_id: str | None,
) -> dict[str, str | None] | None:
    raw = str(signal_id or "").strip()
    if not raw:
        return None
    try:
        parsed = _uuid.UUID(raw)
    except (ValueError, TypeError, AttributeError):
        return None
    row = await pool.fetchrow(
        """
        SELECT id, company_name, vendor_name, review_id
        FROM b2b_company_signals
        WHERE id = $1::uuid
        """,
        parsed,
    )
    if not row:
        return None
    return {
        "signal_id": str(row["id"]),
        "company_name": row["company_name"],
        "vendor_name": row["vendor_name"],
        "review_id": str(row["review_id"]) if row.get("review_id") else None,
    }


async def _resolve_webhook_activity_account_focus(
    pool,
    user: AuthUser | None,
    *,
    vendor_name: str | None,
    company_name: str | None,
    signal_id: str | None = None,
    review_id: str | None = None,
    report_cache: dict[str, Any] | None = None,
    tracked_vendor_cache: dict[str, Any] | None = None,
    signal_cache: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    if not user or not getattr(user, "account_id", None):
        return None

    signal_key = str(signal_id or "").strip()
    signal_context = None
    if signal_key:
        if signal_cache is not None and signal_key in signal_cache:
            signal_context = signal_cache[signal_key]
        else:
            signal_context = await _fetch_company_signal_focus_context(pool, signal_key)
            if signal_cache is not None:
                signal_cache[signal_key] = signal_context

    resolved_vendor = str((signal_context or {}).get("vendor_name") or vendor_name or "").strip()
    resolved_company = str((signal_context or {}).get("company_name") or company_name or "").strip()
    resolved_review_id = str(review_id or (signal_context or {}).get("review_id") or "").strip()
    if not resolved_vendor or not resolved_company:
        return None

    vendor_key = _normalize_vendor_name(resolved_vendor)
    if tracked_vendor_cache is not None and vendor_key in tracked_vendor_cache:
        tracked_row = tracked_vendor_cache[vendor_key]
    else:
        tracked_row = await pool.fetchrow(
            """
            SELECT vendor_name, track_mode
            FROM tracked_vendors
            WHERE account_id = $1::uuid
              AND LOWER(vendor_name) = LOWER($2)
            ORDER BY added_at ASC
            LIMIT 1
            """,
            user.account_id,
            resolved_vendor,
        )
        if tracked_vendor_cache is not None:
            tracked_vendor_cache[vendor_key] = tracked_row
    if not tracked_row:
        return None

    watch_vendor = str(tracked_row["vendor_name"] or resolved_vendor).strip()
    watch_vendor_key = _normalize_vendor_name(watch_vendor)
    if report_cache is not None and watch_vendor_key in report_cache:
        report_row = report_cache[watch_vendor_key]
    else:
        report_row = await _fetch_latest_accounts_in_motion_report(pool, watch_vendor, user)
        if report_cache is not None:
            report_cache[watch_vendor_key] = report_row
    if not report_row:
        return None

    report = _safe_json(report_row["intelligence_data"])
    if not isinstance(report, dict):
        return None
    report_date = str(report_row["report_date"]) if report_row.get("report_date") else ""
    if not report_date:
        return None

    target_company_key = _company_lookup_key(resolved_company)
    target_vendor_key = _normalize_vendor_name(resolved_vendor)
    strong_matches: list[dict[str, Any]] = []
    fallback_matches: list[dict[str, Any]] = []
    for account in report.get("accounts") or []:
        if not isinstance(account, dict):
            continue
        shaped = _shape_persisted_accounts_in_motion_account(account, watch_vendor)
        if _company_lookup_key(shaped.get("company")) != target_company_key:
            continue
        if _normalize_vendor_name(shaped.get("vendor")) != target_vendor_key:
            continue
        source_review_ids = {
            str(item).strip()
            for item in (account.get("source_reviews") or [])
            if str(item or "").strip()
        }
        if resolved_review_id and resolved_review_id in source_review_ids:
            strong_matches.append(shaped)
        else:
            fallback_matches.append(shaped)

    match = None
    if resolved_review_id:
        if len(strong_matches) == 1:
            match = strong_matches[0]
        elif len(strong_matches) > 1:
            return None
    if match is None:
        if len(fallback_matches) == 1:
            match = fallback_matches[0]
        elif len(fallback_matches) > 1:
            return None
    if match is None:
        return None

    return {
        "vendor": str(match.get("vendor") or resolved_vendor),
        "company": str(match.get("company") or resolved_company),
        "report_date": report_date,
        "watch_vendor": watch_vendor,
        "category": str(match.get("category") or ""),
        "track_mode": str(tracked_row.get("track_mode") or ""),
    }


async def _accounts_in_motion_missing_report_detail(pool) -> str:
    if await has_complete_core_run_marker(pool, date.today()):
        return "No persisted accounts-in-motion report found for that vendor"
    return (
        "No persisted accounts-in-motion report found for that vendor because "
        "today's core churn materialization is incomplete"
    )


def _validate_accounts_in_motion_window(window_days: int) -> None:
    configured = settings.b2b_churn.intelligence_window_days
    if window_days != configured:
        raise HTTPException(
            status_code=400,
            detail=(
                "accounts-in-motion uses the persisted daily report window. "
                f"Use /accounts-in-motion/live for custom window_days values (configured={configured})."
            ),
        )


async def _list_accounts_in_motion_from_reviews(
    pool,
    vendor_name: str,
    min_urgency: float,
    window_days: int,
    limit: int,
    user: AuthUser | None,
):
    # APPROVED-ENRICHMENT-READ: churn_signals, urgency_score, buyer_authority, pain_categories, quotable_phrases, competitors_mentioned, reviewer_context
    # Reason: DISTINCT ON per-company dedup + intent_to_leave filter + budget_authority - structurally unique
    conditions = [
        "r.enrichment_status = 'enriched'",
        "r.duplicate_of_review_id IS NULL",
        "(r.enrichment->'churn_signals'->>'intent_to_leave')::boolean = true",
        "r.reviewer_company IS NOT NULL",
        "LENGTH(TRIM(r.reviewer_company)) > 3",
        "(r.enrichment->>'urgency_score')::numeric >= $1",
        "r.enriched_at > NOW() - make_interval(days => $2)",
        "r.vendor_name ILIKE '%' || $3 || '%'",
    ]
    params: list[Any] = [min_urgency, window_days, vendor_name.strip()]
    idx = 4
    if _should_scope(user):
        conditions.append(
            f"r.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(user.account_id)
        idx += 1
    rows = await pool.fetch(
        f"""
        SELECT DISTINCT ON (LOWER(r.reviewer_company))
            r.reviewer_company, r.vendor_name, r.product_category,
            (r.enrichment->>'urgency_score')::numeric AS urgency,
            r.enrichment->'buyer_authority'->>'role_type' AS role_type,
            r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
            (r.enrichment->'buyer_authority'->>'has_budget_authority')::boolean AS budget_authority,
            r.enrichment->>'pain_categories' AS pain_categories,
            r.enrichment->>'quotable_phrases' AS quotable_phrases,
            r.enrichment->'competitors_mentioned' AS competitors,
            r.enrichment->'churn_signals'->>'contract_signal' AS contract_signal,
            r.reviewer_title, r.company_size_raw,
            COALESCE(r.reviewer_industry, r.enrichment->'reviewer_context'->>'industry') AS industry,
            r.enriched_at
        FROM b2b_reviews r
        WHERE {" AND ".join(conditions)}
        ORDER BY LOWER(r.reviewer_company), (r.enrichment->>'urgency_score')::numeric DESC
        """,
        *params,
    )
    accounts: list[dict[str, Any]] = []
    for row in rows[:limit]:
        pain_categories = _safe_json(row["pain_categories"])
        quotes = _safe_json(row["quotable_phrases"])
        competitors = _safe_json(row["competitors"])
        accounts.append(
            {
                "company": row["reviewer_company"],
                "vendor": row["vendor_name"],
                "category": row["product_category"],
                "urgency": _safe_float(row["urgency"], 0),
                "role_type": row["role_type"],
                "buying_stage": row["buying_stage"],
                "budget_authority": row["budget_authority"],
                "pain_categories": [
                    {"category": p.get("category", ""), "severity": p.get("severity", "")}
                    for p in (pain_categories if isinstance(pain_categories, list) else [])
                    if isinstance(p, dict)
                ],
                "evidence": [q for q in (quotes if isinstance(quotes, list) else []) if isinstance(q, str)][:3],
                "alternatives_considering": [
                    {"name": c.get("name", ""), "reason": c.get("reason", "")}
                    for c in (competitors if isinstance(competitors, list) else [])
                    if isinstance(c, dict) and c.get("name")
                ][:5],
                "contract_signal": row["contract_signal"],
                "reviewer_title": row["reviewer_title"],
                "company_size_raw": row["company_size_raw"],
                "industry": row["industry"],
                "enriched_at": str(row["enriched_at"]) if row["enriched_at"] else None,
            }
        )
    accounts = await _enrich_accounts_in_motion_accounts(pool, accounts)
    accounts.sort(key=lambda account: -(account.get("urgency") or 0))
    return {
        "vendor": vendor_name.strip(),
        "accounts": accounts,
        "count": len(accounts),
        "window_days": window_days,
        "min_urgency": min_urgency,
        "data_source": "live_reviews",
    }

@router.get("/accounts-in-motion")
async def list_accounts_in_motion(
    vendor_name: str = Query(..., description="Target vendor (companies leaving this vendor)"),
    min_urgency: float = Query(settings.b2b_churn.accounts_in_motion_min_urgency, ge=0, le=10),
    window_days: int = Query(settings.b2b_churn.intelligence_window_days, ge=1, le=3650),
    limit: int = Query(settings.b2b_churn.accounts_in_motion_max_per_vendor, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    """Ranked list of companies showing churn intent for a specific vendor.

    Each account includes urgency, buyer context, pain evidence, firmographic
    data (from Apollo), and decision-maker contacts with verified emails.
    Designed for SDR prospecting lists.
    """
    vendor_name = _required_query_text(vendor_name, "vendor_name")
    pool = _pool_or_503()
    _validate_accounts_in_motion_window(window_days)
    persisted = await _list_accounts_in_motion_from_report(
        pool,
        vendor_name,
        min_urgency=min_urgency,
        limit=limit,
        user=user,
    )
    if persisted is None:
        raise HTTPException(
            status_code=404,
            detail=await _accounts_in_motion_missing_report_detail(pool),
        )
    return persisted


@router.get("/accounts-in-motion/live")
async def list_accounts_in_motion_live(
    vendor_name: str = Query(..., description="Target vendor (companies leaving this vendor)"),
    min_urgency: float = Query(settings.b2b_churn.accounts_in_motion_min_urgency, ge=0, le=10),
    window_days: int = Query(settings.b2b_churn.intelligence_window_days, ge=1, le=3650),
    limit: int = Query(settings.b2b_churn.accounts_in_motion_max_per_vendor, ge=1, le=100),
    user: AuthUser | None = Depends(optional_auth),
):
    """Live exploratory accounts-in-motion view rebuilt directly from reviews."""
    vendor_name = _required_query_text(vendor_name, "vendor_name")
    pool = _pool_or_503()
    return await _list_accounts_in_motion_from_reviews(
        pool,
        vendor_name,
        min_urgency=min_urgency,
        window_days=window_days,
        limit=limit,
        user=user,
    )
