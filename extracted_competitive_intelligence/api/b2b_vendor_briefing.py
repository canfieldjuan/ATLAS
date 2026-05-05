"""
REST endpoints for Vendor Intelligence Briefings.

Preview, generate+send, email gate, report data, and list sent briefings.
"""

import csv
import io
import json
import logging
import re
from datetime import datetime
from uuid import UUID

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from ..auth.dependencies import AuthUser, require_auth
from ..config import settings
from ..services.crm_provider import get_crm_provider
from ..storage.database import get_db_pool
from ..autonomous.tasks.b2b_vendor_briefing import (
    build_gate_url,
    build_vendor_briefing,
    create_gate_token,
    generate_and_send_briefing,
    reject_briefing,
    send_approved_briefing,
    send_batch_briefings,
    send_vendor_briefing,
)
from ..autonomous.tasks.campaign_suppression import is_suppressed
from ..services.b2b.vendor_briefing_api_ports import (
    VendorBriefingAPIError,
    VendorBriefingAPINotConfigured,
    create_vendor_checkout_session,
    retrieve_vendor_checkout_session,
    send_checkout_confirmation_email,
    send_gated_report_email,
)
from ..templates.email.vendor_briefing import render_vendor_briefing_html

logger = logging.getLogger("atlas.api.b2b_vendor_briefing")

router = APIRouter(prefix="/b2b/briefings", tags=["b2b-briefings"])


def _clean_required_text(value) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("value is required")
    return text


def _clean_optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_required_query_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field_name} is required")
    return text


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
        elif isinstance(val, datetime):
            d[key] = val.isoformat()
        else:
            d[key] = val
    return d


# ---------------------------------------------------------------------------
# Public report redaction helpers
# ---------------------------------------------------------------------------

# Dollar amounts like $180K, $500/month, $120K/year
_DOLLAR_RE = re.compile(r"\$[\d,.]+[KkMm]?(?:/\w+)?")
_PUBLIC_PRODUCT_CLAIM_DANGEROUS_KEYS = {
    "cross_vendor_battles",
    "objection_handlers",
    "recommended_plays",
    "talk_track",
    "vendor_weaknesses",
    "weakness_analysis",
    "top_displacement_targets",
    "market_winners",
    "market_losers",
    "top_battles",
    "category_council",
}
_PUBLIC_UNVALIDATED_PROFILE_KEYS = {
    "strengths",
    "weaknesses",
    "commonly_compared_to",
    "commonly_switched_from",
}


def _redact_quotes(evidence: list) -> list:
    """Filter quotes that contain dollar amounts or are too identifying.

    Keeps the quote but scrubs dollar figures (replaces with '[amount]').
    """
    cleaned = []
    for item in evidence:
        if isinstance(item, dict):
            text = item.get("quote") or item.get("text") or ""
        elif isinstance(item, str):
            text = item
        else:
            continue
        if not text:
            continue
        # Scrub dollar amounts
        text = _DOLLAR_RE.sub("[amount]", text)
        if isinstance(item, dict):
            cleaned.append({**item, "quote": text})
        else:
            cleaned.append(text)
    return cleaned


def _redact_public_account_identity(briefing_data: dict) -> dict:
    named = briefing_data.pop("named_accounts", []) or []

    preview = briefing_data.get("account_reasoning_preview")
    preview_dict = dict(preview) if isinstance(preview, dict) else {}

    raw_priority_names = briefing_data.pop("priority_account_names", None)
    if not raw_priority_names:
        raw_priority_names = preview_dict.get("priority_account_names")
    priority_names = [
        str(item or "").strip()
        for item in (raw_priority_names or [])
        if str(item or "").strip()
    ]

    preview_count = 0
    metrics = preview_dict.get("account_pressure_metrics")
    if isinstance(metrics, dict):
        try:
            preview_count = int(metrics.get("total_accounts") or 0)
        except (TypeError, ValueError):
            preview_count = 0
    if preview_count <= 0:
        reasoning = preview_dict.get("account_reasoning")
        if isinstance(reasoning, dict):
            try:
                preview_count = int(reasoning.get("total_accounts") or 0)
            except (TypeError, ValueError):
                preview_count = 0
    if preview_count <= 0:
        preview_count = len(priority_names)

    briefing_data["named_account_count"] = max(len(named), preview_count)

    if preview_dict:
        preview_dict.pop("priority_account_names", None)
        preview_dict.pop("account_reasoning", None)
        preview_dict.pop("top_accounts", None)
        if preview_dict:
            briefing_data["account_reasoning_preview"] = preview_dict
        else:
            briefing_data.pop("account_reasoning_preview", None)

    return briefing_data


def _is_product_claim_like(value: object) -> bool:
    return (
        isinstance(value, dict)
        and ("report_allowed" in value or "render_allowed" in value)
        and (
            "claim_type" in value
            or "claim_scope" in value
            or "evidence_posture" in value
            or "confidence" in value
        )
    )


def _collect_product_claims(value: object) -> list[dict]:
    if _is_product_claim_like(value):
        return [value]  # type: ignore[list-item]
    if isinstance(value, list):
        claims: list[dict] = []
        for item in value:
            claims.extend(_collect_product_claims(item))
        return claims
    if isinstance(value, dict):
        claims: list[dict] = []
        for item in value.values():
            claims.extend(_collect_product_claims(item))
        return claims
    return []


def _has_report_safe_product_claim_context(value: object) -> bool:
    claims = _collect_product_claims(value)
    return bool(claims) and all(claim.get("report_allowed") is True for claim in claims)


def _public_report_safe_value(value: object) -> object | None:
    if isinstance(value, list):
        kept = [
            _strip_public_unvalidated_sections(item)
            for item in value
            if _has_report_safe_product_claim_context(item)
        ]
        return kept or None
    if _has_report_safe_product_claim_context(value):
        return _strip_public_unvalidated_sections(value)
    return None


def _strip_public_unvalidated_sections(value: object) -> object:
    if isinstance(value, list):
        return [_strip_public_unvalidated_sections(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned: dict = {}
    for key, item in value.items():
        if key in _PUBLIC_PRODUCT_CLAIM_DANGEROUS_KEYS:
            report_safe_item = _public_report_safe_value(item)
            if report_safe_item not in (None, [], {}):
                cleaned[key] = report_safe_item
            continue
        cleaned[key] = _strip_public_unvalidated_sections(item)
    return cleaned


def _strip_public_unvalidated_product_profile(profile: dict | None) -> dict | None:
    if not isinstance(profile, dict):
        return None
    if _has_report_safe_product_claim_context(profile):
        stripped = _strip_public_unvalidated_sections(profile)
        return stripped if isinstance(stripped, dict) else {}
    return {
        key: value
        for key, value in profile.items()
        if key not in _PUBLIC_UNVALIDATED_PROFILE_KEYS
    }


# ---------------------------------------------------------------------------
# Gate token helpers
# ---------------------------------------------------------------------------

_jwt_cfg = settings.saas_auth


def decode_gate_token(token: str) -> dict:
    """Decode and validate a briefing gate token. Raises HTTPException on failure."""
    try:
        claims = pyjwt.decode(
            token, _jwt_cfg.jwt_secret, algorithms=[_jwt_cfg.jwt_algorithm]
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="This link has expired")
    except pyjwt.PyJWTError as exc:
        logger.warning("Gate token decode failed: %s (token=%s...)", exc, token[:30])
        raise HTTPException(status_code=400, detail="Invalid link")

    if claims.get("type") != "briefing_gate":
        raise HTTPException(status_code=400, detail="Invalid token type")

    return claims


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1)
    prospect_mode: bool = Field(False, description="Redact named accounts for sales demos")

    @field_validator("vendor_name", mode="before")
    @classmethod
    def _trim_vendor_name(cls, value):
        return _clean_required_text(value)


class GenerateRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1)
    to_email: str | None = Field(None, min_length=3)

    @field_validator("vendor_name", mode="before")
    @classmethod
    def _trim_vendor_name(cls, value):
        return _clean_required_text(value)

    @field_validator("to_email", mode="before")
    @classmethod
    def _trim_to_email(cls, value):
        return _clean_optional_text(value)


class GateRequest(BaseModel):
    email: str = Field(..., min_length=5)
    token: str = Field(..., min_length=10)

    @field_validator("email", "token", mode="before")
    @classmethod
    def _trim_required_fields(cls, value):
        return _clean_required_text(value)


class VendorCheckoutRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1)
    tier: str = Field(..., pattern="^(standard|pro)$")
    email: str | None = Field(None, min_length=5)

    @field_validator("vendor_name", "tier", mode="before")
    @classmethod
    def _trim_required_fields(cls, value):
        return _clean_required_text(value)

    @field_validator("email", mode="before")
    @classmethod
    def _trim_optional_email(cls, value):
        return _clean_optional_text(value)


class BulkBriefingApproveRequest(BaseModel):
    briefing_ids: list[str] = Field(..., min_length=1)
    action: str = Field("approve", pattern="^(approve|reject)$")


class BulkBriefingRejectRequest(BaseModel):
    briefing_ids: list[str] = Field(..., min_length=1)
    reason: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/preview")
async def preview_briefing(
    body: PreviewRequest,
    user: AuthUser = Depends(require_auth),
):
    """Build and render a briefing as HTML without sending."""
    if not settings.b2b_churn.vendor_briefing_enabled:
        raise HTTPException(status_code=403, detail="Vendor briefings disabled")

    briefing_data = await build_vendor_briefing(body.vendor_name)
    if not briefing_data:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for vendor: {body.vendor_name}",
        )

    if body.prospect_mode:
        briefing_data["prospect_mode"] = True
        briefing_data["gate_url"] = build_gate_url(body.vendor_name)

    html = render_vendor_briefing_html(briefing_data)
    return {"vendor_name": body.vendor_name, "html": html, "data": briefing_data}


@router.post("/generate")
async def generate_briefing(
    body: GenerateRequest,
    user: AuthUser = Depends(require_auth),
):
    """Build, render, send, and persist a vendor briefing."""
    if not settings.b2b_churn.vendor_briefing_enabled:
        raise HTTPException(status_code=403, detail="Vendor briefings disabled")

    result = await generate_and_send_briefing(
        vendor_name=body.vendor_name,
        to_email=body.to_email,
    )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@router.post("/gate")
async def briefing_gate(body: GateRequest):
    """Email gate: validate token, create CRM lead, send full briefing.

    This is a public endpoint (no auth required) -- called from the
    churnsignals.co landing page when a prospect submits their email.
    """
    if not settings.b2b_churn.vendor_briefing_enabled:
        raise HTTPException(status_code=403, detail="Vendor briefings disabled")

    # Validate token
    claims = decode_gate_token(body.token)
    vendor_name = claims["vendor_name"]
    email = body.email.lower()

    pool = _pool_or_503()

    # Rate limit: max 5 gate requests per email per day
    count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM b2b_vendor_briefings
        WHERE LOWER(recipient_email) = $1
          AND created_at > NOW() - INTERVAL '1 day'
        """,
        email,
    )
    if count and count >= 5:
        raise HTTPException(status_code=429, detail="Too many requests -- try again tomorrow")

    # Suppression check
    if await is_suppressed(pool, email=email):
        raise HTTPException(status_code=410, detail="This email address has been unsubscribed")

    # CRM: create/upsert contact as lead
    try:
        crm = get_crm_provider()
        contact = await crm.find_or_create_contact(
            full_name="",
            email=email,
            contact_type="lead",
            source="briefing_gate",
            source_ref=vendor_name,
        )
        contact_id = contact.get("id")
        if contact_id:
            await crm.log_interaction(
                contact_id=str(contact_id),
                interaction_type="briefing_gate",
                summary=f"Requested full briefing for {vendor_name}",
                intent="vendor_briefing_request",
            )
        logger.info("Gate lead created: %s for %s", email, vendor_name)
    except Exception:
        logger.exception("CRM lead creation failed during gate (non-fatal)")

    # Build briefing data for fallback fields
    cached_row = await pool.fetchrow(
        """
        SELECT briefing_data FROM b2b_vendor_briefings
        WHERE LOWER(vendor_name) = LOWER($1) AND status = 'sent'
          AND briefing_data IS NOT NULL
        ORDER BY created_at DESC LIMIT 1
        """,
        vendor_name,
    )
    if cached_row and cached_row["briefing_data"]:
        bd = cached_row["briefing_data"]
        briefing_data = json.loads(bd) if isinstance(bd, str) else bd
    else:
        briefing_data = await build_vendor_briefing(vendor_name)

    if not briefing_data:
        raise HTTPException(status_code=404, detail=f"No data found for vendor: {vendor_name}")

    # Fetch the full exploratory_overview report for deep analysis
    full_report_row = await pool.fetchrow(
        """
        SELECT intelligence_data, executive_summary
        FROM b2b_intelligence
        WHERE report_type = 'exploratory_overview'
        ORDER BY created_at DESC LIMIT 1
        """,
    )
    full_report_data: dict = {}
    if full_report_row:
        rd = full_report_row["intelligence_data"]
        full_report_data = json.loads(rd) if isinstance(rd, str) else rd

    try:
        await send_gated_report_email(
            vendor_name=vendor_name,
            recipient_email=email,
            report_data=full_report_data,
            briefing_data=briefing_data,
        )
    except VendorBriefingAPIError:
        logger.exception("Failed to send gated report email")
        raise HTTPException(status_code=500, detail="Failed to send report")

    # Persist delivery record
    try:
        await pool.execute(
            """
            INSERT INTO b2b_vendor_briefings
                (vendor_name, recipient_email, subject, briefing_data, status)
            VALUES ($1, $2, $3, $4::jsonb, 'sent')
            """,
            vendor_name,
            email,
            f"Your {vendor_name} Churn Intelligence Report",
            json.dumps(briefing_data, default=str),
        )
    except Exception:
        logger.warning("Failed to persist gate delivery record (non-fatal)")

    logger.info("Gate full report (PDF) sent to %s for %s", email, vendor_name)
    return {
        "status": "ok",
        "vendor_name": vendor_name,
        "message": "Full report sent to your email",
        "report_token": body.token,
    }


@router.post("/checkout")
async def vendor_checkout(body: VendorCheckoutRequest):
    """Create a Stripe Checkout Session for vendor retention intel subscription."""
    try:
        session = await create_vendor_checkout_session(
            vendor_name=body.vendor_name,
            tier=body.tier,
            customer_email=body.email,
        )
    except VendorBriefingAPINotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except VendorBriefingAPIError as exc:
        logger.error("Stripe checkout creation failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to create checkout session")

    logger.info(
        "Vendor checkout session: vendor=%s tier=%s session=%s",
        body.vendor_name, body.tier, session.url,
    )
    return {"url": session.url}


@router.get("/checkout-session")
async def checkout_session_info(session_id: str = Query(..., min_length=10)):
    """Retrieve customer email and metadata from a completed Stripe Checkout session.

    Also fires the purchase confirmation email on first call (idempotent).
    """
    session_id = _clean_required_query_text(session_id, field_name="session_id")

    try:
        session = await retrieve_vendor_checkout_session(session_id)
    except VendorBriefingAPINotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except VendorBriefingAPIError as exc:
        logger.warning("Stripe session retrieval failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid session")

    customer_email = session.customer_email
    vendor_name = session.vendor_name
    tier = session.tier

    # Send confirmation email only after successful payment
    # (idempotent -- dedup by session_id in billing_events)
    payment_ok = session.payment_status == "paid"
    if customer_email and session.source == "vendor_briefing_report" and payment_ok:
        pool = get_db_pool()
        dedup_key = f"vendor_checkout_email_{session_id}"
        already_sent = await pool.fetchval(
            "SELECT 1 FROM billing_events WHERE stripe_event_id = $1", dedup_key
        )
        if not already_sent:
            try:
                await send_checkout_confirmation_email(
                    vendor_name=vendor_name,
                    tier=tier,
                    customer_email=customer_email,
                )
                # Mark as sent so webhook doesn't double-send
                await pool.execute(
                    """
                    INSERT INTO billing_events (stripe_event_id, event_type, payload)
                    VALUES ($1, $2, '{}'::jsonb)
                    ON CONFLICT (stripe_event_id) DO NOTHING
                    """,
                    dedup_key,
                    "vendor_checkout_confirmation_email",
                )
                logger.info(
                    "Vendor checkout confirmation sent (direct): email=%s vendor=%s tier=%s",
                    customer_email, vendor_name, tier,
                )
            except Exception:
                logger.exception("Failed to send vendor checkout confirmation (direct)")

    return {
        "email": customer_email,
        "vendor_name": vendor_name,
        "tier": tier,
    }


@router.get("/report-data")
async def report_data(token: str = Query(..., min_length=10)):
    """Public endpoint: return rich intelligence data for a vendor given a valid gate token.

    Called by the churnsignals.co report view page after email gate capture.
    No auth required -- the JWT token IS the access control.
    """
    token = _clean_required_query_text(token, field_name="token")
    claims = decode_gate_token(token)
    vendor_name = claims["vendor_name"]

    pool = _pool_or_503()

    # 1. Base briefing data -- prefer cached from b2b_vendor_briefings (instant)
    #    over build_vendor_briefing() which hits OpenRouter LLM (~60s).
    cached_row = await pool.fetchrow(
        """
        SELECT briefing_data FROM b2b_vendor_briefings
        WHERE LOWER(vendor_name) = LOWER($1) AND status = 'sent'
          AND briefing_data IS NOT NULL
        ORDER BY created_at DESC LIMIT 1
        """,
        vendor_name,
    )
    if cached_row and cached_row["briefing_data"]:
        bd = cached_row["briefing_data"]
        briefing_data = json.loads(bd) if isinstance(bd, str) else bd
        logger.info("report-data: using cached briefing for %s", vendor_name)
    else:
        logger.info("report-data: no cache, building briefing for %s (slow path)", vendor_name)
        briefing_data = await build_vendor_briefing(vendor_name)

    if not briefing_data:
        raise HTTPException(status_code=404, detail=f"No data found for vendor: {vendor_name}")

    # Redact account-identifying fields from the public report while preserving
    # the preview summary/count needed for the public UI.
    briefing_data = _redact_public_account_identity(dict(briefing_data))

    # Strip company-identifying details from quotes
    evidence = briefing_data.get("evidence") or []
    briefing_data["evidence"] = _redact_quotes(evidence)
    stripped_briefing = _strip_public_unvalidated_sections(briefing_data)
    briefing_data = stripped_briefing if isinstance(stripped_briefing, dict) else {}

    # 2. Enrich with b2b_intelligence reports (vendor comparisons, deeper analysis)
    intelligence_reports = []
    rows = await pool.fetch(
        """
        SELECT report_type, executive_summary, intelligence_data,
               report_date, created_at
        FROM b2b_intelligence
        WHERE LOWER(vendor_filter) = LOWER($1)
          OR LOWER(category_filter) = LOWER($1)
        ORDER BY created_at DESC
        LIMIT 5
        """,
        vendor_name,
    )
    for row in rows:
        intel_data = row["intelligence_data"]
        if isinstance(intel_data, str):
            try:
                intel_data = json.loads(intel_data)
            except (json.JSONDecodeError, TypeError):
                intel_data = {}
        # Strip company-identifying fields from public report
        _REDACT_INTEL_KEYS = {
            "primary_company_examples", "comparison_company_examples",
            "primary_quote_highlights", "comparison_quote_highlights",
        }
        redacted_data = {
            k: v for k, v in intel_data.items() if k not in _REDACT_INTEL_KEYS
        }
        if isinstance(redacted_data, dict):
            redacted_data = _redact_public_account_identity(redacted_data)
            redacted_data = _strip_public_unvalidated_sections(redacted_data)
        intelligence_reports.append({
            "report_type": row["report_type"],
            "executive_summary": row["executive_summary"],
            "data": redacted_data,
            "report_date": str(row["report_date"]) if row["report_date"] else None,
        })

    # 3. Product profile
    profile_row = await pool.fetchrow(
        """
        SELECT profile_summary, commonly_compared_to, commonly_switched_from,
               strengths, weaknesses, product_category
        FROM b2b_product_profiles
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC LIMIT 1
        """,
        vendor_name,
    )
    product_profile = None
    if profile_row:
        product_profile = {}
        for key in profile_row.keys():
            val = profile_row[key]
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    product_profile[key] = parsed
                except (json.JSONDecodeError, TypeError):
                    product_profile[key] = val
            else:
                product_profile[key] = val
        product_profile = _strip_public_unvalidated_product_profile(product_profile)

    return {
        "vendor_name": vendor_name,
        "briefing": briefing_data,
        "intelligence_reports": intelligence_reports,
        "product_profile": product_profile,
    }


@router.post("/send-batch")
async def send_batch(user: AuthUser = Depends(require_auth)):
    """Send briefings to all eligible vendor targets."""
    if not settings.b2b_churn.vendor_briefing_enabled:
        raise HTTPException(status_code=403, detail="Vendor briefings disabled")

    result = await send_batch_briefings()

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


@router.get("")
async def list_briefings(
    vendor: str | None = Query(None),
    limit: int = Query(50, le=200),
    user: AuthUser = Depends(require_auth),
):
    """List all briefings, optionally filtered by vendor."""
    pool = _pool_or_503()

    if vendor:
        rows = await pool.fetch(
            """
            SELECT id, vendor_name, recipient_email, subject,
                   resend_id, status, created_at
            FROM b2b_vendor_briefings
            WHERE LOWER(vendor_name) = LOWER($1)
            ORDER BY created_at DESC
            LIMIT $2
            """,
            vendor,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, vendor_name, recipient_email, subject,
                   resend_id, status, created_at
            FROM b2b_vendor_briefings
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )

    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Briefing Review Queue (HITL)
# ---------------------------------------------------------------------------

@router.get("/review-queue/summary")
async def briefing_review_summary(
    user: AuthUser = Depends(require_auth),
):
    """Summary stats for the briefing review queue."""
    pool = _pool_or_503()

    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'pending_approval') AS pending_approval,
            COUNT(*) FILTER (WHERE status = 'sent') AS sent,
            COUNT(*) FILTER (WHERE status = 'rejected') AS rejected,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed,
            EXTRACT(EPOCH FROM (NOW() - MIN(created_at) FILTER (WHERE status = 'pending_approval'))) / 3600
                AS oldest_pending_hours
        FROM b2b_vendor_briefings
        """
    )

    return {
        "pending_approval": row["pending_approval"] or 0,
        "sent": row["sent"] or 0,
        "rejected": row["rejected"] or 0,
        "failed": row["failed"] or 0,
        "oldest_pending_hours": round(float(row["oldest_pending_hours"]), 1) if row["oldest_pending_hours"] else None,
    }


@router.get("/review-queue")
async def briefing_review_queue(
    status: str | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    user: AuthUser = Depends(require_auth),
):
    """List briefings for review, optionally filtered by status."""
    pool = _pool_or_503()

    if status and status != "all":
        rows = await pool.fetch(
            """
            SELECT id, vendor_name, recipient_email, subject,
                   briefing_html, status, target_mode,
                   created_at, approved_at, rejected_at, reject_reason
            FROM b2b_vendor_briefings
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            status,
            limit,
            offset,
        )
        count_val = await pool.fetchval(
            "SELECT COUNT(*) FROM b2b_vendor_briefings WHERE status = $1",
            status,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, vendor_name, recipient_email, subject,
                   briefing_html, status, target_mode,
                   created_at, approved_at, rejected_at, reject_reason
            FROM b2b_vendor_briefings
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        count_val = await pool.fetchval(
            "SELECT COUNT(*) FROM b2b_vendor_briefings"
        )

    return {
        "briefings": [_row_to_dict(r) for r in rows],
        "count": count_val or 0,
    }


@router.post("/bulk-approve")
async def bulk_approve_briefings(
    body: BulkBriefingApproveRequest,
    user: AuthUser = Depends(require_auth),
):
    """Approve and send pending briefings."""
    results = []
    failed = []

    for bid in body.briefing_ids:
        result = await send_approved_briefing(bid)
        if "error" in result:
            failed.append({"id": bid, "reason": result["error"]})
        else:
            results.append(result)

    return {
        "processed": len(results),
        "failed": failed,
    }


@router.post("/bulk-reject")
async def bulk_reject_briefings(
    body: BulkBriefingRejectRequest,
    user: AuthUser = Depends(require_auth),
):
    """Reject pending briefings."""
    rejected = 0
    failed = []

    for bid in body.briefing_ids:
        result = await reject_briefing(bid, body.reason)
        if "error" in result:
            failed.append({"id": bid, "reason": result["error"]})
        else:
            rejected += 1

    return {
        "rejected": rejected,
        "failed": failed,
    }


_EXPORT_LIMIT = 10_000


@router.get("/export")
async def export_briefings(
    status: str | None = Query(None),
    user: AuthUser = Depends(require_auth),
):
    """Export briefings as CSV."""
    pool = _pool_or_503()

    if status and status != "all":
        rows = await pool.fetch(
            """
            SELECT vendor_name, recipient_email, subject, status,
                   target_mode, created_at, approved_at, rejected_at, reject_reason
            FROM b2b_vendor_briefings
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            status,
            _EXPORT_LIMIT,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT vendor_name, recipient_email, subject, status,
                   target_mode, created_at, approved_at, rejected_at, reject_reason
            FROM b2b_vendor_briefings
            ORDER BY created_at DESC
            LIMIT $1
            """,
            _EXPORT_LIMIT,
        )

    data = []
    for r in rows:
        data.append({
            "vendor_name": r["vendor_name"] or "",
            "recipient_email": r["recipient_email"] or "",
            "subject": r["subject"] or "",
            "status": r["status"] or "",
            "target_mode": r["target_mode"] or "",
            "created_at": r["created_at"].isoformat() if r["created_at"] else "",
            "approved_at": r["approved_at"].isoformat() if r["approved_at"] else "",
            "rejected_at": r["rejected_at"].isoformat() if r["rejected_at"] else "",
            "reject_reason": r["reject_reason"] or "",
        })

    if not data:
        buf = io.StringIO()
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="briefings.csv"'},
        )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(data[0].keys()))
    writer.writeheader()
    writer.writerows(data)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="briefings.csv"'},
    )
