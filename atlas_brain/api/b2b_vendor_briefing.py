"""
REST endpoints for Vendor Intelligence Briefings.

Preview, generate+send, email gate, report data, and list sent briefings.
"""

import json
import logging
from datetime import datetime
from uuid import UUID

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, optional_auth
from ..config import settings
from ..services.crm_provider import get_crm_provider
from ..storage.database import get_db_pool
from ..autonomous.tasks.b2b_vendor_briefing import (
    build_gate_url,
    build_vendor_briefing,
    create_gate_token,
    generate_and_send_briefing,
    send_batch_briefings,
    send_vendor_briefing,
)
from ..autonomous.tasks.campaign_suppression import is_suppressed
from ..templates.email.vendor_briefing import render_vendor_briefing_html

logger = logging.getLogger("atlas.api.b2b_vendor_briefing")

router = APIRouter(prefix="/b2b/briefings", tags=["b2b-briefings"])


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


class GenerateRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1)
    to_email: str | None = Field(None, min_length=3)


class GateRequest(BaseModel):
    email: str = Field(..., min_length=5)
    token: str = Field(..., min_length=10)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/preview")
async def preview_briefing(
    body: PreviewRequest,
    user: AuthUser | None = Depends(optional_auth),
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
    user: AuthUser | None = Depends(optional_auth),
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
    email = body.email.strip().lower()

    pool = _pool_or_503()

    # Rate limit: max 3 gate requests per email per day
    count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM b2b_vendor_briefings
        WHERE LOWER(recipient_email) = $1
          AND created_at > NOW() - INTERVAL '1 day'
        """,
        email,
    )
    if count and count >= 3:
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

    # Build full (non-redacted) briefing
    briefing_data = await build_vendor_briefing(vendor_name)
    if not briefing_data:
        raise HTTPException(status_code=404, detail=f"No data found for vendor: {vendor_name}")

    # Mark as gated delivery so template adds "Want this weekly?" footer
    briefing_data["is_gated_delivery"] = True

    briefing_html = render_vendor_briefing_html(briefing_data)

    # Send the full briefing
    result = await send_vendor_briefing(
        to_email=email,
        vendor_name=vendor_name,
        briefing_html=briefing_html,
        briefing_data=briefing_data,
    )

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to send briefing")

    logger.info("Gate briefing sent to %s for %s", email, vendor_name)
    return {
        "status": "ok",
        "vendor_name": vendor_name,
        "message": "Full briefing sent to your email",
        "report_token": body.token,
    }


@router.get("/report-data")
async def report_data(token: str = Query(..., min_length=10)):
    """Public endpoint: return rich intelligence data for a vendor given a valid gate token.

    Called by the churnsignals.co report view page after email gate capture.
    No auth required -- the JWT token IS the access control.
    """
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
        intelligence_reports.append({
            "report_type": row["report_type"],
            "executive_summary": row["executive_summary"],
            "data": intel_data,
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

    return {
        "vendor_name": vendor_name,
        "briefing": briefing_data,
        "intelligence_reports": intelligence_reports,
        "product_profile": product_profile,
    }


@router.post("/send-batch")
async def send_batch(user: AuthUser | None = Depends(optional_auth)):
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
    user: AuthUser | None = Depends(optional_auth),
):
    """List sent briefings, optionally filtered by vendor."""
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
