"""
REST endpoints for Vendor Intelligence Briefings.

Preview, generate+send, and list sent briefings.
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, optional_auth
from ..config import settings
from ..storage.database import get_db_pool
from ..autonomous.tasks.b2b_vendor_briefing import (
    build_vendor_briefing,
    generate_and_send_briefing,
)
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
# Schemas
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1)


class GenerateRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1)
    to_email: str = Field(..., min_length=3)


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
