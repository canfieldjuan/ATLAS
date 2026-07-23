"""Public lead-intake endpoint for the EOM website estimate form.

Issue #2151 Phase 1. The website form POSTs here at submit time (in parallel
with the existing Web3Forms email relay), giving Atlas a real-time CRM write
plus an instant customer acknowledgement. Public by design — same trust model
as the b2b briefing gate (a browser form cannot hold a secret); abuse guards
are the honeypot field, payload length caps, and same-day duplicate
suppression via the contact_interactions dedupe key.
"""

import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/leads", tags=["leads"])

EOM_BUSINESS_CONTEXT_ID = "effingham_maids"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PHONE_DIGITS = 7
# Server-side throttle (PR #2152 review, R3/R8): max submissions per
# email-or-phone identity per rolling day, enforced BEFORE any side effect.
MAX_DAILY_SUBMISSIONS = 5


class LeadIntakeRequest(BaseModel):
    """Payload mirroring the website estimate form."""

    name: str = Field(min_length=1, max_length=200)
    email: str = Field(default="", max_length=320)
    phone: str = Field(default="", max_length=40)
    service: str = Field(default="", max_length=120)
    frequency: str = Field(default="", max_length=120)
    square_feet: str = Field(default="", max_length=40)
    message: str = Field(default="", max_length=8000)
    source_page: str = Field(default="", max_length=300)
    # Honeypot: hidden on the real form; humans leave it empty.
    website: str = Field(default="", max_length=300)


class LeadValidationError(ValueError):
    """Raised when a payload is structurally unusable as a lead."""


class LeadRateLimitedError(RuntimeError):
    """Raised when an identity exceeds the daily submission cap."""


async def _daily_submission_count(email: str, phone: str) -> int:
    """Count today's web_form submissions for this identity (EOM-scoped)."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    return await pool.fetchval(
        """
        SELECT COUNT(*)
        FROM contact_interactions ci
        JOIN contacts c ON c.id = ci.contact_id
        WHERE ci.interaction_type = 'web_form'
          AND ci.occurred_at > NOW() - INTERVAL '1 day'
          AND c.business_context_id = $1
          AND (($2 <> '' AND LOWER(c.email) = $2) OR ($3 <> '' AND c.phone = $3))
        """,
        EOM_BUSINESS_CONTEXT_ID,
        email,
        phone,
    ) or 0


def _build_summary(payload: LeadIntakeRequest) -> str:
    """Full, untruncated interaction summary (the gmail_digest path's 200-char
    truncation is one of the defects this endpoint exists to bypass)."""
    parts = ["Website estimate request"]
    if payload.service:
        parts.append(f"service: {payload.service}")
    if payload.frequency:
        parts.append(f"frequency: {payload.frequency}")
    if payload.square_feet:
        parts.append(f"square feet: {payload.square_feet}")
    if payload.source_page:
        parts.append(f"page: {payload.source_page}")
    summary = " — ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0]
    if payload.message:
        summary += f"\nMessage: {payload.message}"
    return summary


async def _process_lead_intake(
    payload: LeadIntakeRequest,
    crm: Any,
    email_provider: Any,
    daily_count: Optional[Any] = None,
) -> dict[str, Any]:
    """Core intake flow with injectable providers (unit-testable sans HTTP)."""
    if payload.website.strip():
        # Honeypot tripped: report success so bots learn nothing, touch nothing.
        logger.info("lead_intake: honeypot tripped; dropping submission silently")
        return {"success": True, "contact_id": None, "email_sent": False}

    email = payload.email.strip().lower()
    phone = payload.phone.strip()
    # A phone with no dialable digits ("n/a", "-") is not a contact channel
    # (PR #2152 review, R1/R2).
    if phone and len(re.sub(r"\D", "", phone)) < _MIN_PHONE_DIGITS:
        phone = ""
    if not email and not phone:
        raise LeadValidationError(
            "Provide an email address or a phone number with at least "
            f"{_MIN_PHONE_DIGITS} digits"
        )
    if email and not _EMAIL_RE.match(email):
        raise LeadValidationError("Invalid email address")

    # Throttle BEFORE any side effect (CRM write or email).
    if daily_count is not None:
        recent = await daily_count(email, phone)
        if recent >= MAX_DAILY_SUBMISSIONS:
            raise LeadRateLimitedError("Daily submission limit reached")

    contact = await crm.find_or_create_contact(
        full_name=payload.name.strip(),
        email=email or None,
        phone=phone or None,
        contact_type="lead",
        source="web",
        source_ref="website_estimate_form",
        business_context_id=EOM_BUSINESS_CONTEXT_ID,
        tags=["website", "estimate_request"],
    )
    contact_id = contact.get("id")
    if not contact_id:
        raise RuntimeError("CRM returned contact without id")

    interaction = await crm.log_interaction(
        contact_id=str(contact_id),
        interaction_type="web_form",
        summary=_build_summary(payload),
        intent="estimate_request",
        metadata={
            "service": payload.service,
            "frequency": payload.frequency,
            "square_feet": payload.square_feet,
            "source_page": payload.source_page,
        },
    )
    # log_interaction dedupes identical same-day rows and reports it via the
    # public "inserted" flag — a double-submit shouldn't double-email the lead.
    freshly_logged = bool((interaction or {}).get("inserted", True))

    email_sent = False
    if email and freshly_logged:
        try:
            from ..templates.email import BUSINESS_EMAIL, format_request_acknowledgement

            subject, body = format_request_acknowledgement(
                client_name=payload.name,
                service=payload.service,
                frequency=payload.frequency,
            )
            result = await email_provider.send(
                to=[email],
                subject=subject,
                body=body,
                reply_to=BUSINESS_EMAIL,
            )
            email_sent = bool(result.get("success", True)) if isinstance(result, dict) else True
        except Exception:
            # Acknowledgement is best-effort; the CRM write is the source of
            # truth and must not be rolled back or reported as failed.
            logger.exception("lead_intake: acknowledgement email failed for contact %s", contact_id)

    return {"success": True, "contact_id": str(contact_id), "email_sent": email_sent}


@router.post("/intake")
async def lead_intake(payload: LeadIntakeRequest) -> dict[str, Any]:
    """Receive an estimate-form submission from the public website."""
    from ..services.crm_provider import get_crm_provider
    from ..services.email_provider import get_email_provider

    try:
        return await _process_lead_intake(
            payload,
            crm=get_crm_provider(),
            email_provider=get_email_provider(),
            daily_count=_daily_submission_count,
        )
    except LeadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LeadRateLimitedError as exc:
        raise HTTPException(status_code=429, detail="Too many requests — try again tomorrow") from exc
    except Exception:
        logger.exception("lead_intake: intake failed")
        raise HTTPException(status_code=503, detail="Lead intake temporarily unavailable")
