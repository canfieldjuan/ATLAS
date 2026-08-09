"""Public lead-intake endpoint for the EOM website estimate form.

Issue #2151 Phase 1. The website form POSTs here at submit time (in parallel
with the existing Web3Forms email relay), giving Atlas a real-time CRM write
plus an instant customer acknowledgement. Public by design, matching the trust model
as the b2b briefing gate (a browser form cannot hold a secret); abuse guards
are the honeypot field, payload length caps, and same-day duplicate
suppression via the contact_interactions dedupe key.
"""

import logging
import re
from collections.abc import Mapping
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from ..services.eom_lead_ingress import (
    EOM_BUSINESS_CONTEXT_ID,
    resolve_or_create_eom_inbound_lead_and_log_interaction,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/leads", tags=["leads"])

# Route-scoped CORS: only THIS endpoint is exposed to the marketing site, and
# without credentials — an app-wide allowlist entry would grant the site's
# origin credentialed access to every route (PR #2153 round 6, R3).
ALLOWED_FORM_ORIGINS = frozenset({
    "https://effinghamofficemaids.com",
    "https://www.effinghamofficemaids.com",
})


def _form_cors_headers(origin: str) -> dict[str, str]:
    if origin not in ALLOWED_FORM_ORIGINS:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Vary": "Origin",
    }


class LeadIntakeCORSMiddleware:
    """Path-scoped, credential-free CORS for the public intake route.

    Mounted OUTSIDE the app-wide CORSMiddleware (added after it), because
    Starlette's CORSMiddleware consumes browser preflights before routing —
    a route-level OPTIONS handler would never see them (PR #2153 round 7,
    R1/R12). Only the exact intake path is affected; every other route keeps
    the app-wide policy.
    """

    def __init__(self, app: Any, path: str = "/api/v1/leads/intake") -> None:
        self.app = app
        self.path = path

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != self.path:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        cors = _form_cors_headers(headers.get("origin", ""))
        if scope.get("method") == "OPTIONS" and "access-control-request-method" in headers:
            await send({
                "type": "http.response.start",
                "status": 204 if cors else 400,
                "headers": [(k.encode(), v.encode()) for k, v in cors.items()],
            })
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_with_cors(message: Any) -> None:
            if message.get("type") == "http.response.start" and cors:
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (k.encode(), v.encode()) for k, v in cors.items()
                ]
            await send(message)

        await self.app(scope, receive, send_with_cors)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PHONE_DIGITS = 7
# Server-side throttle (PR #2152 review, R3/R8): max submissions per
# email-or-phone identity per rolling day, enforced BEFORE any side effect.
MAX_DAILY_SUBMISSIONS = 5
# Global acknowledgement-send ceiling (PR #2153 review, R3/R8): bounds the
# unsolicited-email blast radius when an abuser streams unique addresses.
# Leads are still captured past the cap; only the outbound email is skipped.
GLOBAL_ACK_HOURLY_CAP = 20
# Global new-lead push ceiling (same unique-address abuse path): the per-identity
# daily throttle does not stop a stream of DISTINCT identities, so cap the
# operator's phone pushes per rolling hour. Leads are still captured past the
# cap; only the notification is skipped.
GLOBAL_NOTIFY_HOURLY_CAP = 30
# Wall-clock bound on the cap's COUNT query so a stalled/unindexed count cannot
# hang the awaited intake route; on timeout the push fails closed (is skipped).
_NOTIFY_VOLUME_TIMEOUT = 2.0


class LeadIntakeRequest(BaseModel):
    """Payload mirroring the website estimate form."""

    name: str = Field(min_length=1, max_length=200)
    email: str = Field(default="", max_length=254)
    phone: str = Field(default="", max_length=32)
    address: str = Field(default="", max_length=300)
    service: str = Field(default="", max_length=120)
    frequency: str = Field(default="", max_length=120)
    square_feet: str = Field(default="", max_length=40)
    message: str = Field(default="", max_length=8000)
    source_page: str = Field(default="", max_length=300)
    utm_source: str = Field(default="", max_length=256)
    utm_medium: str = Field(default="", max_length=256)
    utm_campaign: str = Field(default="", max_length=256)
    utm_term: str = Field(default="", max_length=256)
    utm_content: str = Field(default="", max_length=256)
    gclid: str = Field(default="", max_length=512)
    gbraid: str = Field(default="", max_length=512)
    wbraid: str = Field(default="", max_length=512)
    landing_path: str = Field(default="", max_length=500)
    referrer: str = Field(default="", max_length=500)
    # Honeypot: hidden on the real form; humans leave it empty.
    website: str = Field(default="", max_length=300)

    @model_validator(mode="before")
    @classmethod
    def _route_address_surrogates_to_validation(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        address = value.get("address")
        if not isinstance(address, str):
            return value
        if not any(0xD800 <= ord(char) <= 0xDFFF for char in address):
            return value
        sanitized = dict(value)
        sanitized["address"] = "\x00"
        return sanitized

    @field_validator("address")
    @classmethod
    def _reject_database_invalid_address(cls, value: str) -> str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("address must be valid") from exc
        if b"\x00" in encoded:
            raise ValueError("address must be valid")
        return value


class LeadValidationError(ValueError):
    """Raised when a payload is structurally unusable as a lead."""


class LeadRateLimitedError(RuntimeError):
    """Raised when an identity exceeds the daily submission cap."""


async def _daily_submission_count(email: str, phone_digits: str) -> int:
    """Count today's web_form submissions for this identity (EOM-scoped).

    Phone bucketing mirrors search_contacts' lookup exactly (substring LIKE
    on the last 10 submitted digits) so any submission that would RESOLVE to
    a stored contact also COUNTS against that contact's cap.
    """
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    return await pool.fetchval(
        """
        SELECT COUNT(*)
        FROM contact_interactions ci
        JOIN contacts c ON c.id = ci.contact_id
        WHERE ci.interaction_type = 'web_form'
          AND ci.occurred_at > NOW() - INTERVAL '1 day'
          AND (c.business_context_id = $1 OR c.business_context_id IS NULL)
          AND (($2 <> '' AND LOWER(c.email) = $2)
               OR ($3 <> '' AND regexp_replace(COALESCE(c.phone, ''), '[^0-9]', '', 'g')
                   LIKE '%' || RIGHT($3, 10) || '%'))
        """,
        EOM_BUSINESS_CONTEXT_ID,
        email,
        phone_digits,
    ) or 0


async def _hourly_ack_volume() -> int:
    """Global count of web_form intakes in the last hour (EOM-scoped)."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    return await pool.fetchval(
        """
        SELECT COUNT(*)
        FROM contact_interactions ci
        JOIN contacts c ON c.id = ci.contact_id
        WHERE ci.interaction_type = 'web_form'
          AND ci.occurred_at > NOW() - INTERVAL '1 hour'
          AND (c.business_context_id = $1 OR c.business_context_id IS NULL)
          AND COALESCE(ci.metadata->>'submitted_email', '') <> ''
        """,
        EOM_BUSINESS_CONTEXT_ID,
    ) or 0


async def _hourly_lead_notification_volume() -> int:
    """Global count of web_form leads in the last hour (EOM-scoped), across ALL
    channels — the ceiling for outbound new-lead pushes. Unlike the ack-volume
    query this omits the email filter, so a phone-only flood is bounded too."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    return await pool.fetchval(
        """
        SELECT COUNT(*)
        FROM contact_interactions ci
        JOIN contacts c ON c.id = ci.contact_id
        WHERE ci.interaction_type = 'web_form'
          AND ci.occurred_at > NOW() - INTERVAL '1 hour'
          AND (c.business_context_id = $1 OR c.business_context_id IS NULL)
        """,
        EOM_BUSINESS_CONTEXT_ID,
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


def _summary_with_channels(payload: LeadIntakeRequest, email: str, phone_digits: str) -> str:
    """Summary including the submitted callback channels: they participate in
    the same-day dedupe key, so a resubmission with a CORRECTED email/phone
    is a new interaction (and a fresh acknowledgement target) instead of
    being swallowed by the duplicate guard (PR #2153 round 6, R1/R6)."""
    channels = ", ".join(v for v in (email, phone_digits) if v)
    summary = _build_summary(payload)
    # Prepended, not appended: log_interaction hashes only the leading slice
    # of the normalized summary, so the channels must sit inside the hashed
    # prefix or a long message pushes them out of the dedupe basis
    # (PR #2153 round 7, R1/R6).
    return f"Callback: {channels}\n{summary}" if channels else summary


def _format_phone_for_display(phone_digits: str) -> str:
    """Pretty-print a US 10/11-digit number for the push; pass others through."""
    if len(phone_digits) == 11 and phone_digits.startswith("1"):
        phone_digits = phone_digits[1:]
    if len(phone_digits) == 10:
        return f"({phone_digits[0:3]}) {phone_digits[3:6]}-{phone_digits[6:10]}"
    return phone_digits


def _header_value_ascii(value: str) -> str:
    """Reduce a string to printable ASCII for safe HTTP-header use. Two reasons
    to go all the way to ASCII rather than latin-1: httpx (0.28) raises
    UnicodeEncodeError on non-latin-1 header values, AND ntfy decodes header
    bytes as UTF-8, so a raw latin-1 byte (e.g. "é" -> 0xE9) would render as
    mojibake. Control chars are dropped too (header-injection safe). The FULL,
    unmodified name — any Unicode — still rides the UTF-8 body below."""
    return "".join(ch for ch in value if 0x20 <= ord(ch) <= 0x7E)


def _lead_push_body(payload: LeadIntakeRequest, email: str, phone_digits: str) -> str:
    """Scannable body for the new-lead push: who (exact name, UTF-8), how to
    reach them, what they want, then where. Kept short to read at a glance."""
    lines: list[str] = [payload.name.strip() or "Website visitor"]
    channels = [c for c in (
        _format_phone_for_display(phone_digits) if phone_digits else "",
        email,
    ) if c]
    if channels:
        lines.append(" · ".join(channels))
    service = payload.service.strip()
    frequency = payload.frequency.strip()
    detail = " · ".join(part for part in (service, frequency) if part)
    if detail:
        lines.append(detail)
    address = payload.address.strip()
    if address:
        lines.append(address)
    return "\n".join(lines)


async def _publish_lead_ntfy(title: str, body: str) -> None:
    """Fire-and-forget push to the dedicated leads ntfy topic. Never raises: a
    notification failure must not touch the already-captured lead.

    Two deliberate constraints:
    - Bounded by a TRUE 5s wall-clock deadline via asyncio.wait_for. httpx's own
      Timeout(5) is per-phase (connect/read/write each get 5s), so it alone
      could keep the awaited intake route busy far longer.
    - Never logs the request URL or a raised transport error verbatim: the URL
      embeds the topic, which is the ONLY secret protecting lead PII on the
      public relay. On failure we record a status code or the error class only."""
    try:
        from ..config import settings

        alerts = settings.alerts
        topic = (alerts.leads_ntfy_topic or "").strip()
        if not alerts.ntfy_enabled or not topic:
            return  # feature off (no topic configured) — silently skip

        import asyncio

        import httpx

        # Use the PINNED leads relay, never the runtime-mutable alerts.ntfy_url:
        # ntfy_url is settable via the public PATCH /settings/notifications, so
        # routing lead PII through it would let an attacker redirect it to a host
        # they control (and thereby also learn the secret topic).
        base_url = (alerts.leads_ntfy_url or "").strip()
        if not base_url:
            return  # no pinned relay configured — do not fall back to a mutable URL
        url = f"{base_url.rstrip('/')}/{topic}"
        headers = {"Title": title, "Priority": "high", "Tags": "moneybag"}

        async def _post() -> int:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.post(
                    url, content=body.encode("utf-8"), headers=headers
                )
            return response.status_code

        status = await asyncio.wait_for(_post(), timeout=5.0)
        if not 200 <= status < 300:
            # Only 2xx is delivered. httpx does not follow redirects by default,
            # so a 3xx (proxy/HTTP->HTTPS) means the push was NOT delivered —
            # treat it as a failure, not a silent success. Never surface the
            # exception/URL: it carries the secret topic.
            logger.warning("lead_intake: new-lead ntfy push returned HTTP %s", status)
            return
        logger.info("lead_intake: new-lead push sent to ntfy topic")
    except Exception as exc:
        # Log only the error CLASS: httpx/asyncio exception text can embed the
        # request URL (and thus the secret topic).
        logger.warning(
            "lead_intake: new-lead ntfy push failed (%s); lead still captured",
            type(exc).__name__,
        )


def _lead_push_title(payload: LeadIntakeRequest) -> str:
    """Notification title (an HTTP header, kept strictly ASCII). Pure, tested
    directly. Only a fully-ASCII name goes in the header; a name with any
    non-ASCII character yields a clean generic title and shows in full in the
    UTF-8 body, avoiding a truncated or mojibake header."""
    name = payload.name.strip()
    ascii_name = _header_value_ascii(name).strip()
    if name and name.isascii() and ascii_name:
        return f"New lead: {ascii_name}"
    return "New lead"


async def _default_lead_notifier(
    payload: LeadIntakeRequest, email: str, phone_digits: str
) -> None:
    """Build and send the new-lead push. The title/body builders are pure and
    tested directly; this is thin glue over the transport."""
    await _publish_lead_ntfy(
        title=_lead_push_title(payload),
        body=_lead_push_body(payload, email, phone_digits),
    )


def _leads_push_configured() -> bool:
    """Whether new-lead pushes are actually enabled (ntfy on + a topic set). Used
    to keep the hourly-volume DB query entirely out of the disabled path, so the
    off-by-default configuration stays inert (no extra COUNT/JOIN per lead)."""
    from ..config import settings

    alerts = settings.alerts
    return bool(alerts.ntfy_enabled and (alerts.leads_ntfy_topic or "").strip())


async def _maybe_notify_new_lead(
    notifier: Any,
    notify_volume: Optional[Any],
    payload: LeadIntakeRequest,
    email: str,
    phone_digits: str,
    contact_id: Any,
) -> None:
    """Fire the new-lead push behind a global hourly ceiling. One fire-and-forget
    guard: a volume-check failure OR timeout fails closed (skip the push, lead
    already captured) and a notifier failure never propagates. The volume query
    runs ONLY when pushes are enabled, so the disabled default does no extra DB
    work, and it is itself time-bounded so a stalled/unindexed COUNT cannot hang
    the awaited intake route ahead of the publish deadline."""
    import asyncio

    try:
        if notify_volume is not None and _leads_push_configured():
            count = await asyncio.wait_for(notify_volume(), timeout=_NOTIFY_VOLUME_TIMEOUT)
            if count > GLOBAL_NOTIFY_HOURLY_CAP:
                logger.warning("lead_intake: notification volume cap hit; skipping push")
                return
        await notifier(payload, email, phone_digits)
    except Exception:
        logger.exception(
            "lead_intake: new-lead notification failed for contact %s", contact_id
        )


async def _process_lead_intake(
    payload: LeadIntakeRequest,
    crm: Any,
    email_provider: Any,
    daily_count: Optional[Any] = None,
    ack_volume: Optional[Any] = None,
    email_history: Optional[Any] = None,
    lead_notifier: Optional[Any] = None,
    notify_volume: Optional[Any] = None,
) -> dict[str, Any]:
    """Core intake flow with injectable providers (unit-testable sans HTTP)."""
    if payload.website.strip():
        # Honeypot tripped: report success so bots learn nothing, touch nothing.
        logger.info("lead_intake: honeypot tripped; dropping submission silently")
        return {"success": True, "email_sent": False}

    email = payload.email.strip().lower()
    phone_digits = re.sub(r"\D", "", payload.phone)
    # A phone with no dialable digits ("n/a", "-") is not a contact channel
    # (PR #2152 review, R1/R2).
    if len(phone_digits) < _MIN_PHONE_DIGITS:
        phone_digits = ""
    if not email and not phone_digits:
        raise LeadValidationError(
            "Provide an email address or a phone number with at least "
            f"{_MIN_PHONE_DIGITS} digits"
        )
    if not email and len(phone_digits) < 10:
        raise LeadValidationError(
            "Provide an email address or a phone number with at least 10 digits"
        )
    if email and not _EMAIL_RE.match(email):
        raise LeadValidationError("Invalid email address")

    # Throttle BEFORE any side effect (CRM write or email).
    if daily_count is not None:
        recent = await daily_count(email, phone_digits)
        if recent >= MAX_DAILY_SUBMISSIONS:
            raise LeadRateLimitedError("Daily submission limit reached")

    attribution = {
        key: value.strip()
        for key, value in {
            "utm_source": payload.utm_source,
            "utm_medium": payload.utm_medium,
            "utm_campaign": payload.utm_campaign,
            "utm_term": payload.utm_term,
            "utm_content": payload.utm_content,
            "gclid": payload.gclid,
            "gbraid": payload.gbraid,
            "wbraid": payload.wbraid,
            "landing_path": payload.landing_path,
            "referrer": payload.referrer,
        }.items()
        if value.strip()
    }
    # Recorded on the interaction so the segment a lead arrived as is evidence
    # on the lead itself, not only on the email we happened to send. Adding a
    # key here cannot shift interaction dedupe: the dedupe key reads only the
    # fixed anchor allowlist and ``metadata["attribution"]``
    # (``services/crm_provider.py`` ``_interaction_anchor`` /
    # ``_interaction_attribution_identity``).
    from ..templates.email import classify_ack_variant

    ack_variant = classify_ack_variant(payload.service)
    metadata = {
        "service": payload.service,
        "frequency": payload.frequency,
        "square_feet": payload.square_feet,
        "source_page": payload.source_page,
        "ack_variant": ack_variant,
        # Submitted channels are recorded even when an existing contact
        # is resolved read-only, so a new callback number/email is never
        # lost (PR #2153 round 4, R1/R6).
        "submitted_email": email,
        "submitted_phone": phone_digits,
        "submitted_address": payload.address.strip(),
    }
    if attribution:
        metadata["attribution"] = attribution

    # This is the one shared EOM inbound boundary. The database provider keeps
    # contact resolution and this interaction in one transaction; lightweight
    # protocol fakes preserve the same observable fallback for unit coverage.
    contact, interaction = await resolve_or_create_eom_inbound_lead_and_log_interaction(
        crm,
        full_name=payload.name.strip(),
        email=email or None,
        phone=phone_digits if len(phone_digits) >= 10 else None,
        address=payload.address.strip() or None,
        source="web",
        source_ref="website_estimate_form",
        tags=["website", "estimate_request"],
        interaction_type="web_form",
        summary=_summary_with_channels(payload, email, phone_digits),
        intent="estimate_request",
        metadata=metadata,
    )
    contact_id = contact.get("id")
    if not contact_id:
        raise RuntimeError("CRM returned contact without id")
    # log_interaction dedupes identical same-day rows and reports it via the
    # public "inserted" flag; a double-submit must not double-email the lead.
    freshly_logged = bool((interaction or {}).get("inserted", True))

    # Instant operator heads-up on every NEW lead — independent of the ack
    # email, so a phone-only lead (no email) still pushes. Guarded by
    # freshly_logged so a same-day double-submit does not double-notify, and by
    # a global hourly ceiling so a distinct-identity flood cannot spam the phone.
    # The notifier is explicit (the route wires the production one); when absent
    # the push is a no-op, so direct callers/tests never touch the live topic.
    if freshly_logged and lead_notifier is not None:
        await _maybe_notify_new_lead(
            lead_notifier, notify_volume, payload, email, phone_digits, contact_id
        )

    email_sent = False
    from ..config import settings as _settings

    email_enabled = bool(getattr(getattr(_settings, "email", None), "enabled", False))
    if email and freshly_logged and email_enabled:
        # Global send ceiling: past the hourly cap the lead is still captured
        # but no acknowledgement goes out (PR #2153 review, R3/R8).
        over_volume = False
        if ack_volume is not None:
            try:
                over_volume = (await ack_volume()) > GLOBAL_ACK_HOURLY_CAP
            except Exception:
                # The volume guard is part of the optional email path: if it
                # cannot be evaluated, skip the send (fail-closed for email)
                # but never fail the already-captured lead.
                logger.exception("lead_intake: ack volume check failed; skipping email")
                over_volume = True
        if over_volume:
            logger.warning("lead_intake: global ack volume cap hit; skipping email")
        else:
            try:
                from ..templates.email import (
                    BUSINESS_EMAIL,
                    BUSINESS_NAME,
                    format_request_acknowledgement,
                )

                subject, body = format_request_acknowledgement(
                    client_name=payload.name,
                    service=payload.service,
                    frequency=payload.frequency,
                )
                # Route this transactional acknowledgement through Resend so it
                # comes from the verified brand domain (info@...) rather than
                # the Gmail account. Other Atlas email is unaffected.
                result = await email_provider.send(
                    to=[email],
                    subject=subject,
                    body=body,
                    from_email=f"{BUSINESS_NAME} <{BUSINESS_EMAIL}>",
                    reply_to=BUSINESS_EMAIL,
                    provider="resend",
                )
                email_sent = bool(result.get("success", True)) if isinstance(result, dict) else True
                if email_sent and email_history is not None:
                    try:
                        message_id = None
                        if isinstance(result, dict):
                            message_id = result.get("message_id") or result.get("id")
                        await email_history.create(
                            to_addresses=[email],
                            subject=subject,
                            body=body,
                            template_type="request_acknowledgement",
                            resend_message_id=message_id,
                            metadata={
                                "source": "website_estimate_form",
                                "contact_id": str(contact_id),
                                # Both evidence records keep the derived variant
                                # next to the raw submitted value. Without the
                                # raw value here, a contact with several
                                # requests gives no way to tell which
                                # submission produced a given email — and the
                                # body text is not a durable substitute,
                                # because A2/A3 replace the template.
                                "service": payload.service,
                                "ack_variant": ack_variant,
                            },
                            business_context_id=EOM_BUSINESS_CONTEXT_ID,
                        )
                    except Exception:
                        # Delivery already succeeded. History is secondary
                        # evidence and must not flip the public outcome or
                        # retry the acknowledgement.
                        logger.exception(
                            "lead_intake: acknowledgement history write failed "
                            "for contact %s",
                            contact_id,
                        )
            except Exception:
                # The acknowledgement must never fail the request: the CRM
                # write is the source of truth and has already committed.
                logger.exception(
                    "lead_intake: acknowledgement email failed for contact %s", contact_id
                )

    # Public response carries no CRM identifiers: contacts.py exposes
    # unauthenticated per-id reads, so returning the UUID here would let an
    # attacker map email/phone -> contact id (PR #2153 review, R3).
    return {"success": True, "email_sent": email_sent}


def _crm_dependency() -> Any:
    from ..services.crm_provider import get_crm_provider

    return get_crm_provider()


def _email_dependency() -> Any:
    from ..services.email_provider import get_email_provider

    return get_email_provider()


def _daily_count_dependency() -> Any:
    return _daily_submission_count


def _ack_volume_dependency() -> Any:
    return _hourly_ack_volume


def _email_history_dependency() -> Any:
    from ..storage.repositories.email import get_email_repo

    return get_email_repo()


def _notify_dependency() -> Any:
    return _default_lead_notifier


def _notify_volume_dependency() -> Any:
    return _hourly_lead_notification_volume


@router.post("/intake")
async def lead_intake(
    payload: LeadIntakeRequest,
    crm: Any = Depends(_crm_dependency),
    email_provider: Any = Depends(_email_dependency),
    daily_count: Any = Depends(_daily_count_dependency),
    ack_volume: Any = Depends(_ack_volume_dependency),
    email_history: Any = Depends(_email_history_dependency),
    lead_notifier: Any = Depends(_notify_dependency),
    notify_volume: Any = Depends(_notify_volume_dependency),
) -> dict[str, Any]:
    """Receive an estimate-form submission from the public website.

    CORS for the form origins is applied by LeadIntakeCORSMiddleware in
    main.py (path-scoped, credential-free), including on error responses.
    """
    try:
        return await _process_lead_intake(
            payload,
            crm=crm,
            email_provider=email_provider,
            daily_count=daily_count,
            ack_volume=ack_volume,
            email_history=email_history,
            lead_notifier=lead_notifier,
            notify_volume=notify_volume,
        )
    except LeadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LeadRateLimitedError as exc:
        raise HTTPException(status_code=429, detail="Too many requests — try again tomorrow") from exc
    except Exception:
        logger.exception("lead_intake: intake failed")
        raise HTTPException(status_code=503, detail="Lead intake temporarily unavailable")
