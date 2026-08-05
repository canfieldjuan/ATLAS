"""Approve-and-send orchestration for EOM onboarding email drafts.

Executes the protocol documented in migration 360: claim exactly one
pending draft into 'sending' (readiness predicate inline), send OUTSIDE
any open transaction with the draft id as the transport idempotency key,
and confirm 'sending' -> 'sent' only after transport acceptance. A stuck
'sending' row is operator reconciliation evidence, never a silent retry.

The transport is a direct Resend POST over httpx, mirroring
atlas_brain/content_ops_deflection_delivery.py. The generic email port
cannot serve this path: it exposes no idempotency key, and its send()
lazily imports the atlas_brain.tools registry, whose dependencies are
deliberately absent from the slim EOM Render profile -- failing there
would wedge the freshly claimed row mid-request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from ..templates.email.estimate_confirmation import (
    BUSINESS_EMAIL,
    BUSINESS_NAME,
)

# NOTE: atlas_brain.config is imported lazily inside the functions that
# need it. The slim EOM Render profile asserts that importing main_eom
# does not load atlas_brain.config at module-import time; this module is
# imported by the funnel router, so a top-level settings import would
# break that isolation contract.

logger = logging.getLogger("atlas.services.eom_onboarding_drafts")

RESEND_API_URL = "https://api.resend.com/emails"


class EOMOnboardingDraftError(Exception):
    """HTTP-mappable failure for the private draft-approval commands."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class EOMOnboardingDraftApproval:
    """One office approval command for one draft."""

    draft_id: str
    actor_id: int
    actor_name: str


def onboarding_draft_idempotency_key(draft_id: str) -> str:
    """Deterministic Resend idempotency key for one draft.

    Derived purely from the draft id so any retry of the same draft --
    including an operator-driven retry after a crash between send and
    confirm -- reuses the exact key. Resend dedupes identical keys
    server-side for 24 hours.
    """
    return f"eom-onboarding-draft:{draft_id}"


def _is_idempotency_conflict(response: Any) -> bool:
    """Match Resend's documented invalid_idempotent_request 409 error."""
    try:
        data = response.json()
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    name = str(data.get("name") or "")
    message = str(data.get("message") or "")
    return (
        "invalid_idempotent_request" in name
        or "invalid_idempotent_request" in message
    )


async def send_onboarding_email(
    *,
    to: str,
    subject: str,
    body: str,
    idempotency_key: str,
    http_client: Any | None = None,
) -> dict[str, Any]:
    """POST one onboarding email to Resend with server-side dedupe.

    Returns {"message_id": str | None, "idempotent_replay": bool}. A 409
    invalid_idempotent_request response is positive proof the original
    send for this key was accepted, so it reports delivered rather than
    raising (same semantics as the deflection delivery sender).
    """
    payload = {
        "from": f"{BUSINESS_NAME} <{BUSINESS_EMAIL}>",
        "to": [to],
        "subject": subject,
        "text": body,
        "reply_to": BUSINESS_EMAIL,
    }
    from ..config import settings

    headers = {
        # Stripped to match the preflight admission: a padded key must not
        # reach Resend as a malformed Authorization header.
        "Authorization": f"Bearer {(settings.email.api_key or '').strip()}",
        "Content-Type": "application/json",
        # Resend dedupes identical Idempotency-Key values server-side for
        # 24h, so a retried send cannot produce a second email.
        "Idempotency-Key": idempotency_key,
    }
    if http_client is not None:
        response = await http_client.post(
            RESEND_API_URL, json=payload, headers=headers
        )
    else:
        async with httpx.AsyncClient(timeout=settings.email.timeout) as client:
            response = await client.post(
                RESEND_API_URL, json=payload, headers=headers
            )
    if getattr(response, "status_code", 200) == 409 and _is_idempotency_conflict(
        response
    ):
        return {"message_id": None, "idempotent_replay": True}
    response.raise_for_status()
    data = response.json()
    return {"message_id": str(data["id"]), "idempotent_replay": False}


def _require_transport_configured() -> None:
    """Refuse to claim a draft the transport cannot serve."""
    from ..config import settings

    # A whitespace-only key would pass a truthiness check, claim the draft,
    # and only then fail at Resend -- wedging the row in 'sending' instead
    # of taking the 503-before-claim path this guard promises.
    if not settings.email.enabled or not (settings.email.api_key or "").strip():
        raise EOMOnboardingDraftError(
            503,
            "Email transport is not configured; the draft was not claimed",
        )


def _draft_lifecycle_callables(crm: Any) -> tuple[Any, Any]:
    claimer = getattr(crm, "claim_eom_onboarding_draft", None)
    confirmer = getattr(crm, "confirm_eom_onboarding_draft_sent", None)
    if not callable(claimer) or not callable(confirmer):
        raise RuntimeError(
            "Configured CRM provider cannot approve EOM onboarding drafts"
        )
    return claimer, confirmer


async def _record_send_evidence(
    crm: Any,
    draft: dict[str, Any],
    message_id: str | None,
    *,
    email_history: Any | None = None,
) -> None:
    """Secondary evidence after confirmed delivery; never flips the outcome."""
    try:
        if email_history is None:
            from ..storage.repositories.email import EmailRepository

            # The sent_emails row must land in the same store that owns the
            # draft it describes. The funnel CRM provider is bound to its own
            # connection string in the slim profile, where the global pool
            # may point at a different database (or be uninitialized).
            email_history = EmailRepository(pool=getattr(crm, "pool", None))
        await email_history.create(
            to_addresses=[str(draft["recipient_email"])],
            subject=str(draft["subject"]),
            body=str(draft["body"]),
            template_type="onboarding_welcome",
            resend_message_id=message_id,
            metadata={
                "source": "eom_funnel_onboarding_draft",
                "contact_id": str(draft["contact_id"]),
                "draft_id": str(draft["draft_id"]),
            },
            business_context_id="effingham_maids",
        )
    except Exception:
        logger.warning(
            "Onboarding email sent but sent_emails history write failed "
            "for draft %s",
            draft["draft_id"],
            exc_info=True,
        )
    try:
        log_interaction = getattr(crm, "log_interaction", None)
        if callable(log_interaction):
            await log_interaction(
                str(draft["contact_id"]),
                "email",
                "Onboarding welcome email sent to "
                f"{draft['recipient_email']} (draft {draft['draft_id']})",
            )
    except Exception:
        logger.warning(
            "Onboarding email sent but CRM interaction log failed for "
            "draft %s",
            draft["draft_id"],
            exc_info=True,
        )


async def record_operator_confirmed_send_evidence(
    crm: Any,
    draft: dict[str, Any],
    *,
    email_history: Any | None = None,
) -> None:
    """Sent-email history for the operator confirm-sent recovery path.

    The transport accepted the send before the process died, so the
    message id was never observed: the history row carries a null
    transport id but otherwise the same evidence as the normal approve
    path -- without this, a crash-recovered delivery would be missing
    from customer history forever.
    """
    await _record_send_evidence(crm, draft, None, email_history=email_history)


async def approve_and_send_eom_onboarding_draft(
    crm: Any,
    command: EOMOnboardingDraftApproval,
    *,
    sender: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    email_history: Any | None = None,
) -> dict[str, Any]:
    """Claim, send, then confirm one onboarding draft (migration 360)."""
    claimer, confirmer = _draft_lifecycle_callables(crm)
    if sender is None:
        _require_transport_configured()
        sender = send_onboarding_email

    claim = await claimer(
        draft_id=command.draft_id,
        actor_id=command.actor_id,
        actor_name=command.actor_name,
    )
    if not claim["claimed"]:
        # Already sent: idempotent replay, no second transport call.
        return claim["draft"]
    draft = claim["draft"]

    try:
        send_result = await sender(
            to=str(draft["recipient_email"]),
            subject=str(draft["subject"]),
            body=str(draft["body"]),
            idempotency_key=onboarding_draft_idempotency_key(command.draft_id),
        )
    except Exception as exc:
        # The row stays 'sending' as operator reconciliation evidence
        # (migration 360 step 4): the send outcome is unknown, so neither
        # a silent retry nor a rollback to 'pending' is safe.
        raise EOMOnboardingDraftError(
            502,
            "Onboarding email transport failed; the draft stays in "
            "'sending' and requires reconciliation against the transport "
            "log",
        ) from exc

    confirmed = await confirmer(draft_id=command.draft_id)
    await _record_send_evidence(
        crm,
        confirmed,
        send_result.get("message_id"),
        email_history=email_history,
    )
    result = dict(confirmed)
    result["resend_message_id"] = send_result.get("message_id")
    result["transport_idempotent_replay"] = bool(
        send_result.get("idempotent_replay")
    )
    return result
