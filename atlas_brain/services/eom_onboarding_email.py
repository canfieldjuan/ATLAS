"""Office-approved EOM onboarding email delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class EOMOnboardingEmailError(Exception):
    """HTTP-mappable failure for onboarding email approval."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class EOMOnboardingEmailApproval:
    draft_id: str
    actor_id: int
    actor_name: str


def _blocked_message(draft: dict[str, Any] | None) -> tuple[int, str]:
    if draft is None:
        return 404, "EOM onboarding email draft not found"
    status = str(draft.get("status") or "")
    if status == "sent":
        return 200, "EOM onboarding email already sent"
    if status == "sending":
        return 409, "EOM onboarding email send is already in progress"
    if status == "revoked":
        return 409, "EOM onboarding email draft is revoked"
    blocker = str(draft.get("blocker") or "").strip()
    if blocker:
        return 409, f"EOM onboarding email draft is blocked: {blocker}"
    if not str(draft.get("recipient_email") or "").strip():
        return 409, "EOM onboarding email draft has no recipient"
    return 409, "EOM onboarding email draft is not claimable"


async def approve_and_send_eom_onboarding_email(
    crm: Any,
    email_provider: Any,
    approval: EOMOnboardingEmailApproval,
) -> dict[str, Any]:
    """Claim one pending onboarding email draft, send it, then confirm delivery."""
    claimer = getattr(crm, "claim_eom_onboarding_email_draft", None)
    confirmer = getattr(crm, "confirm_eom_onboarding_email_sent", None)
    getter = getattr(crm, "get_eom_onboarding_email_draft", None)
    sender = getattr(email_provider, "send", None)
    if not callable(claimer) or not callable(confirmer) or not callable(getter):
        raise RuntimeError("Configured CRM provider cannot approve EOM onboarding emails")
    if not callable(sender):
        raise RuntimeError("Configured email provider cannot send EOM onboarding emails")

    claimed = await claimer(
        draft_id=approval.draft_id,
        actor_id=approval.actor_id,
        actor_name=approval.actor_name,
    )
    if claimed is None:
        draft = await getter(draft_id=approval.draft_id)
        status_code, message = _blocked_message(draft)
        if status_code == 200:
            return {
                "draft_id": approval.draft_id,
                "contact_id": str(draft.get("contact_id") or ""),
                "status": "sent",
                "idempotent": True,
            }
        raise EOMOnboardingEmailError(status_code, message)

    await sender(
        to=[str(claimed["recipient_email"])],
        subject=str(claimed["subject"]),
        body=str(claimed["body"]),
        headers={
            "X-Atlas-EOM-Onboarding-Draft-ID": str(claimed["draft_id"]),
        },
    )
    confirmed = await confirmer(
        draft_id=approval.draft_id,
        actor_id=approval.actor_id,
        actor_name=approval.actor_name,
    )
    if confirmed is None:
        raise EOMOnboardingEmailError(
            409,
            "EOM onboarding email was accepted by transport but could not be confirmed",
        )
    return {
        "draft_id": str(confirmed["draft_id"]),
        "contact_id": str(confirmed["contact_id"]),
        "status": str(confirmed["status"]),
        "lead_stage": str(confirmed["lead_stage"]),
        "idempotent": False,
    }
