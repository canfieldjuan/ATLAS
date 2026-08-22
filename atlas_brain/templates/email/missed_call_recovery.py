"""Deterministic EOM residential missed-call recovery messages.

This module deliberately owns copy only.  Delivery eligibility, scheduling,
idempotency, and state all live in ``services.eom_missed_call_recovery`` so a
template render can never create a sequence or send customer mail.
"""

from __future__ import annotations

from dataclasses import dataclass

from .estimate_confirmation import BUSINESS_NAME, BUSINESS_PHONE


@dataclass(frozen=True)
class MissedCallRecoveryEmail:
    """One immutable message snapshot suitable for the durable outbox."""

    step_number: int
    subject: str
    body: str


_SUBJECTS = {
    1: "I tried reaching you about your cleaning estimate",
    2: "Still interested in a cleaning estimate?",
    3: "Should I keep your estimate request open?",
}


def _first_name(value: str) -> str:
    """Return a safe, human greeting from canonical display text.

    Contacts normally require a full name.  The fallback remains friendly for
    historical rows that are malformed or were created by a channel that only
    supplied a phone number; it never changes canonical contact data.
    """

    if not isinstance(value, str):
        return "there"
    parts = value.strip().split()
    return parts[0] if parts else "there"


def render_missed_call_recovery_email(
    *,
    step_number: int,
    full_name: str,
    booking_link: str,
) -> MissedCallRecoveryEmail:
    """Render one approved plaintext recovery email.

    ``booking_link`` is a deploy-time setting supplied by the delivery service;
    it is not a browser argument and is not stored in a response projection.
    """

    if step_number not in _SUBJECTS:
        raise ValueError("missed-call recovery step number is invalid")
    if not isinstance(booking_link, str) or not booking_link.strip():
        raise ValueError("missed-call recovery booking link is required")

    first_name = _first_name(full_name)
    link = booking_link.strip()
    if step_number == 1:
        body = f"""Hi {first_name},

I just tried calling you about your residential cleaning estimate.

Whenever you have a moment, you can call or text me at {BUSINESS_PHONE}. You can also reply to this email or request a convenient estimate time here:

{link}

Requested times are not confirmed until we verify the appointment with you by phone or text.

Thanks,
Juan
{BUSINESS_NAME}
{BUSINESS_PHONE}
"""
    elif step_number == 2:
        body = f"""Hi {first_name},

I wanted to follow up in case we missed each other.

I’d be happy to answer your questions and provide a residential cleaning estimate. You can call or text me at {BUSINESS_PHONE}, reply to this email, or request a time here:

{link}

We’ll confirm the appointment by phone or text before coming out.

Thanks,
Juan
"""
    else:
        body = f"""Hi {first_name},

I haven’t been able to catch you by phone, so I wanted to check in one last time.

If you’re still looking for cleaning service, reply to this email, call or text {BUSINESS_PHONE}, or request an estimate time here:

{link}

If now isn’t the right time, no problem. You’re welcome to reach out whenever you’re ready.

Requested times are not confirmed until we verify them by phone or text.

Thanks,
Juan
"""
    return MissedCallRecoveryEmail(
        step_number=step_number,
        subject=_SUBJECTS[step_number],
        body=body,
    )
