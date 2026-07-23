"""Request-acknowledgement email for website estimate submissions.

Sent immediately when a lead submits the estimate form on
effinghamofficemaids.com (issue #2151 Phase 1). Deliberately price-free and
date-free: at request time no walkthrough has happened, so the message only
sets expectations. Copy guardrails (operator, 2026-07): no dollar figures, no
clock promises beyond "within 24 hours", "estimate" never "quote", sell the
team (separate areas at once) without per-size time claims.
"""

from .estimate_confirmation import (
    BUSINESS_EMAIL,
    BUSINESS_NAME,
    BUSINESS_PHONE,
    BUSINESS_WEBSITE,
)

ACK_SUBJECT = "We received your estimate request - " + BUSINESS_NAME

ACK_TEMPLATE = """Hi {client_name},

Thanks for requesting a free estimate from {business_name} - it just landed \
with a real person, and we'll get back to you within 24 hours.

{request_line}Here's what happens next:

1. We reach out to ask a few quick questions about your home and which rooms
   you'd like covered - the whole home or just the spaces you actually use.
2. You get a free estimate specific to your home. Every home is different,
   so we don't price from a one-size rate card.
3. If it works for you, we schedule your first clean. We send a team that
   works separate areas at once - in and out fast, your space back sooner.

There's no obligation and no pressure - the estimate is free either way.

Questions in the meantime? Call us at {business_phone}, or just reply to
this email.

Talk soon,
The {business_name} Team
{business_phone} | {business_email}
{business_website}
"""


def format_request_acknowledgement(
    client_name: str,
    service: str = "",
    frequency: str = "",
) -> tuple[str, str]:
    """Render (subject, body) for the request acknowledgement.

    ``service``/``frequency`` echo the form selections back when present so
    the lead sees their request was captured accurately; both are optional.
    """
    details = ", ".join(part for part in (service.strip(), frequency.strip()) if part)
    request_line = f"Your request: {details}.\n\n" if details else ""
    body = ACK_TEMPLATE.format(
        client_name=client_name.strip() or "there",
        business_name=BUSINESS_NAME,
        business_phone=BUSINESS_PHONE,
        business_email=BUSINESS_EMAIL,
        business_website=BUSINESS_WEBSITE,
        request_line=request_line,
    )
    return ACK_SUBJECT, body
