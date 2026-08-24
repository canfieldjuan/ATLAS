"""Onboarding welcome email for leads whose first cleaning is booked.

Enqueued as a pending draft when the office books the first cleaning
(lead_stage -> won). Nothing sends from here: the draft sits in
eom_onboarding_email_drafts until the office approves it (issue #2188
office-controlled conversion, 2026-07-26). Copy guardrails follow
request_acknowledgement.py: no dollar figures, no clock promises,
"estimate" never "quote".
"""

from .estimate_confirmation import (
    BUSINESS_EMAIL,
    BUSINESS_NAME,
    BUSINESS_PHONE,
    BUSINESS_WEBSITE,
)

ONBOARDING_SUBJECT = "Welcome aboard - " + BUSINESS_NAME

ONBOARDING_TEMPLATE = """Hi {client_name},

Welcome to {business_name}! Your first cleaning is on the calendar and we're
glad to have you.

A few things to know before your first visit:

1. Your cleaning team arrives with everything they need - supplies and
equipment are on us. If you prefer we use a product you already have, just
leave it out and let us know.

2. You don't need to be there. Many of our clients give us entry
instructions and come back to a finished space. If you'd rather be present
for the first visit, that works too.

3. After we finish, please take a walk through the space and let us know if
anything needs attention. We'll take care of it right away.

4. After the first cleaning, tell us what you thought. We adjust to your
feedback, and there's no long-term contract holding you here - we keep the
schedule because you want us back.

Need to change anything about your first visit? Call us at {business_phone}
or just reply to this email.

See you soon,
The {business_name} Team
{business_phone} | {business_email}
{business_website}
"""


def format_onboarding_welcome(client_name: str) -> tuple[str, str]:
    """Render (subject, body) for the onboarding welcome draft."""
    body = ONBOARDING_TEMPLATE.format(
        client_name=client_name.strip() or "there",
        business_name=BUSINESS_NAME,
        business_phone=BUSINESS_PHONE,
        business_email=BUSINESS_EMAIL,
        business_website=BUSINESS_WEBSITE,
    )
    return ONBOARDING_SUBJECT, body
