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

# Acknowledgement variants (ATLAS #2320). A multi-location company is still a
# commercial customer -- "multi-site" describes the shape of the request, not a
# third customer type -- so this is a lead-time acknowledgement variant and
# deliberately NOT a column on ``contacts``. ``contacts.contact_type`` already
# means lifecycle (lead vs customer); a residential/commercial attribute on the
# contact is a separate decision this feature does not force.
ACK_VARIANT_RESIDENTIAL = "residential"
ACK_VARIANT_COMMERCIAL_SINGLE_SITE = "commercial_single_site"
ACK_VARIANT_COMMERCIAL_MULTI_SITE = "commercial_multi_site"
ACK_VARIANT_GENERAL = "general"

# Every value the website forms can submit maps explicitly. ``other`` is an
# allowlisted form option, not an unknown, so it names ``general`` on purpose
# rather than falling through. Anything unrecognised (including empty) also
# resolves to ``general`` so a new form option can never crash intake or
# silently pick residential copy.
_ACK_VARIANT_BY_SERVICE = {
    "residential": ACK_VARIANT_RESIDENTIAL,
    "deep": ACK_VARIANT_RESIDENTIAL,
    "move": ACK_VARIANT_RESIDENTIAL,
    "commercial": ACK_VARIANT_COMMERCIAL_SINGLE_SITE,
    "multi-location-commercial": ACK_VARIANT_COMMERCIAL_MULTI_SITE,
    "other": ACK_VARIANT_GENERAL,
}


def classify_ack_variant(service: str) -> str:
    """Return the acknowledgement variant for a submitted ``service`` value.

    Deterministic and total: every input returns one of the four variants and
    none raises. Intake supplies a validated ``str``, but the whole non-string
    class is guarded rather than only the falsy part of it — a truthy
    non-string (``1``, ``True``, a list, a dict) would otherwise reach
    ``.strip()`` and raise ``AttributeError``. Classification is decided
    server-side from the submitted value; a browser-supplied template name is
    never trusted.
    """
    if not isinstance(service, str):
        return ACK_VARIANT_GENERAL
    return _ACK_VARIANT_BY_SERVICE.get(service.strip().lower(), ACK_VARIANT_GENERAL)


ACK_SUBJECT = "We received your estimate request - " + BUSINESS_NAME

ACK_TEMPLATE = """Hi {client_name},

Thanks for requesting a free estimate from {business_name} - it just landed \
with a real person, and we'll get back to you within 24 hours.

{request_line}Here's what happens next:

1. We'll give you a call to book a day and time that works for you for a quick \
estimate walkthrough. The walkthrough usually takes less than 20 minutes.

2. We'll send our estimate team out to look over the spaces you want cleaned. \
Your two team members will be Mayra Canfield and Tina Gomez.

3. Show them around and tell them what you'd like cleaned and how often.

4. Before Mayra and Tina leave, they'll give you the cost to clean your space. \
If the pricing works for you, you can schedule your first cleaning right then.

5. Estimates are FREE and there's no obligation. If the pricing isn't right for \
you, we won't try to talk you into it.

Questions in the meantime? Call us at {business_phone}, or just reply to
this email.

Talk soon,
The {business_name} Team
{business_phone} | {business_email}
{business_website}
"""


COMMERCIAL_SINGLE_SITE_TEMPLATE = """Hi {client_name},

Thank you for requesting a free estimate from {business_name}

My name is Juan Canfield and I'll be giving you a call shortly.

{request_line}Here's what happens next:

1. I will give you a call within 24 hours to ask a few questions about \
your facility and to set up a time and day for a walk-through.

2. We'll walk the space with you when you have some free time available. \
We have a flexible schedule for walk-throughs and estimates.

3. Tell us what areas are the most important to you. Which areas need the \
most attention, anything that needs special handling, and how often you \
want us on site.

4. After the walk-through, we'll put together an estimate covering the \
scope of work plus any details covered in the walk-through and email it to \
you for review.

5. Estimates are FREE and there is no obligation. If we're not the right \
fit, we won't try to talk you into it.

If you have any questions in the meantime call me at {business_phone}, or
just reply to this email.

Talk soon,
Juan Canfield
{business_name}
{business_phone} | {business_email}
{business_website}
"""

COMMERCIAL_MULTI_SITE_TEMPLATE = """Hi {client_name},

Thank you for requesting a free estimate from {business_name}

My name is Juan Canfield and I'll be giving you a call shortly.

{request_line}Multi-location work takes a little more planning than a \
single building, so here's what happens next:

1. I will give you a call within 24 hours. That first call is to \
understand the whole picture before we put any numbers to it.

2. We'll go through your locations together - where they are, how they \
differ from each other, and who we'd be working with at each one. Some \
sites need more attention than others, and we'd rather know that up front.

3. From there we'll work out which locations need a walk-through. Not \
every site always does - it depends on how similar they are.

4. Once we understand the locations and the schedules you need, we'll put \
together a scope for each one and email you an estimate for review.

5. Estimates are FREE and there is no obligation. If we're not the right \
fit, we won't try to talk you into it.

If you have any questions in the meantime call me at {business_phone}, or
just reply to this email.

Talk soon,
Juan Canfield
{business_name}
{business_phone} | {business_email}
{business_website}
"""

# "custom" is a real website form option, but it is a placeholder for "we'll
# work it out on the call" rather than a cadence that can be read back in a
# sentence -- "a custom commercial cleaning" does not parse as English. It is
# therefore dropped exactly like a blank frequency, and the request line falls
# back to its cadence-free wording.
_UNSPOKEN_FREQUENCIES = frozenset({"custom"})


def _spoken_frequency(frequency: str) -> str:
    """Return the frequency only when it reads naturally inside a sentence."""
    cleaned = frequency.strip()
    if cleaned.lower() in _UNSPOKEN_FREQUENCIES:
        return ""
    return cleaned


def _commercial_request_line(frequency: str, *, multi_site: bool) -> str:
    """Echo the submitted cadence back in the commercial voice.

    The residential template keeps its raw ``service, frequency`` echo
    unchanged (operator decision, 2026-08-09: leave residential as is). The
    commercial variants speak the cadence in a sentence instead, so the value
    has to be one that reads naturally after "a".
    """
    spoken = _spoken_frequency(frequency)
    if multi_site:
        detail = f"{spoken} cleaning" if spoken else "cleaning"
        return f"You requested {detail} for multiple locations.\n\n"
    detail = f"a {spoken} commercial cleaning" if spoken else "a commercial cleaning"
    return f"You requested {detail}.\n\n"


def format_request_acknowledgement(
    client_name: str,
    service: str = "",
    frequency: str = "",
) -> tuple[str, str]:
    """Render (subject, body) for the request acknowledgement.

    The variant is derived here from ``service`` rather than accepted as an
    argument, so the email a lead receives and the ``ack_variant`` intake
    records cannot disagree: both call ``classify_ack_variant`` on the same
    submitted value. Callers keep the existing signature.

    ``service``/``frequency`` echo the form selections back when present so
    the lead sees their request was captured accurately; both are optional.
    """
    variant = classify_ack_variant(service)

    if variant == ACK_VARIANT_COMMERCIAL_SINGLE_SITE:
        template = COMMERCIAL_SINGLE_SITE_TEMPLATE
        request_line = _commercial_request_line(frequency, multi_site=False)
    elif variant == ACK_VARIANT_COMMERCIAL_MULTI_SITE:
        template = COMMERCIAL_MULTI_SITE_TEMPLATE
        request_line = _commercial_request_line(frequency, multi_site=True)
    else:
        # residential AND general. `general` covers the form's "Other" option
        # and anything unrecognised, where the request is not known to be
        # commercial -- so it keeps exactly the copy those leads receive
        # today rather than being pointed at unapproved wording.
        template = ACK_TEMPLATE
        details = ", ".join(
            part for part in (service.strip(), frequency.strip()) if part
        )
        request_line = f"Your request: {details}.\n\n" if details else ""

    body = template.format(
        client_name=client_name.strip() or "there",
        business_name=BUSINESS_NAME,
        business_phone=BUSINESS_PHONE,
        business_email=BUSINESS_EMAIL,
        business_website=BUSINESS_WEBSITE,
        request_line=request_line,
    )
    return ACK_SUBJECT, body
