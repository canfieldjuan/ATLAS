"""Commercial acknowledgement copy (ATLAS #2320, slices A2 + A3).

A1 classified and recorded without changing any email. This slice is the copy
change: `commercial` and `multi-location-commercial` now render their own
templates. The residential email stays frozen — operator decision 2026-08-09,
"leave it as is" — so the byte-identical anchor from A1 has to keep holding.
"""

import re

import pytest

from atlas_brain.api.leads import _process_lead_intake
from atlas_brain.templates.email import (
    ACK_VARIANT_COMMERCIAL_MULTI_SITE,
    ACK_VARIANT_COMMERCIAL_SINGLE_SITE,
    classify_ack_variant,
    format_request_acknowledgement,
)
from atlas_brain.templates.email.request_acknowledgement import (
    ACK_TEMPLATE,
    COMMERCIAL_MULTI_SITE_TEMPLATE,
    COMMERCIAL_SINGLE_SITE_TEMPLATE,
)

from .test_leads_intake import _crm, _email_history, _email_provider, _payload


@pytest.fixture(autouse=True)
def _email_enabled(monkeypatch):
    """`test_leads_intake`'s autouse fixture does not cross module boundaries."""
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.email, "enabled", True)


# The website form's own frequency options, transcribed from the markup
# (Effingham_Office_Maids_Website: contact.html, commercial-estimate.html)
# rather than read from the implementation. `frequency` is a free-text field
# server-side (`leads.py` `Field(default="", max_length=120)`), so these are
# what real submissions carry, not what the API constrains.
FORM_FREQUENCIES = ("daily", "weekly", "bi-weekly", "monthly", "one-time", "custom")

# "custom" is a placeholder for "we'll work it out on the call", not a cadence
# that can be spoken in a sentence.
UNSPOKEN_FREQUENCIES = ("custom",)
SPOKEN_FREQUENCIES = tuple(f for f in FORM_FREQUENCIES if f not in UNSPOKEN_FREQUENCIES)


# --- template selection ---------------------------------------------------


def test_single_site_commercial_renders_the_single_site_template():
    _, body = format_request_acknowledgement("Gretchen", "commercial", "one-time")
    assert "set up a time and day for a walk-through" in body
    assert "for multiple locations" not in body


def test_multi_site_commercial_renders_the_multi_site_template():
    _, body = format_request_acknowledgement(
        "Acme", "multi-location-commercial", "weekly"
    )
    assert "Multi-location work takes a little more planning" in body
    assert "set up a time and day for a walk-through" not in body


def test_the_two_commercial_templates_are_not_the_same_copy():
    _, single = format_request_acknowledgement("Acme", "commercial", "weekly")
    _, multi = format_request_acknowledgement(
        "Acme", "multi-location-commercial", "weekly"
    )
    assert single != multi


@pytest.mark.parametrize(
    ("service", "expected_variant"),
    [
        ("commercial", ACK_VARIANT_COMMERCIAL_SINGLE_SITE),
        ("Commercial", ACK_VARIANT_COMMERCIAL_SINGLE_SITE),
        ("  MULTI-LOCATION-COMMERCIAL  ", ACK_VARIANT_COMMERCIAL_MULTI_SITE),
    ],
)
def test_rendering_agrees_with_the_recorded_variant(service, expected_variant):
    """The email sent and the `ack_variant` recorded cannot disagree.

    `format_request_acknowledgement` derives the variant from `service` with
    the same `classify_ack_variant` intake uses, rather than accepting a
    caller-supplied variant that could drift from the recorded one.
    """
    assert classify_ack_variant(service) == expected_variant
    _, body = format_request_acknowledgement("Acme", service, "weekly")
    expected_marker = {
        ACK_VARIANT_COMMERCIAL_SINGLE_SITE: "set up a time and day for a walk-through",
        ACK_VARIANT_COMMERCIAL_MULTI_SITE: "Multi-location work takes a little more",
    }[expected_variant]
    assert expected_marker in body


# --- the echoed cadence ---------------------------------------------------


@pytest.mark.parametrize("frequency", SPOKEN_FREQUENCIES)
def test_single_site_speaks_every_spoken_frequency(frequency):
    _, body = format_request_acknowledgement("Gretchen", "commercial", frequency)
    assert f"You requested a {frequency} commercial cleaning." in body


@pytest.mark.parametrize("frequency", SPOKEN_FREQUENCIES)
def test_multi_site_speaks_every_spoken_frequency(frequency):
    _, body = format_request_acknowledgement(
        "Acme", "multi-location-commercial", frequency
    )
    assert f"You requested {frequency} cleaning for multiple locations." in body


@pytest.mark.parametrize("frequency", ["", "   ", "custom", "CUSTOM", "  Custom  "])
def test_unspoken_frequency_falls_back_to_cadence_free_wording(frequency):
    """A blank or `custom` cadence must not produce broken English.

    "You requested a custom commercial cleaning" does not parse, so `custom`
    is dropped exactly like a blank value.
    """
    _, single = format_request_acknowledgement("Gretchen", "commercial", frequency)
    assert "You requested a commercial cleaning." in single

    _, multi = format_request_acknowledgement(
        "Acme", "multi-location-commercial", frequency
    )
    assert "You requested cleaning for multiple locations." in multi


@pytest.mark.parametrize("frequency", FORM_FREQUENCIES + ("", "   "))
def test_commercial_copy_never_double_spaces_or_dangles_the_cadence(frequency):
    """Whatever the cadence, the sentence stays well-formed."""
    for service in ("commercial", "multi-location-commercial"):
        _, body = format_request_acknowledgement("Acme", service, frequency)
        line = next(
            line for line in body.splitlines() if line.startswith("You requested")
        )
        assert "  " not in line, line
        assert line.endswith("."), line
        assert " ." not in line, line


# --- copy guardrails, as executable checks --------------------------------

ALL_TEMPLATES = (
    ACK_TEMPLATE,
    COMMERCIAL_SINGLE_SITE_TEMPLATE,
    COMMERCIAL_MULTI_SITE_TEMPLATE,
)


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_no_template_contains_a_dollar_figure(template):
    """Operator copy guardrail: no dollar figures in public copy, ever."""
    assert re.search(r"\$\s*\d", template) is None


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_no_template_says_quote(template):
    """Operator copy guardrail: it is an "estimate", never a "quote"."""
    assert re.search(r"\bquotes?\b", template, re.IGNORECASE) is None


@pytest.mark.parametrize(
    "template", (COMMERCIAL_SINGLE_SITE_TEMPLATE, COMMERCIAL_MULTI_SITE_TEMPLATE)
)
def test_commercial_templates_name_juan_not_the_residential_estimate_team(template):
    assert "Juan Canfield" in template
    assert "Mayra" not in template
    assert "Tina" not in template


def test_multi_site_makes_none_of_the_promises_it_must_not_make():
    """The multi-site email must not carry single-building promises.

    Spec (issue #2320): no immediate price at the first visit, no claim that
    the estimate team personally inspects every location, no implication of one
    identical checklist for every building.
    """
    body = COMMERCIAL_MULTI_SITE_TEMPLATE
    assert "right then" not in body
    assert "before they leave" not in body.lower()
    assert "every location" not in body.lower()
    assert "each location" not in body.lower()


# Every sentence in which any variant may promise "24 hours", approved
# verbatim. "Within 24 hours" is the INITIAL-RESPONSE promise only: it must
# never attach to walkthrough completion or estimate delivery.
#
# This is an exact whitelist rather than a keyword rule on purpose. A rule like
# "no line containing '24 hours' may also contain 'estimate'" cannot tell
# "requesting a free estimate ... within 24 hours" (fine — that names what was
# requested) from "we'll send the estimate within 24 hours" (a promise EOM
# cannot keep). A keyword guard that flags correct copy gets relaxed until it
# means nothing; an exact list fails on ANY edit to a 24-hour sentence and asks
# a human to re-approve it, which is the actual review this needs.
APPROVED_24_HOUR_SENTENCES = {
    "residential": [
        "Thanks for requesting a free estimate from Effingham Office Maids - it "
        "just landed with a real person, and we'll get back to you within 24 hours."
    ],
    "commercial": [
        "1. I will give you a call within 24 hours to ask a few questions about "
        "your facility and to set up a time and day for a walk-through."
    ],
    "multi-location-commercial": [
        "1. I will give you a call within 24 hours. That first call is to "
        "understand the whole picture before we put any numbers to it."
    ],
}


@pytest.mark.parametrize("service", sorted(APPROVED_24_HOUR_SENTENCES))
def test_only_approved_sentences_promise_24_hours(service):
    _, body = format_request_acknowledgement("Acme", service, "weekly")
    promised = [line for line in body.splitlines() if "24 hours" in line]
    assert promised == APPROVED_24_HOUR_SENTENCES[service]


@pytest.mark.parametrize("service", sorted(APPROVED_24_HOUR_SENTENCES))
def test_estimate_delivery_is_never_inside_the_24_hour_window(service):
    """The estimate is promised after the walkthrough, never within 24 hours.

    Checked as an ordering property rather than a keyword: the sentence that
    delivers the estimate must be a different sentence from the one that
    promises 24 hours.
    """
    _, body = format_request_acknowledgement("Acme", service, "weekly")
    delivery_lines = [
        line
        for line in body.splitlines()
        if re.search(r"\bemail it to you\b|\bemail you an estimate\b", line)
    ]
    for line in delivery_lines:
        assert "24 hours" not in line


@pytest.mark.parametrize("service", ["commercial", "multi-location-commercial"])
def test_rendered_commercial_body_has_no_unfilled_placeholders(service):
    _, body = format_request_acknowledgement("Acme", service, "weekly")
    assert "{" not in body
    assert "}" not in body
    assert "Acme" in body
    assert "(217) 207-3097" in body
    assert "info@effinghamofficemaids.com" in body


@pytest.mark.parametrize("service", ["commercial", "multi-location-commercial"])
def test_missing_client_name_still_renders_a_greeting(service):
    _, body = format_request_acknowledgement("   ", service, "weekly")
    assert body.startswith("Hi there,")


# --- through intake -------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "marker"),
    [
        ("commercial", "set up a time and day for a walk-through"),
        ("multi-location-commercial", "Multi-location work takes a little more"),
    ],
)
async def test_intake_sends_the_commercial_body_for_commercial_services(service, marker):
    """The real intake coroutine sends the commercial copy, not just the renderer.

    A template can be correct while the send path still passes the wrong
    arguments, so this asserts the body actually handed to the provider.
    """
    crm, provider, history = _crm(), _email_provider(), _email_history()

    await _process_lead_intake(
        _payload(service=service, frequency="weekly"),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    provider.send.assert_awaited()
    sent_body = provider.send.await_args.kwargs["body"]
    assert marker in sent_body
    assert "Mayra Canfield and Tina Gomez" not in sent_body

    # The stored copy must be the copy that was sent.
    assert history.create.await_args.kwargs["body"] == sent_body


@pytest.mark.asyncio
async def test_intake_still_sends_the_original_copy_for_residential():
    """The frozen path, asserted through intake rather than only the renderer."""
    crm, provider, history = _crm(), _email_provider(), _email_history()

    await _process_lead_intake(
        _payload(service="residential", frequency="monthly"),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    sent_body = provider.send.await_args.kwargs["body"]
    assert "Mayra Canfield and Tina Gomez" in sent_body
    assert "Juan Canfield" not in sent_body
