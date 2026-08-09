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
    ACK_VARIANT_GENERAL,
    ACK_VARIANT_RESIDENTIAL,
    classify_ack_variant,
    format_request_acknowledgement,
    request_acknowledgement,
)
from atlas_brain.templates.email.request_acknowledgement import (
    ACK_TEMPLATE,
    ACK_TEMPLATE_BY_VARIANT,
    COMMERCIAL_MULTI_SITE_TEMPLATE,
    COMMERCIAL_SINGLE_SITE_TEMPLATE,
    SPEAKABLE_FREQUENCIES,
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
# what the form offers, NOT what the API constrains -- any string up to 120
# characters can arrive.
FORM_FREQUENCIES = ("daily", "weekly", "bi-weekly", "monthly", "one-time", "custom")

# The contracted allowlist, written here as literals rather than imported, so
# the tests check the implementation against the contract instead of against
# itself.
CONTRACT_SPEAKABLE = frozenset({"daily", "weekly", "bi-weekly", "monthly", "one-time"})
SPOKEN_FREQUENCIES = tuple(sorted(CONTRACT_SPEAKABLE))

# Everything the form can offer that is NOT speakable, plus free-text values
# that no form offers but intake accepts.
UNSPEAKABLE_INPUTS = (
    "",
    "   ",
    "custom",
    "CUSTOM",
    "  Custom  ",
    "every other week",
    "twice a month",
    "as needed",
    "whenever you can fit us in",
    "2x/week",
    "<script>alert(1)</script>",
    "x" * 120,
)


def test_implementation_allowlist_equals_the_contract():
    """The speakable set is exactly the contracted one -- no more, no less.

    An extra member would be interpolated into customer copy without any test
    proving it reads correctly; a missing one would silently drop a real
    cadence from the email.
    """
    assert set(SPEAKABLE_FREQUENCIES) == CONTRACT_SPEAKABLE


# Members reviewed and confirmed to read correctly after the hardcoded "a".
# English article choice follows SOUND, not spelling: "a one-time cleaning" is
# correct despite the leading vowel letter, while "an hour" is correct despite
# the leading consonant. A spelling heuristic therefore cannot decide this, so
# this is an explicit reviewed list -- a new member (say "annual") fails here
# and forces someone to check whether the sentence needs "an".
ARTICLE_A_FREQUENCIES = frozenset(
    {"daily", "weekly", "bi-weekly", "monthly", "one-time"}
)


def test_every_speakable_frequency_is_reviewed_for_the_article_a():
    assert set(SPEAKABLE_FREQUENCIES) == ARTICLE_A_FREQUENCIES


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


@pytest.mark.parametrize("frequency", UNSPEAKABLE_INPUTS)
def test_unspeakable_frequency_falls_back_to_cadence_free_wording(frequency):
    """Anything outside the allowlist must not reach customer copy.

    `frequency` is free text server-side, so this fails CLOSED: a denylist
    would let `"every other week"` through and render "You requested a every
    other week commercial cleaning", which is broken English in an email a
    lead actually reads.
    """
    _, single = format_request_acknowledgement("Gretchen", "commercial", frequency)
    assert "You requested a commercial cleaning." in single
    assert frequency.strip() not in single or not frequency.strip()

    _, multi = format_request_acknowledgement(
        "Acme", "multi-location-commercial", frequency
    )
    assert "You requested cleaning for multiple locations." in multi


@pytest.mark.parametrize("frequency", FORM_FREQUENCIES + UNSPEAKABLE_INPUTS)
def test_commercial_copy_never_double_spaces_or_dangles_the_cadence(frequency):
    """Whatever arrives in the field, the sentence stays well-formed."""
    for service in ("commercial", "multi-location-commercial"):
        _, body = format_request_acknowledgement("Acme", service, frequency)
        line = next(
            line for line in body.splitlines() if line.startswith("You requested")
        )
        assert "  " not in line, line
        assert line.endswith("."), line
        assert " ." not in line, line
        assert " a  " not in line, line


@pytest.mark.parametrize("frequency", [None, 0, True, [], {}, 1.5, object()])
def test_non_string_frequency_never_raises(frequency):
    """Defence in depth, mirroring A1's totality guarantee for `service`."""
    for service in ("commercial", "multi-location-commercial"):
        _, body = format_request_acknowledgement("Acme", service, frequency)
        assert "You requested" in body


# --- copy guardrails, as executable checks --------------------------------

# Derived from the canonical selection surface, never hand-maintained. A
# parallel hand-written list would let a template added to the router escape
# the dollar / terminology / turnaround guards while this suite stayed green.
# `test_every_module_template_is_routed` closes the other direction: a
# `*_TEMPLATE` constant that exists but is not routed also fails.
ALL_TEMPLATES = tuple(dict.fromkeys(ACK_TEMPLATE_BY_VARIANT.values()))


def test_the_guard_inventory_is_derived_from_the_router_not_hand_written():
    """Every routed template is covered by the copy guards below."""
    assert set(ALL_TEMPLATES) == set(ACK_TEMPLATE_BY_VARIANT.values())
    assert ACK_TEMPLATE in ALL_TEMPLATES
    assert COMMERCIAL_SINGLE_SITE_TEMPLATE in ALL_TEMPLATES
    assert COMMERCIAL_MULTI_SITE_TEMPLATE in ALL_TEMPLATES


def test_every_module_template_is_routed():
    """A template constant that exists but is unrouted fails closed.

    Without this, adding `SOME_NEW_TEMPLATE` to the module and forgetting to
    route it would leave it unguarded and unnoticed: the guards iterate the
    router, so an unrouted body is invisible to them.
    """
    module_templates = {
        name: value
        for name, value in vars(request_acknowledgement).items()
        if name.endswith("_TEMPLATE") and isinstance(value, str)
    }
    assert module_templates, "no *_TEMPLATE constants found -- test is not looking"
    unrouted = {
        name
        for name, value in module_templates.items()
        if value not in set(ACK_TEMPLATE_BY_VARIANT.values())
    }
    assert unrouted == set(), f"template constants not routed by variant: {unrouted}"


def test_every_variant_is_routed_to_a_template():
    """The other side of the same closure: no variant may be unrouted."""
    from atlas_brain.templates.email.request_acknowledgement import (
        _ACK_VARIANT_BY_SERVICE,
    )

    assert set(ACK_TEMPLATE_BY_VARIANT) == set(_ACK_VARIANT_BY_SERVICE.values()) | {
        ACK_VARIANT_COMMERCIAL_SINGLE_SITE,
        ACK_VARIANT_COMMERCIAL_MULTI_SITE,
    }


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
    # A scope promised "for each one" commits EOM to serving every submitted
    # site before coverage has been established (Codex #2340 R1).
    assert "for each one" not in body.lower()
    assert "for each of" not in body.lower()


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
    ACK_VARIANT_RESIDENTIAL: [
        "Thanks for requesting a free estimate from Effingham Office Maids - it "
        "just landed with a real person, and we'll get back to you within 24 hours."
    ],
    ACK_VARIANT_GENERAL: [
        "Thanks for requesting a free estimate from Effingham Office Maids - it "
        "just landed with a real person, and we'll get back to you within 24 hours."
    ],
    ACK_VARIANT_COMMERCIAL_SINGLE_SITE: [
        "1. I will give you a call within 24 hours to ask a few questions about "
        "your facility and to set up a time and day for a walk-through."
    ],
    ACK_VARIANT_COMMERCIAL_MULTI_SITE: [
        "1. I will give you a call within 24 hours. That first call is to "
        "understand the whole picture before we put any numbers to it."
    ],
}

# One representative service per variant, so the whitelist can be exercised
# through the real rendering path rather than against the raw constants.
SERVICE_FOR_VARIANT = {
    ACK_VARIANT_RESIDENTIAL: "residential",
    ACK_VARIANT_GENERAL: "other",
    ACK_VARIANT_COMMERCIAL_SINGLE_SITE: "commercial",
    ACK_VARIANT_COMMERCIAL_MULTI_SITE: "multi-location-commercial",
}


def test_the_24_hour_whitelist_covers_every_routed_variant():
    """Closure: a new variant cannot slip past the turnaround guard.

    Keyed by variant rather than by service precisely so that adding a route
    to `ACK_TEMPLATE_BY_VARIANT` without approving its 24-hour wording fails
    here instead of shipping an unreviewed promise.
    """
    assert set(APPROVED_24_HOUR_SENTENCES) == set(ACK_TEMPLATE_BY_VARIANT)
    assert set(SERVICE_FOR_VARIANT) == set(ACK_TEMPLATE_BY_VARIANT)


@pytest.mark.parametrize("variant", sorted(APPROVED_24_HOUR_SENTENCES))
def test_only_approved_sentences_promise_24_hours(variant):
    _, body = format_request_acknowledgement(
        "Acme", SERVICE_FOR_VARIANT[variant], "weekly"
    )
    promised = [line for line in body.splitlines() if "24 hours" in line]
    assert promised == APPROVED_24_HOUR_SENTENCES[variant]


@pytest.mark.parametrize("variant", sorted(APPROVED_24_HOUR_SENTENCES))
def test_estimate_delivery_is_never_inside_the_24_hour_window(variant):
    """The estimate is promised after the walkthrough, never within 24 hours.

    Checked as an ordering property rather than a keyword: the sentence that
    delivers the estimate must be a different sentence from the one that
    promises 24 hours.
    """
    _, body = format_request_acknowledgement(
        "Acme", SERVICE_FOR_VARIANT[variant], "weekly"
    )
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
