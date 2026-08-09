"""Acknowledgement-variant classification and recording (ATLAS #2320, slice A1).

This slice classifies and records only. Every submitted service still renders
the one existing acknowledgement template, so the residential email must be
byte-identical to what production sent before this change.
"""

import random
import string

import pytest

from atlas_brain.api.leads import _process_lead_intake
from atlas_brain.templates.email.request_acknowledgement import (
    _ACK_VARIANT_BY_SERVICE,
)
from atlas_brain.templates.email import (
    ACK_VARIANT_COMMERCIAL_MULTI_SITE,
    ACK_VARIANT_COMMERCIAL_SINGLE_SITE,
    ACK_VARIANT_GENERAL,
    ACK_VARIANT_RESIDENTIAL,
    classify_ack_variant,
    format_request_acknowledgement,
)

from .test_leads_intake import _crm, _email_history, _email_provider, _payload


@pytest.fixture(autouse=True)
def _email_enabled(monkeypatch):
    """Mirror of the autouse fixture in ``test_leads_intake``: the endpoint
    honors ``settings.email.enabled``, and an autouse fixture does not cross
    module boundaries, so the send path needs it enabled here too."""
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.email, "enabled", True)


# --- classifier -----------------------------------------------------------

# Independent contract oracle. Both halves of the contract are written here as
# string LITERALS, never as references to the implementation's constants:
#
#   * the six inputs the website forms can submit, transcribed from the form
#     markup (Effingham_Office_Maids_Website: contact.html,
#     commercial-estimate.html, house-cleaning-estimate.html,
#     house-cleaning-services/index.html), and
#   * the four variant strings intake is contracted to persist.
#
# Naming the outputs by literal matters as much as naming the inputs. If the
# oracle said `ACK_VARIANT_RESIDENTIAL` instead of `"residential"`, then editing
# that constant would move the classifier and the oracle together — the suite
# would stay green while intake persisted a value that violates acceptance
# criterion 1. Written as literals, the oracle detects output-contract drift
# (a renamed variant value) as well as added mapping keys.
CONTRACT_RESIDENTIAL = "residential"
CONTRACT_COMMERCIAL_SINGLE_SITE = "commercial_single_site"
CONTRACT_COMMERCIAL_MULTI_SITE = "commercial_multi_site"
CONTRACT_GENERAL = "general"

CONTRACT_SERVICE_VARIANTS = {
    "residential": CONTRACT_RESIDENTIAL,
    "deep": CONTRACT_RESIDENTIAL,
    "move": CONTRACT_RESIDENTIAL,
    "commercial": CONTRACT_COMMERCIAL_SINGLE_SITE,
    "multi-location-commercial": CONTRACT_COMMERCIAL_MULTI_SITE,
    "other": CONTRACT_GENERAL,
}


def test_exported_variant_constants_equal_the_contracted_literals():
    """The exported constants carry the contracted string VALUES.

    These constants are what downstream code (and A2/A3 template selection)
    will branch on, and their values are what intake persists into
    ``contact_interactions.metadata`` / ``sent_emails.metadata``. Pinning them
    to literals here is what makes every other assertion in this file a real
    check rather than a comparison of the implementation against itself.
    """
    assert ACK_VARIANT_RESIDENTIAL == CONTRACT_RESIDENTIAL
    assert ACK_VARIANT_COMMERCIAL_SINGLE_SITE == CONTRACT_COMMERCIAL_SINGLE_SITE
    assert ACK_VARIANT_COMMERCIAL_MULTI_SITE == CONTRACT_COMMERCIAL_MULTI_SITE
    assert ACK_VARIANT_GENERAL == CONTRACT_GENERAL


def test_implementation_mapping_equals_the_contract_oracle():
    """The implementation carries exactly the contracted members — no more.

    Without this, an extra entry would both re-point a real submission and be
    excluded from the generated unrecognised inputs, so nothing would fail.
    """
    assert _ACK_VARIANT_BY_SERVICE == CONTRACT_SERVICE_VARIANTS


@pytest.mark.parametrize(
    ("service", "expected"),
    # Driven by the contract oracle above, so the explicit cases and the
    # generated ones can never disagree about what "recognised" means.
    sorted(CONTRACT_SERVICE_VARIANTS.items()),
)
def test_every_submitted_service_maps_explicitly(service, expected):
    assert classify_ack_variant(service) == expected


@pytest.mark.parametrize(
    "service",
    ["", "   ", "unknown", "Residential Cleaning", "commercial-multi", "1; DROP TABLE"],
)
def test_unrecognised_service_falls_back_to_general(service):
    assert classify_ack_variant(service) == CONTRACT_GENERAL


@pytest.mark.parametrize(
    ("service", "expected"),
    [
        ("  residential  ", CONTRACT_RESIDENTIAL),
        ("RESIDENTIAL", CONTRACT_RESIDENTIAL),
        ("Multi-Location-Commercial", CONTRACT_COMMERCIAL_MULTI_SITE),
    ],
)
def test_classification_is_whitespace_and_case_insensitive(service, expected):
    assert classify_ack_variant(service) == expected


def _generated_unrecognised_strings(rng, count):
    """Yield arbitrary strings that are not one of the six contracted values.

    Grammar-derived rather than a fixed sample: the declared set is OPEN, so a
    handful of literals cannot stand in for "any unrecognised string". The
    exclusion consults the contract oracle, never the implementation.
    """
    alphabet = (
        string.ascii_letters + string.digits + " \t\n-_./\\'\"<>{}[]();:@#$%&*+=|~`^"
        + "áéíóúñü汉字🧹​ "
    )
    known = set(CONTRACT_SERVICE_VARIANTS)  # oracle, not the implementation
    produced = 0
    while produced < count:
        candidate = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        if candidate.strip().lower() in known:
            continue  # a generated collision with a real value is not a counter-example
        produced += 1
        yield candidate


def _generated_non_strings(rng, count):
    """Yield varied non-string shapes, including nested containers."""
    atoms = [
        None, True, False, 0, 1, -3, 0.0, 1.5, float("nan"), float("inf"),
        b"commercial", bytearray(b"residential"), object(), Exception("x"),
        range(2), iter([]), lambda: None, type, NotImplemented, Ellipsis,
    ]
    for _ in range(count):
        shape = rng.randint(0, 5)
        atom = rng.choice(atoms)
        if shape == 0:
            yield atom
        elif shape == 1:
            yield [atom]
        elif shape == 2:
            yield (atom,)
        elif shape == 3:
            yield {"service": atom}
        elif shape == 4:
            yield [[atom], {"nested": atom}]
        else:
            yield {rng.randint(0, 9): [atom]}


def test_generated_unrecognised_strings_all_resolve_to_general():
    """Property: any string outside the known set resolves to ``general``.

    The set is declared OPEN (a website form option can appear without any
    change here), so the guarantee has to hold for arbitrary strings, not for a
    curated list of them.
    """
    rng = random.Random(20260808)  # seeded: reproducible, not flaky
    checked = 0
    for value in _generated_unrecognised_strings(rng, 500):
        assert classify_ack_variant(value) == CONTRACT_GENERAL, repr(value)
        checked += 1
    assert checked == 500


def test_generated_non_strings_never_raise_and_resolve_to_general():
    """Property: the whole non-string class resolves to ``general``.

    Acceptance criterion 2 promises this for every non-string, so the proof is
    generated over varied shapes — atoms, containers, nested containers — rather
    than the fixed examples that previously let a truthy-non-string
    ``AttributeError`` through.
    """
    rng = random.Random(20260808)
    checked = 0
    for value in _generated_non_strings(rng, 500):
        assert classify_ack_variant(value) == CONTRACT_GENERAL, repr(value)
        checked += 1
    assert checked == 500


@pytest.mark.parametrize(
    "value",
    [
        # Retained as named regression anchors for the specific defect Codex
        # found: the falsy half took the old `service or ""` path, while the
        # truthy half reached `.strip()` and raised AttributeError.
        None, 0, False, [], {},
        1, True, 1.5, ["residential"], {"service": "commercial"}, b"commercial",
    ],
)
def test_classifier_is_total_for_the_whole_non_string_class(value):
    assert classify_ack_variant(value) == CONTRACT_GENERAL


# --- recording ------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "expected"),
    [
        ("residential", CONTRACT_RESIDENTIAL),
        ("deep", CONTRACT_RESIDENTIAL),
        ("move", CONTRACT_RESIDENTIAL),
        ("commercial", CONTRACT_COMMERCIAL_SINGLE_SITE),
        ("multi-location-commercial", CONTRACT_COMMERCIAL_MULTI_SITE),
        ("other", CONTRACT_GENERAL),
        ("", CONTRACT_GENERAL),
    ],
)
async def test_variant_recorded_on_interaction_and_email_history(service, expected):
    crm, provider, history = _crm(), _email_provider(), _email_history()

    await _process_lead_intake(
        _payload(service=service),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    interaction_metadata = crm.log_interaction.await_args.kwargs["metadata"]
    assert interaction_metadata["ack_variant"] == expected
    # The raw submitted value stays alongside the derived variant.
    assert interaction_metadata["service"] == service

    history_metadata = history.create.await_args.kwargs["metadata"]
    assert history_metadata["ack_variant"] == expected
    # The raw submitted value rides along in BOTH evidence records, so a
    # contact with several requests can be traced to the submission that
    # produced a given email.
    assert history_metadata["service"] == service


@pytest.mark.asyncio
async def test_variant_recorded_on_interaction_even_when_no_email_is_sent():
    """A phone-only lead gets no acknowledgement; the segment is still evidence."""
    crm, provider, history = _crm(), _email_provider(), _email_history()

    await _process_lead_intake(
        _payload(email="", service="multi-location-commercial"),
        crm=crm,
        email_provider=provider,
        email_history=history,
    )

    provider.send.assert_not_awaited()
    metadata = crm.log_interaction.await_args.kwargs["metadata"]
    assert metadata["ack_variant"] == CONTRACT_COMMERCIAL_MULTI_SITE


# --- no copy change in this slice ----------------------------------------

@pytest.mark.parametrize(
    "service",
    ["residential", "deep", "move", "commercial", "multi-location-commercial", "other", ""],
)
def test_all_variants_still_render_the_single_existing_template(service):
    """A1 classifies and records only. Template selection arrives in A2/A3."""
    baseline_subject, baseline_body = format_request_acknowledgement(
        "Jane", "residential", "monthly"
    )
    subject, body = format_request_acknowledgement("Jane", service, "monthly")

    assert subject == baseline_subject
    # Only the echoed "Your request:" line may differ between services.
    def _without_request_line(text: str) -> list[str]:
        return [line for line in text.splitlines() if not line.startswith("Your request:")]

    assert _without_request_line(body) == _without_request_line(baseline_body)


def test_residential_render_is_byte_identical_to_pre_change_production():
    """Golden anchor: the residential email must not drift while variants land.

    Captured from origin/main before this slice; unchanged across the rebase to 40bb24553.
    """
    subject, body = format_request_acknowledgement("Marie", "residential", "monthly")

    assert subject == "We received your estimate request - Effingham Office Maids"
    assert body == (
        "Hi Marie,\n"
        "\n"
        "Thanks for requesting a free estimate from Effingham Office Maids - it "
        "just landed with a real person, and we'll get back to you within 24 hours.\n"
        "\n"
        "Your request: residential, monthly.\n"
        "\n"
        "Here's what happens next:\n"
        "\n"
        "1. We'll give you a call to book a day and time that works for you for a "
        "quick estimate walkthrough. The walkthrough usually takes less than 20 "
        "minutes.\n"
        "\n"
        "2. We'll send our estimate team out to look over the spaces you want "
        "cleaned. Your two team members will be Mayra Canfield and Tina Gomez.\n"
        "\n"
        "3. Show them around and tell them what you'd like cleaned and how often.\n"
        "\n"
        "4. Before Mayra and Tina leave, they'll give you the cost to clean your "
        "space. If the pricing works for you, you can schedule your first cleaning "
        "right then.\n"
        "\n"
        "5. Estimates are FREE and there's no obligation. If the pricing isn't "
        "right for you, we won't try to talk you into it.\n"
        "\n"
        "Questions in the meantime? Call us at (217) 207-3097, or just reply to\n"
        "this email.\n"
        "\n"
        "Talk soon,\n"
        "The Effingham Office Maids Team\n"
        "(217) 207-3097 | info@effinghamofficemaids.com\n"
        "effinghamofficemaids.com\n"
    )
