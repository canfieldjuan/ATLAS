"""Acknowledgement-variant classification and recording (ATLAS #2320, slice A1).

This slice classifies and records only. Every submitted service still renders
the one existing acknowledgement template, so the residential email must be
byte-identical to what production sent before this change.
"""

import pytest

from atlas_brain.api.leads import _process_lead_intake
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

@pytest.mark.parametrize(
    ("service", "expected"),
    [
        # Every value the website forms can submit.
        ("residential", ACK_VARIANT_RESIDENTIAL),
        ("deep", ACK_VARIANT_RESIDENTIAL),
        ("move", ACK_VARIANT_RESIDENTIAL),
        ("commercial", ACK_VARIANT_COMMERCIAL_SINGLE_SITE),
        ("multi-location-commercial", ACK_VARIANT_COMMERCIAL_MULTI_SITE),
        ("other", ACK_VARIANT_GENERAL),
    ],
)
def test_every_submitted_service_maps_explicitly(service, expected):
    assert classify_ack_variant(service) == expected


@pytest.mark.parametrize(
    "service",
    ["", "   ", "unknown", "Residential Cleaning", "commercial-multi", "1; DROP TABLE"],
)
def test_unrecognised_service_falls_back_to_general(service):
    assert classify_ack_variant(service) == ACK_VARIANT_GENERAL


@pytest.mark.parametrize(
    ("service", "expected"),
    [
        ("  residential  ", ACK_VARIANT_RESIDENTIAL),
        ("RESIDENTIAL", ACK_VARIANT_RESIDENTIAL),
        ("Multi-Location-Commercial", ACK_VARIANT_COMMERCIAL_MULTI_SITE),
    ],
)
def test_classification_is_whitespace_and_case_insensitive(service, expected):
    assert classify_ack_variant(service) == expected


@pytest.mark.parametrize(
    "value",
    [
        # Falsy non-strings — these took the old `service or ""` path.
        None, 0, 0.0, False, [], {}, set(), (),
        # Truthy non-strings — these reached `.strip()` and raised
        # AttributeError before the isinstance guard. Covering only the falsy
        # half is what let the defect through.
        1, True, 1.5, ["residential"], {"service": "commercial"}, {"commercial"},
        ("commercial",), object(), b"commercial",
    ],
)
def test_classifier_is_total_for_the_whole_non_string_class(value):
    """Intake passes a validated str, but the classifier must never raise.

    Acceptance criterion 2 promises every non-string resolves to ``general``,
    so the guard covers the whole class rather than the falsy part of it.
    """
    assert classify_ack_variant(value) == ACK_VARIANT_GENERAL


# --- recording ------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "expected"),
    [
        ("residential", ACK_VARIANT_RESIDENTIAL),
        ("deep", ACK_VARIANT_RESIDENTIAL),
        ("move", ACK_VARIANT_RESIDENTIAL),
        ("commercial", ACK_VARIANT_COMMERCIAL_SINGLE_SITE),
        ("multi-location-commercial", ACK_VARIANT_COMMERCIAL_MULTI_SITE),
        ("other", ACK_VARIANT_GENERAL),
        ("", ACK_VARIANT_GENERAL),
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
    assert metadata["ack_variant"] == ACK_VARIANT_COMMERCIAL_MULTI_SITE


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
