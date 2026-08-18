"""Contract tests for bounded one-run commercial billing candidate overrides."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from atlas_brain.services.commercial_billing_approvals import _invoice_draft
from atlas_brain.services.commercial_billing_candidate_overrides import (
    CommercialBillingCandidateOverrideValidationError,
    MAX_INVOICE_LINE_DESCRIPTION_LENGTH,
    build_effective_snapshot,
    decorate_effective_line_keys,
    decorate_line_keys,
    override_review_fingerprint,
)

_CANDIDATE_KEY = "commercial-billing:acme:2026-03"
_SOURCE_FINGERPRINT = "a" * 64


def _source_snapshot(*, total_cents: int | None = None) -> dict:
    """One blocked hourly candidate with immutable source evidence."""

    return {
        "billingPeriod": "2026-03",
        "blockers": [
            {
                "code": "invalid_rate",
                "eventIds": ["event-1"],
                "message": "The service rate must be a positive cent-precise amount.",
                "serviceId": "service-1",
            },
            {
                "code": "missing_billing_delivery_preference",
                "eventIds": [],
                "message": "No explicit delivery preference.",
                "serviceId": None,
            },
            {
                "code": "missing_billing_email",
                "eventIds": [],
                "message": "A billing email is required.",
                "serviceId": None,
            },
            {
                "code": "missing_hours",
                "eventIds": ["event-1"],
                "message": "Hours must be supplied before this service can be billed.",
                "serviceId": "service-1",
            },
            {
                "code": "zero_or_invalid_total",
                "eventIds": [],
                "message": "The candidate total is zero or invalid.",
                "serviceId": None,
            },
        ],
        "candidateKey": _CANDIDATE_KEY,
        "customer": {
            "contactId": "00000000-0000-0000-0000-000000000001",
            "customerType": "commercial",
            "displayName": "Acme Office",
        },
        "deliveryMethod": None,
        "lineItems": [
            {
                "amountCents": None,
                "description": "Hourly cleaning",
                "eventIds": ["event-1"],
                "locations": ["100 Main St"],
                "quantity": None,
                "quantityUnit": "hour",
                "rateCents": None,
                "serviceId": "service-1",
                "sourceDate": "2026-03-03",
            }
        ],
        "recipient": {
            "contactId": "00000000-0000-0000-0000-000000000001",
            "displayName": "Acme Accounts Payable",
            "email": None,
        },
        "sourceFingerprint": _SOURCE_FINGERPRINT,
        "subtotalCents": total_cents,
        "taxCents": None,
        "taxRateBasisPoints": 750,
        "totalCents": total_cents,
    }


def _complete_override(snapshot: dict) -> dict:
    line_key = decorate_line_keys(snapshot)["lineItems"][0]["lineKey"]
    return build_effective_snapshot(
        snapshot,
        line_overrides=[
            {
                "lineKey": line_key,
                "description": "After-hours cleaning",
                "quantityMinutes": 75,
                "rateCents": 4825,
            }
        ],
        adjustment={
            "kind": "charge",
            "description": "One-time access fee",
            "amountCents": 17,
        },
        recipient={"displayName": "Acme AP", "email": "Billing@Example.Test"},
        delivery_method="gmail_pdf",
    )


def _missing_billing_email_blocker() -> dict:
    return {
        "code": "missing_billing_email",
        "eventIds": [],
        "message": "The canonical commercial customer has no valid billing email.",
        "serviceId": None,
    }


def test_effective_override_is_exact_cent_scoped_and_leaves_source_immutable():
    source = _source_snapshot()
    retained_source = deepcopy(source)

    effective = _complete_override(source)

    assert source == retained_source
    assert effective["sourceFingerprint"] == _SOURCE_FINGERPRINT
    assert effective["deliveryMethod"] == "gmail_pdf"
    assert effective["recipient"] == {
        "contactId": "00000000-0000-0000-0000-000000000001",
        "displayName": "Acme AP",
        "email": "billing@example.test",
    }
    assert effective["blockers"] == []
    assert effective["lineItems"] == [
        {
            **source["lineItems"][0],
            "description": "After-hours cleaning",
            "quantity": "1.25",
            "quantityMinutes": 75,
            "rateCents": 4825,
            "amountCents": 6031,
        },
        {
            "amountCents": 17,
            "description": "One-time access fee",
            "kind": "adjustment",
            "quantity": 1,
            "quantityUnit": "adjustment",
            "rateCents": 17,
            "sourceDate": "2026-03-01",
        },
    ]
    assert (
        effective["subtotalCents"],
        effective["taxCents"],
        effective["totalCents"],
    ) == (
        6048,
        454,
        6502,
    )
    decorated = decorate_effective_line_keys(source, effective)
    assert (
        decorated["lineItems"][0]["lineKey"]
        == decorate_line_keys(source)["lineItems"][0]["lineKey"]
    )
    assert "lineKey" not in decorated["lineItems"][1]


@pytest.mark.parametrize(
    ("line_description", "adjustment_description"),
    (
        ("l" * 257, None),
        (None, "a" * 257),
        ("l" * 512, None),
        (None, "a" * 512),
        (f" {'l' * 257} ", None),
    ),
)
def test_override_description_admission_matches_invoice_line_limit(
    line_description, adjustment_description
):
    """An admitted override cannot contain a line approval will later reject."""

    source = _source_snapshot()
    line_key = decorate_line_keys(source)["lineItems"][0]["lineKey"]
    line_overrides = (
        [{"lineKey": line_key, "description": line_description}]
        if line_description is not None
        else []
    )
    adjustment = (
        {
            "kind": "charge",
            "description": adjustment_description,
            "amountCents": 17,
        }
        if adjustment_description is not None
        else None
    )

    with pytest.raises(
        CommercialBillingCandidateOverrideValidationError,
        match=f"1 to {MAX_INVOICE_LINE_DESCRIPTION_LENGTH} safe characters",
    ):
        build_effective_snapshot(
            source,
            line_overrides=line_overrides,
            adjustment=adjustment,
            recipient=None,
            delivery_method=None,
        )


def test_maximum_override_description_remains_approvable():
    """The shared limit admits the largest line approval can carry forward."""

    source = _source_snapshot()
    line_key = decorate_line_keys(source)["lineItems"][0]["lineKey"]
    description = "l" * MAX_INVOICE_LINE_DESCRIPTION_LENGTH
    effective = build_effective_snapshot(
        source,
        line_overrides=[
            {
                "lineKey": line_key,
                "description": description,
                "quantityMinutes": 60,
                "rateCents": 4825,
            }
        ],
        adjustment=None,
        recipient={"email": "billing@example.test"},
        delivery_method="gmail_pdf",
    )
    billing_run_id = UUID("00000000-0000-0000-0000-000000000004")
    review_fingerprint = override_review_fingerprint(
        billing_run_id, _SOURCE_FINGERPRINT, 1, effective
    )

    draft = _invoice_draft(
        effective,
        billing_run_id=billing_run_id,
        expected_candidate_key=_CANDIDATE_KEY,
        expected_source_fingerprint=_SOURCE_FINGERPRINT,
        expected_review_fingerprint=review_fingerprint,
        actor="Juan Canfield",
        due_days=14,
        issue_date=date(2026, 4, 2),
    )

    assert draft.line_items[0]["description"] == description


@pytest.mark.parametrize(
    "email",
    (
        "billing@example..com",
        "billing@.example.com",
        "billing@example.com.",
        ".billing@example.com",
        "billing..team@example.com",
        "billing@example",
        "billing@exam_ple.com",
        "billing@-example.com",
    ),
)
def test_gmail_recipient_override_uses_canonical_contact_email(email):
    """A recipient that Gmail will reject never clears the billing-email blocker."""

    source = _source_snapshot()
    line_key = decorate_line_keys(source)["lineItems"][0]["lineKey"]

    with pytest.raises(
        CommercialBillingCandidateOverrideValidationError,
        match="Billing recipient email is invalid",
    ):
        build_effective_snapshot(
            source,
            line_overrides=[
                {
                    "lineKey": line_key,
                    "quantityMinutes": 60,
                    "rateCents": 4825,
                }
            ],
            adjustment=None,
            recipient={"email": email},
            delivery_method="gmail_pdf",
        )


@pytest.mark.parametrize(
    ("source_delivery_method", "source_recipient", "source_blockers"),
    (
        ("manual_square", None, []),
        (
            "manual_square",
            {
                "contactId": "00000000-0000-0000-0000-000000000001",
                "displayName": "Acme AP",
                "email": None,
            },
            [],
        ),
        ("manual_square", {}, []),
        (
            "no_invoice_residential_receipt",
            None,
            [
                {
                    "code": "no_invoice_delivery_preference",
                    "eventIds": [],
                    "message": "The recorded delivery preference does not permit a commercial invoice.",
                    "serviceId": None,
                }
            ],
        ),
        (
            None,
            None,
            [
                {
                    "code": "missing_billing_delivery_preference",
                    "eventIds": [],
                    "message": "No explicit billing delivery preference is recorded.",
                    "serviceId": None,
                }
            ],
        ),
        ("manual_square", {"email": "not-an-email"}, []),
    ),
)
def test_gmail_delivery_override_readds_missing_recipient_blocker(
    source_delivery_method, source_recipient, source_blockers
):
    """Delivery changes cannot make a no-recipient candidate approval-eligible."""

    source = _complete_override(_source_snapshot())
    source["deliveryMethod"] = source_delivery_method
    source["recipient"] = deepcopy(source_recipient)
    source["blockers"] = deepcopy(source_blockers)

    effective = build_effective_snapshot(
        source,
        line_overrides=[],
        adjustment=None,
        recipient=None,
        delivery_method="gmail_pdf",
    )

    assert effective["blockers"][0] == _missing_billing_email_blocker()
    assert [blocker["code"] for blocker in effective["blockers"]] == [
        "missing_billing_email",
        "zero_or_invalid_total",
    ]


def test_omitted_invalid_rate_service_stays_blocked_after_an_adjustment_override():
    """A valid retained line cannot silently erase an omitted unpriced service."""

    source = _source_snapshot()
    source["deliveryMethod"] = "manual_square"
    source["recipient"] = None
    source["lineItems"][0].update(
        {
            "amountCents": 4825,
            "quantity": "1",
            "quantityMinutes": 60,
            "rateCents": 4825,
        }
    )
    source["blockers"] = [
        {
            "code": "invalid_rate",
            "eventIds": ["event-unpriced"],
            "message": "The service rate must be a positive cent-precise amount.",
            "serviceId": "service-omitted-by-producer",
        },
        {
            "code": "zero_or_invalid_total",
            "eventIds": [],
            "message": "The candidate total is zero or invalid.",
            "serviceId": None,
        },
    ]
    source["subtotalCents"] = None
    source["taxCents"] = None
    source["totalCents"] = None

    effective = build_effective_snapshot(
        source,
        line_overrides=[],
        adjustment={
            "kind": "charge",
            "description": "Documented after-hours access fee",
            "amountCents": 17,
        },
        recipient=None,
        delivery_method=None,
    )

    assert effective["subtotalCents"] is None
    assert effective["taxCents"] is None
    assert effective["totalCents"] is None
    assert [blocker["code"] for blocker in effective["blockers"]] == [
        "invalid_rate",
        "zero_or_invalid_total",
    ]


def test_effective_override_requires_a_real_permitted_change_and_rejects_source_edits():
    source = _source_snapshot()
    line_key = decorate_line_keys(source)["lineItems"][0]["lineKey"]

    with pytest.raises(
        CommercialBillingCandidateOverrideValidationError,
        match="must change permitted review evidence",
    ):
        build_effective_snapshot(
            source,
            line_overrides=[],
            adjustment=None,
            recipient=None,
            delivery_method=None,
        )

    with pytest.raises(
        CommercialBillingCandidateOverrideValidationError,
        match="unsupported fields",
    ):
        build_effective_snapshot(
            source,
            line_overrides=[
                {
                    "lineKey": line_key,
                    "sourceDate": "2026-03-04",
                }
            ],
            adjustment=None,
            recipient=None,
            delivery_method=None,
        )

    with pytest.raises(
        CommercialBillingCandidateOverrideValidationError,
        match="whole quantityMinutes",
    ):
        build_effective_snapshot(
            source,
            line_overrides=[{"lineKey": line_key, "quantity": 2}],
            adjustment=None,
            recipient=None,
            delivery_method=None,
        )


def test_credit_that_leaves_no_positive_total_stays_blocked_and_unapprovable():
    source = _source_snapshot()
    effective = _complete_override(source)
    hourly_key = decorate_line_keys(source)["lineItems"][0]["lineKey"]

    credited = build_effective_snapshot(
        source,
        line_overrides=[
            {
                "lineKey": hourly_key,
                "quantityMinutes": 60,
                "rateCents": 100,
            }
        ],
        adjustment={
            "kind": "credit",
            "description": "Customer credit",
            "amountCents": 100,
        },
        recipient={"email": "billing@example.test"},
        delivery_method="gmail_pdf",
    )

    assert effective["totalCents"] == 6502
    assert credited["subtotalCents"] is None
    assert credited["taxCents"] is None
    assert credited["totalCents"] is None
    assert [blocker["code"] for blocker in credited["blockers"]] == [
        "zero_or_invalid_total"
    ]


def test_each_override_revision_has_a_new_review_identity_and_invoice_uses_effective_money():
    effective = _complete_override(_source_snapshot())
    first_run = UUID("00000000-0000-0000-0000-000000000002")
    second_run = UUID("00000000-0000-0000-0000-000000000003")
    revision_one = override_review_fingerprint(
        first_run, _SOURCE_FINGERPRINT, 1, effective
    )
    revision_two = override_review_fingerprint(
        first_run, _SOURCE_FINGERPRINT, 2, effective
    )
    same_revision_other_run = override_review_fingerprint(
        second_run, _SOURCE_FINGERPRINT, 1, effective
    )

    assert revision_one != revision_two
    assert revision_one != same_revision_other_run
    draft = _invoice_draft(
        effective,
        billing_run_id=first_run,
        expected_candidate_key=_CANDIDATE_KEY,
        expected_source_fingerprint=_SOURCE_FINGERPRINT,
        expected_review_fingerprint=revision_one,
        actor="Juan Canfield",
        due_days=14,
        issue_date=date(2026, 4, 2),
    )

    assert draft.line_items == [
        {
            "amount": "60.31",
            "date": "2026-03-03",
            "description": "After-hours cleaning",
            "quantity": "1.25",
            "unit_price": "48.25",
        },
        {
            "amount": "0.17",
            "date": "2026-03-01",
            "description": "One-time access fee",
            "quantity": 1,
            "unit_price": "0.17",
        },
    ]
    assert (draft.subtotal, draft.tax_amount, draft.total) == (
        Decimal("60.48"),
        Decimal("4.54"),
        Decimal("65.02"),
    )
    assert draft.metadata["reviewFingerprint"] == revision_one
    assert draft.metadata["sourceFingerprint"] == _SOURCE_FINGERPRINT
    assert draft.metadata["commercialBillingExactLineAmounts"] is True


def test_candidate_override_migration_is_atomic_append_only_and_delivery_free():
    migration = (
        Path(__file__).parents[1]
        / "atlas_brain/storage/migrations/382_commercial_billing_candidate_overrides.sql"
    ).read_text(encoding="utf-8")

    assert migration.startswith("-- atlas: atomic-bookkeeping")
    assert (
        "CREATE TABLE IF NOT EXISTS commercial_billing_candidate_overrides" in migration
    )
    assert (
        "FOREIGN KEY (billing_run_id, candidate_key, source_fingerprint)" in migration
    )
    assert (
        "UNIQUE (billing_run_id, candidate_key, source_fingerprint, revision)"
        in migration
    )
    assert "trg_prevent_commercial_billing_candidate_override_mutation" in migration
    assert "review_fingerprint" in migration
    assert "requires an explicit include decision" in migration
    executable = "\n".join(
        line for line in migration.splitlines() if not line.lstrip().startswith("--")
    )
    assert "DROP TABLE" not in executable
    assert "INSERT INTO invoices" not in executable
    assert "UPDATE invoices" not in executable
    assert "gmail" not in executable.lower()
    assert "email" not in executable.lower()
