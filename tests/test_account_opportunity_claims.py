from __future__ import annotations

from datetime import date

from atlas_brain.services.b2b.account_opportunity_claims import (
    account_opportunity_source_review_count,
    attach_account_opportunity_claim,
    build_account_opportunity_claim,
    serialize_product_claim,
)


def _opportunity_row(source_review_ids: list[str] | None = None) -> dict:
    return {
        "company": "Acme Corp",
        "vendor": "Zendesk",
        "source_review_ids": source_review_ids or ["review-1", "review-2", "review-3"],
        "alternatives": ["Intercom"],
        "buying_stage": "active_purchase",
        "quotes": [{"text": "We are actively evaluating Intercom."}],
    }


def test_account_opportunity_source_review_count_prefers_review_ids():
    assert account_opportunity_source_review_count(_opportunity_row(["a", "b"])) == 2


def test_account_opportunity_source_review_count_falls_back_to_quote_evidence():
    row = _opportunity_row([])
    row["source_review_ids"] = []
    row["quotes"] = [{"text": "We are actively evaluating alternatives."}]

    assert account_opportunity_source_review_count(row) == 1


def test_serialize_product_claim_uses_explicit_source_review_count_not_sample_size():
    claim = build_account_opportunity_claim(
        _opportunity_row(["review-1", "review-2", "review-3"]),
        as_of_date=date(2026, 4, 29),
        analysis_window_days=30,
    )

    serialized = serialize_product_claim(claim, source_review_count=1)

    assert serialized["sample_size"] == 3
    assert serialized["source_review_count"] == 1


def test_attach_account_opportunity_claim_serializes_explicit_source_review_count():
    payload = attach_account_opportunity_claim(
        _opportunity_row(["review-1", "review-2"]),
        as_of_date=date(2026, 4, 29),
        analysis_window_days=30,
    )

    assert payload["opportunity_claim"]["sample_size"] == 2
    assert payload["opportunity_claim"]["source_review_count"] == 2
