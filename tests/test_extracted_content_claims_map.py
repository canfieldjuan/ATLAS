"""Unit tests for slice 3: the deterministic claims map.

Pure value types + matching logic; no DB, no async, no Atlas imports, no LLM.
"""

from __future__ import annotations

from datetime import date

import pytest

from extracted_content_pipeline.claims_map import (
    ClaimStatus,
    ExtractedClaim,
    RegistryClaim,
    blocking_claims,
    build_claims_map,
    is_clear,
    map_claim,
)
from extracted_content_pipeline.review_contract import RiskTier

_AS_OF = date(2026, 6, 6)

_REGISTRY = {
    "pricing.discount.q4": RegistryClaim(
        id="pricing.discount.q4",
        approved_wording="Save up to 30% on eligible annual plans",
        risk_tier=RiskTier.HIGH,
        expiration=date(2026, 1, 15),  # lapsed before _AS_OF
    ),
    "feature.sso": RegistryClaim(
        id="feature.sso",
        approved_wording="SSO is included on every plan",
        risk_tier=RiskTier.MEDIUM,
        expiration=None,  # never lapses
    ),
}


def test_claim_status_values() -> None:
    assert {s.value for s in ClaimStatus} == {
        "match",
        "mismatch",
        "unregistered",
        "expired",
    }


def test_match_when_wording_equals_approved_modulo_normalization() -> None:
    claim = ExtractedClaim(
        text="  SSO is   included on EVERY plan ",  # case + spacing differ
        location="hero",
        registry_id="feature.sso",
    )
    mapped = map_claim(claim, _REGISTRY, as_of=_AS_OF)
    assert mapped.status is ClaimStatus.MATCH
    assert mapped.approved_wording == "SSO is included on every plan"
    assert mapped.risk_tier is RiskTier.MEDIUM


def test_mismatch_when_wording_differs() -> None:
    # The doc's liability example: "all plans" vs "eligible annual plans".
    claim = ExtractedClaim(
        text="Save 30% on all plans",
        location="subject",
        registry_id="pricing.discount.q4",
    )
    # Use a date before expiration so the status is MISMATCH, not EXPIRED.
    mapped = map_claim(claim, _REGISTRY, as_of=date(2026, 1, 1))
    assert mapped.status is ClaimStatus.MISMATCH
    assert mapped.registry_id == "pricing.discount.q4"
    assert mapped.risk_tier is RiskTier.HIGH


def test_unregistered_when_no_or_unknown_registry_id() -> None:
    no_id = map_claim(ExtractedClaim(text="we are the best"), _REGISTRY, as_of=_AS_OF)
    assert no_id.status is ClaimStatus.UNREGISTERED
    assert no_id.registry_id is None
    assert no_id.approved_wording is None

    unknown = map_claim(
        ExtractedClaim(text="x", registry_id="does.not.exist"),
        _REGISTRY,
        as_of=_AS_OF,
    )
    assert unknown.status is ClaimStatus.UNREGISTERED


def test_expired_takes_precedence_even_if_wording_matches() -> None:
    claim = ExtractedClaim(
        text="Save up to 30% on eligible annual plans",  # exact approved wording
        location="hero",
        registry_id="pricing.discount.q4",
    )
    mapped = map_claim(claim, _REGISTRY, as_of=_AS_OF)  # _AS_OF is past expiration
    assert mapped.status is ClaimStatus.EXPIRED


def test_expiration_day_itself_is_still_valid() -> None:
    claim = ExtractedClaim(
        text="Save up to 30% on eligible annual plans",
        registry_id="pricing.discount.q4",
    )
    on_day = map_claim(claim, _REGISTRY, as_of=date(2026, 1, 15))
    assert on_day.status is ClaimStatus.MATCH


def test_build_claims_map_preserves_order() -> None:
    claims = [
        ExtractedClaim(text="SSO is included on every plan", registry_id="feature.sso"),
        ExtractedClaim(text="unbeatable prices", registry_id=None),
    ]
    mapped = build_claims_map(claims, _REGISTRY, as_of=_AS_OF)
    assert [m.status for m in mapped] == [
        ClaimStatus.MATCH,
        ClaimStatus.UNREGISTERED,
    ]


def test_blocking_claims_flags_mismatch_and_expired_only() -> None:
    claims = [
        ExtractedClaim(text="SSO is included on every plan", registry_id="feature.sso"),
        ExtractedClaim(text="Save 30% on all plans", registry_id="pricing.discount.q4"),
        ExtractedClaim(text="brand new", registry_id=None),
    ]
    mapped = build_claims_map(claims, _REGISTRY, as_of=_AS_OF)
    blocking = blocking_claims(mapped)
    statuses = {m.status for m in blocking}
    # pricing claim is EXPIRED at _AS_OF; SSO matches; unregistered does not block.
    assert statuses == {ClaimStatus.EXPIRED}
    assert is_clear(mapped) is False


def test_is_clear_true_when_only_match_and_unregistered() -> None:
    claims = [
        ExtractedClaim(text="SSO is included on every plan", registry_id="feature.sso"),
        ExtractedClaim(text="brand new", registry_id=None),
    ]
    mapped = build_claims_map(claims, _REGISTRY, as_of=_AS_OF)
    assert is_clear(mapped) is True


def test_none_text_normalizes_to_unmatched_not_error() -> None:
    # None text (e.g. JSON null) must not raise; with a real registry id it
    # simply won't match the approved wording.
    claim = ExtractedClaim(text=None, registry_id="feature.sso")  # type: ignore[arg-type]
    mapped = map_claim(claim, _REGISTRY, as_of=_AS_OF)
    assert mapped.status is ClaimStatus.MISMATCH


def test_registry_claim_is_expired_helper() -> None:
    entry = _REGISTRY["pricing.discount.q4"]
    assert entry.is_expired(date(2026, 6, 6)) is True
    assert entry.is_expired(date(2026, 1, 15)) is False  # day-of still valid
    assert _REGISTRY["feature.sso"].is_expired(date(2999, 1, 1)) is False


def test_mapped_claim_is_frozen() -> None:
    mapped = map_claim(
        ExtractedClaim(text="x", registry_id="feature.sso"),
        _REGISTRY,
        as_of=_AS_OF,
    )
    with pytest.raises(Exception):
        mapped.status = ClaimStatus.MATCH  # type: ignore[misc]
