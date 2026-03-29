"""Tests for canonical account intelligence wiring into battle cards."""

import importlib
import pytest

briefing_shared = importlib.import_module(
    "atlas_brain.autonomous.tasks._b2b_shared"
)
battle_cards = importlib.import_module(
    "atlas_brain.autonomous.tasks.b2b_battle_cards"
)


# ---------------------------------------------------------------------------
# _normalize_canonical_accounts_for_battle_card
# ---------------------------------------------------------------------------


def test_normalize_canonical_accounts_basic():
    accounts = [
        {
            "company_name": "Acme Corp",
            "urgency_score": 8.5,
            "buyer_role": "VP Engineering",
            "pain_category": "reliability",
            "buying_stage": "evaluation",
            "company_size": "500-1000",
            "decision_maker": True,
            "confidence_score": 0.8,
            "contract_end": "2026-06-01",
            "source": "g2",
        },
        {
            "company_name": "Beta LLC",
            "urgency_score": 6.0,
            "buyer_role": "unknown",
            "pain_category": "cost",
            "buying_stage": "consideration",
            "company_size": "50-200",
            "decision_maker": False,
            "confidence_score": 0.5,
            "source": "capterra",
        },
    ]
    result = briefing_shared._normalize_canonical_accounts_for_battle_card(
        accounts,
        current_vendor="OtherVendor",
        limit=5,
    )
    assert len(result) == 2
    # Acme should rank first (higher urgency + DM + stage)
    assert result[0]["company"] == "Acme Corp"
    assert result[0]["urgency"] == 8.5
    assert result[0]["role"] == "VP Engineering"
    assert result[0]["decision_maker"] is True
    assert result[0]["contract_end"] == "2026-06-01"
    # Beta: role "unknown" -> None
    assert result[1]["company"] == "Beta LLC"
    assert result[1]["role"] is None


def test_normalize_canonical_accounts_filters_vendor_self():
    accounts = [
        {
            "company_name": "Zendesk",
            "urgency_score": 9.0,
            "buyer_role": "Admin",
            "buying_stage": "evaluation",
            "company_size": "1000+",
        },
    ]
    result = briefing_shared._normalize_canonical_accounts_for_battle_card(
        accounts,
        current_vendor="Zendesk",
        limit=5,
    )
    # Self-vendor should be blocked
    assert len(result) == 0


def test_normalize_canonical_accounts_respects_limit():
    accounts = [
        {
            "company_name": f"Company {i}",
            "urgency_score": float(i),
            "buyer_role": "Manager",
            "buying_stage": "evaluation",
            "company_size": "100-500",
        }
        for i in range(10)
    ]
    result = briefing_shared._normalize_canonical_accounts_for_battle_card(
        accounts,
        current_vendor="OtherVendor",
        limit=3,
    )
    assert len(result) == 3


def test_normalize_canonical_accounts_skips_empty_company():
    accounts = [
        {"company_name": "", "urgency_score": 9.0, "buyer_role": "VP"},
        {"urgency_score": 8.0, "buyer_role": "Director", "buying_stage": "evaluation"},
    ]
    result = briefing_shared._normalize_canonical_accounts_for_battle_card(
        accounts,
        current_vendor="Vendor",
        limit=5,
    )
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Quality status: thin_evidence vs needs_review
# ---------------------------------------------------------------------------


def test_quality_status_thin_evidence_when_no_canonical_accounts():
    status = battle_cards._battle_card_quality_status(
        phase="deterministic",
        hard_blockers=[],
        warnings=["no high-intent account data available for required stages (evaluation)"],
        has_canonical_accounts=False,
    )
    assert status == "thin_evidence"


def test_quality_status_needs_review_when_canonical_accounts_present():
    status = battle_cards._battle_card_quality_status(
        phase="deterministic",
        hard_blockers=[],
        warnings=["no high-intent account data available for required stages (evaluation)"],
        has_canonical_accounts=True,
    )
    assert status == "needs_review"


def test_quality_status_sales_ready_no_warnings():
    status = battle_cards._battle_card_quality_status(
        phase="final",
        hard_blockers=[],
        warnings=[],
        has_canonical_accounts=True,
    )
    assert status == "sales_ready"


def test_quality_status_fallback_with_blockers():
    status = battle_cards._battle_card_quality_status(
        phase="final",
        hard_blockers=["data stale"],
        warnings=[],
        has_canonical_accounts=True,
    )
    assert status == "deterministic_fallback"


def test_quality_status_needs_review_for_non_account_warnings():
    status = battle_cards._battle_card_quality_status(
        phase="deterministic",
        hard_blockers=[],
        warnings=["evidence window is thin; confidence may improve with more data"],
        has_canonical_accounts=False,
    )
    # Non-account warning should still be needs_review, not thin_evidence
    assert status == "needs_review"
