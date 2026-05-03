"""Tests for extracted_quality_gate.campaign_pack.evaluate_campaign.

The function under test is pure: no DB, no clock, no network. Pure
unit tests, no fixtures.
"""

from __future__ import annotations

import pytest

from extracted_quality_gate.campaign_pack import evaluate_campaign
from extracted_quality_gate.types import (
    GateDecision,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)


def _input(
    *,
    subject: str = "Test subject",
    body: str = "Test body",
    cta: str = "Reply YES",
    campaign: dict | None = None,
    required_proof_terms: tuple = (),
    anchor_examples: dict | None = None,
    witness_highlights: tuple = (),
    specificity_blocking_issues: tuple = (),
    specificity_warnings: tuple = (),
) -> QualityInput:
    return QualityInput(
        artifact_type="campaign_email",
        artifact_id="test-campaign",
        content=body,
        context={
            "subject": subject,
            "body": body,
            "cta": cta,
            "campaign": campaign or {},
            "required_proof_terms": required_proof_terms,
            "anchor_examples": anchor_examples or {},
            "witness_highlights": witness_highlights,
            "specificity_blocking_issues": specificity_blocking_issues,
            "specificity_warnings": specificity_warnings,
        },
    )


# ---- Decision shape ----


def test_returns_quality_report():
    report = evaluate_campaign(_input())
    assert isinstance(report, QualityReport)


def test_clean_input_passes():
    report = evaluate_campaign(_input())
    assert report.passed is True
    assert report.decision == GateDecision.PASS
    assert report.findings == ()


def test_metadata_mirrors_legacy_dict_shape():
    report = evaluate_campaign(_input())
    md = report.metadata
    assert md["status"] == "pass"
    assert md["blocking_issues"] == ()
    assert md["warnings"] == ()
    assert "campaign_proof_terms" in md
    assert "required_proof_terms" in md
    assert "used_proof_terms" in md
    assert "unused_proof_terms" in md
    assert md["primary_blocker"] is None


# ---- Specificity pass-through ----


def test_specificity_blockers_pass_through_as_blockers():
    report = evaluate_campaign(
        _input(specificity_blocking_issues=("missing_anchor_support",))
    )
    assert report.decision == GateDecision.BLOCK
    assert "missing_anchor_support" in report.metadata["blocking_issues"]


def test_specificity_warnings_pass_through_as_warnings():
    report = evaluate_campaign(
        _input(specificity_warnings=("anchor_count_below_target",))
    )
    assert report.decision == GateDecision.WARN
    assert "anchor_count_below_target" in report.metadata["warnings"]


# ---- Proof-term coverage ----


def test_missing_required_proof_term_blocks_when_anchors_present():
    body = "We saw improvements after switching."
    report = evaluate_campaign(
        _input(
            body=body,
            required_proof_terms=("dashboard refresh time",),
            anchor_examples={"primary": [{"phrase": "anchor"}]},
        )
    )
    assert report.decision == GateDecision.BLOCK
    assert "missing_exact_proof_term" in report.metadata["blocking_issues"]


def test_proof_term_present_does_not_block():
    body = "Customers report dashboard refresh time dropped 40%."
    report = evaluate_campaign(
        _input(
            body=body,
            required_proof_terms=("dashboard refresh time",),
            anchor_examples={"primary": [{"phrase": "anchor"}]},
        )
    )
    assert "missing_exact_proof_term" not in report.metadata["blocking_issues"]
    assert "dashboard refresh time" in report.metadata["used_proof_terms"]


def test_no_anchors_no_witnesses_does_not_enforce_proof_terms():
    # Without anchor_examples or witness_highlights, the proof-term gate
    # falls back to specificity (caller's responsibility) and the pack
    # does not block here.
    report = evaluate_campaign(
        _input(
            body="Generic message",
            required_proof_terms=("dashboard refresh time",),
        )
    )
    assert "missing_exact_proof_term" not in report.metadata["blocking_issues"]


def test_require_anchor_support_disabled_skips_proof_term_gate():
    # Even with anchors present, policy can disable the gate.
    policy = QualityPolicy(
        name="campaign_email",
        thresholds={"require_anchor_support": False},
    )
    report = evaluate_campaign(
        _input(
            body="Generic message",
            required_proof_terms=("dashboard refresh time",),
            anchor_examples={"primary": [{"phrase": "anchor"}]},
        ),
        policy=policy,
    )
    assert "missing_exact_proof_term" not in report.metadata["blocking_issues"]


# ---- Report tier banned language ----


def test_report_tier_blocks_dashboard():
    body = "Get the dashboard your team needs to compete."
    report = evaluate_campaign(
        _input(body=body, campaign={"tier": "report"})
    )
    blockers = report.metadata["blocking_issues"]
    assert any(b.startswith("report_tier_language:") for b in blockers)


def test_report_tier_blocks_free_trial():
    body = "Sign up for our free trial today."
    report = evaluate_campaign(
        _input(body=body, campaign={"tier": "report"})
    )
    blockers = report.metadata["blocking_issues"]
    assert any("free trial" in b for b in blockers)


def test_non_report_tier_allows_dashboard():
    body = "Get the dashboard your team needs to compete."
    report = evaluate_campaign(
        _input(body=body, campaign={"tier": "outreach"})
    )
    blockers = report.metadata["blocking_issues"]
    assert not any(b.startswith("report_tier_language:") for b in blockers)


def test_subject_is_exempt_from_report_tier_check():
    # The legacy pack only scans body+CTA, not subject.
    report = evaluate_campaign(
        _input(
            subject="Your dashboard insights",
            body="Generic insight content.",
            cta="Reply",
            campaign={"tier": "report"},
        )
    )
    assert not any(
        b.startswith("report_tier_language:")
        for b in report.metadata["blocking_issues"]
    )


def test_tier_falls_back_to_metadata():
    body = "Use the dashboard to monitor."
    report = evaluate_campaign(
        _input(body=body, campaign={"metadata": {"tier": "report"}})
    )
    blockers = report.metadata["blocking_issues"]
    assert any(b.startswith("report_tier_language:") for b in blockers)


# ---- Forbidden competitor names in cold email ----


def test_competitor_name_in_vendor_retention_cold_email_blocks():
    report = evaluate_campaign(
        _input(
            body="Your team is also evaluating Acme.",
            campaign={
                "channel": "email_cold",
                "target_mode": "vendor_retention",
                "competitors_considering": [{"vendor_name": "Acme"}],
            },
        )
    )
    blockers = report.metadata["blocking_issues"]
    assert any("competitor_name_in_email_cold" in b and "Acme" in b for b in blockers)


def test_competitor_name_pulled_from_signal_summary():
    report = evaluate_campaign(
        _input(
            body="Your team is also evaluating Acme.",
            campaign={
                "channel": "email_cold",
                "target_mode": "vendor_retention",
                "signal_summary": {
                    "competitor_distribution": [{"vendor_name": "Acme"}]
                },
            },
        )
    )
    blockers = report.metadata["blocking_issues"]
    assert any("competitor_name_in_email_cold" in b and "Acme" in b for b in blockers)


def test_incumbent_name_in_challenger_intel_blocks():
    report = evaluate_campaign(
        _input(
            body="Switch from Acme to a faster solution.",
            campaign={
                "channel": "email_cold",
                "target_mode": "challenger_intel",
                "signal_summary": {
                    "incumbents_losing": [{"vendor_name": "Acme"}]
                },
            },
        )
    )
    blockers = report.metadata["blocking_issues"]
    assert any("incumbent_name_in_email_cold" in b and "Acme" in b for b in blockers)


def test_incumbent_pulled_from_archetypes_dict():
    report = evaluate_campaign(
        _input(
            body="Switch from Acme to a faster solution.",
            campaign={
                "channel": "email_cold",
                "target_mode": "challenger_intel",
                "incumbent_archetypes": {
                    "category_dominant": [{"name": "Acme"}],
                },
            },
        )
    )
    blockers = report.metadata["blocking_issues"]
    assert any("incumbent_name_in_email_cold" in b and "Acme" in b for b in blockers)


def test_warm_email_does_not_enforce_forbidden_terms():
    # Only cold email is gated.
    report = evaluate_campaign(
        _input(
            body="Your team is also evaluating Acme.",
            campaign={
                "channel": "email_warm",
                "target_mode": "vendor_retention",
                "competitors_considering": [{"vendor_name": "Acme"}],
            },
        )
    )
    assert all(
        "competitor_name_in_email_cold" not in b
        for b in report.metadata["blocking_issues"]
    )


# ---- Private account name leak ----


def test_anchor_company_leak_blocks():
    report = evaluate_campaign(
        _input(
            body="See what GhostCo's team did to fix this.",
            anchor_examples={
                "primary": [
                    {
                        "phrase": "anchor phrase",
                        "reviewer_company": "GhostCo",
                    }
                ]
            },
        )
    )
    blockers = report.metadata["blocking_issues"]
    assert any("private_account_name_leak" in b and "GhostCo" in b for b in blockers)


def test_witness_company_leak_blocks():
    report = evaluate_campaign(
        _input(
            body="See what GhostCo's team did.",
            witness_highlights=({"reviewer_company": "GhostCo"},),
        )
    )
    blockers = report.metadata["blocking_issues"]
    assert any("private_account_name_leak" in b and "GhostCo" in b for b in blockers)


def test_company_not_in_message_does_not_block():
    report = evaluate_campaign(
        _input(
            body="Generic outbound copy without specific names.",
            witness_highlights=({"reviewer_company": "GhostCo"},),
        )
    )
    assert all(
        "private_account_name_leak" not in b
        for b in report.metadata["blocking_issues"]
    )


# ---- Token boundary correctness ----


def test_contains_term_uses_token_boundaries():
    # "Acme" should not match "Acmecorp" — whole-token match only.
    report = evaluate_campaign(
        _input(
            body="We integrate with Acmecorp's API.",
            campaign={
                "channel": "email_cold",
                "target_mode": "vendor_retention",
                "competitors_considering": [{"vendor_name": "Acme"}],
            },
        )
    )
    assert all(
        "competitor_name_in_email_cold" not in b
        for b in report.metadata["blocking_issues"]
    )


# ---- HTML stripping in body normalization ----


def test_html_tags_stripped_from_body_for_matching():
    # The legacy normalizer strips HTML before matching.
    report = evaluate_campaign(
        _input(
            body="<p>We integrate with <b>Acme</b> APIs.</p>",
            campaign={
                "channel": "email_cold",
                "target_mode": "vendor_retention",
                "competitors_considering": [{"vendor_name": "Acme"}],
            },
        )
    )
    blockers = report.metadata["blocking_issues"]
    assert any("competitor_name_in_email_cold" in b and "Acme" in b for b in blockers)


# ---- Decision aggregation ----


def test_blockers_take_precedence_over_warnings():
    report = evaluate_campaign(
        _input(
            specificity_blocking_issues=("hard_failure",),
            specificity_warnings=("soft_warning",),
        )
    )
    assert report.decision == GateDecision.BLOCK


def test_only_warnings_returns_warn():
    report = evaluate_campaign(
        _input(specificity_warnings=("soft_warning",))
    )
    assert report.decision == GateDecision.WARN


# ---- Frozen output ----


def test_quality_report_is_frozen():
    report = evaluate_campaign(_input())
    with pytest.raises(Exception):
        report.passed = False  # type: ignore[misc]
