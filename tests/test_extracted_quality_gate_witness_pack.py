"""Tests for extracted_quality_gate.witness_pack.

Pure unit tests against the lifted specificity helpers and the
``evaluate_witness_specificity`` pack-contract entry point.
"""

from __future__ import annotations

import pytest

from extracted_quality_gate.types import (
    GateDecision,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)
from extracted_quality_gate.witness_pack import (
    campaign_proof_terms_from_audit,
    evaluate_specificity_support,
    evaluate_witness_specificity,
    merge_specificity_contexts,
    specificity_audit_snapshot,
    specificity_signal_terms,
    surface_specificity_context,
)


# ---- surface_specificity_context ----


def test_surface_context_redacts_company_for_blog():
    source = {
        "anchor_examples": {
            "primary": [
                {
                    "witness_id": "w1",
                    "excerpt_text": "GhostCo migrated last quarter",
                    "reviewer_company": "GhostCo",
                }
            ]
        },
        "witness_highlights": [
            {"witness_id": "w2", "reviewer_company": "Acme Corp"},
        ],
    }
    context = surface_specificity_context(source, surface="blog")
    primary = context["anchor_examples"]["primary"][0]
    assert primary["excerpt_text"] == "a customer migrated last quarter"
    assert "reviewer_company" not in primary
    highlight = context["witness_highlights"][0]
    assert "reviewer_company" not in highlight


def test_surface_context_keeps_company_for_internal_surfaces():
    source = {
        "anchor_examples": {
            "primary": [
                {
                    "excerpt_text": "GhostCo migrated last quarter",
                    "reviewer_company": "GhostCo",
                }
            ]
        }
    }
    context = surface_specificity_context(source, surface="battle_card")
    primary = context["anchor_examples"]["primary"][0]
    assert primary["reviewer_company"] == "GhostCo"
    assert "GhostCo" in primary["excerpt_text"]


def test_surface_context_falls_back_to_reasoning_keys():
    source = {
        "reasoning_anchor_examples": {
            "primary": [{"excerpt_text": "switched in Q3"}]
        },
        "reasoning_witness_highlights": [{"excerpt_text": "renewal pressure"}],
        "reasoning_reference_ids": {"witness_ids": ["w1", "w2"]},
    }
    context = surface_specificity_context(source, surface="campaign")
    assert context["anchor_examples"]
    assert context["witness_highlights"]
    assert context["reference_ids"] == {"witness_ids": ["w1", "w2"]}


def test_surface_context_handles_none_source():
    assert surface_specificity_context(None, surface="blog") == {}


# ---- merge_specificity_contexts ----


def test_merge_dedupes_witnesses_by_marker():
    a = {
        "witness_highlights": [
            {"witness_id": "w1", "excerpt_text": "first"},
        ],
    }
    b = {
        "witness_highlights": [
            {"witness_id": "w1", "excerpt_text": "duplicate"},
            {"witness_id": "w2", "excerpt_text": "second"},
        ],
    }
    merged = merge_specificity_contexts(a, b)
    ids = {row["witness_id"] for row in merged["witness_highlights"]}
    assert ids == {"w1", "w2"}


def test_merge_combines_anchor_buckets():
    a = {"anchor_examples": {"primary": [{"witness_id": "w1"}]}}
    b = {"anchor_examples": {"primary": [{"witness_id": "w2"}]}}
    merged = merge_specificity_contexts(a, b)
    ids = {row["witness_id"] for row in merged["anchor_examples"]["primary"]}
    assert ids == {"w1", "w2"}


def test_merge_handles_none_inputs():
    assert merge_specificity_contexts(None, None) == {}


# ---- specificity_signal_terms ----


def test_signal_terms_extracts_groups():
    witnesses = [
        {
            "competitor": "Acme",
            "time_anchor": "renewal",
            "pain_category": "pricing",
            "numeric_literals": {"raw": ["40%"]},
            "reviewer_company": "GhostCo",
            "replacement_mode": "none",
            "operating_model_shift": "reduce_seats",
        }
    ]
    terms = specificity_signal_terms(
        anchor_examples=None,
        witness_highlights=witnesses,
        allow_company_names=True,
    )
    assert "acme" in terms["competitor_terms"]
    assert "renewal" in terms["timing_terms"]
    assert "pricing" in terms["pain_terms"]
    assert "40%" in terms["numeric_terms"]
    assert "ghostco" in terms["companies"]
    # "none" is filtered from workflow_terms
    assert "none" not in terms["workflow_terms"]
    assert "reduce_seats" in terms["workflow_terms"]


def test_signal_terms_redacts_companies_when_disallowed():
    witnesses = [{"reviewer_company": "GhostCo"}]
    terms = specificity_signal_terms(
        anchor_examples=None,
        witness_highlights=witnesses,
        allow_company_names=False,
    )
    assert terms["companies"] == set()


# ---- evaluate_specificity_support ----


def test_evaluate_returns_pass_when_anchors_match():
    witnesses = [{"time_anchor": "renewal", "pain_category": "pricing"}]
    result = evaluate_specificity_support(
        "We saw renewal pressure last quarter",
        witness_highlights=witnesses,
        allow_company_names=False,
    )
    assert result["blocking_issues"] == []
    assert "timing_terms" in result["matched_groups"]


def test_evaluate_blocks_when_anchors_unmatched():
    witnesses = [{"time_anchor": "renewal", "pain_category": "pricing"}]
    result = evaluate_specificity_support(
        "Generic content with no specific anchors",
        witness_highlights=witnesses,
        allow_company_names=False,
    )
    assert any("witness-backed anchor" in issue for issue in result["blocking_issues"])


def test_evaluate_warns_when_witness_refs_without_data():
    result = evaluate_specificity_support(
        "any text",
        reference_ids={"witness_ids": ["w1"]},
        allow_company_names=False,
    )
    assert any(
        "witness references exist" in warning for warning in result["warnings"]
    )


def test_evaluate_blocks_when_timing_required():
    witnesses = [{"time_anchor": "renewal", "pain_category": "pricing"}]
    # text contains pain term but not timing -- with require_timing, blocks
    result = evaluate_specificity_support(
        "We saw pricing problems in many accounts",
        witness_highlights=witnesses,
        allow_company_names=False,
        require_timing_or_numeric_when_available=True,
    )
    assert any(
        "timing or numeric anchor" in issue for issue in result["blocking_issues"]
    )


def test_evaluate_numeric_term_filter_drops_anchors():
    witnesses = [{"numeric_literals": {"raw": ["123"]}}]  # filtered out
    result = evaluate_specificity_support(
        "any text",
        witness_highlights=witnesses,
        allow_company_names=False,
        numeric_term_filter=lambda term: False,
    )
    assert "numeric_terms" not in result["matched_groups"]


def test_evaluate_skips_competitor_warning_when_disabled():
    witnesses = [{"competitor": "Acme"}]
    result = evaluate_specificity_support(
        "any text",
        witness_highlights=witnesses,
        allow_company_names=False,
        include_competitor_terms=False,
    )
    assert all(
        "competitor-backed anchor" not in warning for warning in result["warnings"]
    )


# ---- specificity_audit_snapshot ----


def test_audit_snapshot_pass_status():
    result = specificity_audit_snapshot("any text", allow_company_names=False)
    assert result["status"] == "pass"
    assert result["anchor_count"] == 0
    assert result["highlight_count"] == 0


def test_audit_snapshot_includes_signal_term_lists():
    witnesses = [{"time_anchor": "renewal"}]
    result = specificity_audit_snapshot(
        "renewal pressure",
        witness_highlights=witnesses,
        allow_company_names=False,
    )
    assert "timing_terms" in result["signal_terms"]
    assert "renewal" in result["signal_terms"]["timing_terms"]


def test_audit_snapshot_lists_missing_groups():
    witnesses = [
        {"time_anchor": "renewal", "pain_category": "pricing"},
    ]
    result = specificity_audit_snapshot(
        "renewal pressure",  # only timing matches; pain unmet
        witness_highlights=witnesses,
        allow_company_names=False,
    )
    assert "pain_terms" in result["missing_groups"]
    assert "timing_terms" in result["matched_groups"]


# ---- campaign_proof_terms_from_audit ----


def test_proof_terms_from_audit_prefers_numeric_timing_when_blocking():
    audit = {
        "blocking_issues": [
            "content omits a concrete timing or numeric anchor even though one is available"
        ],
        "signal_terms": {
            "numeric_terms": ["40_percent_lift"],
            "competitor_terms": ["acme"],
            "pain_terms": ["pricing"],
        },
    }
    terms = campaign_proof_terms_from_audit(audit, channel="email_warm", limit=2)
    # Numeric / timing prioritized when blocking issue forces it
    assert "40 percent lift" in terms


def test_proof_terms_from_audit_skips_competitor_for_cold_email():
    audit = {
        "signal_terms": {
            "competitor_terms": ["acme"],
            "pain_terms": ["pricing"],
        },
    }
    terms = campaign_proof_terms_from_audit(audit, channel="email_cold", limit=2)
    assert "acme" not in terms
    assert "pricing" in terms


def test_proof_terms_from_audit_returns_empty_for_invalid_audit():
    assert campaign_proof_terms_from_audit(None, channel="email_cold", limit=2) == []
    assert campaign_proof_terms_from_audit({}, channel="email_cold", limit=2) == []


# ---- evaluate_witness_specificity (pack contract) ----


def _input(text: str = "any text", **context_overrides) -> QualityInput:
    base_context: dict = {
        "surface": "blog",
        "anchor_examples": {},
        "witness_highlights": [],
        "reference_ids": {},
    }
    base_context.update(context_overrides)
    return QualityInput(
        artifact_type="content",
        artifact_id="test",
        content=text,
        context=base_context,
    )


def test_pack_contract_returns_quality_report():
    report = evaluate_witness_specificity(_input())
    assert isinstance(report, QualityReport)


def test_pack_contract_passes_when_no_anchors():
    report = evaluate_witness_specificity(_input())
    assert report.passed is True
    assert report.decision == GateDecision.PASS


def test_pack_contract_blocks_unmatched_anchors():
    witnesses = [{"time_anchor": "renewal", "pain_category": "pricing"}]
    report = evaluate_witness_specificity(
        _input("generic copy", witness_highlights=witnesses)
    )
    assert report.decision == GateDecision.BLOCK
    assert any(
        f.severity == GateSeverity.BLOCKER for f in report.findings
    )


def test_pack_contract_warns_only_returns_warn():
    # No anchors, but witness_ids exist -> warning, not blocker
    report = evaluate_witness_specificity(
        _input("text", reference_ids={"witness_ids": ["w1"]})
    )
    assert report.decision == GateDecision.WARN


def test_pack_contract_metadata_carries_signal_terms():
    witnesses = [{"time_anchor": "renewal", "competitor": "Acme"}]
    report = evaluate_witness_specificity(
        _input("renewal Acme", witness_highlights=witnesses)
    )
    md = report.metadata
    assert "signal_terms" in md
    assert "matched_groups" in md
    # Both timing + competitor matched
    assert "timing_terms" in md["matched_groups"]
    assert "competitor_terms" in md["matched_groups"]


def test_pack_contract_policy_overrides_thresholds():
    witnesses = [{"time_anchor": "renewal"}]
    # require_anchor_support=False -> no block even if unmatched
    policy = QualityPolicy(
        name="witness",
        thresholds={"require_anchor_support": False},
    )
    report = evaluate_witness_specificity(
        _input("generic text", witness_highlights=witnesses),
        policy=policy,
    )
    blockers = [f for f in report.findings if f.severity == GateSeverity.BLOCKER]
    assert blockers == []


def test_pack_contract_surface_drives_company_default():
    # surface=blog -> allow_company_names defaults to False
    witnesses = [{"reviewer_company": "GhostCo"}]
    report = evaluate_witness_specificity(
        _input("text", surface="blog", witness_highlights=witnesses)
    )
    md = report.metadata
    # "companies" group should be empty since names are redacted out
    assert "companies" not in md["available_groups"]


def test_pack_contract_quality_report_is_frozen():
    report = evaluate_witness_specificity(_input())
    with pytest.raises(Exception):
        report.passed = False  # type: ignore[misc]


# ---- atlas re-export sanity ----


def test_atlas_re_export_paths_match():
    # The atlas-side path should expose the same callables as the pack.
    from atlas_brain.autonomous.tasks._b2b_specificity import (
        campaign_proof_terms_from_audit as atlas_campaign_proof,
        evaluate_specificity_support as atlas_evaluate,
        merge_specificity_contexts as atlas_merge,
        specificity_audit_snapshot as atlas_snapshot,
        specificity_signal_terms as atlas_signal_terms,
        surface_specificity_context as atlas_surface,
    )
    assert atlas_campaign_proof is campaign_proof_terms_from_audit
    assert atlas_evaluate is evaluate_specificity_support
    assert atlas_merge is merge_specificity_contexts
    assert atlas_snapshot is specificity_audit_snapshot
    assert atlas_signal_terms is specificity_signal_terms
    assert atlas_surface is surface_specificity_context
