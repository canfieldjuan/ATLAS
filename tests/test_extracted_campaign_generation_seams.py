from __future__ import annotations

from datetime import date

import pytest

from extracted_content_pipeline.autonomous.visibility import emit_event, record_attempt
from extracted_content_pipeline.campaign_ports import VisibilitySink
from extracted_content_pipeline.pipelines.notify import configure_pipeline_notification_sink
from extracted_content_pipeline.services.b2b.account_opportunity_claims import (
    account_opportunity_source_review_count,
    build_account_opportunity_claim,
    serialize_product_claim,
)
from extracted_content_pipeline.services.campaign_quality import campaign_quality_revalidation
from extracted_content_pipeline.services.campaign_reasoning_context import (
    campaign_reasoning_atom_context,
    campaign_reasoning_delta_summary,
    campaign_reasoning_scope_summary,
)
from extracted_content_pipeline.settings import build_settings


class FakeVisibilitySink(VisibilitySink):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, event_type, payload):
        self.events.append((event_type, dict(payload)))


class FakePool:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        if self.fail:
            raise RuntimeError("db unavailable")
        return "INSERT 0 1"


def test_account_opportunity_claim_renders_but_does_not_report_single_review() -> None:
    row = {
        "company": "Acme",
        "vendor": "LegacyCRM",
        "review_id": "rev-1",
        "buying_stage": "evaluation",
        "quotes": [{"text": "We are comparing alternatives."}],
    }

    claim = build_account_opportunity_claim(
        row,
        as_of_date=date(2026, 5, 2),
        analysis_window_days=90,
    )
    serialized = serialize_product_claim(
        claim,
        source_review_count=account_opportunity_source_review_count(row),
    )

    assert claim.render_allowed is True
    assert claim.report_allowed is False
    assert serialized["claim_scope"] == "account"
    assert serialized["claim_type"] == "account_opportunity_readiness"
    assert serialized["source_review_count"] == 1
    assert serialized["suppression_reason"] == "low_confidence"


def test_account_opportunity_claim_requires_identity_source_and_intent() -> None:
    claim = build_account_opportunity_claim(
        {"company": "Acme", "vendor": "LegacyCRM"},
        as_of_date=date(2026, 5, 2),
        analysis_window_days=90,
    )

    assert claim.render_allowed is False
    assert claim.report_allowed is False
    assert claim.evidence_posture == "unverified"


def test_campaign_reasoning_context_extracts_bounded_prompt_material() -> None:
    context = campaign_reasoning_atom_context(
        {
            "theses": [
                {"summary": "Budget pressure", "why_now": "Renewals"},
                {"summary": "Feature gaps"},
                {"summary": "Third item"},
            ],
            "timing_windows": [
                {"window_type": "renewal", "start_or_anchor": "Q3", "urgency": "high"}
            ],
            "account_signals": [
                {"company": "Acme", "primary_pain": "pricing"},
                {"company": "Beta", "competitor_context": "evaluating Rival"},
                {"company": "Gamma"},
            ],
        }
    )

    assert [item["summary"] for item in context["top_theses"]] == [
        "Budget pressure",
        "Feature gaps",
    ]
    assert context["timing_windows"][0]["anchor"] == "Q3"
    assert [item["company"] for item in context["account_signals"]] == ["Acme", "Beta"]


def test_campaign_reasoning_scope_and_delta_summaries_are_defensive() -> None:
    assert campaign_reasoning_scope_summary(None) == {}
    assert campaign_reasoning_scope_summary(
        {
            "selection_strategy": "canary",
            "reviews_in_scope": 12,
            "empty": "",
        }
    ) == {"selection_strategy": "canary", "reviews_in_scope": 12}

    assert campaign_reasoning_delta_summary(
        {
            "changed": True,
            "wedge_changed": "yes",
            "theses_added": ["a", "b", "c", "d"],
        }
    ) == {
        "changed": True,
        "wedge_changed": True,
        "theses_added": ["a", "b", "c"],
    }


def test_campaign_quality_revalidation_returns_expected_envelope() -> None:
    result = campaign_quality_revalidation(
        campaign={
            "subject": "Hi [Name]",
            "body": "We saw pricing pressure.",
            "metadata": {"campaign_proof_terms": ["pricing pressure"]},
        },
        boundary="generation",
        specificity_context={"anchor_examples": {"pricing": []}},
    )

    assert result["audit"]["boundary"] == "generation"
    assert result["audit"]["blocking_issues"] == ["placeholder_token"]
    assert result["audit"]["campaign_proof_terms"] == ["pricing pressure"]
    assert result["metadata"]["latest_specificity_audit"] == result["audit"]
    assert result["specificity_context"] == {"anchor_examples": {"pricing": []}}


def test_campaign_quality_revalidation_blocks_missing_required_anchor_support() -> None:
    result = campaign_quality_revalidation(
        campaign={
            "subject": "Account pressure",
            "body": "We should talk about the account.",
            "tier": "report",
            "target_mode": "vendor_retention",
            "metadata": {},
        },
        boundary="generation",
        specificity_context={
            "anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "w1",
                        "excerpt_text": "The Q2 renewal now carries a $200k/year issue.",
                    }
                ]
            },
            "witness_highlights": [{"witness_id": "w1"}],
            "reference_ids": {"witness_ids": ["w1"]},
        },
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
    )

    assert result["audit"]["status"] == "fail"
    assert "missing_anchor_support" in result["audit"]["blocking_issues"]
    assert "missing_timing_or_numeric" in result["audit"]["blocking_issues"]
    assert result["metadata"]["tier"] == "report"
    assert result["metadata"]["target_mode"] == "vendor_retention"
    assert result["metadata"]["reasoning_reference_ids"] == {"witness_ids": ["w1"]}


def test_campaign_quality_revalidation_enforces_specificity_defaults_when_flags_omitted() -> None:
    result = campaign_quality_revalidation(
        campaign={
            "subject": "Account pressure",
            "body": "We should talk about the account.",
            "metadata": {},
        },
        boundary="generation",
        specificity_context={
            "anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "w1",
                        "excerpt_text": "The Q2 renewal now carries a $200k/year issue.",
                    }
                ]
            }
        },
    )

    assert result["audit"]["status"] == "fail"
    assert "missing_anchor_support" in result["audit"]["blocking_issues"]
    assert "missing_timing_or_numeric" in result["audit"]["blocking_issues"]


def test_campaign_quality_revalidation_passes_with_anchor_and_preserves_metadata() -> None:
    result = campaign_quality_revalidation(
        campaign={
            "subject": "Renewal pressure",
            "body": "The Q2 renewal now carries a $200k/year issue.",
            "tier": "report",
            "target_mode": "vendor_retention",
            "metadata": {"existing": True},
        },
        boundary="manual_approval",
        specificity_context={
            "anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "w1",
                        "excerpt_text": "The Q2 renewal now carries a $200k/year issue.",
                    }
                ]
            },
            "reference_ids": {"witness_ids": ["w1"]},
        },
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
    )

    assert result["audit"]["status"] == "pass"
    assert result["audit"]["blocking_issues"] == []
    assert result["audit"]["used_proof_terms"] == [
        "The Q2 renewal now carries a $200k/year issue."
    ]
    assert result["metadata"]["existing"] is True
    assert result["metadata"]["reasoning_anchor_examples"]["outlier_or_named_account"][0]["witness_id"] == "w1"
    assert result["metadata"]["latest_specificity_audit"]["failure_explanation"]["anchor_count"] == 1


def test_extracted_b2b_campaign_defaults_match_copied_task_contract() -> None:
    cfg = build_settings().b2b_campaign

    assert cfg.specificity_require_anchor_support is True
    assert cfg.specificity_require_timing_or_numeric_when_available is True
    assert cfg.specificity_revision_term_limit == 3
    assert cfg.word_limits["default"]["email_cold"] == [50, 150]
    assert cfg.word_limits["vendor_retention"]["email_cold"] == [50, 125]
    assert cfg.word_limits["churning_company"]["linkedin"] == [0, 100]


@pytest.mark.asyncio
async def test_visibility_helpers_persist_and_emit_to_sink() -> None:
    sink = FakeVisibilitySink()
    previous = configure_pipeline_notification_sink(sink)
    pool = FakePool()
    try:
        event_id = await emit_event(
            pool,
            stage="campaign_generation",
            event_type="generation_failure",
            entity_type="campaign",
            entity_id="cmp-1",
            summary="Generation failed",
        )
        attempt_id = await record_attempt(
            pool,
            artifact_type="campaign",
            artifact_id="cmp-1",
            stage="generation",
            status="failed",
            blocking_issues=["placeholder_token"],
        )
    finally:
        configure_pipeline_notification_sink(previous)

    assert event_id
    assert attempt_id
    assert len(pool.calls) == 2
    assert [event[0] for event in sink.events] == [
        "pipeline_visibility_event",
        "artifact_attempt",
    ]
    assert sink.events[1][1]["blocking_issues"] == ["placeholder_token"]


@pytest.mark.asyncio
async def test_visibility_helpers_are_best_effort_when_pool_fails() -> None:
    sink = FakeVisibilitySink()
    previous = configure_pipeline_notification_sink(sink)
    try:
        attempt_id = await record_attempt(
            FakePool(fail=True),
            artifact_type="campaign",
            stage="generation",
            status="failed",
        )
    finally:
        configure_pipeline_notification_sink(previous)

    assert attempt_id
    assert sink.events[0][0] == "artifact_attempt"
