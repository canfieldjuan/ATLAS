from __future__ import annotations

from collections import UserDict
from datetime import date
from types import MappingProxyType

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
    campaign_reasoning_context_metadata,
    campaign_reasoning_atom_context,
    campaign_reasoning_delta_summary,
    campaign_reasoning_scope_summary,
    normalize_campaign_reasoning_context,
)


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


def test_normalize_campaign_reasoning_context_accepts_host_compressed_fields() -> None:
    context = normalize_campaign_reasoning_context(
        {
            "reasoning_anchor_examples": {
                "named_account": [
                    {"witness_id": "w1", "excerpt_text": "Pricing came up."},
                    "invalid",
                ]
            },
            "reasoning_witness_highlights": [
                {"witness_id": "w1", "excerpt_text": "Pricing came up."}
            ],
            "reasoning_reference_ids": {"witness_ids": ["w1", ""]},
            "reasoning_context": {
                "account_signals": [{"company": "Acme", "primary_pain": "pricing"}],
                "timing_windows": [{"window_type": "renewal", "anchor": "Q3"}],
                "proof_points": [{"label": "pricing_mentions", "value": 12}],
                "coverage_limits": ["thin_account_signals", ""],
            },
            "reasoning_scope_summary": {"selection_strategy": "host_compressed"},
        }
    )

    assert context.anchor_examples["named_account"][0]["witness_id"] == "w1"
    assert context.witness_highlights[0]["excerpt_text"] == "Pricing came up."
    assert context.reference_ids == {"witness_ids": ("w1",)}
    assert context.account_signals[0]["company"] == "Acme"
    assert context.timing_windows[0]["anchor"] == "Q3"
    assert context.proof_points[0]["label"] == "pricing_mentions"
    assert context.coverage_limits == ("thin_account_signals",)
    assert context.scope_summary == {"selection_strategy": "host_compressed"}


def test_normalize_campaign_reasoning_context_accepts_mapping_and_sequence_inputs() -> None:
    context = normalize_campaign_reasoning_context(
        UserDict({
            "reasoning_anchor_examples": MappingProxyType({
                "named_account": (
                    MappingProxyType({
                        "witness_id": "w1",
                        "excerpt_text": "Pricing came up.",
                    }),
                )
            }),
            "reasoning_witness_highlights": (
                MappingProxyType({
                    "witness_id": "w1",
                    "excerpt_text": "Pricing came up.",
                }),
            ),
            "reasoning_reference_ids": MappingProxyType({
                "witness_ids": ("w1", ""),
            }),
            "reasoning_context": MappingProxyType({
                "account_signals": (
                    MappingProxyType({"company": "Acme", "primary_pain": "pricing"}),
                ),
                "timing_windows": (
                    MappingProxyType({"window_type": "renewal", "anchor": "Q3"}),
                ),
                "proof_points": (
                    MappingProxyType({"label": "pricing_mentions", "value": 12}),
                ),
                "coverage_limits": ("thin_account_signals", ""),
            }),
        })
    )

    assert context.anchor_examples["named_account"][0]["witness_id"] == "w1"
    assert context.witness_highlights[0]["excerpt_text"] == "Pricing came up."
    assert context.reference_ids == {"witness_ids": ("w1",)}
    assert context.account_signals[0]["company"] == "Acme"
    assert context.timing_windows[0]["anchor"] == "Q3"
    assert context.proof_points[0]["value"] == 12
    assert context.coverage_limits == ("thin_account_signals",)


def test_campaign_reasoning_context_metadata_uses_campaign_storage_keys() -> None:
    context = normalize_campaign_reasoning_context(
        {
            "anchor_examples": {"proof": [{"witness_id": "w1"}]},
            "witness_highlights": [{"witness_id": "w1"}],
            "reference_ids": {"witness_ids": ["w1"]},
        }
    )

    metadata = campaign_reasoning_context_metadata(context)

    assert metadata["reasoning_anchor_examples"]["proof"][0]["witness_id"] == "w1"
    assert metadata["reasoning_witness_highlights"][0]["witness_id"] == "w1"
    assert metadata["reasoning_reference_ids"] == {"witness_ids": ["w1"]}
    assert metadata["reasoning_context"]["anchor_examples"]["proof"][0]["witness_id"] == "w1"


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
