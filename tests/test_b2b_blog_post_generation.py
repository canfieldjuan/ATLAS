"""Focused tests for evidence-vault overlays in B2B blog generation."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import atlas_brain.autonomous.tasks.b2b_blog_post_generation as blog_mod
from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
    _assemble_and_store,
    _candidate_overlaps_gap_pain,
    _blueprint_churn_report,
    _blueprint_migration_guide,
    _blueprint_vendor_alternative,
    _build_specialized_blog_review_rows_from_evidence_vault,
    _detect_campaign_content_gaps,
    _fetch_negative_quotes,
    _fetch_positive_quotes,
    _gather_data,
    _load_pool_layers_for_blog,
    _merge_blog_quotes_with_evidence_vault,
    _merge_blog_signals_with_evidence_vault,
)
from atlas_brain.storage.models import ScheduledTask


def test_merge_blog_signals_with_evidence_vault_prefers_canonical_rows():
    raw = [
        {"pain_category": "pricing", "signal_count": 3, "avg_urgency": 4.0, "feature_gaps": ["Legacy exports"]},
        {"pain_category": "support", "signal_count": 2, "avg_urgency": 6.2, "feature_gaps": []},
    ]
    vault = {
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "mention_count_total": 12,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.1},
            },
            {
                "key": "custom_roles",
                "label": "Custom roles",
                "evidence_type": "feature_gap",
                "mention_count_total": 5,
            },
        ],
    }
    merged = _merge_blog_signals_with_evidence_vault(raw, vault)
    assert merged[0]["pain_category"] == "Pricing opacity"
    assert merged[0]["signal_count"] == 12
    assert merged[0]["avg_urgency"] == 7.1
    assert merged[0]["feature_gaps"] == ["Custom roles"]
    assert any(item["pain_category"] == "support" for item in merged)


@pytest.mark.asyncio
async def test_batch_slug_check_blocks_recent_rejected_slug(monkeypatch):
    recent = datetime.now(timezone.utc) - timedelta(hours=2)

    class Pool:
        async def fetch(self, *_args):
            return [{
                "slug": "clickup-deep-dive-2026-04",
                "status": "rejected",
                "rejection_count": 1,
                "rejected_at": recent,
            }]

    blocked = await blog_mod._batch_slug_check(Pool(), ["clickup-deep-dive-2026-04"])
    assert blocked == {"clickup-deep-dive-2026-04"}


@pytest.mark.asyncio
async def test_batch_slug_check_allows_rejected_slug_after_cooldown(monkeypatch):
    stale = datetime.now(timezone.utc) - timedelta(hours=30)

    class Pool:
        async def fetch(self, *_args):
            return [{
                "slug": "clickup-deep-dive-2026-04",
                "status": "rejected",
                "rejection_count": 1,
                "rejected_at": stale,
            }]

    blocked = await blog_mod._batch_slug_check(Pool(), ["clickup-deep-dive-2026-04"])
    assert blocked == set()


@pytest.mark.asyncio
async def test_batch_slug_check_allows_failed_slug_retry(monkeypatch):
    class Pool:
        async def fetch(self, *_args):
            return [{
                "slug": "clickup-deep-dive-2026-04",
                "status": "failed",
                "rejection_count": 0,
                "rejected_at": None,
            }]

    blocked = await blog_mod._batch_slug_check(Pool(), ["clickup-deep-dive-2026-04"])
    assert blocked == set()


def test_merge_blog_quotes_with_evidence_vault_prefers_canonical_quotes():
    raw = [
        {
            "phrase": "Pricing opacity kept surprising us",
            "vendor": "Zendesk",
            "urgency": 5.0,
            "role": "Ops Manager",
            "company": "Acme",
            "company_size": "Mid-Market",
            "industry": "SaaS",
            "source_name": "g2",
            "sentiment": "negative",
        },
        {
            "phrase": "The integrations save hours every week",
            "vendor": "Zendesk",
            "urgency": 2.0,
            "role": "RevOps",
            "company": "Acme",
            "company_size": "Mid-Market",
            "industry": "SaaS",
            "source_name": "g2",
            "sentiment": "positive",
        },
    ]
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "best_quote": "Pricing opacity kept surprising us",
                "mention_count_total": 11,
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "company": "Acme"},
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.4},
            },
        ],
        "strength_evidence": [
            {
                "best_quote": "The integrations save hours every week",
                "mention_count_total": 6,
                "quote_source": {"source": "capterra", "reviewer_title": "RevOps", "company": "Acme"},
                "supporting_metrics": {},
            },
        ],
    }
    merged = _merge_blog_quotes_with_evidence_vault(raw, vault)
    assert merged[0]["phrase"] == "Pricing opacity kept surprising us"
    assert merged[0]["source_name"] == "reddit"
    assert merged[0]["sentiment"] == "negative"
    assert any(item["phrase"] == "The integrations save hours every week" and item["source_name"] == "capterra" for item in merged)


def test_build_specialized_blog_review_rows_from_evidence_vault_filters_pricing():
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "best_quote": "The contract cost kept climbing after the add-ons",
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "rating": 2.0},
                "supporting_metrics": {"avg_urgency_when_mentioned": 8.1},
                "mention_count_total": 9,
            },
            {
                "key": "support",
                "label": "Support",
                "evidence_type": "pain_category",
                "best_quote": "Support vanished during onboarding",
                "quote_source": {"source": "g2", "reviewer_title": "Director"},
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.0},
                "mention_count_total": 12,
            },
        ],
    }
    rows = _build_specialized_blog_review_rows_from_evidence_vault(
        vault,
        mode="pricing",
        limit=5,
    )
    assert len(rows) == 1
    assert rows[0]["text"] == "The contract cost kept climbing after the add-ons"
    assert rows[0]["source_name"] == "reddit"


def test_ensure_methodology_context_injects_review_period_and_disclaimer():
    blueprint = blog_mod.PostBlueprint(
        topic_type="vendor_showdown",
        slug="crowdstrike-vs-sentinelone-2026-03",
        suggested_title="CrowdStrike vs SentinelOne",
        tags=["security"],
        data_context={
            "review_period": "2025-06 to 2026-03",
            "data_source_label": "public B2B software review platforms",
        },
        sections=[
            blog_mod.SectionSpec(
                id="hook",
                heading="Introduction",
                goal="Lead with the comparison",
            ),
        ],
        charts=[],
    )
    content = {"content": "# CrowdStrike vs SentinelOne\n\nOpening paragraph."}
    updated = blog_mod._ensure_methodology_context(blueprint, content)

    assert "2025-06 to 2026-03" in updated["content"]
    assert "self-selected" in updated["content"]


@pytest.mark.asyncio
async def test_generate_content_async_traces_blog_business_metadata(monkeypatch):
    blueprint = blog_mod.PostBlueprint(
        topic_type="vendor_deep_dive",
        slug="jira-deep-dive-2026-04",
        suggested_title="Jira Deep Dive",
        tags=["jira", "project-management"],
        data_context={"vendor": "Jira", "category": "Project Management"},
        sections=[],
        charts=[],
    )

    class DummyLLM:
        model = "anthropic/claude-sonnet-4-5"
        name = "openrouter"

        def chat(self, **_kwargs):
            return {
                "response": json.dumps({
                    "title": "Jira Deep Dive",
                    "description": "Desc",
                    "content": "# Jira\n\nBody",
                }),
                "usage": {"input_tokens": 1234, "output_tokens": 456},
                "_trace_meta": {
                    "provider_request_id": "req_blog_123",
                    "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
                },
            }

    trace_calls = []

    class DummyRegistry:
        def get(self, name):
            assert name == "digest/b2b_blog_post_generation"
            return SimpleNamespace(content="system")

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_blog_post_generation.get_skill_registry",
        lambda: DummyRegistry(),
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.skills.registry.get_skill_registry",
        lambda: DummyRegistry(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        lambda *_args, **_kwargs: SimpleNamespace(
            provider="openrouter",
            model="anthropic/claude-sonnet-4-5",
            request_envelope={"messages": []},
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.trace_llm_call",
        lambda *args, **kwargs: trace_calls.append((args, kwargs)),
    )

    usage_out = {}
    result = await blog_mod._generate_content_async(
        DummyLLM(),
        blueprint,
        4096,
        run_id="run-blog-123",
        usage_out=usage_out,
    )

    assert result is not None
    assert len(trace_calls) == 1
    _, kwargs = trace_calls[0]
    assert kwargs["metadata"]["workflow"] == "b2b_blog_post_generation"
    assert kwargs["metadata"]["report_type"] == "vendor_deep_dive"
    assert kwargs["metadata"]["vendor_name"] == "Jira"
    assert kwargs["metadata"]["run_id"] == "run-blog-123"
    assert kwargs["metadata"]["source_name"] == "b2b_blog_post_generation"
    assert kwargs["metadata"]["entity_type"] == "blog_post"
    assert kwargs["metadata"]["entity_id"] == "jira-deep-dive-2026-04"
    assert kwargs["provider_request_id"] == "req_blog_123"
    assert usage_out["input_tokens"] == 1234
    assert usage_out["output_tokens"] == 456
    assert usage_out["provider"] == "openrouter"
    assert usage_out["model"] == "anthropic/claude-sonnet-4-5"
    assert usage_out["provider_request_id"] == "req_blog_123"


@pytest.mark.asyncio
async def test_run_uses_anthropic_batch_for_blog_first_pass(monkeypatch):
    task = ScheduledTask(
        id=uuid4(),
        name="b2b_blog_post_generation",
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=300,
        enabled=True,
        metadata={"builtin_handler": "b2b_blog_post_generation"},
    )

    class FakePool:
        is_initialized = True

        def __init__(self):
            self.fetch = AsyncMock(return_value=[])

    class FakeAnthropicLLM:
        model = "claude-sonnet-4-5"
        name = "anthropic"

    class FakeDirectLLM:
        model = "anthropic/claude-sonnet-4-5"
        name = "openrouter"

    pool = FakePool()
    direct_llm = FakeDirectLLM()
    batch_llm = FakeAnthropicLLM()
    assembled: list[dict[str, object]] = []
    fallback_mark = AsyncMock()
    notify = AsyncMock()
    record_attempt = AsyncMock()

    monkeypatch.setattr(blog_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_enabled", True, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "anthropic_batch_enabled", True, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_anthropic_batch_enabled", True, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_anthropic_batch_min_items", 1, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_max_per_run", 1, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_temperature", 0.7, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_max_tokens", 4096, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_regenerate_mode", False, raising=False)

    monkeypatch.setattr(
        "atlas_brain.services.llm.anthropic.AnthropicLLM",
        FakeAnthropicLLM,
    )

    def _fake_get_pipeline_llm(*, workload, **_kwargs):
        if workload == "anthropic":
            return batch_llm
        return direct_llm

    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        _fake_get_pipeline_llm,
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.notify.send_pipeline_notification",
        notify,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.visibility.record_attempt",
        record_attempt,
    )

    class DummyRegistry:
        def get(self, name):
            assert name == "digest/b2b_blog_post_generation"
            return SimpleNamespace(content="system")

    monkeypatch.setattr(
        "atlas_brain.skills.registry.get_skill_registry",
        lambda: DummyRegistry(),
    )
    monkeypatch.setattr(
        blog_mod,
        "_select_topic",
        AsyncMock(side_effect=[
            ("vendor_deep_dive", {"vendor": "Jira", "slug": "jira-deep-dive-2026-04"}),
            None,
        ]),
    )
    monkeypatch.setattr(blog_mod, "_gather_data", AsyncMock(return_value={}))
    monkeypatch.setattr(blog_mod, "_load_pool_layers_for_blog", AsyncMock(return_value=None))
    monkeypatch.setattr(
        blog_mod,
        "_check_data_sufficiency",
        lambda *_args, **_kwargs: {"sufficient": True},
    )
    monkeypatch.setattr(
        blog_mod,
        "_build_blueprint",
        lambda *_args, **_kwargs: blog_mod.PostBlueprint(
            topic_type="vendor_deep_dive",
            slug="jira-deep-dive-2026-04",
            suggested_title="Jira Deep Dive",
            tags=["jira"],
            data_context={"vendor": "Jira", "topic_ctx": {"vendor": "Jira"}},
            sections=[],
            charts=[],
        ),
    )
    monkeypatch.setattr(blog_mod, "_fetch_related_for_linking", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        lambda *_args, **_kwargs: SimpleNamespace(
            provider="anthropic",
            model="claude-sonnet-4-5",
            request_envelope={"messages": []},
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text",
        AsyncMock(return_value=None),
    )

    async def _fake_run_batch(**kwargs):
        assert kwargs["llm"] is batch_llm
        assert kwargs["stage_id"] == "b2b_blog_post_generation.content"
        assert len(kwargs["items"]) == 1
        item = kwargs["items"][0]
        assert item.artifact_id == "jira-deep-dive-2026-04"
        return SimpleNamespace(
            local_batch_id="batch-1",
            provider_batch_id="provider-batch-1",
            results_by_custom_id={
                item.custom_id: SimpleNamespace(
                    response_text=json.dumps({
                        "title": "Jira Deep Dive",
                        "description": "Desc",
                        "content": "# Jira\n\nBody",
                    }),
                    usage={"input_tokens": 10, "output_tokens": 20},
                    error_text=None,
                ),
            },
            submitted_items=1,
            cache_prefiltered_items=0,
            fallback_single_call_items=0,
            completed_items=1,
            failed_items=0,
        )

    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        _fake_run_batch,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.mark_batch_fallback_result",
        fallback_mark,
    )
    monkeypatch.setattr(
        blog_mod,
        "_enforce_blog_quality_async",
        AsyncMock(return_value=(
            {
                "title": "Jira Deep Dive",
                "description": "Desc",
                "content": "# Jira\n\nBody",
            },
            {"status": "pass", "_retry_requested": False},
        )),
    )
    monkeypatch.setattr(
        blog_mod,
        "_canonicalize_blog_quality",
        lambda *_args, **_kwargs: {
            "score": 88,
            "threshold": 70,
            "blocking_issues": [],
            "warnings": [],
        },
    )

    async def _fake_assemble_and_store(_pool, _blueprint, content, llm, **_kwargs):
        assembled.append({"llm": llm, "content": content})
        return "post-1"

    monkeypatch.setattr(blog_mod, "_assemble_and_store", _fake_assemble_and_store)

    result = await blog_mod.run(task)

    assert result["count"] == 1
    assert result["blog_batch_jobs"] == 1
    assert result["blog_batch_items_submitted"] == 1
    assert result["blog_batch_completed_items"] == 1
    assert result["blog_batch_failed_items"] == 0
    fallback_mark.assert_not_awaited()
    assert len(assembled) == 1
    assert assembled[0]["llm"] is batch_llm
    assert assembled[0]["content"]["title"] == "Jira Deep Dive"
    notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_allows_maintenance_regeneration_override(monkeypatch):
    task = ScheduledTask(
        id=uuid4(),
        name="b2b_blog_post_generation",
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=300,
        enabled=True,
        metadata={
            "builtin_handler": "b2b_blog_post_generation",
            "maintenance_run": True,
            "regenerate_existing_posts": True,
            "test_vendors": ["Asana"],
            "test_topic_types": ["migration_guide"],
        },
    )

    class FakePool:
        is_initialized = True

    class FakeLLM:
        model = "anthropic/claude-sonnet-4-5"
        name = "openrouter"

    regen = AsyncMock(return_value={"count": 1, "regenerated": True})

    monkeypatch.setattr(blog_mod, "get_db_pool", lambda: FakePool())
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_enabled", False, raising=False)
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_regenerate_mode", False, raising=False)
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: FakeLLM(),
    )
    monkeypatch.setattr(blog_mod, "_regenerate_existing_posts", regen)

    result = await blog_mod.run(task)

    assert result == {"count": 1, "regenerated": True}
    regen.assert_awaited_once()


@pytest.mark.asyncio
async def test_regenerate_existing_posts_applies_scoped_filters_before_limit():
    captured: dict[str, object] = {}

    class Pool:
        async def fetch(self, query, *args):
            captured["query"] = query
            captured["args"] = args
            return []

    task = ScheduledTask(
        id=uuid4(),
        name="b2b_blog_post_generation",
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=300,
        enabled=True,
        metadata={
            "maintenance_run": True,
            "regenerate_existing_posts": True,
            "test_vendors": ["Asana"],
            "test_topic_types": ["migration_guide"],
            "test_slugs": ["switch-to-asana-2026-04"],
        },
    )

    result = await blog_mod._regenerate_existing_posts(
        Pool(),
        llm=object(),
        cfg=SimpleNamespace(),
        task=task,
        max_posts=10,
    )

    assert result == {"_skip_synthesis": "No draft posts to regenerate"}
    assert "LOWER(COALESCE(data_context->>'vendor_name', data_context->>'vendor_a', data_context->>'vendor', '')) = ANY($2::text[])" in str(captured["query"])
    assert "topic_type = ANY($3::text[])" in str(captured["query"])
    assert "slug = ANY($4::text[])" in str(captured["query"])
    assert captured["args"][1] == ["asana"]
    assert captured["args"][2] == ["migration_guide"]
    assert captured["args"][3] == ["switch-to-asana-2026-04"]
    assert captured["args"][4] == 10


def _blog_anchor_context() -> dict:
    return {
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "Hack Club said Zendesk tried to charge $200k/year at Q2 renewal.",
                    "reviewer_company": "Hack Club",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Freshdesk",
                    "pain_category": "pricing",
                },
            ],
        },
        "reasoning_witness_highlights": [
            {
                "witness_id": "witness:r1:0",
                "excerpt_text": "Hack Club said Zendesk tried to charge $200k/year at Q2 renewal.",
                "reviewer_company": "Hack Club",
                "time_anchor": "Q2 renewal",
                "numeric_literals": {"currency_mentions": ["$200k/year"]},
                "competitor": "Freshdesk",
                "pain_category": "pricing",
            },
        ],
        "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
        "reasoning_scope_summary": {
            "selection_strategy": "vendor_facet_packet_v1",
            "witnesses_in_scope": 8,
        },
        "reasoning_atom_context": {
            "theses": [{"thesis_id": "primary_wedge", "summary": "Pricing pressure is driving the story."}],
            "timing_windows": [{"window_id": "trigger_1", "start_or_anchor": "Q2 renewal"}],
            "proof_points": [{"label": "switch_volume"}],
            "account_signals": [{"company": "Hack Club"}],
            "counterevidence": [{"counterevidence_id": "counterevidence_1"}],
            "coverage_limits": [{"coverage_limit_id": "limit_1"}],
        },
        "reasoning_delta_summary": {
            "changed": True,
        },
    }


def _long_blog_body(core_sentence: str) -> str:
    intro = (
        "# Zendesk Alternatives\n\n"
        "This analysis reflects self-selected feedback collected between 2025-06 and 2026-03. "
        "Zendesk remains the focus of the narrative throughout the piece.\n\n"
    )
    repeated = " ".join([core_sentence] * 240)
    return f"{intro}{repeated}"


def _section_by_id(blueprint, section_id: str):
    for section in blueprint.sections:
        if section.id == section_id:
            return section
    raise AssertionError(f"missing section {section_id}")


@pytest.mark.asyncio
async def test_fetch_negative_quotes_only_uses_trusted_account_resolution():
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    await _fetch_negative_quotes(
        pool,
        vendor_name="Zendesk",
        category=None,
        sources=["g2"],
        limit=3,
    )

    sql = pool.fetch.await_args.args[0]
    assert "WHEN ar.confidence_label IN ('high', 'medium')" in sql


@pytest.mark.asyncio
async def test_fetch_positive_quotes_only_uses_trusted_account_resolution():
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    await _fetch_positive_quotes(
        pool,
        vendor_name="Zendesk",
        category=None,
        sources=["g2"],
        limit=3,
    )

    sql = pool.fetch.await_args.args[0]
    assert "WHEN ar.confidence_label IN ('high', 'medium')" in sql


@pytest.mark.asyncio
async def test_load_pool_layers_for_blog_injects_anchor_context_and_claim_plan(monkeypatch):
    class _DummyView:
        primary_wedge = SimpleNamespace(value="price_squeeze")
        wedge_label = "Price Squeeze"
        why_they_stay = {"summary": "Ecosystem breadth keeps some teams in place."}
        confidence_posture = {"overall": "medium"}
        evidence_governance = {"coverage_gaps": []}

        def materialized_contracts(self):
            return {
                "vendor_core_reasoning": {
                    "causal_narrative": {"summary": "Pricing pressure is driving the story."},
                    "timing_intelligence": {"best_timing_window": "Q2 renewal"},
                    "why_they_stay": {
                        "summary": "Ecosystem breadth keeps some teams in place.",
                    },
                },
                "category_reasoning": {"market_regime": "price_competition"},
            }

        def consumer_context(self, consumer):
            assert consumer == "blog_reranker"
            return {
                "anchor_examples": _blog_anchor_context()["reasoning_anchor_examples"],
                "witness_highlights": _blog_anchor_context()["reasoning_witness_highlights"],
                "reference_ids": _blog_anchor_context()["reasoning_reference_ids"],
                "scope_manifest": _blog_anchor_context()["reasoning_scope_summary"],
                "theses": _blog_anchor_context()["reasoning_atom_context"]["theses"],
                "timing_windows": _blog_anchor_context()["reasoning_atom_context"]["timing_windows"],
                "proof_points": _blog_anchor_context()["reasoning_atom_context"]["proof_points"],
                "account_signals": _blog_anchor_context()["reasoning_atom_context"]["account_signals"],
                "counterevidence": _blog_anchor_context()["reasoning_atom_context"]["counterevidence"],
                "coverage_limits": _blog_anchor_context()["reasoning_atom_context"]["coverage_limits"],
                "reasoning_delta": _blog_anchor_context()["reasoning_delta_summary"],
            }

        def filtered_consumer_context(self, consumer):
            context = self.consumer_context(consumer)
            context["reasoning_section_disclaimers"] = {
                "timing_intelligence": "Timing guidance is based on limited direct evidence.",
            }
            return context

    monkeypatch.setattr(
        blog_mod,
        "fetch_all_pool_layers",
        AsyncMock(return_value={"Zendesk": {"displacement": []}}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.load_best_reasoning_views",
        AsyncMock(return_value={"Zendesk": _DummyView()}),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.load_cross_vendor_synthesis_lookup",
        AsyncMock(return_value={"battles": {}, "councils": {}, "asymmetries": {}}),
    )

    data = {"data_context": {}}
    await _load_pool_layers_for_blog(
        object(),
        "vendor_alternative",
        {"vendor": "Zendesk"},
        data,
    )

    anchor = data["reasoning_anchor_examples"]["outlier_or_named_account"][0]
    assert anchor["excerpt_text"].startswith("a customer said Zendesk")
    assert "reviewer_company" not in anchor
    assert data["reasoning_reference_ids"]["witness_ids"] == ["witness:r1:0"]
    assert data["reasoning_section_disclaimers"]["timing_intelligence"]
    assert data["blog_claim_plan"]["primary_thesis"] == "Pricing pressure is driving the story."
    assert data["blog_claim_plan"]["timing_hook"] == "Q2 renewal"
    assert data["data_context"]["reasoning_anchor_examples"]["outlier_or_named_account"][0]["excerpt_text"].startswith(
        "a customer said Zendesk"
    )
    assert data["data_context"]["reasoning_witness_highlights"][0]["excerpt_text"].startswith(
        "a customer said Zendesk"
    )
    assert data["data_context"]["reasoning_reference_ids"]["witness_ids"] == ["witness:r1:0"]
    assert data["data_context"]["reasoning_section_disclaimers"]["timing_intelligence"]
    assert data["data_context"]["blog_claim_plan"]["primary_thesis"] == "Pricing pressure is driving the story."
    assert data["reasoning_scope_summary"]["selection_strategy"] == "vendor_facet_packet_v1"
    assert data["reasoning_atom_context"]["top_theses"][0]["summary"] == "Pricing pressure is driving the story."
    assert data["reasoning_delta_summary"]["changed"] is True
    assert data["data_context"]["reasoning_scope_summary"]["witnesses_in_scope"] == 8
    assert data["data_context"]["reasoning_atom_context"]["timing_windows"][0]["anchor"] == "Q2 renewal"


def test_apply_blog_quality_gate_blocks_generic_copy_when_anchors_available():
    blueprint = blog_mod.PostBlueprint(
        topic_type="vendor_alternative",
        slug="zendesk-alternatives-2026-03",
        suggested_title="Zendesk Alternatives",
        tags=["crm"],
        data_context={
            "vendor": "Zendesk",
            "review_period": "2025-06 to 2026-03",
            **_blog_anchor_context(),
        },
        sections=[],
        charts=[],
    )
    content = {
        "title": "Zendesk Alternatives",
        "content": _long_blog_body(
            "Zendesk faces broad commercial pressure and teams describe general friction across the stack."
        ),
    }

    _, report = blog_mod._apply_blog_quality_gate(blueprint, content)

    assert any(
        "witness_specificity:content does not reference any witness-backed anchor" in issue
        for issue in report["blocking_issues"]
    )


def test_apply_blog_quality_gate_accepts_concrete_anchor_usage():
    blueprint = blog_mod.PostBlueprint(
        topic_type="vendor_alternative",
        slug="zendesk-alternatives-2026-03",
        suggested_title="Zendesk Alternatives",
        tags=["crm"],
        data_context={
            "vendor": "Zendesk",
            "review_period": "2025-06 to 2026-03",
            **_blog_anchor_context(),
        },
        sections=[],
        charts=[],
    )
    content = {
        "title": "Zendesk Alternatives",
        "content": _long_blog_body(
            "Zendesk hits a $200k/year flashpoint around the Q2 renewal window while Freshdesk keeps showing up in the displacement pattern."
        ),
    }

    _, report = blog_mod._apply_blog_quality_gate(blueprint, content)

    assert not any("witness_specificity:" in issue for issue in report["blocking_issues"])


@pytest.mark.asyncio
async def test_assemble_and_store_forwards_run_metadata_to_canonical_row(monkeypatch):
    blueprint = blog_mod.PostBlueprint(
        topic_type="vendor_showdown",
        slug="zendesk-vs-freshdesk-2026-03",
        suggested_title="Zendesk vs Freshdesk",
        tags=["helpdesk"],
        data_context={
            "generation_quality": {
                "score": 88,
                "threshold": 70,
                "blocking_issues": [],
                "warnings": ["methodology_disclaimer_missing_self_selected"],
            },
        },
        sections=[],
        charts=[],
    )
    content = {
        "title": "Zendesk vs Freshdesk",
        "description": "Compare Zendesk and Freshdesk.",
        "content": "Body copy",
    }
    pool = AsyncMock()
    upsert = AsyncMock(return_value="post-123")
    monkeypatch.setattr(blog_mod, "_upsert_blog_post_state", upsert)
    monkeypatch.setattr(blog_mod, "_compute_related_slugs", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_post_ui_path", "", raising=False)

    post_id = await _assemble_and_store(
        pool,
        blueprint,
        content,
        SimpleNamespace(model_name="test-model"),
        run_id="run-42",
        attempt_no=3,
    )

    assert post_id == "post-123"
    assert upsert.await_args.kwargs["run_id"] == "run-42"
    assert upsert.await_args.kwargs["attempt_no"] == 3
    assert upsert.await_args.kwargs["score"] == 88
    assert upsert.await_args.kwargs["threshold"] == 70


def test_blueprint_vendor_alternative_uses_reasoning_contracts_for_market_context():
    blueprint = _blueprint_vendor_alternative(
        {"vendor": "Zendesk", "category": "Helpdesk", "urgency": 7.2, "review_count": 42, "slug": "zendesk-alternatives"},
        {
            "profile": {"strengths": []},
            "signals": [{"pain_category": "pricing", "avg_urgency": 7.1, "signal_count": 12, "feature_gaps": []}],
            "partner": None,
            "pool_displacement": [],
            "pool_temporal": {},
            "pool_segment": {},
            "data_context": {},
            "synthesis_contracts": {
                "vendor_core_reasoning": {
                    "segment_playbook": {
                        "priority_segments": [{"segment": "mid-market finance teams", "estimated_reach": {"value": 18}}],
                    },
                    "timing_intelligence": {
                        "best_timing_window": "Before renewal review",
                        "active_eval_signals": {"value": 11},
                        "immediate_triggers": [{"trigger": "Renewal review", "type": "deadline"}],
                    },
                    "causal_narrative": {"trigger": "Price hike", "why_now": "Budget pressure"},
                },
                "displacement_reasoning": {
                    "migration_proof": {
                        "switching_is_real": True,
                        "evidence_type": "explicit_switch",
                        "switch_volume": {"value": 6},
                    },
                },
                "account_reasoning": {
                    "market_summary": "Three strategic accounts are actively evaluating alternatives.",
                    "top_accounts": [{"name": "Acme Corp"}, {"name": "Globex"}],
                },
                "category_reasoning": {"market_regime": "consolidating", "winner": "Freshdesk"},
            },
            "synthesis_wedge": "price_squeeze",
            "synthesis_wedge_label": "Price Squeeze",
        },
    )

    market_context = _section_by_id(blueprint, "market_context")
    verdict = _section_by_id(blueprint, "verdict")

    assert market_context.key_stats["switching_is_real"] is True
    assert market_context.key_stats["timing_summary"]
    assert market_context.key_stats["segment_targeting_summary"]
    assert market_context.key_stats["account_pressure_summary"] == (
        "Three strategic accounts are actively evaluating alternatives."
    )
    assert market_context.key_stats["category_market_regime"] == "consolidating"
    assert verdict.key_stats["market_regime"] == "consolidating"


def test_blueprint_churn_report_uses_contract_sections_when_pool_slices_are_thin():
    blueprint = _blueprint_churn_report(
        {
            "vendor": "Zendesk",
            "category": "Helpdesk",
            "negative_reviews": 14,
            "total_reviews": 42,
            "avg_urgency": 7.2,
            "slug": "zendesk-churn-report",
        },
        {
            "signals": [{"pain_category": "pricing", "signal_count": 12, "avg_urgency": 7.1, "feature_gaps": []}],
            "profile": {},
            "pool_displacement": [],
            "pool_segment": {},
            "pool_temporal": {},
            "category_overview": {},
            "data_context": {},
            "synthesis_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"trigger": "Price hike", "why_now": "Budget pressure"},
                    "segment_playbook": {
                        "priority_segments": [{"segment": "finance teams", "estimated_reach": {"value": 9}}],
                    },
                    "timing_intelligence": {
                        "best_timing_window": "Before renewal review",
                        "active_eval_signals": {"value": 8},
                        "immediate_triggers": [{"trigger": "Renewal review", "type": "deadline"}],
                    },
                },
                "displacement_reasoning": {
                    "migration_proof": {"switching_is_real": True, "switch_volume": {"value": 4}},
                },
                "account_reasoning": {"market_summary": "Two named accounts are in active evaluation."},
                "category_reasoning": {
                    "market_regime": "fragmented",
                    "narrative": "Buyers are actively re-evaluating vendor fit.",
                },
            },
            "synthesis_wedge": "price_squeeze",
        },
    )

    hook = _section_by_id(blueprint, "hook")
    displacement = _section_by_id(blueprint, "displacement")
    buyer_segments = _section_by_id(blueprint, "buyer_segments")
    timing = _section_by_id(blueprint, "timing")
    outlook = _section_by_id(blueprint, "outlook")

    assert hook.key_stats["market_regime"] == "fragmented"
    assert displacement.key_stats["switching_is_real"] is True
    assert buyer_segments.key_stats["segment_targeting_summary"]
    assert timing.key_stats["timing_summary"]
    assert outlook.key_stats["account_pressure_summary"] == (
        "Two named accounts are in active evaluation."
    )
    assert outlook.key_stats["category_narrative"] == (
        "Buyers are actively re-evaluating vendor fit."
    )


def test_blueprint_migration_guide_uses_inbound_metrics_and_drops_outbound():
    """Migration guide takeaway uses inbound-only metrics. Outbound
    migration_proof (switching_is_real, top_destination) must NOT appear."""
    blueprint = _blueprint_migration_guide(
        {
            "vendor": "Zendesk",
            "category": "Helpdesk",
            "switch_count": 9,
            "review_total": 42,
            "slug": "switch-to-zendesk-2026-03",
        },
        {
            "profile": {
                "commonly_switched_from": [
                    {"vendor": "Freshdesk", "count": 3},
                    {"vendor": "Intercom", "count": 2},
                ],
            },
            "signals": [],
            "pool_displacement": [
                # Inbound edge (to_vendor = Zendesk)
                {"from_vendor": "Freshdesk", "to_vendor": "Zendesk",
                 "edge_metrics": {"mention_count": 10, "primary_driver": "support_quality"}},
                # Outbound edge (from_vendor = Zendesk) -- should be filtered
                {"from_vendor": "Zendesk", "to_vendor": "HubSpot",
                 "edge_metrics": {"mention_count": 5, "primary_driver": "pricing"}},
            ],
            "data_context": {},
            "synthesis_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"trigger": "Price hike"},
                },
                "displacement_reasoning": {
                    "migration_proof": {
                        "switching_is_real": True,
                        "switch_volume": {"value": 4},
                        "top_destination": {"value": "Freshdesk"},
                    },
                },
                "category_reasoning": {"market_regime": "consolidating"},
            },
        },
    )

    takeaway = _section_by_id(blueprint, "takeaway")
    hook = _section_by_id(blueprint, "hook")

    # Inbound metrics present
    assert hook.key_stats["inbound_source_count"] == 2
    assert takeaway.key_stats["inbound_source_count"] == 2
    assert takeaway.key_stats["inbound_migration_mentions"] == 5
    assert takeaway.key_stats["top_inbound_driver"] == "support_quality"
    assert takeaway.key_stats["market_regime"] == "consolidating"

    # Outbound migration_proof must NOT be in takeaway
    assert "switching_is_real" not in takeaway.key_stats
    assert "top_destination" not in takeaway.key_stats
    assert "switch_volume" not in takeaway.key_stats
    assert "dm_churn_rate" not in takeaway.key_stats


def test_blueprint_migration_guide_excludes_destination_vendor_from_sources_chart():
    blueprint = _blueprint_migration_guide(
        {
            "vendor": "Shopify",
            "category": "E-commerce",
            "switch_count": 4,
            "review_total": 1490,
            "slug": "switch-to-shopify-2026-03",
        },
        {
            "profile": {
                "commonly_switched_from": [
                    {"vendor": "WooCommerce", "count": 2},
                    {"vendor": "Shopify", "count": 1},
                    {"vendor": "BigCommerce", "count": 1},
                ],
            },
            "signals": [],
            "pool_displacement": [],
            "data_context": {},
            "synthesis_contracts": {},
        },
    )

    source_chart = next(c for c in blueprint.charts if c.chart_id == "sources-bar")
    source_names = {row["name"] for row in source_chart.data}
    assert "Shopify" not in source_names
    assert "WooCommerce" in source_names
    assert "BigCommerce" in source_names


@pytest.mark.asyncio
async def test_gather_data_vendor_alternative_uses_evidence_vault_overlay(monkeypatch):
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "mention_count_total": 14,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.8},
                "best_quote": "Pricing opacity kept surprising us",
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "company": "Acme"},
            },
        ],
    }
    pool = type("Pool", (), {"fetchrow": AsyncMock(return_value={
        "total_reviews": 25,
        "enriched": 20,
        "churn_intent": 8,
        "earliest": "2026-01-01",
        "latest": "2026-03-18",
    })})()

    monkeypatch.setattr(blog_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={"Zendesk": vault}))
    monkeypatch.setattr(blog_mod, "_fetch_product_profile", AsyncMock(return_value={"strengths": []}))
    monkeypatch.setattr(blog_mod, "_fetch_churn_signals", AsyncMock(return_value=[
        {"pain_category": "pricing", "signal_count": 3, "avg_urgency": 4.0, "feature_gaps": []},
    ]))
    monkeypatch.setattr(blog_mod, "_fetch_quotable_reviews", AsyncMock(return_value=[
        {
            "phrase": "Pricing opacity kept surprising us",
            "vendor": "Zendesk",
            "urgency": 5.0,
            "role": "Ops Manager",
            "company": "Acme",
            "company_size": "Mid-Market",
            "industry": "SaaS",
            "source_name": "g2",
            "sentiment": "negative",
        },
    ]))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_source_distribution", AsyncMock(return_value={"g2": 10}))
    monkeypatch.setattr(blog_mod, "_fetch_category_overview_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner_by_category", AsyncMock(return_value=None))

    data = await _gather_data(
        pool,
        "vendor_alternative",
        {"vendor": "Zendesk", "category": "Helpdesk", "review_count": 42, "urgency": 7.2, "slug": "zendesk-alternatives"},
    )

    assert data["signals"][0]["pain_category"] == "Pricing opacity"
    assert data["signals"][0]["signal_count"] == 14
    assert data["quotes"][0]["phrase"] == "Pricing opacity kept surprising us"
    assert data["quotes"][0]["source_name"] == "reddit"
    assert data["data_context"]["evidence_vault_used"] is True
    assert data["data_context"]["evidence_vault_vendors"] == ["Zendesk"]


@pytest.mark.asyncio
async def test_gather_data_pricing_reality_check_uses_evidence_vault_review_fallback(monkeypatch):
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "mention_count_total": 14,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.8},
                "best_quote": "Pricing kept increasing after the initial contract",
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "rating": 2.0},
            },
        ],
        "strength_evidence": [
            {
                "key": "integrations",
                "label": "Integrations",
                "best_quote": "The integrations still save us a lot of time",
                "quote_source": {"source": "capterra", "reviewer_title": "RevOps", "rating": 4.0},
                "mention_count_total": 6,
            },
        ],
    }
    pool = type(
        "Pool",
        (),
        {
            "fetch": AsyncMock(return_value=[]),
            "fetchrow": AsyncMock(return_value={
                "total_reviews": 25,
                "enriched": 20,
                "churn_intent": 8,
                "earliest": "2026-01-01",
                "latest": "2026-03-18",
            }),
        },
    )()

    monkeypatch.setattr(blog_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={"Zendesk": vault}))
    monkeypatch.setattr(blog_mod, "_fetch_product_profile", AsyncMock(return_value={"strengths": []}))
    monkeypatch.setattr(blog_mod, "_fetch_churn_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_fetch_source_distribution", AsyncMock(return_value={"g2": 10}))
    monkeypatch.setattr(blog_mod, "_fetch_category_overview_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner_by_category", AsyncMock(return_value=None))

    data = await _gather_data(
        pool,
        "pricing_reality_check",
        {"vendor": "Zendesk", "category": "Helpdesk", "pricing_complaints": 8, "total_reviews": 42, "avg_urgency": 7.2, "slug": "zendesk-pricing"},
    )

    assert data["pricing_reviews"][0]["text"] == "Pricing kept increasing after the initial contract"
    assert data["pricing_reviews"][0]["source_name"] == "reddit"
    assert data["positive_reviews"][0]["text"] == "The integrations still save us a lot of time"


@pytest.mark.asyncio
async def test_gather_data_switching_story_uses_evidence_vault_review_fallback(monkeypatch):
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "support",
                "label": "Support",
                "evidence_type": "pain_category",
                "mention_count_total": 10,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.5},
                "best_quote": "We moved away after support stopped responding during renewal",
                "quote_source": {"source": "g2", "reviewer_title": "Director", "rating": 2.0},
            },
        ],
    }
    pool = type(
        "Pool",
        (),
        {
            "fetch": AsyncMock(return_value=[]),
            "fetchrow": AsyncMock(return_value={
                "total_reviews": 25,
                "enriched": 20,
                "churn_intent": 8,
                "earliest": "2026-01-01",
                "latest": "2026-03-18",
            }),
        },
    )()

    monkeypatch.setattr(blog_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={"Zendesk": vault}))
    monkeypatch.setattr(blog_mod, "_fetch_product_profile", AsyncMock(return_value={"strengths": []}))
    monkeypatch.setattr(blog_mod, "_fetch_churn_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_fetch_source_distribution", AsyncMock(return_value={"g2": 10}))
    monkeypatch.setattr(blog_mod, "_fetch_category_overview_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner_by_category", AsyncMock(return_value=None))

    data = await _gather_data(
        pool,
        "switching_story",
        {"from_vendor": "Zendesk", "category": "Helpdesk", "switch_mentions": 6, "total_reviews": 42, "avg_urgency": 7.2, "slug": "why-teams-leave-zendesk"},
    )

    assert data["switch_reviews"][0]["text"] == "We moved away after support stopped responding during renewal"
    assert data["switch_reviews"][0]["source_name"] == "g2"
    assert data["quotes"] == data["switch_reviews"]


def test_candidate_overlaps_gap_pain_ignores_vendor_substring_false_positives():
    ctx = {"vendor": "Pricingly CRM", "category": "CRM"}
    assert _candidate_overlaps_gap_pain("vendor_deep_dive", ctx, {"pricing"}) is False
    assert _candidate_overlaps_gap_pain("pricing_reality_check", ctx, {"pricing"}) is True


@pytest.mark.asyncio
async def test_load_pool_layers_for_blog_category_topic_sets_pool_category_without_vendor_query(monkeypatch):
    monkeypatch.setattr(
        blog_mod,
        "fetch_all_pool_layers",
        AsyncMock(return_value={
            "Zendesk": {
                "evidence_vault": {"product_category": "Helpdesk"},
                "category": {"category": "Helpdesk", "market_regime": {"regime_type": "high_churn"}},
            },
        }),
    )
    pool = type("Pool", (), {"fetch": AsyncMock(return_value=[])})()
    data: dict = {}

    await _load_pool_layers_for_blog(
        pool,
        "market_landscape",
        {"category": "Helpdesk"},
        data,
    )

    assert data["pool_category"]["category"] == "Helpdesk"
    # pool.fetch is now called for category-to-vendor resolution (expected behavior)


@pytest.mark.asyncio
async def test_load_pool_layers_for_blog_scopes_synthesis_query_to_requested_vendor(monkeypatch):
    monkeypatch.setattr(
        blog_mod,
        "fetch_all_pool_layers",
        AsyncMock(return_value={"Zendesk": {"segment": {}, "temporal": {}, "accounts": {}, "category": {}, "displacement": []}}),
    )
    # Track which vendor names are passed to the shared loader
    loaded_vendors: list[list[str]] = []
    original_loader = blog_mod.__dict__.get("_load_pool_layers_for_blog")

    async def _fake_load_views(pool, vendor_names, **kwargs):
        loaded_vendors.append(list(vendor_names))
        return {}

    from atlas_brain.autonomous.tasks import _b2b_synthesis_reader as reader_mod
    monkeypatch.setattr(reader_mod, "load_best_reasoning_views", _fake_load_views)

    pool = type("Pool", (), {"fetch": AsyncMock(return_value=[]), "fetchrow": AsyncMock(return_value=None)})()
    data: dict = {}

    await _load_pool_layers_for_blog(
        pool,
        "vendor_alternative",
        {"vendor": "Zendesk", "category": "Helpdesk"},
        data,
    )

    # Verify the shared loader was called with the scoped vendor name
    assert len(loaded_vendors) == 1
    assert "Zendesk" in loaded_vendors[0]
    assert data.get("synthesis_views") == {}


@pytest.mark.asyncio
async def test_detect_campaign_content_gaps_marks_showdown_drafts_as_coverage():
    class _GapPool:
        async def fetch(self, query, *args):
            normalized = " ".join(str(query).split())
            if "FROM b2b_campaigns" in normalized:
                return [{"vendor": "salesforce", "pain": "pricing"}]
            if "FROM blog_posts" in normalized:
                assert "status = ANY($1::text[])" in normalized
                assert "topic_type = ANY($2::text[])" in normalized
                return [{
                    "title": "Salesforce vs HubSpot",
                    "slug": "salesforce-vs-hubspot-2026-03",
                    "topic_type": "vendor_showdown",
                    "tags": ["crm", "comparison"],
                    "data_context": {
                        "vendor_a": "Salesforce",
                        "vendor_b": "HubSpot",
                        "pain_distribution": "pricing",
                    },
                }]
            raise AssertionError(f"Unexpected query: {normalized}")

    gaps = await _detect_campaign_content_gaps(_GapPool())
    assert gaps == {}


def test_blog_post_covers_showdown_pair_from_topic_ctx():
    post = {
        "topic_type": "vendor_showdown",
        "title": "CrowdStrike vs SentinelOne",
        "slug": "crowdstrike-vs-sentinelone-2026-03",
        "data_context": {
            "topic_ctx": {
                "vendor_a": "CrowdStrike",
                "vendor_b": "SentinelOne",
            },
        },
    }

    assert blog_mod._blog_post_covers_showdown_pair(post, "SentinelOne", "CrowdStrike") is True


@pytest.mark.asyncio
async def test_find_outbound_showdown_gap_candidates_skips_existing_pair(monkeypatch):
    monkeypatch.setattr(
        blog_mod,
        "_fetch_existing_showdown_posts",
        AsyncMock(return_value=[
            {
                "topic_type": "vendor_showdown",
                "title": "CrowdStrike vs SentinelOne",
                "slug": "crowdstrike-vs-sentinelone-2026-03",
                "data_context": {
                    "topic_ctx": {
                        "vendor_a": "CrowdStrike",
                        "vendor_b": "SentinelOne",
                    },
                },
            },
        ]),
    )
    monkeypatch.setattr(
        blog_mod,
        "_fetch_outbound_review_queue_candidates",
        AsyncMock(return_value=[
            {
                "company_name": "Pax8",
                "vendor_name": "CrowdStrike",
                "product_category": "Endpoint Protection",
                "opportunity_score": 67,
                "comparison_asset": {
                    "company_safe": True,
                    "pain_categories": ["support"],
                    "incumbent_vendor": "CrowdStrike",
                    "alternative_vendor": "SentinelOne",
                    "primary_blog_post": {
                        "topic_type": "vendor_deep_dive",
                        "title": "SentinelOne Deep Dive",
                    },
                },
            },
        ]),
    )
    build_candidate = AsyncMock(return_value={"vendor_a": "CrowdStrike", "vendor_b": "SentinelOne"})
    monkeypatch.setattr(blog_mod, "_build_outbound_showdown_candidate", build_candidate)

    result = await blog_mod._find_outbound_showdown_gap_candidates(object())

    assert result == []
    build_candidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_select_topic_prioritizes_outbound_showdown_gap(monkeypatch):
    monkeypatch.setattr(
        blog_mod,
        "_find_vendor_alternative_candidates",
        AsyncMock(return_value=[
            {
                "vendor": "HubSpot",
                "category": "CRM",
                "urgency": 8.4,
                "review_count": 42,
            },
        ]),
    )
    monkeypatch.setattr(blog_mod, "_find_vendor_showdown_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        blog_mod,
        "_find_outbound_showdown_gap_candidates",
        AsyncMock(return_value=[
            {
                "vendor_a": "CrowdStrike",
                "vendor_b": "SentinelOne",
                "category": "Endpoint Protection",
                "reviews_a": 71,
                "reviews_b": 64,
                "total_reviews": 135,
                "urgency_a": 7.9,
                "urgency_b": 6.6,
                "pain_diff": 1.3,
                "outbound_gap_priority": True,
            },
        ]),
    )
    monkeypatch.setattr(blog_mod, "_find_churn_report_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_find_migration_guide_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_find_vendor_deep_dive_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_find_market_landscape_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_find_pricing_reality_check_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_find_switching_story_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_find_pain_point_roundup_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_find_best_fit_guide_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_detect_campaign_content_gaps", AsyncMock(return_value={}))
    monkeypatch.setattr(
        blog_mod,
        "_batch_vendor_review_counts",
        AsyncMock(return_value={
            "hubspot": 42,
            "crowdstrike": 71,
            "sentinelone": 64,
        }),
    )
    monkeypatch.setattr(blog_mod, "_batch_slug_check", AsyncMock(return_value=set()))
    monkeypatch.setattr(
        blog_mod,
        "_recently_covered_vendors",
        AsyncMock(return_value={"crowdstrike", "sentinelone"}),
    )

    topic_type, topic_ctx = await blog_mod._select_topic(object())

    assert topic_type == "vendor_showdown"
    assert topic_ctx["vendor_a"] == "CrowdStrike"
    assert topic_ctx["vendor_b"] == "SentinelOne"


# ---------------------------------------------------------------------------
# Phase 7: Reasoning-aware topic reranking
# ---------------------------------------------------------------------------

from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
    _rerank_topic_candidates_with_reasoning,
)


def _make_mock_pool_with_views(views_data: dict):
    """Create a mock pool that returns synthesis views for given vendors."""
    import json as _json
    from unittest.mock import AsyncMock as _AM

    synth_rows = []
    legacy_rows = []
    for vname, raw in views_data.items():
        synth_rows.append({
            "vendor_name": vname,
            "as_of_date": "2026-03-28",
            "schema_version": "v2",
            "synthesis": raw,
        })

    async def _fetch(query, *args):
        if "b2b_reasoning_synthesis" in query:
            return synth_rows
        if "b2b_churn_signals" in query:
            return legacy_rows
        return []

    pool = _AM()
    pool.fetch = _fetch
    pool.fetchrow = _AM(return_value=None)
    return pool


def _synth_with_timing_and_accounts():
    return {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "high",
                },
                "timing_intelligence": {
                    "best_timing_window": "Q2 renewal cycle",
                    "immediate_triggers": [
                        {"type": "deadline", "trigger": "Q2 renewal"},
                    ],
                    "confidence": "medium",
                },
                "why_they_stay": {
                    "summary": "Ecosystem lock-in",
                    "strengths": [{"area": "integrations"}],
                },
                "confidence_posture": {
                    "overall": "medium",
                    "limits": [],
                },
            },
            "displacement_reasoning": {
                "switch_triggers": [
                    {"type": "deadline", "description": "Q2 renewal"},
                ],
            },
            "account_reasoning": {
                "market_summary": "Active evaluation in mid-market",
                "high_intent_count": {"value": 5, "source_id": "accounts:summary:high_intent_count"},
            },
        },
    }


def _synth_with_coverage_gaps():
    return {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "feature_parity",
                    "confidence": "low",
                },
            },
            "evidence_governance": {
                "coverage_gaps": [
                    {"type": "missing_pool", "area": "displacement", "_sid": "gap:missing_pool:displacement"},
                    {"type": "thin_account_signals", "area": "accounts", "_sid": "gap:thin_accounts:accounts"},
                ],
                "contradictions": [
                    {"dimension": "support", "_sid": "segment:contradiction:support"},
                ],
            },
        },
    }


@pytest.mark.asyncio
async def test_reranker_boosts_reasoning_backed_topic():
    """A topic with timing + account intent should outrank one without."""
    pool = _make_mock_pool_with_views({
        "VendorA": _synth_with_timing_and_accounts(),
    })
    candidates = [
        ("slug-a", 50.0, "vendor_alternative", {"vendor": "VendorA"}),
        ("slug-b", 55.0, "vendor_alternative", {"vendor": "VendorB"}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)

    # VendorA started lower (50) but gets timing + account + trigger + retention boosts
    assert result[0][0] == "slug-a"
    assert result[0][1] > 55.0  # boosted above VendorB's score
    adjustments = result[0][3].get("_reasoning_adjustments", [])
    assert "timing_boost" in adjustments
    assert "account_intent_boost" in adjustments
    assert "switch_trigger_boost" in adjustments
    assert "retention_context" in adjustments


@pytest.mark.asyncio
async def test_reranker_penalizes_thin_evidence():
    """A topic with coverage gaps should be deprioritized."""
    pool = _make_mock_pool_with_views({
        "ThinVendor": _synth_with_coverage_gaps(),
    })
    candidates = [
        ("slug-thin", 80.0, "churn_report", {"vendor": "ThinVendor"}),
        ("slug-ok", 70.0, "vendor_alternative", {"vendor": "UnknownVendor"}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)

    # ThinVendor started higher (80) but gets coverage_gap + contradiction penalties
    thin_entry = [c for c in result if c[0] == "slug-thin"][0]
    assert thin_entry[1] < 80.0
    adjustments = thin_entry[3].get("_reasoning_adjustments", [])
    assert any("coverage_gap" in a for a in adjustments)
    assert "contradiction_penalty" in adjustments


@pytest.mark.asyncio
async def test_reranker_handles_no_views_gracefully():
    """When no synthesis exists, candidates returned unchanged."""
    pool = _make_mock_pool_with_views({})
    candidates = [
        ("slug-a", 50.0, "vendor_alternative", {"vendor": "VendorA"}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)
    assert result == candidates


@pytest.mark.asyncio
async def test_reranker_handles_pool_error_gracefully():
    """When pool query fails, candidates returned unchanged."""
    from unittest.mock import AsyncMock as _AM

    pool = _AM()
    pool.fetch = _AM(side_effect=Exception("db error"))
    candidates = [
        ("slug-a", 50.0, "vendor_alternative", {"vendor": "VendorA"}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)
    assert result == candidates


@pytest.mark.asyncio
async def test_reranker_showdown_checks_both_vendors():
    """For showdowns, vendor_b coverage gaps should also penalize."""
    pool = _make_mock_pool_with_views({
        "VendorA": _synth_with_timing_and_accounts(),
        "VendorB": _synth_with_coverage_gaps(),
    })
    candidates = [
        ("slug-showdown", 90.0, "vendor_showdown", {"vendor_a": "VendorA", "vendor_b": "VendorB"}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)

    adjustments = result[0][3].get("_reasoning_adjustments", [])
    # VendorA boosts present
    assert "timing_boost" in adjustments
    # VendorB gap penalty present
    assert any("vendor_b_gap" in a for a in adjustments)


@pytest.mark.asyncio
async def test_reranker_category_only_topic_gets_category_reasoning():
    """Category-only topics (market_landscape) should get category-level reasoning."""
    synth_with_category = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "medium"},
            },
            "category_reasoning": {
                "market_regime": "consolidating",
                "category": "CRM",
                "confidence_score": 0.7,
            },
            "evidence_governance": {
                "coverage_gaps": [],
            },
        },
    }
    pool = _make_mock_pool_with_views({"SomeVendor": synth_with_category})
    candidates = [
        # Category-only topic (no vendor key)
        ("slug-landscape", 60.0, "market_landscape", {"category": "CRM", "vendor_count": 5}),
        # Vendor topic for comparison
        ("slug-alt", 55.0, "vendor_alternative", {"vendor": "SomeVendor"}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)

    landscape = [c for c in result if c[0] == "slug-landscape"][0]
    adjustments = landscape[3].get("_reasoning_adjustments", [])
    assert "category_regime_boost" in adjustments


@pytest.mark.asyncio
async def test_reranker_category_low_confidence_penalized():
    """Category topics with low confidence should be penalized when views
    are loaded from vendor-named candidates in the same batch."""
    synth_low = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "low"},
            },
            "category_reasoning": {
                "market_regime": "fragmented",
                "category": "Niche Tool",
                "confidence_score": 0.1,
            },
            "evidence_governance": {
                "coverage_gaps": [
                    {"type": "thin_account_signals", "area": "accounts", "_sid": "gap:x"},
                ],
            },
        },
    }
    pool = _make_mock_pool_with_views({"NicheVendor": synth_low})
    candidates = [
        # Category-only topic
        ("slug-niche", 70.0, "market_landscape", {"category": "Niche Tool"}),
        # Vendor topic that triggers view loading
        ("slug-vendor", 50.0, "vendor_alternative", {"vendor": "NicheVendor"}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)

    landscape = [c for c in result if c[0] == "slug-niche"][0]
    adjustments = landscape[3].get("_reasoning_adjustments", [])
    assert "category_low_confidence_penalty" in adjustments
    assert landscape[1] < 70.0


@pytest.mark.asyncio
async def test_reranker_pure_category_batch_resolves_vendors():
    """When ALL candidates are category-only (no vendor keys), the reranker
    should resolve categories to vendors via DB query and still apply
    category-level reasoning."""
    from unittest.mock import AsyncMock as _AM

    synth_with_category = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {"primary_wedge": "price_squeeze", "confidence": "medium"},
            },
            "category_reasoning": {
                "market_regime": "consolidating",
                "confidence_score": 0.7,
            },
        },
    }

    # Pool that returns vendor-name resolution for "crm" category
    # and synthesis views for "CRMVendor"
    synth_rows = [{
        "vendor_name": "CRMVendor",
        "as_of_date": "2026-03-28",
        "schema_version": "v2",
        "synthesis": synth_with_category,
    }]
    cat_vendor_rows = [{"vendor_name": "CRMVendor", "product_category": "CRM"}]

    async def _fetch(query, *args):
        q = str(query)
        if "b2b_reasoning_synthesis" in q and "DISTINCT vendor_name" not in q:
            return synth_rows
        if "b2b_product_profiles" in q and "product_category" in q:
            return cat_vendor_rows
        if "b2b_churn_signals" in q:
            return []
        return []

    pool = _AM()
    pool.fetch = _fetch
    pool.fetchrow = _AM(return_value=None)

    candidates = [
        # Pure category-only: no vendor key
        ("slug-landscape", 60.0, "market_landscape", {"category": "CRM", "vendor_count": 5}),
    ]
    result = await _rerank_topic_candidates_with_reasoning(pool, candidates)

    adjustments = result[0][3].get("_reasoning_adjustments", [])
    assert "category_regime_boost" in adjustments


# ---------------------------------------------------------------------------
# Cross-vendor synthesis resolution tests
# ---------------------------------------------------------------------------

from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
    _resolve_blog_battle_summary,
    _resolve_blog_council_summary,
)


class TestResolveBlogBattleSummary:
    _SYNTH_BATTLE = {
        "conclusion": {
            "conclusion": "Zendesk losing to Freshdesk on pricing",
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "confidence": 0.85,
            "durability_assessment": "structural",
            "key_insights": [{"insight": "Price gap", "evidence": "0.28 vs 0.24"}],
        },
    }
    _FALLBACK = {
        "conclusion": "Old pool battle conclusion",
        "winner": "Freshdesk",
    }

    def test_prefers_synthesis_when_present(self):
        xv = {"battles": {("freshdesk", "zendesk"): self._SYNTH_BATTLE}}
        result = _resolve_blog_battle_summary("Zendesk", "Freshdesk", xv, self._FALLBACK)
        assert result["source"] == "synthesis"
        assert result["winner"] == "Freshdesk"
        assert "pricing" in result["conclusion"]

    def test_matches_original_case_keys(self):
        xv = {"battles": {("Freshdesk", "Zendesk"): self._SYNTH_BATTLE}}
        result = _resolve_blog_battle_summary("zendesk", "freshdesk", xv, self._FALLBACK)
        assert result["source"] == "synthesis"

    def test_falls_back_when_no_synthesis(self):
        xv = {"battles": {}}
        result = _resolve_blog_battle_summary("Zendesk", "Freshdesk", xv, self._FALLBACK)
        assert result.get("source") != "synthesis"
        assert result["conclusion"] == "Old pool battle conclusion"

    def test_falls_back_on_empty_synthesis_conclusion(self):
        xv = {"battles": {("freshdesk", "zendesk"): {"conclusion": {"conclusion": ""}}}}
        result = _resolve_blog_battle_summary("Zendesk", "Freshdesk", xv, self._FALLBACK)
        assert result["conclusion"] == "Old pool battle conclusion"

    def test_empty_xv_lookup_returns_fallback(self):
        result = _resolve_blog_battle_summary("A", "B", {}, {"conclusion": "fb"})
        assert result["conclusion"] == "fb"

    def test_output_shape(self):
        xv = {"battles": {("freshdesk", "zendesk"): self._SYNTH_BATTLE}}
        result = _resolve_blog_battle_summary("Zendesk", "Freshdesk", xv, {})
        for key in ("conclusion", "winner", "loser", "confidence", "durability_assessment", "key_insights"):
            assert key in result


class TestResolveBlogCouncilSummary:
    _SYNTH_COUNCIL = {
        "conclusion": {
            "conclusion": "Price war in Helpdesk category",
            "market_regime": "price_competition",
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "confidence": 0.7,
            "durability_assessment": "structural",
            "key_insights": [],
        },
    }
    _FALLBACK = {
        "conclusion": "Pool council fallback",
        "market_regime": "stable",
    }

    def test_prefers_synthesis_when_present(self):
        xv = {"councils": {"Helpdesk": self._SYNTH_COUNCIL}}
        result = _resolve_blog_council_summary("Helpdesk", xv, self._FALLBACK)
        assert result["source"] == "synthesis"
        assert result["market_regime"] == "price_competition"

    def test_case_insensitive_match(self):
        xv = {"councils": {"helpdesk": self._SYNTH_COUNCIL}}
        result = _resolve_blog_council_summary("Helpdesk", xv, self._FALLBACK)
        assert result["source"] == "synthesis"

    def test_falls_back_when_no_synthesis(self):
        xv = {"councils": {}}
        result = _resolve_blog_council_summary("Helpdesk", xv, self._FALLBACK)
        assert result.get("source") != "synthesis"
        assert result["conclusion"] == "Pool council fallback"

    def test_falls_back_on_empty_synthesis(self):
        xv = {"councils": {"Helpdesk": {"conclusion": {"conclusion": "", "market_regime": ""}}}}
        result = _resolve_blog_council_summary("Helpdesk", xv, self._FALLBACK)
        assert result["conclusion"] == "Pool council fallback"

    def test_output_shape(self):
        xv = {"councils": {"Helpdesk": self._SYNTH_COUNCIL}}
        result = _resolve_blog_council_summary("Helpdesk", xv, {})
        for key in ("conclusion", "market_regime", "winner", "confidence"):
            assert key in result
