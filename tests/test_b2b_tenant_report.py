import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
    _canonicalize_vendor,
    _compact_vendor_churn_scores_for_llm,
)
from atlas_brain.autonomous.tasks.b2b_tenant_report import (
    _check_freshness,
    _apply_tenant_report_context,
    _apply_tenant_synthesis_context,
    _build_deterministic_tenant_report,
    _build_deterministic_tenant_report_from_raw,
    _tenant_displacement_backfill_row,
    _apply_tenant_vendor_context,
    _filter_tenant_payload_for_vendors,
    _merge_tenant_chunk_outputs,
    _run_chunked_tenant_synthesis,
    _run_tenant_synthesis_llm,
    _tenant_payload_vendor_chunks,
    _tenant_report_chunk_size,
    _tenant_report_data_density,
    _tenant_report_llm_model,
    run,
)


def test_tenant_report_llm_model_uses_actual_model():
    assert _tenant_report_llm_model({"model": "openai/gpt-oss-120b"}) == "openai/gpt-oss-120b"


def test_tenant_report_llm_model_defaults_when_missing():
    assert _tenant_report_llm_model({}) == "pipeline_deterministic"
    assert _tenant_report_llm_model(None) == "pipeline_deterministic"


def test_tenant_displacement_backfill_uses_overall_dissatisfaction_fallback():
    row = _tenant_displacement_backfill_row(
        {"vendor": "Salesforce", "competitor": "HubSpot", "mention_count": 2},
        reason_lookup={},
    )
    assert row["primary_driver"] == "overall_dissatisfaction"


def test_tenant_report_data_density_includes_llm_and_reasoning_counts():
    result = {
        "vendors_analyzed": 12,
        "high_intent_companies": 3,
        "competitive_flows": 9,
        "pain_categories": 4,
        "feature_gaps": 7,
    }
    density = json.loads(
        _tenant_report_data_density(
            result,
            llm_usage={"input_tokens": 123, "output_tokens": 45},
            narrative_evidence_count=5,
            stratified_reasoning_count=8,
            synthesis_contract_vendor_count=4,
        )
    )
    assert density["vendors_analyzed"] == 12
    assert density["narrative_evidence_vendors"] == 5
    assert density["stratified_reasoning_vendors"] == 8
    assert density["synthesis_contract_vendors"] == 4
    assert density["llm_input_tokens"] == 123
    assert density["llm_output_tokens"] == 45


def test_tenant_report_data_density_includes_batch_metrics():
    density = json.loads(
        _tenant_report_data_density(
            {
                "vendors_analyzed": 2,
                "high_intent_companies": 1,
                "competitive_flows": 1,
            },
            llm_usage={
                "batch_jobs": 1,
                "batch_items_submitted": 3,
                "batch_cache_prefiltered_items": 1,
                "batch_fallback_single_call_items": 1,
                "batch_completed_items": 2,
                "batch_failed_items": 1,
            },
        )
    )
    assert density["llm_batch_jobs"] == 1
    assert density["llm_batch_items_submitted"] == 3
    assert density["llm_batch_cache_prefiltered_items"] == 1
    assert density["llm_batch_fallback_single_call_items"] == 1
    assert density["llm_batch_completed_items"] == 2
    assert density["llm_batch_failed_items"] == 1


@pytest.mark.asyncio
async def test_check_freshness_requires_complete_core_marker(monkeypatch):
    pool = SimpleNamespace()
    marker = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.has_complete_core_run_marker",
        marker,
    )

    assert await _check_freshness(pool) is None
    marker.assert_awaited_once_with(pool, date.today())


@pytest.mark.asyncio
async def test_run_skips_when_core_signals_not_fresh(monkeypatch):
    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock())
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.get_db_pool",
        lambda: pool,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.intelligence_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.has_complete_core_run_marker",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.describe_core_run_gap",
        AsyncMock(return_value="Core churn materialization is incomplete for today"),
    )

    result = await run(SimpleNamespace(id="task-id", metadata={}))

    assert result == {"_skip_synthesis": "Core churn materialization is incomplete for today"}
    assert pool.fetch.await_count == 0


@pytest.mark.asyncio
async def test_run_tenant_synthesis_llm_uses_exact_cache_hit(monkeypatch):
    cached_response = {
        "response_text": json.dumps({"weekly_churn_feed": [{"vendor": "Zendesk"}]}),
        "model": "anthropic/claude-sonnet-4-5",
        "provider": "openrouter",
    }

    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **_kwargs: SimpleNamespace(name="openrouter", model="anthropic/claude-sonnet-4-5"),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.llm_exact_cache.build_skill_request_envelope",
        lambda **_kwargs: ("cache-key", {"messages": [{"role": "user", "content": "cached"}]}, []),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.llm_exact_cache.lookup_cached_text",
        AsyncMock(return_value=cached_response),
    )

    def _unexpected_llm_call(*_args, **_kwargs):
        raise AssertionError("tenant report LLM should not run on exact-cache hit")

    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.call_llm_with_skill",
        _unexpected_llm_call,
    )

    parsed, usage = await _run_tenant_synthesis_llm(
        {"vendor_churn_scores": [{"vendor_name": "Zendesk"}]},
        max_tokens=512,
    )

    assert parsed["weekly_churn_feed"] == [{"vendor": "Zendesk"}]
    assert usage["cache_hit"] is True
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["model"] == "anthropic/claude-sonnet-4-5"
    assert usage["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_run_chunked_tenant_synthesis_uses_anthropic_batch(monkeypatch):
    class FakeAnthropicLLM:
        model = "claude-sonnet-4-5"
        name = "anthropic"

    batch_llm = FakeAnthropicLLM()
    store_cache = AsyncMock()
    fallback_mark = AsyncMock()

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.anthropic_batch_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.tenant_report_anthropic_batch_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.tenant_report_anthropic_batch_min_items",
        1,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm.anthropic.AnthropicLLM",
        FakeAnthropicLLM,
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda *, workload, **_kwargs: batch_llm if workload == "anthropic" else None,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._tenant_payload_vendor_chunks",
        lambda _payload: [["Zendesk"], ["HubSpot"]],
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._prepare_tenant_synthesis_request",
        lambda payload, *, max_tokens, llm=None, provider=None, model=None: (
            SimpleNamespace(
                provider=provider or getattr(llm, "name", "anthropic"),
                model=model or getattr(llm, "model", "claude-sonnet-4-5"),
                request_envelope={"vendor": payload["vendor_churn_scores"][0]["vendor_name"]},
            ),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"), default=str)},
            ],
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text",
        store_cache,
    )

    async def _fake_run_batch(**kwargs):
        assert kwargs["llm"] is batch_llm
        assert kwargs["stage_id"] == "b2b_tenant_report.synthesis_chunk"
        assert len(kwargs["items"]) == 2
        results = {}
        for item in kwargs["items"]:
            vendor = json.loads(item.messages[1]["content"])["vendor_churn_scores"][0]["vendor_name"]
            results[item.custom_id] = SimpleNamespace(
                response_text=json.dumps(
                    {
                        "executive_summary": f"{vendor} summary",
                        "weekly_churn_feed": [{"vendor": vendor, "churn_pressure_score": 40.0, "avg_urgency": 6.0}],
                        "vendor_scorecards": [{"vendor": vendor, "churn_pressure_score": 40.0}],
                        "displacement_map": [],
                        "category_insights": [],
                        "timeline_hot_list": [],
                    }
                ),
                usage={"input_tokens": 100, "output_tokens": 25},
                cached=False,
                error_text=None,
            )
        return SimpleNamespace(
            local_batch_id="batch-1",
            provider_batch_id="provider-batch-1",
            results_by_custom_id=results,
            submitted_items=2,
            cache_prefiltered_items=0,
            fallback_single_call_items=0,
            completed_items=2,
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

    payload = {
        "data_context": {
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "category": "Helpdesk"},
            {"vendor_name": "HubSpot", "category": "CRM"},
        ],
        "competitive_displacement": [],
        "competitor_reasons": [],
    }

    merged, usage = await _run_chunked_tenant_synthesis(
        payload,
        max_tokens=512,
        data_context=payload["data_context"],
        run_id="run-1",
        account_id="acct-1",
        pool=SimpleNamespace(),
    )

    assert [row["vendor"] for row in merged["weekly_churn_feed"]] == ["Zendesk", "HubSpot"]
    assert usage["chunk_count"] == 2
    assert usage["chunk_failures"] == 0
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 50
    assert usage["batch_jobs"] == 1
    assert usage["batch_items_submitted"] == 2
    assert usage["batch_completed_items"] == 2
    assert usage["batch_failed_items"] == 0
    assert store_cache.await_count == 2
    fallback_mark.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_chunked_tenant_synthesis_marks_fallback_with_usage(monkeypatch):
    fallback_mark = AsyncMock()
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.anthropic_batch_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.tenant_report_anthropic_batch_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.tenant_report_anthropic_batch_min_items",
        1,
        raising=False,
    )

    class FakeAnthropicLLM:
        model = "claude-sonnet-4-5"
        name = "anthropic"

    monkeypatch.setattr(
        "atlas_brain.services.llm.anthropic.AnthropicLLM",
        FakeAnthropicLLM,
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda *, workload, **_kwargs: FakeAnthropicLLM() if workload == "anthropic" else object(),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.llm_exact_cache.llm_identity",
        lambda llm: (getattr(llm, "name", ""), getattr(llm, "model", "")),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        lambda _stage, payload, *, max_tokens, provider=None, model=None: (
            SimpleNamespace(
                provider=provider or "anthropic",
                model=model or "claude-sonnet-4-5",
                request_envelope={"vendor": payload["vendor_churn_scores"][0]["vendor_name"]},
            ),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"), default=str)},
            ],
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._tenant_payload_vendor_chunks",
        lambda _payload: [["Zendesk"], ["HubSpot"]],
    )

    async def _fake_run_batch(**_kwargs):
        return SimpleNamespace(
            local_batch_id="batch-tenant-fallback",
            provider_batch_id="provider-batch-fallback",
            results_by_custom_id={
                "tenant_chunk:0": SimpleNamespace(
                    response_text='{"broken_json"',
                    usage={"input_tokens": 100, "output_tokens": 25},
                    cached=False,
                    error_text=None,
                ),
                "tenant_chunk:1": SimpleNamespace(
                    response_text='{"broken_json"',
                    usage={"input_tokens": 100, "output_tokens": 25},
                    cached=False,
                    error_text=None,
                ),
            },
            submitted_items=2,
            cache_prefiltered_items=0,
            fallback_single_call_items=2,
            completed_items=0,
            failed_items=2,
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
        "atlas_brain.autonomous.tasks.b2b_tenant_report._run_tenant_synthesis_llm",
        AsyncMock(
            return_value=(
                {
                    "executive_summary": "Fallback summary",
                    "weekly_churn_feed": [{"vendor": "Zendesk", "churn_pressure_score": 40.0, "avg_urgency": 6.0}],
                    "vendor_scorecards": [{"vendor": "Zendesk", "churn_pressure_score": 40.0}],
                    "displacement_map": [],
                    "category_insights": [],
                    "timeline_hot_list": [],
                },
                {
                    "input_tokens": 77,
                    "output_tokens": 33,
                    "billable_input_tokens": 55,
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                },
            )
        ),
    )

    payload = {
        "data_context": {
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "category": "Helpdesk"},
            {"vendor_name": "HubSpot", "category": "CRM"},
        ],
        "competitive_displacement": [],
        "competitor_reasons": [],
    }

    merged, usage = await _run_chunked_tenant_synthesis(
        payload,
        max_tokens=512,
        data_context=payload["data_context"],
        run_id="run-tenant-fallback",
        account_id="acct-1",
        pool=SimpleNamespace(),
    )

    assert merged["weekly_churn_feed"]
    assert usage["chunk_failures"] == 0
    assert usage["batch_items_submitted"] == 2
    assert fallback_mark.await_count == 2
    kwargs = fallback_mark.await_args_list[0].kwargs
    assert kwargs["succeeded"] is True
    assert kwargs["usage"]["input_tokens"] == 77
    assert kwargs["usage"]["output_tokens"] == 33
    assert kwargs["provider"] == "anthropic"
    assert kwargs["model"] == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_run_chunked_tenant_synthesis_task_metadata_can_enable_batch(monkeypatch):
    class FakeAnthropicLLM:
        model = "claude-sonnet-4-5"
        name = "anthropic"

    batch_llm = FakeAnthropicLLM()

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.anthropic_batch_enabled",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.b2b_churn.tenant_report_anthropic_batch_enabled",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm.anthropic.AnthropicLLM",
        FakeAnthropicLLM,
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda *, workload, **_kwargs: batch_llm if workload == "anthropic" else None,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._tenant_payload_vendor_chunks",
        lambda _payload: [["Zendesk"], ["HubSpot"]],
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._prepare_tenant_synthesis_request",
        lambda payload, *, max_tokens, llm=None, provider=None, model=None: (
            SimpleNamespace(
                provider=provider or getattr(llm, "name", "anthropic"),
                model=model or getattr(llm, "model", "claude-sonnet-4-5"),
                request_envelope={"vendor": payload["vendor_churn_scores"][0]["vendor_name"]},
            ),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": json.dumps(payload, separators=(",", ":"), default=str)},
            ],
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text",
        AsyncMock(),
    )

    async def _fake_run_batch(**kwargs):
        assert kwargs["llm"] is batch_llm
        assert kwargs["min_batch_size"] == 1
        return SimpleNamespace(
            local_batch_id="batch-tenant-metadata",
            provider_batch_id="provider-tenant-metadata",
            results_by_custom_id={
                "tenant_report:0": SimpleNamespace(
                    response_text=json.dumps(
                        {
                            "executive_summary": "Zendesk summary",
                            "weekly_churn_feed": [{"vendor": "Zendesk"}],
                            "vendor_scorecards": [{"vendor": "Zendesk"}],
                            "displacement_map": [],
                            "category_insights": [],
                            "timeline_hot_list": [],
                        }
                    ),
                    usage={"input_tokens": 10, "output_tokens": 5},
                    cached=False,
                    error_text=None,
                ),
                "tenant_report:1": SimpleNamespace(
                    response_text=json.dumps(
                        {
                            "executive_summary": "HubSpot summary",
                            "weekly_churn_feed": [{"vendor": "HubSpot"}],
                            "vendor_scorecards": [{"vendor": "HubSpot"}],
                            "displacement_map": [],
                            "category_insights": [],
                            "timeline_hot_list": [],
                        }
                    ),
                    usage={"input_tokens": 10, "output_tokens": 5},
                    cached=False,
                    error_text=None,
                ),
            },
            submitted_items=2,
            cache_prefiltered_items=0,
            fallback_single_call_items=0,
            completed_items=2,
            failed_items=0,
        )

    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        _fake_run_batch,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.mark_batch_fallback_result",
        AsyncMock(),
    )

    payload = {
        "data_context": {
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "category": "Helpdesk"},
            {"vendor_name": "HubSpot", "category": "CRM"},
        ],
        "competitive_displacement": [],
        "competitor_reasons": [],
    }

    merged, usage = await _run_chunked_tenant_synthesis(
        payload,
        max_tokens=512,
        data_context=payload["data_context"],
        task=SimpleNamespace(
            metadata={
                "anthropic_batch_enabled": True,
                "tenant_report_anthropic_batch_enabled": True,
                "tenant_report_anthropic_batch_min_items": 1,
            }
        ),
        run_id="run-tenant-metadata",
        account_id="acct-tenant-metadata",
        pool=SimpleNamespace(),
    )

    assert [row["vendor"] for row in merged["weekly_churn_feed"]] == ["Zendesk", "HubSpot"]
    assert usage["batch_jobs"] == 1


def test_tenant_payload_vendor_chunks_groups_categories_before_splitting(monkeypatch):
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._tenant_report_chunk_size",
        lambda: 4,
    )
    payload = {
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "product_category": "Helpdesk"},
            {"vendor_name": "Intercom", "product_category": "Helpdesk"},
            {"vendor_name": "ClickUp", "product_category": "Project Management"},
            {"vendor_name": "Asana", "product_category": "Project Management"},
            {"vendor_name": "HubSpot", "product_category": "CRM"},
        ]
    }
    chunks = _tenant_payload_vendor_chunks(payload)
    assert chunks == [["Zendesk", "Intercom", "ClickUp", "Asana"], ["HubSpot"]]


def test_tenant_report_chunk_size_ignores_deprecated_gpt_oss_override(monkeypatch):
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.llm.openrouter_reasoning_model",
        "openai/gpt-oss-120b",
    )
    assert _tenant_report_chunk_size() == 6


def test_filter_tenant_payload_for_vendors_scopes_vendor_lists():
    payload = {
        "date": "2026-03-21",
        "data_context": {"x": 1},
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "category_council": {"winner": "Zoho Desk"}},
            {"vendor_name": "HubSpot"},
        ],
        "competitive_displacement": [
            {"from_vendor": "Zendesk", "to_vendor": "Freshdesk"},
            {"from_vendor": "HubSpot", "to_vendor": "Salesforce"},
        ],
        "high_intent_companies": [
            {"vendor": "Zendesk", "company": "Acme"},
            {"vendor": "HubSpot", "company": "Beta"},
        ],
        "prior_reports": [{"report_type": "weekly_churn_feed"}],
    }
    filtered = _filter_tenant_payload_for_vendors(payload, ["Zendesk"])
    assert filtered["date"] == "2026-03-21"
    assert filtered["data_context"] == {"x": 1}
    assert filtered["prior_reports"] == [
        {"type": "weekly_churn_feed", "date": "", "data": []},
    ]
    assert filtered["vendor_churn_scores"] == [
        {"vendor_name": "Zendesk", "category_council": {"winner": "Zoho Desk"}},
    ]
    assert filtered["high_intent_companies"] == [{"vendor": "Zendesk", "company": "Acme"}]
    assert filtered["competitive_displacement"] == [
        {"from_vendor": "Zendesk", "to_vendor": "Freshdesk"},
    ]


def test_filter_tenant_payload_for_vendors_compacts_prior_reports():
    payload = {
        "vendor_churn_scores": [{"vendor_name": "Zendesk"}],
        "prior_reports": [
            {
                "type": "vendor_scorecard",
                "date": "2026-03-14",
                "data": [
                    {"vendor": "Zendesk", "score": 42},
                    {"vendor": "HubSpot", "score": 30},
                ],
            }
        ],
    }
    filtered = _filter_tenant_payload_for_vendors(payload, ["Zendesk"])
    assert filtered["prior_reports"] == [
        {
            "type": "vendor_scorecard",
            "date": "2026-03-14",
            "data": [{"vendor": "Zendesk", "score": 42}],
        }
    ]


def test_merge_tenant_chunk_outputs_dedupes_and_builds_summary():
    partials = [
        {
            "executive_summary": "chunk 1",
            "weekly_churn_feed": [
                {"vendor": "Zendesk", "churn_pressure_score": 42.0, "avg_urgency": 6.0},
            ],
            "vendor_scorecards": [
                {"vendor": "Zendesk", "churn_pressure_score": 42.0},
            ],
            "displacement_map": [
                {"from_vendor": "Zendesk", "to_vendor": "Freshdesk", "mention_count": 9},
            ],
            "category_insights": [
                {"category": "Helpdesk", "highest_churn_risk": "Zendesk"},
            ],
        },
        {
            "executive_summary": "chunk 2",
            "weekly_churn_feed": [
                {"vendor": "HubSpot", "churn_pressure_score": 30.0, "avg_urgency": 5.0},
                {"vendor": "Zendesk", "churn_pressure_score": 42.0, "avg_urgency": 6.0},
            ],
            "vendor_scorecards": [
                {"vendor": "HubSpot", "churn_pressure_score": 30.0},
            ],
            "displacement_map": [
                {"from_vendor": "HubSpot", "to_vendor": "Salesforce", "mention_count": 4},
            ],
            "category_insights": [
                {"category": "CRM", "highest_churn_risk": "HubSpot"},
            ],
        },
    ]
    merged = _merge_tenant_chunk_outputs(
        partials,
        data_context={
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
    )
    assert [row["vendor"] for row in merged["weekly_churn_feed"]] == ["Zendesk", "HubSpot"]
    assert [row["vendor"] for row in merged["vendor_scorecards"]] == ["Zendesk", "HubSpot"]
    assert len(merged["displacement_map"]) == 2
    assert len(merged["category_insights"]) == 2
    assert merged["executive_summary"]


def test_compact_vendor_churn_scores_for_llm_prefers_specific_category_rows():
    rows = [
        {
            "vendor_name": "Zendesk",
            "product_category": "B2B Software",
            "total_reviews": 300,
            "churn_intent": 50,
            "avg_urgency": 3.2,
            "avg_rating_normalized": 2.1,
            "recommend_yes": 3,
            "recommend_no": 90,
        },
        {
            "vendor_name": "Zendesk",
            "product_category": "Helpdesk",
            "total_reviews": 280,
            "churn_intent": 49,
            "avg_urgency": 3.8,
            "avg_rating_normalized": 2.4,
            "recommend_yes": 3,
            "recommend_no": 90,
        },
    ]
    compact = _compact_vendor_churn_scores_for_llm(
        rows,
        council_lookup={
            (_canonicalize_vendor("Zendesk"), "helpdesk"): {"winner": "Zoho Desk"},
        },
    )
    assert len(compact) == 1
    assert compact[0]["vendor_name"] == "Zendesk"
    assert compact[0]["category"] == "Helpdesk"
    assert compact[0]["category_council"] == {"winner": "Zoho Desk"}


def test_apply_tenant_vendor_context_overrides_generated_category_and_council():
    payload = {
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "category_council": {"winner": "Zoho Desk"},
            },
            {
                "vendor_name": "HubSpot",
                "category": "Marketing Automation",
            },
        ]
    }
    parsed = {
        "weekly_churn_feed": [
            {"vendor": "Zendesk", "category": "B2B Software"},
            {"vendor": "HubSpot", "category": "B2B Software", "category_council": {"winner": "Someone"}},
        ],
        "vendor_scorecards": [
            {"vendor": "Zendesk", "category": "Software", "category_council": {"winner": "Other"}},
            {"vendor": "HubSpot", "category": "Software", "category_council": {"winner": "Other"}},
        ],
    }

    applied = _apply_tenant_vendor_context(parsed, payload)

    assert applied["weekly_churn_feed"][0]["category"] == "Helpdesk"
    assert applied["weekly_churn_feed"][0]["category_council"] == {"winner": "Zoho Desk"}
    assert applied["weekly_churn_feed"][1]["category"] == "Marketing Automation"
    assert "category_council" not in applied["weekly_churn_feed"][1]

    assert applied["vendor_scorecards"][0]["category"] == "Helpdesk"
    assert applied["vendor_scorecards"][0]["category_council"] == {"winner": "Zoho Desk"}
    assert applied["vendor_scorecards"][1]["category"] == "Marketing Automation"
    assert "category_council" not in applied["vendor_scorecards"][1]


def test_apply_tenant_vendor_context_backfills_missing_vendor_rows():
    payload = {
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "reviews": 280,
                "churn": 49,
                "urgency": 3.8,
                "rec_yes": 3,
                "rec_no": 90,
                "category_council": {"winner": "Zoho Desk"},
            },
            {
                "vendor_name": "HubSpot",
                "category": "Marketing Automation",
                "reviews": 341,
                "churn": 91,
                "urgency": 3.8,
                "rec_yes": 20,
                "rec_no": 87,
            },
        ]
    }
    parsed = {
        "weekly_churn_feed": [{"vendor": "HubSpot", "category": "B2B Software"}],
        "vendor_scorecards": [{"vendor": "HubSpot", "category": "B2B Software"}],
    }

    applied = _apply_tenant_vendor_context(parsed, payload)

    weekly_vendors = [row["vendor"] for row in applied["weekly_churn_feed"]]
    scorecard_vendors = [row["vendor"] for row in applied["vendor_scorecards"]]
    assert weekly_vendors == ["HubSpot", "Zendesk"]
    assert scorecard_vendors == ["HubSpot", "Zendesk"]

    zendesk_feed = next(row for row in applied["weekly_churn_feed"] if row["vendor"] == "Zendesk")
    zendesk_scorecard = next(row for row in applied["vendor_scorecards"] if row["vendor"] == "Zendesk")
    assert zendesk_feed["category"] == "Helpdesk"
    assert zendesk_feed["category_council"] == {"winner": "Zoho Desk"}
    assert zendesk_scorecard["category"] == "Helpdesk"
    assert zendesk_scorecard["category_council"] == {"winner": "Zoho Desk"}
    assert zendesk_scorecard["expert_take"]


def test_apply_tenant_report_context_backfills_categories_and_displacement():
    payload = {
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "reviews": 280,
                "churn": 49,
                "urgency": 3.8,
                "rec_yes": 3,
                "rec_no": 90,
                "category_council": {
                    "winner": "Zoho Desk",
                    "market_regime": "price_competition",
                    "conclusion": "Pricing pressure is driving churn across Helpdesk.",
                },
            },
            {
                "vendor_name": "HubSpot",
                "category": "Marketing Automation",
                "reviews": 341,
                "churn": 91,
                "urgency": 3.8,
                "rec_yes": 20,
                "rec_no": 87,
            },
        ],
        "competitive_displacement": [
            {
                "vendor": "Zendesk",
                "competitor": "Freshdesk",
                "mention_count": 20,
                "explicit_switches": 0,
                "active_evaluations": 0,
                "reason_categories": {"pricing": 12},
            }
        ],
        "competitor_reasons": [
            {
                "vendor": "Zendesk",
                "competitor": "Freshdesk",
                "reason_category": "pricing",
            }
        ],
    }
    parsed = {
        "weekly_churn_feed": [],
        "vendor_scorecards": [],
        "category_insights": [{"category": "Helpdesk"}],
        "displacement_map": [],
    }

    applied = _apply_tenant_report_context(parsed, payload)

    categories = [row["category"] for row in applied["category_insights"]]
    edges = [
        (row["from_vendor"], row["to_vendor"])
        for row in applied["displacement_map"]
    ]
    assert categories == ["Marketing Automation", "Helpdesk"]
    assert edges == [("Zendesk", "Freshdesk")]
    helpdesk = next(row for row in applied["category_insights"] if row["category"] == "Helpdesk")
    assert helpdesk["dominant_pain"] == "pricing"
    assert helpdesk["emerging_challenger"] == "Zoho Desk"
    edge = applied["displacement_map"][0]
    assert edge["primary_driver"] == "pricing"
    assert edge["signal_strength"] == "strong"


def test_apply_tenant_synthesis_context_attaches_shared_contracts(monkeypatch):
    calls = []

    def _fake_attach(entry, view, *, consumer_name, requested_as_of, include_displacement):
        calls.append((entry["vendor"], consumer_name, include_displacement, requested_as_of))
        entry["reasoning_source"] = consumer_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_churn_reports._attach_synthesis_contracts_to_report_entry",
        _fake_attach,
    )

    parsed = {
        "weekly_churn_feed": [
            {"vendor": "Zendesk"},
            {"vendor": "HubSpot"},
        ],
        "vendor_scorecards": [
            {"vendor": "Zendesk"},
            {"vendor": "Asana"},
        ],
    }
    attached = _apply_tenant_synthesis_context(
        parsed,
        {"Zendesk": object(), "Asana": object()},
        requested_as_of=date(2026, 3, 21),
    )

    assert attached == 2
    assert parsed["weekly_churn_feed"][0]["reasoning_source"] == "weekly_churn_feed"
    assert "reasoning_source" not in parsed["weekly_churn_feed"][1]
    assert parsed["vendor_scorecards"][0]["reasoning_source"] == "vendor_scorecard"
    assert parsed["vendor_scorecards"][1]["reasoning_source"] == "vendor_scorecard"
    assert calls == [
        ("Zendesk", "weekly_churn_feed", False, date(2026, 3, 21)),
        ("Zendesk", "vendor_scorecard", True, date(2026, 3, 21)),
        ("Asana", "vendor_scorecard", True, date(2026, 3, 21)),
    ]


def test_build_deterministic_tenant_report_prefers_context_then_contracts(monkeypatch):
    calls = []

    def _fake_attach(parsed, synthesis_views, *, requested_as_of):
        calls.append((sorted(synthesis_views.keys()), requested_as_of))
        parsed["weekly_churn_feed"][0]["account_pressure_summary"] = "Two accounts are active."
        return 1

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._apply_tenant_synthesis_context",
        _fake_attach,
    )

    payload = {
        "data_context": {
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "reviews": 280,
                "churn": 49,
                "urgency": 3.8,
                "rec_yes": 3,
                "rec_no": 90,
            }
        ],
        "competitive_displacement": [],
        "competitor_reasons": [],
    }

    parsed, attached = _build_deterministic_tenant_report(
        payload,
        {"Zendesk": object()},
        requested_as_of=date(2026, 3, 21),
    )

    assert attached == 1
    assert parsed["weekly_churn_feed"][0]["vendor"] == "Zendesk"
    assert parsed["weekly_churn_feed"][0]["account_pressure_summary"] == "Two accounts are active."
    assert parsed["executive_summary"]
    assert calls == [(["Zendesk"], date(2026, 3, 21))]


@pytest.mark.asyncio
async def test_build_deterministic_tenant_report_from_raw_uses_shared_builders(monkeypatch):
    async def _fake_read_vendor_intelligence_map(pool, *, as_of, analysis_window_days, vendor_names=None):
        assert vendor_names is None
        return {}

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.read_vendor_intelligence_map",
        _fake_read_vendor_intelligence_map,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared._fetch_latest_evidence_vault",
        AsyncMock(side_effect=AssertionError("deprecated wrapper should not run")),
    )

    payload = {
        "data_context": {
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "category": "Helpdesk"},
        ],
        "competitive_displacement": [],
        "competitor_reasons": [],
    }
    raw_artifacts = {
        "vendor_scores_from_signals": [
            {
                "vendor_name": "Zendesk",
                "product_category": "Helpdesk",
                "total_reviews": 120,
                "churn_intent": 18,
                "avg_urgency": 7.2,
            }
        ],
        "competitive_disp": [],
        "pain_dist": [],
        "feature_gaps": [],
        "negative_counts": [],
        "price_rates": [],
        "dm_rates": [],
        "churning_companies": [],
        "quotable_evidence": [],
        "budget_signals": [],
        "use_case_dist": [],
        "sentiment_traj": [],
        "buyer_auth": [],
        "timeline_signals": [],
        "competitor_reasons": [],
        "data_context": payload["data_context"],
        "prior_reports": [],
        "keyword_spikes": [],
        "product_profiles_raw": [],
        "review_text_aggs": ([], []),
        "department_dist": [],
        "contract_ctx_aggs": ([], []),
        "sentiment_tenure_raw": [],
        "turning_points_raw": [],
    }

    parsed, attached = await _build_deterministic_tenant_report_from_raw(
        pool=None,
        raw_artifacts=raw_artifacts,
        payload=payload,
        synthesis_views={},
        requested_as_of=date(2026, 3, 21),
        analysis_window_days=30,
    )

    assert attached == 0
    assert parsed["weekly_churn_feed"][0]["vendor"] == "Zendesk"
    assert parsed["vendor_scorecards"][0]["vendor"] == "Zendesk"
    assert parsed["vendor_scorecards"][0]["expert_take"]
    assert parsed["category_insights"][0]["category"] == "Helpdesk"
    assert parsed["executive_summary"]
