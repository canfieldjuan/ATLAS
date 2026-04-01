from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlas_brain.api.admin_costs import router


class _FakePool:
    def __init__(self):
        self.is_initialized = True
        self.last_fetch_query = ""
        self.last_fetch_args = ()

    async def fetchrow(self, query, *args):
        if "FROM task_executions e" in query and "JOIN scheduled_tasks t" in query:
            return {
                "id": uuid4(),
                "task_id": uuid4(),
                "task_name": "b2b_enrichment_repair",
                "status": "completed",
                "started_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                "completed_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
                "duration_ms": 61234,
                "retry_count": 0,
                "result_text": '{"generated": 5, "exact_cache_hits": 2}',
                "error": None,
                "metadata": {"progress_current": 5, "progress_total": 5},
            }
        if "FROM llm_usage" in query and "WHERE run_id = $1" in query and "AS first_call_at" in query:
            return {
                "total_calls": 3,
                "total_cost": Decimal("0.321"),
                "total_input": 3200,
                "total_billable_input": 1200,
                "total_cached_tokens": 1800,
                "total_cache_write_tokens": 400,
                "total_output": 510,
                "total_tokens": 3710,
                "cache_hit_calls": 2,
                "cache_write_calls": 1,
                "first_call_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                "last_call_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
            }
        if "today_cost" in query:
            return {"today_cost": Decimal("4.25"), "today_calls": 11}
        if "FROM b2b_llm_exact_cache" in query:
            return {
                "total_rows": 7,
                "total_hits": 19,
                "writes_in_window": 3,
                "rows_hit_in_window": 4,
            }
        if "FROM llm_usage" in query and "AS cached_tokens" in query and "AS total_cached_tokens" not in query:
            return {
                "total_calls": 25,
                "cache_hit_calls": 11,
                "cache_write_calls": 5,
                "cached_tokens": 6200,
                "cache_write_tokens": 1400,
                "billable_input_tokens": 9100,
            }
        if "FROM reasoning_semantic_cache" in query:
            return {
                "active_entries": 44,
                "invalidated_entries": 3,
                "recent_validations": 9,
            }
        if "FROM anthropic_message_batches" in query and "GROUP BY" not in query and "WHERE run_id = $1" not in query:
            return {
                "total_jobs": 3,
                "submitted_jobs": 2,
                "total_items": 21,
                "submitted_items": 18,
                "cache_prefiltered_items": 4,
                "fallback_single_call_items": 2,
                "completed_items": 16,
                "failed_items": 2,
                "estimated_sequential_cost_usd": Decimal("1.2"),
                "estimated_batch_cost_usd": Decimal("0.6"),
            }
        if "FROM anthropic_message_batches" in query and "WHERE run_id = $1" in query:
            return {
                "total_jobs": 1,
                "submitted_jobs": 1,
                "submitted_items": 6,
                "cache_prefiltered_items": 1,
                "fallback_single_call_items": 1,
                "completed_items": 5,
                "failed_items": 1,
                "estimated_sequential_cost_usd": Decimal("0.4"),
                "estimated_batch_cost_usd": Decimal("0.2"),
            }
        if "FROM b2b_vendor_reasoning_packets" in query:
            return {
                "total_rows": 31,
                "writes_in_window": 8,
                "unique_vendors": 12,
                "unique_hashes": 14,
            }
        if "FROM b2b_cross_vendor_conclusions" in query:
            return {
                "total_rows": 18,
                "cached_rows": 7,
                "cached_rows_in_window": 2,
            }
        return {
            "total_cost": Decimal("12.5"),
            "total_input": 12000,
            "total_billable_input": 8000,
            "total_cached_tokens": 3000,
            "total_cache_write_tokens": 1000,
            "total_output": 2100,
            "total_tokens": 14100,
            "total_calls": 17,
            "cache_hit_calls": 9,
            "cache_write_calls": 3,
            "avg_duration_ms": 420.2,
            "avg_tps": 33.7,
        }

    async def fetch(self, query, *args):
        self.last_fetch_query = query
        self.last_fetch_args = args
        if "e.id AS execution_id" in query:
            return [
                {
                    "execution_id": "run-enrich-1",
                    "task_name": "b2b_enrichment",
                    "started_at": datetime(2026, 3, 31, 21, 55, tzinfo=timezone.utc),
                    "result_text": '{"reviews_processed": 10, "witness_rows": 3, "witness_count": 15, "secondary_write_hits": 1, "strict_discussion_candidates_kept": 4, "strict_discussion_candidates_dropped": 0, "low_signal_discussion_skipped": 0, "exact_cache_hits": 2, "generated": 8}',
                },
                {
                    "execution_id": "run-repair-1",
                    "task_name": "b2b_enrichment_repair",
                    "started_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
                    "result_text": '{"reviews_processed": 5, "witness_rows": 2, "witness_count": 8, "secondary_write_hits": 1, "strict_discussion_candidates_kept": 2, "strict_discussion_candidates_dropped": 3, "low_signal_discussion_skipped": 3, "exact_cache_hits": 1, "generated": 4}',
                },
                {
                    "execution_id": "run-reason-1",
                    "task_name": "b2b_reasoning_synthesis",
                    "started_at": datetime(2026, 3, 31, 22, 10, tzinfo=timezone.utc),
                    "result_text": '{"vendors_reasoned": 2, "witness_vendor_rows": 1, "witness_count": 12, "generated": 2}',
                },
            ]
        if "GROUP BY vendor_name" in query and "FROM llm_usage" in query:
            return [
                {
                    "vendor_name": "Slack",
                    "cost": Decimal("0.35"),
                    "input_tokens": 3500,
                    "billable_input_tokens": 1400,
                    "cached_tokens": 1600,
                    "cache_write_tokens": 300,
                    "output_tokens": 450,
                    "total_tokens": 3950,
                    "calls": 3,
                    "cache_hit_calls": 1,
                    "cache_write_calls": 1,
                    "avg_duration_ms": 620.5,
                },
                {
                    "vendor_name": "Zoom",
                    "cost": Decimal("0.08"),
                    "input_tokens": 900,
                    "billable_input_tokens": 500,
                    "cached_tokens": 0,
                    "cache_write_tokens": 100,
                    "output_tokens": 110,
                    "total_tokens": 1010,
                    "calls": 1,
                    "cache_hit_calls": 0,
                    "cache_write_calls": 1,
                    "avg_duration_ms": 440.0,
                },
            ]
        if "FROM llm_usage" in query and "metadata::text ILIKE '%source%'" in query:
            return [
                {
                    "span_name": "task.b2b_enrichment.tier1",
                    "cost_usd": Decimal("0.10"),
                    "vendor_name": "Slack",
                    "run_id": "run-enrich-1",
                    "metadata": {"vendor_name": "Slack", "source": "reddit"},
                    "created_at": datetime(2026, 3, 31, 21, 55, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_enrichment_repair.extraction",
                    "cost_usd": Decimal("0.05"),
                    "vendor_name": "Slack",
                    "run_id": "run-repair-1",
                    "metadata": {"vendor_name": "Slack", "source": "reddit"},
                    "created_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_reasoning_synthesis",
                    "cost_usd": Decimal("0.20"),
                    "vendor_name": "Slack",
                    "run_id": "run-reason-1",
                    "metadata": {"vendor_name": "Slack"},
                    "created_at": datetime(2026, 3, 31, 22, 10, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_enrichment.tier1",
                    "cost_usd": Decimal("0.08"),
                    "vendor_name": None,
                    "run_id": "run-enrich-2",
                    "metadata": {"vendor_name": "Zoom", "source": "g2"},
                    "created_at": datetime(2026, 3, 31, 21, 57, tzinfo=timezone.utc),
                },
            ]
        if "FROM b2b_reviews" in query and "GROUP BY source" in query:
            return [
                {
                    "source": "reddit",
                    "enriched_rows": 100,
                    "repair_triggered_rows": 20,
                    "repair_promoted_rows": 10,
                    "rows_with_spans": 90,
                    "span_count": 180,
                    "low_signal_discussion_skipped_rows": 12,
                    "strict_discussion_candidates_kept_rows": 18,
                },
                {
                    "source": "g2",
                    "enriched_rows": 40,
                    "repair_triggered_rows": 4,
                    "repair_promoted_rows": 2,
                    "rows_with_spans": 35,
                    "span_count": 70,
                    "low_signal_discussion_skipped_rows": 0,
                    "strict_discussion_candidates_kept_rows": 0,
                },
            ]
        if "FROM artifact_attempts" in query and "WHERE run_id = $1" in query:
            return [{
                "id": uuid4(),
                "artifact_type": "enrichment",
                "artifact_id": "batch",
                "run_id": "run-123",
                "attempt_no": 1,
                "stage": "enrichment",
                "status": "succeeded",
                "score": 5,
                "threshold": None,
                "blocker_count": 0,
                "warning_count": 1,
                "blocking_issues": [],
                "warnings": ["quarantined_review"],
                "failure_step": None,
                "error_message": None,
                "started_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                "completed_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
            }]
        if "FROM pipeline_visibility_events" in query and "WHERE run_id = $1" in query:
            return [{
                "id": uuid4(),
                "occurred_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
                "run_id": "run-123",
                "stage": "extraction",
                "event_type": "enrichment_run_summary",
                "severity": "warning",
                "actionable": True,
                "entity_type": "pipeline",
                "entity_id": "enrichment",
                "artifact_type": "enrichment",
                "reason_code": "enrichment_quarantines",
                "rule_code": None,
                "decision": None,
                "summary": "Enrichment summary",
                "detail": '{"quarantined": 1}',
                "fingerprint": "fp_123",
            }]
        if "FROM b2b_llm_exact_cache" in query and "GROUP BY namespace" in query:
            return [{
                "namespace": "b2b_blog_post_generation.content",
                "rows": 2,
                "total_hits": 5,
                "writes_in_window": 1,
                "rows_hit_in_window": 2,
                "last_write_at": datetime(2026, 3, 31, 21, 30, tzinfo=timezone.utc),
                "last_hit_at": datetime(2026, 3, 31, 21, 45, tzinfo=timezone.utc),
                "provider_count": 1,
                "model_count": 1,
            }]
        if "FROM llm_usage" in query and "GROUP BY span_name" in query and "GROUP BY span_name, operation_type, model_name, model_provider" not in query:
            return [{
                "span_name": "task.b2b_blog_post_generation",
                "calls": 4,
                "cache_hit_calls": 3,
                "cache_write_calls": 1,
                "cached_tokens": 5000,
                "cache_write_tokens": 800,
            }]
        if "FROM reasoning_semantic_cache" in query and "GROUP BY pattern_class" in query:
            return [{
                "pattern_class": "battle_card_render",
                "active_entries": 12,
                "recent_validations": 4,
            }]
        if "FROM anthropic_message_batches" in query and "GROUP BY stage_id, task_name" in query:
            return [{
                "stage_id": "b2b_campaign_generation.content",
                "task_name": "b2b_campaign_generation",
                "total_jobs": 2,
                "submitted_jobs": 2,
                "total_items": 18,
                "submitted_items": 18,
                "cache_prefiltered_items": 4,
                "fallback_single_call_items": 2,
                "completed_items": 16,
                "failed_items": 2,
                "estimated_sequential_cost_usd": Decimal("1.2"),
                "estimated_batch_cost_usd": Decimal("0.6"),
                "last_submitted_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                "last_completed_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
            }]
        if "FROM anthropic_message_batches" in query and "completed_at IS NULL" in query:
            return [{
                "id": uuid4(),
                "stage_id": "b2b_campaign_generation.content",
                "task_name": "b2b_campaign_generation",
                "run_id": "run-batch-1",
                "status": "in_progress",
                "provider_batch_id": "msgbatch_stale_1",
                "total_items": 12,
                "submitted_items": 10,
                "completed_items": 3,
                "failed_items": 1,
                "fallback_single_call_items": 0,
                "submitted_at": datetime(2026, 3, 31, 20, 0, tzinfo=timezone.utc),
                "created_at": datetime(2026, 3, 31, 19, 58, tzinfo=timezone.utc),
                "provider_error": None,
            }]
        if "FROM anthropic_message_batch_items i" in query and "request_metadata->>'applying_at'" in query:
            return [{
                "id": uuid4(),
                "batch_id": uuid4(),
                "stage_id": "b2b_campaign_generation.content",
                "task_name": "b2b_campaign_generation",
                "run_id": "run-batch-1",
                "custom_id": "campaign:slack:email",
                "artifact_id": "batch-1:Slack:email",
                "status": "batch_succeeded",
                "provider_batch_id": "msgbatch_stale_1",
                "applying_by": "reconcile:task-1:abc123",
                "applying_at": datetime(2026, 3, 31, 20, 5, tzinfo=timezone.utc),
            }]
        if "FROM anthropic_message_batches" in query and "WHERE run_id = $1" in query:
            return [{
                "id": uuid4(),
                "stage_id": "b2b_campaign_generation.content",
                "task_name": "b2b_campaign_generation",
                "status": "ended",
                "provider_batch_id": "msgbatch_123",
                "total_items": 6,
                "submitted_items": 6,
                "cache_prefiltered_items": 1,
                "fallback_single_call_items": 1,
                "completed_items": 5,
                "failed_items": 1,
                "estimated_sequential_cost_usd": Decimal("0.4"),
                "estimated_batch_cost_usd": Decimal("0.2"),
                "submitted_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                "completed_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
            }]
        if "FROM anthropic_message_batch_items i" in query:
            return [
                {
                    "id": uuid4(),
                    "batch_id": uuid4(),
                    "custom_id": "campaign:slack:email",
                    "stage_id": "b2b_campaign_generation.content",
                    "task_name": "b2b_campaign_generation",
                    "provider_batch_id": "msgbatch_123",
                    "artifact_type": "campaign",
                    "artifact_id": "batch-1:Slack:email",
                    "vendor_name": "Slack",
                    "status": "batch_succeeded",
                    "cache_prefiltered": False,
                    "fallback_single_call": False,
                    "input_tokens": 900,
                    "billable_input_tokens": 400,
                    "cached_tokens": 500,
                    "cache_write_tokens": 0,
                    "output_tokens": 180,
                    "total_tokens": 1080,
                    "cost_usd": Decimal("0.045"),
                    "provider_request_id": "msg_123",
                    "error_text": None,
                    "request_metadata": {
                        "channel": "email",
                        "target_mode": "vendor_retention",
                        "tier": "report",
                        "replay_handler": "campaign_generation",
                        "applied_at": "2026-03-31T22:01:30+00:00",
                        "applied_status": "succeeded",
                    },
                    "created_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                    "completed_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
                },
                {
                    "id": uuid4(),
                    "batch_id": uuid4(),
                    "custom_id": "campaign:zoom:email",
                    "stage_id": "b2b_campaign_generation.content",
                    "task_name": "b2b_campaign_generation",
                    "provider_batch_id": "msgbatch_123",
                    "artifact_type": "campaign",
                    "artifact_id": "batch-1:Zoom:email",
                    "vendor_name": "Zoom",
                    "status": "cache_hit",
                    "cache_prefiltered": True,
                    "fallback_single_call": False,
                    "input_tokens": 850,
                    "billable_input_tokens": 0,
                    "cached_tokens": 850,
                    "cache_write_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 850,
                    "cost_usd": Decimal("0"),
                    "provider_request_id": None,
                    "error_text": None,
                    "request_metadata": {
                        "channel": "email",
                        "target_mode": "vendor_retention",
                        "tier": "report",
                        "replay_handler": "campaign_generation",
                    },
                    "created_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                    "completed_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
                },
            ]
        if "FROM task_executions e" in query and "JOIN scheduled_tasks t" in query:
            return [
                {
                    "name": "b2b_battle_cards",
                    "result_text": None,
                    "metadata": {"cache_hits": 2, "cards_llm_updated": 1},
                },
                {
                    "name": "b2b_churn_reports",
                    "result_text": '{"scorecard_cache_hits": 3, "scorecard_reasoning_reused": 2, "scorecard_llm_generated": 1}',
                    "metadata": {},
                },
                {
                    "name": "b2b_reasoning_synthesis",
                    "result_text": '{"vendors_skipped": 4, "vendors_reasoned": 2, "cross_vendor_succeeded": 1}',
                    "metadata": {},
                },
                {
                    "name": "b2b_enrichment",
                    "result_text": '{"exact_cache_hits": 6, "generated": 9, "tier1_exact_cache_hits": 4, "tier2_exact_cache_hits": 2}',
                    "metadata": {},
                },
                {
                    "name": "b2b_enrichment_repair",
                    "result_text": '{"exact_cache_hits": 5, "generated": 7, "promoted": 3, "shadowed": 1, "failed": 0}',
                    "metadata": {},
                },
            ]
        if "GROUP BY span_name, operation_type, model_name, model_provider" in query:
            return [{
                "span_name": "task.b2b_blog_post_generation",
                "operation_type": "llm_call",
                "model_name": "anthropic/claude-sonnet-4-6",
                "model_provider": "openrouter",
                "cost": Decimal("1.234"),
                "input_tokens": 10000,
                "billable_input_tokens": 4200,
                "cached_tokens": 5000,
                "cache_write_tokens": 800,
                "output_tokens": 1800,
                "total_tokens": 11800,
                "calls": 4,
                "cache_hit_calls": 3,
                "cache_write_calls": 1,
                "avg_duration_ms": 811.5,
                "latest_created_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
            }]
        return [{
            "id": uuid4(),
            "span_name": "task.b2b_blog_post_generation",
            "operation_type": "llm_call",
            "model_name": "anthropic/claude-sonnet-4-6",
            "model_provider": "openrouter",
            "input_tokens": 10000,
            "billable_input_tokens": 4200,
            "cached_tokens": 5000,
            "cache_write_tokens": 800,
            "output_tokens": 1800,
            "total_tokens": 11800,
            "cost_usd": Decimal("0.0483"),
            "duration_ms": 922,
            "ttft_ms": 210,
            "inference_time_ms": 640,
            "queue_time_ms": 12,
            "tokens_per_second": 28.1,
            "status": "completed",
            "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "provider_request_id": "req_123",
            "metadata": '{"vendor_name":"HubSpot"}',
            "created_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
        }]


def _client(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.admin_costs.get_db_pool", lambda: pool)
    return TestClient(app), pool


def test_cost_summary_includes_cache_metrics(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/summary?days=30")
    assert res.status_code == 200
    body = res.json()
    assert body["total_billable_input_tokens"] == 8000
    assert body["total_cached_tokens"] == 3000
    assert body["total_cache_write_tokens"] == 1000
    assert body["cache_hit_calls"] == 9
    assert body["cache_write_calls"] == 3


def test_cost_by_operation_exposes_cache_rollups(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/by-operation?days=7&limit=25")
    assert res.status_code == 200
    row = res.json()["operations"][0]
    assert row["span_name"] == "task.b2b_blog_post_generation"
    assert row["billable_input_tokens"] == 4200
    assert row["cached_tokens"] == 5000
    assert row["cache_write_tokens"] == 800
    assert row["cache_hit_calls"] == 3


def test_recent_calls_returns_granular_cache_fields(monkeypatch):
    client, pool = _client(monkeypatch)
    with client:
        res = client.get(
            "/admin/costs/recent?days=7&provider=openrouter&status=completed&cache_only=true&limit=10"
        )
    assert res.status_code == 200
    row = res.json()["calls"][0]
    assert row["detail"] == "HubSpot"
    assert row["billable_input_tokens"] == 4200
    assert row["cached_tokens"] == 5000
    assert row["cache_write_tokens"] == 800
    assert row["cache_hit"] is True
    assert row["cache_write"] is True
    assert "model_provider = $2" in pool.last_fetch_query
    assert "(cached_tokens > 0 OR cache_write_tokens > 0)" in pool.last_fetch_query


def test_cache_health_rolls_up_exact_prompt_semantic_and_task_reuse(monkeypatch):
    monkeypatch.setattr(
        "atlas_brain.api.admin_costs.settings.b2b_churn.llm_exact_cache_enabled",
        True,
        raising=False,
    )
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/cache-health?days=14&top_n=5")
    assert res.status_code == 200
    body = res.json()
    assert body["period_days"] == 14
    assert body["top_n"] == 5
    assert body["exact_cache"]["enabled"] is True
    assert body["exact_cache"]["total_rows"] == 7
    assert body["exact_cache"]["total_hits"] == 19
    assert body["provider_prompt_cache"]["cache_hit_calls"] == 11
    assert body["provider_prompt_cache"]["top_spans"][0]["span_name"] == "task.b2b_blog_post_generation"
    assert body["anthropic_batching"]["submitted_jobs"] == 2
    assert body["anthropic_batching"]["estimated_savings_usd"] == pytest.approx(0.6)
    assert body["anthropic_batching"]["stages"][0]["stage_id"] == "b2b_campaign_generation.content"
    assert body["anthropic_batching"]["stale_job_threshold_minutes"] == 30
    assert body["anthropic_batching"]["stale_jobs_count"] == 1
    assert body["anthropic_batching"]["stale_jobs"][0]["task_name"] == "b2b_campaign_generation"
    assert body["anthropic_batching"]["stale_jobs"][0]["provider_batch_id"] == "msgbatch_stale_1"
    assert body["anthropic_batching"]["stale_claims_count"] == 1
    assert body["anthropic_batching"]["stale_claims"][0]["custom_id"] == "campaign:slack:email"
    assert body["anthropic_batching"]["stale_claims"][0]["applying_by"] == "reconcile:task-1:abc123"
    assert body["semantic_cache"]["active_entries"] == 44
    assert body["semantic_cache"]["pattern_classes"][0]["pattern_class"] == "battle_card_render"
    assert body["evidence_hash_reuse"]["cross_vendor_cached_rows"] == 7
    tasks = {row["task_name"]: row for row in body["task_reuse"]["tasks"]}
    assert tasks["b2b_battle_cards"]["reused"] == 2
    assert tasks["b2b_battle_cards"]["semantic_cache_hits"] == 2
    assert tasks["b2b_churn_reports"]["reused"] == 5
    assert tasks["b2b_churn_reports"]["exact_cache_hits"] == 3
    assert tasks["b2b_churn_reports"]["evidence_hash_reuse"] == 2
    assert tasks["b2b_reasoning_synthesis"]["reused"] == 4
    assert tasks["b2b_reasoning_synthesis"]["evidence_hash_reuse"] == 4
    assert tasks["b2b_reasoning_synthesis"]["generated"] == 3
    assert tasks["b2b_enrichment"]["reused"] == 6
    assert tasks["b2b_enrichment"]["exact_cache_hits"] == 6
    assert tasks["b2b_enrichment"]["generated"] == 9
    assert tasks["b2b_enrichment_repair"]["reused"] == 5
    assert tasks["b2b_enrichment_repair"]["exact_cache_hits"] == 5
    assert tasks["b2b_enrichment_repair"]["generated"] == 7


def test_cost_run_detail_correlates_execution_usage_attempts_and_events(monkeypatch):
    client, _ = _client(monkeypatch)
    run_id = str(uuid4())
    with client:
        res = client.get(
            f"/admin/costs/runs/{run_id}?call_limit=10&attempt_limit=10&event_limit=10&batch_item_limit=10"
        )
    assert res.status_code == 200
    body = res.json()
    assert body["run_id"] == run_id
    assert body["task_execution"]["task_name"] == "b2b_enrichment_repair"
    assert body["task_execution"]["result"]["generated"] == 5
    assert body["llm_summary"]["total_calls"] == 3
    assert body["llm_summary"]["cache_hit_calls"] == 2
    assert body["batching_summary"]["submitted_jobs"] == 1
    assert body["batching_summary"]["estimated_savings_usd"] == pytest.approx(0.2)
    assert body["operations"][0]["span_name"] == "task.b2b_blog_post_generation"
    assert body["batch_jobs"][0]["stage_id"] == "b2b_campaign_generation.content"
    assert body["batch_items"][0]["status"] == "batch_succeeded"
    assert body["batch_items"][0]["request_metadata"]["channel"] == "email"
    assert body["batch_items"][0]["replay_handler"] == "campaign_generation"
    assert body["batch_items"][0]["replay_contract_state"] == "missing"
    assert body["batch_items"][0]["replay_contract_version"] is None
    assert body["batch_items"][0]["applied_status"] == "succeeded"
    assert body["batch_items"][1]["cache_prefiltered"] is True
    assert body["batch_items"][1]["replay_contract_state"] == "missing"
    assert body["batch_items"][1]["applied_status"] is None
    assert body["calls"][0]["title"] == "task.b2b_blog_post_generation"
    assert body["artifact_attempts"][0]["artifact_type"] == "enrichment"
    assert body["visibility_events"][0]["event_type"] == "enrichment_run_summary"
    assert body["visibility_events"][0]["detail"]["quarantined"] == 1


def test_cost_by_vendor_returns_sorted_rollups(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/by-vendor?days=30&limit=10")
    assert res.status_code == 200
    body = res.json()
    assert body["period_days"] == 30
    assert body["vendors"][0]["vendor_name"] == "Slack"
    assert body["vendors"][0]["cost_usd"] == 0.35
    assert body["vendors"][1]["vendor_name"] == "Zoom"


def test_b2b_efficiency_rolls_up_vendor_source_and_run_metrics(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/b2b-efficiency?days=30&top_n=5&run_limit=5")
    assert res.status_code == 200
    body = res.json()
    assert body["period_days"] == 30
    assert body["top_n"] == 5
    assert body["run_limit"] == 5
    assert body["summary"]["measured_runs"] == 3
    assert body["summary"]["tracked_witness_count"] == 35
    assert body["summary"]["tracked_cost_usd"] == 0.35
    assert body["summary"]["cost_per_witness_usd"] == pytest.approx(0.01)

    vendor_row = body["vendor_passes"][0]
    assert vendor_row["vendor_name"] == "Slack"
    assert vendor_row["extraction_cost_usd"] == 0.1
    assert vendor_row["repair_cost_usd"] == 0.05
    assert vendor_row["reasoning_cost_usd"] == 0.2
    assert vendor_row["total_cost_usd"] == pytest.approx(0.35)

    source_rows = {row["source"]: row for row in body["source_efficiency"]}
    assert source_rows["reddit"]["total_cost_usd"] == pytest.approx(0.15)
    assert source_rows["reddit"]["witness_yield_rate"] == pytest.approx(1.8)
    assert source_rows["reddit"]["repair_trigger_rate"] == pytest.approx(0.2)
    assert source_rows["reddit"]["cost_per_witness_usd"] == pytest.approx(0.000833, rel=1e-3)
    assert source_rows["reddit"]["strict_discussion_candidates_kept_rows"] == 18
    assert source_rows["reddit"]["low_signal_discussion_skipped_rows"] == 12
    assert source_rows["g2"]["total_cost_usd"] == pytest.approx(0.08)

    run_rows = {row["run_id"]: row for row in body["recent_runs"]}
    assert run_rows["run-enrich-1"]["task_name"] == "b2b_enrichment"
    assert run_rows["run-enrich-1"]["total_cost_usd"] == 0.1
    assert run_rows["run-enrich-1"]["witness_count"] == 15
    assert run_rows["run-enrich-1"]["strict_discussion_candidates_kept"] == 4
    assert run_rows["run-repair-1"]["secondary_write_hits"] == 1
    assert run_rows["run-repair-1"]["strict_discussion_candidates_dropped"] == 3
    assert run_rows["run-reason-1"]["reasoning_cost_usd"] == 0.2
