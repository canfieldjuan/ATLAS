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
        self.provider_daily_rows = []
        self.provider_snapshot_rows = []

    async def fetchrow(self, query, *args):
        if "artifact_type = 'cross_vendor_reasoning'" in query and "ORDER BY created_at DESC" in query:
            return {
                "status": "succeeded",
                "failure_step": None,
                "created_at": datetime(2026, 3, 31, 22, 31, tzinfo=timezone.utc),
            }
        if "artifact_type = 'cross_vendor_reasoning'" in query:
            return {
                "succeeded_items": 1,
                "failed_items": 0,
                "input_budget_rejections": 1,
                "last_attempt_at": datetime(2026, 3, 31, 22, 30, tzinfo=timezone.utc),
            }
        if "span_name = 'task.b2b_reasoning_synthesis.cross_vendor'" in query:
            return {
                "model_call_count": 2,
                "total_input_tokens": 1000,
                "total_billable_input_tokens": 900,
                "total_output_tokens": 150,
                "total_cost_usd": Decimal("0.11"),
                "last_call_at": datetime(2026, 3, 31, 22, 29, tzinfo=timezone.utc),
            }
        if "span_name = 'reasoning.process'" in query and "MAX(created_at) AS last_call_at" in query:
            return {
                "model_call_count": 12,
                "total_input_tokens": 54000,
                "total_billable_input_tokens": 42000,
                "total_output_tokens": 9100,
                "total_cost_usd": Decimal("1.75"),
                "last_call_at": datetime(2026, 3, 31, 22, 30, tzinfo=timezone.utc),
            }
        if "span_name = 'reasoning.process'" in query and "total_billable_input_tokens" in query:
            return {
                "total_calls": 12,
                "total_cost": Decimal("1.75"),
                "total_billable_input_tokens": 42000,
                "total_output_tokens": 9100,
            }
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
        if "FROM scheduled_tasks t" in query and "COALESCE(stats.recent_runs, 0) AS recent_runs" in query:
            return [
                {
                    "id": uuid4(),
                    "name": "b2b_enrichment",
                    "last_run_at": datetime(2026, 3, 31, 21, 55, tzinfo=timezone.utc),
                    "last_status": "completed",
                    "recent_runs": 1,
                    "recent_failures": 0,
                },
                {
                    "id": uuid4(),
                    "name": "b2b_enrichment_repair",
                    "last_run_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
                    "last_status": "completed",
                    "recent_runs": 1,
                    "recent_failures": 0,
                },
                {
                    "id": uuid4(),
                    "name": "b2b_reasoning_synthesis",
                    "last_run_at": datetime(2026, 3, 31, 22, 10, tzinfo=timezone.utc),
                    "last_status": "completed",
                    "recent_runs": 1,
                    "recent_failures": 0,
                },
                {
                    "id": uuid4(),
                    "name": "b2b_battle_cards",
                    "last_run_at": datetime(2026, 3, 31, 22, 15, tzinfo=timezone.utc),
                    "last_status": "completed",
                    "recent_runs": 1,
                    "recent_failures": 0,
                },
            ]
        if "SELECT\n            t.name AS task_name,\n            e.id::text AS run_id," in query:
            return [
                {
                    "task_name": "b2b_enrichment",
                    "run_id": "run-enrich-1",
                    "status": "completed",
                    "started_at": datetime(2026, 3, 31, 21, 55, tzinfo=timezone.utc),
                    "retry_count": 0,
                    "result_text": '{"reviews_processed": 10, "witness_count": 15, "generated": 8, "_skip_synthesis": "No new reviews pending enrichment"}',
                    "metadata": {},
                },
                {
                    "task_name": "b2b_enrichment_repair",
                    "run_id": "run-repair-1",
                    "status": "completed",
                    "started_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
                    "retry_count": 1,
                    "result_text": '{"reviews_processed": 5, "strict_discussion_candidates_dropped": 3, "low_signal_discussion_skipped": 3, "generated": 4}',
                    "metadata": {},
                },
                {
                    "task_name": "b2b_reasoning_synthesis",
                    "run_id": "run-reason-1",
                    "status": "completed",
                    "started_at": datetime(2026, 3, 31, 22, 10, tzinfo=timezone.utc),
                    "retry_count": 0,
                    "result_text": '{"vendors_reasoned": 2, "vendors_skipped": 4, "cross_vendor_succeeded": 1, "generated": 3}',
                    "metadata": {"source_name": "b2b_scheduler", "event_type": "nightly_reasoning"},
                },
                {
                    "task_name": "b2b_battle_cards",
                    "run_id": "run-battle-1",
                    "status": "completed",
                    "started_at": datetime(2026, 3, 31, 22, 15, tzinfo=timezone.utc),
                    "retry_count": 0,
                    "result_text": '{"cards_built": 2, "cards_llm_updated": 1, "cache_hits": 1, "llm_failures": 1}',
                    "metadata": {},
                },
            ]
        if "COUNT(*) AS model_call_count" in query and "run_id IS NOT NULL" in query:
            return [
                {
                    "run_id": "run-enrich-1",
                    "span_name": "task.b2b_enrichment.tier1",
                    "last_call_at": datetime(2026, 3, 31, 21, 55, tzinfo=timezone.utc),
                    "model_call_count": 1,
                    "total_input_tokens": 1100,
                    "total_billable_input_tokens": 700,
                    "total_output_tokens": 120,
                    "total_cost_usd": Decimal("0.10"),
                },
                {
                    "run_id": "run-repair-1",
                    "span_name": "task.b2b_enrichment_repair.extraction",
                    "last_call_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
                    "model_call_count": 1,
                    "total_input_tokens": 900,
                    "total_billable_input_tokens": 500,
                    "total_output_tokens": 100,
                    "total_cost_usd": Decimal("0.05"),
                },
                {
                    "run_id": "run-reason-1",
                    "span_name": "task.b2b_reasoning_synthesis",
                    "last_call_at": datetime(2026, 3, 31, 22, 10, tzinfo=timezone.utc),
                    "model_call_count": 1,
                    "total_input_tokens": 2000,
                    "total_billable_input_tokens": 1200,
                    "total_output_tokens": 300,
                    "total_cost_usd": Decimal("0.20"),
                },
                {
                    "run_id": "manual-reason-1",
                    "span_name": "task.b2b_reasoning_synthesis",
                    "last_call_at": datetime(2026, 3, 31, 22, 20, tzinfo=timezone.utc),
                    "model_call_count": 1,
                    "total_input_tokens": 1500,
                    "total_billable_input_tokens": 900,
                    "total_output_tokens": 220,
                    "total_cost_usd": Decimal("0.15"),
                },
                {
                    "run_id": "run-battle-1",
                    "span_name": "b2b.churn_intelligence.battle_card_sales_copy",
                    "last_call_at": datetime(2026, 3, 31, 22, 15, tzinfo=timezone.utc),
                    "model_call_count": 1,
                    "total_input_tokens": 1800,
                    "total_billable_input_tokens": 950,
                    "total_output_tokens": 260,
                    "total_cost_usd": Decimal("0.07"),
                },
            ]
        if "FROM pipeline_visibility_events v" in query and "trigger_reason" in query:
            return [
                {
                    "task_name": "b2b_enrichment_repair",
                    "trigger_reason": "strict_discussion_gate",
                    "trigger_count": 3,
                },
                {
                    "task_name": "b2b_reasoning_synthesis",
                    "trigger_reason": "thin_specific_witness_pool",
                    "trigger_count": 2,
                },
            ]
        if "FROM artifact_attempts" in query and "failure_step = 'input_budget'" in query:
            return [
                {
                    "artifact_type": "reasoning_synthesis",
                    "artifact_id": "Slack",
                    "blocking_issues": [
                        "input token budget exceeded: estimated_input_tokens=21000, cap=20000"
                    ],
                    "error_message": "Vendor reasoning prompt exceeded the configured input token cap",
                    "created_at": datetime(2026, 3, 31, 22, 25, tzinfo=timezone.utc),
                },
                {
                    "artifact_type": "cross_vendor_reasoning",
                    "artifact_id": "resource_asymmetry:slack|teams",
                    "blocking_issues": [
                        "input token budget exceeded: estimated_input_tokens=12437, cap=12000"
                    ],
                    "error_message": "Cross-vendor prompt exceeded the configured input token cap",
                    "created_at": datetime(2026, 3, 31, 22, 30, tzinfo=timezone.utc),
                },
            ]
        if "FROM llm_provider_daily_costs" in query:
            return self.provider_daily_rows
        if "FROM llm_provider_usage_snapshots" in query and "latest_daily AS (" in query:
            return self.provider_snapshot_rows
        if "span_name = 'reasoning.process'" in query and "GROUP BY 1, 2" in query and "entity_id" not in query:
            return [
                {
                    "source_name": "crm_provider",
                    "event_type": "crm.interaction_logged",
                    "calls": 8,
                    "cost_usd": Decimal("1.25"),
                    "billable_input_tokens": 30000,
                    "output_tokens": 7000,
                },
                {
                    "source_name": "email_intake",
                    "event_type": "email.received",
                    "calls": 4,
                    "cost_usd": Decimal("0.5"),
                    "billable_input_tokens": 12000,
                    "output_tokens": 2100,
                },
            ]
        if "span_name = 'reasoning.process'" in query and "GROUP BY 1, 2" in query and "entity_id" in query:
            return [
                {
                    "entity_type": "contact",
                    "entity_id": "contact-123",
                    "calls": 7,
                    "cost_usd": Decimal("1.2"),
                    "billable_input_tokens": 28000,
                    "output_tokens": 6500,
                },
                {
                    "entity_type": "contact",
                    "entity_id": "contact-999",
                    "calls": 2,
                    "cost_usd": Decimal("0.2"),
                    "billable_input_tokens": 5000,
                    "output_tokens": 900,
                },
            ]
        if "span_name = 'reasoning.process'" in query and "GROUP BY 1" in query and "AS source_name" in query:
            return [
                {
                    "source_name": "crm_provider",
                    "calls": 8,
                    "cost_usd": Decimal("1.25"),
                    "billable_input_tokens": 30000,
                    "output_tokens": 7000,
                },
                {
                    "source_name": "email_intake",
                    "calls": 4,
                    "cost_usd": Decimal("0.5"),
                    "billable_input_tokens": 12000,
                    "output_tokens": 2100,
                },
            ]
        if "span_name = 'reasoning.process'" in query and "GROUP BY 1" in query and "AS event_type" in query:
            return [
                {
                    "event_type": "crm.interaction_logged",
                    "calls": 8,
                    "cost_usd": Decimal("1.25"),
                    "billable_input_tokens": 30000,
                    "output_tokens": 7000,
                },
                {
                    "event_type": "email.received",
                    "calls": 4,
                    "cost_usd": Decimal("0.5"),
                    "billable_input_tokens": 12000,
                    "output_tokens": 2100,
                },
            ]
        if "DATE(created_at AT TIME ZONE 'UTC') AS day" in query and "COALESCE(model_provider, 'unknown') AS provider" in query:
            return [
                {
                    "day": "2026-03-31",
                    "provider": "openrouter",
                    "tracked_cost_usd": Decimal("1.75"),
                    "calls": 12,
                },
                {
                    "day": "2026-03-30",
                    "provider": "openrouter",
                    "tracked_cost_usd": Decimal("0.85"),
                    "calls": 5,
                },
            ]
        if "span_name LIKE 'reasoning.stratified.%'" in query:
            return [
                {
                    "span_name": "reasoning.stratified.reason",
                    "pass_type": "full",
                    "pass_number": 1,
                    "cost": Decimal("0.44"),
                    "tokens": 4200,
                    "calls": 3,
                    "avg_duration_ms": 812.2,
                    "changed_count": 1,
                },
                {
                    "span_name": "reasoning.stratified.reason.ground",
                    "pass_type": "ground",
                    "pass_number": 2,
                    "cost": Decimal("0.12"),
                    "tokens": 1200,
                    "calls": 1,
                    "avg_duration_ms": 633.1,
                    "changed_count": 0,
                },
            ]
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
                {
                    "execution_id": "run-battle-1",
                    "task_name": "b2b_battle_cards",
                    "started_at": datetime(2026, 3, 31, 22, 15, tzinfo=timezone.utc),
                    "result_text": '{"cards_built": 2, "cards_llm_updated": 1, "cache_hits": 1, "llm_failures": 1}',
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
                    "billable_input_tokens": 700,
                    "output_tokens": 120,
                    "vendor_name": "Slack",
                    "run_id": "run-enrich-1",
                    "metadata": {"vendor_name": "Slack", "source": "reddit"},
                    "created_at": datetime(2026, 3, 31, 21, 55, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_enrichment.tier2",
                    "cost_usd": Decimal("0.02"),
                    "billable_input_tokens": 600,
                    "output_tokens": 90,
                    "vendor_name": "Slack",
                    "run_id": "run-enrich-1",
                    "metadata": {"vendor_name": "Slack", "source": "reddit"},
                    "created_at": datetime(2026, 3, 31, 21, 56, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_enrichment_repair.extraction",
                    "cost_usd": Decimal("0.05"),
                    "billable_input_tokens": 500,
                    "output_tokens": 100,
                    "vendor_name": "Slack",
                    "run_id": "run-repair-1",
                    "metadata": {"vendor_name": "Slack", "source": "reddit"},
                    "created_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_reasoning_synthesis",
                    "cost_usd": Decimal("0.20"),
                    "billable_input_tokens": 1200,
                    "output_tokens": 300,
                    "vendor_name": "Slack",
                    "run_id": "run-reason-1",
                    "metadata": {"vendor_name": "Slack"},
                    "created_at": datetime(2026, 3, 31, 22, 10, tzinfo=timezone.utc),
                },
                {
                    "span_name": "b2b.churn_intelligence.battle_card_sales_copy",
                    "cost_usd": Decimal("0.07"),
                    "billable_input_tokens": 950,
                    "output_tokens": 260,
                    "vendor_name": "Slack",
                    "run_id": "run-battle-1",
                    "metadata": {
                        "vendor_name": "Slack",
                        "source_name": "b2b_battle_cards",
                        "event_type": "llm_overlay",
                    },
                    "created_at": datetime(2026, 3, 31, 22, 15, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_enrichment.tier1",
                    "cost_usd": Decimal("0.08"),
                    "billable_input_tokens": 350,
                    "output_tokens": 110,
                    "vendor_name": None,
                    "run_id": "run-enrich-2",
                    "metadata": {"vendor_name": "Zoom", "source": "g2"},
                    "created_at": datetime(2026, 3, 31, 21, 57, tzinfo=timezone.utc),
                },
                {
                    "span_name": "task.b2b_enrichment.tier2",
                    "cost_usd": Decimal("0.01"),
                    "billable_input_tokens": 450,
                    "output_tokens": 70,
                    "vendor_name": None,
                    "run_id": "run-enrich-2",
                    "metadata": {"vendor_name": "Zoom", "source": "g2"},
                    "created_at": datetime(2026, 3, 31, 21, 58, tzinfo=timezone.utc),
                },
            ]
        if "SELECT\n          e.id AS execution_id," in query and "JOIN scheduled_tasks t ON t.id = e.task_id" in query:
            return [
                {
                    "execution_id": "run-enrich-1",
                    "task_name": "b2b_enrichment",
                    "started_at": datetime(2026, 3, 31, 21, 55, tzinfo=timezone.utc),
                    "result_text": '{"reviews_processed": 10, "witness_rows": 8, "witness_count": 15, "generated": 8, "strict_discussion_candidates_kept": 4}',
                },
                {
                    "execution_id": "run-repair-1",
                    "task_name": "b2b_enrichment_repair",
                    "started_at": datetime(2026, 3, 31, 22, 5, tzinfo=timezone.utc),
                    "result_text": '{"promoted": 3, "shadowed": 1, "failed": 0, "witness_count": 8, "secondary_write_hits": 1, "strict_discussion_candidates_dropped": 3, "low_signal_discussion_skipped": 3, "generated": 4}',
                },
                {
                    "execution_id": "run-reason-1",
                    "task_name": "b2b_reasoning_synthesis",
                    "started_at": datetime(2026, 3, 31, 22, 10, tzinfo=timezone.utc),
                    "result_text": '{"vendors_reasoned": 2, "vendors_skipped": 4, "cross_vendor_succeeded": 1, "witness_count": 12, "generated": 3}',
                },
                {
                    "execution_id": "run-battle-1",
                    "task_name": "b2b_battle_cards",
                    "started_at": datetime(2026, 3, 31, 22, 15, tzinfo=timezone.utc),
                    "result_text": '{"cards_built": 2, "cards_llm_updated": 1, "cache_hits": 1, "llm_failures": 1}',
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
                    "result_text": '{"cache_hits": 2, "cards_llm_updated": 1, "llm_failures": 1}',
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
        if (
            "FROM llm_usage" in query
            and "LOWER(" in query
            and "source_name" in query
            and "event_type" in query
            and "entity_type" in query
        ):
            return [{
                "id": uuid4(),
                "run_id": "run-battle-1",
                "span_name": "b2b.churn_intelligence.battle_card_sales_copy",
                "operation_type": "llm_call",
                "model_name": "anthropic/claude-sonnet-4-6",
                "model_provider": "openrouter",
                "input_tokens": 14800,
                "billable_input_tokens": 9900,
                "cached_tokens": 2800,
                "cache_write_tokens": 0,
                "output_tokens": 1900,
                "total_tokens": 16700,
                "cost_usd": Decimal("0.0337"),
                "duration_ms": 1012,
                "ttft_ms": 201,
                "inference_time_ms": 711,
                "queue_time_ms": 9,
                "tokens_per_second": 31.4,
                "status": "completed",
                "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "provider_request_id": "req_battle_123",
                "metadata": {
                    "vendor_name": "Slack",
                    "source_name": "b2b_battle_cards",
                    "event_type": "llm_overlay",
                    "entity_type": "battle_card",
                    "entity_id": "Slack",
                },
                "created_at": datetime(2026, 3, 31, 22, 15, tzinfo=timezone.utc),
            }]
        return [{
            "id": uuid4(),
            "run_id": "run-blog-1",
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


def test_cost_by_operation_accepts_recent_context_filters(monkeypatch):
    client, pool = _client(monkeypatch)
    with client:
        res = client.get(
            "/admin/costs/by-operation?days=7&source_name=b2b_battle_cards&event_type=llm_overlay&entity_type=battle_card&limit=25"
        )
    assert res.status_code == 200
    assert "LOWER(" in pool.last_fetch_query
    assert "source_name" in pool.last_fetch_query
    assert "event_type" in pool.last_fetch_query
    assert "entity_type" in pool.last_fetch_query


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
    assert row["run_id"] == "run-blog-1"
    assert row["vendor_name"] == "HubSpot"


def test_recent_calls_filters_and_surfaces_battle_card_context(monkeypatch):
    client, pool = _client(monkeypatch)
    with client:
        res = client.get(
            "/admin/costs/recent?days=7&source_name=b2b_battle_cards&event_type=llm_overlay&entity_type=battle_card&limit=10"
        )
    assert res.status_code == 200
    row = res.json()["calls"][0]
    assert row["title"] == "Battle Card Sales Copy"
    assert row["run_id"] == "run-battle-1"
    assert row["vendor_name"] == "Slack"
    assert row["source_name"] == "b2b_battle_cards"
    assert row["event_type"] == "llm_overlay"
    assert row["entity_type"] == "battle_card"
    assert row["entity_id"] == "Slack"
    assert "LOWER(" in pool.last_fetch_query
    assert "source_name" in pool.last_fetch_query
    assert "event_type" in pool.last_fetch_query
    assert "entity_type" in pool.last_fetch_query


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
    assert tasks["b2b_battle_cards"]["generated"] == 1
    assert tasks["b2b_battle_cards"]["overlay_failures"] == 1
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
    assert body["calls"][0]["run_id"] == run_id
    assert body["calls"][0]["vendor_name"] == "HubSpot"
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
    assert body["summary"]["tracked_cost_usd"] == 0.37
    assert body["summary"]["cost_per_witness_usd"] == pytest.approx(0.010571, rel=1e-3)
    assert body["token_summary"]["total_billable_input_tokens"] == 4750
    assert body["token_summary"]["total_output_tokens"] == 1050

    by_pass = {row["key"]: row for row in body["token_summary"]["by_pass"]}
    assert by_pass["extraction"]["calls"] == 4
    assert by_pass["extraction"]["billable_input_tokens"] == 2100
    assert by_pass["extraction"]["output_tokens"] == 390
    assert by_pass["repair"]["billable_input_tokens"] == 500
    assert by_pass["reasoning"]["billable_input_tokens"] == 1200
    assert by_pass["battle_card_overlay"]["billable_input_tokens"] == 950

    by_tier = {row["key"]: row for row in body["token_summary"]["enrichment_tiers"]}
    assert by_tier["tier1"]["calls"] == 2
    assert by_tier["tier1"]["billable_input_tokens"] == 1050
    assert by_tier["tier1"]["output_tokens"] == 230
    assert by_tier["tier2"]["calls"] == 2
    assert by_tier["tier2"]["billable_input_tokens"] == 1050
    assert by_tier["tier2"]["output_tokens"] == 160

    vendor_row = body["vendor_passes"][0]
    assert vendor_row["vendor_name"] == "Slack"
    assert vendor_row["extraction_cost_usd"] == pytest.approx(0.12)
    assert vendor_row["repair_cost_usd"] == 0.05
    assert vendor_row["reasoning_cost_usd"] == 0.2
    assert vendor_row["battle_card_overlay_cost_usd"] == 0.07
    assert vendor_row["battle_card_overlay_calls"] == 1
    assert vendor_row["total_cost_usd"] == pytest.approx(0.44)

    source_rows = {row["source"]: row for row in body["source_efficiency"]}
    assert source_rows["reddit"]["total_cost_usd"] == pytest.approx(0.17)
    assert source_rows["reddit"]["witness_yield_rate"] == pytest.approx(1.8)
    assert source_rows["reddit"]["repair_trigger_rate"] == pytest.approx(0.2)
    assert source_rows["reddit"]["cost_per_witness_usd"] == pytest.approx(0.000944, rel=1e-3)
    assert source_rows["reddit"]["strict_discussion_candidates_kept_rows"] == 18
    assert source_rows["reddit"]["low_signal_discussion_skipped_rows"] == 12
    assert source_rows["g2"]["total_cost_usd"] == pytest.approx(0.09)

    run_rows = {row["run_id"]: row for row in body["recent_runs"]}
    assert run_rows["run-enrich-1"]["task_name"] == "b2b_enrichment"
    assert run_rows["run-enrich-1"]["total_cost_usd"] == 0.12
    assert run_rows["run-enrich-1"]["total_billable_input_tokens"] == 1300
    assert run_rows["run-enrich-1"]["total_output_tokens"] == 210
    assert run_rows["run-enrich-1"]["witness_count"] == 15
    assert run_rows["run-enrich-1"]["strict_discussion_candidates_kept"] == 4
    assert run_rows["run-enrich-1"]["enrichment_tier1_billable_input_tokens"] == 700
    assert run_rows["run-enrich-1"]["enrichment_tier1_output_tokens"] == 120
    assert run_rows["run-enrich-1"]["enrichment_tier2_billable_input_tokens"] == 600
    assert run_rows["run-enrich-1"]["enrichment_tier2_output_tokens"] == 90
    assert run_rows["run-repair-1"]["secondary_write_hits"] == 1
    assert run_rows["run-repair-1"]["strict_discussion_candidates_dropped"] == 3
    assert run_rows["run-repair-1"]["repair_billable_input_tokens"] == 500
    assert run_rows["run-repair-1"]["repair_output_tokens"] == 100
    assert run_rows["run-reason-1"]["reasoning_cost_usd"] == 0.2
    assert run_rows["run-reason-1"]["reasoning_billable_input_tokens"] == 1200
    assert run_rows["run-reason-1"]["reasoning_output_tokens"] == 300
    assert run_rows["run-battle-1"]["task_name"] == "b2b_battle_cards"
    assert run_rows["run-battle-1"]["battle_card_overlay_cost_usd"] == 0.07
    assert run_rows["run-battle-1"]["battle_card_overlay_calls"] == 1
    assert run_rows["run-battle-1"]["battle_card_overlay_billable_input_tokens"] == 950
    assert run_rows["run-battle-1"]["battle_card_overlay_output_tokens"] == 260
    assert run_rows["run-battle-1"]["battle_card_cache_hits"] == 1
    assert run_rows["run-battle-1"]["battle_card_llm_updated"] == 1
    assert run_rows["run-battle-1"]["battle_card_llm_failures"] == 1


def test_burn_dashboard_rolls_up_task_runs_and_generic_reasoning(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/burn-dashboard?days=30&top_n=25")
    assert res.status_code == 200
    body = res.json()
    assert body["period_days"] == 30
    assert body["top_n"] == 25
    assert body["summary"]["tracked_cost_usd"] == pytest.approx(2.43)
    assert body["summary"]["model_call_count"] == 19
    assert body["summary"]["recent_runs"] == 5
    assert body["summary"]["rows_processed"] == 24
    assert body["summary"]["rows_reprocessed"] == 5
    assert body["summary"]["reprocess_pct"] == pytest.approx(5 / 24, rel=1e-3)
    assert body["reasoning_budget_pressure"]["vendor_rejections"] == 1
    assert body["reasoning_budget_pressure"]["cross_vendor_rejections"] == 1
    assert body["reasoning_budget_pressure"]["last_rejection_at"] == "2026-03-31T22:30:00+00:00"
    assert body["reasoning_budget_pressure"]["max_vendor_estimated_input_tokens"] == 21000
    assert body["reasoning_budget_pressure"]["max_vendor_cap"] == 20000
    assert body["reasoning_budget_pressure"]["max_cross_vendor_estimated_input_tokens"] == 12437
    assert body["reasoning_budget_pressure"]["max_cross_vendor_cap"] == 12000
    budget_rows = {row["artifact_type"]: row for row in body["reasoning_budget_pressure"]["rows"]}
    assert budget_rows["reasoning_synthesis"]["artifact_label"] == "Vendor reasoning"
    assert budget_rows["cross_vendor_reasoning"]["artifact_label"] == "Cross-vendor reasoning"

    rows = {row["task_name"]: row for row in body["rows"]}
    assert rows["generic_reasoning"]["model_call_count"] == 12
    assert rows["generic_reasoning"]["run_id"] is None
    assert rows["generic_reasoning"]["recent_runs"] is None
    assert rows["generic_reasoning"]["top_trigger_reason"] == "crm_provider | crm.interaction_logged"
    assert rows["b2b_enrichment"]["rows_processed"] == 10
    assert rows["b2b_enrichment"]["run_id"] == "run-enrich-1"
    assert rows["b2b_enrichment"]["avg_cost_per_successful_item"] == pytest.approx(0.0125)
    assert rows["b2b_enrichment"]["top_trigger_reason"] == "No new reviews pending enrichment"
    assert rows["b2b_enrichment_repair"]["rows_processed"] == 5
    assert rows["b2b_enrichment_repair"]["rows_skipped"] == 3
    assert rows["b2b_enrichment_repair"]["rows_reprocessed"] == 5
    assert rows["b2b_enrichment_repair"]["retry_count"] == 1
    assert rows["b2b_enrichment_repair"]["reprocess_pct"] == pytest.approx(1.0)
    assert rows["b2b_enrichment_repair"]["top_trigger_reason"] == "strict_discussion_gate"
    assert rows["b2b_battle_cards"]["run_id"] == "run-battle-1"
    assert rows["b2b_reasoning_synthesis"]["recent_runs"] == 2
    assert rows["b2b_reasoning_synthesis"]["last_status"] == "manual"
    assert rows["b2b_reasoning_synthesis"]["model_call_count"] == 2
    assert rows["b2b_reasoning_synthesis"]["total_cost_usd"] == pytest.approx(0.35)
    assert rows["b2b_reasoning_synthesis"]["rows_processed"] == 7
    assert rows["b2b_reasoning_synthesis"]["rows_skipped"] == 4
    assert rows["b2b_reasoning_synthesis"]["successful_items"] == 3
    assert rows["b2b_reasoning_synthesis"]["top_trigger_reason"] == "thin_specific_witness_pool"
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["last_run_at"] == "2026-03-31T22:31:00+00:00"
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["last_status"] == "completed"
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["model_call_count"] == 2
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["total_cost_usd"] == pytest.approx(0.11)
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["rows_processed"] == 2
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["rows_skipped"] == 1
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["successful_items"] == 1
    assert rows["b2b_reasoning_synthesis.cross_vendor"]["top_trigger_reason"] == "input_budget"


def test_generic_reasoning_rolls_up_sources_events_and_entities(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/generic-reasoning?days=30&top_n=5")
    assert res.status_code == 200
    body = res.json()
    assert body["period_days"] == 30
    assert body["summary"]["total_cost_usd"] == 1.75
    assert body["summary"]["total_calls"] == 12
    assert body["summary"]["top_source_name"] == "crm_provider"
    assert body["summary"]["top_event_type"] == "crm.interaction_logged"
    assert body["by_source"][0]["source_name"] == "crm_provider"
    assert body["by_event_type"][0]["event_type"] == "crm.interaction_logged"
    assert body["top_source_events"][0]["source_name"] == "crm_provider"
    assert body["top_entities"][0]["entity_id"] == "contact-123"


def test_reconciliation_returns_missing_provider_data_state(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/reconciliation?days=30")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "missing_provider_data"
    assert body["summary"]["tracked_cost_usd"] == pytest.approx(2.6)
    assert body["summary"]["provider_cost_usd"] is None
    assert body["daily_rows"][0]["provider"] == "openrouter"
    assert body["daily_rows"][0]["status"] == "missing_provider_data"


def test_reconciliation_merges_provider_daily_rows_and_snapshot_deltas(monkeypatch):
    client, pool = _client(monkeypatch)
    pool.provider_daily_rows = [
        {
            "provider": "anthropic",
            "day": "2026-03-31",
            "provider_cost_usd": Decimal("0.40"),
            "imported_at": datetime(2026, 3, 31, 23, 0, tzinfo=timezone.utc),
        }
    ]
    pool.provider_snapshot_rows = [
        {
            "provider": "openrouter",
            "day": "2026-03-31",
            "provider_cost_usd": Decimal("2.00"),
            "imported_at": datetime(2026, 3, 31, 23, 5, tzinfo=timezone.utc),
        },
        {
            "provider": "openrouter",
            "day": "2026-03-30",
            "provider_cost_usd": Decimal("0.95"),
            "imported_at": datetime(2026, 3, 30, 23, 5, tzinfo=timezone.utc),
        },
    ]
    with client:
        res = client.get("/admin/costs/reconciliation?days=30")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "mixed_provider_data"
    assert body["summary"]["tracked_cost_usd"] == pytest.approx(2.6)
    assert body["summary"]["provider_cost_usd"] == pytest.approx(3.35)
    assert body["summary"]["delta_cost_usd"] == pytest.approx(0.75)

    rows = {(row["date"], row["provider"]): row for row in body["daily_rows"]}
    assert rows[("2026-03-31", "openrouter")]["status"] == "provider_snapshot_delta"
    assert rows[("2026-03-31", "openrouter")]["delta_cost_usd"] == pytest.approx(0.25)
    assert rows[("2026-03-30", "openrouter")]["status"] == "provider_snapshot_delta"
    assert rows[("2026-03-30", "openrouter")]["delta_cost_usd"] == pytest.approx(0.1)
    assert rows[("2026-03-31", "anthropic")]["status"] == "provider_daily_cost"
    assert rows[("2026-03-31", "anthropic")]["tracked_cost_usd"] == pytest.approx(0.0)
    assert rows[("2026-03-31", "anthropic")]["provider_cost_usd"] == pytest.approx(0.4)


def test_reasoning_activity_exposes_legacy_stratified_spend(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/reasoning-activity?days=30")
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["total_cost_usd"] == pytest.approx(0.56)
    assert body["summary"]["total_calls"] == 4
    assert body["phases"][0]["span_name"] == "reasoning.stratified.reason"
