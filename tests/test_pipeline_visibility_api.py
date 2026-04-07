"""Focused API tests for pipeline visibility auth and audit behavior."""

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import atlas_brain.api.pipeline_visibility as visibility_api
import atlas_brain.autonomous.visibility as visibility_helpers
from atlas_brain.auth.dependencies import AuthUser, require_auth
from atlas_brain.config import settings


def _make_app():
    app = FastAPI()
    app.include_router(visibility_api.router)
    return app


def _auth_user() -> AuthUser:
    return AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )


def test_visibility_summary_requires_auth(monkeypatch):
    monkeypatch.setattr(settings.saas_auth, "enabled", True, raising=False)
    app = _make_app()

    with TestClient(app) as client:
        response = client.get("/pipeline/visibility/summary")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_visibility_summary_accepts_30_day_window(monkeypatch):
    app = _make_app()
    app.dependency_overrides[require_auth] = _auth_user

    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {"open_actionable": 2, "open_total": 5},
            {
                "failures": 7,
                "quarantines": 3,
                "rejections": 1,
                "recovered_validation_retries": 4,
            },
        ]
    )
    pool.fetch = AsyncMock(return_value=[])
    pool.execute = AsyncMock(return_value="UPDATE 0")
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: pool)
    monkeypatch.setattr(settings.b2b_campaign, "anthropic_batch_detached_enabled", False, raising=False)

    with TestClient(app) as client:
        response = client.get("/pipeline/visibility/summary?hours=720")

    assert response.status_code == 200
    body = response.json()
    assert body["period_hours"] == 720
    assert body["failures_period"] == 7
    assert body["recovered_validation_retries_period"] == 4
    assert pool.fetchrow.await_args_list[0].args[1] == 720


def test_synthesis_validation_endpoint_returns_normalized_rows(monkeypatch):
    app = _make_app()
    app.dependency_overrides[require_auth] = _auth_user

    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": str(uuid4()),
                "vendor_name": "Shopify",
                "as_of_date": "2026-03-30",
                "analysis_window_days": 90,
                "schema_version": "v2",
                "run_id": "run-1",
                "attempt_no": 2,
                "rule_code": "scope_ambiguity",
                "severity": "warning",
                "passed": False,
                "summary": "Differentiate charted source data from broader displacement data.",
                "field_path": "content.sections[1]",
                "detail": {"vendors": ["WooCommerce"]},
                "scope_manifest": {
                    "selection_strategy": "vendor_facet_packet_v1",
                    "reviews_in_scope": 9,
                    "witnesses_in_scope": 12,
                },
                "reasoning_delta": {
                    "wedge_changed": True,
                    "new_timing_windows": ["renewal_window"],
                },
                "payload_component_tokens": {
                    "witness_pack": 3100,
                    "section_packets": 900,
                },
                "evidence_hash": "abc123hash",
                "created_at": "2026-03-30T22:00:00+00:00",
            }
        ]
    )
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.get("/pipeline/visibility/synthesis-validation?vendor_name=Shopify")

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["vendor_name"] == "Shopify"
    assert body["results"][0]["rule_code"] == "scope_ambiguity"
    assert body["results"][0]["detail"]["vendors"] == ["WooCommerce"]
    assert body["results"][0]["scope_manifest"]["witnesses_in_scope"] == 12
    assert body["results"][0]["reasoning_delta"]["wedge_changed"] is True
    assert body["results"][0]["payload_component_tokens"]["witness_pack"] == 3100
    assert body["results"][0]["evidence_hash"] == "abc123hash"


def test_synthesis_validation_retry_only_joins_attempts(monkeypatch):
    app = _make_app()
    app.dependency_overrides[require_auth] = _auth_user

    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.get("/pipeline/visibility/synthesis-validation?retry_only=true")

    assert response.status_code == 200
    query = pool.fetch.await_args.args[0]
    assert "FROM artifact_attempts a_rejected" in query
    assert "FROM artifact_attempts a_succeeded" in query
    assert "a_succeeded.status = 'succeeded'" in query


def test_extraction_health_endpoint_returns_audit_payload(monkeypatch):
    app = _make_app()
    app.dependency_overrides[require_auth] = _auth_user

    summary = {
        "days": 30,
        "top_n": 10,
        "current_snapshot": {
            "enriched_rows": 25277,
            "rows_with_spans": 21000,
            "span_count": 50554,
            "witness_yield_rate": 2.0,
            "repair_triggered_rows": 3200,
            "repair_promoted_rows": 1400,
            "repair_trigger_rate": 0.1266,
            "repair_promoted_rate": 0.0554,
            "secondary_write_hits_window": 12,
            "hard_gap_rows": 0,
            "phrase_arrays_without_spans": 0,
            "blank_replacement_mode": 0,
            "blank_operating_model_shift": 0,
            "blank_productivity_delta_claim": 0,
            "blank_org_pressure_type": 0,
            "missing_or_empty_evidence_spans": 0,
            "blank_evidence_map_hash": 0,
            "empty_salience_flags": 10624,
        },
        "daily_trend": [],
        "top_vendors": [],
        "top_sources": [],
        "recent_runs": [],
    }
    summarize = AsyncMock(return_value=summary)
    monkeypatch.setattr(visibility_api, "summarize_extraction_health", summarize)
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: MagicMock())

    with TestClient(app) as client:
        response = client.get("/pipeline/visibility/extraction-health?days=30&top_n=12")

    assert response.status_code == 200
    body = response.json()
    assert body["current_snapshot"]["enriched_rows"] == 25277
    assert body["current_snapshot"]["empty_salience_flags"] == 10624
    assert summarize.await_args.kwargs["days"] == 30
    assert summarize.await_args.kwargs["top_n"] == 12


def test_visibility_queue_runs_detached_batch_health_sync(monkeypatch):
    app = _make_app()
    app.dependency_overrides[require_auth] = _auth_user

    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock(return_value="UPDATE 0")
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: pool)
    sync = AsyncMock(return_value=None)
    monkeypatch.setattr(visibility_api, "_sync_detached_batch_health", sync)

    with TestClient(app) as client:
        response = client.get("/pipeline/visibility/queue")

    assert response.status_code == 200
    assert sync.await_count == 1


@pytest.mark.asyncio
async def test_sync_detached_batch_health_emits_stale_batch_and_scheduler_events(monkeypatch):
    class _Pool:
        def __init__(self):
            self.executed = []

        async def fetch(self, query, *args):
            if "FROM anthropic_message_batches" in query and "completed_at IS NULL" in query:
                return [
                    {
                        "id": str(uuid4()),
                        "run_id": "run-batch-1",
                        "status": "in_progress",
                        "provider_batch_id": "msgbatch_123",
                        "submitted_items": 10,
                        "completed_items": 2,
                        "failed_items": 1,
                        "fallback_single_call_items": 0,
                        "stale_since": datetime.now(timezone.utc) - timedelta(minutes=61),
                    }
                ]
            if "FROM anthropic_message_batch_items i" in query and "request_metadata->>'applying_at'" in query:
                return [
                    {
                        "id": str(uuid4()),
                        "batch_id": str(uuid4()),
                        "custom_id": "campaign:slack:email",
                        "artifact_id": "batch-1:Slack:email",
                        "status": "batch_succeeded",
                        "applying_by": "reconcile:task-1:abc123",
                        "applying_at": datetime.now(timezone.utc) - timedelta(minutes=64),
                        "run_id": "run-batch-1",
                        "provider_batch_id": "msgbatch_123",
                    }
                ]
            if "FROM pipeline_visibility_reviews r" in query and "e.reason_code = ANY" in query:
                return []
            raise AssertionError(f"Unexpected fetch query: {query}")

        async def fetchrow(self, query, *args):
            if "FROM scheduled_tasks t" in query:
                return {
                    "id": str(uuid4()),
                    "name": "b2b_campaign_batch_reconciliation",
                    "enabled": True,
                    "interval_seconds": 300,
                    "last_run_at": datetime.now(timezone.utc) - timedelta(minutes=25),
                    "next_run_at": datetime.now(timezone.utc) - timedelta(minutes=15),
                    "last_status": "failed",
                    "last_error": "network timeout",
                    "last_started_at": datetime.now(timezone.utc) - timedelta(minutes=26),
                }
            if "FROM pipeline_visibility_reviews r" in query and "e.stage = $1" in query:
                return None
            raise AssertionError(f"Unexpected fetchrow query: {query}")

        async def execute(self, query, *args):
            self.executed.append((query, args))
            return "UPDATE 0"

    pool = _Pool()
    emitted = []

    async def _fake_emit_event(*args, **kwargs):
        emitted.append(kwargs)
        return str(uuid4())

    monkeypatch.setattr(settings.b2b_campaign, "anthropic_batch_detached_enabled", True, raising=False)
    monkeypatch.setattr(visibility_api, "emit_event", _fake_emit_event)

    await visibility_api._sync_detached_batch_health(pool)

    reason_codes = {item["reason_code"] for item in emitted}
    assert "detached_batch_stale" in reason_codes
    assert "detached_batch_item_claim_stale" in reason_codes
    assert "detached_batch_reconciliation_failed" in reason_codes
    stale_event = next(item for item in emitted if item["reason_code"] == "detached_batch_stale")
    assert stale_event["entity_type"] == "batch_job"
    assert stale_event["severity"] == "error"
    stale_claim_event = next(item for item in emitted if item["reason_code"] == "detached_batch_item_claim_stale")
    assert stale_claim_event["entity_type"] == "batch_item"
    assert stale_claim_event["detail"]["applying_by"] == "reconcile:task-1:abc123"
    scheduler_event = next(item for item in emitted if item["reason_code"] == "detached_batch_reconciliation_failed")
    assert scheduler_event["entity_type"] == "task"
    assert scheduler_event["detail"]["last_error"] == "network timeout"


def test_resolve_review_records_actor_identity(monkeypatch):
    app = _make_app()
    user = _auth_user()
    app.dependency_overrides[require_auth] = lambda: user

    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "id": str(uuid4()),
            "fingerprint": "abc123",
            "entity_type": "blog_post",
            "entity_id": "switch-to-shopify-2026-03",
        }
    )
    pool.execute = AsyncMock(return_value="UPDATE 1")
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: pool)

    record_action = AsyncMock(return_value=str(uuid4()))
    emit_event = AsyncMock(return_value=str(uuid4()))
    monkeypatch.setattr(visibility_api, "record_review_action", record_action)
    monkeypatch.setattr(visibility_api, "emit_event", emit_event)

    review_id = uuid4()
    with TestClient(app) as client:
        response = client.post(
            f"/pipeline/visibility/reviews/{review_id}/resolve",
            params={"action": "resolve", "note": "Reviewed and accepted"},
        )

    assert response.status_code == 200
    assert response.json()["action"] == "resolved"
    assert record_action.await_args.kwargs["actor_id"] == user.user_id
    assert emit_event.await_args.kwargs["detail"]["actor_id"] == user.user_id
    assert pool.execute.await_args.args[-1] == user.user_id


@pytest.mark.asyncio
async def test_record_dedup_persists_row_before_event(monkeypatch):
    pool = MagicMock()
    pool.execute = AsyncMock(return_value="INSERT 0 1")
    emit_event = AsyncMock(return_value=str(uuid4()))
    monkeypatch.setattr(visibility_helpers, "emit_event", emit_event)

    event_id = await visibility_helpers.record_dedup(
        pool,
        stage="blog",
        entity_type="blog_post",
        entity_id="switch-to-shopify-2026-03",
        survivor_entity_id="published:switch-to-shopify-2026-03",
        reason="Slug already published",
        run_id="run-99",
        detail={"slug": "switch-to-shopify-2026-03"},
    )

    assert event_id is not None
    insert_call = pool.execute.await_args_list[0]
    assert "INSERT INTO dedup_decisions" in insert_call.args[0]
    assert insert_call.args[1] == "run-99"
    assert insert_call.args[4] == "published:switch-to-shopify-2026-03"
    assert insert_call.args[5] == "switch-to-shopify-2026-03"
    assert emit_event.await_args.kwargs["event_type"] == "dedup_discard"
    assert emit_event.await_args.kwargs["entity_id"] == "switch-to-shopify-2026-03"


def test_dedup_decisions_endpoint_returns_rows(monkeypatch):
    app = _make_app()
    app.dependency_overrides[require_auth] = _auth_user

    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": str(uuid4()),
                "run_id": "run-12",
                "stage": "blog",
                "entity_type": "blog_post",
                "survivor_entity_id": "published:switch-to-shopify-2026-03",
                "discarded_entity_id": "switch-to-shopify-2026-03",
                "reason_code": "dedup_discard",
                "comparison_metrics": {"slug": "switch-to-shopify-2026-03"},
                "actor_type": "system",
                "actor_id": None,
                "decided_at": "2026-03-30T22:00:00+00:00",
            }
        ]
    )
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: pool)

    with TestClient(app) as client:
        response = client.get("/pipeline/visibility/dedup-decisions?stage=blog")

    assert response.status_code == 200
    body = response.json()
    assert body["decisions"][0]["reason_code"] == "dedup_discard"
    assert body["decisions"][0]["discarded_entity_id"] == "switch-to-shopify-2026-03"
