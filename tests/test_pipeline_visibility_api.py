"""Focused API tests for pipeline visibility auth and audit behavior."""

from unittest.mock import AsyncMock, MagicMock
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
    monkeypatch.setattr(visibility_api, "get_db_pool", lambda: pool)

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
