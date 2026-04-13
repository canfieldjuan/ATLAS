from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import atlas_brain.autonomous.tasks.b2b_watchlist_alert_delivery as mod
from atlas_brain.services.b2b import watchlist_alerts as watchlist_alert_service


@pytest.mark.asyncio
async def test_run_skips_when_watchlist_delivery_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_watchlist_delivery, "enabled", False, raising=False)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result == {"_skip_synthesis": "Watchlist alert delivery disabled"}


@pytest.mark.asyncio
async def test_run_processes_due_view_and_advances_schedule(monkeypatch):
    now = datetime.now(timezone.utc)
    scheduled_for = now - timedelta(minutes=5)
    account_id = uuid4()
    view_id = uuid4()
    claim_id = uuid4()

    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[
            {
                "id": view_id,
                "account_id": account_id,
                "name": "Daily CRM watch",
                "vendor_name": "Intercom",
                "category": "Helpdesk",
                "source": "reddit",
                "min_urgency": 8.0,
                "include_stale": False,
                "named_accounts_only": True,
                "changed_wedges_only": False,
                "vendor_alert_threshold": 7.5,
                "account_alert_threshold": 8.5,
                "stale_days_threshold": 2,
                "alert_email_enabled": True,
                "alert_delivery_frequency": "daily",
                "next_alert_delivery_at": scheduled_for,
                "last_alert_delivery_at": None,
                "last_alert_delivery_status": None,
                "last_alert_delivery_summary": None,
                "created_at": now,
                "updated_at": now,
                "account_name": "Acme",
                "product": "b2b_retention",
                "plan_status": "active",
            }
        ]),
        fetchrow=AsyncMock(return_value={"id": claim_id}),
        execute=AsyncMock(return_value="UPDATE 1"),
    )

    monkeypatch.setattr(mod.settings.b2b_watchlist_delivery, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_watchlist_delivery, "max_views_per_run", 10, raising=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "get_campaign_sender", lambda: object())
    monkeypatch.setattr(watchlist_alert_service, "watchlist_alert_sender_configured", lambda: True)
    monkeypatch.setattr(
        watchlist_alert_service,
        "evaluate_watchlist_alert_events_for_view",
        AsyncMock(return_value={
            "watchlist_view_id": str(view_id),
            "watchlist_view_name": "Daily CRM watch",
            "evaluated_at": now.isoformat(),
            "events": [
                {"id": str(uuid4()), "summary": "Intercom crossed the vendor alert threshold at 8.1"},
            ],
            "count": 1,
            "new_open_event_count": 1,
            "resolved_event_count": 0,
        }),
    )
    monkeypatch.setattr(
        watchlist_alert_service,
        "resolve_watchlist_alert_recipients",
        AsyncMock(return_value=[{"email": "owner@example.com", "full_name": "Owner"}]),
    )
    monkeypatch.setattr(watchlist_alert_service, "watchlist_alert_from_email", lambda: "Atlas <atlas@example.com>")
    monkeypatch.setattr(
        watchlist_alert_service,
        "send_watchlist_alert_email",
        AsyncMock(return_value=(True, "msg-1", None, False)),
    )
    monkeypatch.setattr(watchlist_alert_service, "update_watchlist_alert_email_log", AsyncMock())
    monkeypatch.setattr(watchlist_alert_service, "render_watchlist_alert_email_html", lambda **_kwargs: "<p>body</p>")

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result["_skip_synthesis"] == "Watchlist alert delivery complete"
    assert result["attempts_claimed"] == 1
    assert result["sent"] == 1
    assert watchlist_alert_service.update_watchlist_alert_email_log.await_count == 1
    update_sql = pool.execute.await_args.args[0]
    assert "UPDATE b2b_watchlist_views" in update_sql


@pytest.mark.asyncio
async def test_scheduled_delivery_reports_blocked_preview_summary_on_no_events(monkeypatch):
    now = datetime.now(timezone.utc)
    account_id = uuid4()
    view_id = uuid4()
    delivery_log_id = uuid4()
    row = {
        "id": view_id,
        "account_id": account_id,
        "name": "Quiet CRM watch",
        "vendor_name": "Salesforce",
        "category": "CRM",
        "source": "reddit",
        "min_urgency": 8.0,
        "include_stale": False,
        "named_accounts_only": True,
        "changed_wedges_only": False,
        "vendor_alert_threshold": 7.5,
        "account_alert_threshold": 6.0,
        "stale_days_threshold": 2,
        "alert_email_enabled": True,
        "alert_delivery_frequency": "daily",
        "next_alert_delivery_at": now,
        "account_name": "Acme",
        "product": "b2b_retention",
    }
    pool = SimpleNamespace()

    monkeypatch.setattr(
        watchlist_alert_service,
        "evaluate_watchlist_alert_events_for_view",
        AsyncMock(
            return_value={
                "watchlist_view_id": str(view_id),
                "watchlist_view_name": "Quiet CRM watch",
                "events": [],
                "suppressed_preview_summary": {"count": 1},
            }
        ),
    )
    monkeypatch.setattr(
        watchlist_alert_service,
        "resolve_watchlist_alert_recipients",
        AsyncMock(return_value=[{"email": "owner@example.com", "full_name": "Owner"}]),
    )
    monkeypatch.setattr(watchlist_alert_service, "update_watchlist_alert_email_log", AsyncMock())
    monkeypatch.setattr(mod, "_advance_view_schedule", AsyncMock())

    status = await mod._send_scheduled_watchlist_email(pool, object(), row, delivery_log_id)

    assert status == "no_events"
    assert watchlist_alert_service.update_watchlist_alert_email_log.await_args.kwargs["summary"] == (
        "No open alert events to deliver (1 preview-backed account alert blocked by policy)"
    )
    assert mod._advance_view_schedule.await_args.kwargs["summary"] == (
        "No open alert events to deliver (1 preview-backed account alert blocked by policy)"
    )


@pytest.mark.asyncio
async def test_claim_delivery_attempt_matches_partial_unique_index(monkeypatch):
    now = datetime.now(timezone.utc)
    row = {
        "account_id": uuid4(),
        "id": uuid4(),
        "next_alert_delivery_at": now,
        "alert_delivery_frequency": "daily",
    }
    pool = SimpleNamespace(fetchrow=AsyncMock(return_value={"id": uuid4()}))

    await mod._claim_delivery_attempt(pool, row)

    sql = pool.fetchrow.await_args.args[0]
    assert "ON CONFLICT (watchlist_view_id, scheduled_for, delivery_mode)" in sql
    assert "WHERE scheduled_for IS NOT NULL" in sql


def test_build_watchlist_alert_candidates_uses_preview_signal_score_for_preview_accounts():
    candidates = watchlist_alert_service.build_watchlist_alert_candidates(
        watchlist_view={
            "account_alert_threshold": 6.0,
        },
        feed={"signals": []},
        accounts_feed={
            "accounts": [
                {
                    "company": "Concentrix",
                    "vendor": "Salesforce",
                    "category": "CRM",
                    "urgency": None,
                    "preview_signal_score": 6.2,
                    "confidence": 0.41,
                    "budget_authority": True,
                    "account_reasoning_preview_only": True,
                    "account_pressure_disclaimer": "Early account signal only.",
                    "account_alert_eligible": True,
                    "account_alert_policy_reason": None,
                    "account_alert_hit": True,
                    "report_date": "2026-04-05",
                    "watch_vendor": "Salesforce",
                    "track_mode": "competitor",
                    "source_distribution": {"reddit": 1},
                    "reasoning_reference_ids": {"witness_ids": ["w1"]},
                    "source_review_ids": ["r1"],
                }
            ]
        },
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["summary"] == (
        "Early account signal for Concentrix crossed the account alert threshold at 6.2"
    )
    assert candidate["payload"]["urgency"] is None
    assert candidate["payload"]["preview_signal_score"] == pytest.approx(6.2)
    assert candidate["payload"]["account_alert_score"] == pytest.approx(6.2)
    assert candidate["payload"]["account_alert_score_source"] == "preview_signal_score"
    assert candidate["payload"]["account_alert_eligible"] is True
    assert candidate["payload"]["account_alert_policy_reason"] is None
    assert candidate["payload"]["account_reasoning_preview_only"] is True
    assert candidate["payload"]["account_pressure_disclaimer"] == "Early account signal only."


def test_serialize_watchlist_alert_event_surfaces_preview_account_fields():
    event_id = uuid4()
    view_id = uuid4()

    result = watchlist_alert_service.serialize_watchlist_alert_event(
        {
            "id": event_id,
            "watchlist_view_id": view_id,
            "event_type": "account_alert",
            "threshold_field": "account_alert_threshold",
            "entity_type": "account",
            "entity_key": "account_alert:account:salesforce:concentrix:crm:reddit:2026-04-05",
            "vendor_name": "Salesforce",
            "company_name": "Concentrix",
            "category": "CRM",
            "source": "reddit",
            "threshold_value": 6.0,
            "summary": "Early account signal for Concentrix crossed the account alert threshold at 6.2",
            "payload": {
                "urgency": None,
                "preview_signal_score": 6.2,
                "account_alert_score": 6.2,
                "account_alert_score_source": "preview_signal_score",
                "account_alert_eligible": True,
                "account_alert_policy_reason": None,
                "account_reasoning_preview_only": True,
                "account_pressure_disclaimer": "Early account signal only.",
                "reasoning_reference_ids": {"witness_ids": ["w1"]},
                "source_review_ids": ["r1"],
                "account_review_focus": {
                    "vendor": "Salesforce",
                    "company": "Concentrix",
                    "report_date": "2026-04-05",
                    "watch_vendor": "Salesforce",
                    "category": "CRM",
                    "track_mode": "competitor",
                },
            },
            "status": "open",
        }
    )

    assert result["account_alert_score"] == pytest.approx(6.2)
    assert result["account_alert_score_source"] == "preview_signal_score"
    assert result["account_alert_eligible"] is True
    assert result["account_alert_policy_reason"] is None
    assert result["preview_signal_score"] == pytest.approx(6.2)
    assert result["account_reasoning_preview_only"] is True
    assert result["account_pressure_disclaimer"] == "Early account signal only."


def test_summarize_suppressed_preview_accounts_reports_blocked_threshold_hits():
    result = watchlist_alert_service.summarize_suppressed_preview_accounts(
        watchlist_view={
            "account_alert_threshold": 6.0,
            "preview_account_alert_policy": {
                "applies_to_preview_only": True,
                "enabled": True,
                "enabled_source": "view",
                "min_confidence": 0.55,
                "min_confidence_source": "view",
                "require_budget_authority": False,
                "require_budget_authority_source": "view",
                "override_min_confidence": 0.55,
                "override_require_budget_authority": False,
            },
        },
        accounts_feed={
            "accounts": [
                {
                    "company": "Concentrix",
                    "vendor": "Salesforce",
                    "category": "CRM",
                    "preview_signal_score": 6.2,
                    "confidence": 0.2,
                    "budget_authority": True,
                    "account_reasoning_preview_only": True,
                    "account_alert_eligible": False,
                    "account_alert_policy_reason": "preview_low_confidence",
                    "source_distribution": {"reddit": 1},
                    "account_pressure_disclaimer": "Early account signal only.",
                },
                {
                    "company": "Later Corp",
                    "vendor": "Salesforce",
                    "category": "CRM",
                    "preview_signal_score": 5.2,
                    "confidence": 0.2,
                    "budget_authority": True,
                    "account_reasoning_preview_only": True,
                    "account_alert_eligible": False,
                    "account_alert_policy_reason": "preview_low_confidence",
                    "source_distribution": {"reddit": 1},
                },
            ]
        },
    )

    assert result["count"] == 1
    assert result["threshold_value"] == pytest.approx(6.0)
    assert result["preview_account_alert_policy"] == {
        "applies_to_preview_only": True,
        "enabled": True,
        "enabled_source": "view",
        "min_confidence": 0.55,
        "min_confidence_source": "view",
        "require_budget_authority": False,
        "require_budget_authority_source": "view",
        "override_min_confidence": 0.55,
        "override_require_budget_authority": False,
    }
    assert result["reason_details"] == {
        "preview_low_confidence": {
            "summary": "Preview-backed account alerts require confidence >= 0.55.",
            "short_summary": "confidence >= 0.55 required",
            "min_confidence": pytest.approx(0.55),
            "min_confidence_source": "view",
        }
    }
    assert result["reasons"] == {"preview_low_confidence": 1}
    assert result["vendors"] == [{"vendor_name": "Salesforce", "count": 1}]
    assert result["accounts"] == [
        {
            "vendor_name": "Salesforce",
            "company_name": "Concentrix",
            "category": "CRM",
            "source": "reddit",
            "account_alert_score": pytest.approx(6.2),
            "account_alert_score_source": "preview_signal_score",
            "account_alert_policy_reason": "preview_low_confidence",
            "confidence": pytest.approx(0.2),
            "budget_authority": True,
            "account_pressure_disclaimer": "Early account signal only.",
        }
    ]
