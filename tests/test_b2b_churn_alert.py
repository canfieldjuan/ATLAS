"""Tests for the B2B churn alert task."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_settings(
    *,
    enabled: bool = True,
    email_enabled: bool = True,
    ntfy_enabled: bool = False,
    dashboard_base_url: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        b2b_alert=SimpleNamespace(
            enabled=enabled,
            email_enabled=email_enabled,
            sender_name="Atlas Intelligence",
            dashboard_base_url=dashboard_base_url,
            signal_count_threshold=3,
            urgency_spike_threshold=1.5,
            cooldown_hours=24,
        ),
        alerts=SimpleNamespace(
            ntfy_enabled=ntfy_enabled,
            ntfy_url="http://localhost:8090",
            ntfy_topic="atlas-alerts",
        ),
        campaign_sequence=SimpleNamespace(
            sender_type="resend",
            resend_api_key="resend-key",
            resend_from_email="outreach@churnsignals.co",
            ses_from_email="",
        ),
    )


def _make_pool(*, accounts: list[dict], current_row: dict | None, baselines: dict[str, dict | None]):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=accounts)

    async def _fetchrow(query: str, *args):
        if "FROM b2b_reviews" in query:
            return current_row
        if "FROM b2b_alert_baselines" in query:
            return baselines.get(args[2])
        return None

    pool.fetchrow = AsyncMock(side_effect=_fetchrow)
    pool.execute = AsyncMock()
    return pool


def _make_task() -> MagicMock:
    task = MagicMock()
    task.name = "b2b_churn_alert"
    return task


def _make_account() -> dict:
    return {
        "account_id": "acct-1",
        "account_name": "Acme Corp",
        "vendor_name": "Salesforce",
        "owner_email": "owner@example.com",
        "owner_name": "Alex Owner",
    }


class TestB2BChurnAlertTask:
    @pytest.mark.asyncio
    async def test_skip_when_disabled(self):
        with patch("atlas_brain.autonomous.tasks.b2b_churn_alert.settings", _make_settings(enabled=False)):
            from atlas_brain.autonomous.tasks.b2b_churn_alert import run

            result = await run(_make_task())

        assert result["_skip_synthesis"] == "B2B churn alerts disabled"

    @pytest.mark.asyncio
    async def test_seeds_baselines_on_first_run(self):
        pool = _make_pool(
            accounts=[_make_account()],
            current_row={"signal_count": 5, "avg_urgency": 4.0, "displacement_count": 1},
            baselines={"signal_count": None, "avg_urgency": None, "displacement_count": None},
        )

        with (
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.settings", _make_settings()),
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.get_db_pool", return_value=pool),
        ):
            from atlas_brain.autonomous.tasks.b2b_churn_alert import run

            result = await run(_make_task())

        assert result["baselines_seeded"] == 3
        assert result["alerts_sent"] == 0
        assert pool.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_sends_owner_email_and_updates_baseline(self):
        pool = _make_pool(
            accounts=[_make_account()],
            current_row={"signal_count": 5, "avg_urgency": 2.0, "displacement_count": 0},
            baselines={
                "signal_count": {"baseline_value": 1, "last_alerted_at": None},
                "avg_urgency": {"baseline_value": 2.0, "last_alerted_at": None},
                "displacement_count": {"baseline_value": 0, "last_alerted_at": None},
            },
        )

        with (
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.settings", _make_settings()),
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.get_db_pool", return_value=pool),
            patch(
                "atlas_brain.autonomous.tasks.b2b_churn_alert._send_alert_notifications",
                AsyncMock(return_value={"delivered": True, "email": True, "ntfy": False}),
            ),
            patch(
                "atlas_brain.autonomous.tasks.b2b_churn_alert._dispatch_alert_webhook",
                AsyncMock(),
            ),
        ):
            from atlas_brain.autonomous.tasks.b2b_churn_alert import run

            result = await run(_make_task())

        assert result["alerts_sent"] == 1
        assert result["email_alerts_sent"] == 1
        assert result["ntfy_alerts_sent"] == 0
        assert result["alerts_failed"] == 0
        assert pool.execute.await_count == 1
        assert "UPDATE b2b_alert_baselines" in pool.execute.await_args.args[0]

    @pytest.mark.asyncio
    async def test_failed_delivery_does_not_advance_baseline(self):
        pool = _make_pool(
            accounts=[_make_account()],
            current_row={"signal_count": 5, "avg_urgency": 2.0, "displacement_count": 0},
            baselines={
                "signal_count": {"baseline_value": 1, "last_alerted_at": None},
                "avg_urgency": {"baseline_value": 2.0, "last_alerted_at": None},
                "displacement_count": {"baseline_value": 0, "last_alerted_at": None},
            },
        )

        with (
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.settings", _make_settings()),
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.get_db_pool", return_value=pool),
            patch(
                "atlas_brain.autonomous.tasks.b2b_churn_alert._send_alert_notifications",
                AsyncMock(return_value={"delivered": False, "email": False, "ntfy": False}),
            ),
        ):
            from atlas_brain.autonomous.tasks.b2b_churn_alert import run

            result = await run(_make_task())

        assert result["alerts_sent"] == 0
        assert result["alerts_failed"] == 1
        assert pool.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_send_alert_email_uses_campaign_sender(self):
        sender = MagicMock()
        sender.send = AsyncMock(return_value={"id": "msg_123"})
        settings_obj = _make_settings(dashboard_base_url="https://churnsignals.co/dashboard")

        with (
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.settings", settings_obj),
            patch("atlas_brain.autonomous.tasks.b2b_churn_alert.get_campaign_sender", return_value=sender),
        ):
            from atlas_brain.autonomous.tasks.b2b_churn_alert import _send_alert_email

            sent = await _send_alert_email(
                account_name="Acme Corp",
                vendor_name="Salesforce",
                metric="signal_count",
                baseline=1.0,
                current=5.0,
                owner_email="owner@example.com",
                owner_name="Alex Owner",
            )

        assert sent is True
        kwargs = sender.send.await_args.kwargs
        assert kwargs["to"] == "owner@example.com"
        assert kwargs["from_email"] == "Atlas Intelligence <outreach@churnsignals.co>"
        assert kwargs["subject"] == "Churn Alert: Salesforce signal spike"
        assert "Acme Corp" in kwargs["body"]
        assert "Salesforce" in kwargs["body"]
        assert "https://churnsignals.co/dashboard" in kwargs["body"]
