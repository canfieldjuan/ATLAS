from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import atlas_brain.autonomous.tasks.content_ops_deflection_report_delivery as mod


def _set_delivery_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    values = {
        "enabled": True,
        "interval_seconds": 300,
        "limit": 7,
        "dry_run": True,
        "from_email": "Atlas Content Ops <reports@example.com>",
        "reply_to": "support@example.com",
        "subject": "Your FAQ deflection report is ready",
        "result_base_url": "https://juancanfield.com",
        "result_url_template": "",
        "resend_api_key": "",
        "resend_api_url": "https://resend.example.test/emails",
        "resend_timeout_seconds": 3.0,
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setattr(mod.settings.deflection_delivery, name, value, raising=False)


def test_deflection_delivery_task_is_registered_builtin() -> None:
    from atlas_brain.autonomous.tasks import _BUILTIN_TASKS

    assert (
        "content_ops_deflection_report_delivery",
        "run",
        "content_ops_deflection_report_delivery",
    ) in _BUILTIN_TASKS


def test_deflection_delivery_default_task_seed_is_disabled_interval() -> None:
    from atlas_brain.autonomous.scheduler import TaskScheduler

    seed = next(
        (
            task
            for task in TaskScheduler._DEFAULT_TASKS
            if task.get("name") == "content_ops_deflection_report_delivery"
        ),
        None,
    )

    assert seed is not None
    assert seed["task_type"] == "builtin"
    assert seed["schedule_type"] == "interval"
    assert seed["interval_seconds"] is None
    assert seed["enabled"] is False
    assert seed["timeout_seconds"] == 300
    assert seed["metadata"]["builtin_handler"] == "content_ops_deflection_report_delivery"
    assert seed["metadata"]["notify_tags"] == "email,content_ops,deflection"


def test_deflection_delivery_registers_on_runner() -> None:
    class _Recorder:
        def __init__(self) -> None:
            self.registered: dict[str, object] = {}

        def register_builtin(self, name: str, handler: object) -> None:
            self.registered[name] = handler

    from atlas_brain.autonomous.tasks import register_builtin_tasks

    runner = _Recorder()
    register_builtin_tasks(runner)

    handler = runner.registered["content_ops_deflection_report_delivery"]
    assert callable(handler)
    assert getattr(handler, "__name__", "") == "run"
    assert handler.__module__.endswith("content_ops_deflection_report_delivery")


@pytest.mark.asyncio
async def test_run_skips_when_deflection_delivery_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_delivery_settings(monkeypatch, enabled=False)
    get_pool = AsyncMock()
    monkeypatch.setattr(mod, "get_db_pool", get_pool)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result == {"_skip_synthesis": "Deflection report delivery disabled"}
    get_pool.assert_not_called()


@pytest.mark.asyncio
async def test_run_fails_closed_before_db_when_required_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_delivery_settings(
        monkeypatch,
        from_email=" ",
        subject=" ",
        result_base_url="",
        result_url_template="",
    )
    get_pool = AsyncMock()
    monkeypatch.setattr(mod, "get_db_pool", get_pool)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result == {
        "_skip_synthesis": "Deflection report delivery config missing",
        "missing": ["from_email", "subject", "result_base_url_or_template"],
        "dry_run": True,
    }
    get_pool.assert_not_called()


@pytest.mark.asyncio
async def test_run_delegates_to_queue_worker_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_delivery_settings(monkeypatch, result_url_template="https://juancanfield.com/result/{request_id}")
    pool = SimpleNamespace(is_initialized=True)
    calls: list[dict[str, Any]] = []

    async def _send_pending(pool_arg: Any, *, sender: Any, config: Any) -> Any:
        calls.append(
            {
                "pool": pool_arg,
                "sender_type": type(sender).__name__,
                "from_email": config.from_email,
                "reply_to": config.reply_to,
                "subject": config.subject,
                "result_base_url": config.result_base_url,
                "result_url_template": config.result_url_template,
                "limit": config.limit,
                "dry_run": config.dry_run,
            }
        )
        return SimpleNamespace(scanned=1, sent=0, failed=0, dry_run=1)

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "send_pending_deflection_report_deliveries", _send_pending)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result == {
        "_skip_synthesis": "Deflection report delivery complete",
        "scanned": 1,
        "sent": 0,
        "failed": 0,
        "dry_run": 1,
        "dry_run_enabled": True,
    }
    assert calls == [
        {
            "pool": pool,
            "sender_type": "_DryRunSender",
            "from_email": "Atlas Content Ops <reports@example.com>",
            "reply_to": "support@example.com",
            "subject": "Your FAQ deflection report is ready",
            "result_base_url": "https://juancanfield.com",
            "result_url_template": "https://juancanfield.com/result/{request_id}",
            "limit": 7,
            "dry_run": True,
        }
    ]


@pytest.mark.asyncio
async def test_run_uses_resend_sender_for_live_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_delivery_settings(monkeypatch, dry_run=False, resend_api_key="re_live")
    pool = SimpleNamespace(is_initialized=True)
    sender = object()
    calls: list[dict[str, Any]] = []

    def _create_sender(provider: str, config: dict[str, Any]) -> object:
        calls.append({"provider": provider, "config": config})
        return sender

    async def _send_pending(pool_arg: Any, *, sender: Any, config: Any) -> Any:
        calls.append(
            {
                "pool": pool_arg,
                "sender": sender,
                "dry_run": config.dry_run,
                "result_base_url": config.result_base_url,
            }
        )
        return SimpleNamespace(scanned=0, sent=0, failed=0, dry_run=0)

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "create_campaign_sender", _create_sender)
    monkeypatch.setattr(mod, "send_pending_deflection_report_deliveries", _send_pending)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result["_skip_synthesis"] == "No pending deflection report deliveries"
    assert result["dry_run_enabled"] is False
    assert calls == [
        {
            "provider": "resend",
            "config": {
                "api_key": "re_live",
                "api_url": "https://resend.example.test/emails",
                "timeout_seconds": 3.0,
            },
        },
        {
            "pool": pool,
            "sender": sender,
            "dry_run": False,
            "result_base_url": "https://juancanfield.com",
        },
    ]
