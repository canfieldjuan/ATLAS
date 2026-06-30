from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "send_content_ops_deflection_report_deliveries.py"
SPEC = importlib.util.spec_from_file_location("send_deflection_deliveries", SCRIPT)
assert SPEC and SPEC.loader
send_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(send_cli)


class _Pool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _base_args(*extra: str) -> list[str]:
    return [
        "--database-url",
        "postgresql://atlas@example/db",
        "--from-email",
        "reports@example.com",
        "--result-base-url",
        "https://juancanfield.com",
        *extra,
    ]


def test_parse_defaults_to_dry_run_and_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTED_DATABASE_URL", "postgresql://env/db")
    monkeypatch.setenv("ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL", "reports@env.example")
    monkeypatch.setenv("ATLAS_DEFLECTION_PORTFOLIO_BASE_URL", "https://juancanfield.com")

    args = send_cli._parse_args([])

    assert args.database_url == "postgresql://env/db"
    assert args.from_email == "reports@env.example"
    assert args.result_base_url == "https://juancanfield.com"
    assert args.send is False


def test_parse_prefers_typed_delivery_result_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_DEFLECTION_DELIVERY_RESULT_BASE_URL", "https://typed.example")
    monkeypatch.setenv("ATLAS_DEFLECTION_PORTFOLIO_BASE_URL", "https://legacy.example")
    monkeypatch.setenv(
        "ATLAS_DEFLECTION_DELIVERY_RESULT_URL_TEMPLATE",
        "https://typed.example/results/{request_id}",
    )
    monkeypatch.setenv(
        "ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL_TEMPLATE",
        "https://legacy.example/results/{request_id}",
    )

    args = send_cli._parse_args([])

    assert args.result_base_url == "https://typed.example"
    assert args.result_url_template == "https://typed.example/results/{request_id}"


def test_validate_allows_dry_run_without_resend_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY", raising=False)
    args = send_cli._parse_args(_base_args())

    send_cli._validate_args(args)
    config = send_cli._delivery_config(args)

    assert config.dry_run is True


def test_validate_requires_resend_key_for_live_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY", raising=False)
    args = send_cli._parse_args(_base_args("--send"))

    with pytest.raises(SystemExit, match="Missing --resend-api-key"):
        send_cli._validate_args(args)


def test_validate_fails_closed_before_pool_for_missing_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ATLAS_DEFLECTION_DELIVERY_RESULT_BASE_URL",
        "ATLAS_DEFLECTION_DELIVERY_RESULT_URL_TEMPLATE",
        "ATLAS_DEFLECTION_PORTFOLIO_BASE_URL",
        "ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL_TEMPLATE",
    ):
        monkeypatch.delenv(name, raising=False)
    args = send_cli._parse_args(
        [
            "--database-url",
            "postgresql://atlas@example/db",
            "--from-email",
            "reports@example.com",
        ]
    )

    with pytest.raises(SystemExit, match="Missing --result-base-url"):
        send_cli._validate_args(args)


@pytest.mark.asyncio
async def test_main_dry_run_wires_worker_without_resend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool = _Pool()
    calls: list[dict[str, Any]] = []

    async def _create_pool(database_url: str) -> _Pool:
        calls.append({"database_url": database_url})
        return pool

    async def _send_pending(pool_arg: Any, *, sender: Any, config: Any) -> Any:
        calls.append(
            {
                "pool": pool_arg,
                "sender_type": type(sender).__name__,
                "dry_run": config.dry_run,
                "result_base_url": config.result_base_url,
            }
        )
        return SimpleNamespace(scanned=2, sent=0, failed=0, dry_run=2)

    monkeypatch.setattr(send_cli, "_create_pool", _create_pool)
    monkeypatch.setattr(send_cli, "send_pending_deflection_report_deliveries", _send_pending)

    code = await send_cli._main([*_base_args(), "--json"])

    assert code == 0
    assert pool.closed is True
    assert calls == [
        {"database_url": "postgresql://atlas@example/db"},
        {
            "pool": pool,
            "sender_type": "_DryRunSender",
            "dry_run": True,
            "result_base_url": "https://juancanfield.com",
        },
    ]
    assert json.loads(capsys.readouterr().out) == {
        "dry_run": 2,
        "failed": 0,
        "scanned": 2,
        "sent": 0,
    }


@pytest.mark.asyncio
async def test_main_forwards_account_and_request_scope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool = _Pool()
    calls: list[dict[str, Any]] = []

    async def _create_pool(database_url: str) -> _Pool:
        calls.append({"database_url": database_url})
        return pool

    async def _send_pending(
        pool_arg: Any,
        *,
        sender: Any,
        config: Any,
        account_id: str | None = None,
        request_id: str | None = None,
    ) -> Any:
        calls.append(
            {
                "pool": pool_arg,
                "sender_type": type(sender).__name__,
                "dry_run": config.dry_run,
                "account_id": account_id,
                "request_id": request_id,
            }
        )
        return SimpleNamespace(scanned=1, sent=0, failed=0, dry_run=1)

    monkeypatch.setattr(send_cli, "_create_pool", _create_pool)
    monkeypatch.setattr(send_cli, "send_pending_deflection_report_deliveries", _send_pending)

    code = await send_cli._main(
        [
            *_base_args(
                "--account-id",
                "acct-target",
                "--request-id",
                "content-ops-target",
            ),
            "--json",
        ]
    )

    assert code == 0
    assert pool.closed is True
    assert calls == [
        {"database_url": "postgresql://atlas@example/db"},
        {
            "pool": pool,
            "sender_type": "_DryRunSender",
            "dry_run": True,
            "account_id": "acct-target",
            "request_id": "content-ops-target",
        },
    ]
    assert json.loads(capsys.readouterr().out)["dry_run"] == 1


@pytest.mark.asyncio
async def test_main_live_send_uses_resend_sender_and_returns_failed_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool = _Pool()
    sender = object()
    calls: list[dict[str, Any]] = []

    async def _create_pool(_database_url: str) -> _Pool:
        return pool

    def _create_sender(provider: str, config: dict[str, Any]) -> object:
        calls.append({"provider": provider, "config": config})
        return sender

    async def _send_pending(_pool: Any, *, sender: Any, config: Any) -> Any:
        calls.append({"sender": sender, "dry_run": config.dry_run})
        return SimpleNamespace(scanned=1, sent=0, failed=1, dry_run=0)

    monkeypatch.setattr(send_cli, "_create_pool", _create_pool)
    monkeypatch.setattr(send_cli, "create_campaign_sender", _create_sender)
    monkeypatch.setattr(send_cli, "send_pending_deflection_report_deliveries", _send_pending)

    code = await send_cli._main(
        [
            *_base_args(
                "--send",
                "--resend-api-key",
                "re_test",
                "--resend-api-url",
                "https://resend.example.test/emails",
                "--timeout-seconds",
                "3",
            )
        ]
    )

    assert code == 1
    assert pool.closed is True
    assert calls == [
        {
            "provider": "resend",
            "config": {
                "api_key": "re_test",
                "api_url": "https://resend.example.test/emails",
                "timeout_seconds": 3.0,
            },
        },
        {"sender": sender, "dry_run": False},
    ]
    assert "mode=send scanned=1 sent=0 failed=1 dry_run=0" in capsys.readouterr().out
