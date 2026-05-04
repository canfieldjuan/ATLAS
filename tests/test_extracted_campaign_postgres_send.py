from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import SendResult
from extracted_content_pipeline.campaign_postgres_send import (
    send_due_campaigns_from_postgres,
)
from extracted_content_pipeline.campaign_send import CampaignSendConfig, CampaignSendSummary


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/send_extracted_campaigns.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "send_extracted_campaigns",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self, rows=None) -> None:
        self.rows = list(rows or [])
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((str(query), args))
        return None

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return "OK"

    async def close(self):
        self.closed = True


class _Sender:
    def __init__(self) -> None:
        self.requests = []

    async def send(self, request):
        self.requests.append(request)
        return SendResult(provider="test", message_id=f"msg-{request.campaign_id}")


def _row(**overrides):
    row = {
        "id": "campaign_1",
        "sequence_id": "sequence_1",
        "recipient_email": "buyer@example.com",
        "from_email": "",
        "subject": "Pricing signal",
        "body": "<p>Hello</p>",
        "metadata": {"source": "test"},
        "company_name": "Acme",
        "step_number": 1,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_postgres_send_runner_sends_due_campaigns() -> None:
    pool = _Pool(rows=[_row()])
    sender = _Sender()

    summary = await send_due_campaigns_from_postgres(
        pool,
        sender=sender,
        config=CampaignSendConfig(default_from_email="sales@example.com", limit=3),
    )

    assert summary.as_dict() == {"sent": 1, "failed": 0, "suppressed": 0, "skipped": 0}
    query, args = pool.fetch_calls[0]
    assert "FROM b2b_campaigns" in query
    assert "status = 'queued'" in query
    assert args == (3,)
    assert sender.requests[0].from_email == "sales@example.com"
    assert any("status = 'sent'" in call[0] for call in pool.execute_calls)
    assert any("INSERT INTO campaign_audit_log" in call[0] for call in pool.execute_calls)


@pytest.mark.asyncio
async def test_postgres_send_runner_limit_zero_returns_empty_without_query() -> None:
    pool = _Pool(rows=[_row()])
    sender = _Sender()

    summary = await send_due_campaigns_from_postgres(
        pool,
        sender=sender,
        config=CampaignSendConfig(default_from_email="sales@example.com", limit=3),
        limit=0,
    )

    assert summary.as_dict() == {"sent": 0, "failed": 0, "suppressed": 0, "skipped": 0}
    assert pool.fetch_calls == []
    assert sender.requests == []


def test_send_cli_sender_config_uses_ses_from_email_fallback() -> None:
    cli = _load_cli_module()
    args = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--provider",
        "ses",
        "--ses-from-email",
        "ses@example.com",
    ])

    provider, config = cli._sender_config(args)

    assert provider == "ses"
    assert config["from_email"] == "ses@example.com"
    assert config["region"] == "us-east-1"


def test_send_cli_send_args_allows_default_from_for_ses() -> None:
    cli = _load_cli_module()
    args = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--provider",
        "ses",
        "--default-from-email",
        "sales@example.com",
    ])

    cli._validate_send_args(args)


def test_send_cli_reports_invalid_integer_env(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEND_LIMIT", "many")

    with pytest.raises(SystemExit) as exc_info:
        cli._parse_args([
            "--database-url",
            "postgres://example",
            "--resend-api-key",
            "re_key",
        ])

    message = str(exc_info.value)
    assert "Invalid integer for EXTRACTED_CAMPAIGN_SEND_LIMIT" in message
    assert "'many'" in message


def test_send_cli_reports_invalid_float_env(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SENDER_TIMEOUT_SECONDS", "slow")

    with pytest.raises(SystemExit) as exc_info:
        cli._parse_args([
            "--database-url",
            "postgres://example",
            "--resend-api-key",
            "re_key",
        ])

    message = str(exc_info.value)
    assert "Invalid float for EXTRACTED_CAMPAIGN_SENDER_TIMEOUT_SECONDS" in message
    assert "'slow'" in message


@pytest.mark.asyncio
async def test_send_cli_outputs_json_summary_and_closes_pool(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()
    calls: dict[str, object] = {}

    async def create_pool(database_url):
        calls["database_url"] = database_url
        return pool

    def create_sender(provider, config):
        calls["provider"] = provider
        calls["provider_config"] = config
        return _Sender()

    async def send_from_postgres(pool_arg, *, sender, config, limit):
        calls["pool"] = pool_arg
        calls["sender"] = sender
        calls["send_config"] = config
        calls["limit"] = limit
        return CampaignSendSummary(sent=2, failed=1, suppressed=0, skipped=1)

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(cli, "create_campaign_sender", create_sender)
    monkeypatch.setattr(cli, "send_due_campaigns_from_postgres", send_from_postgres)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "send",
            "--database-url",
            "postgres://example",
            "--provider",
            "resend",
            "--resend-api-key",
            "re_key",
            "--default-from-email",
            "sales@example.com",
            "--limit",
            "4",
            "--json",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output == {"failed": 1, "sent": 2, "skipped": 1, "suppressed": 0}
    assert calls["database_url"] == "postgres://example"
    assert calls["provider"] == "resend"
    assert calls["provider_config"] == {
        "api_key": "re_key",
        "api_url": "https://api.resend.com/emails",
        "timeout_seconds": 30.0,
    }
    assert calls["limit"] == 4
    assert calls["pool"] is pool
    assert pool.closed is True
    assert calls["send_config"] == CampaignSendConfig(
        default_from_email="sales@example.com",
        default_reply_to=None,
        unsubscribe_base_url="",
        company_address="",
        limit=4,
    )


@pytest.mark.asyncio
async def test_send_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli.sys, "argv", ["send", "--resend-api-key", "re_key"])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()


@pytest.mark.asyncio
async def test_send_cli_rejects_zero_limit_before_pool_creation(monkeypatch) -> None:
    cli = _load_cli_module()
    pool_called = False

    async def create_pool(database_url):
        nonlocal pool_called
        pool_called = True
        return _Pool()

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "send",
            "--database-url",
            "postgres://example",
            "--resend-api-key",
            "re_key",
            "--default-from-email",
            "sales@example.com",
            "--limit",
            "0",
        ],
    )

    with pytest.raises(SystemExit, match="Invalid --limit"):
        await cli._main()
    assert pool_called is False


@pytest.mark.asyncio
async def test_send_cli_requires_resend_api_key_before_sender_creation(monkeypatch) -> None:
    cli = _load_cli_module()
    for name in cli.RESEND_API_KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "send",
            "--database-url",
            "postgres://example",
            "--provider",
            "resend",
        ],
    )

    with pytest.raises(SystemExit, match="Missing --resend-api-key"):
        await cli._main()


@pytest.mark.asyncio
async def test_send_cli_requires_resend_default_from_email(monkeypatch) -> None:
    cli = _load_cli_module()
    for name in cli.FROM_EMAIL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "send",
            "--database-url",
            "postgres://example",
            "--provider",
            "resend",
            "--resend-api-key",
            "re_key",
        ],
    )

    with pytest.raises(SystemExit, match="Missing --default-from-email"):
        await cli._main()


@pytest.mark.asyncio
async def test_send_cli_requires_ses_from_email_before_sender_creation(monkeypatch) -> None:
    cli = _load_cli_module()
    for name in cli.FROM_EMAIL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("EXTRACTED_SES_FROM_EMAIL", raising=False)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "send",
            "--database-url",
            "postgres://example",
            "--provider",
            "ses",
        ],
    )

    with pytest.raises(SystemExit, match="Missing --ses-from-email"):
        await cli._main()
