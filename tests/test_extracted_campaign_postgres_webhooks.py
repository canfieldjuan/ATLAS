from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_postgres_webhooks import (
    ingest_resend_webhook_from_postgres,
)
from extracted_content_pipeline.campaign_webhooks import (
    CampaignWebhookIngestionResult,
    WebhookVerificationError,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/ingest_extracted_campaign_webhook.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_extracted_campaign_webhook",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _secret() -> str:
    return "whsec_" + base64.b64encode(b"secret").decode("utf-8")


def _body(event_type: str = "email.opened", **data_overrides) -> bytes:
    data = {
        "email_id": "email_1",
        "to": "Buyer@Example.com",
        "created_at": "2026-05-01T12:00:00Z",
    }
    data.update(data_overrides)
    return json.dumps({"type": event_type, "data": data}).encode("utf-8")


def _headers(body: bytes, *, secret: str | None = None, msg_id: str = "msg_1"):
    secret_text = secret or _secret()
    raw_secret = secret_text[6:] if secret_text.startswith("whsec_") else secret_text
    secret_bytes = base64.b64decode(raw_secret)
    timestamp = "1714550400"
    to_sign = f"{msg_id}.{timestamp}.".encode("utf-8") + body
    signature = base64.b64encode(
        hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()
    ).decode("utf-8")
    return {
        "svix-id": msg_id,
        "svix-timestamp": timestamp,
        "svix-signature": f"v1,{signature}",
    }


class _Pool:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, object]] = []
        self.closed = False

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        return "OK"

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_postgres_webhook_runner_records_signed_open_event() -> None:
    pool = _Pool()
    body = _body("email.opened")

    result = await ingest_resend_webhook_from_postgres(
        pool,
        body=body,
        headers=_headers(body),
        signing_secret=_secret(),
    )

    assert result.as_dict() == {
        "status": "ok",
        "event_type": "opened",
        "message_id": "email_1",
        "reason": None,
        "suppressed": False,
    }
    queries = [call["query"] for call in pool.execute_calls]
    assert any("UPDATE b2b_campaigns" in query for query in queries)
    assert sum("INSERT INTO campaign_audit_log" in query for query in queries) == 2
    assert pool.execute_calls[0]["args"][0] == "email_1"


@pytest.mark.asyncio
async def test_postgres_webhook_runner_applies_unsubscribe_suppression() -> None:
    pool = _Pool()
    body = _body("email.unsubscribed")

    result = await ingest_resend_webhook_from_postgres(
        pool,
        body=body,
        headers=_headers(body),
        signing_secret=_secret(),
    )

    assert result.suppressed is True
    suppression_calls = [
        call
        for call in pool.execute_calls
        if "INSERT INTO campaign_suppressions" in call["query"]
    ]
    assert len(suppression_calls) == 1
    assert suppression_calls[0]["args"][:3] == (
        "buyer@example.com",
        "unsubscribe",
        "webhook",
    )


@pytest.mark.asyncio
async def test_postgres_webhook_runner_rejects_bad_signature_before_writes() -> None:
    pool = _Pool()
    body = _body("email.opened")

    with pytest.raises(WebhookVerificationError, match="Invalid webhook signature"):
        await ingest_resend_webhook_from_postgres(
            pool,
            body=body,
            headers={},
            signing_secret=_secret(),
        )

    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_postgres_webhook_runner_requires_secret_when_verifying() -> None:
    pool = _Pool()

    with pytest.raises(ValueError, match="signing_secret is required"):
        await ingest_resend_webhook_from_postgres(
            pool,
            body=b"{}",
            headers={},
            signing_secret="",
        )

    assert pool.execute_calls == []


def test_webhook_cli_reads_headers_json_and_overrides(tmp_path) -> None:
    cli = _load_cli_module()
    headers_file = tmp_path / "headers.json"
    headers_file.write_text(json.dumps({"svix-id": "old", "x-extra": 1}))

    headers = cli._read_headers(
        headers_file,
        [("svix-id", "new"), ("svix-signature", "v1,sig")],
    )

    assert headers == {
        "svix-id": "new",
        "x-extra": "1",
        "svix-signature": "v1,sig",
    }


def test_webhook_cli_requires_database_url(monkeypatch) -> None:
    cli = _load_cli_module()
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    args = cli._parse_args(["--signing-secret", _secret()])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        cli._validate_args(args)


def test_webhook_cli_requires_signing_secret_unless_skipped(monkeypatch) -> None:
    cli = _load_cli_module()
    for name in cli.RESEND_WEBHOOK_SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    args = cli._parse_args(["--database-url", "postgres://example"])

    with pytest.raises(SystemExit, match="Missing --signing-secret"):
        cli._validate_args(args)

    skipped = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--skip-signature-verification",
    ])
    cli._validate_args(skipped)


def test_webhook_cli_rejects_invalid_soft_bounce_days() -> None:
    cli = _load_cli_module()
    args = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--signing-secret",
        _secret(),
        "--soft-bounce-suppression-days",
        "0",
    ])

    with pytest.raises(SystemExit, match="must be positive"):
        cli._validate_args(args)


@pytest.mark.asyncio
async def test_webhook_cli_ingests_body_file_and_outputs_json(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    cli = _load_cli_module()
    body_file = tmp_path / "body.json"
    headers_file = tmp_path / "headers.json"
    body_file.write_bytes(b'{"type":"email.delivered","data":{"email_id":"email_1"}}')
    headers_file.write_text(json.dumps({"svix-id": "msg_1"}))
    pool = _Pool()
    calls: dict[str, object] = {}

    async def create_pool(database_url):
        calls["database_url"] = database_url
        return pool

    async def ingest(
        pool_arg,
        *,
        body,
        headers,
        signing_secret,
        verify_signatures,
        config,
    ):
        calls["pool"] = pool_arg
        calls["body"] = body
        calls["headers"] = headers
        calls["signing_secret"] = signing_secret
        calls["verify_signatures"] = verify_signatures
        calls["config"] = config
        return CampaignWebhookIngestionResult(
            status="ok",
            event_type="delivered",
            message_id="email_1",
            suppressed=False,
        )

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(cli, "ingest_resend_webhook_from_postgres", ingest)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "webhook",
            "--database-url",
            "postgres://example",
            "--body-file",
            str(body_file),
            "--headers-json",
            str(headers_file),
            "--header",
            "svix-id: msg_2",
            "--signing-secret",
            _secret(),
            "--record-unknown-events",
            "--soft-bounce-suppression-days",
            "9",
            "--json",
        ],
    )

    exit_code = await cli._main()

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "event_type": "delivered",
        "message_id": "email_1",
        "reason": None,
        "status": "ok",
        "suppressed": False,
    }
    assert calls["database_url"] == "postgres://example"
    assert calls["pool"] is pool
    assert calls["body"] == body_file.read_bytes()
    assert calls["headers"] == {"svix-id": "msg_2"}
    assert calls["signing_secret"] == _secret()
    assert calls["verify_signatures"] is True
    assert calls["config"].record_unknown_events is True
    assert calls["config"].soft_bounce_suppression_days == 9
    assert pool.closed is True


@pytest.mark.asyncio
async def test_webhook_cli_skip_signature_passes_verify_false(
    monkeypatch,
    tmp_path,
) -> None:
    cli = _load_cli_module()
    body_file = tmp_path / "body.json"
    body_file.write_bytes(b"{}")
    pool = _Pool()
    calls: dict[str, object] = {}

    async def create_pool(database_url):
        return pool

    async def ingest(
        pool_arg,
        *,
        body,
        headers,
        signing_secret,
        verify_signatures,
        config,
    ):
        calls["verify_signatures"] = verify_signatures
        calls["signing_secret"] = signing_secret
        return CampaignWebhookIngestionResult(status="ignored", reason="no_message_id")

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(cli, "ingest_resend_webhook_from_postgres", ingest)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "webhook",
            "--database-url",
            "postgres://example",
            "--body-file",
            str(body_file),
            "--skip-signature-verification",
        ],
    )

    assert await cli._main() == 0
    assert calls == {"verify_signatures": False, "signing_secret": ""}
