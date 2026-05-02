from __future__ import annotations

import importlib
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from extracted_content_pipeline.campaign_ports import SendRequest, SendResult
from extracted_content_pipeline.autonomous.tasks.campaign_suppression import (
    assign_recipient_to_sequence,
    is_suppressed,
)
from extracted_content_pipeline.services import campaign_sender as campaign_sender_module
from extracted_content_pipeline.services.campaign_sender import CampaignSenderAdapter
from extracted_content_pipeline.services.vendor_registry import resolve_vendor_name
from extracted_content_pipeline.services.vendor_target_selection import (
    dedupe_vendor_target_rows,
)
from extracted_content_pipeline.settings import build_settings
from extracted_content_pipeline.templates.email.vendor_briefing import (
    render_vendor_briefing_html,
)


class FakeSender:
    def __init__(self) -> None:
        self.request: SendRequest | None = None

    async def send(self, request: SendRequest) -> SendResult:
        self.request = request
        return SendResult(provider="fake", message_id="msg-123", raw={"ok": True})


class FakeSuppressionPool:
    def __init__(self, *rows: dict[str, object] | None) -> None:
        self._rows = list(rows)
        self.fetchrow_args: list[tuple[object, ...]] = []
        self.fetchrow_queries: list[str] = []

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.fetchrow_queries.append(query)
        self.fetchrow_args.append(args)
        return self._rows.pop(0)


class FakeAssignmentPool:
    def __init__(self, *, conflict_id: UUID | None = None, result: str = "UPDATE 1") -> None:
        self.conflict_id = conflict_id
        self.result = result
        self.executed: tuple[object, ...] | None = None

    async def fetchval(self, query: str, *args: object) -> UUID | None:
        return self.conflict_id

    async def execute(self, query: str, *args: object) -> str:
        self.executed = args
        return self.result


class FakeTransaction:
    def __init__(self, conn: "FakeAssignmentConnection") -> None:
        self.conn = conn

    async def __aenter__(self) -> None:
        self.conn.transaction_entered = True

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.conn.transaction_exited = True


class FakeAcquireContext:
    def __init__(self, conn: "FakeAssignmentConnection") -> None:
        self.conn = conn

    async def __aenter__(self) -> "FakeAssignmentConnection":
        self.conn.acquired = True
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.conn.released = True


class FakeAssignmentConnection:
    def __init__(self, *, conflict_id: UUID | None = None, result: str = "UPDATE 1") -> None:
        self.conflict_id = conflict_id
        self.result = result
        self.acquired = False
        self.released = False
        self.transaction_entered = False
        self.transaction_exited = False
        self.fetchval_args: tuple[object, ...] | None = None
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.assignment_args: tuple[object, ...] | None = None

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def fetchval(self, query: str, *args: object) -> UUID | None:
        self.fetchval_args = args
        return self.conflict_id

    async def execute(self, query: str, *args: object) -> str:
        self.execute_calls.append((query, args))
        if "pg_advisory_xact_lock" not in query:
            self.assignment_args = args
            return self.result
        return "SELECT 1"


class FakeAssignmentAcquirePool:
    def __init__(self, conn: FakeAssignmentConnection) -> None:
        self.conn = conn

    def acquire(self) -> FakeAcquireContext:
        return FakeAcquireContext(self.conn)


def test_vendor_briefing_module_imports_in_standalone_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTED_PIPELINE_STANDALONE", "1")

    module = importlib.import_module(
        "extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing"
    )

    assert hasattr(module, "send_vendor_briefing")


def test_vendor_briefing_jwt_secret_uses_env_or_ephemeral_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXTRACTED_VENDOR_BRIEFING_JWT_SECRET", raising=False)

    generated_secret = build_settings().saas_auth.jwt_secret

    assert generated_secret
    assert generated_secret != "dev-secret"
    monkeypatch.setenv("EXTRACTED_VENDOR_BRIEFING_JWT_SECRET", "configured-secret")
    assert build_settings().saas_auth.jwt_secret == "configured-secret"


def test_build_settings_includes_vendor_briefing_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXTRACTED_VENDOR_BRIEFING_ENABLED", raising=False)
    monkeypatch.delenv("EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_ENABLED", raising=False)
    monkeypatch.delenv("EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_MAX", raising=False)
    monkeypatch.delenv(
        "EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_REASONING_DEPTH",
        raising=False,
    )
    monkeypatch.delenv(
        "EXTRACTED_VENDOR_BRIEFING_SCHEDULED_ACCOUNT_CARDS_REASONING_DEPTH",
        raising=False,
    )
    monkeypatch.delenv(
        "EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_ADAPTIVE_DEPTH",
        raising=False,
    )

    cfg = build_settings().b2b_churn

    assert cfg.vendor_briefing_enabled is True
    assert cfg.vendor_briefing_account_cards_enabled is True
    assert cfg.vendor_briefing_account_cards_max == 3
    assert cfg.vendor_briefing_account_cards_reasoning_depth == 2
    assert cfg.vendor_briefing_scheduled_account_cards_reasoning_depth == 0
    assert cfg.vendor_briefing_account_cards_adaptive_depth is True

    monkeypatch.setenv("EXTRACTED_VENDOR_BRIEFING_ACCOUNT_CARDS_REASONING_DEPTH", "1")
    assert build_settings().b2b_churn.vendor_briefing_account_cards_reasoning_depth == 1
    assert (
        build_settings()
        .b2b_churn
        .vendor_briefing_scheduled_account_cards_reasoning_depth
        == 0
    )

    monkeypatch.setenv(
        "EXTRACTED_VENDOR_BRIEFING_SCHEDULED_ACCOUNT_CARDS_REASONING_DEPTH",
        "2",
    )
    assert (
        build_settings()
        .b2b_churn
        .vendor_briefing_scheduled_account_cards_reasoning_depth
        == 2
    )


def test_dedupe_vendor_target_rows_keeps_best_row_per_company_and_mode() -> None:
    rows = [
        {
            "company_name": "Acme",
            "target_mode": "vendor_retention",
            "contact_email": "",
            "created_at": "2026-01-01",
        },
        {
            "company_name": " acme ",
            "target_mode": "vendor_retention",
            "contact_email": "ops@example.com",
            "created_at": "2026-01-02",
        },
        {
            "company_name": "Beta",
            "target_mode": "challenger_intel",
            "account_id": "acct-1",
            "created_at": "2026-01-01",
        },
    ]

    deduped = dedupe_vendor_target_rows(rows)

    assert [row["company_name"].strip() for row in deduped] == ["acme", "Beta"]
    assert deduped[0]["contact_email"] == "ops@example.com"
    assert deduped[1]["account_id"] == "acct-1"


@pytest.mark.asyncio
async def test_campaign_sender_adapter_converts_legacy_kwargs_to_send_request() -> None:
    inner = FakeSender()
    adapter = CampaignSenderAdapter(inner)

    result = await adapter.send(
        to="buyer@example.com",
        from_email="Atlas <audit@example.com>",
        subject="Briefing",
        body="<p>Body</p>",
        tags=[{"name": "type", "value": "vendor_briefing"}],
        metadata={"campaign_id": "cmp-1"},
    )

    assert result == {"id": "msg-123", "provider": "fake", "raw": {"ok": True}}
    assert inner.request == SendRequest(
        campaign_id="cmp-1",
        to_email="buyer@example.com",
        from_email="Atlas <audit@example.com>",
        subject="Briefing",
        html_body="<p>Body</p>",
        tags=({"name": "type", "value": "vendor_briefing"},),
        metadata={"campaign_id": "cmp-1"},
    )


def test_campaign_sender_resend_error_lists_accepted_configuration_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign_sender_module,
        "settings",
        SimpleNamespace(
            campaign_sequence=SimpleNamespace(
                sender_type="resend",
                resend_api_key="",
            )
        ),
    )

    with pytest.raises(RuntimeError) as exc:
        campaign_sender_module._sender_config_from_settings()

    message = str(exc.value)
    assert "settings.campaign_sequence.resend_api_key" in message
    assert "EXTRACTED_RESEND_API_KEY" in message
    assert "EXTRACTED_CAMPAIGN_RESEND_API_KEY" in message
    assert "EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY" in message


def test_build_settings_aliases_ses_sender_for_copied_vendor_briefing_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_SEQUENCE_SENDER_TYPE", "ses")
    monkeypatch.setenv("EXTRACTED_SES_FROM_EMAIL", "briefings@example.com")
    monkeypatch.delenv("EXTRACTED_RESEND_API_KEY", raising=False)
    monkeypatch.delenv("EXTRACTED_CAMPAIGN_RESEND_API_KEY", raising=False)
    monkeypatch.delenv("EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY", raising=False)
    monkeypatch.delenv("EXTRACTED_RESEND_FROM_EMAIL", raising=False)
    monkeypatch.delenv("EXTRACTED_CAMPAIGN_RESEND_FROM_EMAIL", raising=False)
    monkeypatch.delenv("EXTRACTED_CAMPAIGN_SEQ_RESEND_FROM_EMAIL", raising=False)

    cfg = build_settings().campaign_sequence

    assert cfg.sender_type == "ses"
    assert cfg.ses_from_email == "briefings@example.com"
    assert cfg.resend_from_email == "briefings@example.com"
    assert cfg.resend_api_key == "ses-configured"


@pytest.mark.asyncio
async def test_is_suppressed_checks_email_before_domain() -> None:
    pool = FakeSuppressionPool(None, {"domain": "example.com", "reason": "manual"})

    row = await is_suppressed(pool, email=" Person@Example.com ")

    assert row == {"domain": "example.com", "reason": "manual"}
    assert pool.fetchrow_args == [("person@example.com",), ("example.com",)]
    assert all("active" not in query.lower() for query in pool.fetchrow_queries)


@pytest.mark.asyncio
async def test_assign_recipient_to_sequence_reports_conflicts() -> None:
    sequence_id = uuid4()
    conflict_id = uuid4()
    pool = FakeAssignmentPool(conflict_id=conflict_id)

    result = await assign_recipient_to_sequence(pool, sequence_id, "buyer@example.com")

    assert result.assigned is False
    assert result.sequence_id == sequence_id
    assert result.conflict_with_sequence_id == conflict_id
    assert result.reason == "recipient_already_assigned"
    assert pool.executed is None


@pytest.mark.asyncio
async def test_assign_recipient_to_sequence_updates_active_sequence() -> None:
    sequence_id = UUID("11111111-1111-1111-1111-111111111111")
    pool = FakeAssignmentPool()

    result = await assign_recipient_to_sequence(pool, sequence_id, " Buyer@Example.com ")

    assert result.assigned is True
    assert result.reason is None
    assert pool.executed == (sequence_id, "buyer@example.com")


@pytest.mark.asyncio
async def test_assign_recipient_to_sequence_uses_transaction_scoped_advisory_lock() -> None:
    sequence_id = UUID("11111111-1111-1111-1111-111111111111")
    conn = FakeAssignmentConnection()
    pool = FakeAssignmentAcquirePool(conn)

    result = await assign_recipient_to_sequence(pool, sequence_id, " Buyer@Example.com ")

    assert result.assigned is True
    assert conn.acquired is True
    assert conn.released is True
    assert conn.transaction_entered is True
    assert conn.transaction_exited is True
    assert any("pg_advisory_xact_lock" in query for query, _ in conn.execute_calls)
    assert conn.fetchval_args == ("buyer@example.com", sequence_id)
    assert conn.assignment_args == (sequence_id, "buyer@example.com")


@pytest.mark.asyncio
async def test_resolve_vendor_name_async_alias_uses_local_normalizer() -> None:
    assert await resolve_vendor_name("  Acme  ") == "Acme"
    assert await resolve_vendor_name("aws") == "Amazon Web Services"
    assert await resolve_vendor_name("salesforce.com") == "Salesforce"


def test_render_vendor_briefing_html_escapes_and_gates_quotes() -> None:
    html = render_vendor_briefing_html(
        {
            "vendor_name": "Acme <script>",
            "analyst_summary": "Renewal risk is rising.",
            "evidence": [
                {"quote": "Costs jumped", "phrase_verbatim": True},
                {"quote": "Unmarked legacy quote"},
            ],
        }
    )

    assert "Churn Intelligence Briefing: Acme &lt;script&gt;" in html
    assert "Renewal risk is rising." in html
    assert "&ldquo;Costs jumped&rdquo;" in html
    assert "Unmarked legacy quote" not in html
    assert "<script>" not in html
