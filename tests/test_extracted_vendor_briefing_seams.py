from __future__ import annotations

import importlib
from uuid import UUID, uuid4

import pytest

from extracted_content_pipeline.campaign_ports import SendRequest, SendResult
from extracted_content_pipeline.autonomous.tasks.campaign_suppression import (
    assign_recipient_to_sequence,
    is_suppressed,
)
from extracted_content_pipeline.services.campaign_sender import CampaignSenderAdapter
from extracted_content_pipeline.services.vendor_registry import resolve_vendor_name
from extracted_content_pipeline.services.vendor_target_selection import (
    dedupe_vendor_target_rows,
)
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


def test_vendor_briefing_module_imports_in_standalone_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTED_PIPELINE_STANDALONE", "1")

    module = importlib.import_module(
        "extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing"
    )

    assert hasattr(module, "send_vendor_briefing")


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
async def test_resolve_vendor_name_async_alias_uses_local_normalizer() -> None:
    assert await resolve_vendor_name("  Acme  ") == "Acme"


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
