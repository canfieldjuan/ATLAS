from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Sequence

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import MacroPublishResult
from extracted_content_pipeline.faq_macro_writeback_publish import FAQMacroPublishSummary
from extracted_content_pipeline.faq_macro_writeback_zendesk import ZendeskMacroCredentials
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_macro_live_zendesk.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_macro_live_zendesk",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _CredentialSource:
    def __init__(self, credentials: ZendeskMacroCredentials | None) -> None:
        self.credentials = credentials
        self.calls: list[TenantScope] = []

    async def credentials_for_scope(
        self,
        scope: TenantScope,
    ) -> ZendeskMacroCredentials | None:
        self.calls.append(scope)
        return self.credentials


class _FAQRepo:
    def __init__(self, draft: TicketFAQDraft | None) -> None:
        self.draft = draft
        self.get_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []

    async def get_draft(
        self,
        faq_id: str,
        *,
        scope: TenantScope,
    ) -> TicketFAQDraft | None:
        self.get_calls.append({"faq_id": faq_id, "scope": scope})
        return self.draft

    async def update_status(
        self,
        faq_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        self.update_calls.append({
            "faq_id": faq_id,
            "status": status,
            "scope": scope,
        })
        return True


class _PublishProvider:
    def __init__(self, *, status: str = "published") -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    async def publish(
        self,
        macros: Sequence[Any],
        *,
        scope: TenantScope,
    ) -> Sequence[MacroPublishResult]:
        self.calls.append({"macros": tuple(macros), "scope": scope})
        return tuple(
            MacroPublishResult(
                macro=macro,
                status=self.status,
                external_id=f"macro-{index}",
            )
            for index, macro in enumerate(macros, start=1)
        )


class _AttemptRepo:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def record_attempt(
        self,
        summary: FAQMacroPublishSummary,
        *,
        scope: TenantScope,
    ) -> None:
        self.calls.append({"summary": summary, "scope": scope})


@pytest.mark.asyncio
async def test_live_zendesk_smoke_skips_before_pool_when_confirmation_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_create_pool(database_url: str) -> object:
        raise AssertionError(f"pool should not be opened for {database_url}")

    output = tmp_path / "result.json"
    monkeypatch.setattr(smoke, "_create_pool", fail_create_pool)

    code = await smoke._main([
        "--database-url",
        "postgres://example",
        "--account-id",
        "acct-1",
        "--faq-id",
        "faq-1",
        "--output",
        str(output),
    ])

    assert code == smoke.SKIPPED_EXIT
    assert '"not_run_reason": "missing_confirm_live_zendesk_write"' in output.read_text()


@pytest.mark.asyncio
async def test_live_zendesk_smoke_requires_tenant_credentials_before_provider_call() -> None:
    provider = _PublishProvider()
    code, payload = await smoke.run_live_zendesk_smoke(
        _args(),
        pool=object(),
        credentials_provider=_CredentialSource(None),
        faq_repository=_FAQRepo(_approved_draft()),
        provider=provider,
        attempt_repository=_AttemptRepo(),
    )

    assert code == smoke.SKIPPED_EXIT
    assert payload["not_run_reason"] == "zendesk_credentials_missing"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_live_zendesk_smoke_rejects_unexpected_zendesk_base_url() -> None:
    provider = _PublishProvider()
    code, payload = await smoke.run_live_zendesk_smoke(
        _args(expected_zendesk_base_url="https://safe-sandbox.zendesk.com"),
        pool=object(),
        credentials_provider=_CredentialSource(_credentials()),
        faq_repository=_FAQRepo(_approved_draft()),
        provider=provider,
        attempt_repository=_AttemptRepo(),
    )

    assert code == smoke.SKIPPED_EXIT
    assert payload["not_run_reason"] == "unexpected_zendesk_base_url"
    assert payload["observed_zendesk_base_url"] == "https://acme.zendesk.com"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_live_zendesk_smoke_requires_publishable_faq_before_provider_call() -> None:
    provider = _PublishProvider()
    code, payload = await smoke.run_live_zendesk_smoke(
        _args(),
        pool=object(),
        credentials_provider=_CredentialSource(_credentials()),
        faq_repository=_FAQRepo(_unapproved_draft()),
        provider=provider,
        attempt_repository=_AttemptRepo(),
    )

    assert code == smoke.SKIPPED_EXIT
    assert payload["not_run_reason"] == "no_publishable_macros"
    assert payload["draft_status"] == "draft"
    assert payload["skipped"][0]["reason"] == "draft_not_approved"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_live_zendesk_smoke_publishes_with_service_and_records_attempt() -> None:
    faq_repo = _FAQRepo(_approved_draft())
    provider = _PublishProvider()
    attempt_repo = _AttemptRepo()

    code, payload = await smoke.run_live_zendesk_smoke(
        _args(user_id="operator-1"),
        pool=object(),
        credentials_provider=_CredentialSource(_credentials()),
        faq_repository=faq_repo,
        provider=provider,
        attempt_repository=attempt_repo,
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["skipped"] is False
    assert payload["zendesk_base_url"] == "https://acme.zendesk.com"
    assert payload["summary"]["published_count"] == 1
    assert payload["summary"]["draft_status_updated"] is True
    scope = TenantScope(account_id="acct-1", user_id="operator-1")
    assert provider.calls[0]["scope"] == scope
    assert provider.calls[0]["macros"][0].title == "Why was I charged twice?"
    assert faq_repo.update_calls == [
        {"faq_id": "faq-1", "status": "published", "scope": scope}
    ]
    assert attempt_repo.calls[0]["scope"] == scope


@pytest.mark.asyncio
async def test_live_zendesk_smoke_reports_provider_failures() -> None:
    provider = _PublishProvider(status="failed")

    code, payload = await smoke.run_live_zendesk_smoke(
        _args(),
        pool=object(),
        credentials_provider=_CredentialSource(_credentials()),
        faq_repository=_FAQRepo(_approved_draft()),
        provider=provider,
        attempt_repository=_AttemptRepo(),
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["macro_publish_failed"]
    assert payload["summary"]["failed_count"] == 1


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "database_url": "postgres://example",
        "account_id": "acct-1",
        "faq_id": "faq-1",
        "user_id": "",
        "expected_zendesk_base_url": "",
        "confirm_live_zendesk_write": True,
        "output": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _credentials() -> ZendeskMacroCredentials:
    return ZendeskMacroCredentials(
        email="agent@example.com",
        api_token="tenant-token",
        subdomain="acme",
    )


def _approved_draft() -> TicketFAQDraft:
    return TicketFAQDraft(
        id="faq-1",
        target_id="ticket-faq-report",
        target_mode="support_ticket_faq",
        title="Saved FAQ report",
        markdown="# Saved FAQ report",
        items=(
            {
                "faq_item_id": "faq-item-1",
                "topic": "billing",
                "question": "Why was I charged twice?",
                "resolution_text": "Open Billing and compare settled charges.",
                "answer_evidence_status": "resolution_evidence",
                "source_ids": ("ticket-1",),
            },
        ),
        status="approved",
    )


def _unapproved_draft() -> TicketFAQDraft:
    draft = _approved_draft()
    return TicketFAQDraft(
        id=draft.id,
        target_id=draft.target_id,
        target_mode=draft.target_mode,
        title=draft.title,
        markdown=draft.markdown,
        items=draft.items,
        status="draft",
    )
