from __future__ import annotations

from typing import Sequence, cast

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import (
    MacroPublishResult,
    MacroPublishStatus,
    SupportMacroDraft,
)
from extracted_content_pipeline.faq_macro_writeback_publish import (
    FAQMacroWritebackPublishService,
)
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


class _FAQRepo:
    def __init__(self, draft: TicketFAQDraft | None) -> None:
        self.draft = draft
        self.get_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []

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


class _AttemptRepo:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def record_attempt(
        self,
        summary,
        *,
        scope: TenantScope,
    ) -> None:
        self.calls.append({"summary": summary, "scope": scope})


class _Provider:
    def __init__(self, statuses: tuple[str, ...]) -> None:
        self.statuses = statuses
        self.calls: list[dict[str, object]] = []

    async def publish(
        self,
        macros: Sequence[SupportMacroDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[MacroPublishResult]:
        self.calls.append({"macros": tuple(macros), "scope": scope})
        return tuple(
            MacroPublishResult(
                macro=macro,
                status=cast(MacroPublishStatus, status),
                external_id=f"external-{index}",
            )
            for index, (macro, status) in enumerate(zip(macros, self.statuses), start=1)
        )


class _PendingProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def publish(
        self,
        macros: Sequence[SupportMacroDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[MacroPublishResult]:
        self.calls.append({"macros": tuple(macros), "scope": scope})
        return tuple(
            MacroPublishResult(
                macro=macro,
                status="failed",
                error="zendesk_macro_mapping_pending_reconcile",
            )
            for macro in macros
        )


def _draft(
    *,
    status: str = "approved",
    items: tuple[dict[str, object], ...] | None = None,
    draft_id: str = "faq-draft-1",
) -> TicketFAQDraft:
    return TicketFAQDraft(
        id=draft_id,
        target_id="ticket-faq-report",
        target_mode="support_ticket_faq",
        title="Saved FAQ report",
        markdown="# Saved FAQ report",
        items=items or (
            {
                "faq_item_id": "faq-item-1",
                "topic": "billing",
                "question": "Why was I charged twice?",
                "resolution_text": "Open Billing and compare settled charges.",
                "answer_evidence_status": "resolution_evidence",
            },
        ),
        source_count=1,
        ticket_source_count=1,
        status=status,
    )


@pytest.mark.asyncio
async def test_publish_service_publishes_approved_verified_faq_and_marks_status() -> None:
    scope = TenantScope(account_id="acct-1", user_id="user-1")
    repo = _FAQRepo(_draft())
    provider = _Provider(("published",))
    attempts = _AttemptRepo()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(" faq-draft-1 ", scope=scope)

    assert summary.ok is True
    assert summary.as_dict()["ok"] is True
    assert summary.publishable_count == 1
    assert summary.skipped_count == 0
    assert summary.published_count == 1
    assert summary.draft_status_updated is True
    assert provider.calls[0]["scope"] == scope
    assert repo.get_calls == [{"faq_id": "faq-draft-1", "scope": scope}]
    assert repo.update_calls == [{
        "faq_id": "faq-draft-1",
        "status": "published",
        "scope": scope,
    }]
    assert attempts.calls == [{"summary": summary, "scope": scope}]


@pytest.mark.asyncio
async def test_publish_service_refuses_unapproved_draft_without_provider_call() -> None:
    repo = _FAQRepo(_draft(status="draft"))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.ok is False
    assert summary.publishable_count == 0
    assert summary.skipped_count == 1
    assert summary.skipped[0]["reason"] == "draft_not_approved"
    assert provider.calls == []
    assert repo.update_calls == []


@pytest.mark.asyncio
async def test_publish_service_keeps_status_when_items_are_skipped() -> None:
    repo = _FAQRepo(_draft(items=(
        {
            "faq_item_id": "faq-item-1",
            "question": "Where do I find invoices?",
            "resolution_text": "Open Billing and choose invoices.",
            "answer_evidence_status": "resolution_evidence",
        },
        {
            "faq_item_id": "faq-item-2",
            "question": "How do I export a report?",
            "answer": "Customers mention exports.",
            "answer_evidence_status": "draft_needs_review",
        },
    )))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.ok is False
    assert summary.publishable_count == 1
    assert summary.skipped_count == 1
    assert summary.published_count == 1
    assert summary.skipped[0]["reason"] == "answer_not_verified"
    assert repo.update_calls == []


@pytest.mark.asyncio
async def test_publish_service_surfaces_pending_reconcile_without_status_update() -> None:
    repo = _FAQRepo(_draft())
    provider = _PendingProvider()
    attempts = _AttemptRepo()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.ok is False
    assert summary.failed_count == 1
    assert summary.pending_reconcile_count == 1
    assert summary.results[0]["error"] == "zendesk_macro_mapping_pending_reconcile"
    assert repo.update_calls == []
    assert attempts.calls == [{
        "summary": summary,
        "scope": TenantScope(account_id="acct-1"),
    }]


@pytest.mark.asyncio
async def test_publish_service_reports_missing_draft_without_provider_call() -> None:
    repo = _FAQRepo(None)
    provider = _Provider(("published",))
    attempts = _AttemptRepo()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "missing-faq",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.as_dict() == {
        "faq_id": "missing-faq",
        "found": False,
        "draft_status": "",
        "publishable_count": 0,
        "skipped_count": 0,
        "published_count": 0,
        "updated_count": 0,
        "failed_count": 0,
        "pending_reconcile_count": 0,
        "draft_status_updated": False,
        "skipped": [],
        "results": [],
        "ok": False,
    }
    assert provider.calls == []
    assert repo.update_calls == []
    assert attempts.calls == []


@pytest.mark.asyncio
async def test_publish_service_does_not_mark_dry_run_results_published() -> None:
    repo = _FAQRepo(_draft())
    provider = _Provider(("dry_run",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.ok is False
    assert summary.publishable_count == 1
    assert summary.published_count == 0
    assert summary.updated_count == 0
    assert repo.update_calls == []
