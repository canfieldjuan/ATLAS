from __future__ import annotations

from dataclasses import replace
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
    def __init__(
        self,
        draft: TicketFAQDraft | None,
        *,
        stored_status: str | None = None,
    ) -> None:
        self.draft = draft
        # The DB-authoritative status; may diverge from draft.status to model
        # a concurrent review decision landing after get_draft.
        self.stored_status = (
            stored_status
            if stored_status is not None
            else (draft.status if draft is not None else "")
        )
        self.get_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []

    async def get_draft(
        self,
        faq_id: str,
        *,
        scope: TenantScope,
    ) -> TicketFAQDraft | None:
        self.get_calls.append({"faq_id": faq_id, "scope": scope})
        if self.draft is None:
            return None
        # The first read is the pre-race snapshot the service observed; any
        # re-read (used to classify a compare-and-set miss) reflects the
        # current DB-authoritative status, exactly as Postgres would.
        if len(self.get_calls) == 1:
            return self.draft
        return replace(self.draft, status=self.stored_status)

    async def update_status(
        self,
        faq_id: str,
        status: str,
        *,
        scope: TenantScope,
        expected_status: str | None = None,
    ) -> bool:
        self.update_calls.append({
            "faq_id": faq_id,
            "status": status,
            "scope": scope,
        })
        if expected_status is not None and self.stored_status != expected_status:
            return False
        self.stored_status = status
        return True


class _AttemptRepo:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    async def record_attempt(
        self,
        summary,
        *,
        scope: TenantScope,
    ) -> None:
        if self.fail:
            raise RuntimeError("attempt write failed")
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
        "status_conflict": False,
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


@pytest.mark.asyncio
async def test_publish_service_keeps_success_when_attempt_history_write_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repo = _FAQRepo(_draft())
    provider = _Provider(("published",))
    attempts = _AttemptRepo(fail=True)
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.ok is True
    assert summary.published_count == 1
    assert repo.update_calls == [{
        "faq_id": "faq-draft-1",
        "status": "published",
        "scope": TenantScope(account_id="acct-1"),
    }]
    assert "failed to record FAQ macro publish attempt" in caplog.text


@pytest.mark.asyncio
async def test_publish_service_skips_attempt_history_without_account_scope() -> None:
    repo = _FAQRepo(_draft())
    provider = _Provider(("published",))
    attempts = _AttemptRepo()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(),
    )

    assert summary.ok is True
    assert attempts.calls == []


@pytest.mark.asyncio
async def test_publish_service_approve_draft_promotes_generated_draft_and_publishes() -> None:
    # Producer-real shape: save_drafts persists status='draft'; an explicit
    # tenant publish action with approve_draft=True promotes then publishes.
    scope = TenantScope(account_id="acct-1", user_id="user-1")
    repo = _FAQRepo(_draft(status="draft"))
    provider = _Provider(("published",))
    attempts = _AttemptRepo()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=scope,
        approve_draft=True,
    )

    assert summary.ok is True
    assert summary.published_count == 1
    assert [(c["status"]) for c in repo.update_calls] == ["approved", "published"]
    assert provider.calls[0]["scope"] == scope
    assert len(attempts.calls) == 1


@pytest.mark.asyncio
async def test_publish_service_approve_draft_never_revives_review_decisions() -> None:
    for status in ("rejected", "archived", "expired"):
        repo = _FAQRepo(_draft(status=status))
        provider = _Provider(("published",))
        service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

        summary = await service.publish_faq_draft(
            "faq-draft-1",
            scope=TenantScope(account_id="acct-1"),
            approve_draft=True,
        )

        assert summary.ok is False, status
        assert summary.published_count == 0, status
        assert provider.calls == [], status
        assert repo.update_calls == [], status


@pytest.mark.asyncio
async def test_publish_service_approve_draft_fails_closed_when_promotion_refused() -> None:
    class _RefusingRepo(_FAQRepo):
        async def update_status(self, faq_id, status, *, scope, expected_status=None):
            self.update_calls.append({
                "faq_id": faq_id,
                "status": status,
                "scope": scope,
            })
            return False

    repo = _RefusingRepo(_draft(status="draft"))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is False
    assert summary.status_conflict is True   # a concurrent race, not plain ineligibility
    assert summary.skipped[0]["reason"] == "draft_not_approved"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_publish_service_default_still_refuses_generated_draft() -> None:
    # Without the explicit approve_draft opt-in, behavior is unchanged: the
    # AI Content Station and scheduled paths still require prior approval.
    repo = _FAQRepo(_draft(status="draft"))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.ok is False
    assert summary.skipped[0]["reason"] == "draft_not_approved"
    assert provider.calls == []
    assert repo.update_calls == []


@pytest.mark.asyncio
async def test_publish_service_approve_draft_loses_race_to_concurrent_review() -> None:
    # get_draft observed 'draft', but a reviewer rejected the draft before the
    # promotion ran: the compare-and-set must lose, nothing publishes, and the
    # concurrent decision is never overwritten.
    repo = _FAQRepo(_draft(status="draft"), stored_status="rejected")
    provider = _Provider(("published",))
    attempts = _AttemptRepo()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is False
    assert summary.published_count == 0
    assert summary.skipped[0]["reason"] == "draft_not_approved"
    assert provider.calls == []
    assert repo.stored_status == "rejected"


@pytest.mark.asyncio
async def test_publish_service_approve_draft_retries_published_idempotently() -> None:
    # Lost HTTP response / second click: the row is already 'published'.
    # The retry must flow through the provider's idempotent mapping and
    # succeed, without any promotion write and without demoting the row.
    repo = _FAQRepo(_draft(status="published"))
    provider = _Provider(("updated",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is True
    assert len(provider.calls) == 1
    # a published retry performs NO status write: not a draft->approved
    # promotion, and no approved->published mark (the row is already
    # published). draft_status reports the honest observed 'published'.
    assert repo.update_calls == []
    assert repo.stored_status == "published"
    assert summary.draft_status == "published"
    assert summary.draft_status_updated is False
    assert summary.status_conflict is False


@pytest.mark.asyncio
async def test_publish_service_mark_published_loses_to_midflight_review_decision() -> None:
    # A reject lands while the external publish is in flight: the publish
    # bookkeeping must not overwrite the reviewer's decision.
    repo = _FAQRepo(_draft(status="draft"))

    class _RejectingMidFlightProvider:
        def __init__(self, target: _FAQRepo) -> None:
            self.target = target
            self.calls: list[dict[str, object]] = []

        async def publish(self, macros, *, scope):
            self.calls.append({"macros": tuple(macros), "scope": scope})
            # concurrent reviewer decision during the external call
            self.target.stored_status = "rejected"
            return tuple(
                MacroPublishResult(
                    macro=macro,
                    status=cast(MacroPublishStatus, "published"),
                    external_id="external-1",
                )
                for macro in macros
            )

    provider = _RejectingMidFlightProvider(repo)
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    # the external publish happened and is reported...
    assert summary.published_count == 1
    # ...but the reviewer's decision is preserved, not overwritten, and the
    # caller sees a conflict rather than a clean success (the DB refused to
    # mark the FAQ published).
    assert repo.stored_status == "rejected"
    assert summary.draft_status_updated is False
    assert summary.status_conflict is True
    assert summary.ok is False


@pytest.mark.asyncio
async def test_publish_service_never_approves_draft_with_no_publishable_macros() -> None:
    # Eligibility is decided before any status write: a draft whose items are
    # all non-publishable must stay 'draft' (an approval would also surface it
    # in the approved-filtered FAQ search projection).
    repo = _FAQRepo(_draft(
        status="draft",
        items=(
            {
                "faq_item_id": "faq-item-1",
                "question": "How do I export a report?",
                "answer": "Customers mention exports.",
                "answer_evidence_status": "draft_needs_review",
            },
        ),
    ))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is False
    assert provider.calls == []
    assert repo.update_calls == []
    assert repo.stored_status == "draft"
    # the real per-item reason is surfaced, not masked as draft_not_approved
    assert summary.skipped[0]["reason"] == "answer_not_verified"


@pytest.mark.asyncio
async def test_publish_service_never_approves_partially_publishable_generated_draft() -> None:
    # A generated draft with a mix of one publishable and one skipped item:
    # preview.macros is non-empty, but it is NOT fully publishable, so the
    # promotion must not fire (else the partial draft would be approved and
    # surface in the approved-filtered FAQ search projection while ok==False).
    repo = _FAQRepo(_draft(
        status="draft",
        items=(
            {
                "faq_item_id": "faq-item-1",
                "question": "Why was I charged twice?",
                "resolution_text": "Open Billing and compare settled charges.",
                "answer_evidence_status": "resolution_evidence",
            },
            {
                "faq_item_id": "faq-item-2",
                "question": "How do I export a report?",
                "answer": "Customers mention exports.",
                "answer_evidence_status": "draft_needs_review",
            },
        ),
    ))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is False
    assert summary.skipped_count == 1
    assert provider.calls == []          # nothing published
    assert repo.update_calls == []       # nothing promoted
    assert repo.stored_status == "draft"  # row untouched


@pytest.mark.asyncio
async def test_publish_service_already_approved_mixed_draft_still_publishes_subset() -> None:
    # Contrast: an already-approved (human-reviewed) mixed draft keeps the
    # existing behavior of publishing its publishable subset. The skipped item
    # only blocks the final mark-published, not the publish itself.
    repo = _FAQRepo(_draft(
        status="approved",
        items=(
            {
                "faq_item_id": "faq-item-1",
                "question": "Why was I charged twice?",
                "resolution_text": "Open Billing and compare settled charges.",
                "answer_evidence_status": "resolution_evidence",
            },
            {
                "faq_item_id": "faq-item-2",
                "question": "How do I export a report?",
                "answer": "Customers mention exports.",
                "answer_evidence_status": "draft_needs_review",
            },
        ),
    ))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.published_count == 1   # the publishable item published
    assert summary.skipped_count == 1
    assert provider.calls != []
    assert repo.update_calls == []        # skipped item blocks the mark
    assert repo.stored_status == "approved"


@pytest.mark.asyncio
async def test_publish_service_concurrent_double_publish_is_idempotent_success() -> None:
    # Two publishes race for the same approved FAQ: the other request marks the
    # row 'published' while this one is between provider.publish and the mark
    # CAS. The CAS misses, but re-reading shows the terminal state is reached,
    # so this must report success -- not a conflict.
    repo = _FAQRepo(_draft(status="approved"))

    class _AlreadyPublishedMidFlight:
        def __init__(self, target: _FAQRepo) -> None:
            self.target = target
            self.calls: list[dict[str, object]] = []

        async def publish(self, macros, *, scope):
            self.calls.append({"macros": tuple(macros), "scope": scope})
            self.target.stored_status = "published"  # concurrent request won
            return tuple(
                MacroPublishResult(
                    macro=macro,
                    status=cast(MacroPublishStatus, "published"),
                    external_id="external-1",
                )
                for macro in macros
            )

    provider = _AlreadyPublishedMidFlight(repo)
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert summary.published_count == 1
    assert summary.status_conflict is False
    assert summary.ok is True
    assert summary.draft_status_updated is False  # already published by the racer
    assert repo.stored_status == "published"


@pytest.mark.asyncio
async def test_publish_service_default_caller_retries_published_draft_idempotently() -> None:
    # A published draft is an idempotent retry for EVERY caller, not just
    # approve_draft=True: a lost response / second click from the AI Content
    # Station or scheduled path must not fail closed.
    repo = _FAQRepo(_draft(status="published"))
    provider = _Provider(("updated",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),  # approve_draft defaults False
    )

    assert summary.ok is True
    assert len(provider.calls) == 1
    assert repo.update_calls == []            # no status write on a retry
    assert summary.draft_status == "published"
    assert summary.status_conflict is False


@pytest.mark.asyncio
async def test_publish_service_partial_generated_draft_reports_observed_skip_status() -> None:
    # A partial generated draft is refused (not promoted); the skipped payload
    # must report the real observed 'draft' status, never the synthetic
    # 'approved' used only to evaluate item-level eligibility.
    repo = _FAQRepo(_draft(
        status="draft",
        items=(
            {
                "faq_item_id": "faq-item-1",
                "question": "Why was I charged twice?",
                "resolution_text": "Open Billing and compare settled charges.",
                "answer_evidence_status": "resolution_evidence",
            },
            {
                "faq_item_id": "faq-item-2",
                "question": "How do I export a report?",
                "answer": "Customers mention exports.",
                "answer_evidence_status": "draft_needs_review",
            },
        ),
    ))
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is False
    assert summary.draft_status == "draft"
    assert provider.calls == []
    assert repo.update_calls == []
    # every skipped item reports the real observed status, not 'approved'
    assert all(item["draft_status"] == "draft" for item in summary.skipped)


@pytest.mark.asyncio
async def test_publish_service_promotion_race_already_published_is_idempotent() -> None:
    # Two approve_draft publishes race while the row is still 'draft': the
    # first promotes and fully publishes before the second reaches the
    # promotion CAS. The second loses the draft->approved CAS, re-reads
    # 'published', and must republish idempotently (no mark, no conflict),
    # reporting success rather than failure.
    repo = _FAQRepo(_draft(status="draft"), stored_status="published")
    provider = _Provider(("updated",))
    attempts = _AttemptRepo()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is True
    assert summary.status_conflict is False
    assert len(provider.calls) == 1          # republished idempotently
    assert summary.draft_status == "published"
    assert summary.draft_status_updated is False  # no re-mark
    assert len(attempts.calls) == 1


@pytest.mark.asyncio
async def test_publish_service_promotion_race_concurrently_approved_publishes_and_marks() -> None:
    # A concurrent request approved the draft (draft->approved) before this one
    # reached the promotion CAS. The CAS miss re-reads 'approved', so this
    # request publishes and marks as approved rather than reporting a conflict.
    repo = _FAQRepo(_draft(status="draft"), stored_status="approved")
    provider = _Provider(("published",))
    service = FAQMacroWritebackPublishService(faq_repository=repo, provider=provider)

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
        approve_draft=True,
    )

    assert summary.ok is True
    assert summary.status_conflict is False
    assert summary.published_count == 1
    assert summary.draft_status == "approved"


@pytest.mark.asyncio
async def test_publish_service_provider_exception_records_failed_attempt() -> None:
    # If the provider raises (for example a tenant credential lookup error
    # outside its per-macro catch) after promotion, the exception must not
    # bypass attempt history: the service records a durable failed attempt and
    # does not propagate the exception.
    repo = _FAQRepo(_draft(status="approved"))
    attempts = _AttemptRepo()

    class _RaisingProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def publish(self, macros, *, scope):
            self.calls += 1
            raise RuntimeError("zendesk credential lookup failed")

    provider = _RaisingProvider()
    service = FAQMacroWritebackPublishService(
        faq_repository=repo,
        provider=provider,
        attempt_repository=attempts,
    )

    summary = await service.publish_faq_draft(
        "faq-draft-1",
        scope=TenantScope(account_id="acct-1"),
    )

    assert provider.calls == 1
    assert summary.ok is False
    assert summary.failed_count == 1
    assert "provider_error" in summary.results[0]["error"]
    assert len(attempts.calls) == 1          # durable failed attempt recorded
    assert summary.draft_status_updated is False
