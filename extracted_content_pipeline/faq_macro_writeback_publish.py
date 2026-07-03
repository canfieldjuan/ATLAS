"""Publish approved FAQ drafts to support macro providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import logging
from typing import Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope
from .faq_macro_writeback import (
    APPROVED_FAQ_STATUS,
    DRAFT_FAQ_STATUS,
    MacroPublishProvider,
    MacroPublishResult,
    MacroWritebackPreview,
    build_macro_writeback_preview,
)
from .ticket_faq_ports import TicketFAQRepository


PUBLISHED_FAQ_STATUS = "published"
SUCCESSFUL_MACRO_STATUSES = frozenset({"published", "updated"})
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FAQMacroPublishSummary:
    """Summary returned by the FAQ macro publish trigger."""

    faq_id: str
    found: bool
    draft_status: str = ""
    publishable_count: int = 0
    skipped_count: int = 0
    published_count: int = 0
    updated_count: int = 0
    failed_count: int = 0
    pending_reconcile_count: int = 0
    draft_status_updated: bool = False
    status_conflict: bool = False
    skipped: Sequence[JsonDict] = field(default_factory=tuple)
    results: Sequence[JsonDict] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return (
            self.found
            and self.publishable_count > 0
            and self.skipped_count == 0
            and self.failed_count == 0
            and self.pending_reconcile_count == 0
            and self.publishable_count == self.published_count + self.updated_count
            and not self.status_conflict
        )

    def as_dict(self) -> JsonDict:
        data = asdict(self)
        data["ok"] = self.ok
        data["skipped"] = [dict(item) for item in self.skipped]
        data["results"] = [dict(item) for item in self.results]
        return data


@dataclass(frozen=True)
class FAQMacroPublishAttempt:
    """Persisted FAQ macro publish attempt summary."""

    id: str
    faq_id: str
    draft_status: str
    ok: bool
    publishable_count: int = 0
    skipped_count: int = 0
    published_count: int = 0
    updated_count: int = 0
    failed_count: int = 0
    pending_reconcile_count: int = 0
    draft_status_updated: bool = False
    status_conflict: bool = False
    skipped: Sequence[JsonDict] = field(default_factory=tuple)
    results: Sequence[JsonDict] = field(default_factory=tuple)
    created_at: str = ""

    def as_dict(self) -> JsonDict:
        data = asdict(self)
        data["skipped"] = [dict(item) for item in self.skipped]
        data["results"] = [dict(item) for item in self.results]
        return data


class FAQMacroPublishAttemptRepository(Protocol):
    """Persistence port for append-only FAQ macro publish attempt history."""

    async def record_attempt(
        self,
        summary: FAQMacroPublishSummary,
        *,
        scope: TenantScope,
    ) -> None:
        """Persist one publish attempt summary for a tenant."""

    async def list_attempts(
        self,
        faq_id: str,
        *,
        scope: TenantScope,
        limit: int,
    ) -> Sequence[FAQMacroPublishAttempt]:
        """Return recent publish attempt summaries for one tenant-scoped FAQ."""


@dataclass(frozen=True)
class FAQMacroWritebackPublishService:
    """Product-level trigger for approved FAQ macro writeback."""

    faq_repository: TicketFAQRepository
    provider: MacroPublishProvider
    attempt_repository: FAQMacroPublishAttemptRepository | None = None
    published_status: str = PUBLISHED_FAQ_STATUS

    async def publish_faq_draft(
        self,
        faq_id: str,
        *,
        scope: TenantScope,
        approve_draft: bool = False,
    ) -> FAQMacroPublishSummary:
        cleaned_id = _clean(faq_id)
        if not cleaned_id:
            return FAQMacroPublishSummary(faq_id="", found=False)

        draft = await self.faq_repository.get_draft(cleaned_id, scope=scope)
        if draft is None:
            return FAQMacroPublishSummary(faq_id=cleaned_id, found=False)

        # A paid Resolution Audit publish is the approval for a *generated*
        # draft, so only the exact 'draft' status is promotable, and only
        # behind approve_draft. An already 'published' draft is an idempotent
        # retry through the provider's mapping for *every* caller (a lost
        # response or a second click must not fail closed), so it is not gated
        # on approve_draft. rejected/archived/expired are never revived; every
        # other path keeps the standard "must already be approved" behavior.
        observed_status = _clean(draft.status)
        promote = approve_draft and observed_status == DRAFT_FAQ_STATUS
        published_retry = observed_status == _clean(self.published_status)
        already_approved = observed_status == APPROVED_FAQ_STATUS
        publishable_now = already_approved or promote or published_retry

        # The status the row will hold when we decide whether to mark it
        # published: a real promotion reaches 'approved'; a published retry
        # stays 'published' so it is never re-marked; anything else keeps its
        # observed status (and therefore cannot be marked published).
        effective_status = (
            APPROVED_FAQ_STATUS if (already_approved or promote) else observed_status
        )
        # Publishing requires approved semantics, so evaluate per-item
        # eligibility as if approved on any path that is allowed to publish.
        eligibility_status = (
            APPROVED_FAQ_STATUS if publishable_now else observed_status
        )
        preview = build_macro_writeback_preview(
            [replace(draft, status=eligibility_status)]
        )

        # A generated draft is auto-approved only when it is *fully*
        # publishable, so a partial or empty generated draft is never promoted
        # into the approved-filtered FAQ search projection. Already-approved
        # drafts still publish their publishable subset below.
        if promote and (preview.publishable_count == 0 or preview.skipped_count > 0):
            return await self._finish(
                _summary(
                    faq_id=cleaned_id,
                    found=True,
                    draft_status=observed_status,
                    preview=preview,
                    results=(),
                ),
                scope=scope,
            )

        if not preview.macros:
            return await self._finish(
                _summary(
                    faq_id=cleaned_id,
                    found=True,
                    draft_status=observed_status,
                    preview=preview,
                    results=(),
                ),
                scope=scope,
            )

        if promote:
            # Promote via the shared compare-and-set transition. On a miss the
            # helper re-reads the row so every write site classifies a race the
            # same way: an already-'published' row means a concurrent publish
            # reached the terminal state, so fall through to an idempotent
            # republish with no mark; an already-'approved' row was approved
            # concurrently, so publish and mark as approved; anything else is a
            # review decision (reject/archive) that wins -- a real conflict.
            promoted, current = await self._transition_status(
                cleaned_id,
                to_status=APPROVED_FAQ_STATUS,
                expected_status=DRAFT_FAQ_STATUS,
                scope=scope,
            )
            if not promoted:
                if current == _clean(self.published_status):
                    effective_status = _clean(self.published_status)
                elif current == APPROVED_FAQ_STATUS:
                    effective_status = APPROVED_FAQ_STATUS
                else:
                    return await self._finish(
                        replace(
                            _summary(
                                faq_id=cleaned_id,
                                found=True,
                                draft_status=observed_status,
                                preview=build_macro_writeback_preview([draft]),
                                results=(),
                            ),
                            status_conflict=True,
                        ),
                        scope=scope,
                    )

        # The provider is the external boundary. A raised exception (for
        # example a tenant credential lookup failure outside the provider's
        # per-macro catch) must still record a durable failed attempt rather
        # than bypassing attempt history and leaving a promoted-but-unpublished
        # draft with no trace.
        try:
            results = tuple(await self.provider.publish(preview.macros, scope=scope))
        except Exception as exc:
            logger.exception(
                "FAQ macro publish provider raised faq_id=%s", cleaned_id
            )
            results = tuple(
                MacroPublishResult(
                    macro=macro,
                    status="failed",
                    error=f"provider_error: {type(exc).__name__}",
                )
                for macro in preview.macros
            )

        summary = _summary(
            faq_id=cleaned_id,
            found=True,
            draft_status=effective_status,
            preview=preview,
            results=results,
        )
        if _should_mark_published(summary):
            # Same shared transition: a mark miss is an idempotent success when
            # the row is already 'published', and a conflict otherwise.
            marked, current = await self._transition_status(
                cleaned_id,
                to_status=self.published_status,
                expected_status=APPROVED_FAQ_STATUS,
                scope=scope,
            )
            if marked:
                summary = replace(summary, draft_status_updated=True)
            elif current == _clean(self.published_status):
                summary = replace(summary, draft_status_updated=False)
            else:
                summary = replace(
                    summary,
                    draft_status_updated=False,
                    status_conflict=True,
                )
        return await self._finish(summary, scope=scope)

    async def _transition_status(
        self,
        faq_id: str,
        *,
        to_status: str,
        expected_status: str,
        scope: TenantScope,
    ) -> tuple[bool, str]:
        """Compare-and-set a status; on a miss, re-read the stored status.

        Returns ``(updated, current_status)``. Centralising this so every write
        site (promotion and the publish-mark) classifies a compare-and-set miss
        identically: an already-terminal row is idempotent success, a
        review-decided row is a conflict.
        """
        updated = await self.faq_repository.update_status(
            faq_id,
            to_status,
            scope=scope,
            expected_status=expected_status,
        )
        if updated:
            return True, _clean(to_status)
        draft = await self.faq_repository.get_draft(faq_id, scope=scope)
        return False, _clean(draft.status) if draft is not None else ""

    async def _finish(
        self,
        summary: FAQMacroPublishSummary,
        *,
        scope: TenantScope,
    ) -> FAQMacroPublishSummary:
        """Record the attempt for every return path, then return the summary."""
        await self._record_attempt(summary, scope=scope)
        return summary

    async def _record_attempt(
        self,
        summary: FAQMacroPublishSummary,
        *,
        scope: TenantScope,
    ) -> None:
        if self.attempt_repository is None or not summary.found:
            return
        if not scope.account_id:
            logger.info(
                "skipping FAQ macro publish attempt history without account scope faq_id=%s",
                summary.faq_id,
            )
            return
        try:
            await self.attempt_repository.record_attempt(summary, scope=scope)
        except Exception:
            logger.exception(
                "failed to record FAQ macro publish attempt faq_id=%s",
                summary.faq_id,
            )


def _summary(
    *,
    faq_id: str,
    found: bool,
    draft_status: str,
    preview: MacroWritebackPreview,
    results: Sequence[MacroPublishResult],
) -> FAQMacroPublishSummary:
    pending_reconcile_count = sum(1 for result in results if _is_pending_reconcile(result))
    failed_count = sum(1 for result in results if result.status == "failed")
    return FAQMacroPublishSummary(
        faq_id=faq_id,
        found=found,
        draft_status=draft_status,
        publishable_count=preview.publishable_count,
        skipped_count=preview.skipped_count,
        published_count=sum(1 for result in results if result.status == "published"),
        updated_count=sum(1 for result in results if result.status == "updated"),
        failed_count=failed_count,
        pending_reconcile_count=pending_reconcile_count,
        # The preview evaluates item-level skip reasons under approved
        # semantics, but the reported per-item status must be the row's real
        # (summary) status so a refused/partial draft never claims 'approved'
        # while the summary says 'draft'.
        skipped=tuple(
            {**item.as_dict(), "draft_status": draft_status}
            for item in preview.skipped
        ),
        results=tuple(result.as_dict() for result in results),
    )


def _should_mark_published(summary: FAQMacroPublishSummary) -> bool:
    result_count = len(summary.results)
    return (
        _clean(summary.draft_status) == APPROVED_FAQ_STATUS
        and summary.publishable_count > 0
        and result_count == summary.publishable_count
        and summary.skipped_count == 0
        and summary.failed_count == 0
        and summary.pending_reconcile_count == 0
        and all(
            _clean(result.get("status")) in SUCCESSFUL_MACRO_STATUSES
            for result in summary.results
        )
    )


def _is_pending_reconcile(result: MacroPublishResult) -> bool:
    error = _clean(result.error)
    return result.status == "failed" and error.endswith("pending_reconcile")


def _clean(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "FAQMacroPublishAttempt",
    "FAQMacroPublishAttemptRepository",
    "FAQMacroPublishSummary",
    "FAQMacroWritebackPublishService",
    "PUBLISHED_FAQ_STATUS",
    "SUCCESSFUL_MACRO_STATUSES",
]
