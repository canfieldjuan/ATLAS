"""Publish approved FAQ drafts to support macro providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import logging
from typing import Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope
from .faq_macro_writeback import (
    APPROVED_FAQ_STATUS,
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
        )

    def as_dict(self) -> JsonDict:
        data = asdict(self)
        data["ok"] = self.ok
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
    ) -> FAQMacroPublishSummary:
        cleaned_id = _clean(faq_id)
        if not cleaned_id:
            return FAQMacroPublishSummary(faq_id="", found=False)

        draft = await self.faq_repository.get_draft(cleaned_id, scope=scope)
        if draft is None:
            return FAQMacroPublishSummary(faq_id=cleaned_id, found=False)

        preview = build_macro_writeback_preview([draft])
        if not preview.macros:
            summary = _summary(
                faq_id=cleaned_id,
                found=True,
                draft_status=_clean(draft.status),
                preview=preview,
                results=(),
            )
            await self._record_attempt(summary, scope=scope)
            return summary

        results = tuple(await self.provider.publish(preview.macros, scope=scope))
        summary = _summary(
            faq_id=cleaned_id,
            found=True,
            draft_status=_clean(draft.status),
            preview=preview,
            results=results,
        )
        if _should_mark_published(summary):
            status_updated = await self.faq_repository.update_status(
                cleaned_id,
                self.published_status,
                scope=scope,
            )
            summary = replace(summary, draft_status_updated=status_updated)
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
        skipped=tuple(item.as_dict() for item in preview.skipped),
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
    "FAQMacroPublishAttemptRepository",
    "FAQMacroPublishSummary",
    "FAQMacroWritebackPublishService",
    "PUBLISHED_FAQ_STATUS",
    "SUCCESSFUL_MACRO_STATUSES",
]
