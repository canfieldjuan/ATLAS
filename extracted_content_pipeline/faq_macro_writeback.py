"""Preview support-tool macro writeback from approved FAQ drafts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope
from .ticket_faq_ports import TicketFAQDraft


APPROVED_FAQ_STATUS = "approved"
RESOLUTION_EVIDENCE_STATUS = "resolution_evidence"

MacroPublishStatus = Literal["dry_run", "published", "updated", "skipped", "failed"]
MacroSkipReason = Literal["draft_not_approved", "answer_not_verified", "missing_question", "missing_resolution_body"]


@dataclass(frozen=True)
class SupportMacroDraft:
    """Platform-agnostic support macro / saved reply draft."""

    title: str
    body: str
    category: str = ""
    faq_draft_id: str = ""
    faq_item_id: str = ""
    source_ids: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        data = asdict(self)
        data["source_ids"] = list(self.source_ids)
        return data


@dataclass(frozen=True)
class MacroWritebackSkippedItem:
    """FAQ item that was not eligible for macro writeback."""

    reason: MacroSkipReason
    faq_draft_id: str = ""
    faq_item_id: str = ""
    question: str = ""
    draft_status: str = ""
    answer_evidence_status: str = ""

    def as_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class MacroPublishResult:
    """Per-item result returned by a macro writeback provider."""

    macro: SupportMacroDraft
    status: MacroPublishStatus
    external_id: str = ""
    error: str = ""

    def as_dict(self) -> JsonDict:
        return {
            "macro": self.macro.as_dict(),
            "status": self.status,
            "external_id": self.external_id,
            "error": self.error,
        }


@dataclass(frozen=True)
class MacroWritebackMapping:
    """Tenant-scoped mapping from one FAQ item to one external macro."""

    platform: str
    faq_draft_id: str
    faq_item_id: str
    external_id: str
    external_url: str = ""
    metadata: JsonDict = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class MacroWritebackPreview:
    """Dry-run preview of macros that would be written back."""

    macros: tuple[SupportMacroDraft, ...] = ()
    skipped: tuple[MacroWritebackSkippedItem, ...] = ()

    @property
    def publishable_count(self) -> int:
        return len(self.macros)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def as_dict(self) -> JsonDict:
        return {
            "publishable_count": self.publishable_count,
            "skipped_count": self.skipped_count,
            "macros": [macro.as_dict() for macro in self.macros],
            "skipped": [item.as_dict() for item in self.skipped],
        }


class MacroPublishProvider(Protocol):
    async def publish(
        self,
        macros: Sequence[SupportMacroDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[MacroPublishResult]:
        """Publish or preview support macros for one tenant scope."""


class MacroWritebackMappingRepository(Protocol):
    """Persistence port for macro writeback idempotency mappings."""

    async def get_mapping(
        self,
        *,
        platform: str,
        faq_draft_id: str,
        faq_item_id: str,
        scope: TenantScope,
    ) -> MacroWritebackMapping | None:
        """Return the existing external macro mapping for one FAQ item."""

    async def upsert_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> MacroWritebackMapping:
        """Create or update the external macro mapping for one FAQ item."""


class DryRunMacroPublishProvider:
    async def publish(
        self,
        macros: Sequence[SupportMacroDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[MacroPublishResult]:
        _ = scope
        return tuple(MacroPublishResult(macro=macro, status="dry_run") for macro in macros)


def build_macro_writeback_preview(
    drafts: Sequence[TicketFAQDraft],
) -> MacroWritebackPreview:
    """Return publishable macro drafts and explicit skip reasons."""

    macros: list[SupportMacroDraft] = []
    skipped: list[MacroWritebackSkippedItem] = []

    for draft in drafts:
        for index, item in enumerate(draft.items, start=1):
            item_id = _item_id(draft, item, index)
            question = _clean_text(item.get("question"))
            evidence_status = _clean_text(item.get("answer_evidence_status"))
            draft_status = _clean_text(draft.status)

            def skip(reason: MacroSkipReason) -> None:
                skipped.append(
                    MacroWritebackSkippedItem(
                        reason=reason,
                        faq_draft_id=_clean_text(draft.id),
                        faq_item_id=item_id,
                        question=question,
                        draft_status=draft_status,
                        answer_evidence_status=evidence_status,
                    )
                )

            if draft_status != APPROVED_FAQ_STATUS:
                skip("draft_not_approved")
                continue
            if evidence_status != RESOLUTION_EVIDENCE_STATUS:
                skip("answer_not_verified")
                continue
            if not question:
                skip("missing_question")
                continue
            body = _macro_body(item)
            if not body:
                skip("missing_resolution_body")
                continue
            macros.append(
                SupportMacroDraft(
                    title=question,
                    body=body,
                    category=_clean_text(item.get("topic")),
                    faq_draft_id=_clean_text(draft.id),
                    faq_item_id=item_id,
                    source_ids=_string_tuple(item.get("source_ids")),
                    metadata={
                        "target_id": draft.target_id,
                        "target_mode": draft.target_mode,
                        "draft_title": draft.title,
                    },
                )
            )

    return MacroWritebackPreview(macros=tuple(macros), skipped=tuple(skipped))


def _macro_body(item: Mapping[str, Any]) -> str:
    resolution_text = _clean_text(item.get("resolution_text"))
    if resolution_text:
        return resolution_text

    steps = _string_tuple(item.get("steps"))
    if steps:
        return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))

    return _clean_text(item.get("answer"))


def _item_id(
    draft: TicketFAQDraft,
    item: Mapping[str, Any],
    index: int,
) -> str:
    for key in ("faq_item_id", "id", "source_id"):
        value = _clean_text(item.get(key))
        if value:
            return value
    draft_id = _clean_text(draft.id)
    if draft_id:
        return f"{draft_id}:item-{index}"
    return f"item-{index}"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return (cleaned,) if cleaned else ()
    if isinstance(value, Sequence):
        return tuple(cleaned for item in value if (cleaned := _clean_text(item)))
    return ()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


__all__ = [
    "APPROVED_FAQ_STATUS",
    "RESOLUTION_EVIDENCE_STATUS",
    "DryRunMacroPublishProvider",
    "MacroPublishProvider",
    "MacroPublishResult",
    "MacroPublishStatus",
    "MacroSkipReason",
    "MacroWritebackMapping",
    "MacroWritebackMappingRepository",
    "MacroWritebackPreview",
    "MacroWritebackSkippedItem",
    "SupportMacroDraft",
    "build_macro_writeback_preview",
]
