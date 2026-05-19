"""Standalone ports for persisted ticket FAQ Markdown drafts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope


@dataclass(frozen=True)
class TicketFAQDraft:
    """A generated, not-yet-approved ticket FAQ Markdown document."""

    target_id: str
    target_mode: str
    title: str
    markdown: str
    items: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    source_count: int = 0
    ticket_source_count: int = 0
    output_checks: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: JsonDict = field(default_factory=dict)
    id: str = ""
    status: str = ""

    def as_dict(self) -> JsonDict:
        return {
            "target_id": self.target_id,
            "target_mode": self.target_mode,
            "title": self.title,
            "markdown": self.markdown,
            "items": [dict(item) for item in self.items],
            "source_count": self.source_count,
            "ticket_source_count": self.ticket_source_count,
            "output_checks": dict(self.output_checks),
            "warnings": [dict(warning) for warning in self.warnings],
            "metadata": dict(self.metadata),
            "id": self.id,
            "status": self.status,
        }


class TicketFAQRepository(Protocol):
    """Persistence contract for generated ticket FAQ Markdown drafts."""

    async def save_drafts(
        self,
        drafts: Sequence[TicketFAQDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts and return assigned FAQ draft ids."""

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        limit: int | None = None,
    ) -> Sequence[TicketFAQDraft]:
        """Return drafts filtered by tenant scope and optional facets."""

    async def update_status(
        self,
        faq_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        """Update a draft status and return True on hit."""

    async def update_statuses(
        self,
        faq_ids: Sequence[str],
        status: str,
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Bulk-update draft statuses and return ids that matched scope."""


__all__ = [
    "TicketFAQDraft",
    "TicketFAQRepository",
]
