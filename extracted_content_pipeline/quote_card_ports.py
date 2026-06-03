"""Standalone ports for persisted quote-card drafts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope


@dataclass(frozen=True)
class QuoteCardDraft:
    """A generated, not-yet-approved quote-card draft."""

    target_id: str
    target_mode: str
    theme: str
    quote: str
    attribution: str
    headline: str
    supporting_text: str
    source_id: str = ""
    source_type: str = ""
    company_name: str = ""
    vendor_name: str = ""
    pain_points: Sequence[str] = field(default_factory=tuple)
    metadata: JsonDict = field(default_factory=dict)
    id: str = ""
    status: str = ""

    def as_dict(self) -> JsonDict:
        return {
            "target_id": self.target_id,
            "target_mode": self.target_mode,
            "theme": self.theme,
            "quote": self.quote,
            "attribution": self.attribution,
            "headline": self.headline,
            "supporting_text": self.supporting_text,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "company_name": self.company_name,
            "vendor_name": self.vendor_name,
            "pain_points": list(self.pain_points),
            "metadata": dict(self.metadata),
            "id": self.id,
            "status": self.status,
        }


class QuoteCardRepository(Protocol):
    """Persistence contract for generated quote-card drafts."""

    async def save_drafts(
        self,
        drafts: Sequence[QuoteCardDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts and return assigned quote-card ids."""

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        theme: str | None = None,
        limit: int | None = None,
    ) -> Sequence[QuoteCardDraft]:
        """Return drafts filtered by tenant scope and optional facets."""

    async def update_status(
        self,
        draft_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        """Update a draft status and return True on hit."""

    async def update_statuses(
        self,
        draft_ids: Sequence[str],
        status: str,
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Bulk-update draft statuses and return ids that matched scope."""


__all__ = [
    "QuoteCardDraft",
    "QuoteCardRepository",
]
