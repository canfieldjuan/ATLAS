"""Standalone ports for persisted stat-card drafts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope


@dataclass(frozen=True)
class StatCardDraft:
    """A generated, not-yet-approved stat-card draft."""

    target_id: str
    target_mode: str
    theme: str
    metric_label: str
    metric_value: Any
    metric_display: str
    claim: str
    headline: str
    supporting_text: str
    evidence: str
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
            "metric_label": self.metric_label,
            "metric_value": self.metric_value,
            "metric_display": self.metric_display,
            "claim": self.claim,
            "headline": self.headline,
            "supporting_text": self.supporting_text,
            "evidence": self.evidence,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "company_name": self.company_name,
            "vendor_name": self.vendor_name,
            "pain_points": list(self.pain_points),
            "metadata": dict(self.metadata),
            "id": self.id,
            "status": self.status,
        }


class StatCardRepository(Protocol):
    """Persistence contract for generated stat-card drafts."""

    async def save_drafts(
        self,
        drafts: Sequence[StatCardDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts and return assigned stat-card ids."""

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        theme: str | None = None,
        limit: int | None = None,
    ) -> Sequence[StatCardDraft]:
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
    "StatCardDraft",
    "StatCardRepository",
]
