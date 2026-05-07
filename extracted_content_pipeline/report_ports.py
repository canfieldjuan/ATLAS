"""Standalone ports for the AI Content Ops Reports product.

Reports are the third content asset type in the AI Content Ops surface
(parallel to email campaigns and blog posts). Where ``CampaignDraft`` is
shaped for short-form transactional output (subject + body), a
``ReportDraft`` carries a typed structured payload (title, summary,
ordered sections, cited references) so renderers can consume the report
without parsing markdown.

Storage lives in the ``reports`` table (migration 273) with the same
status-lifecycle semantics as ``b2b_campaigns``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope


@dataclass(frozen=True)
class ReportSection:
    """A single ordered section within a structured report.

    The shape mirrors ``extracted_reasoning_core.types.NarrativePlan.sections``
    so reports produced from a multi-pass reasoning output can pass the
    bridge's pre-structured plan straight through with minimal reshaping.
    """

    id: str
    title: str
    body_markdown: str
    claim_ids: Sequence[str] = field(default_factory=tuple)
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "title": self.title,
            "body_markdown": self.body_markdown,
            "claim_ids": list(self.claim_ids),
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReportDraft:
    """A generated, not-yet-approved structured report.

    Persists into the ``reports`` table via ``ReportRepository.save_drafts``.
    """

    target_id: str
    target_mode: str
    report_type: str
    title: str
    summary: str
    sections: Sequence[ReportSection] = field(default_factory=tuple)
    reference_ids: Sequence[str] = field(default_factory=tuple)
    metadata: JsonDict = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "target_id": self.target_id,
            "target_mode": self.target_mode,
            "report_type": self.report_type,
            "title": self.title,
            "summary": self.summary,
            "sections": [section.as_dict() for section in self.sections],
            "reference_ids": list(self.reference_ids),
            "metadata": dict(self.metadata),
        }


class ReportRepository(Protocol):
    """Persistence contract for generated structured reports."""

    async def save_drafts(
        self,
        drafts: Sequence[ReportDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts and return the assigned report ids."""

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        report_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[ReportDraft]:
        """Return drafts filtered by tenant scope and optional facets."""

    async def update_status(
        self,
        report_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> None:
        """Update a draft's status (draft / queued / approved / rejected / expired)."""


__all__ = [
    "ReportDraft",
    "ReportRepository",
    "ReportSection",
]
