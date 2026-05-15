"""Standalone ports for the AI Content Ops Sales Briefs product.

Sales briefs are the fifth content asset type in the AI Content Ops
surface (parallel to email campaigns, blog posts, structured reports,
and marketing landing pages). Where ``ReportDraft`` is shaped for a
customer-facing analytical document with a long executive summary, a
``SalesBriefDraft`` is tuned for a salesperson preparing for a specific
opportunity: a punchy one-line ``headline`` ("why this account, why
now") plus ordered sections (account context, signals, talking points,
risks, next actions).

Section shape mirrors ``ReportSection`` -- ``id`` / ``title`` /
``body_markdown`` / ``claim_ids`` / ``evidence_ids`` / ``metadata`` --
so renderers can reuse the same component for both reports and briefs.

Storage lives in the ``sales_briefs`` table (migration 275) with the
same status-lifecycle semantics as ``b2b_campaigns`` / ``reports`` /
``landing_pages``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from .campaign_ports import JsonDict, TargetMode, TenantScope


BriefType = Literal[
    "pre_call",
    "renewal",
    "displacement",
    "discovery",
]
"""Recognised ``brief_type`` values for ``SalesBriefDraft.brief_type``.
Documents the v0 taxonomy without locking it -- hosts can extend with
custom snake_case labels (e.g., ``"qbr_prep"``, ``"win_back"``); the
alias is a type-checker hint, not a runtime check."""


@dataclass(frozen=True)
class SalesBriefSection:
    """A single ordered section within a sales brief.

    Shape is identical to ``ReportSection`` so a future shared renderer
    can consume both. The semantic content differs (a brief's section
    is sales-facing copy; a report's is customer-facing prose) but the
    typed shape is the same -- ``claim_ids`` / ``evidence_ids`` allow
    briefs produced from a multi-pass reasoning output to carry through
    the same provenance the reports pipeline already supports.
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
class SalesBriefDraft:
    """A generated, not-yet-approved sales brief.

    Persists into the ``sales_briefs`` table via
    ``SalesBriefRepository.save_drafts``.

    Distinct from ``ReportDraft``:
    - ``brief_type`` (pre_call, renewal, displacement, discovery, ...)
      vs. ``report_type`` (vendor_pressure, market_intel, ...)
    - ``headline`` -- a one-line punchy framing (~140 chars) instead
      of a longer ``summary`` paragraph. The elevator pitch the rep
      reads in the 30 seconds before walking into the meeting.
    """

    target_id: str
    target_mode: str
    brief_type: str
    title: str
    headline: str
    sections: Sequence[SalesBriefSection] = field(default_factory=tuple)
    reference_ids: Sequence[str] = field(default_factory=tuple)
    metadata: JsonDict = field(default_factory=dict)
    id: str = ""
    status: str = ""

    def as_dict(self) -> JsonDict:
        return {
            "target_id": self.target_id,
            "target_mode": self.target_mode,
            "brief_type": self.brief_type,
            "title": self.title,
            "headline": self.headline,
            "sections": [section.as_dict() for section in self.sections],
            "reference_ids": list(self.reference_ids),
            "metadata": dict(self.metadata),
            "id": self.id,
            "status": self.status,
        }


class SalesBriefRepository(Protocol):
    """Persistence contract for generated sales briefs."""

    async def save_drafts(
        self,
        drafts: Sequence[SalesBriefDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts and return the assigned brief ids."""

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        brief_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[SalesBriefDraft]:
        """Return drafts filtered by tenant scope and optional facets."""

    async def update_status(
        self,
        brief_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        """Update a draft's status. Returns True on hit, False on miss."""

    async def update_statuses(
        self,
        brief_ids: Sequence[str],
        status: str,
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Bulk-update draft statuses and return ids that matched the tenant scope."""


__all__ = [
    "BriefType",
    "SalesBriefDraft",
    "SalesBriefRepository",
    "SalesBriefSection",
    "TargetMode",
]
