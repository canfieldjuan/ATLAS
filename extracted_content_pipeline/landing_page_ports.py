"""Standalone ports for the AI Content Ops Landing Pages product.

Landing pages are the fourth content asset type in the AI Content Ops
surface (parallel to email campaigns, blog posts, and structured
reports). Where ``CampaignDraft`` is shaped for short-form
transactional output and ``ReportDraft`` is shaped for structured
vendor-pressure reports, a ``LandingPageDraft`` carries a
marketing-shaped payload: hero block, ordered body sections, CTA,
SEO meta, and the marketing-campaign context (name / persona /
value_prop) the page was generated for.

Trigger shape is per-campaign (not per-opportunity): hosts pass a
:class:`MarketingCampaign` describing the campaign to generate against,
not an opportunity row. This is the primary structural difference
between landing pages and the other content assets.

Storage lives in the ``landing_pages`` table (migration 274) with the
same status-lifecycle semantics as ``b2b_campaigns`` and ``reports``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .campaign_ports import JsonDict, TenantScope


@dataclass(frozen=True)
class MarketingCampaign:
    """The host-provided marketing-campaign input for landing-page generation.

    A ``MarketingCampaign`` is the per-call input shape for the
    landing-page generator. Hosts identify the campaign by ``name``;
    the generator uses ``persona`` and ``value_prop`` plus optional
    ``vendors`` / ``categories`` / ``tags`` to anchor the page.

    ``context`` is a free-form bag for host-specific extras (analytics
    tags, A/B variant labels, etc.) that the generator passes through
    to the LLM payload but does not interpret.
    """

    name: str
    persona: str = ""
    value_prop: str = ""
    vendors: Sequence[str] = field(default_factory=tuple)
    categories: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    context: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "persona": self.persona,
            "value_prop": self.value_prop,
            "vendors": list(self.vendors),
            "categories": list(self.categories),
            "tags": list(self.tags),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class LandingPageSection:
    """A single ordered body section within a landing page.

    Note (deliberate divergence from ``ReportSection``): landing-page
    sections do NOT carry ``claim_ids`` or ``evidence_ids``. Marketing
    copy is rendered as flat prose with optional social-proof blocks;
    per-section claim attribution is a report-level concern. Hosts that
    want to surface citations can use the draft-level
    ``LandingPageDraft.reference_ids`` instead. The flexible
    ``metadata`` bag absorbs any per-section host-specific fields.
    """

    id: str
    title: str
    body_markdown: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "title": self.title,
            "body_markdown": self.body_markdown,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LandingPageDraft:
    """A generated, not-yet-approved marketing landing-page draft.

    Persists into the ``landing_pages`` table via
    ``LandingPageRepository.save_drafts``.

    ``hero``, ``cta``, and ``meta`` are flexible mappings (rather than
    typed sub-dataclasses) so hosts and renderers can extend them with
    product-specific fields (analytics tags, secondary CTAs, etc.)
    without a schema change. Common keys:

      * ``hero``: ``headline``, ``subheadline``, ``cta_label``,
        ``cta_url``, ``image_url``
      * ``cta``: ``label``, ``url``, ``variant``,
        ``secondary_label``, ``secondary_url``
      * ``meta``: ``title_tag``, ``description``, ``og_image_url``,
        ``og_title``, ``canonical_url``
    """

    campaign_name: str
    persona: str
    value_prop: str
    title: str
    slug: str
    hero: Mapping[str, Any] = field(default_factory=dict)
    sections: Sequence[LandingPageSection] = field(default_factory=tuple)
    cta: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    reference_ids: Sequence[str] = field(default_factory=tuple)
    metadata: JsonDict = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "campaign_name": self.campaign_name,
            "persona": self.persona,
            "value_prop": self.value_prop,
            "title": self.title,
            "slug": self.slug,
            "hero": dict(self.hero),
            "sections": [section.as_dict() for section in self.sections],
            "cta": dict(self.cta),
            "meta": dict(self.meta),
            "reference_ids": list(self.reference_ids),
            "metadata": dict(self.metadata),
        }


class LandingPageRepository(Protocol):
    """Persistence contract for generated landing-page drafts."""

    async def save_drafts(
        self,
        drafts: Sequence[LandingPageDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts and return the assigned landing-page ids."""

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        campaign_name: str | None = None,
        slug: str | None = None,
        limit: int | None = None,
    ) -> Sequence[LandingPageDraft]:
        """Return drafts filtered by tenant scope and optional facets."""

    async def update_status(
        self,
        landing_page_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        """Update a draft's status (draft / queued / approved / rejected / expired).

        Returns ``True`` when exactly one row is updated, ``False`` when
        no row matches (missing id or wrong tenant scope). This explicit
        hit/miss return is deliberately stricter than the silent-no-op
        pattern in ``CampaignRepository`` / ``ReportRepository`` -- a
        landing page approval flow that targets the wrong id should
        learn about it at the call site, not silently succeed.
        """


__all__ = [
    "LandingPageDraft",
    "LandingPageRepository",
    "LandingPageSection",
    "MarketingCampaign",
]
