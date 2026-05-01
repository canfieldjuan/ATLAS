"""Pydantic v2 models for the email campaign engine's JSONB blobs.

Models cover four storage surfaces:

  - CampaignMetadata          → b2b_campaigns.metadata
  - CompanyContext            → campaign_sequences.company_context
  - SellingContext            → campaign_sequences.selling_context
  - BriefingData              → b2b_vendor_briefings.briefing_data

Plus one cross-cutting model used by the multi-ESP webhook layer:

  - CanonicalEvent            → normalized email-engagement event

All models open with ``extra='allow'`` and ``schema_version=1``. The
``extra='allow'`` posture lets pre-existing rows round-trip without
validation errors during the soak window described in
docs/progress/email-campaign-pilot-readiness.md (Gap 4). Once the soak
window confirms no unknown keys, flip to ``extra='forbid'`` in a follow-up
PR and remove the read-side fallback paths.

Each model exposes ``model_json_schema()`` for customer integrations;
schema export lives in docs/schemas/ (generated, not hand-written).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


_BASE_CONFIG = ConfigDict(extra="allow", populate_by_name=True)


class CampaignMetadata(BaseModel):
    """Shape of b2b_campaigns.metadata.

    Built primarily by ``_campaign_storage_metadata`` in
    atlas_brain/autonomous/tasks/b2b_campaign_generation.py and amended by
    ``merge_campaign_revalidation_metadata`` in
    atlas_brain/services/campaign_quality.py. Keys reflect the campaign's
    reasoning lineage so quality re-checks at approve / queue / send time
    can replay the same anchors without re-fetching upstream data.
    """

    model_config = _BASE_CONFIG

    schema_version: int = 1
    tier: Optional[str] = None
    target_mode: Optional[str] = None
    reasoning_anchor_examples: Optional[dict[str, Any]] = None
    reasoning_witness_highlights: Optional[list[dict[str, Any]]] = None
    reasoning_reference_ids: Optional[dict[str, Any]] = None
    campaign_proof_terms: Optional[list[str]] = None
    opportunity_claim: Optional[dict[str, Any]] = None
    opportunity_claim_gate: Optional[dict[str, Any]] = None
    opportunity_claims: Optional[list[dict[str, Any]]] = None
    generation_audit: Optional[dict[str, Any]] = None
    latest_specificity_audit: Optional[dict[str, Any]] = None


class PainCategoryEntry(BaseModel):
    """One row in CompanyContext.pain_categories."""

    model_config = _BASE_CONFIG

    category: str
    severity: Optional[str] = None


class CompetitorEntry(BaseModel):
    """One row in CompanyContext.competitors_considering."""

    model_config = _BASE_CONFIG

    name: Optional[str] = None
    vendor_name: Optional[str] = None


class CompanyContext(BaseModel):
    """Shape of campaign_sequences.company_context (frozen at sequence creation).

    Compaction rules at
    atlas_brain/autonomous/tasks/_campaign_sequence_context.py drop
    several keys at storage time (selling, comparison_asset,
    reasoning_contracts, qualification, partner, primary_blog_post,
    supporting_blog_posts) — those live in SellingContext or are
    re-derived per send, so they're intentionally absent here.
    """

    model_config = _BASE_CONFIG

    schema_version: int = 1
    company: Optional[str] = None
    churning_from: Optional[str] = None
    category: Optional[str] = None
    industry: Optional[str] = None
    role_type: Optional[str] = None
    reviewer_title: Optional[str] = None
    company_size: Optional[str] = None
    seat_count: Optional[int] = None
    contract_end: Optional[str] = None
    decision_timeline: Optional[str] = None
    buying_stage: Optional[str] = None
    urgency: Optional[float] = None
    sentiment_direction: Optional[str] = None
    primary_workflow: Optional[str] = None

    pain_categories: Optional[list[PainCategoryEntry]] = None
    competitors_considering: Optional[list[CompetitorEntry]] = None
    feature_gaps: Optional[list[str]] = None
    integration_stack: Optional[list[str]] = None
    key_quotes: Optional[list[str]] = None

    signal_summary: Optional[dict[str, Any]] = None
    briefing_context: Optional[dict[str, Any]] = None
    reasoning_context: Optional[dict[str, Any]] = None
    incumbent_reasoning: Optional[dict[str, Any]] = None
    incumbent_archetypes: Optional[list[dict[str, Any]]] = None
    category_intelligence: Optional[dict[str, Any]] = None
    opportunity_source: Optional[str] = None
    opportunity_claim: Optional[dict[str, Any]] = None
    recommended_alternatives: Optional[list[dict[str, Any]]] = None
    supplemental_recommended_alternatives: Optional[list[dict[str, Any]]] = None


class BlogPostRef(BaseModel):
    """Lightweight blog-post reference embedded in SellingContext."""

    model_config = _BASE_CONFIG

    id: Optional[str] = None
    slug: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    pain_tags: Optional[list[str]] = None


class SellingContext(BaseModel):
    """Shape of campaign_sequences.selling_context.

    Sender identity and the matched content assets used for prompt
    assembly. Frozen at sequence creation so follow-up steps see the
    same primary asset as the cold email.
    """

    model_config = _BASE_CONFIG

    schema_version: int = 1
    sender_name: Optional[str] = None
    sender_title: Optional[str] = None
    sender_company: Optional[str] = None
    booking_url: Optional[str] = None
    product_name: Optional[str] = None
    affiliate_url: Optional[str] = None
    primary_blog_post: Optional[BlogPostRef] = None
    blog_posts: Optional[list[BlogPostRef]] = None


class BriefingData(BaseModel):
    """Shape of b2b_vendor_briefings.briefing_data.

    Conservative model — the briefing structure is still consolidating
    upstream. extra='allow' keeps unknown keys flowing through until the
    upstream writer is itself schema-pinned.
    """

    model_config = _BASE_CONFIG

    schema_version: int = 1
    vendor_name: Optional[str] = None
    pain_categories: Optional[list[str]] = None
    key_quotes: Optional[list[str]] = None
    battle_cards: Optional[list[dict[str, Any]]] = None
    recommended_actions: Optional[list[dict[str, Any]]] = None
    competitors: Optional[list[dict[str, Any]]] = None
    summary: Optional[str] = None


CanonicalEventType = Literal[
    "delivered",
    "opened",
    "clicked",
    "bounced",
    "complained",
    "unsubscribed",
]


class CanonicalEvent(BaseModel):
    """Provider-agnostic email-engagement event.

    Each WebhookProvider implementation in atlas_brain/services/email_webhooks/
    parses its raw payload into this shape. The route handler in
    atlas_brain/api/campaign_webhooks.py only needs to know about
    CanonicalEvent fields when updating b2b_campaigns / campaign_sequences.

    timestamp is an ISO-8601 string at this layer to keep parsing concerns
    inside the provider; the handler converts to TIMESTAMPTZ at write
    time.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    provider: str = Field(..., description="ESP name: resend|ses|sendgrid|postmark|mailgun")
    event_type: CanonicalEventType
    message_id: str = Field(..., description="ESP-side message identifier (esp_message_id)")
    recipient_email: str
    timestamp: str = Field(..., description="ISO-8601 event timestamp from the provider")
    bounce_type: Optional[str] = None
    bounce_subtype: Optional[str] = None
    click_url: Optional[str] = None
    user_agent: Optional[str] = None
    ip: Optional[str] = None
    raw: Optional[dict[str, Any]] = Field(
        default=None,
        description="Provider-specific payload retained for debugging; not persisted",
    )


__all__ = [
    "CampaignMetadata",
    "PainCategoryEntry",
    "CompetitorEntry",
    "CompanyContext",
    "BlogPostRef",
    "SellingContext",
    "BriefingData",
    "CanonicalEvent",
    "CanonicalEventType",
]
