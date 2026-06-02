"""Landing-page SEO/GEO/AEO input contract.

The input keys here are intentionally explicit. They are the fields the public
control surfaces may pass into ``MarketingCampaign.context`` for landing-page
generation; unrelated request fields must not leak into the LLM payload.
"""

from __future__ import annotations

from typing import Any


LANDING_PAGE_INPUT_ASSET = "landing_page"

LANDING_PAGE_SEO_GEO_AEO_INPUT_KEYS: tuple[str, ...] = (
    "target_keyword",
    "secondary_keywords",
    "search_intent",
    "primary_entity",
    "audience_entity",
    "competitors",
    "objections",
    "faq_questions",
    "source_period",
    "internal_links",
    "cta_label",
    "cta_url",
)

LANDING_PAGE_SUPPORT_TICKET_SOURCE_INPUT_KEYS: tuple[str, ...] = (
    "source_row_count",
    "included_ticket_row_count",
    "skipped_ticket_row_count",
    "truncated_ticket_row_count",
    "question_like_ticket_count",
    "has_dated_window",
    "top_ticket_clusters",
    "customer_wording_examples",
    "support_ticket_resolution_evidence_present",
    "support_ticket_resolution_evidence_count",
    "support_ticket_resolution_examples",
    "has_measured_outcomes",
    "measured_outcome_count",
    "measured_outcome_examples",
)

LANDING_PAGE_REVIEW_SOURCE_INPUT_KEYS: tuple[str, ...] = (
    "review_source_material",
    "review_source_count",
)

LANDING_PAGE_COMPETITIVE_SOURCE_INPUT_KEYS: tuple[str, ...] = (
    "competitive_source_material",
    "competitive_source_count",
    "displacement_source_count",
)

LANDING_PAGE_EXISTING_CONTEXT_KEYS: tuple[str, ...] = (
    "industry",
    "pain_points",
    "differentiators",
    "customer_segments",
    "key_metrics",
    "proof_points",
    "competitive_alternatives",
)

LANDING_PAGE_CONTEXT_INPUT_KEYS: frozenset[str] = frozenset((
    *LANDING_PAGE_EXISTING_CONTEXT_KEYS,
    *LANDING_PAGE_SEO_GEO_AEO_INPUT_KEYS,
    *LANDING_PAGE_SUPPORT_TICKET_SOURCE_INPUT_KEYS,
    *LANDING_PAGE_REVIEW_SOURCE_INPUT_KEYS,
    *LANDING_PAGE_COMPETITIVE_SOURCE_INPUT_KEYS,
))


_LANDING_PAGE_SEO_GEO_AEO_INPUT_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "key": "target_keyword",
        "label": "Target keyword",
        "type": "string",
        "placeholder": "customer support FAQ",
    },
    {
        "key": "secondary_keywords",
        "label": "Secondary keywords",
        "type": "string_list",
        "placeholder": "support ticket FAQ\nreduce repeat support tickets",
    },
    {
        "key": "search_intent",
        "label": "Search intent",
        "type": "string",
        "placeholder": "Small SaaS teams looking to turn support tickets into help-center answers.",
    },
    {
        "key": "primary_entity",
        "label": "Primary entity",
        "type": "string",
        "placeholder": "FAQ Report",
    },
    {
        "key": "audience_entity",
        "label": "Audience entity",
        "type": "string",
        "placeholder": "10-50 person SaaS teams",
    },
    {
        "key": "competitors",
        "label": "Competitors or alternatives",
        "type": "string_list",
        "placeholder": "manual help-center cleanup\nchatbot setup",
    },
    {
        "key": "objections",
        "label": "Objections to address",
        "type": "string_list",
        "placeholder": "Will this publish automatically?\nDo we need a docs person?",
    },
    {
        "key": "faq_questions",
        "label": "FAQ questions to include",
        "type": "string_list",
        "placeholder": "What do you need from us?\nWhat happens after upload?",
    },
    {
        "key": "source_period",
        "label": "Source period",
        "type": "string",
        "placeholder": "Last 90 days of support tickets",
    },
    {
        "key": "internal_links",
        "label": "Internal links",
        "type": "string_list",
        "placeholder": "/systems/ai-content-ops\n/systems/ai-content-ops/intake",
    },
    {
        "key": "cta_label",
        "label": "CTA label",
        "type": "string",
        "placeholder": "Upload Ticket CSV -- Free Analysis",
    },
    {
        "key": "cta_url",
        "label": "CTA URL",
        "type": "string",
        "placeholder": "/systems/ai-content-ops/intake",
    },
)


def landing_page_seo_geo_aeo_input_contracts() -> dict[str, dict[str, Any]]:
    """Return wire contracts for landing-page SEO/GEO/AEO inputs."""

    return {
        item["key"]: {
            **item,
            "asset": LANDING_PAGE_INPUT_ASSET,
            "group": "seo_geo_aeo",
        }
        for item in _LANDING_PAGE_SEO_GEO_AEO_INPUT_CONTRACTS
    }


__all__ = [
    "LANDING_PAGE_CONTEXT_INPUT_KEYS",
    "LANDING_PAGE_COMPETITIVE_SOURCE_INPUT_KEYS",
    "LANDING_PAGE_EXISTING_CONTEXT_KEYS",
    "LANDING_PAGE_INPUT_ASSET",
    "LANDING_PAGE_REVIEW_SOURCE_INPUT_KEYS",
    "LANDING_PAGE_SEO_GEO_AEO_INPUT_KEYS",
    "LANDING_PAGE_SUPPORT_TICKET_SOURCE_INPUT_KEYS",
    "landing_page_seo_geo_aeo_input_contracts",
]
