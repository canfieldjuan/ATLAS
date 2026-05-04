"""B2B source impact ledger and consumer wiring baselines.

Static mappings describe which fields each source can reliably contribute,
which canonical pools should absorb that evidence, and which downstream
surfaces are expected to improve. Dynamic helpers add live field-coverage
baselines from ``b2b_reviews``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...config import settings
from ..scraping.capabilities import get_all_capabilities
from ..scraping.sources import (
    ReviewSource,
    display_name as source_display_name,
    parse_source_allowlist,
)


@dataclass(frozen=True)
class SourceImpactProfile:
    source: str
    source_family: str
    expansion_stage: str
    work_type: tuple[str, ...]
    reliable_fields: tuple[str, ...]
    target_pools: tuple[str, ...]
    expected_consumers: tuple[str, ...]
    consumers_without_material_benefit: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "display_name": source_display_name(self.source),
            "source_family": self.source_family,
            "expansion_stage": self.expansion_stage,
            "work_type": list(self.work_type),
            "reliable_fields": list(self.reliable_fields),
            "target_pools": list(self.target_pools),
            "expected_consumers": list(self.expected_consumers),
            "consumers_without_material_benefit": list(
                self.consumers_without_material_benefit
            ),
            "notes": self.notes,
        }


def _profile(
    source: str,
    *,
    source_family: str,
    expansion_stage: str,
    work_type: tuple[str, ...],
    reliable_fields: tuple[str, ...],
    target_pools: tuple[str, ...],
    expected_consumers: tuple[str, ...],
    consumers_without_material_benefit: tuple[str, ...],
    notes: str,
) -> SourceImpactProfile:
    return SourceImpactProfile(
        source=source,
        source_family=source_family,
        expansion_stage=expansion_stage,
        work_type=work_type,
        reliable_fields=reliable_fields,
        target_pools=target_pools,
        expected_consumers=expected_consumers,
        consumers_without_material_benefit=consumers_without_material_benefit,
        notes=notes,
    )


_STRUCTURED_BENEFICIARIES = (
    "watchlists_accounts_in_motion",
    "b2b_accounts_in_motion",
    "crm_push_candidates",
    "b2b_vendor_briefing",
    "b2b_campaign_generation",
)
_TEMPORAL_BENEFICIARIES = (
    "b2b_churn_alert",
    "b2b_churn_reports",
    "b2b_battle_cards",
    "b2b_vendor_briefing",
    "change_events",
)
_FIRMOGRAPHIC_NON_BENEFICIARIES = (
    "crm_push_candidates",
    "firmographic_segment_playbooks",
)
_COMMUNITY_NON_BENEFICIARIES = (
    "crm_push_candidates",
    "named_account_precision",
)

_SOURCE_IMPACT_PROFILES: dict[str, SourceImpactProfile] = {
    "getapp": _profile(
        "getapp",
        source_family="structured_review",
        expansion_stage="recover_zero_row_core_source",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "reviewer_title",
            "reviewer_company",
            "company_size",
            "industry",
            "pricing_context",
            "pain_quotes",
        ),
        target_pools=("segment", "accounts", "evidence_vault"),
        expected_consumers=_STRUCTURED_BENEFICIARIES,
        consumers_without_material_benefit=("change_events",),
        notes="Best recovery target for segment/account depth because parser support exists but live coverage is missing.",
    ),
    "trustradius": _profile(
        "trustradius",
        source_family="structured_review",
        expansion_stage="expand_high_yield_structured_source",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "reviewer_title",
            "reviewer_company",
            "company_size",
            "industry",
            "buying_stage",
            "alternatives_considered",
            "competitive_quotes",
        ),
        target_pools=("segment", "accounts", "displacement", "evidence_vault"),
        expected_consumers=_STRUCTURED_BENEFICIARIES
        + ("b2b_battle_cards", "displacement_map"),
        consumers_without_material_benefit=("change_events",),
        notes="Best current blend of structured identity plus competitive context.",
    ),
    "gartner": _profile(
        "gartner",
        source_family="structured_review",
        expansion_stage="expand_high_yield_structured_source",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "reviewer_title",
            "company_size",
            "role_type",
            "department",
            "buying_stage",
        ),
        target_pools=("segment",),
        expected_consumers=(
            "segment_playbooks",
            "watchlists_accounts_in_motion",
            "b2b_vendor_briefing",
        ),
        consumers_without_material_benefit=("crm_push_candidates", "change_events"),
        notes="Sharpens segment and strategic-role coverage more than named-account coverage.",
    ),
    "peerspot": _profile(
        "peerspot",
        source_family="structured_review",
        expansion_stage="expand_high_yield_structured_source",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "role_type",
            "buying_stage",
            "alternatives_considered",
            "migration_context",
            "competitive_quotes",
        ),
        target_pools=("segment", "displacement", "evidence_vault"),
        expected_consumers=(
            "segment_playbooks",
            "b2b_battle_cards",
            "b2b_vendor_briefing",
            "displacement_map",
        ),
        consumers_without_material_benefit=("crm_push_candidates",),
        notes="High leverage in cloud/security verticals where displacement and evaluation-stage detail matter.",
    ),
    "g2": _profile(
        "g2",
        source_family="structured_review",
        expansion_stage="fix_parser_before_scaling",
        work_type=("parser_quality",),
        reliable_fields=(
            "reviewer_title",
            "company_size",
            "industry",
            "pain_quotes",
            "alternatives_considered",
        ),
        target_pools=("segment", "accounts", "evidence_vault"),
        expected_consumers=_STRUCTURED_BENEFICIARIES,
        consumers_without_material_benefit=("change_events",),
        notes="Parser intent is richer than current live conversion, so extraction quality is the bottleneck.",
    ),
    "capterra": _profile(
        "capterra",
        source_family="structured_review",
        expansion_stage="fix_parser_before_scaling",
        work_type=("parser_quality",),
        reliable_fields=(
            "reviewer_title",
            "reviewer_company",
            "company_size",
            "industry",
            "pain_quotes",
        ),
        target_pools=("segment", "accounts", "evidence_vault"),
        expected_consumers=_STRUCTURED_BENEFICIARIES,
        consumers_without_material_benefit=("change_events",),
        notes="Current parser support should feed segment/account pools, but live field conversion is weak.",
    ),
    "software_advice": _profile(
        "software_advice",
        source_family="structured_review",
        expansion_stage="fix_parser_before_scaling",
        work_type=("parser_quality",),
        reliable_fields=(
            "reviewer_title",
            "reviewer_company",
            "company_size",
            "industry",
            "pain_quotes",
        ),
        target_pools=("segment", "accounts", "evidence_vault"),
        expected_consumers=_STRUCTURED_BENEFICIARIES,
        consumers_without_material_benefit=("change_events",),
        notes="Same Gartner Digital Markets family as Capterra/GetApp; extraction quality matters more than raw page depth.",
    ),
    "reddit": _profile(
        "reddit",
        source_family="community_signal",
        expansion_stage="query_tune_social_source",
        work_type=("query_strategy",),
        reliable_fields=(
            "renewal_language",
            "switching_language",
            "migration_status",
            "competitors_mentioned",
            "pain_quotes",
        ),
        target_pools=("evidence_vault", "temporal", "displacement"),
        expected_consumers=_TEMPORAL_BENEFICIARIES,
        consumers_without_material_benefit=_COMMUNITY_NON_BENEFICIARIES,
        notes="Best scale source for timing and displacement, but poor fit for firmographic precision.",
    ),
    "trustpilot": _profile(
        "trustpilot",
        source_family="community_signal",
        expansion_stage="query_tune_social_source",
        work_type=("query_strategy",),
        reliable_fields=(
            "support_escalation",
            "pricing_pressure",
            "renewal_pain",
            "pain_quotes",
        ),
        target_pools=("evidence_vault", "temporal"),
        expected_consumers=_TEMPORAL_BENEFICIARIES,
        consumers_without_material_benefit=_COMMUNITY_NON_BENEFICIARIES,
        notes="Useful for pain and pricing narratives; weak for reviewer identity and named-account routing.",
    ),
    "hackernews": _profile(
        "hackernews",
        source_family="community_signal",
        expansion_stage="query_tune_social_source",
        work_type=("query_strategy",),
        reliable_fields=(
            "active_evaluation",
            "competitors_mentioned",
            "migration_context",
            "technical_replacement_quotes",
        ),
        target_pools=("evidence_vault", "temporal", "displacement"),
        expected_consumers=_TEMPORAL_BENEFICIARIES,
        consumers_without_material_benefit=_COMMUNITY_NON_BENEFICIARIES,
        notes="High-yield for technical replacement chatter, but identity remains thin.",
    ),
    "twitter": _profile(
        "twitter",
        source_family="community_signal",
        expansion_stage="query_tune_social_source",
        work_type=("query_strategy",),
        reliable_fields=(
            "support_escalation",
            "outage_mentions",
            "switching_language",
            "public_competitor_mentions",
        ),
        target_pools=("evidence_vault", "temporal", "displacement"),
        expected_consumers=_TEMPORAL_BENEFICIARIES,
        consumers_without_material_benefit=_COMMUNITY_NON_BENEFICIARIES,
        notes="Good for fast-moving complaint and switching chatter; weak for contract timing and firmographics.",
    ),
    "github": _profile(
        "github",
        source_family="developer_context",
        expansion_stage="conditional_context_expansion",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "technical_pain_quotes",
            "integration_breakage",
            "migration_issues",
            "release_regressions",
        ),
        target_pools=("evidence_vault", "temporal"),
        expected_consumers=("b2b_vendor_briefing", "b2b_battle_cards"),
        consumers_without_material_benefit=_FIRMOGRAPHIC_NON_BENEFICIARIES,
        notes="Treat as technical context only, not as a firmographic or account-identification source.",
    ),
    "stackoverflow": _profile(
        "stackoverflow",
        source_family="developer_context",
        expansion_stage="conditional_context_expansion",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "technical_pain_quotes",
            "implementation_friction",
            "alternative_mentions",
        ),
        target_pools=("evidence_vault", "temporal"),
        expected_consumers=("b2b_vendor_briefing", "b2b_battle_cards"),
        consumers_without_material_benefit=_FIRMOGRAPHIC_NON_BENEFICIARIES,
        notes="Best used for technical friction and quote inventory in devtools categories.",
    ),
    "rss": _profile(
        "rss",
        source_family="news_context",
        expansion_stage="conditional_context_expansion",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "news_events",
            "vendor_announcements",
            "outage_mentions",
            "pricing_announcements",
        ),
        target_pools=("temporal", "evidence_vault"),
        expected_consumers=("change_events", "b2b_vendor_briefing"),
        consumers_without_material_benefit=_FIRMOGRAPHIC_NON_BENEFICIARIES,
        notes="Useful for external context and event correlation rather than direct churn-intent evidence.",
    ),
    "youtube": _profile(
        "youtube",
        source_family="community_signal",
        expansion_stage="conditional_context_expansion",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "video_review_quotes",
            "migration_walkthroughs",
            "pricing_commentary",
        ),
        target_pools=("evidence_vault", "temporal"),
        expected_consumers=("b2b_vendor_briefing", "b2b_battle_cards"),
        consumers_without_material_benefit=_FIRMOGRAPHIC_NON_BENEFICIARIES,
        notes="Treat as quote inventory and narrative context, not as a reliable account source.",
    ),
    "quora": _profile(
        "quora",
        source_family="community_signal",
        expansion_stage="conditional_context_expansion",
        work_type=("query_strategy",),
        reliable_fields=(
            "alternative_discovery",
            "faq_quotes",
            "switching_language",
        ),
        target_pools=("evidence_vault", "displacement"),
        expected_consumers=("b2b_vendor_briefing", "b2b_battle_cards"),
        consumers_without_material_benefit=_COMMUNITY_NON_BENEFICIARIES,
        notes="Low-confidence alternative discovery source with limited timing and company context.",
    ),
    "producthunt": _profile(
        "producthunt",
        source_family="community_signal",
        expansion_stage="conditional_context_expansion",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "launch_reception",
            "competitive_positioning",
            "feature_request_quotes",
        ),
        target_pools=("evidence_vault",),
        expected_consumers=("b2b_vendor_briefing",),
        consumers_without_material_benefit=_FIRMOGRAPHIC_NON_BENEFICIARIES,
        notes="Useful only in categories where launch/reception sentiment matters.",
    ),
    "sourceforge": _profile(
        "sourceforge",
        source_family="developer_context",
        expansion_stage="conditional_context_expansion",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "open_source_alternatives",
            "developer_pain_quotes",
            "feature_gaps",
        ),
        target_pools=("evidence_vault", "displacement"),
        expected_consumers=("b2b_battle_cards", "b2b_vendor_briefing"),
        consumers_without_material_benefit=_FIRMOGRAPHIC_NON_BENEFICIARIES,
        notes="Best limited to open-source and devtools categories.",
    ),
    "slashdot": _profile(
        "slashdot",
        source_family="developer_context",
        expansion_stage="deferred_conditional_inventory",
        work_type=("scrape_coverage",),
        reliable_fields=(
            "technical_pain_quotes",
            "replacement_discussion",
            "feature_gaps",
        ),
        target_pools=("evidence_vault", "displacement"),
        expected_consumers=("b2b_battle_cards", "b2b_vendor_briefing"),
        consumers_without_material_benefit=_FIRMOGRAPHIC_NON_BENEFICIARIES,
        notes="Useful for technical replacement chatter, but current non-seedable coverage should be treated as deferred inventory rather than active scrape backlog.",
    ),
}

_DEFAULT_IMPACT_PROFILE = SourceImpactProfile(
    source="unknown",
    source_family="unclassified",
    expansion_stage="conditional_context_expansion",
    work_type=("consumer_wiring",),
    reliable_fields=("pain_quotes",),
    target_pools=("evidence_vault",),
    expected_consumers=("b2b_vendor_briefing",),
    consumers_without_material_benefit=("crm_push_candidates",),
    notes="No explicit impact profile is registered for this source yet.",
)


class SupportsFetch(Protocol):
    async def fetch(self, query: str, *args: Any) -> Any: ...


def _profile_for_source(source: str) -> SourceImpactProfile:
    return _SOURCE_IMPACT_PROFILES.get(source, _DEFAULT_IMPACT_PROFILE)


def build_source_impact_ledger(source: str | None = None) -> dict[str, Any]:
    profiles = get_all_capabilities()
    selected_sources = (
        [source.strip().lower()]
        if source and source.strip()
        else [member.value for member in ReviewSource]
    )
    sources_out: list[dict[str, Any]] = []
    for source_name in selected_sources:
        profile = _profile_for_source(source_name)
        entry = profile.to_dict()
        capability = profiles.get(source_name)
        if capability is not None:
            entry["scrape_data_quality"] = capability.data_quality.value
            entry["capabilities"] = capability.to_dict()
        else:
            entry["scrape_data_quality"] = None
        infra_blocked = source_name in parse_source_allowlist(
            settings.b2b_churn.intelligence_infra_blocked_sources
        )
        deferred_inventory = source_name in parse_source_allowlist(
            getattr(settings.b2b_scrape, "deferred_inventory_sources", "")
        )
        entry["operational_status"] = (
            "infra_blocked"
            if infra_blocked
            else "deferred_inventory"
            if deferred_inventory
            else "active"
        )
        sources_out.append(entry)

    summary = {
        "total_sources": len(sources_out),
        "structured_sources": sum(
            1 for entry in sources_out if entry["source_family"] == "structured_review"
        ),
        "community_sources": sum(
            1 for entry in sources_out if entry["source_family"] == "community_signal"
        ),
        "developer_context_sources": sum(
            1 for entry in sources_out if entry["source_family"] == "developer_context"
        ),
        "news_context_sources": sum(
            1 for entry in sources_out if entry["source_family"] == "news_context"
        ),
        "verified_quality_sources": sum(
            1 for entry in sources_out if entry["scrape_data_quality"] == "verified"
        ),
        "structured_quality_sources": sum(
            1 for entry in sources_out if entry["scrape_data_quality"] == "structured"
        ),
        "community_quality_sources": sum(
            1 for entry in sources_out if entry["scrape_data_quality"] == "community"
        ),
        "news_quality_sources": sum(
            1 for entry in sources_out if entry["scrape_data_quality"] == "news"
        ),
        "parser_quality_targets": sorted(
            entry["source"]
            for entry in sources_out
            if "parser_quality" in entry["work_type"]
        ),
        "scrape_coverage_targets": sorted(
            entry["source"]
            for entry in sources_out
            if "scrape_coverage" in entry["work_type"]
        ),
        "query_strategy_targets": sorted(
            entry["source"]
            for entry in sources_out
            if "query_strategy" in entry["work_type"]
        ),
        "deferred_inventory_sources": sorted(
            entry["source"]
            for entry in sources_out
            if entry["operational_status"] == "deferred_inventory"
        ),
    }
    return {"sources": sources_out, "summary": summary}


def get_consumer_wiring_baseline() -> dict[str, Any]:
    consumers = [
        {
            "consumer": "b2b_reasoning_synthesis",
            "status": "canonical_all_pools",
            "primary_inputs": [
                "b2b_evidence_vault",
                "b2b_segment_intelligence",
                "b2b_temporal_intelligence",
                "b2b_displacement_dynamics",
                "b2b_category_dynamics",
                "b2b_account_intelligence",
            ],
            "legacy_fallback": False,
            "expected_gain_from_structured_sources": [
                "segment",
                "accounts",
            ],
            "expected_gain_from_community_sources": [
                "temporal",
                "displacement",
                "evidence_vault",
            ],
            "notes": "Only consumer that deterministically loads all canonical pools together.",
        },
        {
            "consumer": "b2b_accounts_in_motion",
            "status": "mixed_with_raw_review_fallback",
            "primary_inputs": [
                "persisted_accounts_in_motion_reports",
                "reasoning_contracts",
                "b2b_reviews_live_fallback",
            ],
            "legacy_fallback": True,
            "expected_gain_from_structured_sources": [
                "accounts",
                "segment",
            ],
            "expected_gain_from_community_sources": [
                "temporal",
            ],
            "notes": "Still carries a live raw-review path, so account-pool adoption is incomplete.",
        },
        {
            "consumer": "b2b_battle_cards",
            "status": "mixed_contract_plus_legacy_displacement",
            "primary_inputs": [
                "reasoning_contracts",
                "b2b_displacement_dynamics",
                "legacy_review_displacement_reader",
            ],
            "legacy_fallback": True,
            "expected_gain_from_structured_sources": [
                "segment",
                "accounts",
            ],
            "expected_gain_from_community_sources": [
                "displacement",
                "evidence_vault",
            ],
            "notes": "Competitive outputs still risk divergence because displacement has more than one materialization path.",
        },
        {
            "consumer": "b2b_churn_reports",
            "status": "mixed_contract_plus_legacy_displacement",
            "primary_inputs": [
                "b2b_churn_signals",
                "reasoning_contracts",
                "legacy_review_displacement_reader",
            ],
            "legacy_fallback": True,
            "expected_gain_from_structured_sources": [
                "segment",
                "accounts",
            ],
            "expected_gain_from_community_sources": [
                "temporal",
                "displacement",
                "evidence_vault",
            ],
            "notes": "Reads canonical summary tables but still inherits some legacy competitive paths.",
        },
        {
            "consumer": "b2b_vendor_briefing",
            "status": "mostly_canonical_with_live_quote_overlay",
            "primary_inputs": [
                "reasoning_contracts",
                "b2b_account_intelligence",
                "b2b_displacement_dynamics",
                "b2b_reviews_quote_overlay",
            ],
            "legacy_fallback": True,
            "expected_gain_from_structured_sources": [
                "segment",
                "accounts",
            ],
            "expected_gain_from_community_sources": [
                "evidence_vault",
                "temporal",
                "displacement",
            ],
            "notes": "Already synthesis-first for reasoning, but still enriches thin evidence from raw quotes.",
        },
        {
            "consumer": "mcp_signals_and_reviews",
            "status": "surface_mix_of_summary_tables_and_raw_reviews",
            "primary_inputs": [
                "b2b_churn_signals",
                "b2b_reviews",
                "b2b_vendor_snapshots",
                "reasoning_views",
            ],
            "legacy_fallback": True,
            "expected_gain_from_structured_sources": [
                "accounts",
                "segment",
            ],
            "expected_gain_from_community_sources": [
                "temporal",
                "displacement",
                "evidence_vault",
            ],
            "notes": "Operator surfaces expose both canonical aggregates and direct review search, so fragmentation remains visible here.",
        },
    ]
    return {
        "baseline_mode": "static_code_inventory",
        "measured": False,
        "summary": {
            "total_consumers": len(consumers),
            "canonical_consumers": sum(
                1 for consumer in consumers if consumer["legacy_fallback"] is False
            ),
            "mixed_consumers": sum(
                1 for consumer in consumers if consumer["legacy_fallback"] is True
            ),
        },
        "consumers": consumers,
    }


# PR-B5c: the pure helpers below are owned by the
# ``extracted_quality_gate.source_quality_pack`` pack. Atlas-side
# call sites continue to use the underscore-prefixed aliases for
# diff stability; the pack functions take an optional ``precision``
# kwarg that defaults to 3 (the legacy value) so behaviour is
# identical when called without it.
from extracted_quality_gate.source_quality_pack import (
    build_non_empty_text_check as _build_non_empty_text_check,
    compute_coverage_ratio as _compute_coverage_ratio,
    row_count as _row_count,
)


async def summarize_source_field_baseline(
    pool: SupportsFetch,
    *,
    window_days: int = 90,
    source: str | None = None,
) -> dict[str, Any]:
    title_present_sql = _build_non_empty_text_check("reviewer_title")
    company_present_sql = _build_non_empty_text_check("reviewer_company")
    company_size_present_sql = _build_non_empty_text_check(
        """
            COALESCE(
                company_size_raw,
                enrichment->'reviewer_context'->>'company_size_segment'
            )
        """
    )
    industry_present_sql = _build_non_empty_text_check(
        """
            COALESCE(
                reviewer_industry,
                enrichment->'reviewer_context'->>'industry'
            )
        """
    )
    timing_conditions = [
        _build_non_empty_text_check("enrichment->'timeline'->>'contract_end'"),
        _build_non_empty_text_check("enrichment->'timeline'->>'evaluation_deadline'"),
        _build_non_empty_text_check("enrichment->'timeline'->>'decision_timeline'"),
    ]
    timing_present_sql = f"({' OR '.join(timing_conditions)})"
    pain_present_sql = _build_non_empty_text_check(
        "enrichment->>'pain_category'"
    )
    content_classification_present_sql = _build_non_empty_text_check(
        "enrichment->>'content_classification'"
    )
    conditions = [
        "duplicate_of_review_id IS NULL",
        "imported_at >= NOW() - make_interval(days => $1)",
    ]
    params: list[Any] = [window_days]
    if source and source.strip():
        conditions.append("source = $2")
        params.append(source.strip().lower())

    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT
            source,
            COUNT(*) AS total_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched_reviews,
            COUNT(*) FILTER (WHERE {title_present_sql}) AS title_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched' AND {title_present_sql}
            ) AS enriched_title_rows,
            COUNT(*) FILTER (WHERE {company_present_sql}) AS company_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched' AND {company_present_sql}
            ) AS enriched_company_rows,
            COUNT(*) FILTER (WHERE {company_size_present_sql}) AS company_size_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched' AND {company_size_present_sql}
            ) AS enriched_company_size_rows,
            COUNT(*) FILTER (WHERE {industry_present_sql}) AS industry_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched' AND {industry_present_sql}
            ) AS enriched_industry_rows,
            COUNT(*) FILTER (
                WHERE enrichment->'reviewer_context'->>'decision_maker' = 'true'
            ) AS decision_maker_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND enrichment->'reviewer_context'->>'decision_maker' = 'true'
            ) AS enriched_decision_maker_rows,
            COUNT(*) FILTER (
                WHERE jsonb_array_length(
                    COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb)
                ) > 0
            ) AS competitor_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND jsonb_array_length(
                      COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb)
                  ) > 0
            ) AS enriched_competitor_rows,
            COUNT(*) FILTER (
                WHERE {timing_present_sql}
            ) AS timing_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched' AND {timing_present_sql}
            ) AS enriched_timing_rows,
            COUNT(*) FILTER (
                WHERE jsonb_array_length(
                    COALESCE(enrichment->'quotable_phrases', '[]'::jsonb)
                ) > 0
            ) AS quote_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND jsonb_array_length(
                      COALESCE(enrichment->'quotable_phrases', '[]'::jsonb)
                  ) > 0
            ) AS enriched_quote_rows,
            COUNT(*) FILTER (WHERE {pain_present_sql}) AS pain_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched' AND {pain_present_sql}
            ) AS enriched_pain_rows,
            COUNT(*) FILTER (
                WHERE {content_classification_present_sql}
            ) AS content_classification_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND {content_classification_present_sql}
            ) AS enriched_content_classification_rows,
            COUNT(*) FILTER (
                WHERE enrichment->>'support_escalation' = 'true'
            ) AS support_escalation_rows,
            COUNT(*) FILTER (
                WHERE enrichment_status = 'enriched'
                  AND enrichment->>'support_escalation' = 'true'
            ) AS enriched_support_escalation_rows
        FROM b2b_reviews
        WHERE {where}
        GROUP BY source
        ORDER BY total_reviews DESC, source ASC
        """,
        *params,
    )

    baseline_rows: list[dict[str, Any]] = []
    for row in rows:
        total = int(row["total_reviews"] or 0)
        enriched = int(row["enriched_reviews"] or 0)
        enriched_counts = {
            "title_rows": _row_count(
                row,
                "enriched_title_rows",
                fallback_key="title_rows",
            ),
            "company_rows": _row_count(
                row,
                "enriched_company_rows",
                fallback_key="company_rows",
            ),
            "company_size_rows": _row_count(
                row,
                "enriched_company_size_rows",
                fallback_key="company_size_rows",
            ),
            "industry_rows": _row_count(
                row,
                "enriched_industry_rows",
                fallback_key="industry_rows",
            ),
            "decision_maker_rows": _row_count(
                row,
                "enriched_decision_maker_rows",
                fallback_key="decision_maker_rows",
            ),
            "competitor_rows": _row_count(
                row,
                "enriched_competitor_rows",
                fallback_key="competitor_rows",
            ),
            "timing_rows": _row_count(
                row,
                "enriched_timing_rows",
                fallback_key="timing_rows",
            ),
            "quote_rows": _row_count(
                row,
                "enriched_quote_rows",
                fallback_key="quote_rows",
            ),
            "pain_rows": _row_count(
                row,
                "enriched_pain_rows",
                fallback_key="pain_rows",
            ),
            "content_classification_rows": _row_count(
                row,
                "enriched_content_classification_rows",
                fallback_key="content_classification_rows",
            ),
            "support_escalation_rows": _row_count(
                row,
                "enriched_support_escalation_rows",
                fallback_key="support_escalation_rows",
            ),
        }
        total_counts = {
            "title_rows": _row_count(row, "title_rows"),
            "company_rows": _row_count(row, "company_rows"),
            "company_size_rows": _row_count(row, "company_size_rows"),
            "industry_rows": _row_count(row, "industry_rows"),
            "decision_maker_rows": _row_count(row, "decision_maker_rows"),
            "competitor_rows": _row_count(row, "competitor_rows"),
            "timing_rows": _row_count(row, "timing_rows"),
            "quote_rows": _row_count(row, "quote_rows"),
            "pain_rows": _row_count(row, "pain_rows"),
            "content_classification_rows": _row_count(
                row,
                "content_classification_rows",
            ),
            "support_escalation_rows": _row_count(
                row,
                "support_escalation_rows",
            ),
        }
        baseline_rows.append(
            {
                "source": row["source"],
                "display_name": source_display_name(row["source"]),
                "total_reviews": total,
                "enriched_reviews": enriched,
                "enrichment_rate": _compute_coverage_ratio(enriched, total),
                "coverage": {
                    "title": _compute_coverage_ratio(
                        enriched_counts["title_rows"],
                        enriched,
                    ),
                    "company": _compute_coverage_ratio(
                        enriched_counts["company_rows"],
                        enriched,
                    ),
                    "company_size": _compute_coverage_ratio(
                        enriched_counts["company_size_rows"],
                        enriched,
                    ),
                    "industry": _compute_coverage_ratio(
                        enriched_counts["industry_rows"],
                        enriched,
                    ),
                    "decision_maker": _compute_coverage_ratio(
                        enriched_counts["decision_maker_rows"],
                        enriched,
                    ),
                    "competitors": _compute_coverage_ratio(
                        enriched_counts["competitor_rows"],
                        enriched,
                    ),
                    "timing": _compute_coverage_ratio(
                        enriched_counts["timing_rows"],
                        enriched,
                    ),
                    "quotes": _compute_coverage_ratio(
                        enriched_counts["quote_rows"],
                        enriched,
                    ),
                    "pain_category": _compute_coverage_ratio(
                        enriched_counts["pain_rows"],
                        enriched,
                    ),
                    "content_classification": _compute_coverage_ratio(
                        enriched_counts["content_classification_rows"],
                        enriched,
                    ),
                    "support_escalation": _compute_coverage_ratio(
                        enriched_counts["support_escalation_rows"],
                        enriched,
                    ),
                },
                "coverage_of_total_reviews": {
                    "title": _compute_coverage_ratio(
                        total_counts["title_rows"],
                        total,
                    ),
                    "company": _compute_coverage_ratio(
                        total_counts["company_rows"],
                        total,
                    ),
                    "company_size": _compute_coverage_ratio(
                        total_counts["company_size_rows"],
                        total,
                    ),
                    "industry": _compute_coverage_ratio(
                        total_counts["industry_rows"],
                        total,
                    ),
                    "decision_maker": _compute_coverage_ratio(
                        total_counts["decision_maker_rows"],
                        total,
                    ),
                    "competitors": _compute_coverage_ratio(
                        total_counts["competitor_rows"],
                        total,
                    ),
                    "timing": _compute_coverage_ratio(
                        total_counts["timing_rows"],
                        total,
                    ),
                    "quotes": _compute_coverage_ratio(
                        total_counts["quote_rows"],
                        total,
                    ),
                    "pain_category": _compute_coverage_ratio(
                        total_counts["pain_rows"],
                        total,
                    ),
                    "content_classification": _compute_coverage_ratio(
                        total_counts["content_classification_rows"],
                        total,
                    ),
                    "support_escalation": _compute_coverage_ratio(
                        total_counts["support_escalation_rows"],
                        total,
                    ),
                },
                "raw_counts": enriched_counts,
                "raw_counts_of_total_reviews": total_counts,
            }
        )

    return {
        "window_days": window_days,
        "source_filter": source.strip().lower() if source else None,
        "rows": baseline_rows,
        "summary": {
            "total_sources": len(baseline_rows),
            "total_reviews": sum(row["total_reviews"] for row in baseline_rows),
            "total_enriched_reviews": sum(
                row["enriched_reviews"] for row in baseline_rows
            ),
        },
    }
