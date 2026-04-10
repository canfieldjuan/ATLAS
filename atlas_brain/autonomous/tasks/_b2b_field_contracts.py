"""B2B enrichment field ownership contract.

Declares every field in b2b_reviews.enrichment JSONB, its authoritative
owner path, and approved consumers.  The governance test suite
(tests/test_b2b_field_governance.py) enforces this contract.

Owner paths
-----------
pool             Field is aggregated by _b2b_shared.py into a pool table.
live_overlay     Field is read directly from enrichment JSONB by an
                 approved module (not migrated to pools).
witness          Field is produced/consumed by the witness system.
enrichment_internal  Field is consumed only during enrichment itself
                 (validation, derivation, content_type fallback).
"""

from __future__ import annotations

from typing import TypedDict


class FieldContract(TypedDict):
    owner_path: str                     # pool | live_overlay | witness | enrichment_internal
    owner_pool: str | None              # pool table name when owner_path == "pool"
    stranded: bool                      # True = zero downstream consumers
    approved_consumers: tuple[str, ...] # module-qualified references with approved access
    migration_target: str | None        # adapter function name for deprecated reads


# Modules that may read enrichment JSONB directly without markers.
# These are producers, repairers, validators, or the approved wiring layer.
EXEMPT_MODULES: frozenset[str] = frozenset({
    "atlas_brain/autonomous/tasks/b2b_enrichment.py",
    "atlas_brain/autonomous/tasks/b2b_enrichment_repair.py",
    "atlas_brain/services/extraction_health_audit.py",
    "atlas_brain/autonomous/tasks/_b2b_shared.py",
    "atlas_brain/services/b2b/enrichment_repair_policy.py",
    "scripts/b2b_field_access_inventory.py",  # self-referential regex patterns
})


FIELD_CONTRACTS: dict[str, FieldContract] = {
    # ---- Pool-owned: evidence_vault ----
    "urgency_score": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_churn_scores",
            "_b2b_shared._fetch_high_intent_companies",
            "_b2b_shared._fetch_vendor_witness_reviews",
            "_b2b_shared.read_campaign_opportunities",
            "_b2b_shared.read_vendor_quote_evidence",
            "_b2b_shared.read_category_quote_evidence",
            "b2b_campaign_generation._compute_vendor_trend",
            "b2b_blog_post_generation",
            "b2b_product_profiles._fetch_aggregate_metrics",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_churn_alert",
            "b2b_score_calibration",
            "prospect_enrichment._discover_companies",
            "admin_costs",
            "blog_admin",
            "b2b_scrape",
            "b2b_dashboard",
            "b2b_affiliates",
            "backfill_derived_fields",
            "backfill_company_names",
        ),
        "migration_target": "read_high_intent_companies",
    },
    "churn_signals": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_churn_scores",
            "b2b_blog_post_generation._build_blog_context_metadata",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "prospect_enrichment._discover_companies",
            "admin_costs",
            "b2b_scrape",
            "b2b_dashboard._list_accounts_in_motion",
            "b2b_tenant_dashboard",
            "backfill_witness_primitives",
            "cleanup_accounts_in_motion_pollution",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "pain_category": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_pain_lookup",
            "_b2b_shared._fetch_vendor_churn_scores",
            "_b2b_shared.read_vendor_quote_evidence",
            "_b2b_shared.read_category_quote_evidence",
            "b2b_blog_post_generation._fetch_pain_category_urgency",
            "b2b_product_profiles._fetch_satisfaction_by_area",
            "b2b_product_profiles._fetch_pain_distribution",
            "admin_costs",
            "blog_admin",
            "signals",
            "b2b_dashboard",
            "b2b_tenant_dashboard",
            "backfill_derived_fields",
            "cluster_other_pain",
            "re_enrich_other_pain",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "pain_categories": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_pain_lookup",
            "_b2b_shared.read_vendor_quote_evidence",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_blog_post_generation",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_dashboard._list_accounts_in_motion",
            "cluster_other_pain",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "competitors_mentioned": {
        "owner_path": "pool",
        "owner_pool": "b2b_displacement_dynamics",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_displacement_flows",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_churn_intelligence._fetch_company_signal_review_context",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_product_profiles._fetch_competitive_flows",
            "b2b_churn_alert",
            "b2b_score_calibration",
            "b2b_scrape",
            "b2b_dashboard._list_accounts_in_motion",
            "b2b_affiliates",
            "b2b_tenant_dashboard",
            "backfill_derived_fields",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "quotable_phrases": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_review_text_aggregates",
            "_b2b_shared.read_vendor_quote_evidence",
            "_b2b_shared.read_category_quote_evidence",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_churn_intelligence._fetch_company_signal_review_context",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_dashboard._list_accounts_in_motion",
            "cluster_other_pain",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "specific_complaints": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_review_text_aggregates",
            "cluster_other_pain",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "feature_gaps": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_review_text_aggregates",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_churn_intelligence._fetch_company_signal_review_context",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "positive_aspects": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_review_text_aggregates",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "pricing_phrases": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_review_text_aggregates",
            "backfill_derived_fields",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "recommendation_language": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_churn_scores",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "would_recommend": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_churn_scores",
            "b2b_product_profiles._fetch_aggregate_metrics",
            "backfill_derived_fields",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "urgency_indicators": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_churn_scores",
            "backfill_derived_fields",
            "cleanup_accounts_in_motion_pollution",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "insider_signals": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_insider_aggregates",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "pain_cluster": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_pain_lookup",
            "cluster_other_pain",
        ),
        "migration_target": "read_vendor_evidence",
    },
    "churn_intent": {
        "owner_path": "pool",
        "owner_pool": "b2b_evidence_vault",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_vendor_churn_scores",
        ),
        "migration_target": "read_vendor_evidence",
    },

    # ---- Pool-owned: segment_intelligence ----
    "reviewer_context": {
        "owner_path": "pool",
        "owner_pool": "b2b_segment_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_department_distribution",
            "_b2b_shared._fetch_company_size_distribution",
            "_b2b_shared._fetch_buyer_authority_summary",
            # live_overlay: account resolution reads company_name for identity matching
            "b2b_account_resolution._fetch_unresolved_reviews",
            # live_overlay: intelligence reads for witness/report context
            "b2b_churn_intelligence._fetch_company_signal_review_context",
            "_b2b_shared.read_vendor_quote_evidence",
            "_b2b_shared.read_category_quote_evidence",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_product_profiles._fetch_company_size_distribution",
            "b2b_dashboard._list_accounts_in_motion",
            "b2b_affiliates",
            "blog_admin",
            "backfill_witness_primitives",
        ),
        "migration_target": "read_review_details",
    },
    "buyer_authority": {
        "owner_path": "pool",
        "owner_pool": "b2b_segment_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_buyer_authority_summary",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_dashboard._list_accounts_in_motion",
            "b2b_affiliates",
            "backfill_buyer_authority_roles",
            "backfill_witness_primitives",
        ),
        "migration_target": "read_review_details",
    },
    "budget_signals": {
        "owner_path": "pool",
        "owner_pool": "b2b_segment_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_budget_signals",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_score_calibration",
            "b2b_affiliates",
            "backfill_witness_primitives",
        ),
        "migration_target": "read_review_details",
    },
    "use_case": {
        "owner_path": "pool",
        "owner_pool": "b2b_segment_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_use_case_distribution",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_product_profiles._fetch_use_case_distribution",
            "b2b_product_profiles._fetch_integration_stacks",
        ),
        "migration_target": "read_review_details",
    },
    "contract_context": {
        "owner_path": "pool",
        "owner_pool": "b2b_segment_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_budget_signals",
            "b2b_churn_intelligence._fetch_company_signal_review_context",
            "backfill_derived_fields",
        ),
        "migration_target": "read_review_details",
    },

    "sentiment_trajectory": {
        "owner_path": "pool",
        "owner_pool": "b2b_temporal_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_sentiment_turning_points",
            "_b2b_witnesses._witness_salience",
        ),
        "migration_target": "read_vendor_evidence",
    },

    # ---- Pool-owned: temporal_intelligence ----
    "timeline": {
        "owner_path": "pool",
        "owner_pool": "b2b_temporal_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_timeline_entries",
            "_b2b_shared.read_campaign_opportunities",
            "b2b_churn_intelligence._fetch_company_signal_review_context",
            "b2b_churn_intelligence.generate_vendor_report",
            "b2b_churn_intelligence.generate_challenger_report",
            "b2b_affiliates",
            "backfill_witness_primitives",
        ),
        "migration_target": "read_review_details",
    },
    "event_mentions": {
        "owner_path": "pool",
        "owner_pool": "b2b_temporal_intelligence",
        "stranded": False,
        "approved_consumers": (
            "_b2b_shared._fetch_sentiment_turning_points",
        ),
        "migration_target": "read_vendor_evidence",
    },

    # ---- Witness-owned ----
    "evidence_spans": {
        "owner_path": "witness",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "_b2b_witnesses.derive_evidence_spans",
            "_b2b_shared._fetch_vendor_witness_reviews",
            "admin_costs",
            "backfill_witness_primitives",
            "backfill_derived_fields",
        ),
        "migration_target": None,
    },
    "salience_flags": {
        "owner_path": "witness",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "_b2b_witnesses._witness_salience",
            "backfill_witness_primitives",
        ),
        "migration_target": None,
    },
    "replacement_mode": {
        "owner_path": "witness",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "_b2b_witnesses._candidate_types",
            "backfill_witness_primitives",
        ),
        "migration_target": None,
    },
    "operating_model_shift": {
        "owner_path": "witness",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "_b2b_witnesses._candidate_types",
            "backfill_witness_primitives",
        ),
        "migration_target": None,
    },
    "productivity_delta_claim": {
        "owner_path": "witness",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "_b2b_witnesses._candidate_types",
            "backfill_witness_primitives",
        ),
        "migration_target": None,
    },
    "org_pressure_type": {
        "owner_path": "witness",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "_b2b_witnesses._candidate_types",
            "backfill_witness_primitives",
        ),
        "migration_target": None,
    },

    # ---- Enrichment-internal (metadata / consumed at write time) ----
    "enrichment_schema_version": {
        "owner_path": "enrichment_internal",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "backfill_derived_fields",
        ),
        "migration_target": None,
    },
    "evidence_map_hash": {
        "owner_path": "enrichment_internal",
        "owner_pool": None,
        "stranded": False,
        "approved_consumers": (
            "backfill_witness_primitives",
        ),
        "migration_target": None,
    },

    # ---- Stranded (zero downstream consumers) ----
    "content_classification": {
        "owner_path": "enrichment_internal",
        "owner_pool": None,
        "stranded": True,
        "approved_consumers": (),
        "migration_target": None,
    },
    "support_escalation": {
        "owner_path": "enrichment_internal",
        "owner_pool": None,
        "stranded": True,
        "approved_consumers": (),
        "migration_target": None,
    },
}


STRANDED_FIELDS: frozenset[str] = frozenset(
    name for name, c in FIELD_CONTRACTS.items() if c["stranded"]
)


VALID_OWNER_PATHS: frozenset[str] = frozenset({
    "pool", "live_overlay", "witness", "enrichment_internal",
})


def validate_contracts() -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []
    for name, c in FIELD_CONTRACTS.items():
        if c["owner_path"] not in VALID_OWNER_PATHS:
            errors.append(f"{name}: invalid owner_path '{c['owner_path']}'")
        if c["owner_path"] == "pool" and not c["owner_pool"]:
            errors.append(f"{name}: owner_path is 'pool' but owner_pool is None")
        if c["owner_path"] != "pool" and c["owner_pool"]:
            errors.append(f"{name}: owner_path is '{c['owner_path']}' but owner_pool is set")
        if c["stranded"] and c["approved_consumers"]:
            errors.append(f"{name}: stranded=True but has approved_consumers")
    return errors
