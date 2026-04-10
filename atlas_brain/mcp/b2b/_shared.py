"""Shared utilities for B2B Churn MCP domain modules."""

import json
import logging
import uuid as _uuid

logger = logging.getLogger("atlas.mcp.b2b_churn")

VALID_REPORT_TYPES = (
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
    "exploratory_overview",
    "vendor_comparison",
    "account_comparison",
    "account_deep_dive",
    "vendor_retention",
    "challenger_intel",
    "battle_card",
    "accounts_in_motion",
    "challenger_brief",
)

from atlas_brain.services.scraping.sources import ALL_SOURCES, ReviewSource  # noqa: E402

VALID_SOURCES = ALL_SOURCES

from atlas_brain.services.b2b.corrections import (  # noqa: E402
    suppress_predicate as _suppress_predicate,
    apply_field_overrides as _apply_field_overrides,
)


def get_pool():
    """Get the initialized DB pool."""
    from atlas_brain.storage.database import get_db_pool
    return get_db_pool()


def _is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _safe_json(val):
    """Return val if it's already a list/dict, else try json.loads, else return val as-is."""
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val


def _canonical_review_predicate(alias: str = "") -> str:
    """Return the canonical-review filter for analytics surfaces.

    Raw/provenance endpoints should not use this predicate. It is intended for
    analytics/count surfaces that should collapse cross-source duplicates onto
    the canonical surviving review row.
    """
    prefix = f"{alias}." if alias else ""
    return f"{prefix}duplicate_of_review_id IS NULL"


# ---------------------------------------------------------------------------
# Tool group gating: maps group names to the tool function names they contain.
# Used by _apply_tool_gating() in server.py to remove tools from disabled groups.
# ---------------------------------------------------------------------------
TOOL_GROUPS: dict[str, list[str]] = {
    "read_signals": [
        "list_churn_signals",
        "get_churn_signal",
        "list_high_intent_companies",
        "get_vendor_profile",
        "search_reviews",
        "get_review",
        "get_product_profile",
        "get_product_profile_history",
        "match_products_tool",
        "list_displacement_edges",
        "get_displacement_history",
        "list_vendor_pain_points",
        "list_vendor_use_cases",
        "list_vendor_integrations",
        "list_vendor_buyer_profiles",
        "get_vendor_history",
        "list_change_events",
        "compare_vendor_periods",
        "list_concurrent_events",
        "get_vendor_correlation",
        "list_cross_vendor_conclusions",
        "get_cross_vendor_conclusion",
    ],
    "read_reports": [
        "list_reports",
        "get_report",
        "export_report_pdf",
    ],
    "reasoning": [
        "reason_vendor",
        "compare_vendors",
    ],
    "admin": [
        "list_vendors_registry",
        "fuzzy_vendor_search",
        "fuzzy_company_search",
        "add_vendor_to_registry",
        "add_vendor_alias",
        "list_scrape_targets",
        "add_scrape_target",
        "manage_scrape_target",
        "delete_scrape_target",
        "create_data_correction",
        "list_data_corrections",
        "revert_data_correction",
        "get_data_correction",
        "get_correction_stats",
        "get_source_correction_impact",
        "get_pipeline_status",
        "get_parser_version_status",
        "get_source_health",
        "get_source_telemetry",
        "get_source_capabilities",
        "get_source_impact_ledger",
        "get_operational_overview",
        "get_parser_health",
    ],
    "calibration": [
        "record_campaign_outcome",
        "get_signal_effectiveness",
        "get_outcome_distribution",
        "trigger_score_calibration",
        "get_calibration_weights",
    ],
    "webhooks": [
        "list_webhook_subscriptions",
        "send_test_webhook_tool",
        "update_webhook",
        "get_webhook_delivery_summary",
    ],
    "crm_events": [
        "list_crm_pushes",
        "list_crm_events",
        "ingest_crm_event",
        "get_crm_enrichment_stats",
    ],
    "content": [
        "list_blog_posts",
        "get_blog_post",
        "list_affiliate_partners",
    ],
    "read_pools": [
        "get_evidence_vault",
        "list_evidence_vaults",
        "get_segment_intelligence",
        "list_segment_intelligence",
        "get_temporal_intelligence",
        "list_temporal_intelligence",
        "get_displacement_dynamics",
        "list_displacement_dynamics",
        "get_category_dynamics",
        "list_category_dynamics",
        "get_account_intelligence",
        "list_account_intelligence",
    ],
    "write_intelligence": [
        "persist_conclusion",
        "persist_report",
        "build_challenger_brief",
        "build_accounts_in_motion",
    ],
    "campaigns": [
        "draft_campaign",
    ],
}
