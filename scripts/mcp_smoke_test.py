#!/usr/bin/env python3
"""Smoke test for all 61 B2B Churn MCP server tools.

Calls every tool with minimal valid args, reports pass/fail.
Does NOT modify data (skips destructive tools or uses safe args).

Usage:
    cd /home/juan-canfield/Desktop/Atlas
    source .venv/bin/activate
    python scripts/mcp_smoke_test.py
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Suppress noisy loggers
for name in ("httpx", "atlas", "asyncpg"):
    logging.getLogger(name).setLevel(logging.ERROR)


async def run_tests():
    # Init DB pool the same way the MCP server does
    from atlas_brain.storage.database import init_database, get_db_pool, close_database
    await init_database()
    pool = get_db_pool()

    # Import all domain modules (tools are registered at import time)
    import atlas_brain.mcp.b2b.server  # noqa - triggers all domain imports
    from atlas_brain.mcp.b2b import signals, reviews, reports, products
    from atlas_brain.mcp.b2b import displacement, vendor_history, vendor_registry
    from atlas_brain.mcp.b2b import scrape_targets, pipeline, corrections
    from atlas_brain.mcp.b2b import calibration, webhooks, crm_events, content

    # Grab real IDs from DB for tools that need them
    review_row = await pool.fetchrow(
        "SELECT id FROM b2b_reviews WHERE source = 'reddit' LIMIT 1"
    )
    review_id = str(review_row["id"]) if review_row else "00000000-0000-0000-0000-000000000000"

    report_row = await pool.fetchrow(
        "SELECT id FROM b2b_intelligence LIMIT 1"
    )
    report_id = str(report_row["id"]) if report_row else "00000000-0000-0000-0000-000000000000"

    target_row = await pool.fetchrow(
        "SELECT id FROM b2b_scrape_targets WHERE source = 'reddit' LIMIT 1"
    )
    target_id = str(target_row["id"]) if target_row else "00000000-0000-0000-0000-000000000000"

    async def _safe_fetch(query):
        try:
            return await pool.fetchrow(query)
        except Exception:
            return None

    correction_row = await _safe_fetch("SELECT id FROM b2b_data_corrections LIMIT 1")
    correction_id = str(correction_row["id"]) if correction_row else None

    blog_row = await _safe_fetch("SELECT id FROM b2b_blog_posts LIMIT 1")
    blog_id = str(blog_row["id"]) if blog_row else None

    webhook_row = await _safe_fetch("SELECT id FROM b2b_webhook_subscriptions LIMIT 1")
    webhook_id = str(webhook_row["id"]) if webhook_row else None

    # Define all 61 tools with their test args
    # Tools marked SKIP_MUTATE are write operations we don't want to execute
    SKIP_MUTATE = "SKIP_MUTATE"

    tests = [
        # --- Churn Signals (6) ---
        ("list_churn_signals", signals.list_churn_signals, dict(limit=3)),
        ("get_churn_signal", signals.get_churn_signal, dict(vendor_name="Salesforce")),
        ("list_high_intent_companies", signals.list_high_intent_companies, dict(limit=3, window_days=365)),
        ("get_vendor_profile", signals.get_vendor_profile, dict(vendor_name="Salesforce")),
        ("get_vendor_history", vendor_history.get_vendor_history, dict(vendor_name="Salesforce")),
        ("compare_vendor_periods", vendor_history.compare_vendor_periods, dict(vendor_name="Salesforce")),

        # --- Reviews (2) ---
        ("search_reviews", reviews.search_reviews, dict(vendor_name="Salesforce", limit=3, window_days=365)),
        ("get_review", reviews.get_review, dict(review_id=review_id)),

        # --- Reports (3) ---
        ("list_reports", reports.list_reports, dict(limit=3)),
        ("get_report", reports.get_report, dict(report_id=report_id)),
        ("export_report_pdf", reports.export_report_pdf, dict(report_id=report_id)),

        # --- Products (3) ---
        ("get_product_profile", products.get_product_profile, dict(vendor_name="Salesforce")),
        ("get_product_profile_history", products.get_product_profile_history, dict(vendor_name="Salesforce")),
        ("match_products_tool", products.match_products_tool, dict(churning_from="Salesforce", limit=3)),

        # --- Displacement (6) ---
        ("list_displacement_edges", displacement.list_displacement_edges, dict(limit=3, window_days=365)),
        ("get_displacement_history", displacement.get_displacement_history, dict(from_vendor="Salesforce", to_vendor="HubSpot")),
        ("list_vendor_pain_points", displacement.list_vendor_pain_points, dict(vendor_name="Salesforce")),
        ("list_vendor_use_cases", displacement.list_vendor_use_cases, dict(vendor_name="Salesforce")),
        ("list_vendor_integrations", displacement.list_vendor_integrations, dict(vendor_name="Salesforce")),
        ("list_vendor_buyer_profiles", displacement.list_vendor_buyer_profiles, dict(vendor_name="Salesforce")),

        # --- Registry (5) ---
        ("list_vendors_registry", vendor_registry.list_vendors_registry, dict(limit=3)),
        ("fuzzy_vendor_search", vendor_registry.fuzzy_vendor_search, dict(query="sales")),
        ("fuzzy_company_search", vendor_registry.fuzzy_company_search, dict(query="acme")),
        ("add_vendor_to_registry", None, SKIP_MUTATE),
        ("add_vendor_alias", None, SKIP_MUTATE),

        # --- Scrape Admin (4) ---
        ("list_scrape_targets", scrape_targets.list_scrape_targets, dict(limit=3)),
        ("add_scrape_target", None, SKIP_MUTATE),
        ("manage_scrape_target", None, SKIP_MUTATE),
        ("delete_scrape_target", None, SKIP_MUTATE),

        # --- Pipeline Health (7) ---
        ("get_pipeline_status", pipeline.get_pipeline_status, {}),
        ("get_parser_version_status", pipeline.get_parser_version_status, {}),
        ("get_source_health", pipeline.get_source_health, dict(source="reddit")),
        ("get_source_telemetry", pipeline.get_source_telemetry, dict(source="reddit")),
        ("get_source_capabilities", pipeline.get_source_capabilities, dict(source="reddit")),
        ("get_operational_overview", pipeline.get_operational_overview, {}),
        ("get_parser_health", pipeline.get_parser_health, {}),

        # --- Events (3) ---
        ("list_change_events", vendor_history.list_change_events, dict(limit=3)),
        ("list_concurrent_events", vendor_history.list_concurrent_events, dict(days=90, limit=3)),
        ("get_vendor_correlation", vendor_history.get_vendor_correlation, dict(vendor_a="Salesforce", vendor_b="HubSpot")),

        # --- Campaigns & Calibration (5) ---
        ("record_campaign_outcome", None, SKIP_MUTATE),
        ("get_signal_effectiveness", calibration.get_signal_effectiveness, {}),
        ("get_outcome_distribution", calibration.get_outcome_distribution, {}),
        ("trigger_score_calibration", None, SKIP_MUTATE),
        ("get_calibration_weights", calibration.get_calibration_weights, {}),

        # --- Data Corrections (6) ---
        ("create_data_correction", None, SKIP_MUTATE),
        ("list_data_corrections", corrections.list_data_corrections, dict(limit=3)),
        ("revert_data_correction", None, SKIP_MUTATE),
        ("get_data_correction", corrections.get_data_correction, dict(correction_id=correction_id or "00000000-0000-0000-0000-000000000000")),
        ("get_correction_stats", corrections.get_correction_stats, {}),
        ("get_source_correction_impact", corrections.get_source_correction_impact, {}),

        # --- Webhooks (4) ---
        ("list_webhook_subscriptions", webhooks.list_webhook_subscriptions, {}),
        ("send_test_webhook_tool", None, SKIP_MUTATE),
        ("update_webhook", None, SKIP_MUTATE),
        ("get_webhook_delivery_summary", webhooks.get_webhook_delivery_summary, {}),

        # --- Content (3) ---
        ("list_blog_posts", content.list_blog_posts, dict(limit=3)),
        ("get_blog_post", content.get_blog_post, dict(post_id=blog_id or "00000000-0000-0000-0000-000000000000")),
        ("list_affiliate_partners", content.list_affiliate_partners, dict(limit=3)),

        # --- CRM (4) ---
        ("list_crm_pushes", crm_events.list_crm_pushes, dict(limit=3)),
        ("list_crm_events", crm_events.list_crm_events, dict(limit=3)),
        ("ingest_crm_event", None, SKIP_MUTATE),
        ("get_crm_enrichment_stats", crm_events.get_crm_enrichment_stats, {}),
    ]

    passed = 0
    failed = 0
    skipped = 0
    errors = []

    print(f"{'='*70}")
    print(f"MCP Smoke Test -- {len(tests)} tools")
    print(f"{'='*70}")

    for name, func, args in tests:
        if args == SKIP_MUTATE:
            print(f"  SKIP  {name:45s}  (write operation)")
            skipped += 1
            continue

        t0 = time.monotonic()
        try:
            result = await func(**args)
            elapsed = (time.monotonic() - t0) * 1000

            # Validate result is parseable JSON string
            if isinstance(result, str):
                parsed = json.loads(result)
                # Check for error key
                if isinstance(parsed, dict) and parsed.get("error"):
                    # Some tools return error for missing data -- that's OK
                    # as long as it didn't crash
                    if "not found" in str(parsed["error"]).lower():
                        print(f"  PASS  {name:45s}  {elapsed:6.0f}ms  (not found - OK)")
                        passed += 1
                    else:
                        print(f"  WARN  {name:45s}  {elapsed:6.0f}ms  error: {str(parsed['error'])[:60]}")
                        passed += 1  # still didn't crash
                else:
                    print(f"  PASS  {name:45s}  {elapsed:6.0f}ms")
                    passed += 1
            else:
                print(f"  PASS  {name:45s}  {elapsed:6.0f}ms  (non-string return)")
                passed += 1

        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            print(f"  FAIL  {name:45s}  {elapsed:6.0f}ms  {type(exc).__name__}: {exc}")
            errors.append((name, exc))
            failed += 1

    print(f"{'='*70}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (write ops)")
    print(f"{'='*70}")

    if errors:
        print(f"\nFailed tools:")
        for name, exc in errors:
            print(f"  {name}: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    await close_database()
    return failed


if __name__ == "__main__":
    failures = asyncio.run(run_tests())
    sys.exit(1 if failures else 0)
