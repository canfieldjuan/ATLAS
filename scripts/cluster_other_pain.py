#!/usr/bin/env python3
"""Assign thematic cluster labels to reviews that remain tagged 'other' after re-enrichment.

Reads reviews where pain_category = 'other' AND pain_cluster IS NULL, runs a
cheap Haiku pass to assign one of five thematic clusters, and stores the result
in enrichment['pain_cluster']. This feeds _fetch_pain_distribution so reports
show 'Product Stagnation (Cluster)' instead of a bare 'Other' bar.

Clusters:
  product_stagnation    -- Product not evolving; roadmap unresponsive; feature requests ignored
  ecosystem_fatigue     -- Too many required add-ons, plugins, or integrations to function
  policy_corporate      -- Legal, compliance, corporate policy, vendor policy, procurement friction
  competitive_inferiority -- Losing to competitors on core value, not a specific pain category
  general_dissatisfaction -- Vague dissatisfaction with no identifiable theme

Usage:
    # Validate on 3 test vendors first
    ATLAS_B2B_CHURN_ENABLED=true python scripts/cluster_other_pain.py \\
        --vendors "Shopify,Salesforce,Jira"

    # Dry run to count scope
    ATLAS_B2B_CHURN_ENABLED=true python scripts/cluster_other_pain.py \\
        --vendors "Shopify,Salesforce,Jira" --dry-run

    # Full run (all vendors) after validation
    ATLAS_B2B_CHURN_ENABLED=true python scripts/cluster_other_pain.py

    # Reprocess rows that already have a cluster (re-cluster)
    ATLAS_B2B_CHURN_ENABLED=true python scripts/cluster_other_pain.py \\
        --vendors "Shopify" --recluster
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cluster_other_pain")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Ordered list (not set) so the regex fallback scan is deterministic.
# More specific labels come first so they win over general_dissatisfaction.
VALID_CLUSTERS_ORDERED = [
    "product_stagnation",
    "ecosystem_fatigue",
    "policy_corporate",
    "competitive_inferiority",
    "general_dissatisfaction",
]
VALID_CLUSTERS = set(VALID_CLUSTERS_ORDERED)

CLUSTER_SYSTEM_PROMPT = (
    "You assign thematic cluster labels to B2B software reviews that contain vague complaints.\n\n"
    "Return ONLY a JSON object with a single key 'cluster':\n"
    '{\"cluster\": \"<label>\"}\n\n'
    "Valid labels and when to use each:\n"
    "- product_stagnation: Product not evolving, roadmap ignored, feature requests closed without action, "
    "'they never build what we ask for', 'falling behind competitors on features'\n"
    "- ecosystem_fatigue: Requires too many third-party tools, add-ons, plugins, or integrations just to "
    "function. The core product alone is insufficient. 'need 5 plugins just to do X', 'nickel-and-dimed'\n"
    "- policy_corporate: Friction from legal, compliance, corporate policy, vendor lock-in policies, "
    "procurement process, licensing terms, DPA/GDPR issues, enterprise red tape\n"
    "- competitive_inferiority: Reviewer believes a competitor is simply better at the core job. "
    "No specific technical pain -- just 'X does this better', 'switched to Y because it outperforms'\n"
    "- general_dissatisfaction: Vague unhappiness with no identifiable theme. Use as last resort only "
    "when none of the above fit.\n\n"
    "Rules:\n"
    "- Pick exactly one label\n"
    "- Use 'general_dissatisfaction' only when you cannot assign a more specific label\n"
    "- Do NOT explain your choice"
)


async def main():
    ap = argparse.ArgumentParser(description="Assign thematic cluster labels to 'other' pain reviews")
    ap.add_argument(
        "--vendors",
        default=None,
        help="Comma-separated vendor names to target (default: all vendors)",
    )
    ap.add_argument("--limit", type=int, default=50000, help="Max reviews to process (default: 50000 -- effectively unlimited)")
    ap.add_argument("--concurrency", type=int, default=12, help="Concurrent API calls")
    ap.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model ID (default: claude-haiku-4-5-20251001)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Count scope without making changes")
    ap.add_argument(
        "--recluster",
        action="store_true",
        help="Re-assign clusters even if pain_cluster is already set",
    )
    args = ap.parse_args()

    from atlas_brain.config import settings
    from atlas_brain.storage.database import init_database, get_db_pool, close_database
    from atlas_brain.pipelines.llm import clean_llm_output, parse_json_response

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ATLAS_LLM_ANTHROPIC_API_KEY")
        or ""
    )
    if not api_key:
        logger.error("ANTHROPIC_API_KEY (or ATLAS_LLM_ANTHROPIC_API_KEY) not set")
        sys.exit(1)

    await init_database()
    pool = get_db_pool()

    vendor_list = None
    if args.vendors:
        vendor_list = [v.strip() for v in args.vendors.split(",") if v.strip()]

    # Build query depending on --recluster flag
    cluster_condition = "" if args.recluster else "AND enrichment->>'pain_cluster' IS NULL"

    if vendor_list:
        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name,
                   enrichment->>'specific_complaints' AS specific_complaints_raw,
                   enrichment->>'quotable_phrases'    AS quotable_phrases_raw,
                   enrichment->>'pain_categories'     AS pain_categories_raw,
                   review_text, cons, enrichment
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment->>'pain_category' = 'other'
              {cluster_condition}
              AND LOWER(vendor_name) = ANY($1::text[])
            ORDER BY vendor_name, id
            LIMIT $2
            """,
            [v.lower() for v in vendor_list],
            args.limit,
        )
    else:
        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name,
                   enrichment->>'specific_complaints' AS specific_complaints_raw,
                   enrichment->>'quotable_phrases'    AS quotable_phrases_raw,
                   enrichment->>'pain_categories'     AS pain_categories_raw,
                   review_text, cons, enrichment
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment->>'pain_category' = 'other'
              {cluster_condition}
            ORDER BY vendor_name, id
            LIMIT $1
            """,
            args.limit,
        )

    if not rows:
        logger.info("No unclustered 'other' reviews found for the specified scope")
        await close_database()
        return

    # Group by vendor for reporting
    vendor_counts: dict[str, int] = {}
    for r in rows:
        vn = r["vendor_name"]
        vendor_counts[vn] = vendor_counts.get(vn, 0) + 1

    logger.info("Found %d unclustered 'other' reviews across %d vendors:", len(rows), len(vendor_counts))
    for vn, cnt in sorted(vendor_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-30s %d", vn, cnt)

    if args.dry_run:
        logger.info("DRY RUN -- no changes made")
        await close_database()
        return

    # Stats
    clustered = 0
    general_count = 0
    api_errors = 0
    parse_errors = 0
    done = 0
    cluster_dist: dict[str, int] = {c: 0 for c in VALID_CLUSTERS}

    sem = asyncio.Semaphore(args.concurrency)

    def _parse_enrichment(row) -> dict:
        raw = row.get("enrichment")
        if raw is None:
            return {}
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        return dict(raw)

    def _parse_json_field(raw) -> Any:
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        return raw

    async def cluster_review(client: httpx.AsyncClient, row) -> None:
        nonlocal clustered, general_count, api_errors, parse_errors, done

        async with sem:
            review_id = row["id"]
            existing_enrichment = _parse_enrichment(row)

            # Build minimal evidence payload for the model
            complaints = _parse_json_field(row.get("specific_complaints_raw")) or []
            quotes = _parse_json_field(row.get("quotable_phrases_raw")) or []
            pain_cats = _parse_json_field(row.get("pain_categories_raw")) or []

            if complaints or quotes:
                user_content = json.dumps({
                    "vendor_name": row["vendor_name"],
                    "specific_complaints": complaints[:5],
                    "quotable_phrases": quotes[:3],
                    "cons": (row.get("cons") or "")[:300],
                })
            else:
                # Fall back to raw text snippet
                review_snippet = (row.get("review_text") or "")[:600]
                cons_snippet = (row.get("cons") or "")[:300]
                user_content = json.dumps({
                    "vendor_name": row["vendor_name"],
                    "review_text": review_snippet,
                    "cons": cons_snippet,
                })

            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": args.model,
                        "max_tokens": 64,
                        "system": CLUSTER_SYSTEM_PROMPT,
                        "messages": [
                            {"role": "user", "content": user_content},
                            # Prefill forces valid JSON output
                            {"role": "assistant", "content": '{"cluster":'},
                        ],
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                body = resp.json()
                content_blocks = body.get("content") or []
                raw_text = '{"cluster":' + (
                    content_blocks[0].get("text", "") if content_blocks else ""
                )
            except Exception as exc:
                logger.warning("API error for review %s: %s", review_id, exc)
                api_errors += 1
                done += 1
                return

            raw_text = clean_llm_output(raw_text.strip())
            parsed = parse_json_response(raw_text, recover_truncated=True)

            # Regex fallback: Haiku sometimes outputs just the label name without
            # proper JSON (e.g. ' "general_dissatisfaction"' or just the bare word).
            # Since output is one of 5 known values, extract directly from raw text.
            cluster = None
            if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                raw_cluster = parsed.get("cluster")
                # Guard: model occasionally returns nested dict instead of a string
                if isinstance(raw_cluster, str):
                    cluster = raw_cluster
            if not cluster or cluster not in VALID_CLUSTERS:
                # Try to find any valid cluster name anywhere in the raw response.
                # Iterate VALID_CLUSTERS_ORDERED (not the set) so the first match
                # is deterministic -- more specific labels beat general_dissatisfaction.
                raw_lower = raw_text.lower()
                cluster = None
                for label in VALID_CLUSTERS_ORDERED:
                    if label in raw_lower:
                        cluster = label
                        break

            if not cluster or cluster not in VALID_CLUSTERS:
                parse_errors += 1
                done += 1
                return

            # Patch pain_cluster in existing enrichment
            updated_enrichment = dict(existing_enrichment)
            updated_enrichment["pain_cluster"] = cluster

            # Do NOT touch enriched_at -- pain_cluster is a derived label added
            # on top of existing enrichment; updating enriched_at would make
            # historical reviews appear freshly enriched and distort downstream
            # freshness windows (dashboards, churn alerts).
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment = $1
                WHERE id = $2
                """,
                json.dumps(updated_enrichment),
                review_id,
            )

            clustered += 1
            cluster_dist[cluster] = cluster_dist.get(cluster, 0) + 1
            if cluster == "general_dissatisfaction":
                general_count += 1
            done += 1

            if done % 50 == 0:
                logger.info(
                    "Progress: %d/%d | clustered=%d errors=%d (api=%d parse=%d)",
                    done, len(rows), clustered, api_errors + parse_errors,
                    api_errors, parse_errors,
                )

    logger.info(
        "Starting clustering: %d reviews, model=%s, concurrency=%d",
        len(rows), args.model, args.concurrency,
    )

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[cluster_review(client, row) for row in rows],
            return_exceptions=True,
        )
    # Log any unhandled exceptions that escaped cluster_review (e.g. pool errors)
    unhandled = [r for r in results if isinstance(r, BaseException)]
    if unhandled:
        logger.error("%d unhandled task exceptions (not counted in api_errors/parse_errors):", len(unhandled))
        for exc in unhandled[:5]:
            logger.error("  %s: %s", type(exc).__name__, exc)

    total = len(rows)
    logger.info("")
    logger.info("=" * 70)
    logger.info("PAIN CLUSTER ASSIGNMENT COMPLETE")
    logger.info("=" * 70)
    logger.info("  Model:              %s", args.model)
    logger.info("  Reviews processed:  %d", total)
    logger.info("  Clustered:          %d (%.1f%%)", clustered, clustered * 100 / total if total else 0)
    logger.info("  API errors:         %d", api_errors)
    logger.info("  Parse errors:       %d", parse_errors)
    logger.info("")
    logger.info("  Cluster distribution:")
    for cluster_name, cnt in sorted(cluster_dist.items(), key=lambda x: -x[1]):
        pct = cnt * 100 / clustered if clustered else 0
        label = f"  ({pct:.1f}% of clustered)" if cnt else ""
        logger.info("    %-30s %d%s", cluster_name, cnt, label)
    logger.info("")
    logger.info("  general_dissatisfaction rate: %.1f%% (lower is better)",
                general_count * 100 / clustered if clustered else 0)
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next step: re-run intelligence to see cluster labels in reports:")
    logger.info("  ATLAS_B2B_CHURN_ENABLED=true python scripts/test_intelligence_subset.py \\")
    logger.info('      --vendors "%s"', args.vendors or "all")

    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
