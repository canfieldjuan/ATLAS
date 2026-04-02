#!/usr/bin/env python3
"""Re-classify pain categories for reviews tagged 'other' using the expanded taxonomy.

Targets only reviews where pain_category = 'other', re-runs Tier 2 classification
with the updated extraction prompt (which includes outcome_gap, admin_burden,
ai_hallucination, integration_debt), and updates only the pain_category and
pain_categories fields in the enrichment JSONB.

Does NOT change enrichment_status or re-run full extraction.

Usage:
    # OpenRouter (default, gpt-oss-120b)
    ATLAS_B2B_CHURN_ENABLED=true python scripts/re_enrich_other_pain.py \
        --vendors "Shopify,Salesforce,Jira"

    # Anthropic Haiku (compare reclassification rate)
    ATLAS_B2B_CHURN_ENABLED=true python scripts/re_enrich_other_pain.py \
        --vendors "Shopify,Salesforce,Jira" --provider anthropic --model haiku

    # After validating, run all vendors (no --vendors flag)
    ATLAS_B2B_CHURN_ENABLED=true python scripts/re_enrich_other_pain.py

    # Dry run to count scope
    ATLAS_B2B_CHURN_ENABLED=true python scripts/re_enrich_other_pain.py \
        --vendors "Shopify,Salesforce,Jira" --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
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
logger = logging.getLogger("re_enrich_other_pain")
logging.getLogger("httpx").setLevel(logging.WARNING)

VALID_PAIN_CATEGORIES = {
    "pricing", "features", "reliability", "support", "integration",
    "performance", "security", "ux", "onboarding", "technical_debt",
    "contract_lock_in", "data_migration", "api_limitations",
    "outcome_gap", "admin_burden", "ai_hallucination", "integration_debt",
    "other",
}


async def main():
    ap = argparse.ArgumentParser(description="Re-classify 'other' pain reviews with expanded taxonomy")
    ap.add_argument(
        "--vendors",
        default=None,
        help="Comma-separated vendor names to target (default: all vendors)",
    )
    ap.add_argument("--limit", type=int, default=2000, help="Max reviews to process")
    ap.add_argument("--concurrency", type=int, default=8, help="Concurrent API calls")
    ap.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "anthropic"],
        help="LLM provider (default: openrouter)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model ID. Shorthands: 'haiku' -> claude-haiku-4-5-20251001 (anthropic), "
             "'sonnet' -> claude-sonnet-4-5 (anthropic). Default: from config (openrouter).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Count scope without making changes")
    ap.add_argument(
        "--include-arrays-with-other",
        action="store_true",
        help="Also reclassify reviews that have 'other' in pain_categories array even if primary is not 'other'",
    )
    args = ap.parse_args()

    from atlas_brain.config import settings
    from atlas_brain.storage.database import init_database, get_db_pool, close_database
    from atlas_brain.pipelines.llm import clean_llm_output, parse_json_response
    from atlas_brain.autonomous.tasks.b2b_enrichment import _finalize_enrichment_for_persist

    cfg = settings.b2b_churn

    # Resolve model shorthand and API key for the chosen provider
    _ANTHROPIC_SHORTHANDS = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5",
        "opus": "claude-opus-4-5",
    }

    if args.provider == "anthropic":
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ATLAS_LLM_ANTHROPIC_API_KEY")
            or ""
        )
        if not api_key:
            logger.error("ANTHROPIC_API_KEY (or ATLAS_LLM_ANTHROPIC_API_KEY) not set")
            sys.exit(1)
        model_id = _ANTHROPIC_SHORTHANDS.get(args.model or "", args.model) or "claude-haiku-4-5-20251001"
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY") or cfg.openrouter_api_key or ""
        if not api_key:
            logger.error("OPENROUTER_API_KEY not set and not in config")
            sys.exit(1)
        model_id = args.model or cfg.enrichment_openrouter_model or "openai/gpt-oss-120b"

    await init_database()
    pool = get_db_pool()

    # Minimal pain-only system prompt -- avoids the ~1500-token full Tier 2 response
    # by asking ONLY for pain_categories. Output is ~60-120 tokens max.
    PAIN_SYSTEM_PROMPT = (
        "You classify B2B software review complaints into pain categories.\n\n"
        "Return ONLY a JSON object with a single key 'pain_categories':\n"
        '{"pain_categories": [{"category": "...", "severity": "..."}]}\n\n'
        "Valid categories: pricing, features, reliability, support, integration, "
        "performance, security, ux, onboarding, technical_debt, contract_lock_in, "
        "data_migration, api_limitations, outcome_gap, admin_burden, "
        "ai_hallucination, integration_debt, other\n\n"
        "Category definitions:\n"
        "- outcome_gap: product fails to deliver promised ROI or business outcomes\n"
        "- admin_burden: excessive admin overhead, complex config, high maintenance cost\n"
        "- ai_hallucination: AI features produce unreliable or fabricated outputs\n"
        "- integration_debt: brittle integrations that break frequently, high maintenance\n\n"
        "Severity: primary (root cause), secondary (contributing), minor (passing mention)\n\n"
        "Rules:\n"
        "- First entry must have severity 'primary'\n"
        "- 'other' ONLY when no complaint maps to any named category\n"
        "- Pricing beats others only when dollar amounts or 'too expensive' stated\n"
        "- Return empty array [] if no complaints present"
    )

    # Build vendor filter
    vendor_list = None
    if args.vendors:
        vendor_list = [v.strip() for v in args.vendors.split(",") if v.strip()]

    # APPROVED-ENRICHMENT-READ: pain_category
    # Reason: backfill/migration script — direct enrichment access required
    # Fetch reviews to reclassify
    if vendor_list:
        rows = await pool.fetch(
            """
            SELECT id, vendor_name, product_name, product_category, source,
                   raw_metadata, rating, rating_max, summary, review_text,
                   pros, cons, reviewer_title, reviewer_company, company_size_raw,
                   reviewer_industry, content_type, enrichment
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment->>'pain_category' = 'other'
              AND LOWER(vendor_name) = ANY($1::text[])
            ORDER BY vendor_name, id
            LIMIT $2
            """,
            [v.lower() for v in vendor_list],
            args.limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, vendor_name, product_name, product_category, source,
                   raw_metadata, rating, rating_max, summary, review_text,
                   pros, cons, reviewer_title, reviewer_company, company_size_raw,
                   reviewer_industry, content_type, enrichment
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment->>'pain_category' = 'other'
            ORDER BY vendor_name, id
            LIMIT $1
            """,
            args.limit,
        )

    if not rows:
        logger.info("No 'other' reviews found for the specified scope")
        await close_database()
        return

    # Group by vendor for reporting
    vendor_counts: dict[str, int] = {}
    for r in rows:
        vn = r["vendor_name"]
        vendor_counts[vn] = vendor_counts.get(vn, 0) + 1

    logger.info("Found %d 'other' reviews across %d vendors:", len(rows), len(vendor_counts))
    for vn, cnt in sorted(vendor_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-30s %d", vn, cnt)

    if args.dry_run:
        logger.info("DRY RUN -- no changes made")
        await close_database()
        return

    # Stats
    reclassified = 0
    still_other = 0
    api_errors = 0
    parse_errors = 0
    done = 0

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

    async def reclassify_review(client: httpx.AsyncClient, row) -> None:
        nonlocal reclassified, still_other, api_errors, parse_errors, done

        async with sem:
            review_id = row["id"]
            existing_enrichment = _parse_enrichment(row)

            # Build a minimal user payload: only the evidence the model needs
            # to classify pain. Keeps input tokens small and response ~100 tokens.
            complaints = existing_enrichment.get("specific_complaints") or []
            quotes = existing_enrichment.get("quotable_phrases") or []
            # Fall back to raw review text snippet if no extracted complaints
            if not complaints and not quotes:
                review_snippet = (row.get("review_text") or "")[:800]
                cons_snippet = (row.get("cons") or "")[:300]
                user_content = json.dumps({
                    "vendor_name": row["vendor_name"],
                    "review_text": review_snippet,
                    "cons": cons_snippet,
                })
            else:
                user_content = json.dumps({
                    "vendor_name": row["vendor_name"],
                    "specific_complaints": complaints,
                    "quotable_phrases": quotes,
                    "cons": (row.get("cons") or "")[:300],
                })

            t0 = time.monotonic()
            try:
                if args.provider == "anthropic":
                    # Anthropic Messages API with JSON prefill to guarantee clean output
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_id,
                            "max_tokens": 256,
                            "system": PAIN_SYSTEM_PROMPT,
                            "messages": [
                                {"role": "user", "content": user_content},
                                # Prefill forces the model to start with valid JSON
                                {"role": "assistant", "content": '{"pain_categories":'},
                            ],
                        },
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    # Anthropic response: content[0].text — prefill is prepended back
                    content_blocks = body.get("content") or []
                    raw_text = '{"pain_categories":' + (
                        content_blocks[0].get("text", "") if content_blocks else ""
                    )
                else:
                    # OpenRouter (OpenAI-compatible)
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://atlas-brain.local",
                            "X-Title": "Atlas Pain Reclassify",
                        },
                        json={
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": PAIN_SYSTEM_PROMPT},
                                {"role": "user", "content": user_content},
                            ],
                            "max_tokens": 256,
                            "temperature": 0.0,
                            "response_format": {"type": "json_object"},
                        },
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    choices = body.get("choices") or []
                    if not choices:
                        api_errors += 1
                        done += 1
                        return
                    msg = choices[0].get("message") or {}
                    raw_text = msg.get("content") or ""
                    # Reasoning models (gpt-oss-120b, o1/o3) may put output in reasoning field
                    if not raw_text and msg.get("reasoning"):
                        import re as _re
                        m = _re.search(r"\{[\s\S]*\}", msg["reasoning"])
                        if m:
                            raw_text = m.group(0)

            except Exception as exc:
                logger.warning("API error for review %s: %s", review_id, exc)
                api_errors += 1
                done += 1
                return

            raw_text = clean_llm_output(raw_text.strip())

            parsed = parse_json_response(raw_text, recover_truncated=True)
            if not isinstance(parsed, dict) or parsed.get("_parse_fallback"):
                parse_errors += 1
                done += 1
                return

            # Extract pain_categories from Tier 2 response
            pain_cats = parsed.get("pain_categories")
            if not pain_cats or not isinstance(pain_cats, list):
                still_other += 1
                done += 1
                return

            # Validate categories
            valid_cats = [
                c for c in pain_cats
                if isinstance(c, dict)
                and c.get("category") in VALID_PAIN_CATEGORIES
                and c.get("severity") in ("primary", "secondary", "minor")
            ]
            if not valid_cats:
                still_other += 1
                done += 1
                return

            primary_category = valid_cats[0]["category"]

            # Only update if we resolved to something other than 'other'
            if primary_category == "other":
                still_other += 1
                done += 1
                return

            # Patch only pain fields in the existing enrichment JSONB
            updated_enrichment = dict(existing_enrichment)
            updated_enrichment["pain_categories"] = valid_cats
            updated_enrichment["pain_category"] = primary_category
            finalized, _ = _finalize_enrichment_for_persist(updated_enrichment, dict(row))
            if not finalized:
                parse_errors += 1
                done += 1
                return

            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment = $1,
                    enriched_at = $2
                WHERE id = $3
                """,
                json.dumps(finalized),
                datetime.now(timezone.utc),
                review_id,
            )

            reclassified += 1
            done += 1

            if done % 50 == 0:
                logger.info(
                    "Progress: %d/%d | reclassified=%d still_other=%d errors=%d (api=%d parse=%d)",
                    done, len(rows), reclassified, still_other, api_errors + parse_errors,
                    api_errors, parse_errors,
                )

    logger.info(
        "Starting reclassification: %d reviews, provider=%s model=%s, concurrency=%d",
        len(rows), args.provider, model_id, args.concurrency,
    )

    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *[reclassify_review(client, row) for row in rows],
            return_exceptions=True,
        )

    total = len(rows)
    logger.info("")
    logger.info("=" * 70)
    logger.info("PAIN RECLASSIFICATION COMPLETE")
    logger.info("=" * 70)
    logger.info("  Provider:           %s", args.provider)
    logger.info("  Model:              %s", model_id)
    logger.info("  Reviews processed:  %d", total)
    logger.info("  Reclassified:       %d (%.1f%%)", reclassified, reclassified * 100 / total if total else 0)
    logger.info("  Still 'other':      %d (%.1f%%)", still_other, still_other * 100 / total if total else 0)
    logger.info("  API errors:         %d", api_errors)
    logger.info("  Parse errors:       %d", parse_errors)
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next step: re-run intelligence to recompute signals with updated pain data:")
    logger.info("  ATLAS_B2B_CHURN_ENABLED=true python scripts/test_intelligence_subset.py \\")
    logger.info('      --vendors "%s"', args.vendors or "all")

    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
