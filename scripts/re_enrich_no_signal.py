#!/usr/bin/env python3
"""Re-enrich no_signal reviews using a second-pass model (Hunter Alpha).

Finds reviews that a primary model triaged as no_signal, resets them to
pending, and runs them through a different model to catch missed signal.

Does NOT touch the production enrichment task or the global LLM registry.
Uses its own dedicated OpenRouter client.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/re_enrich_no_signal.py [--model openrouter/hunter-alpha] [--limit 500] [--concurrency 10] [--primary-model openai/gpt-4.1]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("re_enrich")
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main():
    ap = argparse.ArgumentParser(description="Re-enrich no_signal reviews with a second-pass model")
    ap.add_argument("--model", default="openrouter/hunter-alpha",
                    help="OpenRouter model for second pass (default: hunter-alpha)")
    ap.add_argument("--primary-model", default="openai/gpt-4.1",
                    help="Primary model whose no_signal reviews to re-process")
    ap.add_argument("--limit", type=int, default=500,
                    help="Max reviews to re-enrich")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="Concurrent API calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be done without making changes")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return

    from atlas_brain.autonomous.tasks.b2b_enrichment import (
        _build_classify_payload,
        _validate_enrichment,
        _MIN_REVIEW_TEXT_LENGTH,
    )
    from atlas_brain.config import settings
    from atlas_brain.pipelines.llm import clean_llm_output
    from atlas_brain.services.scraping.sources import VERIFIED_SOURCES
    from atlas_brain.skills import get_skill_registry
    from atlas_brain.storage.database import init_database, get_db_pool, close_database

    await init_database()
    pool = get_db_pool()

    # Load skills
    registry = get_skill_registry()
    extraction_skill = registry.get("digest/b2b_churn_extraction")
    triage_skill = registry.get("digest/b2b_churn_triage")
    if not extraction_skill or not triage_skill:
        logger.error("Missing required skills")
        await close_database()
        return

    # Find no_signal reviews from primary model
    rows = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category, source,
               raw_metadata, rating, rating_max, summary, review_text,
               pros, cons, reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment_attempts
        FROM b2b_reviews
        WHERE enrichment_status = 'no_signal'
          AND enrichment_model = $1
          AND length(COALESCE(review_text, '')) >= $2
        ORDER BY random()
        LIMIT $3
        """,
        args.primary_model,
        _MIN_REVIEW_TEXT_LENGTH,
        args.limit,
    )

    if not rows:
        logger.info("No no_signal reviews found for model %s", args.primary_model)
        await close_database()
        return

    logger.info("Found %d no_signal reviews from %s to re-process with %s",
                len(rows), args.primary_model, args.model)

    if args.dry_run:
        logger.info("DRY RUN — would re-enrich %d reviews", len(rows))
        await close_database()
        return

    # Stats
    enriched = 0
    still_no_signal = 0
    triage_failed = 0
    extraction_failed = 0
    api_errors = 0
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def call_openrouter(client, system_prompt, user_content, max_tokens, temperature):
        t0 = time.monotonic()
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://atlas-brain.local",
                    "X-Title": "Atlas Re-Enrich",
                },
                json=body,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            return {"success": True, "content": content, "latency_ms": int((time.monotonic() - t0) * 1000)}
        except Exception as e:
            return {"success": False, "content": "", "error": str(e), "latency_ms": int((time.monotonic() - t0) * 1000)}

    def parse_json(raw):
        try:
            text = clean_llm_output(raw or "")
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    async def process_review(client, row):
        nonlocal enriched, still_no_signal, triage_failed, extraction_failed, api_errors, done
        async with sem:
            review_id = row["id"]
            source = (row.get("source") or "").lower().strip()
            is_verified = source in VERIFIED_SOURCES

            # Step 1: Triage (skip for verified sources)
            if not is_verified:
                review_text = (row.get("review_text") or "")[:1500]
                triage_payload = json.dumps({
                    "vendor_name": row["vendor_name"],
                    "source": source,
                    "content_type": row.get("content_type") or "review",
                    "rating": float(row["rating"]) if row.get("rating") is not None else None,
                    "summary": row.get("summary") or "",
                    "review_text": review_text,
                    "pros": (row.get("pros") or "")[:300],
                    "cons": (row.get("cons") or "")[:300],
                })

                triage_result = await call_openrouter(
                    client, triage_skill.content, triage_payload,
                    max_tokens=256, temperature=0.0,
                )

                if not triage_result["success"]:
                    api_errors += 1
                    done += 1
                    return

                triage = parse_json(triage_result["content"])
                if triage is None or "signal" not in triage:
                    triage_failed += 1
                    done += 1
                    return

                if not triage.get("signal", True):
                    # Second model also says no signal — leave as no_signal
                    still_no_signal += 1
                    done += 1
                    return

            # Step 2: Full extraction
            payload = json.dumps(_build_classify_payload(
                row, settings.b2b_churn.review_truncate_length,
            ))

            ext_result = await call_openrouter(
                client, extraction_skill.content, payload,
                max_tokens=settings.b2b_churn.enrichment_max_tokens,
                temperature=0.1,
            )

            if not ext_result["success"]:
                api_errors += 1
                done += 1
                return

            parsed = parse_json(ext_result["content"])
            if parsed and _validate_enrichment(parsed, row):
                await pool.execute(
                    """
                    UPDATE b2b_reviews
                    SET enrichment = $1,
                        enrichment_status = 'enriched',
                        enrichment_attempts = enrichment_attempts + 1,
                        enriched_at = $2,
                        enrichment_model = $3
                    WHERE id = $4
                    """,
                    json.dumps(parsed),
                    datetime.now(timezone.utc),
                    args.model,
                    review_id,
                )
                enriched += 1
            else:
                extraction_failed += 1

            done += 1
            if done % 25 == 0:
                logger.info(
                    "Progress: %d/%d done | enriched=%d still_no_signal=%d failed=%d",
                    done, len(rows), enriched, still_no_signal, extraction_failed,
                )

    logger.info("Starting re-enrichment: %d reviews, model=%s, concurrency=%d",
                len(rows), args.model, args.concurrency)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *[process_review(client, row) for row in rows],
            return_exceptions=True,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("RE-ENRICHMENT COMPLETE")
    logger.info("=" * 70)
    logger.info("  Model:            %s", args.model)
    logger.info("  Reviews processed: %d", len(rows))
    logger.info("  Newly enriched:    %d (%.1f%%)", enriched, enriched * 100 / len(rows) if rows else 0)
    logger.info("  Still no_signal:   %d", still_no_signal)
    logger.info("  Triage failed:     %d", triage_failed)
    logger.info("  Extraction failed: %d", extraction_failed)
    logger.info("  API errors:        %d", api_errors)
    logger.info("=" * 70)

    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
