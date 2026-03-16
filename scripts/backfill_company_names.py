#!/usr/bin/env python3
"""Targeted company name extraction from enriched reviews.

Uses qwen3-30b on local vLLM to extract reviewer company names from
review text. Only updates reviewer_company when the column is currently
NULL/empty. Does NOT re-run full enrichment.

Key distinction: extracts the REVIEWER'S company (who wrote the review),
not the VENDOR being reviewed.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/backfill_company_names.py
  python scripts/backfill_company_names.py --source trustradius,gartner,g2 --limit 500
  python scripts/backfill_company_names.py --batch-size 20 --concurrency 5
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/backfill_company_names.log"),
    ],
)
logger = logging.getLogger("backfill_company")

VLLM_URL = "http://localhost:8082/v1/chat/completions"

SYSTEM_PROMPT = (
    "Extract the reviewer's company name from this review. "
    "The REVIEWER is the person who WROTE the review, NOT the vendor/product being reviewed. "
    "For example, if someone reviews Salesforce and says 'We use it at Acme Corp', "
    "the reviewer's company is 'Acme Corp', NOT 'Salesforce'.\n\n"
    "Rules:\n"
    "- Only extract proper company/organization names explicitly stated in the text\n"
    "- Never return the vendor/product being reviewed as the company\n"
    "- Return null if no company name is explicitly mentioned\n"
    "- Do not infer or guess from industry, role, company size, or context\n"
    "- Ignore generic terms: 'my company', 'our org', 'the firm', 'a startup'\n"
    "- Ignore vague descriptions: 'a financial company', 'tech startup in SF'\n"
    "- Ignore person names -- only return organization/business names\n"
    "- The company name must be a proper noun that could be looked up\n\n"
    "Respond with ONLY valid JSON: {\"company\": \"Company Name\"} or {\"company\": null}\n"
    "No thinking, no markdown, no explanation. JSON only."
)


async def _extract_company(session, review_text: str, vendor_name: str,
                           reviewer_title: str, model: str) -> str | None:
    """Call vLLM to extract company name from review text."""
    # Truncate to keep it fast
    text = review_text[:1500] if review_text else ""
    if not text:
        return None

    user_msg = (
        f"Vendor being reviewed: {vendor_name}\n"
        f"Reviewer title: {reviewer_title or 'unknown'}\n\n"
        f"Review text:\n{text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 50,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(VLLM_URL, json=payload)
        if resp.status_code != 200:
            return None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        company = (parsed.get("company") or "").strip()
        if not company:
            return None
        cl = company.lower()
        # Reject vendor name
        if cl == vendor_name.lower():
            return None
        # Reject descriptions (contains common filler words)
        _reject = ["company in", "startup in", "firm in", "agency in",
                    "a company", "a startup", "a firm", "my company",
                    "our company", "the company", "acme corp",
                    "example", "anonymous", "n/a", "none", "null"]
        if any(r in cl for r in _reject):
            return None
        # Reject if too short or too long
        if len(company) < 2 or len(company) > 100:
            return None
        return company
    except Exception:
        return None


async def run(args):
    from atlas_brain.storage.database import init_database, get_db_pool, close_database
    await init_database()
    pool = get_db_pool()

    # Determine which model to use
    model = args.model

    # Build source filter
    source_filter = ""
    params = []
    idx = 1
    if args.source:
        sources = [s.strip() for s in args.source.split(",")]
        placeholders = ", ".join(f"${i}" for i in range(idx, idx + len(sources)))
        source_filter = f"AND source IN ({placeholders})"
        params.extend(sources)
        idx += len(sources)

    # Fetch reviews needing company extraction
    params.append(args.limit)
    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, source, review_text, reviewer_title,
               reviewer_industry, company_size_raw
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND (reviewer_company IS NULL OR reviewer_company = '')
          AND review_text IS NOT NULL
          AND LENGTH(review_text) > 50
          {source_filter}
        ORDER BY
            CASE WHEN source IN ('trustradius', 'gartner', 'g2', 'peerspot', 'capterra') THEN 0 ELSE 1 END,
            (enrichment->>'urgency_score')::numeric DESC NULLS LAST
        LIMIT ${idx}
        """,
        *params,
    )

    logger.info("Found %d reviews to process (model=%s)", len(rows), model)

    sem = asyncio.Semaphore(args.concurrency)
    extracted = 0
    skipped = 0
    failed = 0
    started = time.monotonic()

    async def _process_one(row):
        nonlocal extracted, skipped, failed
        async with sem:
            company = await _extract_company(
                None, row["review_text"], row["vendor_name"],
                row.get("reviewer_title") or "", model,
            )
            if company:
                try:
                    await pool.execute(
                        "UPDATE b2b_reviews SET reviewer_company = $1 WHERE id = $2",
                        company, row["id"],
                    )
                    extracted += 1
                    if extracted % 50 == 0:
                        logger.info("Extracted %d companies so far...", extracted)
                except Exception:
                    failed += 1
            else:
                skipped += 1

    # Process in batches
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        await asyncio.gather(*[_process_one(r) for r in batch])
        if i + args.batch_size < len(rows):
            await asyncio.sleep(0.5)

    duration = round(time.monotonic() - started, 1)
    logger.info(
        "COMPLETE: %d extracted, %d skipped (no company), %d failed, %.1fs",
        extracted, skipped, failed, duration,
    )

    # Show what we found
    if extracted > 0:
        sample = await pool.fetch("""
            SELECT vendor_name, reviewer_company, source
            FROM b2b_reviews
            WHERE reviewer_company IS NOT NULL AND reviewer_company != ''
            ORDER BY enriched_at DESC NULLS LAST
            LIMIT 10
        """)
        logger.info("Sample extracted companies:")
        for r in sample:
            logger.info("  [%s] %s -> %s", r["source"], r["vendor_name"], r["reviewer_company"])

    await close_database()


def main():
    ap = argparse.ArgumentParser(description="Backfill reviewer company names via vLLM")
    ap.add_argument("--source", default=None,
                    help="Comma-separated sources to target (default: all)")
    ap.add_argument("--limit", type=int, default=5000,
                    help="Max reviews to process (default: 5000)")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="Batch size for concurrent processing (default: 20)")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="Max concurrent vLLM requests (default: 10)")
    ap.add_argument("--model", default="stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
                    help="vLLM model name")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
