"""
Parallel complaint enrichment blaster.

Runs N concurrent workers. Each worker:
  1. Claims a batch (UPDATE ... SET enrichment_status='processing' ... SKIP LOCKED)
  2. Releases the transaction immediately
  3. Classifies via LLM (slow part -- no locks held)
  4. Writes results back

Usage:
    python scripts/blast_enrichment.py                # 3 workers (default)
    python scripts/blast_enrichment.py --workers 5
    python scripts/blast_enrichment.py --workers 3 --batch 50 --reviews-per-call 10
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env so API keys are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("blast_enrichment")

_enriched = 0
_failed = 0
_lock = asyncio.Lock()


async def worker(
    worker_id: int,
    pool,
    llm,
    skill_content: str,
    batch_size: int,
    max_attempts: int,
    reviews_per_call: int,
    max_tokens_single: int = 256,
    max_tokens_batch_mult: int = 300,
):
    global _enriched, _failed

    while True:
        # Step 1: Claim a batch (fast transaction, immediately released)
        async with pool.transaction() as conn:
            rows = await conn.fetch(
                """
                UPDATE product_reviews
                SET enrichment_status = 'processing'
                WHERE id IN (
                    SELECT id FROM product_reviews
                    WHERE enrichment_status = 'pending'
                      AND enrichment_attempts < $1
                    ORDER BY imported_at ASC
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, asin, rating, summary, review_text,
                          hardware_category, issue_types, enrichment_attempts
                """,
                max_attempts,
                batch_size,
            )

        if not rows:
            return  # Nothing left

        # Step 2: Classify outside transaction (slow LLM calls, no locks)
        for i in range(0, len(rows), reviews_per_call):
            chunk = rows[i:i + reviews_per_call]

            classifications = None
            if len(chunk) > 1:
                classifications = await asyncio.to_thread(
                    _classify_batch, llm, skill_content, chunk, max_tokens_batch_mult
                )

            if classifications and len(classifications) == len(chunk):
                for row, cls in zip(chunk, classifications):
                    if cls and cls.get("root_cause"):
                        await _apply(pool, row, cls)
                        async with _lock:
                            _enriched += 1
                    else:
                        await _mark_failed_or_retry(pool, row, max_attempts)
                        if row["enrichment_attempts"] + 1 >= max_attempts:
                            async with _lock:
                                _failed += 1
            else:
                # Single fallback (or single mode)
                for row in chunk:
                    cls = await asyncio.to_thread(
                        _classify_single, llm, skill_content, row, max_tokens_single
                    )
                    if cls:
                        await _apply(pool, row, cls)
                        async with _lock:
                            _enriched += 1
                    else:
                        await _mark_failed_or_retry(pool, row, max_attempts)
                        if row["enrichment_attempts"] + 1 >= max_attempts:
                            async with _lock:
                                _failed += 1


async def _apply(pool, row, classification: dict) -> None:
    await pool.execute(
        """
        UPDATE product_reviews
        SET root_cause = $1, specific_complaint = $2, severity = $3,
            pain_score = $4, time_to_failure = $5,
            workaround_found = $6, workaround_text = $7,
            alternative_mentioned = $8, alternative_asin = $9,
            alternative_name = $10, actionable_for_manufacturing = $11,
            manufacturing_suggestion = $12,
            enrichment_status = 'enriched',
            enrichment_attempts = $13, enriched_at = $14
        WHERE id = $15
        """,
        classification.get("root_cause"),
        classification.get("specific_complaint"),
        classification.get("severity"),
        classification.get("pain_score"),
        classification.get("time_to_failure"),
        classification.get("workaround_found"),
        classification.get("workaround_text"),
        classification.get("alternative_mentioned"),
        classification.get("alternative_asin"),
        classification.get("alternative_name"),
        classification.get("actionable_for_manufacturing"),
        classification.get("manufacturing_suggestion"),
        row["enrichment_attempts"] + 1,
        datetime.now(timezone.utc),
        row["id"],
    )


async def _mark_failed_or_retry(pool, row, max_attempts: int) -> None:
    new = row["enrichment_attempts"] + 1
    if new >= max_attempts:
        await pool.execute(
            "UPDATE product_reviews SET enrichment_status = 'failed', enrichment_attempts = $1 WHERE id = $2",
            new, row["id"],
        )
    else:
        # Put back to pending so it can be retried
        await pool.execute(
            "UPDATE product_reviews SET enrichment_status = 'pending', enrichment_attempts = $1 WHERE id = $2",
            new, row["id"],
        )


# ------------------------------------------------------------------
# LLM calls
# ------------------------------------------------------------------

_BATCH_SUFFIX = """

## BATCH MODE

You will receive a JSON array of reviews. Classify EACH review independently.
Return a JSON array of classification objects in the SAME ORDER as the input.
Each object must follow the single-review output format above.
Return ONLY the JSON array -- no prose, no markdown fencing."""


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


def _classify_single(llm, skill_content: str, row, max_tokens: int = 256) -> dict | None:
    from atlas_brain.services.protocols import Message

    payload = json.dumps({
        "asin": row["asin"],
        "rating": float(row["rating"]),
        "summary": row["summary"] or "",
        "review_text": (row["review_text"] or "")[:2000],
        "hardware_category": list(row["hardware_category"] or []),
        "issue_types": list(row["issue_types"] or []),
    })
    try:
        result = llm.chat(
            messages=[
                Message(role="system", content=skill_content),
                Message(role="user", content=payload),
            ],
            max_tokens=max_tokens, temperature=0.1,
        )
        text = _clean(result.get("response", ""))
        if not text:
            return None
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "root_cause" in parsed:
            return parsed
    except Exception:
        pass
    return None


def _classify_batch(llm, skill_content: str, rows, max_tokens_per_review: int = 300) -> list[dict] | None:
    from atlas_brain.services.protocols import Message

    reviews = [
        {
            "asin": r["asin"],
            "rating": float(r["rating"]),
            "summary": r["summary"] or "",
            "review_text": (r["review_text"] or "")[:2000],
            "hardware_category": list(r["hardware_category"] or []),
            "issue_types": list(r["issue_types"] or []),
        }
        for r in rows
    ]
    try:
        result = llm.chat(
            messages=[
                Message(role="system", content=skill_content + _BATCH_SUFFIX),
                Message(role="user", content=json.dumps(reviews)),
            ],
            max_tokens=max_tokens_per_review * len(rows), temperature=0.1,
        )
        text = _clean(result.get("response", ""))
        if not text:
            return None
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == len(rows):
            if all(isinstance(x, dict) and "root_cause" in x for x in parsed):
                return parsed
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# Progress monitor
# ------------------------------------------------------------------


async def monitor(pool, start_time: float, initial_enriched: int):
    while True:
        await asyncio.sleep(30)
        row = await pool.fetchrow(
            "SELECT count(*) FILTER (WHERE enrichment_status = 'enriched') AS done, "
            "count(*) FILTER (WHERE enrichment_status = 'pending') AS pending, "
            "count(*) FILTER (WHERE enrichment_status = 'failed') AS failed, "
            "count(*) FILTER (WHERE enrichment_status = 'processing') AS processing "
            "FROM product_reviews"
        )
        done = row["done"]
        pending = row["pending"]
        failed = row["failed"]
        processing = row["processing"]
        elapsed = time.monotonic() - start_time
        session_done = done - initial_enriched
        rate = session_done / elapsed * 3600 if elapsed > 0 else 0
        eta_hours = (pending + processing) / rate if rate > 0 else 0
        print(
            f"[{elapsed:6.0f}s] enriched: {done:,} (+{session_done:,}) | "
            f"processing: {processing:,} | pending: {pending:,} | failed: {failed:,} | "
            f"rate: {rate:,.0f}/hr | ETA: {eta_hours:.1f}h",
            flush=True,
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--reviews-per-call", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. qwen3:14b, anthropic/claude-haiku)")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "openrouter", "groq", "together"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--api-key", type=str, default=None, help="API key override (reads env var by default)")
    parser.add_argument("--max-tokens-single", type=int, default=256, help="Max tokens for single-review calls")
    parser.add_argument("--max-tokens-batch-mult", type=int, default=300, help="Tokens per review in batch calls")
    args = parser.parse_args()

    from atlas_brain.config import settings
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.services import llm_registry
    from atlas_brain.skills import get_skill_registry

    pool = get_db_pool()
    await pool.initialize()

    provider = args.provider
    if provider == "ollama":
        model = args.model or settings.llm.ollama_model
        llm_registry.activate("ollama", model=model, base_url=settings.llm.ollama_url)
    elif provider == "openrouter":
        model = args.model or "anthropic/claude-haiku"
        activate_kwargs = {"model": model}
        if args.api_key:
            activate_kwargs["api_key"] = args.api_key
        llm_registry.activate("openrouter", **activate_kwargs)
    elif provider == "groq":
        model = args.model or "llama-3.3-70b-versatile"
        activate_kwargs = {"model": model}
        if args.api_key:
            activate_kwargs["api_key"] = args.api_key
        llm_registry.activate("groq", **activate_kwargs)
    elif provider == "together":
        model = args.model or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        activate_kwargs = {"model": model}
        if args.api_key:
            activate_kwargs["api_key"] = args.api_key
        llm_registry.activate("together", **activate_kwargs)

    llm = llm_registry.get_active()
    if not llm:
        print("ERROR: No LLM available")
        return

    skill = get_skill_registry().get("digest/complaint_classification")
    if not skill:
        print("ERROR: complaint_classification skill not found")
        return

    row = await pool.fetchrow(
        "SELECT count(*) FILTER (WHERE enrichment_status = 'enriched') AS done, "
        "count(*) FILTER (WHERE enrichment_status = 'pending') AS pending "
        "FROM product_reviews"
    )
    initial = row["done"]
    pending = row["pending"]

    # Reset any stuck 'processing' rows from prior crashed runs
    await pool.execute(
        "UPDATE product_reviews SET enrichment_status = 'pending' WHERE enrichment_status = 'processing'"
    )

    print(
        f"Starting {args.workers} workers | provider={provider} | model={model} | "
        f"batch={args.batch} | reviews_per_call={args.reviews_per_call} | pending={pending:,}",
        flush=True,
    )

    start = time.monotonic()
    mon = asyncio.create_task(monitor(pool, start, initial))

    workers = [
        asyncio.create_task(
            worker(i, pool, llm, skill.content, args.batch,
                   args.max_attempts, args.reviews_per_call,
                   args.max_tokens_single, args.max_tokens_batch_mult)
        )
        for i in range(args.workers)
    ]

    await asyncio.gather(*workers)
    mon.cancel()

    elapsed = time.monotonic() - start
    print(f"\nDone in {elapsed:.0f}s | enriched: {_enriched:,} | failed: {_failed:,}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
