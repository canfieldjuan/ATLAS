"""
Parallel deep enrichment blaster.

Single-pass extraction: 32 rich fields per review using local LLM (qwen3:14b).
Single-review mode only (no batching -- output is ~1200 tokens, batching risks truncation).

Runs N concurrent workers. Each worker:
  1. Claims a batch (UPDATE ... SET deep_enrichment_status='processing' ... SKIP LOCKED)
  2. Releases the transaction immediately
  3. Extracts via LLM (slow part -- no locks held)
  4. Writes JSONB results back

Usage:
    python scripts/blast_deep_enrichment.py                         # 3 workers, all tiers
    python scripts/blast_deep_enrichment.py --tier 1 --workers 2    # tier 1 only (50+ reviews/ASIN)
    python scripts/blast_deep_enrichment.py --validate 10           # inspect 10 sample outputs
    python scripts/blast_deep_enrichment.py --dry-run               # show tier counts, exit
"""

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env", override=True)
load_dotenv(_root / ".env.local", override=True)


def _resolve_vllm_url() -> str:
    explicit = os.getenv("VLLM_BASE_URL")
    if explicit:
        return explicit
    host = os.getenv("VLLM_HOST", "localhost")
    port = os.getenv("VLLM_PORT", "8080")
    return f"http://{host}:{port}"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("blast_deep_enrichment")

_enriched = 0
_failed = 0
_lock = asyncio.Lock()

# Required keys and their expected types for validation (32 fields across 3 sections)
_REQUIRED_KEYS = {
    # Section A: Product Analysis (10)
    "sentiment_aspects": list,
    "feature_requests": list,
    "failure_details": (dict, type(None)),
    "product_comparisons": list,
    "product_name_mentioned": str,
    "buyer_context": dict,
    "quotable_phrases": list,
    "would_repurchase": (bool, type(None)),
    "external_references": list,
    "positive_aspects": list,
    # Section B: Buyer Psychology (12)
    "expertise_level": str,
    "frustration_threshold": str,
    "discovery_channel": str,
    "consideration_set": list,
    "buyer_household": str,
    "profession_hint": (str, type(None)),
    "budget_type": str,
    "use_intensity": str,
    "research_depth": str,
    "community_mentions": list,
    "consequence_severity": str,
    "replacement_behavior": str,
    # Section C: Extended Context (10)
    "brand_loyalty_depth": str,
    "ecosystem_lock_in": dict,
    "safety_flag": dict,
    "bulk_purchase_signal": dict,
    "review_delay_signal": str,
    "sentiment_trajectory": str,
    "occasion_context": str,
    "switching_barrier": dict,
    "amplification_intent": dict,
    "review_sentiment_openness": dict,
}

# Enum sets for Section B
_VALID_EXPERTISE = {"novice", "intermediate", "expert", "professional"}
_VALID_FRUSTRATION = {"low", "medium", "high"}
_VALID_DISCOVERY = {"amazon_organic", "youtube", "reddit", "friend", "amazon_choice", "unknown"}
_VALID_HOUSEHOLD = {"single", "family", "professional", "gift", "bulk"}
_VALID_BUDGET = {"budget_constrained", "value_seeker", "premium_willing", "unknown"}
_VALID_INTENSITY = {"light", "moderate", "heavy"}
_VALID_RESEARCH = {"impulse", "light", "moderate", "deep"}
_VALID_CONSEQUENCE = {"inconvenience", "workflow_impact", "financial_loss", "safety_concern"}
_VALID_REPLACEMENT = {"returned", "replaced_same", "switched_brand", "kept_broken", "unknown"}

# Enum sets for Section A sub-objects
_VALID_SENTIMENT = {"positive", "negative", "mixed"}
_VALID_DIRECTION = {"switched_to", "switched_from", "considered", "compared"}
_VALID_PRICE_SENTIMENT = {"expensive", "fair", "cheap", "not_mentioned"}

# Enum sets for Section C
_VALID_LOYALTY = {"first_time", "occasional", "loyal", "long_term_loyal"}
_VALID_DELAY = {"immediate", "days", "weeks", "months", "unknown"}
_VALID_TRAJECTORY = {"always_bad", "degraded", "mixed_then_bad", "initially_positive", "unknown"}
_VALID_OCCASION = {"none", "gift", "replacement", "upgrade", "first_in_category", "seasonal"}

# Enum sets for Section C sub-objects
_VALID_ECOSYSTEM_LEVEL = {"free", "partially", "fully"}
_VALID_BARRIER_LEVEL = {"none", "low", "medium", "high"}
_VALID_AMPLIFICATION = {"quiet", "private", "social"}
_VALID_BULK_TYPE = {"single", "multi"}

# Tier definitions: (min_reviews, max_reviews_or_none)
_TIERS = {
    "1": (50, None),
    "2": (20, 49),
    "3": (10, 19),
}


def _tier_subquery(tier: str) -> str:
    """Build the ASIN filter subquery for a given tier."""
    if tier == "all":
        return ""
    min_rev, max_rev = _TIERS[tier]
    having = f"HAVING count(*) >= {min_rev}"
    if max_rev is not None:
        having += f" AND count(*) <= {max_rev}"
    return (
        f"AND pr.asin IN ("
        f"SELECT asin FROM product_reviews "
        f"WHERE enrichment_status = 'enriched' "
        f"GROUP BY asin {having})"
    )


def _validate_extraction(data: dict) -> bool:
    """Check all 32 required keys present with correct types, enums, and sub-objects."""
    for key, expected in _REQUIRED_KEYS.items():
        if key not in data:
            return False
        val = data[key]
        if isinstance(expected, tuple):
            if not isinstance(val, expected):
                return False
        else:
            if not isinstance(val, expected):
                return False

    # Section A sub-object: buyer_context
    bc = data.get("buyer_context")
    if isinstance(bc, dict):
        for field in ("use_case", "buyer_type", "price_sentiment"):
            if field not in bc:
                return False
        ps = bc.get("price_sentiment")
        if ps is not None and ps not in _VALID_PRICE_SENTIMENT:
            return False

    # Section A sub-object: sentiment_aspects enum values
    aspects = data.get("sentiment_aspects")
    if isinstance(aspects, list):
        for a in aspects:
            if isinstance(a, dict) and a.get("sentiment") not in _VALID_SENTIMENT:
                return False

    # Section A sub-object: product_comparisons enum values
    comparisons = data.get("product_comparisons")
    if isinstance(comparisons, list):
        for c in comparisons:
            if isinstance(c, dict) and c.get("direction") not in _VALID_DIRECTION:
                return False

    # Section B enum validation
    if data.get("expertise_level") not in _VALID_EXPERTISE:
        return False
    if data.get("frustration_threshold") not in _VALID_FRUSTRATION:
        return False
    if data.get("discovery_channel") not in _VALID_DISCOVERY:
        return False
    if data.get("buyer_household") not in _VALID_HOUSEHOLD:
        return False
    if data.get("budget_type") not in _VALID_BUDGET:
        return False
    if data.get("use_intensity") not in _VALID_INTENSITY:
        return False
    if data.get("research_depth") not in _VALID_RESEARCH:
        return False
    if data.get("consequence_severity") not in _VALID_CONSEQUENCE:
        return False
    if data.get("replacement_behavior") not in _VALID_REPLACEMENT:
        return False

    # Section C enum validation
    if data.get("brand_loyalty_depth") not in _VALID_LOYALTY:
        return False
    if data.get("review_delay_signal") not in _VALID_DELAY:
        return False
    if data.get("sentiment_trajectory") not in _VALID_TRAJECTORY:
        return False
    if data.get("occasion_context") not in _VALID_OCCASION:
        return False

    # Section C sub-object key + enum checks
    eco = data.get("ecosystem_lock_in", {})
    if not isinstance(eco, dict) or "level" not in eco or "ecosystem" not in eco:
        return False
    if eco["level"] not in _VALID_ECOSYSTEM_LEVEL:
        return False
    safety = data.get("safety_flag", {})
    if not isinstance(safety, dict) or "flagged" not in safety or "description" not in safety:
        return False
    bulk = data.get("bulk_purchase_signal", {})
    if not isinstance(bulk, dict) or "type" not in bulk or "estimated_qty" not in bulk:
        return False
    if bulk["type"] not in _VALID_BULK_TYPE:
        return False
    barrier = data.get("switching_barrier", {})
    if not isinstance(barrier, dict) or "level" not in barrier or "reason" not in barrier:
        return False
    if barrier["level"] not in _VALID_BARRIER_LEVEL:
        return False
    amp = data.get("amplification_intent", {})
    if not isinstance(amp, dict) or "intent" not in amp or "context" not in amp:
        return False
    if amp["intent"] not in _VALID_AMPLIFICATION:
        return False
    openness = data.get("review_sentiment_openness", {})
    if not isinstance(openness, dict) or "open" not in openness or "condition" not in openness:
        return False

    return True


# ------------------------------------------------------------------
# Per-worker metadata cache
# ------------------------------------------------------------------

class MetadataCache:
    """Bounded LRU-ish cache for product metadata lookups."""

    def __init__(self, pool, max_size: int = 1000):
        self._pool = pool
        self._cache: dict[str, tuple] = {}
        self._max_size = max_size

    async def get(self, asin: str) -> tuple[str, str]:
        """Return (title, brand) for an ASIN, hitting DB on miss."""
        if asin in self._cache:
            return self._cache[asin]

        row = await self._pool.fetchrow(
            "SELECT title, brand FROM product_metadata WHERE asin = $1",
            asin,
        )
        if row:
            val = (row["title"] or "", row["brand"] or "")
        else:
            val = ("", "")

        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[asin] = val
        return val


# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------


async def worker(
    worker_id: int,
    pool,
    llm,
    skill_content: str,
    batch_size: int,
    max_attempts: int,
    max_tokens: int,
    tier: str,
    metadata_cache: MetadataCache,
    limit: int,
):
    global _enriched, _failed

    tier_filter = _tier_subquery(tier)
    processed = 0

    while True:
        if limit > 0 and processed >= limit:
            return

        claim_limit = batch_size
        if limit > 0:
            claim_limit = min(batch_size, limit - processed)

        async with pool.transaction() as conn:
            rows = await conn.fetch(
                f"""
                UPDATE product_reviews
                SET deep_enrichment_status = 'processing'
                WHERE id IN (
                    SELECT pr.id FROM product_reviews pr
                    WHERE pr.deep_enrichment_status = 'pending'
                      AND pr.deep_enrichment_attempts < $1
                      {tier_filter}
                    ORDER BY pr.imported_at ASC
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, asin, rating, summary, review_text,
                          root_cause, severity, pain_score,
                          deep_enrichment_attempts
                """,
                max_attempts,
                claim_limit,
            )

        if not rows:
            return

        for row in rows:
            title, brand = await metadata_cache.get(row["asin"])
            extraction = await asyncio.to_thread(
                _extract_single, llm, skill_content, row, title, brand, max_tokens
            )

            if extraction and _validate_extraction(extraction):
                await _apply(pool, row, extraction)
                async with _lock:
                    _enriched += 1
            else:
                await _mark_failed_or_retry(pool, row, max_attempts)
                if row["deep_enrichment_attempts"] + 1 >= max_attempts:
                    async with _lock:
                        _failed += 1

            processed += 1
            if limit > 0 and processed >= limit:
                return


async def _apply(pool, row, extraction: dict) -> None:
    await pool.execute(
        """
        UPDATE product_reviews
        SET deep_extraction = $1,
            deep_enrichment_status = 'enriched',
            deep_enrichment_attempts = $2,
            deep_enriched_at = $3
        WHERE id = $4
        """,
        json.dumps(extraction),
        row["deep_enrichment_attempts"] + 1,
        datetime.now(timezone.utc),
        row["id"],
    )


async def _mark_failed_or_retry(pool, row, max_attempts: int) -> None:
    new = row["deep_enrichment_attempts"] + 1
    if new >= max_attempts:
        await pool.execute(
            "UPDATE product_reviews SET deep_enrichment_status = 'deep_failed', deep_enrichment_attempts = $1 WHERE id = $2",
            new, row["id"],
        )
    else:
        await pool.execute(
            "UPDATE product_reviews SET deep_enrichment_status = 'pending', deep_enrichment_attempts = $1 WHERE id = $2",
            new, row["id"],
        )


# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------


def _extract_single(llm, skill_content: str, row, title: str, brand: str, max_tokens: int) -> dict | None:
    from atlas_brain.services.protocols import Message
    from atlas_brain.pipelines.llm import clean_llm_output

    payload = json.dumps({
        "review_text": (row["review_text"] or "")[:3000],
        "summary": row["summary"] or "",
        "rating": float(row["rating"]),
        "product_name": title or row["asin"],
        "brand": brand or "",
        "root_cause": row["root_cause"] or "",
        "severity": row["severity"] or "",
        "pain_score": float(row["pain_score"]) if row["pain_score"] else 0.0,
    })

    try:
        result = llm.chat(
            messages=[
                Message(role="system", content=skill_content),
                Message(role="user", content=payload),
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        text = clean_llm_output(result.get("response", ""))
        if not text:
            return None
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        logger.debug("LLM extraction failed for review %s", row["id"], exc_info=True)
    return None


# ------------------------------------------------------------------
# Progress monitor
# ------------------------------------------------------------------


async def monitor(pool, start_time: float, initial_enriched: int, tier: str):
    while True:
        await asyncio.sleep(30)
        row = await pool.fetchrow(
            "SELECT count(*) FILTER (WHERE deep_enrichment_status = 'enriched') AS done, "
            "count(*) FILTER (WHERE deep_enrichment_status = 'pending') AS pending, "
            "count(*) FILTER (WHERE deep_enrichment_status = 'deep_failed') AS failed, "
            "count(*) FILTER (WHERE deep_enrichment_status = 'processing') AS processing "
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
            f"[{elapsed:6.0f}s] deep_enriched: {done:,} (+{session_done:,}) | "
            f"processing: {processing:,} | pending: {pending:,} | deep_failed: {failed:,} | "
            f"rate: {rate:,.0f}/hr | ETA: {eta_hours:.1f}h | tier: {tier}",
            flush=True,
        )


# ------------------------------------------------------------------
# Validate mode
# ------------------------------------------------------------------


async def validate_mode(pool, llm, skill_content: str, n: int, tier: str, max_tokens: int, max_attempts: int):
    """Process N reviews, print formatted results, exit."""
    tier_filter = _tier_subquery(tier)
    metadata_cache = MetadataCache(pool)

    rows = await pool.fetch(
        f"""
        SELECT pr.id, pr.asin, pr.rating, pr.summary, pr.review_text,
               pr.root_cause, pr.severity, pr.pain_score,
               pr.deep_enrichment_attempts
        FROM product_reviews pr
        WHERE pr.deep_enrichment_status = 'pending'
          AND pr.deep_enrichment_attempts < $1
          {tier_filter}
        ORDER BY pr.imported_at ASC
        LIMIT $2
        """,
        max_attempts,
        n,
    )

    if not rows:
        print("No pending reviews found for validation.")
        return

    print(f"\n{'='*80}")
    print(f"VALIDATING {len(rows)} REVIEWS (tier={tier})")
    print(f"{'='*80}\n")

    ok = 0
    bad = 0
    for i, row in enumerate(rows, 1):
        title, brand = await metadata_cache.get(row["asin"])
        extraction = await asyncio.to_thread(
            _extract_single, llm, skill_content, row, title, brand, max_tokens
        )

        valid = extraction and _validate_extraction(extraction)
        status = "OK" if valid else "FAIL"
        if valid:
            ok += 1
        else:
            bad += 1

        print(f"--- Review {i}/{len(rows)} [{status}] ---")
        print(f"  ASIN: {row['asin']} | Rating: {row['rating']} | Product: {title or 'N/A'}")
        print(f"  Summary: {(row['summary'] or '')[:80]}")
        print(f"  Review: {(row['review_text'] or '')[:120]}...")

        if extraction:
            print(f"  Keys present ({len(extraction.keys())}/32): {sorted(extraction.keys())}")
            missing = set(_REQUIRED_KEYS) - set(extraction.keys())
            if missing:
                print(f"  MISSING: {missing}")

            # Section A highlights
            if "feature_requests" in extraction:
                print(f"  Feature requests: {extraction['feature_requests']}")
            if "quotable_phrases" in extraction:
                print(f"  Quotes: {extraction['quotable_phrases']}")
            if "product_name_mentioned" in extraction:
                print(f"  Product name: {extraction['product_name_mentioned']}")
            if "would_repurchase" in extraction:
                print(f"  Would repurchase: {extraction['would_repurchase']}")
            if "sentiment_aspects" in extraction:
                aspects = extraction["sentiment_aspects"]
                print(f"  Sentiment aspects ({len(aspects)}): {[a.get('aspect','?') for a in aspects if isinstance(a, dict)]}")
            if "product_comparisons" in extraction:
                comps = extraction["product_comparisons"]
                if comps:
                    print(f"  Comparisons: {[c.get('product_name','?') for c in comps if isinstance(c, dict)]}")
            if "buyer_context" in extraction:
                bc = extraction["buyer_context"]
                if isinstance(bc, dict):
                    print(f"  Buyer: {bc.get('use_case','?')} / {bc.get('buyer_type','?')} / {bc.get('price_sentiment','?')}")

            # Section B highlights
            if "expertise_level" in extraction:
                print(f"  Expertise: {extraction['expertise_level']} | Frustration: {extraction.get('frustration_threshold','?')}")
            if "profession_hint" in extraction:
                print(f"  Profession: {extraction['profession_hint']}")
            if "consequence_severity" in extraction:
                print(f"  Consequence: {extraction['consequence_severity']} | Replacement: {extraction.get('replacement_behavior','?')}")

            # Section C highlights
            if "brand_loyalty_depth" in extraction:
                print(f"  Brand loyalty: {extraction['brand_loyalty_depth']} | Trajectory: {extraction.get('sentiment_trajectory','?')}")
            if "safety_flag" in extraction:
                sf = extraction["safety_flag"]
                if isinstance(sf, dict) and sf.get("flagged"):
                    print(f"  SAFETY FLAG: {sf.get('description','?')}")
            if "amplification_intent" in extraction:
                ai = extraction["amplification_intent"]
                if isinstance(ai, dict) and ai.get("intent") != "quiet":
                    print(f"  Amplification: {ai.get('intent','?')} -- {ai.get('context','?')}")
        else:
            print("  No extraction returned (LLM failure)")

        print()

    print(f"{'='*80}")
    print(f"Results: {ok} OK, {bad} FAIL out of {len(rows)}")
    print(f"{'='*80}")


# ------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------


async def dry_run(pool):
    """Show tier counts and exit."""
    for tier_name, (min_rev, max_rev) in _TIERS.items():
        having = f"HAVING count(*) >= {min_rev}"
        if max_rev is not None:
            having += f" AND count(*) <= {max_rev}"

        row = await pool.fetchrow(f"""
            SELECT count(*) AS review_count FROM product_reviews
            WHERE enrichment_status = 'enriched'
              AND deep_enrichment_status = 'pending'
              AND asin IN (
                  SELECT asin FROM product_reviews
                  WHERE enrichment_status = 'enriched'
                  GROUP BY asin {having}
              )
        """)
        print(f"Tier {tier_name} ({min_rev}-{max_rev or '+'} reviews/ASIN): {row['review_count']:,} reviews pending")

    total = await pool.fetchval(
        "SELECT count(*) FROM product_reviews WHERE deep_enrichment_status = 'pending'"
    )
    enriched = await pool.fetchval(
        "SELECT count(*) FROM product_reviews WHERE deep_enrichment_status = 'enriched'"
    )
    failed = await pool.fetchval(
        "SELECT count(*) FROM product_reviews WHERE deep_enrichment_status = 'deep_failed'"
    )
    na = await pool.fetchval(
        "SELECT count(*) FROM product_reviews WHERE deep_enrichment_status = 'not_applicable'"
    )
    print(f"\nTotal: pending={total:,} | enriched={enriched:,} | deep_failed={failed:,} | not_applicable={na:,}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="Deep enrichment blast script")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers (default: 3)")
    parser.add_argument("--batch", type=int, default=30, help="Reviews claimed per worker per round (default: 30)")
    parser.add_argument("--tier", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="ASIN tier filter: 1=50+, 2=20-49, 3=10-19, all=no filter")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts before marking deep_failed")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for LLM output")
    parser.add_argument("--limit", type=int, default=0, help="Max total reviews to process (0=unlimited)")
    parser.add_argument("--validate", type=int, default=0, metavar="N",
                        help="Process N reviews, print formatted results, exit")
    parser.add_argument("--dry-run", action="store_true", help="Show tier counts and exit")
    parser.add_argument("--model", type=str, default=None, help="Model override (HuggingFace format for vllm, e.g. Qwen/Qwen3-14B)")
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "vllm"],
                        help="LLM provider: ollama (default) or vllm (continuous batching)")
    parser.add_argument("--base-url", type=str, default=None, help="Override server URL for the chosen provider")
    args = parser.parse_args()

    from atlas_brain.config import settings
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.services import llm_registry
    from atlas_brain.skills import get_skill_registry

    pool = get_db_pool()
    await pool.initialize()

    if args.dry_run:
        await dry_run(pool)
        return

    # Activate LLM
    if args.provider == "vllm":
        model = args.model or "Qwen/Qwen3-14B"
        base_url = args.base_url or _resolve_vllm_url()
        llm_registry.activate("vllm", model=model, base_url=base_url, timeout=settings.llm.ollama_timeout)
    else:
        model = args.model or settings.llm.ollama_model
        base_url = args.base_url or settings.llm.ollama_url
        llm_registry.activate("ollama", model=model, base_url=base_url, timeout=settings.llm.ollama_timeout)
    llm = llm_registry.get_active()
    if not llm:
        print("ERROR: No LLM available")
        return

    skill = get_skill_registry().get("digest/deep_extraction")
    if not skill:
        print("ERROR: deep_extraction skill not found")
        return

    # Validate mode
    if args.validate > 0:
        await validate_mode(pool, llm, skill.content, args.validate, args.tier, args.max_tokens, args.max_attempts)
        return

    # Reset stuck processing rows from prior crashed runs
    reset = await pool.execute(
        "UPDATE product_reviews SET deep_enrichment_status = 'pending' WHERE deep_enrichment_status = 'processing'"
    )
    if reset and reset != "UPDATE 0":
        print(f"Reset stuck rows: {reset}", flush=True)

    row = await pool.fetchrow(
        "SELECT count(*) FILTER (WHERE deep_enrichment_status = 'enriched') AS done, "
        "count(*) FILTER (WHERE deep_enrichment_status = 'pending') AS pending "
        "FROM product_reviews"
    )
    initial = row["done"]
    pending = row["pending"]

    print(
        f"Starting {args.workers} workers | provider={args.provider} | model={model} | "
        f"batch={args.batch} | tier={args.tier} | max_tokens={args.max_tokens} | "
        f"pending={pending:,}" + (f" | limit={args.limit}" if args.limit else ""),
        flush=True,
    )

    start = time.monotonic()
    mon = asyncio.create_task(monitor(pool, start, initial, args.tier))

    metadata_cache = MetadataCache(pool)
    per_worker_limit = args.limit // args.workers if args.limit > 0 else 0

    workers = [
        asyncio.create_task(
            worker(
                i, pool, llm, skill.content, args.batch,
                args.max_attempts, args.max_tokens, args.tier,
                metadata_cache, per_worker_limit,
            )
        )
        for i in range(args.workers)
    ]

    await asyncio.gather(*workers)
    mon.cancel()

    elapsed = time.monotonic() - start
    rate = _enriched / elapsed * 3600 if elapsed > 0 else 0
    print(
        f"\nDone in {elapsed:.0f}s | deep_enriched: {_enriched:,} | "
        f"deep_failed: {_failed:,} | rate: {rate:,.0f}/hr",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
