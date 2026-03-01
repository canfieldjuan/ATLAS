"""
Unified full-pipeline review enrichment blaster.

Each review goes through both passes in one shot:
  1. First-pass classification (complaint or praise, 256 tok)
  2. Deep extraction (32 fields, ~1500 tok)

Every review that lands in the DB is fully enriched — no waiting
for a waterfall of separate scripts.

Uses 200/ASIN cap with balanced rating sampling to avoid redundantly
processing 10k reviews for one product. Prioritizes diversity.

Runs N concurrent workers against a vLLM server. vLLM's continuous
batching handles the concurrency — workers at different pipeline
stages interleave naturally in the GPU batch.

Usage:
    python scripts/blast_full_pipeline.py                          # 20 workers, 200/ASIN cap
    python scripts/blast_full_pipeline.py --workers 30 --cap 500   # more workers, higher cap
    python scripts/blast_full_pipeline.py --dry-run                # show target counts
    python scripts/blast_full_pipeline.py --validate 5             # process 5 reviews, print results
"""

import argparse
import asyncio
import json
import logging
import os
import re
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
    port = os.getenv("VLLM_PORT", "8082")
    return f"http://{host}:{port}"


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("blast_full_pipeline")

# Counters
_first_pass_done = 0
_deep_done = 0
_failed = 0
_lock = asyncio.Lock()


# ------------------------------------------------------------------
# Deep extraction validation (copied from blast_deep_enrichment.py)
# ------------------------------------------------------------------

_REQUIRED_KEYS = {
    "sentiment_aspects": list, "feature_requests": list,
    "failure_details": (dict, type(None)), "product_comparisons": list,
    "product_name_mentioned": str, "buyer_context": dict,
    "quotable_phrases": list, "would_repurchase": (bool, type(None)),
    "external_references": list, "positive_aspects": list,
    "expertise_level": str, "frustration_threshold": str,
    "discovery_channel": str, "consideration_set": list,
    "buyer_household": str, "profession_hint": (str, type(None)),
    "budget_type": str, "use_intensity": str, "research_depth": str,
    "community_mentions": list, "consequence_severity": str,
    "replacement_behavior": str, "brand_loyalty_depth": str,
    "ecosystem_lock_in": dict, "safety_flag": dict,
    "bulk_purchase_signal": dict, "review_delay_signal": str,
    "sentiment_trajectory": str, "occasion_context": str,
    "switching_barrier": dict, "amplification_intent": dict,
    "review_sentiment_openness": dict,
}

_VALID_EXPERTISE = {"novice", "intermediate", "expert", "professional"}
_VALID_FRUSTRATION = {"low", "medium", "high"}
_VALID_DISCOVERY = {"amazon_organic", "youtube", "reddit", "friend", "amazon_choice", "unknown"}
_VALID_HOUSEHOLD = {"single", "family", "professional", "gift", "bulk"}
_VALID_BUDGET = {"budget_constrained", "value_seeker", "premium_willing", "unknown"}
_VALID_INTENSITY = {"light", "moderate", "heavy"}
_VALID_RESEARCH = {"impulse", "light", "moderate", "deep"}
_VALID_CONSEQUENCE = {"none", "positive_impact", "inconvenience", "workflow_impact", "financial_loss", "safety_concern"}
_VALID_REPLACEMENT = {"returned", "replaced_same", "switched_brand", "switched_to", "avoided", "kept_broken", "kept_using", "repurchased", "unknown"}
_VALID_SENTIMENT = {"positive", "negative", "mixed", "neutral"}
_VALID_PRICE_SENTIMENT = {"expensive", "fair", "cheap", "not_mentioned"}
_VALID_LOYALTY = {"first_time", "occasional", "loyal", "long_term_loyal"}
_VALID_DELAY = {"immediate", "days", "weeks", "months", "unknown"}
_VALID_TRAJECTORY = {"always_negative", "degraded", "mixed_then_negative", "mixed_then_positive", "improved", "always_positive", "unknown"}
_VALID_OCCASION = {"none", "gift", "replacement", "upgrade", "first_in_category", "seasonal", "event", "professional_use"}
_VALID_ECOSYSTEM_LEVEL = {"free", "partially", "fully"}
_VALID_BARRIER_LEVEL = {"none", "low", "medium", "high"}
_VALID_AMPLIFICATION = {"quiet", "private", "social"}
_VALID_BULK_TYPE = {"single", "multi"}


def _validate_deep(data: dict) -> bool:
    """Validate all 32 deep extraction fields."""
    for key, expected in _REQUIRED_KEYS.items():
        if key not in data:
            return False
        val = data[key]
        if isinstance(expected, tuple):
            if not isinstance(val, expected):
                return False
        elif not isinstance(val, expected):
            return False

    bc = data.get("buyer_context")
    if isinstance(bc, dict):
        for field in ("use_case", "buyer_type", "price_sentiment"):
            if field not in bc:
                return False
        ps = bc.get("price_sentiment")
        if ps is not None and ps not in _VALID_PRICE_SENTIMENT:
            return False

    aspects = data.get("sentiment_aspects")
    if isinstance(aspects, list):
        for a in aspects:
            if isinstance(a, dict) and a.get("sentiment") not in _VALID_SENTIMENT:
                return False

    if data.get("expertise_level") not in _VALID_EXPERTISE: return False
    if data.get("frustration_threshold") not in _VALID_FRUSTRATION: return False
    if data.get("discovery_channel") not in _VALID_DISCOVERY: return False
    if data.get("buyer_household") not in _VALID_HOUSEHOLD: return False
    if data.get("budget_type") not in _VALID_BUDGET: return False
    if data.get("use_intensity") not in _VALID_INTENSITY: return False
    if data.get("research_depth") not in _VALID_RESEARCH: return False
    if data.get("consequence_severity") not in _VALID_CONSEQUENCE: return False
    if data.get("replacement_behavior") not in _VALID_REPLACEMENT: return False
    if data.get("brand_loyalty_depth") not in _VALID_LOYALTY: return False
    if data.get("review_delay_signal") not in _VALID_DELAY: return False
    if data.get("sentiment_trajectory") not in _VALID_TRAJECTORY: return False
    if data.get("occasion_context") not in _VALID_OCCASION: return False

    eco = data.get("ecosystem_lock_in", {})
    if not isinstance(eco, dict) or "level" not in eco or "ecosystem" not in eco: return False
    if eco["level"] not in _VALID_ECOSYSTEM_LEVEL: return False
    safety = data.get("safety_flag", {})
    if not isinstance(safety, dict) or "flagged" not in safety or "description" not in safety: return False
    bulk = data.get("bulk_purchase_signal", {})
    if not isinstance(bulk, dict) or "type" not in bulk or "estimated_qty" not in bulk: return False
    if bulk["type"] not in _VALID_BULK_TYPE: return False
    barrier = data.get("switching_barrier", {})
    if not isinstance(barrier, dict) or "level" not in barrier or "reason" not in barrier: return False
    if barrier["level"] not in _VALID_BARRIER_LEVEL: return False
    amp = data.get("amplification_intent", {})
    if not isinstance(amp, dict) or "intent" not in amp or "context" not in amp: return False
    if amp["intent"] not in _VALID_AMPLIFICATION: return False
    openness = data.get("review_sentiment_openness", {})
    if not isinstance(openness, dict) or "open" not in openness or "condition" not in openness: return False

    return True


# ------------------------------------------------------------------
# LLM helpers
# ------------------------------------------------------------------

def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


def _classify_single(llm, skill_content: str, row) -> dict | None:
    """First-pass classification (complaint or praise)."""
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
            max_tokens=256, temperature=0.1,
        )
        text = _clean(result.get("response", ""))
        if not text:
            return None
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "root_cause" in parsed:
            return parsed
    except Exception:
        logger.debug("First-pass classify failed", exc_info=True)
    return None


def _extract_deep(llm, skill_content: str, row, title: str, brand: str,
                  first_pass: dict, max_tokens: int) -> dict | None:
    """Deep extraction (32 fields) using first-pass results as context."""
    from atlas_brain.services.protocols import Message
    from atlas_brain.pipelines.llm import clean_llm_output

    payload = json.dumps({
        "review_text": (row["review_text"] or "")[:3000],
        "summary": row["summary"] or "",
        "rating": float(row["rating"]),
        "product_name": title or row["asin"],
        "brand": brand or "",
        "root_cause": first_pass.get("root_cause", ""),
        "severity": first_pass.get("severity") or "",
        "pain_score": float(first_pass.get("pain_score", 0) or 0),
    })
    try:
        result = llm.chat(
            messages=[
                Message(role="system", content=skill_content),
                Message(role="user", content=payload),
            ],
            max_tokens=max_tokens, temperature=0.1,
        )
        text = clean_llm_output(result.get("response", ""))
        if not text:
            return None
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        logger.debug("Deep extraction failed", exc_info=True)
    return None


# ------------------------------------------------------------------
# Metadata cache
# ------------------------------------------------------------------

class MetadataCache:
    def __init__(self, pool, max_size: int = 2000):
        self._pool = pool
        self._cache: dict[str, tuple] = {}
        self._max_size = max_size

    async def get(self, asin: str) -> tuple[str, str]:
        if asin in self._cache:
            return self._cache[asin]
        row = await self._pool.fetchrow(
            "SELECT title, brand FROM product_metadata WHERE asin = $1", asin,
        )
        val = (row["title"] or "", row["brand"] or "") if row else ("", "")
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[asin] = val
        return val


# ------------------------------------------------------------------
# Worker: full pipeline per review
# ------------------------------------------------------------------

async def worker(
    worker_id: int,
    pool,
    llm,
    classify_skills: dict[str, str],  # {"complaint": content, "praise": content}
    deep_skill_content: str,
    batch_size: int,
    max_attempts: int,
    deep_max_tokens: int,
    metadata_cache: MetadataCache,
    target_asins: set[str] | None,
):
    global _first_pass_done, _deep_done, _failed

    while True:
        # Claim reviews that need BOTH passes: not yet first-pass enriched
        # OR already first-pass enriched but not yet deep enriched.
        # We handle both cases: the review might already have first-pass from
        # a previous run, or it might be completely fresh.
        rows = None
        for attempt in range(4):
            try:
                async with pool.transaction() as conn:
                    if target_asins is not None:
                        rows = await conn.fetch(
                            """
                            UPDATE product_reviews
                            SET deep_enrichment_status = 'processing'
                            WHERE id IN (
                                SELECT pr.id FROM product_reviews pr
                                WHERE pr.deep_enrichment_status = 'pending'
                                  AND pr.deep_enrichment_attempts < $1
                                  AND pr.asin = ANY($2::text[])
                                ORDER BY pr.imported_at ASC
                                LIMIT $3
                                FOR UPDATE SKIP LOCKED
                            )
                            RETURNING id, asin, rating, summary, review_text,
                                      root_cause, severity, pain_score,
                                      hardware_category, issue_types,
                                      enrichment_status, enrichment_attempts,
                                      deep_enrichment_attempts
                            """,
                            max_attempts,
                            list(target_asins),
                            batch_size,
                            timeout=120,
                        )
                    else:
                        rows = await conn.fetch(
                            """
                            UPDATE product_reviews
                            SET deep_enrichment_status = 'processing'
                            WHERE id IN (
                                SELECT pr.id FROM product_reviews pr
                                WHERE pr.deep_enrichment_status = 'pending'
                                  AND pr.deep_enrichment_attempts < $1
                                ORDER BY pr.imported_at ASC
                                LIMIT $2
                                FOR UPDATE SKIP LOCKED
                            )
                            RETURNING id, asin, rating, summary, review_text,
                                      root_cause, severity, pain_score,
                                      hardware_category, issue_types,
                                      enrichment_status, enrichment_attempts,
                                      deep_enrichment_attempts
                            """,
                            max_attempts,
                            batch_size,
                            timeout=120,
                        )
                break
            except (TimeoutError, asyncio.TimeoutError):
                wait = 2 ** attempt
                logger.warning("Worker %d: claim query timed out (attempt %d/4), retrying in %ds", worker_id, attempt + 1, wait)
                await asyncio.sleep(wait)
            except Exception as exc:
                if "QueryCanceled" in type(exc).__name__:
                    wait = 2 ** attempt
                    logger.warning("Worker %d: claim query cancelled (attempt %d/4), retrying in %ds", worker_id, attempt + 1, wait)
                    await asyncio.sleep(wait)
                else:
                    raise

        if not rows:
            return

        for row in rows:
            review_id = row["id"]
            try:
                await _process_review(
                    pool, llm, classify_skills, deep_skill_content,
                    deep_max_tokens, max_attempts, metadata_cache, row,
                )
            except Exception:
                logger.exception("Failed review %s", review_id)
                try:
                    await pool.execute(
                        """UPDATE product_reviews
                           SET deep_enrichment_status = 'pending',
                               deep_enrichment_attempts = deep_enrichment_attempts + 1
                           WHERE id = $1""",
                        review_id,
                    )
                except Exception:
                    pass
                async with _lock:
                    _failed += 1


async def _process_review(
    pool, llm, classify_skills, deep_skill_content,
    deep_max_tokens, max_attempts, metadata_cache, row,
):
    global _first_pass_done, _deep_done, _failed
    review_id = row["id"]
    rating = float(row["rating"])

    # ------ Step 1: First-pass classification ------
    # Skip if already enriched from a previous run
    if row["enrichment_status"] == "enriched" and row["root_cause"]:
        first_pass = {
            "root_cause": row["root_cause"],
            "severity": row["severity"],
            "pain_score": float(row["pain_score"]) if row["pain_score"] else 0.0,
            "specific_complaint": None,
            "time_to_failure": None,
            "workaround_found": None,
            "workaround_text": None,
            "alternative_mentioned": None,
            "alternative_asin": None,
            "alternative_name": None,
            "actionable_for_manufacturing": None,
            "manufacturing_suggestion": None,
        }
    else:
        skill_key = "complaint" if rating <= 3.0 else "praise"
        first_pass = await asyncio.to_thread(
            _classify_single, llm, classify_skills[skill_key], row,
        )
        if not first_pass or "root_cause" not in first_pass:
            # First-pass failed — bump attempts, put back to pending
            new_attempts = row["deep_enrichment_attempts"] + 1
            status = "deep_failed" if new_attempts >= max_attempts else "pending"
            await pool.execute(
                f"UPDATE product_reviews SET deep_enrichment_status = $1, "
                f"deep_enrichment_attempts = $2 WHERE id = $3",
                status, new_attempts, review_id,
            )
            async with _lock:
                _failed += 1
            return

        # Write first-pass results
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
            first_pass.get("root_cause"),
            first_pass.get("specific_complaint"),
            first_pass.get("severity"),
            first_pass.get("pain_score"),
            first_pass.get("time_to_failure"),
            first_pass.get("workaround_found"),
            first_pass.get("workaround_text"),
            first_pass.get("alternative_mentioned"),
            first_pass.get("alternative_asin"),
            first_pass.get("alternative_name"),
            first_pass.get("actionable_for_manufacturing"),
            first_pass.get("manufacturing_suggestion"),
            row["enrichment_attempts"] + 1,
            datetime.now(timezone.utc),
            review_id,
        )
        async with _lock:
            _first_pass_done += 1

    # ------ Step 2: Deep extraction ------
    title, brand = await metadata_cache.get(row["asin"])
    extraction = await asyncio.to_thread(
        _extract_deep, llm, deep_skill_content, row, title, brand,
        first_pass, deep_max_tokens,
    )

    if extraction and _validate_deep(extraction):
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
            review_id,
        )
        async with _lock:
            _deep_done += 1
    else:
        new_attempts = row["deep_enrichment_attempts"] + 1
        status = "deep_failed" if new_attempts >= max_attempts else "pending"
        await pool.execute(
            "UPDATE product_reviews SET deep_enrichment_status = $1, "
            "deep_enrichment_attempts = $2 WHERE id = $3",
            status, new_attempts, review_id,
        )
        if new_attempts >= max_attempts:
            async with _lock:
                _failed += 1


# ------------------------------------------------------------------
# ASIN cap: select balanced sample of ASINs with rating diversity
# ------------------------------------------------------------------

async def build_target_asins(pool, cap_per_asin: int, categories: list[str] | None) -> set[str]:
    """
    For each ASIN, mark up to `cap_per_asin` reviews as deep_enrichment_status='pending'
    using balanced rating sampling. Returns the set of target ASINs.

    Strategy: for each ASIN, take cap/5 reviews from each star bucket (1-5)
    to ensure rating diversity.
    """
    cat_filter = ""
    params: list[Any] = []
    if categories:
        cat_filter = "AND source_category = ANY($1::text[])"
        params.append(categories)

    # Get all ASINs with their review counts
    idx = len(params) + 1
    asins = await pool.fetch(
        f"""
        SELECT asin, COUNT(*) AS cnt
        FROM product_reviews
        WHERE deep_enrichment_status = 'pending'
          {cat_filter}
        GROUP BY asin
        ORDER BY cnt DESC
        """,
        *params,
    )

    target_asins = set()
    per_bucket = max(cap_per_asin // 5, 1)
    capped_count = 0
    uncapped_count = 0

    for row in asins:
        asin = row["asin"]
        cnt = row["cnt"]
        target_asins.add(asin)

        if cnt <= cap_per_asin:
            uncapped_count += cnt
            continue

        # This ASIN exceeds cap — mark excess as 'not_applicable'
        # Keep a balanced sample across rating buckets
        for star in range(1, 6):
            low = float(star) - 0.5 if star > 1 else 0.0
            high = float(star) + 0.49

            # Keep per_bucket reviews in this star range, mark rest not_applicable
            await pool.execute(
                f"""
                UPDATE product_reviews
                SET deep_enrichment_status = 'not_applicable'
                WHERE id IN (
                    SELECT id FROM product_reviews
                    WHERE asin = $1
                      AND deep_enrichment_status = 'pending'
                      AND rating >= $2 AND rating <= $3
                    ORDER BY imported_at ASC
                    OFFSET $4
                )
                """,
                asin, low, high, per_bucket,
            )

        capped_count += cap_per_asin

    total_target = await pool.fetchval(
        "SELECT COUNT(*) FROM product_reviews WHERE deep_enrichment_status = 'pending'"
    )

    print(
        f"Target ASINs: {len(target_asins):,} | "
        f"Reviews after capping: {total_target:,} | "
        f"Marked not_applicable: excess reviews beyond {cap_per_asin}/ASIN",
        flush=True,
    )

    return target_asins


# ------------------------------------------------------------------
# Progress monitor
# ------------------------------------------------------------------

async def monitor(pool, start_time: float, initial_first: int, initial_deep: int):
    while True:
        await asyncio.sleep(30)
        row = await pool.fetchrow(
            "SELECT "
            "count(*) FILTER (WHERE enrichment_status = 'enriched') AS first_done, "
            "count(*) FILTER (WHERE deep_enrichment_status = 'enriched') AS deep_done, "
            "count(*) FILTER (WHERE deep_enrichment_status = 'pending') AS pending, "
            "count(*) FILTER (WHERE deep_enrichment_status = 'processing') AS processing, "
            "count(*) FILTER (WHERE deep_enrichment_status = 'deep_failed') AS deep_failed "
            "FROM product_reviews"
        )
        elapsed = time.monotonic() - start_time
        session_deep = row["deep_done"] - initial_deep
        session_first = row["first_done"] - initial_first
        rate = session_deep / elapsed * 3600 if elapsed > 0 else 0
        pending = row["pending"] + row["processing"]
        eta = pending / rate if rate > 0 else 0
        print(
            f"[{elapsed:6.0f}s] first_pass: +{session_first:,} | "
            f"deep: +{session_deep:,} | pending: {pending:,} | "
            f"failed: {row['deep_failed']:,} | "
            f"rate: {rate:,.0f} full/hr | ETA: {eta:.1f}h",
            flush=True,
        )


# ------------------------------------------------------------------
# Dry run
# ------------------------------------------------------------------

async def dry_run(pool, cap: int, categories: list[str] | None):
    cat_filter = ""
    params: list[Any] = []
    if categories:
        cat_filter = "AND source_category = ANY($1::text[])"
        params.append(categories)

    rows = await pool.fetch(
        f"""
        SELECT bucket, COUNT(*) AS asin_count, SUM(reviews) AS total_reviews FROM (
            SELECT asin, COUNT(*) AS reviews,
                   CASE
                       WHEN COUNT(*) > 10000 THEN '10k+'
                       WHEN COUNT(*) > 5000 THEN '5k-10k'
                       WHEN COUNT(*) > 2000 THEN '2k-5k'
                       WHEN COUNT(*) > 1000 THEN '1k-2k'
                       WHEN COUNT(*) > 500 THEN '500-1k'
                       WHEN COUNT(*) > 200 THEN '200-500'
                       ELSE 'under 200'
                   END AS bucket
            FROM product_reviews
            WHERE deep_enrichment_status = 'pending' {cat_filter}
            GROUP BY asin
        ) sub GROUP BY bucket ORDER BY MIN(reviews) DESC
        """,
        *params,
    )
    total = sum(r["total_reviews"] for r in rows)
    capped = sum(min(r["total_reviews"], r["asin_count"] * cap) for r in rows)

    print(f"\n{'Bucket':<15} {'ASINs':>7} {'Total Reviews':>15}")
    print("-" * 40)
    for r in rows:
        print(f"{r['bucket']:<15} {r['asin_count']:>7} {r['total_reviews']:>15,}")
    print(f"\nTotal reviews pending: {total:,}")
    print(f"After {cap}/ASIN cap:  ~{capped:,}")
    print(f"Est. time at ~3.5k full/hr: {capped / 3500:.0f}h ({capped / 3500 / 24:.1f} days)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Full pipeline enrichment blast")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent workers (default: 20)")
    parser.add_argument("--batch", type=int, default=10, help="Reviews per worker per round (default: 10)")
    parser.add_argument("--cap", type=int, default=200, help="Max reviews per ASIN (default: 200, 0=no cap)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--deep-max-tokens", type=int, default=2048)
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category filter (e.g. 'cell phones,electronics')")
    parser.add_argument("--validate", type=int, default=0, metavar="N",
                        help="Process N reviews and print results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cap", action="store_true", help="Disable ASIN cap (process all reviews)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "ollama"])
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    from atlas_brain.config import settings
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.services import llm_registry
    from atlas_brain.skills import get_skill_registry

    pool = get_db_pool()
    await pool.initialize()

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    if args.dry_run:
        await dry_run(pool, args.cap, categories)
        return

    # Activate LLM
    if args.provider == "vllm":
        model = args.model or "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"
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

    # Load skills
    skill_reg = get_skill_registry()
    complaint_skill = skill_reg.get("digest/complaint_classification")
    praise_skill = skill_reg.get("digest/praise_classification")
    deep_skill = skill_reg.get("digest/deep_extraction")
    if not complaint_skill or not praise_skill or not deep_skill:
        print("ERROR: Missing required skills (complaint_classification, praise_classification, deep_extraction)")
        return

    classify_skills = {
        "complaint": complaint_skill.content,
        "praise": praise_skill.content,
    }

    # Validate mode
    if args.validate > 0:
        metadata_cache = MetadataCache(pool)
        rows = await pool.fetch(
            "SELECT pr.id, pr.asin, pr.rating, pr.summary, pr.review_text, "
            "pr.root_cause, pr.severity, pr.pain_score, "
            "pr.hardware_category, pr.issue_types, "
            "pr.enrichment_status, pr.enrichment_attempts, "
            "pr.deep_enrichment_attempts "
            "FROM product_reviews pr "
            "WHERE pr.deep_enrichment_status = 'pending' "
            "ORDER BY pr.imported_at ASC LIMIT $1",
            args.validate,
        )
        print(f"\n{'='*80}\nVALIDATING {len(rows)} REVIEWS (full pipeline)\n{'='*80}\n")
        ok, fail = 0, 0
        for i, row in enumerate(rows, 1):
            rating = float(row["rating"])
            skill_key = "complaint" if rating <= 3.0 else "praise"

            # Step 1
            fp = await asyncio.to_thread(_classify_single, llm, classify_skills[skill_key], row)
            fp_ok = fp and "root_cause" in fp

            # Step 2
            de = None
            if fp_ok:
                title, brand = await metadata_cache.get(row["asin"])
                de = await asyncio.to_thread(
                    _extract_deep, llm, deep_skill.content, row, title, brand, fp, args.deep_max_tokens
                )
            de_ok = de and _validate_deep(de)

            status = "OK" if (fp_ok and de_ok) else "FAIL"
            if fp_ok and de_ok:
                ok += 1
            else:
                fail += 1

            print(f"--- Review {i}/{len(rows)} [{status}] ---")
            print(f"  Rating: {rating} | ASIN: {row['asin']}")
            print(f"  Summary: {(row['summary'] or '')[:80]}")
            if fp_ok:
                print(f"  First-pass: root={fp.get('root_cause')} severity={fp.get('severity')} pain={fp.get('pain_score')}")
            else:
                print(f"  First-pass: FAILED")
            if de_ok:
                print(f"  Deep: {len(de)} keys | loyalty={de.get('brand_loyalty_depth')} trajectory={de.get('sentiment_trajectory')}")
                aspects = de.get("sentiment_aspects", [])
                print(f"  Aspects: {[a.get('aspect') for a in aspects[:4] if isinstance(a, dict)]}")
                comps = de.get("product_comparisons", [])
                if comps:
                    print(f"  Comparisons: {[c.get('product_name') for c in comps if isinstance(c, dict)]}")
                quotes = de.get("quotable_phrases", [])
                if quotes:
                    print(f"  Quotes: {quotes[:2]}")
            else:
                print(f"  Deep: FAILED")
            print()

        print(f"{'='*80}\nResults: {ok} OK, {fail} FAIL\n{'='*80}")
        return

    # Reset stuck rows
    reset = await pool.execute(
        "UPDATE product_reviews SET deep_enrichment_status = 'pending' "
        "WHERE deep_enrichment_status = 'processing'"
    )
    if reset and reset != "UPDATE 0":
        print(f"Reset stuck: {reset}", flush=True)

    # Apply ASIN cap
    target_asins = None
    if args.cap > 0 and not args.no_cap:
        target_asins = await build_target_asins(pool, args.cap, categories)

    # Snapshot
    row = await pool.fetchrow(
        "SELECT "
        "count(*) FILTER (WHERE enrichment_status = 'enriched') AS first_done, "
        "count(*) FILTER (WHERE deep_enrichment_status = 'enriched') AS deep_done, "
        "count(*) FILTER (WHERE deep_enrichment_status = 'pending') AS pending "
        "FROM product_reviews"
    )
    initial_first = row["first_done"]
    initial_deep = row["deep_done"]
    pending = row["pending"]

    print(
        f"Starting {args.workers} workers | provider={args.provider} | model={model} | "
        f"batch={args.batch} | cap={args.cap}/ASIN | pending: {pending:,}",
        flush=True,
    )

    metadata_cache = MetadataCache(pool)
    start = time.monotonic()
    mon = asyncio.create_task(monitor(pool, start, initial_first, initial_deep))

    workers = [
        asyncio.create_task(
            worker(i, pool, llm, classify_skills, deep_skill.content,
                   args.batch, args.max_attempts, args.deep_max_tokens,
                   metadata_cache, target_asins)
        )
        for i in range(args.workers)
    ]

    await asyncio.gather(*workers)
    mon.cancel()

    elapsed = time.monotonic() - start
    print(
        f"\nDone in {elapsed:.0f}s | first_pass: {_first_pass_done:,} | "
        f"deep: {_deep_done:,} | failed: {_failed:,}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
