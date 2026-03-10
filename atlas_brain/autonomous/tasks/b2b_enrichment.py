"""
B2B review enrichment: extract churn signals from pending reviews via LLM
using the b2b_churn_extraction skill.

Single-pass enrichment (one LLM call per review). Polls b2b_reviews WHERE
enrichment_status = 'pending', calls LLM, stores result in enrichment JSONB
column, sets status to 'enriched'.

Runs on an interval (default 5 min). Returns _skip_synthesis so the
runner does not double-synthesize.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...services.scraping.sources import VERIFIED_SOURCES as _VERIFIED_SOURCES

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


async def _notify_high_urgency(
    vendor_name: str,
    reviewer_company: str,
    urgency: float,
    pain_category: str,
    intent_to_leave: bool,
) -> None:
    """Send ntfy push when a newly enriched review exceeds the urgency threshold."""
    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    company_part = f" at {reviewer_company}" if reviewer_company else ""
    intent_part = " | Intent to leave" if intent_to_leave else ""
    pain_part = f" | Pain: {pain_category}" if pain_category else ""

    message = (
        f"Urgency {urgency:.0f}/10{company_part}\n"
        f"Vendor: {vendor_name}{pain_part}{intent_part}"
    )

    headers: dict[str, str] = {
        "Title": f"High-Urgency Signal: {vendor_name}",
        "Priority": "high",
        "Tags": "rotating_light,b2b,churn",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("ntfy high-urgency alert sent for %s (urgency=%s)", vendor_name, urgency)
    except Exception as exc:
        logger.warning("ntfy high-urgency alert failed for %s: %s", vendor_name, exc)


async def enrich_batch(batch_id: str) -> dict[str, Any]:
    """Enrich all pending reviews from a specific import batch immediately.

    Called inline after scrape insertion so reviews are enriched on arrival
    rather than waiting for the scheduler.
    """
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"skipped": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"skipped": "DB not ready"}

    max_attempts = cfg.enrichment_max_attempts

    rows = await pool.fetch(
        """
        WITH batch AS (
            SELECT id
            FROM b2b_reviews
            WHERE import_batch_id = $1
              AND enrichment_status = 'pending'
              AND enrichment_attempts < $2
            FOR UPDATE SKIP LOCKED
        )
        UPDATE b2b_reviews r
        SET enrichment_status = 'enriching'
        FROM batch
        WHERE r.id = batch.id
        RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                  r.source, r.raw_metadata,
                  r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                  r.reviewer_title, r.reviewer_company, r.company_size_raw,
                  r.reviewer_industry, r.enrichment_attempts
        """,
        batch_id,
        max_attempts,
    )

    if not rows:
        return {"total": 0, "enriched": 0, "failed": 0}

    return await _enrich_rows(rows, cfg, pool)


async def _enrich_rows(rows, cfg, pool) -> dict[str, Any]:
    """Enrich a list of claimed rows concurrently."""
    max_attempts = cfg.enrichment_max_attempts
    enriched = 0
    failed = 0

    sem = asyncio.Semaphore(30)

    async def _bounded_enrich(row):
        async with sem:
            return await _enrich_single(pool, row, max_attempts, cfg.enrichment_local_only,
                                        cfg.enrichment_max_tokens, cfg.review_truncate_length)

    results = await asyncio.gather(
        *[_bounded_enrich(row) for row in rows],
        return_exceptions=True,
    )

    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            logger.error("Unexpected enrichment error for %s: %s", row["id"], result, exc_info=result)
            if (row["enrichment_attempts"] + 1) >= max_attempts:
                failed += 1
        elif result:
            enriched += 1
        elif (row["enrichment_attempts"] + 1) >= max_attempts:
            failed += 1

    # Count how many were triaged out in this batch
    no_signal = await pool.fetchval(
        "SELECT count(*) FROM b2b_reviews WHERE id = ANY($1::uuid[]) AND enrichment_status = 'no_signal'",
        [row["id"] for row in rows],
    )

    logger.info(
        "B2B enrichment: %d enriched, %d no_signal, %d failed (of %d)",
        enriched, no_signal or 0, failed, len(rows),
    )

    return {
        "total": len(rows),
        "enriched": enriched,
        "no_signal": no_signal or 0,
        "failed": failed,
    }


async def _queue_version_upgrades(pool) -> int:
    """Reset enrichment_status to 'pending' for reviews scraped with outdated parser versions.

    Compares each review's parser_version against the currently registered
    parser version.  Reviews with older versions are re-queued for enrichment.
    Returns the number of reviews re-queued.
    """
    try:
        from ...services.scraping.parsers import get_all_parsers

        parsers = get_all_parsers()
        if not parsers:
            return 0

        total_requeued = 0
        for source_name, parser in parsers.items():
            current_version = getattr(parser, "version", None)
            if not current_version:
                continue

            # Find enriched reviews with an older parser version
            count = await pool.fetchval(
                """
                UPDATE b2b_reviews
                SET enrichment_status = 'pending',
                    enrichment_attempts = 0
                WHERE source = $1
                  AND parser_version IS NOT NULL
                  AND parser_version != $2
                  AND enrichment_status IN ('enriched', 'no_signal')
                RETURNING COUNT(*)
                """,
                source_name,
                current_version,
            )
            if count and count > 0:
                logger.info(
                    "Re-queued %d %s reviews for re-enrichment (parser %s -> %s)",
                    count, source_name, "old", current_version,
                )
                total_requeued += count

        return total_requeued
    except Exception:
        logger.debug("Version upgrade check skipped", exc_info=True)
        return 0


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich pending B2B reviews (fallback for anything missed)."""
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Auto re-process reviews scraped with outdated parser versions
    requeued = await _queue_version_upgrades(pool)

    max_batch = min(cfg.enrichment_max_per_batch, 500)
    max_attempts = cfg.enrichment_max_attempts

    total_enriched = 0
    total_failed = 0
    total_no_signal = 0
    rounds = 0

    while True:
        rows = await pool.fetch(
            """
            WITH batch AS (
                SELECT id
                FROM b2b_reviews
                WHERE enrichment_status = 'pending'
                  AND enrichment_attempts < $1
                ORDER BY imported_at ASC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            UPDATE b2b_reviews r
            SET enrichment_status = 'enriching'
            FROM batch
            WHERE r.id = batch.id
            RETURNING r.id, r.vendor_name, r.product_name, r.product_category,
                      r.source, r.raw_metadata,
                      r.rating, r.rating_max, r.summary, r.review_text, r.pros, r.cons,
                      r.reviewer_title, r.reviewer_company, r.company_size_raw,
                      r.reviewer_industry, r.enrichment_attempts
            """,
            max_attempts,
            max_batch,
        )

        if not rows:
            break

        result = await _enrich_rows(rows, cfg, pool)
        total_enriched += result.get("enriched", 0)
        batch_failed = result.get("failed", 0)
        total_failed += batch_failed
        total_no_signal += result.get("no_signal", 0)
        rounds += 1

        # If most of the batch failed, vLLM is likely overwhelmed — stop the loop
        if batch_failed > len(rows) * 0.5:
            logger.warning("B2B enrichment: >50%% failures in batch (%d/%d), stopping loop",
                           batch_failed, len(rows))
            break

        await asyncio.sleep(2)  # breathing room between batches

    if rounds == 0:
        return {"_skip_synthesis": "No B2B reviews to enrich"}

    result = {
        "enriched": total_enriched,
        "failed": total_failed,
        "no_signal": total_no_signal,
        "rounds": rounds,
        "_skip_synthesis": "B2B enrichment complete",
    }
    if requeued:
        result["version_upgrade_requeued"] = requeued
    return result


_MIN_REVIEW_TEXT_LENGTH = 80  # Skip LLM calls for reviews shorter than this

# Verified review platforms -- every review gets full extraction (skip triage).


async def _enrich_single(pool, row, max_attempts: int, local_only: bool,
                         max_tokens: int, truncate_length: int = 3000) -> bool:
    """Enrich a single B2B review with churn signals. Returns True on success."""
    review_id = row["id"]

    # Skip reviews with insufficient text — title-only scrapes can't yield 47 fields
    review_text = row.get("review_text") or ""
    if len(review_text) < _MIN_REVIEW_TEXT_LENGTH:
        await pool.execute(
            "UPDATE b2b_reviews SET enrichment_status = 'not_applicable' WHERE id = $1",
            review_id,
        )
        return False

    source = (row.get("source") or "").lower().strip()
    is_verified = source in _VERIFIED_SOURCES

    try:
        if is_verified:
            # Verified sources: skip triage, go straight to full extraction.
            # Every review on these platforms has signal worth extracting.
            logger.debug("Verified source %s for %s -- skipping triage", source, review_id)
        else:
            # Stage 1: Fast triage -- is this review worth full extraction?
            triage, triage_model_id = await asyncio.wait_for(
                _triage_review(row, local_only),
                timeout=30,
            )

            if triage is None:
                # Triage failed (no LLM, skill missing, parse error).
                # Don't fall through to full extraction -- that will likely fail too.
                # Leave as pending for retry on next cycle.
                logger.debug("Triage returned None for %s, deferring to next cycle", review_id)
                await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
                return False

            if not triage.get("signal", True):
                # No churn signal -- skip full extraction
                await pool.execute(
                    """
                    UPDATE b2b_reviews
                    SET enrichment_status = 'no_signal',
                        enrichment_attempts = enrichment_attempts + 1,
                        enrichment = $1,
                        enrichment_model = $3
                    WHERE id = $2
                    """,
                    json.dumps({"triage": triage}),
                    review_id,
                    triage_model_id,
                )
                return False

        # Stage 2: Full 47-field extraction (only for reviews that passed triage)
        result, model_id = await asyncio.wait_for(
            _classify_review_async(row, local_only, max_tokens, truncate_length),
            timeout=120,
        )

        if result and _validate_enrichment(result):
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment = $1,
                    enrichment_status = 'enriched',
                    enrichment_attempts = enrichment_attempts + 1,
                    enriched_at = $2,
                    enrichment_model = $4
                WHERE id = $3
                """,
                json.dumps(result),
                datetime.now(timezone.utc),
                review_id,
                model_id,
            )

            # Fire ntfy notification for high-urgency signals (must never
            # break enrichment -- wrapped in its own try/except)
            try:
                urgency = result.get("urgency_score", 0)
                threshold = settings.b2b_churn.high_churn_urgency_threshold
                if urgency >= threshold:
                    signals = result.get("churn_signals", {})
                    await _notify_high_urgency(
                        vendor_name=row["vendor_name"],
                        reviewer_company=row.get("reviewer_company") or "",
                        urgency=urgency,
                        pain_category=result.get("pain_category", ""),
                        intent_to_leave=bool(signals.get("intent_to_leave")),
                    )
            except Exception:
                logger.warning("ntfy notification failed for review %s, enrichment preserved", review_id)

            return True
        else:
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return False

    except Exception:
        logger.exception("Failed to enrich B2B review %s", review_id)
        try:
            # Reset from 'enriching' back to 'pending' (or 'failed' if exhausted)
            new_status = "failed" if (row["enrichment_attempts"] + 1) >= max_attempts else "pending"
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment_attempts = enrichment_attempts + 1,
                    enrichment_status = $1
                WHERE id = $2
                """,
                new_status, review_id,
            )
        except Exception:
            pass
        return False


def _smart_truncate(text: str, max_len: int = 3000) -> str:
    """Truncate preserving both beginning and end of review text.

    Churn signals often appear at the end ("I'm switching to X next quarter"),
    so naive head-only truncation loses them.
    """
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 15
    return text[:half] + "\n[...truncated...]\n" + text[-half:]


def _build_classify_payload(row, truncate_length: int = 3000) -> dict[str, Any]:
    """Build the JSON payload for the churn extraction skill."""
    review_text = _smart_truncate(row["review_text"] or "", max_len=truncate_length)

    raw_meta = row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}

    return {
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"] or "",
        "product_category": row["product_category"] or "",
        "source_name": row.get("source") or "",
        "source_weight": raw_meta.get("source_weight", 0.7),
        "source_type": raw_meta.get("source_type", "unknown"),
        "rating": float(row["rating"]) if row["rating"] is not None else None,
        "rating_max": int(row["rating_max"]),
        "summary": row["summary"] or "",
        "review_text": review_text,
        "pros": row["pros"] or "",
        "cons": row["cons"] or "",
        "reviewer_title": row["reviewer_title"] or "",
        "reviewer_company": row["reviewer_company"] or "",
        "company_size_raw": row["company_size_raw"] or "",
        "reviewer_industry": row["reviewer_industry"] or "",
    }


async def _triage_review(row, local_only: bool) -> tuple[dict[str, Any] | None, str | None]:
    """Fast LLM triage: does this review contain churn signals worth extracting?

    Returns ({"signal": bool, "confidence": int, "reason": str}, model_id) or (None, None) on error.
    ~50 tokens output -- runs 5-10x faster than full extraction.
    """
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output

    skill = get_skill_registry().get("digest/b2b_churn_triage")
    if not skill:
        # Triage skill missing -- fall through to full extraction
        return None, None

    llm = get_pipeline_llm(prefer_cloud=False, try_openrouter=False, auto_activate_ollama=True)
    if llm is None:
        return None, None

    model_id = getattr(llm, "model_id", None) or getattr(llm, "model", None)

    # Compact payload -- triage doesn't need all fields
    review_text = (row.get("review_text") or "")[:1500]
    payload = json.dumps({
        "vendor_name": row["vendor_name"],
        "source": row.get("source") or "",
        "rating": float(row["rating"]) if row.get("rating") is not None else None,
        "summary": row.get("summary") or "",
        "review_text": review_text,
        "pros": (row.get("pros") or "")[:300],
        "cons": (row.get("cons") or "")[:300],
    })

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=payload),
    ]

    try:
        if hasattr(llm, "chat_async"):
            text = await llm.chat_async(messages=messages, max_tokens=64, temperature=0.0)
        else:
            result = await asyncio.to_thread(
                llm.chat, messages=messages, max_tokens=64, temperature=0.0,
            )
            text = result.get("response", "").strip()

        if not text:
            return None, model_id
        text = clean_llm_output(text)
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "signal" in parsed:
            return parsed, model_id
    except (json.JSONDecodeError, Exception):
        pass
    # On any failure, return None (fall through to full extraction -- safe default)
    return None, model_id


async def _classify_review_async(row, local_only: bool, max_tokens: int,
                                  truncate_length: int = 3000) -> tuple[dict[str, Any] | None, str | None]:
    """Async LLM call -- no thread pool, pure async for maximum vLLM throughput.

    Returns (enrichment_dict, model_id) tuple.
    """
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output

    skill = get_skill_registry().get("digest/b2b_churn_extraction")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction' not found")
        return None, None

    llm = get_pipeline_llm(
        prefer_cloud=False,
        try_openrouter=False,
        auto_activate_ollama=True,
    )
    if llm is None:
        logger.warning("No LLM available for B2B churn extraction")
        return None, None

    model_id = getattr(llm, "model_id", None) or getattr(llm, "model", None)

    payload = _build_classify_payload(row, truncate_length)
    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload)),
    ]

    try:
        # Use async chat if available (vLLM, httpx-based backends)
        if hasattr(llm, "chat_async"):
            text = await llm.chat_async(messages=messages, max_tokens=max_tokens, temperature=0.1)
        else:
            # Fallback to sync in thread for backends without async
            result = await asyncio.to_thread(
                llm.chat, messages=messages, max_tokens=max_tokens, temperature=0.1
            )
            text = result.get("response", "").strip()

        if not text:
            return None, model_id
        text = clean_llm_output(text)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None, model_id
        return parsed, model_id
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse B2B enrichment JSON for review %s (vendor=%s)",
            row["id"], row["vendor_name"],
        )
        return None, model_id
    except Exception:
        logger.exception("B2B enrichment LLM call failed")
        return None, None


_KNOWN_PAIN_CATEGORIES = {
    "pricing", "features", "reliability", "support", "integration",
    "performance", "security", "ux", "onboarding", "other",
}


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a value to bool. Returns None if unrecognizable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
    return None


_CHURN_SIGNAL_BOOL_FIELDS = (
    "intent_to_leave",
    "actively_evaluating",
    "migration_in_progress",
    "support_escalation",
    "contract_renewal_mentioned",
)

_KNOWN_SEVERITY_LEVELS = {"primary", "secondary", "minor"}
_KNOWN_LOCK_IN_LEVELS = {"high", "medium", "low", "unknown"}
_KNOWN_SENTIMENT_DIRECTIONS = {"declining", "consistently_negative", "improving", "stable_positive", "unknown"}
_KNOWN_ROLE_TYPES = {"economic_buyer", "champion", "evaluator", "end_user", "unknown"}
_KNOWN_BUYING_STAGES = {"active_purchase", "evaluation", "renewal_decision", "post_purchase", "unknown"}
_KNOWN_DECISION_TIMELINES = {"immediate", "within_quarter", "within_year", "unknown"}
_KNOWN_CONTRACT_VALUE_SIGNALS = {"enterprise_high", "enterprise_mid", "mid_market", "smb", "unknown"}


def _validate_enrichment(result: dict) -> bool:
    """Validate enrichment output structure and data consistency."""
    if "churn_signals" not in result:
        return False
    if "urgency_score" not in result:
        return False
    if not isinstance(result.get("churn_signals"), dict):
        return False

    # Type check: urgency_score must be numeric
    urgency = result.get("urgency_score")
    if isinstance(urgency, str):
        try:
            urgency = float(urgency)
            result["urgency_score"] = urgency
        except (ValueError, TypeError):
            logger.warning("urgency_score is non-numeric string: %r", urgency)
            return False

    if not isinstance(urgency, (int, float)):
        logger.warning("urgency_score has unexpected type: %s", type(urgency).__name__)
        return False

    # Range check: 0-10
    if urgency < 0 or urgency > 10:
        logger.warning("urgency_score out of range [0,10]: %s", urgency)
        return False

    # Boolean coercion: churn_signals fields used in ::boolean casts
    signals = result["churn_signals"]
    for field in _CHURN_SIGNAL_BOOL_FIELDS:
        if field in signals:
            coerced = _coerce_bool(signals[field])
            if coerced is None:
                logger.warning("churn_signals.%s unrecognizable bool: %r -- rejecting", field, signals[field])
                return False
            signals[field] = coerced

    # Consistency warning: high urgency with no intent_to_leave
    intent = signals.get("intent_to_leave")
    if urgency >= 9 and intent is False:
        logger.warning(
            "Contradictory: urgency=%s but intent_to_leave=false -- accepting with warning",
            urgency,
        )

    # Boolean coercion: reviewer_context.decision_maker (used in ::boolean cast)
    reviewer_ctx = result.get("reviewer_context")
    if isinstance(reviewer_ctx, dict) and "decision_maker" in reviewer_ctx:
        coerced = _coerce_bool(reviewer_ctx["decision_maker"])
        if coerced is None:
            logger.warning("reviewer_context.decision_maker unrecognizable bool: %r -- rejecting", reviewer_ctx["decision_maker"])
            return False
        reviewer_ctx["decision_maker"] = coerced

    # Boolean coercion: would_recommend (used in ::boolean cast in vendor_churn_scores)
    if "would_recommend" in result:
        coerced = _coerce_bool(result["would_recommend"])
        if coerced is None:
            # null/None is valid (reviewer didn't express preference) -- keep as null
            result["would_recommend"] = None
        else:
            result["would_recommend"] = coerced

    # Type check: competitors_mentioned must be list; items must be dicts with "name"
    competitors = result.get("competitors_mentioned")
    if competitors is not None and not isinstance(competitors, list):
        logger.warning("competitors_mentioned is not a list: %s", type(competitors).__name__)
        result["competitors_mentioned"] = []
    elif isinstance(competitors, list):
        result["competitors_mentioned"] = [
            c for c in competitors
            if isinstance(c, dict) and "name" in c
        ]

    # Type check: quotable_phrases must be list if present
    qp = result.get("quotable_phrases")
    if qp is not None and not isinstance(qp, list):
        logger.warning("quotable_phrases is not a list: %s", type(qp).__name__)
        result["quotable_phrases"] = []

    # Type check: feature_gaps must be list if present
    fg = result.get("feature_gaps")
    if fg is not None and not isinstance(fg, list):
        logger.warning("feature_gaps is not a list: %s", type(fg).__name__)
        result["feature_gaps"] = []

    # Coerce unknown pain_category to "other"
    pain = result.get("pain_category")
    if pain and pain not in _KNOWN_PAIN_CATEGORIES:
        logger.warning("Unknown pain_category: %r -- coercing to 'other'", pain)
        result["pain_category"] = "other"

    # --- New expanded field validation (permissive: coerce, never reject) ---

    # pain_categories: list of {category, severity}
    pc = result.get("pain_categories")
    if pc is not None:
        if not isinstance(pc, list):
            result["pain_categories"] = []
        else:
            cleaned = []
            for item in pc:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category", "other")
                if cat not in _KNOWN_PAIN_CATEGORIES:
                    cat = "other"
                sev = item.get("severity", "minor")
                if sev not in _KNOWN_SEVERITY_LEVELS:
                    sev = "minor"
                cleaned.append({"category": cat, "severity": sev})
            result["pain_categories"] = cleaned

    # budget_signals: dict with known keys
    bs = result.get("budget_signals")
    if bs is not None:
        if not isinstance(bs, dict):
            result["budget_signals"] = {}
        else:
            if "seat_count" in bs and bs["seat_count"] is not None:
                try:
                    seat = int(bs["seat_count"])
                    bs["seat_count"] = seat if 1 <= seat <= 1_000_000 else None
                except (ValueError, TypeError):
                    bs["seat_count"] = None
            if "price_increase_mentioned" in bs:
                coerced = _coerce_bool(bs["price_increase_mentioned"])
                bs["price_increase_mentioned"] = coerced if coerced is not None else False

    # use_case: dict with lists and lock_in_level
    uc = result.get("use_case")
    if uc is not None:
        if not isinstance(uc, dict):
            result["use_case"] = {}
        else:
            if "modules_mentioned" in uc and not isinstance(uc["modules_mentioned"], list):
                uc["modules_mentioned"] = []
            if "integration_stack" in uc and not isinstance(uc["integration_stack"], list):
                uc["integration_stack"] = []
            lil = uc.get("lock_in_level")
            if lil and lil not in _KNOWN_LOCK_IN_LEVELS:
                uc["lock_in_level"] = "unknown"

    # sentiment_trajectory: dict with direction
    st = result.get("sentiment_trajectory")
    if st is not None:
        if not isinstance(st, dict):
            result["sentiment_trajectory"] = {}
        else:
            d = st.get("direction")
            if d and d not in _KNOWN_SENTIMENT_DIRECTIONS:
                st["direction"] = "unknown"

    # buyer_authority: dict with role_type, booleans, buying_stage
    ba = result.get("buyer_authority")
    if ba is not None:
        if not isinstance(ba, dict):
            result["buyer_authority"] = {}
        else:
            rt = ba.get("role_type")
            if rt and rt not in _KNOWN_ROLE_TYPES:
                ba["role_type"] = "unknown"
            for bool_field in ("has_budget_authority", "executive_sponsor_mentioned"):
                if bool_field in ba:
                    coerced = _coerce_bool(ba[bool_field])
                    ba[bool_field] = coerced if coerced is not None else False
            bstage = ba.get("buying_stage")
            if bstage and bstage not in _KNOWN_BUYING_STAGES:
                ba["buying_stage"] = "unknown"

    # timeline: dict with decision_timeline
    tl = result.get("timeline")
    if tl is not None:
        if not isinstance(tl, dict):
            result["timeline"] = {}
        else:
            dt = tl.get("decision_timeline")
            if dt and dt not in _KNOWN_DECISION_TIMELINES:
                tl["decision_timeline"] = "unknown"

    # contract_context: dict with contract_value_signal
    cc = result.get("contract_context")
    if cc is not None:
        if not isinstance(cc, dict):
            result["contract_context"] = {}
        else:
            cvs = cc.get("contract_value_signal")
            if cvs and cvs not in _KNOWN_CONTRACT_VALUE_SIGNALS:
                cc["contract_value_signal"] = "unknown"

    return True


async def _increment_attempts(pool, review_id, current_attempts: int, max_attempts: int) -> None:
    """Bump attempts atomically; reset to pending or mark failed if exhausted."""
    new_status = "failed" if (current_attempts + 1) >= max_attempts else "pending"
    await pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment_attempts = enrichment_attempts + 1,
            enrichment_status = $1
        WHERE id = $2
        """,
        new_status, review_id,
    )
