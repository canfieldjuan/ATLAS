"""
Deep enrichment pipeline: single-pass extraction of 32 rich fields per review.

Simplified version of blast_deep_enrichment.py for ongoing incremental
enrichment of new reviews. Processes a small batch per interval (default 5
reviews every 10 min). Uses local LLM only (no cloud).

Covers all three extraction sections in one LLM call:
  Section A: Product Analysis (10 fields)
  Section B: Buyer Psychology (12 fields)
  Section C: Extended Context (10 fields)

Returns _skip_synthesis always -- results go to DB, not to ntfy.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.deep_enrichment")

# Required keys and their expected types (32 fields across 3 sections)
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
_VALID_CONSEQUENCE = {"none", "positive_impact", "inconvenience", "workflow_impact", "financial_loss", "safety_concern"}
_VALID_REPLACEMENT = {"returned", "replaced_same", "switched_brand", "switched_to", "avoided", "kept_broken", "kept_using", "repurchased", "unknown"}

# Enum sets for Section C
_VALID_LOYALTY = {"first_time", "occasional", "loyal", "long_term_loyal"}
_VALID_DELAY = {"immediate", "days", "weeks", "months", "unknown"}
_VALID_TRAJECTORY = {"always_negative", "degraded", "mixed_then_negative", "mixed_then_positive", "improved", "always_positive", "unknown"}
_VALID_OCCASION = {"none", "gift", "replacement", "upgrade", "first_in_category", "seasonal", "event", "professional_use"}


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

    # Section C sub-object key checks
    eco = data.get("ecosystem_lock_in", {})
    if not isinstance(eco, dict) or "level" not in eco or "ecosystem" not in eco:
        return False
    safety = data.get("safety_flag", {})
    if not isinstance(safety, dict) or "flagged" not in safety or "description" not in safety:
        return False
    bulk = data.get("bulk_purchase_signal", {})
    if not isinstance(bulk, dict) or "type" not in bulk or "estimated_qty" not in bulk:
        return False
    barrier = data.get("switching_barrier", {})
    if not isinstance(barrier, dict) or "level" not in barrier or "reason" not in barrier:
        return False
    amp = data.get("amplification_intent", {})
    if not isinstance(amp, dict) or "intent" not in amp or "context" not in amp:
        return False
    openness = data.get("review_sentiment_openness", {})
    if not isinstance(openness, dict) or "open" not in openness or "condition" not in openness:
        return False

    return True


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: deep-enrich pending product reviews."""
    cfg = settings.external_data
    if not cfg.complaint_mining_enabled:
        return {"_skip_synthesis": "Complaint mining disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    max_batch = cfg.deep_enrichment_max_per_batch
    max_attempts = cfg.deep_enrichment_max_attempts
    max_tokens = cfg.deep_enrichment_max_tokens

    rows = await pool.fetch(
        """
        SELECT pr.id, pr.asin, pr.rating, pr.summary, pr.review_text,
               pr.root_cause, pr.severity, pr.pain_score,
               pr.deep_enrichment_attempts,
               pm.title AS product_title, pm.brand AS product_brand
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'pending'
          AND pr.deep_enrichment_attempts < $1
        ORDER BY pr.imported_at ASC
        LIMIT $2
        """,
        max_attempts,
        max_batch,
    )

    if not rows:
        return {"_skip_synthesis": "No reviews to deep-enrich"}

    enriched = 0
    failed = 0

    for row in rows:
        ok = await _enrich_single(pool, row, max_attempts, max_tokens)
        if ok:
            enriched += 1
        elif (row["deep_enrichment_attempts"] + 1) >= max_attempts:
            failed += 1

    logger.info(
        "Deep enrichment: %d enriched, %d failed (of %d)",
        enriched, failed, len(rows),
    )

    return {
        "_skip_synthesis": "Deep enrichment complete",
        "total": len(rows),
        "enriched": enriched,
        "failed": failed,
    }


async def _enrich_single(pool, row, max_attempts: int, max_tokens: int) -> bool:
    """Extract and store deep fields for a single review. Returns True on success."""
    review_id = row["id"]

    try:
        extraction = await _extract_review(row, max_tokens)

        if extraction and _validate_extraction(extraction):
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
            return True
        else:
            await _increment_attempts(pool, row, max_attempts)
            return False

    except Exception:
        logger.exception("Failed to deep-enrich review %s", review_id)
        try:
            await pool.execute(
                "UPDATE product_reviews SET deep_enrichment_attempts = deep_enrichment_attempts + 1 WHERE id = $1",
                review_id,
            )
        except Exception:
            pass
        return False


async def _increment_attempts(pool, row, max_attempts: int) -> None:
    """Bump attempts; mark deep_failed if exhausted."""
    review_id = row["id"]
    new_attempts = row["deep_enrichment_attempts"] + 1
    await pool.execute(
        "UPDATE product_reviews SET deep_enrichment_attempts = $1 WHERE id = $2",
        new_attempts, review_id,
    )
    if new_attempts >= max_attempts:
        await pool.execute(
            "UPDATE product_reviews SET deep_enrichment_status = 'deep_failed' WHERE id = $1",
            review_id,
        )


async def _extract_review(row, max_tokens: int) -> dict[str, Any] | None:
    """Call LLM with deep_extraction skill to extract 32 fields."""
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output

    skill = get_skill_registry().get("digest/deep_extraction")
    if not skill:
        logger.warning("Skill 'digest/deep_extraction' not found")
        return None

    llm = get_pipeline_llm(
        prefer_cloud=False,
        try_openrouter=False,
        auto_activate_ollama=True,
    )
    if llm is None:
        logger.warning("No LLM available for deep extraction")
        return None

    payload = json.dumps({
        "review_text": (row["review_text"] or "")[:3000],
        "summary": row["summary"] or "",
        "rating": float(row["rating"]),
        "product_name": row["product_title"] or row["asin"],
        "brand": row["product_brand"] or "",
        "root_cause": row["root_cause"] or "",
        "severity": row["severity"] or "",
        "pain_score": float(row["pain_score"]) if row["pain_score"] else 0.0,
    })

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=payload),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=0.1)
        text = clean_llm_output(result.get("response", ""))
        if not text:
            return None
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        logger.debug("Failed to parse deep extraction JSON")
    except Exception:
        logger.exception("Deep extraction LLM call failed")
    return None
