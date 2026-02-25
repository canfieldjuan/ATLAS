"""
Deep enrichment pipeline: second-pass extraction of 10+ rich fields per review.

Simplified version of blast_deep_enrichment.py for ongoing incremental
enrichment of new reviews. Processes a small batch per interval (default 5
reviews every 10 min). Uses local LLM only (no cloud).

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

# Required keys and their expected types
_REQUIRED_KEYS = {
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
}


def _validate_extraction(data: dict) -> bool:
    """Check all 10 required keys present with correct types."""
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
    bc = data.get("buyer_context")
    if isinstance(bc, dict):
        for field in ("use_case", "buyer_type", "price_sentiment"):
            if field not in bc:
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
    """Call LLM with deep_extraction skill to extract 10 fields."""
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
