"""
B2B review enrichment: extract churn signals from pending reviews via LLM
using the b2b_churn_extraction skill.

Single-pass enrichment (one LLM call per review). Polls b2b_reviews WHERE
enrichment_status = 'pending', calls LLM, stores result in enrichment JSONB
column, sets status to 'enriched'.

Runs on an interval (default 5 min). Returns _skip_synthesis so the
runner does not double-synthesize.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: enrich pending B2B reviews with churn signals."""
    cfg = settings.b2b_churn
    if not cfg.enabled:
        return {"_skip_synthesis": "B2B churn pipeline disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    max_batch = cfg.enrichment_max_per_batch
    max_attempts = cfg.enrichment_max_attempts

    rows = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, enrichment_attempts
        FROM b2b_reviews
        WHERE enrichment_status = 'pending'
          AND enrichment_attempts < $1
        ORDER BY imported_at ASC
        LIMIT $2
        """,
        max_attempts,
        max_batch,
    )

    if not rows:
        return {"_skip_synthesis": "No B2B reviews to enrich"}

    enriched = 0
    failed = 0

    for row in rows:
        ok = await _enrich_single(pool, row, max_attempts, cfg.enrichment_local_only,
                                  cfg.enrichment_max_tokens)
        if ok:
            enriched += 1
        elif (row["enrichment_attempts"] + 1) >= max_attempts:
            failed += 1

    logger.info(
        "B2B enrichment: %d enriched, %d failed (of %d)",
        enriched, failed, len(rows),
    )

    return {
        "_skip_synthesis": "B2B enrichment complete",
        "total": len(rows),
        "enriched": enriched,
        "failed": failed,
    }


async def _enrich_single(pool, row, max_attempts: int, local_only: bool,
                         max_tokens: int) -> bool:
    """Enrich a single B2B review with churn signals. Returns True on success."""
    review_id = row["id"]

    try:
        result = _classify_review(row, local_only, max_tokens)

        if result and _validate_enrichment(result):
            await pool.execute(
                """
                UPDATE b2b_reviews
                SET enrichment = $1,
                    enrichment_status = 'enriched',
                    enrichment_attempts = enrichment_attempts + 1,
                    enriched_at = $2
                WHERE id = $3
                """,
                json.dumps(result),
                datetime.now(timezone.utc),
                review_id,
            )
            return True
        else:
            await _increment_attempts(pool, review_id, row["enrichment_attempts"], max_attempts)
            return False

    except Exception:
        logger.exception("Failed to enrich B2B review %s", review_id)
        try:
            await pool.execute(
                "UPDATE b2b_reviews SET enrichment_attempts = enrichment_attempts + 1 WHERE id = $1",
                review_id,
            )
        except Exception:
            pass
        return False


def _classify_review(row, local_only: bool, max_tokens: int) -> dict[str, Any] | None:
    """Call LLM with b2b_churn_extraction skill."""
    from ...skills import get_skill_registry
    from ...services.protocols import Message
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output

    skill = get_skill_registry().get("digest/b2b_churn_extraction")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction' not found")
        return None

    llm = get_pipeline_llm(
        prefer_cloud=not local_only,
        try_openrouter=False,
        auto_activate_ollama=True,
    )
    if llm is None:
        logger.warning("No LLM available for B2B churn extraction")
        return None

    review_text = (row["review_text"] or "")[:3000]
    payload = {
        "vendor_name": row["vendor_name"],
        "product_name": row["product_name"] or "",
        "product_category": row["product_category"] or "",
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

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload)),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=0.1)
        text = result.get("response", "").strip()
        if not text:
            return None
        text = clean_llm_output(text)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except json.JSONDecodeError:
        logger.debug("Failed to parse B2B enrichment JSON")
        return None
    except Exception:
        logger.exception("B2B enrichment LLM call failed")
        return None


def _validate_enrichment(result: dict) -> bool:
    """Basic validation of enrichment output structure."""
    if "churn_signals" not in result:
        return False
    if "urgency_score" not in result:
        return False
    if not isinstance(result.get("churn_signals"), dict):
        return False
    return True


async def _increment_attempts(pool, review_id, current_attempts: int, max_attempts: int) -> None:
    """Bump attempts; mark failed if exhausted."""
    new_attempts = current_attempts + 1
    await pool.execute(
        "UPDATE b2b_reviews SET enrichment_attempts = $1 WHERE id = $2",
        new_attempts, review_id,
    )
    if new_attempts >= max_attempts:
        await pool.execute(
            "UPDATE b2b_reviews SET enrichment_status = 'failed' WHERE id = $1",
            review_id,
        )
