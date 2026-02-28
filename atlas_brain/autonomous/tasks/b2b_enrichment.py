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
               source, raw_metadata,
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
                                  cfg.enrichment_max_tokens, cfg.review_truncate_length)
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
                         max_tokens: int, truncate_length: int = 3000) -> bool:
    """Enrich a single B2B review with churn signals. Returns True on success."""
    review_id = row["id"]

    try:
        result = _classify_review(row, local_only, max_tokens, truncate_length)

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


def _smart_truncate(text: str, max_len: int = 3000) -> str:
    """Truncate preserving both beginning and end of review text.

    Churn signals often appear at the end ("I'm switching to X next quarter"),
    so naive head-only truncation loses them.
    """
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 15
    return text[:half] + "\n[...truncated...]\n" + text[-half:]


def _classify_review(row, local_only: bool, max_tokens: int,
                     truncate_length: int = 3000) -> dict[str, Any] | None:
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

    review_text = _smart_truncate(row["review_text"] or "", max_len=truncate_length)

    # Extract source context from raw_metadata
    raw_meta = row.get("raw_metadata") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            raw_meta = {}

    payload = {
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
            logger.warning("would_recommend unrecognizable bool: %r -- rejecting", result["would_recommend"])
            return False
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
                    bs["seat_count"] = int(bs["seat_count"])
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
