"""Follow-up task: build per-(incumbent, challenger) pair challenger briefs.

Runs after all other B2B follow-up tasks. Reads persisted artifacts from
b2b_displacement_edges, b2b_intelligence (battle_card, accounts_in_motion),
b2b_product_profiles, shared cross-vendor reasoning lookups, and
b2b_churn_signals. Assembles one b2b_intelligence row per
(incumbent, challenger) pair with report_type='challenger_brief'.

No LLM calls -- purely deterministic assembly from pre-computed artifacts.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import date
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_shared import (
    _fetch_latest_evidence_vault,
    _segment_targeting_summary,
    _timing_summary_payload,
)
from ._execution_progress import _update_execution_progress

logger = logging.getLogger("atlas.tasks.b2b_challenger_brief")

_STAGE_SELECTING_PAIRS = "selecting_pairs"
_STAGE_BUILDING_BRIEFS = "building_briefs"
_STAGE_FINALIZING = "finalizing"


# ---------------------------------------------------------------------------
# Freshness check
# ---------------------------------------------------------------------------

async def _check_freshness(pool) -> date | None:
    """Return today's date if the core run completed for today."""
    today = date.today()
    marker = await pool.fetchval(
        "SELECT 1 FROM b2b_intelligence "
        "WHERE report_type = 'core_run_complete' AND report_date = $1",
        today,
    )
    if not marker:
        logger.info("Core run not complete for %s, skipping", today)
        return None
    return today


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

async def _select_displacement_pairs(
    pool,
    *,
    min_mentions: int,
    max_per_incumbent: int,
    window_days: int,
) -> list[dict[str, Any]]:
    """Query displacement edges for pairs with sufficient evidence."""
    from ._b2b_shared import _canonicalize_vendor

    rows = await pool.fetch(
        """
        SELECT from_vendor, to_vendor,
               SUM(mention_count) AS total_mentions,
               MAX(signal_strength) AS max_signal,
               MAX(confidence_score) AS max_confidence
        FROM b2b_displacement_edges
        WHERE computed_date > NOW() - make_interval(days => $1)
        GROUP BY from_vendor, to_vendor
        HAVING SUM(mention_count) >= $2
        ORDER BY SUM(mention_count) DESC
        """,
        window_days,
        min_mentions,
    )

    # Canonicalize and deduplicate
    pairs_by_incumbent: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for r in rows:
        inc = _canonicalize_vendor(r["from_vendor"])
        chal = _canonicalize_vendor(r["to_vendor"])
        if not inc or not chal:
            continue
        # Skip self-flow
        if inc.lower() == chal.lower():
            continue
        key = (inc.lower(), chal.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs_by_incumbent[inc.lower()].append({
            "incumbent": inc,
            "challenger": chal,
            "total_mentions": int(r["total_mentions"]),
            "max_signal": r["max_signal"],
            "max_confidence": float(r["max_confidence"]) if r["max_confidence"] is not None else 0,
        })

    # Rank per incumbent, keep top N
    result: list[dict] = []
    for _inc_key, pairs in pairs_by_incumbent.items():
        pairs.sort(key=lambda p: p["total_mentions"], reverse=True)
        result.extend(pairs[:max_per_incumbent])

    return result


# ---------------------------------------------------------------------------
# Data fetchers (all read from persisted artifacts)
# ---------------------------------------------------------------------------

async def _fetch_persisted_report(
    pool, report_type: str, vendor_filter: str, today: date,
    *, fallback_days: int = 7,
) -> dict | None:
    """Fetch the latest intelligence_data for a given report_type + vendor."""
    record = await _fetch_persisted_report_record(
        pool,
        report_type,
        vendor_filter,
        today,
        fallback_days=fallback_days,
    )
    return record["data"] if record else None


async def _fetch_persisted_report_record(
    pool, report_type: str, vendor_filter: str, today: date,
    *, fallback_days: int = 7,
) -> dict[str, Any] | None:
    """Fetch a persisted artifact plus source-date metadata."""
    row = await pool.fetchrow(
        """
        SELECT intelligence_data, report_date
        FROM b2b_intelligence
        WHERE report_type = $1
          AND LOWER(COALESCE(vendor_filter, '')) = LOWER($2)
          AND report_date >= $3::date - make_interval(days => $4::int)
        ORDER BY report_date DESC, created_at DESC
        LIMIT 1
        """,
        report_type,
        vendor_filter,
        today,
        int(fallback_days),
    )
    if not row:
        return None
    data = row["intelligence_data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return {
        "data": data,
        "report_date": row["report_date"],
        "stale": bool(row["report_date"] and row["report_date"] != today),
    }


async def _fetch_displacement_detail(
    pool, from_vendor: str, to_vendor: str, window_days: int,
) -> dict:
    """Aggregate displacement edge detail for a specific pair."""
    rows = await pool.fetch(
        """
        SELECT id, mention_count, signal_strength, confidence_score,
               primary_driver, key_quote, source_distribution,
               sample_review_ids,
               computed_date
        FROM b2b_displacement_edges
        WHERE LOWER(from_vendor) = LOWER($1)
          AND LOWER(to_vendor) = LOWER($2)
          AND computed_date > NOW() - make_interval(days => $3)
        ORDER BY computed_date DESC
        """,
        from_vendor,
        to_vendor,
        window_days,
    )
    if not rows:
        return {
            "total_mentions": 0,
            "signal_strength": "none",
            "confidence_score": 0,
            "primary_driver": None,
            "key_quote": None,
            "source_distribution": {},
            "sample_review_ids": [],
        }

    total_mentions = sum(r["mention_count"] for r in rows)

    # Use the most recent row for scalar fields
    latest = rows[0]
    signal_strength = latest["signal_strength"] or "none"
    confidence_score = float(latest["confidence_score"]) if latest["confidence_score"] is not None else 0
    primary_driver = latest["primary_driver"]
    key_quote = latest["key_quote"]

    # Merge source distributions
    merged_sources: dict[str, int] = {}
    sample_review_ids: list[str] = []
    for r in rows:
        sd = r["source_distribution"]
        if isinstance(sd, str):
            try:
                sd = json.loads(sd)
            except (json.JSONDecodeError, TypeError):
                sd = {}
        if isinstance(sd, dict):
            for src, count in sd.items():
                merged_sources[src] = merged_sources.get(src, 0) + (int(count) if count else 0)
        for review_id in (r.get("sample_review_ids") or []):
            review_id_text = str(review_id or "").strip()
            if review_id_text and review_id_text not in sample_review_ids:
                sample_review_ids.append(review_id_text)

    return {
        "total_mentions": total_mentions,
        "signal_strength": signal_strength,
        "confidence_score": round(confidence_score, 2),
        "primary_driver": primary_driver,
        "key_quote": key_quote,
        "source_distribution": merged_sources,
        "sample_review_ids": sample_review_ids,
    }


def _reference_ids_from_displacement_detail(
    displacement_detail: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Build fallback witness refs from displacement-edge provenance."""
    if not isinstance(displacement_detail, dict):
        return {}
    witness_ids = [
        str(review_id).strip()
        for review_id in (displacement_detail.get("sample_review_ids") or [])
        if str(review_id).strip()
    ]
    if not witness_ids:
        return {}
    return {"witness_ids": list(dict.fromkeys(witness_ids))}


async def _fetch_product_profile(pool, vendor: str) -> dict | None:
    """Fetch the latest product profile for a vendor."""
    row = await pool.fetchrow(
        """
        SELECT strengths, weaknesses, pain_addressed, commonly_switched_from,
               top_integrations, profile_summary, product_category
        FROM b2b_product_profiles
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor,
    )
    if not row:
        return None

    def _parse(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val
        return val

    return {
        "strengths": _parse(row["strengths"]) or [],
        "weaknesses": _parse(row["weaknesses"]) or [],
        "pain_addressed": _parse(row["pain_addressed"]) or [],
        "commonly_switched_from": _parse(row["commonly_switched_from"]) or [],
        "top_integrations": _parse(row["top_integrations"]) or [],
        "profile_summary": row["profile_summary"] or "",
        "category": row["product_category"] or "",
    }


def _quote_token_set(text: Any) -> set[str]:
    """Tokenize quote text for lightweight similarity checks."""
    tokens = {
        token for token in str(text or "").lower().replace("/", " ").split()
        if len(token) > 2
    }
    return tokens


def _quotes_are_similar(left: Any, right: Any, *, threshold: float) -> bool:
    """Return True when two quotes are near-duplicates."""
    left_text = " ".join(str(left or "").lower().split())
    right_text = " ".join(str(right or "").lower().split())
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if left_text in right_text or right_text in left_text:
        return True
    left_tokens = _quote_token_set(left_text)
    right_tokens = _quote_token_set(right_text)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return bool(union) and (overlap / union) >= threshold


def _extract_pain_quotes_from_reviews(
    review_rows: list[dict[str, Any]],
    *,
    vendor: str,
    max_quotes: int,
    min_urgency: float,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """Build battle-card-compatible pain quotes from enriched review rows."""
    vendor_lower = vendor.lower().strip()
    candidates: list[dict[str, Any]] = []
    for row in review_rows:
        if str(row.get("vendor_name") or "").lower().strip() != vendor_lower:
            continue
        urgency = float(row.get("urgency") or 0)
        if urgency < min_urgency:
            continue
        phrases = row.get("phrases")
        if isinstance(phrases, str):
            try:
                phrases = json.loads(phrases)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(phrases, list):
            continue
        for phrase in phrases:
            quote = str(phrase or "").strip()
            if not quote:
                continue
            candidates.append({
                "quote": quote,
                "urgency": urgency,
                "pain_category": row.get("pain_category") or "",
                "company": row.get("company") or "",
                "role": row.get("role") or "",
                "source_site": row.get("source") or "",
            })
    candidates.sort(key=lambda item: (float(item.get("urgency") or 0), len(str(item.get("quote") or ""))), reverse=True)
    deduped: list[dict[str, Any]] = []
    for quote in candidates:
        if any(_quotes_are_similar(quote["quote"], seen["quote"], threshold=similarity_threshold) for seen in deduped):
            continue
        deduped.append(quote)
        if len(deduped) >= max_quotes:
            break
    return deduped


async def _fetch_review_pain_quotes(
    pool,
    vendor: str,
    *,
    window_days: int = 90,
    limit: int = 5,
    candidate_limit: int = 25,
    min_urgency: float = 6.0,
    similarity_threshold: float = 0.7,
) -> list[dict]:
    """Fetch top pain quotes directly from enriched reviews for a vendor.

    Uses shared adapter for enrichment reads. Post-processing (dedup,
    similarity filtering) stays local.
    """
    from atlas_brain.autonomous.tasks._b2b_shared import read_vendor_quote_evidence

    rows = await read_vendor_quote_evidence(
        pool,
        vendor_name=vendor,
        window_days=window_days,
        min_urgency=min_urgency,
        limit=candidate_limit,
        require_quotes=True,
        recency_column="imported_at",
    )
    normalized_rows = [{
        "vendor_name": r["vendor_name"],
        "source": r["source"],
        "company": r["reviewer_company"],
        "role": r["role_level"],
        "pain_category": r["pain_category"],
        "phrases": r["quotable_phrases"],
        "urgency": r["urgency"],
    } for r in rows]
    return _extract_pain_quotes_from_reviews(
        normalized_rows,
        vendor=vendor,
        max_quotes=limit,
        min_urgency=min_urgency,
        similarity_threshold=similarity_threshold,
    )


async def _fetch_churn_signal(pool, vendor: str, today: date) -> dict | None:
    """Fetch the latest churn signal for a vendor.

    The b2b_churn_signals table uses:
    - reasoning_risk_level (not risk_level)
    - reasoning_key_signals (not key_signals)
    - decision_maker_churn_rate (not dm_churn_rate)
    - last_computed_at (no signal_date column)
    churn_pressure_score and sentiment_direction live in the battle card
    intelligence_data, not in this table. We read what we can from the
    signal row and supplement from the battle card later.
    """
    row = await pool.fetchrow(
        """
        SELECT archetype, archetype_confidence,
               reasoning_risk_level, reasoning_key_signals,
               price_complaint_rate, decision_maker_churn_rate,
               total_reviews, churn_intent_count, avg_urgency_score,
               top_competitors, sentiment_distribution
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor,
    )
    if not row:
        return None

    key_signals = row["reasoning_key_signals"]
    if isinstance(key_signals, str):
        try:
            key_signals = json.loads(key_signals)
        except (json.JSONDecodeError, TypeError):
            key_signals = []

    # Compute churn_pressure_score from available columns
    from ._b2b_shared import _compute_churn_pressure_score
    total_reviews = int(row["total_reviews"] or 0)
    churn_intent = int(row["churn_intent_count"] or 0)
    churn_density = round((churn_intent * 100.0 / total_reviews), 1) if total_reviews else 0.0
    avg_urgency = float(row["avg_urgency_score"] or 0)
    dm_rate = float(row["decision_maker_churn_rate"] or 0)
    pr_rate = float(row["price_complaint_rate"] or 0)
    top_comps = row["top_competitors"]
    if isinstance(top_comps, str):
        try:
            top_comps = json.loads(top_comps)
        except (json.JSONDecodeError, TypeError):
            top_comps = []
    disp_count = len(top_comps) if isinstance(top_comps, list) else 0
    churn_pressure_score = round(_compute_churn_pressure_score(
        churn_density=churn_density,
        avg_urgency=avg_urgency,
        dm_churn_rate=dm_rate,
        displacement_mention_count=disp_count,
        price_complaint_rate=pr_rate,
        total_reviews=total_reviews,
        archetype=row["archetype"],
    ), 1)

    # Derive sentiment_direction from sentiment_distribution
    sent_dist = row["sentiment_distribution"]
    if isinstance(sent_dist, str):
        try:
            sent_dist = json.loads(sent_dist)
        except (json.JSONDecodeError, TypeError):
            sent_dist = {}
    sentiment_direction = None
    if isinstance(sent_dist, dict) and sent_dist:
        _POS = {"stable_positive", "improving"}
        _NEG = {"consistently_negative", "declining"}
        pos = sum(v for k, v in sent_dist.items() if k in _POS)
        neg = sum(v for k, v in sent_dist.items() if k in _NEG)
        if neg > pos * 1.5:
            sentiment_direction = "consistently_negative"
        elif pos > neg * 1.5:
            sentiment_direction = "stable_positive"
        else:
            sentiment_direction = "mixed"

    return {
        "archetype": row["archetype"],
        "archetype_confidence": float(row["archetype_confidence"]) if row["archetype_confidence"] is not None else None,
        "risk_level": row["reasoning_risk_level"],
        "key_signals": key_signals or [],
        "churn_pressure_score": churn_pressure_score,
        "sentiment_direction": sentiment_direction,
        "price_complaint_rate": float(row["price_complaint_rate"]) if row["price_complaint_rate"] is not None else None,
        "dm_churn_rate": float(row["decision_maker_churn_rate"]) if row["decision_maker_churn_rate"] is not None else None,
    }


async def _fetch_synthesis_view(
    pool,
    vendor: str,
    today: date,
    analysis_window_days: int,
) -> Any | None:
    """Fetch best reasoning view for one vendor (synthesis-first, legacy fallback)."""
    from ._b2b_synthesis_reader import load_best_reasoning_view

    return await load_best_reasoning_view(
        pool, vendor, as_of=today, analysis_window_days=analysis_window_days,
    )


async def _resolve_cross_vendor_battle(
    pool,
    vendor_a: str,
    vendor_b: str,
    today: date,
    xv_lookup: dict[str, Any],
) -> dict | None:
    """Resolve a pairwise battle from the merged cross-vendor lookup."""
    del pool, today
    battles = xv_lookup.get("battles") or {}
    needle = tuple(sorted([vendor_a.strip().lower(), vendor_b.strip().lower()]))

    entry = None
    # Try exact key, then case-insensitive scan
    for k, v in battles.items():
        if tuple(sorted(s.lower() for s in k)) == needle:
            entry = v
            break

    if entry is not None:
        conclusion = entry.get("conclusion") or {}
        if isinstance(conclusion, dict) and conclusion.get("conclusion"):
            key_insights = conclusion.get("key_insights") or []
            if isinstance(key_insights, list):
                normalized: list[dict[str, str]] = []
                for item in key_insights:
                    if isinstance(item, str) and item:
                        normalized.append({"insight": item, "evidence": item})
                    elif isinstance(item, dict):
                        text = item.get("insight") or item.get("text") or ""
                        evidence = item.get("evidence") or text
                        if text:
                            normalized.append({"insight": text, "evidence": evidence})
                key_insights = normalized
            result = {
                "conclusion": conclusion.get("conclusion") or "",
                "winner": conclusion.get("winner") or "",
                "loser": conclusion.get("loser") or "",
                "durability": conclusion.get("durability_assessment") or "",
                "key_insights": key_insights,
                "confidence": float(conclusion["confidence"]) if conclusion.get("confidence") is not None else None,
            }
            reference_ids = entry.get("reference_ids")
            if isinstance(reference_ids, dict) and reference_ids:
                result["reference_ids"] = reference_ids
            return result

    return None


async def _retire_unselected_challenger_briefs(
    pool,
    *,
    today: date,
    pairs: list[dict[str, Any]],
) -> int:
    """Delete same-day challenger briefs for pairs no longer selected."""
    active_pairs = {
        (
            str(pair.get("incumbent") or "").strip().lower(),
            str(pair.get("challenger") or "").strip().lower(),
        )
        for pair in pairs
        if str(pair.get("incumbent") or "").strip()
        and str(pair.get("challenger") or "").strip()
    }
    rows = await pool.fetch(
        """
        SELECT id, vendor_filter, category_filter
        FROM b2b_intelligence
        WHERE report_type = 'challenger_brief'
          AND report_date = $1
        """,
        today,
    )
    stale_ids = [
        row["id"]
        for row in rows
        if (
            str(row["vendor_filter"] or "").strip().lower(),
            str(row["category_filter"] or "").strip().lower(),
        ) not in active_pairs
    ]
    if not stale_ids:
        return 0
    await pool.execute(
        "DELETE FROM b2b_intelligence WHERE id = ANY($1::uuid[])",
        stale_ids,
    )
    return len(stale_ids)


# ---------------------------------------------------------------------------
# Compute helpers (deterministic, no DB)
# ---------------------------------------------------------------------------

def _normalize_product_profile_weaknesses(profile_weaknesses: list[Any]) -> list[dict[str, Any]]:
    """Convert product-profile weaknesses to battle-card-compatible shape."""
    normalized: list[dict[str, Any]] = []
    for item in profile_weaknesses or []:
        if not isinstance(item, dict):
            continue
        aspect = item.get("aspect") or item.get("area") or item.get("category") or item.get("name") or item.get("weakness") or ""
        if not aspect:
            continue
        score = float(item.get("score") or item.get("avg_score") or 0)
        count = int(item.get("review_count") or item.get("evidence_count") or item.get("mentions") or item.get("count") or 0)
        evidence = (
            "Satisfaction score %.1f/5.0 across %d reviews" % (score, count)
            if score and count else
            "Structured weakness signal from product profile"
        )
        normalized.append({
            "weakness": aspect,
            "area": aspect,
            "evidence": evidence,
            "mention_count": count,
            "count": count,
            "source": "product_profile",
        })
    normalized.sort(key=lambda item: int(item.get("mention_count") or 0), reverse=True)
    return normalized[:5]


def _build_challenger_vault_weaknesses(
    vault: dict[str, Any] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Convert canonical vault weakness rows to brief-compatible weakness rows."""
    if not isinstance(vault, dict):
        return []
    recent_window_days = int(vault.get("recent_window_days") or 0)
    normalized: list[dict[str, Any]] = []
    for item in vault.get("weakness_evidence") or []:
        if not isinstance(item, dict):
            continue
        area = str(item.get("label") or item.get("key") or "").strip()
        if not area:
            continue
        total = int(item.get("mention_count_total") or 0)
        recent = item.get("mention_count_recent")
        trend = item.get("trend") if isinstance(item.get("trend"), dict) else {}
        evidence_parts = ["%d mentions" % total] if total else []
        if recent is not None and recent_window_days:
            evidence_parts.append("%d in last %d days" % (int(recent), recent_window_days))
        direction = str(trend.get("direction") or "").strip()
        if direction:
            evidence_parts.append("trend %s" % direction)
        normalized.append({
            "weakness": area,
            "area": area,
            "evidence": ", ".join(evidence_parts) or "Canonical weakness evidence from evidence vault",
            "mention_count": total,
            "count": total,
            "evidence_count": total,
            "customer_quote": item.get("best_quote") or "",
            "source": "evidence_vault",
            "evidence_type": item.get("evidence_type") or "",
        })
    normalized.sort(key=lambda row: int(row.get("count") or 0), reverse=True)
    return normalized[:limit]


def _build_challenger_vault_strengths(
    vault: dict[str, Any] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Convert canonical vault strength rows to brief-compatible strength rows.

    Mirrors ``_build_challenger_vault_weaknesses`` so the incumbent profile
    can surface what the vendor does well -- enabling reps to anticipate
    objections and identify defensible areas.
    """
    if not isinstance(vault, dict):
        return []
    recent_window_days = int(vault.get("recent_window_days") or 0)
    normalized: list[dict[str, Any]] = []
    for item in vault.get("strength_evidence") or []:
        if not isinstance(item, dict):
            continue
        area = str(item.get("label") or item.get("key") or "").strip()
        if not area:
            continue
        total = int(item.get("mention_count_total") or 0)
        recent = item.get("mention_count_recent")
        trend = item.get("trend") if isinstance(item.get("trend"), dict) else {}
        evidence_parts = ["%d mentions" % total] if total else []
        if recent is not None and recent_window_days:
            evidence_parts.append("%d in last %d days" % (int(recent), recent_window_days))
        direction = str(trend.get("direction") or "").strip()
        if direction:
            evidence_parts.append("trend %s" % direction)
        normalized.append({
            "strength": area,
            "area": area,
            "evidence": ", ".join(evidence_parts) or "Canonical strength evidence from evidence vault",
            "mention_count": total,
            "count": total,
            "customer_quote": item.get("best_quote") or "",
            "source": "evidence_vault",
            "evidence_type": item.get("evidence_type") or "",
        })
    normalized.sort(key=lambda row: int(row.get("count") or 0), reverse=True)
    return normalized[:limit]


def _build_challenger_vault_pain_quotes(
    vault: dict[str, Any] | None,
    *,
    max_quotes: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """Convert canonical vault weakness quotes to brief-compatible pain quotes."""
    if not isinstance(vault, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for item in vault.get("weakness_evidence") or []:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("best_quote") or "").strip()
        if not quote:
            continue
        source = item.get("quote_source") if isinstance(item.get("quote_source"), dict) else {}
        metrics = item.get("supporting_metrics") if isinstance(item.get("supporting_metrics"), dict) else {}
        candidates.append({
            "quote": quote,
            "urgency": metrics.get("avg_urgency_when_mentioned"),
            "pain_category": item.get("key") or item.get("label") or "",
            "company": source.get("company") or "",
            "role": source.get("reviewer_title") or "",
            "source_site": source.get("source") or "",
            "reviewed_at": source.get("reviewed_at"),
            "rating": source.get("rating"),
            "_total": int(item.get("mention_count_total") or 0),
            "_recent": int(item.get("mention_count_recent") or 0),
        })
    candidates.sort(
        key=lambda row: (
            row["_total"],
            row["_recent"],
            str(row.get("reviewed_at") or ""),
            len(str(row.get("quote") or "")),
        ),
        reverse=True,
    )
    deduped: list[dict[str, Any]] = []
    for item in candidates:
        if any(_quotes_are_similar(item["quote"], seen["quote"], threshold=similarity_threshold) for seen in deduped):
            continue
        deduped.append({k: v for k, v in item.items() if not k.startswith("_")})
        if len(deduped) >= max_quotes:
            break
    return deduped


def _compute_weakness_coverage(
    incumbent_weaknesses: list[dict],
    challenger_pain_addressed: list,
) -> list[dict]:
    """Match incumbent weaknesses against challenger's pain_addressed.

    Returns list of {incumbent_weakness, challenger_strength_score, match_quality}.
    """
    if not incumbent_weaknesses:
        return []

    # Normalize challenger pain areas for matching
    challenger_areas: set[str] = set()
    for item in (challenger_pain_addressed or []):
        if isinstance(item, dict):
            area = (item.get("area") or item.get("pain") or item.get("name") or "").lower().strip()
        else:
            area = str(item).lower().strip()
        if area:
            challenger_areas.add(area)

    coverage: list[dict] = []
    for w in incumbent_weaknesses:
        if isinstance(w, dict):
            weakness_area = (w.get("area") or w.get("weakness") or w.get("name") or "").lower().strip()
            weakness_score = float(w.get("score") or w.get("severity") or 0)
            evidence_count = int(w.get("evidence_count") or w.get("mentions") or 0)
        else:
            weakness_area = str(w).lower().strip()
            weakness_score = 0
            evidence_count = 0

        if not weakness_area:
            continue

        # Check if challenger addresses this weakness
        # Use substring matching for flexibility
        match_score = 0.0
        for ca in challenger_areas:
            if ca == weakness_area:
                match_score = 1.0
                break
            if ca in weakness_area or weakness_area in ca:
                match_score = max(match_score, 0.7)

        if match_score >= 0.7:
            match_quality = "strong" if match_score >= 0.8 else "moderate"
        else:
            match_quality = "none"
            match_score = 0.0

        if match_score > 0:
            coverage.append({
                "incumbent_weakness": weakness_area,
                "challenger_strength_score": round(match_score, 2),
                "match_quality": match_quality,
            })

    return coverage


def _compute_defensible_areas(
    incumbent_strengths: list[dict],
    challenger_profile: dict | None,
) -> list[dict]:
    """Identify incumbent strengths the challenger cannot match.

    Inverse of ``_compute_weakness_coverage``: instead of matching incumbent
    weaknesses to challenger capabilities, this matches incumbent *strengths*
    against challenger pain_addressed + strengths.  Areas with no overlap are
    defensible -- the rep should avoid attacking there.
    """
    if not incumbent_strengths:
        return []

    challenger_areas: set[str] = set()
    if challenger_profile:
        for item in (challenger_profile.get("pain_addressed") or []):
            if isinstance(item, dict):
                area = (item.get("area") or item.get("pain") or item.get("name") or "").lower().strip()
            else:
                area = str(item).lower().strip()
            if area:
                challenger_areas.add(area)
        for item in (challenger_profile.get("strengths") or []):
            if isinstance(item, dict):
                area = (item.get("aspect") or item.get("area") or item.get("name") or "").lower().strip()
            else:
                area = str(item).lower().strip()
            if area:
                challenger_areas.add(area)

    defensible: list[dict] = []
    for s in incumbent_strengths:
        if not isinstance(s, dict):
            continue
        strength_area = (s.get("area") or s.get("strength") or "").lower().strip()
        if not strength_area:
            continue

        matched = False
        for ca in challenger_areas:
            if ca == strength_area or ca in strength_area or strength_area in ca:
                matched = True
                break

        if not matched:
            defensible.append({
                "incumbent_strength": strength_area,
                "mention_count": int(s.get("count") or s.get("mention_count") or 0),
                "customer_quote": s.get("customer_quote") or "",
            })

    defensible.sort(key=lambda d: d["mention_count"], reverse=True)
    return defensible


def _filter_target_accounts(
    accounts_data: dict | None,
    challenger: str,
    *,
    max_accounts: int,
) -> tuple[list[dict], int, int]:
    """Filter accounts from accounts_in_motion that consider the challenger.

    Returns (target_accounts, total_count, considering_challenger_count).
    """
    if not accounts_data:
        return [], 0, 0

    accounts = accounts_data.get("accounts") or []
    if not accounts:
        return [], 0, 0

    total = len(accounts)
    challenger_lower = challenger.lower()
    # Fuzzy tokens for substring matching (e.g. "BigCommerce" matches "big commerce")
    challenger_tokens = challenger_lower.split()

    targets: list[dict] = []
    considering_count = 0

    for acct in accounts:
        alts = acct.get("alternatives_considering") or []
        considers = False
        for a in alts:
            if not isinstance(a, str):
                continue
            al = a.lower()
            # Exact match, substring in either direction, or token overlap
            if al == challenger_lower or challenger_lower in al or al in challenger_lower:
                considers = True
                break
            # Token overlap (e.g. "big commerce" matches "BigCommerce")
            if any(t in al for t in challenger_tokens if len(t) > 2):
                considers = True
                break
        if considers:
            considering_count += 1

        targets.append({
            "company": acct.get("company") or "",
            "opportunity_score": acct.get("opportunity_score") or 0,
            "urgency": float(acct.get("urgency") or 0),
            "buying_stage": acct.get("buying_stage") or "",
            "seat_count": acct.get("seat_count"),
            "contract_end": acct.get("contract_end"),
            "industry": acct.get("industry"),
            "domain": acct.get("domain"),
            "annual_revenue_range": acct.get("annual_revenue_range"),
            "top_quote": acct.get("top_quote"),
            "considers_challenger": considers,
        })

    # Sort by score DESC
    targets.sort(key=lambda a: a.get("opportunity_score", 0), reverse=True)
    targets = targets[:max_accounts]

    return targets, total, considering_count


def _reasoning_int(value: Any) -> int | None:
    """Unwrap a traced numeric contract field into an integer."""
    raw = value.get("value") if isinstance(value, dict) else value
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None


def _account_reasoning_summary_payload(
    account_reasoning: dict[str, Any] | None,
) -> tuple[str, dict[str, int]]:
    """Derive readable summary fields from account reasoning."""
    if not isinstance(account_reasoning, dict):
        return "", {}
    summary = str(account_reasoning.get("market_summary") or "").strip()
    metrics: dict[str, int] = {}
    for key in ("total_accounts", "high_intent_count", "active_eval_count"):
        value = _reasoning_int(account_reasoning.get(key))
        if value is not None:
            metrics[key] = value
    return summary, metrics


def _fallback_target_accounts_from_reasoning(
    account_reasoning: dict[str, Any] | None,
    *,
    max_accounts: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Build lightweight target accounts from account reasoning when report rows are missing."""
    if not isinstance(account_reasoning, dict):
        return [], 0, 0
    targets: list[dict[str, Any]] = []
    for item in account_reasoning.get("top_accounts") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        try:
            intent_score = float(item.get("intent_score") or 0)
        except (TypeError, ValueError):
            intent_score = 0.0
        targets.append({
            "company": name,
            "opportunity_score": max(0, min(100, round(intent_score * 100))),
            "urgency": round(max(0.0, min(10.0, intent_score * 10)), 1),
            "buying_stage": "",
            "seat_count": None,
            "contract_end": None,
            "industry": None,
            "domain": None,
            "annual_revenue_range": None,
            "top_quote": None,
            "considers_challenger": False,
            "reasoning_backed": True,
            "source_id": item.get("source_id"),
        })
        if len(targets) >= max_accounts:
            break
    total_accounts = _reasoning_int(account_reasoning.get("total_accounts"))
    return targets, total_accounts if total_accounts is not None else len(targets), 0


def _fallback_target_accounts_from_battle_card(
    battle_card: dict[str, Any] | None,
    *,
    max_accounts: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Build lightweight target accounts from battle-card account signals."""
    if not isinstance(battle_card, dict):
        return [], 0, 0
    targets_by_company: dict[str, dict[str, Any]] = {}

    for item in battle_card.get("high_intent_companies") or []:
        if not isinstance(item, dict):
            continue
        company = str(item.get("company") or "").strip()
        if not company:
            continue
        key = company.casefold()
        try:
            urgency = float(item.get("urgency") or 0)
        except (TypeError, ValueError):
            urgency = 0.0
        targets_by_company[key] = {
            "company": company,
            "opportunity_score": max(0, min(100, round(urgency * 10))),
            "urgency": round(max(0.0, min(10.0, urgency)), 1),
            "buying_stage": item.get("buying_stage") or item.get("stage") or "",
            "seat_count": item.get("seat_count"),
            "contract_end": item.get("contract_end"),
            "industry": item.get("industry"),
            "domain": item.get("domain"),
            "annual_revenue_range": item.get("annual_revenue_range"),
            "top_quote": item.get("top_quote"),
            "considers_challenger": False,
            "battle_card_backed": True,
        }

    for item in battle_card.get("active_evaluation_deadlines") or []:
        if not isinstance(item, dict):
            continue
        company = str(item.get("company") or "").strip()
        if not company:
            continue
        key = company.casefold()
        try:
            urgency = float(item.get("urgency") or 0)
        except (TypeError, ValueError):
            urgency = 0.0
        target = targets_by_company.get(key, {
            "company": company,
            "opportunity_score": max(0, min(100, round(urgency * 10))),
            "urgency": round(max(0.0, min(10.0, urgency)), 1),
            "buying_stage": item.get("buying_stage") or "",
            "seat_count": item.get("seat_count"),
            "contract_end": item.get("contract_end"),
            "industry": item.get("industry"),
            "domain": item.get("domain"),
            "annual_revenue_range": item.get("annual_revenue_range"),
            "top_quote": item.get("top_quote"),
            "considers_challenger": False,
            "battle_card_backed": True,
        })
        if not target.get("contract_end") and item.get("contract_end"):
            target["contract_end"] = item.get("contract_end")
        if not target.get("buying_stage") and item.get("buying_stage"):
            target["buying_stage"] = item.get("buying_stage")
        if not target.get("top_quote") and item.get("top_quote"):
            target["top_quote"] = item.get("top_quote")
        target["opportunity_score"] = max(
            int(target.get("opportunity_score") or 0),
            max(0, min(100, round(urgency * 10))),
        )
        target["urgency"] = max(float(target.get("urgency") or 0), round(max(0.0, min(10.0, urgency)), 1))
        targets_by_company[key] = target

    targets = sorted(
        targets_by_company.values(),
        key=lambda item: (float(item.get("urgency") or 0), int(item.get("opportunity_score") or 0)),
        reverse=True,
    )
    return targets[:max_accounts], len(targets), 0


def _challenger_playbook_primary_segment(
    segment_playbook: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(segment_playbook, dict):
        return {}
    priority = segment_playbook.get("priority_segments") or []
    for item in priority:
        if isinstance(item, dict) and (
            item.get("segment") or item.get("best_opening_angle") or item.get("why_vulnerable")
        ):
            return item
    return {}


def _fallback_sales_playbook(
    *,
    incumbent: str,
    challenger: str,
    battle_card: dict[str, Any] | None,
    segment_playbook: dict[str, Any] | None,
    timing_summary: str,
    displacement_detail: dict[str, Any],
    head_to_head: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic sales playbook fields when persisted seller copy is empty."""
    existing = {
        "discovery_questions": [],
        "landmine_questions": [],
        "objection_handlers": [],
        "talk_track": {},
        "recommended_plays": [],
    }
    if isinstance(battle_card, dict):
        existing = {
            "discovery_questions": battle_card.get("discovery_questions") or [],
            "landmine_questions": battle_card.get("landmine_questions") or [],
            "objection_handlers": battle_card.get("objection_handlers") or [],
            "talk_track": battle_card.get("talk_track") or battle_card.get("elevator_pitch") or {},
            "recommended_plays": battle_card.get("recommended_plays") or [],
        }
    discovery = existing["discovery_questions"] if isinstance(existing["discovery_questions"], list) else []
    landmine = existing["landmine_questions"] if isinstance(existing["landmine_questions"], list) else []
    objections = existing["objection_handlers"] if isinstance(existing["objection_handlers"], list) else []
    plays = existing["recommended_plays"] if isinstance(existing["recommended_plays"], list) else []
    talk_track = existing["talk_track"] if isinstance(existing["talk_track"], dict) else {}

    primary_segment = _challenger_playbook_primary_segment(segment_playbook if isinstance(segment_playbook, dict) else {})
    segment_name = str(primary_segment.get("segment") or "the highest-friction evaluation segment").strip()
    opening_angle = str(primary_segment.get("best_opening_angle") or "a cleaner alternative with less operational drag").strip()
    why_vulnerable = str(primary_segment.get("why_vulnerable") or displacement_detail.get("primary_driver") or "").strip()
    disqualifier = str(primary_segment.get("disqualifier") or "").strip()
    battle_conclusion = str(head_to_head.get("conclusion") or "").strip()
    timing_text = timing_summary.strip() or "Use this motion when active evaluation or renewal pressure is visible."

    if not plays:
        play: dict[str, Any] = {
            "play": f"Target {segment_name} with {opening_angle}.",
            "target_segment": segment_name,
            "key_message": why_vulnerable or f"Show why {challenger} is the cleaner alternative to {incumbent}.",
            "timing": timing_text,
        }
        secondary_play = {
            "play": f"Run a side-by-side benchmark against {incumbent} before the next decision checkpoint.",
            "target_segment": "active evaluation accounts",
            "key_message": battle_conclusion or f"Give the buyer a concrete comparison between {challenger} and {incumbent}.",
            "timing": "Before the next stakeholder review on fit, cost, or migration timing.",
        }
        plays = [play, secondary_play]

    if not discovery:
        discovery = [
            f"What is driving this team to re-evaluate {incumbent} right now in {segment_name}?",
            f"Which current workflow or cost issue would make {challenger} worth a formal benchmark?",
        ]

    if not landmine:
        landmine = [
            f"What happens if the current issues with {incumbent} remain unresolved through the next renewal or budget cycle?",
            (
                f"Where would {challenger} still be a bad fit for this account?"
                if not disqualifier else
                f"Would this account be disqualified by the current constraint: {disqualifier}?"
            ),
        ]

    if not objections:
        objections = [{
            "objection": f"We already know how to operate {incumbent}.",
            "acknowledge": "Operational familiarity is real, and switching always has a cost.",
            "pivot": (
                f"The better comparison is whether the current team can keep absorbing "
                f"{why_vulnerable or 'the recurring friction'} without a clearer alternative."
            ),
            "proof_point": battle_conclusion or timing_text,
        }]

    if not talk_track or not all(str(talk_track.get(key) or "").strip() for key in ("opening", "mid_call_pivot", "closing")):
        talk_track = {
            "opening": (
                battle_conclusion
                or f"Teams are re-evaluating {incumbent} where {segment_name} is exposed to the current friction."
            ),
            "mid_call_pivot": (
                f"Once that pain is confirmed, move to a concrete benchmark around {opening_angle.lower()}."
            ),
            "closing": (
                f"Close on a short working session before the next decision checkpoint. {timing_text}"
            ),
        }

    return {
        "discovery_questions": discovery,
        "landmine_questions": landmine,
        "objection_handlers": objections,
        "talk_track": talk_track,
        "recommended_plays": plays,
    }


def _build_integration_comparison(
    incumbent_profile: dict | None,
    challenger_profile: dict | None,
) -> dict:
    """Compute shared/exclusive integration sets between two vendors."""
    inc_integrations: set[str] = set()
    chal_integrations: set[str] = set()

    if incumbent_profile:
        for item in (incumbent_profile.get("top_integrations") or []):
            name = item if isinstance(item, str) else (item.get("name") or "") if isinstance(item, dict) else str(item)
            if name:
                inc_integrations.add(name)

    if challenger_profile:
        for item in (challenger_profile.get("top_integrations") or []):
            name = item if isinstance(item, str) else (item.get("name") or "") if isinstance(item, dict) else str(item)
            if name:
                chal_integrations.add(name)

    return {
        "shared": sorted(inc_integrations & chal_integrations),
        "challenger_exclusive": sorted(chal_integrations - inc_integrations),
        "incumbent_exclusive": sorted(inc_integrations - chal_integrations),
    }


def _build_challenger_brief(
    *,
    incumbent: str,
    challenger: str,
    displacement_detail: dict,
    battle_card: dict | None,
    accounts_in_motion: dict | None,
    incumbent_profile: dict | None,
    challenger_profile: dict | None,
    incumbent_evidence_vault: dict | None = None,
    churn_signal: dict | None,
    incumbent_synthesis_view: Any | None = None,
    cross_vendor_battle: dict | None,
    review_pain_quotes: list[dict] | None = None,
    battle_card_metadata: dict[str, Any] | None = None,
    quote_similarity_threshold: float | None = None,
    max_target_accounts: int,
    requested_as_of: date | None = None,
) -> dict:
    """Assemble the full challenger brief from pre-computed artifacts."""

    # Category (best effort from profiles)
    category = ""
    if incumbent_profile:
        category = incumbent_profile.get("category") or ""
    if not category and challenger_profile:
        category = challenger_profile.get("category") or ""

    # --- displacement_summary ---
    displacement_summary = {
        "total_mentions": displacement_detail.get("total_mentions", 0),
        "signal_strength": displacement_detail.get("signal_strength", "none"),
        "confidence_score": displacement_detail.get("confidence_score", 0),
        "primary_driver": displacement_detail.get("primary_driver"),
        "key_quote": displacement_detail.get("key_quote"),
        "source_distribution": displacement_detail.get("source_distribution", {}),
    }

    # --- incumbent_profile section ---
    # Battle card field names: vendor_weaknesses, customer_pain_quotes,
    # objection_data (contains dm_churn_rate, price_complaint_rate,
    # sentiment_direction, churn_pressure_score via parent card).
    inc_weaknesses = []
    inc_pain_quotes = []
    bc_churn_pressure_score = None
    bc_sentiment_direction = None
    bc_price_complaint_rate = None
    bc_dm_churn_rate = None
    bc_risk_level = None
    used_evidence_vault = False
    if battle_card:
        inc_weaknesses = battle_card.get("vendor_weaknesses") or []
        inc_pain_quotes = battle_card.get("customer_pain_quotes") or []
        bc_churn_pressure_score = battle_card.get("churn_pressure_score")
        bc_risk_level = battle_card.get("risk_level")
        obj_data = battle_card.get("objection_data") or {}
        bc_sentiment_direction = obj_data.get("sentiment_direction")
        bc_price_complaint_rate = obj_data.get("price_complaint_rate")
        bc_dm_churn_rate = obj_data.get("dm_churn_rate")

    if not inc_weaknesses and incumbent_evidence_vault:
        inc_weaknesses = _build_challenger_vault_weaknesses(
            incumbent_evidence_vault,
            limit=10,
        )
        used_evidence_vault = bool(inc_weaknesses) or used_evidence_vault

    if not inc_weaknesses and incumbent_profile:
        inc_weaknesses = _normalize_product_profile_weaknesses(
            incumbent_profile.get("weaknesses") or [],
        )

    similarity_threshold = (
        float(quote_similarity_threshold)
        if quote_similarity_threshold is not None else
        float(settings.b2b_churn.challenger_brief_quote_similarity_threshold)
    )
    if not inc_pain_quotes and incumbent_evidence_vault:
        inc_pain_quotes = _build_challenger_vault_pain_quotes(
            incumbent_evidence_vault,
            max_quotes=5,
            similarity_threshold=similarity_threshold,
        )
        used_evidence_vault = bool(inc_pain_quotes) or used_evidence_vault

    # Fallback chain for pain quotes:
    # 1. Battle card quotes (already handled above)
    # 2. Canonical evidence vault quotes
    # 3. Direct review quotes (from b2b_reviews)
    # 4. Displacement key_quote (single quote, last resort)
    if not inc_pain_quotes and review_pain_quotes:
        inc_pain_quotes = review_pain_quotes
    if not inc_pain_quotes and displacement_detail.get("key_quote"):
        inc_pain_quotes = [{
            "quote": displacement_detail["key_quote"],
            "source_site": "displacement_evidence",
        }]

    # --- incumbent strengths (from evidence vault) ---
    inc_strengths: list[dict[str, Any]] = []
    if incumbent_evidence_vault:
        inc_strengths = _build_challenger_vault_strengths(
            incumbent_evidence_vault,
            limit=5,
        )
        used_evidence_vault = bool(inc_strengths) or used_evidence_vault

    # Prefer churn_signal data, fall back to battle card objection_data.
    # Use explicit ``is not None`` checks so legitimate 0 / 0.0 values
    # are preserved instead of falling through to the battle-card fallback.
    def _coalesce(primary, fallback):
        return primary if primary is not None else fallback

    cs = churn_signal or {}
    vault_snapshot = (
        (incumbent_evidence_vault.get("metric_snapshot") or {})
        if isinstance(incumbent_evidence_vault, dict) else {}
    )
    incumbent_section = {
        "archetype": cs.get("archetype"),
        "archetype_confidence": cs.get("archetype_confidence"),
        "churn_pressure_score": _coalesce(cs.get("churn_pressure_score"), bc_churn_pressure_score),
        "risk_level": _coalesce(cs.get("risk_level"), bc_risk_level),
        "key_signals": cs.get("key_signals") or [],
        "top_weaknesses": inc_weaknesses[:10],
        "top_strengths": inc_strengths[:5],
        "top_pain_quotes": inc_pain_quotes[:5],
        "price_complaint_rate": _coalesce(cs.get("price_complaint_rate"), bc_price_complaint_rate),
        "dm_churn_rate": _coalesce(cs.get("dm_churn_rate"), bc_dm_churn_rate),
        "sentiment_direction": _coalesce(cs.get("sentiment_direction"), bc_sentiment_direction),
    }
    _rr = vault_snapshot.get("recommend_ratio")
    if _rr is not None:
        try:
            incumbent_section["recommend_ratio"] = round(float(_rr), 2)
        except (TypeError, ValueError):
            pass

    vendor_core_reasoning: dict[str, Any] = {}
    displacement_reasoning: dict[str, Any] = {}
    account_reasoning: dict[str, Any] = {}
    reasoning_contracts: dict[str, Any] = {}
    segment_playbook: dict[str, Any] = {}
    reasoning_contract_gaps: list[str] = []
    evidence_window: dict[str, Any] = {}
    evidence_window_days: int | None = None
    account_pressure_summary = ""
    account_pressure_metrics: dict[str, int] = {}
    timing_summary = ""
    timing_metrics: dict[str, Any] = {}
    priority_timing_triggers: list[str] = []
    synthesis_wedge = ""
    synthesis_wedge_label = ""
    reasoning_anchor_examples: dict[str, list[dict[str, Any]]] = {}
    reasoning_witness_highlights: list[dict[str, Any]] = []
    reasoning_reference_ids: dict[str, Any] = {}
    if incumbent_synthesis_view is not None:
        from ._b2b_synthesis_reader import (
            inject_synthesis_freshness,
        )
        consumer_context = incumbent_synthesis_view.filtered_consumer_context("challenger_brief")
        reasoning_contracts = consumer_context.get("reasoning_contracts") or {}
        vendor_core_reasoning = (
            consumer_context.get("vendor_core_reasoning")
            if isinstance(consumer_context.get("vendor_core_reasoning"), dict)
            else {}
        )
        displacement_reasoning = (
            consumer_context.get("displacement_reasoning")
            if isinstance(consumer_context.get("displacement_reasoning"), dict)
            else {}
        )
        account_reasoning = (
            consumer_context.get("account_reasoning")
            if isinstance(consumer_context.get("account_reasoning"), dict)
            else {}
        )
        reasoning_anchor_examples = (
            consumer_context.get("anchor_examples")
            if isinstance(consumer_context.get("anchor_examples"), dict)
            else {}
        )
        reasoning_witness_highlights = (
            consumer_context.get("witness_highlights")
            if isinstance(consumer_context.get("witness_highlights"), list)
            else []
        )
        reasoning_reference_ids = (
            consumer_context.get("reference_ids")
            if isinstance(consumer_context.get("reference_ids"), dict)
            else {}
        )
        if isinstance(vendor_core_reasoning, dict):
            segment_playbook = (
                vendor_core_reasoning.get("segment_playbook")
                if isinstance(vendor_core_reasoning.get("segment_playbook"), dict)
                else {}
            )
        evidence_window = incumbent_synthesis_view.meta
        synthesis_wedge = (
            incumbent_synthesis_view.primary_wedge.value
            if incumbent_synthesis_view.primary_wedge else ""
        )
        synthesis_wedge_label = incumbent_synthesis_view.wedge_label or ""
        if evidence_window:
            ew_start = evidence_window.get("evidence_window_start")
            ew_end = evidence_window.get("evidence_window_end")
            if ew_start and ew_end:
                try:
                    evidence_window_days = (
                        date.fromisoformat(str(ew_end)[:10])
                        - date.fromisoformat(str(ew_start)[:10])
                    ).days
                except (TypeError, ValueError):
                    evidence_window_days = None

        if vendor_core_reasoning:
            if not incumbent_section.get("key_signals"):
                cn = incumbent_synthesis_view.section("causal_narrative")
                fallback_signals = [
                    value for value in (
                        cn.get("trigger") if isinstance(cn, dict) else None,
                        cn.get("why_now") if isinstance(cn, dict) else None,
                    )
                    if value
                ]
                if fallback_signals:
                    incumbent_section["key_signals"] = fallback_signals
        account_pressure_summary, account_pressure_metrics = (
            _account_reasoning_summary_payload(account_reasoning)
        )
        if account_pressure_summary:
            incumbent_section["account_pressure_summary"] = account_pressure_summary
        if account_pressure_metrics:
            incumbent_section["account_pressure_metrics"] = account_pressure_metrics
        timing_intelligence = (
            vendor_core_reasoning.get("timing_intelligence")
            if isinstance(vendor_core_reasoning.get("timing_intelligence"), dict)
            else {}
        )
        timing_summary, timing_metrics, priority_timing_triggers = (
            _timing_summary_payload(timing_intelligence)
        )
        if timing_intelligence:
            incumbent_section["timing_intelligence"] = timing_intelligence
        if timing_summary:
            incumbent_section["timing_summary"] = timing_summary
        if timing_metrics:
            incumbent_section["timing_metrics"] = timing_metrics
        if priority_timing_triggers:
            incumbent_section["priority_timing_triggers"] = priority_timing_triggers
        contract_gaps = consumer_context.get("reasoning_contract_gaps") or []
        reasoning_contract_gaps = contract_gaps
        if contract_gaps:
            incumbent_section["reasoning_contract_gaps"] = contract_gaps
        section_disclaimers = consumer_context.get("reasoning_section_disclaimers")
        if isinstance(section_disclaimers, dict) and section_disclaimers:
            incumbent_section["reasoning_section_disclaimers"] = section_disclaimers
        if reasoning_anchor_examples:
            incumbent_section["reasoning_anchor_examples"] = reasoning_anchor_examples
        if reasoning_witness_highlights:
            incumbent_section["reasoning_witness_highlights"] = reasoning_witness_highlights
        if reasoning_reference_ids:
            incumbent_section["reasoning_reference_ids"] = reasoning_reference_ids

        # Phase 3 governance fields
        wts = incumbent_synthesis_view.why_they_stay
        if wts:
            incumbent_section["why_they_stay"] = wts
        cp = incumbent_synthesis_view.confidence_posture
        if cp:
            incumbent_section["confidence_posture"] = cp
            limits = cp.get("limits")
            if limits:
                incumbent_section["confidence_limits"] = limits
        st = incumbent_synthesis_view.switch_triggers
        if st:
            incumbent_section["switch_triggers"] = st
        eg = incumbent_synthesis_view.evidence_governance
        if eg:
            cg = eg.get("coverage_gaps")
            if isinstance(cg, list) and cg:
                incumbent_section["coverage_gaps"] = cg

    # --- challenger_advantage ---
    challenger_strengths = []
    challenger_commonly_switched_from = []
    challenger_summary = ""
    if challenger_profile:
        challenger_strengths = challenger_profile.get("strengths") or []
        challenger_commonly_switched_from = challenger_profile.get("commonly_switched_from") or []
        challenger_summary = challenger_profile.get("profile_summary") or ""

    challenger_pain_addressed = (challenger_profile.get("pain_addressed") or []) if challenger_profile else []
    weakness_coverage = _compute_weakness_coverage(inc_weaknesses, challenger_pain_addressed)

    defensible_areas = _compute_defensible_areas(inc_strengths, challenger_profile)

    challenger_advantage = {
        "strengths": challenger_strengths[:10],
        "weakness_coverage": weakness_coverage,
        "defensible_areas": defensible_areas,
        "commonly_switched_from": challenger_commonly_switched_from[:10],
        "profile_summary": challenger_summary,
    }

    # --- head_to_head ---
    head_to_head: dict[str, Any] = {}
    if cross_vendor_battle:
        head_to_head = {
            "winner": cross_vendor_battle.get("winner") or "",
            "loser": cross_vendor_battle.get("loser") or "",
            "conclusion": cross_vendor_battle.get("conclusion") or "",
            "durability": cross_vendor_battle.get("durability") or "",
            "confidence": cross_vendor_battle.get("confidence"),
            "key_insights": cross_vendor_battle.get("key_insights") or [],
        }
        cross_vendor_reference_ids = cross_vendor_battle.get("reference_ids")
        if isinstance(cross_vendor_reference_ids, dict) and cross_vendor_reference_ids:
            head_to_head["reference_ids"] = cross_vendor_reference_ids
    else:
        # Synthesize from displacement data when no pairwise battle exists
        driver = displacement_detail.get("primary_driver") or "unknown"
        mentions = displacement_detail.get("total_mentions", 0)
        signal = displacement_detail.get("signal_strength", "none")
        confidence = displacement_detail.get("confidence_score", 0)
        insights: list[dict[str, str]] = []
        if driver and driver != "unknown":
            insights.append({"insight": "Primary displacement driver: %s" % driver, "evidence": driver})
        if mentions:
            insights.append({"insight": "%d displacement mentions across sources" % mentions, "evidence": "%d mentions" % mentions})
        # Add archetype context if available
        archetype = (churn_signal or {}).get("archetype")
        if archetype:
            insights.append({"insight": "%s classified as %s archetype" % (incumbent, archetype), "evidence": archetype})
        # Add top weakness from incumbent profile
        if inc_weaknesses:
            top_w = inc_weaknesses[0]
            area = top_w.get("area") or top_w.get("weakness") or ""
            count = top_w.get("count") or top_w.get("evidence_count") or 0
            if area:
                insights.append({"insight": "Top incumbent weakness: %s" % area, "evidence": "%d mentions" % count if count else "product profile"})
        head_to_head = {
            "winner": challenger if mentions > 0 and signal in ("strong", "moderate", "emerging") else "",
            "loser": incumbent if mentions > 0 and signal in ("strong", "moderate", "emerging") else "",
            "conclusion": "%s displacing %s driven by %s (%d mentions, %s signal)." % (
                challenger, incumbent, driver, mentions, signal,
            ),
            "durability": "uncertain",
            "confidence": confidence,
            "key_insights": insights,
            "synthesized": True,
        }
    if not (
        isinstance(head_to_head.get("reference_ids"), dict)
        and head_to_head.get("reference_ids")
    ):
        fallback_reference_ids = _reference_ids_from_displacement_detail(displacement_detail)
        if fallback_reference_ids:
            head_to_head["reference_ids"] = fallback_reference_ids

    category_council: dict[str, Any] = {}
    if accounts_in_motion and isinstance(accounts_in_motion.get("category_council"), dict):
        src = accounts_in_motion.get("category_council") or {}
        if any(
            [
                src.get("winner"),
                src.get("loser"),
                src.get("conclusion"),
                src.get("market_regime"),
                src.get("key_insights"),
            ]
        ):
            category_council = {
                "winner": src.get("winner") or "",
                "loser": src.get("loser") or "",
                "conclusion": src.get("conclusion") or "",
                "market_regime": src.get("market_regime") or "",
                "durability": src.get("durability") or "",
                "confidence": src.get("confidence"),
                "key_insights": src.get("key_insights") or [],
            }

    # --- target_accounts ---
    target_accounts_source = ""
    target_accounts, total_target, considering_count = _filter_target_accounts(
        accounts_in_motion, challenger, max_accounts=max_target_accounts,
    )
    if target_accounts:
        target_accounts_source = "accounts_in_motion"
    elif account_reasoning:
        target_accounts, total_target, considering_count = (
            _fallback_target_accounts_from_reasoning(
                account_reasoning,
                max_accounts=max_target_accounts,
            )
        )
        if target_accounts or total_target:
            target_accounts_source = "account_reasoning"
    elif battle_card:
        target_accounts, total_target, considering_count = (
            _fallback_target_accounts_from_battle_card(
                battle_card,
                max_accounts=max_target_accounts,
            )
        )
        if target_accounts or total_target:
            target_accounts_source = "battle_card"

    # --- sales_playbook (from battle card if available, fallback to segment_playbook) ---
    # LLM-enriched fields are added directly to the card dict (not nested).
    sales_playbook: dict[str, Any] = {}
    if battle_card:
        sales_playbook = {
            "discovery_questions": battle_card.get("discovery_questions") or [],
            "landmine_questions": battle_card.get("landmine_questions") or [],
            "objection_handlers": battle_card.get("objection_handlers") or [],
            "talk_track": battle_card.get("talk_track") or battle_card.get("elevator_pitch") or "",
            "recommended_plays": battle_card.get("recommended_plays") or [],
        }
    if (
        not sales_playbook
        or not any(
            sales_playbook.get(key)
            for key in (
                "discovery_questions",
                "landmine_questions",
                "objection_handlers",
                "recommended_plays",
            )
        )
        or not (
            isinstance(sales_playbook.get("talk_track"), dict)
            and any(str(sales_playbook["talk_track"].get(key) or "").strip() for key in ("opening", "mid_call_pivot", "closing"))
        )
    ):
        sales_playbook = _fallback_sales_playbook(
            incumbent=incumbent,
            challenger=challenger,
            battle_card=battle_card,
            segment_playbook=segment_playbook,
            timing_summary=timing_summary,
            displacement_detail=displacement_detail,
            head_to_head=head_to_head,
        )

    # --- integration_comparison ---
    integration_comparison = _build_integration_comparison(
        incumbent_profile, challenger_profile,
    )

    used_reasoning_synthesis = bool(
        vendor_core_reasoning
        or displacement_reasoning
        or account_reasoning
        or synthesis_wedge
        or evidence_window
    )

    # --- data_sources ---
    data_sources = {
        "battle_card": battle_card is not None,
        "reasoning_synthesis": used_reasoning_synthesis,
        "evidence_vault": used_evidence_vault,
        "accounts_in_motion": accounts_in_motion is not None,
        "account_reasoning": bool(account_reasoning),
        "product_profiles": (incumbent_profile is not None or challenger_profile is not None),
        "cross_vendor_conclusion": cross_vendor_battle is not None,
        "review_quotes": bool(review_pain_quotes),
    }

    # --- executive summary ---
    mentions = displacement_summary["total_mentions"]
    exec_summary = (
        f"Challenger brief: {challenger} vs {incumbent}. "
        f"{mentions} displacement mentions, "
        f"{total_target} target accounts ({considering_count} considering {challenger})."
    )
    if account_pressure_summary:
        exec_summary = "%s %s" % (exec_summary, account_pressure_summary)
    if timing_summary:
        exec_summary = "%s %s" % (exec_summary, timing_summary)
    if target_accounts_source == "account_reasoning":
        exec_summary = "%s Target list uses reasoning-backed account fallback." % exec_summary

    result = {
        "incumbent": incumbent,
        "challenger": challenger,
        "category": category,
        "report_date": str(date.today()),
        "displacement_summary": displacement_summary,
        "incumbent_profile": incumbent_section,
        "challenger_advantage": challenger_advantage,
        "head_to_head": head_to_head,
        "target_accounts": target_accounts,
        "total_target_accounts": total_target,
        "accounts_considering_challenger": considering_count,
        "sales_playbook": sales_playbook,
        "integration_comparison": integration_comparison,
        "data_sources": data_sources,
        "_executive_summary": exec_summary,
    }
    if category_council:
        result["category_council"] = category_council
    if target_accounts_source:
        result["target_accounts_source"] = target_accounts_source
    if reasoning_contracts:
        result["reasoning_contracts"] = reasoning_contracts
    if reasoning_anchor_examples:
        result["reasoning_anchor_examples"] = reasoning_anchor_examples
    if reasoning_witness_highlights:
        result["reasoning_witness_highlights"] = reasoning_witness_highlights
    if reasoning_reference_ids:
        result["reasoning_reference_ids"] = reasoning_reference_ids
    if account_reasoning:
        result["account_reasoning"] = account_reasoning
    if timing_summary or timing_metrics or priority_timing_triggers:
        timing_intelligence = (
            vendor_core_reasoning.get("timing_intelligence")
            if isinstance(vendor_core_reasoning.get("timing_intelligence"), dict)
            else {}
        )
        if timing_intelligence:
            result["timing_intelligence"] = timing_intelligence
        if timing_summary:
            result["timing_summary"] = timing_summary
        if timing_metrics:
            result["timing_metrics"] = timing_metrics
        if priority_timing_triggers:
            result["priority_timing_triggers"] = priority_timing_triggers
    if segment_playbook:
        result["segment_playbook"] = segment_playbook
        timing_intelligence = (
            vendor_core_reasoning.get("timing_intelligence")
            if isinstance(vendor_core_reasoning.get("timing_intelligence"), dict)
            else None
        )
        targeting_summary = _segment_targeting_summary(
            segment_playbook,
            timing_intelligence,
        )
        if targeting_summary:
            result["segment_targeting_summary"] = targeting_summary
    if synthesis_wedge:
        result["synthesis_wedge"] = synthesis_wedge
        result["synthesis_wedge_label"] = synthesis_wedge_label
    if evidence_window:
        result["evidence_window"] = evidence_window
    if evidence_window_days is not None:
        result["evidence_window_days"] = evidence_window_days
    if incumbent_synthesis_view is not None:
        result["synthesis_schema_version"] = incumbent_synthesis_view.schema_version
    if used_reasoning_synthesis:
        result["reasoning_source"] = "b2b_reasoning_synthesis"
        inject_synthesis_freshness(
            result,
            incumbent_synthesis_view,
            requested_as_of=requested_as_of,
        )
    if reasoning_contract_gaps:
        result["reasoning_contract_gaps"] = reasoning_contract_gaps
    if used_reasoning_synthesis:
        section_disclaimers = (
            incumbent_section.get("reasoning_section_disclaimers")
            if isinstance(incumbent_section, dict)
            else None
        )
        if isinstance(section_disclaimers, dict) and section_disclaimers:
            result["reasoning_section_disclaimers"] = section_disclaimers
    if battle_card_metadata:
        if battle_card_metadata.get("battle_card_date"):
            result["battle_card_date"] = battle_card_metadata["battle_card_date"]
        if battle_card_metadata.get("battle_card_stale") is not None:
            result["battle_card_stale"] = bool(battle_card_metadata["battle_card_stale"])
    return result


# ---------------------------------------------------------------------------
# Main task entry point
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict[str, Any]:
    """Build per-(incumbent, challenger) pair challenger briefs."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    today = await _check_freshness(pool)
    if today is None:
        return {"_skip_synthesis": "Core signals not fresh for today"}

    from .b2b_churn_intelligence import _normalize_test_vendors

    window_days = cfg.intelligence_window_days
    min_mentions = cfg.challenger_brief_min_displacement_mentions
    max_per_incumbent = cfg.challenger_brief_max_pairs_per_incumbent
    max_target_accounts = cfg.challenger_brief_max_target_accounts
    fallback_days = cfg.challenger_brief_report_fallback_days
    quote_limit = cfg.challenger_brief_quote_fallback_limit
    quote_candidate_limit = cfg.challenger_brief_quote_candidate_limit
    quote_similarity_threshold = cfg.challenger_brief_quote_similarity_threshold
    min_quote_urgency = cfg.quotable_phrase_min_urgency
    scoped_vendors = _normalize_test_vendors((task.metadata or {}).get("test_vendors"))
    vendor_scope = {vendor.lower() for vendor in scoped_vendors}

    # --- Phase 1: Select displacement pairs ---
    await _update_execution_progress(
        task,
        stage=_STAGE_SELECTING_PAIRS,
        progress_message="Selecting displacement pairs for challenger briefs.",
    )
    pairs = await _select_displacement_pairs(
        pool,
        min_mentions=min_mentions,
        max_per_incumbent=max_per_incumbent,
        window_days=window_days,
    )
    if vendor_scope:
        raw_pair_count = len(pairs)
        pairs = [
            pair for pair in pairs
            if str(pair.get("incumbent") or "").strip().lower() in vendor_scope
        ]
        logger.info(
            "Scoped challenger briefs to %d/%d pairs for vendors: %s",
            len(pairs), raw_pair_count, sorted(scoped_vendors),
        )
    if not pairs:
        logger.info("No displacement pairs above threshold, skipping")
        return {"_skip_synthesis": "No displacement pairs above threshold"}

    logger.info("Selected %d displacement pairs for challenger briefs", len(pairs))
    briefs_retired = await _retire_unselected_challenger_briefs(
        pool,
        today=today,
        pairs=pairs,
    )

    try:
        evidence_vault_lookup = await _fetch_latest_evidence_vault(
            pool,
            as_of=today,
            analysis_window_days=window_days,
        )
    except Exception:
        logger.warning("Failed to load evidence vault for challenger briefs", exc_info=True)
        evidence_vault_lookup = {}

    # Load merged cross-vendor reasoning (canonical synthesis first, legacy gaps filled).
    xv_lookup: dict[str, Any] = {"battles": {}, "councils": {}, "asymmetries": {}}
    try:
        from ._b2b_cross_vendor_synthesis import load_best_cross_vendor_lookup
        xv_lookup = await load_best_cross_vendor_lookup(
            pool, as_of=today, analysis_window_days=window_days,
        )
    except Exception:
        logger.debug("Cross-vendor lookup load failed", exc_info=True)

    # --- Phase 2: Fetch artifacts and build briefs ---
    persisted = 0
    await _update_execution_progress(
        task,
        stage=_STAGE_BUILDING_BRIEFS,
        progress_current=0,
        progress_total=len(pairs),
        progress_message="Building challenger briefs from persisted artifacts.",
        pairs=len(pairs),
        persisted=0,
    )
    for idx, pair in enumerate(pairs, start=1):
        incumbent = pair["incumbent"]
        challenger = pair["challenger"]

        try:
            # Parallel fetch of all artifacts for this pair
            (
                battle_card_record,
                accounts_in_motion,
                displacement_detail,
                incumbent_profile,
                challenger_profile,
                churn_signal,
                incumbent_synthesis_view,
                cross_vendor_battle,
                review_pain_quotes,
            ) = await asyncio.gather(
                _fetch_persisted_report_record(
                    pool, "battle_card", incumbent, today, fallback_days=fallback_days,
                ),
                _fetch_persisted_report(
                    pool, "accounts_in_motion", incumbent, today, fallback_days=fallback_days,
                ),
                _fetch_displacement_detail(pool, incumbent, challenger, window_days),
                _fetch_product_profile(pool, incumbent),
                _fetch_product_profile(pool, challenger),
                _fetch_churn_signal(pool, incumbent, today),
                _fetch_synthesis_view(pool, incumbent, today, window_days),
                _resolve_cross_vendor_battle(
                    pool, incumbent, challenger, today, xv_lookup,
                ),
                _fetch_review_pain_quotes(
                    pool,
                    incumbent,
                    window_days=window_days,
                    limit=quote_limit,
                    candidate_limit=quote_candidate_limit,
                    min_urgency=min_quote_urgency,
                    similarity_threshold=quote_similarity_threshold,
                ),
            )
            battle_card = battle_card_record["data"] if battle_card_record else None
            battle_card_metadata = {
                "battle_card_date": str(battle_card_record["report_date"]) if battle_card_record else "",
                "battle_card_stale": bool(battle_card_record["stale"]) if battle_card_record else False,
            }

            brief = _build_challenger_brief(
                incumbent=incumbent,
                challenger=challenger,
                displacement_detail=displacement_detail,
                battle_card=battle_card,
                accounts_in_motion=accounts_in_motion,
                incumbent_profile=incumbent_profile,
                challenger_profile=challenger_profile,
                incumbent_evidence_vault=evidence_vault_lookup.get(incumbent),
                churn_signal=churn_signal,
                incumbent_synthesis_view=incumbent_synthesis_view,
                cross_vendor_battle=cross_vendor_battle,
                review_pain_quotes=review_pain_quotes,
                battle_card_metadata=battle_card_metadata if battle_card_record else None,
                quote_similarity_threshold=quote_similarity_threshold,
                max_target_accounts=max_target_accounts,
                requested_as_of=today,
            )

            exec_summary = brief.pop("_executive_summary", "")
            total_mentions = brief["displacement_summary"]["total_mentions"]
            sources_present = sum(1 for v in brief["data_sources"].values() if v)

            execution_id = str(
                getattr(task, "id", None)
                or (getattr(task, "metadata", {}) or {}).get("_execution_id")
                or ""
            ) or None
            report_status = "published" if total_mentions > 0 else "failed"
            latest_failure_step = None if total_mentions > 0 else "no_data"
            latest_error_code = None if total_mentions > 0 else "no_data"
            latest_error_summary = None if total_mentions > 0 else "Challenger brief has no displacement mentions"
            sql = """
                INSERT INTO b2b_intelligence (
                    report_date, report_type, vendor_filter, category_filter,
                    intelligence_data, executive_summary, data_density, status, llm_model,
                    source_review_count, source_distribution,
                    latest_run_id, latest_attempt_no, latest_failure_step,
                    latest_error_code, latest_error_summary,
                    blocker_count, warning_count
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                    $12, $13, $14, $15, $16, $17, $18
                )
                ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')),
                             LOWER(COALESCE(category_filter,'')),
                             COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
                DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                              executive_summary = EXCLUDED.executive_summary,
                              data_density = EXCLUDED.data_density,
                              status = EXCLUDED.status,
                              source_review_count = EXCLUDED.source_review_count,
                              source_distribution = EXCLUDED.source_distribution,
                              latest_run_id = EXCLUDED.latest_run_id,
                              latest_attempt_no = EXCLUDED.latest_attempt_no,
                              latest_failure_step = EXCLUDED.latest_failure_step,
                              latest_error_code = EXCLUDED.latest_error_code,
                              latest_error_summary = EXCLUDED.latest_error_summary,
                              blocker_count = EXCLUDED.blocker_count,
                              warning_count = EXCLUDED.warning_count,
                              created_at = now()
                RETURNING id
            """
            sql_args = (
                today,
                "challenger_brief",
                incumbent,
                challenger,
                json.dumps(brief, default=str),
                exec_summary,
                json.dumps({
                    "total_mentions": total_mentions,
                    "sources_present": sources_present,
                    "target_accounts": brief["total_target_accounts"],
                }),
                report_status,
                "pipeline_deterministic",
                total_mentions,
                json.dumps(brief["displacement_summary"].get("source_distribution", {})),
                execution_id,
                1,
                latest_failure_step,
                latest_error_code,
                latest_error_summary,
                0 if total_mentions > 0 else 1,
                0,
            )
            fetchrow = getattr(pool, "fetchrow", None)
            report_row = await fetchrow(sql, *sql_args) if callable(fetchrow) else None
            if report_row is None:
                await pool.execute(sql.replace(" RETURNING id", ""), *sql_args)
            persisted += 1
            from ..visibility import record_attempt
            await record_attempt(
                pool,
                artifact_type="churn_report",
                artifact_id=str(report_row["id"]) if report_row else f"{today}:challenger_brief:{incumbent}:{challenger}",
                run_id=execution_id,
                stage="persistence",
                status="succeeded" if total_mentions > 0 else "failed",
                failure_step=latest_failure_step,
                error_message=latest_error_summary,
            )
            if total_mentions <= 0:
                from ..visibility import emit_event
                await emit_event(
                    pool,
                        stage="reports",
                        event_type="report_failed",
                        entity_type="churn_report",
                    entity_id=str(report_row["id"]) if report_row else f"{today}:challenger_brief:{incumbent}:{challenger}",
                    artifact_type="churn_report",
                    run_id=execution_id,
                    severity="warning",
                    actionable=False,
                    reason_code="no_data",
                    summary=f"challenger_brief produced no displacement mentions for {incumbent} vs {challenger}",
                    source_table="b2b_intelligence",
                    source_id=str(report_row["id"]) if report_row else None,
                )
        except Exception:
            logger.exception(
                "Failed to build/persist challenger brief for %s -> %s",
                incumbent, challenger,
            )
        await _update_execution_progress(
            task,
            stage=_STAGE_BUILDING_BRIEFS,
            progress_current=idx,
            progress_total=len(pairs),
            progress_message="Building challenger briefs from persisted artifacts.",
            pairs=len(pairs),
            persisted=persisted,
        )

    logger.info(
        "Challenger briefs: %d pairs, %d persisted, %d retired",
        len(pairs), persisted, briefs_retired,
    )
    await _update_execution_progress(
        task,
        stage=_STAGE_FINALIZING,
        progress_current=len(pairs),
        progress_total=len(pairs),
        progress_message="Finalizing challenger-brief execution status.",
        pairs=len(pairs),
        persisted=persisted,
    )

    return {
        "_skip_synthesis": "Challenger briefs complete",
        "pairs": len(pairs),
        "persisted": persisted,
        "briefs_retired": briefs_retired,
    }
