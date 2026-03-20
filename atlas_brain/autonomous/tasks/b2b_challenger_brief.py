"""Follow-up task: build per-(incumbent, challenger) pair challenger briefs.

Runs after all other B2B follow-up tasks. Reads persisted artifacts from
b2b_displacement_edges, b2b_intelligence (battle_card, accounts_in_motion),
b2b_product_profiles, b2b_cross_vendor_conclusions, and b2b_churn_signals.
Assembles one b2b_intelligence row per (incumbent, challenger) pair with
report_type='challenger_brief'.

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
from ._b2b_shared import _fetch_latest_evidence_vault
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
        SELECT mention_count, signal_strength, confidence_score,
               primary_driver, key_quote, source_distribution,
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

    return {
        "total_mentions": total_mentions,
        "signal_strength": signal_strength,
        "confidence_score": round(confidence_score, 2),
        "primary_driver": primary_driver,
        "key_quote": key_quote,
        "source_distribution": merged_sources,
    }


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

    Reads ``enrichment->'quotable_phrases'`` plus pain metadata from b2b_reviews.
    Used as fallback when no battle card quotes are available.
    """
    rows = await pool.fetch(
        """
        SELECT vendor_name, source, reviewer_company, reviewer_title,
               COALESCE(reviewer_title, enrichment->'reviewer_context'->>'role_level') AS role,
               enrichment->>'pain_category' AS pain_category,
               enrichment->'quotable_phrases' AS phrases,
               (enrichment->>'urgency_score')::float AS urgency
        FROM b2b_reviews
        WHERE LOWER(vendor_name) = LOWER($1)
          AND enrichment_status = 'enriched'
          AND enrichment->'quotable_phrases' IS NOT NULL
          AND jsonb_array_length(enrichment->'quotable_phrases') > 0
          AND (enrichment->>'urgency_score')::float >= $4
          AND imported_at > NOW() - make_interval(days => $2)
        ORDER BY (enrichment->>'urgency_score')::float DESC
        LIMIT $3
        """,
        vendor,
        window_days,
        candidate_limit,
        min_urgency,
    )
    normalized_rows = [{
        "vendor_name": row["vendor_name"],
        "source": row["source"],
        "company": row["reviewer_company"],
        "role": row["role"],
        "pain_category": row["pain_category"],
        "phrases": row["phrases"],
        "urgency": row["urgency"],
    } for row in rows]
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
               price_complaint_rate, decision_maker_churn_rate
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

    return {
        "archetype": row["archetype"],
        "archetype_confidence": float(row["archetype_confidence"]) if row["archetype_confidence"] is not None else None,
        "risk_level": row["reasoning_risk_level"],
        "key_signals": key_signals or [],
        "churn_pressure_score": None,  # populated from battle card
        "sentiment_direction": None,   # populated from battle card
        "price_complaint_rate": float(row["price_complaint_rate"]) if row["price_complaint_rate"] is not None else None,
        "dm_churn_rate": float(row["decision_maker_churn_rate"]) if row["decision_maker_churn_rate"] is not None else None,
    }


async def _fetch_cross_vendor_battle(
    pool, vendor_a: str, vendor_b: str, today: date,
) -> dict | None:
    """Fetch the pairwise battle conclusion between two vendors.

    The table schema:
    - vendors is TEXT[] (not JSONB), use @> ARRAY[...] for containment
    - computed_date (not analysis_date)
    - conclusion is JSONB containing winner, durability_assessment, key_insights
    """
    row = await pool.fetchrow(
        """
        SELECT conclusion, confidence
        FROM b2b_cross_vendor_conclusions
        WHERE analysis_type = 'pairwise_battle'
          AND computed_date <= $1
          AND vendors @> ARRAY[$2, $3]::text[]
        ORDER BY computed_date DESC, created_at DESC
        LIMIT 1
        """,
        today,
        vendor_a,
        vendor_b,
    )
    if not row:
        return None

    conclusion = row["conclusion"]
    if isinstance(conclusion, str):
        try:
            conclusion = json.loads(conclusion)
        except (json.JSONDecodeError, TypeError):
            conclusion = {}
    if not isinstance(conclusion, dict):
        conclusion = {}

    key_insights = conclusion.get("key_insights") or []
    if isinstance(key_insights, list):
        # Normalize: items can be strings or dicts with "insight"+"evidence"
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

    return {
        "conclusion": conclusion.get("conclusion") or "",
        "winner": conclusion.get("winner") or "",
        "loser": conclusion.get("loser") or "",
        "durability": conclusion.get("durability_assessment") or conclusion.get("displacement_flows_nature") or "",
        "key_insights": key_insights,
        "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
    }


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
    cross_vendor_battle: dict | None,
    review_pain_quotes: list[dict] | None = None,
    battle_card_metadata: dict[str, Any] | None = None,
    quote_similarity_threshold: float | None = None,
    max_target_accounts: int,
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

    # Prefer churn_signal data, fall back to battle card objection_data.
    # Use explicit ``is not None`` checks so legitimate 0 / 0.0 values
    # are preserved instead of falling through to the battle-card fallback.
    def _coalesce(primary, fallback):
        return primary if primary is not None else fallback

    cs = churn_signal or {}
    incumbent_section = {
        "archetype": cs.get("archetype"),
        "archetype_confidence": cs.get("archetype_confidence"),
        "churn_pressure_score": _coalesce(cs.get("churn_pressure_score"), bc_churn_pressure_score),
        "risk_level": _coalesce(cs.get("risk_level"), bc_risk_level),
        "key_signals": cs.get("key_signals") or [],
        "top_weaknesses": inc_weaknesses[:10],
        "top_pain_quotes": inc_pain_quotes[:5],
        "price_complaint_rate": _coalesce(cs.get("price_complaint_rate"), bc_price_complaint_rate),
        "dm_churn_rate": _coalesce(cs.get("dm_churn_rate"), bc_dm_churn_rate),
        "sentiment_direction": _coalesce(cs.get("sentiment_direction"), bc_sentiment_direction),
    }

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

    challenger_advantage = {
        "strengths": challenger_strengths[:10],
        "weakness_coverage": weakness_coverage,
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
    else:
        # Synthesize from displacement data when no pairwise battle exists
        driver = displacement_detail.get("primary_driver") or "unknown"
        mentions = displacement_detail.get("total_mentions", 0)
        signal = displacement_detail.get("signal_strength", "none")
        confidence = displacement_detail.get("confidence_score", 0)
        insights: list[dict[str, str]] = []
        if driver and driver != "unknown":
            insights.append({"insight": "Primary displacement driver: %s" % driver, "evidence": "displacement_detail.primary_driver"})
        if mentions:
            insights.append({"insight": "%d displacement mentions across sources" % mentions, "evidence": "total_mentions: %d" % mentions})
        # Add archetype context if available
        archetype = (churn_signal or {}).get("archetype")
        if archetype:
            insights.append({"insight": "%s classified as %s archetype" % (incumbent, archetype), "evidence": "stratified_reasoning.archetype"})
        # Add top weakness from incumbent profile
        if inc_weaknesses:
            top_w = inc_weaknesses[0]
            area = top_w.get("area") or top_w.get("weakness") or ""
            count = top_w.get("count") or top_w.get("evidence_count") or 0
            if area:
                insights.append({"insight": "Top incumbent weakness: %s" % area, "evidence": "%d mentions" % count if count else "product_profile"})
        head_to_head = {
            "winner": challenger if mentions >= 5 and signal in ("strong", "moderate") else "",
            "loser": incumbent if mentions >= 5 and signal in ("strong", "moderate") else "",
            "conclusion": "%s displacing %s driven by %s (%d mentions, %s signal)." % (
                challenger, incumbent, driver, mentions, signal,
            ),
            "durability": "uncertain",
            "confidence": confidence,
            "key_insights": insights,
            "synthesized": True,
        }

    # --- target_accounts ---
    target_accounts, total_target, considering_count = _filter_target_accounts(
        accounts_in_motion, challenger, max_accounts=max_target_accounts,
    )

    # --- sales_playbook (from battle card if available) ---
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

    # --- integration_comparison ---
    integration_comparison = _build_integration_comparison(
        incumbent_profile, challenger_profile,
    )

    # --- data_sources ---
    data_sources = {
        "battle_card": battle_card is not None,
        "evidence_vault": used_evidence_vault,
        "accounts_in_motion": accounts_in_motion is not None,
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

    window_days = cfg.intelligence_window_days
    min_mentions = cfg.challenger_brief_min_displacement_mentions
    max_per_incumbent = cfg.challenger_brief_max_pairs_per_incumbent
    max_target_accounts = cfg.challenger_brief_max_target_accounts
    fallback_days = cfg.challenger_brief_report_fallback_days
    quote_limit = cfg.challenger_brief_quote_fallback_limit
    quote_candidate_limit = cfg.challenger_brief_quote_candidate_limit
    quote_similarity_threshold = cfg.challenger_brief_quote_similarity_threshold
    min_quote_urgency = cfg.quotable_phrase_min_urgency

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
    if not pairs:
        logger.info("No displacement pairs above threshold, skipping")
        return {"_skip_synthesis": "No displacement pairs above threshold"}

    logger.info("Selected %d displacement pairs for challenger briefs", len(pairs))

    try:
        evidence_vault_lookup = await _fetch_latest_evidence_vault(
            pool,
            as_of=today,
            analysis_window_days=window_days,
        )
    except Exception:
        logger.warning("Failed to load evidence vault for challenger briefs", exc_info=True)
        evidence_vault_lookup = {}

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
                _fetch_cross_vendor_battle(pool, incumbent, challenger, today),
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
                cross_vendor_battle=cross_vendor_battle,
                review_pain_quotes=review_pain_quotes,
                battle_card_metadata=battle_card_metadata if battle_card_record else None,
                quote_similarity_threshold=quote_similarity_threshold,
                max_target_accounts=max_target_accounts,
            )

            exec_summary = brief.pop("_executive_summary", "")
            total_mentions = brief["displacement_summary"]["total_mentions"]
            sources_present = sum(1 for v in brief["data_sources"].values() if v)

            await pool.execute(
                """
                INSERT INTO b2b_intelligence (
                    report_date, report_type, vendor_filter, category_filter,
                    intelligence_data, executive_summary, data_density, status, llm_model,
                    source_review_count, source_distribution
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')),
                             LOWER(COALESCE(category_filter,'')),
                             COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
                DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                              executive_summary = EXCLUDED.executive_summary,
                              data_density = EXCLUDED.data_density,
                              source_review_count = EXCLUDED.source_review_count,
                              source_distribution = EXCLUDED.source_distribution,
                              created_at = now()
                """,
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
                "published",
                "pipeline_deterministic",
                total_mentions,
                json.dumps(brief["displacement_summary"].get("source_distribution", {})),
            )
            persisted += 1
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
        "Challenger briefs: %d pairs, %d persisted",
        len(pairs), persisted,
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
    }
