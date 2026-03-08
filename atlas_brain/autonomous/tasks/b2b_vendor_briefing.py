"""
Vendor Intelligence Briefing -- build, render, and send.

Assembles a deterministic vendor churn briefing from existing DB tables
(no LLM calls) and sends it via Resend.

Data sources (in priority order):
1. b2b_intelligence (weekly_churn_feed) -- richest, pre-aggregated
2. b2b_churn_signals -- fallback aggregated metrics
3. b2b_product_profiles -- enrichment (profile summary, competitive context)
4. b2b_reviews -- high-urgency quotes if feed evidence is insufficient
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...templates.email.vendor_briefing import render_vendor_briefing_html

logger = logging.getLogger("atlas.b2b.vendor_briefing")


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------

async def build_vendor_briefing(vendor_name: str) -> dict[str, Any] | None:
    """
    Build a briefing data dict for *vendor_name* from existing DB tables.

    Returns None if the vendor is not found in any source.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        return None

    briefing: dict[str, Any] = {
        "vendor_name": vendor_name,
        "report_date": date.today().isoformat(),
        "booking_url": settings.b2b_churn.vendor_briefing_booking_url,
    }

    found = False

    # ------------------------------------------------------------------
    # Source 1: weekly_churn_feed from b2b_intelligence
    # ------------------------------------------------------------------
    feed_entry = await _extract_feed_entry(pool, vendor_name)
    if feed_entry:
        found = True
        briefing.update({
            "churn_pressure_score": feed_entry.get("churn_pressure_score", 0),
            "churn_signal_density": feed_entry.get("churn_signal_density", 0),
            "avg_urgency": feed_entry.get("avg_urgency", 0),
            "review_count": feed_entry.get("total_reviews", 0),
            "dm_churn_rate": feed_entry.get("dm_churn_rate", 0),
            "pain_breakdown": feed_entry.get("pain_breakdown", []),
            "top_displacement_targets": _normalize_displacement(
                feed_entry.get("top_displacement_targets", [])
            ),
            "evidence": feed_entry.get("evidence", []),
            "trend": feed_entry.get("trend"),
            "named_accounts": feed_entry.get("named_accounts", []),
            "top_feature_gaps": feed_entry.get("top_feature_gaps", []),
            "category": feed_entry.get("category", "Software"),
            "budget_context": feed_entry.get("budget_context"),
        })

    # ------------------------------------------------------------------
    # Source 2: b2b_churn_signals (fallback if no feed entry)
    # ------------------------------------------------------------------
    if not found:
        signals = await _fetch_churn_signals(pool, vendor_name)
        if signals:
            found = True
            # top_feature_gaps from signals is [{feature, count}] -- extract names
            raw_gaps = _jsonb_list(signals.get("top_feature_gaps"))
            feature_gap_strings = [
                g.get("feature", str(g)) if isinstance(g, dict) else str(g)
                for g in raw_gaps[:5]
            ]
            briefing.update({
                "churn_pressure_score": _compute_pressure(signals),
                "churn_signal_density": _safe_density(signals),
                "avg_urgency": float(signals.get("avg_urgency_score", 0)),
                "review_count": signals.get("total_reviews", 0),
                "dm_churn_rate": float(signals.get("decision_maker_churn_rate", 0) or 0) * 100,
                "pain_breakdown": _jsonb_list(signals.get("top_pain_categories")),
                "top_displacement_targets": _normalize_displacement(
                    _jsonb_list(signals.get("top_competitors"))
                ),
                "evidence": _jsonb_list(signals.get("quotable_evidence")),
                "trend": None,
                "named_accounts": _jsonb_list(signals.get("company_churn_list")),
                "top_feature_gaps": feature_gap_strings,
                "category": signals.get("product_category") or "Software",
            })

    if not found:
        return None

    # ------------------------------------------------------------------
    # Source 3: b2b_product_profiles (enrichment)
    # ------------------------------------------------------------------
    profile = await _fetch_product_profile(pool, vendor_name)
    if profile:
        if not briefing.get("category") or briefing["category"] == "Software":
            briefing["category"] = profile.get("product_category") or "Software"
        if profile.get("profile_summary"):
            briefing["profile_summary"] = profile["profile_summary"]
        # Feature gaps from weaknesses if not already populated
        if not briefing.get("top_feature_gaps"):
            weaknesses = _jsonb_list(profile.get("weaknesses"))
            briefing["top_feature_gaps"] = [
                w.get("area", str(w)) if isinstance(w, dict) else str(w)
                for w in weaknesses[:3]
            ]

    # ------------------------------------------------------------------
    # Source 4: b2b_reviews (high-urgency quotes if evidence thin)
    # ------------------------------------------------------------------
    evidence = briefing.get("evidence") or []
    if len(evidence) < 2:
        extra_quotes = await _fetch_high_urgency_quotes(pool, vendor_name, limit=5)
        # Deduplicate against existing evidence
        existing = set(evidence)
        for q in extra_quotes:
            if q not in existing:
                evidence.append(q)
                existing.add(q)
        briefing["evidence"] = evidence

    return briefing


async def _extract_feed_entry(pool: Any, vendor_name: str) -> dict | None:
    """Find vendor in the latest weekly_churn_feed."""
    row = await pool.fetchrow(
        """
        SELECT intelligence_data
        FROM b2b_intelligence
        WHERE report_type = 'weekly_churn_feed'
        ORDER BY report_date DESC, created_at DESC
        LIMIT 1
        """,
    )
    if not row:
        return None

    data = row["intelligence_data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    # intelligence_data is the full report; the feed is a list inside it
    feed = data if isinstance(data, list) else data.get("weekly_churn_feed", data)
    if not isinstance(feed, list):
        return None

    vn_lower = vendor_name.lower()
    for entry in feed:
        if not isinstance(entry, dict):
            continue
        entry_vendor = entry.get("vendor", "") or entry.get("vendor_name", "")
        if entry_vendor.lower() == vn_lower:
            return entry

    return None


async def _fetch_churn_signals(pool: Any, vendor_name: str) -> dict | None:
    """Fetch latest churn signal row for vendor."""
    row = await pool.fetchrow(
        """
        SELECT total_reviews, negative_reviews, churn_intent_count,
               avg_urgency_score, top_pain_categories, top_competitors,
               top_feature_gaps, price_complaint_rate,
               decision_maker_churn_rate, company_churn_list,
               quotable_evidence, product_category
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor_name,
    )
    return dict(row) if row else None


async def _fetch_product_profile(pool: Any, vendor_name: str) -> dict | None:
    """Fetch product profile for vendor."""
    row = await pool.fetchrow(
        """
        SELECT profile_summary, commonly_compared_to, commonly_switched_from,
               weaknesses, product_category
        FROM b2b_product_profiles
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor_name,
    )
    return dict(row) if row else None


async def _fetch_high_urgency_quotes(
    pool: Any, vendor_name: str, limit: int = 5
) -> list[str]:
    """Fetch quotable phrases from high-urgency reviews."""
    rows = await pool.fetch(
        """
        SELECT enrichment->'quotable_phrases' AS phrases
        FROM b2b_reviews
        WHERE LOWER(vendor_name) = LOWER($1)
          AND enrichment_status = 'enriched'
          AND enrichment IS NOT NULL
          AND (enrichment->>'urgency_score')::float >= 7
        ORDER BY enriched_at DESC
        LIMIT $2
        """,
        vendor_name,
        limit,
    )
    quotes: list[str] = []
    for row in rows:
        phrases = row["phrases"]
        if isinstance(phrases, str):
            try:
                phrases = json.loads(phrases)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(phrases, list):
            for p in phrases:
                if isinstance(p, str) and p.strip():
                    quotes.append(p.strip())
    return quotes


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_displacement(targets: list) -> list[dict]:
    """Normalize displacement target dicts to {competitor, count}."""
    result = []
    for t in (targets or []):
        if isinstance(t, dict):
            result.append({
                "competitor": t.get("competitor", t.get("name", "Unknown")),
                "count": t.get("count", t.get("mentions", 0)),
            })
        elif isinstance(t, str):
            result.append({"competitor": t, "count": 0})
    return result


def _jsonb_list(val: Any) -> list:
    """Parse a JSONB value that should be a list."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _safe_density(signals: dict) -> float:
    """Compute churn signal density % from signals row."""
    total = signals.get("total_reviews", 0)
    churn = signals.get("churn_intent_count", 0)
    if total <= 0:
        return 0.0
    return round((churn / total) * 100, 1)


def _compute_pressure(signals: dict) -> float:
    """Compute a pressure score from churn signals (0-100 scale)."""
    density = _safe_density(signals)
    urgency = float(signals.get("avg_urgency_score", 0))
    dm_rate = float(signals.get("decision_maker_churn_rate", 0) or 0) * 100
    # Weighted composite: density (40%), urgency*10 (40%), DM rate (20%)
    score = (density * 0.4) + (urgency * 10 * 0.4) + (dm_rate * 0.2)
    return min(round(score, 1), 100.0)


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

async def send_vendor_briefing(
    *,
    to_email: str,
    vendor_name: str,
    briefing_html: str,
    briefing_data: dict,
) -> dict | None:
    """Send a vendor briefing email via Resend and persist to DB."""
    cfg = settings.campaign_sequence
    if not cfg.resend_api_key or not cfg.resend_from_email:
        logger.warning("Resend not configured -- cannot send briefing")
        return None

    sender_name = settings.b2b_churn.vendor_briefing_sender_name
    from_addr = f"{sender_name} <{cfg.resend_from_email}>"
    subject = f"Churn Intelligence Briefing: {vendor_name}"

    resend_id: str | None = None
    status = "sent"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {cfg.resend_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": from_addr,
                    "to": [to_email],
                    "subject": subject,
                    "html": briefing_html,
                },
            )
            resp.raise_for_status()
            resend_id = resp.json().get("id")
    except Exception as exc:
        logger.warning("Failed to send briefing to %s: %s", to_email, exc)
        status = "failed"

    # Persist delivery record
    pool = get_db_pool()
    if pool.is_initialized:
        try:
            await pool.execute(
                """
                INSERT INTO b2b_vendor_briefings
                    (vendor_name, recipient_email, subject, briefing_data, resend_id, status)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6)
                """,
                vendor_name,
                to_email,
                subject,
                json.dumps(briefing_data, default=str),
                resend_id,
                status,
            )
        except Exception as exc:
            logger.warning("Failed to persist briefing record: %s", exc)

    if status == "failed":
        return None

    return {"resend_id": resend_id, "status": status, "subject": subject}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def generate_and_send_briefing(
    *,
    vendor_name: str,
    to_email: str,
) -> dict[str, Any]:
    """Build, render, send, and return summary for a vendor briefing."""
    briefing_data = await build_vendor_briefing(vendor_name)
    if not briefing_data:
        return {"error": f"No data found for vendor: {vendor_name}"}

    briefing_html = render_vendor_briefing_html(briefing_data)

    result = await send_vendor_briefing(
        to_email=to_email,
        vendor_name=vendor_name,
        briefing_html=briefing_html,
        briefing_data=briefing_data,
    )

    if result is None:
        return {"error": "Failed to send briefing email (check Resend config)"}

    return {
        "vendor_name": vendor_name,
        "to_email": to_email,
        "resend_id": result.get("resend_id"),
        "status": result.get("status"),
        "subject": result.get("subject"),
        "sections": {
            "has_pain_breakdown": bool(briefing_data.get("pain_breakdown")),
            "has_displacement": bool(briefing_data.get("top_displacement_targets")),
            "has_quotes": bool(briefing_data.get("evidence")),
            "has_named_accounts": bool(briefing_data.get("named_accounts")),
            "has_feature_gaps": bool(briefing_data.get("top_feature_gaps")),
        },
    }
