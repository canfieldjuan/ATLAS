"""
B2B churn intelligence: aggregate enriched review data, feed to LLM
for synthesis, persist intelligence products, and notify.

Runs weekly (default Sunday 9 PM). Produces 4 report types:
  - weekly_churn_feed: ranked companies showing churn intent
  - vendor_scorecard: per-vendor health metrics
  - displacement_report: competitive flow map
  - category_overview: cross-vendor trends

Handles its own LLM call, report persistence, churn_signals upserts,
and ntfy notification -- returns _skip_synthesis so the runner does
not double-synthesize.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...services.scraping.sources import parse_source_allowlist, display_name as _source_display_name
from ...services.vendor_registry import (
    resolve_vendor_name_cached,
    _ensure_cache as _warm_vendor_cache,
)

logger = logging.getLogger("atlas.autonomous.tasks.b2b_churn_intelligence")

_EXPLORATORY_OVERVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "exploratory_summary": {"type": "string"},
        "timeline_hot_list": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "vendor": {"type": "string"},
                    "contract_end": {"type": ["string", "null"]},
                    "urgency": {"type": "number"},
                    "action": {"type": "string"},
                    "buyer_role": {"type": "string"},
                    "budget_authority": {"type": "boolean"},
                },
                "required": [
                    "company", "vendor", "contract_end", "urgency",
                    "action", "buyer_role", "budget_authority",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["exploratory_summary", "timeline_hot_list"],
    "additionalProperties": False,
}


def _canonicalize_competitor(raw: str) -> str:
    """Normalize competitor name via vendor registry, then title-case."""
    return resolve_vendor_name_cached(raw)


def _canonicalize_vendor(raw: str) -> str:
    """Normalize vendor labels using the same alias handling as competitors."""
    return resolve_vendor_name_cached(raw)


def _intelligence_source_allowlist() -> list[str]:
    """Return the configured intelligence source allowlist for SQL ANY() binding."""
    return parse_source_allowlist(settings.b2b_churn.intelligence_source_allowlist)


def _executive_source_list() -> list[str]:
    """Return curated executive sources for headline-facing queries."""
    return parse_source_allowlist(settings.b2b_churn.intelligence_executive_sources)


def _eligible_review_filters(*, window_param: int | None = 1, source_param: int = 2, alias: str = "") -> str:
    """Build a reusable SQL predicate for eligible intelligence review rows.

    When *alias* is set (e.g. ``"r"``), column references are prefixed
    with the table alias so the predicate works inside JOINed queries.
    """
    p = f"{alias}." if alias else ""
    parts = [f"{p}enrichment_status = 'enriched'"]
    if window_param is not None:
        parts.append(f"{p}enriched_at > NOW() - make_interval(days => ${window_param})")
    parts.append(f"{p}source = ANY(${source_param}::text[])")
    parts.append(f"COALESCE({p}raw_metadata->>'extraction_method', '') != 'jsonld_aggregate'")
    return "\n          AND ".join(parts)


def _validate_report(
    parsed: dict[str, Any],
    *,
    source_high_intent: list[dict[str, Any]],
    source_quotable: list[dict[str, Any]],
    source_displacement: list[dict[str, Any]],
    report_date: date,
) -> list[str]:
    """Post-LLM validation: drop fabricated quotes, unmatched displacements, stale dates.

    Mutates *parsed* in place and returns a list of warning strings.
    """
    warnings: list[str] = []

    # Build lookup of real quotes from source data
    real_quotes: set[str] = set()
    for h in source_high_intent:
        for q in h.get("quotes", []):
            if isinstance(q, str):
                real_quotes.add(q)
    for qe in source_quotable:
        for q in qe.get("quotes", []):
            if isinstance(q, str):
                real_quotes.add(q)

    # 1. Summary vendor check -- warn if exec summary names a vendor not in feed
    exec_summary = parsed.get("executive_summary", "")
    feed = parsed.get("weekly_churn_feed", [])
    feed_vendors = {e.get("vendor", "") for e in feed if isinstance(e, dict)}
    feed_vendors.discard("")
    if isinstance(exec_summary, str) and feed_vendors:
        # Collect all vendor names from source data (superset of feed)
        all_known_vendors = {h.get("vendor", "") for h in source_high_intent}
        all_known_vendors.discard("")
        for vendor in all_known_vendors:
            if vendor in exec_summary and vendor not in feed_vendors:
                warnings.append(
                    f"Executive summary mentions {vendor!r} which is not in weekly_churn_feed"
                )

    # 2. Quote verification -- weekly_churn_feed
    if isinstance(feed, list):
        for entry in feed:
            kq = entry.get("key_quote")
            if kq and kq not in real_quotes:
                warnings.append(f"Fabricated key_quote in weekly_churn_feed for {entry.get('vendor') or entry.get('company')}: {kq[:80]}")
                entry["key_quote"] = None

    # 2b. Quote verification -- displacement_map
    disp_map = parsed.get("displacement_map", [])
    if isinstance(disp_map, list):
        for entry in disp_map:
            kq = entry.get("key_quote")
            if kq and kq not in real_quotes:
                warnings.append(f"Fabricated key_quote in displacement_map for {entry.get('from_vendor')}->{entry.get('to_vendor')}: {kq[:80]}")
                entry["key_quote"] = None

    # 3. Displacement pair verification (add both orderings for direction ambiguity)
    source_pairs: set[tuple[str, str]] = set()
    for d in source_displacement:
        vendor_l = _canonicalize_vendor(d.get("vendor", "")).lower()
        comp_l = _canonicalize_competitor(d.get("competitor", "")).lower()
        source_pairs.add((vendor_l, comp_l))
        source_pairs.add((comp_l, vendor_l))
    if isinstance(disp_map, list):
        valid_disp = []
        for entry in disp_map:
            pair = (
                _canonicalize_vendor(entry.get("from_vendor") or "").lower(),
                _canonicalize_competitor(entry.get("to_vendor") or "").lower(),
            )
            if pair in source_pairs:
                valid_disp.append(entry)
            else:
                warnings.append(f"Unmatched displacement pair: {entry.get('from_vendor')}->{entry.get('to_vendor')}")
        parsed["displacement_map"] = valid_disp

    # 4. Stale date detection in timeline_hot_list
    timeline = parsed.get("timeline_hot_list", [])
    if isinstance(timeline, list):
        valid_tl = []
        for entry in timeline:
            contract_end = entry.get("contract_end")
            if contract_end:
                try:
                    end_date = datetime.strptime(str(contract_end)[:10], "%Y-%m-%d").date()
                    if end_date < report_date:
                        warnings.append(f"Stale contract_end {contract_end} for {entry.get('company')} (before report date {report_date})")
                        continue
                except (ValueError, TypeError):
                    warnings.append(
                        f"Unparseable contract_end {contract_end!r} for {entry.get('company')} -- kept in timeline"
                    )
            valid_tl.append(entry)
        parsed["timeline_hot_list"] = valid_tl

    return warnings


def _build_validated_executive_summary(
    parsed: dict[str, Any],
    *,
    data_context: dict[str, Any],
    executive_sources: list[str],
) -> str:
    """Build a concise deterministic executive summary from validated structured data."""
    feed = parsed.get("weekly_churn_feed", [])
    if not isinstance(feed, list) or not feed:
        return parsed.get("executive_summary", "")

    period = data_context.get("enrichment_period") or {}
    start = period.get("earliest")
    end = period.get("latest")
    window_label = f"Between {start} and {end}" if start and end else "In the current analysis window"

    source_dist = data_context.get("source_distribution") or {}
    executive_review_count = sum(
        int((source_dist.get(source) or {}).get("reviews") or 0)
        for source in executive_sources
    )
    source_labels = [_source_display_name(source) for source in executive_sources]
    source_label_text = ", ".join(source_labels)

    top_entries = feed[:3]
    top_vendors: list[str] = []
    top_pains: list[str] = []
    top_alternatives: list[str] = []
    quote = None
    has_named_accounts = False

    for entry in top_entries:
        vendor = entry.get("vendor")
        churn_density = entry.get("churn_signal_density") or entry.get("churn_density")
        total_reviews = entry.get("total_reviews")
        if vendor:
            parts = [str(vendor)]
            if churn_density is not None:
                parts[0] += f" ({churn_density}% churn density"
                if total_reviews:
                    parts[0] += f", {total_reviews} reviews)"
                else:
                    parts[0] += ")"
            top_vendors.append(parts[0])
        # Extract pains from pain_breakdown or top_pain
        pain_breakdown = entry.get("pain_breakdown", [])
        if pain_breakdown:
            for pb in pain_breakdown[:2]:
                p = pb.get("category", "")
                if p and p not in top_pains:
                    top_pains.append(str(p))
        elif entry.get("top_pain"):
            p = str(entry["top_pain"])
            if p not in top_pains:
                top_pains.append(p)
        # Extract alternatives from displacement targets
        for dt in entry.get("top_displacement_targets", []) or []:
            comp = dt.get("competitor", "")
            if comp and comp not in top_alternatives:
                top_alternatives.append(comp)
        if not quote and entry.get("key_quote"):
            quote = str(entry["key_quote"])
        if entry.get("named_accounts"):
            has_named_accounts = True

    lines = [
        (
            f"{window_label}, Atlas identified {len(feed)} vendors under elevated churn pressure "
            f"from {executive_review_count} executive-source reviews across {source_label_text}."
            if executive_review_count
            else f"{window_label}, Atlas identified {len(feed)} vendors under elevated churn pressure across {source_label_text}."
        )
    ]

    if top_vendors:
        lines.append("Strongest vendor-level churn signals: " + "; ".join(top_vendors) + ".")
    if top_pains or top_alternatives:
        pain_text = _join_summary_terms(top_pains[:3]) if top_pains else "mixed issues"
        alt_text = _join_summary_terms(top_alternatives[:4]) if top_alternatives else "multiple alternatives"
        lines.append(
            f"The dominant churn drivers are {pain_text}. The most visible active alternatives are {alt_text}."
        )
    if quote:
        lines.append(f"Representative evidence: \"{quote}\"")
    if has_named_accounts:
        lines.append("Named accounts identified in some vendor feeds -- see individual entries for details.")

    lines.append(
        "Confidence is highest for vendors with 50+ reviews; broader market-level conclusions should be treated as directional because source mix still varies across vendors."
    )
    return " ".join(lines)


def _extract_alternative_names(alternatives: list[Any]) -> list[str]:
    """Normalize alternative names from review enrichment objects."""
    names: list[str] = []
    for alt in alternatives or []:
        if isinstance(alt, dict):
            label = alt.get("name", "")
        else:
            label = str(alt) if alt is not None else ""
        label = _canonicalize_competitor(label)
        if label and label not in names:
            names.append(label)
    return names


def _join_summary_terms(items: list[str]) -> str:
    """Join short label lists into readable executive-summary phrasing."""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _build_buyer_action(vendor: str, pain: str | None, alternatives: list[str]) -> str:
    """Deterministic buyer-facing recommendation for weekly churn feed entries."""
    alt_text = ", ".join(alternatives[:2]) if alternatives else "competing alternatives"
    pain_l = (pain or "").lower()
    if pain_l == "pricing":
        return f"Teams on {vendor} should benchmark pricing against {alt_text} before renewal and model migration costs before signing any extension."
    if pain_l == "reliability":
        return f"Teams on {vendor} should demand incident reviews and SLA remedies immediately while validating {alt_text} as contingency options."
    if pain_l == "features":
        return f"Teams on {vendor} should map missing requirements against {alt_text} and require roadmap commitments before renewal."
    if pain_l == "ux":
        return f"Teams on {vendor} should compare admin burden and end-user adoption against {alt_text} before committing to another term."
    return f"Teams on {vendor} should compare current fit against {alt_text} and validate switching costs before the next renewal decision."


def _compute_churn_pressure_score(
    *,
    churn_density: float,
    avg_urgency: float,
    dm_churn_rate: float,
    displacement_mention_count: int,
    price_complaint_rate: float,
    total_reviews: int,
) -> float:
    """Composite 0-100 score for ranking vendors by churn pressure.

    Weights: churn density 30%, urgency 25%, DM churn rate 20%,
    displacement mentions 15%, price complaints 10%.
    Confidence multiplier: 1.0 (50+), 0.85 (20-49), 0.65 (<20).
    """
    raw = (
        min(churn_density, 100.0) * 0.30
        + min(avg_urgency, 10.0) * 10.0 * 0.25
        + min(dm_churn_rate, 1.0) * 100.0 * 0.20
        + min(displacement_mention_count, 50) * 2.0 * 0.15
        + min(price_complaint_rate, 1.0) * 100.0 * 0.10
    )
    if total_reviews >= 50:
        confidence = 1.0
    elif total_reviews >= 20:
        confidence = 0.85
    else:
        confidence = 0.65
    return round(min(raw * confidence, 100.0), 1)


def _build_deterministic_weekly_feed(high_intent: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    """Build weekly churn feed directly from validated high-intent rows."""
    seen: set[tuple[str, str]] = set()
    ordered = sorted(
        high_intent,
        key=lambda r: (
            -(r.get("urgency") or 0),
            -(1 if r.get("decision_maker") else 0),
            str(r.get("company") or ""),
        ),
    )
    results: list[dict[str, Any]] = []
    for row in ordered:
        company = row.get("company") or ""
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        if not company or not vendor:
            continue
        key = (company, vendor)
        if key in seen:
            continue
        seen.add(key)
        alternatives = _extract_alternative_names(row.get("alternatives") or [])
        quotes = [q for q in (row.get("quotes") or []) if isinstance(q, str)]
        results.append({
            "company": company,
            "vendor": vendor,
            "urgency": row.get("urgency") or 0,
            "pain": row.get("pain") or "unknown",
            "alternatives_evaluating": alternatives,
            "key_quote": quotes[0] if quotes else None,
            "evidence": quotes[:3],
            "action_recommendation": _build_buyer_action(vendor, row.get("pain"), alternatives),
            "buyer_role": row.get("role_level") or "",
            "buyer_authority": bool(row.get("decision_maker")),
        })
        if len(results) >= limit:
            break
    return results


def _build_deterministic_vendor_feed(
    vendor_scores: list[dict[str, Any]],
    *,
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list[str]],
    budget_lookup: dict[str, dict],
    sentiment_lookup: dict[str, dict[str, int]],
    buyer_auth_lookup: dict[str, dict],
    dm_lookup: dict[str, float],
    price_lookup: dict[str, float],
    company_lookup: dict[str, list],
    keyword_spike_lookup: dict[str, dict],
    prior_reports: list[dict[str, Any]],
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Build vendor-level weekly churn feed from aggregated data.

    Uses ALL enriched reviews (not just those with named companies).
    Each entry represents one vendor's churn pressure profile.
    """
    # Build prior vendor metrics for trend comparison
    prior_vendor_metrics: dict[str, dict[str, float]] = {}
    for report in prior_reports:
        if report.get("report_type") != "weekly_churn_feed":
            continue
        data = report.get("intelligence_data") or []
        if not isinstance(data, list):
            continue
        for row in data:
            vendor = row.get("vendor")
            if vendor and vendor not in prior_vendor_metrics:
                prior_vendor_metrics[vendor] = {
                    "churn_signal_density": float(row.get("churn_signal_density") or row.get("churn_density") or 0),
                    "avg_urgency": float(row.get("avg_urgency") or row.get("urgency") or 0),
                }

    # Aggregate (vendor_name, product_category) rows into one row per vendor.
    # Sums reviews/churn_intent, weighted-averages urgency, picks dominant category.
    merged: dict[str, dict[str, Any]] = {}
    for row in vendor_scores:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        reviews = int(row.get("total_reviews") or 0)
        churn = int(row.get("churn_intent") or 0)
        urgency = float(row.get("avg_urgency") or 0)
        category = row.get("product_category") or "Unknown"
        if vendor not in merged:
            merged[vendor] = {
                "total_reviews": reviews,
                "churn_intent": churn,
                "urgency_weighted_sum": urgency * reviews,
                "category": category,
                "category_reviews": reviews,
            }
        else:
            m = merged[vendor]
            m["total_reviews"] += reviews
            m["churn_intent"] += churn
            m["urgency_weighted_sum"] += urgency * reviews
            # Keep category with most reviews
            if reviews > m["category_reviews"]:
                m["category"] = category
                m["category_reviews"] = reviews

    candidates: list[dict[str, Any]] = []
    for vendor, m in merged.items():
        total_reviews = m["total_reviews"]
        churn_intent = m["churn_intent"]
        churn_density = round((churn_intent * 100.0 / total_reviews), 1) if total_reviews else 0.0
        avg_urgency = round(m["urgency_weighted_sum"] / total_reviews, 1) if total_reviews else 0.0
        category = m["category"]
        dm_rate = float(dm_lookup.get(vendor, 0))
        price_rate = float(price_lookup.get(vendor, 0))

        # Filter: include if meaningful signal
        if churn_density < 15 and avg_urgency < 6 and dm_rate < 0.3:
            continue

        # Confidence label
        if total_reviews >= 50:
            confidence = "high"
        elif total_reviews >= 20:
            confidence = "medium"
        else:
            confidence = "low"

        # Displacement mention total for this vendor
        comp_entries = competitor_lookup.get(vendor, [])
        displacement_mentions = sum(c.get("mentions", 0) for c in comp_entries)

        score = _compute_churn_pressure_score(
            churn_density=churn_density,
            avg_urgency=avg_urgency,
            dm_churn_rate=dm_rate,
            displacement_mention_count=displacement_mentions,
            price_complaint_rate=price_rate,
            total_reviews=total_reviews,
        )

        # Pain breakdown (top 3)
        pains = pain_lookup.get(vendor, [])
        top_pain = pains[0]["category"] if pains else "unknown"
        pain_breakdown = [{"category": p["category"], "count": p["count"]} for p in pains[:3]]

        # Feature gaps (top 3)
        gaps = feature_gap_lookup.get(vendor, [])
        top_feature_gaps = [g["feature"] for g in gaps[:3]]

        # Displacement targets (top 3)
        top_displacement = [{"competitor": c["name"], "mentions": c["mentions"]} for c in comp_entries[:3]]

        # Quotes
        quotes = quote_lookup.get(vendor, [])
        key_quote = quotes[0] if quotes else None
        evidence = quotes[:3]

        # Dominant buyer role
        ba = buyer_auth_lookup.get(vendor, {})
        role_types = ba.get("role_types", {})
        dominant_role = max(role_types.items(), key=lambda x: x[1])[0] if role_types else "unknown"

        # Sentiment direction
        sentiment_counts = sentiment_lookup.get(vendor, {})
        if total_reviews < 10 or not sentiment_counts:
            sentiment_direction = "insufficient_history"
        else:
            sentiment_direction = max(sentiment_counts.items(), key=lambda x: x[1])[0]

        # Trend from prior reports
        prior = prior_vendor_metrics.get(vendor)
        if not prior:
            trend = "new"
        else:
            delta_density = churn_density - prior.get("churn_signal_density", 0)
            delta_urgency = avg_urgency - prior.get("avg_urgency", 0)
            if delta_density >= 5 or delta_urgency >= 1:
                trend = "worsening"
            elif delta_density <= -5 or delta_urgency <= -1:
                trend = "improving"
            else:
                trend = "stable"

        # Named accounts (may be empty)
        companies = company_lookup.get(vendor, [])
        named_accounts = [
            {"company": c.get("company", c) if isinstance(c, dict) else str(c),
             "urgency": c.get("urgency", 0) if isinstance(c, dict) else 0}
            for c in companies[:5]
        ]

        # Alternatives for action recommendation
        alt_names = [c["name"] for c in comp_entries[:2]] if comp_entries else []

        candidates.append({
            "vendor": vendor,
            "category": category,
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "avg_urgency": avg_urgency,
            "sample_size_confidence": confidence,
            "churn_pressure_score": score,
            "top_pain": top_pain,
            "pain_breakdown": pain_breakdown,
            "top_feature_gaps": top_feature_gaps,
            "dm_churn_rate": round(dm_rate, 2),
            "price_complaint_rate": round(price_rate, 2),
            "dominant_buyer_role": dominant_role,
            "top_displacement_targets": top_displacement,
            "key_quote": key_quote,
            "evidence": evidence,
            "sentiment_direction": sentiment_direction,
            "trend": trend,
            "budget_context": budget_lookup.get(vendor, {}),
            "action_recommendation": _build_buyer_action(vendor, top_pain, alt_names),
            "named_accounts": named_accounts,
        })

    candidates.sort(key=lambda x: -x["churn_pressure_score"])
    return candidates[:limit]


def _build_reason_lookup(competitor_reasons: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    """Map (vendor, competitor) to ordered reason strings."""
    lookup: dict[tuple[str, str], list[str]] = {}
    for row in competitor_reasons:
        vendor = _canonicalize_vendor(row.get("vendor", ""))
        competitor = _canonicalize_competitor(row.get("competitor", ""))
        reason = row.get("reason") or ""
        if not vendor or not competitor or not reason:
            continue
        key = (vendor, competitor)
        lookup.setdefault(key, [])
        if reason not in lookup[key]:
            lookup[key].append(reason)
    return lookup


def _infer_driver_from_reasons(reasons: list[str], fallback: str = "other") -> str:
    """Classify a driver label from competitor-reason text."""
    text = " ".join(reasons).lower()
    keyword_map = {
        "pricing": ("price", "pricing", "cost", "cheaper", "affordable", "budget"),
        "reliability": ("outage", "uptime", "reliable", "stability", "incident", "support"),
        "ux": ("ui", "ux", "easy", "simpler", "interface", "adoption", "learning curve"),
        "features": ("feature", "workflow", "automation", "integration", "reporting", "capability"),
    }
    for label, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return label
    return fallback


def _pick_displacement_quote(
    *,
    vendor: str,
    competitor: str,
    reasons: list[str],
    quote_lookup: dict[str, list[str]],
) -> str | None:
    """Choose a quote matching the competitor or reason text when possible."""
    quotes = quote_lookup.get(vendor, [])
    competitor_l = competitor.lower()
    for quote in quotes:
        if competitor_l in quote.lower():
            return quote
    for reason in reasons:
        for token in reason.lower().split():
            if len(token) >= 5:
                for quote in quotes:
                    if token in quote.lower():
                        return quote
    return quotes[0] if quotes else None


def _build_deterministic_displacement_map(
    competitive_disp: list[dict[str, Any]],
    competitor_reasons: list[dict[str, Any]],
    quote_lookup: dict[str, list[str]],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Build displacement report from deterministic aggregated flows."""
    reason_lookup = _build_reason_lookup(competitor_reasons)
    results: list[dict[str, Any]] = []
    for row in competitive_disp:
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        competitor = _canonicalize_competitor(row.get("competitor") or "")
        if not vendor or not competitor or vendor.lower() == competitor.lower():
            continue
        reasons = reason_lookup.get((vendor, competitor), [])
        mention_count = int(row.get("mention_count") or 0)
        if mention_count >= 5:
            strength = "strong"
        elif mention_count >= 3:
            strength = "moderate"
        else:
            strength = "emerging"
        driver = _infer_driver_from_reasons(reasons)
        results.append({
            "from_vendor": vendor,
            "to_vendor": competitor,
            "mention_count": mention_count,
            "primary_driver": driver,
            "signal_strength": strength,
            "key_quote": _pick_displacement_quote(
                vendor=vendor,
                competitor=competitor,
                reasons=reasons,
                quote_lookup=quote_lookup,
            ),
        })
    results.sort(key=lambda x: x["mention_count"], reverse=True)
    return results[:limit]


def _build_deterministic_vendor_scorecards(
    vendor_scores: list[dict[str, Any]],
    *,
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    budget_lookup: dict[str, dict],
    sentiment_lookup: dict[str, dict[str, int]],
    prior_reports: list[dict[str, Any]],
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Build vendor scorecards directly from aggregated numeric data."""
    prior_vendor_metrics: dict[str, dict[str, float]] = {}
    for report in prior_reports:
        if report.get("report_type") != "vendor_scorecard":
            continue
        data = report.get("intelligence_data") or []
        if not isinstance(data, list):
            continue
        for row in data:
            vendor = row.get("vendor")
            if vendor and vendor not in prior_vendor_metrics:
                prior_vendor_metrics[vendor] = {
                    "churn_signal_density": float(row.get("churn_signal_density") or 0),
                    "avg_urgency": float(row.get("avg_urgency") or 0),
                }

    results: list[dict[str, Any]] = []
    for row in vendor_scores:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        total_reviews = int(row.get("total_reviews") or 0)
        churn_intent = int(row.get("churn_intent") or 0)
        churn_density = round((churn_intent * 100.0 / total_reviews), 1) if total_reviews else 0.0
        avg_urgency = round(float(row.get("avg_urgency") or 0), 1)
        positive_pct = row.get("positive_review_pct")
        recommend_yes = int(row.get("recommend_yes") or 0)
        recommend_no = int(row.get("recommend_no") or 0)
        recommend_ratio = round(((recommend_yes - recommend_no) / total_reviews) * 100, 1) if total_reviews else 0.0
        if total_reviews >= 50:
            confidence = "high"
        elif total_reviews >= 20:
            confidence = "medium"
        else:
            confidence = "low"

        prior = prior_vendor_metrics.get(vendor)
        if not prior:
            trend = "new"
        else:
            delta_density = churn_density - prior.get("churn_signal_density", 0)
            delta_urgency = avg_urgency - prior.get("avg_urgency", 0)
            if delta_density >= 5 or delta_urgency >= 1:
                trend = "worsening"
            elif delta_density <= -5 or delta_urgency <= -1:
                trend = "improving"
            else:
                trend = "stable"

        sentiment_counts = sentiment_lookup.get(vendor, {})
        if total_reviews < 10 or not sentiment_counts:
            sentiment_direction = "insufficient_history"
        else:
            sentiment_direction = max(sentiment_counts.items(), key=lambda item: item[1])[0]

        top_competitor = competitor_lookup.get(vendor, [])[:1]
        if top_competitor:
            comp = top_competitor[0]
            top_competitor_text = f"{comp['name']} ({comp['mentions']} mentions)"
        else:
            top_competitor_text = "Insufficient displacement data"

        top_pain = (pain_lookup.get(vendor, [{}])[0] or {}).get("category", "unknown")
        results.append({
            "vendor": vendor,
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "positive_review_pct": float(positive_pct) if positive_pct is not None else None,
            "avg_urgency": avg_urgency,
            "recommend_ratio": recommend_ratio,
            "sample_size_confidence": confidence,
            "top_pain": top_pain,
            "top_competitor_threat": top_competitor_text,
            "trend": trend,
            "budget_context": budget_lookup.get(vendor, {}),
            "sentiment_direction": sentiment_direction,
        })
    results.sort(key=lambda x: (-(x["avg_urgency"]), -(x["churn_signal_density"]), x["vendor"]))
    return results[:limit]


def _build_deterministic_category_overview(
    vendor_scores: list[dict[str, Any]],
    *,
    pain_lookup: dict[str, list[dict]],
    competitive_disp: list[dict[str, Any]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Build category overview directly from scorecards and competitive flows."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in vendor_scores:
        category = row.get("product_category") or "Unknown"
        by_category.setdefault(category, []).append(row)

    results: list[dict[str, Any]] = []
    for category, rows in by_category.items():
        ranked = sorted(
            rows,
            key=lambda r: (
                -((r.get("churn_intent") or 0) / max((r.get("total_reviews") or 1), 1)),
                -(r.get("avg_urgency") or 0),
            ),
        )
        top_vendor = ranked[0]
        highest_vendor = _canonicalize_vendor(top_vendor.get("vendor_name") or "")
        dominant_pain = (pain_lookup.get(highest_vendor, [{}])[0] or {}).get("category", "unknown")

        category_flows: dict[str, int] = {}
        for flow in competitive_disp:
            source_vendor = _canonicalize_vendor(flow.get("vendor") or "")
            if any(_canonicalize_vendor(r.get("vendor_name") or "") == source_vendor for r in rows):
                competitor = _canonicalize_competitor(flow.get("competitor") or "")
                category_flows[competitor] = category_flows.get(competitor, 0) + int(flow.get("mention_count") or 0)
        emerging = max(category_flows.items(), key=lambda item: item[1])[0] if category_flows else "Insufficient data"

        total_reviews = int(top_vendor.get("total_reviews") or 0)
        churn_density = round((int(top_vendor.get("churn_intent") or 0) * 100.0 / total_reviews), 1) if total_reviews else 0.0
        results.append({
            "category": category,
            "highest_churn_risk": highest_vendor,
            "emerging_challenger": emerging,
            "dominant_pain": dominant_pain,
            "market_shift_signal": (
                f"Based on {total_reviews} reviews, {highest_vendor} shows {churn_density}% churn-signal density in {category}. "
                f"{emerging} is the most visible challenger flow in this category."
            ),
        })
    results.sort(key=lambda x: x["category"])
    return results[:limit]


def _build_exploratory_overview(
    parsed: dict[str, Any],
    *,
    payload: dict[str, Any],
    validation_warnings: list[str],
    llm_model_id: str,
) -> dict[str, Any]:
    """Store exploratory LLM output separately from executive reports."""
    return {
        "scope": "exploratory",
        "llm_model": llm_model_id,
        "parse_fallback": bool(parsed.get("_parse_fallback")),
        "model_analysis": parsed.get("analysis_text", ""),
        "exploratory_summary": parsed.get("exploratory_summary", ""),
        "executive_summary": parsed.get("executive_summary", ""),
        "weekly_churn_feed": parsed.get("weekly_churn_feed", []),
        "vendor_scorecards": parsed.get("vendor_scorecards", []),
        "displacement_map": parsed.get("displacement_map", []),
        "category_insights": parsed.get("category_insights", []),
        "timeline_hot_list": parsed.get("timeline_hot_list", []),
        "validation_warnings": validation_warnings,
        "source_distribution": (payload.get("data_context") or {}).get("source_distribution", {}),
    }


def _trim_quote_bundles(
    rows: list[dict[str, Any]],
    *,
    outer_limit: int,
    quote_limit: int,
) -> list[dict[str, Any]]:
    """Trim quote bundles while preserving vendor labels."""
    trimmed: list[dict[str, Any]] = []
    for row in rows[:outer_limit]:
        trimmed.append({
            "vendor": row.get("vendor"),
            "quotes": list(row.get("quotes") or [])[:quote_limit],
        })
    return trimmed


def _trim_company_bundles(
    rows: list[dict[str, Any]],
    *,
    outer_limit: int,
    company_limit: int,
) -> list[dict[str, Any]]:
    """Trim churning-company bundles while preserving vendor labels."""
    trimmed: list[dict[str, Any]] = []
    for row in rows[:outer_limit]:
        trimmed.append({
            "vendor": row.get("vendor"),
            "companies": list(row.get("companies") or [])[:company_limit],
        })
    return trimmed


def _trim_use_case_distribution(
    rows: list[dict[str, Any]],
    *,
    inner_limit: int,
) -> list[dict[str, Any]]:
    """Trim nested use-case datasets without changing their structure."""
    trimmed: list[dict[str, Any]] = []
    for row in rows:
        trimmed.append({
            "type": row.get("type"),
            "data": list(row.get("data") or [])[:inner_limit],
        })
    return trimmed


def _build_exploratory_payload(
    cfg,
    *,
    today: date,
    window_days: int,
    data_context: dict[str, Any],
    vendor_scores: list[dict[str, Any]],
    high_intent: list[dict[str, Any]],
    competitive_disp: list[dict[str, Any]],
    pain_dist: list[dict[str, Any]],
    feature_gaps: list[dict[str, Any]],
    negative_counts: list[dict[str, Any]],
    price_rates: list[dict[str, Any]],
    dm_rates: list[dict[str, Any]],
    timeline_signals: list[dict[str, Any]],
    competitor_reasons: list[dict[str, Any]],
    prior_reports: list[dict[str, Any]],
    quotable_evidence: list[dict[str, Any]],
    budget_signals: list[dict[str, Any]],
    use_case_dist: list[dict[str, Any]],
    sentiment_traj: list[dict[str, Any]],
    buyer_auth: list[dict[str, Any]],
    churning_companies: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Build a trimmed exploratory payload that fits within the configured budget."""
    generic_limit = max(1, cfg.intelligence_exploratory_generic_limit)
    vendor_limit = max(1, cfg.intelligence_exploratory_vendor_limit)
    high_intent_limit = max(1, cfg.intelligence_exploratory_high_intent_limit)
    quote_vendor_limit = max(1, cfg.intelligence_exploratory_quote_vendor_limit)
    quote_limit = max(1, cfg.intelligence_exploratory_quotes_per_vendor)
    use_case_limit = max(1, cfg.intelligence_exploratory_use_case_limit)
    company_limit = max(1, cfg.intelligence_exploratory_company_limit)
    prior_limit = max(1, cfg.intelligence_exploratory_prior_report_limit)

    def _make_payload() -> dict[str, Any]:
        llm_vendors = [
            {
                "vendor": v["vendor_name"],
                "category": v["product_category"],
                "reviews": v["total_reviews"],
                "churn": v["churn_intent"],
                "urgency": round(v["avg_urgency"], 1),
                "rating": round(v["avg_rating_normalized"], 2) if v["avg_rating_normalized"] else None,
                "rec_yes": v["recommend_yes"],
                "rec_no": v["recommend_no"],
                "positive_pct": v.get("positive_review_pct"),
            }
            for v in vendor_scores[:vendor_limit]
        ]
        llm_high_intent = [
            {
                "company": h["company"],
                "vendor": h["vendor"],
                "urgency": h["urgency"],
                "pain": h["pain"],
                "dm": h.get("decision_maker"),
                "role": h.get("role_level"),
                "alts": [a.get("name", a) if isinstance(a, dict) else a for a in h.get("alternatives", [])[:3]],
                "signal": h.get("contract_signal"),
                "quotes": h.get("quotes", [])[:quote_limit],
            }
            for h in high_intent[:high_intent_limit]
        ]
        known_companies = [h["company"] for h in high_intent[:high_intent_limit] if h.get("company")]
        llm_prior = [
            {
                "type": p["report_type"],
                "date": p["report_date"],
                "data": p.get("intelligence_data", {}),
            }
            for p in prior_reports[:prior_limit]
        ]
        return {
            "date": str(today),
            "data_context": data_context,
            "analysis_window_days": window_days,
            "vendor_churn_scores": llm_vendors,
            "high_intent_companies": llm_high_intent,
            "competitive_displacement": competitive_disp[:generic_limit],
            "pain_distribution": pain_dist[:generic_limit],
            "feature_gaps": feature_gaps[:generic_limit],
            "negative_review_counts": negative_counts[:generic_limit],
            "price_complaint_rates": price_rates[:generic_limit],
            "decision_maker_churn_rates": dm_rates[:generic_limit],
            "timeline_signals": timeline_signals[:generic_limit],
            "competitor_reasons": competitor_reasons[:generic_limit],
            "prior_reports": llm_prior,
            "quotable_evidence": _trim_quote_bundles(
                quotable_evidence,
                outer_limit=quote_vendor_limit,
                quote_limit=quote_limit,
            ),
            "budget_signals": budget_signals[:generic_limit],
            "use_case_distribution": _trim_use_case_distribution(
                use_case_dist,
                inner_limit=use_case_limit,
            ),
            "sentiment_trajectory": sentiment_traj[:generic_limit],
            "buyer_authority": buyer_auth[:generic_limit],
            "churning_companies": _trim_company_bundles(
                churning_companies,
                outer_limit=generic_limit,
                company_limit=company_limit,
            ),
            "known_companies": known_companies,
        }

    payload = _make_payload()
    payload_size = len(json.dumps(payload, default=str))

    while payload_size > cfg.intelligence_exploratory_char_budget:
        changed = False
        if vendor_limit > 6:
            vendor_limit -= 2
            changed = True
        if high_intent_limit > 4:
            high_intent_limit -= 1
            changed = True
        if generic_limit > 6:
            generic_limit -= 2
            changed = True
        if quote_vendor_limit > 6:
            quote_vendor_limit -= 1
            changed = True
        if use_case_limit > 4:
            use_case_limit -= 1
            changed = True
        if company_limit > 3:
            company_limit -= 1
            changed = True
        if quote_limit > 1:
            quote_limit -= 1
            changed = True
        if prior_limit > 1:
            prior_limit -= 1
            changed = True
        if not changed:
            break
        payload = _make_payload()
        payload_size = len(json.dumps(payload, default=str))

    return payload, payload_size


def _safe_json(value: Any, default: Any = None) -> Any:
    """Safely deserialize a JSON value, returning *default* on failure."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Malformed JSON in aggregation data: %.100r", value)
            return default
    return default


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: weekly B2B churn intelligence."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Warm vendor registry cache so sync resolve_vendor_name_cached() calls
    # throughout this function hit the DB-backed cache rather than bootstrap.
    await _warm_vendor_cache()

    window_days = cfg.intelligence_window_days
    min_reviews = cfg.intelligence_min_reviews
    urgency_threshold = cfg.high_churn_urgency_threshold
    neg_threshold = cfg.negative_review_threshold
    fg_min_mentions = cfg.feature_gap_min_mentions
    quote_min_urgency = cfg.quotable_phrase_min_urgency
    tl_limit = cfg.timeline_signals_limit
    prior_limit = cfg.prior_reports_limit
    today = date.today()

    # Gather all 17 data sources + data_context in parallel
    (
        vendor_scores, high_intent, competitive_disp,
        pain_dist, feature_gaps,
        negative_counts, price_rates, dm_rates,
        churning_companies, quotable_evidence,
        budget_signals, use_case_dist, sentiment_traj,
        buyer_auth, timeline_signals, competitor_reasons,
        keyword_spikes, data_context,
    ) = await asyncio.gather(
        _fetch_vendor_churn_scores(pool, window_days, min_reviews),
        _fetch_high_intent_companies(pool, urgency_threshold, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_pain_distribution(pool, window_days),
        _fetch_feature_gaps(pool, window_days, min_mentions=fg_min_mentions),
        _fetch_negative_review_counts(pool, window_days, threshold=neg_threshold),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_churning_companies(pool, window_days),
        _fetch_quotable_evidence(pool, window_days, min_urgency=quote_min_urgency),
        _fetch_budget_signals(pool, window_days),
        _fetch_use_case_distribution(pool, window_days),
        _fetch_sentiment_trajectory(pool, window_days),
        _fetch_buyer_authority_summary(pool, window_days),
        _fetch_timeline_signals(pool, window_days, limit=tl_limit),
        _fetch_competitor_reasons(pool, window_days),
        _fetch_keyword_spikes(pool),
        _fetch_data_context(pool, window_days),
        return_exceptions=True,
    )

    # Convert exceptions to empty values, track failures
    fetcher_failures = 0

    def _safe(val: Any, name: str) -> list:
        nonlocal fetcher_failures
        if isinstance(val, Exception):
            fetcher_failures += 1
            logger.error("%s fetch failed: %s", name, val, exc_info=val)
            return []
        return val

    vendor_scores = _safe(vendor_scores, "vendor_scores")
    high_intent = _safe(high_intent, "high_intent")
    competitive_disp = _aggregate_competitive_disp(_safe(competitive_disp, "competitive_disp"))
    pain_dist = _safe(pain_dist, "pain_dist")
    feature_gaps = _safe(feature_gaps, "feature_gaps")
    negative_counts = _safe(negative_counts, "negative_counts")
    price_rates = _safe(price_rates, "price_rates")
    dm_rates = _safe(dm_rates, "dm_rates")
    churning_companies = _safe(churning_companies, "churning_companies")
    quotable_evidence = _safe(quotable_evidence, "quotable_evidence")
    budget_signals = _safe(budget_signals, "budget_signals")
    use_case_dist = _safe(use_case_dist, "use_case_dist")
    sentiment_traj = _safe(sentiment_traj, "sentiment_traj")
    buyer_auth = _safe(buyer_auth, "buyer_auth")
    timeline_signals = _safe(timeline_signals, "timeline_signals")
    competitor_reasons = _safe(competitor_reasons, "competitor_reasons")
    keyword_spikes = _safe(keyword_spikes, "keyword_spikes")
    if isinstance(data_context, Exception):
        logger.warning("data_context fetch failed: %s", data_context)
        data_context = {}

    # Check if there's enough data
    if not vendor_scores and not high_intent:
        return {"_skip_synthesis": "No enriched B2B reviews to analyze"}

    # Fetch prior reports for trend comparison
    prior_reports = await _fetch_prior_reports(pool, limit=prior_limit)

    payload, payload_size = _build_exploratory_payload(
        cfg,
        today=today,
        window_days=window_days,
        data_context=data_context,
        vendor_scores=vendor_scores,
        high_intent=high_intent,
        competitive_disp=competitive_disp,
        pain_dist=pain_dist,
        feature_gaps=feature_gaps,
        negative_counts=negative_counts,
        price_rates=price_rates,
        dm_rates=dm_rates,
        timeline_signals=timeline_signals,
        competitor_reasons=competitor_reasons,
        prior_reports=prior_reports,
        quotable_evidence=quotable_evidence,
        budget_signals=budget_signals,
        use_case_dist=use_case_dist,
        sentiment_traj=sentiment_traj,
        buyer_auth=buyer_auth,
        churning_companies=churning_companies,
    )

    from ...pipelines.llm import call_llm_with_skill, parse_json_response, get_pipeline_llm

    # Resolve model name before the call so we can record it in the DB
    # Configurable: ATLAS_B2B_CHURN_INTELLIGENCE_LLM_BACKEND=vllm|anthropic|auto
    llm_backend = cfg.intelligence_llm_backend
    _llm_workload = {"vllm": "vllm", "anthropic": "anthropic", "auto": "synthesis"}.get(
        llm_backend, "vllm"
    )
    _resolved_llm = get_pipeline_llm(workload=_llm_workload)
    llm_model_id = getattr(_resolved_llm, "model_id", "unknown") if _resolved_llm else "unknown"
    exploratory_max_tokens = cfg.intelligence_exploratory_max_tokens

    logger.info("B2B intelligence payload: %d chars, max_tokens=%d, model=%s",
                payload_size, exploratory_max_tokens, llm_model_id)

    parsed: dict[str, Any] = {}
    analysis = None
    if payload_size > cfg.intelligence_exploratory_char_budget:
        logger.warning(
            "Skipping exploratory_overview LLM call: payload size %d exceeds budget %d",
            payload_size,
            cfg.intelligence_exploratory_char_budget,
        )
    elif cfg.intelligence_exploratory_enabled:
        try:
            analysis = await asyncio.wait_for(
                asyncio.to_thread(
                    call_llm_with_skill,
                    "digest/b2b_exploratory_overview", payload,
                    max_tokens=exploratory_max_tokens, temperature=0.4,
                    response_format={"type": "json_object"},
                    guided_json=(
                        _EXPLORATORY_OVERVIEW_SCHEMA
                        if _llm_workload == "vllm" and settings.llm.vllm_guided_json_enabled
                        else None
                    ),
                    workload=_llm_workload,
                ),
                timeout=300,
            )
        except asyncio.TimeoutError:
            logger.error("LLM call timed out after 300s for b2b_churn_intelligence")

    had_llm_analysis = bool(analysis)
    if had_llm_analysis:
        parsed = parse_json_response(analysis, recover_truncated=True)

    # --- Post-processing: validate company names and flag fabricated quotes ---
    known_set = set(payload.get("known_companies", []))

    # Validate weekly_churn_feed company names
    feed = parsed.get("weekly_churn_feed", [])
    if isinstance(feed, list):
        valid_feed = []
        for entry in feed:
            company = entry.get("company", "")
            if company in known_set:
                valid_feed.append(entry)
            else:
                logger.warning(
                    "Dropped fabricated company from weekly_churn_feed: %r (not in known_companies)",
                    company,
                )
        parsed["weekly_churn_feed"] = valid_feed

    # Validate timeline_hot_list company names
    timeline = parsed.get("timeline_hot_list", [])
    if isinstance(timeline, list):
        valid_timeline = []
        for entry in timeline:
            company = entry.get("company", "")
            if company in known_set:
                valid_timeline.append(entry)
            else:
                logger.warning(
                    "Dropped fabricated company from timeline_hot_list: %r",
                    company,
                )
        parsed["timeline_hot_list"] = valid_timeline

    # --- Post-LLM validation: quotes, displacement pairs, stale dates ---
    validation_warnings = _validate_report(
        parsed,
        source_high_intent=high_intent,
        source_quotable=quotable_evidence,
        source_displacement=competitive_disp,
        report_date=today,
    )
    if validation_warnings:
        logger.info("Report validation: %d warnings", len(validation_warnings))
        for w in validation_warnings:
            logger.warning("Validation: %s", w)

    pain_lookup = _build_pain_lookup(pain_dist)
    competitor_lookup = _build_competitor_lookup(competitive_disp)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    neg_lookup = {r["vendor"]: r["negative_count"] for r in negative_counts}
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    company_lookup = {r["vendor"]: r["companies"] for r in churning_companies}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
    use_case_lookup = _build_use_case_lookup(use_case_dist)
    integration_lookup = _build_integration_lookup(use_case_dist)
    sentiment_lookup = _build_sentiment_lookup(sentiment_traj)
    buyer_auth_lookup = _build_buyer_auth_lookup(buyer_auth)
    timeline_lookup = _build_timeline_lookup(timeline_signals)
    keyword_spike_lookup = _build_keyword_spike_lookup(keyword_spikes)

    deterministic_weekly_feed = _build_deterministic_vendor_feed(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitor_lookup=competitor_lookup,
        feature_gap_lookup=feature_gap_lookup,
        quote_lookup=quote_lookup,
        budget_lookup=budget_lookup,
        sentiment_lookup=sentiment_lookup,
        buyer_auth_lookup=buyer_auth_lookup,
        dm_lookup=dm_lookup,
        price_lookup=price_lookup,
        company_lookup=company_lookup,
        keyword_spike_lookup=keyword_spike_lookup,
        prior_reports=prior_reports,
    )
    deterministic_vendor_scorecards = _build_deterministic_vendor_scorecards(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitor_lookup=competitor_lookup,
        budget_lookup=budget_lookup,
        sentiment_lookup=sentiment_lookup,
        prior_reports=prior_reports,
    )
    deterministic_displacement_map = _build_deterministic_displacement_map(
        competitive_disp,
        competitor_reasons,
        quote_lookup,
    )
    deterministic_category_overview = _build_deterministic_category_overview(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitive_disp=competitive_disp,
    )

    parsed["executive_summary"] = _build_validated_executive_summary(
        {"weekly_churn_feed": deterministic_weekly_feed},
        data_context=data_context,
        executive_sources=_executive_source_list(),
    )
    parsed["weekly_churn_feed"] = deterministic_weekly_feed
    parsed["vendor_scorecards"] = deterministic_vendor_scorecards
    parsed["displacement_map"] = deterministic_displacement_map
    parsed["category_insights"] = deterministic_category_overview

    # Persist intelligence reports
    report_types = [
        ("weekly_churn_feed", deterministic_weekly_feed),
        ("vendor_scorecard", deterministic_vendor_scorecards),
        ("displacement_report", deterministic_displacement_map),
        ("category_overview", deterministic_category_overview),
    ]

    data_density = json.dumps({
        "vendors_analyzed": len(vendor_scores),
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "pain_categories": len(pain_dist),
        "feature_gaps": len(feature_gaps),
    })
    exec_summary = parsed.get("executive_summary", "")
    exploratory_persisted = False

    try:
        async with pool.transaction() as conn:
            for report_type, data in report_types:
                await conn.execute(
                    """
                    INSERT INTO b2b_intelligence (
                        report_date, report_type, intelligence_data,
                        executive_summary, data_density, status, llm_model
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    today,
                    report_type,
                    json.dumps(data, default=str),
                    exec_summary,
                    data_density,
                    "published",
                    llm_model_id,
                )
    except Exception:
        logger.exception("Failed to store intelligence reports (rolled back)")

    if had_llm_analysis:
        exploratory_data = _build_exploratory_overview(
            parsed,
            payload=payload,
            validation_warnings=validation_warnings,
            llm_model_id=llm_model_id,
        )
        try:
            await pool.execute(
                """
                INSERT INTO b2b_intelligence (
                    report_date, report_type, intelligence_data,
                    executive_summary, data_density, status, llm_model
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                today,
                "exploratory_overview",
                json.dumps(exploratory_data, default=str),
                exploratory_data.get("executive_summary", ""),
                json.dumps({**json.loads(data_density), "scope": "exploratory"}),
                "published",
                llm_model_id,
            )
            exploratory_persisted = True
        except Exception:
            logger.exception("Failed to store exploratory_overview")

    # Upsert per-vendor churn signals
    upsert_failures = await _upsert_churn_signals(
        pool, vendor_scores,
        neg_lookup, pain_lookup, competitor_lookup, feature_gap_lookup,
        price_lookup, dm_lookup, company_lookup, quote_lookup,
        budget_lookup, use_case_lookup, integration_lookup,
        sentiment_lookup, buyer_auth_lookup, timeline_lookup,
        keyword_spike_lookup,
    )

    # Send ntfy notification
    await _send_notification(task, parsed, high_intent)

    # Emit reasoning events (no-op when reasoning disabled)
    await _emit_reasoning_events(parsed, high_intent, vendor_scores)

    return {
        "_skip_synthesis": "B2B churn intelligence complete",
        "date": str(today),
        "vendors_analyzed": len(vendor_scores),
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "report_types": len(report_types) + (1 if exploratory_persisted else 0),
        "fetcher_failures": fetcher_failures,
        "upsert_failures": upsert_failures,
    }


# ------------------------------------------------------------------
# Reasoning events
# ------------------------------------------------------------------


async def _emit_reasoning_events(
    parsed: dict[str, Any],
    high_intent: list[dict[str, Any]],
    vendor_scores: list[dict[str, Any]],
) -> None:
    """Emit B2B events for the reasoning agent (no-op when disabled)."""
    from ...reasoning.producers import emit_if_enabled
    from ...reasoning.events import EventType

    # One report-level event per run
    await emit_if_enabled(
        EventType.B2B_INTELLIGENCE_GENERATED,
        source="b2b_churn_intelligence",
        payload={
            "vendors_analyzed": len(vendor_scores),
            "high_intent_count": len(high_intent),
            "executive_summary": parsed.get("executive_summary", ""),
        },
    )

    # One event per high-intent company (cap at 10)
    for company in high_intent[:10]:
        await emit_if_enabled(
            EventType.B2B_HIGH_INTENT_DETECTED,
            source="b2b_churn_intelligence",
            payload={
                "company": company.get("company", ""),
                "vendor": company.get("vendor", ""),
                "urgency": company.get("urgency", 0),
                "pain": company.get("pain", ""),
                "alternatives": company.get("alternatives", []),
            },
            entity_type="company",
            entity_id=company.get("company", ""),
        )


# ------------------------------------------------------------------
# Data fetchers
# ------------------------------------------------------------------


async def _fetch_data_context(pool, window_days: int) -> dict[str, Any]:
    """Compute temporal metadata and source composition for the LLM."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=None, source_param=2)
    row = await pool.fetchrow(
        f"""
        SELECT
            count(*) AS total_enriched,
            count(*) FILTER (WHERE enriched_at > NOW() - make_interval(days => $1)) AS in_window,
            min(enriched_at) AS earliest_enriched,
            max(enriched_at) AS latest_enriched,
            count(DISTINCT vendor_name) AS vendor_count,
            count(DISTINCT reviewer_company) FILTER (
                WHERE reviewer_company IS NOT NULL AND reviewer_company != ''
            ) AS company_count
        FROM b2b_reviews
        WHERE {filters}
        """,
        window_days,
        sources,
    )

    # Source distribution so LLM knows the composition
    source_rows = await pool.fetch(
        f"""
        SELECT source, count(*) AS cnt,
            count(*) FILTER (
                WHERE (enrichment->>'urgency_score')::int >= 7
            ) AS high_urgency
        FROM b2b_reviews
        WHERE {filters}
            AND enriched_at > NOW() - make_interval(days => $1)
        GROUP BY source ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )
    source_dist = {
        r["source"]: {"reviews": r["cnt"], "high_urgency": r["high_urgency"]}
        for r in source_rows
    }

    return {
        "total_enriched_reviews": row["total_enriched"],
        "reviews_in_analysis_window": row["in_window"],
        "analysis_window_days": window_days,
        "enrichment_period": {
            "earliest": str(row["earliest_enriched"].date()) if row["earliest_enriched"] else None,
            "latest": str(row["latest_enriched"].date()) if row["latest_enriched"] else None,
        },
        "unique_vendors": row["vendor_count"],
        "unique_companies": row["company_count"],
        "source_distribution": source_dist,
    }


async def _fetch_vendor_churn_scores(pool, window_days: int, min_reviews: int) -> list[dict[str, Any]]:
    """Per-vendor health metrics from enriched reviews."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=3)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name, product_category,
            count(*) AS total_reviews,
            count(*) FILTER (
                WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS churn_intent,
            -- Source-weighted urgency: weighted avg preserving 0-10 scale
            -- Falls back to 0.7 for pre-existing reviews without source_weight
            avg(
                (enrichment->>'urgency_score')::numeric
                * COALESCE((raw_metadata->>'source_weight')::numeric, 0.7)
            ) / NULLIF(avg(COALESCE((raw_metadata->>'source_weight')::numeric, 0.7)), 0)
            AS avg_urgency,
            avg(rating / NULLIF(rating_max, 0)) AS avg_rating_normalized,
            count(*) FILTER (
                WHERE (enrichment->>'would_recommend')::boolean = true
            ) AS recommend_yes,
            count(*) FILTER (
                WHERE (enrichment->>'would_recommend')::boolean = false
            ) AS recommend_no,
            -- Positive review percentage (rating >= 70% of max)
            ROUND(
                count(*) FILTER (
                    WHERE rating IS NOT NULL AND rating_max > 0
                      AND (rating / rating_max) >= 0.7
                ) * 100.0 / NULLIF(count(*) FILTER (
                    WHERE rating IS NOT NULL AND rating_max > 0
                ), 0),
                1
            ) AS positive_review_pct
        FROM b2b_reviews
        WHERE {filters}
        GROUP BY vendor_name, product_category
        HAVING count(*) >= $2
        ORDER BY avg((enrichment->>'urgency_score')::numeric) DESC
        """,
        window_days,
        min_reviews,
        sources,
    )
    return [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "total_reviews": r["total_reviews"],
            "churn_intent": r["churn_intent"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
            "avg_rating_normalized": float(r["avg_rating_normalized"]) if r["avg_rating_normalized"] else None,
            "recommend_yes": r["recommend_yes"],
            "recommend_no": r["recommend_no"],
            "positive_review_pct": float(r["positive_review_pct"]) if r["positive_review_pct"] is not None else None,
        }
        for r in rows
    ]


async def _fetch_high_intent_companies(pool, urgency_threshold: int, window_days: int) -> list[dict[str, Any]]:
    """Companies showing high churn intent -- the money feed."""
    sources = _executive_source_list()
    filters = _eligible_review_filters(window_param=2, source_param=3)
    rows = await pool.fetch(
        f"""
        SELECT reviewer_company, vendor_name, product_category,
            enrichment->'reviewer_context'->>'role_level' AS role_level,
            (enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
            (enrichment->>'urgency_score')::numeric AS urgency,
            enrichment->>'pain_category' AS pain,
            enrichment->'competitors_mentioned' AS alternatives,
            enrichment->'quotable_phrases' AS quotes,
            enrichment->'contract_context'->>'contract_value_signal' AS value_signal
        FROM b2b_reviews
        WHERE {filters}
          AND (enrichment->>'urgency_score')::numeric >= $1
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        """,
        urgency_threshold,
        window_days,
        sources,
    )
    results = []
    for r in rows:
        try:
            urgency = float(r["urgency"]) if r["urgency"] is not None else 0
        except (ValueError, TypeError):
            urgency = 0
        results.append({
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "category": r["product_category"],
            "role_level": r["role_level"],
            "decision_maker": r["is_dm"],
            "urgency": urgency,
            "pain": r["pain"],
            "alternatives": _safe_json(r["alternatives"]),
            "quotes": _safe_json(r["quotes"]),
            "contract_signal": r["value_signal"],
        })
    return results


async def _fetch_competitive_displacement(pool, window_days: int) -> list[dict[str, Any]]:
    """Who's winning from whom -- competitive flows with 2+ mentions (per skill rules)."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            comp.value->>'name' AS competitor,
            comp.value->>'context' AS direction,
            count(*) AS mention_count
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'competitors_mentioned') AS comp(value)
        WHERE {filters}
        GROUP BY vendor_name, comp.value->>'name', comp.value->>'context'
        HAVING count(*) >= 2
        ORDER BY mention_count DESC
        """,
        window_days,
        sources,
    )
    # Post-process: canonicalize competitors, filter self-flows, re-aggregate
    merged: dict[tuple[str, str, str | None], int] = {}
    for r in rows:
        canon = _canonicalize_competitor(r["competitor"] or "")
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        # Filter self-flows (vendor mentioning itself)
        if canon and vendor and canon.lower() == vendor.lower():
            continue
        key = (vendor, canon, r["direction"])
        merged[key] = merged.get(key, 0) + r["mention_count"]

    # Re-apply HAVING count >= 2, sort by mention_count DESC
    results = [
        {"vendor": k[0], "competitor": k[1], "direction": k[2], "mention_count": cnt}
        for k, cnt in merged.items()
        if cnt >= 2
    ]
    results.sort(key=lambda x: x["mention_count"], reverse=True)
    return results


async def _fetch_pain_distribution(pool, window_days: int) -> list[dict[str, Any]]:
    """What's driving churn per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            enrichment->>'pain_category' AS pain,
            count(*) AS complaint_count,
            avg((enrichment->>'urgency_score')::numeric) AS avg_urgency
        FROM b2b_reviews
        WHERE {filters}
        GROUP BY vendor_name, enrichment->>'pain_category'
        ORDER BY complaint_count DESC
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "pain": r["pain"],
            "complaint_count": r["complaint_count"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
        }
        for r in rows
    ]


async def _fetch_feature_gaps(pool, window_days: int, *, min_mentions: int = 2) -> list[dict[str, Any]]:
    """Most-mentioned missing features per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=3)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            gap.value #>> '{{}}' AS feature_gap,
            count(*) AS mentions
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'feature_gaps') AS gap(value)
        WHERE {filters}
        GROUP BY vendor_name, gap.value #>> '{{}}'
        HAVING count(*) >= $2
        ORDER BY mentions DESC
        """,
        window_days,
        min_mentions,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "feature_gap": r["feature_gap"],
            "mentions": r["mentions"],
        }
        for r in rows
    ]


async def _fetch_negative_review_counts(pool, window_days: int, *, threshold: float = 0.5) -> list[dict[str, Any]]:
    """Count reviews with below-threshold ratings per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=3)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name, count(*) AS negative_count
        FROM b2b_reviews
        WHERE {filters}
          AND rating IS NOT NULL AND rating_max > 0
          AND (rating / rating_max) < $2
        GROUP BY vendor_name
        """,
        window_days,
        threshold,
        sources,
    )
    return [{"vendor": r["vendor_name"], "negative_count": r["negative_count"]} for r in rows]


async def _fetch_price_complaint_rates(pool, window_days: int) -> list[dict[str, Any]]:
    """Fraction of reviews with pain_category='pricing' per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            count(*) FILTER (WHERE enrichment->>'pain_category' = 'pricing') AS pricing_count,
            count(*) AS total
        FROM b2b_reviews
        WHERE {filters}
        GROUP BY vendor_name
        HAVING count(*) > 0
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "price_complaint_rate": r["pricing_count"] / r["total"] if r["total"] else 0,
        }
        for r in rows
    ]


async def _fetch_dm_churn_rates(pool, window_days: int) -> list[dict[str, Any]]:
    """Decision-maker churn rate: DMs with intent_to_leave / total DMs, per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            count(*) FILTER (
                WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
                  AND (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS dm_churning,
            count(*) FILTER (
                WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
            ) AS dm_total
        FROM b2b_reviews
        WHERE {filters}
        GROUP BY vendor_name
        HAVING count(*) FILTER (
            WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean = true
        ) > 0
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "dm_churn_rate": r["dm_churning"] / r["dm_total"] if r["dm_total"] else 0,
        }
        for r in rows
    ]


async def _fetch_churning_companies(pool, window_days: int) -> list[dict[str, Any]]:
    """Companies with high churn intent, aggregated per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            jsonb_agg(jsonb_build_object(
                'company', reviewer_company,
                'urgency', (enrichment->>'urgency_score')::numeric,
                'role', enrichment->'reviewer_context'->>'role_level',
                'pain', enrichment->>'pain_category'
            ) ORDER BY (enrichment->>'urgency_score')::numeric DESC)
            AS companies
        FROM b2b_reviews
        WHERE {filters}
          AND (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
        GROUP BY vendor_name
        """,
        window_days,
        sources,
    )
    results = []
    for r in rows:
        companies = _safe_json(r["companies"])
        results.append({"vendor": r["vendor_name"], "companies": companies})
    return results


async def _fetch_quotable_evidence(pool, window_days: int, *, min_urgency: float = 6) -> list[dict[str, Any]]:
    """Top quotable phrases per vendor (highest urgency, deduplicated)."""
    sources = _executive_source_list()
    filters = _eligible_review_filters(window_param=1, source_param=3)
    rows = await pool.fetch(
        f"""
        WITH ranked_quotes AS (
            SELECT vendor_name, phrase.value AS quote,
                (enrichment->>'urgency_score')::numeric AS urgency,
                ROW_NUMBER() OVER (
                    PARTITION BY vendor_name
                    ORDER BY (enrichment->>'urgency_score')::numeric DESC
                ) AS rn
            FROM b2b_reviews
            CROSS JOIN LATERAL jsonb_array_elements_text(
                COALESCE(enrichment->'quotable_phrases', '[]'::jsonb)
            ) AS phrase(value)
            WHERE {filters}
              AND (enrichment->>'urgency_score')::numeric >= $2
        )
        SELECT vendor_name, jsonb_agg(quote ORDER BY urgency DESC) AS quotes
        FROM ranked_quotes WHERE rn <= 15
        GROUP BY vendor_name
        """,
        window_days,
        min_urgency,
        sources,
    )
    results = []
    for r in rows:
        quotes = _safe_json(r["quotes"])
        results.append({"vendor": r["vendor_name"], "quotes": quotes})
    return results


async def _fetch_budget_signals(pool, window_days: int) -> list[dict[str, Any]]:
    """Aggregate budget signals: seat_count stats and price-increase mentions per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            avg(NULLIF(
                CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                     THEN (enrichment->'budget_signals'->>'seat_count')::numeric END,
                0)) AS avg_seat_count,
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY NULLIF(
                    CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                         THEN (enrichment->'budget_signals'->>'seat_count')::numeric END,
                    0)
            ) AS median_seat_count,
            max(NULLIF(
                CASE WHEN enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                     THEN (enrichment->'budget_signals'->>'seat_count')::numeric END,
                0)) AS max_seat_count,
            count(*) FILTER (
                WHERE (enrichment->'budget_signals'->>'price_increase_mentioned')::boolean = true
            ) AS price_increase_count,
            count(*) AS total
        FROM b2b_reviews
        WHERE {filters}
          AND enrichment->'budget_signals' IS NOT NULL
          AND enrichment->'budget_signals' != 'null'::jsonb
        GROUP BY vendor_name
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "avg_seat_count": float(r["avg_seat_count"]) if r["avg_seat_count"] else None,
            "median_seat_count": float(r["median_seat_count"]) if r["median_seat_count"] else None,
            "max_seat_count": float(r["max_seat_count"]) if r["max_seat_count"] else None,
            "price_increase_count": r["price_increase_count"],
            "price_increase_rate": r["price_increase_count"] / r["total"] if r["total"] else 0,
        }
        for r in rows
    ]


async def _fetch_use_case_distribution(pool, window_days: int) -> list[dict[str, Any]]:
    """Explode use_case modules and integration stacks, count per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    module_rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            mod.value #>> '{{}}' AS module_name,
            count(*) AS mentions
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(enrichment->'use_case'->'modules_mentioned', '[]'::jsonb)
        ) AS mod(value)
        WHERE {filters}
        GROUP BY vendor_name, mod.value #>> '{{}}'
        HAVING count(*) >= 2
        ORDER BY mentions DESC
        """,
        window_days,
        sources,
    )
    stack_rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            tool.value #>> '{{}}' AS tool_name,
            count(*) AS mentions
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(enrichment->'use_case'->'integration_stack', '[]'::jsonb)
        ) AS tool(value)
        WHERE {filters}
        GROUP BY vendor_name, tool.value #>> '{{}}'
        HAVING count(*) >= 2
        ORDER BY mentions DESC
        """,
        window_days,
        sources,
    )
    lock_rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE {filters}
          AND enrichment->'use_case'->>'lock_in_level' IS NOT NULL
        GROUP BY vendor_name, enrichment->'use_case'->>'lock_in_level'
        ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )
    return [
        {"type": "modules", "data": [dict(r) for r in module_rows]},
        {"type": "stacks", "data": [dict(r) for r in stack_rows]},
        {"type": "lock_in", "data": [dict(r) for r in lock_rows]},
    ]


async def _fetch_sentiment_trajectory(pool, window_days: int) -> list[dict[str, Any]]:
    """Count reviews per sentiment direction per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            enrichment->'sentiment_trajectory'->>'direction' AS direction,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE {filters}
          AND enrichment->'sentiment_trajectory'->>'direction' IS NOT NULL
        GROUP BY vendor_name, enrichment->'sentiment_trajectory'->>'direction'
        ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "direction": r["direction"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_buyer_authority_summary(pool, window_days: int) -> list[dict[str, Any]]:
    """Count reviews per role_type and buying_stage per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            enrichment->'buyer_authority'->>'role_type' AS role_type,
            enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE {filters}
          AND enrichment->'buyer_authority' IS NOT NULL
          AND enrichment->'buyer_authority' != 'null'::jsonb
        GROUP BY vendor_name,
            enrichment->'buyer_authority'->>'role_type',
            enrichment->'buyer_authority'->>'buying_stage'
        ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "role_type": r["role_type"],
            "buying_stage": r["buying_stage"],
            "count": r["cnt"],
        }
        for r in rows
    ]


async def _fetch_timeline_signals(pool, window_days: int, *, limit: int = 50) -> list[dict[str, Any]]:
    """Extract reviews with non-null contract_end or evaluation_deadline -- hottest leads."""
    sources = _executive_source_list()
    filters = _eligible_review_filters(window_param=1, source_param=3)
    rows = await pool.fetch(
        f"""
        SELECT reviewer_company, vendor_name,
            enrichment->'timeline'->>'contract_end' AS contract_end,
            enrichment->'timeline'->>'evaluation_deadline' AS evaluation_deadline,
            enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
            (enrichment->>'urgency_score')::numeric AS urgency
        FROM b2b_reviews
        WHERE {filters}
          AND (
              enrichment->'timeline'->>'contract_end' IS NOT NULL
              OR enrichment->'timeline'->>'evaluation_deadline' IS NOT NULL
          )
        ORDER BY (enrichment->>'urgency_score')::numeric DESC
        LIMIT $2
        """,
        window_days,
        limit,
        sources,
    )
    return [
        {
            "company": r["reviewer_company"],
            "vendor": r["vendor_name"],
            "contract_end": r["contract_end"],
            "evaluation_deadline": r["evaluation_deadline"],
            "decision_timeline": r["decision_timeline"],
            "urgency": float(r["urgency"]) if r["urgency"] else 0,
        }
        for r in rows
    ]


async def _fetch_competitor_reasons(pool, window_days: int) -> list[dict[str, Any]]:
    """Top reasons per vendor/competitor pair (aggregated, not raw dump)."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        WITH ranked_reasons AS (
            SELECT vendor_name,
                comp.value->>'name' AS competitor,
                comp.value->>'context' AS direction,
                comp.value->>'reason' AS reason,
                count(*) AS mention_count,
                ROW_NUMBER() OVER (
                    PARTITION BY vendor_name, comp.value->>'name'
                    ORDER BY count(*) DESC
                ) AS rn
            FROM b2b_reviews
            CROSS JOIN LATERAL jsonb_array_elements(enrichment->'competitors_mentioned') AS comp(value)
                        WHERE {filters}
              AND comp.value->>'reason' IS NOT NULL
            GROUP BY vendor_name, comp.value->>'name', comp.value->>'context', comp.value->>'reason'
        )
        SELECT vendor_name, competitor, direction, reason, mention_count
        FROM ranked_reasons WHERE rn <= 3
        ORDER BY mention_count DESC
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "competitor": r["competitor"],
            "direction": r["direction"],
            "reason": r["reason"],
            "mention_count": r["mention_count"],
        }
        for r in rows
    ]


async def _fetch_prior_reports(pool, *, limit: int = 4) -> list[dict[str, Any]]:
    """Fetch most recent prior intelligence reports for trend comparison.

    Includes both weekly_churn_feed and vendor_scorecard, with full
    intelligence_data so the LLM can compute trends from actual numbers
    instead of guessing from prose.
    """
    rows = await pool.fetch(
        """
        SELECT report_type, intelligence_data, executive_summary, report_date
        FROM b2b_intelligence
        WHERE report_type IN ('weekly_churn_feed', 'vendor_scorecard')
        ORDER BY report_date DESC
        LIMIT $1
        """,
        limit,
    )
    results = []
    for r in rows:
        intel_data = r["intelligence_data"]
        # asyncpg auto-deserializes JSONB to dict/list, but handle string fallback
        if isinstance(intel_data, str):
            try:
                intel_data = json.loads(intel_data)
            except (json.JSONDecodeError, TypeError):
                intel_data = {}
        results.append({
            "report_type": r["report_type"],
            "report_date": str(r["report_date"]),
            "executive_summary": r["executive_summary"],
            "intelligence_data": intel_data,
        })
    return results


# ------------------------------------------------------------------
# Public aggregation entry point (reused by b2b_tenant_report)
# ------------------------------------------------------------------


def _vendor_match(value: str, vendor_set: set[str]) -> bool:
    """Case-insensitive ILIKE-style match: vendor_set entry is contained in value."""
    vl = value.lower()
    return any(name in vl for name in vendor_set)


def _filter_by_vendors(data: list[dict], vendor_names: list[str]) -> list[dict]:
    """Post-filter fetcher results to only include rows matching vendor_names.

    Handles two structures:
    - Flat dicts with a ``vendor`` / ``vendor_name`` key (most fetchers)
    - Nested dicts like ``use_case_distribution`` with ``{"type": ..., "data": [...]}``
      where vendor data lives inside ``data[*]["vendor_name"]``
    """
    lowered = {v.lower() for v in vendor_names}
    filtered = []
    for row in data:
        vn = row.get("vendor") or row.get("vendor_name") or ""
        if vn:
            # Standard flat row
            if _vendor_match(vn, lowered):
                filtered.append(row)
        elif "data" in row and isinstance(row["data"], list):
            # Nested structure (use_case_distribution): filter inner data
            inner = [r for r in row["data"] if _vendor_match(
                r.get("vendor_name") or r.get("vendor") or "", lowered
            )]
            if inner:
                filtered.append({**row, "data": inner})
        # else: no vendor key at all, skip row
    return filtered


async def gather_intelligence_data(
    pool,
    window_days: int = 30,
    min_reviews: int = 3,
    vendor_names: list[str] | None = None,
) -> dict[str, Any]:
    """Gather all 17 intelligence data sources, optionally scoped to vendors.

    Returns a trimmed payload dict that fits the LLM token budget. Used by both
    the global ``run()`` handler and per-tenant report generation.
    """
    cfg = settings.b2b_churn
    urgency_threshold = cfg.high_churn_urgency_threshold
    neg_threshold = cfg.negative_review_threshold
    fg_min_mentions = cfg.feature_gap_min_mentions
    quote_min_urgency = cfg.quotable_phrase_min_urgency
    tl_limit = cfg.timeline_signals_limit
    prior_limit = cfg.prior_reports_limit

    results = await asyncio.gather(
        _fetch_vendor_churn_scores(pool, window_days, min_reviews),
        _fetch_high_intent_companies(pool, urgency_threshold, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_pain_distribution(pool, window_days),
        _fetch_feature_gaps(pool, window_days, min_mentions=fg_min_mentions),
        _fetch_negative_review_counts(pool, window_days, threshold=neg_threshold),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_churning_companies(pool, window_days),
        _fetch_quotable_evidence(pool, window_days, min_urgency=quote_min_urgency),
        _fetch_budget_signals(pool, window_days),
        _fetch_use_case_distribution(pool, window_days),
        _fetch_sentiment_trajectory(pool, window_days),
        _fetch_buyer_authority_summary(pool, window_days),
        _fetch_timeline_signals(pool, window_days, limit=tl_limit),
        _fetch_competitor_reasons(pool, window_days),
        _fetch_data_context(pool, window_days),
        return_exceptions=True,
    )

    names = [
        "vendor_scores", "high_intent", "competitive_disp", "pain_dist",
        "feature_gaps", "negative_counts", "price_rates", "dm_rates",
        "churning_companies", "quotable_evidence", "budget_signals",
        "use_case_dist", "sentiment_traj", "buyer_auth",
        "timeline_signals", "competitor_reasons", "data_context",
    ]

    fetcher_failures = 0
    data: dict[str, Any] = {}
    for name, val in zip(names, results):
        if isinstance(val, Exception):
            fetcher_failures += 1
            logger.error("%s fetch failed: %s", name, val, exc_info=val)
            data[name] = {} if name == "data_context" else []
        else:
            data[name] = val

    # Post-filter by vendor names if scoped
    if vendor_names:
        for key in data:
            val = data[key]
            if isinstance(val, list) and val and isinstance(val[0], dict):
                data[key] = _filter_by_vendors(val, vendor_names)

    prior_reports = await _fetch_prior_reports(pool, limit=prior_limit)

    # Trim payload to fit ~4k token input budget (8k context - 4k output)
    llm_vendors = [
        {
            "vendor": v["vendor_name"],
            "category": v["product_category"],
            "reviews": v["total_reviews"],
            "churn": v["churn_intent"],
            "urgency": round(v["avg_urgency"], 1),
            "rating": round(v["avg_rating_normalized"], 2) if v["avg_rating_normalized"] else None,
            "rec_yes": v["recommend_yes"],
            "rec_no": v["recommend_no"],
        }
        for v in data["vendor_scores"][:15]
    ]
    llm_high_intent = [
        {
            "company": h["company"],
            "vendor": h["vendor"],
            "urgency": h["urgency"],
            "pain": h["pain"],
            "dm": h.get("decision_maker"),
            "role": h.get("role_level"),
            "alts": [a.get("name", a) if isinstance(a, dict) else a for a in h.get("alternatives", [])[:3]],
            "signal": h.get("contract_signal"),
        }
        for h in data["high_intent"][:10]
    ]
    llm_prior = [
        {
            "type": p["report_type"],
            "date": p["report_date"],
            "data": p.get("intelligence_data", {}),
        }
        for p in prior_reports[:2]
    ]
    payload = {
        "date": str(date.today()),
        "data_context": data["data_context"],
        "analysis_window_days": window_days,
        "vendor_churn_scores": llm_vendors,
        "high_intent_companies": llm_high_intent,
        "competitive_displacement": data["competitive_disp"][:10],
        "pain_distribution": data["pain_dist"][:10],
        "feature_gaps": data["feature_gaps"][:8],
        "negative_review_counts": data["negative_counts"][:10],
        "price_complaint_rates": data["price_rates"][:10],
        "decision_maker_churn_rates": data["dm_rates"][:10],
        "timeline_signals": data["timeline_signals"][:8],
        "competitor_reasons": data["competitor_reasons"][:8],
        "prior_reports": llm_prior,
    }

    return {
        "payload": payload,
        "fetcher_failures": fetcher_failures,
        "vendors_analyzed": len(data["vendor_scores"]),
        "high_intent_companies": len(data["high_intent"]),
        "competitive_flows": len(data["competitive_disp"]),
        "pain_categories": len(data["pain_dist"]),
        "feature_gaps": len(data["feature_gaps"]),
    }


# ------------------------------------------------------------------
# Lookup builders (pure Python, no DB)
# ------------------------------------------------------------------


def _build_pain_lookup(pain_dist: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {category, count, avg_urgency}."""
    lookup: dict[str, list[dict]] = {}
    for row in pain_dist:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "category": row.get("pain", "other"),
            "count": row.get("complaint_count", 0),
            "avg_urgency": round(row.get("avg_urgency", 0), 1),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["count"], reverse=True)
    return lookup


def _aggregate_competitive_disp(competitive_disp: list[dict]) -> list[dict]:
    """Merge rows with same (vendor, competitor) across directions, summing mentions."""
    agg: dict[tuple[str, str], int] = {}
    for row in competitive_disp:
        key = (row.get("vendor", ""), row.get("competitor", ""))
        agg[key] = agg.get(key, 0) + int(row.get("mention_count") or 0)
    return [
        {"vendor": v, "competitor": c, "mention_count": m}
        for (v, c), m in sorted(agg.items(), key=lambda x: x[1], reverse=True)
    ]


def _build_competitor_lookup(competitive_disp: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {name, mentions} (aggregated across directions)."""
    # First pass: sum mentions per (vendor, competitor) across all directions.
    agg: dict[str, dict[str, int]] = {}
    for row in competitive_disp:
        vendor = row.get("vendor", "")
        comp = row.get("competitor", "")
        mentions = row.get("mention_count", 0)
        agg.setdefault(vendor, {})
        agg[vendor][comp] = agg[vendor].get(comp, 0) + mentions
    # Second pass: build sorted list per vendor.
    lookup: dict[str, list[dict]] = {}
    for vendor, comps in agg.items():
        lookup[vendor] = sorted(
            [{"name": c, "mentions": m} for c, m in comps.items()],
            key=lambda x: x["mentions"],
            reverse=True,
        )
    return lookup


def _build_feature_gap_lookup(feature_gaps: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {feature, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for row in feature_gaps:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "feature": row.get("feature_gap", ""),
            "mentions": row.get("mentions", 0),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_use_case_lookup(use_case_dist: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {module, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for entry in use_case_dist:
        if entry.get("type") != "modules":
            continue
        for row in entry.get("data", []):
            vendor = row.get("vendor_name", "")
            lookup.setdefault(vendor, []).append({
                "module": row.get("module_name", ""),
                "mentions": row.get("mentions", 0),
            })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_integration_lookup(use_case_dist: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {tool, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for entry in use_case_dist:
        if entry.get("type") != "stacks":
            continue
        for row in entry.get("data", []):
            vendor = row.get("vendor_name", "")
            lookup.setdefault(vendor, []).append({
                "tool": row.get("tool_name", ""),
                "mentions": row.get("mentions", 0),
            })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["mentions"], reverse=True)
    return lookup


def _build_sentiment_lookup(sentiment_traj: list[dict]) -> dict[str, dict[str, int]]:
    """vendor -> {direction: count}."""
    lookup: dict[str, dict[str, int]] = {}
    for row in sentiment_traj:
        vendor = row.get("vendor", "")
        direction = row.get("direction", "unknown")
        lookup.setdefault(vendor, {})[direction] = row.get("count", 0)
    return lookup


def _build_buyer_auth_lookup(buyer_auth: list[dict]) -> dict[str, dict]:
    """vendor -> {role_types: {type: count}, buying_stages: {stage: count}}."""
    lookup: dict[str, dict] = {}
    for row in buyer_auth:
        vendor = row.get("vendor", "")
        if vendor not in lookup:
            lookup[vendor] = {"role_types": {}, "buying_stages": {}}
        rt = row.get("role_type", "unknown")
        bs = row.get("buying_stage", "unknown")
        cnt = row.get("count", 0)
        lookup[vendor]["role_types"][rt] = lookup[vendor]["role_types"].get(rt, 0) + cnt
        lookup[vendor]["buying_stages"][bs] = lookup[vendor]["buying_stages"].get(bs, 0) + cnt
    return lookup


def _build_timeline_lookup(timeline_signals: list[dict]) -> dict[str, list[dict]]:
    """vendor -> list of timeline entries."""
    lookup: dict[str, list[dict]] = {}
    for row in timeline_signals:
        vendor = row.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "company": row.get("company"),
            "contract_end": row.get("contract_end"),
            "evaluation_deadline": row.get("evaluation_deadline"),
            "decision_timeline": row.get("decision_timeline"),
            "urgency": row.get("urgency", 0),
        })
    return lookup


async def _fetch_keyword_spikes(pool) -> list[dict[str, Any]]:
    """Fetch recent keyword spikes from b2b_keyword_signals.

    Returns one row per vendor with spike count and spike keywords.
    Uses only the latest snapshot per keyword (most recent week) to avoid
    JSONB_OBJECT_AGG duplicate key non-determinism.
    """
    rows = await pool.fetch(
        """
        WITH latest AS (
            SELECT DISTINCT ON (vendor_name, keyword)
                   vendor_name, keyword, volume_relative,
                   volume_change_pct, is_spike
            FROM b2b_keyword_signals
            WHERE snapshot_week >= CURRENT_DATE - INTERVAL '28 days'
            ORDER BY vendor_name, keyword, snapshot_week DESC
        )
        SELECT vendor_name,
               COUNT(*) FILTER (WHERE is_spike) AS spike_count,
               ARRAY_AGG(DISTINCT keyword) FILTER (WHERE is_spike) AS spike_keywords,
               JSONB_OBJECT_AGG(
                   keyword,
                   JSONB_BUILD_OBJECT(
                       'volume', volume_relative,
                       'change_pct', volume_change_pct,
                       'is_spike', is_spike
                   )
               ) AS trend_summary
        FROM latest
        GROUP BY vendor_name
        """
    )
    return [dict(r) for r in rows]


def _build_keyword_spike_lookup(
    keyword_spikes: list[dict],
) -> dict[str, dict]:
    """vendor -> {spike_count, spike_keywords, trend_summary}."""
    lookup: dict[str, dict] = {}
    for row in keyword_spikes:
        vendor = row.get("vendor_name", "")
        lookup[vendor] = {
            "spike_count": row.get("spike_count", 0),
            "spike_keywords": row.get("spike_keywords") or [],
            "trend_summary": row.get("trend_summary") or {},
        }
    return lookup


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------


async def _upsert_churn_signals(
    pool,
    vendor_scores: list[dict],
    neg_lookup: dict[str, int],
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    price_lookup: dict[str, float],
    dm_lookup: dict[str, float],
    company_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list[str]],
    budget_lookup: dict[str, dict] | None = None,
    use_case_lookup: dict[str, list[dict]] | None = None,
    integration_lookup: dict[str, list[dict]] | None = None,
    sentiment_lookup: dict[str, dict[str, int]] | None = None,
    buyer_auth_lookup: dict[str, dict] | None = None,
    timeline_lookup: dict[str, list[dict]] | None = None,
    keyword_spike_lookup: dict[str, dict] | None = None,
) -> int:
    """Upsert b2b_churn_signals (25 columns incl. keyword signals). Returns failure count."""
    now = datetime.now(timezone.utc)
    budget_lookup = budget_lookup or {}
    use_case_lookup = use_case_lookup or {}
    integration_lookup = integration_lookup or {}
    sentiment_lookup = sentiment_lookup or {}
    buyer_auth_lookup = buyer_auth_lookup or {}
    timeline_lookup = timeline_lookup or {}
    keyword_spike_lookup = keyword_spike_lookup or {}
    failures = 0

    for vs in vendor_scores:
        vendor = vs["vendor_name"]
        category = vs.get("product_category")

        total = vs["total_reviews"]
        recommend_yes = vs.get("recommend_yes", 0)
        recommend_no = vs.get("recommend_no", 0)
        nps = ((recommend_yes - recommend_no) / total * 100) if total > 0 else None

        try:
            kw_data = keyword_spike_lookup.get(vendor, {})
            await pool.execute(
                """
                INSERT INTO b2b_churn_signals (
                    vendor_name, product_category,
                    total_reviews, negative_reviews, churn_intent_count,
                    avg_urgency_score, avg_rating_normalized, nps_proxy,
                    top_pain_categories, top_competitors, top_feature_gaps,
                    price_complaint_rate, decision_maker_churn_rate,
                    company_churn_list, quotable_evidence,
                    top_use_cases, top_integration_stacks,
                    budget_signal_summary, sentiment_distribution,
                    buyer_authority_summary, timeline_summary,
                    keyword_spike_count, keyword_spike_keywords,
                    keyword_trend_summary,
                    last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                          $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                          $22, $23, $24, $25)
                ON CONFLICT (vendor_name, COALESCE(product_category, '')) DO UPDATE SET
                    total_reviews = EXCLUDED.total_reviews,
                    negative_reviews = EXCLUDED.negative_reviews,
                    churn_intent_count = EXCLUDED.churn_intent_count,
                    avg_urgency_score = EXCLUDED.avg_urgency_score,
                    avg_rating_normalized = EXCLUDED.avg_rating_normalized,
                    nps_proxy = EXCLUDED.nps_proxy,
                    top_pain_categories = EXCLUDED.top_pain_categories,
                    top_competitors = EXCLUDED.top_competitors,
                    top_feature_gaps = EXCLUDED.top_feature_gaps,
                    price_complaint_rate = EXCLUDED.price_complaint_rate,
                    decision_maker_churn_rate = EXCLUDED.decision_maker_churn_rate,
                    company_churn_list = EXCLUDED.company_churn_list,
                    quotable_evidence = EXCLUDED.quotable_evidence,
                    top_use_cases = EXCLUDED.top_use_cases,
                    top_integration_stacks = EXCLUDED.top_integration_stacks,
                    budget_signal_summary = EXCLUDED.budget_signal_summary,
                    sentiment_distribution = EXCLUDED.sentiment_distribution,
                    buyer_authority_summary = EXCLUDED.buyer_authority_summary,
                    timeline_summary = EXCLUDED.timeline_summary,
                    keyword_spike_count = EXCLUDED.keyword_spike_count,
                    keyword_spike_keywords = EXCLUDED.keyword_spike_keywords,
                    keyword_trend_summary = EXCLUDED.keyword_trend_summary,
                    last_computed_at = EXCLUDED.last_computed_at
                """,
                vendor,
                category,
                total,
                neg_lookup.get(vendor, 0),
                vs.get("churn_intent", 0),
                vs.get("avg_urgency", 0),
                vs.get("avg_rating_normalized"),
                nps,
                json.dumps(pain_lookup.get(vendor, [])[:5]),
                json.dumps(competitor_lookup.get(vendor, [])[:5]),
                json.dumps(feature_gap_lookup.get(vendor, [])[:5]),
                price_lookup.get(vendor),
                dm_lookup.get(vendor),
                json.dumps(company_lookup.get(vendor, [])[:20]),
                json.dumps(quote_lookup.get(vendor, [])[:10]),
                json.dumps(use_case_lookup.get(vendor, [])[:10]),
                json.dumps(integration_lookup.get(vendor, [])[:10]),
                json.dumps(budget_lookup.get(vendor, {})),
                json.dumps(sentiment_lookup.get(vendor, {})),
                json.dumps(buyer_auth_lookup.get(vendor, {})),
                json.dumps(timeline_lookup.get(vendor, [])[:10]),
                kw_data.get("spike_count", 0),
                json.dumps(kw_data.get("spike_keywords", [])),
                json.dumps(kw_data.get("trend_summary", {})),
                now,
            )
        except Exception:
            failures += 1
            logger.exception("Failed to upsert churn signal for %s", vendor)

    return failures


# ------------------------------------------------------------------
# Notification
# ------------------------------------------------------------------


async def _send_notification(task: ScheduledTask, parsed: dict, high_intent: list) -> None:
    """Send ntfy push notification with executive summary."""
    from ...pipelines.notify import send_pipeline_notification

    # Build a custom notification body for churn intelligence
    parts: list[str] = []

    summary = parsed.get("executive_summary", "")
    if summary:
        parts.append(summary.strip())

    # Top vendors under churn pressure
    feed = parsed.get("weekly_churn_feed", [])
    if feed and isinstance(feed, list):
        items = []
        for entry in feed[:5]:
            if isinstance(entry, dict):
                vendor = entry.get("vendor", "Unknown")
                churn_density = entry.get("churn_signal_density", "?")
                urgency = entry.get("avg_urgency") or entry.get("urgency", "?")
                pain = entry.get("top_pain") or entry.get("pain", "")
                score = entry.get("churn_pressure_score", "")
                line = f"- **{vendor}** -- {churn_density}% churn density, urgency {urgency}/10"
                if score:
                    line += f", score {score}"
                if pain:
                    line += f"\n  Top pain: {pain}"
                named = entry.get("named_accounts", [])
                if named:
                    acct_names = [a.get("company", "") for a in named[:3] if a.get("company")]
                    if acct_names:
                        line += f"\n  Named accounts: {', '.join(acct_names)}"
                items.append(line)
        if items:
            parts.append("\n**Vendors Under Churn Pressure**\n" + "\n".join(items))

    message = "\n\n".join(parts) if parts else "Weekly churn intelligence report generated."

    vendor_count = len(feed) if isinstance(feed, list) else 0
    title = f"Atlas: Weekly Churn Feed ({vendor_count} vendor{'s' if vendor_count != 1 else ''} under churn pressure)"

    await send_pipeline_notification(
        message, task,
        title=title,
        default_tags="brain,chart_with_downwards_trend",
    )


# ------------------------------------------------------------------
# Vendor-scoped intelligence report (P1: Vendor Retention)
# ------------------------------------------------------------------


async def generate_vendor_report(
    pool,
    vendor_name: str,
    window_days: int = 90,
) -> dict[str, Any] | None:
    """Generate a structured intelligence report for a specific vendor.

    Returns the report dict (also stored in b2b_intelligence) or None on failure.
    Called by the vendor_targets API or campaign generation pipeline.
    """
    today = date.today()
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=3, alias="r")

    # Fetch signals for this vendor
    rows = await pool.fetch(
        f"""
        SELECT r.id AS review_id, r.vendor_name, r.reviewer_company, r.product_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
               r.enrichment->'buyer_authority'->>'role_type' AS role_type,
               r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
               CASE WHEN r.enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                    THEN (r.enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
               r.enrichment->'timeline'->>'contract_end' AS contract_end,
               r.enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
               r.enrichment->'competitors_mentioned' AS competitors_json,
               r.enrichment->'pain_categories' AS pain_json,
               r.enrichment->'quotable_phrases' AS quotable_phrases,
               r.enrichment->'feature_gaps' AS feature_gaps,
               r.enrichment->'sentiment_trajectory'->>'direction' AS sentiment_direction
        FROM b2b_reviews r
                WHERE {filters}
          AND r.vendor_name ILIKE '%' || $2 || '%'
          AND (r.enrichment->>'urgency_score')::numeric >= 3
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT 500
        """,
        window_days,
        vendor_name,
                sources,
    )

    if not rows:
        return None

    signals = []
    for r in rows:
        d = dict(r)
        d["urgency"] = float(d.get("urgency") or 0)
        comps = d.get("competitors_json")
        if isinstance(comps, str):
            try:
                comps = json.loads(comps)
            except (json.JSONDecodeError, TypeError):
                comps = []
        d["competitors"] = comps if isinstance(comps, list) else []
        signals.append(d)

    total = len(signals)
    high_urgency = [s for s in signals if s["urgency"] >= 8]
    medium_urgency = [s for s in signals if 5 <= s["urgency"] < 8]

    # Pain distribution
    pain_counts: dict[str, int] = {}
    for s in signals:
        pain = _safe_json(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Competitive displacement
    comp_counts: dict[str, int] = {}
    for s in signals:
        for c in s["competitors"]:
            if isinstance(c, dict) and c.get("name"):
                comp_counts[c["name"]] = comp_counts.get(c["name"], 0) + 1

    # Feature gaps
    gap_counts: dict[str, int] = {}
    for s in signals:
        gaps = _safe_json(s.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label:
                gap_counts[label] = gap_counts.get(label, 0) + 1

    # Anonymized quotes (high-urgency only)
    anon_quotes: list[str] = []
    for s in high_urgency[:20]:
        phrases = _safe_json(s.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in anon_quotes:
                anon_quotes.append(text)
            if len(anon_quotes) >= 10:
                break

    report_data = {
        "vendor_name": vendor_name,
        "report_date": str(today),
        "window_days": window_days,
        "signal_count": total,
        "high_urgency_count": len(high_urgency),
        "medium_urgency_count": len(medium_urgency),
        "pain_categories": sorted(
            [{"category": k, "count": v} for k, v in pain_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "competitive_displacement": sorted(
            [{"competitor": k, "count": v} for k, v in comp_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "top_feature_gaps": sorted(
            [{"feature": k, "count": v} for k, v in gap_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "anonymized_quotes": anon_quotes[:10],
    }

    # Persist to b2b_intelligence
    try:
        await pool.execute(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter,
                intelligence_data, executive_summary, data_density, status, llm_model
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            today,
            "vendor_retention",
            vendor_name,
            json.dumps(report_data, default=str),
            f"{total} accounts showing churn signals for {vendor_name}. "
            f"{len(high_urgency)} at critical urgency.",
            json.dumps({
                "signal_count": total,
                "pain_categories": len(pain_counts),
                "competitors": len(comp_counts),
                "feature_gaps": len(gap_counts),
            }),
            "published",
            "pipeline_aggregation",
        )
    except Exception:
        logger.exception("Failed to store vendor report for %s", vendor_name)

    return report_data


async def _fetch_vendor_comparison_rows(
    pool,
    vendor_name: str,
    window_days: int,
) -> list[dict[str, Any]]:
    """Fetch enriched review rows for one vendor comparison side."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=3, alias="r")
    rows = await pool.fetch(
        f"""
        SELECT r.vendor_name, r.reviewer_company, r.product_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (r.enrichment->>'would_recommend')::boolean AS would_recommend,
               r.rating, r.rating_max,
               r.enrichment->'competitors_mentioned' AS competitors_json,
               r.enrichment->'pain_categories' AS pain_json,
               r.enrichment->'quotable_phrases' AS quotable_phrases,
               r.enrichment->'feature_gaps' AS feature_gaps
        FROM b2b_reviews r
        WHERE {filters}
          AND r.vendor_name ILIKE '%' || $2 || '%'
          AND (r.enrichment->>'urgency_score')::numeric >= 3
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT 500
        """,
        window_days,
        vendor_name,
        sources,
    )
    signals: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["urgency"] = float(item.get("urgency") or 0)
        comps = item.get("competitors_json")
        if isinstance(comps, str):
            try:
                comps = json.loads(comps)
            except (json.JSONDecodeError, TypeError):
                comps = []
        item["competitors"] = comps if isinstance(comps, list) else []
        signals.append(item)
    return signals


def _build_vendor_comparison_snapshot(
    vendor_name: str,
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize comparison metrics for one vendor."""
    pain_counts: dict[str, int] = {}
    comp_counts: dict[str, int] = {}
    gap_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    quote_highlights: list[str] = []
    company_examples: list[str] = []
    positive_reviews = recommend_yes = recommend_no = churn_intent = 0
    rating_sum = rating_count = 0
    for signal in signals:
        if signal.get("intent_to_leave") is True:
            churn_intent += 1
        if signal.get("would_recommend") is True:
            recommend_yes += 1
        elif signal.get("would_recommend") is False:
            recommend_no += 1
        rating = signal.get("rating")
        rating_max = signal.get("rating_max")
        if rating is not None and rating_max:
            normalized = float(rating) / float(rating_max)
            rating_sum += normalized
            rating_count += 1
            if normalized >= 0.7:
                positive_reviews += 1
        category = signal.get("product_category")
        if category:
            category_counts[str(category)] = category_counts.get(str(category), 0) + 1
        company = signal.get("reviewer_company")
        if company and company not in company_examples:
            company_examples.append(str(company))
        for pain in _safe_json(signal.get("pain_json")):
            if isinstance(pain, dict) and pain.get("category"):
                key = str(pain["category"])
                pain_counts[key] = pain_counts.get(key, 0) + 1
        for comp in signal.get("competitors", []):
            if isinstance(comp, dict) and comp.get("name"):
                name = _canonicalize_competitor(str(comp["name"]))
                comp_counts[name] = comp_counts.get(name, 0) + 1
        for gap in _safe_json(signal.get("feature_gaps")):
            label = gap if isinstance(gap, str) else (gap.get("feature", "") if isinstance(gap, dict) else "")
            if label:
                gap_counts[str(label)] = gap_counts.get(str(label), 0) + 1
        for phrase in _safe_json(signal.get("quotable_phrases")):
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in quote_highlights:
                quote_highlights.append(str(text))
            if len(quote_highlights) >= 5:
                break
    total = len(signals)
    avg_urgency = round(sum(float(s.get("urgency") or 0) for s in signals) / total, 2) if total else 0.0
    avg_rating = round((rating_sum / rating_count) * 100, 1) if rating_count else None
    positive_pct = round((positive_reviews * 100.0) / rating_count, 1) if rating_count else None
    recommend_ratio = round(((recommend_yes - recommend_no) * 100.0) / total, 1) if total else 0.0
    return {
        "vendor_name": vendor_name,
        "signal_count": total,
        "high_urgency_count": sum(1 for s in signals if float(s.get("urgency") or 0) >= 8),
        "churn_intent_count": churn_intent,
        "churn_signal_density": round((churn_intent * 100.0) / total, 1) if total else 0.0,
        "avg_urgency_score": avg_urgency,
        "avg_rating_normalized": avg_rating,
        "positive_review_pct": positive_pct,
        "recommend_ratio": recommend_ratio,
        "product_categories": sorted([{"category": k, "count": v} for k, v in category_counts.items()], key=lambda x: x["count"], reverse=True)[:5],
        "top_pain_categories": sorted([{"category": k, "count": v} for k, v in pain_counts.items()], key=lambda x: x["count"], reverse=True)[:5],
        "top_competitors": sorted([{"competitor": k, "count": v} for k, v in comp_counts.items()], key=lambda x: x["count"], reverse=True)[:5],
        "top_feature_gaps": sorted([{"feature": k, "count": v} for k, v in gap_counts.items()], key=lambda x: x["count"], reverse=True)[:5],
        "company_examples": company_examples[:10],
        "quote_highlights": quote_highlights[:5],
    }


async def _fetch_vendor_head_to_head(
    pool,
    primary_vendor: str,
    comparison_vendor: str,
    window_days: int,
) -> list[dict[str, Any]]:
    """Count direct displacement mentions between two vendors."""
    primary_norm = _canonicalize_vendor(primary_vendor).lower()
    comparison_norm = _canonicalize_vendor(comparison_vendor).lower()
    primary_rows = await _fetch_vendor_comparison_rows(pool, primary_vendor, window_days)
    comparison_rows = await _fetch_vendor_comparison_rows(pool, comparison_vendor, window_days)
    counts = {
        (primary_vendor, comparison_vendor): {"count": 0, "companies": []},
        (comparison_vendor, primary_vendor): {"count": 0, "companies": []},
    }
    for from_vendor, to_vendor, rows in [
        (primary_vendor, comparison_vendor, primary_rows),
        (comparison_vendor, primary_vendor, comparison_rows),
    ]:
        target_norm = comparison_norm if from_vendor == primary_vendor else primary_norm
        bucket = counts[(from_vendor, to_vendor)]
        for row in rows:
            names = [
                _canonicalize_competitor(str(comp.get("name") or "")).lower()
                for comp in row.get("competitors", []) if isinstance(comp, dict)
            ]
            if target_norm in names:
                bucket["count"] += 1
                company = row.get("reviewer_company")
                if company and company not in bucket["companies"]:
                    bucket["companies"].append(str(company))
    return [
        {
            "name": f"{from_vendor} -> {to_vendor}",
            "count": bucket["count"],
            "companies": bucket["companies"][:5],
        }
        for (from_vendor, to_vendor), bucket in counts.items()
    ]


def _build_vendor_comparison_summary(
    primary_snapshot: dict[str, Any],
    comparison_snapshot: dict[str, Any],
    head_to_head: list[dict[str, Any]],
) -> str:
    """Build a concise executive summary for a head-to-head vendor comparison."""
    snapshots = [primary_snapshot, comparison_snapshot]
    higher_risk = max(snapshots, key=lambda item: (item.get("churn_signal_density", 0), item.get("avg_urgency_score", 0)))
    lower_risk = comparison_snapshot if higher_risk is primary_snapshot else primary_snapshot
    stronger_sentiment = max(
        snapshots,
        key=lambda item: (item.get("positive_review_pct") or 0, item.get("recommend_ratio") or 0),
    )
    weaker_sentiment = comparison_snapshot if stronger_sentiment is primary_snapshot else primary_snapshot
    flow_text = "; ".join(
        f"{flow['name']} has {flow['count']} direct mentions"
        for flow in head_to_head if int(flow.get("count") or 0) > 0
    ) or "No direct displacement mentions were observed between the two vendors"
    high_pain = (higher_risk.get("top_pain_categories") or [{}])[0]
    low_pain = (lower_risk.get("top_pain_categories") or [{}])[0]
    return (
        f"{higher_risk['vendor_name']} shows the heavier churn signal pressure versus {lower_risk['vendor_name']}, "
        f"with {higher_risk['churn_intent_count']} of {higher_risk['signal_count']} reviews ({higher_risk['churn_signal_density']}%) "
        f"mentioning churn intent compared with {lower_risk['churn_intent_count']} of {lower_risk['signal_count']} "
        f"({lower_risk['churn_signal_density']}%). {flow_text}. Sentiment currently favors {stronger_sentiment['vendor_name']} "
        f"over {weaker_sentiment['vendor_name']} on positive-review share ({stronger_sentiment.get('positive_review_pct')}% vs "
        f"{weaker_sentiment.get('positive_review_pct')}%). Top pain themes diverge between {higher_risk['vendor_name']} "
        f"({high_pain.get('category', 'insufficient_data')}) and {lower_risk['vendor_name']} "
        f"({low_pain.get('category', 'insufficient_data')})."
    )


async def generate_vendor_comparison_report(
    pool,
    primary_vendor: str,
    comparison_vendor: str,
    window_days: int = 90,
    persist: bool = True,
) -> dict[str, Any] | None:
    """Generate a deterministic head-to-head comparison report for two vendors."""
    primary_name = primary_vendor.strip()
    comparison_name = comparison_vendor.strip()
    if not primary_name or not comparison_name:
        return None
    if _canonicalize_vendor(primary_name).lower() == _canonicalize_vendor(comparison_name).lower():
        return None
    primary_rows = await _fetch_vendor_comparison_rows(pool, primary_name, window_days)
    comparison_rows = await _fetch_vendor_comparison_rows(pool, comparison_name, window_days)
    if not primary_rows or not comparison_rows:
        return None
    today = date.today()
    primary_snapshot = _build_vendor_comparison_snapshot(primary_name, primary_rows)
    comparison_snapshot = _build_vendor_comparison_snapshot(comparison_name, comparison_rows)
    head_to_head = await _fetch_vendor_head_to_head(pool, primary_name, comparison_name, window_days)
    shared_pains = sorted(
        {row["category"] for row in primary_snapshot["top_pain_categories"]} &
        {row["category"] for row in comparison_snapshot["top_pain_categories"]}
    )
    report_data = {
        "primary_vendor": primary_name,
        "comparison_vendor": comparison_name,
        "report_date": str(today),
        "window_days": window_days,
        "executive_summary": _build_vendor_comparison_summary(primary_snapshot, comparison_snapshot, head_to_head),
        "primary_metrics": {k: v for k, v in primary_snapshot.items() if k not in {"top_pain_categories", "top_competitors", "top_feature_gaps", "company_examples", "quote_highlights", "product_categories"}},
        "comparison_metrics": {k: v for k, v in comparison_snapshot.items() if k not in {"top_pain_categories", "top_competitors", "top_feature_gaps", "company_examples", "quote_highlights", "product_categories"}},
        "primary_top_pains": primary_snapshot["top_pain_categories"],
        "comparison_top_pains": comparison_snapshot["top_pain_categories"],
        "primary_top_competitors": primary_snapshot["top_competitors"],
        "comparison_top_competitors": comparison_snapshot["top_competitors"],
        "primary_top_feature_gaps": primary_snapshot["top_feature_gaps"],
        "comparison_top_feature_gaps": comparison_snapshot["top_feature_gaps"],
        "primary_product_categories": primary_snapshot["product_categories"],
        "comparison_product_categories": comparison_snapshot["product_categories"],
        "primary_company_examples": primary_snapshot["company_examples"],
        "comparison_company_examples": comparison_snapshot["company_examples"],
        "primary_quote_highlights": primary_snapshot["quote_highlights"],
        "comparison_quote_highlights": comparison_snapshot["quote_highlights"],
        "direct_displacement": head_to_head,
        "shared_pain_categories": shared_pains,
    }
    if persist:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter, category_filter,
                intelligence_data, executive_summary, data_density, status, llm_model
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            today,
            "vendor_comparison",
            primary_name,
            comparison_name,
            json.dumps(report_data, default=str),
            report_data["executive_summary"],
            json.dumps({
                "primary_signal_count": primary_snapshot["signal_count"],
                "comparison_signal_count": comparison_snapshot["signal_count"],
                "shared_pain_count": len(shared_pains),
                "direct_flow_mentions": sum(int(item.get("count") or 0) for item in head_to_head),
            }),
            "published",
            "pipeline_aggregation",
        )
        if row:
            report_data["report_id"] = str(row["id"])
    return report_data


async def _fetch_company_comparison_rows(
    pool,
    company_name: str,
    window_days: int,
) -> list[dict[str, Any]]:
    """Fetch enriched review rows for a named reviewer company."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=3, alias="r")
    rows = await pool.fetch(
        f"""
        SELECT r.vendor_name, r.reviewer_company, r.product_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
               r.enrichment->'reviewer_context'->>'role_level' AS role_level,
               r.enrichment->>'pain_category' AS pain,
               r.enrichment->'competitors_mentioned' AS competitors_json,
               r.enrichment->'quotable_phrases' AS quotable_phrases,
               r.enrichment->'feature_gaps' AS feature_gaps,
               r.enrichment->'timeline'->>'contract_end' AS contract_end,
               r.enrichment->'timeline'->>'evaluation_deadline' AS evaluation_deadline,
               r.enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
               r.enrichment->'contract_context'->>'contract_value_signal' AS contract_value_signal
        FROM b2b_reviews r
        WHERE {filters}
          AND LOWER(r.reviewer_company) = LOWER($2)
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT 100
        """,
        window_days,
        company_name,
        sources,
    )
    return [dict(row) for row in rows]


def _build_company_comparison_snapshot(
    company_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic company-level churn snapshot."""
    vendors: dict[str, int] = {}
    categories: dict[str, int] = {}
    pains: dict[str, int] = {}
    alternatives: dict[str, int] = {}
    gaps: dict[str, int] = {}
    role_levels: dict[str, int] = {}
    timeline_signals: list[dict[str, Any]] = []
    contract_signals: list[str] = []
    quote_highlights: list[str] = []
    decision_maker_count = 0
    churn_mentions = 0
    urgencies: list[float] = []
    for row in rows:
        vendor_name = str(row.get("vendor_name") or "")
        if vendor_name:
            vendors[vendor_name] = vendors.get(vendor_name, 0) + 1
        category = str(row.get("product_category") or "")
        if category:
            categories[category] = categories.get(category, 0) + 1
        pain = str(row.get("pain") or "")
        if pain:
            pains[pain] = pains.get(pain, 0) + 1
        if row.get("intent_to_leave") is True:
            churn_mentions += 1
        if row.get("is_dm") is True:
            decision_maker_count += 1
        role_level = str(row.get("role_level") or "")
        if role_level:
            role_levels[role_level] = role_levels.get(role_level, 0) + 1
        urgency = float(row.get("urgency") or 0)
        urgencies.append(urgency)
        for comp in _safe_json(row.get("competitors_json")):
            if isinstance(comp, dict) and comp.get("name"):
                name = _canonicalize_competitor(str(comp["name"]))
                alternatives[name] = alternatives.get(name, 0) + 1
        for gap in _safe_json(row.get("feature_gaps")):
            label = gap if isinstance(gap, str) else (gap.get("feature", "") if isinstance(gap, dict) else "")
            if label:
                gaps[str(label)] = gaps.get(str(label), 0) + 1
        for phrase in _safe_json(row.get("quotable_phrases")):
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in quote_highlights:
                quote_highlights.append(str(text))
            if len(quote_highlights) >= 5:
                break
        timeline_item = {
            "vendor": vendor_name,
            "contract_end": row.get("contract_end"),
            "evaluation_deadline": row.get("evaluation_deadline"),
            "decision_timeline": row.get("decision_timeline"),
            "urgency": urgency,
        }
        if timeline_item["contract_end"] or timeline_item["evaluation_deadline"] or timeline_item["decision_timeline"]:
            timeline_signals.append(timeline_item)
        contract_signal = str(row.get("contract_value_signal") or "")
        if contract_signal and contract_signal not in contract_signals:
            contract_signals.append(contract_signal)
    signal_count = len(rows)
    return {
        "company_name": company_name,
        "signal_count": signal_count,
        "avg_urgency_score": round(sum(urgencies) / signal_count, 2) if signal_count else 0.0,
        "max_urgency_score": max(urgencies) if urgencies else 0.0,
        "decision_maker_signals": decision_maker_count,
        "churn_intent_count": churn_mentions,
        "current_vendors": sorted([{"vendor": k, "count": v} for k, v in vendors.items()], key=lambda x: x["count"], reverse=True)[:5],
        "product_categories": sorted([{"category": k, "count": v} for k, v in categories.items()], key=lambda x: x["count"], reverse=True)[:5],
        "top_pain_categories": sorted([{"category": k, "count": v} for k, v in pains.items()], key=lambda x: x["count"], reverse=True)[:5],
        "alternatives_considered": sorted([{"name": k, "count": v} for k, v in alternatives.items()], key=lambda x: x["count"], reverse=True)[:5],
        "top_feature_gaps": sorted([{"feature": k, "count": v} for k, v in gaps.items()], key=lambda x: x["count"], reverse=True)[:5],
        "role_levels": sorted([{"role": k, "count": v} for k, v in role_levels.items()], key=lambda x: x["count"], reverse=True)[:5],
        "timeline_signals": timeline_signals[:5],
        "contract_value_signals": contract_signals[:5],
        "quote_highlights": quote_highlights[:5],
    }


def _build_company_comparison_summary(
    primary_snapshot: dict[str, Any],
    comparison_snapshot: dict[str, Any],
) -> str:
    """Build a concise account-vs-account executive summary."""
    primary_vendor = ((primary_snapshot.get("current_vendors") or [{}])[0] or {}).get("vendor", "unknown vendor")
    comparison_vendor = ((comparison_snapshot.get("current_vendors") or [{}])[0] or {}).get("vendor", "unknown vendor")
    higher_urgency = primary_snapshot if primary_snapshot.get("avg_urgency_score", 0) >= comparison_snapshot.get("avg_urgency_score", 0) else comparison_snapshot
    lower_urgency = comparison_snapshot if higher_urgency is primary_snapshot else primary_snapshot
    primary_pain = ((primary_snapshot.get("top_pain_categories") or [{}])[0] or {}).get("category", "insufficient_data")
    comparison_pain = ((comparison_snapshot.get("top_pain_categories") or [{}])[0] or {}).get("category", "insufficient_data")
    shared_alts = sorted(
        {item["name"] for item in primary_snapshot.get("alternatives_considered", [])} &
        {item["name"] for item in comparison_snapshot.get("alternatives_considered", [])}
    )
    shared_alt_text = ", ".join(shared_alts[:3]) if shared_alts else "no shared alternatives"
    return (
        f"{higher_urgency['company_name']} is the hotter account signal versus {lower_urgency['company_name']}, "
        f"with average urgency {higher_urgency['avg_urgency_score']} against {lower_urgency['avg_urgency_score']}. "
        f"{primary_snapshot['company_name']} is currently tied to {primary_vendor} and is primarily citing {primary_pain}, "
        f"while {comparison_snapshot['company_name']} is tied to {comparison_vendor} and is primarily citing {comparison_pain}. "
        f"The two accounts share {shared_alt_text} in their evaluation sets, with churn intent appearing in "
        f"{primary_snapshot['churn_intent_count']} of {primary_snapshot['signal_count']} versus "
        f"{comparison_snapshot['churn_intent_count']} of {comparison_snapshot['signal_count']} company records."
    )


def _build_company_deep_dive_summary(snapshot: dict[str, Any]) -> str:
    """Build a concise executive summary for a single account deep dive."""
    top_vendor = ((snapshot.get("current_vendors") or [{}])[0] or {}).get("vendor", "unknown vendor")
    top_pain = ((snapshot.get("top_pain_categories") or [{}])[0] or {}).get("category", "insufficient_data")
    top_alt = ((snapshot.get("alternatives_considered") or [{}])[0] or {}).get("name", "no named alternative")
    return (
        f"{snapshot['company_name']} currently shows {snapshot['signal_count']} company-level review signals tied most strongly to {top_vendor}, "
        f"with average urgency {snapshot['avg_urgency_score']} and churn intent present in {snapshot['churn_intent_count']} of those records. "
        f"The leading pain theme is {top_pain}, while the most visible evaluated alternative is {top_alt}. "
        f"Decision-maker participation appears in {snapshot['decision_maker_signals']} records, highlighting how active this account appears in renewal or migration evaluation."
    )


async def generate_company_deep_dive_report(
    pool,
    company_name: str,
    window_days: int = 90,
    persist: bool = True,
    account_id: Any = None,
) -> dict[str, Any] | None:
    """Generate a deterministic deep-dive report for one reviewer company."""
    normalized_name = company_name.strip()
    if not normalized_name:
        return None
    rows = await _fetch_company_comparison_rows(pool, normalized_name, window_days)
    if not rows:
        return None
    today = date.today()
    snapshot = _build_company_comparison_snapshot(normalized_name, rows)
    report_data = {
        "company_name": normalized_name,
        "report_date": str(today),
        "window_days": window_days,
        "executive_summary": _build_company_deep_dive_summary(snapshot),
        "company_metrics": snapshot,
    }
    if persist:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter, category_filter,
                intelligence_data, executive_summary, data_density, status, llm_model, account_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """,
            today,
            "account_deep_dive",
            normalized_name,
            None,
            json.dumps(report_data, default=str),
            report_data["executive_summary"],
            json.dumps({
                "signal_count": snapshot["signal_count"],
                "decision_maker_signals": snapshot["decision_maker_signals"],
                "timeline_signal_count": len(snapshot["timeline_signals"]),
                "alternative_count": len(snapshot["alternatives_considered"]),
            }),
            "published",
            "pipeline_aggregation",
            account_id,
        )
        if row:
            report_data["report_id"] = str(row["id"])
    return report_data


async def generate_company_comparison_report(
    pool,
    primary_company: str,
    comparison_company: str,
    window_days: int = 90,
    persist: bool = True,
    account_id: Any = None,
) -> dict[str, Any] | None:
    """Generate a deterministic reviewer-company versus reviewer-company report."""
    primary_name = primary_company.strip()
    comparison_name = comparison_company.strip()
    if not primary_name or not comparison_name:
        return None
    if primary_name.lower() == comparison_name.lower():
        return None
    primary_rows = await _fetch_company_comparison_rows(pool, primary_name, window_days)
    comparison_rows = await _fetch_company_comparison_rows(pool, comparison_name, window_days)
    if not primary_rows or not comparison_rows:
        return None
    today = date.today()
    primary_snapshot = _build_company_comparison_snapshot(primary_name, primary_rows)
    comparison_snapshot = _build_company_comparison_snapshot(comparison_name, comparison_rows)
    shared_alternatives = sorted(
        {item["name"] for item in primary_snapshot["alternatives_considered"]} &
        {item["name"] for item in comparison_snapshot["alternatives_considered"]}
    )
    shared_vendors = sorted(
        {item["vendor"] for item in primary_snapshot["current_vendors"]} &
        {item["vendor"] for item in comparison_snapshot["current_vendors"]}
    )
    report_data = {
        "primary_company": primary_name,
        "comparison_company": comparison_name,
        "report_date": str(today),
        "window_days": window_days,
        "executive_summary": _build_company_comparison_summary(primary_snapshot, comparison_snapshot),
        "primary_company_metrics": primary_snapshot,
        "comparison_company_metrics": comparison_snapshot,
        "shared_alternatives": shared_alternatives,
        "shared_vendors": shared_vendors,
        "urgency_gap": round(abs(primary_snapshot["avg_urgency_score"] - comparison_snapshot["avg_urgency_score"]), 2),
    }
    if persist:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter, category_filter,
                intelligence_data, executive_summary, data_density, status, llm_model, account_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """,
            today,
            "account_comparison",
            primary_name,
            comparison_name,
            json.dumps(report_data, default=str),
            report_data["executive_summary"],
            json.dumps({
                "primary_signal_count": primary_snapshot["signal_count"],
                "comparison_signal_count": comparison_snapshot["signal_count"],
                "shared_alternative_count": len(shared_alternatives),
                "shared_vendor_count": len(shared_vendors),
            }),
            "published",
            "pipeline_aggregation",
            account_id,
        )
        if row:
            report_data["report_id"] = str(row["id"])
    return report_data


# ------------------------------------------------------------------
# Challenger-scoped intelligence report (P2: Challenger Intel)
# ------------------------------------------------------------------


async def generate_challenger_report(
    pool,
    challenger_name: str,
    window_days: int = 90,
) -> dict[str, Any] | None:
    """Generate a structured intelligence report for a challenger target.

    Queries reviews where *challenger_name* appears in the enrichment
    ``competitors_mentioned`` array (i.e. reviewers of *other* vendors
    who are considering switching to this challenger).

    Returns the report dict (also stored in b2b_intelligence) or None
    when no matching signals exist.
    """
    today = date.today()
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=3, alias="r")

    rows = await pool.fetch(
        f"""
        SELECT r.id AS review_id, r.vendor_name, r.reviewer_company, r.product_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
               r.enrichment->'buyer_authority'->>'role_type' AS role_type,
               r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
               CASE WHEN r.enrichment->'budget_signals'->>'seat_count' ~ '^\\d+$'
                    THEN (r.enrichment->'budget_signals'->>'seat_count')::int END AS seat_count,
               r.enrichment->'competitors_mentioned' AS competitors_json,
               r.enrichment->'pain_categories' AS pain_json,
               r.enrichment->'quotable_phrases' AS quotable_phrases,
               r.enrichment->'feature_gaps' AS feature_gaps
        FROM b2b_reviews r
                WHERE {filters}
          AND (r.enrichment->>'urgency_score')::numeric >= 3
          AND EXISTS (
                SELECT 1 FROM jsonb_array_elements(r.enrichment->'competitors_mentioned') AS comp(value)
                WHERE comp.value->>'name' ILIKE '%' || $2 || '%'
              )
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT 500
        """,
        window_days,
        challenger_name,
        sources,
    )

    if not rows:
        return None

    signals = []
    for r in rows:
        d = dict(r)
        d["urgency"] = float(d.get("urgency") or 0)
        comps = d.get("competitors_json")
        if isinstance(comps, str):
            try:
                comps = json.loads(comps)
            except (json.JSONDecodeError, TypeError):
                comps = []
        d["competitors"] = comps if isinstance(comps, list) else []
        signals.append(d)

    total = len(signals)
    high_urgency = [s for s in signals if s["urgency"] >= 8]
    medium_urgency = [s for s in signals if 5 <= s["urgency"] < 8]

    # Buying stage distribution
    stage_counts: dict[str, int] = {}
    for s in signals:
        stage = s.get("buying_stage")
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    by_buying_stage = {
        "active_purchase": stage_counts.get("active_purchase", 0),
        "evaluation": stage_counts.get("evaluation", 0),
        "renewal_decision": stage_counts.get("renewal_decision", 0),
    }

    # Role distribution
    role_counts: dict[str, int] = {}
    for s in signals:
        role = s.get("role_type")
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1

    # Pain driving switch
    pain_counts: dict[str, int] = {}
    for s in signals:
        pain = _safe_json(s.get("pain_json"))
        for p in pain:
            if isinstance(p, dict) and p.get("category"):
                pain_counts[p["category"]] = pain_counts.get(p["category"], 0) + 1

    # Incumbents losing (the vendor_name on each review is the incumbent)
    incumbent_counts: dict[str, int] = {}
    for s in signals:
        vname = s.get("vendor_name")
        if vname:
            incumbent_counts[vname] = incumbent_counts.get(vname, 0) + 1

    # Seat count distribution
    large = mid = small = 0
    for s in signals:
        sc = s.get("seat_count")
        if sc is not None:
            if sc >= 500:
                large += 1
            elif sc >= 100:
                mid += 1
            else:
                small += 1

    # Incumbent feature gaps (what incumbents are missing)
    gap_counts: dict[str, int] = {}
    for s in signals:
        gaps = _safe_json(s.get("feature_gaps"))
        for g in gaps:
            label = g if isinstance(g, str) else (g.get("feature", "") if isinstance(g, dict) else "")
            if label:
                gap_counts[label] = gap_counts.get(label, 0) + 1

    # Feature mentions (challenger features reviewers cite)
    feature_set: list[str] = []
    for s in signals:
        for c in s["competitors"]:
            if isinstance(c, dict):
                cname = (c.get("name") or "").lower()
                if cname and challenger_name.lower() in cname:
                    for feat in c.get("features", []):
                        if isinstance(feat, str) and feat not in feature_set:
                            feature_set.append(feat)

    # Anonymized quotes (high-urgency only)
    anon_quotes: list[str] = []
    for s in high_urgency[:20]:
        phrases = _safe_json(s.get("quotable_phrases"))
        for phrase in phrases:
            text = phrase if isinstance(phrase, str) else (phrase.get("text", "") if isinstance(phrase, dict) else "")
            if text and text not in anon_quotes:
                anon_quotes.append(text)
            if len(anon_quotes) >= 10:
                break

    report_data = {
        "challenger_name": challenger_name,
        "report_date": str(today),
        "window_days": window_days,
        "signal_count": total,
        "high_urgency_count": len(high_urgency),
        "medium_urgency_count": len(medium_urgency),
        "by_buying_stage": by_buying_stage,
        "role_distribution": sorted(
            [{"role": k, "count": v} for k, v in role_counts.items()],
            key=lambda x: x["count"], reverse=True,
        ),
        "pain_driving_switch": sorted(
            [{"category": k, "count": v} for k, v in pain_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "incumbents_losing": sorted(
            [{"name": k, "count": v} for k, v in incumbent_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "seat_count_signals": {
            "large_500plus": large,
            "mid_100_499": mid,
            "small_under_100": small,
        },
        "incumbent_feature_gaps": sorted(
            [{"feature": k, "count": v} for k, v in gap_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:10],
        "feature_mentions": feature_set[:20],
        "anonymized_quotes": anon_quotes[:10],
    }

    # Persist to b2b_intelligence
    try:
        await pool.execute(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter,
                intelligence_data, executive_summary, data_density, status, llm_model
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            today,
            "challenger_intel",
            challenger_name,
            json.dumps(report_data, default=str),
            f"{total} accounts mentioning {challenger_name} as alternative. "
            f"{len(high_urgency)} at critical urgency.",
            json.dumps({
                "signal_count": total,
                "buying_stages": len(stage_counts),
                "incumbents": len(incumbent_counts),
                "feature_gaps": len(gap_counts),
            }),
            "published",
            "pipeline_aggregation",
        )
    except Exception:
        logger.exception("Failed to store challenger report for %s", challenger_name)

    return report_data
