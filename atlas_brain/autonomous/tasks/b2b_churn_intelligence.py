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
import math
import uuid as _uuid
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...services.tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)
from ...services.company_normalization import normalize_company_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...services.scraping.sources import parse_source_allowlist, display_name as _source_display_name, VERIFIED_SOURCES
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


def _build_llm_trace_metadata(
    phase: str,
    *,
    report_type: str | None = None,
    vendor_name: str | None = None,
) -> dict[str, Any]:
    """Build compact trace metadata for churn-intelligence LLM phases."""
    metadata: dict[str, Any] = {
        "workflow": "b2b_churn_intelligence",
        "phase": phase,
    }
    if report_type:
        metadata["report_type"] = report_type
    if vendor_name:
        metadata["vendor_name"] = vendor_name
    business = build_business_trace_context(
        workflow="b2b_churn_intelligence",
        report_type=report_type,
        vendor_name=vendor_name,
    )
    if business:
        metadata["business"] = business
    return metadata


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
    parts.append(
        f"NOT EXISTS (SELECT 1 FROM data_corrections dc"
        f" WHERE dc.entity_type = 'review' AND dc.entity_id = {p}id"
        f" AND dc.correction_type = 'suppress' AND dc.status = 'applied')"
    )
    # Exclude reviews from suppressed sources (global or vendor-scoped)
    parts.append(
        f"NOT EXISTS (SELECT 1 FROM data_corrections dc2"
        f" WHERE dc2.entity_type = 'source'"
        f" AND dc2.correction_type = 'suppress_source'"
        f" AND dc2.status = 'applied'"
        f" AND LOWER(dc2.metadata->>'source_name') = LOWER({p}source)"
        f" AND (dc2.field_name IS NULL OR LOWER(dc2.field_name) = LOWER({p}vendor_name))"
        f")"
    )
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

    # Build lookup of real quotes from source data (handles both str and dict quotes)
    real_quotes: set[str] = set()
    for h in source_high_intent:
        for q in h.get("quotes", []):
            text = _quote_text(q)
            if text:
                real_quotes.add(text)
    for qe in source_quotable:
        for q in qe.get("quotes", []):
            text = _quote_text(q)
            if text:
                real_quotes.add(text)

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
    report_type: str = "weekly_churn_feed",
) -> str:
    """Build a concise deterministic executive summary from validated structured data.

    Each *report_type* gets a tailored summary.  Shared helper ``_summary_preamble``
    produces the opening sentence; the body varies per type.
    """
    feed = parsed.get("weekly_churn_feed", [])
    if not isinstance(feed, list) or not feed:
        return parsed.get("executive_summary", "")

    period = data_context.get("enrichment_period") or {}
    start = period.get("earliest")
    end = period.get("latest")
    window_label = f"Between {start} and {end}" if start and end else "In the current analysis window"

    source_dist = data_context.get("source_distribution") or {}
    total_review_count = sum(
        int((source_dist.get(source) or {}).get("reviews") or 0)
        for source in executive_sources
    )
    source_labels = [_source_display_name(source) for source in executive_sources]
    source_label_text = ", ".join(source_labels)

    # --- per-report-type summaries ------------------------------------

    if report_type == "vendor_scorecard":
        scorecards = parsed.get("vendor_scorecards", [])
        n = len(scorecards)
        if not scorecards:
            return ""

        # Risk distribution
        risk_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for s in scorecards:
            rl = s.get("risk_level", "low")
            if rl in risk_counts:
                risk_counts[rl] += 1
        risk_parts = []
        for level in ("high", "medium", "low"):
            if risk_counts[level]:
                risk_parts.append(f"{risk_counts[level]} {level}-risk")

        # Sentence 1: scale + risk distribution
        lines = [
            f"{window_label}, Atlas scored {n} vendors on churn pressure"
            + (f" from {total_review_count:,} reviews across {source_label_text}"
               if total_review_count else f" across {source_label_text}")
            + f" \u2014 {', '.join(risk_parts)}."
        ]

        # Sentence 2: top-pressure vendor, score, driver, DM rate
        top = scorecards[0]  # sorted by urgency+density descending
        top_vendor = top.get("vendor", "")
        top_score = top.get("churn_pressure_score", 0)
        top_density = top.get("churn_signal_density", 0)
        top_pain = top.get("top_pain", "")
        dm_rate = top.get("dm_churn_rate", 0)

        s2 = f"{top_vendor} scored highest at {top_score:.1f}"
        s2 += f" ({top_density}% churn density \u2014 share of reviews containing explicit switching signals)"
        if top_pain and top_pain.lower() not in ("other", "unknown"):
            s2 += f", driven primarily by {top_pain} complaints"
        if dm_rate and dm_rate >= 0.3:
            s2 += f"; {dm_rate:.0%} of switching signals came from decision-makers"
        s2 += "."
        lines.append(s2)

        # Sentence 3: trend direction (only when there's a clear skew)
        trend_counts: dict[str, int] = {"worsening": 0, "improving": 0, "stable": 0, "new": 0}
        for s in scorecards:
            t = s.get("trend", "stable")
            if t in trend_counts:
                trend_counts[t] += 1
        if trend_counts["worsening"] > trend_counts["improving"]:
            lines.append(
                f"{trend_counts['worsening']} vendors are trending worse than last period"
                + (f", {trend_counts['improving']} improving." if trend_counts["improving"] else ".")
            )
        elif trend_counts["improving"] > trend_counts["worsening"]:
            lines.append(
                f"{trend_counts['improving']} vendors are improving"
                + (f", {trend_counts['worsening']} worsening." if trend_counts["worsening"] else ".")
            )

        # Sentence 4: displacement direction for highest-risk vendor
        top_threat = top.get("top_competitor_threat", "")
        if top_threat and "Insufficient" not in top_threat:
            lines.append(f"High-risk vendors are losing ground primarily to {top_threat}.")

        return " ".join(lines)

    if report_type == "displacement_report":
        disp = parsed.get("displacement_map", [])
        if not disp:
            return ""
        n_edges = len(disp)
        from_vendors = {e.get("from_vendor") for e in disp if e.get("from_vendor")}
        total_mentions = sum(e.get("mention_count", 0) for e in disp)

        # Sentence 1: scale
        lines = [
            f"{window_label}, Atlas tracked {n_edges} competitive displacement flows"
            f" across {len(from_vendors)} vendors"
            + (f" from {total_review_count:,} reviews" if total_review_count else "")
            + f", totaling {total_mentions:,} switching mentions."
        ]

        # Sentence 2: strongest flow
        top = disp[0]  # sorted by mention_count descending
        top_from = top.get("from_vendor", "")
        top_to = top.get("to_vendor", "")
        top_count = top.get("mention_count", 0)
        top_driver = top.get("primary_driver", "")
        s2 = f"The strongest flow is {top_from} \u2192 {top_to} ({top_count} mentions)"
        if top_driver and top_driver.lower() != "other":
            s2 += f", driven by {top_driver}"
        s2 += "."
        lines.append(s2)

        # Sentence 3: driver distribution (weighted by mention count)
        driver_counts: dict[str, int] = {}
        for e in disp:
            d = e.get("primary_driver", "other")
            if d.lower() != "other":
                driver_counts[d] = driver_counts.get(d, 0) + e.get("mention_count", 0)
        if driver_counts and total_mentions:
            ranked_drivers = sorted(driver_counts.items(), key=lambda x: -x[1])
            top_d = ranked_drivers[0]
            pct = round(top_d[1] * 100 / total_mentions)
            parts = [f"{top_d[0]} drives {pct}% of all displacement flow"]
            for d_name, d_count in ranked_drivers[1:3]:
                parts.append(f"{d_name} ({round(d_count * 100 / total_mentions)}%)")
            lines.append(", followed by ".join(parts) + "." if len(parts) > 1 else parts[0] + ".")

        # Sentence 4: signal strength breakdown
        strength_counts: dict[str, int] = {"strong": 0, "moderate": 0, "emerging": 0}
        for e in disp:
            ss = e.get("signal_strength", "emerging")
            if ss in strength_counts:
                strength_counts[ss] += 1
        strength_parts = []
        for level, label in [("strong", "strong signal (5+ mentions)"), ("moderate", "moderate"), ("emerging", "emerging")]:
            if strength_counts[level]:
                strength_parts.append(f"{strength_counts[level]} {label}")
        if strength_parts:
            lines.append(", ".join(strength_parts) + ".")

        return " ".join(lines)

    if report_type == "category_overview":
        cats = parsed.get("category_insights", [])
        if not cats:
            return ""
        n = len(cats)

        # Count total unique vendors across all category rankings
        all_cat_vendors: set[str] = set()
        for cat in cats:
            for vr in cat.get("vendor_rankings", []):
                v = vr.get("vendor")
                if v:
                    all_cat_vendors.add(v)

        # Sentence 1: scale
        lines = [
            f"{window_label}, Atlas analyzed churn trends across {n} product categories"
            + (f" covering {len(all_cat_vendors)} vendors from {total_review_count:,} reviews."
               if total_review_count and all_cat_vendors
               else f" covering {len(all_cat_vendors)} vendors." if all_cat_vendors
               else ".")
        ]

        # Sentence 2: hottest category (highest churn density)
        hottest = None
        hottest_density = 0.0
        for cat in cats:
            rankings = cat.get("vendor_rankings", [])
            if rankings:
                cd = rankings[0].get("churn_signal_density", 0)
                if cd > hottest_density:
                    hottest_density = cd
                    hottest = cat
        if hottest:
            cat_name = hottest.get("category", "")
            risk_vendor = hottest.get("highest_churn_risk", "")
            pain = hottest.get("dominant_pain", "")
            s2 = (
                f"{cat_name} shows the highest pressure \u2014 {risk_vendor}"
                f" at {hottest_density}% churn density"
                " (share of reviews with explicit switching signals)"
            )
            if pain and pain.lower() not in ("other", "unknown"):
                s2 += f", driven by {pain}"
            s2 += "."
            lines.append(s2)

        # Sentence 3: market movement — categories with clear challengers
        movements = []
        for cat in cats:
            challenger = cat.get("emerging_challenger", "")
            incumbent = cat.get("highest_churn_risk", "")
            cat_name = cat.get("category", "")
            if challenger and incumbent and "Insufficient" not in challenger:
                # Skip the hottest category (already covered in sentence 2)
                if hottest and cat_name == hottest.get("category"):
                    continue
                movements.append(f"in {cat_name}, {challenger} is emerging as the primary alternative to {incumbent}")
        if movements:
            lines.append("; ".join(movements[:2]).capitalize() + ".")

        # Sentence 4: cross-category pain pattern
        pain_counts: dict[str, int] = {}
        for cat in cats:
            p = cat.get("dominant_pain", "")
            if p and p.lower() not in ("other", "unknown"):
                pain_counts[p] = pain_counts.get(p, 0) + 1
        if pain_counts:
            top_pain_name, top_pain_ct = max(pain_counts.items(), key=lambda x: x[1])
            if top_pain_ct >= 2:
                lines.append(f"{top_pain_name.capitalize()} is the dominant churn driver in {top_pain_ct} of {n} categories.")

        return " ".join(lines)

    # --- default: weekly_churn_feed -----------------------------------

    top_entries = feed[:3]
    top_vendors: list[str] = []
    top_pains: list[str] = []
    top_alternatives: list[str] = []
    quote = None
    churn_density_defined = False

    for entry in top_entries:
        vendor = entry.get("vendor")
        churn_density = entry.get("churn_signal_density") or entry.get("churn_density")
        total_reviews = entry.get("total_reviews")
        if vendor:
            parts = [str(vendor)]
            if churn_density is not None:
                if not churn_density_defined:
                    parts[0] += f" ({churn_density}% churn density -- share of reviews containing explicit switching signals"
                    churn_density_defined = True
                else:
                    parts[0] += f" ({churn_density}%"
                if total_reviews:
                    parts[0] += f", {total_reviews} reviews)"
                else:
                    parts[0] += ")"
            top_vendors.append(parts[0])
        # Extract pains from pain_breakdown or top_pain -- skip vague "other"
        pain_breakdown = entry.get("pain_breakdown", [])
        if pain_breakdown:
            for pb in pain_breakdown[:2]:
                p = pb.get("category", "")
                if p and p.lower() != "other" and p not in top_pains:
                    top_pains.append(str(p))
        elif entry.get("top_pain"):
            p = str(entry["top_pain"])
            if p.lower() != "other" and p not in top_pains:
                top_pains.append(p)
        # Extract alternatives from displacement targets
        for dt in entry.get("top_displacement_targets", []) or []:
            comp = dt.get("competitor", "")
            if comp and comp not in top_alternatives:
                top_alternatives.append(comp)
        if not quote and entry.get("key_quote"):
            quote = str(entry["key_quote"])

    lines = [
        (
            f"{window_label}, Atlas identified {len(feed)} vendors under elevated churn pressure "
            f"from {total_review_count} reviews across {source_label_text}."
            if total_review_count
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

    lines.append(
        "Confidence is highest for vendors with 50+ reviews; smaller samples should be treated as directional."
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


def _build_buyer_action(vendor: str, pain: str | None, alternatives: list[str], archetype: str | None = None) -> str:
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
    if archetype == "acquisition_decay":
        return f"Teams on {vendor} should assess post-acquisition roadmap stability against {alt_text} before committing to another term."
    if archetype == "support_collapse":
        return f"Teams on {vendor} should escalate support SLA concerns immediately and benchmark response times against {alt_text}."
    if archetype == "integration_break":
        return f"Teams on {vendor} should audit API/integration stability and evaluate migration paths to {alt_text} before renewal."
    return f"Teams on {vendor} should compare current fit against {alt_text} and validate switching costs before the next renewal decision."


# Default scoring weights for churn pressure score.
_DEFAULT_WEIGHTS = {
    "churn_density": 0.30,
    "urgency": 0.25,
    "dm_churn_rate": 0.20,
    "displacement": 0.15,
    "price_complaints": 0.10,
}

# Archetype-specific weight overrides.  When the stratified reasoner
# classifies a vendor into an archetype, the corresponding weight set
# is used instead of the default, biasing the score toward the signal
# that matters most for that churn pattern.
_ARCHETYPE_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
    "pricing_shock":        {"churn_density": 0.20, "urgency": 0.20, "dm_churn_rate": 0.15, "displacement": 0.10, "price_complaints": 0.35},
    "feature_gap":          {"churn_density": 0.25, "urgency": 0.20, "dm_churn_rate": 0.15, "displacement": 0.30, "price_complaints": 0.10},
    "support_collapse":     {"churn_density": 0.30, "urgency": 0.35, "dm_churn_rate": 0.15, "displacement": 0.10, "price_complaints": 0.10},
    "acquisition_decay":    {"churn_density": 0.35, "urgency": 0.25, "dm_churn_rate": 0.20, "displacement": 0.15, "price_complaints": 0.05},
    "category_disruption":  {"churn_density": 0.20, "urgency": 0.15, "dm_churn_rate": 0.15, "displacement": 0.40, "price_complaints": 0.10},
    "integration_break":    {"churn_density": 0.30, "urgency": 0.30, "dm_churn_rate": 0.20, "displacement": 0.15, "price_complaints": 0.05},
    "leadership_redesign":  {"churn_density": 0.30, "urgency": 0.25, "dm_churn_rate": 0.15, "displacement": 0.20, "price_complaints": 0.10},
    "compliance_gap":       {"churn_density": 0.25, "urgency": 0.30, "dm_churn_rate": 0.25, "displacement": 0.10, "price_complaints": 0.10},
}


def _compute_churn_pressure_score(
    *,
    churn_density: float,
    avg_urgency: float,
    dm_churn_rate: float,
    displacement_mention_count: int,
    price_complaint_rate: float,
    total_reviews: int,
    archetype: str | None = None,
) -> float:
    """Composite 0-100 score for ranking vendors by churn pressure.

    Default weights: churn density 30%, urgency 25%, DM churn rate 20%,
    displacement mentions 15%, price complaints 10%.

    When *archetype* is provided (from the stratified reasoner), the weights
    shift to emphasise the signal most relevant to that churn pattern --
    e.g. ``pricing_shock`` boosts price_complaints to 35%.

    Confidence multiplier: 1.0 (50+), 0.85 (20-49), 0.65 (<20).
    """
    w = _ARCHETYPE_WEIGHT_OVERRIDES.get(archetype or "", _DEFAULT_WEIGHTS)
    raw = (
        min(churn_density, 100.0) * w["churn_density"]
        + min(avg_urgency, 10.0) * 10.0 * w["urgency"]
        + min(dm_churn_rate, 1.0) * 100.0 * w["dm_churn_rate"]
        + min(displacement_mention_count, 50) * 2.0 * w["displacement"]
        + min(price_complaint_rate, 1.0) * 100.0 * w["price_complaints"]
    )
    if total_reviews >= 50:
        confidence = 1.0
    elif total_reviews >= 20:
        confidence = 0.85
    else:
        confidence = 0.65
    return round(min(raw * confidence, 100.0), 1)


def _build_vendor_evidence(
    vs: dict[str, Any],
    *,
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    insider_lookup: dict[str, dict],
    keyword_spike_lookup: dict[str, dict],
    temporal_lookup: dict[str, dict] | None = None,
    archetype_lookup: dict[str, list[dict]] | None = None,
    dm_lookup: dict[str, float] | None = None,
    price_lookup: dict[str, float] | None = None,
    quote_lookup: dict[str, list] | None = None,
    budget_lookup: dict[str, dict] | None = None,
    buyer_auth_lookup: dict[str, dict] | None = None,
    use_case_lookup: dict[str, list[dict]] | None = None,
) -> dict[str, Any]:
    """Merge vendor score row + all available lookups into a single evidence
    dict for the stratified reasoner.

    Includes pain, competitors, feature gaps, insider signals, keyword spikes,
    temporal velocities, archetype pre-scores, and optionally DM rate, price
    complaint rate, budget signals, buyer authority, use cases, and quotes.
    """
    vendor = _canonicalize_vendor(vs.get("vendor_name") or "")
    total = int(vs.get("total_reviews") or 0)
    churn = int(vs.get("churn_intent") or 0)
    churn_density = round((churn * 100.0 / total), 1) if total else 0.0
    avg_urgency = round(float(vs.get("avg_urgency") or 0), 1)

    evidence: dict[str, Any] = {
        "vendor_name": vendor,
        "product_category": vs.get("product_category") or "",
        "total_reviews": total,
        "churn_intent": churn,
        "churn_density": churn_density,
        "avg_urgency": avg_urgency,
        "recommend_yes": int(vs.get("recommend_yes") or 0),
        "recommend_no": int(vs.get("recommend_no") or 0),
    }

    pains = pain_lookup.get(vendor, [])
    if pains:
        evidence["pain_categories"] = [
            {"category": p.get("category", ""), "count": p.get("count", 0)}
            for p in pains[:5]
        ]

    comps = competitor_lookup.get(vendor, [])
    if comps:
        evidence["competitors"] = [
            {"name": c.get("name", ""), "mentions": c.get("mentions", 0)}
            for c in comps[:5]
        ]
        evidence["displacement_mention_count"] = sum(c.get("mentions", 0) for c in comps)

    gaps = feature_gap_lookup.get(vendor, [])
    if gaps:
        evidence["feature_gaps"] = [
            {"feature": g.get("feature", ""), "mentions": g.get("count", g.get("mentions", 0))}
            for g in gaps[:5]
        ]

    insider = insider_lookup.get(vendor, {})
    if insider:
        evidence["insider_signal_count"] = insider.get("signal_count", 0)
        evidence["insider_talent_drain_rate"] = insider.get("talent_drain_rate")

    kw = keyword_spike_lookup.get(vendor, {})
    if kw.get("spike_count"):
        evidence["keyword_spike_count"] = kw["spike_count"]
        evidence["keyword_spike_keywords"] = kw.get("spike_keywords", [])

    if temporal_lookup:
        td = temporal_lookup.get(vendor, {})
        if td:
            evidence.update(td)

    if archetype_lookup:
        arch = archetype_lookup.get(vendor, [])
        if arch:
            evidence["archetype_scores"] = arch

    # Additional context (when available)
    if dm_lookup:
        dm = dm_lookup.get(vendor)
        if dm is not None:
            evidence["dm_churn_rate"] = dm
    if price_lookup:
        pr = price_lookup.get(vendor)
        if pr is not None:
            evidence["price_complaint_rate"] = pr
    if quote_lookup:
        quotes = quote_lookup.get(vendor, [])
        if quotes:
            evidence["quote_count"] = len(quotes)
            evidence["top_quote"] = (quotes[0].get("quote", str(quotes[0])) if isinstance(quotes[0], dict) else str(quotes[0]))[:200] if quotes else None
    if budget_lookup:
        budget = budget_lookup.get(vendor, {})
        if budget:
            evidence["budget_context"] = budget
    if buyer_auth_lookup:
        ba = buyer_auth_lookup.get(vendor, {})
        if ba:
            evidence["buyer_authority"] = ba
    if use_case_lookup:
        ucs = use_case_lookup.get(vendor, [])
        if ucs:
            evidence["top_use_cases"] = [u.get("use_case", u.get("name", "")) for u in ucs[:3] if isinstance(u, dict)]

    return evidence


def _classify_trend(
    vendor: str,
    churn_density: float,
    avg_urgency: float,
    prior_metrics: dict[str, float] | None,
    temporal_lookup: dict[str, dict] | None = None,
) -> str:
    """Classify vendor trend using temporal z-scores when available, falling
    back to hardcoded delta thresholds.

    Returns one of: ``"new"``, ``"worsening"``, ``"improving"``, ``"stable"``.
    """
    if not prior_metrics:
        return "new"

    # Prefer temporal z-score anomalies when available
    td = (temporal_lookup or {}).get(vendor, {})
    anomalies = td.get("anomalies", [])
    if anomalies:
        anomalies_by_metric: dict[str, dict] = {}
        for a in anomalies:
            if isinstance(a, dict):
                anomalies_by_metric[a.get("metric", "")] = a
        cd_a = anomalies_by_metric.get("churn_density", {})
        urg_a = anomalies_by_metric.get("avg_urgency", {})
        if (cd_a.get("is_anomaly") and cd_a.get("z_score", 0) > 0) or \
           (urg_a.get("is_anomaly") and urg_a.get("z_score", 0) > 0):
            return "worsening"
        if (cd_a.get("is_anomaly") and cd_a.get("z_score", 0) < 0) or \
           (urg_a.get("is_anomaly") and urg_a.get("z_score", 0) < 0):
            return "improving"

    # Fallback: hardcoded delta thresholds
    delta_density = churn_density - prior_metrics.get("churn_signal_density", 0)
    delta_urgency = avg_urgency - prior_metrics.get("avg_urgency", 0)
    if delta_density >= 5 or delta_urgency >= 1:
        return "worsening"
    if delta_density <= -5 or delta_urgency <= -1:
        return "improving"
    return "stable"


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
    quote_lookup: dict[str, list],
    budget_lookup: dict[str, dict],
    sentiment_lookup: dict[str, dict[str, int]],
    buyer_auth_lookup: dict[str, dict],
    dm_lookup: dict[str, float],
    price_lookup: dict[str, float],
    company_lookup: dict[str, list],
    keyword_spike_lookup: dict[str, dict],
    prior_reports: list[dict[str, Any]],
    reasoning_lookup: dict[str, dict] | None = None,
    temporal_lookup: dict[str, dict] | None = None,
    limit: int = 50,
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
        # Boost confidence when reasoning provides corroborating evidence
        _rc = (reasoning_lookup or {}).get(vendor, {})
        if _rc.get("confidence", 0) >= 0.8 and confidence == "medium":
            confidence = "high"

        # Displacement mention total for this vendor
        comp_entries = competitor_lookup.get(vendor, [])
        displacement_mentions = sum(c.get("mentions", 0) for c in comp_entries)

        _vendor_archetype = (reasoning_lookup or {}).get(vendor, {}).get("archetype")
        score = _compute_churn_pressure_score(
            churn_density=churn_density,
            avg_urgency=avg_urgency,
            dm_churn_rate=dm_rate,
            displacement_mention_count=displacement_mentions,
            price_complaint_rate=price_rate,
            total_reviews=total_reviews,
            archetype=_vendor_archetype,
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

        # Quotes (items may be dicts with review_id or plain strings)
        quotes = quote_lookup.get(vendor, [])
        key_quote = _quote_text(quotes[0]) if quotes else None
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

        # Trend from prior reports (z-score aware when temporal data available)
        trend = _classify_trend(
            vendor, churn_density, avg_urgency,
            prior_vendor_metrics.get(vendor),
            temporal_lookup=temporal_lookup,
        )

        # Named accounts (may be empty)
        companies = company_lookup.get(vendor, [])
        named_accounts = [
            {"company": c.get("company", c) if isinstance(c, dict) else str(c),
             "urgency": c.get("urgency", 0) if isinstance(c, dict) else 0,
             "title": c.get("title") if isinstance(c, dict) else None,
             "company_size": c.get("company_size") if isinstance(c, dict) else None,
             "industry": c.get("industry") if isinstance(c, dict) else None}
            for c in companies[:5]
        ]

        # Risk level -- prefer reasoning conclusion, fall back to deterministic
        _rc_risk = (reasoning_lookup or {}).get(vendor, {}).get("risk_level", "")
        if _rc_risk:
            risk_level = _rc_risk
        elif score >= 70:
            risk_level = "high"
        elif score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Affected segments (aggregate industry/size from churning companies)
        _industry_counts: dict[str, int] = {}
        _size_counts: dict[str, int] = {}
        for c in companies:
            if not isinstance(c, dict):
                continue
            ind = c.get("industry")
            if ind and ind != "unknown":
                _industry_counts[ind] = _industry_counts.get(ind, 0) + 1
            sz = c.get("company_size")
            if sz:
                _size_counts[sz] = _size_counts.get(sz, 0) + 1
        affected_segments = {
            "industries": sorted(
                [{"industry": k, "count": v} for k, v in _industry_counts.items()],
                key=lambda x: -x["count"],
            )[:5],
            "company_sizes": sorted(
                [{"size": k, "count": v} for k, v in _size_counts.items()],
                key=lambda x: -x["count"],
            )[:5],
        }

        # Alternatives for action recommendation
        alt_names = [c["name"] for c in comp_entries[:2]] if comp_entries else []

        entry = {
            "vendor": vendor,
            "category": category,
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "avg_urgency": avg_urgency,
            "sample_size_confidence": confidence,
            "churn_pressure_score": score,
            "risk_level": risk_level,
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
            "action_recommendation": _build_buyer_action(vendor, top_pain, alt_names, archetype=_vendor_archetype),
            "named_accounts": named_accounts,
            "affected_segments": affected_segments,
        }
        rc = (reasoning_lookup or {}).get(vendor, {})
        if rc:
            entry["archetype"] = rc.get("archetype", "")
            entry["archetype_confidence"] = rc.get("confidence", 0)
            entry["archetype_risk_level"] = rc.get("risk_level", "")
            entry["reasoning_mode"] = rc.get("mode", "")
        candidates.append(entry)

    # Reasoning-weighted sort: vendors with high-confidence archetypes get a boost
    def _sort_key(x: dict) -> tuple:
        score = x["churn_pressure_score"]
        rc = (reasoning_lookup or {}).get(x.get("vendor", ""), {})
        # High-confidence reasoning adds up to 5 points to sort priority
        reasoning_boost = min(rc.get("confidence", 0) * 5, 5.0) if rc.get("archetype") else 0
        return (-(score + reasoning_boost),)
    candidates.sort(key=_sort_key)
    return candidates[:limit]


def _build_reason_lookup(competitor_reasons: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    """Map (vendor, competitor) to ordered reason strings.

    Prefers structured ``reason_category`` when available; falls back to
    free-text ``reason`` for legacy data.
    """
    lookup: dict[tuple[str, str], list[str]] = {}
    for row in competitor_reasons:
        vendor = _canonicalize_vendor(row.get("vendor", ""))
        competitor = _canonicalize_competitor(row.get("competitor", ""))
        reason = row.get("reason_category") or row.get("reason") or ""
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


def _compute_evidence_confidence(
    mention_count: int,
    source_distribution: dict[str, int],
) -> float:
    """Evidence-based confidence score for provenance-tracked entities.

    Three equally-weighted signals (each 0-1, averaged):
      - mention_weight:  log-scaled mention count (caps at 20)
      - source_weight:   number of distinct sources (3+ = 1.0)
      - quality_weight:  proportion from VERIFIED_SOURCES
    """
    _VERIFIED = {s.value for s in VERIFIED_SOURCES}

    # Mention weight: log2(count)/log2(20), capped at 1.0
    mention_weight = min(math.log2(max(mention_count, 1)) / math.log2(20), 1.0)

    # Source diversity: n_sources / 3, capped at 1.0
    n_sources = len(source_distribution)
    source_weight = min(n_sources / 3.0, 1.0)

    # Quality weight: fraction of mentions from verified sources
    total = sum(source_distribution.values()) or 1
    verified_total = sum(
        cnt for src, cnt in source_distribution.items() if src in _VERIFIED
    )
    quality_weight = verified_total / total

    score = (mention_weight + source_weight + quality_weight) / 3.0
    return round(score, 2)


def _pick_displacement_quote(
    *,
    vendor: str,
    competitor: str,
    reasons: list[str],
    quote_lookup: dict[str, list],
) -> str | None:
    """Choose a quote matching the competitor or reason text when possible."""
    quotes = quote_lookup.get(vendor, [])
    competitor_l = competitor.lower()
    for q in quotes:
        text = _quote_text(q) or ""
        if competitor_l in text.lower():
            return text
    for reason in reasons:
        for token in reason.lower().split():
            if len(token) >= 5:
                for q in quotes:
                    text = _quote_text(q) or ""
                    if token in text.lower():
                        return text
    return _quote_text(quotes[0]) if quotes else None


def _build_deterministic_displacement_map(
    competitive_disp: list[dict[str, Any]],
    competitor_reasons: list[dict[str, Any]],
    quote_lookup: dict[str, list],
    *,
    reasoning_lookup: dict[str, dict] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Build displacement report from evidence-quality-filtered aggregated flows.

    Quality gate: only edges with at least one explicit_switch or active_evaluation
    survive.  Pure implied_preference edges are market-pain data, not displacement.
    """
    reason_lookup = _build_reason_lookup(competitor_reasons)
    _rl = reasoning_lookup or {}
    results: list[dict[str, Any]] = []
    for row in competitive_disp:
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        competitor = _canonicalize_competitor(row.get("competitor") or "")
        if not vendor or not competitor or vendor.lower() == competitor.lower():
            continue

        mention_count = int(row.get("mention_count") or 0)
        explicit = int(row.get("explicit_switches") or 0)
        active_eval = int(row.get("active_evaluations") or 0)
        implied = int(row.get("implied_preferences") or 0)

        # Quality gate: require at least one explicit switch or active evaluation
        if explicit == 0 and active_eval == 0:
            continue

        # Primary driver: use structured reason_categories; fall back to keyword inference
        reason_cats = row.get("reason_categories") or {}
        if reason_cats:
            driver = max(reason_cats.items(), key=lambda x: x[1])[0]
        else:
            reasons = reason_lookup.get((vendor, competitor), [])
            driver = _infer_driver_from_reasons(reasons)

        # Signal strength: weight evidence types (explicit 3x, active_eval 2x, implied 1x)
        weighted_signal = explicit * 3 + active_eval * 2 + implied * 1
        if weighted_signal >= 15 or explicit >= 3:
            strength = "strong"
        elif weighted_signal >= 6 or explicit >= 1:
            strength = "moderate"
        else:
            strength = "emerging"
        # Boost signal strength when source vendor has category_disruption or feature_gap archetype
        src_arch = _rl.get(vendor, {}).get("archetype", "")
        if src_arch in ("category_disruption", "feature_gap") and strength == "emerging":
            strength = "moderate"

        reasons_for_quote = reason_lookup.get((vendor, competitor), [])
        edge_entry: dict[str, Any] = {
            "from_vendor": vendor,
            "to_vendor": competitor,
            "mention_count": mention_count,
            "primary_driver": driver,
            "signal_strength": strength,
            "evidence_breakdown": {
                "explicit_switches": explicit,
                "active_evaluations": active_eval,
                "implied_preferences": implied,
            },
            "key_quote": _pick_displacement_quote(
                vendor=vendor,
                competitor=competitor,
                reasons=reasons_for_quote,
                quote_lookup=quote_lookup,
            ),
            "industries": row.get("industries", []),
            "company_sizes": row.get("company_sizes", []),
        }
        _src_rc = _rl.get(vendor, {})
        _tgt_rc = _rl.get(competitor, {})
        if _src_rc.get("archetype"):
            edge_entry["source_archetype"] = _src_rc["archetype"]
            edge_entry["source_archetype_confidence"] = _src_rc.get("confidence", 0)
        if _tgt_rc.get("archetype"):
            edge_entry["target_archetype"] = _tgt_rc["archetype"]
            edge_entry["target_archetype_confidence"] = _tgt_rc.get("confidence", 0)
        results.append(edge_entry)
    results.sort(key=lambda x: x["mention_count"], reverse=True)
    return results[:limit]


def _build_deterministic_vendor_scorecards(
    vendor_scores: list[dict[str, Any]],
    *,
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list],
    budget_lookup: dict[str, dict],
    sentiment_lookup: dict[str, dict[str, int]],
    buyer_auth_lookup: dict[str, dict],
    dm_lookup: dict[str, float],
    price_lookup: dict[str, float],
    company_lookup: dict[str, list[dict]],
    product_profile_lookup: dict[str, dict],
    prior_reports: list[dict[str, Any]],
    reasoning_lookup: dict[str, dict] | None = None,
    temporal_lookup: dict[str, dict] | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Build vendor deep-dive scorecards from aggregated numeric data.

    Merges multi-category rows into one entry per vendor, then enriches
    with feature analysis, churn predictors, competitor overlap, and
    customer profile data.
    """
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

    # ── Merge multi-category rows into one per vendor ──────────────
    merged: dict[str, dict[str, Any]] = {}
    for row in vendor_scores:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        reviews = int(row.get("total_reviews") or 0)
        churn = int(row.get("churn_intent") or 0)
        urgency = float(row.get("avg_urgency") or 0)
        pos_pct = row.get("positive_review_pct")
        rec_yes = int(row.get("recommend_yes") or 0)
        rec_no = int(row.get("recommend_no") or 0)
        category = row.get("product_category") or "Unknown"
        if vendor not in merged:
            merged[vendor] = {
                "total_reviews": reviews,
                "churn_intent": churn,
                "urgency_weighted_sum": urgency * reviews,
                "recommend_yes": rec_yes,
                "recommend_no": rec_no,
                "positive_pct_sum": (float(pos_pct) * reviews) if pos_pct is not None else 0,
                "positive_pct_count": reviews if pos_pct is not None else 0,
                "category": category,
                "category_reviews": reviews,
            }
        else:
            m = merged[vendor]
            m["total_reviews"] += reviews
            m["churn_intent"] += churn
            m["urgency_weighted_sum"] += urgency * reviews
            m["recommend_yes"] += rec_yes
            m["recommend_no"] += rec_no
            if pos_pct is not None:
                m["positive_pct_sum"] += float(pos_pct) * reviews
                m["positive_pct_count"] += reviews
            if reviews > m["category_reviews"]:
                m["category"] = category
                m["category_reviews"] = reviews

    # ── Build enriched scorecard per vendor ─────────────────────────
    results: list[dict[str, Any]] = []
    for vendor, m in merged.items():
        total_reviews = m["total_reviews"]
        churn_intent = m["churn_intent"]
        churn_density = round((churn_intent * 100.0 / total_reviews), 1) if total_reviews else 0.0
        avg_urgency = round(m["urgency_weighted_sum"] / total_reviews, 1) if total_reviews else 0.0
        positive_pct = round(m["positive_pct_sum"] / m["positive_pct_count"], 1) if m["positive_pct_count"] else None
        recommend_yes = m["recommend_yes"]
        recommend_no = m["recommend_no"]
        recommend_ratio = round(((recommend_yes - recommend_no) / total_reviews) * 100, 1) if total_reviews else 0.0

        if total_reviews >= 50:
            confidence = "high"
        elif total_reviews >= 20:
            confidence = "medium"
        else:
            confidence = "low"
        # Boost confidence when reasoning provides corroborating evidence
        _rc = (reasoning_lookup or {}).get(vendor, {})
        if _rc.get("confidence", 0) >= 0.8 and confidence == "medium":
            confidence = "high"

        # Trend (z-score aware when temporal data available)
        trend = _classify_trend(
            vendor, churn_density, avg_urgency,
            prior_vendor_metrics.get(vendor),
            temporal_lookup=temporal_lookup,
        )

        sentiment_counts = sentiment_lookup.get(vendor, {})
        if total_reviews < 10 or not sentiment_counts:
            sentiment_direction = "insufficient_history"
        else:
            sentiment_direction = max(sentiment_counts.items(), key=lambda item: item[1])[0]

        # Pain breakdown (top 5)
        pains = pain_lookup.get(vendor, [])
        top_pain = pains[0]["category"] if pains else "unknown"
        pain_breakdown = [{"category": p["category"], "count": p["count"]} for p in pains[:5]]

        # Competitor overlap (top 5)
        comp_entries = competitor_lookup.get(vendor, [])
        top_competitor = comp_entries[:1]
        if top_competitor:
            comp = top_competitor[0]
            top_competitor_text = f"{comp['name']} ({comp['mentions']} mentions)"
        else:
            top_competitor_text = "Insufficient displacement data"
        competitor_overlap = [
            {"competitor": c["name"], "mentions": c["mentions"]}
            for c in comp_entries[:5]
        ]

        # Churn pressure score
        dm_rate = float(dm_lookup.get(vendor, 0))
        price_rate = float(price_lookup.get(vendor, 0))
        displacement_mentions = sum(c.get("mentions", 0) for c in comp_entries)
        _vendor_archetype = (reasoning_lookup or {}).get(vendor, {}).get("archetype")
        churn_pressure_score = _compute_churn_pressure_score(
            churn_density=churn_density,
            avg_urgency=avg_urgency,
            dm_churn_rate=dm_rate,
            displacement_mention_count=displacement_mentions,
            price_complaint_rate=price_rate,
            total_reviews=total_reviews,
            archetype=_vendor_archetype,
        )

        # Risk level
        if churn_pressure_score >= 70:
            risk_level = "high"
        elif churn_pressure_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Feature analysis (loved from product profile, hated from feature gaps)
        profile = product_profile_lookup.get(vendor, {})
        strengths_raw = profile.get("strengths") or []
        if isinstance(strengths_raw, list):
            loved = [
                {"feature": s.get("area", s) if isinstance(s, dict) else str(s),
                 "score": s.get("score") if isinstance(s, dict) else None,
                 "source": "product_profile"}
                for s in strengths_raw[:5]
            ]
        else:
            loved = []
        gaps = feature_gap_lookup.get(vendor, [])
        hated = [
            {"feature": g["feature"],
             "mentions": g.get("count", g.get("mentions", 0)),
             "source": "reviews"}
            for g in gaps[:5]
        ]
        feature_analysis = {"loved": loved, "hated": hated}

        # Churn predictors (high-urgency segment correlations)
        companies = company_lookup.get(vendor, [])
        high_urg_industries: dict[str, int] = {}
        high_urg_sizes: dict[str, int] = {}
        for c in companies:
            if not isinstance(c, dict):
                continue
            urg = float(c.get("urgency", 0))
            if urg >= 7:
                ind = c.get("industry")
                if ind and ind != "unknown":
                    high_urg_industries[ind] = high_urg_industries.get(ind, 0) + 1
                sz = c.get("company_size")
                if sz:
                    high_urg_sizes[sz] = high_urg_sizes.get(sz, 0) + 1
        churn_predictors = {
            "high_risk_industries": sorted(
                [{"industry": k, "count": v} for k, v in high_urg_industries.items()],
                key=lambda x: -x["count"],
            )[:3],
            "high_risk_sizes": sorted(
                [{"size": k, "count": v} for k, v in high_urg_sizes.items()],
                key=lambda x: -x["count"],
            )[:3],
            "dm_churn_rate": round(dm_rate, 2),
            "price_complaint_rate": round(price_rate, 2),
        }

        # Evidence (top quotes)
        quotes = quote_lookup.get(vendor, [])
        evidence = quotes[:5]

        # Named accounts
        named_accounts = [
            {"company": c.get("company", c) if isinstance(c, dict) else str(c),
             "urgency": c.get("urgency", 0) if isinstance(c, dict) else 0,
             "title": c.get("title") if isinstance(c, dict) else None,
             "company_size": c.get("company_size") if isinstance(c, dict) else None,
             "industry": c.get("industry") if isinstance(c, dict) else None}
            for c in companies[:5]
        ]

        # Industry and company size distributions
        industry_counts: dict[str, int] = {}
        size_counts: dict[str, int] = {}
        for c in companies:
            ind = c.get("industry") if isinstance(c, dict) else None
            if ind and ind != "unknown":
                industry_counts[ind] = industry_counts.get(ind, 0) + 1
            sz = c.get("company_size") if isinstance(c, dict) else None
            if sz:
                size_counts[sz] = size_counts.get(sz, 0) + 1
        industry_dist = sorted(
            [{"industry": k, "count": v} for k, v in industry_counts.items()],
            key=lambda x: -x["count"],
        )[:5]
        size_dist = sorted(
            [{"size": k, "count": v} for k, v in size_counts.items()],
            key=lambda x: -x["count"],
        )[:5]

        # Customer profile (from product profile + company data)
        customer_profile = {
            "typical_industries": profile.get("typical_industries") or [],
            "typical_company_size": profile.get("typical_company_size") or [],
            "primary_use_cases": profile.get("primary_use_cases") or [],
            "top_integrations": profile.get("top_integrations") or [],
            "industry_distribution": industry_dist,
            "company_size_distribution": size_dist,
        }

        # Buyer authority
        ba = buyer_auth_lookup.get(vendor, {})
        role_types = ba.get("role_types", {})
        dominant_role = max(role_types.items(), key=lambda x: x[1])[0] if role_types else "unknown"

        sc_entry = {
            "vendor": vendor,
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "positive_review_pct": float(positive_pct) if positive_pct is not None else None,
            "avg_urgency": avg_urgency,
            "recommend_ratio": recommend_ratio,
            "sample_size_confidence": confidence,
            "churn_pressure_score": churn_pressure_score,
            "risk_level": risk_level,
            "top_pain": top_pain,
            "pain_breakdown": pain_breakdown,
            "top_competitor_threat": top_competitor_text,
            "competitor_overlap": competitor_overlap,
            "trend": trend,
            "budget_context": budget_lookup.get(vendor, {}),
            "sentiment_direction": sentiment_direction,
            "dm_churn_rate": round(dm_rate, 2),
            "price_complaint_rate": round(price_rate, 2),
            "feature_analysis": feature_analysis,
            "churn_predictors": churn_predictors,
            "evidence": evidence,
            "named_accounts": named_accounts,
            "customer_profile": customer_profile,
            "dominant_buyer_role": dominant_role,
            "industry_distribution": industry_dist,
            "company_size_distribution": size_dist,
        }
        rc = (reasoning_lookup or {}).get(vendor, {})
        if rc:
            sc_entry["archetype"] = rc.get("archetype", "")
            sc_entry["archetype_confidence"] = rc.get("confidence", 0)
            sc_entry["reasoning_summary"] = rc.get("executive_summary", "")
            sc_entry["falsification_conditions"] = rc.get("falsification_conditions", [])
            sc_entry["uncertainty_sources"] = rc.get("uncertainty_sources", [])
            sc_entry["reasoning_mode"] = rc.get("mode", "")
            rc_risk = rc.get("risk_level", "")
            if rc_risk:
                sc_entry["risk_level"] = rc_risk
        results.append(sc_entry)
    def _sc_sort_key(x: dict) -> tuple:
        rc = (reasoning_lookup or {}).get(x.get("vendor", ""), {})
        reasoning_boost = min(rc.get("confidence", 0) * 5, 5.0) if rc.get("archetype") else 0
        return (-(x["avg_urgency"] + reasoning_boost), -(x["churn_signal_density"]), x["vendor"])
    results.sort(key=_sc_sort_key)
    return results[:limit]


def _build_market_shift_signal(
    category: str,
    highest_vendor: str,
    churn_density: float,
    total_reviews: int,
    emerging: str,
    reasoning_lookup: dict[str, dict],
) -> str:
    """Build market shift signal with archetype context when available."""
    base = (
        f"Based on {total_reviews} reviews, {highest_vendor} shows "
        f"{churn_density}% churn-signal density in {category}."
    )
    arch = reasoning_lookup.get(highest_vendor, {}).get("archetype", "")
    if arch:
        base += f" Classified as {arch} pattern."
    challenger_part = (
        f" {emerging} is the most visible challenger flow in this category."
        if emerging and "Insufficient" not in emerging
        else ""
    )
    return base + challenger_part


def _build_deterministic_category_overview(
    vendor_scores: list[dict[str, Any]],
    *,
    pain_lookup: dict[str, list[dict]],
    competitive_disp: list[dict[str, Any]],
    company_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list],
    feature_gap_lookup: dict[str, list[dict]],
    dm_lookup: dict[str, float],
    price_lookup: dict[str, float],
    competitor_lookup: dict[str, list[dict]],
    reasoning_lookup: dict[str, dict] | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Build industry-specific category overview with vendor rankings and case studies."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in vendor_scores:
        category = row.get("product_category") or "Unknown"
        by_category.setdefault(category, []).append(row)

    results: list[dict[str, Any]] = []
    _rl = reasoning_lookup or {}
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

        # Aggregate industry/size across all vendors in category
        cat_industries: dict[str, int] = {}
        cat_sizes: dict[str, int] = {}
        for r in rows:
            v = _canonicalize_vendor(r.get("vendor_name") or "")
            for c in company_lookup.get(v, []):
                if not isinstance(c, dict):
                    continue
                ind = c.get("industry")
                if ind and ind != "unknown":
                    cat_industries[ind] = cat_industries.get(ind, 0) + 1
                sz = c.get("company_size")
                if sz:
                    cat_sizes[sz] = cat_sizes.get(sz, 0) + 1
        top_industries = sorted(
            [{"industry": k, "count": v} for k, v in cat_industries.items()],
            key=lambda x: -x["count"],
        )[:5]
        top_sizes = sorted(
            [{"size": k, "count": v} for k, v in cat_sizes.items()],
            key=lambda x: -x["count"],
        )[:5]

        # ── Vendor rankings (all vendors in category ranked by churn pressure) ──
        vendor_rankings: list[dict[str, Any]] = []
        for r in ranked[:8]:
            v = _canonicalize_vendor(r.get("vendor_name") or "")
            rev = int(r.get("total_reviews") or 0)
            ci = int(r.get("churn_intent") or 0)
            cd = round((ci * 100.0 / rev), 1) if rev else 0.0
            urg = round(float(r.get("avg_urgency") or 0), 1)
            dm_rate = float(dm_lookup.get(v, 0))
            pr_rate = float(price_lookup.get(v, 0))
            comp_ent = competitor_lookup.get(v, [])
            disp_m = sum(c.get("mentions", 0) for c in comp_ent)
            _v_arch = _rl.get(v, {}).get("archetype")
            vr_score = _compute_churn_pressure_score(
                churn_density=cd, avg_urgency=urg, dm_churn_rate=dm_rate,
                displacement_mention_count=disp_m, price_complaint_rate=pr_rate,
                total_reviews=rev,
                archetype=_v_arch,
            )
            # Use reasoning risk_level when available (Gap #26)
            _rc_risk = _rl.get(v, {}).get("risk_level", "")
            _det_risk = "high" if vr_score >= 70 else ("medium" if vr_score >= 40 else "low")
            vendor_rankings.append({
                "vendor": v,
                "churn_pressure_score": vr_score,
                "churn_signal_density": cd,
                "total_reviews": rev,
                "risk_level": _rc_risk or _det_risk,
            })
        vendor_rankings.sort(key=lambda x: -x["churn_pressure_score"])

        # ── Case studies (top quotes per category with company context) ──
        cat_quotes: list[dict[str, Any]] = []
        for r in rows:
            v = _canonicalize_vendor(r.get("vendor_name") or "")
            for q in quote_lookup.get(v, [])[:2]:
                if isinstance(q, dict):
                    cat_quotes.append({
                        "vendor": v,
                        "quote": str(q.get("quote", q.get("text", str(q))))[:200],
                        "company": q.get("company", "Anonymous"),
                        "urgency": q.get("urgency", 0),
                        "title": q.get("title"),
                    })
                elif isinstance(q, str):
                    cat_quotes.append({
                        "vendor": v, "quote": q[:200],
                        "company": "Anonymous", "urgency": 0,
                    })
        cat_quotes.sort(key=lambda x: -float(x.get("urgency", 0)))
        case_studies = cat_quotes[:3]

        # ── Top feature gaps (aggregated across category) ──
        cat_gaps: dict[str, int] = {}
        for r in rows:
            v = _canonicalize_vendor(r.get("vendor_name") or "")
            for g in feature_gap_lookup.get(v, []):
                feat = g.get("feature", "")
                if feat:
                    cat_gaps[feat] = cat_gaps.get(feat, 0) + g.get("count", g.get("mentions", 1))
        top_gaps = sorted(
            [{"feature": k, "mentions": v} for k, v in cat_gaps.items()],
            key=lambda x: -x["mentions"],
        )[:5]

        # Archetype distribution: count + avg confidence per archetype in category
        arch_counts: dict[str, int] = {}
        arch_conf_sums: dict[str, float] = {}
        for r in rows:
            v = _canonicalize_vendor(r.get("vendor_name") or "")
            _rc = _rl.get(v, {})
            arch = _rc.get("archetype", "")
            if arch:
                arch_counts[arch] = arch_counts.get(arch, 0) + 1
                arch_conf_sums[arch] = arch_conf_sums.get(arch, 0) + _rc.get("confidence", 0)
        archetype_distribution = (
            [
                {"archetype": k, "count": v, "avg_confidence": round(arch_conf_sums.get(k, 0) / v, 2)}
                for k, v in sorted(arch_counts.items(), key=lambda x: -x[1])
            ]
            if arch_counts else []
        )

        results.append({
            "category": category,
            "highest_churn_risk": highest_vendor,
            "emerging_challenger": emerging,
            "dominant_pain": dominant_pain,
            "market_shift_signal": _build_market_shift_signal(
                category, highest_vendor, churn_density, total_reviews, emerging, _rl,
            ),
            "industry_distribution": top_industries,
            "company_size_distribution": top_sizes,
            "vendor_rankings": vendor_rankings,
            "case_studies": case_studies,
            "top_feature_gaps": top_gaps,
            "archetype_distribution": archetype_distribution,
        })
    results.sort(key=lambda x: x["category"])
    return results[:limit]


def _build_deterministic_battle_cards(
    vendor_scores: list[dict[str, Any]],
    *,
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list],
    price_lookup: dict[str, float],
    budget_lookup: dict[str, dict],
    sentiment_lookup: dict[str, dict[str, int]],
    dm_lookup: dict[str, float],
    company_lookup: dict[str, list],
    product_profile_lookup: dict[str, dict],
    competitive_disp: list[dict[str, Any]],
    competitor_reasons: list[dict[str, Any]],
    reasoning_lookup: dict[str, dict] | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Build per-vendor battle cards from aggregated data.

    Each card is a concise, sales-oriented one-pager with:
    - Top weaknesses (merged from product profile + pain categories)
    - Customer pain quotes (verbatim, with provenance)
    - Competitor differentiators (who they lose to, and why)
    - Objection data (raw metrics for LLM-generated sales copy)
    """
    # Build reason lookup for inferring primary driver on differentiators
    # (reuses _build_reason_lookup and _infer_driver_from_reasons from displacement map)
    reason_lookup = _build_reason_lookup(competitor_reasons)

    # Merge multi-category rows per vendor (same pattern as vendor feed builder)
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
            if reviews > m["category_reviews"]:
                m["category"] = category
                m["category_reviews"] = reviews

    cards: list[dict[str, Any]] = []
    for vendor, m in merged.items():
        total_reviews = m["total_reviews"]
        churn_intent = m["churn_intent"]
        churn_density = round(churn_intent * 100.0 / total_reviews, 1) if total_reviews else 0.0
        avg_urgency = round(
            m["urgency_weighted_sum"] / total_reviews, 1
        ) if total_reviews else 0.0
        dm_rate = float(dm_lookup.get(vendor, 0))
        price_rate = float(price_lookup.get(vendor, 0))

        # Qualification gate -- reasoning can lower thresholds for high-confidence archetypes
        _rc = (reasoning_lookup or {}).get(vendor, {})
        _has_reasoning = bool(_rc.get("archetype")) and _rc.get("confidence", 0) >= 0.7
        _density_gate = 10 if _has_reasoning else 15
        _urgency_gate = 5 if _has_reasoning else 6
        _dm_gate = 0.2 if _has_reasoning else 0.3
        if churn_density < _density_gate and avg_urgency < _urgency_gate and dm_rate < _dm_gate:
            continue

        # Confidence label
        if total_reviews >= 50:
            confidence = "high"
        elif total_reviews >= 20:
            confidence = "medium"
        else:
            confidence = "low"
        # Boost confidence when reasoning provides corroborating evidence
        _rc = (reasoning_lookup or {}).get(vendor, {})
        if _rc.get("confidence", 0) >= 0.8 and confidence == "medium":
            confidence = "high"

        comp_entries = competitor_lookup.get(vendor, [])
        displacement_mentions = sum(c.get("mentions", 0) for c in comp_entries)

        _vendor_archetype = (reasoning_lookup or {}).get(vendor, {}).get("archetype")
        score = _compute_churn_pressure_score(
            churn_density=churn_density,
            avg_urgency=avg_urgency,
            dm_churn_rate=dm_rate,
            displacement_mention_count=displacement_mentions,
            price_complaint_rate=price_rate,
            total_reviews=total_reviews,
            archetype=_vendor_archetype,
        )

        # -- Section 1: Vendor Weaknesses --
        # Merge product profile weaknesses with pain categories
        weaknesses: list[dict[str, Any]] = []
        seen_areas: set[str] = set()

        profile = product_profile_lookup.get(vendor, {})
        for w in (profile.get("weaknesses") or []):
            if not isinstance(w, dict):
                continue
            area = w.get("area", "")
            if area and area not in seen_areas:
                seen_areas.add(area)
                weaknesses.append({
                    "area": area,
                    "score": w.get("score"),
                    "evidence_count": w.get("evidence_count", 0),
                    "source": "product_profile",
                })

        for p in pain_lookup.get(vendor, []):
            area = p.get("category", "")
            if area and area not in seen_areas:
                seen_areas.add(area)
                weaknesses.append({
                    "area": area,
                    "count": p.get("count", 0),
                    "source": "pain_category",
                })

        for g in feature_gap_lookup.get(vendor, []):
            feature = g.get("feature", "")
            if feature and feature not in seen_areas:
                seen_areas.add(feature)
                weaknesses.append({
                    "area": feature,
                    "count": g.get("mentions", 0),
                    "source": "feature_gap",
                })

        # Sort by evidence (higher = more actionable), take top 5
        weaknesses.sort(
            key=lambda w: -(w.get("evidence_count") or w.get("count") or 0),
        )
        weaknesses = weaknesses[:5]

        # -- Section 2: Customer Pain Quotes --
        quotes_raw = quote_lookup.get(vendor, [])
        pain_quotes: list[dict[str, Any]] = []
        for q in quotes_raw[:5]:
            if isinstance(q, dict):
                pain_quotes.append({
                    "quote": q.get("quote", ""),
                    "urgency": q.get("urgency", 0),
                    "source_site": q.get("source_site", ""),
                    "company": q.get("company", ""),
                    "title": q.get("title", ""),
                    "company_size": q.get("company_size", ""),
                    "industry": q.get("industry", ""),
                })
            elif isinstance(q, str):
                pain_quotes.append({"quote": q, "urgency": 0})

        # -- Section 3: Competitor Differentiators --
        differentiators: list[dict[str, Any]] = []
        for c in comp_entries[:5]:
            comp_name = c.get("name", "")
            if not comp_name:
                continue
            comp_profile = product_profile_lookup.get(
                _canonicalize_vendor(comp_name), {},
            )
            # Find which of this vendor's weaknesses the competitor solves
            solves = None
            comp_pain_addressed = comp_profile.get("pain_addressed") or {}
            if isinstance(comp_pain_addressed, dict) and weaknesses:
                for w in weaknesses:
                    area = w.get("area", "")
                    if area in comp_pain_addressed and comp_pain_addressed[area] >= 0.6:
                        solves = area
                        break

            # Switch count from competitor's commonly_switched_from
            switch_count = 0
            for sf in (comp_profile.get("commonly_switched_from") or []):
                if isinstance(sf, dict):
                    sf_vendor = _canonicalize_vendor(sf.get("vendor", ""))
                    if sf_vendor == vendor:
                        switch_count = sf.get("count", 0)
                        break

            # Infer primary driver from competitor reasons (same as displacement map)
            reasons = reason_lookup.get((vendor, _canonicalize_competitor(comp_name)), [])
            driver = _infer_driver_from_reasons(reasons)

            differentiators.append({
                "competitor": comp_name,
                "mentions": c.get("mentions", 0),
                "primary_driver": driver,
                "solves_weakness": solves,
                "switch_count": switch_count,
            })

        # -- Section 4: Objection Data (raw metrics for LLM) --
        sentiment_counts = sentiment_lookup.get(vendor, {})
        if total_reviews < 10 or not sentiment_counts:
            sentiment_dir = "insufficient_data"
        else:
            sentiment_dir = max(sentiment_counts.items(), key=lambda x: x[1])[0]

        top_gaps = [
            {"feature": g.get("feature", ""), "mentions": g.get("mentions", 0)}
            for g in feature_gap_lookup.get(vendor, [])[:5]
        ]

        objection_data = {
            "price_complaint_rate": round(price_rate, 3),
            "dm_churn_rate": round(dm_rate, 3),
            "sentiment_direction": sentiment_dir,
            "top_feature_gaps": top_gaps,
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "avg_urgency": avg_urgency,
            "budget_context": budget_lookup.get(vendor, {}),
        }

        card_entry = {
            "vendor": vendor,
            "category": m.get("category") or "",
            "churn_pressure_score": score,
            "total_reviews": total_reviews,
            "confidence": confidence,
            "vendor_weaknesses": weaknesses,
            "customer_pain_quotes": pain_quotes,
            "competitor_differentiators": differentiators,
            "objection_data": objection_data,
            # Populated by LLM pass in run():
            "objection_handlers": [],
            "recommended_plays": [],
        }
        rc = (reasoning_lookup or {}).get(vendor, {})
        if rc:
            card_entry["archetype"] = rc.get("archetype", "")
            card_entry["archetype_confidence"] = rc.get("confidence", 0)
            card_entry["archetype_risk_level"] = rc.get("risk_level", "")
            card_entry["archetype_key_signals"] = rc.get("key_signals", [])
            card_entry["falsification_conditions"] = rc.get("falsification_conditions", [])
            card_entry["uncertainty_sources"] = rc.get("uncertainty_sources", [])
        cards.append(card_entry)

    def _bc_sort_key(x: dict) -> tuple:
        score = x["churn_pressure_score"]
        rc = (reasoning_lookup or {}).get(x.get("vendor", ""), {})
        reasoning_boost = min(rc.get("confidence", 0) * 5, 5.0) if rc.get("archetype") else 0
        return (-(score + reasoning_boost),)
    cards.sort(key=_bc_sort_key)
    return cards[:limit]


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
    strip_ids: bool = False,
) -> list[dict[str, Any]]:
    """Trim quote bundles while preserving vendor labels.

    When *strip_ids* is True, returns plain strings (for LLM payloads).
    """
    trimmed: list[dict[str, Any]] = []
    for row in rows[:outer_limit]:
        raw_quotes = list(row.get("quotes") or [])[:quote_limit]
        trimmed.append({
            "vendor": row.get("vendor"),
            "quotes": _strip_quote_ids(raw_quotes) if strip_ids else raw_quotes,
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
                strip_ids=True,
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


def _quote_text(q: Any) -> str | None:
    """Extract plain quote text from a quote item (dict or string)."""
    if isinstance(q, str):
        return q
    if isinstance(q, dict):
        return q.get("quote")
    return None


def _strip_quote_ids(quotes: list) -> list[str]:
    """Strip review_ids from quote dicts, returning plain strings for LLM payloads."""
    return [t for q in quotes if (t := _quote_text(q))]


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


async def _persist_vendor_snapshots(
    pool,
    vendor_scores: list[dict[str, Any]],
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    high_intent: list[dict[str, Any]],
    today: date,
    price_lookup: dict[str, float] | None = None,
    dm_lookup: dict[str, float] | None = None,
) -> int:
    """Persist daily vendor health snapshots and clean up old data."""
    cfg = settings.b2b_churn

    # Build per-vendor high-intent counts
    hi_counts: dict[str, int] = {}
    for hi in high_intent:
        vendor = hi.get("vendor", "")
        if vendor:
            hi_counts[vendor] = hi_counts.get(vendor, 0) + 1

    # Fetch displacement edge counts for today (count edges where vendor is the source)
    disp_rows = await pool.fetch(
        "SELECT from_vendor, count(*) AS cnt FROM b2b_displacement_edges "
        "WHERE computed_date = $1 GROUP BY from_vendor",
        today,
    )
    disp_counts: dict[str, int] = {r["from_vendor"]: r["cnt"] for r in disp_rows}

    persisted = 0
    for row in vendor_scores:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        total_reviews = int(row.get("total_reviews") or 0)
        churn_intent = int(row.get("churn_intent") or 0)
        churn_density = round((churn_intent * 100.0 / total_reviews), 1) if total_reviews else 0.0
        avg_urgency = round(float(row.get("avg_urgency") or 0), 1)
        positive_pct = row.get("positive_review_pct")
        recommend_yes = int(row.get("recommend_yes") or 0)
        recommend_no = int(row.get("recommend_no") or 0)
        recommend_ratio = round(((recommend_yes - recommend_no) / total_reviews) * 100, 1) if total_reviews else 0.0

        pains = pain_lookup.get(vendor, [])
        top_pain = (pains[0] if pains else {}).get("category")
        comps = competitor_lookup.get(vendor, [])
        top_competitor = comps[0]["name"] if comps else None

        _dm_rate = (dm_lookup or {}).get(vendor)
        _price_rate = (price_lookup or {}).get(vendor)
        _disp_cnt = disp_counts.get(vendor, 0)
        _pressure = _compute_churn_pressure_score(
            churn_density=churn_density,
            avg_urgency=avg_urgency,
            dm_churn_rate=_dm_rate or 0.0,
            displacement_mention_count=_disp_cnt,
            price_complaint_rate=_price_rate or 0.0,
            total_reviews=total_reviews,
        )

        try:
            await pool.execute(
                """
                INSERT INTO b2b_vendor_snapshots (
                    vendor_name, snapshot_date, total_reviews, churn_intent,
                    churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                    top_pain, top_competitor, pain_count, competitor_count,
                    displacement_edge_count, high_intent_company_count,
                    pressure_score, dm_churn_rate, price_complaint_rate
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                          $15, $16, $17)
                ON CONFLICT (vendor_name, snapshot_date) DO UPDATE SET
                    total_reviews = EXCLUDED.total_reviews,
                    churn_intent = EXCLUDED.churn_intent,
                    churn_density = EXCLUDED.churn_density,
                    avg_urgency = EXCLUDED.avg_urgency,
                    positive_review_pct = EXCLUDED.positive_review_pct,
                    recommend_ratio = EXCLUDED.recommend_ratio,
                    top_pain = EXCLUDED.top_pain,
                    top_competitor = EXCLUDED.top_competitor,
                    pain_count = EXCLUDED.pain_count,
                    competitor_count = EXCLUDED.competitor_count,
                    displacement_edge_count = EXCLUDED.displacement_edge_count,
                    high_intent_company_count = EXCLUDED.high_intent_company_count,
                    pressure_score = EXCLUDED.pressure_score,
                    dm_churn_rate = EXCLUDED.dm_churn_rate,
                    price_complaint_rate = EXCLUDED.price_complaint_rate
                """,
                vendor, today, total_reviews, churn_intent,
                churn_density, avg_urgency,
                float(positive_pct) if positive_pct is not None else None,
                recommend_ratio,
                top_pain, top_competitor,
                len(pains), len(comps),
                _disp_cnt,
                hi_counts.get(vendor, 0),
                _pressure, _dm_rate, _price_rate,
            )
            persisted += 1
        except Exception:
            logger.warning("Failed to persist snapshot for vendor %s", vendor)

    # Retention cleanup
    await pool.execute(
        "DELETE FROM b2b_vendor_snapshots WHERE snapshot_date < CURRENT_DATE - $1::int",
        cfg.snapshot_retention_days,
    )
    await pool.execute(
        "DELETE FROM b2b_change_events WHERE event_date < CURRENT_DATE - $1::int",
        cfg.change_event_retention_days,
    )

    return persisted


async def _detect_change_events(
    pool,
    vendor_scores: list[dict[str, Any]],
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    today: date,
    price_lookup: dict[str, float] | None = None,
    dm_lookup: dict[str, float] | None = None,
    temporal_lookup: dict[str, dict] | None = None,
) -> int:
    """Compare today's vendor data against prior snapshots and log change events.

    When *temporal_lookup* is provided (from TemporalEngine), z-score anomalies
    are used instead of hardcoded delta thresholds for urgency and churn density.
    A new ``velocity_acceleration`` event fires when a metric has both high
    velocity AND positive acceleration.
    """
    detected = 0

    for row in vendor_scores:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue

        # Fetch most recent prior snapshot
        prior = await pool.fetchrow(
            "SELECT * FROM b2b_vendor_snapshots "
            "WHERE vendor_name = $1 AND snapshot_date < $2 "
            "ORDER BY snapshot_date DESC LIMIT 1",
            vendor, today,
        )
        if not prior:
            continue

        # Compute current metrics inline (same as _persist_vendor_snapshots)
        total_reviews = int(row.get("total_reviews") or 0)
        churn_intent = int(row.get("churn_intent") or 0)
        churn_density = round((churn_intent * 100.0 / total_reviews), 1) if total_reviews else 0.0
        avg_urgency = round(float(row.get("avg_urgency") or 0), 1)
        recommend_yes = int(row.get("recommend_yes") or 0)
        recommend_no = int(row.get("recommend_no") or 0)
        recommend_ratio = round(((recommend_yes - recommend_no) / total_reviews) * 100, 1) if total_reviews else 0.0

        pains = pain_lookup.get(vendor, [])
        top_pain = (pains[0] if pains else {}).get("category")
        comps = competitor_lookup.get(vendor, [])
        top_competitor = comps[0]["name"] if comps else None

        events: list[tuple[str, str, float | None, float | None, float | None]] = []

        # Temporal evidence for this vendor (z-scores, velocities, accelerations)
        td = (temporal_lookup or {}).get(vendor, {})
        anomalies_by_metric: dict[str, dict] = {}
        for a in td.get("anomalies", []):
            if isinstance(a, dict):
                anomalies_by_metric[a.get("metric", "")] = a

        # Urgency spike/drop -- prefer z-score anomaly when available
        prior_urg = float(prior["avg_urgency"] or 0)
        urg_delta = avg_urgency - prior_urg
        urg_anomaly = anomalies_by_metric.get("avg_urgency", {})
        if urg_anomaly.get("is_anomaly") and urg_anomaly.get("z_score", 0) > 0:
            z = urg_anomaly["z_score"]
            events.append(("urgency_spike", f"Avg urgency rose from {prior_urg} to {avg_urgency} (z={z:.1f})", prior_urg, avg_urgency, urg_delta))
        elif urg_anomaly.get("is_anomaly") and urg_anomaly.get("z_score", 0) < 0:
            z = urg_anomaly["z_score"]
            events.append(("urgency_drop", f"Avg urgency fell from {prior_urg} to {avg_urgency} (z={z:.1f})", prior_urg, avg_urgency, urg_delta))
        elif urg_delta >= 1.0:
            events.append(("urgency_spike", f"Avg urgency rose from {prior_urg} to {avg_urgency}", prior_urg, avg_urgency, urg_delta))
        elif urg_delta <= -1.0:
            events.append(("urgency_drop", f"Avg urgency fell from {prior_urg} to {avg_urgency}", prior_urg, avg_urgency, urg_delta))

        # Churn density spike -- prefer z-score anomaly when available
        prior_cd = float(prior["churn_density"] or 0)
        cd_delta = churn_density - prior_cd
        cd_anomaly = anomalies_by_metric.get("churn_density", {})
        if cd_anomaly.get("is_anomaly") and cd_anomaly.get("z_score", 0) > 0:
            z = cd_anomaly["z_score"]
            events.append(("churn_density_spike", f"Churn density rose from {prior_cd}% to {churn_density}% (z={z:.1f})", prior_cd, churn_density, cd_delta))
        elif cd_delta >= 5.0:
            events.append(("churn_density_spike", f"Churn density rose from {prior_cd}% to {churn_density}%", prior_cd, churn_density, cd_delta))

        # Velocity acceleration event: metric has both high velocity AND positive acceleration
        for metric_key in ("churn_density", "avg_urgency", "pressure_score"):
            vel = td.get(f"velocity_{metric_key}")
            accel = td.get(f"accel_{metric_key}")
            if vel is not None and accel is not None and vel > 0 and accel > 0:
                events.append((
                    "velocity_acceleration",
                    f"{metric_key} accelerating: velocity={vel:.3f}/day, acceleration={accel:.3f}/day^2",
                    vel, accel, None,
                ))

        # NPS / recommend ratio shift -- prefer z-score when available
        prior_rr = float(prior["recommend_ratio"] or 0)
        rr_delta = recommend_ratio - prior_rr
        rr_anomaly = anomalies_by_metric.get("recommend_ratio", {})
        if rr_anomaly.get("is_anomaly"):
            z = rr_anomaly["z_score"]
            direction = "improved" if rr_delta > 0 else "declined"
            events.append(("nps_shift", f"Recommend ratio {direction} from {prior_rr} to {recommend_ratio} (z={z:.1f})", prior_rr, recommend_ratio, rr_delta))
        elif abs(rr_delta) >= 10.0:
            direction = "improved" if rr_delta > 0 else "declined"
            events.append(("nps_shift", f"Recommend ratio {direction} from {prior_rr} to {recommend_ratio}", prior_rr, recommend_ratio, rr_delta))

        # Review volume spike -- prefer z-score when available
        prior_tr = int(prior["total_reviews"] or 0)
        if prior_tr > 0:
            vol_pct = ((total_reviews - prior_tr) / prior_tr) * 100
            vol_anomaly = anomalies_by_metric.get("total_reviews", {})
            if vol_anomaly.get("is_anomaly") and vol_anomaly.get("z_score", 0) > 0:
                z = vol_anomaly["z_score"]
                events.append(("review_volume_spike", f"Review count jumped from {prior_tr} to {total_reviews} (+{vol_pct:.0f}%, z={z:.1f})", float(prior_tr), float(total_reviews), vol_pct))
            elif vol_pct >= 25.0:
                events.append(("review_volume_spike", f"Review count jumped from {prior_tr} to {total_reviews} (+{vol_pct:.0f}%)", float(prior_tr), float(total_reviews), vol_pct))

        # Pressure score spike (threshold: 10.0 points on 0-100 scale)
        prior_ps = float(prior["pressure_score"] or 0)
        _dm_rate = (dm_lookup or {}).get(vendor, 0.0)
        _price_rate = (price_lookup or {}).get(vendor, 0.0)
        cur_ps = _compute_churn_pressure_score(
            churn_density=churn_density,
            avg_urgency=avg_urgency,
            dm_churn_rate=_dm_rate,
            displacement_mention_count=0,  # not available here, use snapshot delta
            price_complaint_rate=_price_rate,
            total_reviews=total_reviews,
        )
        ps_delta = cur_ps - prior_ps
        ps_anomaly = anomalies_by_metric.get("pressure_score", {})
        if ps_anomaly.get("is_anomaly") and ps_anomaly.get("z_score", 0) > 0:
            z = ps_anomaly["z_score"]
            events.append(("pressure_score_spike", f"Pressure score rose from {prior_ps} to {cur_ps} (z={z:.1f})", prior_ps, cur_ps, ps_delta))
        elif ps_delta >= 10.0:
            events.append(("pressure_score_spike", f"Pressure score rose from {prior_ps} to {cur_ps}", prior_ps, cur_ps, ps_delta))

        # Decision-maker churn rate spike -- prefer z-score when available
        prior_dm = float(prior["dm_churn_rate"] or 0)
        dm_delta = _dm_rate - prior_dm
        dm_anomaly = anomalies_by_metric.get("dm_churn_rate", {})
        if dm_anomaly.get("is_anomaly") and dm_anomaly.get("z_score", 0) > 0:
            z = dm_anomaly["z_score"]
            events.append(("dm_churn_spike", f"DM churn rate rose from {prior_dm:.2%} to {_dm_rate:.2%} (z={z:.1f})", prior_dm, _dm_rate, dm_delta))
        elif dm_delta >= 0.15:
            events.append(("dm_churn_spike", f"DM churn rate rose from {prior_dm:.2%} to {_dm_rate:.2%}", prior_dm, _dm_rate, dm_delta))

        # New pain category
        prior_pain = prior["top_pain"]
        if top_pain and prior_pain and top_pain != prior_pain:
            events.append(("new_pain_category", f"Top pain shifted from '{prior_pain}' to '{top_pain}'", None, None, None))

        # New competitor
        prior_comp = prior["top_competitor"]
        if top_competitor and prior_comp and top_competitor != prior_comp:
            events.append(("new_competitor", f"Top competitor shifted from '{prior_comp}' to '{top_competitor}'", None, None, None))

        webhook_events: list[tuple[str, dict]] = []
        for event_type, description, old_val, new_val, delta in events:
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_change_events (vendor_name, event_date, event_type, description, old_value, new_value, delta)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    vendor, today, event_type, description, old_val, new_val, delta,
                )
                detected += 1
                webhook_events.append((vendor, {
                    "event_type": event_type,
                    "vendor_name": vendor,
                    "description": description,
                    "old_value": old_val,
                    "new_value": new_val,
                    "delta": delta,
                    "event_date": str(today),
                }))
            except Exception:
                logger.warning("Failed to persist change event %s for %s", event_type, vendor)

        # Dispatch webhooks for change events (fire-and-forget, never raises)
        if webhook_events:
            try:
                from ...services.b2b.webhook_dispatcher import dispatch_webhooks_multi
                await dispatch_webhooks_multi(pool, "change_event", webhook_events)
            except Exception:
                logger.debug("Webhook dispatch skipped for change events")

    # Cross-vendor correlation: detect concurrent shifts
    detected += await _detect_concurrent_shifts(pool, today)

    return detected


async def _detect_concurrent_shifts(pool, today: date) -> int:
    """Detect dates where 3+ vendors had the same event type -- signals market trend."""
    detected = 0
    try:
        rows = await pool.fetch(
            """
            SELECT event_type, COUNT(DISTINCT vendor_name) AS vendor_count,
                   ARRAY_AGG(DISTINCT vendor_name ORDER BY vendor_name) AS vendors,
                   AVG(delta) AS avg_delta
            FROM b2b_change_events
            WHERE event_date = $1
            GROUP BY event_type
            HAVING COUNT(DISTINCT vendor_name) >= 3
            """,
            today,
        )
        for row in rows:
            event_type = row["event_type"]
            vendor_count = row["vendor_count"]
            vendors = row["vendors"]
            avg_delta = round(float(row["avg_delta"] or 0), 2)
            vendor_list = ", ".join(vendors[:5])
            suffix = f" +{vendor_count - 5} more" if vendor_count > 5 else ""
            description = (
                f"Concurrent {event_type} across {vendor_count} vendors: "
                f"{vendor_list}{suffix} (avg delta: {avg_delta})"
            )
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_change_events
                        (vendor_name, event_date, event_type, description, delta, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    "__market__",
                    today,
                    "concurrent_shift",
                    description,
                    avg_delta,
                    json.dumps({
                        "original_event_type": event_type,
                        "vendor_count": vendor_count,
                        "vendors": vendors,
                    }),
                )
                detected += 1
            except Exception:
                logger.debug("Failed to persist concurrent_shift for %s", event_type)
    except Exception:
        logger.debug("Concurrent shift detection skipped", exc_info=True)
    return detected


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
    from ...pipelines.llm import get_pipeline_llm as _get_plm
    _ci_llm = _get_plm(workload="vllm")
    span = tracer.start_span(
        span_name="b2b.churn_intelligence.run",
        operation_type="intelligence",
        model_name=getattr(_ci_llm, "model", getattr(_ci_llm, "model_id", None)) if _ci_llm else None,
        model_provider=getattr(_ci_llm, "name", None) if _ci_llm else None,
        metadata={
            "business": build_business_trace_context(
                workflow="b2b_churn_intelligence",
                report_type="weekly_churn_feed",
            ),
        },
    )

    # Gather all data sources + data_context + provenance in parallel
    (
        vendor_scores, high_intent, competitive_disp,
        pain_dist, feature_gaps,
        negative_counts, price_rates, dm_rates,
        churning_companies, quotable_evidence,
        budget_signals, use_case_dist, sentiment_traj,
        buyer_auth, timeline_signals, competitor_reasons,
        keyword_spikes, data_context, vendor_provenance,
        displacement_provenance,
        pain_provenance, use_case_provenance, integration_provenance,
        buyer_profile_provenance,
        insider_aggregates_raw,
        product_profiles_raw,
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
        _fetch_vendor_provenance(pool, window_days),
        _fetch_displacement_provenance(pool, window_days),
        _fetch_pain_provenance(pool, window_days),
        _fetch_use_case_provenance(pool, window_days),
        _fetch_integration_provenance(pool, window_days),
        _fetch_buyer_profile_provenance(pool, window_days),
        _fetch_insider_aggregates(pool, window_days),
        _fetch_product_profiles(pool),
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
    if isinstance(vendor_provenance, Exception):
        logger.warning("vendor_provenance fetch failed: %s", vendor_provenance)
        vendor_provenance = {}
    if isinstance(displacement_provenance, Exception):
        logger.warning("displacement_provenance fetch failed: %s", displacement_provenance)
        displacement_provenance = {}
    if isinstance(pain_provenance, Exception):
        logger.warning("pain_provenance fetch failed: %s", pain_provenance)
        pain_provenance = {}
    if isinstance(use_case_provenance, Exception):
        logger.warning("use_case_provenance fetch failed: %s", use_case_provenance)
        use_case_provenance = {}
    if isinstance(integration_provenance, Exception):
        logger.warning("integration_provenance fetch failed: %s", integration_provenance)
        integration_provenance = {}
    if isinstance(buyer_profile_provenance, Exception):
        logger.warning("buyer_profile_provenance fetch failed: %s", buyer_profile_provenance)
        buyer_profile_provenance = {}
    if isinstance(insider_aggregates_raw, Exception):
        logger.warning("insider_aggregates fetch failed: %s", insider_aggregates_raw)
        insider_aggregates_raw = []
    if isinstance(product_profiles_raw, Exception):
        logger.warning("product_profiles fetch failed: %s", product_profiles_raw)
        product_profiles_raw = []
    insider_lookup = _build_insider_lookup(insider_aggregates_raw)

    # Check if there's enough data
    if not vendor_scores and not high_intent:
        tracer.end_span(span, status="completed", output_data={"skipped": "no enriched reviews"})
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

    # Enrich payload with temporal analysis + archetype pre-scores per vendor
    _temporal_lookup: dict[str, dict] = {}
    _archetype_lookup: dict[str, list[dict]] = {}
    try:
        from atlas_brain.reasoning.temporal import TemporalEngine
        from atlas_brain.reasoning.archetypes import enrich_evidence_with_archetypes

        temporal_engine = TemporalEngine(pool)
        temporal_summaries = []
        for vs in vendor_scores[:cfg.intelligence_exploratory_vendor_limit]:
            vname = vs["vendor_name"]
            try:
                te = await temporal_engine.analyze_vendor(vname)
                td = TemporalEngine.to_evidence_dict(te)
                enriched = enrich_evidence_with_archetypes(
                    {"vendor_name": vname, **td}, td,
                )
                # Extract per-metric velocities and accelerations
                velocities = {k: v for k, v in td.items() if k.startswith("velocity_")}
                accelerations = {k: v for k, v in td.items() if k.startswith("accel_")}
                temporal_summaries.append({
                    "vendor": vname,
                    "velocities": velocities,
                    "accelerations": accelerations,
                    "anomalies": td.get("anomalies", []),
                    "archetype_scores": enriched.get("archetype_scores", []),
                    "insufficient_data": td.get("temporal_status") == "insufficient_data",
                })
                # Store for _build_vendor_evidence
                _temporal_lookup[vname] = td
                arch_scores = enriched.get("archetype_scores", [])
                if arch_scores:
                    _archetype_lookup[vname] = arch_scores
            except Exception:
                logger.debug("Temporal enrichment skipped for %s", vname)
        if temporal_summaries:
            payload["temporal_analysis"] = temporal_summaries
            payload_size = len(json.dumps(payload, default=str))
    except Exception:
        logger.debug("Temporal/archetype enrichment unavailable", exc_info=True)

    # --- Stratified reasoning: recall/reconstitute/reason per vendor ---
    # Runs BEFORE reports so conclusions feed into all builders + LLM prompts.
    # When stratified_reasoning_enabled, builds rich evidence dicts first.
    reasoning_lookup: dict[str, dict] = {}
    evidence_for_reasoning: dict[str, dict[str, Any]] | None = None
    if cfg.stratified_reasoning_enabled:
        _pre_pain = _build_pain_lookup(pain_dist)
        _pre_comp = _build_competitor_lookup(competitive_disp)
        _pre_fg = _build_feature_gap_lookup(feature_gaps)
        _pre_kw = _build_keyword_spike_lookup(keyword_spikes)
        _pre_dm = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
        _pre_price = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
        _pre_quote = {r["vendor"]: r["quotes"] for r in quotable_evidence}
        _pre_budget = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
        _pre_ba = _build_buyer_auth_lookup(buyer_auth)
        _pre_uc = _build_use_case_lookup(use_case_dist)

        evidence_for_reasoning = {}
        for vs in vendor_scores:
            vname = _canonicalize_vendor(vs.get("vendor_name") or "")
            if vname:
                evidence_for_reasoning[vname] = _build_vendor_evidence(
                    vs,
                    pain_lookup=_pre_pain,
                    competitor_lookup=_pre_comp,
                    feature_gap_lookup=_pre_fg,
                    insider_lookup=insider_lookup,
                    keyword_spike_lookup=_pre_kw,
                    temporal_lookup=_temporal_lookup or None,
                    archetype_lookup=_archetype_lookup or None,
                    dm_lookup=_pre_dm,
                    price_lookup=_pre_price,
                    quote_lookup=_pre_quote,
                    budget_lookup=_pre_budget,
                    buyer_auth_lookup=_pre_ba,
                    use_case_lookup=_pre_uc,
                )

    try:
        from atlas_brain.reasoning import get_stratified_reasoner, init_stratified_reasoner
        from atlas_brain.reasoning.tiers import Tier, gather_tier_context

        reasoner = get_stratified_reasoner()
        if reasoner is None:
            # Lazy-init for standalone/manual runs (main app lifespan not active)
            await init_stratified_reasoner(pool)
            reasoner = get_stratified_reasoner()
        if reasoner is not None:
            sem = asyncio.Semaphore(cfg.stratified_reasoning_concurrency)
            total_tokens = 0
            mode_counts: dict[str, int] = {}

            async def _analyze_one(vs: dict[str, Any]) -> None:
                nonlocal total_tokens
                vname = _canonicalize_vendor(vs.get("vendor_name") or "")
                if not vname:
                    return
                category = vs.get("product_category", "")
                evidence = (evidence_for_reasoning or {}).get(vname, vs)
                async with sem:
                    try:
                        tier_ctx = await gather_tier_context(
                            reasoner._cache, Tier.VENDOR_ARCHETYPE,
                            vendor_name=vname, product_category=category,
                        )
                        sr = await reasoner.analyze(
                            vendor_name=vname,
                            evidence=evidence,
                            product_category=category,
                            tier_context=tier_ctx,
                        )
                        conclusion = sr.conclusion or {}
                        reasoning_lookup[vname] = {
                            "archetype": conclusion.get("archetype", ""),
                            "confidence": sr.confidence,
                            "risk_level": conclusion.get("risk_level", ""),
                            "executive_summary": conclusion.get("executive_summary", ""),
                            "key_signals": conclusion.get("key_signals", []),
                            "falsification_conditions": conclusion.get("falsification_conditions", []),
                            "uncertainty_sources": conclusion.get("uncertainty_sources", []),
                            "mode": sr.mode,
                            "tokens_used": sr.tokens_used,
                        }
                        total_tokens += sr.tokens_used
                        mode_counts[sr.mode] = mode_counts.get(sr.mode, 0) + 1
                    except Exception:
                        logger.debug("Stratified reasoning failed for %s", vname, exc_info=True)

            await asyncio.gather(*[
                _analyze_one(vs)
                for vs in vendor_scores[:cfg.intelligence_exploratory_vendor_limit]
            ])

            if reasoning_lookup:
                mode_summary = ", ".join(f"{m}={c}" for m, c in sorted(mode_counts.items()))
                logger.info(
                    "Stratified reasoning: %d vendors, modes: %s, total_tokens: %d",
                    len(reasoning_lookup), mode_summary, total_tokens,
                )
                # Inject summary into payload for exploratory overview LLM
                payload["stratified_intelligence"] = [
                    {
                        "vendor": vname,
                        "archetype": rc.get("archetype", ""),
                        "confidence": rc.get("confidence", 0),
                        "risk_level": rc.get("risk_level", ""),
                        "executive_summary": rc.get("executive_summary", ""),
                        "key_signals": rc.get("key_signals", []),
                    }
                    for vname, rc in reasoning_lookup.items()
                ]
                payload_size = len(json.dumps(payload, default=str))
    except Exception:
        logger.debug("Stratified reasoning integration skipped", exc_info=True)

    # --- Cross-vendor ecosystem analysis (category-level intelligence) ---
    ecosystem_evidence: dict[str, Any] = {}
    if cfg.stratified_reasoning_enabled and reasoning_lookup:
        try:
            from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer
            eco = EcosystemAnalyzer(pool)
            eco_results = await eco.analyze_all_categories()
            for cat, ev in eco_results.items():
                ecosystem_evidence[cat] = {
                    "category": cat,
                    "hhi": getattr(ev.health, "hhi", None),
                    "market_structure": getattr(ev.health, "market_structure", None),
                    "displacement_intensity": getattr(ev.health, "displacement_intensity", None),
                    "dominant_archetype": getattr(ev.health, "dominant_archetype", None),
                    "archetype_distribution": ev.archetype_distribution or {},
                }
            if ecosystem_evidence:
                logger.info("Ecosystem analysis: %d categories", len(ecosystem_evidence))
        except Exception:
            logger.debug("Ecosystem analysis skipped", exc_info=True)

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
    llm_usage: dict[str, Any] = {}
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
                    usage_out=llm_usage,
                    span_name="b2b.churn_intelligence.exploratory_overview",
                    trace_metadata=_build_llm_trace_metadata(
                        "exploratory_overview",
                        report_type="weekly_churn_feed",
                    ),
                ),
                timeout=300,
            )
        except asyncio.TimeoutError:
            logger.error("LLM call timed out after 300s for b2b_churn_intelligence")

    if llm_usage.get("input_tokens"):
        logger.info("b2b_churn_intelligence LLM tokens: in=%d out=%d model=%s",
                     llm_usage["input_tokens"], llm_usage["output_tokens"],
                     llm_usage.get("model", ""))

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
    product_profile_lookup: dict[str, dict] = {}
    for pp in product_profiles_raw:
        vn = _canonicalize_vendor(pp.get("vendor_name", ""))
        if vn and vn not in product_profile_lookup:
            product_profile_lookup[vn] = pp

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
        reasoning_lookup=reasoning_lookup,
        temporal_lookup=_temporal_lookup or None,
    )
    deterministic_vendor_scorecards = _build_deterministic_vendor_scorecards(
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
        product_profile_lookup=product_profile_lookup,
        prior_reports=prior_reports,
        reasoning_lookup=reasoning_lookup,
        temporal_lookup=_temporal_lookup or None,
    )
    deterministic_displacement_map = _build_deterministic_displacement_map(
        competitive_disp,
        competitor_reasons,
        quote_lookup,
        reasoning_lookup=reasoning_lookup,
    )

    # Enrich displacement edges with provenance and confidence
    for edge in deterministic_displacement_map:
        prov_key = (edge["from_vendor"], edge["to_vendor"])
        prov = displacement_provenance.get(prov_key, {})
        src_dist = prov.get("source_distribution", {})
        edge["source_distribution"] = src_dist
        edge["sample_review_ids"] = prov.get("sample_review_ids", [])
        edge["confidence_score"] = _compute_evidence_confidence(
            edge["mention_count"], src_dist,
        )

    deterministic_category_overview = _build_deterministic_category_overview(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitive_disp=competitive_disp,
        company_lookup=company_lookup,
        quote_lookup=quote_lookup,
        feature_gap_lookup=feature_gap_lookup,
        dm_lookup=dm_lookup,
        price_lookup=price_lookup,
        competitor_lookup=competitor_lookup,
        reasoning_lookup=reasoning_lookup,
    )

    # Enrich category overview with ecosystem evidence (HHI, market structure)
    if ecosystem_evidence:
        for cat_entry in deterministic_category_overview:
            cat_name = cat_entry.get("category", "")
            eco = ecosystem_evidence.get(cat_name)
            if eco:
                cat_entry["ecosystem"] = {
                    "hhi": eco.get("hhi"),
                    "market_structure": eco.get("market_structure"),
                    "displacement_intensity": eco.get("displacement_intensity"),
                    "dominant_archetype": eco.get("dominant_archetype"),
                }

    # LLM pass: generate expert take for each vendor scorecard.
    # When stratified reasoning provided an executive_summary, use it directly
    # as expert_take (skip per-vendor LLM call -- saves ~300 tokens * N vendors).
    scorecard_llm_failures = 0
    scorecard_reasoning_reused = 0
    for sc in deterministic_vendor_scorecards:
        reasoning_summary = sc.get("reasoning_summary", "")
        if reasoning_summary and cfg.stratified_reasoning_enabled:
            sc["expert_take"] = reasoning_summary
            scorecard_reasoning_reused += 1
            continue
        try:
            llm_input = {k: sc[k] for k in (
                "vendor", "churn_pressure_score", "risk_level", "churn_signal_density",
                "avg_urgency", "feature_analysis", "churn_predictors", "competitor_overlap",
                "trend", "sentiment_direction",
            ) if k in sc}
            if sc.get("archetype"):
                llm_input["reasoning_conclusion"] = {
                    "archetype": sc["archetype"],
                    "confidence": sc.get("archetype_confidence", 0),
                    "executive_summary": reasoning_summary,
                    "key_signals": (reasoning_lookup or {}).get(sc.get("vendor", ""), {}).get("key_signals", []),
                }
            narrative = await asyncio.wait_for(
                asyncio.to_thread(
                    call_llm_with_skill,
                    "digest/vendor_deep_dive_narrative",
                    json.dumps(llm_input, default=str),
                    max_tokens=300, temperature=0.3,
                    response_format={"type": "json_object"},
                    workload=_llm_workload,
                    span_name="b2b.churn_intelligence.scorecard_narrative",
                    trace_metadata=_build_llm_trace_metadata(
                        "scorecard_narrative",
                        report_type="vendor_scorecard",
                        vendor_name=sc.get("vendor"),
                    ),
                ),
                timeout=45,
            )
            parsed_narrative = parse_json_response(narrative)
            sc["expert_take"] = parsed_narrative.get("expert_take", "")
        except Exception:
            scorecard_llm_failures += 1
            logger.warning("Vendor deep dive LLM failed for %s", sc.get("vendor"))
            sc["expert_take"] = ""
    if scorecard_llm_failures or scorecard_reasoning_reused:
        logger.info("Vendor scorecard LLM: %d/%d failed, %d reused reasoning summary",
                     scorecard_llm_failures, len(deterministic_vendor_scorecards),
                     scorecard_reasoning_reused)

    parsed["weekly_churn_feed"] = deterministic_weekly_feed
    parsed["vendor_scorecards"] = deterministic_vendor_scorecards
    parsed["displacement_map"] = deterministic_displacement_map
    parsed["category_insights"] = deterministic_category_overview

    # Build per-report-type executive summaries
    _exec_sources = _executive_source_list()
    _exec_summaries: dict[str, str] = {}
    for _rt in ("weekly_churn_feed", "vendor_scorecard", "displacement_report", "category_overview"):
        _exec_summaries[_rt] = _build_validated_executive_summary(
            parsed,
            data_context=data_context,
            executive_sources=_exec_sources,
            report_type=_rt,
        )
    # LLM-synthesized executive summaries (when enabled, upgrades deterministic)
    if cfg.executive_summary_llm_enabled and reasoning_lookup:
        _reasoning_digest = [
            {
                "vendor": vname,
                "archetype": rc.get("archetype", ""),
                "risk_level": rc.get("risk_level", ""),
                "key_signals": rc.get("key_signals", [])[:3],
            }
            for vname, rc in list(reasoning_lookup.items())[:15]
        ]
        _report_data_map = {
            "weekly_churn_feed": deterministic_weekly_feed,
            "vendor_scorecard": deterministic_vendor_scorecards,
            "displacement_report": deterministic_displacement_map,
            "category_overview": deterministic_category_overview,
        }
        for _rt in ("weekly_churn_feed", "vendor_scorecard", "displacement_report", "category_overview"):
            try:
                _es_input = {
                    "report_type": _rt,
                    "report_data": _report_data_map[_rt][:10],
                    "reasoning_summary": _reasoning_digest,
                    "data_context": {
                        k: data_context.get(k)
                        for k in ("enrichment_period", "source_distribution",
                                  "reviews_in_analysis_window", "vendor_count")
                        if data_context.get(k)
                    },
                }
                _es_raw = await asyncio.wait_for(
                    asyncio.to_thread(
                        call_llm_with_skill,
                        "digest/b2b_executive_summary",
                        json.dumps(_es_input, default=str),
                        max_tokens=512, temperature=0.3,
                        response_format={"type": "json_object"},
                        workload=_llm_workload,
                        span_name="b2b.churn_intelligence.executive_summary",
                        trace_metadata=_build_llm_trace_metadata(
                            "executive_summary",
                            report_type=_rt,
                        ),
                    ),
                    timeout=60,
                )
                _es_parsed = parse_json_response(_es_raw)
                llm_summary = _es_parsed.get("executive_summary", "")
                if llm_summary:
                    _exec_summaries[_rt] = llm_summary
            except Exception:
                logger.debug("LLM executive summary failed for %s, using deterministic", _rt)

    _fallback_summary = _exec_summaries.get("weekly_churn_feed", "")

    # Build battle cards (per-vendor, persisted separately)
    deterministic_battle_cards = _build_deterministic_battle_cards(
        vendor_scores,
        pain_lookup=pain_lookup,
        competitor_lookup=competitor_lookup,
        feature_gap_lookup=feature_gap_lookup,
        quote_lookup=quote_lookup,
        price_lookup=price_lookup,
        budget_lookup=budget_lookup,
        sentiment_lookup=sentiment_lookup,
        dm_lookup=dm_lookup,
        company_lookup=company_lookup,
        product_profile_lookup=product_profile_lookup,
        competitive_disp=competitive_disp,
        competitor_reasons=competitor_reasons,
        reasoning_lookup=reasoning_lookup,
    )
    logger.info("Built %d battle cards", len(deterministic_battle_cards))

    # Enrich battle cards with ecosystem context
    if ecosystem_evidence:
        for card in deterministic_battle_cards:
            cat = card.get("category", "")
            eco = ecosystem_evidence.get(cat)
            if eco:
                card["ecosystem_context"] = {
                    "market_structure": eco.get("market_structure"),
                    "dominant_archetype": eco.get("dominant_archetype"),
                    "displacement_intensity": eco.get("displacement_intensity"),
                }

    # LLM pass: generate sales copy for each battle card (with caching)
    from ...reasoning.semantic_cache import SemanticCache, CacheEntry, compute_evidence_hash

    _bc_cache = SemanticCache(pool)
    battle_card_llm_failures = 0
    battle_card_cache_hits = 0
    _bc_llm_fields = (
        "executive_summary", "weakness_analysis", "discovery_questions",
        "landmine_questions", "objection_handlers", "competitive_landscape",
        "talk_track", "recommended_plays",
    )
    for card in deterministic_battle_cards:
        card_hash = compute_evidence_hash({
            "vendor": card.get("vendor"),
            "churn_pressure_score": card.get("churn_pressure_score"),
            "vendor_weaknesses": card.get("vendor_weaknesses"),
            "customer_pain_quotes": card.get("customer_pain_quotes"),
            "competitor_differentiators": card.get("competitor_differentiators"),
            "objection_data": card.get("objection_data"),
        })
        pattern_sig = f"battle_card:{card.get('vendor')}:{card_hash}"

        cached = await _bc_cache.lookup(pattern_sig)
        if cached:
            for _cf in cached.conclusion:
                card[_cf] = cached.conclusion[_cf]
            await _bc_cache.validate(pattern_sig)
            battle_card_cache_hits += 1
            continue

        try:
            sales_copy = await asyncio.wait_for(
                asyncio.to_thread(
                    call_llm_with_skill,
                    "digest/battle_card_sales_copy",
                    json.dumps(card, default=str),
                    max_tokens=3000, temperature=0.5,
                    response_format={"type": "json_object"},
                    workload=_llm_workload,
                    span_name="b2b.churn_intelligence.battle_card_sales_copy",
                    trace_metadata=_build_llm_trace_metadata(
                        "battle_card_sales_copy",
                        report_type="battle_card",
                        vendor_name=card.get("vendor"),
                    ),
                ),
                timeout=90,
            )
            parsed_copy = parse_json_response(sales_copy)
            for _f in _bc_llm_fields:
                if _f in parsed_copy:
                    card[_f] = parsed_copy[_f]
        except Exception:
            battle_card_llm_failures += 1
            logger.warning(
                "Battle card LLM failed for %s, using data-only card",
                card.get("vendor"),
            )
            continue

        try:
            await _bc_cache.store(CacheEntry(
                pattern_sig=pattern_sig,
                pattern_class="battle_card_sales_copy",
                conclusion={_f: card[_f] for _f in _bc_llm_fields if _f in card},
                confidence=0.95,
                evidence_hash=card_hash,
                vendor_name=card.get("vendor"),
                conclusion_type="sales_copy",
            ))
        except Exception:
            logger.warning(
                "Failed to cache battle card for %s",
                card.get("vendor"),
            )
    logger.info(
        "Battle card LLM: %d cache hits, %d generated, %d failed (of %d)",
        battle_card_cache_hits,
        len(deterministic_battle_cards) - battle_card_cache_hits - battle_card_llm_failures,
        battle_card_llm_failures,
        len(deterministic_battle_cards),
    )

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
    exploratory_persisted = False

    # Provenance for intelligence reports
    report_source_review_count = data_context.get("reviews_in_analysis_window")
    report_source_dist = json.dumps(
        {src: info["reviews"] for src, info in data_context.get("source_distribution", {}).items()}
    )

    displacement_report_id = None
    try:
        async with pool.transaction() as conn:
            for report_type, data in report_types:
                rid = await conn.fetchval(
                    """
                    INSERT INTO b2b_intelligence (
                        report_date, report_type, intelligence_data,
                        executive_summary, data_density, status, llm_model,
                        source_review_count, source_distribution
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')), LOWER(COALESCE(category_filter,'')), COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
                    DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                                  executive_summary = EXCLUDED.executive_summary,
                                  data_density = EXCLUDED.data_density,
                                  source_review_count = EXCLUDED.source_review_count,
                                  source_distribution = EXCLUDED.source_distribution,
                                  created_at = now()
                    RETURNING id
                    """,
                    today,
                    report_type,
                    json.dumps(data, default=str),
                    _exec_summaries.get(report_type, _fallback_summary),
                    data_density,
                    "published",
                    llm_model_id,
                    report_source_review_count,
                    report_source_dist,
                )
                if report_type == "displacement_report":
                    displacement_report_id = rid
    except Exception:
        logger.exception("Failed to store intelligence reports (rolled back)")

    # Persist battle cards (one row per vendor, using vendor_filter)
    battle_cards_persisted = 0
    for card in deterministic_battle_cards:
        vendor = card.get("vendor", "")
        if not vendor:
            continue
        try:
            await pool.execute(
                """
                INSERT INTO b2b_intelligence (
                    report_date, report_type, vendor_filter,
                    intelligence_data, executive_summary, data_density, status, llm_model,
                    source_review_count, source_distribution
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
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
                "battle_card",
                vendor,
                json.dumps(card, default=str),
                (
                    f"Battle card for {vendor}: "
                    f"score {card.get('churn_pressure_score', 0):.0f}, "
                    f"{len(card.get('vendor_weaknesses', []))} weaknesses, "
                    f"{len(card.get('competitor_differentiators', []))} competitors."
                ),
                data_density,
                "published",
                llm_model_id,
                report_source_review_count,
                report_source_dist,
            )
            battle_cards_persisted += 1
        except Exception:
            logger.exception("Failed to persist battle card for %s", vendor)
    if deterministic_battle_cards:
        logger.info("Persisted %d/%d battle cards",
                     battle_cards_persisted, len(deterministic_battle_cards))

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
                    executive_summary, data_density, status, llm_model,
                    source_review_count, source_distribution
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')), LOWER(COALESCE(category_filter,'')), COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
                DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                              executive_summary = EXCLUDED.executive_summary,
                              data_density = EXCLUDED.data_density,
                              source_review_count = EXCLUDED.source_review_count,
                              source_distribution = EXCLUDED.source_distribution,
                              created_at = now()
                """,
                today,
                "exploratory_overview",
                json.dumps(exploratory_data, default=str),
                exploratory_data.get("executive_summary", ""),
                json.dumps({**json.loads(data_density), "scope": "exploratory"}),
                "published",
                llm_model_id,
                report_source_review_count,
                report_source_dist,
            )
            exploratory_persisted = True
        except Exception:
            logger.exception("Failed to store exploratory_overview")

    # Persist displacement edges to first-class table
    displacement_edges_persisted = 0
    try:
        async with pool.transaction() as conn:
            for edge in deterministic_displacement_map:
                sample_ids = [
                    _uuid.UUID(rid) for rid in edge.get("sample_review_ids", [])
                    if rid
                ]
                await conn.execute(
                    """
                    INSERT INTO b2b_displacement_edges (
                        from_vendor, to_vendor, mention_count,
                        primary_driver, signal_strength, key_quote,
                        source_distribution, sample_review_ids,
                        confidence_score, computed_date, report_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::uuid[], $9, $10, $11)
                    ON CONFLICT (from_vendor, to_vendor, computed_date)
                    DO UPDATE SET
                        mention_count = EXCLUDED.mention_count,
                        primary_driver = EXCLUDED.primary_driver,
                        signal_strength = EXCLUDED.signal_strength,
                        key_quote = EXCLUDED.key_quote,
                        source_distribution = EXCLUDED.source_distribution,
                        sample_review_ids = EXCLUDED.sample_review_ids,
                        confidence_score = EXCLUDED.confidence_score,
                        report_id = EXCLUDED.report_id
                    """,
                    edge["from_vendor"],
                    edge["to_vendor"],
                    edge["mention_count"],
                    edge.get("primary_driver"),
                    edge.get("signal_strength"),
                    edge.get("key_quote"),
                    json.dumps(edge.get("source_distribution", {})),
                    sample_ids,
                    edge.get("confidence_score", 0),
                    today,
                    displacement_report_id,
                )
                displacement_edges_persisted += 1
    except Exception:
        displacement_edges_persisted = 0
        logger.exception("Failed to persist displacement edges")

    # Upsert per-vendor churn signals
    upsert_failures = await _upsert_churn_signals(
        pool, vendor_scores,
        neg_lookup, pain_lookup, competitor_lookup, feature_gap_lookup,
        price_lookup, dm_lookup, company_lookup, quote_lookup,
        budget_lookup, use_case_lookup, integration_lookup,
        sentiment_lookup, buyer_auth_lookup, timeline_lookup,
        keyword_spike_lookup,
        provenance_lookup=vendor_provenance,
        insider_lookup=insider_lookup,
        reasoning_lookup=reasoning_lookup,
    )

    # Persist company signals to first-class table
    company_signals_persisted = 0
    try:
        async with pool.transaction() as conn:
            for hi in high_intent:
                review_id = None
                if hi.get("review_id"):
                    try:
                        review_id = _uuid.UUID(hi["review_id"])
                    except (ValueError, TypeError):
                        pass
                # Confidence for company signal: source quality + data completeness
                _src = hi.get("source", "")
                _src_dist = {_src: 1} if _src else {}
                _cs_conf = _compute_evidence_confidence(1, _src_dist)
                # Boost for richer data (decision_maker, buying_stage, seat_count)
                _filled = sum(1 for f in (hi.get("decision_maker"), hi.get("buying_stage"), hi.get("seat_count")) if f is not None)
                _cs_conf = round(min(_cs_conf + _filled * 0.05, 1.0), 2)

                await conn.execute(
                    """
                    INSERT INTO b2b_company_signals (
                        company_name, vendor_name, urgency_score,
                        pain_category, buyer_role, decision_maker,
                        seat_count, contract_end, buying_stage,
                        review_id, source, confidence_score, last_seen_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now())
                    ON CONFLICT (company_name, vendor_name)
                    DO UPDATE SET
                        urgency_score = GREATEST(b2b_company_signals.urgency_score, EXCLUDED.urgency_score),
                        pain_category = COALESCE(EXCLUDED.pain_category, b2b_company_signals.pain_category),
                        buyer_role = COALESCE(EXCLUDED.buyer_role, b2b_company_signals.buyer_role),
                        decision_maker = COALESCE(EXCLUDED.decision_maker, b2b_company_signals.decision_maker),
                        seat_count = COALESCE(EXCLUDED.seat_count, b2b_company_signals.seat_count),
                        contract_end = COALESCE(EXCLUDED.contract_end, b2b_company_signals.contract_end),
                        buying_stage = COALESCE(EXCLUDED.buying_stage, b2b_company_signals.buying_stage),
                        review_id = COALESCE(EXCLUDED.review_id, b2b_company_signals.review_id),
                        source = COALESCE(EXCLUDED.source, b2b_company_signals.source),
                        confidence_score = GREATEST(b2b_company_signals.confidence_score, EXCLUDED.confidence_score),
                        last_seen_at = EXCLUDED.last_seen_at
                    """,
                    normalize_company_name(hi.get("company", "")),
                    hi.get("vendor", ""),
                    hi.get("urgency"),
                    hi.get("pain"),
                    hi.get("role_level"),
                    hi.get("decision_maker"),
                    hi.get("seat_count"),
                    hi.get("contract_end"),
                    hi.get("buying_stage"),
                    review_id,
                    hi.get("source"),
                    _cs_conf,
                )
                company_signals_persisted += 1
    except Exception:
        company_signals_persisted = 0
        logger.exception("Failed to persist company signals")

    # Persist vendor pain points to first-class table
    pain_points_persisted = 0
    try:
        async with pool.transaction() as conn:
            for (vendor, pain_cat), prov in pain_provenance.items():
                sample_ids = [
                    _uuid.UUID(rid) for rid in prov.get("sample_review_ids", [])
                    if rid
                ]
                confidence = _compute_evidence_confidence(
                    prov["mention_count"],
                    prov.get("source_distribution", {}),
                )
                await conn.execute(
                    """
                    INSERT INTO b2b_vendor_pain_points (
                        vendor_name, pain_category, mention_count,
                        primary_count, secondary_count, minor_count,
                        avg_urgency, avg_rating,
                        source_distribution, sample_review_ids,
                        confidence_score, last_seen_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::uuid[], $11, NOW())
                    ON CONFLICT (vendor_name, pain_category) DO UPDATE SET
                        mention_count = EXCLUDED.mention_count,
                        primary_count = EXCLUDED.primary_count,
                        secondary_count = EXCLUDED.secondary_count,
                        minor_count = EXCLUDED.minor_count,
                        avg_urgency = EXCLUDED.avg_urgency,
                        avg_rating = EXCLUDED.avg_rating,
                        source_distribution = EXCLUDED.source_distribution,
                        sample_review_ids = EXCLUDED.sample_review_ids,
                        confidence_score = EXCLUDED.confidence_score,
                        last_seen_at = EXCLUDED.last_seen_at
                    """,
                    vendor,
                    pain_cat,
                    prov["mention_count"],
                    prov.get("primary_count", 0),
                    prov.get("secondary_count", 0),
                    prov.get("minor_count", 0),
                    prov.get("avg_urgency"),
                    prov.get("avg_rating"),
                    json.dumps(prov.get("source_distribution", {})),
                    sample_ids,
                    confidence,
                )
                pain_points_persisted += 1
    except Exception:
        pain_points_persisted = 0
        logger.exception("Failed to persist vendor pain points")

    # Persist vendor use cases to first-class table
    use_cases_persisted = 0
    try:
        async with pool.transaction() as conn:
            for (vendor, use_case_name), prov in use_case_provenance.items():
                sample_ids = [
                    _uuid.UUID(rid) for rid in prov.get("sample_review_ids", [])
                    if rid
                ]
                confidence = _compute_evidence_confidence(
                    prov["mention_count"],
                    prov.get("source_distribution", {}),
                )
                await conn.execute(
                    """
                    INSERT INTO b2b_vendor_use_cases (
                        vendor_name, use_case_name, mention_count,
                        avg_urgency, lock_in_distribution,
                        source_distribution, sample_review_ids,
                        confidence_score, last_seen_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::uuid[], $8, NOW())
                    ON CONFLICT (vendor_name, use_case_name) DO UPDATE SET
                        mention_count = EXCLUDED.mention_count,
                        avg_urgency = EXCLUDED.avg_urgency,
                        lock_in_distribution = EXCLUDED.lock_in_distribution,
                        source_distribution = EXCLUDED.source_distribution,
                        sample_review_ids = EXCLUDED.sample_review_ids,
                        confidence_score = EXCLUDED.confidence_score,
                        last_seen_at = EXCLUDED.last_seen_at
                    """,
                    vendor,
                    use_case_name,
                    prov["mention_count"],
                    prov.get("avg_urgency"),
                    json.dumps(prov.get("lock_in_distribution", {})),
                    json.dumps(prov.get("source_distribution", {})),
                    sample_ids,
                    confidence,
                )
                use_cases_persisted += 1
    except Exception:
        use_cases_persisted = 0
        logger.exception("Failed to persist vendor use cases")

    # Persist vendor integrations to first-class table
    integrations_persisted = 0
    try:
        async with pool.transaction() as conn:
            for (vendor, integration_name), prov in integration_provenance.items():
                sample_ids = [
                    _uuid.UUID(rid) for rid in prov.get("sample_review_ids", [])
                    if rid
                ]
                confidence = _compute_evidence_confidence(
                    prov["mention_count"],
                    prov.get("source_distribution", {}),
                )
                await conn.execute(
                    """
                    INSERT INTO b2b_vendor_integrations (
                        vendor_name, integration_name, mention_count,
                        source_distribution, sample_review_ids,
                        confidence_score, last_seen_at
                    ) VALUES ($1, $2, $3, $4, $5::uuid[], $6, NOW())
                    ON CONFLICT (vendor_name, integration_name) DO UPDATE SET
                        mention_count = EXCLUDED.mention_count,
                        source_distribution = EXCLUDED.source_distribution,
                        sample_review_ids = EXCLUDED.sample_review_ids,
                        confidence_score = EXCLUDED.confidence_score,
                        last_seen_at = EXCLUDED.last_seen_at
                    """,
                    vendor,
                    integration_name,
                    prov["mention_count"],
                    json.dumps(prov.get("source_distribution", {})),
                    sample_ids,
                    confidence,
                )
                integrations_persisted += 1
    except Exception:
        integrations_persisted = 0
        logger.exception("Failed to persist vendor integrations")

    # Persist buyer profiles to first-class table
    buyer_profiles_persisted = 0
    try:
        async with pool.transaction() as conn:
            for (vendor, role_type, buying_stage), prov in buyer_profile_provenance.items():
                sample_ids = [
                    _uuid.UUID(rid) for rid in prov.get("sample_review_ids", [])
                    if rid
                ]
                confidence = _compute_evidence_confidence(
                    prov["review_count"],
                    prov.get("source_distribution", {}),
                )
                await conn.execute(
                    """
                    INSERT INTO b2b_vendor_buyer_profiles (
                        vendor_name, role_type, buying_stage,
                        review_count, dm_count, avg_urgency,
                        source_distribution, sample_review_ids,
                        confidence_score, last_seen_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::uuid[], $9, NOW())
                    ON CONFLICT (vendor_name, role_type, buying_stage) DO UPDATE SET
                        review_count = EXCLUDED.review_count,
                        dm_count = EXCLUDED.dm_count,
                        avg_urgency = EXCLUDED.avg_urgency,
                        source_distribution = EXCLUDED.source_distribution,
                        sample_review_ids = EXCLUDED.sample_review_ids,
                        confidence_score = EXCLUDED.confidence_score,
                        last_seen_at = EXCLUDED.last_seen_at
                    """,
                    vendor,
                    role_type,
                    buying_stage,
                    prov["review_count"],
                    prov["dm_count"],
                    prov.get("avg_urgency"),
                    json.dumps(prov.get("source_distribution", {})),
                    sample_ids,
                    confidence,
                )
                buyer_profiles_persisted += 1
    except Exception:
        buyer_profiles_persisted = 0
        logger.exception("Failed to persist buyer profiles")

    # Persist vendor health snapshots + detect change events
    snapshots_persisted = 0
    change_events_detected = 0
    if cfg.snapshot_enabled:
        try:
            snapshots_persisted = await _persist_vendor_snapshots(
                pool, vendor_scores, pain_lookup, competitor_lookup,
                high_intent, today,
                price_lookup=price_lookup, dm_lookup=dm_lookup,
            )
            if cfg.change_detection_enabled:
                change_events_detected = await _detect_change_events(
                    pool, vendor_scores, pain_lookup, competitor_lookup, today,
                    price_lookup=price_lookup, dm_lookup=dm_lookup,
                    temporal_lookup=_temporal_lookup or None,
                )
        except Exception:
            logger.exception("Failed to persist vendor snapshots / change events")

    # Send ntfy notification
    await _send_notification(task, parsed, high_intent)

    # Emit reasoning events (no-op when reasoning disabled)
    await _emit_reasoning_events(parsed, high_intent, vendor_scores)

    response = {
        "_skip_synthesis": "B2B churn intelligence complete",
        "date": str(today),
        "vendors_analyzed": len(vendor_scores),
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "report_types": len(report_types) + (1 if exploratory_persisted else 0),
        "fetcher_failures": fetcher_failures,
        "upsert_failures": upsert_failures,
        "displacement_edges_persisted": displacement_edges_persisted,
        "company_signals_persisted": company_signals_persisted,
        "pain_points_persisted": pain_points_persisted,
        "use_cases_persisted": use_cases_persisted,
        "integrations_persisted": integrations_persisted,
        "buyer_profiles_persisted": buyer_profiles_persisted,
        "snapshots_persisted": snapshots_persisted,
        "change_events_detected": change_events_detected,
    }
    tracer.end_span(
        span,
        status="completed",
        output_data=response,
        input_tokens=llm_usage.get("input_tokens"),
        output_tokens=llm_usage.get("output_tokens"),
        metadata={
            "reasoning": build_reasoning_trace_context(
                decision={"report_types": len(report_types) + (1 if exploratory_persisted else 0)},
                evidence={
                    "fetcher_failures": fetcher_failures,
                    "upsert_failures": upsert_failures,
                    "competitive_flows": len(competitive_disp),
                },
                rationale=parsed.get("executive_summary"),
            ),
        },
    )

    # Flush metacognition counters to DB after each intelligence run
    try:
        from atlas_brain.reasoning import get_stratified_reasoner
        reasoner = get_stratified_reasoner()
        if reasoner and reasoner._meta:
            await reasoner._meta.flush()
    except Exception:
        logger.debug("Metacognition flush skipped", exc_info=True)

    return response


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


async def _fetch_vendor_provenance(pool, window_days: int) -> dict[str, dict]:
    """Per-vendor provenance: source distribution, sample review IDs, and review window.

    Returns {vendor_name: {"source_distribution": {...}, "sample_review_ids": [...],
             "review_window_start": dt, "review_window_end": dt}}.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)

    # Source distribution per vendor
    dist_rows = await pool.fetch(
        f"""
        SELECT vendor_name, source, count(*) AS cnt
        FROM b2b_reviews
        WHERE {filters}
        GROUP BY vendor_name, source
        """,
        window_days,
        sources,
    )
    dist: dict[str, dict[str, int]] = {}
    for r in dist_rows:
        dist.setdefault(r["vendor_name"], {})[r["source"]] = r["cnt"]

    # Sample review IDs (top 50 by urgency) + window per vendor
    sample_rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            (ARRAY_AGG(id ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST))[1:50]
                AS sample_ids,
            MIN(enriched_at) AS window_start,
            MAX(enriched_at) AS window_end
        FROM b2b_reviews
        WHERE {filters}
        GROUP BY vendor_name
        """,
        window_days,
        sources,
    )

    result: dict[str, dict] = {}
    for r in sample_rows:
        vendor = r["vendor_name"]
        result[vendor] = {
            "source_distribution": dist.get(vendor, {}),
            "sample_review_ids": r["sample_ids"] or [],
            "review_window_start": r["window_start"],
            "review_window_end": r["window_end"],
        }
    # Fill in vendors that appear in dist but not in sample (shouldn't happen, but safe)
    for vendor in dist:
        if vendor not in result:
            result[vendor] = {
                "source_distribution": dist[vendor],
                "sample_review_ids": [],
                "review_window_start": None,
                "review_window_end": None,
            }
    return result


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
        SELECT id AS review_id, source,
            reviewer_company, vendor_name, product_category,
            enrichment->'reviewer_context'->>'role_level' AS role_level,
            (enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
            (enrichment->>'urgency_score')::numeric AS urgency,
            enrichment->>'pain_category' AS pain,
            enrichment->'competitors_mentioned' AS alternatives,
            enrichment->'quotable_phrases' AS quotes,
            enrichment->'contract_context'->>'contract_value_signal' AS value_signal,
            enrichment->'budget_signals'->>'seat_count' AS seat_count,
            enrichment->'timeline'->>'contract_end' AS contract_end,
            enrichment->'buyer_authority'->>'buying_stage' AS buying_stage
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
        # Parse seat_count safely
        seat_count = None
        if r["seat_count"]:
            try:
                seat_count = int(r["seat_count"])
            except (ValueError, TypeError):
                pass
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
            "review_id": str(r["review_id"]) if r["review_id"] else None,
            "source": r["source"],
            "seat_count": seat_count,
            "contract_end": r["contract_end"],
            "buying_stage": r["buying_stage"],
        })
    return results


async def _fetch_competitive_displacement(pool, window_days: int) -> list[dict[str, Any]]:
    """Competitive displacement flows — filtered to real displacement evidence only.

    Uses evidence_type (new schema) with COALESCE fallback to context (legacy).
    Excludes reverse_flow, neutral_mention, and low-confidence implied_preference.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            comp.value->>'name' AS competitor,
            COALESCE(
                comp.value->>'evidence_type',
                CASE comp.value->>'context'
                    WHEN 'switched_to' THEN 'explicit_switch'
                    WHEN 'considering' THEN 'active_evaluation'
                    WHEN 'compared' THEN 'implied_preference'
                    WHEN 'switched_from' THEN 'reverse_flow'
                    ELSE 'neutral_mention'
                END
            ) AS evidence_type,
            COALESCE(
                comp.value->>'displacement_confidence', 'low'
            ) AS displacement_confidence,
            comp.value->>'reason_category' AS reason_category,
            comp.value->>'context' AS direction,
            count(*) AS mention_count,
            array_agg(DISTINCT company_size_raw) FILTER (WHERE company_size_raw IS NOT NULL) AS sizes,
            array_agg(DISTINCT COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry'))
                FILTER (WHERE COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') IS NOT NULL) AS industries
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(enrichment->'competitors_mentioned') AS comp(value)
        WHERE {filters}
          AND COALESCE(
                comp.value->>'evidence_type',
                CASE comp.value->>'context'
                    WHEN 'switched_to' THEN 'explicit_switch'
                    WHEN 'considering' THEN 'active_evaluation'
                    WHEN 'compared' THEN 'implied_preference'
                    WHEN 'switched_from' THEN 'reverse_flow'
                    ELSE 'neutral_mention'
                END
              ) IN ('explicit_switch', 'active_evaluation', 'implied_preference')
          AND NOT (
              COALESCE(
                comp.value->>'evidence_type',
                CASE comp.value->>'context'
                    WHEN 'compared' THEN 'implied_preference'
                    ELSE 'neutral_mention'
                END
              ) = 'implied_preference'
              AND COALESCE(comp.value->>'displacement_confidence', 'low') = 'low'
          )
        GROUP BY vendor_name, comp.value->>'name',
                 evidence_type, displacement_confidence, reason_category,
                 comp.value->>'context'
        HAVING count(*) >= 2
        ORDER BY mention_count DESC
        """,
        window_days,
        sources,
    )
    # Post-process: canonicalize competitors, filter self-flows, re-aggregate
    MergeKey = tuple[str, str, str]  # (vendor, competitor, evidence_type)
    merged: dict[MergeKey, int] = {}
    merged_industries: dict[MergeKey, list[str]] = {}
    merged_sizes: dict[MergeKey, list[str]] = {}
    merged_reason_cats: dict[MergeKey, dict[str, int]] = {}
    for r in rows:
        canon = _canonicalize_competitor(r["competitor"] or "")
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        if canon and vendor and canon.lower() == vendor.lower():
            continue
        et = r["evidence_type"] or "implied_preference"
        key: MergeKey = (vendor, canon, et)
        merged[key] = merged.get(key, 0) + r["mention_count"]
        if r.get("industries"):
            merged_industries.setdefault(key, []).extend(r["industries"])
        if r.get("sizes"):
            merged_sizes.setdefault(key, []).extend(r["sizes"])
        rc = r.get("reason_category")
        if rc:
            cats = merged_reason_cats.setdefault(key, {})
            cats[rc] = cats.get(rc, 0) + r["mention_count"]

    # Re-apply HAVING count >= 2, sort by mention_count DESC
    results = []
    for k, cnt in merged.items():
        if cnt < 2:
            continue
        ind_list = merged_industries.get(k, [])
        sz_list = merged_sizes.get(k, [])
        results.append({
            "vendor": k[0],
            "competitor": k[1],
            "evidence_type": k[2],
            "mention_count": cnt,
            "reason_categories": merged_reason_cats.get(k, {}),
            "industries": sorted(set(i for i in ind_list if i and i != "unknown")),
            "company_sizes": sorted(set(s for s in sz_list if s)),
        })
    results.sort(key=lambda x: x["mention_count"], reverse=True)
    return results


async def _fetch_displacement_provenance(pool, window_days: int) -> dict[tuple[str, str], dict]:
    """Per-edge source distribution and sample review IDs for confidence scoring.

    Returns a dict keyed by ``(from_vendor, to_vendor)`` with:
      - ``source_distribution``: ``{source: count}``
      - ``sample_review_ids``:   list of UUID strings (top 20 by urgency)
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2, alias="r")
    rows = await pool.fetch(
        f"""
        SELECT r.vendor_name,
            comp.value->>'name' AS competitor,
            r.source,
            count(*) AS cnt,
            array_agg(r.id ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST)
                FILTER (WHERE r.id IS NOT NULL) AS review_ids
        FROM b2b_reviews r
        CROSS JOIN LATERAL jsonb_array_elements(r.enrichment->'competitors_mentioned') AS comp(value)
        WHERE {filters}
        GROUP BY r.vendor_name, comp.value->>'name', r.source
        HAVING count(*) >= 1
        ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )

    # Aggregate by canonicalized (from_vendor, to_vendor)
    result: dict[tuple[str, str], dict] = {}
    for r in rows:
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        competitor = _canonicalize_competitor(r["competitor"] or "")
        if not vendor or not competitor or vendor.lower() == competitor.lower():
            continue
        key = (vendor, competitor)
        if key not in result:
            result[key] = {"source_distribution": {}, "sample_review_ids": []}
        entry = result[key]
        source = r["source"] or "unknown"
        entry["source_distribution"][source] = entry["source_distribution"].get(source, 0) + r["cnt"]
        # Collect review IDs (cap at 20 per edge)
        rids = r["review_ids"] or []
        for rid in rids:
            if len(entry["sample_review_ids"]) < 20 and str(rid) not in entry["sample_review_ids"]:
                entry["sample_review_ids"].append(str(rid))

    return result


# ------------------------------------------------------------------
# Provenance fetchers for first-class entity tables (Sprint 2)
# ------------------------------------------------------------------


async def _fetch_pain_provenance(
    pool, window_days: int,
) -> dict[tuple[str, str], dict]:
    """Per-vendor/pain_category provenance for the b2b_vendor_pain_points table.

    Returns ``{(vendor, pain_category): {mention_count, primary_count,
    secondary_count, minor_count, avg_urgency, avg_rating,
    source_distribution, sample_review_ids}}``.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2, alias="r")

    # 1. Core counts + averages grouped by vendor, pain_category, source
    core_rows = await pool.fetch(
        f"""
        SELECT r.vendor_name,
            r.enrichment->>'pain_category' AS pain_category,
            r.source,
            count(*) AS cnt,
            avg((r.enrichment->>'urgency_score')::numeric) AS avg_urgency,
            avg(r.rating) AS avg_rating,
            array_agg(r.id ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST)
                FILTER (WHERE r.id IS NOT NULL) AS review_ids
        FROM b2b_reviews r
        WHERE {filters}
          AND r.enrichment->>'pain_category' IS NOT NULL
        GROUP BY r.vendor_name, r.enrichment->>'pain_category', r.source
        """,
        window_days,
        sources,
    )

    # 2. Severity breakdown from the pain_categories array
    severity_rows = await pool.fetch(
        f"""
        SELECT r.vendor_name,
            p.value->>'category' AS pain_category,
            p.value->>'severity' AS severity,
            count(*) AS cnt
        FROM b2b_reviews r
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(r.enrichment->'pain_categories', '[]'::jsonb)
        ) AS p(value)
        WHERE {filters}
          AND p.value->>'category' IS NOT NULL
        GROUP BY r.vendor_name, p.value->>'category', p.value->>'severity'
        """,
        window_days,
        sources,
    )

    result: dict[tuple[str, str], dict] = {}

    # Aggregate core rows
    for r in core_rows:
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        pain = r["pain_category"] or ""
        if not vendor or not pain:
            continue
        key = (vendor, pain)
        if key not in result:
            result[key] = {
                "mention_count": 0,
                "primary_count": 0,
                "secondary_count": 0,
                "minor_count": 0,
                "avg_urgency": 0.0,
                "avg_rating": 0.0,
                "source_distribution": {},
                "sample_review_ids": [],
                "_urgency_sum": 0.0,
                "_rating_sum": 0.0,
                "_total": 0,
            }
        entry = result[key]
        cnt = r["cnt"]
        entry["mention_count"] += cnt
        entry["_total"] += cnt
        entry["_urgency_sum"] += float(r["avg_urgency"] or 0) * cnt
        entry["_rating_sum"] += float(r["avg_rating"] or 0) * cnt
        source = r["source"] or "unknown"
        entry["source_distribution"][source] = (
            entry["source_distribution"].get(source, 0) + cnt
        )
        for rid in (r["review_ids"] or []):
            if len(entry["sample_review_ids"]) < 20 and str(rid) not in entry["sample_review_ids"]:
                entry["sample_review_ids"].append(str(rid))

    # Aggregate severity breakdown
    for r in severity_rows:
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        pain = r["pain_category"] or ""
        if not vendor or not pain:
            continue
        key = (vendor, pain)
        if key not in result:
            continue  # only augment keys from core query
        severity = (r["severity"] or "").lower()
        cnt = r["cnt"]
        if severity == "primary":
            result[key]["primary_count"] += cnt
        elif severity == "secondary":
            result[key]["secondary_count"] += cnt
        elif severity == "minor":
            result[key]["minor_count"] += cnt

    # Finalize averages and clean internal fields
    for entry in result.values():
        total = entry.pop("_total", 0) or 1
        entry["avg_urgency"] = round(entry.pop("_urgency_sum", 0) / total, 1)
        entry["avg_rating"] = round(entry.pop("_rating_sum", 0) / total, 2)

    return result


async def _fetch_use_case_provenance(
    pool, window_days: int,
) -> dict[tuple[str, str], dict]:
    """Per-vendor/module provenance for the b2b_vendor_use_cases table.

    Returns ``{(vendor, module_name): {mention_count, avg_urgency,
    lock_in_distribution, source_distribution, sample_review_ids}}``.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2, alias="r")

    # 1. Module mentions with source + urgency + sample IDs
    module_rows = await pool.fetch(
        f"""
        SELECT r.vendor_name,
            mod.value #>> '{{}}' AS module_name,
            r.source,
            count(*) AS cnt,
            avg((r.enrichment->>'urgency_score')::numeric) AS avg_urgency,
            array_agg(r.id ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST)
                FILTER (WHERE r.id IS NOT NULL) AS review_ids
        FROM b2b_reviews r
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(r.enrichment->'use_case'->'modules_mentioned', '[]'::jsonb)
        ) AS mod(value)
        WHERE {filters}
        GROUP BY r.vendor_name, mod.value #>> '{{}}', r.source
        """,
        window_days,
        sources,
    )

    # 2. Lock-in distribution per vendor/module
    lock_rows = await pool.fetch(
        f"""
        SELECT r.vendor_name,
            mod.value #>> '{{}}' AS module_name,
            r.enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
            count(*) AS cnt
        FROM b2b_reviews r
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(r.enrichment->'use_case'->'modules_mentioned', '[]'::jsonb)
        ) AS mod(value)
        WHERE {filters}
          AND r.enrichment->'use_case'->>'lock_in_level' IS NOT NULL
        GROUP BY r.vendor_name, mod.value #>> '{{}}', r.enrichment->'use_case'->>'lock_in_level'
        """,
        window_days,
        sources,
    )

    result: dict[tuple[str, str], dict] = {}

    for r in module_rows:
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        module = r["module_name"] or ""
        if not vendor or not module:
            continue
        key = (vendor, module)
        if key not in result:
            result[key] = {
                "mention_count": 0,
                "avg_urgency": 0.0,
                "lock_in_distribution": {},
                "source_distribution": {},
                "sample_review_ids": [],
                "_urgency_sum": 0.0,
                "_total": 0,
            }
        entry = result[key]
        cnt = r["cnt"]
        entry["mention_count"] += cnt
        entry["_total"] += cnt
        entry["_urgency_sum"] += float(r["avg_urgency"] or 0) * cnt
        source = r["source"] or "unknown"
        entry["source_distribution"][source] = (
            entry["source_distribution"].get(source, 0) + cnt
        )
        for rid in (r["review_ids"] or []):
            if len(entry["sample_review_ids"]) < 20 and str(rid) not in entry["sample_review_ids"]:
                entry["sample_review_ids"].append(str(rid))

    for r in lock_rows:
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        module = r["module_name"] or ""
        key = (vendor, module)
        if key not in result:
            continue
        level = r["lock_in_level"] or "unknown"
        result[key]["lock_in_distribution"][level] = (
            result[key]["lock_in_distribution"].get(level, 0) + r["cnt"]
        )

    for entry in result.values():
        total = entry.pop("_total", 0) or 1
        entry["avg_urgency"] = round(entry.pop("_urgency_sum", 0) / total, 1)

    return result


async def _fetch_integration_provenance(
    pool, window_days: int,
) -> dict[tuple[str, str], dict]:
    """Per-vendor/integration provenance for the b2b_vendor_integrations table.

    Returns ``{(vendor, tool_name): {mention_count, source_distribution,
    sample_review_ids}}``.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2, alias="r")

    rows = await pool.fetch(
        f"""
        SELECT r.vendor_name,
            tool.value #>> '{{}}' AS tool_name,
            r.source,
            count(*) AS cnt,
            array_agg(r.id ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST)
                FILTER (WHERE r.id IS NOT NULL) AS review_ids
        FROM b2b_reviews r
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(r.enrichment->'use_case'->'integration_stack', '[]'::jsonb)
        ) AS tool(value)
        WHERE {filters}
        GROUP BY r.vendor_name, tool.value #>> '{{}}', r.source
        """,
        window_days,
        sources,
    )

    result: dict[tuple[str, str], dict] = {}
    for r in rows:
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        tool = r["tool_name"] or ""
        if not vendor or not tool:
            continue
        key = (vendor, tool)
        if key not in result:
            result[key] = {
                "mention_count": 0,
                "source_distribution": {},
                "sample_review_ids": [],
            }
        entry = result[key]
        cnt = r["cnt"]
        entry["mention_count"] += cnt
        source = r["source"] or "unknown"
        entry["source_distribution"][source] = (
            entry["source_distribution"].get(source, 0) + cnt
        )
        for rid in (r["review_ids"] or []):
            if len(entry["sample_review_ids"]) < 20 and str(rid) not in entry["sample_review_ids"]:
                entry["sample_review_ids"].append(str(rid))

    return result


async def _fetch_buyer_profile_provenance(
    pool, window_days: int,
) -> dict[tuple[str, str, str], dict]:
    """Per-vendor/role_type/buying_stage provenance for buyer profiles.

    Returns ``{(vendor, role_type, buying_stage): {review_count, dm_count,
    avg_urgency, source_distribution, sample_review_ids}}``.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)

    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            COALESCE(enrichment->'buyer_authority'->>'role_type', 'unknown') AS role_type,
            COALESCE(enrichment->'buyer_authority'->>'buying_stage', 'unknown') AS buying_stage,
            source,
            count(*) AS cnt,
            count(*) FILTER (
                WHERE (enrichment->'buyer_authority'->>'decision_maker')::boolean IS TRUE
            ) AS dm_cnt,
            avg((enrichment->>'urgency_score')::numeric) AS avg_urg,
            array_agg(id ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST)
                FILTER (WHERE id IS NOT NULL) AS review_ids
        FROM b2b_reviews
        WHERE {filters}
          AND enrichment->'buyer_authority' IS NOT NULL
          AND enrichment->'buyer_authority' != 'null'::jsonb
        GROUP BY vendor_name,
            enrichment->'buyer_authority'->>'role_type',
            enrichment->'buyer_authority'->>'buying_stage',
            source
        """,
        window_days,
        sources,
    )

    result: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        vendor = _canonicalize_vendor(r["vendor_name"] or "")
        role = r["role_type"] or "unknown"
        stage = r["buying_stage"] or "unknown"
        if not vendor:
            continue
        key = (vendor, role, stage)
        if key not in result:
            result[key] = {
                "review_count": 0,
                "dm_count": 0,
                "avg_urgency": 0.0,
                "source_distribution": {},
                "sample_review_ids": [],
                "_weighted_urgency_sum": 0.0,
            }
        entry = result[key]
        cnt = r["cnt"]
        entry["review_count"] += cnt
        entry["dm_count"] += r["dm_cnt"]
        entry["_weighted_urgency_sum"] += float(r["avg_urg"] or 0) * cnt
        source = r["source"] or "unknown"
        entry["source_distribution"][source] = (
            entry["source_distribution"].get(source, 0) + cnt
        )
        for rid in (r["review_ids"] or []):
            if len(entry["sample_review_ids"]) < 20 and str(rid) not in entry["sample_review_ids"]:
                entry["sample_review_ids"].append(str(rid))

    # Compute weighted avg urgency
    for entry in result.values():
        total = entry["review_count"]
        entry["avg_urgency"] = round(entry.pop("_weighted_urgency_sum") / total, 1) if total else 0.0

    return result


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
                'pain', enrichment->>'pain_category',
                'title', reviewer_title,
                'company_size', company_size_raw,
                'industry', COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry')
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
    """Top quotable phrases per vendor (highest urgency, deduplicated).

    Each quote is a dict with 'quote', 'urgency', and 'review_id' for provenance.
    """
    sources = _executive_source_list()
    filters = _eligible_review_filters(window_param=1, source_param=3)
    rows = await pool.fetch(
        f"""
        WITH ranked_quotes AS (
            SELECT vendor_name, id AS review_id, phrase.value AS quote,
                (enrichment->>'urgency_score')::numeric AS urgency,
                reviewer_company, reviewer_title, company_size_raw,
                COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry,
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
        SELECT vendor_name,
            jsonb_agg(
                jsonb_build_object(
                    'quote', quote,
                    'urgency', urgency,
                    'review_id', review_id,
                    'company', reviewer_company,
                    'title', reviewer_title,
                    'company_size', company_size_raw,
                    'industry', industry
                ) ORDER BY urgency DESC
            ) AS quotes
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


async def _fetch_insider_aggregates(pool, window_days: int) -> list[dict[str, Any]]:
    """Aggregate insider account signals per vendor.

    Reads b2b_reviews WHERE content_type = 'insider_account' and extracts:
    - signal count
    - org health summary (mode of bureaucracy_level / leadership_quality / innovation_climate)
    - talent drain rate (fraction mentioning departures)
    - top quotable phrases from high-urgency insider posts
    """
    rows = await pool.fetch(
        """
        SELECT
            vendor_name,
            COUNT(DISTINCT id)::int AS signal_count,
            ROUND(
                COUNT(DISTINCT CASE WHEN enrichment->'insider_signals'->'talent_drain'->>'departures_mentioned' = 'true'
                         THEN id END)::numeric
                / NULLIF(COUNT(DISTINCT id), 0)::numeric,
                4
            ) AS talent_drain_rate,
            jsonb_agg(DISTINCT
                enrichment->'insider_signals'->'org_health'
            ) FILTER (WHERE enrichment->'insider_signals'->'org_health' IS NOT NULL) AS org_health_array,
            jsonb_agg(
                jsonb_build_object(
                    'quote', ph.value,
                    'urgency', (enrichment->>'urgency_score')::numeric,
                    'review_id', id::text
                )
                ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            ) FILTER (WHERE ph.value IS NOT NULL) AS quotable_phrases
        FROM b2b_reviews
        LEFT JOIN LATERAL jsonb_array_elements_text(
            COALESCE(enrichment->'quotable_phrases', '[]'::jsonb)
        ) AS ph(value) ON true
        WHERE content_type = 'insider_account'
          AND enrichment_status = 'enriched'
          AND imported_at > NOW() - make_interval(days => $1)
        GROUP BY vendor_name
        """,
        window_days,
    )
    return list(rows)


def _build_insider_lookup(rows: list[Any]) -> dict[str, dict]:
    """Build vendor → insider aggregate dict from _fetch_insider_aggregates rows."""
    result: dict[str, dict] = {}
    for r in rows:
        vendor = r["vendor_name"]
        org_health_array = _safe_json(r["org_health_array"]) if r["org_health_array"] else []
        quotable_raw = _safe_json(r["quotable_phrases"]) if r["quotable_phrases"] else []

        # Mode of org health fields across all insider posts
        bureaucracy = Counter(
            h.get("bureaucracy_level") for h in org_health_array if h and h.get("bureaucracy_level")
        )
        leadership = Counter(
            h.get("leadership_quality") for h in org_health_array if h and h.get("leadership_quality")
        )
        innovation = Counter(
            h.get("innovation_climate") for h in org_health_array if h and h.get("innovation_climate")
        )

        # Flatten all culture_indicators
        culture_indicators: list[str] = []
        for h in org_health_array:
            if h and isinstance(h.get("culture_indicators"), list):
                culture_indicators.extend(h["culture_indicators"])
        top_culture = [item for item, _ in Counter(culture_indicators).most_common(10)]

        org_health_summary = {
            "bureaucracy_level": bureaucracy.most_common(1)[0][0] if bureaucracy else "unknown",
            "leadership_quality": leadership.most_common(1)[0][0] if leadership else "unknown",
            "innovation_climate": innovation.most_common(1)[0][0] if innovation else "unknown",
            "culture_indicators": top_culture,
        }

        # Keep top 5 quotable phrases, deduplicate
        seen_quotes: set[str] = set()
        quotable_evidence = []
        for q in quotable_raw:
            if isinstance(q, dict) and q.get("quote") and q["quote"] not in seen_quotes:
                seen_quotes.add(q["quote"])
                quotable_evidence.append(q)
                if len(quotable_evidence) >= 5:
                    break

        result[vendor] = {
            "signal_count": r["signal_count"] or 0,
            "org_health_summary": org_health_summary,
            "talent_drain_rate": float(r["talent_drain_rate"]) if r["talent_drain_rate"] is not None else None,
            "quotable_evidence": quotable_evidence,
        }
    return result


async def _fetch_product_profiles(pool) -> list[dict[str, Any]]:
    """Fetch pre-computed product profiles for battle card enrichment."""
    rows = await pool.fetch("""
        SELECT vendor_name, product_category, strengths, weaknesses,
               pain_addressed, commonly_compared_to, commonly_switched_from,
               total_reviews_analyzed, avg_rating, recommend_rate,
               primary_use_cases, typical_company_size, typical_industries,
               top_integrations, profile_summary
        FROM b2b_product_profiles
        ORDER BY total_reviews_analyzed DESC
    """)
    return [dict(r) for r in rows]


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
            (enrichment->>'urgency_score')::numeric AS urgency,
            reviewer_title, company_size_raw,
            COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
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
            "title": r["reviewer_title"],
            "company_size": r["company_size_raw"],
            "industry": r["industry"],
        }
        for r in rows
    ]


async def _fetch_competitor_reasons(pool, window_days: int) -> list[dict[str, Any]]:
    """Top reasons per vendor/competitor pair — prefers structured reason_category."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        WITH ranked_reasons AS (
            SELECT vendor_name,
                comp.value->>'name' AS competitor,
                comp.value->>'context' AS direction,
                COALESCE(comp.value->>'reason_category', comp.value->>'reason') AS reason,
                comp.value->>'reason_category' AS reason_category,
                comp.value->>'reason_detail' AS reason_detail,
                count(*) AS mention_count,
                ROW_NUMBER() OVER (
                    PARTITION BY vendor_name, comp.value->>'name'
                    ORDER BY count(*) DESC
                ) AS rn
            FROM b2b_reviews
            CROSS JOIN LATERAL jsonb_array_elements(enrichment->'competitors_mentioned') AS comp(value)
                        WHERE {filters}
              AND COALESCE(comp.value->>'reason_category', comp.value->>'reason') IS NOT NULL
            GROUP BY vendor_name, comp.value->>'name', comp.value->>'context',
                     reason, reason_category, reason_detail
        )
        SELECT vendor_name, competitor, direction, reason, reason_category, reason_detail, mention_count
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
            "reason_category": r["reason_category"],
            "reason_detail": r["reason_detail"],
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
    """Merge rows with same (vendor, competitor), preserving evidence breakdown."""
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for row in competitive_disp:
        key = (row.get("vendor", ""), row.get("competitor", ""))
        if key not in agg:
            agg[key] = {
                "mention_count": 0,
                "explicit_switches": 0,
                "active_evaluations": 0,
                "implied_preferences": 0,
                "reason_categories": {},
            }
        entry = agg[key]
        cnt = int(row.get("mention_count") or 0)
        entry["mention_count"] += cnt
        et = row.get("evidence_type", "implied_preference")
        if et == "explicit_switch":
            entry["explicit_switches"] += cnt
        elif et == "active_evaluation":
            entry["active_evaluations"] += cnt
        else:
            entry["implied_preferences"] += cnt
        # Merge reason_categories from this row
        for rc, rc_cnt in (row.get("reason_categories") or {}).items():
            entry["reason_categories"][rc] = entry["reason_categories"].get(rc, 0) + rc_cnt

    results = []
    for (v, c), data in sorted(agg.items(), key=lambda x: x[1]["mention_count"], reverse=True):
        results.append({
            "vendor": v,
            "competitor": c,
            "mention_count": data["mention_count"],
            "explicit_switches": data["explicit_switches"],
            "active_evaluations": data["active_evaluations"],
            "implied_preferences": data["implied_preferences"],
            "reason_categories": data["reason_categories"],
        })
    return results


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
            "title": row.get("title"),
            "company_size": row.get("company_size"),
            "industry": row.get("industry"),
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
    quote_lookup: dict[str, list],
    budget_lookup: dict[str, dict] | None = None,
    use_case_lookup: dict[str, list[dict]] | None = None,
    integration_lookup: dict[str, list[dict]] | None = None,
    sentiment_lookup: dict[str, dict[str, int]] | None = None,
    buyer_auth_lookup: dict[str, dict] | None = None,
    timeline_lookup: dict[str, list[dict]] | None = None,
    keyword_spike_lookup: dict[str, dict] | None = None,
    provenance_lookup: dict[str, dict] | None = None,
    insider_lookup: dict[str, dict] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
) -> int:
    """Upsert b2b_churn_signals (33 columns incl. provenance + insider + reasoning). Returns failure count."""
    now = datetime.now(timezone.utc)
    budget_lookup = budget_lookup or {}
    use_case_lookup = use_case_lookup or {}
    integration_lookup = integration_lookup or {}
    sentiment_lookup = sentiment_lookup or {}
    buyer_auth_lookup = buyer_auth_lookup or {}
    timeline_lookup = timeline_lookup or {}
    keyword_spike_lookup = keyword_spike_lookup or {}
    reasoning_lookup = reasoning_lookup or {}
    provenance_lookup = provenance_lookup or {}
    insider_lookup = insider_lookup or {}
    failures = 0

    for vs in vendor_scores:
        vendor = vs["vendor_name"]
        category = vs.get("product_category")

        total = vs["total_reviews"]
        recommend_yes = vs.get("recommend_yes", 0)
        recommend_no = vs.get("recommend_no", 0)
        nps = ((recommend_yes - recommend_no) / total * 100) if total > 0 else None

        prov = provenance_lookup.get(vendor, {})
        insider = insider_lookup.get(vendor, {})

        try:
            kw_data = keyword_spike_lookup.get(vendor, {})
            src_dist = prov.get("source_distribution", {})
            signal_confidence = _compute_evidence_confidence(total, src_dist)
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
                    source_distribution, sample_review_ids,
                    review_window_start, review_window_end,
                    confidence_score,
                    insider_signal_count, insider_org_health_summary,
                    insider_talent_drain_rate, insider_quotable_evidence,
                    archetype, archetype_confidence,
                    reasoning_mode, falsification_conditions,
                    last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                          $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                          $22, $23, $24, $25, $26, $27, $28, $29,
                          $30, $31, $32, $33,
                          $34, $35, $36, $37, $38)
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
                    source_distribution = EXCLUDED.source_distribution,
                    sample_review_ids = EXCLUDED.sample_review_ids,
                    review_window_start = EXCLUDED.review_window_start,
                    review_window_end = EXCLUDED.review_window_end,
                    confidence_score = EXCLUDED.confidence_score,
                    insider_signal_count = EXCLUDED.insider_signal_count,
                    insider_org_health_summary = EXCLUDED.insider_org_health_summary,
                    insider_talent_drain_rate = EXCLUDED.insider_talent_drain_rate,
                    insider_quotable_evidence = EXCLUDED.insider_quotable_evidence,
                    archetype = EXCLUDED.archetype,
                    archetype_confidence = EXCLUDED.archetype_confidence,
                    reasoning_mode = EXCLUDED.reasoning_mode,
                    falsification_conditions = EXCLUDED.falsification_conditions,
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
                json.dumps(src_dist),
                prov.get("sample_review_ids", []),
                prov.get("review_window_start"),
                prov.get("review_window_end"),
                signal_confidence,
                # Insider aggregate columns (migration 133)
                insider.get("signal_count", 0),
                json.dumps(insider.get("org_health_summary", {})),
                insider.get("talent_drain_rate"),
                json.dumps(insider.get("quotable_evidence", [])[:5]),
                # Reasoning columns (migration 139)
                reasoning_lookup.get(vendor, {}).get("archetype"),
                reasoning_lookup.get(vendor, {}).get("confidence"),
                reasoning_lookup.get(vendor, {}).get("mode"),
                json.dumps(reasoning_lookup.get(vendor, {}).get("falsification_conditions", [])) if reasoning_lookup.get(vendor) else None,
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
               r.enrichment->'sentiment_trajectory'->>'direction' AS sentiment_direction,
               r.reviewer_title, r.company_size_raw,
               COALESCE(r.reviewer_industry, r.enrichment->'reviewer_context'->>'industry') AS industry
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
            ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')), LOWER(COALESCE(category_filter,'')), COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                          executive_summary = EXCLUDED.executive_summary,
                          data_density = EXCLUDED.data_density,
                          created_at = now()
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
               r.enrichment->'feature_gaps' AS feature_gaps,
               r.reviewer_title, r.company_size_raw,
               COALESCE(r.reviewer_industry, r.enrichment->'reviewer_context'->>'industry') AS industry
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


def _switching_triggers(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract common reasons customers leave a vendor, grouped by competitor.

    Correlates pain categories with competitor mentions across review signals
    to identify the primary driver behind each competitive displacement.
    """
    pain_by_competitor: dict[str, dict[str, int]] = {}
    for s in signals:
        for comp in s.get("competitors", []):
            if not isinstance(comp, dict):
                continue
            cname = _canonicalize_competitor(str(comp.get("name", "")))
            if not cname:
                continue
            for pain in _safe_json(s.get("pain_json")):
                if isinstance(pain, dict) and pain.get("category"):
                    cat = str(pain["category"])
                    pain_by_competitor.setdefault(cname, {})
                    pain_by_competitor[cname][cat] = pain_by_competitor[cname].get(cat, 0) + 1
    triggers: list[dict[str, Any]] = []
    for comp, pains in sorted(pain_by_competitor.items(), key=lambda x: -sum(x[1].values()))[:5]:
        if not pains:
            continue
        top_pain = max(pains.items(), key=lambda x: x[1])
        triggers.append({
            "competitor": comp,
            "primary_reason": top_pain[0],
            "mention_count": top_pain[1],
            "total_mentions": sum(pains.values()),
        })
    return triggers


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

    # ── Competitive Benchmark enrichments ──────────────────────────

    # Strengths/weaknesses from product profiles
    profiles_raw = await _fetch_product_profiles(pool)
    _profile_lookup: dict[str, dict] = {}
    for pp in profiles_raw:
        vn = _canonicalize_vendor(pp.get("vendor_name", ""))
        if vn and vn not in _profile_lookup:
            _profile_lookup[vn] = pp

    def _extract_profile_list(vendor_name: str, field: str) -> list[dict[str, Any]]:
        canon = _canonicalize_vendor(vendor_name)
        raw = _profile_lookup.get(canon, {}).get(field) or []
        if not isinstance(raw, list):
            return []
        return [
            {"area": item.get("area", item) if isinstance(item, dict) else str(item),
             "score": item.get("score") if isinstance(item, dict) else None}
            for item in raw[:5]
        ]

    primary_strengths = _extract_profile_list(primary_name, "strengths")
    primary_weaknesses = _extract_profile_list(primary_name, "weaknesses")
    comparison_strengths = _extract_profile_list(comparison_name, "strengths")
    comparison_weaknesses = _extract_profile_list(comparison_name, "weaknesses")

    # Switching triggers from review data
    primary_triggers = _switching_triggers(primary_rows)
    comparison_triggers = _switching_triggers(comparison_rows)

    # Trend analysis from prior comparison reports
    trend_analysis = None
    try:
        prior_row = await pool.fetchrow("""
            SELECT intelligence_data FROM b2b_intelligence
            WHERE report_type = 'vendor_comparison'
              AND vendor_filter = $1 AND category_filter = $2
              AND report_date < $3
            ORDER BY report_date DESC LIMIT 1
        """, primary_name, comparison_name, today)
        if prior_row and prior_row["intelligence_data"]:
            prior_data = prior_row["intelligence_data"]
            if isinstance(prior_data, str):
                prior_data = json.loads(prior_data)
            if isinstance(prior_data, dict):
                prior_pm = prior_data.get("primary_metrics", {})
                prior_cm = prior_data.get("comparison_metrics", {})
                trend_analysis = {
                    "primary_churn_density_change": round(
                        primary_snapshot.get("churn_signal_density", 0) - float(prior_pm.get("churn_signal_density", 0)), 1),
                    "comparison_churn_density_change": round(
                        comparison_snapshot.get("churn_signal_density", 0) - float(prior_cm.get("churn_signal_density", 0)), 1),
                    "primary_urgency_change": round(
                        primary_snapshot.get("avg_urgency_score", 0) - float(prior_pm.get("avg_urgency_score", 0)), 1),
                    "comparison_urgency_change": round(
                        comparison_snapshot.get("avg_urgency_score", 0) - float(prior_cm.get("avg_urgency_score", 0)), 1),
                    "prior_report_date": str(prior_data.get("report_date", "")),
                }
    except Exception:
        logger.warning("Failed to fetch prior comparison for trend analysis")

    # ── Assemble report ────────────────────────────────────────────

    _snapshot_exclude = {"top_pain_categories", "top_competitors", "top_feature_gaps",
                         "company_examples", "quote_highlights", "product_categories"}
    report_data = {
        "primary_vendor": primary_name,
        "comparison_vendor": comparison_name,
        "report_date": str(today),
        "window_days": window_days,
        "executive_summary": _build_vendor_comparison_summary(primary_snapshot, comparison_snapshot, head_to_head),
        "primary_metrics": {k: v for k, v in primary_snapshot.items() if k not in _snapshot_exclude},
        "comparison_metrics": {k: v for k, v in comparison_snapshot.items() if k not in _snapshot_exclude},
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
        # Competitive Benchmark enrichments
        "primary_strengths": primary_strengths,
        "primary_weaknesses": primary_weaknesses,
        "comparison_strengths": comparison_strengths,
        "comparison_weaknesses": comparison_weaknesses,
        "primary_switching_triggers": primary_triggers,
        "comparison_switching_triggers": comparison_triggers,
        "trend_analysis": trend_analysis,
    }
    if persist:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, vendor_filter, category_filter,
                intelligence_data, executive_summary, data_density, status, llm_model
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')), LOWER(COALESCE(category_filter,'')), COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                          executive_summary = EXCLUDED.executive_summary,
                          data_density = EXCLUDED.data_density,
                          created_at = now()
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
               r.enrichment->'contract_context'->>'contract_value_signal' AS contract_value_signal,
               r.reviewer_title, r.company_size_raw,
               COALESCE(r.reviewer_industry, r.enrichment->'reviewer_context'->>'industry') AS industry
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
    industries: dict[str, int] = {}
    timeline_signals: list[dict[str, Any]] = []
    contract_signals: list[str] = []
    quote_highlights: list[str] = []
    decision_maker_count = 0
    churn_mentions = 0
    urgencies: list[float] = []
    company_size_val: str | None = None
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
        ind = str(row.get("industry") or "")
        if ind:
            industries[ind] = industries.get(ind, 0) + 1
        if not company_size_val and row.get("company_size_raw"):
            company_size_val = str(row["company_size_raw"])
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
            "title": row.get("reviewer_title"),
            "company_size": row.get("company_size_raw"),
            "industry": row.get("industry"),
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
        "industries": sorted([{"industry": k, "count": v} for k, v in industries.items()], key=lambda x: x["count"], reverse=True)[:5],
        "company_size": company_size_val,
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
            ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')), LOWER(COALESCE(category_filter,'')), COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                          executive_summary = EXCLUDED.executive_summary,
                          data_density = EXCLUDED.data_density,
                          created_at = now()
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
            ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')), LOWER(COALESCE(category_filter,'')), COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                          executive_summary = EXCLUDED.executive_summary,
                          data_density = EXCLUDED.data_density,
                          created_at = now()
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
               r.enrichment->'feature_gaps' AS feature_gaps,
               r.reviewer_title, r.company_size_raw,
               COALESCE(r.reviewer_industry, r.enrichment->'reviewer_context'->>'industry') AS industry
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
            ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')), LOWER(COALESCE(category_filter,'')), COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data,
                          executive_summary = EXCLUDED.executive_summary,
                          data_density = EXCLUDED.data_density,
                          created_at = now()
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
