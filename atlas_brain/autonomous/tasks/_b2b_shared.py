"""Shared fetch helpers, lookup builders, and deterministic report builders.

Extracted from b2b_churn_intelligence.py to enable follow-up tasks
(reports, battle cards, article correlation) to re-use these without
importing the monolith.
"""

import asyncio
import json
import logging
import math
import re
from collections import Counter
from datetime import date
from typing import Any

from ...config import settings
from ...services.apollo_company_overrides import fetch_company_override_map
from ...services.company_normalization import normalize_company_name
from ...services.scraping.sources import (
    parse_source_allowlist,
    display_name as _source_display_name,
    VERIFIED_SOURCES,
)
from ...services.tracing import build_business_trace_context
from ...services.vendor_registry import resolve_vendor_name_cached

logger = logging.getLogger("atlas.autonomous.tasks.b2b_shared")


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


def _battle_card_high_priority_score_min() -> float:
    """Return the configured score threshold for high-priority language."""
    return float(settings.b2b_churn.battle_card_high_priority_score_min)


def _battle_card_high_priority_urgency_min() -> float:
    """Return the configured urgency threshold for high-priority language."""
    return float(settings.b2b_churn.battle_card_high_priority_urgency_min)


def _battle_card_feature_gap_headline_min_mentions() -> int:
    """Return the configured feature-gap mention threshold for headlines."""
    return int(settings.b2b_churn.battle_card_feature_gap_headline_min_mentions)


def _battle_card_leaving_patterns() -> tuple[str, ...]:
    """Return configured phrases that imply unsupported switching claims."""
    patterns = settings.b2b_churn.battle_card_leaving_patterns or []
    return tuple(str(item).strip().lower() for item in patterns if str(item).strip())


def _synthesis_reference_confidence_min() -> float:
    """Return the configured confidence floor for synthesis references."""
    return float(settings.b2b_churn.synthesis_reference_confidence_min)


def _synthesis_expert_take_max_words() -> int:
    """Return the configured max word count for scorecard expert_take."""
    return int(settings.b2b_churn.synthesis_expert_take_max_words)


def _trim_words(text: str, limit: int) -> str:
    """Trim text to a max word count while preserving sentence punctuation."""
    words = str(text or "").split()
    if len(words) <= limit:
        return str(text or "").strip()
    return " ".join(words[:limit]).rstrip(" ,;") + "."


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


# ------------------------------------------------------------------
# Layer 0: no internal dependencies
# ------------------------------------------------------------------


def _canonicalize_competitor(raw: str) -> str:
    """Normalize competitor name via vendor registry, then title-case."""
    return resolve_vendor_name_cached(raw)


def _canonicalize_vendor(raw: str) -> str:
    """Normalize vendor labels using the same alias handling as competitors."""
    return resolve_vendor_name_cached(raw)


async def _sync_vendor_firmographics(pool, *, as_of: date) -> int:
    """Best-effort vendor -> org-cache sync for firmographic reasoning inputs."""
    override_map = await fetch_company_override_map(pool)
    rows = await pool.fetch(
        """
        SELECT DISTINCT vendor_name
        FROM b2b_reviews
        WHERE vendor_name IS NOT NULL AND vendor_name <> ''
        ORDER BY vendor_name
        """
    )
    synced = 0
    for row in rows:
        vendor_name = _canonicalize_vendor(row["vendor_name"] or "")
        vendor_name_norm = normalize_company_name(vendor_name)
        if not vendor_name or not vendor_name_norm:
            continue
        override = override_map.get(vendor_name_norm) or {}
        candidate_names = [vendor_name_norm]
        for alias in override.get("search_names", []) or []:
            alias_norm = normalize_company_name(str(alias))
            if alias_norm and alias_norm not in candidate_names:
                candidate_names.append(alias_norm)
        candidate_domains = []
        for domain in override.get("domains", []) or []:
            cleaned = str(domain).strip().lower().removeprefix("www.")
            if cleaned and cleaned not in candidate_domains:
                candidate_domains.append(cleaned)
        org = await pool.fetchrow(
            """
            SELECT id, company_name_raw, company_name_norm, domain,
                   industry, employee_count, annual_revenue_range
            FROM prospect_org_cache
            WHERE status = 'enriched'
              AND (
                company_name_norm = ANY($1::text[])
                OR (
                  domain IS NOT NULL
                  AND domain <> ''
                  AND LOWER(domain) = ANY($2::text[])
                )
              )
            ORDER BY
                CASE
                    WHEN company_name_norm = $3 THEN 0
                    WHEN company_name_norm = ANY($1::text[]) THEN 1
                    ELSE 2
                END,
                employee_count DESC NULLS LAST,
                updated_at DESC
            LIMIT 1
            """,
            candidate_names,
            candidate_domains,
            vendor_name_norm,
        )
        if not org:
            continue
        await pool.execute(
            """
            INSERT INTO b2b_vendor_firmographics (
                vendor_name, vendor_name_norm, company_name_raw, company_name_norm,
                org_cache_id, domain, industry, employee_count,
                annual_revenue_range, source, match_confidence, last_synced_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'prospect_org_cache', 1.0, NOW(), NOW())
            ON CONFLICT (vendor_name_norm) DO UPDATE SET
                vendor_name = EXCLUDED.vendor_name,
                company_name_raw = EXCLUDED.company_name_raw,
                company_name_norm = EXCLUDED.company_name_norm,
                org_cache_id = EXCLUDED.org_cache_id,
                domain = EXCLUDED.domain,
                industry = EXCLUDED.industry,
                employee_count = EXCLUDED.employee_count,
                annual_revenue_range = EXCLUDED.annual_revenue_range,
                last_synced_at = NOW(),
                updated_at = NOW()
            """,
            vendor_name,
            vendor_name_norm,
            org["company_name_raw"],
            org["company_name_norm"],
            org["id"],
            org["domain"],
            org["industry"],
            org["employee_count"],
            org["annual_revenue_range"],
        )
        if org["employee_count"] is not None:
            await pool.execute(
                """
                INSERT INTO b2b_vendor_firmographic_snapshots (
                    vendor_name, vendor_name_norm, snapshot_date,
                    employee_count, annual_revenue_range, industry, source
                ) VALUES ($1, $2, $3, $4, $5, $6, 'prospect_org_cache')
                ON CONFLICT (vendor_name_norm, snapshot_date) DO UPDATE SET
                    vendor_name = EXCLUDED.vendor_name,
                    employee_count = EXCLUDED.employee_count,
                    annual_revenue_range = EXCLUDED.annual_revenue_range,
                    industry = EXCLUDED.industry
                """,
                vendor_name,
                vendor_name_norm,
                as_of,
                org["employee_count"],
                org["annual_revenue_range"],
                org["industry"],
            )
        synced += 1
    return synced


def _battle_card_quote_sort_key(raw_quote: Any) -> tuple[float, int, int]:
    """Rank quotes for battle cards by urgency, specificity, and metadata richness."""
    if not isinstance(raw_quote, dict):
        text = str(raw_quote or "")
        return (0.0, min(len(text), 240), 0)

    text = str(raw_quote.get("quote") or raw_quote.get("text") or "")
    urgency = float(raw_quote.get("urgency") or 0.0)
    metadata_points = sum(
        1 for key in ("company", "title", "company_size", "industry", "source_site")
        if raw_quote.get(key)
    )
    # Longer, more concrete quotes usually perform better than generic one-liners.
    specificity = min(len(text.strip()), 240)
    return (urgency, specificity, metadata_points)


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


def _battle_card_iter_text(value: Any, path: str = ""):
    """Yield flattened string fields from nested battle-card payloads."""
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield from _battle_card_iter_text(inner, next_path)
        return
    if isinstance(value, list):
        for idx, inner in enumerate(value):
            next_path = f"{path}[{idx}]"
            yield from _battle_card_iter_text(inner, next_path)


def _battle_card_numeric_paths(path: str) -> bool:
    """Return True when a path should contain only input-supported claims."""
    return (
        path == "executive_summary"
        or ".evidence" in path
        or ".proof_point" in path
        or path.startswith("competitive_landscape.")
    )


def _battle_card_headline_paths(path: str) -> bool:
    """Return True for top-line summary fields that should stay on strong evidence."""
    return path == "executive_summary" or path.startswith("weakness_analysis[0].")


def _battle_card_numeric_tokens(text: str) -> set[str]:
    """Extract numeric tokens from narrative sections for validation."""
    return set(re.findall(r"\b\d[\d,]*(?:\.\d+)?%?", text or ""))


def _battle_card_add_claim(claims: set[str], value: Any, *, pct: bool = False) -> None:
    """Add an allowed numeric claim token derived from source data."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return
    if pct:
        num *= 100.0
    rounded = round(num, 1)
    if float(rounded).is_integer():
        base = f"{int(round(rounded))}"
    else:
        base = f"{rounded:.1f}"
    claims.add(base)
    claims.add(f"{int(round(num)):,}" if not pct and num >= 1000 else base)
    if pct:
        claims.add(f"{base}%")
        claims.add(f"{int(round(num))}%")
    elif num >= 1000:
        claims.add(f"{int(round(num))}")


def _battle_card_allowed_claims(card: dict[str, Any]) -> set[str]:
    """Build the set of numeric claims supported by deterministic card input."""
    claims: set[str] = _battle_card_numeric_tokens(json.dumps(card, default=str))
    _battle_card_add_claim(claims, card.get("total_reviews"))
    _battle_card_add_claim(claims, card.get("churn_pressure_score"))
    data = card.get("objection_data") or {}
    for key in ("price_complaint_rate", "dm_churn_rate"):
        _battle_card_add_claim(claims, data.get(key), pct=True)
    for key in ("churn_signal_density", "avg_urgency", "total_reviews"):
        _battle_card_add_claim(claims, data.get(key))
    for item in card.get("vendor_weaknesses") or []:
        _battle_card_add_claim(claims, item.get("evidence_count") or item.get("count"))
    for item in card.get("competitor_differentiators") or []:
        _battle_card_add_claim(claims, item.get("mentions"))
        _battle_card_add_claim(claims, item.get("switch_count"))
    for item in data.get("top_feature_gaps") or []:
        _battle_card_add_claim(claims, item.get("mentions"))
    for key in ("avg_seat_count", "max_seat_count", "median_seat_count", "price_increase_count"):
        _battle_card_add_claim(claims, (data.get("budget_context") or {}).get(key))
    _battle_card_add_claim(claims, (data.get("budget_context") or {}).get("price_increase_rate"), pct=True)
    return claims


def _battle_card_competitor_names(card: dict[str, Any], *, limit: int = 2) -> list[str]:
    """Return top competitor names already present in deterministic card input."""
    names: list[str] = []
    for item in card.get("competitor_differentiators") or []:
        competitor = str(item.get("competitor") or "").strip()
        if competitor and competitor not in names:
            names.append(competitor)
        if len(names) >= limit:
            break
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


def _battle_card_safe_summary(card: dict[str, Any]) -> str:
    """Build a grounded executive summary when generated copy overclaims."""
    vendor = str(card.get("vendor") or "This vendor").strip() or "This vendor"
    weaknesses = [
        str(item.get("area") or item.get("weakness") or "").strip().lower()
        for item in card.get("vendor_weaknesses") or []
        if str(item.get("area") or item.get("weakness") or "").strip()
    ]
    pain = weaknesses[0] if weaknesses else "customer fit"
    competitors = _battle_card_competitor_names(card)
    if competitors:
        return (
            f"{vendor} is showing churn pressure around {pain} while buyers "
            f"re-evaluate fit against {_join_summary_terms(competitors)}."
        )
    return f"{vendor} is showing churn pressure around {pain} in recent buyer feedback."


def _battle_card_safe_text(card: dict[str, Any], path: str) -> str:
    """Return grounded replacement text for numeric-sensitive paths."""
    if path == "executive_summary":
        return _battle_card_safe_summary(card)
    if path.endswith(".evidence"):
        return "Supported by recurring customer complaints and churn-oriented review evidence."
    if path.endswith(".proof_point"):
        return "The input shows recurring buyer friction and credible evaluation pressure."
    if path.startswith("competitive_landscape."):
        competitors = _battle_card_competitor_names(card)
        if "top_alternatives" in path and competitors:
            return (
                "Alternatives appearing most often in buyer evaluation sets include "
                f"{_join_summary_terms(competitors)}."
            )
        return "Competitive pressure is present where buyers are re-evaluating fit and value."
    return ""


def _sanitize_battle_card_text(
    card: dict[str, Any],
    path: str,
    text: str,
    *,
    allowed_claims: set[str],
    source_text: str,
    max_switch: int,
    score: float,
    urgency: float,
    low_gap_terms: list[str],
) -> str:
    """Rewrite overclaiming battle-card strings into validator-safe text."""
    cleaned = str(text or "")
    if _battle_card_numeric_paths(path):
        bad = sorted(tok for tok in _battle_card_numeric_tokens(cleaned) if tok not in allowed_claims)
        if bad:
            cleaned = _battle_card_safe_text(card, path)
    years = [year for year in re.findall(r"\b20\d{2}\b", cleaned) if year not in source_text]
    for year in years:
        cleaned = re.sub(rf"\s*\b{re.escape(year)}\b", "", cleaned)
    if max_switch == 0:
        replacements = (
            ("customers are leaving", "buyers are evaluating alternatives"),
            ("customer are leaving", "buyers are evaluating alternatives"),
            ("are leaving for", "are evaluating"),
            ("capturing defectors", "winning evaluations"),
            ("capture defectors", "win evaluations"),
            ("defectors", "evaluators"),
        )
        lowered = cleaned.lower()
        for src, dst in replacements:
            if src in lowered:
                cleaned = re.sub(re.escape(src), dst, cleaned, flags=re.IGNORECASE)
                lowered = cleaned.lower()
    if score < _battle_card_high_priority_score_min() or urgency < _battle_card_high_priority_urgency_min():
        cleaned = re.sub(r"high[- ]priority target:?\s*", "Emerging vulnerability: ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b(a|an)\s+Emerging vulnerability:\s*",
            "an emerging vulnerability for ",
            cleaned,
            flags=re.IGNORECASE,
        )
    if _battle_card_headline_paths(path):
        for term in low_gap_terms:
            if term and term in cleaned.lower():
                return _battle_card_safe_summary(card) if path == "executive_summary" else "Workflow friction is showing up in customer feedback."
    return re.sub(r"\s+", " ", cleaned).strip()


def _sanitize_battle_card_sales_copy(card: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """Deterministically rewrite near-miss sales copy before final rejection."""
    if not isinstance(generated, dict):
        return {}
    allowed = _battle_card_allowed_claims(card)
    source_text = json.dumps(card, default=str).lower()
    max_switch = max(
        (int(c.get("switch_count") or 0) for c in card.get("competitor_differentiators") or []),
        default=0,
    )
    score = float(card.get("churn_pressure_score") or 0)
    urgency = float(((card.get("objection_data") or {}).get("avg_urgency")) or 0)
    low_gap_terms = [
        str(g.get("feature") or "").strip().lower()
        for g in ((card.get("objection_data") or {}).get("top_feature_gaps") or [])
        if int(g.get("mentions") or 0) < _battle_card_feature_gap_headline_min_mentions()
    ]

    def _walk(value: Any, path: str = "") -> Any:
        if isinstance(value, str):
            return _sanitize_battle_card_text(
                card,
                path,
                value,
                allowed_claims=allowed,
                source_text=source_text,
                max_switch=max_switch,
                score=score,
                urgency=urgency,
                low_gap_terms=low_gap_terms,
            )
        if isinstance(value, list):
            return [_walk(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
        if isinstance(value, dict):
            return {
                key: _walk(inner, f"{path}.{key}" if path else str(key))
                for key, inner in value.items()
            }
        return value

    return _walk(generated)


def _best_cross_vendor_comparison(scorecard: dict[str, Any]) -> dict[str, Any] | None:
    """Return the highest-confidence cross-vendor comparison above the ref floor."""
    floor = _synthesis_reference_confidence_min()
    comparisons = [
        comp for comp in (scorecard.get("cross_vendor_comparisons") or [])
        if float(comp.get("confidence") or 0) >= floor
    ]
    if not comparisons:
        return None
    return max(comparisons, key=lambda comp: float(comp.get("confidence") or 0))


def _build_scorecard_locked_facts(scorecard: dict[str, Any]) -> dict[str, Any]:
    """Build source-of-truth synthesis constraints for scorecard narratives."""
    locked: dict[str, Any] = {
        "vendor": str(scorecard.get("vendor") or ""),
        "risk_level": str(scorecard.get("risk_level") or ""),
    }
    if float(scorecard.get("archetype_confidence") or 0) >= _synthesis_reference_confidence_min():
        if scorecard.get("archetype"):
            locked["archetype"] = scorecard["archetype"]
    allowed_opponents = [
        str(comp.get("opponent") or "")
        for comp in (scorecard.get("cross_vendor_comparisons") or [])
        if str(comp.get("opponent") or "")
        and float(comp.get("confidence") or 0) >= _synthesis_reference_confidence_min()
    ]
    if allowed_opponents:
        locked["allowed_opponents"] = allowed_opponents
    best = _best_cross_vendor_comparison(scorecard)
    if best:
        locked["comparison"] = {
            "opponent": best.get("opponent", ""),
            "resource_advantage": best.get("resource_advantage", ""),
        }
    return {k: v for k, v in locked.items() if v not in ("", [], None)}


def _build_battle_card_locked_facts(card: dict[str, Any]) -> dict[str, Any]:
    """Build source-of-truth synthesis constraints for battle-card copy."""
    objection_data = card.get("objection_data") or {}
    allowed_opponents: list[str] = []
    for comp in card.get("competitor_differentiators") or []:
        name = str(comp.get("competitor") or "").strip()
        if name and name not in allowed_opponents:
            allowed_opponents.append(name)
    for battle in card.get("cross_vendor_battles") or []:
        name = str(battle.get("opponent") or "").strip()
        if name and name not in allowed_opponents:
            allowed_opponents.append(name)
    asymmetry = card.get("resource_asymmetry") or {}
    asym_opponent = str(asymmetry.get("opponent") or "").strip()
    if asym_opponent and asym_opponent not in allowed_opponents:
        allowed_opponents.append(asym_opponent)
    locked: dict[str, Any] = {
        "vendor": str(card.get("vendor") or ""),
        "priority_language_allowed": (
            float(card.get("churn_pressure_score") or 0) >= _battle_card_high_priority_score_min()
            and float(objection_data.get("avg_urgency") or 0) >= _battle_card_high_priority_urgency_min()
        ),
    }
    if card.get("archetype"):
        locked["archetype"] = card["archetype"]
    if card.get("archetype_risk_level"):
        locked["archetype_risk_level"] = card["archetype_risk_level"]
    if allowed_opponents:
        locked["allowed_opponents"] = allowed_opponents
    if asymmetry.get("resource_advantage"):
        locked["resource_advantage"] = asymmetry.get("resource_advantage")
    return locked


def _fallback_scorecard_expert_take(scorecard: dict[str, Any]) -> str:
    """Build a deterministic, buyer-facing scorecard narrative fallback."""
    vendor = str(scorecard.get("vendor") or "this vendor").strip() or "this vendor"
    reasoning = str(scorecard.get("reasoning_summary") or "").strip()
    if reasoning:
        base = reasoning
    else:
        top_pain = str(scorecard.get("top_pain") or "customer fit").strip().lower()
        trend = str(scorecard.get("trend") or "stable").strip().lower() or "stable"
        competitor_overlap = scorecard.get("competitor_overlap") or []
        top_comp = ""
        if competitor_overlap and isinstance(competitor_overlap[0], dict):
            top_comp = str(competitor_overlap[0].get("competitor") or "").strip()
        if top_comp:
            base = (
                f"Buyers considering {vendor} should pressure-test {top_pain} because "
                f"recent feedback is {trend} and evaluation pressure is clustering around {top_comp}."
            )
        else:
            base = (
                f"Buyers considering {vendor} should pressure-test {top_pain} because "
                f"recent buyer feedback shows a {trend} churn pattern."
            )
    best = _best_cross_vendor_comparison(scorecard)
    if best and best.get("opponent"):
        opponent = str(best.get("opponent") or "").strip()
        advantage = str(best.get("resource_advantage") or "").strip()
        if advantage:
            base += f" Relative to {opponent}, the market comparison points to {advantage}."
        else:
            base += f" Relative to {opponent}, buyers appear to be weighing competitive differences more directly."
    return _trim_words(base, _synthesis_expert_take_max_words())


def _validate_scorecard_expert_take(scorecard: dict[str, Any], expert_take: str) -> list[str]:
    """Reject synthesized scorecard narratives that contradict locked facts."""
    text = str(expert_take or "").strip()
    if not text:
        return ["expert_take is empty"]
    warnings: list[str] = []
    lower = text.lower()
    if len(text.split()) > _synthesis_expert_take_max_words():
        warnings.append("expert_take exceeds max word count")
    locked = _build_scorecard_locked_facts(scorecard)
    allowed_archetype = str(locked.get("archetype") or "").lower()
    from ...reasoning.archetypes import ARCHETYPES
    for archetype_name in ARCHETYPES:
        if archetype_name in lower:
            if not allowed_archetype:
                warnings.append("expert_take references an archetype without sufficient reasoning confidence")
            elif archetype_name != allowed_archetype:
                warnings.append(f"expert_take references archetype '{archetype_name}' instead of '{allowed_archetype}'")
            break
    all_opponents = [
        str(comp.get("opponent") or "").strip()
        for comp in (scorecard.get("cross_vendor_comparisons") or [])
        if str(comp.get("opponent") or "").strip()
    ]
    allowed_opponents = {str(name).lower() for name in (locked.get("allowed_opponents") or [])}
    for opponent in all_opponents:
        if opponent.lower() in lower and opponent.lower() not in allowed_opponents:
            warnings.append(f"expert_take references opponent '{opponent}' without high-confidence comparison support")
            break
    return warnings


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


# ------------------------------------------------------------------
# Layer 1: depends on Layer 0
# ------------------------------------------------------------------


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
                    from datetime import datetime
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


def _validate_battle_card_sales_copy(
    card: dict[str, Any],
    generated: dict[str, Any],
) -> list[str]:
    """Reject battle-card copy that overclaims beyond deterministic evidence."""
    if not isinstance(generated, dict):
        return ["battle card sales copy is not a JSON object"]
    warnings: list[str] = []
    allowed = _battle_card_allowed_claims(card)
    source_text = json.dumps(card, default=str).lower()
    max_switch = max((int(c.get("switch_count") or 0) for c in card.get("competitor_differentiators") or []), default=0)
    score = float(card.get("churn_pressure_score") or 0)
    urgency = float(((card.get("objection_data") or {}).get("avg_urgency")) or 0)
    low_gap_terms = [
        str(g.get("feature") or "").strip().lower()
        for g in ((card.get("objection_data") or {}).get("top_feature_gaps") or [])
        if int(g.get("mentions") or 0) < _battle_card_feature_gap_headline_min_mentions()
    ]
    for path, text in _battle_card_iter_text(generated):
        lowered = text.lower()
        if _battle_card_numeric_paths(path):
            bad = sorted(tok for tok in _battle_card_numeric_tokens(text) if tok not in allowed)
            if bad:
                warnings.append(f"{path} uses unsupported numeric claims: {', '.join(bad[:4])}")
        years = re.findall(r"\b20\d{2}\b", text)
        if any(year not in source_text for year in years):
            warnings.append(f"{path} references a year not present in source data")
        if max_switch == 0 and any(term in lowered for term in _battle_card_leaving_patterns()):
            warnings.append(f"{path} implies switching despite zero switch_count evidence")
        if score < _battle_card_high_priority_score_min() or urgency < _battle_card_high_priority_urgency_min():
            if "high-priority target" in lowered or "high priority target" in lowered:
                warnings.append(f"{path} overstates urgency for a moderate-priority card")
        if _battle_card_headline_paths(path):
            for term in low_gap_terms:
                if term and term in lowered:
                    warnings.append(f"{path} elevates low-evidence feature gap '{term}' to a headline")
    return warnings


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


def _compute_churn_pressure_score(
    *,
    churn_density: float,
    avg_urgency: float,
    dm_churn_rate: float,
    displacement_mention_count: int,
    price_complaint_rate: float,
    total_reviews: int,
    archetype: str | None = None,
    displacement_velocity: float | None = None,
) -> float:
    """Composite 0-100 score for ranking vendors by churn pressure.

    Default weights: churn density 30%, urgency 25%, DM churn rate 20%,
    displacement mentions 15%, price complaints 10%.

    When *archetype* is provided (from the stratified reasoner), the weights
    shift to emphasise the signal most relevant to that churn pattern --
    e.g. ``pricing_shock`` boosts price_complaints to 35%.

    Optional *displacement_velocity* adds a 0-5 point bonus for vendors
    with accelerating competitive displacement (leading indicator).

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
    # Displacement velocity bonus: 0-5 points on top of weighted score
    if displacement_velocity is not None and displacement_velocity > 0:
        raw += min(displacement_velocity / 5.0, 1.0) * 5.0
    if total_reviews >= 50:
        confidence = 1.0
    elif total_reviews >= 20:
        confidence = 0.85
    else:
        confidence = 0.65
    return round(min(raw * confidence, 100.0), 1)


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
            + f" -- {', '.join(risk_parts)}."
        ]

        # Sentence 2: top-pressure vendor, score, driver, DM rate
        top = scorecards[0]  # sorted by urgency+density descending
        top_vendor = top.get("vendor", "")
        top_score = top.get("churn_pressure_score", 0)
        top_density = top.get("churn_signal_density", 0)
        top_pain = top.get("top_pain", "")
        dm_rate = top.get("dm_churn_rate", 0)

        s2 = f"{top_vendor} scored highest at {top_score:.1f}"
        s2 += f" ({top_density}% churn density -- share of reviews containing explicit switching signals)"
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
        s2 = f"The strongest flow is {top_from} -> {top_to} ({top_count} mentions)"
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
                f"{cat_name} shows the highest pressure -- {risk_vendor}"
                f" at {hottest_density}% churn density"
                " (share of reviews with explicit switching signals)"
            )
            if pain and pain.lower() not in ("other", "unknown"):
                s2 += f", driven by {pain}"
            s2 += "."
            lines.append(s2)

        # Sentence 3: market movement -- categories with clear challengers
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


# ------------------------------------------------------------------
# Layer 2: async fetch helpers
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
        WITH review_scores AS (
            SELECT vendor_name, product_category,
                count(*) AS total_reviews,
                count(*) FILTER (
                    WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
                ) AS churn_intent,
                avg(
                    (enrichment->>'urgency_score')::numeric
                    * COALESCE(source_weight, 0.7)
                    * (0.7 + 0.3 * COALESCE(relevance_score, 0.5))
                ) / NULLIF(avg(
                    COALESCE(source_weight, 0.7)
                    * (0.7 + 0.3 * COALESCE(relevance_score, 0.5))
                ), 0)
                AS avg_urgency,
                avg(author_churn_score) FILTER (WHERE author_churn_score IS NOT NULL)
                AS avg_author_churn_score,
                avg(rating / NULLIF(rating_max, 0)) AS avg_rating_normalized,
                count(*) FILTER (
                    WHERE (enrichment->>'would_recommend')::boolean = true
                ) AS recommend_yes,
                count(*) FILTER (
                    WHERE (enrichment->>'would_recommend')::boolean = false
                ) AS recommend_no,
                ROUND(
                    count(*) FILTER (
                        WHERE rating IS NOT NULL AND rating_max > 0
                          AND (rating / rating_max) >= 0.7
                    ) * 100.0 / NULLIF(count(*) FILTER (
                        WHERE rating IS NOT NULL AND rating_max > 0
                    ), 0),
                    1
                ) AS positive_review_pct,
                AVG(rating / NULLIF(rating_max, 0)) FILTER (
                    WHERE review_text ILIKE '%support%' OR review_text ILIKE '%service%'
                ) AS support_sentiment,
                AVG(rating / NULLIF(rating_max, 0)) FILTER (
                    WHERE review_text ILIKE '%legacy%' OR review_text ILIKE '%old version%' OR review_text ILIKE '%deprecated%'
                ) AS legacy_support_score,
                COUNT(*) FILTER (
                    WHERE review_text ILIKE '%new feature%' OR review_text ILIKE '%update%' OR review_text ILIKE '%release%'
                ) * 1.0 / NULLIF(count(*), 0) AS new_feature_velocity
            FROM b2b_reviews
            WHERE {filters}
            GROUP BY vendor_name, product_category
            HAVING count(*) >= $2
        )
        SELECT rs.*,
               vf.employee_count,
               vf.industry AS vendor_industry,
               vf.annual_revenue_range,
               CASE
                   WHEN prev.employee_count IS NOT NULL
                        AND prev.employee_count > 0
                        AND vf.employee_count IS NOT NULL
                   THEN (vf.employee_count - prev.employee_count)::numeric / prev.employee_count
                   ELSE NULL
               END AS employee_growth_rate
        FROM review_scores rs
        LEFT JOIN b2b_vendor_firmographics vf
          ON LOWER(vf.vendor_name) = LOWER(rs.vendor_name)
        LEFT JOIN LATERAL (
            SELECT employee_count
            FROM b2b_vendor_firmographic_snapshots snap
            WHERE snap.vendor_name_norm = vf.vendor_name_norm
              AND snap.snapshot_date < CURRENT_DATE
              AND snap.employee_count IS NOT NULL
            ORDER BY snap.snapshot_date DESC
            LIMIT 1
        ) prev ON TRUE
        ORDER BY rs.avg_urgency DESC
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
            "avg_author_churn_score": float(r["avg_author_churn_score"]) if r["avg_author_churn_score"] is not None else None,
            "support_sentiment": float(r["support_sentiment"]) if r["support_sentiment"] is not None else None,
            "legacy_support_score": float(r["legacy_support_score"]) if r["legacy_support_score"] is not None else None,
            "new_feature_velocity": float(r["new_feature_velocity"]) if r["new_feature_velocity"] is not None else None,
            "employee_growth_rate": float(r["employee_growth_rate"]) if r["employee_growth_rate"] is not None else None,
            "employee_count": r["employee_count"],
            "vendor_industry": r["vendor_industry"],
            "annual_revenue_range": r["annual_revenue_range"],
        }
        for r in rows
    ]


async def _fetch_high_intent_companies(pool, urgency_threshold: int, window_days: int) -> list[dict[str, Any]]:
    """Companies showing high churn intent -- the money feed."""
    sources = _intelligence_source_allowlist()
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
            enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
            relevance_score,
            author_churn_score
        FROM b2b_reviews
        WHERE {filters}
          AND (enrichment->>'urgency_score')::numeric >= $1
          AND reviewer_company IS NOT NULL AND reviewer_company != ''
          AND COALESCE(relevance_score, 0.5) >= 0.3
        ORDER BY (enrichment->>'urgency_score')::numeric
                 * (0.7 + 0.3 * COALESCE(relevance_score, 0.5)) DESC
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
            "relevance_score": float(r["relevance_score"]) if r["relevance_score"] is not None else None,
            "author_churn_score": float(r["author_churn_score"]) if r["author_churn_score"] is not None else None,
        })
    return results


async def _fetch_competitive_displacement(pool, window_days: int) -> list[dict[str, Any]]:
    """Competitive displacement flows -- filtered to real displacement evidence only.

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
            count(*) AS total,
            array_agg(DISTINCT enrichment->'budget_signals'->>'annual_spend_estimate')
                FILTER (WHERE enrichment->'budget_signals'->>'annual_spend_estimate' IS NOT NULL
                        AND enrichment->'budget_signals'->>'annual_spend_estimate' != '')
                AS annual_spend_values,
            array_agg(DISTINCT enrichment->'budget_signals'->>'price_per_seat')
                FILTER (WHERE enrichment->'budget_signals'->>'price_per_seat' IS NOT NULL
                        AND enrichment->'budget_signals'->>'price_per_seat' != '')
                AS price_per_seat_values
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
            "annual_spend_signals": [v for v in (r["annual_spend_values"] or []) if v],
            "price_per_seat_signals": [v for v in (r["price_per_seat_values"] or []) if v],
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
            sentiment_direction AS direction,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE {filters}
          AND sentiment_direction IS NOT NULL
        GROUP BY vendor_name, sentiment_direction
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


async def _fetch_sentiment_tenure(pool, window_days: int) -> list[dict[str, Any]]:
    """Aggregate customer tenure from sentiment_trajectory per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            sentiment_tenure AS tenure,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE {filters}
          AND sentiment_tenure IS NOT NULL
          AND sentiment_tenure != ''
        GROUP BY vendor_name, sentiment_tenure
        ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )
    return [
        {"vendor": r["vendor_name"], "tenure": r["tenure"], "count": r["cnt"]}
        for r in rows
    ]


async def _fetch_turning_points(pool, window_days: int) -> list[dict[str, Any]]:
    """Aggregate churn turning points from sentiment_trajectory per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            sentiment_turning_point AS turning_point,
            count(*) AS cnt
        FROM b2b_reviews
        WHERE {filters}
          AND sentiment_turning_point IS NOT NULL
          AND sentiment_turning_point != ''
        GROUP BY vendor_name, sentiment_turning_point
        ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )
    return [
        {"vendor": r["vendor_name"], "trigger": r["turning_point"], "mentions": r["cnt"]}
        for r in rows
    ]


async def _fetch_review_text_aggregates(pool, window_days: int) -> tuple[list[dict], list[dict]]:
    """Aggregate specific_complaints and positive_aspects per vendor.

    Returns (complaint_rows, positive_rows) where each row has
    vendor, text, mentions. Only includes items with 2+ mentions.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)

    complaint_rows, positive_rows = await asyncio.gather(
        pool.fetch(
            f"""
            SELECT vendor_name, c.value #>> '{{}}' AS text, count(*) AS mentions
            FROM b2b_reviews
            CROSS JOIN LATERAL jsonb_array_elements(
                COALESCE(enrichment->'specific_complaints', '[]'::jsonb)
            ) AS c(value)
            WHERE {filters}
            GROUP BY vendor_name, c.value #>> '{{}}'
            HAVING count(*) >= 2
            ORDER BY mentions DESC
            """,
            window_days,
            sources,
        ),
        pool.fetch(
            f"""
            SELECT vendor_name, a.value #>> '{{}}' AS text, count(*) AS mentions
            FROM b2b_reviews
            CROSS JOIN LATERAL jsonb_array_elements(
                COALESCE(enrichment->'positive_aspects', '[]'::jsonb)
            ) AS a(value)
            WHERE {filters}
            GROUP BY vendor_name, a.value #>> '{{}}'
            HAVING count(*) >= 2
            ORDER BY mentions DESC
            """,
            window_days,
            sources,
        ),
    )

    complaints = [
        {"vendor": r["vendor_name"], "text": r["text"], "mentions": r["mentions"]}
        for r in complaint_rows
    ]
    positives = [
        {"vendor": r["vendor_name"], "aspect": r["text"], "mentions": r["mentions"]}
        for r in positive_rows
    ]
    return complaints, positives


async def _fetch_department_distribution(pool, window_days: int) -> list[dict[str, Any]]:
    """Count reviews and churn rate per department per vendor."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            enrichment->'reviewer_context'->>'department' AS department,
            count(*) AS review_count,
            count(*) FILTER (
                WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS churning_count,
            round(avg((enrichment->>'urgency_score')::numeric), 1) AS avg_urgency
        FROM b2b_reviews
        WHERE {filters}
          AND enrichment->'reviewer_context'->>'department' IS NOT NULL
          AND enrichment->'reviewer_context'->>'department' != ''
          AND enrichment->'reviewer_context'->>'department' != 'unknown'
        GROUP BY vendor_name, enrichment->'reviewer_context'->>'department'
        ORDER BY review_count DESC
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "department": r["department"],
            "review_count": r["review_count"],
            "churn_rate": round(r["churning_count"] / r["review_count"], 2) if r["review_count"] else 0,
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
        }
        for r in rows
    ]


async def _fetch_contract_context_distribution(pool, window_days: int) -> tuple[list[dict], list[dict]]:
    """Aggregate contract_value_signal and usage_duration per vendor.

    Returns (value_signal_rows, duration_rows).
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)

    value_rows, duration_rows = await asyncio.gather(
        pool.fetch(
            f"""
            SELECT vendor_name,
                enrichment->'contract_context'->>'contract_value_signal' AS segment,
                count(*) AS cnt,
                count(*) FILTER (
                    WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
                ) AS churning
            FROM b2b_reviews
            WHERE {filters}
              AND enrichment->'contract_context'->>'contract_value_signal' IS NOT NULL
              AND enrichment->'contract_context'->>'contract_value_signal' NOT IN ('unknown', '')
            GROUP BY vendor_name, enrichment->'contract_context'->>'contract_value_signal'
            ORDER BY cnt DESC
            """,
            window_days,
            sources,
        ),
        pool.fetch(
            f"""
            SELECT vendor_name,
                enrichment->'contract_context'->>'usage_duration' AS duration,
                count(*) AS cnt,
                count(*) FILTER (
                    WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
                ) AS churning
            FROM b2b_reviews
            WHERE {filters}
              AND enrichment->'contract_context'->>'usage_duration' IS NOT NULL
              AND enrichment->'contract_context'->>'usage_duration' != ''
            GROUP BY vendor_name, enrichment->'contract_context'->>'usage_duration'
            ORDER BY cnt DESC
            """,
            window_days,
            sources,
        ),
    )

    values = [
        {
            "vendor": r["vendor_name"],
            "segment": r["segment"],
            "count": r["cnt"],
            "churn_rate": round(r["churning"] / r["cnt"], 2) if r["cnt"] else 0,
        }
        for r in value_rows
    ]
    durations = [
        {
            "vendor": r["vendor_name"],
            "duration": r["duration"],
            "count": r["cnt"],
            "churn_rate": round(r["churning"] / r["cnt"], 2) if r["cnt"] else 0,
        }
        for r in duration_rows
    ]
    return values, durations


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
    """Top reasons per vendor/competitor pair -- prefers structured reason_category."""
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


# ------------------------------------------------------------------
# Layer 3: lookup builders (pure Python, no DB)
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


def _build_complaint_lookup(rows: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {text, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r.get("vendor", "")
        lookup.setdefault(vendor, []).append({"text": r["text"], "mentions": r["mentions"]})
    for v in lookup:
        lookup[v].sort(key=lambda x: -x["mentions"])
    return lookup


def _build_positive_lookup(rows: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {aspect, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r.get("vendor", "")
        lookup.setdefault(vendor, []).append({"aspect": r["aspect"], "mentions": r["mentions"]})
    for v in lookup:
        lookup[v].sort(key=lambda x: -x["mentions"])
    return lookup


def _build_department_lookup(rows: list[dict]) -> dict[str, list[dict]]:
    """vendor -> sorted list of {department, review_count, churn_rate, avg_urgency}."""
    lookup: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "department": r["department"],
            "review_count": r["review_count"],
            "churn_rate": r["churn_rate"],
            "avg_urgency": r["avg_urgency"],
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: -x["review_count"])
    return lookup


def _build_contract_value_lookup(rows: list[dict]) -> dict[str, list[dict]]:
    """vendor -> list of {segment, count, churn_rate}."""
    lookup: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "segment": r["segment"],
            "count": r["count"],
            "churn_rate": r["churn_rate"],
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: -x["count"])
    return lookup


def _build_usage_duration_lookup(rows: list[dict]) -> dict[str, list[dict]]:
    """vendor -> list of {duration, count, churn_rate}."""
    lookup: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r.get("vendor", "")
        lookup.setdefault(vendor, []).append({
            "duration": r["duration"],
            "count": r["count"],
            "churn_rate": r["churn_rate"],
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: -x["count"])
    return lookup


def _build_tenure_lookup(rows: list[dict]) -> dict[str, list[dict]]:
    """vendor -> list of {tenure, count}."""
    lookup: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r.get("vendor", "")
        lookup.setdefault(vendor, []).append({"tenure": r["tenure"], "count": r["count"]})
    for v in lookup:
        lookup[v].sort(key=lambda x: -x["count"])
    return lookup


def _build_turning_point_lookup(rows: list[dict]) -> dict[str, list[dict]]:
    """vendor -> list of {trigger, mentions}."""
    lookup: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r.get("vendor", "")
        lookup.setdefault(vendor, []).append({"trigger": r["trigger"], "mentions": r["mentions"]})
    for v in lookup:
        lookup[v].sort(key=lambda x: -x["mentions"])
    return lookup


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


def _build_insider_lookup(rows: list[Any]) -> dict[str, dict]:
    """Build vendor -> insider aggregate dict from _fetch_insider_aggregates rows."""
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


# ------------------------------------------------------------------
# Layer 4: deterministic builders (depend on all above)
# ------------------------------------------------------------------


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
    velocity_lookup: dict[str, dict] | None = None,
    market_regime_lookup: dict[str, dict[str, Any]] | None = None,
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
    for raw_key in (
        "support_sentiment",
        "legacy_support_score",
        "new_feature_velocity",
        "employee_growth_rate",
        "positive_review_pct",
        "avg_rating_normalized",
    ):
        raw_val = vs.get(raw_key)
        if raw_val is not None:
            evidence[raw_key] = raw_val

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
    if market_regime_lookup:
        regime = market_regime_lookup.get(vendor)
        if regime:
            evidence["market_regime"] = regime

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
    if velocity_lookup:
        vel = velocity_lookup.get(vendor, {})
        if vel:
            evidence["displacement_velocity_7d"] = vel.get("velocity_7d")
            evidence["displacement_velocity_30d"] = vel.get("velocity_30d")
            v7 = vel.get("velocity_7d") or 0
            evidence["velocity_trend"] = (
                "accelerating" if v7 > 0
                else "decelerating" if v7 < 0
                else "stable"
            )

    return evidence


async def _fetch_latest_category_overview_entry(
    pool,
    category: str,
) -> dict[str, Any] | None:
    """Fetch the latest persisted category overview entry for a category."""
    if not category:
        return None

    row = await pool.fetchrow(
        """
        SELECT intelligence_data
        FROM b2b_intelligence
        WHERE report_type = 'category_overview'
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

    entries = data if isinstance(data, list) else data.get("category_overview", data)
    if not isinstance(entries, list):
        return None

    wanted = str(category).strip().lower()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("category", "")).strip().lower() == wanted:
            return entry
    return None


async def fetch_vendor_evidence(
    pool,
    vendor_name: str,
    *,
    window_days: int = 90,
) -> dict[str, Any] | None:
    """Build a rich evidence dict for a single vendor from current DB data.

    Used for on-demand reasoning (API + MCP).  Returns None if the vendor
    has no churn signal rows.
    """
    from ...config import settings
    cfg = settings.b2b_churn
    min_reviews = cfg.intelligence_min_reviews
    canonical = _canonicalize_vendor(vendor_name)
    if not canonical:
        return None

    # Fetch vendor score rows
    all_scores = await _fetch_vendor_churn_scores(pool, window_days, min_reviews)
    vs_match = [v for v in all_scores if _canonicalize_vendor(v.get("vendor_name", "")) == canonical]
    if not vs_match:
        return None
    vs = vs_match[0]

    # Parallel fetches for lookups
    (
        pain_dist, competitive_disp, feature_gaps,
        price_rates, dm_rates, keyword_spikes,
        insider_raw, quotable_evidence, budget_signals,
        use_case_dist, buyer_auth,
    ) = await asyncio.gather(
        _fetch_pain_distribution(pool, window_days),
        _fetch_competitive_displacement(pool, window_days),
        _fetch_feature_gaps(pool, window_days),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_keyword_spikes(pool),
        _fetch_insider_aggregates(pool, window_days),
        _fetch_quotable_evidence(pool, window_days),
        _fetch_budget_signals(pool, window_days),
        _fetch_use_case_distribution(pool, window_days),
        _fetch_buyer_authority_summary(pool, window_days),
    )

    pain_lookup = _build_pain_lookup(pain_dist)
    comp_lookup = _build_competitor_lookup(competitive_disp)
    fg_lookup = _build_feature_gap_lookup(feature_gaps)
    kw_lookup = _build_keyword_spike_lookup(keyword_spikes)
    insider_lookup = _build_insider_lookup(insider_raw)
    dm_lookup = {r["vendor"]: r["dm_churn_rate"] for r in dm_rates}
    price_lookup = {r["vendor"]: r["price_complaint_rate"] for r in price_rates}
    quote_lookup = {r["vendor"]: r["quotes"] for r in quotable_evidence}
    budget_lookup = {r["vendor"]: {k: v for k, v in r.items() if k != "vendor"} for r in budget_signals}
    ba_lookup = _build_buyer_auth_lookup(buyer_auth)
    uc_lookup = _build_use_case_lookup(use_case_dist)

    # Temporal analysis (non-fatal)
    temporal_lookup: dict[str, dict] = {}
    archetype_lookup: dict[str, list[dict]] = {}
    market_regime_lookup: dict[str, dict[str, Any]] = {}
    try:
        from ...reasoning.temporal import TemporalEngine
        from ...reasoning.archetypes import enrich_evidence_with_archetypes
        te = TemporalEngine(pool)
        td_result = await te.analyze_vendor(canonical)
        td = TemporalEngine.to_evidence_dict(td_result)
        temporal_lookup[canonical] = td
        evidence_seed = {
            "vendor_name": canonical,
            "support_sentiment": vs.get("support_sentiment"),
            "legacy_support_score": vs.get("legacy_support_score"),
            "new_feature_velocity": vs.get("new_feature_velocity"),
            "employee_growth_rate": vs.get("employee_growth_rate"),
            **td,
        }
        enriched = enrich_evidence_with_archetypes(evidence_seed, td)
        arch_scores = enriched.get("archetype_scores", [])
        if arch_scores:
            archetype_lookup[canonical] = arch_scores
    except Exception:
        logger.debug("Temporal analysis unavailable for %s", canonical)

    try:
        cat_entry = await _fetch_latest_category_overview_entry(
            pool, str(vs.get("product_category") or ""),
        )
        regime = (cat_entry or {}).get("cross_vendor_analysis", {}).get("market_regime")
        if regime:
            market_regime_lookup[canonical] = regime
    except Exception:
        logger.debug("Market regime unavailable for %s", canonical, exc_info=True)

    return _build_vendor_evidence(
        vs,
        pain_lookup=pain_lookup,
        competitor_lookup=comp_lookup,
        feature_gap_lookup=fg_lookup,
        insider_lookup=insider_lookup,
        keyword_spike_lookup=kw_lookup,
        temporal_lookup=temporal_lookup or None,
        archetype_lookup=archetype_lookup or None,
        dm_lookup=dm_lookup,
        price_lookup=price_lookup,
        quote_lookup=quote_lookup,
        budget_lookup=budget_lookup,
        buyer_auth_lookup=ba_lookup,
        use_case_lookup=uc_lookup,
        market_regime_lookup=market_regime_lookup or None,
    )


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
    timeline_lookup: dict[str, list[dict]] | None = None,
    use_case_lookup: dict[str, list[dict]] | None = None,
    complaint_lookup: dict[str, list[dict]] | None = None,
    positive_lookup: dict[str, list[dict]] | None = None,
    department_lookup: dict[str, list[dict]] | None = None,
    contract_value_lookup: dict[str, list[dict]] | None = None,
    turning_point_lookup: dict[str, list[dict]] | None = None,
    tenure_lookup: dict[str, list[dict]] | None = None,
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

    # -- Merge multi-category rows into one per vendor --------------
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

    # -- Build enriched scorecard per vendor -------------------------
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
        # Idle field enrichments (Tier 1: wired from existing lookups)
        tl_entries = (timeline_lookup or {}).get(vendor, [])
        eval_deadlines = [t for t in tl_entries if t.get("evaluation_deadline")]
        if eval_deadlines:
            sc_entry["upcoming_evaluation_deadlines"] = eval_deadlines[:5]
        uc_entries = (use_case_lookup or {}).get(vendor, [])
        if uc_entries:
            sc_entry["product_depth"] = uc_entries[:10]
        # Tier 3: new aggregation lookups
        complaints = (complaint_lookup or {}).get(vendor, [])
        if complaints:
            sc_entry["top_complaints"] = complaints[:5]
        positives = (positive_lookup or {}).get(vendor, [])
        if positives:
            sc_entry["retention_signals"] = positives[:5]
        departments = (department_lookup or {}).get(vendor, [])
        if departments:
            sc_entry["department_distribution"] = departments[:5]
        deal_sizes = (contract_value_lookup or {}).get(vendor, [])
        if deal_sizes:
            sc_entry["deal_size_distribution"] = deal_sizes
        triggers = (turning_point_lookup or {}).get(vendor, [])
        if triggers:
            sc_entry["churn_triggers"] = triggers[:5]
        tenures = (tenure_lookup or {}).get(vendor, [])
        if tenures:
            sc_entry["customer_tenure_profile"] = tenures[:5]

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

        # -- Vendor rankings (all vendors in category ranked by churn pressure) --
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

        # -- Case studies (top quotes per category with company context) --
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

        # -- Top feature gaps (aggregated across category) --
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
    timeline_lookup: dict[str, list[dict]] | None = None,
    use_case_lookup: dict[str, list[dict]] | None = None,
    positive_lookup: dict[str, list[dict]] | None = None,
    department_lookup: dict[str, list[dict]] | None = None,
    usage_duration_lookup: dict[str, list[dict]] | None = None,
    buyer_auth_lookup: dict[str, dict] | None = None,
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

    # Pick the primary category row per vendor (most reviews) instead of
    # summing across categories, which inflates totals and averages.
    merged: dict[str, dict[str, Any]] = {}
    for row in vendor_scores:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        reviews = int(row.get("total_reviews") or 0)
        churn = int(row.get("churn_intent") or 0)
        urgency = float(row.get("avg_urgency") or 0)
        category = row.get("product_category") or "Unknown"
        if vendor not in merged or reviews > merged[vendor]["total_reviews"]:
            merged[vendor] = {
                "total_reviews": reviews,
                "churn_intent": churn,
                "avg_urgency": urgency,
                "category": category,
            }

    cards: list[dict[str, Any]] = []
    for vendor, m in merged.items():
        total_reviews = m["total_reviews"]
        churn_intent = m["churn_intent"]
        churn_density = round(churn_intent * 100.0 / total_reviews, 1) if total_reviews else 0.0
        avg_urgency = round(m["avg_urgency"], 1)
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

        # -- Section 2: Customer Pain Quotes (deduplicated by review + reviewer) --
        quotes_raw = sorted(
            quote_lookup.get(vendor, []),
            key=_battle_card_quote_sort_key,
            reverse=True,
        )
        pain_quotes: list[dict[str, Any]] = []
        seen_review_ids: set[str] = set()
        seen_reviewers: set[str] = set()
        for q in quotes_raw:
            if len(pain_quotes) >= 5:
                break
            if isinstance(q, dict):
                # Dedupe by review_id (exact same review)
                rid = q.get("review_id", "")
                if rid and rid in seen_review_ids:
                    continue
                # Dedupe by reviewer identity (same person, different reviews)
                reviewer_key = (
                    f"{(q.get('company') or '').lower().strip()}"
                    f":{(q.get('title') or '').lower().strip()}"
                )
                if reviewer_key != ":" and reviewer_key in seen_reviewers:
                    continue
                if rid:
                    seen_review_ids.add(rid)
                if reviewer_key != ":":
                    seen_reviewers.add(reviewer_key)
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
        # Exclude "unknown" to find the dominant *known* sentiment direction
        known_sentiment = {k: v for k, v in sentiment_counts.items() if k != "unknown"}
        if total_reviews < 10 or not known_sentiment:
            sentiment_dir = "insufficient_data"
        else:
            sentiment_dir = max(known_sentiment.items(), key=lambda x: x[1])[0]

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

        # -- Section 5: High-intent companies --
        hi_companies = company_lookup.get(vendor, [])[:5]

        # -- Section 6: Integration stack --
        integrations = (product_profile_lookup.get(vendor, {}).get("top_integrations") or [])[:8]

        # -- Section 7: Buyer authority summary --
        buyer_authority = (buyer_auth_lookup or {}).get(vendor, {})

        # -- Budget: prefer median for "typical" size --
        budget_ctx = budget_lookup.get(vendor, {})

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
            "high_intent_companies": hi_companies,
            "integration_stack": integrations,
            "buyer_authority": buyer_authority or None,
            # Populated by LLM pass in run():
            "objection_handlers": [],
            "recommended_plays": [],
        }
        # Idle field enrichments for battle cards
        positives = (positive_lookup or {}).get(vendor, [])
        if positives:
            card_entry["retention_signals"] = positives[:5]
        tl_entries = (timeline_lookup or {}).get(vendor, [])
        eval_deadlines = [t for t in tl_entries if t.get("evaluation_deadline")]
        if eval_deadlines:
            card_entry["active_evaluation_deadlines"] = eval_deadlines[:5]
        uc_entries = (use_case_lookup or {}).get(vendor, [])
        if uc_entries:
            card_entry["objection_data"]["product_depth"] = uc_entries[:5]
        departments = (department_lookup or {}).get(vendor, [])
        if departments:
            card_entry["objection_data"]["department_context"] = departments[:3]
        durations = (usage_duration_lookup or {}).get(vendor, [])
        if durations:
            card_entry["objection_data"]["tenure_churn_pattern"] = durations[:5]

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
