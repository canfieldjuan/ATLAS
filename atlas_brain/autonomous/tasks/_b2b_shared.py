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
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

from ...config import settings
from ...services.apollo_company_overrides import fetch_company_override_map
from ...services.b2b.corrections import suppress_predicate
from ...services.company_normalization import normalize_company_name
from ...services.scraping.sources import (
    parse_source_allowlist,
    filter_deprecated_sources,
    display_name as _source_display_name,
    VERIFIED_SOURCES,
)
from ...services.tracing import build_business_trace_context
from ...services.vendor_registry import resolve_vendor_name_cached

logger = logging.getLogger("atlas.autonomous.tasks.b2b_shared")

_INTELLIGENCE_ELIGIBLE_STATUSES: tuple[str, ...] = (
    "enriched",
    "no_signal",
    "quarantined",
)


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



def filter_vendors_by_focus_categories(
    vendor_scores: list[dict],
    focus_categories_raw: str,
) -> list[dict]:
    """Filter vendor_scores to only include vendors in focus categories.

    Returns the full list unchanged when focus is 'all' or empty.
    Category matching is case-insensitive.
    """
    raw = (focus_categories_raw or "").strip().lower()
    if not raw or raw == "all":
        return vendor_scores
    focus = {c.strip() for c in raw.split(",") if c.strip()}
    if not focus:
        return vendor_scores
    return [
        vs for vs in vendor_scores
        if (vs.get("product_category") or "").strip().lower() in focus
    ]


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


# Overreaching absolute phrases paired with evidence-calibrated replacements.
# Checked in the validator (raises a warning) and in the sanitizer (auto-replaced).
_BATTLE_CARD_OVERREACH_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("inability to execute", "execution risk is emerging"),
    ("shows an inability to", "shows emerging risk of"),
    ("inability to deliver", "delivery risk is present"),
    ("strongest loser", "losing momentum in evaluated segments"),
    ("biggest loser", "losing momentum in evaluated segments"),
    ("clear loser", "losing ground in evaluated segments"),
    ("failing across the board", "showing broad evaluation pressure"),
    ("failing at scale", "facing scaling pressure"),
    ("customers are fleeing", "buyers are actively evaluating alternatives"),
    ("customers are abandoning", "buyers are evaluating alternatives"),
    ("customers are leaving in droves", "buyers are evaluating alternatives at scale"),
    ("is collapsing", "is under increasing pressure"),
    ("is crumbling", "is showing structural weakness"),
    ("cannot execute", "faces execution risk"),
    ("has lost the market", "is losing momentum in evaluated segments"),
)


def _battle_card_overreach_violations(text: str) -> list[str]:
    """Return banned overreach phrases found in text."""
    lowered = text.lower()
    return [phrase for phrase, _ in _BATTLE_CARD_OVERREACH_REPLACEMENTS if phrase in lowered]


def _battle_card_replace_overreach(text: str) -> str:
    """Replace overreaching phrases with evidence-calibrated language."""
    result = text
    for phrase, replacement in _BATTLE_CARD_OVERREACH_REPLACEMENTS:
        lowered = result.lower()
        if phrase not in lowered:
            continue
        idx = lowered.index(phrase)
        # Preserve sentence casing: if phrase starts a sentence, capitalize replacement.
        if idx == 0 or lowered[idx - 1] in ".!?\n":
            replacement = replacement[0].upper() + replacement[1:]
        result = result[:idx] + replacement + result[idx + len(phrase):]
    return result


def _synthesis_expert_take_max_words() -> int:
    """Return the configured max word count for scorecard expert_take."""
    return int(settings.b2b_churn.synthesis_expert_take_max_words)


def _evidence_vault_supporting_review_limit() -> int:
    """Return the max supporting-review IDs stored per evidence item."""
    return int(settings.b2b_churn.intelligence_evidence_vault_supporting_review_limit)


def _evidence_vault_segment_limit() -> int:
    """Return the max affected segments stored per evidence item."""
    return int(settings.b2b_churn.intelligence_evidence_vault_segment_limit)


def _evidence_vault_role_limit() -> int:
    """Return the max affected roles stored per evidence item."""
    return int(settings.b2b_churn.intelligence_evidence_vault_role_limit)


def _evidence_vault_trend_accelerating_ratio() -> float:
    """Return the ratio needed to mark a trend as accelerating."""
    return float(settings.b2b_churn.intelligence_evidence_vault_trend_accelerating_ratio)


def _evidence_vault_trend_declining_ratio() -> float:
    """Return the ratio at or below which a trend is declining."""
    return float(settings.b2b_churn.intelligence_evidence_vault_trend_declining_ratio)


def _evidence_vault_trend_new_min_recent() -> int:
    """Return the minimum recent mentions needed to call a trend new."""
    return int(settings.b2b_churn.intelligence_evidence_vault_trend_new_min_recent)


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

# Archetype-specific weight overrides. When vendor reasoning identifies
# an archetype-style churn pattern, the corresponding weight set is used
# instead of the default, biasing the score toward the most relevant signal.
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
          AND duplicate_of_review_id IS NULL
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


_PAIN_SIGNAL_TERMS: frozenset[str] = frozenset([
    "expensive", "pricey", "costly", "overpriced",
    "difficult", "frustrating", "complicated", "confusing", "clunky",
    "crashes", "crash", "bugs", "broken", "unreliable", "outage",
    "poor", "terrible", "awful", "disappointing", "subpar",
    "canceling", "cancelling", "switching to", "switched to",
    "migrating to", "as a replacement", "looking to replace",
    "looking for alternative", "considering switching",
    "evaluating alternatives", "evaluating a replacement",
    "unfortunately",
    "not user-friendly", "not intuitive", "not great", "not good",
    "unable to", "needs improvement",
    "missing feature", "limited feature",
    "price increase", "prices have", "pricing issues", "pricing concerns",
    "issues with", "problems with",
])

_POSITIVE_ONLY_PHRASES: tuple[str, ...] = (
    "perfect score",
    "rate customer support a perfect",
    "customer support is good",
    "support is excellent",
    "support is amazing",
    "support is outstanding",
    "very positive experience",
    "highly recommend",
    "no complaints",
    "no issues",
    "love everything about",
    "could not be happier",
    "best tool",
    "best crm",
)


def _quote_has_pain_signal(
    quote_text: str,
    urgency: float = 0.0,
    rating: float | None = None,
    rating_max: float | None = None,
) -> bool:
    """Return True if the quote is appropriate for a pain/weakness section.

    Rejects clearly-positive testimonials. High-urgency quotes (>= 6.5) are
    kept even when ambiguous to avoid discarding thin evidence pools.
    """
    lowered = quote_text.lower().strip()
    if not lowered:
        return False
    if rating is not None and rating_max and float(rating_max) > 0:
        if float(rating) / float(rating_max) >= 0.80:
            return False
    for phrase in _POSITIVE_ONLY_PHRASES:
        if phrase in lowered:
            if not any(t in lowered for t in _PAIN_SIGNAL_TERMS):
                return False
    if any(t in lowered for t in _PAIN_SIGNAL_TERMS):
        return True
    return urgency >= 6.5


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
    return filter_deprecated_sources(
        parse_source_allowlist(settings.b2b_churn.intelligence_source_allowlist),
        settings.b2b_churn.deprecated_review_sources,
    )


def _executive_source_list() -> list[str]:
    """Return curated executive sources for headline-facing queries."""
    return filter_deprecated_sources(
        parse_source_allowlist(settings.b2b_churn.intelligence_executive_sources),
        settings.b2b_churn.deprecated_review_sources,
    )


def _company_signal_skip_sources() -> set[str]:
    """Return sources that should never seed canonical named-account artifacts."""
    blocked: set[str] = set()
    if settings.b2b_churn.company_signal_skip_deprecated_sources:
        blocked.update(
            parse_source_allowlist(settings.b2b_scrape.deprecated_sources)
        )
    return blocked


def _company_signal_low_trust_sources() -> set[str]:
    """Return configured low-trust sources for canonical company signals."""
    return {
        str(source).strip().lower()
        for source in (settings.b2b_churn.company_signal_low_trust_sources or [])
        if str(source).strip()
    }


def _normalize_company_signal_confidence(value: Any) -> float | None:
    """Normalize confidence values to a 0-1 unit interval."""
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0:
        confidence /= 10.0
    return round(min(1.0, max(0.0, confidence)), 3)


def _eligible_review_timestamp_expr(*, alias: str = "") -> str:
    """Stable review-occurrence timestamp for intelligence windows."""
    p = f"{alias}." if alias else ""
    return f"COALESCE({p}reviewed_at, {p}imported_at, {p}enriched_at)"


def _eligible_review_filters(*, window_param: int | None = 1, source_param: int = 2, alias: str = "") -> str:
    """Build a reusable SQL predicate for eligible intelligence review rows.

    When *alias* is set (e.g. ``"r"``), column references are prefixed
    with the table alias so the predicate works inside JOINed queries.
    """
    p = f"{alias}." if alias else ""
    time_expr = _eligible_review_timestamp_expr(alias=alias)
    status_list = ", ".join(
        f"'{status}'"
        for status in _INTELLIGENCE_ELIGIBLE_STATUSES
        if re.fullmatch(r"[a-z_]+", status)
    )
    parts = [f"{p}enrichment_status IN ({status_list})"]
    parts.append(f"{p}duplicate_of_review_id IS NULL")
    if window_param is not None:
        parts.append(f"{time_expr} > NOW() - make_interval(days => ${window_param})")
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


async def _fetch_review_funnel_audit(pool, window_days: int) -> dict[str, Any]:
    """Return end-to-end review funnel counts for the active intelligence sources."""
    sources = _intelligence_source_allowlist()
    eligible_filters = _eligible_review_filters(window_param=1, source_param=2)
    status_rows = await pool.fetch(
        """
        SELECT enrichment_status, count(*) AS ct
        FROM b2b_reviews
        WHERE duplicate_of_review_id IS NULL
          AND source = ANY($1::text[])
          AND COALESCE(reviewed_at, imported_at, enriched_at) > NOW() - make_interval(days => $2)
        GROUP BY enrichment_status
        """,
        sources,
        window_days,
    )
    status_counts = {
        str(row["enrichment_status"] or ""): int(row["ct"] or 0)
        for row in status_rows
    }
    eligible_row = await pool.fetchrow(
        f"""
        SELECT
            count(*) AS intelligence_eligible_reviews,
            count(*) FILTER (
                WHERE reviewer_company_norm IS NOT NULL
                  AND reviewer_company_norm <> ''
            ) AS company_signal_eligible_reviews,
            count(*) FILTER (
                WHERE reviewer_company_norm IS NOT NULL
                  AND reviewer_company_norm <> ''
                  AND source <> ALL($3::text[])
            ) AS high_confidence_named_account_reviews
        FROM b2b_reviews
        WHERE {eligible_filters}
        """,
        window_days,
        sources,
        list(_company_signal_low_trust_sources()),
    )

    def _row_count(key: str) -> int:
        if not eligible_row:
            return 0
        return int((eligible_row[key] or 0))

    intelligence_eligible_reviews = _row_count("intelligence_eligible_reviews")
    company_signal_eligible_reviews = _row_count("company_signal_eligible_reviews")
    high_conf_named_account_reviews = _row_count("high_confidence_named_account_reviews")
    return {
        "found": sum(status_counts.values()),
        "enriched": status_counts.get("enriched", 0),
        "no_signal": status_counts.get("no_signal", 0),
        "quarantined": status_counts.get("quarantined", 0),
        "raw_only": status_counts.get("raw_only", 0),
        "pending": status_counts.get("pending", 0),
        "failed": status_counts.get("failed", 0),
        "not_applicable": status_counts.get("not_applicable", 0),
        "duplicate": status_counts.get("duplicate", 0),
        "intelligence_eligible": intelligence_eligible_reviews,
        "company_signal_eligible": company_signal_eligible_reviews,
        "high_confidence_named_account": high_conf_named_account_reviews,
    }


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
        or path.startswith("recommended_plays[")
        or path.startswith("talk_track.")
    )


def _battle_card_headline_paths(path: str) -> bool:
    """Return True for top-line summary fields that should stay on strong evidence."""
    return path == "executive_summary" or path.startswith("weakness_analysis[0].")


def _battle_card_numeric_tokens(text: str) -> set[str]:
    """Extract numeric tokens from narrative sections for validation."""
    tokens: set[str] = set()
    for match in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?", text or ""):
        token = match.group(0)
        start, end = match.span()
        next_char = text[end] if end < len(text or "") else ""
        next_next = text[end + 1] if end + 1 < len(text or "") else ""
        normalized = token.replace(",", "").rstrip("%")
        # Ignore UI-style singleton compounds like "1-click" that are not
        # economic or corpus-level claims. Multi-digit timeline claims such as
        # "12-month" still flow through validation.
        if len(normalized) == 1 and next_char == "-" and next_next.isalpha():
            continue
        tokens.add(token)
    return tokens


def _battle_card_normalize_numeric_token(token: str) -> str:
    """Canonicalize equivalent numeric strings like 27 and 27.0%."""
    text = str(token or "").strip()
    if not text:
        return text
    has_pct = text.endswith("%")
    raw = text[:-1] if has_pct else text
    raw = raw.replace(",", "")
    try:
        numeric = float(raw)
    except ValueError:
        return text
    if numeric.is_integer():
        base = str(int(numeric))
    else:
        base = f"{numeric:.1f}".rstrip("0").rstrip(".")
    return f"{base}%" if has_pct else base


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


def _battle_card_add_wrapper_claim(claims: set[str], wrapper: Any) -> None:
    """Add numeric tokens from a {value, source_id} wrapper when possible."""
    if not isinstance(wrapper, dict):
        return
    _battle_card_add_claim(claims, wrapper.get("value"))


def _battle_card_add_text_numeric_claims(claims: set[str], text: Any) -> None:
    """Add normalized numeric tokens pulled from witness-backed text."""
    excerpt = str(text or "")
    if not excerpt:
        return
    for token in re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", excerpt):
        normalized = _battle_card_normalize_numeric_token(token.lstrip("$"))
        if normalized:
            claims.add(normalized)


def _battle_card_add_witness_numeric_claims(claims: set[str], witness_blob: Any) -> None:
    """Import numeric claims from raw witness/anchor payloads."""
    if isinstance(witness_blob, dict):
        if "excerpt_text" in witness_blob or "quote" in witness_blob:
            _battle_card_add_text_numeric_claims(claims, witness_blob.get("excerpt_text"))
            _battle_card_add_text_numeric_claims(claims, witness_blob.get("quote"))
            return
        for value in witness_blob.values():
            _battle_card_add_witness_numeric_claims(claims, value)
        return
    if isinstance(witness_blob, list):
        for item in witness_blob:
            _battle_card_add_witness_numeric_claims(claims, item)
        return
    return


def _battle_card_allowed_claims(card: dict[str, Any]) -> set[str]:
    """Build the set of numeric claims supported by deterministic card input."""
    claims: set[str] = set()
    _battle_card_add_claim(claims, card.get("total_reviews"))
    _battle_card_add_claim(claims, card.get("churn_pressure_score"))
    acct = card.get("account_pressure_metrics") or {}
    for key in ("total_accounts", "active_eval_count", "high_intent_count"):
        _battle_card_add_claim(claims, acct.get(key))
    for item in card.get("high_intent_companies") or []:
        if isinstance(item, dict):
            _battle_card_add_claim(claims, item.get("urgency"))
    data = card.get("objection_data") or {}
    for key in ("price_complaint_rate", "dm_churn_rate"):
        _battle_card_add_claim(claims, data.get(key), pct=True)
    for key in ("churn_signal_density", "avg_urgency", "total_reviews"):
        _battle_card_add_claim(claims, data.get(key))
    for item in card.get("vendor_weaknesses") or []:
        _battle_card_add_claim(claims, item.get("evidence_count") or item.get("count"))
    for item in _battle_card_aggregated_competitors(card).values():
        _battle_card_add_claim(claims, item.get("mentions"))
        _battle_card_add_claim(claims, item.get("switch_count"))
    for item in data.get("top_feature_gaps") or []:
        _battle_card_add_claim(claims, item.get("mentions"))
    for key in ("avg_seat_count", "max_seat_count", "median_seat_count", "price_increase_count"):
        _battle_card_add_claim(claims, (data.get("budget_context") or {}).get(key))
    _battle_card_add_claim(claims, (data.get("budget_context") or {}).get("price_increase_rate"), pct=True)
    contracts = card.get("reasoning_contracts")
    if isinstance(contracts, dict):
        vendor_core = contracts.get("vendor_core_reasoning") or {}
        if isinstance(vendor_core, dict):
            timing = vendor_core.get("timing_intelligence") or {}
            if isinstance(timing, dict):
                _battle_card_add_wrapper_claim(claims, timing.get("active_eval_signals"))
            segments = vendor_core.get("segment_playbook") or {}
            if isinstance(segments, dict):
                for segment in segments.get("priority_segments") or []:
                    if not isinstance(segment, dict):
                        continue
                    _battle_card_add_wrapper_claim(claims, segment.get("estimated_reach"))
                    _battle_card_add_claim(claims, segment.get("sample_size"))
        account_reasoning = contracts.get("account_reasoning") or {}
        if isinstance(account_reasoning, dict):
            _battle_card_add_wrapper_claim(claims, account_reasoning.get("total_accounts"))
            _battle_card_add_wrapper_claim(claims, account_reasoning.get("active_eval_count"))
            _battle_card_add_wrapper_claim(claims, account_reasoning.get("high_intent_count"))
        displacement = contracts.get("displacement_reasoning") or {}
        if isinstance(displacement, dict):
            migration = displacement.get("migration_proof") or {}
            if isinstance(migration, dict):
                _battle_card_add_wrapper_claim(claims, migration.get("switch_volume"))
                _battle_card_add_wrapper_claim(claims, migration.get("active_evaluation_volume"))
                _battle_card_add_wrapper_claim(claims, migration.get("displacement_mention_volume"))
            reframes = displacement.get("competitive_reframes") or {}
            if isinstance(reframes, dict):
                for item in reframes.get("reframes") or []:
                    if not isinstance(item, dict):
                        continue
                    proof = item.get("proof_point") or {}
                    if isinstance(proof, dict):
                        _battle_card_add_claim(claims, proof.get("value"))
        category = contracts.get("category_reasoning") or {}
        if isinstance(category, dict):
            _battle_card_add_claim(claims, category.get("vendor_count"))
            _battle_card_add_claim(claims, category.get("displacement_flow_count"))
    _battle_card_add_witness_numeric_claims(claims, card.get("anchor_examples"))
    _battle_card_add_witness_numeric_claims(claims, card.get("witness_highlights"))
    return claims


def _battle_card_contract(card: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a battle-card reasoning contract from canonical storage."""
    contracts = card.get("reasoning_contracts")
    if isinstance(contracts, dict):
        contract = contracts.get(name)
        if isinstance(contract, dict) and contract:
            return contract
        if contracts:
            return {}
    return {}


def _battle_card_contract_section(
    card: dict[str, Any],
    contract_name: str,
    section_name: str,
    flat_name: str,
) -> dict[str, Any]:
    """Resolve a section from contracts first, then flat compatibility fields."""
    contract = _battle_card_contract(card, contract_name)
    if contract:
        section = contract.get(section_name)
        if isinstance(section, dict) and section:
            return section
    contracts = card.get("reasoning_contracts")
    if isinstance(contracts, dict) and contracts:
        return {}
    flat = card.get(flat_name)
    if isinstance(flat, dict) and flat:
        return flat
    return {}


def _battle_card_validator_source(card: dict[str, Any]) -> dict[str, Any]:
    """Return the contract-first source-of-truth view for validation.

    The renderer now consumes a compact contract-first packet. Validation should
    use the same authoritative view so stale flat mirrors do not expand the set
    of accepted claims.
    """
    base_keys = (
        "vendor",
        "category",
        "churn_pressure_score",
        "risk_level",
        "total_reviews",
        "confidence",
        "vendor_weaknesses",
        "customer_pain_quotes",
        "competitor_differentiators",
        "weakness_analysis",
        "competitive_landscape",
        "archetype",
        "synthesis_wedge",
        "synthesis_wedge_label",
        "archetype_risk_level",
        "archetype_key_signals",
        "evidence_depth_warning",
        "objection_data",
        "cross_vendor_battles",
        "category_council",
        "resource_asymmetry",
        "ecosystem_context",
        "high_intent_companies",
        "integration_stack",
        "buyer_authority",
        "keyword_spikes",
        "retention_signals",
        "active_evaluation_deadlines",
        "falsification_conditions",
        "uncertainty_sources",
        "evidence_window",
        "evidence_window_days",
        "reasoning_source",
        "synthesis_schema_version",
        "render_packet_version",
    )
    source = {
        key: card[key]
        for key in base_keys
        if key in card
    }
    reasoning_contracts = card.get("reasoning_contracts")
    if isinstance(reasoning_contracts, dict) and reasoning_contracts:
        source["reasoning_contracts"] = reasoning_contracts

    for contract_name, section_name, flat_name in (
        ("vendor_core_reasoning", "causal_narrative", "causal_narrative"),
        ("vendor_core_reasoning", "segment_playbook", "segment_playbook"),
        ("vendor_core_reasoning", "timing_intelligence", "timing_intelligence"),
        ("displacement_reasoning", "competitive_reframes", "competitive_reframes"),
        ("displacement_reasoning", "migration_proof", "migration_proof"),
    ):
        section = _battle_card_contract_section(card, contract_name, section_name, flat_name)
        if section:
            source[flat_name] = section
    return source


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


def _battle_card_segment_evidence_is_thin(card: dict[str, Any]) -> bool:
    """Return True when segment targeting should stay explicitly tentative."""
    if card.get("high_intent_companies"):
        return False
    segment_playbook = _battle_card_contract_section(
        card,
        "vendor_core_reasoning",
        "segment_playbook",
        "segment_playbook",
    )
    if not isinstance(segment_playbook, dict) or not segment_playbook:
        return False
    data_gaps = [
        str(item).strip().lower()
        for item in (segment_playbook.get("data_gaps") or [])
        if str(item).strip()
    ]
    return any("no account-level intelligence" in gap for gap in data_gaps)


def _battle_card_segment_playbook(card: dict[str, Any]) -> dict[str, Any]:
    """Return the contract-first segment playbook section."""
    return _battle_card_contract_section(
        card,
        "vendor_core_reasoning",
        "segment_playbook",
        "segment_playbook",
    )


_SEGMENT_STRATEGIC_ROLE_TYPES = frozenset((
    "decision_maker",
    "economic_buyer",
    "champion",
    "evaluator",
))

_SEGMENT_ROLE_LABELS: dict[str, str] = {
    "decision_maker": "decision-makers",
    "economic_buyer": "economic buyers",
    "champion": "internal champions",
    "evaluator": "evaluators",
    "end_user": "end users",
}

_SEGMENT_DEPARTMENT_LABELS: dict[str, str] = {
    "it": "IT",
    "hr": "HR",
    "qa": "QA",
    "bi": "BI",
}

_SEGMENT_GENERIC_CONTRACTS = frozenset((
    "smb",
    "mid market",
    "enterprise",
    "enterprise mid",
    "enterprise high",
))

_SEGMENT_SAFE_USE_CASE_ACRONYMS = frozenset((
    "crm", "erp", "hr", "hris", "it", "bi", "qa", "seo", "sms",
    "etl", "api", "csm", "plg", "okr", "kpi",
))

_SEGMENT_PRIORITY_LABEL_OVERRIDES: dict[str, str] = {
    "small business": "Small Business",
    "mid market": "Mid-Market",
}

_SEGMENT_OPENING_ANGLE_LOWERCASE_WORDS = frozenset((
    "advanced",
    "benchmark",
    "compare",
    "cost",
    "demonstrate",
    "emphasize",
    "feature",
    "focus",
    "frame",
    "highlight",
    "lead",
    "license",
    "offer",
    "open",
    "position",
    "reliable",
    "show",
    "simplified",
    "target",
    "use",
))

_TIMING_GENERIC_TRIGGER_PATTERNS = (
    "active evaluation signal",
    "active evaluation signals",
    "signal detected",
    "price increase signal",
    "price increase count spike",
    "timeline signal",
    "contract renewal deadline",
    "within quarter evaluation deadline",
)


def _clean_segment_label(value: Any) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "").strip())
    return re.sub(r"\s+", " ", text).strip()


def _join_segment_labels(labels: list[str]) -> str:
    items = [str(label).strip() for label in labels if str(label).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _segment_role_display_label(role_type: Any) -> str:
    role_name = str(role_type or "").strip() or "unknown"
    return _SEGMENT_ROLE_LABELS.get(role_name, _clean_segment_label(role_name) or "buyers")


def _segment_priority_display_label(value: Any) -> str:
    text = re.sub(r"\([^)]*\d[^)]*\)", "", _clean_segment_label(value)).strip(" -")
    if not text:
        return ""
    lower = re.sub(r"\s+role$", "", text.lower()).strip()
    role_key = lower.replace(" ", "_")
    if role_key in _SEGMENT_ROLE_LABELS:
        return _segment_role_display_label(role_key)
    override = _SEGMENT_PRIORITY_LABEL_OVERRIDES.get(lower)
    if override:
        return override
    return text


def _segment_opening_angle_phrase(value: Any) -> str:
    text = _clean_segment_label(value)
    if not text:
        return ""
    first_word = text.split(" ", 1)[0].strip().lower()
    if first_word in _SEGMENT_OPENING_ANGLE_LOWERCASE_WORDS:
        return text[:1].lower() + text[1:]
    return text


def _segment_playbook_supporting_list(
    segment_playbook: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    if not isinstance(segment_playbook, dict):
        return []
    supporting = segment_playbook.get("supporting_evidence") or {}
    items = supporting.get(key) or []
    return [dict(item) for item in items if isinstance(item, dict)]


def _segment_role_sort_value(item: dict[str, Any]) -> tuple[float, float, float, str]:
    role_type = str(item.get("role_type") or "").strip() or "unknown"
    try:
        priority = float(item.get("priority_score") or 0)
    except (TypeError, ValueError):
        priority = 0.0
    try:
        review_count = float(item.get("review_count") or 0)
    except (TypeError, ValueError):
        review_count = 0.0
    return (priority, review_count, _SEGMENT_ROLE_SCORES.get(role_type, 0.0), role_type)


def _segment_playbook_strategic_roles(
    segment_playbook: dict[str, Any],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    roles = _segment_playbook_supporting_list(segment_playbook, "top_strategic_roles")
    if not roles:
        roles = [
            item for item in _segment_playbook_supporting_list(segment_playbook, "top_roles")
            if str(item.get("role_type") or "").strip() in _SEGMENT_STRATEGIC_ROLE_TYPES
        ]
    roles.sort(key=_segment_role_sort_value, reverse=True)
    return roles[:limit]


def _segment_playbook_context_label(
    segment_playbook: dict[str, Any],
    key: str,
    field: str,
) -> str:
    items = _segment_playbook_supporting_list(segment_playbook, key)
    if not items:
        return ""
    return _clean_segment_label(items[0].get(field))


def _segment_department_label(value: Any) -> str:
    text = _clean_segment_label(value).lower()
    if not text:
        return ""
    return _SEGMENT_DEPARTMENT_LABELS.get(text, text.title())


def _segment_context_key(value: Any) -> str:
    text = _clean_segment_label(value).lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _segment_contract_clause(segment_playbook: dict[str, Any]) -> tuple[int, str]:
    text = _segment_playbook_context_label(segment_playbook, "top_contract_segments", "segment")
    if not text:
        return (0, "")
    priority = 35 if text.lower() in _SEGMENT_GENERIC_CONTRACTS else 50
    return (priority, f"{text} contracts")


def _segment_safe_use_case_label(value: Any) -> str:
    text = _clean_segment_label(value)
    if not text:
        return ""
    raw_tokens = [tok for tok in re.split(r"\s+", text) if tok]
    if not raw_tokens:
        return ""
    has_title_case = any(
        any(ch.isupper() for ch in tok[1:]) or (tok[:1].isupper() and tok[1:].islower())
        for tok in raw_tokens
    )
    if has_title_case:
        return ""
    normalized = text.lower()
    compact = re.sub(r"[^a-z0-9/+-]", "", normalized)
    if compact in _SEGMENT_SAFE_USE_CASE_ACRONYMS:
        return text.upper()
    words = [tok for tok in re.split(r"[^a-z0-9/+-]+", normalized) if tok]
    if not words:
        return ""
    if len(words) == 1 and len(words[0]) > 12:
        return ""
    return text


def _segment_context_clauses(
    segment_playbook: dict[str, Any],
    *,
    limit: int = 2,
) -> list[tuple[str, str]]:
    candidates: list[tuple[int, str, str]] = []
    department = _segment_department_label(
        _segment_playbook_context_label(segment_playbook, "top_departments", "department"),
    )
    if department:
        candidates.append((90, "in", f"{department} teams"))
    size = _segment_playbook_context_label(segment_playbook, "top_company_sizes", "segment")
    size_key = _segment_context_key(size)
    if size:
        candidates.append((80, "in", f"{size} accounts"))
    safe_use_case = ""
    for item in _segment_playbook_supporting_list(segment_playbook, "top_use_cases"):
        safe_use_case = _segment_safe_use_case_label(item.get("use_case"))
        if safe_use_case:
            break
    if safe_use_case:
        candidates.append((65, "plain", f"for {safe_use_case} workflows"))
    duration = _segment_playbook_context_label(segment_playbook, "top_usage_durations", "duration")
    if duration:
        candidates.append((55, "plain", f"after {duration} of usage"))
    contract_priority, contract_clause = _segment_contract_clause(segment_playbook)
    contract_key = _segment_context_key(
        _segment_playbook_context_label(segment_playbook, "top_contract_segments", "segment"),
    )
    if contract_clause and contract_key != size_key:
        candidates.append((contract_priority, "in", contract_clause))
    candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
    clauses: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, mode, clause in candidates:
        norm = clause.lower()
        if norm in seen:
            continue
        seen.add(norm)
        clauses.append((mode, clause))
        if len(clauses) >= limit:
            break
    return clauses


def _segment_best_timing_sentence(timing_intelligence: dict[str, Any] | None) -> str:
    if not isinstance(timing_intelligence, dict):
        return ""
    window = _clean_segment_label(timing_intelligence.get("best_timing_window"))
    if not window:
        return ""
    lower = window[:1].lower() + window[1:] if window else ""
    if re.match(r"^(during|before|after|when|within|immediately|at)\b", lower):
        return f"Best tested {lower}."
    return f"Best tested during {lower}."


def _segment_targeting_summary(
    segment_playbook: dict[str, Any],
    timing_intelligence: dict[str, Any] | None = None,
) -> str:
    roles = _segment_playbook_strategic_roles(segment_playbook, limit=2)
    if roles:
        labels = [_segment_role_display_label(item.get("role_type")) for item in roles]
        contexts = _segment_context_clauses(segment_playbook, limit=2)
        summary = f"Strongest current pressure is surfacing with {_join_segment_labels(labels)}"
        in_clauses = [text for mode, text in contexts if mode == "in"]
        plain_clauses = [text for mode, text in contexts if mode == "plain"]
        if in_clauses and plain_clauses:
            sentence = (
                f"{summary}, especially in {_join_segment_labels(in_clauses)}, "
                f"and {_join_segment_labels(plain_clauses)}."
            )
        elif in_clauses:
            sentence = f"{summary}, especially in {_join_segment_labels(in_clauses)}."
        elif plain_clauses:
            sentence = f"{summary}, especially {_join_segment_labels(plain_clauses)}."
        else:
            sentence = summary + "."
        timing = _segment_best_timing_sentence(timing_intelligence)
        context_texts = [text for _, text in contexts]
        if timing and (not context_texts or all("contracts" in ctx.lower() for ctx in context_texts)):
            return f"{sentence} {timing}"
        return sentence
    for segment in segment_playbook.get("priority_segments") or []:
        if not isinstance(segment, dict):
            continue
        name = _segment_priority_display_label(segment.get("segment"))
        angle = _segment_opening_angle_phrase(segment.get("best_opening_angle"))
        if name and angle:
            timing = _segment_best_timing_sentence(timing_intelligence)
            sentence = f"Best current segment wedge is {name}, led with {angle}."
            return f"{sentence} {timing}".strip() if timing else sentence
        if name:
            timing = _segment_best_timing_sentence(timing_intelligence)
            sentence = f"Best current segment wedge is {name}."
            return f"{sentence} {timing}".strip() if timing else sentence
    return ""


def _timing_wrapper_int(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _timing_wrapper_float(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_sentence(text: Any) -> str:
    sentence = str(text or "").strip()
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _timing_window_conflicts_with_active_eval(
    best_window: Any,
    active_eval: int | None,
) -> bool:
    if active_eval is None or active_eval <= 0:
        return False
    text = str(best_window or "").strip().lower()
    if not text:
        return False
    if text.startswith("no strong timing signal"):
        return True
    if "no active evaluation" in text:
        return True
    if text.startswith("none") and "no" in text:
        return True
    return False


def _timing_summary_key(text: Any) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = re.sub(r"[_-]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9% ]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _timing_window_is_generic(best_window: Any) -> bool:
    text = _timing_summary_key(best_window)
    if not text:
        return False
    if re.search(r"\b(after|before|following|during|when)\b", text):
        return False
    return text.startswith((
        "immediate",
        "immediately",
        "within",
        "next 30 days",
        "current quarter",
        "q4",
        "this week",
    ))


def _timing_trigger_is_generic(label: Any) -> bool:
    text = _timing_summary_key(label)
    if not text:
        return True
    return any(pattern in text for pattern in _TIMING_GENERIC_TRIGGER_PATTERNS)


def _timing_trigger_sentence(
    best_window: Any,
    trigger_labels: list[str],
) -> str:
    best_key = _timing_summary_key(best_window)
    if not trigger_labels:
        return ""
    for label in trigger_labels:
        text = str(label or "").strip()
        key = _timing_summary_key(text)
        if not key or _timing_trigger_is_generic(text):
            continue
        if best_key and (key in best_key or best_key in key):
            continue
        return f"Key trigger: {_summary_sentence(text).rstrip('.') }."
    return ""


def _timing_summary_payload(
    timing_intelligence: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], list[str]]:
    if not isinstance(timing_intelligence, dict) or not timing_intelligence:
        return ("", {}, [])

    metrics: dict[str, Any] = {}
    triggers = timing_intelligence.get("immediate_triggers") or []
    trigger_labels: list[str] = []
    seen_triggers: set[str] = set()
    for item in triggers:
        if not isinstance(item, dict):
            continue
        label = str(item.get("trigger") or item.get("label") or "").strip()
        if not label:
            continue
        norm = label.casefold()
        if norm in seen_triggers:
            continue
        seen_triggers.add(norm)
        trigger_labels.append(label)
        if len(trigger_labels) >= 3:
            break
    if triggers:
        metrics["immediate_trigger_count"] = len(triggers)

    active_eval = _timing_wrapper_int(timing_intelligence.get("active_eval_signals"))
    if active_eval is not None:
        metrics["active_eval_signals"] = active_eval

    supporting = timing_intelligence.get("supporting_evidence") or {}
    timeline = supporting.get("timeline_signal_summary") or {}
    for field in (
        "evaluation_deadline_signals",
        "contract_end_signals",
        "renewal_signals",
        "budget_cycle_signals",
    ):
        value = _timing_wrapper_int(timeline.get(field))
        if value is not None:
            metrics[field] = value

    sentiment_direction = str(
        timing_intelligence.get("sentiment_direction") or ""
    ).strip().lower()
    if sentiment_direction:
        metrics["sentiment_direction"] = sentiment_direction

    sentiment = supporting.get("sentiment_snapshot") or {}
    for field in ("declining_pct", "improving_pct"):
        value = _timing_wrapper_float(sentiment.get(field))
        if value is not None:
            metrics[field] = round(value, 2)

    best_window = str(timing_intelligence.get("best_timing_window") or "").strip()
    summary_parts: list[str] = []
    if best_window and not _timing_window_conflicts_with_active_eval(best_window, active_eval):
        summary_parts.append(_summary_sentence(best_window))
    trigger_sentence = ""
    if not best_window or _timing_window_is_generic(best_window):
        trigger_sentence = _timing_trigger_sentence(best_window, trigger_labels)
    if trigger_sentence:
        summary_parts.append(trigger_sentence)
    if active_eval is not None and active_eval > 0:
        summary_parts.append(
            f"{active_eval} active evaluation signals are visible right now."
        )
    elif metrics.get("immediate_trigger_count"):
        count = int(metrics["immediate_trigger_count"])
        summary_parts.append(
            f"{count} immediate timing triggers are currently open."
        )

    if sentiment_direction == "declining":
        summary_parts.append("Review sentiment is skewing more negative.")
    elif sentiment_direction == "improving":
        summary_parts.append(
            "Review sentiment is improving, so outreach should stay tied to concrete events."
        )

    return (" ".join(summary_parts).strip(), metrics, trigger_labels)


def _battle_card_timing_intelligence(card: dict[str, Any]) -> dict[str, Any]:
    """Return the contract-first timing intelligence section."""
    return _battle_card_contract_section(
        card,
        "vendor_core_reasoning",
        "timing_intelligence",
        "timing_intelligence",
    )


def _battle_card_primary_weakness(card: dict[str, Any]) -> str:
    """Return the strongest weakness label available for safe seller phrasing."""
    weaknesses = card.get("vendor_weaknesses") or []
    if weaknesses and isinstance(weaknesses[0], dict):
        area = str(
            weaknesses[0].get("area")
            or weaknesses[0].get("weakness")
            or ""
        ).strip().lower()
        if area:
            return area
    return "fit and value"


def _battle_card_structured_proof_text(card: dict[str, Any]) -> str:
    """Return the strongest supported proof-point sentence for seller copy."""
    timing = _battle_card_contract_section(
        card,
        "vendor_core_reasoning",
        "timing_intelligence",
        "timing_intelligence",
    )
    active_eval = {}
    if isinstance(timing, dict):
        active_eval = timing.get("active_eval_signals") or {}
    if isinstance(active_eval, dict):
        value = active_eval.get("value")
        try:
            count = int(round(float(value)))
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            weakness = _battle_card_primary_weakness(card)
            if weakness:
                return f"{count} active evaluation signals show recurring buyer pressure around {weakness}."
            return f"{count} active evaluation signals show recurring buyer pressure."

    weaknesses = card.get("vendor_weaknesses") or []
    if weaknesses and isinstance(weaknesses[0], dict):
        top = weaknesses[0]
        area = str(top.get("area") or top.get("weakness") or "customer fit").strip().lower()
        count = top.get("evidence_count") or top.get("count")
        try:
            count_int = int(round(float(count)))
        except (TypeError, ValueError):
            count_int = 0
        if count_int > 0:
            return f"{count_int} recurring complaints point to buyer friction around {area}."

    return "The input shows recurring buyer friction and credible evaluation pressure."


def _battle_card_safe_play_text(card: dict[str, Any], path: str) -> str:
    """Return grounded fallback text for recommended plays and talk tracks."""
    weakness = _battle_card_primary_weakness(card)
    index_match = re.search(r"\[(\d+)\]", path)
    index = int(index_match.group(1)) if index_match else 0
    if path.endswith(".target_segment"):
        if weakness == "pricing":
            return "Support, finance, and operations teams already feeling pricing pressure."
        if weakness == "support":
            return "Support leaders already dealing with service friction and renewal pressure."
        return "Teams already showing evaluation pressure around fit and value."
    if path.endswith(".key_message"):
        if weakness == "pricing":
            variants = [
                "Lead with pricing clarity, spend control, and fewer forced add-ons.",
                "Lead with packaging clarity, budget predictability, and less renewal surprise.",
                "Lead with tighter spend control, clearer packaging, and a cleaner renewal story.",
            ]
            return variants[index % len(variants)]
        if weakness == "support":
            variants = [
                "Lead with more responsive support operations and clearer accountability.",
                "Lead with faster escalations, cleaner ownership, and less day-to-day support friction.",
                "Lead with stronger service responsiveness and clearer accountability at renewal.",
            ]
            return variants[index % len(variants)]
        variants = [
            "Lead with a simpler path to better fit, lower friction, and clearer value.",
            "Lead with cleaner fit, less operational drag, and a clearer path to value.",
            "Lead with lower friction, stronger day-to-day fit, and easier value validation.",
        ]
        return variants[index % len(variants)]
    if path.endswith(".timing"):
        timing_variants = [
            "Best tested during active evaluation windows, renewal review, or planning cycles.",
            "Best timed to renewal planning, budget review, or any fresh fit-and-value checkpoint.",
            "Use this when buyers are reassessing fit, budgets, or switching timing ahead of the next renewal motion.",
        ]
        return timing_variants[index % len(timing_variants)]
    if path.endswith(".play"):
        prefix = "Best tested on" if _battle_card_segment_evidence_is_thin(card) else "Target"
        if weakness == "pricing":
            return f"{prefix} teams facing pricing pressure with a focused pricing and packaging benchmark."
        if weakness == "support":
            return f"{prefix} support teams with a support-operations benchmark against current escalation pain."
        return f"{prefix} teams showing fit and renewal pressure with a focused benchmark."
    if path.startswith("talk_track."):
        if path.endswith(".opening"):
            vendor = str(card.get("vendor") or "the incumbent").strip() or "the incumbent"
            anchor_phrase = _battle_card_anchor_phrase_from_card(card)
            opening = (
                f"Buyers are actively pressure-testing {vendor} because {weakness} concerns keep resurfacing during evaluation."
            )
            if anchor_phrase:
                opening = f"{opening} The clearest live signal is coming from {anchor_phrase}."
            return opening
        if path.endswith(".mid_call_pivot"):
            return "Once pain is confirmed, pivot to the recurring friction and evaluation pressure visible in the current evidence."
        if path.endswith(".closing"):
            return "Close with a working session to benchmark current fit, costs, and switching timing before renewal."
    return ""


def _battle_card_normalize_recommended_play_segment(value: Any) -> str:
    """Normalize a recommended-play target segment for duplicate detection."""
    segment = str(value or "").strip().lower()
    segment = re.sub(r"\s*\(sample n=\d+\)\s*$", "", segment)
    return re.sub(r"\s+", " ", segment).strip()


def _battle_card_normalize_recommended_play_text(value: Any) -> str:
    """Normalize recommended-play text for duplicate detection."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[.]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _repair_battle_card_recommended_play_duplicates(card: dict[str, Any], generated: dict[str, Any]) -> None:
    """Deterministically diversify duplicate recommended-play rows before retry."""
    plays = generated.get("recommended_plays")
    if not isinstance(plays, list) or not plays:
        return
    fallback_candidates = _battle_card_fallback_recommended_plays(
        card,
        limit=max(len(plays) + 2, 2),
    )
    repaired: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    seen_plays: set[str] = set()
    seen_messages: set[str] = set()

    def _row_signals(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            _battle_card_normalize_recommended_play_segment(row.get("target_segment")),
            _battle_card_normalize_recommended_play_text(row.get("play")),
            _battle_card_normalize_recommended_play_text(row.get("key_message")),
        )

    def _next_fallback() -> dict[str, Any] | None:
        for candidate in fallback_candidates:
            if not isinstance(candidate, dict):
                continue
            segment_key, play_key, message_key = _row_signals(candidate)
            if segment_key and segment_key in seen_segments:
                continue
            if play_key and play_key in seen_plays:
                continue
            if message_key and message_key in seen_messages:
                continue
            return dict(candidate)
        return None

    for idx, item in enumerate(plays):
        row = dict(item) if isinstance(item, dict) else {}
        if not row:
            continue
        segment_key, play_key, message_key = _row_signals(row)
        if segment_key and segment_key in seen_segments:
            replacement = _next_fallback()
            if replacement is not None:
                row = replacement
                segment_key, play_key, message_key = _row_signals(row)
        if play_key and play_key in seen_plays:
            row["play"] = _battle_card_safe_play_text(card, f"recommended_plays[{idx}].play").rstrip(".") + "."
            play_key = _battle_card_normalize_recommended_play_text(row.get("play"))
        if message_key and message_key in seen_messages:
            row["key_message"] = _battle_card_safe_play_text(card, f"recommended_plays[{idx}].key_message").rstrip(".") + "."
            message_key = _battle_card_normalize_recommended_play_text(row.get("key_message"))
        if (segment_key and segment_key in seen_segments) or (play_key and play_key in seen_plays) or (message_key and message_key in seen_messages):
            replacement = _next_fallback()
            if replacement is not None:
                row = replacement
                segment_key, play_key, message_key = _row_signals(row)
        repaired.append(row)
        if segment_key:
            seen_segments.add(segment_key)
        if play_key:
            seen_plays.add(play_key)
        if message_key:
            seen_messages.add(message_key)
    if repaired:
        generated["recommended_plays"] = repaired


def _battle_card_role_target_segment(
    segment_playbook: dict[str, Any],
    role: dict[str, Any],
) -> str:
    role_label = _segment_role_display_label(role.get("role_type"))
    contexts = _segment_context_clauses(segment_playbook, limit=1)
    if contexts:
        mode, clause = contexts[0]
        if mode == "plain":
            return f"{role_label} {clause}"
        return f"{role_label} in {clause}"
    return role_label


def _battle_card_role_opening_angle(card: dict[str, Any], role_type: str) -> str:
    weakness = _battle_card_primary_weakness(card)
    if role_type in {"decision_maker", "economic_buyer"} and weakness == "pricing":
        return "a finance-ready pricing and renewal benchmark"
    if role_type in {"decision_maker", "economic_buyer"}:
        return "a decision-ready fit, risk, and renewal benchmark"
    if role_type == "champion":
        return "an adoption and internal-alignment review"
    if role_type == "evaluator":
        return "a side-by-side evaluation on fit and switching friction"
    return "a focused benchmark on fit and value"


def _battle_card_role_key_message(card: dict[str, Any], role_type: str) -> str:
    weakness = _battle_card_primary_weakness(card)
    if role_type in {"decision_maker", "economic_buyer"} and weakness == "pricing":
        return "Lead with pricing predictability, spend control, and fewer surprise costs"
    if role_type in {"decision_maker", "economic_buyer"}:
        return "Lead with clearer vendor accountability, lower friction, and more predictable value"
    if role_type == "champion":
        return "Lead with smoother adoption, fewer workarounds, and less internal pushback"
    if role_type == "evaluator":
        return "Lead with faster evaluation clarity, cleaner validation, and fewer edge-case surprises"
    return _battle_card_safe_play_text(card, "recommended_plays[0].key_message").rstrip(".")


def _battle_card_account_stage_timing(stage: Any, default_timing: str) -> str:
    normalized = _normalize_buying_stage(stage)
    if normalized == "renewal_decision":
        return "Before the next renewal decision checkpoint."
    if normalized == "evaluation":
        return "During active evaluation and renewal review."
    if normalized == "consideration":
        return "As the evaluation moves into shortlist review."
    if normalized == "procurement":
        return "Before procurement locks the renewal plan."
    return default_timing.rstrip(".") + "."


def _battle_card_safe_fallback_timing(value: Any) -> str:
    text = str(value or "").strip()
    default = "Best tested during active evaluation windows, renewal review, or planning cycles."
    if not text:
        return default
    if len(text) > 140:
        return default
    if re.search(r"\b20\d{2}\b", text):
        return default
    if any(token in text.lower() for token in ("march ", "april ", "may ", "june ", "july ", "august ", "september ", "october ", "november ", "december ", "january ", "february ")):
        return default
    return text.rstrip(".") + "."


def _battle_card_account_play_text(card: dict[str, Any], company: str) -> str:
    weakness = _battle_card_primary_weakness(card)
    if weakness == "pricing":
        return f"Run a pricing benchmark workshop with {company} before renewal pressure hardens."
    if weakness == "support":
        return f"Run a support-risk review with {company} before the next renewal checkpoint."
    return f"Run a fit-and-risk benchmark with {company} before the next evaluation checkpoint."


def _battle_card_generic_fallback_roles(card: dict[str, Any]) -> list[dict[str, str]]:
    weakness = _battle_card_primary_weakness(card)
    if weakness == "pricing":
        return [
            {"role_type": "economic_buyer"},
            {"role_type": "evaluator"},
        ]
    if weakness == "support":
        return [
            {"role_type": "champion"},
            {"role_type": "economic_buyer"},
        ]
    return [
        {"role_type": "evaluator"},
        {"role_type": "economic_buyer"},
    ]


def _battle_card_fallback_recommended_plays(
    card: dict[str, Any],
    *,
    limit: int = 2,
) -> list[dict[str, str]]:
    """Build deterministic recommended-play fallbacks from segment contracts."""
    segment_playbook = _battle_card_segment_playbook(card)
    timing = _battle_card_timing_intelligence(card)
    best_timing_window = str(timing.get("best_timing_window") or "").strip()
    default_timing = _battle_card_safe_fallback_timing(best_timing_window)
    thin = _battle_card_segment_evidence_is_thin(card)
    prefix = "Best tested on" if thin else "Target"
    plays: list[dict[str, str]] = []
    seen_segments: set[str] = set()
    for segment in segment_playbook.get("priority_segments") or []:
        if not isinstance(segment, dict):
            continue
        segment_name = str(segment.get("segment") or "").strip()
        if not segment_name:
            continue
        display_segment = re.sub(r"\([^)]*\d[^)]*\)", "", segment_name).strip()
        display_segment = re.sub(r"\s+", " ", display_segment).strip()
        norm = re.sub(r"\s+", " ", display_segment.lower()).strip()
        if not norm or norm in seen_segments:
            continue
        seen_segments.add(norm)
        opening = str(segment.get("best_opening_angle") or "").strip()
        sample_size = None
        try:
            sample_size = int(round(float(segment.get("sample_size"))))
        except (TypeError, ValueError):
            sample_size = None
        segment_label = display_segment
        if sample_size and sample_size > 0:
            segment_label = f"{display_segment} (sample n={sample_size})"
        play_text = f"{prefix} {display_segment} with {opening or 'a focused benchmark on fit and value'}"
        target_text = segment_label
        key_message = opening or _battle_card_safe_play_text(card, "recommended_plays[0].key_message")
        timing_text = default_timing
        plays.append({
            "play": play_text.rstrip(".") + ".",
            "target_segment": target_text,
            "key_message": key_message.rstrip(".") + ".",
            "timing": timing_text.rstrip(".") + ".",
        })
        if len(plays) >= limit:
            break
    for role in _segment_playbook_strategic_roles(segment_playbook, limit=limit):
        if len(plays) >= limit:
            break
        role_type = str(role.get("role_type") or "").strip()
        target_text = _battle_card_role_target_segment(segment_playbook, role)
        norm = re.sub(r"\s+", " ", target_text.lower()).strip()
        if not norm or norm in seen_segments:
            continue
        seen_segments.add(norm)
        plays.append({
            "play": f"{prefix} {target_text} with {_battle_card_role_opening_angle(card, role_type)}.",
            "target_segment": target_text,
            "key_message": _battle_card_role_key_message(card, role_type).rstrip(".") + ".",
            "timing": default_timing.rstrip(".") + ".",
        })
    if len(plays) < limit:
        for account in _rank_high_intent_companies(card.get("high_intent_companies") or []):
            if len(plays) >= limit:
                break
            if not isinstance(account, dict):
                continue
            company = str(account.get("company") or account.get("company_name") or "").strip()
            if not company:
                continue
            stage = account.get("buying_stage") or account.get("stage")
            target_text = f"{company} renewal stakeholders"
            norm = re.sub(r"\s+", " ", target_text.lower()).strip()
            if not norm or norm in seen_segments:
                continue
            seen_segments.add(norm)
            plays.append({
                "play": _battle_card_account_play_text(card, company),
                "target_segment": target_text,
                "key_message": _battle_card_safe_play_text(card, "recommended_plays[0].key_message").rstrip(".") + ".",
                "timing": _battle_card_account_stage_timing(stage, default_timing),
            })
    for role in _battle_card_generic_fallback_roles(card):
        if len(plays) >= limit:
            break
        role_type = str(role.get("role_type") or "").strip()
        if not role_type:
            continue
        target_text = _battle_card_role_target_segment(segment_playbook, role)
        norm = re.sub(r"\s+", " ", target_text.lower()).strip()
        if not norm or norm in seen_segments:
            continue
        seen_segments.add(norm)
        plays.append({
            "play": f"{prefix} {target_text} with {_battle_card_role_opening_angle(card, role_type)}.",
            "target_segment": target_text,
            "key_message": _battle_card_role_key_message(card, role_type).rstrip(".") + ".",
            "timing": default_timing.rstrip(".") + ".",
        })
    return plays


def _battle_card_has_duplicate_recommended_play_segments(generated: dict[str, Any]) -> bool:
    """Return True when recommended plays collapse onto the same target segment."""
    plays = generated.get("recommended_plays")
    if not isinstance(plays, list):
        return False
    seen: set[str] = set()
    for item in plays:
        if not isinstance(item, dict):
            continue
        segment = str(item.get("target_segment") or "").strip().lower()
        segment = re.sub(r"\s*\(sample n=\d+\)\s*$", "", segment)
        segment = re.sub(r"\s+", " ", segment)
        if not segment:
            continue
        if segment in seen:
            return True
        seen.add(segment)
    return False


def _battle_card_safe_text(card: dict[str, Any], path: str) -> str:
    """Return grounded replacement text for numeric-sensitive paths."""
    if path == "executive_summary":
        return _battle_card_safe_summary(card)
    if path.startswith("objection_handlers[") and path.endswith(".pivot"):
        weakness = _battle_card_primary_weakness(card)
        return (
            f"The better question is whether {weakness} is creating enough drag to justify a cleaner alternative before renewal."
        )
    if path.startswith("why_they_stay.strengths["):
        index_match = re.search(r"\[(\d+)\]", path)
        index = int(index_match.group(1)) if index_match else 0
        neutralize_variants = [
            "Acknowledge the familiar setup, then redirect to the renewal risks already showing up in current evaluation motion.",
            "Reframe the conversation around operational predictability, simpler administration, and less day-to-day friction.",
            "Keep the focus on reducing switching risk, spend surprises, and the manual work buyers are already flagging.",
        ]
        evidence_variants = [
            "Customers still cite familiar workflows as a reason to stay.",
            "Some teams still value the incumbent because the current setup feels established and good enough.",
            "Retention signals show the incumbent still gets credit for familiarity and continuity.",
        ]
        if path.endswith(".how_to_neutralize"):
            return neutralize_variants[index % len(neutralize_variants)]
        if path.endswith(".evidence"):
            return evidence_variants[index % len(evidence_variants)]
        if path.endswith(".summary"):
            return "The incumbent still holds on where familiarity and continuity outweigh switching effort."
    if path.endswith(".evidence"):
        return "Supported by recurring customer complaints and churn-oriented review evidence."
    if path.endswith(".proof_point"):
        base = _battle_card_structured_proof_text(card).rstrip(".")
        weakness = _battle_card_primary_weakness(card)
        index_match = re.search(r"\[(\d+)\]", path)
        index = int(index_match.group(1)) if index_match else 0
        variants = [
            f"{base}.",
            f"{base}, which keeps reinforcing buyer scrutiny around {weakness}.",
            f"{base}. That pattern is showing up again in current evaluation motion.",
            f"{base}. It is one of the clearest signals behind current switching pressure.",
        ]
        return variants[index % len(variants)]
    if path.startswith("recommended_plays[") or path.startswith("talk_track."):
        return _battle_card_safe_play_text(card, path)
    if path.startswith("competitive_landscape."):
        competitors = _battle_card_competitor_names(card)
        if "top_alternatives" in path and competitors:
            return (
                "Alternatives appearing most often in buyer evaluation sets include "
                f"{_join_summary_terms(competitors)}."
            )
        return "Competitive pressure is present where buyers are re-evaluating fit and value."
    return ""


def _battle_card_anchor_phrase_from_card(card: dict[str, Any]) -> str:
    """Return a seller-safe anchor phrase from raw anchor_examples."""
    raw = card.get("anchor_examples")
    if not isinstance(raw, dict):
        return ""
    for rows in raw.values():
        if not isinstance(rows, list):
            continue
        for witness in rows:
            if not isinstance(witness, dict):
                continue
            company = str(witness.get("reviewer_company") or "").strip()
            competitor = str(witness.get("competitor") or "").strip()
            time_anchor = _battle_card_specific_time_anchor(witness.get("time_anchor"))
            parts: list[str] = []
            if company:
                parts.append(f"accounts like {company}")
            if competitor:
                parts.append(f"while evaluating {competitor}")
            if time_anchor:
                parts.append(f"during {time_anchor}")
            excerpt = str(witness.get("excerpt_text") or "")
            numeric_tokens = re.findall(r"\$\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?%", excerpt)
            if numeric_tokens:
                preview = ", ".join(numeric_tokens[:3])
                if parts:
                    parts.append(f"with pricing callouts like {preview}")
                else:
                    parts.append(f"pricing callouts like {preview} in current review evidence")
            if parts:
                return " ".join(parts)
    return ""


def _battle_card_anchor_terms_from_card(card: dict[str, Any]) -> set[str]:
    """Collect non-numeric anchor terms from raw anchor_examples."""
    raw = card.get("anchor_examples")
    if not isinstance(raw, dict):
        return set()
    terms: set[str] = set()
    for rows in raw.values():
        if not isinstance(rows, list):
            continue
        for witness in rows:
            if not isinstance(witness, dict):
                continue
            for value in (
                witness.get("reviewer_company"),
                witness.get("competitor"),
                witness.get("time_anchor"),
            ):
                if value == witness.get("time_anchor"):
                    term = _battle_card_specific_time_anchor(value).lower()
                else:
                    term = str(value or "").strip().lower()
                if term:
                    terms.add(term)
            excerpt = str(witness.get("excerpt_text") or "")
            for token in re.findall(r"\$\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?%", excerpt):
                normalized = token.strip().lower()
                if normalized:
                    terms.add(normalized)
    return terms


def _battle_card_render_text_from_generated(generated: dict[str, Any]) -> str:
    """Flatten generated seller copy into comparable lowercase text."""
    return " ".join(text for _, text in _battle_card_iter_text(generated) if text).lower()


def _battle_card_set_generated_path(generated: dict[str, Any], path: str, value: str) -> None:
    """Set a nested generated path like talk_track.opening or objection_handlers[1].proof_point."""
    current: Any = generated
    tokens = re.findall(r"([^\.\[\]]+)|\[(\d+)\]", path)
    if not tokens:
        return
    for idx, (key_token, index_token) in enumerate(tokens):
        is_last = idx == len(tokens) - 1
        if index_token:
            if not isinstance(current, list):
                return
            index = int(index_token)
            if index >= len(current):
                return
            if is_last:
                current[index] = value
                return
            current = current[index]
            continue
        key = str(key_token)
        if not isinstance(current, dict) or key not in current:
            return
        if is_last:
            current[key] = value
            return
        current = current[key]


def _repair_battle_card_duplicate_copy(card: dict[str, Any], generated: dict[str, Any]) -> None:
    """Replace duplicate long-form sections with grounded fallback phrasing."""
    long_strings: list[tuple[str, str]] = []
    for path, text in _battle_card_iter_text(generated):
        stripped = text.strip()
        if len(stripped) >= 80:
            long_strings.append((path, stripped))
    seen_prefixes: dict[str, str] = {}
    for path, text in long_strings:
        prefix = text[:80].lower()
        first_path = seen_prefixes.get(prefix)
        if first_path and first_path != path:
            replacement = _battle_card_safe_text(card, path)
            if replacement and replacement.strip().lower() != text.lower():
                _battle_card_set_generated_path(generated, path, replacement)
        else:
            seen_prefixes[prefix] = path


def _repair_battle_card_missing_anchor(card: dict[str, Any], generated: dict[str, Any]) -> None:
    """Inject a witness-backed anchor when seller copy stays too generic."""
    anchor_phrase = _battle_card_anchor_phrase_from_card(card)
    anchor_terms = _battle_card_anchor_terms_from_card(card)
    if not anchor_phrase or not anchor_terms:
        return
    render_text = _battle_card_render_text_from_generated(generated)
    raw = card.get("anchor_examples")
    companies: set[str] = set()
    competitor_terms: set[str] = set()
    timing_terms: set[str] = set()
    numeric_terms: set[str] = set()
    if isinstance(raw, dict):
        for rows in raw.values():
            if not isinstance(rows, list):
                continue
            for witness in rows:
                if not isinstance(witness, dict):
                    continue
                company = str(witness.get("reviewer_company") or "").strip().lower()
                competitor = str(witness.get("competitor") or "").strip().lower()
                time_anchor = _battle_card_specific_time_anchor(witness.get("time_anchor")).lower()
                if company:
                    companies.add(company)
                if competitor:
                    competitor_terms.add(competitor)
                if time_anchor:
                    timing_terms.add(time_anchor)
                excerpt = str(witness.get("excerpt_text") or "")
                for token in re.findall(r"\$\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?%", excerpt):
                    normalized = token.strip().lower()
                    if normalized:
                        numeric_terms.add(normalized)
    anchor_categories_missing = any((
        companies and not any(term in render_text for term in companies),
        competitor_terms and not any(term in render_text for term in competitor_terms),
        timing_terms and not any(term in render_text for term in timing_terms),
        numeric_terms and not any(term in render_text for term in numeric_terms),
    ))
    if not anchor_categories_missing and any(term in render_text for term in anchor_terms):
        return
    anchor_sentence = f"The clearest live signal is coming from {anchor_phrase}."
    summary = str(generated.get("executive_summary") or "").strip()
    if summary:
        generated["executive_summary"] = f"{summary.rstrip('.')} {anchor_sentence}".strip()
        return
    talk_track = generated.get("talk_track")
    if isinstance(talk_track, dict):
        opening = str(talk_track.get("opening") or "").strip()
        if opening:
            talk_track["opening"] = f"{opening.rstrip('.')} {anchor_sentence}".strip()


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
    normalized_allowed_claims = {
        _battle_card_normalize_numeric_token(token) for token in (allowed_claims or set())
    }
    if path.startswith("recommended_plays[") and path.endswith(".timing"):
        if _battle_card_numeric_tokens(cleaned):
            cleaned = _battle_card_safe_text(card, path)
    if _battle_card_numeric_paths(path):
        bad = sorted(
            tok
            for tok in _battle_card_numeric_tokens(cleaned)
            if _battle_card_normalize_numeric_token(tok) not in normalized_allowed_claims
        )
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
    if _battle_card_segment_evidence_is_thin(card) and path.endswith(".play"):
        cleaned = re.sub(r"^\s*target\b", "Best tested on", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*engage\b", "Best tested on", cleaned, flags=re.IGNORECASE)
    if _battle_card_headline_paths(path):
        for term in low_gap_terms:
            if term and term in cleaned.lower():
                return _battle_card_safe_summary(card) if path == "executive_summary" else "Workflow friction is showing up in customer feedback."
    if _battle_card_overreach_violations(cleaned):
        cleaned = _battle_card_replace_overreach(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _battle_card_allowed_quotes(card: dict[str, Any]) -> list[str]:
    """Return exact customer pain quotes that synthesis may reuse."""
    quotes: list[str] = []
    for item in card.get("customer_pain_quotes") or []:
        if isinstance(item, dict):
            quote = str(item.get("quote") or "").strip()
        else:
            quote = str(item or "").strip()
        if quote and quote not in quotes:
            quotes.append(quote)
    return quotes


def _battle_card_quote_entries(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unique quote entries with any attached metadata preserved."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in card.get("customer_pain_quotes") or []:
        if isinstance(item, dict):
            quote = str(item.get("quote") or "").strip()
            urgency = float(item.get("urgency") or 0)
            rating = item.get("rating")
            rating_max = item.get("rating_max")
        else:
            quote = str(item or "").strip()
            urgency = 0.0
            rating = None
            rating_max = None
        if quote and quote not in seen:
            seen.add(quote)
            entries.append({
                "quote": quote,
                "urgency": urgency,
                "rating": rating,
                "rating_max": rating_max,
            })
    return entries


def _battle_card_best_supported_quote(
    card: dict[str, Any],
    context: str = "",
    *,
    preferred_terms: list[str] | None = None,
    excluded_quotes: set[str] | None = None,
    require_preferred_match: bool = False,
) -> str:
    """Pick the most relevant exact quote from source data for a weakness context."""
    entries = _battle_card_quote_entries(card)
    if excluded_quotes:
        entries = [entry for entry in entries if entry["quote"] not in excluded_quotes]
    entries = [
        e for e in entries
        if _quote_has_pain_signal(
            e["quote"],
            urgency=float(e.get("urgency") or 0),
            rating=e.get("rating"),
            rating_max=e.get("rating_max"),
        )
    ]
    if not entries:
        return ""
    context_tokens = {
        token for token in re.findall(r"[a-z0-9]+", context.lower())
        if len(token) >= 4
    }
    preferred = [term.lower() for term in (preferred_terms or []) if term]
    ranked = sorted(
        entries,
        key=lambda entry: (
            sum(1 for term in preferred if term in entry["quote"].lower()),
            len(context_tokens & set(re.findall(r"[a-z0-9]+", entry["quote"].lower()))),
            float(entry.get("urgency") or 0),
            len(entry["quote"]),
        ),
        reverse=True,
    )
    if preferred and require_preferred_match:
        best = ranked[0]
        if not any(term in best["quote"].lower() for term in preferred):
            return ""
    return ranked[0]["quote"]


def _battle_card_weakness_headline(area: str, *, source: str = "") -> str:
    """Map a weakness area into a rep-facing battle-card wedge."""
    lower = area.lower()
    if any(token in lower for token in ("price", "pricing", "cost", "budget")):
        return "Pricing pressure is creating renewal scrutiny"
    if any(token in lower for token in ("support", "service", "response", "help")):
        return "Support friction is undermining buyer confidence"
    if any(token in lower for token in ("reliability", "uptime", "outage", "stability", "performance")):
        return "Reliability issues are increasing switching risk"
    if source == "feature_gap" or any(token in lower for token in ("feature", "workflow", "reporting", "automation")):
        return "Feature gaps are pushing teams to evaluate alternatives"
    if any(token in lower for token in ("integration", "plugin", "api", "stack")):
        return "Integration complexity is adding operational drag"
    return f"{area.title()} concerns keep resurfacing in buyer feedback"


def _battle_card_winning_position(area: str, *, source: str = "") -> str:
    """Return a vendor-agnostic capability emphasis for a weakness."""
    lower = area.lower()
    if any(token in lower for token in ("price", "pricing", "cost", "budget")):
        return "Emphasize transparent pricing, packaging clarity, and spend controls."
    if any(token in lower for token in ("support", "service", "response", "help")):
        return "Emphasize responsive support operations and clearer accountability on escalations."
    if any(token in lower for token in ("reliability", "uptime", "outage", "stability", "performance")):
        return "Emphasize reliability, incident response discipline, and operational resilience."
    if source == "feature_gap" or any(token in lower for token in ("feature", "workflow", "reporting", "automation")):
        return "Emphasize broader native capability coverage and fewer workaround-heavy workflows."
    if any(token in lower for token in ("integration", "plugin", "api", "stack")):
        return "Emphasize native integrations, simpler administration, and less app sprawl."
    return "Emphasize predictable operations, easier adoption, and lower day-to-day friction."


def _battle_card_weakness_evidence(card: dict[str, Any], weakness: dict[str, Any]) -> str:
    """Build a deterministic evidence line for a weakness entry."""
    area = str(weakness.get("area") or weakness.get("weakness") or "").strip()
    source = str(weakness.get("source") or "").strip()
    count = int(weakness.get("evidence_count") or weakness.get("count") or 0)
    data = card.get("objection_data") or {}
    total_reviews = int(card.get("total_reviews") or data.get("total_reviews") or 0)
    if any(token in area.lower() for token in ("price", "pricing", "cost", "budget")) and data.get("price_complaint_rate") is not None:
        return f"{float(data['price_complaint_rate']):.1%} price complaint rate across {total_reviews:,} reviews"
    for gap in data.get("top_feature_gaps") or []:
        if str(gap.get("feature") or "").strip().lower() == area.lower():
            mentions = int(gap.get("mentions") or 0)
            return f"{mentions} feature-gap mentions tied to {area}"
    if source == "product_profile" and weakness.get("score") is not None:
        return f"Satisfaction score {float(weakness['score']):.1f} with {count} supporting reviews"
    if count:
        unit = "reviews" if source == "product_profile" else "mentions"
        return f"{count} supporting {unit} tied to {area}"
    return f"Recurring customer friction tied to {area}"


def _battle_card_quote_terms(area: str, *, source: str = "") -> list[str]:
    """Return quote-preference keywords for a weakness area."""
    lower = area.lower()
    if any(token in lower for token in ("price", "pricing", "cost", "budget")):
        return ["cost", "costs", "price", "pricing", "budget", "spend"]
    if any(token in lower for token in ("support", "service", "response", "help")):
        return ["support", "service", "response", "escalation", "help"]
    if any(token in lower for token in ("reliability", "uptime", "outage", "stability", "performance")):
        return ["outage", "uptime", "reliability", "stability", "incident", "performance", "downtime", "down"]
    if source == "feature_gap" or any(token in lower for token in ("feature", "workflow", "reporting", "automation")):
        return ["feature", "workflow", "reporting", "automation", "capability"]
    if any(token in lower for token in ("integration", "plugin", "api", "stack")):
        return ["integration", "plugin", "api", "stack"]
    return []


def _battle_card_specific_time_anchor(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if re.search(r"\d", lower):
        return text
    if any(
        term in lower
        for term in (
            "renewal",
            "review",
            "planning",
            "quarter",
            "month",
            "week",
            "deadline",
            "window",
            "cycle",
            "decision",
        )
    ):
        return text
    return ""


def _battle_card_competitor_bucket_key(raw_label: str, buckets: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Return a stable dedupe key and display label for a competitor mention."""
    label = _canonicalize_competitor(raw_label) or raw_label
    norm = normalize_company_name(label)
    if not norm:
        return label, label
    for existing_norm, bucket in buckets.items():
        if existing_norm == norm:
            return existing_norm, str(bucket.get("competitor") or label)
        if existing_norm.endswith(f" {norm}") or norm.endswith(f" {existing_norm}"):
            existing_label = str(bucket.get("competitor") or label)
            display_label = label if len(label) > len(existing_label) else existing_label
            return existing_norm, display_label
    return norm, label


def _battle_card_competitor_is_eligible(label: str) -> bool:
    """Return True when a competitor label is seller-usable in battle cards."""
    lower = str(label or "").strip().lower()
    if not lower:
        return False
    blocked_terms = (
        " integration",
        "integrations",
        "plugin",
        "plug-in",
        "add-on",
        "addon",
        "workflow",
        "workaround",
        "custom ",
        "internal ",
        "in-house",
    )
    return not any(term in lower for term in blocked_terms)


def _battle_card_aggregated_competitors(card: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Aggregate seller-usable competitor counts for battle-card surfaces."""
    aggregated_competitors: dict[str, dict[str, Any]] = {}
    for item in card.get("competitor_differentiators") or []:
        if not isinstance(item, dict):
            continue
        raw_label = str(item.get("competitor") or "").strip()
        bucket_key, label = _battle_card_competitor_bucket_key(raw_label, aggregated_competitors)
        if not label or not bucket_key or not _battle_card_competitor_is_eligible(label):
            continue
        bucket = aggregated_competitors.setdefault(
            bucket_key,
            {"competitor": label, "mentions": 0, "switch_count": 0, "driver_counts": Counter()},
        )
        if len(label) > len(str(bucket.get("competitor") or "")):
            bucket["competitor"] = label
        bucket["mentions"] += int(item.get("mentions") or 0)
        bucket["switch_count"] += int(item.get("switch_count") or 0)
        driver = str(item.get("primary_driver") or "buyer fit").strip()
        if driver:
            bucket["driver_counts"][driver] += max(int(item.get("mentions") or 0), 1)
    return aggregated_competitors


def _sanitize_battle_card_sales_copy(card: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """Deterministically rewrite near-miss sales copy before final rejection."""
    if not isinstance(generated, dict):
        return {}
    allowed = _battle_card_allowed_claims(card)
    source_text = json.dumps(_battle_card_validator_source(card), default=str).lower()
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

    sanitized = _walk(generated)
    if isinstance(sanitized, dict) and _battle_card_has_duplicate_recommended_play_segments(sanitized):
        fallback_plays = _battle_card_fallback_recommended_plays(card)
        if fallback_plays:
            sanitized["recommended_plays"] = fallback_plays
    if isinstance(sanitized, dict):
        _repair_battle_card_recommended_play_duplicates(card, sanitized)
        _repair_battle_card_duplicate_copy(card, sanitized)
        _repair_battle_card_missing_anchor(card, sanitized)
    allowed_quotes = _battle_card_allowed_quotes(card)
    pain_quote_meta: dict[str, dict[str, Any]] = {}
    for _pq in card.get("customer_pain_quotes") or []:
        if isinstance(_pq, dict):
            _qt = str(_pq.get("quote") or "").strip()
            if _qt:
                pain_quote_meta[_qt] = _pq
    weaknesses = sanitized.get("weakness_analysis") if isinstance(sanitized, dict) else None
    if allowed_quotes and isinstance(weaknesses, list):
        for item in weaknesses:
            if not isinstance(item, dict):
                continue
            customer_quote = str(item.get("customer_quote") or "").strip()
            needs_replacement = bool(customer_quote and customer_quote not in allowed_quotes)
            if not needs_replacement and customer_quote:
                _meta = pain_quote_meta.get(customer_quote) or {}
                if not _quote_has_pain_signal(
                    customer_quote,
                    urgency=float(_meta.get("urgency") or 0),
                    rating=_meta.get("rating"),
                    rating_max=_meta.get("rating_max"),
                ):
                    needs_replacement = True
            if needs_replacement:
                context = "%s %s" % (item.get("weakness") or "", item.get("evidence") or "")
                excluded = {customer_quote} if customer_quote else None
                item["customer_quote"] = _battle_card_best_supported_quote(
                    card, context, excluded_quotes=excluded,
                )
    return sanitized


def _build_deterministic_battle_card_weakness_analysis(
    card: dict[str, Any],
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Build battle-card weakness analysis from deterministic evidence only."""
    items: list[dict[str, str]] = []
    used_quotes: set[str] = set()
    for weakness in (card.get("vendor_weaknesses") or [])[:limit]:
        if not isinstance(weakness, dict):
            continue
        area = str(weakness.get("area") or weakness.get("weakness") or "").strip()
        if not area:
            continue
        source = str(weakness.get("source") or "").strip()
        evidence = _battle_card_weakness_evidence(card, weakness)
        hint = area
        lower = area.lower()
        if any(token in lower for token in ("price", "pricing", "cost", "budget")):
            hint += " pricing cost budget spend renewal"
        elif any(token in lower for token in ("support", "service", "response", "help")):
            hint += " support service response escalation"
        elif any(token in lower for token in ("reliability", "uptime", "outage", "stability", "performance")):
            hint += " outage uptime reliability stability incident"
        elif source == "feature_gap" or any(token in lower for token in ("feature", "workflow", "reporting", "automation")):
            hint += " feature workflow capability reporting automation"
        elif any(token in lower for token in ("integration", "plugin", "api", "stack")):
            hint += " integration plugin api stack"
        quote_terms = _battle_card_quote_terms(area, source=source)
        quote = _battle_card_best_supported_quote(
            card,
            f"{hint} {evidence}",
            preferred_terms=quote_terms,
            excluded_quotes=used_quotes,
            require_preferred_match=bool(quote_terms),
        )
        if quote:
            used_quotes.add(quote)
        items.append({
            "weakness": _battle_card_weakness_headline(area, source=source),
            "evidence": evidence,
            "customer_quote": quote,
            "winning_position": _battle_card_winning_position(area, source=source),
        })
    return items


def _build_deterministic_battle_card_competitive_landscape(
    card: dict[str, Any],
    *,
    trigger_limit: int = 3,
    alt_limit: int = 3,
) -> dict[str, Any]:
    """Build battle-card competitive landscape from deterministic inputs."""
    data = card.get("objection_data") or {}
    budget = data.get("budget_context") or {}
    sentiment = str(data.get("sentiment_direction") or "").strip()
    council = card.get("category_council") or {}
    category_reasoning = {}
    contracts = card.get("reasoning_contracts")
    if isinstance(contracts, dict):
        contract = contracts.get("category_reasoning")
        if isinstance(contract, dict):
            category_reasoning = contract
    if not category_reasoning:
        raw_category = card.get("category_reasoning")
        if isinstance(raw_category, dict):
            category_reasoning = raw_category
    market_regime = (
        str(council.get("market_regime") or "").strip()
        or str(category_reasoning.get("market_regime") or "").strip()
    )
    aggregated_competitors = _battle_card_aggregated_competitors(card)
    alternatives: list[str] = []
    ranked_competitors = sorted(
        aggregated_competitors.values(),
        key=lambda item: (int(item["switch_count"]), int(item["mentions"]), item["competitor"]),
        reverse=True,
    )
    for item in ranked_competitors[:alt_limit]:
        label = str(item["competitor"]).strip()
        switches = int(item["switch_count"] or 0)
        mentions = int(item["mentions"] or 0)
        if switches <= 0 and mentions <= 0:
            continue
        driver_counts = item.get("driver_counts") or Counter()
        driver = driver_counts.most_common(1)[0][0] if driver_counts else "buyer fit"
        if switches > 0:
            alternatives.append(f"{label} ({switches} explicit switches; primary driver: {driver})")
        else:
            alternatives.append(f"{label} ({mentions} mentions in evaluation sets; primary driver: {driver})")
    window_bits: list[str] = []
    if sentiment == "declining":
        window_bits.append("Buyer sentiment is declining")
    if float(budget.get("price_increase_rate") or 0) > 0:
        window_bits.append("recent price increases are creating renewal scrutiny")
    if card.get("active_evaluation_deadlines"):
        window_bits.append("near-term evaluation timing is visible in review signals")
    if market_regime:
        window_bits.append(f"the category backdrop is {market_regime}")
    if not window_bits:
        window_bits.append("buyers are actively re-evaluating fit and value")
    triggers: list[str] = []
    if float(budget.get("price_increase_rate") or 0) > 0:
        triggers.append("Renewal cycles after recent price increases or packaging changes")
    if card.get("active_evaluation_deadlines"):
        triggers.append("Accounts showing explicit evaluation timelines or near-term decision windows")
    if any("support" in str(item.get("area") or "").lower() for item in (card.get("vendor_weaknesses") or [])):
        triggers.append("Support escalations or unresolved service issues")
    if not triggers and alternatives:
        triggers.append(f"Direct head-to-head evaluations against {alternatives[0].split(' (', 1)[0]}")
    return {
        "vulnerability_window": ". ".join(bit[:1].upper() + bit[1:] for bit in window_bits) + ".",
        "top_alternatives": alternatives,
        "displacement_triggers": triggers[:trigger_limit],
    }


def _best_cross_vendor_comparison(scorecard: dict[str, Any]) -> dict[str, Any] | None:
    """Return the highest-confidence cross-vendor comparison above the ref floor."""
    floor = _synthesis_reference_confidence_min()
    comparisons = [
        comp for comp in (scorecard.get("cross_vendor_comparisons") or [])
        if float(comp.get("confidence") or 0) >= floor
    ]
    if not comparisons:
        return None
    return max(
        comparisons,
        key=lambda comp: (
            float(comp.get("confidence") or 0),
            str(comp.get("opponent") or "").strip().lower(),
        ),
    )


def _build_scorecard_locked_facts(scorecard: dict[str, Any]) -> dict[str, Any]:
    """Build source-of-truth synthesis constraints for scorecard narratives."""
    locked: dict[str, Any] = {
        "vendor": str(scorecard.get("vendor") or ""),
        "risk_level": str(scorecard.get("risk_level") or ""),
    }
    if float(scorecard.get("archetype_confidence") or 0) >= _synthesis_reference_confidence_min():
        if scorecard.get("archetype"):
            locked["archetype"] = scorecard["archetype"]
    allowed_opponents = sorted({
        str(comp.get("opponent") or "").strip()
        for comp in (scorecard.get("cross_vendor_comparisons") or [])
        if str(comp.get("opponent") or "").strip()
        and float(comp.get("confidence") or 0) >= _synthesis_reference_confidence_min()
    })
    if allowed_opponents:
        locked["allowed_opponents"] = allowed_opponents
    best = _best_cross_vendor_comparison(scorecard)
    if best:
        locked["comparison"] = {
            "opponent": best.get("opponent", ""),
            "resource_advantage": best.get("resource_advantage", ""),
        }
    return {k: v for k, v in locked.items() if v not in ("", [], None)}


def _build_metric_ledger(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a scoped metric ledger so the LLM knows exactly what each number means.

    Each entry: {label, value, scope, wording}
    scope options: all_reviews, pricing_mentions, decision_makers,
                   active_eval_accounts, segment_sample, budget_data
    """
    entries: list[dict[str, Any]] = []
    data = card.get("objection_data") or {}
    total_reviews = int(data.get("total_reviews") or card.get("total_reviews") or 0)
    if total_reviews:
        entries.append({
            "label": "Reviews analyzed",
            "value": total_reviews,
            "scope": "all_reviews",
            "wording": f"{total_reviews:,} reviews analyzed across sources",
        })
    price_rate = data.get("price_complaint_rate")
    if price_rate is not None:
        pct = round(float(price_rate) * 100, 1)
        base = int(round(float(price_rate) * total_reviews)) if total_reviews else None
        base_text = f" ({base:,} of {total_reviews:,} reviews)" if base else ""
        entries.append({
            "label": "Price complaint rate",
            "value": f"{pct}%",
            "scope": "pricing_mentions",
            "wording": f"{pct}% of reviews mention pricing as a pain point{base_text}",
        })
    dm_rate = data.get("dm_churn_rate")
    if dm_rate is not None:
        pct = round(float(dm_rate) * 100, 1)
        entries.append({
            "label": "Decision-maker churn signal rate",
            "value": f"{pct}%",
            "scope": "decision_makers",
            "wording": f"{pct}% of decision-maker reviews show churn signals",
        })
    density = data.get("churn_signal_density")
    if density is not None:
        entries.append({
            "label": "Churn signal density",
            "value": round(float(density), 1),
            "scope": "all_reviews",
            "wording": f"{round(float(density), 1)}% of all reviews contain a churn signal",
        })
    avg_urgency = data.get("avg_urgency")
    if avg_urgency is not None:
        entries.append({
            "label": "Average urgency score",
            "value": round(float(avg_urgency), 1),
            "scope": "all_reviews",
            "wording": (
                f"average urgency {round(float(avg_urgency), 1)}/10 across all reviews"
            ),
        })
    acct_metrics = card.get("account_pressure_metrics") or {}
    eval_count = acct_metrics.get("active_eval_count")
    if eval_count is not None:
        entries.append({
            "label": "Active evaluation accounts",
            "value": int(eval_count),
            "scope": "active_eval_accounts",
            "wording": f"{int(eval_count)} accounts actively evaluating alternatives",
        })
    budget = data.get("budget_context") or {}
    price_inc_count = budget.get("price_increase_count")
    price_inc_rate = budget.get("price_increase_rate")
    if price_inc_count and price_inc_rate is not None:
        pct = round(float(price_inc_rate) * 100, 1)
        entries.append({
            "label": "Price increase mentions",
            "value": int(price_inc_count),
            "scope": "budget_data",
            "wording": (
                f"{int(price_inc_count)} reviews mention a price increase "
                f"({pct}% of all reviews)"
            ),
        })
    for seg in (card.get("segment_playbook") or {}).get("priority_segments") or []:
        if not isinstance(seg, dict):
            continue
        seg_name = str(seg.get("segment") or "").strip()
        sample = seg.get("sample_size")
        if seg_name and sample is not None:
            entries.append({
                "label": f"Segment sample: {seg_name}",
                "value": int(sample),
                "scope": "segment_sample",
                "wording": (
                    f"{int(sample)} reviews from {seg_name} accounts "
                    f"(sample n={int(sample)})"
                ),
            })
    return entries


def _build_battle_card_locked_facts(card: dict[str, Any]) -> dict[str, Any]:
    """Build source-of-truth synthesis constraints for battle-card copy."""
    objection_data = card.get("objection_data") or {}
    allowed_opponents: list[str] = []
    for comp in _battle_card_aggregated_competitors(card).values():
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
    # Cross-vendor comparison context is available in the separate
    # cross_vendor_comparisons field on the scorecard.  Appending it to the
    # expert_take prose leaked internal LLM labels (e.g. "Neither - Resource
    # Parity") into buyer-facing copy, so we no longer interpolate it here.
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


def _normalize_scorecard_expert_take(expert_take: str) -> str:
    """Clamp synthesized scorecard narratives to the configured word budget."""
    text = str(expert_take or "").strip()
    if not text:
        return ""
    return _trim_words(text, _synthesis_expert_take_max_words())


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


def _infer_driver_from_reasons(reasons: list[str], fallback: str | None = None) -> str | None:
    """Classify a driver label from competitor-reason text."""
    for reason in reasons:
        normalized = _normalize_displacement_driver_label(reason)
        if normalized:
            return normalized
    text = " ".join(reasons).lower()
    keyword_map = {
        "pricing": ("price", "pricing", "cost", "cheaper", "affordable", "budget"),
        "support": ("support", "service", "response", "help desk", "ticket"),
        "reliability": ("outage", "uptime", "reliable", "stability", "incident"),
        "ux": ("ui", "ux", "easy", "simpler", "interface", "adoption", "learning curve"),
        "integration": ("integration", "api", "connector", "webhook", "plugin", "stack"),
        "features": ("feature", "workflow", "automation", "reporting", "capability"),
        "security": ("security", "privacy", "governance", "server", "encryption"),
        "compliance": ("compliance", "gdpr", "hipaa", "soc2", "soc 2"),
    }
    for label, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return label
    return fallback


def _get_vendor_reasoning(
    vendor: str,
    *,
    synthesis_views: dict[str, Any] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Get reasoning entry for a vendor, preferring SynthesisView.

    Returns the same dict shape as reasoning_lookup entries: archetype,
    confidence, executive_summary, key_signals, etc.  When a SynthesisView
    is available, reads directly from it; otherwise falls back to the
    pre-built reasoning_lookup shim.
    """
    if synthesis_views:
        view = synthesis_views.get(vendor)
        # Canonicalized fallback: try lowered/stripped key
        if view is None and vendor:
            canon = vendor.strip().lower()
            for vn, v in synthesis_views.items():
                if vn.strip().lower() == canon:
                    view = v
                    break
        if view is not None:
            from ._b2b_synthesis_reader import synthesis_view_to_reasoning_entry
            return synthesis_view_to_reasoning_entry(view)
    return (reasoning_lookup or {}).get(vendor, {})


def _build_market_shift_signal(
    category: str,
    highest_vendor: str,
    churn_density: float,
    total_reviews: int,
    emerging: str,
    reasoning_lookup: dict[str, dict],
    synthesis_views: dict[str, Any] | None = None,
) -> str:
    """Build market shift signal with archetype context when available."""
    base = (
        f"Based on {total_reviews} reviews, {highest_vendor} shows "
        f"{churn_density}% churn-signal density in {category}."
    )
    _rc = _get_vendor_reasoning(highest_vendor, synthesis_views=synthesis_views, reasoning_lookup=reasoning_lookup)
    arch = _rc.get("archetype", "")
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
    if _battle_card_has_duplicate_recommended_play_segments(generated):
        warnings.append("recommended_plays repeat the same target segment")
    allowed = _battle_card_allowed_claims(card)
    source_text = json.dumps(_battle_card_validator_source(card), default=str).lower()
    max_switch = max((int(c.get("switch_count") or 0) for c in card.get("competitor_differentiators") or []), default=0)
    score = float(card.get("churn_pressure_score") or 0)
    urgency = float(((card.get("objection_data") or {}).get("avg_urgency")) or 0)
    low_gap_terms = [
        str(g.get("feature") or "").strip().lower()
        for g in ((card.get("objection_data") or {}).get("top_feature_gaps") or [])
        if int(g.get("mentions") or 0) < _battle_card_feature_gap_headline_min_mentions()
    ]
    allowed_quotes = set(_battle_card_allowed_quotes(card))
    for path, text in _battle_card_iter_text(generated):
        lowered = text.lower()
        if _battle_card_numeric_paths(path):
            bad = sorted(
                tok
                for tok in _battle_card_numeric_tokens(text)
                if _battle_card_normalize_numeric_token(tok) not in allowed
            )
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
        if _battle_card_segment_evidence_is_thin(card) and path.endswith(".play"):
            if lowered.startswith("target ") or lowered.startswith("engage "):
                warnings.append(f"{path} overstates segment certainty without account-level intelligence")
        if _battle_card_headline_paths(path):
            for term in low_gap_terms:
                if term and term in lowered:
                    warnings.append(f"{path} elevates low-evidence feature gap '{term}' to a headline")
        overreach = _battle_card_overreach_violations(text)
        if overreach:
            warnings.append(
                f"{path} uses overreaching language: {overreach[0]!r}"
            )
    pain_quote_meta: dict[str, dict[str, Any]] = {}
    for _pq in card.get("customer_pain_quotes") or []:
        if isinstance(_pq, dict):
            _qt = str(_pq.get("quote") or "").strip()
            if _qt:
                pain_quote_meta[_qt] = _pq
    weaknesses = generated.get("weakness_analysis")
    if allowed_quotes and isinstance(weaknesses, list):
        for idx, item in enumerate(weaknesses):
            if not isinstance(item, dict):
                continue
            customer_quote = str(item.get("customer_quote") or "").strip()
            if customer_quote and customer_quote not in allowed_quotes:
                warnings.append(
                    f"weakness_analysis[{idx}].customer_quote is not an exact source quote"
                )
            if customer_quote:
                _meta = pain_quote_meta.get(customer_quote) or {}
                if not _quote_has_pain_signal(
                    customer_quote,
                    urgency=float(_meta.get("urgency") or 0),
                    rating=_meta.get("rating"),
                    rating_max=_meta.get("rating_max"),
                ):
                    warnings.append(
                        f"weakness_analysis[{idx}].customer_quote appears positive, not a pain signal"
                    )
    # Detect near-duplicate sections: collect all long string values and
    # flag when the same substantial prefix appears in more than one field.
    long_strings: list[tuple[str, str]] = []
    for path, text in _battle_card_iter_text(generated):
        if len(text.strip()) >= 80:
            long_strings.append((path, text.strip()))
    seen_prefixes: dict[str, str] = {}
    for path, text in long_strings:
        prefix = text[:80].lower()
        if prefix in seen_prefixes and seen_prefixes[prefix] != path:
            warnings.append(
                f"{path} duplicates content already present in {seen_prefixes[prefix]}"
            )
        else:
            seen_prefixes[prefix] = path
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

    When *archetype* is provided from a reasoning layer, the weights
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


def _normalize_generic_pain_label(label: Any) -> str:
    raw = str(label or "").strip().lower()
    if raw in {"other", "general_dissatisfaction", "overall_dissatisfaction"}:
        return "overall_dissatisfaction"
    return raw


def _is_generic_pain_label(label: Any) -> bool:
    return _normalize_generic_pain_label(label) == "overall_dissatisfaction"


def _executive_summary_representative_quote(
    feed: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> str | None:
    """Pick a churn-aligned representative quote from the weekly feed."""
    candidates: list[tuple[int, float, int, str]] = []
    fallback_quotes: list[tuple[int, float, int, str]] = []
    for idx, entry in enumerate(feed[: max(limit, 1)]):
        if not isinstance(entry, dict):
            continue
        quote = str(entry.get("key_quote") or "").strip()
        if not quote:
            continue
        try:
            urgency = float(entry.get("avg_urgency") or 0.0)
        except (TypeError, ValueError):
            urgency = 0.0
        ranked = (idx, urgency, len(quote), quote)
        if _quote_has_pain_signal(quote, urgency=urgency):
            candidates.append(ranked)
        else:
            fallback_quotes.append(ranked)
    if candidates:
        candidates.sort(key=lambda item: (-item[1], item[0], -item[2]))
        return candidates[0][3]
    if fallback_quotes:
        fallback_quotes.sort(key=lambda item: (item[0], -item[1], -item[2]))
        return fallback_quotes[0][3]
    return None


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
        if top_pain and not _is_generic_pain_label(top_pain) and top_pain.lower() != "unknown":
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
        if top_driver and not _is_generic_pain_label(top_driver):
            s2 += f", driven by {top_driver}"
        s2 += "."
        lines.append(s2)

        # Sentence 3: driver distribution (weighted by mention count)
        driver_counts: dict[str, int] = {}
        for e in disp:
            d = e.get("primary_driver") or None
            if d and not _is_generic_pain_label(d):
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
            if pain and not _is_generic_pain_label(pain) and pain.lower() != "unknown":
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
            if p and not _is_generic_pain_label(p) and p.lower() != "unknown":
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
    quote = _executive_summary_representative_quote(top_entries)
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
        # Extract pains from pain_breakdown or top_pain -- skip generic fallback labels
        pain_breakdown = entry.get("pain_breakdown", [])
        if pain_breakdown:
            for pb in pain_breakdown[:2]:
                p = pb.get("category", "")
                if p and not _is_generic_pain_label(p) and p not in top_pains:
                    top_pains.append(str(p))
        elif entry.get("top_pain"):
            p = str(entry["top_pain"])
            if not _is_generic_pain_label(p) and p not in top_pains:
                top_pains.append(p)
        # Extract alternatives from displacement targets
        for dt in entry.get("top_displacement_targets", []) or []:
            comp = dt.get("competitor", "")
            if comp and comp not in top_alternatives:
                top_alternatives.append(comp)

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
    funnel_audit = await _fetch_review_funnel_audit(pool, window_days)
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
                WHERE (enrichment->>'urgency_score')::numeric >= 7
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
        "funnel_audit": funnel_audit,
    }


async def _fetch_vendor_provenance(pool, window_days: int) -> dict[str, dict]:
    """Per-vendor provenance: source distribution, sample review IDs, and review window.

    Returns {vendor_name: {"source_distribution": {...}, "sample_review_ids": [...],
             "review_window_start": dt, "review_window_end": dt}}.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    time_expr = _eligible_review_timestamp_expr()

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
            MIN({time_expr}) AS window_start,
            MAX({time_expr}) AS window_end
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
            SELECT vendor_name,
                MODE() WITHIN GROUP (ORDER BY product_category) AS product_category,
                count(*) AS total_reviews,
                count(*) FILTER (
                    WHERE COALESCE((enrichment->>'urgency_score')::numeric, 0) > 0
                       OR (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
                       OR jsonb_array_length(COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb)) > 0
                ) AS signal_reviews,
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
                count(*) FILTER (
                    WHERE enrichment->>'would_recommend' IS NOT NULL
                ) AS recommend_total,
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
                ) * 1.0 / NULLIF(count(*), 0) AS new_feature_velocity,
                -- v2 urgency indicator counts (from three-layer extraction)
                count(*) FILTER (
                    WHERE (enrichment->'urgency_indicators'->>'explicit_cancel_language')::boolean = true
                ) AS indicator_cancel_count,
                count(*) FILTER (
                    WHERE (enrichment->'urgency_indicators'->>'active_migration_language')::boolean = true
                ) AS indicator_migration_count,
                count(*) FILTER (
                    WHERE (enrichment->'urgency_indicators'->>'active_evaluation_language')::boolean = true
                ) AS indicator_evaluation_count,
                count(*) FILTER (
                    WHERE (enrichment->'urgency_indicators'->>'completed_switch_language')::boolean = true
                ) AS indicator_switch_count,
                count(*) FILTER (
                    WHERE (enrichment->'urgency_indicators'->>'named_alternative_with_reason')::boolean = true
                ) AS indicator_named_alt_count,
                count(*) FILTER (
                    WHERE (enrichment->'urgency_indicators'->>'decision_maker_language')::boolean = true
                ) AS indicator_dm_language_count,
                -- v2 pricing phrase count
                count(*) FILTER (
                    WHERE jsonb_array_length(COALESCE(enrichment->'pricing_phrases', '[]'::jsonb)) > 0
                ) AS has_pricing_phrases_count,
                -- v2 recommendation language count
                count(*) FILTER (
                    WHERE jsonb_array_length(COALESCE(enrichment->'recommendation_language', '[]'::jsonb)) > 0
                ) AS has_recommendation_language_count
            FROM b2b_reviews
            WHERE {filters}
            GROUP BY vendor_name
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
            "signal_reviews": r["signal_reviews"],
            "churn_intent": r["churn_intent"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
            "avg_rating_normalized": float(r["avg_rating_normalized"]) if r["avg_rating_normalized"] else None,
            "recommend_yes": r["recommend_yes"],
            "recommend_no": r["recommend_no"],
            "recommend_total": r["recommend_total"],
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


def _is_generic_vendor_score_category(category: str | None) -> bool:
    """Return True for generic vendor-score categories we should down-rank."""
    return str(category or "").strip().lower() in {"", "unknown", "b2b software"}


def _canonicalize_vendor_name_filters(vendor_names: Iterable[Any] | None) -> list[str]:
    """Normalize an optional vendor filter list for adapter-backed reads."""
    return sorted(
        {
            canonical.lower()
            for name in (vendor_names or [])
            for canonical in [_canonicalize_vendor(name)]
            if canonical
        }
    )


async def read_vendor_scorecards(
    pool,
    *,
    window_days: int,
    min_reviews: int,
    vendor_names: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Read derived vendor scorecards from ``b2b_churn_signals``.

    This is the canonical adapter for vendor-level ranking and score reads.
    Feature code should prefer this adapter instead of querying
    ``b2b_churn_signals`` directly.
    """
    requested_vendors = _canonicalize_vendor_name_filters(vendor_names)
    vendor_filter_clause = "AND LOWER(vendor_name) = ANY($3::text[])" if requested_vendors else ""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
               product_category,
               total_reviews,
               signal_reviews,
               churn_intent_count AS churn_intent,
               avg_urgency_score AS avg_urgency,
               last_computed_at,
               review_window_end
        FROM b2b_churn_signals
        WHERE total_reviews >= $2
          AND COALESCE(review_window_end, last_computed_at::date) >= CURRENT_DATE - $1::int
          """ + vendor_filter_clause + """
        ORDER BY vendor_name, total_reviews DESC, last_computed_at DESC
        """,
        window_days,
        min_reviews,
        *([requested_vendors] if requested_vendors else []),
    )
    mapped = [
        {
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "total_reviews": int(r["total_reviews"] or 0),
            "churn_intent": int(r["churn_intent"] or 0),
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] is not None else 0.0,
        }
        for r in rows
        if r.get("vendor_name")
    ]

    categories_by_vendor: dict[str, set[str]] = {}
    for row in mapped:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        categories_by_vendor.setdefault(vendor, set()).add(
            str(row.get("product_category") or "").strip()
        )

    filtered: list[dict[str, Any]] = []
    for row in mapped:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        category = str(row.get("product_category") or "").strip()
        categories = categories_by_vendor.get(vendor) or set()
        has_specific_category = any(
            not _is_generic_vendor_score_category(value)
            for value in categories
        )
        if has_specific_category and _is_generic_vendor_score_category(category):
            continue
        filtered.append(row)
    return filtered


async def read_vendor_scorecard(
    pool,
    vendor_name: str,
    *,
    window_days: int,
    min_reviews: int,
) -> dict[str, Any] | None:
    """Read the derived scorecard row for a single vendor."""
    rows = await read_vendor_scorecards(
        pool,
        window_days=window_days,
        min_reviews=min_reviews,
        vendor_names=[vendor_name],
    )
    canonical = _canonicalize_vendor(vendor_name)
    if not canonical:
        return None
    for row in rows:
        if _canonicalize_vendor(row.get("vendor_name") or "") == canonical:
            return row
    return None


async def read_vendor_scorecard_details(
    pool,
    *,
    vendor_names: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Read detailed derived scorecard rows from ``b2b_churn_signals``.

    This adapter is for vendor-level fallback consumers that need the richer
    compatibility columns from the derived scorecard table.
    """
    requested_vendors = _canonicalize_vendor_name_filters(vendor_names)
    vendor_filter_clause = "WHERE LOWER(vendor_name) = ANY($1::text[])" if requested_vendors else ""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
               product_category,
               total_reviews,
               negative_reviews,
               churn_intent_count,
               avg_urgency_score,
               top_pain_categories,
               top_competitors,
               top_feature_gaps,
               price_complaint_rate,
               decision_maker_churn_rate,
               company_churn_list,
               quotable_evidence,
               materialization_run_id,
               last_computed_at,
               review_window_end
        FROM b2b_churn_signals
        """ + vendor_filter_clause + """
        ORDER BY vendor_name, total_reviews DESC, last_computed_at DESC
        """,
        *([requested_vendors] if requested_vendors else []),
    )
    mapped = [dict(row) for row in rows if row.get("vendor_name")]

    categories_by_vendor: dict[str, set[str]] = {}
    for row in mapped:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        categories_by_vendor.setdefault(vendor, set()).add(
            str(row.get("product_category") or "").strip()
        )

    filtered: list[dict[str, Any]] = []
    for row in mapped:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        category = str(row.get("product_category") or "").strip()
        categories = categories_by_vendor.get(vendor) or set()
        has_specific_category = any(
            not _is_generic_vendor_score_category(value)
            for value in categories
        )
        if has_specific_category and _is_generic_vendor_score_category(category):
            continue
        filtered.append(row)
    return filtered


async def read_vendor_scorecard_detail(
    pool,
    vendor_name: str,
) -> dict[str, Any] | None:
    """Read the detailed derived scorecard row for a single vendor."""
    rows = await read_vendor_scorecard_details(
        pool,
        vendor_names=[vendor_name],
    )
    canonical = _canonicalize_vendor(vendor_name)
    if not canonical:
        return None
    for row in rows:
        if _canonicalize_vendor(row.get("vendor_name") or "") == canonical:
            return row
    return None


async def read_vendor_signal_detail(
    pool,
    *,
    vendor_name_query: str,
    product_category: str | None = None,
    tracked_account_id: Any | None = None,
    include_snapshot_metrics: bool = False,
    exclude_suppressed: bool = False,
) -> dict[str, Any] | None:
    """Read one vendor signal detail row with optional scope and snapshot fields."""
    normalized_query = str(vendor_name_query or "").strip()
    if not normalized_query:
        return None

    conditions = ["sig.vendor_name ILIKE '%' || $1 || '%'"]
    params: list[Any] = [normalized_query]
    idx = 2

    if product_category:
        conditions.append(f"sig.product_category = ${idx}")
        params.append(product_category)
        idx += 1

    if tracked_account_id is not None:
        conditions.append(
            f"sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(tracked_account_id)
        idx += 1

    if exclude_suppressed:
        conditions.append(
            suppress_predicate(
                "churn_signal",
                id_expr="sig.id",
                vendor_expr="sig.vendor_name",
            )
        )

    snapshot_select = ""
    snapshot_join = ""
    if include_snapshot_metrics:
        snapshot_select = """
               , snap.support_sentiment AS support_sentiment
               , snap.legacy_support_score AS legacy_support_score
               , snap.new_feature_velocity AS new_feature_velocity
               , snap.employee_growth_rate AS employee_growth_rate
        """
        snapshot_join = """
        LEFT JOIN LATERAL (
            SELECT support_sentiment,
                   legacy_support_score,
                   new_feature_velocity,
                   employee_growth_rate
            FROM b2b_vendor_snapshots snap
            WHERE snap.vendor_name = sig.vendor_name
            ORDER BY snap.snapshot_date DESC
            LIMIT 1
        ) snap ON TRUE
        """

    row = await pool.fetchrow(
        """
        SELECT sig.*
               """
        + snapshot_select
        + """
        FROM b2b_churn_signals sig
        """
        + snapshot_join
        + """
        WHERE """
        + " AND ".join(conditions)
        + """
        ORDER BY sig.avg_urgency_score DESC
        LIMIT 1
        """,
        *params,
    )
    return dict(row) if row else None


async def read_vendor_signal_detail_exact(
    pool,
    *,
    vendor_name: str,
    product_category: str | None = None,
    tracked_account_id: Any | None = None,
    include_snapshot_metrics: bool = False,
    exclude_suppressed: bool = False,
) -> dict[str, Any] | None:
    """Read one exact-match vendor signal detail row with optional scope."""
    normalized_vendor = _canonicalize_vendor(vendor_name) or str(vendor_name or "").strip()
    if not normalized_vendor:
        return None

    conditions = ["LOWER(sig.vendor_name) = LOWER($1)"]
    params: list[Any] = [normalized_vendor]
    idx = 2

    if product_category:
        conditions.append(f"sig.product_category = ${idx}")
        params.append(product_category)
        idx += 1

    if tracked_account_id is not None:
        conditions.append(
            f"sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(tracked_account_id)
        idx += 1

    if exclude_suppressed:
        conditions.append(
            suppress_predicate(
                "churn_signal",
                id_expr="sig.id",
                vendor_expr="sig.vendor_name",
            )
        )

    snapshot_select = ""
    snapshot_join = ""
    if include_snapshot_metrics:
        snapshot_select = """
               , snap.support_sentiment AS support_sentiment
               , snap.legacy_support_score AS legacy_support_score
               , snap.new_feature_velocity AS new_feature_velocity
               , snap.employee_growth_rate AS employee_growth_rate
        """
        snapshot_join = """
        LEFT JOIN LATERAL (
            SELECT support_sentiment,
                   legacy_support_score,
                   new_feature_velocity,
                   employee_growth_rate
            FROM b2b_vendor_snapshots snap
            WHERE snap.vendor_name = sig.vendor_name
            ORDER BY snap.snapshot_date DESC
            LIMIT 1
        ) snap ON TRUE
        """

    row = await pool.fetchrow(
        """
        SELECT sig.*
               """
        + snapshot_select
        + """
        FROM b2b_churn_signals sig
        """
        + snapshot_join
        + """
        WHERE """
        + " AND ".join(conditions)
        + """
        ORDER BY sig.last_computed_at DESC NULLS LAST,
                 sig.total_reviews DESC NULLS LAST,
                 sig.product_category ASC NULLS LAST
        LIMIT 1
        """,
        *params,
    )
    return dict(row) if row else None


def _normalize_vendor_name_list(vendor_names: Iterable[Any] | None) -> list[str]:
    return _canonicalize_vendor_name_filters(vendor_names)


async def read_vendor_signal_rows(
    pool,
    *,
    vendor_name_query: str | None = None,
    vendor_names: Iterable[Any] | None = None,
    min_urgency: float = 0.0,
    product_category: str | None = None,
    tracked_account_id: Any | None = None,
    include_snapshot_metrics: bool = False,
    exclude_suppressed: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Read vendor signal list rows with optional scope and snapshot fields."""
    normalized_vendor_names = _normalize_vendor_name_list(vendor_names)
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if tracked_account_id is not None:
        conditions.append(
            f"sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(tracked_account_id)
        idx += 1

    if normalized_vendor_names:
        conditions.append(f"LOWER(sig.vendor_name) = ANY(${idx}::text[])")
        params.append(normalized_vendor_names)
        idx += 1

    normalized_query = str(vendor_name_query or "").strip()
    if normalized_query:
        conditions.append(f"sig.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(normalized_query)
        idx += 1

    if min_urgency > 0:
        conditions.append(f"sig.avg_urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    if product_category:
        conditions.append(f"sig.product_category = ${idx}")
        params.append(product_category)
        idx += 1

    if exclude_suppressed:
        conditions.append(
            suppress_predicate(
                "churn_signal",
                id_expr="sig.id",
                vendor_expr="sig.vendor_name",
            )
        )

    snapshot_select = ""
    snapshot_join = ""
    if include_snapshot_metrics:
        snapshot_select = """
               , snap.support_sentiment AS support_sentiment
               , snap.legacy_support_score AS legacy_support_score
               , snap.new_feature_velocity AS new_feature_velocity
               , snap.employee_growth_rate AS employee_growth_rate
        """
        snapshot_join = """
        LEFT JOIN LATERAL (
            SELECT support_sentiment,
                   legacy_support_score,
                   new_feature_velocity,
                   employee_growth_rate
            FROM b2b_vendor_snapshots snap
            WHERE snap.vendor_name = sig.vendor_name
            ORDER BY snap.snapshot_date DESC
            LIMIT 1
        ) snap ON TRUE
        """

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    capped_limit = max(1, min(int(limit or 20), 10_000))
    rows = await pool.fetch(
        """
        SELECT sig.vendor_name,
               sig.product_category,
               sig.total_reviews,
               sig.churn_intent_count,
               sig.avg_urgency_score,
               sig.avg_rating_normalized,
               sig.nps_proxy,
               sig.price_complaint_rate,
               sig.decision_maker_churn_rate,
               sig.keyword_spike_count,
               sig.insider_signal_count,
               sig.last_computed_at
               """
        + snapshot_select
        + """
        FROM b2b_churn_signals sig
        """
        + snapshot_join
        + """
        """
        + where_clause
        + f"""
        ORDER BY sig.avg_urgency_score DESC
        LIMIT ${idx}
        """,
        *params,
        capped_limit,
    )
    return [dict(row) for row in rows]


async def read_best_vendor_signal_rows(
    pool,
    *,
    vendor_name_query: str | None = None,
    vendor_names: Iterable[Any] | None = None,
    tracked_account_id: Any | None = None,
    exclude_suppressed: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Read one best-fit vendor signal row per vendor."""
    normalized_vendor_names = _normalize_vendor_name_list(vendor_names)
    if vendor_names is not None and not normalized_vendor_names:
        return []
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if tracked_account_id is not None:
        conditions.append(
            f"sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(tracked_account_id)
        idx += 1

    if normalized_vendor_names:
        conditions.append(f"LOWER(sig.vendor_name) = ANY(${idx}::text[])")
        params.append(normalized_vendor_names)
        idx += 1

    normalized_query = str(vendor_name_query or "").strip()
    if normalized_query:
        conditions.append(f"sig.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(normalized_query)
        idx += 1

    if exclude_suppressed:
        conditions.append(
            suppress_predicate(
                "churn_signal",
                id_expr="sig.id",
                vendor_expr="sig.vendor_name",
            )
        )

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    capped_limit = max(1, min(int(limit or 20), 5000))
    rows = await pool.fetch(
        """
        WITH ranked_signals AS (
            SELECT sig.vendor_name,
                   sig.product_category,
                   sig.total_reviews,
                   sig.churn_intent_count,
                   sig.avg_urgency_score,
                   sig.nps_proxy,
                   sig.last_computed_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY sig.vendor_name
                       ORDER BY sig.avg_urgency_score DESC,
                                sig.total_reviews DESC,
                                sig.last_computed_at DESC NULLS LAST,
                                sig.product_category ASC NULLS LAST
                   ) AS vendor_row_rank
            FROM b2b_churn_signals sig
        """
        + where_clause
        + """
        )
        SELECT vendor_name,
               product_category,
               total_reviews,
               churn_intent_count,
               avg_urgency_score,
               nps_proxy,
               last_computed_at
        FROM ranked_signals
        WHERE vendor_row_rank = 1
        ORDER BY total_reviews DESC,
                 avg_urgency_score DESC,
                 last_computed_at DESC NULLS LAST,
                 vendor_name ASC
        LIMIT $"""
        + str(idx),
        *params,
        capped_limit,
    )
    return [dict(row) for row in rows]


async def read_vendor_signal_summary(
    pool,
    *,
    vendor_name_query: str | None = None,
    vendor_names: Iterable[Any] | None = None,
    min_urgency: float = 0.0,
    product_category: str | None = None,
    tracked_account_id: Any | None = None,
    exclude_suppressed: bool = False,
) -> dict[str, Any]:
    """Read aggregate summary stats for vendor signal list surfaces."""
    normalized_vendor_names = _normalize_vendor_name_list(vendor_names)
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if tracked_account_id is not None:
        conditions.append(
            f"sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(tracked_account_id)
        idx += 1

    if normalized_vendor_names:
        conditions.append(f"LOWER(sig.vendor_name) = ANY(${idx}::text[])")
        params.append(normalized_vendor_names)
        idx += 1

    normalized_query = str(vendor_name_query or "").strip()
    if normalized_query:
        conditions.append(f"sig.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(normalized_query)
        idx += 1

    if min_urgency > 0:
        conditions.append(f"sig.avg_urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    if product_category:
        conditions.append(f"sig.product_category = ${idx}")
        params.append(product_category)
        idx += 1

    if exclude_suppressed:
        conditions.append(
            suppress_predicate(
                "churn_signal",
                id_expr="sig.id",
                vendor_expr="sig.vendor_name",
            )
        )

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    row = await pool.fetchrow(
        """
        SELECT COUNT(DISTINCT sig.vendor_name) AS total_vendors,
               COUNT(*) FILTER (WHERE sig.avg_urgency_score >= 7) AS high_urgency_count,
               COALESCE(SUM(sig.total_reviews), 0) AS total_signal_reviews
        FROM b2b_churn_signals sig
        """
        + where_clause,
        *params,
    )
    return dict(row) if row else {
        "total_vendors": 0,
        "high_urgency_count": 0,
        "total_signal_reviews": 0,
    }


async def read_vendor_signal_overview(
    pool,
    *,
    vendor_name_query: str | None = None,
    vendor_names: Iterable[Any] | None = None,
    min_urgency: float = 0.0,
    product_category: str | None = None,
    tracked_account_id: Any | None = None,
    exclude_suppressed: bool = False,
) -> dict[str, Any]:
    """Read overview rollups for vendor signal dashboard surfaces."""
    normalized_vendor_names = _normalize_vendor_name_list(vendor_names)
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if tracked_account_id is not None:
        conditions.append(
            f"sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(tracked_account_id)
        idx += 1

    if normalized_vendor_names:
        conditions.append(f"LOWER(sig.vendor_name) = ANY(${idx}::text[])")
        params.append(normalized_vendor_names)
        idx += 1

    normalized_query = str(vendor_name_query or "").strip()
    if normalized_query:
        conditions.append(f"sig.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(normalized_query)
        idx += 1

    if min_urgency > 0:
        conditions.append(f"sig.avg_urgency_score >= ${idx}")
        params.append(min_urgency)
        idx += 1

    if product_category:
        conditions.append(f"sig.product_category = ${idx}")
        params.append(product_category)
        idx += 1

    if exclude_suppressed:
        conditions.append(
            suppress_predicate(
                "churn_signal",
                id_expr="sig.id",
                vendor_expr="sig.vendor_name",
            )
        )

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    row = await pool.fetchrow(
        """
        SELECT COALESCE(AVG(sig.avg_urgency_score), 0) AS avg_urgency,
               COALESCE(SUM(sig.churn_intent_count), 0) AS total_churn_signals,
               COALESCE(SUM(sig.total_reviews), 0) AS total_reviews
        FROM b2b_churn_signals sig
        """
        + where_clause,
        *params,
    )
    return dict(row) if row else {
        "avg_urgency": 0,
        "total_churn_signals": 0,
        "total_reviews": 0,
    }


async def read_ranked_vendor_signal_rows(
    pool,
    *,
    vendor_name_query: str | None = None,
    vendor_names: Iterable[Any] | None = None,
    product_category: str | None = None,
    tracked_account_id: Any | None = None,
    exclude_suppressed: bool = False,
    require_snapshot_activity: bool = False,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Read one ranked vendor signal row per vendor for slow-burn/watchlist surfaces."""
    normalized_vendor_names = _normalize_vendor_name_list(vendor_names)
    conditions: list[str] = []
    params: list[Any] = []
    idx = 1

    if require_snapshot_activity:
        conditions.append(
            "("
            "snap.support_sentiment IS NOT NULL OR "
            "snap.legacy_support_score IS NOT NULL OR "
            "snap.new_feature_velocity IS NOT NULL OR "
            "snap.employee_growth_rate IS NOT NULL"
            ")"
        )

    if tracked_account_id is not None:
        conditions.append(
            f"sig.vendor_name IN (SELECT vendor_name FROM tracked_vendors WHERE account_id = ${idx}::uuid)"
        )
        params.append(tracked_account_id)
        idx += 1

    if normalized_vendor_names:
        conditions.append(f"LOWER(sig.vendor_name) = ANY(${idx}::text[])")
        params.append(normalized_vendor_names)
        idx += 1

    normalized_query = str(vendor_name_query or "").strip()
    if normalized_query:
        conditions.append(f"sig.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(normalized_query)
        idx += 1

    if product_category:
        conditions.append(f"sig.product_category = ${idx}")
        params.append(product_category)
        idx += 1

    if exclude_suppressed:
        conditions.append(
            suppress_predicate(
                "churn_signal",
                id_expr="sig.id",
                vendor_expr="sig.vendor_name",
            )
        )

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    capped_limit = max(1, min(int(limit or 10), 100))
    rows = await pool.fetch(
        """
        WITH ranked_signals AS (
            SELECT sig.vendor_name,
                   sig.product_category,
                   sig.total_reviews,
                   sig.churn_intent_count,
                   sig.avg_urgency_score,
                   sig.avg_rating_normalized,
                   sig.nps_proxy,
                   sig.price_complaint_rate,
                   sig.decision_maker_churn_rate,
                   sig.keyword_spike_count,
                   sig.insider_signal_count,
                   sig.last_computed_at,
                   snap.support_sentiment AS support_sentiment,
                   snap.legacy_support_score AS legacy_support_score,
                   snap.new_feature_velocity AS new_feature_velocity,
                   snap.employee_growth_rate AS employee_growth_rate,
                   ROW_NUMBER() OVER (
                       PARTITION BY sig.vendor_name
                       ORDER BY sig.avg_urgency_score DESC,
                                sig.total_reviews DESC,
                                sig.last_computed_at DESC NULLS LAST,
                                sig.product_category ASC NULLS LAST
                   ) AS vendor_row_rank
            FROM b2b_churn_signals sig
            LEFT JOIN LATERAL (
                SELECT support_sentiment,
                       legacy_support_score,
                       new_feature_velocity,
                       employee_growth_rate
                FROM b2b_vendor_snapshots snap
                WHERE snap.vendor_name = sig.vendor_name
                ORDER BY snap.snapshot_date DESC
                LIMIT 1
            ) snap ON TRUE
        """
        + where_clause
        + f"""
        )
        SELECT vendor_name,
               product_category,
               total_reviews,
               churn_intent_count,
               avg_urgency_score,
               avg_rating_normalized,
               nps_proxy,
               price_complaint_rate,
               decision_maker_churn_rate,
               keyword_spike_count,
               insider_signal_count,
               last_computed_at,
               support_sentiment,
               legacy_support_score,
               new_feature_velocity,
               employee_growth_rate
        FROM ranked_signals
        WHERE vendor_row_rank = 1
        ORDER BY employee_growth_rate DESC NULLS LAST,
                 support_sentiment ASC NULLS LAST,
                 legacy_support_score ASC NULLS LAST,
                 new_feature_velocity DESC NULLS LAST,
                 avg_urgency_score DESC,
                 last_computed_at DESC NULLS LAST
        LIMIT ${idx}
        """,
        *params,
        capped_limit,
    )
    return [dict(row) for row in rows]


async def read_vendor_top_competitor_map(
    pool,
    *,
    vendor_names: Iterable[Any] | None = None,
) -> dict[str, str]:
    """Read the top named competitor per vendor from derived scorecard rows."""
    rows = await read_vendor_scorecard_details(
        pool,
        vendor_names=vendor_names,
    )
    result: dict[str, str] = {}
    for row in rows:
        vendor = str(row.get("vendor_name") or "").strip()
        if not vendor or vendor in result:
            continue
        flat_competitor = str(row.get("top_competitor") or "").strip()
        if flat_competitor:
            result[vendor] = flat_competitor
            continue
        competitors = _safe_json(row.get("top_competitors"), default=[])
        if not isinstance(competitors, list) or not competitors:
            continue
        first = competitors[0]
        if isinstance(first, dict):
            competitor = str(first.get("name") or first.get("competitor") or "").strip()
        else:
            competitor = str(first or "").strip()
        if competitor:
            result[vendor] = competitor
    return result


async def read_company_churn_context(
    pool,
    *,
    company_hint: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Read derived churn context rows matching a company hint."""
    normalized_hint = str(company_hint or "").strip()
    if not normalized_hint:
        return []
    rows = await pool.fetch(
        """
        SELECT vendor_name,
               product_category,
               avg_urgency_score,
               top_pain_categories,
               top_competitors,
               decision_maker_churn_rate,
               price_complaint_rate
        FROM b2b_churn_signals
        WHERE EXISTS (
            SELECT 1
            FROM jsonb_array_elements(company_churn_list) AS company_row
            WHERE company_row->>'company' ILIKE '%' || $1 || '%'
        )
        ORDER BY avg_urgency_score DESC
        LIMIT $2
        """,
        normalized_hint,
        limit,
    )
    return [
        {
            "vendor_name": row["vendor_name"],
            "product_category": row["product_category"],
            "avg_urgency_score": float(row["avg_urgency_score"]) if row["avg_urgency_score"] else 0.0,
            "top_pain_categories": _safe_json(row.get("top_pain_categories"), default=[]),
            "top_competitors": _safe_json(row.get("top_competitors"), default=[]),
            "decision_maker_churn_rate": (
                float(row["decision_maker_churn_rate"])
                if row["decision_maker_churn_rate"] is not None
                else None
            ),
            "price_complaint_rate": (
                float(row["price_complaint_rate"])
                if row["price_complaint_rate"] is not None
                else None
            ),
        }
        for row in rows
        if row.get("vendor_name")
    ]


async def read_signal_product_categories(
    pool,
) -> list[str]:
    """Read distinct non-empty product categories from vendor scorecards."""
    rows = await pool.fetch(
        """
        SELECT DISTINCT product_category
        FROM b2b_churn_signals
        WHERE product_category IS NOT NULL
          AND TRIM(product_category) != ''
        ORDER BY product_category
        """
    )
    categories: list[str] = []
    seen: set[str] = set()
    for row in rows:
        category = str(row.get("product_category") or "").strip()
        if not category:
            continue
        key = category.lower()
        if key in seen:
            continue
        seen.add(key)
        categories.append(category)
    return categories


async def read_category_vendor_signal_rows(
    pool,
    *,
    product_category: str,
) -> list[dict[str, Any]]:
    """Read one scorecard-backed vendor row per vendor within a category."""
    normalized_category = str(product_category or "").strip()
    if not normalized_category:
        return []

    rows = await pool.fetch(
        """
        WITH ranked_signals AS (
            SELECT vendor_name,
                   total_reviews,
                   avg_urgency_score,
                   confidence_score,
                   last_computed_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY vendor_name
                       ORDER BY total_reviews DESC,
                                last_computed_at DESC NULLS LAST,
                                product_category ASC NULLS LAST
                   ) AS vendor_row_rank
            FROM b2b_churn_signals
            WHERE LOWER(product_category) = LOWER($1)
        )
        SELECT rs.vendor_name,
               rs.total_reviews,
               rs.avg_urgency_score AS avg_urgency,
               rs.confidence_score,
               snap.churn_density,
               snap.positive_review_pct,
               snap.displacement_edge_count,
               COALESCE(d_out.cnt, 0) AS displacement_out,
               COALESCE(d_in.cnt, 0) AS displacement_in
        FROM ranked_signals rs
        LEFT JOIN (
            SELECT DISTINCT ON (vendor_name) *
            FROM b2b_vendor_snapshots
            ORDER BY vendor_name, snapshot_date DESC
        ) snap ON LOWER(rs.vendor_name) = LOWER(snap.vendor_name)
        LEFT JOIN (
            SELECT from_vendor, SUM(mention_count) AS cnt
            FROM b2b_displacement_edges
            GROUP BY from_vendor
        ) d_out ON LOWER(rs.vendor_name) = LOWER(d_out.from_vendor)
        LEFT JOIN (
            SELECT to_vendor, SUM(mention_count) AS cnt
            FROM b2b_displacement_edges
            GROUP BY to_vendor
        ) d_in ON LOWER(rs.vendor_name) = LOWER(d_in.to_vendor)
        WHERE rs.vendor_row_rank = 1
        ORDER BY rs.total_reviews DESC, rs.vendor_name ASC
        """,
        normalized_category,
    )
    return [dict(row) for row in rows if row.get("vendor_name")]


async def read_vendor_graph_sync_rows(
    pool,
) -> list[dict[str, Any]]:
    """Read vendor rows for knowledge-graph sync from canonical scorecard state."""
    rows = await pool.fetch(
        """
        WITH ranked_signals AS (
            SELECT vendor_name,
                   product_category,
                   total_reviews,
                   avg_urgency_score,
                   confidence_score,
                   last_computed_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY vendor_name
                       ORDER BY total_reviews DESC,
                                last_computed_at DESC NULLS LAST,
                                product_category ASC NULLS LAST
                   ) AS vendor_row_rank
            FROM b2b_churn_signals
        )
        SELECT v.canonical_name,
               v.aliases,
               rs.product_category,
               rs.total_reviews,
               rs.avg_urgency_score AS avg_urgency,
               rs.confidence_score,
               snap.churn_density,
               snap.positive_review_pct,
               snap.recommend_ratio,
               snap.pain_count,
               snap.competitor_count
        FROM b2b_vendors v
        LEFT JOIN ranked_signals rs
            ON LOWER(v.canonical_name) = LOWER(rs.vendor_name)
           AND rs.vendor_row_rank = 1
        LEFT JOIN (
            SELECT DISTINCT ON (vendor_name) *
            FROM b2b_vendor_snapshots
            ORDER BY vendor_name, snapshot_date DESC
        ) snap ON LOWER(v.canonical_name) = LOWER(snap.vendor_name)
        ORDER BY v.canonical_name
        """
    )
    return [dict(row) for row in rows if row.get("canonical_name")]


async def read_vendor_scorecard_inventory_rows(
    pool,
) -> list[dict[str, Any]]:
    """Read scorecard-derived inventory rows for scrape coverage planning."""
    rows = await pool.fetch(
        """
        SELECT vendor_name,
               COALESCE(NULLIF(TRIM(product_category), ''), 'Unknown') AS product_category,
               MAX(total_reviews) AS total_reviews_analyzed,
               0.0::double precision AS confidence_score,
               NULL::timestamptz AS last_computed_at,
               'b2b_churn_signals' AS inventory_source
        FROM b2b_churn_signals
        WHERE vendor_name IS NOT NULL
          AND TRIM(vendor_name) != ''
        GROUP BY vendor_name, COALESCE(NULLIF(TRIM(product_category), ''), 'Unknown')
        """
    )
    return [dict(row) for row in rows]


async def read_vendor_scorecard_metrics(
    pool,
    *,
    vendor_name: str,
) -> dict[str, Any] | None:
    """Read the latest scorecard metrics row for one vendor."""
    row = await pool.fetchrow(
        """
        SELECT price_complaint_rate,
               decision_maker_churn_rate,
               total_reviews,
               signal_reviews,
               churn_intent_count,
               avg_urgency_score,
               avg_rating_normalized,
               top_competitors,
               sentiment_distribution,
               materialization_run_id
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor_name,
    )
    return dict(row) if row else None


async def read_vendor_scorecard_archetypes(
    pool,
    *,
    vendor_names: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Read latest non-null vendor archetypes from the derived scorecard table."""
    requested_vendors = _canonicalize_vendor_name_filters(vendor_names)
    if not requested_vendors:
        return []
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name,
               archetype,
               archetype_confidence
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = ANY($1::text[])
          AND archetype IS NOT NULL
        ORDER BY vendor_name,
                 last_computed_at DESC NULLS LAST,
                 total_reviews DESC NULLS LAST
        """,
        requested_vendors,
    )
    return [
        dict(row)
        for row in rows
        if row.get("vendor_name") and row.get("archetype")
    ]


async def read_vendor_intelligence_records_latest(
    pool,
    *,
    vendor_names: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Read the latest canonical vendor intelligence row per vendor across all windows."""
    requested_vendors = _canonicalize_vendor_name_filters(vendor_names)
    if not requested_vendors:
        return []
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name,
               as_of_date,
               analysis_window_days,
               schema_version,
               materialization_run_id,
               vault,
               created_at
        FROM b2b_evidence_vault
        WHERE LOWER(vendor_name) = ANY($1::text[])
        ORDER BY vendor_name, as_of_date DESC, created_at DESC
        """,
        requested_vendors,
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _normalize_vendor_intelligence_record(row)
        if record is not None:
            records.append(record)
    return records


async def read_market_landscape_candidates(
    pool,
    *,
    min_vendor_profiles: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Read category-level market landscape candidates from profile + scorecard joins."""
    rows = await pool.fetch(
        """
        SELECT
            pp.product_category AS category,
            COUNT(DISTINCT pp.vendor_name) AS vendor_count,
            COALESCE(SUM(cs.total_reviews), 0) AS total_reviews,
            ROUND(AVG(cs.avg_urgency_score)::numeric, 1) AS avg_urgency
        FROM b2b_product_profiles pp
        JOIN b2b_churn_signals cs
          ON LOWER(cs.vendor_name) = LOWER(pp.vendor_name)
         AND LOWER(COALESCE(cs.product_category, '')) = LOWER(COALESCE(pp.product_category, ''))
        WHERE pp.product_category IS NOT NULL AND pp.product_category != ''
        GROUP BY pp.product_category
        HAVING COUNT(DISTINCT pp.vendor_name) >= $1
        ORDER BY COUNT(DISTINCT pp.vendor_name) DESC, COALESCE(SUM(cs.total_reviews), 0) DESC
        LIMIT $2
        """,
        min_vendor_profiles,
        limit,
    )
    return [
        {
            "category": row["category"],
            "vendor_count": row["vendor_count"],
            "total_reviews": row["total_reviews"],
            "avg_urgency": float(row["avg_urgency"]),
        }
        for row in rows
    ]


async def read_vendor_alternative_candidates(
    pool,
    *,
    min_avg_urgency: float = 6.0,
    min_total_reviews: int = 5,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Read vendor-alternative blog candidates from churn scorecards."""
    rows = await pool.fetch(
        """
        SELECT
            cs.vendor_name AS vendor,
            cs.product_category AS category,
            cs.avg_urgency_score AS urgency,
            cs.total_reviews AS review_count,
            ap.id AS affiliate_id,
            ap.name AS affiliate_name,
            ap.product_name AS affiliate_product,
            ap.affiliate_url
        FROM b2b_churn_signals cs
        LEFT JOIN affiliate_partners ap
            ON LOWER(ap.category) = LOWER(cs.product_category)
            AND ap.enabled = true
        WHERE cs.avg_urgency_score >= $1
          AND cs.total_reviews >= $2
        ORDER BY cs.avg_urgency_score * cs.total_reviews DESC
        LIMIT $3
        """,
        min_avg_urgency,
        min_total_reviews,
        limit,
    )
    return [
        {
            "vendor": row["vendor"],
            "category": row["category"],
            "urgency": float(row["urgency"]),
            "review_count": row["review_count"],
            "has_affiliate": row["affiliate_id"] is not None,
            "affiliate_id": str(row["affiliate_id"]) if row["affiliate_id"] else None,
            "affiliate_name": row["affiliate_name"],
            "affiliate_product": row["affiliate_product"],
            "affiliate_url": row["affiliate_url"],
        }
        for row in rows
    ]


async def read_vendor_showdown_candidates(
    pool,
    *,
    min_total_reviews: int = 10,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Read vendor-showdown blog candidates from churn scorecards."""
    rows = await pool.fetch(
        """
        SELECT
            a.vendor_name AS vendor_a, b.vendor_name AS vendor_b,
            a.product_category AS category,
            a.total_reviews AS reviews_a, b.total_reviews AS reviews_b,
            (a.total_reviews + b.total_reviews) AS total_reviews,
            a.avg_urgency_score AS urgency_a, b.avg_urgency_score AS urgency_b,
            ABS(a.avg_urgency_score - b.avg_urgency_score) AS pain_diff
        FROM b2b_churn_signals a
        JOIN b2b_churn_signals b
            ON a.product_category = b.product_category
            AND a.vendor_name < b.vendor_name
        WHERE a.total_reviews >= $1 AND b.total_reviews >= $1
        ORDER BY (a.total_reviews + b.total_reviews) DESC
        LIMIT $2
        """,
        min_total_reviews,
        limit,
    )
    return [
        {
            "vendor_a": row["vendor_a"],
            "vendor_b": row["vendor_b"],
            "category": row["category"],
            "reviews_a": row["reviews_a"],
            "reviews_b": row["reviews_b"],
            "total_reviews": row["total_reviews"],
            "urgency_a": round(float(row["urgency_a"]), 1),
            "urgency_b": round(float(row["urgency_b"]), 1),
            "pain_diff": round(float(row["pain_diff"]), 1),
        }
        for row in rows
    ]


async def read_churn_report_candidates(
    pool,
    *,
    min_negative_reviews: int = 8,
    min_avg_urgency: float = 6.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Read churn-report blog candidates from churn scorecards."""
    rows = await pool.fetch(
        """
        SELECT
            vendor_name AS vendor,
            product_category AS category,
            negative_reviews,
            avg_urgency_score AS avg_urgency,
            total_reviews
        FROM b2b_churn_signals
        WHERE negative_reviews >= $1
          AND avg_urgency_score >= $2
        ORDER BY negative_reviews * avg_urgency_score DESC
        LIMIT $3
        """,
        min_negative_reviews,
        min_avg_urgency,
        limit,
    )
    return [
        {
            "vendor": row["vendor"],
            "category": row["category"],
            "negative_reviews": row["negative_reviews"],
            "avg_urgency": round(float(row["avg_urgency"]), 1),
            "total_reviews": row["total_reviews"],
        }
        for row in rows
    ]


async def read_category_vendor_rows(
    pool,
    *,
    category_names: Iterable[Any],
    limit: int | None = None,
    require_scorecard_match: bool = False,
) -> list[dict[str, Any]]:
    """Read vendor rows for one or more product categories."""
    normalized_categories = sorted(
        {
            str(value or "").strip().lower()
            for value in category_names
            if str(value or "").strip()
        }
    )
    if not normalized_categories:
        return []

    limit_clause = " LIMIT $2" if limit is not None else ""
    if require_scorecard_match:
        query = (
            """
            SELECT DISTINCT pp.vendor_name, pp.product_category
            FROM b2b_product_profiles pp
            JOIN b2b_churn_signals cs
              ON LOWER(cs.vendor_name) = LOWER(pp.vendor_name)
             AND LOWER(COALESCE(cs.product_category, '')) = LOWER(COALESCE(pp.product_category, ''))
            WHERE LOWER(pp.product_category) = ANY($1::text[])
            ORDER BY pp.vendor_name
            """
            + limit_clause
        )
    else:
        query = (
            """
            SELECT DISTINCT vendor_name, product_category
            FROM b2b_product_profiles
            WHERE LOWER(product_category) = ANY($1::text[])
            ORDER BY vendor_name
            """
            + limit_clause
        )
    rows = await pool.fetch(
        query,
        normalized_categories,
        *([limit] if limit is not None else []),
    )
    return [dict(row) for row in rows if row.get("vendor_name")]


async def read_known_vendor_names(
    pool,
) -> list[str]:
    """Read the canonical known-vendor list from churn scorecards."""
    rows = await pool.fetch(
        """
        SELECT DISTINCT vendor_name
        FROM b2b_churn_signals
        WHERE vendor_name IS NOT NULL AND TRIM(vendor_name) != ''
        ORDER BY vendor_name
        """
    )
    return [str(row["vendor_name"]) for row in rows if row.get("vendor_name")]


async def _fetch_vendor_churn_scores_from_signals(
    pool,
    window_days: int,
    min_reviews: int,
) -> list[dict[str, Any]]:
    """Deprecated wrapper. Use ``read_vendor_scorecards`` instead."""
    return await read_vendor_scorecards(
        pool,
        window_days=window_days,
        min_reviews=min_reviews,
    )


async def _fetch_high_intent_companies(
    pool,
    urgency_threshold: float,
    window_days: int,
    *,
    vendor_name: str | None = None,
    scoped_vendors: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Companies showing high churn intent -- the money feed."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=2, source_param=3, alias="r")
    params: list = [urgency_threshold, window_days, sources]
    extra_where = ""
    signal_where = ""
    if _high_intent_signal_evidence_enabled():
        signal_where = """
          AND (
                COALESCE((r.enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
             OR COALESCE((r.enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
             OR COALESCE((r.enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
             OR COALESCE((r.enrichment->'urgency_indicators'->>'explicit_cancel_language')::boolean, false)
             OR COALESCE((r.enrichment->'urgency_indicators'->>'active_migration_language')::boolean, false)
             OR COALESCE((r.enrichment->'urgency_indicators'->>'active_evaluation_language')::boolean, false)
             OR COALESCE((r.enrichment->'urgency_indicators'->>'completed_switch_language')::boolean, false)
          )
        """
    idx = 4
    if vendor_name:
        extra_where += f"\n          AND r.vendor_name ILIKE '%' || ${idx} || '%'"
        params.append(vendor_name)
        idx += 1
    if scoped_vendors is not None:
        if not scoped_vendors:
            return []  # scoped user with no tracked vendors = zero results
        extra_where += f"\n          AND r.vendor_name = ANY(${idx}::text[])"
        params.append(scoped_vendors)
        idx += 1
    limit_clause = ""
    if limit is not None:
        limit_clause = f"\n        LIMIT ${idx}"
        params.append(limit)
        idx += 1
    rows = await pool.fetch(
        f"""
        SELECT r.id AS review_id, r.source,
            COALESCE(
                CASE
                    WHEN ar.confidence_label IN ('high', 'medium')
                    THEN ar.resolved_company_name
                    ELSE NULL
                END,
                r.reviewer_company
            ) AS reviewer_company,
            r.reviewer_company AS raw_reviewer_company,
            ar.confidence_label AS resolution_confidence,
            r.vendor_name, r.product_category,
            r.reviewer_title,
            r.company_size_raw,
            COALESCE(poc.industry, r.reviewer_industry,
                     r.enrichment->'reviewer_context'->>'industry') AS industry,
            poc.employee_count AS verified_employee_count,
            poc.country AS company_country,
            poc.domain AS company_domain,
            poc.annual_revenue_range AS revenue_range,
            poc.founded_year,
            poc.total_funding,
            poc.latest_funding_stage,
            poc.headcount_growth_6m,
            poc.headcount_growth_12m,
            poc.headcount_growth_24m,
            poc.publicly_traded_exchange,
            poc.publicly_traded_symbol,
            poc.short_description AS company_description,
            r.enrichment->'reviewer_context'->>'role_level' AS role_level,
            (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS is_dm,
            (r.enrichment->>'urgency_score')::numeric AS urgency,
            r.enrichment->>'pain_category' AS pain,
            r.enrichment->'competitors_mentioned' AS alternatives,
            r.enrichment->'quotable_phrases' AS quotes,
            r.enrichment->'contract_context'->>'contract_value_signal' AS value_signal,
            r.enrichment->'budget_signals'->>'seat_count' AS seat_count,
            r.enrichment->'use_case'->>'lock_in_level' AS lock_in_level,
            r.enrichment->'timeline'->>'contract_end' AS contract_end,
            r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
            r.relevance_score,
            r.author_churn_score,
            (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
            (r.enrichment->'churn_signals'->>'actively_evaluating')::boolean AS actively_evaluating,
            (r.enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean AS contract_renewal_mentioned,
            -- v2 urgency indicators for richer account intelligence
            (r.enrichment->'urgency_indicators'->>'explicit_cancel_language')::boolean AS indicator_cancel,
            (r.enrichment->'urgency_indicators'->>'active_migration_language')::boolean AS indicator_migration,
            (r.enrichment->'urgency_indicators'->>'active_evaluation_language')::boolean AS indicator_evaluation,
            (r.enrichment->'urgency_indicators'->>'completed_switch_language')::boolean AS indicator_switch
        FROM b2b_reviews r
        LEFT JOIN b2b_account_resolution ar
            ON ar.review_id = r.id AND ar.resolution_status = 'resolved'
        LEFT JOIN prospect_org_cache poc
            ON poc.company_name_norm = CASE
                WHEN ar.confidence_label IN ('high', 'medium')
                THEN ar.normalized_company_name
                ELSE NULL
            END
        WHERE {filters}{extra_where}
          AND (r.enrichment->>'urgency_score')::numeric >= $1
          AND r.reviewer_company IS NOT NULL AND r.reviewer_company != ''
          AND COALESCE(r.relevance_score, 0.5) >= 0.3
          {signal_where}
        ORDER BY (r.enrichment->>'urgency_score')::numeric
                 * (0.7 + 0.3 * COALESCE(r.relevance_score, 0.5)) DESC{limit_clause}
        """,
        *params,
    )
    results = []
    for r in rows:
        alternatives = _safe_json(r["alternatives"])
        blocked_names = {
            normalize_company_name(name)
            for name in _extract_alternative_names(alternatives)
            if normalize_company_name(name)
        }
        if _company_signal_exclusion_reason(
            r["reviewer_company"],
            current_vendor=r["vendor_name"],
            blocked_names=blocked_names,
            source=r["source"],
            confidence_score=None,
        ):
            continue
        if _high_intent_signal_evidence_enabled() and not _high_intent_row_has_signal_evidence(dict(r)):
            continue
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
            "raw_company": r["raw_reviewer_company"],
            "resolution_confidence": r["resolution_confidence"],
            "vendor": r["vendor_name"],
            "category": r["product_category"],
            "title": r["reviewer_title"],
            "company_size": r["company_size_raw"],
            "industry": r["industry"],
            "verified_employee_count": r["verified_employee_count"],
            "company_country": r["company_country"],
            "company_domain": r["company_domain"],
            "revenue_range": r["revenue_range"],
            "founded_year": r["founded_year"],
            "total_funding": r["total_funding"],
            "funding_stage": r["latest_funding_stage"],
            "headcount_growth_6m": float(r["headcount_growth_6m"]) if r["headcount_growth_6m"] is not None else None,
            "headcount_growth_12m": float(r["headcount_growth_12m"]) if r["headcount_growth_12m"] is not None else None,
            "headcount_growth_24m": float(r["headcount_growth_24m"]) if r["headcount_growth_24m"] is not None else None,
            "publicly_traded": r["publicly_traded_exchange"] or None,
            "ticker": r["publicly_traded_symbol"] or None,
            "company_description": r["company_description"],
            "role_level": r["role_level"],
            "decision_maker": r["is_dm"],
            "urgency": urgency,
            "pain": r["pain"],
            "alternatives": alternatives,
            "quotes": _safe_json(r["quotes"]),
            "contract_signal": r["value_signal"],
            "review_id": str(r["review_id"]) if r["review_id"] else None,
            "source": r["source"],
            "seat_count": seat_count,
            "lock_in_level": r["lock_in_level"],
            "contract_end": r["contract_end"],
            "buying_stage": r["buying_stage"],
            "relevance_score": float(r["relevance_score"]) if r["relevance_score"] is not None else None,
            "author_churn_score": float(r["author_churn_score"]) if r["author_churn_score"] is not None else None,
            "intent_signals": {
                "cancel": bool(r.get("indicator_cancel")),
                "migration": bool(r.get("indicator_migration")),
                "evaluation": bool(r.get("indicator_evaluation")),
                "completed_switch": bool(r.get("indicator_switch")),
            },
        })
    return results


async def _fetch_existing_company_signals(
    pool,
    *,
    window_days: int,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch canonical company-signal rows still active in the analysis window."""
    rows = await pool.fetch(
        """
        SELECT cs.company_name, cs.vendor_name, cs.urgency_score,
               pain_category, buyer_role, decision_maker,
               seat_count, contract_end, buying_stage,
               cs.review_id, cs.source, cs.confidence_score,
               cs.first_seen_at, cs.last_seen_at,
               r.content_type,
               (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (r.enrichment->'churn_signals'->>'actively_evaluating')::boolean AS actively_evaluating,
               (r.enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean AS contract_renewal_mentioned,
               (r.enrichment->'urgency_indicators'->>'explicit_cancel_language')::boolean AS indicator_cancel,
               (r.enrichment->'urgency_indicators'->>'active_migration_language')::boolean AS indicator_migration,
               (r.enrichment->'urgency_indicators'->>'active_evaluation_language')::boolean AS indicator_evaluation,
               (r.enrichment->'urgency_indicators'->>'completed_switch_language')::boolean AS indicator_switch,
               r.reviewer_title,
               r.company_size_raw,
               COALESCE(
                   poc.industry,
                   r.reviewer_industry,
                   r.enrichment->'reviewer_context'->>'industry'
               ) AS industry,
               poc.employee_count AS verified_employee_count,
               poc.country AS company_country,
               poc.domain AS company_domain,
               poc.founded_year,
               poc.total_funding,
               poc.latest_funding_stage,
               poc.headcount_growth_6m,
               poc.headcount_growth_12m,
               poc.headcount_growth_24m,
               poc.publicly_traded_exchange,
               poc.publicly_traded_symbol,
               poc.short_description AS company_description,
               r.enrichment->'competitors_mentioned' AS alternatives,
               r.enrichment->'quotable_phrases' AS quotes
        FROM b2b_company_signals cs
        LEFT JOIN b2b_reviews r ON r.id = cs.review_id
        LEFT JOIN b2b_account_resolution ar
            ON ar.review_id = cs.review_id AND ar.resolution_status = 'resolved'
        LEFT JOIN prospect_org_cache poc
            ON poc.company_name_norm = ar.normalized_company_name
        WHERE cs.last_seen_at >= NOW() - make_interval(days => $1)
        ORDER BY cs.vendor_name, cs.urgency_score DESC NULLS LAST, cs.last_seen_at DESC
        """,
        window_days,
    )
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        if _high_intent_signal_evidence_enabled() and not _high_intent_row_has_signal_evidence(dict(row)):
            continue
        lookup.setdefault(vendor, []).append({
            "company_name": row.get("company_name"),
            "vendor_name": vendor,
            "urgency_score": float(row["urgency_score"]) if row.get("urgency_score") is not None else None,
            "pain_category": row.get("pain_category"),
            "buyer_role": row.get("buyer_role"),
            "decision_maker": row.get("decision_maker"),
            "seat_count": row.get("seat_count"),
            "contract_end": str(row.get("contract_end") or "") or None,
            "buying_stage": row.get("buying_stage"),
            "review_id": str(row.get("review_id") or "") or None,
            "source": row.get("source"),
            "content_type": row.get("content_type"),
            "confidence_score": float(row["confidence_score"]) if row.get("confidence_score") is not None else None,
            "first_seen_at": str(row.get("first_seen_at") or "") or None,
            "last_seen_at": str(row.get("last_seen_at") or "") or None,
            "title": row.get("reviewer_title"),
            "company_size": row.get("company_size_raw"),
            "industry": row.get("industry"),
            "verified_employee_count": row.get("verified_employee_count"),
            "company_country": row.get("company_country"),
            "company_domain": row.get("company_domain"),
            "founded_year": row.get("founded_year"),
            "total_funding": row.get("total_funding"),
            "funding_stage": row.get("latest_funding_stage"),
            "headcount_growth_6m": float(row["headcount_growth_6m"]) if row.get("headcount_growth_6m") is not None else None,
            "headcount_growth_12m": float(row["headcount_growth_12m"]) if row.get("headcount_growth_12m") is not None else None,
            "headcount_growth_24m": float(row["headcount_growth_24m"]) if row.get("headcount_growth_24m") is not None else None,
            "publicly_traded": row.get("publicly_traded_exchange") or None,
            "ticker": row.get("publicly_traded_symbol") or None,
            "company_description": row.get("company_description"),
            "alternatives": _safe_json(row.get("alternatives"), default=[]),
            "quotes": _safe_json(row.get("quotes"), default=[]),
        })
    return lookup


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
                WHERE (enrichment->'reviewer_context'->>'decision_maker')::boolean IS TRUE
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
    """What's driving churn per vendor.

    Uses the multi-label pain_categories array when present (primary=1.0,
    secondary=0.4, minor=0.1 weights) so that secondary signals are counted
    rather than dropped.  Falls back to the singular pain_category field for
    reviews that pre-date the multi-label schema.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        WITH base AS (
            SELECT vendor_name,
                   (enrichment->>'urgency_score')::numeric AS urgency,
                   enrichment->'pain_categories'           AS cats,
                   enrichment->>'pain_category'            AS fallback_pain,
                   enrichment->>'pain_cluster'             AS pain_cluster
            FROM b2b_reviews
            WHERE {filters}
        ),
        pain_labels AS (
            -- Multi-label path: unnest pain_categories array with severity weights.
            -- When a label is the generic fallback bucket and pain_cluster is set,
            -- substitute the cluster name so reports show something more specific.
            SELECT b.vendor_name,
                   b.urgency,
                   CASE
                       WHEN p.value->>'category' IN ('other', 'general_dissatisfaction', 'overall_dissatisfaction') AND b.pain_cluster IS NOT NULL
                       THEN b.pain_cluster
                       WHEN p.value->>'category' IN ('other', 'general_dissatisfaction', 'overall_dissatisfaction')
                       THEN 'overall_dissatisfaction'
                       ELSE p.value->>'category'
                   END AS pain,
                   CASE p.value->>'severity'
                       WHEN 'primary'   THEN 1.0
                       WHEN 'secondary' THEN 0.4
                       WHEN 'minor'     THEN 0.1
                       ELSE 1.0
                   END AS weight
            FROM base b
            CROSS JOIN LATERAL jsonb_array_elements(
                COALESCE(b.cats, '[]'::jsonb)
            ) AS p(value)
            WHERE jsonb_array_length(COALESCE(b.cats, '[]'::jsonb)) > 0
              AND p.value->>'category' IS NOT NULL
              AND p.value->>'category' != ''

            UNION ALL

            -- Fallback: reviews with no pain_categories array.
            -- Same cluster substitution for generic fallback labels.
            SELECT b.vendor_name,
                   b.urgency,
                   CASE
                       WHEN b.fallback_pain IN ('other', 'general_dissatisfaction', 'overall_dissatisfaction') AND b.pain_cluster IS NOT NULL
                       THEN b.pain_cluster
                       WHEN b.fallback_pain IN ('other', 'general_dissatisfaction', 'overall_dissatisfaction')
                       THEN 'overall_dissatisfaction'
                       ELSE b.fallback_pain
                   END AS pain,
                   1.0 AS weight
            FROM base b
            WHERE jsonb_array_length(COALESCE(b.cats, '[]'::jsonb)) = 0
              AND b.fallback_pain IS NOT NULL
              AND b.fallback_pain != ''
        )
        SELECT vendor_name,
               pain,
               round(sum(weight)::numeric, 1) AS complaint_count,
               avg(urgency)                   AS avg_urgency
        FROM pain_labels
        GROUP BY vendor_name, pain
        ORDER BY complaint_count DESC
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "pain": r["pain"],
            "complaint_count": float(r["complaint_count"]),
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
            count(*) FILTER (
                WHERE (enrichment->'contract_context'->>'price_complaint')::boolean = true
            ) AS price_complaint_count,
            count(*) FILTER (
                WHERE jsonb_array_length(COALESCE(enrichment->'pricing_phrases', '[]'::jsonb)) > 0
            ) AS pricing_phrases_count,
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
            "price_complaint_rate": max(
                r["pricing_count"] / r["total"] if r["total"] else 0,
                r["price_complaint_count"] / r["total"] if r["total"] else 0,
            ),
            "pricing_phrases_rate": r["pricing_phrases_count"] / r["total"] if r["total"] else 0,
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
                'decision_maker', (enrichment->'reviewer_context'->>'decision_maker')::boolean,
                'buying_stage', enrichment->'buyer_authority'->>'buying_stage',
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


async def _fetch_quotable_evidence(pool, window_days: int, *, min_urgency: float = 4.5) -> list[dict[str, Any]]:
    """Top quotable phrases per vendor (highest urgency, deduplicated).

    Each quote is a dict with 'quote', 'urgency', and 'review_id' for provenance.
    """
    sources = _executive_source_list()
    filters = _eligible_review_filters(window_param=1, source_param=3)
    rows = await pool.fetch(
        f"""
        WITH review_best AS (
            SELECT DISTINCT ON (vendor_name, id)
                vendor_name, id AS review_id, phrase.value AS quote,
                (enrichment->>'urgency_score')::numeric AS urgency,
                source, reviewed_at, rating, rating_max,
                reviewer_company, reviewer_title, company_size_raw,
                COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry,
                enrichment->'churn_signals' AS churn_signals,
                enrichment->'salience_flags' AS salience_flags
            FROM b2b_reviews
            CROSS JOIN LATERAL jsonb_array_elements_text(
                COALESCE(enrichment->'quotable_phrases', '[]'::jsonb)
            ) AS phrase(value)
            WHERE {filters}
              AND (enrichment->>'urgency_score')::numeric >= $2
            ORDER BY vendor_name, id, length(phrase.value) DESC
        ),
        ranked_quotes AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY vendor_name
                    ORDER BY
                        CASE WHEN (churn_signals->>'intent_to_leave')::boolean IS TRUE THEN 0 ELSE 1 END,
                        jsonb_array_length(COALESCE(salience_flags, '[]'::jsonb)) DESC,
                        urgency DESC
                ) AS rn
            FROM review_best
        )
        SELECT vendor_name,
            jsonb_agg(
                jsonb_build_object(
                    'quote', quote,
                    'urgency', urgency,
                    'review_id', review_id,
                    'source_site', source,
                    'reviewed_at', reviewed_at,
                    'rating', rating,
                    'rating_max', rating_max,
                    'company', reviewer_company,
                    'title', reviewer_title,
                    'company_size', company_size_raw,
                    'industry', industry
                ) ORDER BY rn
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


async def _fetch_evidence_vault_review_rows(
    pool,
    window_days: int,
) -> list[dict[str, Any]]:
    """Fetch review-level evidence rows used for pass-2 vault aggregation."""
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT id AS review_id,
            vendor_name,
            source,
            reviewed_at,
            enriched_at,
            rating,
            rating_max,
            reviewer_title,
            company_size_raw,
            COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry,
            enrichment->'reviewer_context'->>'role_level' AS role_level,
            enrichment->>'pain_category' AS pain_category,
            COALESCE(enrichment->'feature_gaps', '[]'::jsonb) AS feature_gaps,
            COALESCE(enrichment->'positive_aspects', '[]'::jsonb) AS positive_aspects,
            (enrichment->>'urgency_score')::numeric AS urgency
        FROM b2b_reviews
        WHERE {filters}
        """,
        window_days,
        sources,
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append({
            "review_id": str(row["review_id"]) if row.get("review_id") else None,
            "vendor_name": row.get("vendor_name"),
            "source": row.get("source"),
            "reviewed_at": row.get("reviewed_at"),
            "enriched_at": row.get("enriched_at"),
            "rating": float(row["rating"]) if row.get("rating") is not None else None,
            "rating_max": float(row["rating_max"]) if row.get("rating_max") is not None else None,
            "reviewer_title": row.get("reviewer_title"),
            "company_size_raw": row.get("company_size_raw"),
            "industry": row.get("industry"),
            "role_level": row.get("role_level"),
            "pain_category": row.get("pain_category"),
            "feature_gaps": _safe_json(row.get("feature_gaps"), default=[]),
            "positive_aspects": _safe_json(row.get("positive_aspects"), default=[]),
            "urgency": float(row["urgency"]) if row.get("urgency") is not None else 0.0,
        })
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
                COUNT(DISTINCT CASE
                    WHEN (enrichment->'insider_signals'->>'departures_mentioned')::boolean = true
                      OR (enrichment->'insider_signals'->'talent_drain'->>'departures_mentioned')::boolean = true
                    THEN id END)::numeric
                / NULLIF(COUNT(DISTINCT id), 0)::numeric,
                4
            ) AS talent_drain_rate,
            -- v2 flattened insider fields (with v1 nested fallback)
            jsonb_agg(DISTINCT jsonb_build_object(
                'bureaucracy_level', COALESCE(
                    enrichment->'insider_signals'->>'bureaucracy_level',
                    enrichment->'insider_signals'->'org_health'->>'bureaucracy_level'
                ),
                'leadership_quality', COALESCE(
                    enrichment->'insider_signals'->>'leadership_quality',
                    enrichment->'insider_signals'->'org_health'->>'leadership_quality'
                ),
                'innovation_climate', COALESCE(
                    enrichment->'insider_signals'->>'innovation_climate',
                    enrichment->'insider_signals'->'org_health'->>'innovation_climate'
                ),
                'morale', COALESCE(
                    enrichment->'insider_signals'->>'morale',
                    enrichment->'insider_signals'->'talent_drain'->>'morale'
                )
            )) FILTER (WHERE enrichment->'insider_signals' IS NOT NULL) AS org_health_array,
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
          AND duplicate_of_review_id IS NULL
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
    normalized: list[dict[str, Any]] = []
    for row in rows:
        profile = dict(row)
        profile["strengths"] = _safe_json(profile.get("strengths"), default=[])
        profile["weaknesses"] = _safe_json(profile.get("weaknesses"), default=[])
        profile["pain_addressed"] = _safe_json(profile.get("pain_addressed"), default={})
        profile["commonly_compared_to"] = _safe_json(profile.get("commonly_compared_to"), default=[])
        profile["commonly_switched_from"] = _safe_json(profile.get("commonly_switched_from"), default=[])
        profile["primary_use_cases"] = _safe_json(profile.get("primary_use_cases"), default=[])
        profile["typical_company_size"] = _safe_json(profile.get("typical_company_size"), default=[])
        profile["typical_industries"] = _safe_json(profile.get("typical_industries"), default=[])
        profile["top_integrations"] = _safe_json(profile.get("top_integrations"), default=[])
        normalized.append(profile)
    return normalized


def _normalize_vendor_intelligence_record(row: Any) -> dict[str, Any] | None:
    """Normalize a latest-row evidence-vault record into a stable dict."""
    if not row:
        return None
    vendor_name = str(row.get("vendor_name") or "").strip()
    if not vendor_name:
        return None
    vault = _safe_json(row.get("vault"), default={})
    if not isinstance(vault, dict):
        vault = {}
    return {
        "vendor_name": vendor_name,
        "as_of_date": row.get("as_of_date"),
        "analysis_window_days": row.get("analysis_window_days"),
        "schema_version": row.get("schema_version"),
        "materialization_run_id": row.get("materialization_run_id"),
        "created_at": row.get("created_at"),
        "vault": vault,
    }


async def read_vendor_intelligence_records(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_names: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """Read the latest canonical vendor intelligence row per vendor."""
    requested_vendors = _canonicalize_vendor_name_filters(vendor_names)
    vendor_filter_clause = "AND LOWER(vendor_name) = ANY($3::text[])" if requested_vendors else ""
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name,
               as_of_date,
               analysis_window_days,
               schema_version,
               materialization_run_id,
               vault,
               created_at
        FROM b2b_evidence_vault
        WHERE as_of_date <= $1
          AND analysis_window_days = $2
          """ + vendor_filter_clause + """
        ORDER BY vendor_name, as_of_date DESC, created_at DESC
        """,
        as_of,
        analysis_window_days,
        *([requested_vendors] if requested_vendors else []),
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _normalize_vendor_intelligence_record(row)
        if record is not None:
            records.append(record)
    return records


async def read_vendor_intelligence_record(
    pool,
    vendor_name: str,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, Any] | None:
    """Read the latest canonical vendor intelligence row for one vendor."""
    canonical_vendor = _canonicalize_vendor(vendor_name)
    if not canonical_vendor:
        return None
    records = await read_vendor_intelligence_records(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
        vendor_names=[vendor_name],
    )
    for record in records:
        if _canonicalize_vendor(record.get("vendor_name") or "") == canonical_vendor:
            return record
    return None


async def search_vendor_intelligence_record(
    pool,
    *,
    vendor_query: str,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, Any] | None:
    """Read the latest canonical vendor intelligence row matching a partial vendor query."""
    normalized_query = str(vendor_query or "").strip()
    if not normalized_query:
        return None
    row = await pool.fetchrow(
        """
        SELECT vendor_name,
               as_of_date,
               analysis_window_days,
               schema_version,
               materialization_run_id,
               vault,
               created_at
        FROM b2b_evidence_vault
        WHERE vendor_name ILIKE '%' || $1 || '%'
          AND as_of_date <= $2
          AND analysis_window_days = $3
        ORDER BY as_of_date DESC, created_at DESC
        LIMIT 1
        """,
        normalized_query,
        as_of,
        analysis_window_days,
    )
    return _normalize_vendor_intelligence_record(row)


async def search_vendor_intelligence_records(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_query: str | None = None,
) -> list[dict[str, Any]]:
    """Read the latest canonical vendor intelligence row for each matching vendor."""
    conditions = [
        "as_of_date <= $1",
        "analysis_window_days = $2",
    ]
    params: list[Any] = [as_of, analysis_window_days]
    if str(vendor_query or "").strip():
        conditions.append("vendor_name ILIKE '%' || $3 || '%'")
        params.append(str(vendor_query).strip())
    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT DISTINCT ON (vendor_name)
               vendor_name,
               as_of_date,
               analysis_window_days,
               schema_version,
               materialization_run_id,
               vault,
               created_at
        FROM b2b_evidence_vault
        WHERE {where}
        ORDER BY vendor_name, as_of_date DESC, created_at DESC
        """,
        *params,
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _normalize_vendor_intelligence_record(row)
        if record is not None:
            records.append(record)
    return records


async def read_vendor_intelligence_record_nearest_window(
    pool,
    *,
    vendor_name: str,
    analysis_window_days: int,
) -> dict[str, Any] | None:
    """Read the nearest-window canonical vendor intelligence row for one vendor."""
    normalized_vendor = _canonicalize_vendor(vendor_name) or str(vendor_name or "").strip()
    if not normalized_vendor:
        return None
    row = await pool.fetchrow(
        """
        SELECT vendor_name,
               as_of_date,
               analysis_window_days,
               schema_version,
               materialization_run_id,
               vault,
               created_at
        FROM b2b_evidence_vault
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY
            CASE WHEN analysis_window_days = $2 THEN 0 ELSE 1 END,
            ABS(analysis_window_days - $2),
            as_of_date DESC,
            created_at DESC
        LIMIT 1
        """,
        normalized_vendor,
        analysis_window_days,
    )
    return _normalize_vendor_intelligence_record(row)


async def read_vendor_intelligence_map(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_names: Iterable[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read canonical vendor intelligence objects from ``b2b_evidence_vault``."""
    records = await read_vendor_intelligence_records(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
        vendor_names=vendor_names,
    )
    vault_lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        vendor = _canonicalize_vendor(record.get("vendor_name") or "")
        if not vendor:
            continue
        vault_lookup[vendor] = record.get("vault") or {}
    return vault_lookup


async def read_vendor_intelligence(
    pool,
    vendor_name: str,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, Any] | None:
    """Read the canonical vendor intelligence object for a single vendor."""
    lookup = await read_vendor_intelligence_map(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
        vendor_names=[vendor_name],
    )
    canonical = _canonicalize_vendor(vendor_name)
    if not canonical:
        return None
    return lookup.get(canonical)


async def _fetch_latest_evidence_vault(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, dict[str, Any]]:
    """Deprecated wrapper. Use ``read_vendor_intelligence_map`` instead."""
    return await read_vendor_intelligence_map(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


async def fetch_all_pool_layers(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load all 6 pool layers and merge into one dict per vendor.

    Returns: {vendor_name: {evidence_vault: {...}, segment: {...},
    temporal: {...}, displacement: [...], category: {...}, accounts: {...}}}
    """
    _POOL_TABLES = [
        ("b2b_evidence_vault", "vault", "evidence_vault"),
        ("b2b_segment_intelligence", "segments", "segment"),
        ("b2b_temporal_intelligence", "temporal", "temporal"),
        ("b2b_account_intelligence", "accounts", "accounts"),
    ]
    generic_categories = {"", "unknown", "b2b software"}
    result: dict[str, dict[str, Any]] = {}
    requested_vendors = sorted(
        {
            _canonicalize_vendor(name).lower()
            for name in (vendor_names or [])
            if _canonicalize_vendor(name)
        }
    )
    vendor_filter_clause = "AND LOWER(vendor_name) = ANY($3::text[])" if requested_vendors else ""
    vendor_filter_args: list[Any] = [requested_vendors] if requested_vendors else []

    for table, col, key in _POOL_TABLES:
        try:
            rows = await pool.fetch(
                f"""
                SELECT DISTINCT ON (vendor_name)
                       vendor_name, {col}
                FROM {table}
                WHERE as_of_date <= $1
                  AND analysis_window_days = $2
                  {vendor_filter_clause}
                ORDER BY vendor_name, as_of_date DESC, created_at DESC
                """,
                as_of,
                analysis_window_days,
                *vendor_filter_args,
            )
            for row in rows:
                vn = _canonicalize_vendor(row.get("vendor_name") or "")
                if not vn:
                    continue
                data = _safe_json(row.get(col), default={})
                if isinstance(data, dict):
                    result.setdefault(vn, {})[key] = data
        except Exception:
            logger.debug("Pool layer %s unavailable", table, exc_info=True)

    # Displacement dynamics: keyed by (from_vendor, to_vendor)
    try:
        displacement_filter_clause = "AND LOWER(from_vendor) = ANY($3::text[])" if requested_vendors else ""
        disp_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (from_vendor, to_vendor)
                   from_vendor, to_vendor, dynamics
            FROM b2b_displacement_dynamics
            WHERE as_of_date <= $1
              AND analysis_window_days = $2
              """ + displacement_filter_clause + """
            ORDER BY from_vendor, to_vendor, as_of_date DESC, created_at DESC
            """,
            as_of,
            analysis_window_days,
            *vendor_filter_args,
        )
        for row in disp_rows:
            fv = _canonicalize_vendor(row.get("from_vendor") or "")
            data = _safe_json(row.get("dynamics"), default={})
            if fv and isinstance(data, dict):
                result.setdefault(fv, {}).setdefault(
                    "displacement", [],
                ).append(data)
    except Exception:
        logger.debug("Displacement dynamics unavailable", exc_info=True)

    # Category dynamics: keyed by category
    try:
        if requested_vendors:
            profile_rows = await pool.fetch(
                """
                SELECT DISTINCT ON (vendor_name)
                       vendor_name, product_category
                FROM b2b_product_profiles
                WHERE LOWER(vendor_name) = ANY($1::text[])
                ORDER BY vendor_name,
                         CASE
                             WHEN lower(COALESCE(product_category, '')) IN ('', 'unknown', 'b2b software')
                             THEN 1 ELSE 0
                         END,
                         total_reviews_analyzed DESC NULLS LAST
                """,
                requested_vendors,
            )
        else:
            profile_rows = await pool.fetch(
                """
                SELECT DISTINCT ON (vendor_name)
                       vendor_name, product_category
                FROM b2b_product_profiles
                ORDER BY vendor_name,
                         CASE
                             WHEN lower(COALESCE(product_category, '')) IN ('', 'unknown', 'b2b software')
                             THEN 1 ELSE 0
                         END,
                         total_reviews_analyzed DESC NULLS LAST
                """,
            )
        preferred_vendor_categories: dict[str, str] = {}
        for row in profile_rows:
            vendor = _canonicalize_vendor(row.get("vendor_name") or "")
            category = str(row.get("product_category") or "").strip()
            if vendor and category:
                preferred_vendor_categories[vendor] = category

        cat_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (category)
                   category, dynamics
            FROM b2b_category_dynamics
            WHERE as_of_date <= $1
              AND analysis_window_days = $2
            ORDER BY category, as_of_date DESC, created_at DESC
            """,
            as_of,
            analysis_window_days,
        )
        cat_map: dict[str, dict] = {}
        for row in cat_rows:
            cat = row.get("category") or ""
            data = _safe_json(row.get("dynamics"), default={})
            if cat and isinstance(data, dict):
                cat_map[cat] = data
        # Attach category to each vendor based on evidence_vault
        for vn, layers in result.items():
            ev = layers.get("evidence_vault") or {}
            cat = str(ev.get("product_category") or "").strip()
            preferred = str(preferred_vendor_categories.get(vn) or "").strip()
            if preferred and preferred in cat_map and cat.lower() in generic_categories:
                cat = preferred
            if cat.lower() in generic_categories:
                continue
            if cat and cat in cat_map:
                layers["category"] = cat_map[cat]
    except Exception:
        logger.debug("Category dynamics unavailable", exc_info=True)

    # Review candidates for witness-pack building.
    try:
        review_filter_clause = "AND LOWER(r.vendor_name) = ANY($3::text[])" if requested_vendors else ""
        review_rows = await pool.fetch(
            """
                SELECT
                    r.vendor_name,
                    r.id,
                    r.source,
                rating,
                rating_max,
                summary,
                review_text,
                pros,
                cons,
                r.reviewer_title,
                COALESCE(
                    r.reviewer_company,
                    CASE
                        WHEN ar.confidence_label IN ('high', 'medium')
                        THEN ar.resolved_company_name
                        ELSE NULL
                    END
                ) AS reviewer_company,
                r.reviewer_company AS raw_reviewer_company,
                ar.confidence_label AS resolution_confidence,
                r.reviewed_at,
                r.imported_at,
                r.raw_metadata,
                r.enrichment
                FROM b2b_reviews r
                LEFT JOIN b2b_account_resolution ar
                  ON ar.review_id = r.id AND ar.resolution_status = 'resolved'
                WHERE r.enrichment_status = 'enriched'
                  AND r.duplicate_of_review_id IS NULL
                  AND COALESCE(r.reviewed_at, r.imported_at) <= ($1::date + INTERVAL '1 day')
                  AND COALESCE(r.reviewed_at, r.imported_at) >= ($1::date - ($2::int * INTERVAL '1 day'))
                  """ + review_filter_clause + """
                ORDER BY
                    r.vendor_name,
                    COALESCE(r.reviewed_at, r.imported_at) DESC,
                r.imported_at DESC,
                r.id DESC
            """,
            as_of,
            analysis_window_days,
            *vendor_filter_args,
        )
        for row in review_rows:
            vendor = _canonicalize_vendor(row.get("vendor_name") or "")
            if not vendor:
                continue
            review = {
                "id": str(row.get("id") or ""),
                "source": row.get("source"),
                "rating": row.get("rating"),
                "rating_max": row.get("rating_max"),
                "summary": row.get("summary"),
                "review_text": row.get("review_text"),
                "pros": row.get("pros"),
                "cons": row.get("cons"),
                "reviewer_title": row.get("reviewer_title"),
                "reviewer_company": row.get("reviewer_company"),
                "raw_reviewer_company": row.get("raw_reviewer_company"),
                "resolution_confidence": row.get("resolution_confidence"),
                "reviewed_at": row.get("reviewed_at"),
                "imported_at": row.get("imported_at"),
                "raw_metadata": _safe_json(row.get("raw_metadata"), default={}),
                "enrichment": _safe_json(row.get("enrichment"), default={}),
            }
            result.setdefault(vendor, {}).setdefault("reviews", []).append(review)
    except Exception:
        logger.debug("Review candidates unavailable", exc_info=True)

    return result


async def _fetch_latest_account_intelligence(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, dict[str, Any]]:
    """Load latest canonical account-intelligence rows per vendor."""
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name, accounts
        FROM b2b_account_intelligence
        WHERE as_of_date <= $1
          AND analysis_window_days = $2
        ORDER BY vendor_name, as_of_date DESC, created_at DESC
        """,
        as_of,
        analysis_window_days,
    )
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        vendor = _canonicalize_vendor(row.get("vendor_name") or "")
        if not vendor:
            continue
        data = _safe_json(row.get("accounts"), default={})
        if isinstance(data, dict):
            lookup[vendor] = data
    return lookup


async def _fetch_latest_displacement_dynamics(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, list[dict[str, Any]]]:
    """Load latest canonical displacement dynamics rows keyed by vendor."""
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (from_vendor, to_vendor)
               from_vendor, to_vendor, dynamics
        FROM b2b_displacement_dynamics
        WHERE as_of_date <= $1
          AND analysis_window_days = $2
        ORDER BY from_vendor, to_vendor, as_of_date DESC, created_at DESC
        """,
        as_of,
        analysis_window_days,
    )
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        from_vendor = _canonicalize_vendor(row.get("from_vendor") or "")
        to_vendor = str(row.get("to_vendor") or "").strip()
        data = _safe_json(row.get("dynamics"), default={})
        if not from_vendor or not to_vendor or not isinstance(data, dict):
            continue
        data.setdefault("from_vendor", row.get("from_vendor") or from_vendor)
        data.setdefault("to_vendor", row.get("to_vendor") or to_vendor)
        lookup.setdefault(from_vendor, []).append(data)
    return lookup


def _normalize_displacement_driver_label(value: Any) -> str:
    """Map free-text displacement reasons into stable seller-safe driver buckets."""
    text = str(value or "").strip()
    lower = text.lower()
    if not lower:
        return ""
    if any(token in lower for token in ("price", "pricing", "cost", "budget", "cheap", "cheaper", "affordable", "economics", "refund", "free", "fee")):
        return "pricing"
    if any(token in lower for token in ("feature", "capability", "automation", "workflow", "ai", "reporting", "dashboard", "customization")):
        return "features"
    if any(token in lower for token in ("integrat", "api", "connector", "webhook", "plugin", "stack")):
        return "integration"
    if any(token in lower for token in ("ux", "ui", "usability", "easy", "easier", "simple", "onboarding", "setup", "interface")):
        return "ux"
    if any(token in lower for token in ("support", "service", "response", "help desk", "ticket")):
        return "support"
    if any(token in lower for token in ("reliab", "uptime", "outage", "stability", "downtime")):
        return "reliability"
    if any(token in lower for token in ("security", "privacy", "server", "governance", "data residency", "encryption")):
        return "security"
    if any(token in lower for token in ("compliance", "gdpr", "hipaa", "soc2", "soc 2", "regulatory", "certification")):
        return "compliance"
    if any(token in lower for token in ("performance", "slow", "latency", "speed")):
        return "performance"
    if any(token in lower for token in ("migrat", "import", "export")):
        return "migration"
    return ""


def _competitive_disp_from_dynamics(
    displacement_lookup: dict[str, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Normalize canonical displacement dynamics into legacy competitor rows."""
    def _normalize_evidence_type(value: Any) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if text in {"explicit_switch", "switch", "switched_to"}:
            return "explicit_switch"
        if text in {"active_evaluation", "evaluation", "considering", "trial", "poc"}:
            return "active_evaluation"
        if text in {"implied_preference", "compared", "preference"}:
            return "implied_preference"
        return None

    def _direction_to_evidence_type(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return "implied_preference"
        if text in {"switched_to", "migrated_to", "replaced_with"}:
            return "explicit_switch"
        if text in {
            "considering", "evaluation", "evaluating", "active_evaluation",
            "trial", "poc", "active_purchase",
        }:
            return "active_evaluation"
        if "switch" in text and "from" not in text:
            return "explicit_switch"
        if "consider" in text or "evaluat" in text or "trial" in text:
            return "active_evaluation"
        return "implied_preference"

    def _intish(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    rows: list[dict[str, Any]] = []
    for vendor, flows in (displacement_lookup or {}).items():
        for flow in flows:
            competitor = _canonicalize_competitor(str(
                flow.get("to_vendor") or flow.get("competitor") or "",
            ).strip())
            if not competitor:
                continue
            flow_summary = flow.get("flow_summary") or {}
            edge_metrics = flow.get("edge_metrics") or {}
            reason_categories: dict[str, int] = {}
            industries: list[str] = []
            company_sizes: list[str] = []
            normalized_breakdown: list[dict[str, Any]] = []
            breakdown_counts = {
                "explicit_switch": 0,
                "active_evaluation": 0,
                "implied_preference": 0,
            }
            for item in flow.get("evidence_breakdown") or []:
                if not isinstance(item, dict):
                    continue
                evidence_type = _normalize_evidence_type(item.get("evidence_type"))
                cats = item.get("reason_categories") or {}
                mention_count = _intish(item.get("mention_count"))
                if mention_count <= 0 and isinstance(cats, dict):
                    mention_count = sum(_intish(v) for v in cats.values())
                if evidence_type and mention_count > 0:
                    normalized_breakdown.append({
                        "evidence_type": evidence_type,
                        "mention_count": mention_count,
                        "reason_categories": {
                            str(k): _intish(v)
                            for k, v in cats.items()
                            if str(k or "").strip() and _intish(v) > 0
                        },
                        "industries": [
                            str(v).strip()
                            for v in (item.get("industries") or [])
                            if str(v or "").strip() and str(v).strip().lower() != "unknown"
                        ],
                        "company_sizes": [
                            str(v).strip()
                            for v in (item.get("company_sizes") or [])
                            if str(v or "").strip()
                        ],
                    })
                    breakdown_counts[evidence_type] += mention_count
                industries.extend(
                    str(v).strip()
                    for v in (item.get("industries") or [])
                    if str(v or "").strip() and str(v).strip().lower() != "unknown"
                )
                company_sizes.extend(
                    str(v).strip()
                    for v in (item.get("company_sizes") or [])
                    if str(v or "").strip()
                )

            reason_direction_counts = {
                "explicit_switch": 0,
                "active_evaluation": 0,
                "implied_preference": 0,
            }
            for item in flow.get("switch_reasons") or []:
                if not isinstance(item, dict):
                    continue
                key = str(
                    item.get("reason_category") or item.get("reason") or "",
                ).strip()
                reason_count = _intish(item.get("mention_count"), default=1)
                if not key:
                    key = str(item.get("reason_detail") or "").strip()
                key = _normalize_displacement_driver_label(key)
                if key:
                    reason_categories[key] = (
                        reason_categories.get(key, 0) + reason_count
                    )
                evidence_type = _direction_to_evidence_type(item.get("direction"))
                reason_direction_counts[evidence_type] += reason_count
            mention_count = max(
                _intish(
                    flow_summary.get("total_flow_mentions")
                    or edge_metrics.get("mention_count"),
                ),
                sum(breakdown_counts.values()),
                sum(reason_direction_counts.values()),
            )
            explicit_switches = max(
                _intish(flow_summary.get("explicit_switch_count")),
                breakdown_counts["explicit_switch"],
                reason_direction_counts["explicit_switch"],
            )
            active_evaluations = max(
                _intish(flow_summary.get("active_evaluation_count")),
                breakdown_counts["active_evaluation"],
                reason_direction_counts["active_evaluation"],
            )
            implied_preferences = max(
                mention_count - explicit_switches - active_evaluations,
                breakdown_counts["implied_preference"],
                reason_direction_counts["implied_preference"],
            )
            if not normalized_breakdown:
                for evidence_type, count in (
                    ("explicit_switch", explicit_switches),
                    ("active_evaluation", active_evaluations),
                    ("implied_preference", implied_preferences),
                ):
                    if count <= 0:
                        continue
                    normalized_breakdown.append({
                        "evidence_type": evidence_type,
                        "mention_count": count,
                        "reason_categories": {},
                        "industries": [],
                        "company_sizes": [],
                    })
            if not mention_count:
                mention_count = sum(item["mention_count"] for item in normalized_breakdown)
            rows.append({
                "vendor": vendor,
                "competitor": competitor,
                "mention_count": mention_count,
                "explicit_switches": explicit_switches,
                "active_evaluations": active_evaluations,
                "implied_preferences": implied_preferences,
                "reason_categories": reason_categories,
                "industries": sorted(set(i for i in industries if i)),
                "company_sizes": sorted(set(s for s in company_sizes if s)),
                "evidence_breakdown": normalized_breakdown,
            })
    return rows


async def _fetch_competitive_displacement_source_of_truth(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
) -> list[dict[str, Any]]:
    """Prefer canonical displacement dynamics, fall back to legacy review query."""
    try:
        lookup = await _fetch_latest_displacement_dynamics(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )
    except Exception:
        logger.debug("Displacement dynamics lookup failed", exc_info=True)
        lookup = {}
    rows = _competitive_disp_from_dynamics(lookup)
    if rows:
        return rows
    return await _fetch_competitive_displacement(pool, analysis_window_days)


def _merge_pain_lookup_with_evidence_vault(
    raw_pain_lookup: dict[str, list[dict[str, Any]]],
    evidence_vault_lookup: dict[str, dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Prefer canonical vault pain rows while preserving raw fallback entries."""
    merged: dict[str, list[dict[str, Any]]] = {}
    vendors = set(raw_pain_lookup) | set((evidence_vault_lookup or {}))
    for vendor in vendors:
        raw_entries = list(raw_pain_lookup.get(vendor, []))
        vault_entries: list[dict[str, Any]] = []
        for item in ((evidence_vault_lookup or {}).get(vendor, {}) or {}).get("weakness_evidence") or []:
            if str(item.get("evidence_type") or "") != "pain_category":
                continue
            key = str(item.get("key") or "").strip().lower()
            if not key:
                continue
            if key in {"other", "general_dissatisfaction"}:
                key = "overall_dissatisfaction"
            metrics = item.get("supporting_metrics") or {}
            vault_entries.append({
                "category": key,
                "count": int(item.get("mention_count_total") or 0),
                "avg_urgency": metrics.get("avg_urgency") or metrics.get("avg_urgency_when_mentioned"),
            })
        if not vault_entries:
            if raw_entries:
                merged[vendor] = raw_entries
            continue
        seen = {str(item.get("category") or "").strip().lower() for item in vault_entries}
        for item in raw_entries:
            category = str(item.get("category") or "").strip().lower()
            if category and category not in seen:
                vault_entries.append(item)
        vault_entries.sort(key=lambda item: int(item.get("count") or 0), reverse=True)
        merged[vendor] = vault_entries
    return merged


def _merge_feature_gap_lookup_with_evidence_vault(
    raw_feature_gap_lookup: dict[str, list[dict[str, Any]]],
    evidence_vault_lookup: dict[str, dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Prefer canonical vault feature-gap rows while preserving raw fallback entries."""
    merged: dict[str, list[dict[str, Any]]] = {}
    vendors = set(raw_feature_gap_lookup) | set((evidence_vault_lookup or {}))
    for vendor in vendors:
        raw_entries = list(raw_feature_gap_lookup.get(vendor, []))
        vault_entries: list[dict[str, Any]] = []
        for item in ((evidence_vault_lookup or {}).get(vendor, {}) or {}).get("weakness_evidence") or []:
            if str(item.get("evidence_type") or "") != "feature_gap":
                continue
            label = str(item.get("label") or item.get("key") or "").strip()
            if not label:
                continue
            vault_entries.append({
                "feature": label,
                "mentions": int(item.get("mention_count_total") or 0),
            })
        if not vault_entries:
            if raw_entries:
                merged[vendor] = raw_entries
            continue
        seen = {str(item.get("feature") or "").strip().lower() for item in vault_entries}
        for item in raw_entries:
            feature = str(item.get("feature") or "").strip().lower()
            if feature and feature not in seen:
                vault_entries.append(item)
        vault_entries.sort(key=lambda item: int(item.get("mentions") or 0), reverse=True)
        merged[vendor] = vault_entries
    return merged


def _merge_company_lookup_with_evidence_vault(
    raw_company_lookup: dict[str, list[dict[str, Any]]],
    evidence_vault_lookup: dict[str, dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Merge raw company context with canonical vault company signals."""
    merged: dict[str, list[dict[str, Any]]] = {}
    vendors = set(raw_company_lookup) | set((evidence_vault_lookup or {}))
    for vendor in vendors:
        bucket: dict[str, dict[str, Any]] = {}
        for item in raw_company_lookup.get(vendor, []):
            if not isinstance(item, dict):
                continue
            company = str(item.get("company") or item.get("company_name") or "").strip()
            key = normalize_company_name(company)
            if not key:
                continue
            bucket[key] = dict(item)
            bucket[key]["company"] = company or key
        for item in ((evidence_vault_lookup or {}).get(vendor, {}) or {}).get("company_signals") or []:
            key = normalize_company_name(item.get("company_name") or item.get("company") or "")
            if not key:
                continue
            current = bucket.get(key, {})
            current_urgency = current.get("urgency")
            signal_urgency = item.get("urgency_score")
            if current_urgency is None:
                urgency = signal_urgency
            elif signal_urgency is None:
                urgency = current_urgency
            else:
                urgency = max(float(current_urgency), float(signal_urgency))
            bucket[key] = {
                **current,
                "company": current.get("company") or str(item.get("company_name") or key),
                "urgency": urgency,
                "title": current.get("title") or item.get("buyer_role"),
                "company_size": current.get("company_size"),
                "industry": current.get("industry"),
                "source": item.get("source") or current.get("source"),
                "buying_stage": item.get("buying_stage") or current.get("buying_stage"),
                "confidence_score": item.get("confidence_score") if item.get("confidence_score") is not None else current.get("confidence_score"),
                "decision_maker": item.get("decision_maker") if item.get("decision_maker") is not None else current.get("decision_maker"),
                "first_seen_at": item.get("first_seen_at") or current.get("first_seen_at"),
                "last_seen_at": item.get("last_seen_at") or current.get("last_seen_at"),
                "review_id": item.get("review_id") or current.get("review_id"),
            }
        ordered = sorted(
            bucket.values(),
            key=lambda item: (
                -(float(item.get("urgency") or 0)),
                -(float(item.get("confidence_score") or 0)),
                str(item.get("company") or ""),
            ),
        )
        if ordered:
            merged[vendor] = ordered
    return merged


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
          AND sentiment_direction != 'unknown'
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
    results = [
        {"vendor": r["vendor_name"], "trigger": r["turning_point"], "mentions": r["cnt"]}
        for r in rows
    ]

    # Supplement with v2 event_mentions for richer temporal data
    event_rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            ev.value->>'event' AS event_text,
            ev.value->>'timeframe' AS timeframe,
            count(*) AS cnt
        FROM b2b_reviews
        CROSS JOIN LATERAL jsonb_array_elements(
            COALESCE(enrichment->'event_mentions', '[]'::jsonb)
        ) AS ev(value)
        WHERE {filters}
          AND jsonb_array_length(COALESCE(enrichment->'event_mentions', '[]'::jsonb)) > 0
        GROUP BY vendor_name, ev.value->>'event', ev.value->>'timeframe'
        HAVING count(*) >= 2
        ORDER BY cnt DESC
        """,
        window_days,
        sources,
    )
    seen_triggers = {(r["vendor"], r["trigger"]) for r in results}
    for r in event_rows:
        trigger = r["event_text"] or ""
        if not trigger:
            continue
        if r["timeframe"]:
            trigger = f"{trigger} ({r['timeframe']})"
        if (r["vendor_name"], trigger) not in seen_triggers:
            seen_triggers.add((r["vendor_name"], trigger))
            results.append({
                "vendor": r["vendor_name"],
                "trigger": trigger,
                "mentions": r["cnt"],
            })

    results.sort(key=lambda x: x.get("mentions", 0), reverse=True)
    return results


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


async def _fetch_company_size_distribution(pool, window_days: int) -> list[dict[str, Any]]:
    """Count reviews per normalized company-size segment per vendor.

    Segment source priority:
      1. Enrichment LLM label (startup/smb/mid_market/enterprise) when not 'unknown'
      2. Apollo-verified employee_count bucket via account resolution
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2, alias="r")
    rows = await pool.fetch(
        f"""
        SELECT r.vendor_name,
            COALESCE(
                NULLIF(NULLIF(r.enrichment->'reviewer_context'->>'company_size_segment', 'unknown'), ''),
                CASE
                    WHEN poc.employee_count >= 1001 THEN 'enterprise'
                    WHEN poc.employee_count >= 201  THEN 'mid_market'
                    WHEN poc.employee_count >= 11   THEN 'smb'
                    WHEN poc.employee_count >= 1    THEN 'startup'
                END
            ) AS segment,
            count(*) AS review_count,
            count(*) FILTER (
                WHERE (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean = true
            ) AS churning_count
        FROM b2b_reviews r
        LEFT JOIN b2b_account_resolution ar
            ON ar.review_id = r.id AND ar.resolution_status = 'resolved'
        LEFT JOIN prospect_org_cache poc
            ON poc.company_name_norm = ar.normalized_company_name
        WHERE {filters}
        GROUP BY r.vendor_name,
            COALESCE(
                NULLIF(NULLIF(r.enrichment->'reviewer_context'->>'company_size_segment', 'unknown'), ''),
                CASE
                    WHEN poc.employee_count >= 1001 THEN 'enterprise'
                    WHEN poc.employee_count >= 201  THEN 'mid_market'
                    WHEN poc.employee_count >= 11   THEN 'smb'
                    WHEN poc.employee_count >= 1    THEN 'startup'
                END
            )
        HAVING COALESCE(
                NULLIF(NULLIF(r.enrichment->'reviewer_context'->>'company_size_segment', 'unknown'), ''),
                CASE
                    WHEN poc.employee_count >= 1001 THEN 'enterprise'
                    WHEN poc.employee_count >= 201  THEN 'mid_market'
                    WHEN poc.employee_count >= 11   THEN 'smb'
                    WHEN poc.employee_count >= 1    THEN 'startup'
                END
            ) IS NOT NULL
        ORDER BY review_count DESC
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "segment": r["segment"],
            "review_count": r["review_count"],
            "churn_rate": round(r["churning_count"] / r["review_count"], 2) if r["review_count"] else 0,
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


async def _fetch_role_churn_summary(pool, window_days: int) -> list[dict[str, Any]]:
    """Per-vendor, per-role_type churn rate and dominant pain category.

    Returns one row per (vendor, role_type) with the fraction of reviews
    showing churn intent and the most common pain_category for that role.
    """
    sources = _intelligence_source_allowlist()
    filters = _eligible_review_filters(window_param=1, source_param=2)
    rows = await pool.fetch(
        f"""
        SELECT vendor_name,
            enrichment->'buyer_authority'->>'role_type' AS role_type,
            count(*) AS total,
            count(*) FILTER (
                WHERE (enrichment->>'churn_intent')::boolean IS TRUE
            ) AS churn_count,
            mode() WITHIN GROUP (
                ORDER BY enrichment->>'pain_category'
            ) FILTER (
                WHERE enrichment->>'pain_category' IS NOT NULL
            ) AS top_pain
        FROM b2b_reviews
        WHERE {filters}
          AND enrichment->'buyer_authority' IS NOT NULL
          AND enrichment->'buyer_authority' != 'null'::jsonb
        GROUP BY vendor_name,
            enrichment->'buyer_authority'->>'role_type'
        HAVING count(*) >= 3
        """,
        window_days,
        sources,
    )
    return [
        {
            "vendor": r["vendor_name"],
            "role_type": r["role_type"],
            "total": r["total"],
            "churn_count": r["churn_count"],
            "churn_rate": round(r["churn_count"] / r["total"], 3) if r["total"] else 0.0,
            "top_pain": r["top_pain"],
        }
        for r in rows
    ]


def _build_role_churn_lookup(
    role_churn_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """vendor -> role_type -> {churn_rate, top_pain}."""
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for row in role_churn_rows:
        vendor = row.get("vendor", "")
        rt = row.get("role_type", "unknown")
        lookup.setdefault(vendor, {})[rt] = {
            "churn_rate": row.get("churn_rate"),
            "top_pain": row.get("top_pain"),
        }
    return lookup


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
              OR NULLIF(enrichment->'timeline'->>'decision_timeline', '') IS NOT NULL
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
            "category": _normalize_generic_pain_label(row.get("pain")) or "overall_dissatisfaction",
            "count": row.get("complaint_count", 0),
            "avg_urgency": round(row.get("avg_urgency", 0), 1),
        })
    for v in lookup:
        lookup[v].sort(key=lambda x: x["count"], reverse=True)
    return lookup


def _aggregate_competitive_disp(competitive_disp: list[dict]) -> list[dict]:
    """Merge rows with same (vendor, competitor), preserving evidence breakdown."""
    agg: dict[tuple[str, str], dict[str, Any]] = {}

    def _intish(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _normalize_evidence_type(value: Any) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if text in {"explicit_switch", "switch", "switched_to"}:
            return "explicit_switch"
        if text in {"active_evaluation", "evaluation", "considering", "trial", "poc"}:
            return "active_evaluation"
        if text in {"implied_preference", "compared", "preference"}:
            return "implied_preference"
        return None

    def _merge_breakdown_item(
        entry: dict[str, Any],
        evidence_type: str | None,
        mention_count: int,
        reason_categories: dict[str, Any] | None = None,
        industries: list[str] | None = None,
        company_sizes: list[str] | None = None,
    ) -> None:
        if not evidence_type or mention_count <= 0:
            return
        bucket = entry["evidence_breakdown"].setdefault(
            evidence_type,
            {
                "evidence_type": evidence_type,
                "mention_count": 0,
                "reason_categories": {},
                "industries": set(),
                "company_sizes": set(),
            },
        )
        bucket["mention_count"] += mention_count
        for rc, rc_cnt in (reason_categories or {}).items():
            if not str(rc or "").strip():
                continue
            count = _intish(rc_cnt)
            if count <= 0:
                continue
            bucket["reason_categories"][rc] = (
                bucket["reason_categories"].get(rc, 0) + count
            )
        for value in industries or []:
            text = str(value or "").strip()
            if text and text.lower() != "unknown":
                bucket["industries"].add(text)
        for value in company_sizes or []:
            text = str(value or "").strip()
            if text:
                bucket["company_sizes"].add(text)

    for row in competitive_disp:
        key = (row.get("vendor", ""), row.get("competitor", ""))
        if key not in agg:
            agg[key] = {
                "mention_count": 0,
                "explicit_switches": 0,
                "active_evaluations": 0,
                "implied_preferences": 0,
                "reason_categories": {},
                "industries": set(),
                "company_sizes": set(),
                "evidence_breakdown": {},
            }
        entry = agg[key]
        cnt = _intish(row.get("mention_count") or 0)
        explicit = _intish(row.get("explicit_switches") or 0)
        active_eval = _intish(row.get("active_evaluations") or 0)
        implied = _intish(row.get("implied_preferences") or 0)
        breakdown_seen = False
        for item in row.get("evidence_breakdown") or []:
            if not isinstance(item, dict):
                continue
            evidence_type = _normalize_evidence_type(item.get("evidence_type"))
            reason_categories = item.get("reason_categories") or {}
            mention_count = _intish(item.get("mention_count") or 0)
            if mention_count <= 0 and isinstance(reason_categories, dict):
                mention_count = sum(_intish(v) for v in reason_categories.values())
            if not evidence_type or mention_count <= 0:
                continue
            breakdown_seen = True
            _merge_breakdown_item(
                entry,
                evidence_type,
                mention_count,
                reason_categories,
                item.get("industries") or [],
                item.get("company_sizes") or [],
            )
        if not breakdown_seen:
            evidence_type = _normalize_evidence_type(row.get("evidence_type"))
            if explicit <= 0 and active_eval <= 0 and implied <= 0 and evidence_type:
                if evidence_type == "explicit_switch":
                    explicit = cnt
                elif evidence_type == "active_evaluation":
                    active_eval = cnt
                else:
                    implied = cnt
            if cnt <= 0:
                cnt = explicit + active_eval + implied
            for evidence_type, count in (
                ("explicit_switch", explicit),
                ("active_evaluation", active_eval),
                ("implied_preference", implied),
            ):
                _merge_breakdown_item(
                    entry,
                    evidence_type,
                    count,
                    row.get("reason_categories") or {},
                    row.get("industries") or [],
                    row.get("company_sizes") or [],
                )
        entry["mention_count"] += cnt or (explicit + active_eval + implied)
        entry["explicit_switches"] += explicit
        entry["active_evaluations"] += active_eval
        entry["implied_preferences"] += implied
        # Merge reason_categories from this row
        for rc, rc_cnt in (row.get("reason_categories") or {}).items():
            count = _intish(rc_cnt)
            if count <= 0:
                continue
            entry["reason_categories"][rc] = entry["reason_categories"].get(rc, 0) + count
        for value in row.get("industries") or []:
            text = str(value or "").strip()
            if text and text.lower() != "unknown":
                entry["industries"].add(text)
        for value in row.get("company_sizes") or []:
            text = str(value or "").strip()
            if text:
                entry["company_sizes"].add(text)

    results = []
    for (v, c), data in sorted(agg.items(), key=lambda x: x[1]["mention_count"], reverse=True):
        breakdown = sorted(
            (
                {
                    "evidence_type": item["evidence_type"],
                    "mention_count": item["mention_count"],
                    "reason_categories": dict(
                        sorted(
                            item["reason_categories"].items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        ),
                    ),
                    "industries": sorted(item["industries"]),
                    "company_sizes": sorted(item["company_sizes"]),
                }
                for item in data["evidence_breakdown"].values()
                if item["mention_count"] > 0
            ),
            key=lambda item: item["mention_count"],
            reverse=True,
        )
        if not breakdown:
            for evidence_type, count in (
                ("explicit_switch", data["explicit_switches"]),
                ("active_evaluation", data["active_evaluations"]),
                ("implied_preference", data["implied_preferences"]),
            ):
                if count <= 0:
                    continue
                breakdown.append({
                    "evidence_type": evidence_type,
                    "mention_count": count,
                    "reason_categories": {},
                    "industries": [],
                    "company_sizes": [],
                })
        results.append({
            "vendor": v,
            "competitor": c,
            "mention_count": data["mention_count"],
            "explicit_switches": data["explicit_switches"],
            "active_evaluations": data["active_evaluations"],
            "implied_preferences": data["implied_preferences"],
            "reason_categories": data["reason_categories"],
            "industries": sorted(data["industries"]),
            "company_sizes": sorted(data["company_sizes"]),
            "evidence_breakdown": breakdown,
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


def _build_inbound_displacement_lookup(competitive_disp: list[dict]) -> dict[str, int]:
    """Sum mentions where vendor is the target (inbound displacement)."""
    inbound: dict[str, int] = {}
    for row in competitive_disp:
        comp = row.get("competitor", "")
        mentions = row.get("mention_count", 0)
        if comp:
            inbound[comp] = inbound.get(comp, 0) + mentions
    return inbound


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


def _build_lock_in_lookup(use_case_dist: list[dict]) -> dict[str, str]:
    """vendor -> dominant lock_in_level (high/medium/low)."""
    vendor_counts: dict[str, dict[str, int]] = {}
    for entry in use_case_dist:
        if entry.get("type") != "lock_in":
            continue
        for row in entry.get("data", []):
            vendor = row.get("vendor_name", "")
            level = str(row.get("lock_in_level") or "unknown").lower().strip()
            if not vendor or level in ("unknown", "none", "null"):
                continue
            vendor_counts.setdefault(vendor, {})[level] = (
                vendor_counts.get(vendor, {}).get(level, 0) + int(row.get("cnt") or 0)
            )
    lookup: dict[str, str] = {}
    for vendor, counts in vendor_counts.items():
        if counts:
            lookup[vendor] = max(counts, key=counts.get)
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
    """vendor -> role and buying-stage summaries for segment intelligence."""
    lookup: dict[str, dict] = {}
    for row in buyer_auth:
        vendor = row.get("vendor", "")
        if vendor not in lookup:
            lookup[vendor] = {
                "role_types": {},
                "buying_stages": {},
                "role_buying_stages": {},
            }
        rt = row.get("role_type", "unknown")
        bs = row.get("buying_stage", "unknown")
        cnt = row.get("count", 0)
        lookup[vendor]["role_types"][rt] = lookup[vendor]["role_types"].get(rt, 0) + cnt
        lookup[vendor]["buying_stages"][bs] = lookup[vendor]["buying_stages"].get(bs, 0) + cnt
        role_stage_counts = lookup[vendor]["role_buying_stages"].setdefault(rt, {})
        role_stage_counts[bs] = role_stage_counts.get(bs, 0) + cnt
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


def _parse_timeline_date(raw: Any) -> date | None:
    """Parse common timeline date formats from review enrichment."""
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text[:32], fmt).date()
        except ValueError:
            continue
    return None


def _build_active_evaluation_deadlines(
    timeline_entries: list[dict[str, Any]],
    *,
    limit: int,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Convert raw timeline entries into actionable deadline/timing signals."""
    today = today or date.today()
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in timeline_entries:
        eval_date = _parse_timeline_date(row.get("evaluation_deadline"))
        contract_date = _parse_timeline_date(row.get("contract_end"))
        timeline = str(row.get("decision_timeline") or "").strip().lower()
        if eval_date and eval_date < today:
            eval_date = None
        if contract_date and contract_date < today:
            contract_date = None
        trigger = "deadline" if eval_date else ("contract_end" if contract_date else "timeline_signal")
        if trigger == "timeline_signal" and timeline in {"", "unknown", "none", "n/a"}:
            continue
        dedupe = (str(row.get("company") or ""), str(eval_date or contract_date or ""), timeline)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        entries.append({
            "company": row.get("company"),
            "evaluation_deadline": eval_date.isoformat() if eval_date else None,
            "contract_end": contract_date.isoformat() if contract_date else None,
            "decision_timeline": row.get("decision_timeline"),
            "urgency": row.get("urgency", 0),
            "title": row.get("title"),
            "company_size": row.get("company_size"),
            "industry": row.get("industry"),
            "trigger_type": trigger,
        })
    entries.sort(key=lambda item: (0 if item.get("evaluation_deadline") else 1, item.get("evaluation_deadline") or item.get("contract_end") or "9999-12-31", -float(item.get("urgency") or 0)))
    return entries[:limit]


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


def _normalize_vault_feature_key(raw: Any) -> str:
    """Normalize a feature-gap label into the vault key format."""
    return str(raw or "").strip().lower().replace(" ", "_")[:50]


def _normalize_vault_aspect_key(raw: Any) -> str:
    """Normalize a positive-aspect label into the vault key format."""
    return str(raw or "").strip().lower()


def _evidence_rollup_review_date(row: dict[str, Any]) -> date | None:
    """Return the best available event date for review-level vault rollups."""
    raw = row.get("reviewed_at") or row.get("enriched_at")
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    parsed = _parse_timeline_date(raw)
    return parsed if parsed else None


def _quote_reviewed_at_text(raw: Any) -> str | None:
    """Convert quote provenance timestamps into a stable ISO date string."""
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    text = str(raw or "").strip()
    return text[:10] if text else None


def _build_evidence_vault_trend(
    recent_count: int,
    prior_count: int,
) -> dict[str, Any]:
    """Classify evidence trend from recent and prior same-length windows."""
    direction = "stable"
    if prior_count <= 0 and recent_count >= _evidence_vault_trend_new_min_recent():
        direction = "new"
    elif prior_count > 0:
        ratio = recent_count / prior_count
        if recent_count > prior_count and ratio >= _evidence_vault_trend_accelerating_ratio():
            direction = "accelerating"
        elif recent_count < prior_count and ratio <= _evidence_vault_trend_declining_ratio():
            direction = "declining"
    return {
        "direction": direction,
        "prior_count": prior_count,
        "recent_count": recent_count,
        "delta_pct": round((recent_count - prior_count) / prior_count, 2) if prior_count > 0 else None,
        "basis": "recent_vs_prior_window",
    }


def _build_evidence_quote_lookup(
    quotes_by_vendor: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group vendor quotes by review ID for weakness/strength matching."""
    lookup: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for raw_vendor, quotes in (quotes_by_vendor or {}).items():
        vendor = _canonicalize_vendor(raw_vendor or "")
        if not vendor:
            continue
        bucket: dict[str, list[dict[str, Any]]] = {}
        for item in quotes or []:
            if not isinstance(item, dict):
                continue
            review_id = str(item.get("review_id") or "").strip()
            if not review_id:
                continue
            bucket.setdefault(review_id, []).append(item)
        for review_id, items in bucket.items():
            items.sort(key=_battle_card_quote_sort_key, reverse=True)
            bucket[review_id] = items
        lookup[vendor] = bucket
    return lookup


def _build_evidence_vault_pass2_rollups(
    review_rows: list[dict[str, Any]],
    quotes_by_vendor: dict[str, list[dict[str, Any]]],
    *,
    recent_window_days: int,
    today: date | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate review-level pass-2 rollups for the evidence vault."""
    today = today or date.today()
    recent_cutoff = today - timedelta(days=max(int(recent_window_days), 1))
    prior_cutoff = recent_cutoff - timedelta(days=max(int(recent_window_days), 1))
    supporting_limit = _evidence_vault_supporting_review_limit()
    segment_limit = _evidence_vault_segment_limit()
    role_limit = _evidence_vault_role_limit()

    quote_lookup = _build_evidence_quote_lookup(quotes_by_vendor)
    vendor_rollups: dict[str, dict[str, Any]] = {}

    def _vendor_bucket(vendor: str) -> dict[str, Any]:
        return vendor_rollups.setdefault(vendor, {
            "reviews_in_recent_window": 0,
            "_recent_review_ids": set(),
            "weaknesses": {},
            "strengths": {},
        })

    def _entry(bucket: dict[str, Any], evidence_type: str, key: str) -> dict[str, Any]:
        return bucket.setdefault((evidence_type, key), {
            "mention_count_recent": 0,
            "_prior_count": 0,
            "_urgency_sum": 0.0,
            "_urgency_count": 0,
            "_rating_sum": 0.0,
            "_rating_count": 0,
            "_segment_counts": {},
            "_role_counts": {},
            "_supporting_reviews": [],
            "_supporting_seen": set(),
            "best_quote": None,
            "quote_source": None,
            "_best_quote_key": None,
        })

    def _update_entry(
        item: dict[str, Any],
        *,
        is_recent: bool,
        is_prior: bool,
        review_id: str | None,
        urgency: float,
        rating_5: float | None,
        segment: str | None,
        role: str | None,
        quote_item: dict[str, Any] | None,
    ) -> None:
        if is_recent:
            item["mention_count_recent"] += 1
        elif is_prior:
            item["_prior_count"] += 1
        item["_urgency_sum"] += urgency
        item["_urgency_count"] += 1
        if rating_5 is not None:
            item["_rating_sum"] += rating_5
            item["_rating_count"] += 1
        if segment:
            seg = item["_segment_counts"].setdefault(segment, {"count": 0, "recent_count": 0})
            seg["count"] += 1
            if is_recent:
                seg["recent_count"] += 1
        if role:
            role_entry = item["_role_counts"].setdefault(role, {"count": 0, "recent_count": 0})
            role_entry["count"] += 1
            if is_recent:
                role_entry["recent_count"] += 1
        if review_id and review_id not in item["_supporting_seen"]:
            item["_supporting_seen"].add(review_id)
            item["_supporting_reviews"].append((urgency, review_id))
        if quote_item:
            quote_key = _battle_card_quote_sort_key(quote_item)
            if item["_best_quote_key"] is None or quote_key > item["_best_quote_key"]:
                item["_best_quote_key"] = quote_key
                item["best_quote"] = quote_item.get("quote")
                item["quote_source"] = {
                    "review_id": str(quote_item.get("review_id") or "") or None,
                    "source": quote_item.get("source"),
                    "company": quote_item.get("company"),
                    "reviewer_title": quote_item.get("title"),
                    "company_size": quote_item.get("company_size"),
                    "industry": quote_item.get("industry"),
                    "reviewed_at": _quote_reviewed_at_text(quote_item.get("reviewed_at")),
                    "rating": float(quote_item["rating"]) if quote_item.get("rating") is not None else None,
                }

    for row in review_rows or []:
        vendor = _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
        if not vendor:
            continue
        vendor_bucket = _vendor_bucket(vendor)
        review_id = str(row.get("review_id") or "").strip() or None
        review_date = _evidence_rollup_review_date(row)
        is_recent = bool(review_date and review_date >= recent_cutoff)
        is_prior = bool(review_date and prior_cutoff <= review_date < recent_cutoff)
        if review_id and is_recent and review_id not in vendor_bucket["_recent_review_ids"]:
            vendor_bucket["_recent_review_ids"].add(review_id)
            vendor_bucket["reviews_in_recent_window"] += 1

        urgency = float(row.get("urgency") or 0.0)
        rating = row.get("rating")
        rating_max = row.get("rating_max")
        rating_5 = None
        if rating is not None and rating_max:
            try:
                rating_5 = round((float(rating) / float(rating_max)) * 5.0, 2)
            except (TypeError, ValueError, ZeroDivisionError):
                rating_5 = None

        segment = str(row.get("company_size_raw") or "").strip() or None
        role = str(row.get("reviewer_title") or row.get("role_level") or "").strip() or None
        quote_item = None
        if review_id:
            quote_item = (quote_lookup.get(vendor, {}).get(review_id) or [None])[0]

        pain_key = str(row.get("pain_category") or "").strip().lower()
        if pain_key:
            item = _entry(vendor_bucket["weaknesses"], "pain_category", pain_key)
            _update_entry(
                item,
                is_recent=is_recent,
                is_prior=is_prior,
                review_id=review_id,
                urgency=urgency,
                rating_5=rating_5,
                segment=segment,
                role=role,
                quote_item=quote_item,
            )

        feature_gaps = {
            str(gap).strip() for gap in (_safe_json(row.get("feature_gaps"), default=[]) or [])
            if str(gap).strip()
        }
        for raw_gap in feature_gaps:
            key = _normalize_vault_feature_key(raw_gap)
            if not key:
                continue
            item = _entry(vendor_bucket["weaknesses"], "feature_gap", key)
            _update_entry(
                item,
                is_recent=is_recent,
                is_prior=is_prior,
                review_id=review_id,
                urgency=urgency,
                rating_5=rating_5,
                segment=segment,
                role=role,
                quote_item=quote_item,
            )

        positive_aspects = {
            str(aspect).strip() for aspect in (_safe_json(row.get("positive_aspects"), default=[]) or [])
            if str(aspect).strip()
        }
        for raw_aspect in positive_aspects:
            key = _normalize_vault_aspect_key(raw_aspect)
            if not key:
                continue
            item = _entry(vendor_bucket["strengths"], "retention_signal", key)
            _update_entry(
                item,
                is_recent=is_recent,
                is_prior=is_prior,
                review_id=review_id,
                urgency=urgency,
                rating_5=rating_5,
                segment=segment,
                role=role,
                quote_item=quote_item,
            )

    for vendor, bucket in vendor_rollups.items():
        bucket.pop("_recent_review_ids", None)
        for section_name in ("weaknesses", "strengths"):
            finalized: dict[tuple[str, str], dict[str, Any]] = {}
            for section_key, item in bucket[section_name].items():
                segments = sorted(
                    item["_segment_counts"].items(),
                    key=lambda entry: (-entry[1]["count"], -entry[1]["recent_count"], entry[0]),
                )
                roles = sorted(
                    item["_role_counts"].items(),
                    key=lambda entry: (-entry[1]["count"], -entry[1]["recent_count"], entry[0]),
                )
                supporting = sorted(
                    item["_supporting_reviews"],
                    key=lambda entry: (-entry[0], entry[1]),
                )
                metrics: dict[str, Any] = {}
                if item["_urgency_count"]:
                    metrics["avg_urgency_when_mentioned"] = round(
                        item["_urgency_sum"] / item["_urgency_count"], 1,
                    )
                if item["_rating_count"]:
                    metrics["avg_rating_when_mentioned"] = round(
                        item["_rating_sum"] / item["_rating_count"], 2,
                    )
                finalized[section_key] = {
                    "best_quote": item["best_quote"],
                    "quote_source": item["quote_source"],
                    "mention_count_recent": item["mention_count_recent"],
                    "trend": _build_evidence_vault_trend(
                        item["mention_count_recent"],
                        item["_prior_count"],
                    ),
                    "affected_segments": [
                        {
                            "segment": segment,
                            "count": counts["count"],
                            "recent_count": counts["recent_count"],
                        }
                        for segment, counts in segments[:segment_limit]
                    ] or None,
                    "affected_roles": [
                        {
                            "role": role_name,
                            "count": counts["count"],
                            "recent_count": counts["recent_count"],
                        }
                        for role_name, counts in roles[:role_limit]
                    ] or None,
                    "supporting_metrics": metrics,
                    "supporting_review_ids": [
                        review_id for _, review_id in supporting[:supporting_limit]
                    ],
                }
            bucket[section_name] = finalized
    return vendor_rollups


def _compute_company_signal_confidence(signal: dict[str, Any]) -> float:
    """Compute the canonical confidence score for a company-signal row."""
    source = str(signal.get("source") or "").strip()
    source_dist = {source: 1} if source else {}
    score = _compute_evidence_confidence(1, source_dist)
    completeness = sum(
        1
        for field in (
            signal.get("decision_maker"),
            signal.get("buying_stage"),
            signal.get("seat_count"),
        )
        if field is not None
    )
    return round(min(score + completeness * 0.05, 1.0), 2)


def _merge_canonical_company_signals(
    current_high_intent: list[dict[str, Any]],
    persisted_lookup: dict[str, list[dict[str, Any]]] | None,
    *,
    as_of: datetime | None = None,
    blocked_names_by_vendor: dict[str, set[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Merge current high-intent rows with canonical persisted company signals."""
    as_of = as_of or datetime.now(timezone.utc)
    merged: dict[str, dict[str, dict[str, Any]]] = {}

    def _signal_rank(signal: dict[str, Any]) -> tuple[float, float, int, int, int]:
        urgency = signal.get("urgency_score")
        if urgency is None:
            urgency = signal.get("urgency")
        try:
            urgency_value = float(urgency) if urgency is not None else 0.0
        except (TypeError, ValueError):
            urgency_value = 0.0
        confidence = signal.get("confidence_score")
        if confidence is None:
            confidence_value = _compute_company_signal_confidence(signal)
        else:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0
        return (
            urgency_value,
            confidence_value,
            1 if signal.get("decision_maker") else 0,
            1 if signal.get("review_id") else 0,
            1 if signal.get("buying_stage") else 0,
        )

    def _merge_record_list(primary: Any, secondary: Any, *, quote_mode: bool = False) -> list[Any]:
        merged_items: list[Any] = []
        seen_keys: set[str] = set()
        for source in (primary, secondary):
            if not isinstance(source, list):
                continue
            for item in source:
                if quote_mode:
                    text = _quote_text(item) if isinstance(item, dict) else str(item or "").strip()
                    key = text.lower()
                else:
                    name = item.get("name") if isinstance(item, dict) else item
                    key = normalize_company_name(name) or str(name or "").strip().lower()
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_items.append(item)
        return merged_items

    for vendor, items in (persisted_lookup or {}).items():
        vendor_bucket = merged.setdefault(vendor, {})
        base_blocked_names = (blocked_names_by_vendor or {}).get(vendor) or set()
        for item in items or []:
            company_name = normalize_company_name(item.get("company_name") or "")
            if not company_name:
                continue
            blocked_names = set(base_blocked_names)
            blocked_names.update(
                normalize_company_name(name)
                for name in _extract_alternative_names(item.get("alternatives") or [])
                if normalize_company_name(name)
            )
            if not _company_signal_name_is_eligible(
                item.get("company_name"),
                current_vendor=vendor,
                blocked_names=blocked_names,
            ):
                continue
            current = dict(item)
            current["company_name"] = company_name
            current["_merge_rank"] = _signal_rank(current)
            vendor_bucket[company_name] = current

    for hi in current_high_intent or []:
        vendor = _canonicalize_vendor(hi.get("vendor") or hi.get("vendor_name") or "")
        company_name = normalize_company_name(hi.get("company") or hi.get("company_name") or "")
        if not vendor or not company_name:
            continue
        blocked_names = set((blocked_names_by_vendor or {}).get(vendor) or set())
        blocked_names.update(
            normalize_company_name(name)
            for name in _extract_alternative_names(hi.get("alternatives") or [])
            if normalize_company_name(name)
        )
        if not _company_signal_name_is_eligible(
            hi.get("company") or hi.get("company_name"),
            current_vendor=vendor,
            blocked_names=blocked_names,
        ):
            continue
        vendor_bucket = merged.setdefault(vendor, {})
        existing = vendor_bucket.get(company_name, {})
        current_confidence = _compute_company_signal_confidence(hi)
        current_rank = _signal_rank({
            "urgency_score": hi.get("urgency_score"),
            "urgency": hi.get("urgency"),
            "confidence_score": current_confidence,
            "decision_maker": hi.get("decision_maker"),
            "review_id": hi.get("review_id"),
            "buying_stage": hi.get("buying_stage"),
        })
        existing_rank = existing.get("_merge_rank") or (0.0, 0.0, 0, 0, 0)
        field_source = hi if current_rank >= existing_rank else existing
        fallback_source = existing if field_source is hi else hi
        now_text = as_of.isoformat()
        urgency_score = hi.get("urgency")
        if urgency_score is None:
            urgency_score = hi.get("urgency_score")
        existing_urgency = existing.get("urgency_score")
        if existing_urgency is None:
            merged_urgency = urgency_score
        elif urgency_score is None:
            merged_urgency = existing_urgency
        else:
            merged_urgency = max(float(existing_urgency), float(urgency_score))
        vendor_bucket[company_name] = {
            "company_name": company_name,
            "vendor_name": vendor,
            "urgency_score": float(merged_urgency) if merged_urgency is not None else None,
            "pain_category": (
                field_source.get("pain")
                or field_source.get("pain_category")
                or fallback_source.get("pain")
                or fallback_source.get("pain_category")
            ),
            "buyer_role": (
                field_source.get("role_level")
                or field_source.get("buyer_role")
                or fallback_source.get("role_level")
                or fallback_source.get("buyer_role")
            ),
            "decision_maker": (
                field_source.get("decision_maker")
                if field_source.get("decision_maker") is not None
                else fallback_source.get("decision_maker")
            ),
            "seat_count": (
                field_source.get("seat_count")
                if field_source.get("seat_count") is not None
                else fallback_source.get("seat_count")
            ),
            "contract_end": (
                str(field_source.get("contract_end") or "")
                or str(fallback_source.get("contract_end") or "")
                or None
            ),
            "buying_stage": field_source.get("buying_stage") or fallback_source.get("buying_stage"),
            "review_id": (
                str(field_source.get("review_id") or "")
                or str(fallback_source.get("review_id") or "")
                or None
            ),
            "title": field_source.get("title") or fallback_source.get("title"),
            "company_size": field_source.get("company_size") or fallback_source.get("company_size"),
            "industry": field_source.get("industry") or fallback_source.get("industry"),
            "source": field_source.get("source") or fallback_source.get("source"),
            "confidence_score": max(
                float(existing.get("confidence_score") or 0),
                current_confidence,
            ),
            "first_seen_at": existing.get("first_seen_at") or now_text,
            "last_seen_at": now_text,
            "alternatives": _merge_record_list(
                field_source.get("alternatives"),
                fallback_source.get("alternatives"),
            ),
            "quotes": _merge_record_list(
                field_source.get("quotes"),
                fallback_source.get("quotes"),
                quote_mode=True,
            ),
            "_merge_rank": max(existing_rank, current_rank),
        }

    result: dict[str, list[dict[str, Any]]] = {}
    for vendor, items in merged.items():
        ordered = sorted(
            items.values(),
            key=lambda item: (
                -(float(item.get("urgency_score") or 0)),
                -(float(item.get("confidence_score") or 0)),
                str(item.get("company_name") or ""),
            ),
        )
        cleaned: list[dict[str, Any]] = []
        for item in ordered:
            current = dict(item)
            current.pop("_merge_rank", None)
            cleaned.append(current)
        result[vendor] = cleaned
    return result


# ------------------------------------------------------------------
# Layer 4: deterministic builders (depend on all above)
# ------------------------------------------------------------------


_EVIDENCE_VAULT_SCHEMA_VERSION = "v1"

_WEAKNESS_CORRELATION_MIN_OVERLAP = 0.3
_WEAKNESS_CORRELATION_MIN_SHARED = 3


def _correlate_weakness_clusters(
    weakness_evidence: list[dict[str, Any]],
    *,
    min_overlap_ratio: float = _WEAKNESS_CORRELATION_MIN_OVERLAP,
    min_shared_reviews: int = _WEAKNESS_CORRELATION_MIN_SHARED,
) -> None:
    """Detect co-mentioned weaknesses and annotate correlation metadata.

    Two weaknesses are correlated when they share a meaningful fraction of
    their supporting reviews -- indicating a common root cause.  For example,
    "poor support" and "slow response times" often co-occur in the same
    reviews, so they should be treated as one amplified signal rather than
    two independent items.

    Mutates *weakness_evidence* in place: adds ``correlated_with`` (list of
    related keys) and ``correlation_bonus`` (float added to confidence).
    """
    if len(weakness_evidence) < 2:
        return

    # Build review-ID sets per weakness (index-keyed for speed)
    id_sets: list[set[str]] = []
    for w in weakness_evidence:
        ids = w.get("supporting_review_ids") or []
        id_sets.append({str(rid) for rid in ids if rid})

    n = len(weakness_evidence)

    # Pairwise overlap check
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        if not id_sets[i]:
            continue
        for j in range(i + 1, n):
            if not id_sets[j]:
                continue
            shared = id_sets[i] & id_sets[j]
            if len(shared) < min_shared_reviews:
                continue
            smaller = min(len(id_sets[i]), len(id_sets[j])) or 1
            ratio = len(shared) / smaller
            if ratio >= min_overlap_ratio:
                edges.append((i, j, ratio))

    if not edges:
        return

    # Union-find to group into clusters
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, _ in edges:
        union(i, j)

    # Build cluster membership
    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    # Annotate each weakness with its cluster peers
    for members in clusters.values():
        if len(members) < 2:
            continue
        member_keys = [weakness_evidence[m]["key"] for m in members]
        combined_ids: set[str] = set()
        for m in members:
            combined_ids |= id_sets[m]
        for m in members:
            own_key = weakness_evidence[m]["key"]
            peers = [k for k in member_keys if k != own_key]
            weakness_evidence[m]["correlated_with"] = peers
            weakness_evidence[m]["cluster_review_count"] = len(combined_ids)
            weakness_evidence[m]["correlation_bonus"] = round(
                min(0.1, 0.05 * (len(members) - 1)), 2,
            )


def build_evidence_vault(
    vendor_name: str,
    vs: dict[str, Any],
    *,
    pain_entries: list[dict],
    feature_gap_entries: list[dict],
    quotes: list[dict],
    positive_entries: list[dict],
    product_profile: dict | None = None,
    keyword_spikes: dict | None = None,
    company_signals: list[dict],
    provenance: dict | None = None,
    data_context: dict | None = None,
    pass2_rollups: dict[str, Any] | None = None,
    dm_rate: float | None = None,
    price_rate: float | None = None,
    product_category: str | None = None,
    blocked_names: set[str] | None = None,
    analysis_window_days: int = 90,
    recent_window_days: int = 30,
) -> dict[str, Any]:
    """Build a canonical evidence vault object for a single vendor.

    Pass 2 populates review-level recency, trend, per-weakness quote matching,
    affected segments, affected roles, supporting review IDs, and recent-review
    counts when direct review evidence is available.
    """
    from datetime import date

    as_of = date.today().isoformat()
    pass2 = pass2_rollups or {}
    weakness_rollups = pass2.get("weaknesses") if isinstance(pass2.get("weaknesses"), dict) else {}
    strength_rollups = pass2.get("strengths") if isinstance(pass2.get("strengths"), dict) else {}

    # --- Weakness evidence ---
    weakness_evidence: list[dict] = []
    _seen_keys: set[str] = set()

    # 1. Pain categories
    for p in pain_entries:
        key = str(p.get("category") or p.get("pain") or "").lower().strip()
        if not key or key in _seen_keys:
            continue
        _seen_keys.add(key)
        label = key.replace("_", " ").title()
        rollup = weakness_rollups.get(("pain_category", key), {})
        metrics = {"avg_urgency": p.get("avg_urgency")}
        metrics.update(rollup.get("supporting_metrics") or {})
        weakness_evidence.append({
            "key": key,
            "label": label,
            "evidence_type": "pain_category",
            "best_quote": rollup.get("best_quote"),
            "quote_source": rollup.get("quote_source"),
            "mention_count_total": p.get("complaint_count") or p.get("count") or 0,
            "mention_count_recent": rollup.get("mention_count_recent"),
            "trend": rollup.get("trend"),
            "affected_segments": rollup.get("affected_segments"),
            "affected_roles": rollup.get("affected_roles"),
            "supporting_metrics": metrics,
            "supporting_review_ids": rollup.get("supporting_review_ids") or [],
            "confidence_score": None,
        })

    # 2. Feature gaps
    for fg in feature_gap_entries:
        raw = str(fg.get("feature_gap") or fg.get("feature") or "").strip()
        if not raw:
            continue
        key = _normalize_vault_feature_key(raw)
        if key in _seen_keys:
            continue
        _seen_keys.add(key)
        rollup = weakness_rollups.get(("feature_gap", key), {})
        weakness_evidence.append({
            "key": key,
            "label": raw,
            "evidence_type": "feature_gap",
            "best_quote": rollup.get("best_quote"),
            "quote_source": rollup.get("quote_source"),
            "mention_count_total": fg.get("mentions") or 0,
            "mention_count_recent": rollup.get("mention_count_recent"),
            "trend": rollup.get("trend"),
            "affected_segments": rollup.get("affected_segments"),
            "affected_roles": rollup.get("affected_roles"),
            "supporting_metrics": rollup.get("supporting_metrics") or {},
            "supporting_review_ids": rollup.get("supporting_review_ids") or [],
            "confidence_score": None,
        })

    # 3. Product profile weaknesses
    if product_profile:
        pp_weaknesses = _safe_json(product_profile.get("weaknesses"), default=[])
        for w in pp_weaknesses:
            if isinstance(w, str):
                area = w.lower().strip()
                score = None
                evidence_count = 0
            elif isinstance(w, dict):
                area = str(w.get("area") or "").lower().strip()
                score = w.get("score")
                evidence_count = w.get("evidence_count") or 0
            else:
                continue
            if not area or area in _seen_keys:
                continue
            _seen_keys.add(area)
            rollup = weakness_rollups.get(("satisfaction_area", area), {})
            metrics = {"satisfaction_score": score}
            metrics.update(rollup.get("supporting_metrics") or {})
            weakness_evidence.append({
                "key": area,
                "label": area.replace("_", " ").title(),
                "evidence_type": "satisfaction_area",
                "best_quote": rollup.get("best_quote"),
                "quote_source": rollup.get("quote_source"),
                "mention_count_total": evidence_count,
                "mention_count_recent": rollup.get("mention_count_recent"),
                "trend": rollup.get("trend"),
                "affected_segments": rollup.get("affected_segments"),
                "affected_roles": rollup.get("affected_roles"),
                "supporting_metrics": metrics,
                "supporting_review_ids": rollup.get("supporting_review_ids") or [],
                "confidence_score": None,
            })

    # Sort weaknesses by composite signal: mention count + urgency weight.
    # This prevents a high-urgency weakness with fewer mentions from being
    # buried below a low-urgency weakness with more mentions.
    def _weakness_sort_key(w: dict) -> float:
        mc = float(w.get("mention_count_total") or 0)
        sm = w.get("supporting_metrics") if isinstance(w.get("supporting_metrics"), dict) else {}
        urg = float(sm.get("avg_urgency_when_mentioned") or sm.get("avg_urgency") or 0)
        # Normalize urgency (0-10) to a 0-1 scale and weight it at 30% of
        # the sort signal so urgency influences order without dominating.
        return mc + (min(urg, 10.0) / 10.0) * max(mc, 1) * 0.3

    weakness_evidence.sort(key=_weakness_sort_key, reverse=True)

    # Detect co-mentioned weaknesses (shared root cause clustering)
    _correlate_weakness_clusters(weakness_evidence)

    # Derive pass-1 confidence: mention share + type + volume + correlation
    # + recency + urgency bonuses
    total_mentions = sum(w["mention_count_total"] or 0 for w in weakness_evidence) or 1
    for w in weakness_evidence:
        mc = w["mention_count_total"] or 0
        share = mc / total_mentions
        type_bonus = 0.1 if w["evidence_type"] == "pain_category" else 0.0
        volume_bonus = 0.1 if mc >= 10 else 0.0
        corr_bonus = float(w.get("correlation_bonus") or 0)
        trend = w.get("trend") if isinstance(w.get("trend"), dict) else {}
        direction = str(trend.get("direction") or "").strip()
        if direction == "accelerating":
            recency_bonus = 0.1
        elif direction == "new":
            recency_bonus = 0.1
        elif direction == "declining":
            recency_bonus = -0.05
        else:
            recency_bonus = 0.0
        sm = w.get("supporting_metrics") if isinstance(w.get("supporting_metrics"), dict) else {}
        urg = float(sm.get("avg_urgency_when_mentioned") or sm.get("avg_urgency") or 0)
        urgency_bonus = 0.1 if urg >= 7.0 else (0.05 if urg >= 4.5 else 0.0)
        w["confidence_score"] = round(min(0.95, max(0.0, share + type_bonus + volume_bonus + corr_bonus + recency_bonus + urgency_bonus)), 2)

    # Fallback: assign vendor quotes to weaknesses that lack a best_quote
    # from pass-2 review evidence.  Uses keyword matching to pair each quote
    # to the weakness it is most relevant to, instead of blindly assigning
    # the first quote to the top weakness.
    if weakness_evidence and quotes:
        unquoted = [w for w in weakness_evidence if not w.get("best_quote")]
        if unquoted:
            used_quote_ids: set[str] = set()
            for w in unquoted:
                wkey = str(w.get("key") or w.get("label") or "").lower()
                wlabel = str(w.get("label") or w.get("key") or "").lower()
                w_tokens = set(wkey.replace("_", " ").split()) | set(wlabel.replace("_", " ").split())
                best_match = None
                best_overlap = 0
                for q in quotes:
                    qid = str(q.get("review_id") or id(q))
                    if qid in used_quote_ids:
                        continue
                    qtext = str(q.get("quote") or "").lower()
                    overlap = sum(1 for t in w_tokens if len(t) >= 3 and t in qtext)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = q
                        best_match_id = qid
                if best_match and best_overlap > 0:
                    used_quote_ids.add(best_match_id)
                    w["best_quote"] = best_match.get("quote")
                    w["quote_source"] = {
                        "review_id": str(best_match.get("review_id") or ""),
                        "source": best_match.get("source"),
                        "company": best_match.get("company"),
                        "reviewer_title": best_match.get("title"),
                        "company_size": best_match.get("company_size"),
                        "industry": best_match.get("industry"),
                        "reviewed_at": _quote_reviewed_at_text(best_match.get("reviewed_at")),
                        "rating": float(best_match["rating"]) if best_match.get("rating") is not None else None,
                    }
            # Last resort: if no keyword match found for the top weakness,
            # assign the first unused quote so the vault is never quote-less
            if not weakness_evidence[0].get("best_quote") and quotes:
                for q in quotes:
                    qid = str(q.get("review_id") or id(q))
                    if qid not in used_quote_ids:
                        weakness_evidence[0]["best_quote"] = q.get("quote")
                        weakness_evidence[0]["quote_source"] = {
                            "review_id": str(q.get("review_id") or ""),
                            "source": q.get("source"),
                            "company": q.get("company"),
                            "reviewer_title": q.get("title"),
                            "company_size": q.get("company_size"),
                            "industry": q.get("industry"),
                            "reviewed_at": _quote_reviewed_at_text(q.get("reviewed_at")),
                            "rating": float(q["rating"]) if q.get("rating") is not None else None,
                        }
                        break

    # --- Strength evidence ---
    strength_evidence: list[dict] = []
    _seen_strengths: set[str] = set()

    # 1. Positive aspects
    for pa in positive_entries:
        aspect = str(pa.get("aspect") or "").lower().strip()
        if not aspect or aspect in _seen_strengths:
            continue
        _seen_strengths.add(aspect)
        rollup = strength_rollups.get(("retention_signal", aspect), {})
        strength_evidence.append({
            "key": aspect,
            "label": aspect.replace("_", " ").title(),
            "evidence_type": "retention_signal",
            "best_quote": rollup.get("best_quote"),
            "quote_source": rollup.get("quote_source"),
            "mention_count_total": pa.get("mentions") or 0,
            "mention_count_recent": rollup.get("mention_count_recent"),
            "trend": rollup.get("trend"),
            "affected_segments": rollup.get("affected_segments"),
            "affected_roles": rollup.get("affected_roles"),
            "supporting_metrics": rollup.get("supporting_metrics") or {},
            "supporting_review_ids": rollup.get("supporting_review_ids") or [],
            "confidence_score": None,
        })

    # 2. Product profile strengths
    if product_profile:
        pp_strengths = _safe_json(product_profile.get("strengths"), default=[])
        for s in pp_strengths:
            if isinstance(s, str):
                area = s.lower().strip()
                score = None
                evidence_count = 0
            elif isinstance(s, dict):
                area = str(s.get("area") or "").lower().strip()
                score = s.get("score")
                evidence_count = s.get("evidence_count") or 0
            else:
                continue
            if not area or area in _seen_strengths:
                continue
            _seen_strengths.add(area)
            rollup = strength_rollups.get(("satisfaction_area", area), {})
            metrics = {"satisfaction_score": score}
            metrics.update(rollup.get("supporting_metrics") or {})
            strength_evidence.append({
                "key": area,
                "label": area.replace("_", " ").title(),
                "evidence_type": "satisfaction_area",
                "best_quote": rollup.get("best_quote"),
                "quote_source": rollup.get("quote_source"),
                "mention_count_total": evidence_count,
                "mention_count_recent": rollup.get("mention_count_recent"),
                "trend": rollup.get("trend"),
                "affected_segments": rollup.get("affected_segments"),
                "affected_roles": rollup.get("affected_roles"),
                "supporting_metrics": metrics,
                "supporting_review_ids": rollup.get("supporting_review_ids") or [],
                "confidence_score": None,
            })

    strength_evidence.sort(key=lambda s: s["mention_count_total"] or 0, reverse=True)

    # Derive pass-1 confidence for strengths
    total_str_mentions = sum(s["mention_count_total"] or 0 for s in strength_evidence) or 1
    for s in strength_evidence:
        mc = s["mention_count_total"] or 0
        share = mc / total_str_mentions
        s["confidence_score"] = round(min(0.95, share + (0.1 if mc >= 10 else 0.0)), 2)

    # --- Metric snapshot ---
    def _sf(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    kw = keyword_spikes or {}
    metric_snapshot = {
        "snapshot_date": as_of,
        "total_reviews": vs.get("total_reviews") or 0,
        "reviews_in_analysis_window": vs.get("total_reviews") or 0,
        "reviews_in_recent_window": int(pass2.get("reviews_in_recent_window") or 0),
        "churn_density": _sf(vs.get("churn_density")),
        "avg_urgency": _sf(vs.get("avg_urgency")),
        "recommend_yes": vs.get("recommend_yes") or 0,
        "recommend_no": vs.get("recommend_no") or 0,
        "recommend_ratio": _sf(vs.get("recommend_ratio")) or (
            round((vs.get("recommend_yes") or 0) / max((vs.get("recommend_yes") or 0) + (vs.get("recommend_no") or 0), 1), 2)
        ),
        "price_complaint_rate": _sf(price_rate),
        "dm_churn_rate": _sf(dm_rate),
        "positive_review_pct": _sf(vs.get("positive_review_pct")),
        "displacement_mention_count": vs.get("displacement_mention_count") or 0,
        "keyword_spike_count": kw.get("spike_count") or 0,
        # v2 indicator-based signal counts
        "indicator_counts": {
            "cancel": vs.get("indicator_cancel_count") or 0,
            "migration": vs.get("indicator_migration_count") or 0,
            "evaluation": vs.get("indicator_evaluation_count") or 0,
            "completed_switch": vs.get("indicator_switch_count") or 0,
            "named_alternative": vs.get("indicator_named_alt_count") or 0,
            "decision_maker_language": vs.get("indicator_dm_language_count") or 0,
        },
        "has_pricing_phrases_count": vs.get("has_pricing_phrases_count") or 0,
        "has_recommendation_language_count": vs.get("has_recommendation_language_count") or 0,
    }

    # --- Company signals (pre-filtered for this vendor) ---
    cs_out: list[dict] = []
    for cs in company_signals:
        company_name = cs.get("company") or cs.get("company_name") or ""
        if _company_signal_exclusion_reason(
            company_name,
            current_vendor=vendor_name,
            blocked_names=blocked_names,
            source=cs.get("source"),
            confidence_score=cs.get("confidence_score"),
        ):
            continue
        cs_out.append({
            "company_name": company_name,
            "signal_type": "churning",
            "urgency_score": _sf(cs.get("urgency") or cs.get("urgency_score")),
            "pain_category": cs.get("pain") or cs.get("pain_category"),
            "buyer_role": cs.get("role_level") or cs.get("buyer_role"),
            "decision_maker": cs.get("decision_maker"),
            "seat_count": cs.get("seat_count"),
            "contract_end": str(cs.get("contract_end") or "") or None,
            "buying_stage": cs.get("buying_stage"),
            "review_id": str(cs.get("review_id") or ""),
            "source": cs.get("source"),
            "confidence_score": _sf(cs.get("confidence_score")),
            "first_seen_at": str(cs.get("first_seen_at") or "") or None,
            "last_seen_at": str(cs.get("last_seen_at") or "") or None,
        })

    # --- Provenance ---
    prov = provenance or {}
    dc = data_context or {}
    enrichment_period = dc.get("enrichment_period") or {}
    provenance_out = {
        "sources": list((prov.get("source_distribution") or {}).keys()),
        "source_distribution": prov.get("source_distribution") or {},
        "sample_review_ids": [str(rid) for rid in (prov.get("sample_review_ids") or [])[:5]],
        "enrichment_window_start": str(prov.get("review_window_start") or enrichment_period.get("earliest") or ""),
        "enrichment_window_end": str(prov.get("review_window_end") or enrichment_period.get("latest") or ""),
    }

    return {
        "vendor_name": vendor_name,
        "schema_version": _EVIDENCE_VAULT_SCHEMA_VERSION,
        "as_of_date": as_of,
        "analysis_window_days": analysis_window_days,
        "recent_window_days": recent_window_days,
        "product_category": product_category or "",
        "weakness_evidence": weakness_evidence,
        "strength_evidence": strength_evidence,
        "company_signals": cs_out,
        "metric_snapshot": metric_snapshot,
        "provenance": provenance_out,
    }


_SEGMENT_INTELLIGENCE_SCHEMA_VERSION = "v1"
_SEGMENT_ROLE_SCORES: dict[str, float] = {
    "decision_maker": 20.0,
    "economic_buyer": 15.0,
    "champion": 15.0,
    "evaluator": 10.0,
    "end_user": 0.0,
    "unknown": 0.0,
}
_SEGMENT_ROLE_SCORE_DIVISOR = 5.0


def _segment_role_multiplier(role_type: Any) -> float:
    role_name = str(role_type or "").strip() or "unknown"
    role_score = _SEGMENT_ROLE_SCORES.get(role_name, 0.0)
    return max(1.0, role_score / _SEGMENT_ROLE_SCORE_DIVISOR)


def _segment_role_priority_score(role_type: Any, count: Any) -> float:
    try:
        raw_count = float(count or 0)
    except (TypeError, ValueError):
        raw_count = 0.0
    return raw_count * _segment_role_multiplier(role_type)


def _segment_role_count_value(count: Any) -> float:
    try:
        return float(count or 0)
    except (TypeError, ValueError):
        return 0.0


def _sorted_segment_role_counts(role_types: dict[str, Any] | None) -> list[tuple[str, Any]]:
    entries = list((role_types or {}).items())
    return sorted(
        entries,
        key=lambda item: (
            _segment_role_priority_score(item[0], item[1]),
            _segment_role_count_value(item[1]),
            _SEGMENT_ROLE_SCORES.get(str(item[0] or "").strip() or "unknown", 0.0),
            str(item[0] or ""),
        ),
        reverse=True,
    )


def _known_buying_stage_counts(stage_counts: dict[str, Any] | None) -> dict[str, Any]:
    """Prefer explicit buying stages over placeholder labels like 'unknown'."""
    raw = stage_counts or {}
    known: dict[str, Any] = {}
    unknown_total = 0
    for stage, count in raw.items():
        stage_name = str(stage or "").strip() or "unknown"
        if stage_name.lower() in {"unknown", "none", "null", "n/a", "na"}:
            unknown_total += count or 0
            continue
        known[stage_name] = known.get(stage_name, 0) + (count or 0)
    if known:
        return known
    if unknown_total:
        return {"unknown": unknown_total}
    return {}


def _dominant_segment_role(role_types: dict[str, Any] | None) -> str:
    ranked = _sorted_segment_role_counts(role_types)
    if not ranked:
        return "unknown"
    return str(ranked[0][0] or "").strip() or "unknown"


def build_segment_intelligence(
    vendor_name: str,
    *,
    buyer_auth: dict | None = None,
    department_entries: list[dict] | None = None,
    company_size_entries: list[dict] | None = None,
    budget: dict | None = None,
    dm_rate: float | None = None,
    contract_value_entries: list[dict] | None = None,
    usage_duration_entries: list[dict] | None = None,
    use_case_entries: list[dict] | None = None,
    role_churn: dict[str, dict[str, Any]] | None = None,
    vendor_lock_in_level: str | None = None,
    analysis_window_days: int = 90,
) -> dict[str, Any]:
    """Build a canonical segment intelligence object for a single vendor.

    Pass 1: populates affected_roles, departments, company sizes, budget
    pressure, contract/duration segments, and top use cases under pressure.
    Fields requiring cross-joins or LLM synthesis are set to null.
    """
    from datetime import date

    as_of = date.today().isoformat()
    ba = buyer_auth or {}
    bg = budget or {}

    def _sf(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # --- Affected roles (from buyer_auth_lookup) ---
    affected_roles: list[dict] = []
    role_types_raw = ba.get("role_types") or {}
    known_role_types: dict[str, Any] = {}
    unknown_role_count = 0
    for role_type, count in role_types_raw.items():
        role_name = str(role_type or "").strip() or "unknown"
        if role_name.lower() in {"unknown", "none", "null", "n/a", "na"}:
            unknown_role_count += count or 0
            continue
        known_role_types[role_name] = (
            known_role_types.get(role_name, 0) + (count or 0)
        )
    if known_role_types:
        role_types = known_role_types
    elif unknown_role_count:
        role_types = {"unknown": unknown_role_count}
    else:
        role_types = {}
    buying_stages = _known_buying_stage_counts(ba.get("buying_stages") or {})
    role_buying_stages = ba.get("role_buying_stages") or {}
    for rt, count in _sorted_segment_role_counts(role_types):
        top_stage = None
        per_role_stages = _known_buying_stage_counts(role_buying_stages.get(rt) or {})
        if per_role_stages:
            top_stage = max(
                per_role_stages.items(),
                key=lambda item: (item[1], str(item[0] or "")),
            )[0]
        rc = (role_churn or {}).get(rt) or {}
        affected_roles.append({
            "role_type": rt,
            "review_count": count,
            "priority_score": round(_segment_role_priority_score(rt, count), 2),
            "top_buying_stage": top_stage,
            "churn_rate": _sf(rc.get("churn_rate")),
            "top_pain": rc.get("top_pain") or None,
        })
    # Fill remaining gaps with the dominant global stage.
    if affected_roles and buying_stages:
        top_stage = max(buying_stages, key=buying_stages.get)
        for role in affected_roles:
            if not role.get("top_buying_stage"):
                role["top_buying_stage"] = top_stage

    # --- Affected departments ---
    affected_departments: list[dict] = []
    for d in (department_entries or []):
        affected_departments.append({
            "department": d.get("department", ""),
            "review_count": d.get("review_count", 0),
            "churn_rate": _sf(d.get("churn_rate")),
            "avg_urgency": _sf(d.get("avg_urgency")),
        })

    size_distribution = [
        {
            "segment": entry.get("segment", ""),
            "review_count": entry.get("review_count", 0),
            "churn_rate": _sf(entry.get("churn_rate")),
        }
        for entry in (company_size_entries or [])
        if entry.get("segment")
    ]

    # --- Company size signals ---
    affected_company_sizes = {
        "avg_seat_count": _sf(bg.get("avg_seat_count")),
        "median_seat_count": _sf(bg.get("median_seat_count")),
        "max_seat_count": _sf(bg.get("max_seat_count")),
        "size_distribution": size_distribution or None,
    }

    # --- Budget pressure ---
    budget_pressure = {
        "price_increase_count": bg.get("price_increase_count") or 0,
        "price_increase_rate": _sf(bg.get("price_increase_rate")),
        "dm_churn_rate": _sf(dm_rate),
        "annual_spend_signals": bg.get("annual_spend_signals") or [],
        "price_per_seat_signals": bg.get("price_per_seat_signals") or [],
    }

    # --- Contract segments ---
    contract_segments: list[dict] = []
    for cv in (contract_value_entries or []):
        contract_segments.append({
            "segment": cv.get("segment", ""),
            "count": cv.get("count", 0),
            "churn_rate": _sf(cv.get("churn_rate")),
        })

    # --- Usage duration segments ---
    usage_duration_segments: list[dict] = []
    for ud in (usage_duration_entries or []):
        usage_duration_segments.append({
            "duration": ud.get("duration", ""),
            "count": ud.get("count", 0),
            "churn_rate": _sf(ud.get("churn_rate")),
        })

    # --- Top use cases under pressure ---
    _lock_in = vendor_lock_in_level or None
    top_use_cases: list[dict] = []
    for uc in (use_case_entries or [])[:10]:
        top_use_cases.append({
            "use_case": uc.get("module") or uc.get("use_case_name") or uc.get("use_case", ""),
            "mention_count": uc.get("mentions") or uc.get("mention_count", 0),
            "lock_in_level": _lock_in,
            "confidence_score": _sf(uc.get("confidence_score")),
        })

    # --- Buying stage distribution ---
    buying_stage_distribution: list[dict] = []
    for stage, count in sorted(buying_stages.items(), key=lambda x: -x[1]):
        buying_stage_distribution.append({
            "stage": stage,
            "count": count,
        })

    return {
        "vendor_name": vendor_name,
        "schema_version": _SEGMENT_INTELLIGENCE_SCHEMA_VERSION,
        "as_of_date": as_of,
        "analysis_window_days": analysis_window_days,
        "affected_roles": affected_roles,
        "affected_departments": affected_departments,
        "affected_company_sizes": affected_company_sizes,
        "budget_pressure": budget_pressure,
        "contract_segments": contract_segments,
        "usage_duration_segments": usage_duration_segments,
        "top_use_cases_under_pressure": top_use_cases,
        "buying_stage_distribution": buying_stage_distribution,
        "best_fit_challenger_segments": None,
    }


_TEMPORAL_INTELLIGENCE_SCHEMA_VERSION = "v1"


def build_temporal_intelligence(
    vendor_name: str,
    *,
    timeline_entries: list[dict] | None = None,
    keyword_spikes: dict | None = None,
    sentiment: dict | None = None,
    sentiment_tenure: list[dict[str, Any]] | None = None,
    turning_points: list[dict[str, Any]] | None = None,
    analysis_window_days: int = 90,
) -> dict[str, Any]:
    """Build a canonical temporal intelligence object for a single vendor.

    Pass 1: populates evaluation deadlines, keyword spikes, sentiment
    trajectory, and timeline signal counts.  Structured trend per-weakness
    and acceleration metrics are deferred to pass 2.
    """
    from datetime import date

    as_of = date.today().isoformat()
    kw = keyword_spikes or {}
    sent = sentiment or {}

    # --- Active evaluation deadlines (processed from raw timeline) ---
    eval_deadlines = _build_active_evaluation_deadlines(
        timeline_entries or [],
        limit=10,
    )

    # --- Immediate triggers (entries with concrete dates) ---
    immediate_triggers: list[dict] = []
    for ed in eval_deadlines:
        if ed.get("evaluation_deadline") or ed.get("contract_end"):
            immediate_triggers.append({
                "company": ed.get("company"),
                "trigger_type": ed.get("trigger_type", "unknown"),
                "date": ed.get("evaluation_deadline") or ed.get("contract_end"),
                "urgency": ed.get("urgency", 0),
                "title": ed.get("title"),
                "company_size": ed.get("company_size"),
            })

    # --- Keyword spike details ---
    spike_keywords: list[dict] = []
    trend_summary = kw.get("trend_summary") or {}
    if isinstance(trend_summary, str):
        trend_summary = _safe_json(trend_summary) or {}
    if not isinstance(trend_summary, dict):
        trend_summary = {}
    for keyword, detail in trend_summary.items():
        if isinstance(detail, dict):
            spike_keywords.append({
                "keyword": keyword,
                "volume": detail.get("volume", 0),
                "change_pct": detail.get("change_pct", 0),
                "is_spike": bool(detail.get("is_spike")),
            })
    spike_keywords.sort(
        key=lambda x: abs(x.get("change_pct") or 0), reverse=True,
    )

    # --- Sentiment trajectory ---
    total_sentiment = sum(sent.values()) or 1
    sentiment_trajectory = {
        "declining": sent.get("declining", 0),
        "stable": sent.get("stable", 0),
        "improving": sent.get("improving", 0),
        "total": sum(sent.values()),
        "declining_pct": round(
            sent.get("declining", 0) / total_sentiment, 2,
        ),
        "improving_pct": round(
            sent.get("improving", 0) / total_sentiment, 2,
        ),
    }

    # --- Timeline signal summary ---
    raw_entries = timeline_entries or []
    contract_end_count = sum(
        1 for e in raw_entries if e.get("contract_end")
    )
    eval_deadline_count = sum(
        1 for e in raw_entries if e.get("evaluation_deadline")
    )
    # Renewal signals: contract_end dates within the analysis window
    _renewal_horizon = analysis_window_days
    _today = date.today()
    renewal_count = 0
    for e in raw_entries:
        ce = _parse_timeline_date(e.get("contract_end"))
        if ce is not None and _today <= ce <= _today + timedelta(days=_renewal_horizon):
            renewal_count += 1
    # Budget cycle signals: decision_timeline mentions of budget/planning terms
    _budget_terms = ("budget", "planning", "fiscal", "annual review", "quarterly review")
    budget_cycle_count = sum(
        1 for e in raw_entries
        if any(t in str(e.get("decision_timeline") or "").lower() for t in _budget_terms)
    )

    tenure_distribution = [
        {
            "tenure": item.get("tenure"),
            "count": item.get("count", 0),
        }
        for item in (sentiment_tenure or [])
        if isinstance(item, dict) and item.get("tenure")
    ]
    turning_point_summary = [
        {
            "trigger": item.get("trigger"),
            "mentions": item.get("mentions", 0),
        }
        for item in (turning_points or [])
        if isinstance(item, dict) and item.get("trigger")
    ]

    return {
        "vendor_name": vendor_name,
        "schema_version": _TEMPORAL_INTELLIGENCE_SCHEMA_VERSION,
        "as_of_date": as_of,
        "analysis_window_days": analysis_window_days,
        "immediate_triggers": immediate_triggers,
        "evaluation_deadlines": eval_deadlines,
        "keyword_spikes": {
            "spike_count": kw.get("spike_count", 0),
            "spike_keywords": kw.get("spike_keywords", []),
            "keyword_details": spike_keywords,
        },
        "sentiment_trajectory": sentiment_trajectory,
        "sentiment_tenure": tenure_distribution[:10],
        "turning_points": turning_point_summary[:10],
        "timeline_signal_summary": {
            "total_signals": len(raw_entries),
            "contract_end_signals": contract_end_count,
            "evaluation_deadline_signals": eval_deadline_count,
            "renewal_signals": renewal_count,
            "budget_cycle_signals": budget_cycle_count,
        },
        "trend_per_weakness": None,
        "acceleration_metrics": None,
    }


_DISPLACEMENT_DYNAMICS_SCHEMA_VERSION = "v1"


def build_displacement_dynamics(
    from_vendor: str,
    to_vendor: str,
    *,
    edge: dict | None = None,
    displacement_flows: list[dict] | None = None,
    reasons: list[dict] | None = None,
    battle_conclusion: dict | None = None,
    analysis_window_days: int = 90,
) -> dict[str, Any]:
    """Build a canonical displacement dynamics object for a vendor pair.

    Pass 1: populates edge metrics, switch reasons, evidence types,
    and battle conclusion summary.  Segment-level displacement breakdowns
    and trend acceleration are deferred to pass 2.

    Args:
        from_vendor: the vendor losing customers.
        to_vendor: the vendor gaining customers.
        edge: persisted displacement edge row (from b2b_displacement_edges).
        displacement_flows: competitive_disp entries for this pair.
        reasons: competitor_reasons entries for this pair.
        battle_conclusion: cross-vendor pairwise battle conclusion (jsonb).
    """
    from datetime import date

    as_of = date.today().isoformat()
    eg = edge or {}

    def _sf(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _intish(v: Any, default: int = 0) -> int:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    def _normalize_evidence_type(value: Any) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if text in {"explicit_switch", "switch", "switched_to"}:
            return "explicit_switch"
        if text in {"active_evaluation", "evaluation", "considering", "trial", "poc"}:
            return "active_evaluation"
        if text in {"implied_preference", "compared", "preference"}:
            return "implied_preference"
        return None

    # --- Edge metrics (from persisted displacement edges) ---
    edge_metrics = {
        "mention_count": eg.get("mention_count") or 0,
        "primary_driver": eg.get("primary_driver") if not _is_generic_pain_label(eg.get("primary_driver")) else None,
        "signal_strength": eg.get("signal_strength"),
        "key_quote": eg.get("key_quote"),
        "confidence_score": _sf(eg.get("confidence_score")),
        "velocity_7d": eg.get("velocity_7d") or 0,
        "velocity_30d": eg.get("velocity_30d") or 0,
    }

    # --- Evidence type breakdown (from competitive_disp flows) ---
    evidence_breakdown: list[dict] = []
    flows = displacement_flows or []
    for f in flows:
        row_breakdown = f.get("evidence_breakdown") or []
        if isinstance(row_breakdown, list) and row_breakdown:
            for item in row_breakdown:
                if not isinstance(item, dict):
                    continue
                evidence_type = _normalize_evidence_type(item.get("evidence_type"))
                mention_count = _intish(item.get("mention_count") or 0)
                reason_categories = item.get("reason_categories") or {}
                if mention_count <= 0 and isinstance(reason_categories, dict):
                    mention_count = sum(_intish(v) for v in reason_categories.values())
                if not evidence_type or mention_count <= 0:
                    continue
                evidence_breakdown.append({
                    "evidence_type": evidence_type,
                    "mention_count": mention_count,
                    "reason_categories": reason_categories,
                    "industries": item.get("industries") or f.get("industries") or [],
                    "company_sizes": item.get("company_sizes") or f.get("company_sizes") or [],
                })
            continue

        explicit = _intish(f.get("explicit_switches") or 0)
        active_eval = _intish(f.get("active_evaluations") or 0)
        implied = _intish(f.get("implied_preferences") or 0)
        mention_count = _intish(f.get("mention_count") or 0)
        if mention_count <= 0:
            mention_count = explicit + active_eval + implied
        if explicit <= 0 and active_eval <= 0 and implied <= 0:
            evidence_type = _normalize_evidence_type(f.get("evidence_type")) or "implied_preference"
            if mention_count > 0:
                evidence_breakdown.append({
                    "evidence_type": evidence_type,
                    "mention_count": mention_count,
                    "reason_categories": f.get("reason_categories") or {},
                    "industries": f.get("industries") or [],
                    "company_sizes": f.get("company_sizes") or [],
                })
            continue
        for evidence_type, count in (
            ("explicit_switch", explicit),
            ("active_evaluation", active_eval),
            ("implied_preference", max(implied, mention_count - explicit - active_eval)),
        ):
            if count <= 0:
                continue
            evidence_breakdown.append({
                "evidence_type": evidence_type,
                "mention_count": count,
                "reason_categories": {},
                "industries": f.get("industries") or [],
                "company_sizes": f.get("company_sizes") or [],
            })
    evidence_breakdown.sort(
        key=lambda x: x["mention_count"], reverse=True,
    )

    # --- Switch reasons (from competitor_reasons) ---
    switch_reasons: list[dict] = []
    for r in (reasons or []):
        switch_reasons.append({
            "reason": r.get("reason") or r.get("reason_category", ""),
            "reason_category": r.get("reason_category"),
            "reason_detail": r.get("reason_detail"),
            "direction": r.get("direction"),
            "mention_count": r.get("mention_count", 0),
        })
    switch_reasons.sort(
        key=lambda x: x["mention_count"], reverse=True,
    )

    # --- Battle conclusion summary (from cross-vendor reasoning) ---
    bc = battle_conclusion or {}
    battle_summary = None
    if bc:
        battle_summary = {
            "winner": bc.get("winner"),
            "loser": bc.get("loser"),
            "conclusion": bc.get("conclusion"),
            "confidence": _sf(bc.get("confidence")),
            "durability_assessment": bc.get("durability_assessment"),
            "key_insights": bc.get("key_insights") or [],
            "resource_advantage": bc.get("resource_advantage"),
        }

    # --- Net flow direction ---
    total_explicit = sum(
        f["mention_count"] for f in evidence_breakdown
        if f["evidence_type"] == "explicit_switch"
    )
    total_eval = sum(
        f["mention_count"] for f in evidence_breakdown
        if f["evidence_type"] == "active_evaluation"
    )
    total_mentions = max(
        _intish(edge_metrics.get("mention_count")),
        sum(f["mention_count"] for f in evidence_breakdown),
    )

    return {
        "from_vendor": from_vendor,
        "to_vendor": to_vendor,
        "schema_version": _DISPLACEMENT_DYNAMICS_SCHEMA_VERSION,
        "as_of_date": as_of,
        "analysis_window_days": analysis_window_days,
        "edge_metrics": edge_metrics,
        "evidence_breakdown": evidence_breakdown,
        "switch_reasons": switch_reasons,
        "battle_summary": battle_summary,
        "flow_summary": {
            "explicit_switch_count": total_explicit,
            "active_evaluation_count": total_eval,
            "total_flow_mentions": total_mentions,
        },
        "segment_displacement": None,
        "trend_acceleration": None,
    }


_CATEGORY_DYNAMICS_SCHEMA_VERSION = "v1"


def build_category_dynamics(
    category: str,
    *,
    market_regime: dict | None = None,
    council_conclusion: dict | None = None,
    vendor_count: int = 0,
    displacement_flow_count: int = 0,
    analysis_window_days: int = 90,
) -> dict[str, Any]:
    """Build a canonical category dynamics object.

    Pass 1: populates market regime, council conclusion summary,
    vendor/flow counts, and structural indicators.
    Cross-category comparisons deferred to pass 2.

    Args:
        category: product category name.
        market_regime: MarketRegime as dict (from asdict()).
        council_conclusion: cross-vendor category_council conclusion (jsonb).
    """
    from datetime import date

    as_of = date.today().isoformat()
    mr = market_regime or {}
    cc = council_conclusion or {}

    def _sf(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # --- Market regime (from MarketPulseReasoner) ---
    regime = {
        "regime_type": mr.get("regime_type", "unknown"),
        "confidence": _sf(mr.get("confidence")),
        "avg_churn_velocity": _sf(mr.get("avg_churn_velocity")),
        "avg_price_pressure": _sf(mr.get("avg_price_pressure")),
        "outlier_vendors": mr.get("outlier_vendors") or [],
        "narrative": mr.get("narrative") or "",
    }

    # --- Council conclusion (from cross-vendor LLM reasoning) ---
    council_summary = None
    if cc:
        council_summary = {
            "market_regime": cc.get("market_regime"),
            "winner": cc.get("winner"),
            "loser": cc.get("loser"),
            "conclusion": cc.get("conclusion"),
            "confidence": _sf(cc.get("confidence")),
            "key_insights": cc.get("key_insights") or [],
            "durability_assessment": cc.get("durability_assessment"),
            "segment_dynamics": cc.get("segment_dynamics")
            or cc.get("segment_winners_losers"),
            "category_default": cc.get("category_default"),
        }

    return {
        "category": category,
        "schema_version": _CATEGORY_DYNAMICS_SCHEMA_VERSION,
        "as_of_date": as_of,
        "analysis_window_days": analysis_window_days,
        "market_regime": regime,
        "council_summary": council_summary,
        "vendor_count": vendor_count,
        "displacement_flow_count": displacement_flow_count,
        "cross_category_comparison": None,
    }


_ACCOUNT_INTELLIGENCE_SCHEMA_VERSION = "v1"


def build_account_intelligence(
    vendor_name: str,
    *,
    high_intent_entries: list[dict] | None = None,
    persisted_signals: list[dict] | None = None,
    blocked_names: set[str] | None = None,
    analysis_window_days: int = 90,
) -> dict[str, Any]:
    """Build a canonical account intelligence object for a single vendor.

    Aggregates company-level signals from high_intent (in-memory) and
    b2b_company_signals (persisted).  Per-company scoring and enrichment
    deferred to the accounts-in-motion task.
    """
    from datetime import date

    as_of = date.today().isoformat()

    def _sf(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _is_active_eval_stage(stage: Any) -> bool:
        text = str(stage or "").strip().lower()
        if not text:
            return False
        if text in {"evaluation", "active_purchase", "consideration", "trial", "poc"}:
            return True
        return "evaluat" in text or "consider" in text

    # Merge: prefer persisted signals, supplement with high_intent
    seen_companies: set[str] = set()
    accounts: list[dict] = []

    for ps in (persisted_signals or []):
        cn = (ps.get("company_name") or "").strip()
        if not cn:
            continue
        if _company_signal_exclusion_reason(
            cn,
            current_vendor=vendor_name,
            blocked_names=blocked_names,
            source=ps.get("source"),
            confidence_score=ps.get("confidence_score"),
        ):
            continue
        key = cn.lower()
        if key in seen_companies:
            continue
        seen_companies.add(key)
        acct: dict[str, Any] = {
            "company_name": cn,
            "urgency_score": _sf(ps.get("urgency_score")),
            "pain_category": ps.get("pain_category"),
            "buyer_role": ps.get("buyer_role"),
            "decision_maker": ps.get("decision_maker"),
            "seat_count": ps.get("seat_count"),
            "contract_end": (
                str(ps.get("contract_end") or "") or None
            ),
            "buying_stage": ps.get("buying_stage"),
            "source": ps.get("source"),
            "content_type": ps.get("content_type"),
            "confidence_score": _sf(ps.get("confidence_score")),
            "first_seen_at": (
                str(ps.get("first_seen_at") or "") or None
            ),
            "last_seen_at": (
                str(ps.get("last_seen_at") or "") or None
            ),
        }
        if ps.get("title"):
            acct["title"] = ps["title"]
        if ps.get("company_size"):
            acct["company_size"] = ps["company_size"]
        if ps.get("industry"):
            acct["industry"] = ps["industry"]
        ps_alts = ps.get("alternatives")
        if isinstance(ps_alts, list) and ps_alts:
            acct["alternatives"] = ps_alts
        ps_quotes = ps.get("quotes")
        if isinstance(ps_quotes, list) and ps_quotes:
            acct["quotes"] = ps_quotes
        if ps.get("review_id"):
            acct["review_id"] = str(ps["review_id"])
        accounts.append(acct)

    for hi in (high_intent_entries or []):
        cn = (hi.get("company") or "").strip()
        if not cn:
            continue
        blocked = set(blocked_names or set())
        blocked.update(
            normalize_company_name(name)
            for name in _extract_alternative_names(hi.get("alternatives") or [])
            if normalize_company_name(name)
        )
        if _company_signal_exclusion_reason(
            cn,
            current_vendor=vendor_name,
            blocked_names=blocked,
            source=hi.get("source"),
            confidence_score=hi.get("confidence_score"),
        ):
            continue
        key = cn.lower()
        if key in seen_companies:
            continue
        seen_companies.add(key)
        acct_hi: dict[str, Any] = {
            "company_name": cn,
            "urgency_score": _sf(hi.get("urgency")),
            "pain_category": hi.get("pain"),
            "buyer_role": hi.get("role_level"),
            "decision_maker": hi.get("decision_maker"),
            "seat_count": hi.get("seat_count"),
            "contract_end": (
                str(hi.get("contract_end") or "") or None
            ),
            "buying_stage": hi.get("buying_stage"),
            "source": hi.get("source"),
            "confidence_score": None,
            "first_seen_at": None,
            "last_seen_at": None,
        }
        if hi.get("title"):
            acct_hi["title"] = hi["title"]
        if hi.get("company_size"):
            acct_hi["company_size"] = hi["company_size"]
        if hi.get("industry"):
            acct_hi["industry"] = hi["industry"]
        if hi.get("review_id"):
            acct_hi["review_id"] = str(hi["review_id"])
        alts = hi.get("alternatives")
        if alts:
            acct_hi["alternatives"] = alts if isinstance(alts, list) else []
        hi_quotes = hi.get("quotes")
        if hi_quotes:
            acct_hi["quotes"] = hi_quotes if isinstance(hi_quotes, list) else []
        accounts.append(acct_hi)

    accounts.sort(
        key=lambda a: a.get("urgency_score") or 0, reverse=True,
    )

    # Summary stats
    dm_count = sum(1 for a in accounts if a.get("decision_maker"))
    high_intent_count = sum(
        1 for a in accounts if float(a.get("urgency_score") or 0) >= settings.b2b_churn.high_churn_urgency_threshold
    )
    active_eval_signal_count = sum(
        1 for a in accounts if _is_active_eval_stage(a.get("buying_stage"))
    )
    with_contract_end = sum(
        1 for a in accounts if a.get("contract_end")
    )
    with_seat_count = sum(
        1 for a in accounts if a.get("seat_count")
    )

    return {
        "vendor_name": vendor_name,
        "schema_version": _ACCOUNT_INTELLIGENCE_SCHEMA_VERSION,
        "as_of_date": as_of,
        "analysis_window_days": analysis_window_days,
        "accounts": accounts,
        "summary": {
            "total_accounts": len(accounts),
            "decision_maker_count": dm_count,
            "high_intent_count": high_intent_count,
            "active_eval_signal_count": active_eval_signal_count,
            "with_contract_end": with_contract_end,
            "with_seat_count": with_seat_count,
        },
    }


async def read_high_intent_companies(
    pool,
    *,
    min_urgency: float = 7.0,
    window_days: int = 30,
    vendor_name: str | None = None,
    scoped_vendors: list[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Shared adapter for high-intent company reads.

    Replaces direct enrichment reads in:
      - api.b2b_dashboard.list_high_intent / export_high_intent
      - api.b2b_tenant_dashboard.list_leads
      - mcp.b2b.signals.list_high_intent_companies

    Returns dicts with keys: company, raw_company, resolution_confidence,
    vendor, category, title, company_size, industry, verified_employee_count,
    company_country, company_domain, revenue_range, founded_year,
    total_funding, funding_stage, headcount_growth_6m/12m/24m,
    publicly_traded, ticker, company_description, role_level, decision_maker,
    urgency, pain, alternatives, quotes, contract_signal, review_id, source,
    seat_count, lock_in_level, contract_end, buying_stage, relevance_score,
    author_churn_score, intent_signals.
    """
    return await _fetch_high_intent_companies(
        pool,
        urgency_threshold=min_urgency,
        window_days=window_days,
        vendor_name=vendor_name,
        scoped_vendors=scoped_vendors,
        limit=limit,
    )


async def read_review_details(
    pool,
    *,
    window_days: int = 30,
    vendor_name: str | None = None,
    scoped_vendors: list[str] | None = None,
    pain_category: str | None = None,
    min_urgency: float | None = None,
    company: str | None = None,
    has_churn_intent: bool | None = None,
    min_relevance: float | None = None,
    exclude_low_fidelity: bool = False,
    content_type: str | None = None,
    recency_column: str = "enriched_at",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Shared adapter for review detail reads.

    Replaces direct enrichment reads in:
      - api.b2b_dashboard.search_reviews / export_reviews
      - api.b2b_tenant_dashboard.list_tenant_reviews
      - mcp.b2b.reviews.search_reviews

    Returns dicts with keys: id, vendor_name, product_category,
    reviewer_company, rating, source, reviewed_at, enriched_at,
    urgency_score, pain_category, intent_to_leave, decision_maker,
    role_level, buying_stage, sentiment_direction, industry,
    reviewer_title, company_size, content_type, thread_id,
    competitors_mentioned, quotable_phrases, positive_aspects,
    specific_complaints, relevance_score, author_churn_score,
    low_fidelity, low_fidelity_reasons.
    """
    from atlas_brain.services.b2b.corrections import suppress_predicate

    if recency_column == "enriched_at":
        recency_expr = "r.enriched_at"
    else:
        recency_expr = "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)"
    conditions = [
        "r.enrichment_status = 'enriched'",
        "r.duplicate_of_review_id IS NULL",
        f"{recency_expr} > NOW() - make_interval(days => $1)",
    ]
    params: list = [window_days]
    idx = 2

    if scoped_vendors is not None:
        if not scoped_vendors:
            return []  # scoped user with no tracked vendors = zero results
        conditions.append(f"r.vendor_name = ANY(${idx}::text[])")
        params.append(scoped_vendors)
        idx += 1
    if vendor_name:
        conditions.append(f"r.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1
    if pain_category:
        conditions.append(f"r.enrichment->>'pain_category' = ${idx}")
        params.append(pain_category)
        idx += 1
    if min_urgency is not None:
        conditions.append(f"(r.enrichment->>'urgency_score')::numeric >= ${idx}")
        params.append(min_urgency)
        idx += 1
    if company:
        conditions.append(f"r.reviewer_company ILIKE '%' || ${idx} || '%'")
        params.append(company)
        idx += 1
    if has_churn_intent is not None:
        conditions.append(
            f"(r.enrichment->'churn_signals'->>'intent_to_leave')::boolean = ${idx}"
        )
        params.append(has_churn_intent)
        idx += 1
    if min_relevance is not None:
        conditions.append(f"COALESCE(r.relevance_score, 0.5) >= ${idx}")
        params.append(min_relevance)
        idx += 1
    if exclude_low_fidelity:
        conditions.append("(r.low_fidelity IS NULL OR r.low_fidelity = false)")

    if content_type:
        conditions.append(f"r.content_type = ${idx}")
        params.append(content_type)
        idx += 1
    else:
        conditions.append(
            suppress_predicate(
                "review", id_expr="r.id", source_expr="r.source",
                vendor_expr="r.vendor_name",
            )
        )
    params.append(limit)
    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT r.id, r.vendor_name, r.product_category, r.reviewer_company,
               r.rating, r.source, r.reviewed_at, r.enriched_at,
               (r.enrichment->>'urgency_score')::numeric AS urgency_score,
               r.enrichment->>'pain_category' AS pain_category,
               (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
               (r.enrichment->'reviewer_context'->>'decision_maker')::boolean AS decision_maker,
               r.enrichment->'reviewer_context'->>'role_level' AS role_level,
               r.enrichment->'buyer_authority'->>'buying_stage' AS buying_stage,
               r.sentiment_direction,
               COALESCE(r.reviewer_industry,
                        r.enrichment->'reviewer_context'->>'industry') AS industry,
               r.reviewer_title, r.company_size_raw,
               r.content_type, r.thread_id,
               r.enrichment->'competitors_mentioned' AS competitors_raw,
               r.enrichment->'quotable_phrases' AS quotable_raw,
               r.enrichment->'positive_aspects' AS positive_raw,
               r.enrichment->'specific_complaints' AS complaints_raw,
               r.relevance_score, r.author_churn_score,
               r.low_fidelity, r.low_fidelity_reasons
        FROM b2b_reviews r
        WHERE {where}
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT ${idx}
        """,
        *params,
    )

    results = []
    for r in rows:
        urg = None
        if r["urgency_score"] is not None:
            try:
                urg = float(r["urgency_score"])
            except (ValueError, TypeError):
                pass
        results.append({
            "id": str(r["id"]) if r["id"] else None,
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "reviewer_company": r["reviewer_company"],
            "rating": float(r["rating"]) if r["rating"] is not None else None,
            "source": r["source"],
            "reviewed_at": r["reviewed_at"],
            "enriched_at": r["enriched_at"],
            "urgency_score": urg,
            "pain_category": r["pain_category"],
            "intent_to_leave": bool(r["intent_to_leave"]) if r["intent_to_leave"] is not None else None,
            "decision_maker": bool(r["decision_maker"]) if r["decision_maker"] is not None else None,
            "role_level": r["role_level"] if r["role_level"] != "unknown" else None,
            "buying_stage": r["buying_stage"] if r["buying_stage"] != "unknown" else None,
            "sentiment_direction": r["sentiment_direction"],
            "industry": r["industry"],
            "reviewer_title": r["reviewer_title"],
            "company_size": r["company_size_raw"],
            "content_type": r["content_type"],
            "thread_id": r["thread_id"],
            "competitors_mentioned": _safe_json(r["competitors_raw"]),
            "quotable_phrases": _safe_json(r["quotable_raw"]),
            "positive_aspects": _safe_json(r["positive_raw"]),
            "specific_complaints": _safe_json(r["complaints_raw"]),
            "relevance_score": float(r["relevance_score"]) if r["relevance_score"] is not None else None,
            "author_churn_score": float(r["author_churn_score"]) if r["author_churn_score"] is not None else None,
            "low_fidelity": bool(r["low_fidelity"]) if r["low_fidelity"] is not None else None,
            "low_fidelity_reasons": _safe_json(r["low_fidelity_reasons"]),
        })
    return results


async def read_campaign_opportunities(
    pool,
    *,
    window_days: int = 90,
    min_urgency: float = 5.0,
    vendor_name: str | None = None,
    company: str | None = None,
    dm_only: bool = True,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Shared adapter for campaign opportunity enrichment reads.

    Replaces direct enrichment reads in:
      - tasks.b2b_campaign_generation._fetch_opportunities
      - api.b2b_affiliates.list_opportunities

    Returns review-level dicts with buyer authority, timeline, competitor
    context, pain categories, and quotable phrases extracted from enrichment.
    Consumers add domain-specific logic (scoring, affiliate matching, etc.).
    """
    from atlas_brain.services.b2b.corrections import suppress_predicate

    conditions = [
        "r.enrichment_status = 'enriched'",
        "r.duplicate_of_review_id IS NULL",
        "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)"
        " > NOW() - make_interval(days => $1)",
        "(r.enrichment->>'urgency_score')::numeric >= $2",
    ]
    params: list = [window_days, min_urgency]
    idx = 3

    if vendor_name:
        conditions.append(f"r.vendor_name ILIKE '%' || ${idx} || '%'")
        params.append(vendor_name)
        idx += 1
    if company:
        conditions.append(
            f"COALESCE(NULLIF(BTRIM(r.reviewer_company), ''),"
            f" NULLIF(BTRIM(r.reviewer_company_norm), ''))"
            f" ILIKE '%' || ${idx} || '%'"
        )
        params.append(company)
        idx += 1
    if dm_only:
        conditions.append(
            "(r.enrichment->'reviewer_context'->>'decision_maker')::boolean = true"
        )

    conditions.append(
        suppress_predicate(
            "review", id_expr="r.id", source_expr="r.source",
            vendor_expr="r.vendor_name",
        )
    )
    params.append(limit)
    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT r.id AS review_id,
               r.vendor_name,
               COALESCE(NULLIF(BTRIM(r.reviewer_company), ''),
                        NULLIF(BTRIM(r.reviewer_company_norm), '')) AS reviewer_company,
               r.product_category, r.source, r.reviewed_at,
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
               r.enrichment->'use_case'->>'primary_workflow' AS primary_workflow,
               r.enrichment->'use_case'->'integration_stack' AS integration_stack,
               r.sentiment_direction,
               COALESCE(r.reviewer_industry,
                        r.enrichment->'reviewer_context'->>'industry') AS industry,
               r.reviewer_title, r.company_size_raw,
               NULLIF(BTRIM(r.reviewer_name), '') AS reviewer_name
        FROM b2b_reviews r
        WHERE {where}
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC
        LIMIT ${idx}
        """,
        *params,
    )

    results = []
    for r in rows:
        competitors = _safe_json(r["competitors_json"])
        if not isinstance(competitors, list):
            competitors = []
        seat_count = r["seat_count"]
        results.append({
            "review_id": str(r["review_id"]) if r["review_id"] else None,
            "vendor_name": r["vendor_name"],
            "reviewer_company": r["reviewer_company"],
            "reviewer_name": r["reviewer_name"],
            "product_category": r["product_category"],
            "source": r["source"],
            "reviewed_at": r["reviewed_at"],
            "urgency": float(r["urgency"]) if r["urgency"] is not None else None,
            "is_dm": bool(r["is_dm"]) if r["is_dm"] is not None else None,
            "role_type": r["role_type"],
            "buying_stage": r["buying_stage"],
            "seat_count": seat_count,
            "contract_end": r["contract_end"],
            "decision_timeline": r["decision_timeline"],
            "competitors": competitors,
            "competitors_json": r["competitors_json"],
            "pain_json": r["pain_json"],
            "quotable_phrases": _safe_json(r["quotable_phrases"]),
            "feature_gaps": _safe_json(r["feature_gaps"]),
            "primary_workflow": r["primary_workflow"],
            "integration_stack": _safe_json(r["integration_stack"]),
            "sentiment_direction": r["sentiment_direction"],
            "industry": r["industry"],
            "reviewer_title": r["reviewer_title"],
            "company_size_raw": r["company_size_raw"],
        })
    return results


# ---------------------------------------------------------------------------
# Class 2: Shared SQL fragment helpers for repeated aggregate patterns
# ---------------------------------------------------------------------------

def _vendor_evidence_base_filters(
    *,
    alias: str = "r",
    window_param: int = 1,
    recency_column: str = "enriched_at",
) -> str:
    """Base WHERE clause for vendor evidence queries.

    Centralizes enrichment status check, recency semantics, and suppression.
    Returns a SQL fragment with $<window_param> as the window_days placeholder.
    """
    from atlas_brain.services.b2b.corrections import suppress_predicate

    if recency_column == "enriched_at":
        recency = f"{alias}.enriched_at"
    else:
        recency = f"COALESCE({alias}.reviewed_at, {alias}.imported_at, {alias}.enriched_at)"

    return (
        f"{alias}.enrichment_status = 'enriched'"
        f" AND {recency} > NOW() - make_interval(days => ${window_param})"
        f" AND {suppress_predicate('review', id_expr=f'{alias}.id', source_expr=f'{alias}.source', vendor_expr=f'{alias}.vendor_name')}"
    )


def _competitor_unnest_sql(alias: str = "r") -> str:
    """SQL fragment for CROSS JOIN LATERAL on competitors_mentioned."""
    return (
        f"CROSS JOIN LATERAL jsonb_array_elements("
        f"CASE WHEN jsonb_typeof({alias}.enrichment->'competitors_mentioned') = 'array'"
        f" THEN {alias}.enrichment->'competitors_mentioned'"
        f" ELSE '[]'::jsonb END) AS comp(value)"
    )


def _integration_stack_unnest_sql(alias: str = "r") -> str:
    """SQL fragment for jsonb_array_elements_text on use_case.integration_stack."""
    return (
        f"jsonb_array_elements_text("
        f"CASE WHEN jsonb_typeof({alias}.enrichment->'use_case'->'integration_stack') = 'array'"
        f" THEN {alias}.enrichment->'use_case'->'integration_stack'"
        f" ELSE '[]'::jsonb END)"
    )


# ---------------------------------------------------------------------------
# Class 1: Row-level evidence readers
# ---------------------------------------------------------------------------

async def read_vendor_quote_evidence(
    pool,
    *,
    vendor_name: str,
    window_days: int = 90,
    min_urgency: float = 5.0,
    limit: int = 10,
    sources: list[str] | None = None,
    pain_filter: str | None = None,
    require_quotes: bool = False,
    recency_column: str = "enriched_at",
) -> list[dict[str, Any]]:
    """Row-level quote evidence for a vendor.

    Replaces direct enrichment reads in:
      - b2b_blog_post_generation._resolve_blog_battle_summary (pricing/switching)
      - b2b_blog_post_generation._fetch_high_urgency_quotes (vendor variant)
      - b2b_challenger_brief._fetch_review_pain_quotes

    Args:
      require_quotes: When True, only returns reviews with non-empty
          quotable_phrases (matches challenger_brief semantics).
      recency_column: 'enriched_at' (default), 'imported_at' (challenger_brief),
          or 'coalesce' (reviewed_at -> imported_at -> enriched_at).

    Returns dicts with: vendor_name, source, reviewer_company, reviewer_title,
    role_level, pain_category, urgency, review_text, quotable_phrases, rating.
    """
    from atlas_brain.services.b2b.corrections import suppress_predicate

    if recency_column == "imported_at":
        recency_expr = "r.imported_at"
    elif recency_column == "coalesce":
        recency_expr = "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)"
    else:
        recency_expr = "r.enriched_at"

    conditions = [
        "r.enrichment_status = 'enriched'",
        "r.duplicate_of_review_id IS NULL",
        f"{recency_expr} > NOW() - make_interval(days => $1)",
        "LOWER(r.vendor_name) = LOWER($2)",
    ]
    params: list = [window_days, vendor_name]
    idx = 3

    if min_urgency > 0:
        conditions.append(f"(r.enrichment->>'urgency_score')::numeric >= ${idx}")
        params.append(min_urgency)
        idx += 1
    if sources:
        conditions.append(f"r.source = ANY(${idx}::text[])")
        params.append(sources)
        idx += 1
    if pain_filter:
        conditions.append(f"r.enrichment->>'pain_categories' ILIKE '%' || ${idx} || '%'")
        params.append(pain_filter)
        idx += 1
    if require_quotes:
        conditions.append(
            "r.enrichment->'quotable_phrases' IS NOT NULL"
            " AND jsonb_array_length(r.enrichment->'quotable_phrases') > 0"
        )

    conditions.append(
        suppress_predicate("review", id_expr="r.id", source_expr="r.source", vendor_expr="r.vendor_name")
    )
    params.append(limit)
    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT r.vendor_name, r.source, r.reviewer_company, r.reviewer_title,
               COALESCE(r.reviewer_title, r.enrichment->'reviewer_context'->>'role_level') AS role_level,
               r.enrichment->>'pain_category' AS pain_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               r.review_text, r.rating,
               r.enrichment->'quotable_phrases' AS quotable_raw
        FROM b2b_reviews r
        WHERE {where}
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST
        LIMIT ${idx}
        """,
        *params,
    )

    return [
        {
            "vendor_name": r["vendor_name"],
            "source": r["source"],
            "reviewer_company": r["reviewer_company"],
            "reviewer_title": r["reviewer_title"],
            "role_level": r["role_level"],
            "pain_category": r["pain_category"],
            "urgency": float(r["urgency"]) if r["urgency"] is not None else None,
            "review_text": r["review_text"],
            "rating": float(r["rating"]) if r["rating"] is not None else None,
            "quotable_phrases": _safe_json(r["quotable_raw"]),
        }
        for r in rows
    ]


async def read_category_quote_evidence(
    pool,
    *,
    product_category: str,
    window_days: int = 90,
    min_urgency: float = 5.0,
    limit: int = 10,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Row-level quote evidence for a product category.

    Replaces direct enrichment reads in:
      - b2b_blog_post_generation._fetch_high_urgency_quotes (category variant)

    Same return shape as read_vendor_quote_evidence.
    """
    from atlas_brain.services.b2b.corrections import suppress_predicate

    conditions = [
        "r.enrichment_status = 'enriched'",
        "r.duplicate_of_review_id IS NULL",
        "r.enriched_at > NOW() - make_interval(days => $1)",
        "r.product_category = $2",
    ]
    params: list = [window_days, product_category]
    idx = 3

    if min_urgency > 0:
        conditions.append(f"(r.enrichment->>'urgency_score')::numeric >= ${idx}")
        params.append(min_urgency)
        idx += 1
    if sources:
        conditions.append(f"r.source = ANY(${idx}::text[])")
        params.append(sources)
        idx += 1

    conditions.append(
        suppress_predicate("review", id_expr="r.id", source_expr="r.source", vendor_expr="r.vendor_name")
    )
    params.append(limit)
    where = " AND ".join(conditions)

    rows = await pool.fetch(
        f"""
        SELECT r.vendor_name, r.source, r.reviewer_company, r.reviewer_title,
               COALESCE(r.reviewer_title, r.enrichment->'reviewer_context'->>'role_level') AS role_level,
               r.enrichment->>'pain_category' AS pain_category,
               (r.enrichment->>'urgency_score')::numeric AS urgency,
               r.review_text, r.rating,
               r.enrichment->'quotable_phrases' AS quotable_raw
        FROM b2b_reviews r
        WHERE {where}
        ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST
        LIMIT ${idx}
        """,
        *params,
    )

    return [
        {
            "vendor_name": r["vendor_name"],
            "source": r["source"],
            "reviewer_company": r["reviewer_company"],
            "reviewer_title": r["reviewer_title"],
            "role_level": r["role_level"],
            "pain_category": r["pain_category"],
            "urgency": float(r["urgency"]) if r["urgency"] is not None else None,
            "review_text": r["review_text"],
            "rating": float(r["rating"]) if r["rating"] is not None else None,
            "quotable_phrases": _safe_json(r["quotable_raw"]),
        }
        for r in rows
    ]


def _battle_card_weaknesses_from_evidence_vault(
    vault: dict[str, Any] | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Normalize vault weakness evidence into the current battle-card shape."""
    if not isinstance(vault, dict):
        return []
    weaknesses: list[dict[str, Any]] = []
    for item in vault.get("weakness_evidence") or []:
        if not isinstance(item, dict):
            continue
        area = str(item.get("label") or item.get("key") or "").strip()
        if not area:
            continue
        entry = {
            "area": area,
            "evidence_count": int(item.get("mention_count_total") or 0),
            "source": str(item.get("evidence_type") or "evidence_vault"),
        }
        if item.get("supporting_metrics", {}).get("satisfaction_score") is not None:
            entry["score"] = item["supporting_metrics"]["satisfaction_score"]
        weaknesses.append(entry)
    weaknesses.sort(key=lambda item: -int(item.get("evidence_count") or 0))
    return weaknesses[:limit]


def _battle_card_strengths_from_evidence_vault(
    vault: dict[str, Any] | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Normalize vault strength evidence into battle-card shape.

    Mirrors ``_battle_card_weaknesses_from_evidence_vault`` so the LLM can
    ground objection handler ``acknowledge`` fields in real incumbent strengths.
    """
    if not isinstance(vault, dict):
        return []
    strengths: list[dict[str, Any]] = []
    for item in vault.get("strength_evidence") or []:
        if not isinstance(item, dict):
            continue
        area = str(item.get("label") or item.get("key") or "").strip()
        if not area:
            continue
        total = int(item.get("mention_count_total") or 0)
        trend = item.get("trend") if isinstance(item.get("trend"), dict) else {}
        entry: dict[str, Any] = {
            "area": area,
            "mention_count": total,
            "source": str(item.get("evidence_type") or "evidence_vault"),
        }
        direction = str(trend.get("direction") or "").strip()
        if direction:
            entry["trend"] = direction
        quote = str(item.get("best_quote") or "").strip()
        if quote:
            entry["customer_quote"] = quote
        strengths.append(entry)
    strengths.sort(key=lambda s: -int(s.get("mention_count") or 0))
    return strengths[:limit]


def _looks_like_company_domain(raw_name: str) -> bool:
    text = str(raw_name or "").strip().lower()
    if not text or " " in text or "/" in text or "@" in text:
        return False
    return bool(re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", text))


_GENERIC_COMPANY_PATTERNS = (
    re.compile(r"^(msp|saas|fintech|startup|start up|non-?profit|university|school|government|public sector)$", re.I),
    re.compile(r"(^| )(software company|tech company|saas company|consulting company)( |$)", re.I),
    re.compile(r"(^| )(government agency|b2c fintech|b2c fintech company|b2b fintech|b2b fintech company)( |$)", re.I),
    re.compile(
        r"(^| )(b2b|b2c|e-?commerce|ecommerce|support|video|marketing|design|creative|recruitment|retail|fintech)( |-|_).*(startup|agency|studio|company|team|business|organization|department)( |$)",
        re.I,
    ),
    re.compile(r"(^| )(agency|organization|vendor|provider|department)( |$)", re.I),
    re.compile(r"(^| )\d+\s+employees?( |$)", re.I),
)

_GENERIC_COMPANY_DESCRIPTOR_PREFIXES = {
    "small",
    "medium",
    "midsized",
    "mid",
    "large",
    "enterprise",
    "multinational",
    "global",
    "regional",
    "local",
    "private",
    "public",
    "independent",
    "boutique",
}

_GENERIC_COMPANY_DESCRIPTOR_TOKENS = {
    "agency",
    "bank",
    "banking",
    "business",
    "company",
    "consulting",
    "creative",
    "department",
    "design",
    "ecommerce",
    "erp",
    "fintech",
    "firm",
    "health",
    "healthcare",
    "insurance",
    "management",
    "manufacturing",
    "marketing",
    "organization",
    "pharmaceutical",
    "pharma",
    "project",
    "provider",
    "recruitment",
    "retail",
    "saas",
    "software",
    "solution",
    "solutions",
    "support",
    "sized",
    "team",
    "technical",
    "vendor",
    "video",
}

_PLACEHOLDER_COMPANY_NAMES = {
    "company",
    "mycompany",
    "my company",
    "ourcompany",
    "our company",
    "ourdomain",
    "customer",
    "client",
}

_LOCATION_LIKE_COMPANY_NAMES = {
    "costa rica",
    "united states",
    "usa",
    "united kingdom",
    "uk",
    "canada",
    "australia",
    "india",
    "germany",
    "france",
    "mexico",
    "brazil",
}

_PARTNERISH_COMPANY_PATTERNS = (
    re.compile(r".*partners?$", re.I),
    re.compile(r".*(reseller|consulting|consultancy|implementer|implementation)$", re.I),
)


def _looks_like_generic_company_descriptor(raw_name: Any) -> bool:
    """Return True for descriptive labels that are not seller-usable org names."""
    text = str(raw_name or "").strip()
    if not text:
        return True
    normalized = normalize_company_name(text)
    if not normalized:
        return True
    if normalized in _PLACEHOLDER_COMPANY_NAMES:
        return True
    tokens = [token for token in re.split(r"[\s/_-]+", normalized) if token]
    if tokens and tokens[0] in {"a", "an"}:
        tokens = tokens[1:]
    if len(tokens) >= 2 and tokens[0] in _GENERIC_COMPANY_DESCRIPTOR_PREFIXES:
        if all(token in _GENERIC_COMPANY_DESCRIPTOR_TOKENS for token in tokens[1:]):
            return True
    return any(pattern.search(normalized) for pattern in _GENERIC_COMPANY_PATTERNS)


def _looks_like_location_label(raw_name: Any) -> bool:
    normalized = normalize_company_name(raw_name)
    if not normalized:
        return False
    return normalized in _LOCATION_LIKE_COMPANY_NAMES


def _looks_like_partner_or_tool_label(raw_name: Any) -> bool:
    normalized = normalize_company_name(raw_name)
    if not normalized:
        return False
    return any(pattern.fullmatch(normalized) for pattern in _PARTNERISH_COMPANY_PATTERNS)


def _build_company_signal_blocked_names_by_vendor(
    vendor_names: Iterable[Any],
    *,
    high_intent_entries: list[dict[str, Any]] | None = None,
    integration_lookup: dict[str, list[Any]] | None = None,
) -> dict[str, set[str]]:
    """Build vendor-specific blocklists for non-prospect company labels."""
    vendor_list = list(vendor_names or [])
    normalized_vendors = {
        normalize_company_name(name)
        for name in vendor_list
        if normalize_company_name(name)
    }
    blocked_by_vendor: dict[str, set[str]] = {}
    for name in vendor_list:
        vendor = _canonicalize_vendor(name or "")
        if vendor:
            blocked_by_vendor[vendor] = set(normalized_vendors)

    for vendor, items in (integration_lookup or {}).items():
        vendor_key = _canonicalize_vendor(vendor or "")
        if not vendor_key:
            continue
        bucket = blocked_by_vendor.setdefault(vendor_key, set(normalized_vendors))
        for item in items or []:
            if isinstance(item, dict):
                raw_name = (
                    item.get("integration_name")
                    or item.get("integration")
                    or item.get("name")
                    or ""
                )
            else:
                raw_name = item
            normalized = normalize_company_name(raw_name)
            if normalized:
                bucket.add(normalized)

    for hi in high_intent_entries or []:
        vendor = _canonicalize_vendor(hi.get("vendor") or hi.get("vendor_name") or "")
        if not vendor:
            continue
        bucket = blocked_by_vendor.setdefault(vendor, set(normalized_vendors))
        for name in _extract_alternative_names(hi.get("alternatives") or []):
            normalized = normalize_company_name(name)
            if normalized:
                bucket.add(normalized)

    return blocked_by_vendor


def _company_signal_name_is_eligible(
    raw_name: Any,
    *,
    current_vendor: str = "",
    blocked_names: set[str] | None = None,
) -> bool:
    """Return True when a company-like label is safe to keep as a signal."""
    name = str(raw_name or "").strip()
    if not name:
        return False
    normalized = normalize_company_name(name)
    if not normalized:
        return False
    if _looks_like_company_domain(name):
        return False
    if _looks_like_generic_company_descriptor(name):
        return False
    if _looks_like_location_label(name):
        return False
    if _looks_like_partner_or_tool_label(name):
        return False
    if current_vendor and normalized == normalize_company_name(current_vendor):
        return False
    blocked = blocked_names or set()
    if normalized in blocked:
        return False
    return True


def _company_signal_exclusion_reason(
    raw_name: Any,
    *,
    current_vendor: str = "",
    blocked_names: set[str] | None = None,
    source: Any = None,
    confidence_score: Any = None,
) -> str | None:
    """Return a stable exclusion reason for non-actionable company signals."""
    if not _company_signal_name_is_eligible(
        raw_name,
        current_vendor=current_vendor,
        blocked_names=blocked_names,
    ):
        return "ineligible_company_name"

    normalized_source = str(source or "").strip().lower()
    if normalized_source and normalized_source in _company_signal_skip_sources():
        return "deprecated_source"

    if normalized_source and normalized_source in _company_signal_low_trust_sources():
        confidence = _normalize_company_signal_confidence(confidence_score)
        if confidence is None or confidence < float(settings.b2b_churn.company_signal_low_trust_min_confidence):
            return "low_confidence_low_trust_source"

    return None


def _high_intent_signal_evidence_enabled() -> bool:
    return bool(getattr(settings.b2b_churn, "high_intent_require_signal_evidence", True))


def _high_intent_row_has_signal_evidence(row: dict[str, Any]) -> bool:
    return any(
        bool(row.get(key))
        for key in (
            "intent_to_leave",
            "actively_evaluating",
            "contract_renewal_mentioned",
            "indicator_cancel",
            "indicator_migration",
            "indicator_evaluation",
            "indicator_switch",
        )
    )


def _battle_card_company_is_display_safe(
    raw_name: Any,
    *,
    current_vendor: str = "",
    blocked_names: set[str] | None = None,
    role: Any = None,
    company_size: Any = None,
    buying_stage: Any = None,
) -> bool:
    if not _company_signal_name_is_eligible(
        raw_name,
        current_vendor=current_vendor,
        blocked_names=blocked_names,
    ):
        return False
    # Seller-facing rows need at least one qualifier beyond just a name.
    return any(bool(val) for val in (role, company_size, buying_stage))


def _normalize_buying_stage(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return text.replace("-", "_").replace(" ", "_")


def _battle_card_required_buying_stages() -> set[str]:
    raw = getattr(settings.b2b_churn, "battle_card_quality_required_stages", None)
    if not isinstance(raw, list):
        return set()
    return {_normalize_buying_stage(item) for item in raw if str(item or "").strip()}


def _battle_card_min_high_intent_urgency() -> float:
    return float(getattr(settings.b2b_churn, "battle_card_quality_min_high_intent_urgency", 7.0))


def _rank_high_intent_companies(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_stages = _battle_card_required_buying_stages()
    min_urgency = _battle_card_min_high_intent_urgency()

    def _score(item: dict[str, Any]) -> tuple[int, float, int, float, str]:
        try:
            urgency = float(item.get("urgency") or 0)
        except (TypeError, ValueError):
            urgency = 0.0
        stage = _normalize_buying_stage(item.get("buying_stage"))
        stage_qualified = int(bool(required_stages) and stage in required_stages and urgency >= min_urgency)
        dm = int(bool(item.get("decision_maker")))
        try:
            confidence = float(item.get("confidence_score") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        name = str(item.get("company") or "").strip().lower()
        return (stage_qualified, urgency, dm, confidence, name)

    return sorted(companies, key=_score, reverse=True)


def _normalize_canonical_accounts_for_battle_card(
    accounts: list[dict[str, Any]],
    *,
    current_vendor: str = "",
    blocked_names: set[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Convert canonical account-intelligence rows to battle-card company shape."""
    candidates: list[dict[str, Any]] = []
    for a in accounts:
        if not isinstance(a, dict):
            continue
        company = a.get("company_name") or a.get("name") or ""
        if not company:
            continue
        role = a.get("buyer_role") or a.get("title") or ""
        if role.lower() in ("unknown", ""):
            role = ""
        entry: dict[str, Any] = {
            "company": company,
            "urgency": float(a.get("urgency_score") or a.get("urgency") or 0),
            "role": role or None,
            "pain": a.get("pain_category") or None,
            "company_size": a.get("company_size"),
            "source": a.get("source"),
            "buying_stage": a.get("buying_stage"),
            "decision_maker": bool(a.get("decision_maker")),
            "confidence_score": float(a.get("confidence_score") or 0),
            "contract_end": a.get("contract_end"),
            "industry": a.get("industry"),
        }
        if not _battle_card_company_is_display_safe(
            entry["company"],
            current_vendor=current_vendor,
            blocked_names=blocked_names,
            role=entry.get("role"),
            company_size=entry.get("company_size"),
            buying_stage=entry.get("buying_stage"),
        ):
            continue
        candidates.append(entry)
    return _rank_high_intent_companies(candidates)[:limit]


def _battle_card_companies_from_evidence_vault(
    vault: dict[str, Any] | None,
    *,
    current_vendor: str = "",
    blocked_names: set[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Normalize vault company signals into the current battle-card shape."""
    if not isinstance(vault, dict):
        return []
    companies: list[dict[str, Any]] = []
    for item in vault.get("company_signals") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("company_name") or "").strip()
        if _company_signal_exclusion_reason(
            name,
            current_vendor=current_vendor,
            blocked_names=blocked_names,
            source=item.get("source"),
            confidence_score=item.get("confidence_score"),
        ):
            continue
        if not _battle_card_company_is_display_safe(
            name,
            current_vendor=current_vendor,
            blocked_names=blocked_names,
            role=item.get("buyer_role"),
            company_size=item.get("seat_count"),
            buying_stage=item.get("buying_stage"),
        ):
            continue
        entry: dict[str, Any] = {
            "company": name,
            "urgency": item.get("urgency_score"),
            "role": item.get("buyer_role"),
            "pain": item.get("pain_category"),
            "company_size": item.get("seat_count"),
            "source": item.get("source"),
            "buying_stage": item.get("buying_stage"),
        }
        if item.get("decision_maker") is not None:
            entry["decision_maker"] = item["decision_maker"]
        if item.get("confidence_score") is not None:
            entry["confidence_score"] = item["confidence_score"]
        if item.get("contract_end"):
            entry["contract_end"] = str(item["contract_end"])
        companies.append(entry)
    return _rank_high_intent_companies(companies)[:limit]


def _battle_card_provenance_from_evidence_vault(
    vault: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map vault provenance into the battle-card provenance attachment shape."""
    if not isinstance(vault, dict):
        return {}
    provenance = vault.get("provenance") or {}
    if not isinstance(provenance, dict):
        return {}
    mapped: dict[str, Any] = {}
    if provenance.get("source_distribution"):
        mapped["source_distribution"] = provenance["source_distribution"]
    if provenance.get("sample_review_ids"):
        mapped["sample_review_ids"] = provenance["sample_review_ids"]
    if provenance.get("enrichment_window_start"):
        mapped["review_window_start"] = provenance["enrichment_window_start"]
    if provenance.get("enrichment_window_end"):
        mapped["review_window_end"] = provenance["enrichment_window_end"]
    return mapped


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
    dict for synthesis and deterministic reasoning consumers.

    Includes pain, competitors, feature gaps, insider signals, keyword spikes,
    temporal velocities, archetype pre-scores, and optionally DM rate, price
    complaint rate, budget signals, buyer authority, use cases, and quotes.
    """
    vendor = _canonicalize_vendor(vs.get("vendor_name") or "")
    total = int(vs.get("total_reviews") or 0)
    signal_total = int(vs.get("signal_reviews") or 0) or total
    churn = int(vs.get("churn_intent") or 0)
    churn_density = round((churn * 100.0 / signal_total), 1) if signal_total else 0.0
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
        "displacement_mention_count": 0,
        "quote_count": 0,
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
            q0 = quotes[0]
            if isinstance(q0, dict):
                evidence["top_quote"] = str(q0.get("quote") or q0.get("text") or "")[:200] or None
            elif isinstance(q0, str):
                evidence["top_quote"] = q0[:200]
            else:
                evidence["top_quote"] = None
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
    all_scores = await read_vendor_scorecards(
        pool,
        window_days=window_days,
        min_reviews=min_reviews,
        vendor_names=[vendor_name],
    )
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
        _fetch_competitive_displacement_source_of_truth(
            pool,
            as_of=date.today(),
            analysis_window_days=window_days,
        ),
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
    synthesis_views: dict[str, Any] | None = None,
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
        sig_reviews = int(row.get("signal_reviews") or 0)
        if vendor not in merged:
            merged[vendor] = {
                "total_reviews": reviews,
                "signal_reviews": sig_reviews,
                "churn_intent": churn,
                "urgency_weighted_sum": urgency * reviews,
                "category": category,
                "category_reviews": reviews,
            }
        else:
            m = merged[vendor]
            m["total_reviews"] += reviews
            m["signal_reviews"] = m.get("signal_reviews", 0) + sig_reviews
            m["churn_intent"] += churn
            m["urgency_weighted_sum"] += urgency * reviews
            # Keep category with most reviews
            if reviews > m["category_reviews"]:
                m["category"] = category
                m["category_reviews"] = reviews

    candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    for vendor, m in merged.items():
        total_reviews = m["total_reviews"]
        signal_reviews = int(m.get("signal_reviews") or 0) or total_reviews
        churn_intent = m["churn_intent"]
        churn_density = round((churn_intent * 100.0 / signal_reviews), 1) if signal_reviews else 0.0
        avg_urgency = round(m["urgency_weighted_sum"] / total_reviews, 1) if total_reviews else 0.0
        category = m["category"]
        dm_rate = float(dm_lookup.get(vendor, 0))
        price_rate = float(price_lookup.get(vendor, 0))
        # Load reasoning once per vendor
        _rc = _get_vendor_reasoning(vendor, synthesis_views=synthesis_views, reasoning_lookup=reasoning_lookup)
        reasoning_confidence = float(_rc.get("confidence", 0) or 0)

        passes_primary_gate = not (
            churn_density < 15 and avg_urgency < 6 and dm_rate < 0.3
        )
        passes_secondary_signal = (
            reasoning_confidence >= 0.75
            or price_rate >= 0.18
            or dm_rate >= 0.08
            or churn_density >= 5.0
        )
        if not passes_primary_gate and not (
            total_reviews >= 50 and passes_secondary_signal
        ):
            continue

        # Confidence label
        if total_reviews >= 50:
            confidence = "high"
        elif total_reviews >= 20:
            confidence = "medium"
        else:
            confidence = "low"

        # Boost confidence when reasoning provides corroborating evidence
        if reasoning_confidence >= 0.8 and confidence == "medium":
            confidence = "high"

        # Displacement mention total for this vendor
        comp_entries = competitor_lookup.get(vendor, [])
        displacement_mentions = sum(c.get("mentions", 0) for c in comp_entries)

        _vendor_archetype = _rc.get("archetype")
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
        total_pain = sum(int(p.get("count") or 0) for p in pains)
        pain_breakdown = [
            {
                "category": p["category"],
                "count": p["count"],
                "pct": round(int(p.get("count") or 0) / max(total_pain, 1), 3),
            }
            for p in pains[:3]
        ]

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
        dominant_role = _dominant_segment_role(role_types)

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
             "industry": c.get("industry") if isinstance(c, dict) else None,
             "source": c.get("source") if isinstance(c, dict) else None,
             "buying_stage": c.get("buying_stage") if isinstance(c, dict) else None,
             "confidence_score": c.get("confidence_score") if isinstance(c, dict) else None,
             "decision_maker": c.get("decision_maker") if isinstance(c, dict) else None,
             "first_seen_at": c.get("first_seen_at") if isinstance(c, dict) else None,
             "last_seen_at": c.get("last_seen_at") if isinstance(c, dict) else None}
            for c in companies[:5]
        ]

        # Risk level -- prefer reasoning conclusion, fall back to deterministic
        _rc_risk = _rc.get("risk_level", "")
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
        if _rc:
            entry["archetype"] = _rc.get("archetype", "")
            entry["archetype_confidence"] = reasoning_confidence
            entry["archetype_risk_level"] = _rc.get("risk_level", "")
            entry["reasoning_mode"] = _rc.get("mode", "")
            # Synthesis-native fields when available
            if _rc.get("mode") == "synthesis":
                entry["reasoning_source"] = "synthesis"
                summary = _rc.get("executive_summary", "")
                if summary:
                    entry["reasoning_summary"] = summary
        if passes_primary_gate:
            candidates.append(entry)
        else:
            entry["selection_mode"] = "secondary_signal"
            fallback_candidates.append(entry)

    # Reasoning-weighted sort: vendors with high-confidence archetypes get a boost
    def _sort_key(x: dict) -> tuple:
        score = x["churn_pressure_score"]
        conf = x.get("archetype_confidence", 0)
        has_arch = bool(x.get("archetype"))
        reasoning_boost = min(conf * 5, 5.0) if has_arch else 0
        return (-(score + reasoning_boost),)
    candidates.sort(key=_sort_key)
    if not candidates and fallback_candidates:
        fallback_candidates.sort(key=_sort_key)
        return fallback_candidates[:limit]
    return candidates[:limit]


def _build_deterministic_displacement_map(
    competitive_disp: list[dict[str, Any]],
    competitor_reasons: list[dict[str, Any]],
    quote_lookup: dict[str, list],
    *,
    synthesis_views: dict[str, Any] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Build displacement report from evidence-quality-filtered aggregated flows.

    Quality gate: only edges with at least one explicit_switch or active_evaluation
    survive.  Pure implied_preference edges are market-pain data, not displacement.
    """
    reason_lookup = _build_reason_lookup(competitor_reasons)
    def _rl_get(v: str) -> dict:
        return _get_vendor_reasoning(v, synthesis_views=synthesis_views, reasoning_lookup=reasoning_lookup)

    def _source_vendor_wedge(vendor_name: str) -> str:
        canon = _canonicalize_vendor(vendor_name)
        if not canon:
            return ""
        if synthesis_views:
            view = synthesis_views.get(canon)
            if view is None:
                for key, candidate in synthesis_views.items():
                    if _canonicalize_vendor(key) == canon:
                        view = candidate
                        break
            if view is not None:
                wedge = getattr(view, "primary_wedge", None)
                if wedge is not None:
                    return str(getattr(wedge, "value", "") or "")
                section = getattr(view, "section", None)
                if callable(section):
                    cn = section("causal_narrative")
                    if isinstance(cn, dict):
                        return str(cn.get("primary_wedge") or "")
        return str(_rl_get(canon).get("archetype") or "")

    results: list[dict[str, Any]] = []
    for row in competitive_disp:
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        competitor = _canonicalize_competitor(row.get("competitor") or "")
        if not vendor or not competitor or vendor.lower() == competitor.lower():
            continue
        if not _battle_card_competitor_is_eligible(competitor):
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
            normalized_reason_cats: dict[str, int] = {}
            for key, count in reason_cats.items():
                label = _normalize_displacement_driver_label(key)
                if not label:
                    continue
                normalized_reason_cats[label] = normalized_reason_cats.get(label, 0) + int(count or 0)
            if normalized_reason_cats:
                canonical_priority = {
                    "pricing": 9,
                    "features": 8,
                    "integration": 7,
                    "ux": 6,
                    "support": 5,
                    "reliability": 4,
                    "security": 3,
                    "compliance": 2,
                    "performance": 1,
                    "migration": 1,
                }
                driver = max(
                    normalized_reason_cats.items(),
                    key=lambda x: (x[1], canonical_priority.get(x[0], 0)),
                )[0]
            else:
                # All keys failed normalization -- try keyword inference on raw keys
                driver = _infer_driver_from_reasons(list(reason_cats.keys()))
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
        src_arch = _source_vendor_wedge(vendor)
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
        _src_rc = _rl_get(vendor)
        _tgt_rc = _rl_get(competitor)
        if _src_rc.get("archetype"):
            edge_entry["source_archetype"] = _src_rc["archetype"]
            edge_entry["source_archetype_confidence"] = _src_rc.get("confidence", 0)
        if _tgt_rc.get("archetype"):
            edge_entry["target_archetype"] = _tgt_rc["archetype"]
            edge_entry["target_archetype_confidence"] = _tgt_rc.get("confidence", 0)
        results.append(edge_entry)
    results.sort(key=lambda x: x["mention_count"], reverse=True)
    return results[:limit]


def _structure_displacement_report(flows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap the flat flow list into a narrative-first report structure.

    Sections:
      market_losers   -- vendors bleeding the most customers (net outbound)
      market_winners  -- vendors gaining the most customers (net inbound)
      top_battles     -- highest-signal flows, with battle_conclusion when available
      driver_summary  -- what's driving displacement across all flows
      meta            -- headline numbers
      flows           -- full sorted list preserved for backward compat
    """
    if not flows:
        return {
            "market_losers": [],
            "market_winners": [],
            "top_battles": [],
            "driver_summary": [],
            "meta": {
                "total_flows": 0,
                "total_mentions": 0,
                "dominant_driver": None,
                "most_displaced_vendor": None,
                "biggest_winner": None,
            },
            "flows": [],
        }

    # --- Compute per-vendor net flow ---
    inbound: dict[str, int] = {}
    outbound: dict[str, int] = {}
    # top destination/source and driver per vendor
    out_flows: dict[str, list[dict]] = {}
    in_flows: dict[str, list[dict]] = {}
    driver_mentions: dict[str, int] = {}
    driver_flow_count: dict[str, int] = {}

    for f in flows:
        fv = f.get("from_vendor", "")
        tv = f.get("to_vendor", "")
        mc = int(f.get("mention_count") or 0)
        drv = f.get("primary_driver") or "unspecified"

        outbound[fv] = outbound.get(fv, 0) + mc
        inbound[tv] = inbound.get(tv, 0) + mc
        out_flows.setdefault(fv, []).append(f)
        in_flows.setdefault(tv, []).append(f)
        driver_mentions[drv] = driver_mentions.get(drv, 0) + mc
        driver_flow_count[drv] = driver_flow_count.get(drv, 0) + 1

    all_vendors = set(inbound) | set(outbound)
    net = {v: inbound.get(v, 0) - outbound.get(v, 0) for v in all_vendors}
    total_mentions = sum(f.get("mention_count", 0) for f in flows)

    def _top_driver_for(flow_list: list[dict]) -> str | None:
        counts: dict[str, int] = {}
        for f in flow_list:
            d = f.get("primary_driver") or "unspecified"
            counts[d] = counts.get(d, 0) + int(f.get("mention_count") or 0)
        return max(counts, key=lambda x: counts[x]) if counts else None

    # --- Market losers (most net-negative vendors) ---
    losers = sorted(
        [(v, n) for v, n in net.items() if n < 0],
        key=lambda x: x[1],
    )[:10]
    market_losers = []
    for vendor, net_flow in losers:
        vout = sorted(out_flows.get(vendor, []), key=lambda x: x.get("mention_count", 0), reverse=True)
        market_losers.append({
            "vendor": vendor,
            "net_flow": net_flow,
            "outbound_mentions": outbound.get(vendor, 0),
            "inbound_mentions": inbound.get(vendor, 0),
            "top_destination": vout[0]["to_vendor"] if vout else None,
            "top_driver": _top_driver_for(out_flows.get(vendor, [])),
        })

    # --- Market winners (most net-positive vendors) ---
    winners = sorted(
        [(v, n) for v, n in net.items() if n > 0],
        key=lambda x: -x[1],
    )[:10]
    market_winners = []
    for vendor, net_flow in winners:
        vin = sorted(in_flows.get(vendor, []), key=lambda x: x.get("mention_count", 0), reverse=True)
        market_winners.append({
            "vendor": vendor,
            "net_flow": net_flow,
            "outbound_mentions": outbound.get(vendor, 0),
            "inbound_mentions": inbound.get(vendor, 0),
            "top_source": vin[0]["from_vendor"] if vin else None,
            "top_driver": _top_driver_for(in_flows.get(vendor, [])),
        })

    # --- Top battles: flows with battle_conclusion first, then high-signal remainder ---
    with_battle = [f for f in flows if f.get("battle_conclusion")]
    without_battle = [f for f in flows if not f.get("battle_conclusion")]
    # Sort each group by signal quality: strong > moderate > emerging, then mention_count
    _strength_rank = {"strong": 3, "moderate": 2, "emerging": 1}
    def _battle_sort(f: dict) -> tuple:
        return (
            -_strength_rank.get(f.get("signal_strength", ""), 0),
            -int(f.get("mention_count") or 0),
        )
    top_battles = sorted(with_battle, key=_battle_sort)[:8]
    if len(top_battles) < 8:
        needed = 8 - len(top_battles)
        top_battles += sorted(without_battle, key=_battle_sort)[:needed]

    # Keep only the fields useful for narrative display; drop raw ids
    _battle_keep = {
        "from_vendor", "to_vendor", "mention_count", "primary_driver",
        "signal_strength", "confidence_score", "key_quote",
        "battle_conclusion", "durability", "source_archetype", "target_archetype",
        "reference_ids", "data_as_of_date", "reasoning_source",
    }
    top_battles = [{k: v for k, v in f.items() if k in _battle_keep} for f in top_battles]

    # --- Driver summary ---
    driver_summary = []
    for drv, mentions in sorted(driver_mentions.items(), key=lambda x: -x[1]):
        pct = round(mentions * 100 / total_mentions) if total_mentions else 0
        driver_summary.append({
            "driver": drv,
            "mentions": mentions,
            "pct": pct,
            "flow_count": driver_flow_count.get(drv, 0),
        })

    dominant_driver = driver_summary[0]["driver"] if driver_summary else None
    most_displaced = market_losers[0]["vendor"] if market_losers else None
    biggest_winner = market_winners[0]["vendor"] if market_winners else None

    return {
        "market_losers": market_losers,
        "market_winners": market_winners,
        "top_battles": top_battles,
        "driver_summary": driver_summary,
        "meta": {
            "total_flows": len(flows),
            "total_mentions": total_mentions,
            "dominant_driver": dominant_driver,
            "pricing_pct": next(
                (d["pct"] for d in driver_summary if d["driver"] == "pricing"), 0
            ),
            "most_displaced_vendor": most_displaced,
            "biggest_winner": biggest_winner,
        },
        "flows": flows,
    }


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
    inbound_displacement_lookup: dict[str, int] | None = None,
    synthesis_views: dict[str, Any] | None = None,
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
    evidence_vault_lookup: dict[str, dict[str, Any]] | None = None,
    limit: int | None = 15,
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
        rec_total = int(row.get("recommend_total") or 0)
        category = row.get("product_category") or "Unknown"
        sig_reviews = int(row.get("signal_reviews") or 0)
        if vendor not in merged:
            merged[vendor] = {
                "total_reviews": reviews,
                "signal_reviews": sig_reviews,
                "churn_intent": churn,
                "urgency_weighted_sum": urgency * reviews,
                "recommend_yes": rec_yes,
                "recommend_no": rec_no,
                "recommend_total": rec_total,
                "positive_pct_sum": (float(pos_pct) * reviews) if pos_pct is not None else 0,
                "positive_pct_count": reviews if pos_pct is not None else 0,
                "category": category,
                "category_reviews": reviews,
            }
        else:
            m = merged[vendor]
            m["total_reviews"] += reviews
            m["signal_reviews"] = m.get("signal_reviews", 0) + sig_reviews
            m["churn_intent"] += churn
            m["urgency_weighted_sum"] += urgency * reviews
            m["recommend_yes"] += rec_yes
            m["recommend_no"] += rec_no
            m["recommend_total"] += rec_total
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
        signal_reviews = int(m.get("signal_reviews") or 0) or total_reviews
        churn_intent = m["churn_intent"]
        churn_density = round((churn_intent * 100.0 / signal_reviews), 1) if signal_reviews else 0.0
        avg_urgency = round(m["urgency_weighted_sum"] / total_reviews, 1) if total_reviews else 0.0
        positive_pct = round(m["positive_pct_sum"] / m["positive_pct_count"], 1) if m["positive_pct_count"] else None
        recommend_yes = m["recommend_yes"]
        recommend_no = m["recommend_no"]
        recommend_total = m.get("recommend_total", 0)
        if recommend_total:
            recommend_ratio = round(((recommend_yes - recommend_no) / recommend_total) * 100, 1)
        else:
            # Fall back to evidence vault metric_snapshot when signal table
            # lacks recommend columns (the common path for reports).
            vault = (evidence_vault_lookup or {}).get(vendor) or {}
            ms = vault.get("metric_snapshot") or {}
            vault_yes = int(ms.get("recommend_yes") or 0)
            vault_no = int(ms.get("recommend_no") or 0)
            vault_total = vault_yes + vault_no
            if vault_total:
                recommend_yes = vault_yes
                recommend_no = vault_no
                recommend_ratio = round(((vault_yes - vault_no) / vault_total) * 100, 1)
            else:
                recommend_ratio = 0.0

        if total_reviews >= 50:
            confidence = "high"
        elif total_reviews >= 20:
            confidence = "medium"
        else:
            confidence = "low"
        # Boost confidence when reasoning provides corroborating evidence
        _rc = _get_vendor_reasoning(vendor, synthesis_views=synthesis_views, reasoning_lookup=reasoning_lookup)
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
        total_pain = sum(int(p.get("count") or 0) for p in pains)
        pain_breakdown = [
            {
                "category": p["category"],
                "count": p["count"],
                "pct": round(int(p.get("count") or 0) / max(total_pain, 1), 3),
            }
            for p in pains[:5]
        ]

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
        _vendor_archetype = _rc.get("archetype")
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
             "industry": c.get("industry") if isinstance(c, dict) else None,
             "source": c.get("source") if isinstance(c, dict) else None,
             "buying_stage": c.get("buying_stage") if isinstance(c, dict) else None,
             "confidence_score": c.get("confidence_score") if isinstance(c, dict) else None,
             "decision_maker": c.get("decision_maker") if isinstance(c, dict) else None,
             "first_seen_at": c.get("first_seen_at") if isinstance(c, dict) else None,
             "last_seen_at": c.get("last_seen_at") if isinstance(c, dict) else None}
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
        dominant_role = _dominant_segment_role(role_types)

        # Net Flow calculation: Inbound Displacement - Outbound Churn
        inbound_count = (inbound_displacement_lookup or {}).get(vendor, 0)
        net_flow = inbound_count - churn_intent

        sc_entry = {
            "vendor": vendor,
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "inbound_displacement_count": inbound_count,
            "net_flow": net_flow,
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

        if _rc:
            sc_entry["archetype"] = _rc.get("archetype", "")
            sc_entry["archetype_confidence"] = _rc.get("confidence", 0)
            sc_entry["reasoning_summary"] = _rc.get("executive_summary", "")
            sc_entry["falsification_conditions"] = _rc.get("falsification_conditions", [])
            sc_entry["uncertainty_sources"] = _rc.get("uncertainty_sources", [])
            sc_entry["reasoning_mode"] = _rc.get("mode", "")
            rc_risk = _rc.get("risk_level", "")
            if rc_risk:
                sc_entry["risk_level"] = rc_risk
            if _rc.get("mode") == "synthesis":
                sc_entry["reasoning_source"] = "synthesis"
        results.append(sc_entry)
    def _sc_sort_key(x: dict) -> tuple:
        conf = x.get("archetype_confidence", 0)
        has_arch = bool(x.get("archetype"))
        reasoning_boost = min(conf * 5, 5.0) if has_arch else 0
        return (-(x["avg_urgency"] + reasoning_boost), -(x["churn_signal_density"]), x["vendor"])
    results.sort(key=_sc_sort_key)
    if limit is None:
        return results
    return results[:limit]


def _build_vendor_deep_dives(
    vendor_scores: list[dict[str, Any]],
    *,
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    feature_gap_lookup: dict[str, list[dict]],
    quote_lookup: dict[str, list],
    company_lookup: dict[str, list[dict]],
    dm_lookup: dict[str, float],
    price_lookup: dict[str, float],
    sentiment_lookup: dict[str, dict[str, int]],
    buyer_auth_lookup: dict[str, dict] | None = None,
    synthesis_views: dict[str, Any] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
    limit: int = 60,
) -> list[dict[str, Any]]:
    """Build comprehensive per-vendor deep dive reports, sorted by churn pressure."""
    def _rl_get(v: str) -> dict:
        return _get_vendor_reasoning(v, synthesis_views=synthesis_views, reasoning_lookup=reasoning_lookup)
    _ba = buyer_auth_lookup or {}
    results: list[dict[str, Any]] = []
    _SENT_POS = {"stable_positive", "improving"}
    _SENT_NEG = {"consistently_negative", "declining"}

    for row in vendor_scores:
        v = _canonicalize_vendor(row.get("vendor_name") or "")
        if not v:
            continue

        total_reviews = int(row.get("total_reviews") or 0)
        signal_reviews = int(row.get("signal_reviews") or 0) or total_reviews
        churn_intent = int(row.get("churn_intent") or 0)
        churn_density = round((churn_intent * 100.0 / signal_reviews), 1) if signal_reviews else 0.0
        avg_urgency = round(float(row.get("avg_urgency") or 0), 1)
        dm_rate = float(dm_lookup.get(v, 0))
        pr_rate = float(price_lookup.get(v, 0))
        comp_entries = competitor_lookup.get(v, [])
        disp_mentions = sum(int(c.get("mentions") or 0) for c in comp_entries)
        _rc = _rl_get(v)

        pressure = _compute_churn_pressure_score(
            churn_density=churn_density,
            avg_urgency=avg_urgency,
            dm_churn_rate=dm_rate,
            displacement_mention_count=disp_mentions,
            price_complaint_rate=pr_rate,
            total_reviews=total_reviews,
            archetype=_rc.get("archetype"),
        )
        risk_level = _rc.get("risk_level") or (
            "high" if pressure >= 70 else ("medium" if pressure >= 40 else "low")
        )

        # Pain breakdown
        pains = pain_lookup.get(v, [])
        total_pain = sum(int(p.get("count") or 0) for p in pains)
        pain_breakdown = sorted(
            [
                {
                    "category": p.get("category") or p.get("key") or "unknown",
                    "count": int(p.get("count") or 0),
                    "pct": round(int(p.get("count") or 0) / max(total_pain, 1), 3),
                }
                for p in pains
                if p.get("category") or p.get("key")
            ],
            key=lambda x: -x["count"],
        )[:10]

        # Displacement targets
        displacement_targets = sorted(
            [
                {
                    "vendor": c.get("name", ""),
                    "mention_count": int(c.get("mentions") or c.get("mention_count") or 0),
                    "primary_driver": c.get("primary_driver"),
                }
                for c in comp_entries
                if c.get("name")
            ],
            key=lambda x: -x["mention_count"],
        )[:12]

        # Feature gaps
        feature_gaps = sorted(
            [
                {
                    "feature": g.get("feature") or g.get("gap") or "",
                    "mentions": int(g.get("count") or g.get("mentions") or 0),
                }
                for g in feature_gap_lookup.get(v, [])
                if g.get("feature") or g.get("gap")
            ],
            key=lambda x: -x["mentions"],
        )[:10]

        # Customer profile
        companies = company_lookup.get(v, [])
        ind_counts: dict[str, int] = {}
        size_counts: dict[str, int] = {}
        for c in companies:
            if not isinstance(c, dict):
                continue
            ind = c.get("industry")
            if ind and ind != "unknown":
                ind_counts[ind] = ind_counts.get(ind, 0) + 1
            sz = c.get("company_size")
            if sz:
                size_counts[sz] = size_counts.get(sz, 0) + 1
        industry_distribution = sorted(
            [{"industry": k, "count": n} for k, n in ind_counts.items()],
            key=lambda x: -x["count"],
        )[:8]
        size_distribution = sorted(
            [{"size": k, "count": n} for k, n in size_counts.items()],
            key=lambda x: -x["count"],
        )[:6]

        # Case studies
        raw_quotes = quote_lookup.get(v, [])
        case_studies: list[dict[str, Any]] = []
        for q in raw_quotes:
            if isinstance(q, dict):
                case_studies.append({
                    "quote": str(q.get("quote") or q.get("text") or "")[:300],
                    "company": q.get("company", "Anonymous"),
                    "urgency": float(q.get("urgency") or 0),
                    "title": q.get("title"),
                })
            elif isinstance(q, str):
                case_studies.append({
                    "quote": q[:300],
                    "company": "Anonymous",
                    "urgency": 0.0,
                    "title": None,
                })
        case_studies.sort(key=lambda x: -x["urgency"])

        # Sentiment breakdown -- map direction strings to pos/neg/neutral buckets
        sent = sentiment_lookup.get(v, {})
        _sent_pos = sum(cnt for d, cnt in sent.items() if d in _SENT_POS)
        _sent_neg = sum(cnt for d, cnt in sent.items() if d in _SENT_NEG)
        _sent_neu = sum(cnt for d, cnt in sent.items() if d not in _SENT_POS and d not in _SENT_NEG)

        # Buyer role -- skip "unknown" entries
        ba = _ba.get(v, {})
        if isinstance(ba, dict) and ba.get("role_types"):
            _rt = {k: n for k, n in ba["role_types"].items() if k and k != "unknown"}
            dominant_buyer_role = max(_rt, key=_rt.get, default=None) if _rt else None
        else:
            dominant_buyer_role = None

        results.append({
            "vendor": v,
            "category": row.get("product_category") or "Unknown",
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "churn_pressure_score": round(pressure, 1),
            "avg_urgency": avg_urgency,
            "risk_level": risk_level,
            "sentiment_direction": _rc.get("sentiment_direction") or "",
            "trend": _rc.get("trend") or "",
            "archetype": _rc.get("archetype"),
            "archetype_confidence": _rc.get("confidence"),
            "dm_churn_rate": round(dm_rate, 3),
            "price_complaint_rate": round(pr_rate, 3),
            "dominant_buyer_role": dominant_buyer_role,
            "pain_breakdown": pain_breakdown,
            "displacement_targets": displacement_targets,
            "feature_gaps": feature_gaps,
            "industry_distribution": industry_distribution,
            "company_size_distribution": size_distribution,
            "case_studies": case_studies[:5],
            "sentiment_breakdown": {
                "positive": _sent_pos,
                "negative": _sent_neg,
                "neutral": _sent_neu,
            },
        })

    results.sort(key=lambda x: -x["churn_pressure_score"])
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
    synthesis_views: dict[str, Any] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Build industry-specific category overview with vendor rankings and case studies."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in vendor_scores:
        category = row.get("product_category") or "Unknown"
        by_category.setdefault(category, []).append(row)

    results: list[dict[str, Any]] = []
    def _rl_get(v: str) -> dict:
        return _get_vendor_reasoning(v, synthesis_views=synthesis_views, reasoning_lookup=reasoning_lookup)
    for category, rows in by_category.items():
        ranked = sorted(
            rows,
            key=lambda r: (
                -((r.get("churn_intent") or 0) / max((r.get("signal_reviews") or r.get("total_reviews") or 1), 1)),
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
                if competitor != highest_vendor:
                    category_flows[competitor] = category_flows.get(competitor, 0) + int(flow.get("mention_count") or 0)
        emerging = max(category_flows.items(), key=lambda item: item[1])[0] if category_flows else "Insufficient data"

        total_reviews = int(top_vendor.get("total_reviews") or 0)
        signal_reviews = int(top_vendor.get("signal_reviews") or 0) or total_reviews
        churn_density = round((int(top_vendor.get("churn_intent") or 0) * 100.0 / signal_reviews), 1) if signal_reviews else 0.0

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
            _v_arch = _rl_get(v).get("archetype")
            vr_score = _compute_churn_pressure_score(
                churn_density=cd, avg_urgency=urg, dm_churn_rate=dm_rate,
                displacement_mention_count=disp_m, price_complaint_rate=pr_rate,
                total_reviews=rev,
                archetype=_v_arch,
            )
            # Use reasoning risk_level when available (Gap #26)
            _rc_risk = _rl_get(v).get("risk_level", "")
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
                        "quote": str(q.get("quote") or q.get("text") or "")[:200],
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
            _rc = _rl_get(v)
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
                category, highest_vendor, churn_density, total_reviews, emerging,
                reasoning_lookup or {},
                synthesis_views=synthesis_views,
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


def _get_battle_card_reasoning_state(
    vendor: str,
    *,
    synthesis_views: dict[str, Any] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Unified reasoning state for battle card builders.

    Resolves vendor reasoning through synthesis_views (preferred) or
    reasoning_lookup (fallback).  No card mutation here.
    """
    rc = _get_vendor_reasoning(
        vendor, synthesis_views=synthesis_views, reasoning_lookup=reasoning_lookup,
    )
    archetype = rc.get("archetype", "")
    confidence = rc.get("confidence", 0)
    risk_level = rc.get("risk_level", "")
    key_signals = rc.get("key_signals", [])
    falsification = rc.get("falsification_conditions", [])
    uncertainty = rc.get("uncertainty_sources", [])
    mode = rc.get("mode", "")
    executive_summary = rc.get("executive_summary", "")

    # Determine reasoning source
    reasoning_source = "synthesis" if mode == "synthesis" else (
        "synthesis_fallback" if mode == "synthesis_fallback" else "legacy"
    )

    # Confident reasoning: high-confidence archetype from unified helper,
    # or confident synthesis view (wedge + medium/high confidence)
    has_confident = bool(archetype) and confidence >= 0.7
    if not has_confident and synthesis_views:
        view = synthesis_views.get(vendor)
        if view is None and vendor:
            canon = vendor.strip().lower()
            for vn, v in synthesis_views.items():
                if vn.strip().lower() == canon:
                    view = v
                    break
        if view is not None:
            w = getattr(view, "primary_wedge", None)
            c = view.confidence("causal_narrative") if hasattr(view, "confidence") else ""
            has_confident = bool(w) and c in ("medium", "high")

    return {
        "archetype": archetype,
        "confidence": confidence,
        "risk_level": risk_level,
        "key_signals": key_signals,
        "falsification_conditions": falsification,
        "uncertainty_sources": uncertainty,
        "executive_summary": executive_summary,
        "reasoning_source": reasoning_source,
        "reasoning_mode": mode,
        "has_confident_reasoning": has_confident,
    }


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
    synthesis_views: dict[str, Any] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
    timeline_lookup: dict[str, list[dict]] | None = None,
    use_case_lookup: dict[str, list[dict]] | None = None,
    positive_lookup: dict[str, list[dict]] | None = None,
    department_lookup: dict[str, list[dict]] | None = None,
    usage_duration_lookup: dict[str, list[dict]] | None = None,
    buyer_auth_lookup: dict[str, dict] | None = None,
    keyword_spike_lookup: dict[str, dict] | None = None,
    evidence_vault_lookup: dict[str, dict[str, Any]] | None = None,
    synthesis_requested_as_of: date | None = None,
    category_dynamics_lookup: dict[str, dict[str, Any]] | None = None,
    account_intel_lookup: dict[str, dict[str, Any]] | None = None,
    limit: int | None = None,
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
                "signal_reviews": int(row.get("signal_reviews") or 0),
                "churn_intent": churn,
                "avg_urgency": urgency,
                "category": category,
                "vendor_score": row,
            }

    cards: list[dict[str, Any]] = []
    for vendor, m in merged.items():
        vendor_vault = (evidence_vault_lookup or {}).get(vendor, {})
        total_reviews = m["total_reviews"]
        signal_reviews = int(m.get("signal_reviews") or 0) or total_reviews
        churn_intent = m["churn_intent"]
        churn_density = round(churn_intent * 100.0 / signal_reviews, 1) if signal_reviews else 0.0
        avg_urgency = round(m["avg_urgency"], 1)
        dm_rate = float(dm_lookup.get(vendor, 0))
        price_rate = float(price_lookup.get(vendor, 0))
        acct_intel = (account_intel_lookup or {}).get(vendor, {})
        acct_summary = acct_intel.get("summary") or {} if isinstance(acct_intel, dict) else {}
        acct_high_intent = int(acct_summary.get("high_intent_count") or 0)
        acct_active_eval = int(acct_summary.get("active_eval_signal_count") or 0)
        account_pressure_override = acct_high_intent >= 3 or acct_active_eval >= 2

        # Resolve reasoning state once per vendor
        reasoning_state = _get_battle_card_reasoning_state(
            vendor,
            synthesis_views=synthesis_views,
            reasoning_lookup=reasoning_lookup,
        )

        # Qualification gate -- reasoning can lower thresholds
        _has_reasoning = reasoning_state["has_confident_reasoning"]
        _density_gate = 10 if _has_reasoning else 15
        _urgency_gate = 2.5 if _has_reasoning else 3.0
        _dm_gate = 0.2 if _has_reasoning else 0.3
        if (
            churn_density < _density_gate
            and avg_urgency < _urgency_gate
            and dm_rate < _dm_gate
            and not account_pressure_override
        ):
            continue

        # Confidence label
        if total_reviews >= 50:
            confidence = "high"
        elif total_reviews >= 20:
            confidence = "medium"
        else:
            confidence = "low"
        if reasoning_state["confidence"] >= 0.8 and confidence == "medium":
            confidence = "high"

        comp_entries = competitor_lookup.get(vendor, [])
        displacement_mentions = sum(c.get("mentions", 0) for c in comp_entries)

        _vendor_archetype = reasoning_state["archetype"]
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
        weaknesses = _battle_card_weaknesses_from_evidence_vault(vendor_vault, limit=5)
        if not weaknesses:
            weaknesses = []
            seen_areas: set[str] = set()

            profile = product_profile_lookup.get(vendor, {})
            for w in (profile.get("weaknesses") or []):
                if not isinstance(w, dict):
                    continue
                area = w.get("area", "")
                if area and area not in seen_areas:
                    seen_areas.add(area)
                    evidence_count = int(w.get("evidence_count") or 0)
                    weaknesses.append({
                        "area": area,
                        "score": w.get("score"),
                        "evidence_count": evidence_count,
                        "source": "product_profile",
                    })

            for p in pain_lookup.get(vendor, []):
                area = p.get("category", "")
                if area and area not in seen_areas:
                    seen_areas.add(area)
                    evidence_count = int(p.get("count") or 0)
                    weaknesses.append({
                        "area": area,
                        "count": evidence_count,
                        "evidence_count": evidence_count,
                        "source": "pain_category",
                    })

            for g in feature_gap_lookup.get(vendor, []):
                feature = g.get("feature", "")
                if feature and feature not in seen_areas:
                    seen_areas.add(feature)
                    evidence_count = int(g.get("mentions") or 0)
                    weaknesses.append({
                        "area": feature,
                        "count": evidence_count,
                        "evidence_count": evidence_count,
                        "source": "feature_gap",
                    })

            weaknesses.sort(
                key=lambda w: -int(w.get("evidence_count") or 0),
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
                quote_text = str(q.get("quote") or "")
                urgency = float(q.get("urgency") or 0)
                if not _quote_has_pain_signal(
                    quote_text,
                    urgency=urgency,
                    rating=q.get("rating"),
                    rating_max=q.get("rating_max"),
                ):
                    continue
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
                    "quote": quote_text,
                    "urgency": urgency,
                    "source_site": q.get("source_site", ""),
                    "company": q.get("company", ""),
                    "title": q.get("title", ""),
                    "company_size": q.get("company_size", ""),
                    "industry": q.get("industry", ""),
                    "rating": q.get("rating"),
                    "rating_max": q.get("rating_max"),
                })
            elif isinstance(q, str):
                if _quote_has_pain_signal(q):
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

        vault_snapshot = vendor_vault.get("metric_snapshot") if isinstance(vendor_vault, dict) else None
        vault_snapshot = vault_snapshot if isinstance(vault_snapshot, dict) else {}
        objection_data: dict[str, Any] = {
            "price_complaint_rate": round(price_rate, 3),
            "dm_churn_rate": round(dm_rate, 3),
            "sentiment_direction": sentiment_dir,
            "top_feature_gaps": top_gaps,
            "total_reviews": total_reviews,
            "churn_signal_density": churn_density,
            "avg_urgency": avg_urgency,
            "budget_context": budget_lookup.get(vendor, {}),
        }
        _rr = vault_snapshot.get("recommend_ratio")
        if _rr is not None:
            try:
                objection_data["recommend_ratio"] = round(float(_rr), 2)
            except (TypeError, ValueError):
                pass
        _pp = vault_snapshot.get("positive_review_pct")
        if _pp is not None:
            try:
                objection_data["positive_review_pct"] = round(float(_pp), 1)
            except (TypeError, ValueError):
                pass

        integrations = (product_profile_lookup.get(vendor, {}).get("top_integrations") or [])[:8]
        blocked_company_names = {
            normalize_company_name(vendor),
            *(
                normalize_company_name(str(item.get("competitor") or ""))
                for item in differentiators
            ),
            *(normalize_company_name(str(name)) for name in integrations),
        }

        # -- Section 5: High-intent companies --
        # Priority: canonical account intelligence > evidence vault > churning_companies
        canonical_accounts = (acct_intel.get("accounts") or []) if isinstance(acct_intel, dict) else []
        hi_companies = _normalize_canonical_accounts_for_battle_card(
            canonical_accounts,
            current_vendor=vendor,
            blocked_names=blocked_company_names,
            limit=5,
        ) if canonical_accounts else []
        if not hi_companies:
            has_vault_company_signals = isinstance(vendor_vault, dict) and "company_signals" in vendor_vault
            hi_companies = _battle_card_companies_from_evidence_vault(
                vendor_vault,
                current_vendor=vendor,
                blocked_names=blocked_company_names,
                limit=5,
            )
            if not hi_companies and not has_vault_company_signals:
                fallback_candidates = [
                    item for item in (company_lookup.get(vendor, []))
                    if _battle_card_company_is_display_safe(
                        item.get("company"),
                        current_vendor=vendor,
                        blocked_names=blocked_company_names,
                        role=item.get("role") or item.get("title"),
                        company_size=item.get("company_size"),
                        buying_stage=item.get("buying_stage"),
                    )
                ]
                hi_companies = _rank_high_intent_companies(fallback_candidates)[:5]

        # -- Section 6: Integration stack --

        # -- Section 7: Buyer authority summary --
        buyer_authority = (buyer_auth_lookup or {}).get(vendor, {})
        keyword_spikes = (keyword_spike_lookup or {}).get(vendor, {})

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
        vault_strengths = _battle_card_strengths_from_evidence_vault(vendor_vault, limit=5)
        if vault_strengths:
            card_entry["incumbent_strengths"] = vault_strengths
        if keyword_spikes.get("spike_count"):
            card_entry["keyword_spikes"] = {
                "spike_count": int(keyword_spikes.get("spike_count") or 0),
                "keywords": list(keyword_spikes.get("spike_keywords") or [])[:10],
                "trend_summary": keyword_spikes.get("trend_summary") or {},
            }
        tl_entries = (timeline_lookup or {}).get(vendor, [])
        eval_deadlines = _build_active_evaluation_deadlines(
            tl_entries,
            limit=5,
        )
        # Merge canonical account contract_end dates into deadlines
        # Use hi_companies (display-safe) not raw canonical_accounts
        if hi_companies and len(eval_deadlines) < 5:
            seen_companies = {(d.get("company") or "").lower() for d in eval_deadlines}
            for hc in hi_companies:
                if not isinstance(hc, dict) or len(eval_deadlines) >= 5:
                    break
                c_end = hc.get("contract_end")
                if not c_end:
                    continue
                c_name = hc.get("company") or ""
                if not c_name or c_name.lower() in seen_companies:
                    continue
                eval_deadlines.append({
                    "company": c_name,
                    "evaluation_deadline": None,
                    "contract_end": str(c_end),
                    "decision_timeline": None,
                    "urgency": float(hc.get("urgency") or 0),
                    "title": hc.get("role"),
                    "company_size": hc.get("company_size"),
                    "industry": hc.get("industry"),
                    "trigger_type": "contract_end",
                    "buying_stage": hc.get("buying_stage"),
                    "role": hc.get("role"),
                    "pain": hc.get("pain"),
                    "source": hc.get("source") or "account_intelligence",
                })
                seen_companies.add(c_name.lower())
            eval_deadlines.sort(
                key=lambda d: (-float(d.get("urgency") or 0),),
            )
        if eval_deadlines:
            card_entry["active_evaluation_deadlines"] = eval_deadlines[:5]
        # Derive account pressure summary from canonical account intelligence
        if canonical_accounts:
            acct_summary = acct_intel.get("summary") or {} if isinstance(acct_intel, dict) else {}
            hi_count = int(acct_summary.get("high_intent_count") or 0)
            ae_count = int(acct_summary.get("active_eval_signal_count") or 0)
            dm_ct = int(acct_summary.get("decision_maker_count") or 0)
            pparts: list[str] = []
            if hi_count:
                pparts.append(f"{hi_count} high-intent account{'s' if hi_count != 1 else ''}")
            if ae_count:
                pparts.append(f"{ae_count} active evaluation signal{'s' if ae_count != 1 else ''}")
            if dm_ct:
                pparts.append(f"{dm_ct} decision-maker signal{'s' if dm_ct != 1 else ''}")
            if pparts:
                card_entry["account_pressure_summary"] = "Detected: " + ", ".join(pparts) + "."
                card_entry["account_pressure_metrics"] = {
                    "high_intent_count": hi_count,
                    "active_eval_count": ae_count,
                    "decision_maker_count": dm_ct,
                    "total_accounts": int(acct_summary.get("total_accounts") or 0),
                }
            priority_names = [
                ca.get("company_name") or ca.get("name") or ""
                for ca in canonical_accounts
                if isinstance(ca, dict) and (
                    float(ca.get("urgency_score") or 0) >= 6.0
                    or ca.get("decision_maker")
                )
            ][:5]
            if priority_names:
                card_entry["priority_account_names"] = [n for n in priority_names if n]
        uc_entries = (use_case_lookup or {}).get(vendor, [])
        if uc_entries:
            card_entry["objection_data"]["product_depth"] = uc_entries[:5]
        departments = (department_lookup or {}).get(vendor, [])
        if departments:
            card_entry["objection_data"]["department_context"] = departments[:3]
        durations = (usage_duration_lookup or {}).get(vendor, [])
        if durations:
            card_entry["objection_data"]["tenure_churn_pattern"] = durations[:5]

        if reasoning_state.get("archetype"):
            card_entry["archetype"] = reasoning_state["archetype"]
            card_entry["archetype_confidence"] = reasoning_state["confidence"]
            card_entry["archetype_risk_level"] = reasoning_state["risk_level"]
            card_entry["archetype_key_signals"] = reasoning_state["key_signals"]
            card_entry["falsification_conditions"] = reasoning_state["falsification_conditions"]
            card_entry["uncertainty_sources"] = reasoning_state["uncertainty_sources"]
            card_entry["reasoning_source"] = reasoning_state["reasoning_source"]

        # Inject category dynamics pool data for downstream council resolution
        _cat_key = card_entry.get("category") or ""
        _cat_dyn = (category_dynamics_lookup or {}).get(_cat_key)
        if isinstance(_cat_dyn, dict):
            card_entry["category_dynamics"] = _cat_dyn

        vendor_evidence = _build_vendor_evidence(
            m.get("vendor_score") or {},
            pain_lookup=pain_lookup,
            competitor_lookup=competitor_lookup,
            feature_gap_lookup=feature_gap_lookup,
            insider_lookup={},
            keyword_spike_lookup=keyword_spike_lookup or {},
            dm_lookup=dm_lookup,
            price_lookup=price_lookup,
            quote_lookup=quote_lookup,
            budget_lookup=budget_lookup,
            buyer_auth_lookup=buyer_auth_lookup,
            use_case_lookup=use_case_lookup,
        )

        # Inject reasoning synthesis via typed reader contract
        _synth_view = (synthesis_views or {}).get(vendor)
        if _synth_view is None and vendor:
            # Canonicalized fallback
            canon = vendor.strip().lower()
            for vn, v in (synthesis_views or {}).items():
                if vn.strip().lower() == canon:
                    _synth_view = v
                    break
        if _synth_view is not None:
            from ._b2b_synthesis_reader import inject_synthesis_into_card
            inject_synthesis_into_card(
                card_entry,
                _synth_view,
                requested_as_of=synthesis_requested_as_of,
                vendor_evidence=vendor_evidence,
            )
        segment_playbook = _battle_card_segment_playbook(card_entry)
        if segment_playbook:
            card_entry["segment_playbook"] = segment_playbook
        timing_intelligence = _battle_card_timing_intelligence(card_entry)
        if timing_intelligence:
            card_entry["timing_intelligence"] = timing_intelligence
            timing_summary, timing_metrics, priority_triggers = (
                _timing_summary_payload(timing_intelligence)
            )
            if timing_summary:
                card_entry["timing_summary"] = timing_summary
            if timing_metrics:
                card_entry["timing_metrics"] = timing_metrics
            if priority_triggers:
                card_entry["priority_timing_triggers"] = priority_triggers
        if segment_playbook:
            targeting_summary = _segment_targeting_summary(
                segment_playbook,
                timing_intelligence if timing_intelligence else None,
            )
            if targeting_summary:
                card_entry["segment_targeting_summary"] = targeting_summary
        cards.append(card_entry)

    def _bc_sort_key(x: dict) -> tuple:
        score = x["churn_pressure_score"]
        conf = x.get("archetype_confidence", 0)
        has_arch = bool(x.get("archetype"))
        reasoning_boost = min(conf * 5, 5.0) if has_arch else 0
        return (-(score + reasoning_boost),)
    cards.sort(key=_bc_sort_key)
    if limit is None:
        return cards
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
