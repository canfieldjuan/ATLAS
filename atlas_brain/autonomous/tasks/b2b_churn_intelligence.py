"""
B2B churn intelligence: aggregate enriched review data, persist canonical pools,
and hand off reasoning/narrative synthesis to follow-up tasks.

Runs weekly (default Sunday 9 PM). Its job is deterministic:
  - build and persist churn signals
  - persist evidence, segment, temporal, displacement, category, and account pools
  - write the core completion marker consumed by downstream synthesis/report tasks

Vendor reasoning and cross-vendor reasoning no longer run here. Those LLM stages
now live in ``b2b_reasoning_synthesis`` and the synthesis-first consumers that
follow it.
"""

import asyncio
import json
import logging
import re
import time
import uuid as _uuid
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
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
from ._execution_progress import _update_execution_progress

logger = logging.getLogger("atlas.autonomous.tasks.b2b_churn_intelligence")

_STAGE_LOADING_INPUTS = "loading_inputs"
_STAGE_REASONING = "reasoning"
_STAGE_PERSISTING_SIGNALS = "persisting_signals"
_STAGE_FINALIZING = "finalizing"
_PERSISTENCE_PHASES = (
    "displacement_edges",
    "churn_signals",
    "company_signals",
    "pain_points",
    "use_cases",
    "integrations",
    "buyer_profiles",
    "snapshots",
    "core_marker",
)


def _normalize_report_pain_category(category: Any) -> str:
    return _normalize_generic_pain_label(category)

# ---------------------------------------------------------------------------
# Shared helpers (extracted to _b2b_shared.py)
# ---------------------------------------------------------------------------
from ._b2b_shared import (  # noqa: E402
    _canonicalize_vendor,
    _canonicalize_competitor,
    _compute_churn_pressure_score,
    _compute_evidence_confidence,
    _safe_json,
    _intelligence_source_allowlist,
    _eligible_review_filters,
    _normalize_generic_pain_label,
    _build_deterministic_vendor_feed,
    _build_deterministic_displacement_map,
    _build_exploratory_payload,
    _build_pain_lookup,
    _build_competitor_lookup,
    _build_feature_gap_lookup,
    _build_use_case_lookup,
    _build_integration_lookup,
    _build_sentiment_lookup,
    _build_buyer_auth_lookup,
    _build_role_churn_lookup,
    _fetch_role_churn_summary,
    _build_lock_in_lookup,
    _build_timeline_lookup,
    _build_keyword_spike_lookup,
    _build_complaint_lookup,
    _build_positive_lookup,
    _build_department_lookup,
    _build_contract_value_lookup,
    _build_usage_duration_lookup,
    _build_tenure_lookup,
    _build_turning_point_lookup,
    _build_insider_lookup,
    _build_evidence_vault_pass2_rollups,
    _build_company_signal_blocked_names_by_vendor,
    _company_signal_name_is_eligible,
    _battle_card_competitor_is_eligible,
    _merge_canonical_company_signals,
    build_evidence_vault,
    build_segment_intelligence,
    build_temporal_intelligence,
    build_displacement_dynamics,
    build_category_dynamics,
    build_account_intelligence,
    _aggregate_competitive_disp,
    _fetch_data_context,
    _fetch_vendor_provenance,
    _fetch_vendor_churn_scores,
    _fetch_vendor_churn_scores_from_signals,
    _sync_vendor_firmographics,
    _fetch_high_intent_companies,
    _fetch_existing_company_signals,
    _fetch_competitive_displacement_source_of_truth,
    _fetch_displacement_provenance,
    _fetch_pain_provenance,
    _fetch_use_case_provenance,
    _fetch_integration_provenance,
    _fetch_buyer_profile_provenance,
    _fetch_pain_distribution,
    _fetch_feature_gaps,
    _fetch_negative_review_counts,
    _fetch_price_complaint_rates,
    _fetch_dm_churn_rates,
    _fetch_churning_companies,
    _fetch_quotable_evidence,
    _fetch_evidence_vault_review_rows,
    _fetch_insider_aggregates,
    _fetch_product_profiles,
    _fetch_budget_signals,
    _fetch_use_case_distribution,
    _fetch_sentiment_trajectory,
    _fetch_sentiment_tenure,
    _fetch_turning_points,
    _fetch_review_text_aggregates,
    _fetch_department_distribution,
    _fetch_company_size_distribution,
    _fetch_contract_context_distribution,
    _fetch_buyer_authority_summary,
    _fetch_timeline_signals,
    _fetch_competitor_reasons,
    _fetch_prior_reports,
    _fetch_keyword_spikes,
    _build_validated_executive_summary,
    _executive_source_list,
)


class _TaskTimer:
    """Track elapsed wall-clock time for observability logging.

    For testing, pass ``_clock`` to inject a callable returning monotonic seconds.
    """

    __slots__ = ("_start", "_clock")

    def __init__(self, *, _clock=time.monotonic):
        self._clock = _clock
        self._start = _clock()

    def elapsed(self) -> float:
        return self._clock() - self._start


_GENERIC_PRODUCT_CATEGORIES = {"", "unknown", "b2b software"}


def _is_generic_product_category(category: str) -> bool:
    return str(category or "").strip().lower() in _GENERIC_PRODUCT_CATEGORIES


def _resolve_vendor_category(
    vendor_name: str,
    raw_category: str,
    preferred_categories: dict[str, str] | None = None,
) -> str:
    """Prefer a specific profile-backed category over generic vendor labels."""
    raw = str(raw_category or "").strip()
    preferred = str((preferred_categories or {}).get(_canonicalize_vendor(vendor_name or "")) or "").strip()
    if _is_generic_product_category(raw) and preferred:
        return preferred
    return raw or preferred


def _should_persist_category_dynamics(scoped_vendors: list[str] | None) -> bool:
    return not bool(scoped_vendors)


def _should_use_cross_vendor_category(category: str) -> bool:
    text = str(category or "").strip()
    return bool(text) and not _is_generic_product_category(text)


_PLACEHOLDER_VENDOR_REFS = {"vendor a", "vendor b", "vendor_a", "vendor_b"}


def _normalized_vendor_refs(vendors: list[str] | tuple[str, ...] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for vendor in vendors or []:
        text = str(vendor or "").strip()
        key = _canonicalize_vendor(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _is_placeholder_vendor_ref(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("-", "_")
    return text in _PLACEHOLDER_VENDOR_REFS


def _conclusion_mentions_vendor(value: Any, vendors: list[str]) -> bool:
    haystack = str(value or "").strip().lower()
    if not haystack:
        return False
    for vendor in vendors:
        token = str(vendor or "").strip().lower()
        if token and token in haystack:
            return True
    return False


def _cross_vendor_entry_quality(
    analysis_type: str,
    entry: dict[str, Any],
) -> float:
    conclusion = entry.get("conclusion") or {}
    score = 0.0
    if conclusion.get("conclusion"):
        score += 4.0
    score += min(float(entry.get("confidence") or 0.0), 1.0) * 2.0
    score += min(len(conclusion.get("key_insights") or []), 3) * 0.5
    if analysis_type == "pairwise_battle":
        if conclusion.get("winner"):
            score += 1.0
        if conclusion.get("loser"):
            score += 1.0
        if conclusion.get("durability_assessment"):
            score += 0.5
    elif analysis_type == "category_council":
        if conclusion.get("market_regime"):
            score += 1.5
        if conclusion.get("winner"):
            score += 0.5
        if conclusion.get("loser"):
            score += 0.5
    elif analysis_type == "resource_asymmetry":
        if conclusion.get("resource_advantage"):
            score += 1.5
        if conclusion.get("durability_assessment"):
            score += 0.5
    return score


def _normalize_cross_vendor_conclusion(
    raw: Any,
    *,
    analysis_type: str = "",
    vendors: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Coerce JSONB read-back into a normalized dict for downstream consumers."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    normalized = dict(raw)
    key_insights = raw.get("key_insights") or []
    if isinstance(key_insights, list):
        insights: list[dict[str, str]] = []
        for item in key_insights:
            if isinstance(item, str) and item:
                insights.append({"insight": item, "evidence": item})
            elif isinstance(item, dict):
                text = item.get("insight") or item.get("text") or ""
                evidence = item.get("evidence") or text
                if text:
                    insights.append({"insight": text, "evidence": evidence})
        normalized["key_insights"] = insights
    durability = normalized.get("durability_assessment") or raw.get("displacement_flows_nature")
    if isinstance(durability, str) and durability:
        normalized["durability_assessment"] = durability
    clean_vendors = _normalized_vendor_refs(vendors)
    vendor_keys = {_canonicalize_vendor(v) for v in clean_vendors}
    for field in ("winner", "loser"):
        value = str(normalized.get(field) or "").strip()
        if _is_placeholder_vendor_ref(value):
            normalized[field] = ""
            continue
        if value and vendor_keys and _canonicalize_vendor(value) not in vendor_keys:
            normalized[field] = ""
    category_default = normalized.get("category_default")
    if isinstance(category_default, dict):
        default_vendor = str(category_default.get("vendor") or "").strip()
        if (
            default_vendor
            and vendor_keys
            and _canonicalize_vendor(default_vendor) not in vendor_keys
        ):
            category_default["vendor"] = ""
    if analysis_type == "resource_asymmetry":
        resource_advantage = str(normalized.get("resource_advantage") or "").strip()
        if resource_advantage:
            if (
                ("vendor a" in resource_advantage.lower() or "vendor b" in resource_advantage.lower())
                and not _conclusion_mentions_vendor(resource_advantage, clean_vendors)
            ):
                normalized["resource_advantage"] = ""
    return normalized


def _compact_vendor_churn_scores_for_llm(
    vendor_scores: list[dict[str, Any]],
    *,
    council_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Build one compact tenant-LLM vendor row per vendor, preferring specific categories."""
    ordered_vendors: list[str] = []
    best_rows: dict[str, tuple[tuple[int, int, float], dict[str, Any]]] = {}

    for row in vendor_scores:
        if not isinstance(row, dict):
            continue
        vendor_name = str(row.get("vendor_name") or row.get("vendor") or "").strip()
        if not vendor_name:
            continue
        vendor_key = _canonicalize_vendor(vendor_name)
        if not vendor_key:
            continue
        category = str(row.get("product_category") or row.get("category") or "").strip()
        category_key = category.lower()
        reviews = int(row.get("total_reviews") or 0)
        compact_row = {
            "vendor": vendor_name,
            "vendor_name": vendor_name,
            "category": category,
            "product_category": category,
            "reviews": reviews,
            "churn": row.get("churn_intent"),
            "urgency": round(float(row.get("avg_urgency") or 0), 1),
            "rating": round(row["avg_rating_normalized"], 2) if row.get("avg_rating_normalized") else None,
            "rec_yes": row.get("recommend_yes"),
            "rec_no": row.get("recommend_no"),
            "category_council": (council_lookup or {}).get((vendor_key, category_key)),
        }
        score = (
            0 if _is_generic_product_category(category) else 1,
            reviews,
            float(row.get("avg_urgency") or 0),
        )
        if vendor_key not in best_rows:
            ordered_vendors.append(vendor_key)
            best_rows[vendor_key] = (score, compact_row)
        elif score > best_rows[vendor_key][0]:
            best_rows[vendor_key] = (score, compact_row)

    return [best_rows[key][1] for key in ordered_vendors[:limit] if key in best_rows]


def _json_list_or_default(raw: Any) -> list[Any]:
    """Parse JSON-backed list columns while preserving already-decoded lists."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_profile_competitor_entry(item: Any) -> tuple[str, int] | None:
    """Parse product-profile competitor entries shaped as vendor/mentions or name/count."""
    if isinstance(item, dict):
        raw_name = item.get("vendor") or item.get("name") or ""
        raw_count = item.get("mentions")
        if raw_count is None:
            raw_count = item.get("count")
    elif isinstance(item, str):
        raw_name = item
        raw_count = 1
    else:
        return None
    name = _canonicalize_competitor(str(raw_name or "").strip())
    if not name:
        return None
    try:
        count = int(raw_count or 1)
    except (TypeError, ValueError):
        count = 1
    return name, max(count, 1)


def _switching_trigger_competitor_is_eligible(label: Any) -> bool:
    """Return True for seller-usable competitor labels in switching-trigger reports."""
    text = str(label or "").strip()
    if not text or not _battle_card_competitor_is_eligible(text):
        return False
    lower = text.lower()
    blocked_terms = (
        "homegrown",
        "utility",
        "tooling",
        "manual process",
    )
    return not any(term in lower for term in blocked_terms)


async def _fetch_vendor_vault_row(
    pool,
    vendor_name: str,
    window_days: int,
) -> tuple[dict[str, Any], int] | tuple[None, None]:
    """Fetch the exact or nearest evidence-vault row for a vendor/window."""
    row = await pool.fetchrow(
        """
        SELECT vault, analysis_window_days
        FROM b2b_evidence_vault
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY
            CASE WHEN analysis_window_days = $2 THEN 0 ELSE 1 END,
            ABS(analysis_window_days - $2),
            as_of_date DESC,
            created_at DESC
        LIMIT 1
        """,
        vendor_name,
        window_days,
    )
    if not row:
        return None, None
    vault = _safe_json(row["vault"], default={})
    if not isinstance(vault, dict):
        return None, None
    try:
        resolved_window = int(row["analysis_window_days"])
    except (TypeError, ValueError):
        resolved_window = window_days
    return vault, resolved_window


async def _fetch_company_signal_review_context(
    pool,
    review_ids: list[_uuid.UUID],
) -> dict[str, dict[str, Any]]:
    """Enrich pooled company signals with targeted review-context lookups by id."""
    if not review_ids:
        return {}
    rows = await pool.fetch(
        """
        SELECT id, vendor_name, product_category,
               enrichment->'competitors_mentioned' AS competitors_json,
               enrichment->'quotable_phrases' AS quotable_phrases,
               enrichment->'feature_gaps' AS feature_gaps,
               enrichment->'timeline'->>'evaluation_deadline' AS evaluation_deadline,
               enrichment->'timeline'->>'decision_timeline' AS decision_timeline,
               enrichment->'contract_context'->>'contract_value_signal' AS contract_value_signal,
               reviewer_title, company_size_raw,
               COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
        FROM b2b_reviews
        WHERE id = ANY($1::uuid[])
          AND enrichment_status = 'enriched'
        """,
        review_ids,
    )
    context: dict[str, dict[str, Any]] = {}
    for row in rows:
        review_id = row.get("id")
        if not review_id:
            continue
        context[str(review_id)] = {
            "vendor_name": row.get("vendor_name"),
            "product_category": row.get("product_category"),
            "competitors_json": _safe_json(row.get("competitors_json"), default=[]),
            "quotable_phrases": _safe_json(row.get("quotable_phrases"), default=[]),
            "feature_gaps": _safe_json(row.get("feature_gaps"), default=[]),
            "evaluation_deadline": row.get("evaluation_deadline"),
            "decision_timeline": row.get("decision_timeline"),
            "contract_value_signal": row.get("contract_value_signal"),
            "reviewer_title": row.get("reviewer_title"),
            "company_size_raw": row.get("company_size_raw"),
            "industry": row.get("industry"),
        }
    return context


def _normalize_test_vendors(raw: Any) -> list[str]:
    """Normalize optional runtime vendor scope from task metadata."""
    if not raw:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(part or "").strip() for part in raw]
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        canon = _canonicalize_vendor(value) or value
        key = canon.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(canon)
    return normalized


def _count_analyzed_vendors(vendor_scores: Any) -> int:
    """Count unique canonical vendors from vendor-score style rows."""
    if not isinstance(vendor_scores, list):
        return 0
    seen: set[str] = set()
    for row in vendor_scores:
        if not isinstance(row, dict):
            continue
        vendor_name = row.get("vendor_name") or row.get("vendor") or ""
        canon = _canonicalize_vendor(vendor_name)
        if canon:
            seen.add(canon.strip().lower())
    return len(seen)


def _vendor_scope_contains(value: Any, vendor_scope: set[str]) -> bool:
    """Return whether a vendor-ish value falls inside the normalized scope."""
    text = str(value or "").strip()
    if not text:
        return False
    canon = _canonicalize_vendor(text) or text
    return canon.strip().lower() in vendor_scope


def _filter_rows_for_vendor_scope(
    rows: Any,
    vendor_names: list[str],
    *,
    vendor_fields: tuple[str, ...] = ("vendor", "vendor_name"),
) -> Any:
    """Filter list-of-dict fetcher rows to a vendor scope."""
    if not isinstance(rows, list):
        return rows
    vendor_scope = {v.lower() for v in vendor_names}
    if not vendor_scope:
        return rows

    filtered: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if any(_vendor_scope_contains(row.get(field), vendor_scope) for field in vendor_fields):
            filtered.append(row)
            continue
        nested = row.get("data")
        if isinstance(nested, list):
            inner = [
                item
                for item in nested
                if isinstance(item, dict)
                and any(_vendor_scope_contains(item.get(field), vendor_scope) for field in vendor_fields)
            ]
            if inner:
                filtered.append({**row, "data": inner})
    return filtered


def _filter_mapping_for_vendor_scope(data: Any, vendor_names: list[str]) -> Any:
    """Filter vendor-keyed or pair-keyed mappings to a vendor scope."""
    if not isinstance(data, dict):
        return data
    vendor_scope = {v.lower() for v in vendor_names}
    if not vendor_scope:
        return data

    filtered: dict[Any, Any] = {}
    for key, value in data.items():
        if isinstance(key, tuple):
            if any(_vendor_scope_contains(part, vendor_scope) for part in key if isinstance(part, str)):
                filtered[key] = value
            continue
        if _vendor_scope_contains(key, vendor_scope):
            filtered[key] = value
    return filtered


def _apply_vendor_scope_to_churn_inputs(
    data: dict[str, Any],
    vendor_names: list[str] | str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply a runtime vendor scope across churn-intelligence source bundles."""
    scoped_vendors = _normalize_test_vendors(vendor_names)
    if not scoped_vendors:
        return data, []

    scoped = dict(data)
    list_key_fields = {
        "vendor_scores": ("vendor_name", "vendor"),
        "vendor_scores_from_signals": ("vendor_name", "vendor"),
        "high_intent": ("vendor", "vendor_name"),
        "competitive_disp": ("vendor", "vendor_name", "from_vendor"),
        "pain_dist": ("vendor", "vendor_name"),
        "feature_gaps": ("vendor", "vendor_name"),
        "negative_counts": ("vendor", "vendor_name"),
        "price_rates": ("vendor", "vendor_name"),
        "dm_rates": ("vendor", "vendor_name"),
        "churning_companies": ("vendor", "vendor_name"),
        "quotable_evidence": ("vendor", "vendor_name"),
        "evidence_vault_review_rows": ("vendor", "vendor_name"),
        "budget_signals": ("vendor", "vendor_name"),
        "use_case_dist": ("vendor", "vendor_name"),
        "sentiment_traj": ("vendor", "vendor_name"),
        "buyer_auth": ("vendor", "vendor_name"),
        "timeline_signals": ("vendor", "vendor_name"),
        "competitor_reasons": ("vendor", "vendor_name"),
        "keyword_spikes": ("vendor", "vendor_name"),
        "insider_aggregates_raw": ("vendor_name", "vendor"),
        "product_profiles_raw": ("vendor_name", "vendor"),
        "department_dist": ("vendor_name", "vendor"),
        "sentiment_tenure_raw": ("vendor_name", "vendor"),
        "turning_points_raw": ("vendor_name", "vendor"),
    }
    for key, vendor_fields in list_key_fields.items():
        if key in scoped:
            scoped[key] = _filter_rows_for_vendor_scope(
                scoped.get(key),
                scoped_vendors,
                vendor_fields=vendor_fields,
            )

    for key in (
        "existing_company_signals",
        "vendor_provenance",
        "displacement_provenance",
        "pain_provenance",
        "use_case_provenance",
        "integration_provenance",
        "buyer_profile_provenance",
    ):
        if key in scoped:
            scoped[key] = _filter_mapping_for_vendor_scope(
                scoped.get(key),
                scoped_vendors,
            )

    review_text_aggs = scoped.get("review_text_aggs")
    if isinstance(review_text_aggs, tuple) and len(review_text_aggs) == 2:
        scoped["review_text_aggs"] = tuple(
            _filter_rows_for_vendor_scope(rows, scoped_vendors, vendor_fields=("vendor_name", "vendor"))
            for rows in review_text_aggs
        )

    contract_ctx_aggs = scoped.get("contract_ctx_aggs")
    if isinstance(contract_ctx_aggs, tuple) and len(contract_ctx_aggs) == 2:
        scoped["contract_ctx_aggs"] = tuple(
            _filter_rows_for_vendor_scope(rows, scoped_vendors, vendor_fields=("vendor_name", "vendor"))
            for rows in contract_ctx_aggs
        )

    return scoped, scoped_vendors

async def _persist_vendor_snapshots(
    pool,
    vendor_scores: list[dict[str, Any]],
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    high_intent: list[dict[str, Any]],
    today: date,
    price_lookup: dict[str, float] | None = None,
    dm_lookup: dict[str, float] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
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

    # Fetch displacement velocity per vendor from latest edges (leading indicator)
    vel_rows = await pool.fetch(
        """
        SELECT from_vendor, SUM(COALESCE(velocity_7d, 0)) AS total_velocity
        FROM b2b_displacement_edges
        WHERE computed_date = (SELECT MAX(computed_date) FROM b2b_displacement_edges)
        GROUP BY from_vendor
        """,
    )
    velocity_lookup: dict[str, float] = {r["from_vendor"]: float(r["total_velocity"]) for r in vel_rows}

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
        recommend_denom = recommend_yes + recommend_no
        recommend_ratio = round(((recommend_yes - recommend_no) / recommend_denom) * 100, 1) if recommend_denom else 0.0

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
            displacement_velocity=velocity_lookup.get(vendor),
        )

        _rl = (reasoning_lookup or {}).get(vendor, {})
        _archetype = _rl.get("archetype")
        _arch_conf = _rl.get("confidence")
        
        # New metrics for deeper reasoning
        _support_sent = round(float(row.get("support_sentiment") or 0.0), 2)
        _legacy_supp = round(float(row.get("legacy_support_score") or 0.0), 2)
        _new_feat = round(float(row.get("new_feature_velocity") or 0.0), 2)
        _emp_growth = float(row.get("employee_growth_rate")) if row.get("employee_growth_rate") is not None else None

        try:
            await pool.execute(
                """
                INSERT INTO b2b_vendor_snapshots (
                    vendor_name, snapshot_date, total_reviews, churn_intent,
                    churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                    top_pain, top_competitor, pain_count, competitor_count,
                    displacement_edge_count, high_intent_company_count,
                    pressure_score, dm_churn_rate, price_complaint_rate,
                    archetype, archetype_confidence,
                    support_sentiment, legacy_support_score, new_feature_velocity, employee_growth_rate
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                          $15, $16, $17, $18, $19, $20, $21, $22, $23)
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
                    price_complaint_rate = EXCLUDED.price_complaint_rate,
                    archetype = EXCLUDED.archetype,
                    archetype_confidence = EXCLUDED.archetype_confidence,
                    support_sentiment = EXCLUDED.support_sentiment,
                    legacy_support_score = EXCLUDED.legacy_support_score,
                    new_feature_velocity = EXCLUDED.new_feature_velocity,
                    employee_growth_rate = EXCLUDED.employee_growth_rate
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
                _archetype, _arch_conf,
                _support_sent, _legacy_supp, _new_feat, _emp_growth,
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


async def reconstruct_reasoning_lookup(
    pool,
    as_of: date | None = None,
) -> dict[str, dict]:
    """Rebuild reasoning_lookup from persisted b2b_churn_signals.

    Returns the legacy archetype-style dict shape expected by downstream
    compatibility readers, rebuilt from persisted churn-signal fields.

    When *as_of* is None, uses the most recent last_computed_at watermark.
    """
    if as_of is None:
        watermark = await pool.fetchval(
            "SELECT MAX(last_computed_at)::date FROM b2b_churn_signals"
        )
        if not watermark:
            return {}
        as_of = watermark

    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name,
               archetype, archetype_confidence, reasoning_mode,
               falsification_conditions,
               reasoning_risk_level, reasoning_executive_summary,
               reasoning_key_signals, reasoning_uncertainty_sources
        FROM b2b_churn_signals
        WHERE last_computed_at::date >= $1
          AND archetype IS NOT NULL
        ORDER BY vendor_name, archetype_confidence DESC NULLS LAST, last_computed_at DESC
        """,
        as_of,
    )
    lookup: dict[str, dict] = {}
    for r in rows:
        lookup[r["vendor_name"]] = {
            "archetype": r["archetype"],
            "confidence": float(r["archetype_confidence"]) if r["archetype_confidence"] else 0,
            "mode": r["reasoning_mode"] or "",
            "risk_level": r["reasoning_risk_level"] or "",
            "executive_summary": r["reasoning_executive_summary"] or "",
            "key_signals": _json_list_or_default(r["reasoning_key_signals"]),
            "falsification_conditions": _json_list_or_default(r["falsification_conditions"]),
            "uncertainty_sources": _json_list_or_default(r["reasoning_uncertainty_sources"]),
        }

    # Fallback: for vendors absent from churn_signals (or missing archetype),
    # pull synthesis_wedge from b2b_reasoning_synthesis (schema v2.2+).
    # This keeps the two tasks in sync even when synthesis runs before intelligence.
    synth_rows = await pool.fetch(
        """
        SELECT DISTINCT ON (vendor_name)
               vendor_name,
               synthesis->>'synthesis_wedge'       AS synthesis_wedge,
               synthesis->>'synthesis_wedge_label' AS synthesis_wedge_label
        FROM b2b_reasoning_synthesis
        WHERE as_of_date >= $1
          AND synthesis->>'synthesis_wedge' IS NOT NULL
          AND synthesis->>'synthesis_wedge' != ''
        ORDER BY vendor_name, as_of_date DESC
        """,
        as_of,
    )
    for r in synth_rows:
        vname = r["vendor_name"]
        if vname not in lookup:
            lookup[vname] = {
                "archetype": r["synthesis_wedge"],
                "confidence": 0.0,
                "mode": "synthesis_fallback",
                "risk_level": "",
                "executive_summary": "",
                "key_signals": [],
                "falsification_conditions": [],
                "uncertainty_sources": [],
            }

    return lookup


async def reconstruct_evidence_volatility(
    pool,
    days: int = 14,
) -> dict[str, dict]:
    """Read evidence volatility per vendor from persisted diffs.

    Returns vendor -> {avg_diff_ratio, max_diff_ratio, core_contradictions,
    days_tracked, latest_decision, latest_contradicted_fields,
    latest_component_scores}.
    Used by vulnerability_report and accounts_in_motion for instability signals.
    """
    rows = await pool.fetch(
        """
        SELECT vendor_name,
               AVG(weighted_diff_ratio) FILTER (WHERE compared) AS avg_diff,
               MAX(weighted_diff_ratio) FILTER (WHERE compared) AS max_diff,
               SUM(CASE WHEN has_core_contradiction AND compared THEN 1 ELSE 0 END) AS core_contradictions,
               COUNT(*) FILTER (WHERE compared) AS days_compared,
               COUNT(*) AS days_tracked,
               (ARRAY_AGG(decision ORDER BY computed_date DESC))[1] AS latest_decision,
               (ARRAY_AGG(component_scores ORDER BY computed_date DESC NULLS LAST)
                FILTER (WHERE compared))[1] AS latest_component_scores,
               (ARRAY_AGG(contradicted_fields ORDER BY computed_date DESC NULLS LAST)
                FILTER (WHERE compared))[1] AS latest_contradicted
        FROM reasoning_evidence_diffs
        WHERE computed_date >= CURRENT_DATE - $1::int
        GROUP BY vendor_name
        ORDER BY AVG(weighted_diff_ratio) FILTER (WHERE compared) DESC NULLS LAST
        """,
        days,
    )
    return {
        r["vendor_name"]: {
            "avg_diff_ratio": round(float(r["avg_diff"] or 0), 4),
            "max_diff_ratio": round(float(r["max_diff"] or 0), 4),
            "core_contradictions": r["core_contradictions"] or 0,
            "days_compared": r["days_compared"] or 0,
            "days_tracked": r["days_tracked"] or 0,
            "latest_decision": r["latest_decision"] or "",
            "latest_component_scores": r["latest_component_scores"] or {},
            "latest_contradicted_fields": r["latest_contradicted"] or [],
        }
        for r in rows
    }


async def reconstruct_cross_vendor_lookup(
    pool,
    as_of: date | None = None,
) -> dict[str, dict]:
    """Rebuild cross-vendor conclusions from b2b_cross_vendor_conclusions.

    Returns a dict with three keys:
      - ``battles``:     {(vendor_a, vendor_b): conclusion_dict, ...}
      - ``councils``:    {category: conclusion_dict, ...}
      - ``asymmetries``: {(vendor_a, vendor_b): conclusion_dict, ...}

    Keyed by sorted vendor tuple (battles/asymmetries) or category name
    (councils).  When *as_of* is None the most recent computed_date is used.
    """
    if as_of is None:
        watermark = await pool.fetchval(
            "SELECT MAX(computed_date) FROM b2b_cross_vendor_conclusions"
        )
        if not watermark:
            return {"battles": {}, "councils": {}, "asymmetries": {}}
        as_of = watermark

    rows = await pool.fetch(
        """
        SELECT analysis_type, vendors, category, conclusion, confidence, computed_date
        FROM b2b_cross_vendor_conclusions
        WHERE computed_date <= $1
        ORDER BY computed_date DESC, confidence DESC
        """,
        as_of,
    )

    battles: dict[tuple[str, ...], dict] = {}
    councils: dict[str, dict] = {}
    asymmetries: dict[tuple[str, ...], dict] = {}

    for r in rows:
        vendors = _normalized_vendor_refs(list(r["vendors"]) if r["vendors"] else [])
        atype = r["analysis_type"]
        if atype in {"pairwise_battle", "resource_asymmetry"} and len(vendors) < 2:
            continue
        conclusion = _normalize_cross_vendor_conclusion(
            r["conclusion"],
            analysis_type=atype,
            vendors=vendors,
        )
        if atype == "resource_asymmetry" and not (
            conclusion.get("conclusion") or conclusion.get("resource_advantage")
        ):
            continue
        entry = {
            "conclusion": conclusion,
            "confidence": float(r["confidence"]) if r["confidence"] is not None else 0,
            "vendors": vendors,
            "category": r["category"],
            "computed_date": r.get("computed_date") if hasattr(r, "get") else r["computed_date"],
        }
        key = tuple(sorted(vendors))
        if atype == "pairwise_battle":
            existing = battles.get(key)
            if existing is None:
                battles[key] = entry
            elif (
                existing.get("computed_date") == entry.get("computed_date")
                and _cross_vendor_entry_quality(atype, entry)
                > _cross_vendor_entry_quality(atype, existing)
            ):
                battles[key] = entry
        elif atype == "category_council":
            cat = r["category"] or ""
            if cat and not _should_use_cross_vendor_category(cat):
                continue
            existing = councils.get(cat)
            if existing is None:
                councils[cat] = entry
            elif (
                existing.get("computed_date") == entry.get("computed_date")
                and _cross_vendor_entry_quality(atype, entry)
                > _cross_vendor_entry_quality(atype, existing)
            ):
                councils[cat] = entry
        elif atype == "resource_asymmetry":
            existing = asymmetries.get(key)
            if existing is None:
                asymmetries[key] = entry
            elif (
                existing.get("computed_date") == entry.get("computed_date")
                and _cross_vendor_entry_quality(atype, entry)
                > _cross_vendor_entry_quality(atype, existing)
            ):
                asymmetries[key] = entry

    return {"battles": battles, "councils": councils, "asymmetries": asymmetries}


async def _detect_change_events(
    pool,
    vendor_scores: list[dict[str, Any]],
    pain_lookup: dict[str, list[dict]],
    competitor_lookup: dict[str, list[dict]],
    today: date,
    price_lookup: dict[str, float] | None = None,
    dm_lookup: dict[str, float] | None = None,
    temporal_lookup: dict[str, dict] | None = None,
    reasoning_lookup: dict[str, dict] | None = None,
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
        recommend_denom = recommend_yes + recommend_no
        recommend_ratio = round(((recommend_yes - recommend_no) / recommend_denom) * 100, 1) if recommend_denom else 0.0

        pains = pain_lookup.get(vendor, [])
        top_pain = (pains[0] if pains else {}).get("category")
        comps = competitor_lookup.get(vendor, [])
        top_competitor = comps[0]["name"] if comps else None

        # (event_type, description, old_val, new_val, delta, z_score)
        events: list[tuple[str, str, float | None, float | None, float | None, float | None]] = []

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
            events.append(("urgency_spike", f"Avg urgency rose from {prior_urg} to {avg_urgency} (z={z:.1f})", prior_urg, avg_urgency, urg_delta, z))
        elif urg_anomaly.get("is_anomaly") and urg_anomaly.get("z_score", 0) < 0:
            z = urg_anomaly["z_score"]
            events.append(("urgency_drop", f"Avg urgency fell from {prior_urg} to {avg_urgency} (z={z:.1f})", prior_urg, avg_urgency, urg_delta, z))
        elif urg_delta >= 1.0:
            events.append(("urgency_spike", f"Avg urgency rose from {prior_urg} to {avg_urgency}", prior_urg, avg_urgency, urg_delta, None))
        elif urg_delta <= -1.0:
            events.append(("urgency_drop", f"Avg urgency fell from {prior_urg} to {avg_urgency}", prior_urg, avg_urgency, urg_delta, None))

        # Churn density spike -- prefer z-score anomaly when available
        prior_cd = float(prior["churn_density"] or 0)
        cd_delta = churn_density - prior_cd
        cd_anomaly = anomalies_by_metric.get("churn_density", {})
        if cd_anomaly.get("is_anomaly") and cd_anomaly.get("z_score", 0) > 0:
            z = cd_anomaly["z_score"]
            events.append(("churn_density_spike", f"Churn density rose from {prior_cd}% to {churn_density}% (z={z:.1f})", prior_cd, churn_density, cd_delta, z))
        elif cd_delta >= 5.0:
            events.append(("churn_density_spike", f"Churn density rose from {prior_cd}% to {churn_density}%", prior_cd, churn_density, cd_delta, None))

        # Velocity acceleration event: metric has both high velocity AND positive acceleration
        for metric_key in ("churn_density", "avg_urgency", "pressure_score"):
            vel = td.get(f"velocity_{metric_key}")
            accel = td.get(f"accel_{metric_key}")
            if vel is not None and accel is not None and vel > 0 and accel > 0:
                events.append((
                    "velocity_acceleration",
                    f"{metric_key} accelerating: velocity={vel:.3f}/day, acceleration={accel:.3f}/day^2",
                    vel, accel, None, None,
                ))

        # NPS / recommend ratio shift -- prefer z-score when available
        prior_rr = float(prior["recommend_ratio"] or 0)
        rr_delta = recommend_ratio - prior_rr
        rr_anomaly = anomalies_by_metric.get("recommend_ratio", {})
        if rr_anomaly.get("is_anomaly"):
            z = rr_anomaly["z_score"]
            direction = "improved" if rr_delta > 0 else "declined"
            events.append(("nps_shift", f"Recommend ratio {direction} from {prior_rr} to {recommend_ratio} (z={z:.1f})", prior_rr, recommend_ratio, rr_delta, z))
        elif abs(rr_delta) >= 10.0:
            direction = "improved" if rr_delta > 0 else "declined"
            events.append(("nps_shift", f"Recommend ratio {direction} from {prior_rr} to {recommend_ratio}", prior_rr, recommend_ratio, rr_delta, None))

        # Review volume spike -- prefer z-score when available
        prior_tr = int(prior["total_reviews"] or 0)
        if prior_tr > 0:
            vol_pct = ((total_reviews - prior_tr) / prior_tr) * 100
            vol_anomaly = anomalies_by_metric.get("total_reviews", {})
            if vol_anomaly.get("is_anomaly") and vol_anomaly.get("z_score", 0) > 0:
                z = vol_anomaly["z_score"]
                events.append(("review_volume_spike", f"Review count jumped from {prior_tr} to {total_reviews} (+{vol_pct:.0f}%, z={z:.1f})", float(prior_tr), float(total_reviews), vol_pct, z))
            elif vol_pct >= 25.0:
                events.append(("review_volume_spike", f"Review count jumped from {prior_tr} to {total_reviews} (+{vol_pct:.0f}%)", float(prior_tr), float(total_reviews), vol_pct, None))

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
            events.append(("pressure_score_spike", f"Pressure score rose from {prior_ps} to {cur_ps} (z={z:.1f})", prior_ps, cur_ps, ps_delta, z))
        elif ps_delta >= 10.0:
            events.append(("pressure_score_spike", f"Pressure score rose from {prior_ps} to {cur_ps}", prior_ps, cur_ps, ps_delta, None))

        # Decision-maker churn rate spike -- prefer z-score when available
        prior_dm = float(prior["dm_churn_rate"] or 0)
        dm_delta = _dm_rate - prior_dm
        dm_anomaly = anomalies_by_metric.get("dm_churn_rate", {})
        if dm_anomaly.get("is_anomaly") and dm_anomaly.get("z_score", 0) > 0:
            z = dm_anomaly["z_score"]
            events.append(("dm_churn_spike", f"DM churn rate rose from {prior_dm:.2%} to {_dm_rate:.2%} (z={z:.1f})", prior_dm, _dm_rate, dm_delta, z))
        elif dm_delta >= 0.15:
            events.append(("dm_churn_spike", f"DM churn rate rose from {prior_dm:.2%} to {_dm_rate:.2%}", prior_dm, _dm_rate, dm_delta, None))

        # New pain category
        prior_pain = prior["top_pain"]
        if top_pain and prior_pain and top_pain != prior_pain:
            events.append(("new_pain_category", f"Top pain shifted from '{prior_pain}' to '{top_pain}'", None, None, None, None))

        # New competitor
        prior_comp = prior["top_competitor"]
        if top_competitor and prior_comp and top_competitor != prior_comp:
            events.append(("new_competitor", f"Top competitor shifted from '{prior_comp}' to '{top_competitor}'", None, None, None, None))

        # Archetype shift (requires vendor reasoning context when available)
        if reasoning_lookup:
            current_rc = reasoning_lookup.get(vendor, {})
            current_arch = current_rc.get("archetype")
            current_conf = current_rc.get("confidence", 0)
            prior_arch = prior.get("archetype")
            prior_arch_conf = float(prior["archetype_confidence"]) if prior.get("archetype_confidence") else None
            if current_arch and prior_arch and current_arch != prior_arch:
                conf_was = f"{prior_arch_conf:.2f}" if prior_arch_conf is not None else "?"
                events.append((
                    "archetype_shift",
                    f"Classification shifted from '{prior_arch}' to '{current_arch}' "
                    f"(confidence {conf_was} -> {current_conf:.2f})",
                    None, None, None, None,
                ))

        _SEVERITY_RANK = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
        _min_sev = settings.b2b_webhook.min_change_event_severity
        _min_rank = _SEVERITY_RANK.get(_min_sev, 1)

        webhook_events: list[tuple[str, dict]] = []
        for event_type, description, old_val, new_val, delta, z_score in events:
            # Derive severity from z_score magnitude
            severity = None
            if z_score is not None:
                az = abs(z_score)
                if az >= 3.0:
                    severity = "critical"
                elif az >= 2.0:
                    severity = "high"
                elif az >= 1.5:
                    severity = "moderate"
                else:
                    severity = "low"
            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_change_events
                        (vendor_name, event_date, event_type, description,
                         old_value, new_value, delta, z_score, severity)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    vendor, today, event_type, description,
                    old_val, new_val, delta, z_score, severity,
                )
                detected += 1
                # Severity gating: only dispatch webhooks for events at or above threshold.
                # Events without z_score (categorical: new_pain_category, archetype_shift)
                # have severity=None and always dispatch.
                if severity is not None and _SEVERITY_RANK.get(severity, 0) < _min_rank:
                    continue
                webhook_events.append((vendor, {
                    "event_type": event_type,
                    "vendor_name": vendor,
                    "description": description,
                    "old_value": old_val,
                    "new_value": new_val,
                    "delta": delta,
                    "z_score": z_score,
                    "severity": severity,
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


# SORAM channel alignment with archetype patterns
_ARCHETYPE_SORAM_WEIGHTS: dict[str, dict[str, float]] = {
    "pricing_shock":       {"operational": 0.4, "regulatory": 0.2, "media": 0.3},
    "feature_gap":         {"operational": 0.5, "media": 0.3},
    "support_collapse":    {"operational": 0.6, "media": 0.3},
    "leadership_redesign": {"alignment": 0.6, "media": 0.3},
    "acquisition_decay":   {"alignment": 0.5, "operational": 0.3, "media": 0.2},
    "integration_break":   {"operational": 0.6, "societal": 0.2},
    "category_disruption": {"societal": 0.3, "regulatory": 0.3, "media": 0.3},
    "compliance_gap":      {"regulatory": 0.6, "alignment": 0.2},
}


async def _correlate_articles_with_archetypes(
    pool,
    vendor_names: list[str],
    reasoning_lookup: dict[str, dict],
    today: date,
    window_days: int = 30,
) -> int:
    """Find articles mentioning tracked vendors and correlate with archetypes.

    For each vendor with a current archetype:
    1. Find articles from the last *window_days* whose entities_detected
       overlap the vendor name (case-insensitive array match).
    2. Score SORAM channel alignment between article and archetype.
    3. If an archetype_shift change event exists for this vendor today,
       link the article to that event (temporal correlation).
    4. Persist to b2b_article_correlations (deduplicated by article+vendor).
    """
    if not vendor_names or not reasoning_lookup:
        return 0

    correlated = 0

    # Fetch today's archetype_shift events for temporal linking
    shift_events = await pool.fetch(
        """
        SELECT id, vendor_name FROM b2b_change_events
        WHERE event_type = 'archetype_shift' AND event_date = $1
        """,
        today,
    )
    shift_by_vendor: dict[str, str] = {
        r["vendor_name"]: str(r["id"]) for r in shift_events
    }

    for vendor in vendor_names:
        rc = reasoning_lookup.get(vendor, {})
        archetype = rc.get("archetype")
        if not archetype or archetype in ("stable", "mixed"):
            continue
        confidence = rc.get("confidence", 0)

        # Find articles mentioning this vendor (entity match, case-insensitive)
        articles = await pool.fetch(
            """
            SELECT id, title, soram_channels, pressure_direction,
                   entities_detected, published_at
            FROM news_articles
            WHERE enrichment_status = 'classified'
              AND created_at > CURRENT_DATE - $2::int
              AND EXISTS (
                  SELECT 1 FROM unnest(entities_detected) e
                  WHERE LOWER(e) = LOWER($1)
              )
            ORDER BY created_at DESC
            LIMIT 10
            """,
            vendor,
            window_days,
        )

        if not articles:
            continue

        soram_weights = _ARCHETYPE_SORAM_WEIGHTS.get(archetype, {})
        change_event_id = shift_by_vendor.get(vendor)

        for article in articles:
            # Score SORAM alignment
            soram = article["soram_channels"] or {}
            if isinstance(soram, str):
                try:
                    soram = json.loads(soram)
                except (json.JSONDecodeError, TypeError):
                    soram = {}

            alignment_score = 0.0
            alignment_detail: dict[str, float] = {}
            for channel, weight in soram_weights.items():
                channel_val = float(soram.get(channel, 0))
                contribution = channel_val * weight
                alignment_score += contribution
                if contribution > 0:
                    alignment_detail[channel] = round(contribution, 3)

            # Relevance: 60% SORAM alignment + 40% entity match (always 1.0 if here)
            relevance = round(alignment_score * 0.6 + 0.4, 3)

            # Bump relevance if there's a temporal archetype shift today
            correlation_type = "entity_match"
            if change_event_id:
                correlation_type = "temporal_shift"
                relevance = min(relevance + 0.15, 1.0)
            elif alignment_score >= 0.3:
                correlation_type = "soram_alignment"

            try:
                await pool.execute(
                    """
                    INSERT INTO b2b_article_correlations
                        (article_id, vendor_name, correlation_type,
                         archetype, archetype_confidence,
                         change_event_id, soram_alignment,
                         relevance_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
                    ON CONFLICT DO NOTHING
                    """,
                    article["id"],
                    vendor,
                    correlation_type,
                    archetype,
                    confidence,
                    change_event_id,
                    json.dumps(alignment_detail) if alignment_detail else None,
                    relevance,
                )
                correlated += 1
            except Exception:
                logger.debug(
                    "Failed to persist article correlation for %s / %s",
                    vendor, article["id"],
                )

    if correlated:
        logger.info("Article-archetype correlations: %d persisted", correlated)

    return correlated


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: weekly B2B churn intelligence."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    budget = _TaskTimer()
    synced_firmographics = 0
    await _update_execution_progress(
        task,
        stage=_STAGE_LOADING_INPUTS,
        progress_message="Loading churn-intelligence source artifacts.",
    )

    # Warm vendor registry cache so sync resolve_vendor_name_cached() calls
    # throughout this function hit the DB-backed cache rather than bootstrap.
    await _warm_vendor_cache()

    try:
        synced_firmographics = await _sync_vendor_firmographics(pool, as_of=date.today())
        if synced_firmographics:
            logger.info("Vendor firmographics synced: %d vendors", synced_firmographics)
    except Exception:
        logger.debug("Vendor firmographics sync skipped", exc_info=True)

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
    _llm_workload = cfg.intelligence_llm_backend if cfg.intelligence_llm_backend in ("vllm", "anthropic", "auto") else "vllm"
    _ci_llm = _get_plm(workload=_llm_workload)
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
        vendor_scores, high_intent, existing_company_signals,
        competitive_disp,
        pain_dist, feature_gaps,
        negative_counts, price_rates, dm_rates,
        churning_companies, quotable_evidence, evidence_vault_review_rows,
        budget_signals, use_case_dist, sentiment_traj,
        buyer_auth, timeline_signals, competitor_reasons,
        keyword_spikes, data_context, vendor_provenance,
        displacement_provenance,
        pain_provenance, use_case_provenance, integration_provenance,
        buyer_profile_provenance,
        insider_aggregates_raw,
        product_profiles_raw,
        # Idle field aggregations
        _review_text_aggs,
        _department_dist,
        _company_size_dist,
        _contract_ctx_aggs,
        _sentiment_tenure_raw,
        _turning_points_raw,
    ) = await asyncio.gather(
        _fetch_vendor_churn_scores(pool, window_days, min_reviews),
        _fetch_high_intent_companies(pool, urgency_threshold, window_days),
        _fetch_existing_company_signals(pool, window_days=window_days),
        _fetch_competitive_displacement_source_of_truth(
            pool,
            as_of=today,
            analysis_window_days=window_days,
        ),
        _fetch_pain_distribution(pool, window_days),
        _fetch_feature_gaps(pool, window_days, min_mentions=fg_min_mentions),
        _fetch_negative_review_counts(pool, window_days, threshold=neg_threshold),
        _fetch_price_complaint_rates(pool, window_days),
        _fetch_dm_churn_rates(pool, window_days),
        _fetch_churning_companies(pool, window_days),
        _fetch_quotable_evidence(pool, window_days, min_urgency=quote_min_urgency),
        _fetch_evidence_vault_review_rows(pool, window_days),
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
        # Idle field fetches
        _fetch_review_text_aggregates(pool, window_days),
        _fetch_department_distribution(pool, window_days),
        _fetch_company_size_distribution(pool, window_days),
        _fetch_contract_context_distribution(pool, window_days),
        _fetch_sentiment_tenure(pool, window_days),
        _fetch_turning_points(pool, window_days),
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
    if isinstance(existing_company_signals, Exception):
        logger.warning("existing_company_signals fetch failed: %s", existing_company_signals)
        existing_company_signals = {}
    competitive_disp = _aggregate_competitive_disp(_safe(competitive_disp, "competitive_disp"))
    pain_dist = _safe(pain_dist, "pain_dist")
    feature_gaps = _safe(feature_gaps, "feature_gaps")
    negative_counts = _safe(negative_counts, "negative_counts")
    price_rates = _safe(price_rates, "price_rates")
    dm_rates = _safe(dm_rates, "dm_rates")
    churning_companies = _safe(churning_companies, "churning_companies")
    quotable_evidence = _safe(quotable_evidence, "quotable_evidence")
    evidence_vault_review_rows = _safe(evidence_vault_review_rows, "evidence_vault_review_rows")
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
    # Idle field aggregations -- safe handling for tuple returns
    if isinstance(_review_text_aggs, Exception):
        logger.warning("review_text_aggregates fetch failed: %s", _review_text_aggs)
        _review_text_aggs = ([], [])
    if isinstance(_department_dist, Exception):
        logger.warning("department_distribution fetch failed: %s", _department_dist)
        _department_dist = []
    if isinstance(_company_size_dist, Exception):
        logger.warning("company_size_distribution fetch failed: %s", _company_size_dist)
        _company_size_dist = []
    if isinstance(_contract_ctx_aggs, Exception):
        logger.warning("contract_context_distribution fetch failed: %s", _contract_ctx_aggs)
        _contract_ctx_aggs = ([], [])
    if isinstance(_sentiment_tenure_raw, Exception):
        logger.warning("sentiment_tenure fetch failed: %s", _sentiment_tenure_raw)
        _sentiment_tenure_raw = []
    if isinstance(_turning_points_raw, Exception):
        logger.warning("turning_points fetch failed: %s", _turning_points_raw)
        _turning_points_raw = []

    scoped_vendors = _normalize_test_vendors((task.metadata or {}).get("test_vendors"))
    if scoped_vendors:
        raw_vendor_count = _count_analyzed_vendors(vendor_scores)
        scoped_data, scoped_vendors = _apply_vendor_scope_to_churn_inputs(
            {
                "vendor_scores": vendor_scores,
                "high_intent": high_intent,
                "existing_company_signals": existing_company_signals,
                "competitive_disp": competitive_disp,
                "pain_dist": pain_dist,
                "feature_gaps": feature_gaps,
                "negative_counts": negative_counts,
                "price_rates": price_rates,
                "dm_rates": dm_rates,
                "churning_companies": churning_companies,
                "quotable_evidence": quotable_evidence,
                "evidence_vault_review_rows": evidence_vault_review_rows,
                "budget_signals": budget_signals,
                "use_case_dist": use_case_dist,
                "sentiment_traj": sentiment_traj,
                "buyer_auth": buyer_auth,
                "timeline_signals": timeline_signals,
                "competitor_reasons": competitor_reasons,
                "keyword_spikes": keyword_spikes,
                "vendor_provenance": vendor_provenance,
                "displacement_provenance": displacement_provenance,
                "pain_provenance": pain_provenance,
                "use_case_provenance": use_case_provenance,
                "integration_provenance": integration_provenance,
                "buyer_profile_provenance": buyer_profile_provenance,
                "insider_aggregates_raw": insider_aggregates_raw,
                "product_profiles_raw": product_profiles_raw,
                "review_text_aggs": _review_text_aggs,
                "department_dist": _department_dist,
                "contract_ctx_aggs": _contract_ctx_aggs,
                "sentiment_tenure_raw": _sentiment_tenure_raw,
                "turning_points_raw": _turning_points_raw,
            },
            scoped_vendors,
        )
        vendor_scores = scoped_data["vendor_scores"]
        high_intent = scoped_data["high_intent"]
        existing_company_signals = scoped_data["existing_company_signals"]
        competitive_disp = scoped_data["competitive_disp"]
        pain_dist = scoped_data["pain_dist"]
        feature_gaps = scoped_data["feature_gaps"]
        negative_counts = scoped_data["negative_counts"]
        price_rates = scoped_data["price_rates"]
        dm_rates = scoped_data["dm_rates"]
        churning_companies = scoped_data["churning_companies"]
        quotable_evidence = scoped_data["quotable_evidence"]
        evidence_vault_review_rows = scoped_data["evidence_vault_review_rows"]
        budget_signals = scoped_data["budget_signals"]
        use_case_dist = scoped_data["use_case_dist"]
        sentiment_traj = scoped_data["sentiment_traj"]
        buyer_auth = scoped_data["buyer_auth"]
        timeline_signals = scoped_data["timeline_signals"]
        competitor_reasons = scoped_data["competitor_reasons"]
        keyword_spikes = scoped_data["keyword_spikes"]
        vendor_provenance = scoped_data["vendor_provenance"]
        displacement_provenance = scoped_data["displacement_provenance"]
        pain_provenance = scoped_data["pain_provenance"]
        use_case_provenance = scoped_data["use_case_provenance"]
        integration_provenance = scoped_data["integration_provenance"]
        buyer_profile_provenance = scoped_data["buyer_profile_provenance"]
        insider_aggregates_raw = scoped_data["insider_aggregates_raw"]
        product_profiles_raw = scoped_data["product_profiles_raw"]
        _review_text_aggs = scoped_data["review_text_aggs"]
        _department_dist = scoped_data["department_dist"]
        _contract_ctx_aggs = scoped_data["contract_ctx_aggs"]
        _sentiment_tenure_raw = scoped_data["sentiment_tenure_raw"]
        _turning_points_raw = scoped_data["turning_points_raw"]
        logger.info(
            "Scoped churn intelligence to %d/%d vendors for test run: %s",
            _count_analyzed_vendors(vendor_scores),
            raw_vendor_count,
            sorted(scoped_vendors),
        )
    analyzed_vendor_count = _count_analyzed_vendors(vendor_scores)
    insider_lookup = _build_insider_lookup(insider_aggregates_raw)

    # Build idle field lookups
    _complaint_rows, _positive_rows = _review_text_aggs if isinstance(_review_text_aggs, tuple) else ([], [])
    complaint_lookup = _build_complaint_lookup(_complaint_rows)
    positive_lookup = _build_positive_lookup(_positive_rows)
    department_lookup = _build_department_lookup(_department_dist if isinstance(_department_dist, list) else [])
    _cv_rows, _dur_rows = _contract_ctx_aggs if isinstance(_contract_ctx_aggs, tuple) else ([], [])
    contract_value_lookup = _build_contract_value_lookup(_cv_rows)
    usage_duration_lookup = _build_usage_duration_lookup(_dur_rows)
    tenure_lookup = _build_tenure_lookup(_sentiment_tenure_raw if isinstance(_sentiment_tenure_raw, list) else [])
    turning_point_lookup = _build_turning_point_lookup(_turning_points_raw if isinstance(_turning_points_raw, list) else [])

    # Check if there's enough data
    if not vendor_scores and not high_intent:
        tracer.end_span(span, status="completed", output_data={"skipped": "no enriched reviews"})
        return {"_skip_synthesis": "No enriched B2B reviews to analyze"}

    _reasoning_seen: set[str] = set()
    reasoning_target = 0
    for vs in vendor_scores:
        vn = _canonicalize_vendor(vs.get("vendor_name") or "")
        if not vn or vn in _reasoning_seen:
            continue
        _reasoning_seen.add(vn)
        reasoning_target += 1

    await _update_execution_progress(
        task,
        stage=_STAGE_REASONING,
        progress_current=0,
        progress_total=reasoning_target,
        progress_message="Building deterministic temporal context before synthesis handoff.",
        vendors_analyzed=analyzed_vendor_count,
        high_intent_companies=len(high_intent),
        fetcher_failures=fetcher_failures,
        synced_firmographics=synced_firmographics,
    )

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

    preferred_profile_categories: dict[str, str] = {}
    for pp in product_profiles_raw or []:
        vn = _canonicalize_vendor(pp.get("vendor_name") or "")
        if not vn:
            continue
        category = str(pp.get("product_category") or "").strip()
        if not category or _is_generic_product_category(category):
            continue
        preferred_profile_categories.setdefault(vn, category)

    # Enrich payload with temporal analysis + archetype pre-scores per vendor
    _temporal_lookup: dict[str, dict] = {}
    _archetype_lookup: dict[str, list[dict]] = {}
    _market_regime_lookup: dict[str, dict[str, Any]] = {}
    try:
        from atlas_brain.reasoning.temporal import TemporalEngine
        from atlas_brain.reasoning.archetypes import enrich_evidence_with_archetypes

        temporal_engine = TemporalEngine(pool)
        temporal_summaries = []
        _temporal_vendors = vendor_scores[:cfg.temporal_analysis_vendor_limit]
        for vs in _temporal_vendors:
            vname = vs["vendor_name"]
            try:
                te = await temporal_engine.analyze_vendor(vname)
                td = TemporalEngine.to_evidence_dict(te)
                evidence_seed = {
                    "vendor_name": vname,
                    "support_sentiment": vs.get("support_sentiment"),
                    "legacy_support_score": vs.get("legacy_support_score"),
                    "new_feature_velocity": vs.get("new_feature_velocity"),
                    "employee_growth_rate": vs.get("employee_growth_rate"),
                    **td,
                }
                enriched = enrich_evidence_with_archetypes(evidence_seed, td)
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

    market_regimes: dict[str, Any] = {}
    if _temporal_lookup:
        try:
            from atlas_brain.reasoning.market_pulse import MarketPulseReasoner
            from atlas_brain.reasoning.temporal import TemporalEvidence, VendorVelocity

            mp_reasoner = MarketPulseReasoner()
            vendors_by_cat: dict[str, list[tuple[str, dict[str, Any]]]] = {}
            for vs in vendor_scores:
                vname = _canonicalize_vendor(vs.get("vendor_name") or "")
                td = _temporal_lookup.get(vname)
                if not td:
                    continue
                category = _resolve_vendor_category(
                    vname,
                    str(vs.get("product_category") or ""),
                    preferred_profile_categories,
                )
                if category:
                    vendors_by_cat.setdefault(category, []).append((vname, td))

            for category, vendor_tds in vendors_by_cat.items():
                te_list = []
                for vname, td in vendor_tds:
                    velocities = []
                    for key, value in td.items():
                        if not key.startswith("velocity_"):
                            continue
                        velocities.append(VendorVelocity(
                            vendor_name=vname,
                            metric=key.replace("velocity_", ""),
                            current_value=0,
                            previous_value=0,
                            velocity=float(value),
                            days_between=1,
                        ))
                    te_list.append(TemporalEvidence(
                        vendor_name=vname,
                        snapshot_days=td.get("snapshot_days", 0),
                        velocities=velocities,
                    ))
                regime = mp_reasoner.analyze_category(category, te_list)
                market_regimes[category] = regime

            for vs in vendor_scores:
                vname = _canonicalize_vendor(vs.get("vendor_name") or "")
                category = _resolve_vendor_category(
                    vname,
                    str(vs.get("product_category") or ""),
                    preferred_profile_categories,
                )
                regime = market_regimes.get(category)
                if vname and regime is not None:
                    _market_regime_lookup[vname] = asdict(regime)
        except Exception:
            logger.debug("Market pulse analysis skipped", exc_info=True)

    # Vendor and cross-vendor LLM reasoning now run in the dedicated
    # b2b_reasoning_synthesis follow-up task. Keep the shared fallback shape
    # stable here so snapshots, feeds, and change detection still build.
    reasoning_lookup: dict[str, dict] = {}
    logger.info(
        "Vendor and cross-vendor reasoning is deferred to b2b_reasoning_synthesis",
    )

    await _update_execution_progress(
        task,
        stage=_STAGE_REASONING,
        progress_current=reasoning_target,
        progress_total=reasoning_target,
        progress_message="Prepared deterministic vendor context for b2b_reasoning_synthesis.",
        vendors_analyzed=analyzed_vendor_count,
        high_intent_companies=len(high_intent),
        fetcher_failures=fetcher_failures,
        synced_firmographics=synced_firmographics,
        reasoning_vendors=0,
    )

    # Exploratory overview LLM, post-processing, and report validation are
    # now handled by the b2b_churn_reports follow-up task.
    parsed: dict[str, Any] = {}

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
    lock_in_lookup = _build_lock_in_lookup(use_case_dist)
    sentiment_lookup = _build_sentiment_lookup(sentiment_traj)
    buyer_auth_lookup = _build_buyer_auth_lookup(buyer_auth)
    try:
        _role_churn_rows = await _fetch_role_churn_summary(pool, window_days)
    except Exception:
        logger.warning("role_churn_summary fetch failed", exc_info=True)
        _role_churn_rows = []
    role_churn_lookup = _build_role_churn_lookup(_role_churn_rows)
    company_size_lookup: dict[str, list[dict[str, Any]]] = {}
    for row in (_company_size_dist or []):
        vn = _canonicalize_vendor(row.get("vendor") or "")
        if not vn:
            continue
        company_size_lookup.setdefault(vn, []).append({
            "segment": row.get("segment"),
            "review_count": row.get("review_count", 0),
            "churn_rate": row.get("churn_rate"),
        })
    timeline_lookup = _build_timeline_lookup(timeline_signals)
    keyword_spike_lookup = _build_keyword_spike_lookup(keyword_spikes)
    product_profile_lookup: dict[str, dict] = {}
    for pp in product_profiles_raw:
        vn = _canonicalize_vendor(pp.get("vendor_name", ""))
        if vn and vn not in product_profile_lookup:
            product_profile_lookup[vn] = pp
    evidence_vault_pass2_lookup = _build_evidence_vault_pass2_rollups(
        evidence_vault_review_rows,
        quote_lookup,
        recent_window_days=cfg.intelligence_recent_window_days,
    )

    # ---------------------------------------------------------------
    # Evidence vault: canonical per-vendor intelligence objects
    # consumed by all downstream products (battle cards, reports, briefs).
    # ---------------------------------------------------------------
    vault_persisted = 0
    try:
        from datetime import date as _date

        _hi_by_vendor: dict[str, list[dict]] = {}
        for hi in high_intent:
            vn = _canonicalize_vendor(hi.get("vendor") or "")
            if vn:
                _hi_by_vendor.setdefault(vn, []).append(hi)
        _company_signal_blocked_names = _build_company_signal_blocked_names_by_vendor(
            (vs.get("vendor_name") for vs in vendor_scores),
            high_intent_entries=high_intent,
            integration_lookup=integration_lookup,
        )
        _canonical_company_signals = _merge_canonical_company_signals(
            high_intent,
            existing_company_signals if isinstance(existing_company_signals, dict) else {},
            blocked_names_by_vendor=_company_signal_blocked_names,
        )

        _vault_rows: list[tuple] = []
        _vault_seen: set[str] = set()
        for vs in vendor_scores:
            vn = _canonicalize_vendor(vs.get("vendor_name") or "")
            if not vn or vn in _vault_seen:
                continue
            _vault_seen.add(vn)
            vault = build_evidence_vault(
                vendor_name=vn,
                vs=vs,
                pain_entries=pain_lookup.get(vn, []),
                feature_gap_entries=feature_gap_lookup.get(vn, []),
                quotes=quote_lookup.get(vn, []),
                positive_entries=positive_lookup.get(vn, []),
                product_profile=product_profile_lookup.get(vn),
                keyword_spikes=keyword_spike_lookup.get(vn),
                company_signals=_canonical_company_signals.get(vn) or _hi_by_vendor.get(vn, []),
                provenance=(vendor_provenance or {}).get(vn),
                data_context=data_context if isinstance(data_context, dict) else None,
                pass2_rollups=evidence_vault_pass2_lookup.get(vn),
                dm_rate=dm_lookup.get(vn),
                price_rate=price_lookup.get(vn),
                product_category=vs.get("product_category"),
                blocked_names=_company_signal_blocked_names.get(vn),
                analysis_window_days=window_days,
                recent_window_days=cfg.intelligence_recent_window_days,
            )
            _vault_rows.append((
                vn,
                _date.today(),
                window_days,
                vault["schema_version"],
                json.dumps(vault, default=str),
            ))

        if _vault_rows:
            await pool.executemany(
                """
                INSERT INTO b2b_evidence_vault
                    (vendor_name, as_of_date, analysis_window_days, schema_version, vault)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (vendor_name, as_of_date, analysis_window_days, schema_version)
                DO UPDATE SET vault = EXCLUDED.vault, created_at = NOW()
                """,
                _vault_rows,
            )
            vault_persisted = len(_vault_rows)
            logger.info("Evidence vault: persisted %d vendor vaults", vault_persisted)
    except Exception:
        logger.warning("Evidence vault persistence failed (non-fatal)", exc_info=True)

    # ---------------------------------------------------------------
    # Segment intelligence: canonical per-vendor buyer segment objects.
    # ---------------------------------------------------------------
    segment_persisted = 0
    try:
        from datetime import date as _seg_date

        _seg_rows: list[tuple] = []
        _seg_seen: set[str] = set()
        for vs in vendor_scores:
            vn = _canonicalize_vendor(vs.get("vendor_name") or "")
            if not vn or vn in _seg_seen:
                continue
            _seg_seen.add(vn)
            seg = build_segment_intelligence(
                vendor_name=vn,
                buyer_auth=buyer_auth_lookup.get(vn),
                department_entries=department_lookup.get(vn, []),
                company_size_entries=company_size_lookup.get(vn, []),
                budget=budget_lookup.get(vn),
                dm_rate=dm_lookup.get(vn),
                contract_value_entries=contract_value_lookup.get(vn, []),
                usage_duration_entries=usage_duration_lookup.get(vn, []),
                use_case_entries=use_case_lookup.get(vn, []),
                role_churn=role_churn_lookup.get(vn),
                vendor_lock_in_level=lock_in_lookup.get(vn),
                analysis_window_days=window_days,
            )
            _seg_rows.append((
                vn,
                _seg_date.today(),
                window_days,
                seg["schema_version"],
                json.dumps(seg, default=str),
            ))

        if _seg_rows:
            await pool.executemany(
                """
                INSERT INTO b2b_segment_intelligence
                    (vendor_name, as_of_date, analysis_window_days, schema_version, segments)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (vendor_name, as_of_date, analysis_window_days, schema_version)
                DO UPDATE SET segments = EXCLUDED.segments, created_at = NOW()
                """,
                _seg_rows,
            )
            segment_persisted = len(_seg_rows)
            logger.info("Segment intelligence: persisted %d vendor segments", segment_persisted)
    except Exception:
        logger.warning("Segment intelligence persistence failed (non-fatal)", exc_info=True)

    # ---------------------------------------------------------------
    # Temporal intelligence: canonical per-vendor timing and trend objects.
    # ---------------------------------------------------------------
    temporal_persisted = 0
    try:
        from datetime import date as _temp_date

        _temp_rows: list[tuple] = []
        _temp_seen: set[str] = set()
        for vs in vendor_scores:
            vn = _canonicalize_vendor(vs.get("vendor_name") or "")
            if not vn or vn in _temp_seen:
                continue
            _temp_seen.add(vn)
            temp = build_temporal_intelligence(
                vendor_name=vn,
                timeline_entries=timeline_lookup.get(vn, []),
                keyword_spikes=keyword_spike_lookup.get(vn),
                sentiment=sentiment_lookup.get(vn),
                sentiment_tenure=tenure_lookup.get(vn, []),
                turning_points=turning_point_lookup.get(vn, []),
                analysis_window_days=window_days,
            )
            _temp_rows.append((
                vn,
                _temp_date.today(),
                window_days,
                temp["schema_version"],
                json.dumps(temp, default=str),
            ))

        if _temp_rows:
            await pool.executemany(
                """
                INSERT INTO b2b_temporal_intelligence
                    (vendor_name, as_of_date, analysis_window_days,
                     schema_version, temporal)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (vendor_name, as_of_date,
                             analysis_window_days, schema_version)
                DO UPDATE SET temporal = EXCLUDED.temporal,
                              created_at = NOW()
                """,
                _temp_rows,
            )
            temporal_persisted = len(_temp_rows)
            logger.info(
                "Temporal intelligence: persisted %d vendor temporals",
                temporal_persisted,
            )
    except Exception:
        logger.warning(
            "Temporal intelligence persistence failed (non-fatal)",
            exc_info=True,
        )

    # ---------------------------------------------------------------
    # Displacement dynamics: canonical per-pair competitive flow objects.
    # ---------------------------------------------------------------
    displacement_dynamics_persisted = 0
    try:
        from datetime import date as _disp_date

        # Read persisted edges for velocity/quote data
        _edge_rows = await pool.fetch(
            """
            SELECT from_vendor, to_vendor, mention_count,
                   primary_driver, signal_strength, key_quote,
                   confidence_score, velocity_7d, velocity_30d
            FROM b2b_displacement_edges
            WHERE computed_date = (
                SELECT MAX(computed_date) FROM b2b_displacement_edges
            )
            """,
        )
        _edge_map: dict[tuple[str, str], dict] = {}
        for er in _edge_rows:
            _edge_map[(er["from_vendor"], er["to_vendor"])] = dict(er)

        # Group competitive_disp flows by (vendor, competitor)
        _flow_map: dict[tuple[str, str], list[dict]] = {}
        for f in competitive_disp:
            key = (f.get("vendor", ""), f.get("competitor", ""))
            if key[0] and key[1]:
                _flow_map.setdefault(key, []).append(f)

        # Group competitor_reasons by (vendor, competitor)
        _reason_map: dict[tuple[str, str], list[dict]] = {}
        for r in competitor_reasons:
            key = (r.get("vendor", ""), r.get("competitor", ""))
            if key[0] and key[1]:
                _reason_map.setdefault(key, []).append(r)

        # Read recent battle conclusions
        _battle_rows = await pool.fetch(
            """
            SELECT vendors, conclusion, confidence
            FROM b2b_cross_vendor_conclusions
            WHERE analysis_type = 'pairwise_battle'
              AND computed_date = (
                  SELECT MAX(computed_date)
                  FROM b2b_cross_vendor_conclusions
                  WHERE analysis_type = 'pairwise_battle'
              )
            """,
        )
        _battle_map: dict[tuple[str, str], dict] = {}
        for br in _battle_rows:
            vs = br["vendors"] or []
            if len(vs) >= 2:
                conclusion = (
                    br["conclusion"]
                    if isinstance(br["conclusion"], dict)
                    else {}
                )
                _battle_map[(vs[0], vs[1])] = conclusion
                # Do NOT map the reverse direction: the conclusion prose is
                # generated for (vs[0], vs[1]) specifically and would be
                # misleading if attached to the opposite directed edge.

        # Build and persist per unique pair
        _all_pairs: set[tuple[str, str]] = set()
        _all_pairs.update(_edge_map.keys())
        _all_pairs.update(_flow_map.keys())

        _disp_rows: list[tuple] = []
        for from_v, to_v in _all_pairs:
            dyn = build_displacement_dynamics(
                from_vendor=from_v,
                to_vendor=to_v,
                edge=_edge_map.get((from_v, to_v)),
                displacement_flows=_flow_map.get((from_v, to_v), []),
                reasons=_reason_map.get((from_v, to_v), []),
                battle_conclusion=_battle_map.get((from_v, to_v)),
                analysis_window_days=window_days,
            )
            _disp_rows.append((
                from_v,
                to_v,
                _disp_date.today(),
                window_days,
                dyn["schema_version"],
                json.dumps(dyn, default=str),
            ))

        if _disp_rows:
            await pool.executemany(
                """
                INSERT INTO b2b_displacement_dynamics
                    (from_vendor, to_vendor, as_of_date,
                     analysis_window_days, schema_version, dynamics)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (from_vendor, to_vendor, as_of_date,
                             analysis_window_days, schema_version)
                DO UPDATE SET dynamics = EXCLUDED.dynamics,
                              created_at = NOW()
                """,
                _disp_rows,
            )
            displacement_dynamics_persisted = len(_disp_rows)
            logger.info(
                "Displacement dynamics: persisted %d pair dynamics",
                displacement_dynamics_persisted,
            )
    except Exception:
        logger.warning(
            "Displacement dynamics persistence failed (non-fatal)",
            exc_info=True,
        )

    # ---------------------------------------------------------------
    # Category dynamics: canonical per-category market regime objects.
    # ---------------------------------------------------------------
    category_dynamics_persisted = 0
    if not _should_persist_category_dynamics(scoped_vendors):
        logger.info(
            "Skipping category dynamics persistence for scoped run covering %d vendors",
            len(scoped_vendors),
        )
    else:
        try:
            from datetime import date as _cat_date
            from dataclasses import asdict as _cat_asdict

            # Read category council conclusions from DB
            _council_rows = await pool.fetch(
                """
                SELECT category, conclusion, confidence
                FROM b2b_cross_vendor_conclusions
                WHERE analysis_type = 'category_council'
                  AND computed_date = (
                      SELECT MAX(computed_date)
                      FROM b2b_cross_vendor_conclusions
                      WHERE analysis_type = 'category_council'
                  )
                ORDER BY confidence DESC
                """,
            )
            _council_map: dict[str, dict] = {}
            for cr in _council_rows:
                cat = cr["category"] or ""
                if cat and cat not in _council_map:
                    _council_map[cat] = (
                        cr["conclusion"]
                        if isinstance(cr["conclusion"], dict)
                        else {}
                    )

            # Count vendors and displacement flows per category
            _cat_vendor_counts: dict[str, int] = {}
            _cat_flow_counts: dict[str, int] = {}
            for vs in vendor_scores:
                cat = _resolve_vendor_category(
                    str(vs.get("vendor_name") or ""),
                    str(vs.get("product_category") or ""),
                    preferred_profile_categories,
                )
                if cat and not _is_generic_product_category(cat):
                    _cat_vendor_counts[cat] = (
                        _cat_vendor_counts.get(cat, 0) + 1
                    )
            for f in competitive_disp:
                # Use vendor lookup to find category
                vn = f.get("vendor", "")
                for vs in vendor_scores:
                    if _canonicalize_vendor(
                        vs.get("vendor_name") or ""
                    ) == vn:
                        cat = _resolve_vendor_category(
                            vn,
                            str(vs.get("product_category") or ""),
                            preferred_profile_categories,
                        )
                        if cat and not _is_generic_product_category(cat):
                            _cat_flow_counts[cat] = (
                                _cat_flow_counts.get(cat, 0) + 1
                            )
                        break

            # Collect all categories
            _all_cats: set[str] = set()
            _all_cats.update(
                cat
                for cat in market_regimes.keys()
                if cat and not _is_generic_product_category(cat)
            )
            _all_cats.update(
                cat
                for cat in _council_map.keys()
                if cat and not _is_generic_product_category(cat)
            )
            _all_cats.update(
                cat
                for cat in _cat_vendor_counts
                if cat and not _is_generic_product_category(cat)
            )

            _cat_rows: list[tuple] = []
            for cat in sorted(_all_cats):
                if not cat:
                    continue
                mr_dict = None
                regime_obj = market_regimes.get(cat)
                if regime_obj is not None:
                    try:
                        mr_dict = _cat_asdict(regime_obj)
                    except (TypeError, AttributeError):
                        mr_dict = (
                            regime_obj
                            if isinstance(regime_obj, dict)
                            else None
                        )
                dyn = build_category_dynamics(
                    category=cat,
                    market_regime=mr_dict,
                    council_conclusion=_council_map.get(cat),
                    vendor_count=_cat_vendor_counts.get(cat, 0),
                    displacement_flow_count=_cat_flow_counts.get(
                        cat, 0
                    ),
                    analysis_window_days=window_days,
                )
                _cat_rows.append((
                    cat,
                    _cat_date.today(),
                    window_days,
                    dyn["schema_version"],
                    json.dumps(dyn, default=str),
                ))

            if _cat_rows:
                await pool.executemany(
                    """
                    INSERT INTO b2b_category_dynamics
                        (category, as_of_date, analysis_window_days,
                         schema_version, dynamics)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT (category, as_of_date,
                                 analysis_window_days, schema_version)
                    DO UPDATE SET dynamics = EXCLUDED.dynamics,
                                  created_at = NOW()
                    """,
                    _cat_rows,
                )
                category_dynamics_persisted = len(_cat_rows)
                logger.info(
                    "Category dynamics: persisted %d categories",
                    category_dynamics_persisted,
                )
        except Exception:
            logger.warning(
                "Category dynamics persistence failed (non-fatal)",
                exc_info=True,
            )

    # ---------------------------------------------------------------
    # Account intelligence: canonical per-vendor account signal objects.
    # ---------------------------------------------------------------
    account_intelligence_persisted = 0
    try:
        from datetime import date as _acct_date

        # Collect all vendors with signals
        _acct_vendors: set[str] = set()
        _acct_vendors.update(_canonical_company_signals.keys())
        # Include all vendors from vendor_scores so every vendor gets an
        # account intelligence record (even if empty) for pool completeness.
        _acct_vendors.update(
            vs.get("vendor_name", "") for vs in vendor_scores
        )

        _acct_rows: list[tuple] = []
        for vn in sorted(_acct_vendors):
            if not vn:
                continue
            acct = build_account_intelligence(
                vendor_name=vn,
                persisted_signals=_canonical_company_signals.get(vn, []),
                blocked_names=_company_signal_blocked_names.get(vn),
                analysis_window_days=window_days,
            )
            _acct_rows.append((
                vn,
                _acct_date.today(),
                window_days,
                acct["schema_version"],
                json.dumps(acct, default=str),
            ))

        if _acct_rows:
            await pool.executemany(
                """
                INSERT INTO b2b_account_intelligence
                    (vendor_name, as_of_date, analysis_window_days,
                     schema_version, accounts)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (vendor_name, as_of_date,
                             analysis_window_days, schema_version)
                DO UPDATE SET accounts = EXCLUDED.accounts,
                              created_at = NOW()
                """,
                _acct_rows,
            )
            account_intelligence_persisted = len(_acct_rows)
            logger.info(
                "Account intelligence: persisted %d vendor accounts",
                account_intelligence_persisted,
            )
    except Exception:
        logger.warning(
            "Account intelligence persistence failed (non-fatal)",
            exc_info=True,
        )

    # ---------------------------------------------------------------
    # Report building, battle cards, scorecards, executive summaries,
    # and exploratory overview persistence are handled by follow-up
    # tasks (b2b_churn_reports.py, b2b_battle_cards.py).
    # Core task only persists signals, snapshots, displacement edges,
    # and change events.
    # ---------------------------------------------------------------

    displacement_edges_persisted = 0
    upsert_failures = 0
    company_signals_persisted = 0
    pain_points_persisted = 0
    use_cases_persisted = 0
    integrations_persisted = 0
    buyer_profiles_persisted = 0
    snapshots_persisted = 0
    change_events_detected = 0
    persistence_progress = 0

    async def _update_persist_progress() -> None:
        await _update_execution_progress(
            task,
            stage=_STAGE_PERSISTING_SIGNALS,
            progress_current=persistence_progress,
            progress_total=len(_PERSISTENCE_PHASES),
            progress_message="Persisting churn intelligence outputs.",
            vendors_analyzed=analyzed_vendor_count,
            high_intent_companies=len(high_intent),
            fetcher_failures=fetcher_failures,
            upsert_failures=upsert_failures,
            displacement_edges_persisted=displacement_edges_persisted,
            company_signals_persisted=company_signals_persisted,
            pain_points_persisted=pain_points_persisted,
            use_cases_persisted=use_cases_persisted,
            integrations_persisted=integrations_persisted,
            buyer_profiles_persisted=buyer_profiles_persisted,
            snapshots_persisted=snapshots_persisted,
            vault_persisted=vault_persisted,
            segment_persisted=segment_persisted,
            temporal_persisted=temporal_persisted,
            displacement_dynamics_persisted=displacement_dynamics_persisted,
            category_dynamics_persisted=category_dynamics_persisted,
            account_intelligence_persisted=account_intelligence_persisted,
            change_events_detected=change_events_detected,
        )

    await _update_persist_progress()

    # Build displacement map (needed for edge persistence below)
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

    displacement_report_id = None  # set by follow-up report task

    # Persist displacement edges to first-class table
    try:
        # Pre-fetch prior mention counts for velocity computation (single query)
        prior_rows = await pool.fetch(
            """
            SELECT from_vendor, to_vendor, computed_date, mention_count
            FROM b2b_displacement_edges
            WHERE computed_date >= ($1::date - 30)
              AND computed_date < $1
            """,
            today,
        )
        # Build lookup: (from, to) -> {date: mention_count}
        _prior: dict[tuple[str, str], dict] = {}
        for pr in prior_rows:
            key = (pr["from_vendor"], pr["to_vendor"])
            _prior.setdefault(key, {})[pr["computed_date"]] = pr["mention_count"]

        async with pool.transaction() as conn:
            for edge in deterministic_displacement_map:
                sample_ids = [
                    _uuid.UUID(rid) for rid in edge.get("sample_review_ids", [])
                    if rid
                ]

                # Velocity: compare current mention_count to closest prior value
                pair_key = (edge["from_vendor"], edge["to_vendor"])
                pair_history = _prior.get(pair_key, {})
                cur_mentions = edge["mention_count"]

                velocity_7d = None
                velocity_30d = None
                if pair_history:
                    # Find the closest date within each window
                    for window, attr_name in [(7, "velocity_7d"), (30, "velocity_30d")]:
                        best_date = None
                        for d in pair_history:
                            age = (today - d).days
                            if age <= 0 or age > window:
                                continue
                            if best_date is None or abs(age - window) < abs((today - best_date).days - window):
                                best_date = d
                        if best_date is not None:
                            prior_val = pair_history[best_date]
                            if attr_name == "velocity_7d":
                                velocity_7d = cur_mentions - prior_val
                            else:
                                velocity_30d = cur_mentions - prior_val

                await conn.execute(
                    """
                    INSERT INTO b2b_displacement_edges (
                        from_vendor, to_vendor, mention_count,
                        primary_driver, signal_strength, key_quote,
                        source_distribution, sample_review_ids,
                        confidence_score, computed_date, report_id,
                        velocity_7d, velocity_30d
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::uuid[], $9, $10, $11, $12, $13)
                    ON CONFLICT (from_vendor, to_vendor, computed_date)
                    DO UPDATE SET
                        mention_count = EXCLUDED.mention_count,
                        primary_driver = EXCLUDED.primary_driver,
                        signal_strength = EXCLUDED.signal_strength,
                        key_quote = EXCLUDED.key_quote,
                        source_distribution = EXCLUDED.source_distribution,
                        sample_review_ids = EXCLUDED.sample_review_ids,
                        confidence_score = EXCLUDED.confidence_score,
                        report_id = EXCLUDED.report_id,
                        velocity_7d = EXCLUDED.velocity_7d,
                        velocity_30d = EXCLUDED.velocity_30d
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
                    velocity_7d,
                    velocity_30d,
                )
                displacement_edges_persisted += 1
    except Exception:
        displacement_edges_persisted = 0
        logger.exception("Failed to persist displacement edges")
    persistence_progress += 1
    await _update_persist_progress()

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
    persistence_progress += 1
    await _update_persist_progress()

    # Persist company signals to first-class table
    try:
        async with pool.transaction() as conn:
            for vendor_name, signals in _canonical_company_signals.items():
                blocked_names = _company_signal_blocked_names.get(vendor_name) or set()
                for hi in signals or []:
                    company_name = hi.get("company_name") or hi.get("company") or ""
                    if not _company_signal_name_is_eligible(
                        company_name,
                        current_vendor=vendor_name,
                        blocked_names=blocked_names,
                    ):
                        continue
                    review_id = None
                    if hi.get("review_id"):
                        try:
                            review_id = _uuid.UUID(str(hi["review_id"]))
                        except (ValueError, TypeError):
                            pass
                    # Confidence for company signal: source quality + data completeness
                    _src = hi.get("source", "")
                    _src_dist = {_src: 1} if _src else {}
                    _cs_conf = hi.get("confidence_score")
                    if _cs_conf is None:
                        _cs_conf = _compute_evidence_confidence(1, _src_dist)
                        _filled = sum(
                            1
                            for f in (
                                hi.get("decision_maker"),
                                hi.get("buying_stage"),
                                hi.get("seat_count"),
                            )
                            if f is not None
                        )
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
                        normalize_company_name(company_name),
                        vendor_name,
                        hi.get("urgency") if hi.get("urgency") is not None else hi.get("urgency_score"),
                        hi.get("pain") or hi.get("pain_category"),
                        hi.get("role_level") or hi.get("buyer_role"),
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
    persistence_progress += 1
    await _update_persist_progress()

    # Persist vendor pain points to first-class table
    try:
        async with pool.transaction() as conn:
            for (vendor, pain_cat), prov in pain_provenance.items():
                sample_ids = [
                    _uuid.UUID(rid) for rid in prov.get("sample_review_ids", [])
                    if rid
                ]
                confidence = min(_compute_evidence_confidence(
                    prov["mention_count"],
                    prov.get("source_distribution", {}),
                ), 1.0)
                raw_avg_rating = prov.get("avg_rating")
                avg_rating = min(float(raw_avg_rating), 9.99) if raw_avg_rating is not None else None
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
                    avg_rating,
                    json.dumps(prov.get("source_distribution", {})),
                    sample_ids,
                    confidence,
                )
                pain_points_persisted += 1
    except Exception:
        pain_points_persisted = 0
        logger.exception("Failed to persist vendor pain points")
    persistence_progress += 1
    await _update_persist_progress()

    # Persist vendor use cases to first-class table
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
    persistence_progress += 1
    await _update_persist_progress()

    # Persist vendor integrations to first-class table
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
    persistence_progress += 1
    await _update_persist_progress()

    # Persist buyer profiles to first-class table
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
    persistence_progress += 1
    await _update_persist_progress()

    # Persist vendor health snapshots + detect change events
    if cfg.snapshot_enabled:
        try:
            snapshots_persisted = await _persist_vendor_snapshots(
                pool, vendor_scores, pain_lookup, competitor_lookup,
                high_intent, today,
                price_lookup=price_lookup, dm_lookup=dm_lookup,
                reasoning_lookup=reasoning_lookup,
            )
            if cfg.change_detection_enabled:
                change_events_detected = await _detect_change_events(
                    pool, vendor_scores, pain_lookup, competitor_lookup, today,
                    price_lookup=price_lookup, dm_lookup=dm_lookup,
                    temporal_lookup=_temporal_lookup or None,
                    reasoning_lookup=reasoning_lookup or None,
                )
        except Exception:
            logger.exception("Failed to persist vendor snapshots / change events")
    persistence_progress += 1
    await _update_persist_progress()

    # Article-archetype correlation is now handled by follow-up tasks.

    # Write completion marker so follow-up tasks know core finished
    try:
        await pool.execute(
            """
            INSERT INTO b2b_intelligence (
                report_date, report_type, intelligence_data, status, llm_model
            ) VALUES ($1, 'core_run_complete', $2::jsonb, 'published', 'pipeline_core')
            ON CONFLICT (report_date, report_type, LOWER(COALESCE(vendor_filter,'')),
                         LOWER(COALESCE(category_filter,'')),
                         COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET intelligence_data = EXCLUDED.intelligence_data, created_at = now()
            """,
            today,
            json.dumps({
                "vendors_analyzed": analyzed_vendor_count,
                "upsert_failures": upsert_failures,
                "elapsed_seconds": round(budget.elapsed(), 1),
            }),
        )
    except Exception:
        logger.warning("Failed to write core_run_complete marker")
    persistence_progress += 1
    await _update_persist_progress()

    notification_feed = _build_deterministic_vendor_feed(
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
        reasoning_lookup=reasoning_lookup or None,
        temporal_lookup=_temporal_lookup or None,
    )
    notification_payload = {"weekly_churn_feed": notification_feed}
    notification_payload["executive_summary"] = _build_validated_executive_summary(
        notification_payload,
        data_context=data_context,
        executive_sources=_executive_source_list(),
        report_type="weekly_churn_feed",
    )

    await _update_execution_progress(
        task,
        stage=_STAGE_FINALIZING,
        progress_current=len(_PERSISTENCE_PHASES),
        progress_total=len(_PERSISTENCE_PHASES),
        progress_message="Finalizing churn-intelligence execution status.",
        vendors_analyzed=analyzed_vendor_count,
        high_intent_companies=len(high_intent),
        fetcher_failures=fetcher_failures,
        upsert_failures=upsert_failures,
        displacement_edges_persisted=displacement_edges_persisted,
        company_signals_persisted=company_signals_persisted,
        pain_points_persisted=pain_points_persisted,
        use_cases_persisted=use_cases_persisted,
        integrations_persisted=integrations_persisted,
        buyer_profiles_persisted=buyer_profiles_persisted,
        snapshots_persisted=snapshots_persisted,
        change_events_detected=change_events_detected,
        elapsed_seconds=round(budget.elapsed(), 1),
    )

    # Send ntfy notification
    await _send_notification(task, notification_payload, high_intent)

    # Emit reasoning events (no-op when reasoning disabled)
    await _emit_reasoning_events(notification_payload, high_intent, vendor_scores)

    response = {
        "_skip_synthesis": "B2B churn intelligence complete",
        "date": str(today),
        "vendors_analyzed": analyzed_vendor_count,
        "high_intent_companies": len(high_intent),
        "competitive_flows": len(competitive_disp),
        "fetcher_failures": fetcher_failures,
        "upsert_failures": upsert_failures,
        "displacement_edges_persisted": displacement_edges_persisted,
        "company_signals_persisted": company_signals_persisted,
        "pain_points_persisted": pain_points_persisted,
        "use_cases_persisted": use_cases_persisted,
        "integrations_persisted": integrations_persisted,
        "buyer_profiles_persisted": buyer_profiles_persisted,
        "snapshots_persisted": snapshots_persisted,
        "vault_persisted": vault_persisted,
        "segment_persisted": segment_persisted,
        "temporal_persisted": temporal_persisted,
        "displacement_dynamics_persisted": displacement_dynamics_persisted,
        "category_dynamics_persisted": category_dynamics_persisted,
        "account_intelligence_persisted": account_intelligence_persisted,
        "change_events_detected": change_events_detected,
        "elapsed_seconds": round(budget.elapsed(), 1),
    }
    tracer.end_span(
        span,
        status="completed",
        output_data=response,
        metadata={
            "reasoning": build_reasoning_trace_context(
                decision={"vendors_analyzed": analyzed_vendor_count},
                evidence={
                    "fetcher_failures": fetcher_failures,
                    "upsert_failures": upsert_failures,
                    "competitive_flows": len(competitive_disp),
                },
                rationale=notification_payload.get("executive_summary"),
            ),
        },
    )

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
            "vendors_analyzed": _count_analyzed_vendors(vendor_scores),
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


async def gather_intelligence_data(
    pool,
    window_days: int = 30,
    min_reviews: int = 3,
    vendor_names: list[str] | None = None,
    *,
    include_raw_artifacts: bool = False,
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
        _fetch_competitive_displacement_source_of_truth(
            pool,
            as_of=date.today(),
            analysis_window_days=window_days,
        ),
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

    if include_raw_artifacts:
        raw_results = await asyncio.gather(
            _fetch_vendor_churn_scores_from_signals(pool, window_days, min_reviews),
            _fetch_keyword_spikes(pool),
            _fetch_product_profiles(pool),
            _fetch_review_text_aggregates(pool, window_days),
            _fetch_department_distribution(pool, window_days),
            _fetch_contract_context_distribution(pool, window_days),
            _fetch_sentiment_tenure(pool, window_days),
            _fetch_turning_points(pool, window_days),
            _fetch_displacement_provenance(pool, window_days),
            return_exceptions=True,
        )
        raw_names = [
            "vendor_scores_from_signals",
            "keyword_spikes",
            "product_profiles_raw",
            "review_text_aggs",
            "department_dist",
            "contract_ctx_aggs",
            "sentiment_tenure_raw",
            "turning_points_raw",
            "displacement_provenance",
        ]
        for name, val in zip(raw_names, raw_results):
            if isinstance(val, Exception):
                fetcher_failures += 1
                logger.error("%s fetch failed: %s", name, val, exc_info=val)
                if name in {"review_text_aggs", "contract_ctx_aggs"}:
                    data[name] = ([], [])
                elif name == "displacement_provenance":
                    data[name] = {}
                else:
                    data[name] = []
            else:
                data[name] = val

    data, scoped_vendors = _apply_vendor_scope_to_churn_inputs(data, vendor_names)
    if scoped_vendors:
        logger.info(
            "Scoped gathered churn payload to vendors: %s",
            sorted(scoped_vendors),
        )

    council_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        xv_lookup = await reconstruct_cross_vendor_lookup(pool, as_of=date.today())
        for row in data["vendor_scores"]:
            if not isinstance(row, dict):
                continue
            category = str(row.get("product_category") or row.get("category") or "").strip()
            if not _should_use_cross_vendor_category(category):
                continue
            council = (xv_lookup.get("councils") or {}).get(category)
            if not isinstance(council, dict):
                continue
            conclusion = council.get("conclusion") or {}
            if not any(
                [
                    conclusion.get("winner"),
                    conclusion.get("loser"),
                    conclusion.get("conclusion"),
                    conclusion.get("market_regime"),
                    conclusion.get("key_insights"),
                ]
            ):
                continue
            vendor_key = _canonicalize_vendor(row.get("vendor_name") or row.get("vendor") or "")
            category_key = str(category).strip().lower()
            council_lookup[(vendor_key, category_key)] = {
                "winner": conclusion.get("winner") or "",
                "loser": conclusion.get("loser") or "",
                "conclusion": conclusion.get("conclusion") or "",
                "market_regime": conclusion.get("market_regime") or "",
                "durability": conclusion.get("durability_assessment") or "",
                "confidence": council.get("confidence"),
                "key_insights": list(conclusion.get("key_insights") or [])[:3],
            }
    except Exception:
        logger.debug("Cross-vendor council enrichment skipped for gathered payload", exc_info=True)

    prior_reports = await _fetch_prior_reports(pool, limit=prior_limit)

    # Trim payload to fit ~4k token input budget (8k context - 4k output)
    vendor_limit = 15
    if vendor_names:
        vendor_limit = max(
            15,
            len(
                {
                    _canonicalize_vendor(name)
                    for name in vendor_names
                    if _canonicalize_vendor(name)
                }
            ),
        )

    llm_vendors = _compact_vendor_churn_scores_for_llm(
        data["vendor_scores"],
        council_lookup=council_lookup,
        limit=vendor_limit,
    )
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

    raw_artifacts = None
    if include_raw_artifacts:
        raw_artifacts = {
            "vendor_scores_from_signals": list(data.get("vendor_scores_from_signals") or []),
            "vendor_scores": list(data.get("vendor_scores") or []),
            "high_intent": list(data.get("high_intent") or []),
            "competitive_disp": list(data.get("competitive_disp") or []),
            "pain_dist": list(data.get("pain_dist") or []),
            "feature_gaps": list(data.get("feature_gaps") or []),
            "negative_counts": list(data.get("negative_counts") or []),
            "price_rates": list(data.get("price_rates") or []),
            "dm_rates": list(data.get("dm_rates") or []),
            "churning_companies": list(data.get("churning_companies") or []),
            "quotable_evidence": list(data.get("quotable_evidence") or []),
            "budget_signals": list(data.get("budget_signals") or []),
            "use_case_dist": list(data.get("use_case_dist") or []),
            "sentiment_traj": list(data.get("sentiment_traj") or []),
            "buyer_auth": list(data.get("buyer_auth") or []),
            "timeline_signals": list(data.get("timeline_signals") or []),
            "competitor_reasons": list(data.get("competitor_reasons") or []),
            "data_context": dict(data.get("data_context") or {}),
            "prior_reports": list(prior_reports),
            "keyword_spikes": list(data.get("keyword_spikes") or []),
            "product_profiles_raw": list(data.get("product_profiles_raw") or []),
            "review_text_aggs": data.get("review_text_aggs") or ([], []),
            "department_dist": list(data.get("department_dist") or []),
            "contract_ctx_aggs": data.get("contract_ctx_aggs") or ([], []),
            "sentiment_tenure_raw": list(data.get("sentiment_tenure_raw") or []),
            "turning_points_raw": list(data.get("turning_points_raw") or []),
            "displacement_provenance": dict(data.get("displacement_provenance") or {}),
        }

    result = {
        "payload": payload,
        "fetcher_failures": fetcher_failures,
        "vendors_analyzed": _count_analyzed_vendors(data["vendor_scores"]),
        "high_intent_companies": len(data["high_intent"]),
        "competitive_flows": len(data["competitive_disp"]),
        "pain_categories": len(data["pain_dist"]),
        "feature_gaps": len(data["feature_gaps"]),
    }
    if raw_artifacts is not None:
        result["raw_artifacts"] = raw_artifacts
    return result


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
                    reasoning_risk_level, reasoning_executive_summary,
                    reasoning_key_signals, reasoning_uncertainty_sources,
                    last_computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                          $12, $13, $14, $15, $16, $17, $18, $19, $20, $21,
                          $22, $23, $24, $25, $26, $27, $28, $29,
                          $30, $31, $32, $33,
                          $34, $35, $36, $37,
                          $38, $39, $40, $41,
                          $42)
                ON CONFLICT (vendor_name) DO UPDATE SET
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
                    archetype = COALESCE(EXCLUDED.archetype, b2b_churn_signals.archetype),
                    archetype_confidence = COALESCE(EXCLUDED.archetype_confidence, b2b_churn_signals.archetype_confidence),
                    reasoning_mode = COALESCE(EXCLUDED.reasoning_mode, b2b_churn_signals.reasoning_mode),
                    falsification_conditions = COALESCE(EXCLUDED.falsification_conditions, b2b_churn_signals.falsification_conditions),
                    reasoning_risk_level = COALESCE(EXCLUDED.reasoning_risk_level, b2b_churn_signals.reasoning_risk_level),
                    reasoning_executive_summary = COALESCE(EXCLUDED.reasoning_executive_summary, b2b_churn_signals.reasoning_executive_summary),
                    reasoning_key_signals = COALESCE(EXCLUDED.reasoning_key_signals, b2b_churn_signals.reasoning_key_signals),
                    reasoning_uncertainty_sources = COALESCE(EXCLUDED.reasoning_uncertainty_sources, b2b_churn_signals.reasoning_uncertainty_sources),
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
                # Reasoning columns (migration 139 + 144)
                reasoning_lookup.get(vendor, {}).get("archetype"),
                reasoning_lookup.get(vendor, {}).get("confidence"),
                reasoning_lookup.get(vendor, {}).get("mode"),
                json.dumps(reasoning_lookup.get(vendor, {}).get("falsification_conditions", [])) if reasoning_lookup.get(vendor) else None,
                reasoning_lookup.get(vendor, {}).get("risk_level"),
                reasoning_lookup.get(vendor, {}).get("executive_summary"),
                json.dumps(reasoning_lookup.get(vendor, {}).get("key_signals", [])) if reasoning_lookup.get(vendor) else None,
                json.dumps(reasoning_lookup.get(vendor, {}).get("uncertainty_sources", [])) if reasoning_lookup.get(vendor) else None,
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
               r.sentiment_direction,
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
                category = _normalize_report_pain_category(p["category"])
                pain_counts[category] = pain_counts.get(category, 0) + 1

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


async def _vendor_snapshot_from_pools(
    pool,
    vendor_name: str,
    window_days: int,
) -> dict[str, Any] | None:
    """Build a vendor comparison snapshot from pre-computed pool tables."""
    vault, resolved_window = await _fetch_vendor_vault_row(
        pool,
        vendor_name,
        window_days,
    )
    if not vault:
        return None

    ms = vault.get("metric_snapshot", {})

    # Churn signals (for fields not always in vault)
    cs_row = await pool.fetchrow(
        """
        SELECT total_reviews, churn_intent_count, avg_urgency_score,
               avg_rating_normalized
        FROM b2b_churn_signals
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY last_computed_at DESC
        LIMIT 1
        """,
        vendor_name,
    )

    # Product profile (for competitors and categories)
    pp_row = await pool.fetchrow(
        """
        SELECT commonly_compared_to, product_category
        FROM b2b_product_profiles
        WHERE LOWER(vendor_name) = LOWER($1)
        ORDER BY created_at DESC
        LIMIT 1
        """,
        vendor_name,
    )

    total_reviews = int(ms.get("total_reviews") or (cs_row["total_reviews"] if cs_row else 0) or 0)
    churn_density = float(ms.get("churn_density") or 0)
    avg_urgency = float(ms.get("avg_urgency") or 0)
    positive_pct = ms.get("positive_review_pct")
    recommend_ratio = float(ms.get("recommend_ratio") or 0)
    avg_rating = float(cs_row["avg_rating_normalized"]) if cs_row and cs_row["avg_rating_normalized"] is not None else None

    churn_intent = int(cs_row["churn_intent_count"]) if cs_row and cs_row["churn_intent_count"] is not None else int(round(churn_density * total_reviews / 100.0)) if total_reviews else 0

    # Count high-urgency from window-compatible company signals when available.
    company_signals = vault.get("company_signals", [])
    high_urgency_count = 0
    for cs in company_signals:
        if not isinstance(cs, dict):
            continue
        try:
            if float(cs.get("urgency_score") or 0) >= 8:
                high_urgency_count += 1
        except (TypeError, ValueError):
            continue
    if not high_urgency_count:
        high_urgency_row = await pool.fetchrow(
            """
            SELECT COUNT(*) AS cnt
            FROM b2b_company_signals
            WHERE LOWER(vendor_name) = LOWER($1)
              AND urgency_score >= 8
              AND last_seen_at >= NOW() - INTERVAL '1 day' * $2
            """,
            vendor_name,
            resolved_window or window_days,
        )
        high_urgency_count = int(high_urgency_row["cnt"]) if high_urgency_row else 0

    # Pain categories + feature gaps from weakness_evidence
    weakness_ev = vault.get("weakness_evidence", [])
    pain_counts: dict[str, int] = {}
    gap_counts: dict[str, int] = {}
    quote_highlights: list[str] = []
    for ev in weakness_ev:
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("evidence_type", "")
        label = ev.get("label", "")
        count = int(ev.get("mention_count_total") or 1)
        if ev_type == "feature_gap":
            if label:
                gap_counts[label] = gap_counts.get(label, 0) + count
        else:
            cat = ev.get("key") or label
            if cat:
                category = _normalize_report_pain_category(cat)
                pain_counts[category] = pain_counts.get(category, 0) + count
        best_quote = ev.get("best_quote")
        if best_quote and best_quote not in quote_highlights and len(quote_highlights) < 5:
            quote_highlights.append(str(best_quote))

    # Company examples from vault company_signals
    company_examples: list[str] = []
    for cs in company_signals:
        if isinstance(cs, dict):
            name = cs.get("company_name")
            if name and name not in company_examples:
                company_examples.append(str(name))
            if len(company_examples) >= 10:
                break

    # Competitors from product_profiles
    comp_list: list[dict[str, Any]] = []
    if pp_row:
        compared_to = _safe_json(pp_row["commonly_compared_to"], default=[])
        for item in compared_to[:5]:
            parsed = _parse_profile_competitor_entry(item)
            if not parsed:
                continue
            name, count = parsed
            comp_list.append({"competitor": name, "count": count})

    # Product categories from product_profiles
    cat_list: list[dict[str, Any]] = []
    if pp_row and pp_row["product_category"]:
        cat_list.append({"category": str(pp_row["product_category"]), "count": total_reviews})

    return {
        "vendor_name": vendor_name,
        "signal_count": total_reviews,
        "high_urgency_count": high_urgency_count,
        "churn_intent_count": churn_intent,
        "churn_signal_density": round(churn_density, 1),
        "avg_urgency_score": round(avg_urgency, 2),
        "avg_rating_normalized": round(avg_rating * 100, 1) if avg_rating is not None else None,
        "positive_review_pct": round(float(positive_pct), 1) if positive_pct is not None else None,
        "recommend_ratio": round(recommend_ratio, 1),
        "product_categories": cat_list,
        "top_pain_categories": sorted(
            [{"category": k, "count": v} for k, v in pain_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "top_competitors": comp_list[:5],
        "top_feature_gaps": sorted(
            [{"feature": k, "count": v} for k, v in gap_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "company_examples": company_examples[:10],
        "quote_highlights": quote_highlights[:5],
    }


async def _head_to_head_from_edges(
    pool,
    vendor_a: str,
    vendor_b: str,
    window_days: int | None = None,
) -> list[dict[str, Any]]:
    """Read head-to-head displacement from dynamics, with edge-backed company examples."""
    edge_rows = await pool.fetch(
        """
        SELECT from_vendor, to_vendor, mention_count, sample_review_ids
        FROM b2b_displacement_edges
        WHERE computed_date = (SELECT MAX(computed_date) FROM b2b_displacement_edges)
          AND ((LOWER(from_vendor) = LOWER($1) AND LOWER(to_vendor) = LOWER($2))
            OR (LOWER(from_vendor) = LOWER($2) AND LOWER(to_vendor) = LOWER($1)))
        """,
        vendor_a,
        vendor_b,
    )
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in edge_rows:
        edge_map[(row["from_vendor"], row["to_vendor"])] = {
            "count": int(row["mention_count"] or 0),
            "sample_review_ids": list(row.get("sample_review_ids") or []),
        }

    dynamics_counts: dict[tuple[str, str], int] = {}
    if window_days is not None:
        dyn_rows = await pool.fetch(
            """
            WITH resolved_window AS (
                SELECT analysis_window_days
                FROM b2b_displacement_dynamics
                WHERE ((LOWER(from_vendor) = LOWER($1) AND LOWER(to_vendor) = LOWER($2))
                    OR (LOWER(from_vendor) = LOWER($2) AND LOWER(to_vendor) = LOWER($1)))
                GROUP BY analysis_window_days
                ORDER BY
                    CASE WHEN analysis_window_days = $3 THEN 0 ELSE 1 END,
                    ABS(analysis_window_days - $3),
                    analysis_window_days DESC
                LIMIT 1
            )
            SELECT DISTINCT ON (from_vendor, to_vendor)
                   from_vendor, to_vendor, dynamics
            FROM b2b_displacement_dynamics
            WHERE ((LOWER(from_vendor) = LOWER($1) AND LOWER(to_vendor) = LOWER($2))
                OR (LOWER(from_vendor) = LOWER($2) AND LOWER(to_vendor) = LOWER($1)))
              AND analysis_window_days = (SELECT analysis_window_days FROM resolved_window)
            ORDER BY from_vendor, to_vendor, as_of_date DESC, created_at DESC
            """,
            vendor_a,
            vendor_b,
            window_days,
        )
        for row in dyn_rows:
            dyn = _safe_json(row.get("dynamics"), default={})
            if not isinstance(dyn, dict):
                continue
            metrics = dyn.get("edge_metrics") or {}
            dynamics_counts[(row["from_vendor"], row["to_vendor"])] = int(
                metrics.get("mention_count") or 0
            )

    sample_review_ids: list[_uuid.UUID] = []
    for payload in edge_map.values():
        for review_id in payload.get("sample_review_ids") or []:
            if review_id and review_id not in sample_review_ids:
                sample_review_ids.append(review_id)

    company_lookup: dict[tuple[str, str], str] = {}
    if sample_review_ids:
        company_rows = await pool.fetch(
            """
            SELECT vendor_name, review_id, company_name
            FROM b2b_company_signals
            WHERE review_id = ANY($1::uuid[])
            ORDER BY urgency_score DESC NULLS LAST, last_seen_at DESC
            """,
            sample_review_ids,
        )
        for row in company_rows:
            review_id = row.get("review_id")
            company_name = row.get("company_name")
            vendor_name = row.get("vendor_name")
            if not review_id or not company_name or not vendor_name:
                continue
            company_lookup[(str(vendor_name).lower(), str(review_id))] = str(company_name)
        review_company_rows = await pool.fetch(
            """
            SELECT vendor_name, id AS review_id, reviewer_company
            FROM b2b_reviews
            WHERE id = ANY($1::uuid[])
              AND enrichment_status = 'enriched'
              AND reviewer_company IS NOT NULL
              AND BTRIM(reviewer_company) <> ''
            """,
            sample_review_ids,
        )
        for row in review_company_rows:
            review_id = row.get("review_id")
            company_name = row.get("reviewer_company")
            vendor_name = row.get("vendor_name")
            if not review_id or not company_name or not vendor_name:
                continue
            company_lookup.setdefault(
                (str(vendor_name).lower(), str(review_id)),
                str(company_name),
            )

    def _companies_for_pair(from_vendor: str, to_vendor: str) -> list[str]:
        payload = edge_map.get((from_vendor, to_vendor), {}) or {}
        companies: list[str] = []
        for review_id in payload.get("sample_review_ids") or []:
            company = company_lookup.get((str(from_vendor).lower(), str(review_id)))
            if (
                company
                and _company_signal_name_is_eligible(
                    company,
                    current_vendor=from_vendor,
                    blocked_names=None,
                )
                and company not in companies
            ):
                companies.append(company)
            if len(companies) >= 5:
                break
        return companies

    def _count_for_pair(from_vendor: str, to_vendor: str) -> int:
        if (from_vendor, to_vendor) in dynamics_counts:
            return dynamics_counts[(from_vendor, to_vendor)]
        return int((edge_map.get((from_vendor, to_vendor), {}) or {}).get("count") or 0)

    return [
        {
            "name": f"{vendor_a} -> {vendor_b}",
            "count": _count_for_pair(vendor_a, vendor_b),
            "companies": _companies_for_pair(vendor_a, vendor_b),
        },
        {
            "name": f"{vendor_b} -> {vendor_a}",
            "count": _count_for_pair(vendor_b, vendor_a),
            "companies": _companies_for_pair(vendor_b, vendor_a),
        },
    ]


async def _switching_triggers_from_dynamics(
    pool,
    vendor_name: str,
    window_days: int | None = None,
) -> list[dict[str, Any]]:
    """Extract switching triggers from pre-computed displacement dynamics."""
    params: list[Any] = [vendor_name]
    sql = """
        SELECT DISTINCT ON (to_vendor)
               to_vendor, dynamics
        FROM b2b_displacement_dynamics
        WHERE LOWER(from_vendor) = LOWER($1)
    """
    if window_days is not None:
        sql += """
          AND analysis_window_days = (
              SELECT analysis_window_days
              FROM b2b_displacement_dynamics
              WHERE LOWER(from_vendor) = LOWER($1)
              GROUP BY analysis_window_days
              ORDER BY
                  CASE WHEN analysis_window_days = $2 THEN 0 ELSE 1 END,
                  ABS(analysis_window_days - $2),
                  analysis_window_days DESC
              LIMIT 1
          )
        """
        params.append(window_days)
    else:
        params.append(None)
    sql += """
          AND as_of_date = (
              SELECT MAX(as_of_date)
              FROM b2b_displacement_dynamics
              WHERE LOWER(from_vendor) = LOWER($1)
                AND (
                    $2::int IS NULL
                    OR analysis_window_days = (
                        SELECT analysis_window_days
                        FROM b2b_displacement_dynamics
                        WHERE LOWER(from_vendor) = LOWER($1)
                        GROUP BY analysis_window_days
                        ORDER BY
                            CASE WHEN analysis_window_days = $2 THEN 0 ELSE 1 END,
                            ABS(analysis_window_days - $2),
                            analysis_window_days DESC
                        LIMIT 1
                    )
                )
          )
        ORDER BY to_vendor, created_at DESC
    """
    rows = await pool.fetch(sql, *params)
    trigger_map: dict[str, dict[str, Any]] = {}
    for r in rows:
        dyn = _safe_json(r["dynamics"], default={})
        if not isinstance(dyn, dict):
            continue
        to_vendor = _canonicalize_competitor(str(r["to_vendor"] or "").strip())
        if not to_vendor or not _switching_trigger_competitor_is_eligible(to_vendor):
            continue
        edge_metrics = dyn.get("edge_metrics", {})
        primary_driver = edge_metrics.get("primary_driver", "")
        switch_reasons = dyn.get("switch_reasons", [])
        total_mentions = int(edge_metrics.get("mention_count", 0))
        top_count = 0
        top_reason = primary_driver
        if switch_reasons and isinstance(switch_reasons, list):
            for sr in switch_reasons:
                if isinstance(sr, dict):
                    cnt = int(sr.get("mention_count") or sr.get("count") or 0)
                    candidate = str(
                        sr.get("reason_category")
                        or sr.get("reason")
                        or sr.get("reason_detail")
                        or primary_driver
                        or ""
                    ).strip()
                    if len(candidate) > 80 and primary_driver:
                        candidate = str(primary_driver).strip()
                    if cnt > top_count:
                        top_count = cnt
                        top_reason = candidate or primary_driver
        if not top_reason:
            top_reason = "unspecified"
        bucket = trigger_map.setdefault(
            to_vendor.lower(),
            {
                "competitor": to_vendor,
                "primary_reason": top_reason,
                "mention_count": 0,
                "total_mentions": 0,
                "_reason_strength": 0,
            },
        )
        bucket["total_mentions"] += total_mentions
        bucket["mention_count"] += top_count or total_mentions
        if (top_count or total_mentions) >= int(bucket.get("_reason_strength") or 0):
            bucket["primary_reason"] = top_reason
            bucket["_reason_strength"] = top_count or total_mentions
    triggers = list(trigger_map.values())
    for item in triggers:
        item.pop("_reason_strength", None)
    triggers.sort(key=lambda x: (-x["total_mentions"], x["competitor"]))
    return triggers[:5]


async def _company_snapshot_from_signals(
    pool,
    company_name: str,
    window_days: int,
) -> dict[str, Any] | None:
    """Build a company comparison snapshot from pre-computed pool tables."""
    company_norm = normalize_company_name(company_name)
    if not company_norm:
        return None

    rows = await pool.fetch(
        """
        SELECT vendor_name, urgency_score, pain_category, buyer_role,
               decision_maker, seat_count, contract_end, buying_stage, source,
               review_id
        FROM b2b_company_signals
        WHERE company_name = $1
          AND last_seen_at >= NOW() - INTERVAL '1 day' * $2
        ORDER BY urgency_score DESC
        """,
        company_norm,
        window_days,
    )
    if not rows:
        return None

    review_ids: list[_uuid.UUID] = []
    for row in rows:
        review_id = row.get("review_id")
        if review_id and review_id not in review_ids:
            review_ids.append(review_id)
    review_context = await _fetch_company_signal_review_context(pool, review_ids)

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
    urgencies: list[float] = []
    decision_maker_count = 0
    churn_mentions = 0
    seat_count_val: str | None = None

    for r in rows:
        vname = r["vendor_name"] or ""
        ctx = review_context.get(str(r.get("review_id") or ""), {})
        if vname:
            vendors[vname] = vendors.get(vname, 0) + 1
        category = ctx.get("product_category")
        if category:
            categories[str(category)] = categories.get(str(category), 0) + 1
        pain = r["pain_category"] or ""
        if pain:
            category = _normalize_report_pain_category(pain)
            pains[category] = pains.get(category, 0) + 1
        role = r["buyer_role"] or ""
        if role:
            role_levels[role] = role_levels.get(role, 0) + 1
        urg = float(r["urgency_score"] or 0)
        urgencies.append(urg)
        if urg >= 7:
            churn_mentions += 1
        if r["decision_maker"] is True:
            decision_maker_count += 1
        industry = str(ctx.get("industry") or "").strip()
        if industry:
            industries[industry] = industries.get(industry, 0) + 1
        if not seat_count_val:
            if ctx.get("company_size_raw"):
                seat_count_val = str(ctx["company_size_raw"])
            elif r["seat_count"] is not None:
                seat_count_val = str(r["seat_count"])
        for comp in ctx.get("competitors_json") or []:
            if isinstance(comp, dict):
                comp_name = _canonicalize_competitor(
                    str(comp.get("name") or comp.get("vendor") or "")
                )
            else:
                comp_name = _canonicalize_competitor(str(comp or ""))
            if comp_name:
                alternatives[comp_name] = alternatives.get(comp_name, 0) + 1
        for gap in ctx.get("feature_gaps") or []:
            label = gap if isinstance(gap, str) else (
                gap.get("feature", "") if isinstance(gap, dict) else ""
            )
            if label:
                gaps[str(label)] = gaps.get(str(label), 0) + 1
        for phrase in ctx.get("quotable_phrases") or []:
            text = phrase if isinstance(phrase, str) else (
                phrase.get("text", "") if isinstance(phrase, dict) else ""
            )
            if text and text not in quote_highlights:
                quote_highlights.append(str(text))
            if len(quote_highlights) >= 5:
                break
        contract_signal = str(ctx.get("contract_value_signal") or "").strip()
        if contract_signal and contract_signal not in contract_signals:
            contract_signals.append(contract_signal)
        if r["contract_end"] or ctx.get("evaluation_deadline") or ctx.get("decision_timeline"):
            timeline_signals.append({
                "vendor": vname,
                "contract_end": str(r["contract_end"]),
                "evaluation_deadline": ctx.get("evaluation_deadline"),
                "decision_timeline": ctx.get("decision_timeline"),
                "urgency": urg,
                "title": ctx.get("reviewer_title"),
                "company_size": ctx.get("company_size_raw") or seat_count_val,
                "industry": industry or None,
            })

    signal_count = len(rows)
    vendor_names = list(vendors.keys())

    # Product categories + alternatives from product profiles of current vendors
    if vendor_names:
        pp_rows = await pool.fetch(
            """
            SELECT vendor_name, product_category, commonly_compared_to
            FROM b2b_product_profiles
            WHERE vendor_name = ANY($1)
            """,
            vendor_names,
        )
        for pp in pp_rows:
            cat = pp["product_category"]
            if cat and not categories:
                categories[str(cat)] = categories.get(str(cat), 0) + vendors.get(pp["vendor_name"], 1)
            if not alternatives:
                compared = _safe_json(pp["commonly_compared_to"], default=[])
                for item in compared:
                    parsed = _parse_profile_competitor_entry(item)
                    if not parsed:
                        continue
                    name, cnt = parsed
                    alternatives[name] = alternatives.get(name, 0) + cnt

    # Feature gaps + quotes from evidence vault for each vendor
    if vendor_names:
        vault_rows = await pool.fetch(
            """
            SELECT DISTINCT ON (vendor_name) vendor_name, vault
            FROM b2b_evidence_vault
            WHERE vendor_name = ANY($1)
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
            """,
            vendor_names,
        )
        for vr in vault_rows:
            vault = _safe_json(vr["vault"], default={})
            if not isinstance(vault, dict):
                continue
            for ev in vault.get("weakness_evidence", []):
                if not isinstance(ev, dict):
                    continue
                if not gaps and ev.get("evidence_type") == "feature_gap":
                    label = ev.get("label", "")
                    if label:
                        gaps[label] = gaps.get(label, 0) + int(ev.get("mention_count_total", 1))
                bq = ev.get("best_quote") if len(quote_highlights) < 5 else None
                if bq and bq not in quote_highlights and len(quote_highlights) < 5:
                    quote_highlights.append(str(bq))

    return {
        "company_name": company_name,
        "signal_count": signal_count,
        "avg_urgency_score": round(sum(urgencies) / signal_count, 2) if signal_count else 0.0,
        "max_urgency_score": max(urgencies) if urgencies else 0.0,
        "decision_maker_signals": decision_maker_count,
        "churn_intent_count": churn_mentions,
        "current_vendors": sorted(
            [{"vendor": k, "count": v} for k, v in vendors.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "product_categories": sorted(
            [{"category": k, "count": v} for k, v in categories.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "top_pain_categories": sorted(
            [{"category": k, "count": v} for k, v in pains.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "alternatives_considered": sorted(
            [{"name": k, "count": v} for k, v in alternatives.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "top_feature_gaps": sorted(
            [{"feature": k, "count": v} for k, v in gaps.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "role_levels": sorted(
            [{"role": k, "count": v} for k, v in role_levels.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "industries": sorted(
            [{"industry": k, "count": v} for k, v in industries.items()],
            key=lambda x: x["count"], reverse=True,
        )[:5],
        "company_size": seat_count_val,
        "timeline_signals": timeline_signals[:5],
        "contract_value_signals": contract_signals[:5],
        "quote_highlights": quote_highlights[:5],
    }


def _build_vendor_comparison_summary(
    primary_snapshot: dict[str, Any],
    comparison_snapshot: dict[str, Any],
    head_to_head: list[dict[str, Any]],
    primary_archetype: str | None = None,
    comparison_archetype: str | None = None,
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
    archetype_text = ""
    if primary_archetype or comparison_archetype:
        parts = []
        if primary_archetype:
            parts.append(f"{primary_snapshot['vendor_name']} classified as {primary_archetype}")
        if comparison_archetype:
            parts.append(f"{comparison_snapshot['vendor_name']} classified as {comparison_archetype}")
        archetype_text = f" Archetype classification: {'; '.join(parts)}."
    return (
        f"{higher_risk['vendor_name']} shows the heavier churn signal pressure versus {lower_risk['vendor_name']}, "
        f"with {higher_risk['churn_intent_count']} of {higher_risk['signal_count']} reviews ({higher_risk['churn_signal_density']}%) "
        f"mentioning churn intent compared with {lower_risk['churn_intent_count']} of {lower_risk['signal_count']} "
        f"({lower_risk['churn_signal_density']}%). {flow_text}. Sentiment currently favors {stronger_sentiment['vendor_name']} "
        f"over {weaker_sentiment['vendor_name']} on positive-review share ({stronger_sentiment.get('positive_review_pct')}% vs "
        f"{weaker_sentiment.get('positive_review_pct')}%). Top pain themes diverge between {higher_risk['vendor_name']} "
        f"({high_pain.get('category', 'insufficient_data')}) and {lower_risk['vendor_name']} "
        f"({low_pain.get('category', 'insufficient_data')}).{archetype_text}"
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
    primary_snapshot, comparison_snapshot, head_to_head = await asyncio.gather(
        _vendor_snapshot_from_pools(pool, primary_name, window_days),
        _vendor_snapshot_from_pools(pool, comparison_name, window_days),
        _head_to_head_from_edges(pool, primary_name, comparison_name, window_days),
    )
    if not primary_snapshot or not comparison_snapshot:
        return None
    today = date.today()
    shared_pains = sorted(
        {row["category"] for row in primary_snapshot["top_pain_categories"]} &
        {row["category"] for row in comparison_snapshot["top_pain_categories"]}
    )

    # -- Competitive Benchmark enrichments --------------------------

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

    # Switching triggers from displacement dynamics
    primary_triggers, comparison_triggers = await asyncio.gather(
        _switching_triggers_from_dynamics(pool, primary_name, window_days),
        _switching_triggers_from_dynamics(pool, comparison_name, window_days),
    )

    # Current/prior reasoning context (synthesis-first, legacy fallback)
    from ._b2b_synthesis_reader import (
        load_best_reasoning_views,
        load_prior_reasoning_snapshots,
        synthesis_view_to_reasoning_entry,
    )

    _current_views = await load_best_reasoning_views(
        pool,
        [primary_name, comparison_name],
        as_of=today,
        analysis_window_days=window_days,
    )
    _prior_archs = await load_prior_reasoning_snapshots(
        pool,
        [primary_name, comparison_name],
        before_date=today,
        analysis_window_days=window_days,
    )

    def _view_for(vendor_name: str):
        lowered = str(vendor_name or "").strip().lower()
        for current_name, view in _current_views.items():
            if str(current_name or "").strip().lower() == lowered:
                return view
        return None

    primary_view = _view_for(primary_name)
    comparison_view = _view_for(comparison_name)
    primary_reasoning = (
        synthesis_view_to_reasoning_entry(primary_view)
        if primary_view is not None else {}
    )
    comparison_reasoning = (
        synthesis_view_to_reasoning_entry(comparison_view)
        if comparison_view is not None else {}
    )
    primary_falsification = [
        fc.get("condition", fc) if isinstance(fc, dict) else fc
        for fc in (primary_view.falsification_conditions() if primary_view is not None else [])
    ]
    comparison_falsification = [
        fc.get("condition", fc) if isinstance(fc, dict) else fc
        for fc in (comparison_view.falsification_conditions() if comparison_view is not None else [])
    ]

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

    # -- Assemble report --------------------------------------------

    _snapshot_exclude = {"top_pain_categories", "top_competitors", "top_feature_gaps",
                         "company_examples", "quote_highlights", "product_categories"}
    report_data = {
        "primary_vendor": primary_name,
        "comparison_vendor": comparison_name,
        "report_date": str(today),
        "window_days": window_days,
        "executive_summary": _build_vendor_comparison_summary(
            primary_snapshot, comparison_snapshot, head_to_head,
            primary_archetype=primary_reasoning.get("archetype") or None,
            comparison_archetype=comparison_reasoning.get("archetype") or None,
        ),
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
        # Archetype context
        "primary_archetype": primary_reasoning.get("archetype") or None,
        "primary_archetype_confidence": primary_reasoning.get("confidence"),
        "primary_archetype_was": _prior_archs.get(primary_name, {}).get("archetype"),
        "primary_falsification": primary_falsification,
        "comparison_archetype": comparison_reasoning.get("archetype") or None,
        "comparison_archetype_confidence": comparison_reasoning.get("confidence"),
        "comparison_archetype_was": _prior_archs.get(comparison_name, {}).get("archetype"),
        "comparison_falsification": comparison_falsification,
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
    snapshot = await _company_snapshot_from_signals(pool, normalized_name, window_days)
    if not snapshot:
        return None
    today = date.today()

    # Fetch vendor archetypes for the company's current vendors
    vendor_names = [v["vendor"] for v in (snapshot.get("current_vendors") or []) if v.get("vendor")]
    vendor_archetypes: dict[str, dict] = {}
    if vendor_names:
        arch_rows = await pool.fetch(
            "SELECT vendor_name, archetype, archetype_confidence "
            "FROM b2b_churn_signals WHERE vendor_name = ANY($1) AND archetype IS NOT NULL",
            vendor_names,
        )
        vendor_archetypes = {
            r["vendor_name"]: {
                "archetype": r["archetype"],
                "confidence": float(r["archetype_confidence"]) if r["archetype_confidence"] else None,
            }
            for r in arch_rows
        }

    report_data = {
        "company_name": normalized_name,
        "report_date": str(today),
        "window_days": window_days,
        "executive_summary": _build_company_deep_dive_summary(snapshot),
        "company_metrics": snapshot,
        "vendor_archetypes": vendor_archetypes or None,
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
    primary_snapshot, comparison_snapshot = await asyncio.gather(
        _company_snapshot_from_signals(pool, primary_name, window_days),
        _company_snapshot_from_signals(pool, comparison_name, window_days),
    )
    if not primary_snapshot or not comparison_snapshot:
        return None
    today = date.today()
    shared_alternatives = sorted(
        {item["name"] for item in primary_snapshot["alternatives_considered"]} &
        {item["name"] for item in comparison_snapshot["alternatives_considered"]}
    )
    shared_vendors = sorted(
        {item["vendor"] for item in primary_snapshot["current_vendors"]} &
        {item["vendor"] for item in comparison_snapshot["current_vendors"]}
    )

    # Fetch vendor archetypes for both companies' current vendors
    all_vendor_names = list({
        v["vendor"]
        for s in (primary_snapshot, comparison_snapshot)
        for v in (s.get("current_vendors") or [])
        if v.get("vendor")
    })
    company_vendor_archetypes: dict[str, dict] = {}
    if all_vendor_names:
        arch_rows = await pool.fetch(
            "SELECT vendor_name, archetype, archetype_confidence "
            "FROM b2b_churn_signals WHERE vendor_name = ANY($1) AND archetype IS NOT NULL",
            all_vendor_names,
        )
        company_vendor_archetypes = {
            r["vendor_name"]: {
                "archetype": r["archetype"],
                "confidence": float(r["archetype_confidence"]) if r["archetype_confidence"] else None,
            }
            for r in arch_rows
        }

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
        "vendor_archetypes": company_vendor_archetypes or None,
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
                category = _normalize_report_pain_category(p["category"])
                pain_counts[category] = pain_counts.get(category, 0) + 1

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
