"""Follow-up task: build per-vendor accounts-in-motion prospecting lists.

Runs after b2b_churn_core. Reads persisted artifacts from b2b_churn_signals,
b2b_reviews, and b2b_company_signals. Merges company profiles from multiple
sources, scores each account, and persists one b2b_intelligence row per vendor
with report_type='accounts_in_motion'.
"""

import asyncio
import json
import logging
from collections import Counter
from datetime import date
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._b2b_shared import _segment_targeting_summary, _timing_summary_payload, _reasoning_int
from ._execution_progress import _update_execution_progress

logger = logging.getLogger("atlas.tasks.b2b_accounts_in_motion")

_STAGE_LOADING_INPUTS = "loading_inputs"
_STAGE_BUILDING_ACCOUNTS = "building_accounts"
_STAGE_PERSISTING_REPORTS = "persisting_reports"
_STAGE_FINALIZING = "finalizing"


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_ROLE_SCORES: dict[str, int] = {
    "executive": 20,
    "economic_buyer": 15,
    "champion": 15,
    "evaluator": 10,
}

_STAGE_SCORES: dict[str, int] = {
    "active_purchase": 25,
    "evaluation": 20,
    "renewal_decision": 15,
    "post_purchase": 5,
}


def _compute_account_opportunity_score(account: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Score an account profile on a 0-100 scale.

    Returns (total_score, component_breakdown).
    """
    # Urgency: (urgency - 5) * 6, clamped [0, 30]
    urgency_raw = float(account.get("urgency") or 0)
    urgency_pts = max(0, min(30, int((urgency_raw - 5) * 6)))

    # Role: DM=20, economic_buyer/champion=15, evaluator=10
    role_pts = 0
    if account.get("decision_maker"):
        role_pts = 20
    else:
        role_level = (account.get("role_level") or "").lower()
        role_pts = _ROLE_SCORES.get(role_level, 0)

    # Stage: active_purchase=25, evaluation=20, renewal_decision=15, post_purchase=5
    stage = (account.get("buying_stage") or "").lower()
    stage_pts = _STAGE_SCORES.get(stage, 0)

    # Seats: 500+=15, 100+=10, 20+=5
    seat_count = account.get("seat_count") or 0
    if seat_count >= 500:
        seat_pts = 15
    elif seat_count >= 100:
        seat_pts = 10
    elif seat_count >= 20:
        seat_pts = 5
    else:
        seat_pts = 0

    # Alternatives: 3+=10, 2=7, 1=4
    alts = account.get("alternatives_considering") or []
    alt_count = len(alts) if isinstance(alts, list) else 0
    if alt_count >= 3:
        alt_pts = 10
    elif alt_count == 2:
        alt_pts = 7
    elif alt_count == 1:
        alt_pts = 4
    else:
        alt_pts = 0

    total = urgency_pts + role_pts + stage_pts + seat_pts + alt_pts
    components = {
        "urgency": urgency_pts,
        "role": role_pts,
        "stage": stage_pts,
        "seats": seat_pts,
        "alternatives": alt_pts,
    }
    return total, components


def _normalize_company_key(company: str | None) -> str:
    """Normalize a company name for dedup key purposes."""
    return (company or "").strip().lower()


def _extract_account_quote(quotes: Any) -> str | None:
    """Extract the first usable quote from account-scoped quote data."""
    if not isinstance(quotes, list):
        return None
    for item in quotes:
        if isinstance(item, str) and item.strip():
            return item.strip()[:500]
        if isinstance(item, dict):
            quote = str(item.get("quote") or item.get("text") or "").strip()
            if quote:
                return quote[:500]
    return None


def _clean_alternatives(
    alternatives: Any,
    *,
    company: str,
    vendor: str,
    invalid_terms: list[str],
) -> list[str]:
    """Remove duplicates, self-references, and configured non-vendor alternatives."""
    blocked = {
        _normalize_company_key(company),
        _normalize_company_key(vendor),
    }
    invalid = {_normalize_company_key(term) for term in invalid_terms if term}
    cleaned: list[str] = []
    seen: set[str] = set()
    if not isinstance(alternatives, list):
        return cleaned
    for item in alternatives:
        name = item.get("name") if isinstance(item, dict) else item
        value = str(name or "").strip()
        norm = _normalize_company_key(value)
        if not value or norm in seen or norm in blocked or norm in invalid:
            continue
        seen.add(norm)
        cleaned.append(value)
    return cleaned


def _account_reasoning_summary_payload(
    account_reasoning: dict[str, Any] | None,
) -> tuple[str, dict[str, int], list[str]]:
    """Derive readable summary fields from account reasoning."""
    if not isinstance(account_reasoning, dict):
        return "", {}, []
    summary = str(account_reasoning.get("market_summary") or "").strip()
    metrics: dict[str, int] = {}
    for key in ("total_accounts", "high_intent_count", "active_eval_count"):
        value = _reasoning_int(account_reasoning.get(key))
        if value is not None:
            metrics[key] = value
    priority_names: list[str] = []
    for item in account_reasoning.get("top_accounts") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name and name not in priority_names:
            priority_names.append(name)
    return summary, metrics, priority_names[:3]


def _normalize_account_pressure_metrics(
    result: dict[str, Any],
    accounts: list[dict[str, Any]],
) -> None:
    """Keep emitted account pressure metrics consistent with the account payload."""
    metrics = result.get("account_pressure_metrics")
    if not isinstance(metrics, dict):
        return
    normalized = dict(metrics)
    priority_names = result.get("priority_account_names")
    priority_count = len(priority_names) if isinstance(priority_names, list) else 0
    account_count = len(accounts)
    total_floor = max(account_count, priority_count)
    current_total = int(normalized.get("total_accounts") or 0)
    if total_floor and current_total < total_floor:
        normalized["total_accounts"] = total_floor
    if accounts:
        high_urgency_count = sum(1 for a in accounts if float(a.get("urgency") or 0) >= 7)
        active_eval_count = sum(1 for a in accounts if (a.get("buying_stage") or "").lower() in (
            "active_purchase", "evaluation", "renewal_decision",
        ))
        normalized["high_intent_count"] = max(
            int(normalized.get("high_intent_count") or 0),
            high_urgency_count,
        )
        normalized["active_eval_count"] = max(
            int(normalized.get("active_eval_count") or 0),
            active_eval_count,
        )
    result["account_pressure_metrics"] = normalized


def _apply_account_quality_adjustments(account: dict[str, Any], cfg) -> tuple[int, dict[str, int], list[str]]:
    """Apply quality bonuses/penalties to the base account score."""
    delta = 0
    components: dict[str, int] = {}
    flags: list[str] = []
    evidence_count = int(account.get("evidence_count") or 0)
    if evidence_count > 1:
        bonus = min(
            cfg.accounts_in_motion_repeat_evidence_bonus_max,
            (evidence_count - 1) * cfg.accounts_in_motion_repeat_evidence_bonus,
        )
        delta += bonus
        components["repeat_evidence"] = bonus
    confidence = account.get("confidence")
    if confidence is not None and float(confidence) < cfg.accounts_in_motion_low_confidence_threshold:
        penalty = cfg.accounts_in_motion_low_confidence_penalty
        delta -= penalty
        components["low_confidence"] = -penalty
        flags.append("low_confidence")
    if not account.get("domain"):
        delta -= cfg.accounts_in_motion_missing_domain_penalty
        components["missing_domain"] = -cfg.accounts_in_motion_missing_domain_penalty
        flags.append("missing_domain")
    if not account.get("title"):
        delta -= cfg.accounts_in_motion_missing_title_penalty
        components["missing_title"] = -cfg.accounts_in_motion_missing_title_penalty
        flags.append("missing_title")
    if not account.get("top_quote"):
        delta -= cfg.accounts_in_motion_missing_quote_penalty
        components["missing_quote"] = -cfg.accounts_in_motion_missing_quote_penalty
        flags.append("missing_quote")
    return delta, components, flags


def _build_fallback_intent_rows(
    churning_companies: list[dict[str, Any]],
    *,
    min_urgency: float,
) -> list[dict[str, Any]]:
    """Build high-intent-shaped fallback rows from churning-company aggregates."""
    fallback: list[dict[str, Any]] = []
    for vendor_row in churning_companies:
        vendor = str(vendor_row.get("vendor") or "").strip()
        if not vendor:
            continue
        for company_row in (vendor_row.get("companies") or []):
            company = str(company_row.get("company") or "").strip()
            if not company:
                continue
            try:
                urgency = float(company_row.get("urgency") or 0)
            except (TypeError, ValueError):
                urgency = 0.0
            if urgency < float(min_urgency):
                continue
            fallback.append({
                "company": company,
                "vendor": vendor,
                "category": None,
                "title": company_row.get("title"),
                "company_size": company_row.get("company_size"),
                "industry": company_row.get("industry"),
                "role_level": company_row.get("role"),
                "decision_maker": False,
                "urgency": urgency,
                "pain": company_row.get("pain"),
                "alternatives": [],
                "quotes": [],
                "contract_signal": None,
                "review_id": None,
                "source": "churning_companies_fallback",
                "seat_count": None,
                "contract_end": None,
                "buying_stage": None,
            })
    return fallback


def _build_signal_metadata_fallback_rows(
    signal_metadata: list[dict[str, Any]],
    *,
    min_urgency: float,
) -> list[dict[str, Any]]:
    """Build fallback intake rows from persisted company-signal metadata."""
    fallback: list[dict[str, Any]] = []
    for row in signal_metadata:
        company = str(row.get("company") or "").strip()
        vendor = str(row.get("vendor") or "").strip()
        if not company or not vendor:
            continue
        try:
            urgency = float(row.get("confidence") or 0)
        except (TypeError, ValueError):
            urgency = 0.0
        if urgency < float(min_urgency):
            continue
        fallback.append({
            "company": company,
            "vendor": vendor,
            "category": None,
            "title": None,
            "company_size": None,
            "industry": None,
            "role_level": None,
            "decision_maker": False,
            "urgency": urgency,
            "pain": None,
            "alternatives": [],
            "quotes": [],
            "contract_signal": None,
            "review_id": None,
            "source": "company_signals_fallback",
            "seat_count": None,
            "contract_end": None,
            "buying_stage": None,
        })
    return fallback


def _build_account_pool_seed_rows(
    account_pool_lookup: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Flatten canonical account-intelligence rows into merge seed records."""
    seed_rows: list[dict[str, Any]] = []
    for vendor, payload in (account_pool_lookup or {}).items():
        if not isinstance(payload, dict):
            continue
        for row in payload.get("accounts") or []:
            if not isinstance(row, dict):
                continue
            company = str(row.get("company_name") or row.get("company") or "").strip()
            if not company:
                continue
            seed_rows.append({
                "company": company,
                "vendor": vendor,
                "category": payload.get("category"),
                "title": row.get("title"),
                "company_size": row.get("company_size"),
                "industry": row.get("industry"),
                "role_level": row.get("buyer_role"),
                "decision_maker": bool(row.get("decision_maker")),
                "urgency": row.get("urgency_score"),
                "pain": row.get("pain_category"),
                "alternatives": row.get("alternatives") or [],
                "quotes": row.get("quotes") or [],
                "review_id": row.get("review_id"),
                "source": row.get("source") or "account_pool",
                "seat_count": row.get("seat_count"),
                "contract_end": row.get("contract_end"),
                "buying_stage": row.get("buying_stage"),
                "first_seen": row.get("first_seen_at"),
                "last_seen": row.get("last_seen_at"),
                "confidence": row.get("confidence_score"),
            })
    return seed_rows


# ---------------------------------------------------------------------------
# New lightweight query: company signal metadata
# ---------------------------------------------------------------------------

async def _fetch_apollo_org_lookup(pool) -> dict[str, dict[str, Any]]:
    """Load Apollo org data keyed by normalized company name.

    Includes 'enriched' rows and 'pending' rows that already have a domain
    (org data arrived via person reveal but person search is incomplete).
    """
    try:
        rows = await pool.fetch(
            """
            SELECT company_name_norm, domain, industry,
                   employee_count, annual_revenue_range
            FROM prospect_org_cache
            WHERE status = 'enriched'
               OR (status = 'pending' AND domain IS NOT NULL)
            """
        )
    except Exception:
        logger.debug("prospect_org_cache not available, skipping Apollo lookup")
        return {}
    return {
        r["company_name_norm"]: {
            "domain": r["domain"],
            "industry": r["industry"],
            "employee_count": r["employee_count"],
            "annual_revenue_range": r["annual_revenue_range"],
        }
        for r in rows
    }


async def _fetch_company_signal_metadata(pool, window_days: int = 90) -> list[dict[str, Any]]:
    """Read first_seen_at, last_seen_at, confidence from b2b_company_signals."""
    rows = await pool.fetch(
        """
        SELECT company_name, vendor_name,
               first_seen_at, last_seen_at,
               urgency_score AS confidence
        FROM b2b_company_signals
        WHERE last_seen_at > NOW() - make_interval(days => $1)
        """,
        window_days,
    )
    return [
        {
            "company": r["company_name"],
            "vendor": r["vendor_name"],
            "first_seen": r["first_seen_at"].isoformat() if r["first_seen_at"] else None,
            "last_seen": r["last_seen_at"].isoformat() if r["last_seen_at"] else None,
            "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
        }
        for r in rows
    ]


async def _fetch_latest_synthesis_views(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch best reasoning views (synthesis-first, legacy fallback).

    When ``vendor_names`` is provided, scopes to those vendors. Otherwise
    loads all vendors with synthesis rows, then fills gaps from legacy.
    """
    from ._b2b_synthesis_reader import discover_reasoning_vendor_names, load_best_reasoning_views

    if vendor_names:
        return await load_best_reasoning_views(
            pool, vendor_names,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )

    all_names = await discover_reasoning_vendor_names(
        pool, as_of=as_of, analysis_window_days=analysis_window_days,
    )
    if not all_names:
        return {}
    return await load_best_reasoning_views(
        pool, all_names, as_of=as_of, analysis_window_days=analysis_window_days,
    )


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _merge_company_profiles(
    high_intent: list[dict[str, Any]],
    timeline_signals: list[dict[str, Any]],
    churning_companies: list[dict[str, Any]],
    quotable_evidence: list[dict[str, Any]],
    signal_metadata: list[dict[str, Any]],
    *,
    seed_rows: list[dict[str, Any]] | None = None,
    min_urgency: float = 5.0,
    apollo_org_lookup: dict[str, dict[str, Any]] | None = None,
    invalid_alternative_terms: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge company profiles from multiple data sources.

    Key: (normalize_company_key(company), canonicalized vendor)

    Returns: {key: merged_profile_dict, ...}
    """
    from ._b2b_shared import _canonicalize_vendor

    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_alternative_terms = invalid_alternative_terms or []
    seed_rows = seed_rows or []

    for row in seed_rows:
        company = row.get("company") or ""
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        if not company or not vendor:
            continue
        urgency = float(row.get("urgency") or 0)
        if urgency < min_urgency:
            continue
        alts = _clean_alternatives(
            row.get("alternatives") or [],
            company=company,
            vendor=vendor,
            invalid_terms=invalid_alternative_terms,
        )
        review_ids: list[str] = []
        if row.get("review_id"):
            review_ids.append(str(row["review_id"]))
        account_quote = _extract_account_quote(row.get("quotes"))
        source_distribution = {row["source"]: 1} if row.get("source") else {}
        key = (_normalize_company_key(company), vendor)
        if key in profiles:
            continue
        profiles[key] = {
            "company": company,
            "vendor": vendor,
            "category": row.get("category"),
            "urgency": urgency,
            "pain_category": row.get("pain"),
            "role_level": row.get("role_level"),
            "decision_maker": bool(row.get("decision_maker")),
            "buying_stage": row.get("buying_stage"),
            "seat_count": row.get("seat_count"),
            "contract_end": row.get("contract_end"),
            "alternatives_considering": alts,
            "source_reviews": review_ids,
            "evidence_count": 0,
            "title": row.get("title"),
            "company_size": row.get("company_size"),
            "industry": row.get("industry"),
            "evaluation_deadline": None,
            "decision_timeline": None,
            "top_quote": account_quote,
            "quote_match_type": "review" if account_quote else None,
            "domain": None,
            "annual_revenue_range": None,
            "first_seen": row.get("first_seen"),
            "last_seen": row.get("last_seen"),
            "confidence": row.get("confidence"),
            "source_distribution": source_distribution,
            "pool_seeded": True,
        }

    # 1. Base from high_intent -- group by key, take highest-urgency
    for row in high_intent:
        company = row.get("company") or ""
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        if not company or not vendor:
            continue
        key = (_normalize_company_key(company), vendor)

        existing = profiles.get(key)
        urgency = float(row.get("urgency") or 0)

        if urgency < min_urgency:
            continue

        if existing is None:
            alts = _clean_alternatives(
                row.get("alternatives") or [],
                company=company,
                vendor=vendor,
                invalid_terms=invalid_alternative_terms,
            )
            review_ids = []
            if row.get("review_id"):
                review_ids.append(row["review_id"])
            account_quote = _extract_account_quote(row.get("quotes"))
            profiles[key] = {
                "company": company,
                "vendor": vendor,
                "category": row.get("category"),
                "urgency": urgency,
                "pain_category": row.get("pain"),
                "role_level": row.get("role_level"),
                "decision_maker": bool(row.get("decision_maker")),
                "buying_stage": row.get("buying_stage"),
                "seat_count": row.get("seat_count"),
                "contract_end": row.get("contract_end"),
                "alternatives_considering": alts,
                "source_reviews": review_ids,
                "evidence_count": 1,
                "title": row.get("title"),
                "company_size": row.get("company_size"),
                "industry": row.get("industry"),
                "evaluation_deadline": None,
                "decision_timeline": None,
                "top_quote": account_quote,
                "quote_match_type": "review" if account_quote else None,
                "domain": None,
                "annual_revenue_range": None,
                "first_seen": None,
                "last_seen": None,
                "confidence": None,
                "source_distribution": {row["source"]: 1} if row.get("source") else {},
            }
        else:
            # Merge: union alternatives, collect review_ids, max urgency
            existing["urgency"] = max(existing["urgency"], urgency)
            current_count = int(existing.get("evidence_count") or 0)
            if existing.get("pool_seeded") and current_count == 0:
                existing["evidence_count"] = 1
            else:
                existing["evidence_count"] = current_count + 1
            if row.get("review_id") and row["review_id"] not in existing["source_reviews"]:
                existing["source_reviews"].append(row["review_id"])
            # Merge alternatives
            new_alts = _clean_alternatives(
                row.get("alternatives") or [],
                company=company,
                vendor=vendor,
                invalid_terms=invalid_alternative_terms,
            )
            for name in new_alts:
                if name not in existing["alternatives_considering"]:
                    existing["alternatives_considering"].append(name)
            if row.get("source"):
                dist = existing.setdefault("source_distribution", {})
                dist[row["source"]] = dist.get(row["source"], 0) + 1
            # Fill nulls from higher-urgency row
            if row.get("category") and not existing.get("category"):
                existing["category"] = row["category"]
            if row.get("pain") and not existing.get("pain_category"):
                existing["pain_category"] = row["pain"]
            if row.get("decision_maker") and not existing.get("decision_maker"):
                existing["decision_maker"] = True
            if row.get("role_level") and not existing.get("role_level"):
                existing["role_level"] = row["role_level"]
            if row.get("buying_stage") and not existing.get("buying_stage"):
                existing["buying_stage"] = row["buying_stage"]
            if row.get("seat_count") and not existing.get("seat_count"):
                existing["seat_count"] = row["seat_count"]
            if row.get("contract_end") and not existing.get("contract_end"):
                existing["contract_end"] = row["contract_end"]
            if row.get("title") and not existing.get("title"):
                existing["title"] = row["title"]
            if row.get("company_size") and not existing.get("company_size"):
                existing["company_size"] = row["company_size"]
            if row.get("industry") and not existing.get("industry"):
                existing["industry"] = row["industry"]
            if not existing.get("top_quote"):
                account_quote = _extract_account_quote(row.get("quotes"))
                if account_quote:
                    existing["top_quote"] = account_quote
                    existing["quote_match_type"] = "review"

    # 2. Fill from timeline signals
    for row in timeline_signals:
        company = row.get("company") or ""
        vendor = _canonicalize_vendor(row.get("vendor") or "")
        key = (_normalize_company_key(company), vendor)
        prof = profiles.get(key)
        if prof is None:
            continue
        if row.get("contract_end") and not prof.get("contract_end"):
            prof["contract_end"] = row["contract_end"]
        if row.get("evaluation_deadline") and not prof.get("evaluation_deadline"):
            prof["evaluation_deadline"] = row["evaluation_deadline"]
        if row.get("decision_timeline") and not prof.get("decision_timeline"):
            prof["decision_timeline"] = row["decision_timeline"]
        if row.get("title") and not prof.get("title"):
            prof["title"] = row["title"]
        if row.get("company_size") and not prof.get("company_size"):
            prof["company_size"] = row["company_size"]
        if row.get("industry") and not prof.get("industry"):
            prof["industry"] = row["industry"]

    # 3. Fill from churning companies (vendor -> [{company, title, company_size, industry, ...}])
    for vendor_row in churning_companies:
        vendor = _canonicalize_vendor(vendor_row.get("vendor") or "")
        for c in (vendor_row.get("companies") or []):
            company = c.get("company") or ""
            key = (_normalize_company_key(company), vendor)
            prof = profiles.get(key)
            if prof is None:
                continue
            if c.get("title") and not prof.get("title"):
                prof["title"] = c["title"]
            if c.get("company_size") and not prof.get("company_size"):
                prof["company_size"] = c["company_size"]
            if c.get("industry") and not prof.get("industry"):
                prof["industry"] = c["industry"]

    # 4. Attach quote (highest-urgency per vendor, match by company)
    quote_by_vendor: dict[str, list[dict]] = {}
    for qr in quotable_evidence:
        vendor = _canonicalize_vendor(qr.get("vendor") or "")
        for q in (qr.get("quotes") or []):
            quote_by_vendor.setdefault(vendor, []).append(q)

    for key, prof in profiles.items():
        _, vendor = key
        company_key = key[0]
        quotes = quote_by_vendor.get(vendor, [])
        # Find quote matching this company
        best_quote = None
        best_urgency = -1
        for q in quotes:
            qc = _normalize_company_key(q.get("company") or "")
            if qc == company_key:
                qu = float(q.get("urgency") or 0)
                if qu > best_urgency:
                    best_urgency = qu
                    best_quote = q.get("quote")
        if best_quote:
            prof["top_quote"] = best_quote
            prof["quote_match_type"] = "company_match"

    # 5. Attach metadata from b2b_company_signals
    meta_lookup: dict[tuple[str, str], dict] = {}
    for m in signal_metadata:
        mk = (_normalize_company_key(m.get("company") or ""), _canonicalize_vendor(m.get("vendor") or ""))
        meta_lookup[mk] = m

    for key, prof in profiles.items():
        meta = meta_lookup.get(key)
        if meta:
            prof["first_seen"] = meta.get("first_seen")
            prof["last_seen"] = meta.get("last_seen")
            if meta.get("confidence") is not None and prof.get("confidence") is None:
                prof["confidence"] = meta["confidence"]

    # 6. Fill from Apollo org cache (lowest priority -- review data wins)
    #    Apollo keys use normalize_company_name (strips legal suffixes),
    #    profile keys use _normalize_company_key (strip+lower only).
    #    Try both to maximize matches.
    if apollo_org_lookup:
        from ...services.company_normalization import normalize_company_name

        for key, prof in profiles.items():
            company_key = key[0]  # _normalize_company_key output
            apollo = apollo_org_lookup.get(company_key)
            if not apollo:
                apollo = apollo_org_lookup.get(normalize_company_name(prof.get("company") or ""))
            if not apollo:
                continue
            if not prof.get("industry") and apollo.get("industry"):
                prof["industry"] = apollo["industry"]
            if not prof.get("company_size") and apollo.get("employee_count"):
                prof["company_size"] = str(apollo["employee_count"])
            if apollo.get("domain"):
                prof["domain"] = apollo["domain"]
            if apollo.get("annual_revenue_range"):
                prof["annual_revenue_range"] = apollo["annual_revenue_range"]

    for prof in profiles.values():
        prof.pop("pool_seeded", None)

    return profiles


# ---------------------------------------------------------------------------
# Aggregate builder
# ---------------------------------------------------------------------------


def _summarize_vendor_evidence(accounts: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
    """Summarize vendor-level supporting evidence from merged account rows."""
    review_ids: set[str] = set()
    source_distribution: dict[str, int] = {}
    for account in accounts:
        for review_id in account.get("source_reviews") or []:
            if review_id:
                review_ids.add(str(review_id))
        for source, count in (account.get("source_distribution") or {}).items():
            source_distribution[source] = source_distribution.get(source, 0) + int(count or 0)
    return len(review_ids), source_distribution


def _accounts_in_motion_feature_gaps_from_evidence_vault(
    vault: dict[str, Any] | None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Map canonical vault feature gaps into accounts-in-motion aggregate shape."""
    gaps: list[dict[str, Any]] = []
    for item in ((vault or {}).get("weakness_evidence") or []):
        if str(item.get("evidence_type") or "") != "feature_gap":
            continue
        label = str(item.get("label") or item.get("key") or "").strip()
        if not label:
            continue
        gaps.append({
            "feature": label,
            "mentions": int(item.get("mention_count_total") or 0),
        })
    gaps.sort(key=lambda row: int(row.get("mentions") or 0), reverse=True)
    return gaps[:limit]


def _infer_top_destination_from_accounts(
    vendor: str,
    accounts: list[dict[str, Any]],
) -> str | None:
    """Infer likely top destination from account-level alternatives."""
    vendor_norm = _normalize_company_key(vendor)
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for account in accounts:
        for alt in (account.get("alternatives_considering") or []):
            if not isinstance(alt, str):
                continue
            name = alt.strip()
            norm = _normalize_company_key(name)
            if not norm or norm == vendor_norm:
                continue
            counts[norm] += 1
            display.setdefault(norm, name)
    if not counts:
        return None
    top_norm, _ = counts.most_common(1)[0]
    return display.get(top_norm)


def _build_vendor_aggregate(
    vendor: str,
    accounts: list[dict[str, Any]],
    *,
    category: str | None = None,
    reasoning_lookup: dict[str, dict],
    xv_lookup: dict[str, dict],
    feature_gap_lookup: dict[str, list[dict]],
    price_lookup: dict[str, float],
    budget_lookup: dict[str, dict],
    competitor_lookup: dict[str, list[dict]],
    evidence_vault_lookup: dict[str, dict[str, Any]] | None = None,
    account_pool_lookup: dict[str, dict[str, Any]] | None = None,
    synthesis_views: dict[str, Any] | None = None,
    requested_as_of: date | None = None,
) -> dict[str, Any]:
    """Build the full accounts_in_motion aggregate for one vendor."""
    source_review_count, source_distribution = _summarize_vendor_evidence(accounts)
    vendor_vault = (evidence_vault_lookup or {}).get(vendor, {}) or {}
    evidence_vault_used = False
    # Archetype from reasoning
    r_info = reasoning_lookup.get(vendor, {})
    archetype = r_info.get("archetype")
    archetype_confidence = r_info.get("confidence", 0)

    # Feature gaps (top 10)
    feature_gaps = _accounts_in_motion_feature_gaps_from_evidence_vault(vendor_vault)
    if feature_gaps:
        evidence_vault_used = True
    else:
        vendor_gaps = feature_gap_lookup.get(vendor, [])[:10]
        feature_gaps = [
            {"feature": g.get("feature", ""), "mentions": g.get("mentions", 0)}
            for g in vendor_gaps
        ]

    # Pricing pressure
    price_rate = price_lookup.get(vendor, 0)
    if not price_rate:
        metric_snapshot = vendor_vault.get("metric_snapshot") or {}
        if metric_snapshot.get("price_complaint_rate") is not None:
            price_rate = float(metric_snapshot.get("price_complaint_rate") or 0)
            evidence_vault_used = True
    budget = budget_lookup.get(vendor, {})
    pricing_pressure = {
        "price_complaint_rate": round(price_rate, 3),
        "price_increase_rate": round(budget.get("price_increase_rate", 0), 3),
        "avg_seat_count": round(budget.get("avg_seat_count") or 0),
    }

    # Cross-vendor context
    top_dest = None
    competitors = competitor_lookup.get(vendor, [])
    if competitors:
        top_dest = competitors[0].get("name")
    if not top_dest:
        top_dest = _infer_top_destination_from_accounts(vendor, accounts)

    # Battle conclusion from xv_lookup
    battle_conclusion = None
    market_regime = None
    category_council = None
    preferred_pair = None
    if top_dest:
        preferred_pair = tuple(sorted((vendor, top_dest)))
    preferred_battle = None
    if preferred_pair:
        preferred_battle = (xv_lookup.get("battles", {}) or {}).get(preferred_pair)
    if preferred_battle:
        bc = preferred_battle.get("conclusion", {})
        battle_conclusion = bc.get("conclusion", "")
    else:
        for pair_key, battle in xv_lookup.get("battles", {}).items():
            if vendor in pair_key:
                bc = battle.get("conclusion", {})
                battle_conclusion = bc.get("conclusion", "")
                break
    if category:
        council = xv_lookup.get("councils", {}).get(category, {})
        cc = council.get("conclusion", {})
        if any(
            [
                cc.get("winner"),
                cc.get("loser"),
                cc.get("conclusion"),
                cc.get("market_regime"),
                cc.get("key_insights"),
            ]
        ):
            category_council = {
                "winner": cc.get("winner") or "",
                "loser": cc.get("loser") or "",
                "conclusion": cc.get("conclusion") or "",
                "market_regime": cc.get("market_regime") or "",
                "durability": cc.get("durability_assessment") or "",
                "confidence": council.get("confidence"),
                "key_insights": cc.get("key_insights") or [],
            }
        if cc.get("market_regime"):
            market_regime = cc["market_regime"]

    # Synthesize battle_conclusion from displacement data if missing
    if not battle_conclusion and top_dest and competitors:
        top_comp = competitors[0]
        mentions = top_comp.get("mentions", 0)
        driver = top_comp.get("primary_driver", "")
        parts = ["%s displacing %s" % (top_dest, vendor)]
        if mentions:
            parts.append("%d mentions" % mentions)
        if driver:
            parts.append("driven by %s" % driver)
        battle_conclusion = ", ".join(parts) + "."
    if not battle_conclusion and top_dest:
        battle_conclusion = (
            "%s appears in active evaluation sets against %s based on account-level alternatives."
            % (top_dest, vendor)
        )
    if not battle_conclusion:
        battle_conclusion = "No directional displacement evidence found in the current analysis window."

    # Synthesize market_regime from category if missing
    if not market_regime and category:
        market_regime = "active_displacement"

    cross_vendor_context = {
        "top_destination": top_dest,
        "battle_conclusion": battle_conclusion,
        "market_regime": market_regime,
    }

    result = {
        "vendor": vendor,
        "category": category,
        "archetype": archetype,
        "archetype_confidence": round(archetype_confidence, 2) if archetype_confidence else None,
        "total_accounts_in_motion": len(accounts),
        "accounts": accounts,
        "pricing_pressure": pricing_pressure,
        "feature_gaps": feature_gaps,
        "cross_vendor_context": cross_vendor_context,
        "category_council": category_council,
        "source_review_count": source_review_count,
        "source_distribution": source_distribution,
        "data_sources": {
            "evidence_vault": evidence_vault_used,
            "account_pool": bool((account_pool_lookup or {}).get(vendor)),
        },
    }
    view = (synthesis_views or {}).get(vendor)
    if view is None:
        view = (synthesis_views or {}).get(_normalize_company_key(vendor))
    if view is not None:
        from ._b2b_synthesis_reader import (
            contract_gaps_for_consumer,
            inject_synthesis_freshness,
        )
        reasoning_contracts: dict[str, Any] = {}
        materialized_contracts = getattr(view, "materialized_contracts", None)
        if callable(materialized_contracts):
            reasoning_contracts = materialized_contracts() or {}
        else:
            for name in (
                "vendor_core_reasoning",
                "displacement_reasoning",
                "category_reasoning",
                "account_reasoning",
            ):
                contract = view.contract(name)
                if contract:
                    reasoning_contracts[name] = contract
            if reasoning_contracts:
                reasoning_contracts["schema_version"] = "v1"
        if reasoning_contracts:
            result["reasoning_contracts"] = reasoning_contracts
            reference_ids = getattr(view, "reference_ids", None)
            if isinstance(reference_ids, dict) and reference_ids:
                result["reference_ids"] = reference_ids
            vendor_core = reasoning_contracts.get("vendor_core_reasoning") or {}
            account_reasoning = reasoning_contracts.get("account_reasoning") or {}
            if isinstance(account_reasoning, dict) and account_reasoning:
                result["account_reasoning"] = account_reasoning
                summary_text, summary_metrics, priority_names = (
                    _account_reasoning_summary_payload(account_reasoning)
                )
                if summary_text:
                    result["account_pressure_summary"] = summary_text
                if summary_metrics:
                    result["account_pressure_metrics"] = summary_metrics
                if priority_names:
                    result["priority_account_names"] = priority_names
            if isinstance(vendor_core, dict):
                segment_playbook = vendor_core.get("segment_playbook") or {}
                timing_intelligence = vendor_core.get("timing_intelligence") or {}
                if isinstance(timing_intelligence, dict) and timing_intelligence:
                    result["timing_intelligence"] = timing_intelligence
                    timing_summary, timing_metrics, priority_triggers = (
                        _timing_summary_payload(timing_intelligence)
                    )
                    if timing_summary:
                        result["timing_summary"] = timing_summary
                    if timing_metrics:
                        result["timing_metrics"] = timing_metrics
                    if priority_triggers:
                        result["priority_timing_triggers"] = priority_triggers
                if isinstance(segment_playbook, dict) and segment_playbook:
                    result["segment_playbook"] = segment_playbook
                    targeting_summary = _segment_targeting_summary(
                        segment_playbook,
                        timing_intelligence if isinstance(timing_intelligence, dict) else None,
                    )
                    if targeting_summary:
                        result["segment_targeting_summary"] = targeting_summary
        if view.primary_wedge:
            result["synthesis_wedge"] = view.primary_wedge.value
            result["synthesis_wedge_label"] = view.wedge_label
        if view.meta:
            result["evidence_window"] = view.meta
        result["synthesis_schema_version"] = view.schema_version
        result["reasoning_source"] = "b2b_reasoning_synthesis"
        inject_synthesis_freshness(
            result,
            view,
            requested_as_of=requested_as_of,
        )
        contract_gaps = contract_gaps_for_consumer(view, "accounts_in_motion")
        if contract_gaps:
            result["reasoning_contract_gaps"] = contract_gaps

        # Phase 3 governance fields
        wts = getattr(view, "why_they_stay", None)
        if isinstance(wts, dict) and wts:
            result["why_they_stay"] = wts
        cp = getattr(view, "confidence_posture", None)
        if isinstance(cp, dict) and cp:
            result["confidence_posture"] = cp
            limits = cp.get("limits")
            if limits:
                result["confidence_limits"] = limits
        st = getattr(view, "switch_triggers", None)
        if isinstance(st, list) and st:
            result["switch_triggers"] = st
        cg = getattr(view, "coverage_gaps", None)
        if isinstance(cg, list) and cg:
            result["coverage_gaps"] = cg

    # Backfill account-pressure fields from raw accounts when synthesis didn't
    # populate them.  Uses the same field names so consumers see a consistent
    # schema regardless of the data path.
    if not result.get("account_pressure_summary") and accounts:
        high_urgency = [a for a in accounts if float(a.get("urgency") or 0) >= 7]
        active_eval = [a for a in accounts if (a.get("buying_stage") or "").lower() in (
            "active_purchase", "evaluation", "renewal_decision",
        )]
        top_pain = ""
        pain_counts: dict[str, int] = {}
        for a in accounts:
            pc = a.get("pain_category") or ""
            if pc:
                pain_counts[pc] = pain_counts.get(pc, 0) + 1
        if pain_counts:
            top_pain = max(pain_counts, key=pain_counts.get)  # type: ignore[arg-type]
        parts = [f"{len(accounts)} accounts in motion"]
        if high_urgency:
            parts.append(f"{len(high_urgency)} high-urgency")
        if active_eval:
            parts.append(f"{len(active_eval)} in active evaluation")
        if top_pain:
            parts.append(f"top pain: {top_pain}")
        result["account_pressure_summary"] = ", ".join(parts) + "."
    if not result.get("account_pressure_metrics") and accounts:
        high_urgency_count = sum(1 for a in accounts if float(a.get("urgency") or 0) >= 7)
        active_eval_count = sum(1 for a in accounts if (a.get("buying_stage") or "").lower() in (
            "active_purchase", "evaluation", "renewal_decision",
        ))
        result["account_pressure_metrics"] = {
            "total_accounts": len(accounts),
            "high_intent_count": high_urgency_count,
            "active_eval_count": active_eval_count,
        }
    if not result.get("priority_account_names") and accounts:
        result["priority_account_names"] = [
            a.get("company") or a.get("company_name") or ""
            for a in sorted(accounts, key=lambda a: a.get("opportunity_score", 0), reverse=True)[:3]
            if a.get("company") or a.get("company_name")
        ]
    _normalize_account_pressure_metrics(result, accounts)
    return result


# ---------------------------------------------------------------------------
# Freshness check (same pattern as battle cards)
# ---------------------------------------------------------------------------

async def _check_freshness(pool) -> date | None:
    """Return today's date if core task wrote a completion marker, else None."""
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
# Main task entry point
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict[str, Any]:
    """Build per-vendor accounts-in-motion prospecting lists."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.intelligence_enabled:
        return {"_skip_synthesis": "B2B churn intelligence disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    today = await _check_freshness(pool)
    if today is None:
        return {"_skip_synthesis": "Core signals not fresh for today"}

    from ._b2b_shared import (
        _build_competitor_lookup,
        _build_feature_gap_lookup,
        _canonicalize_vendor,
        _fetch_budget_signals,
        _fetch_churning_companies,
        _fetch_competitive_displacement_source_of_truth,
        _fetch_latest_account_intelligence,
        _fetch_latest_evidence_vault,
        _fetch_feature_gaps,
        _fetch_high_intent_companies,
        _fetch_price_complaint_rates,
        _fetch_quotable_evidence,
        _fetch_timeline_signals,
        _aggregate_competitive_disp,
    )
    from .b2b_churn_intelligence import (
        _normalize_test_vendors,
        reconstruct_reasoning_lookup,
    )

    window_days = cfg.intelligence_window_days
    min_urgency = cfg.accounts_in_motion_min_urgency
    max_per_vendor = cfg.accounts_in_motion_max_per_vendor
    scoped_vendors = _normalize_test_vendors((task.metadata or {}).get("test_vendors"))
    vendor_scope = {vendor.lower() for vendor in scoped_vendors}

    # --- Phase 1: Parallel data fetch ---
    await _update_execution_progress(
        task,
        stage=_STAGE_LOADING_INPUTS,
        progress_message="Loading accounts-in-motion source artifacts.",
    )
    try:
        (
            high_intent,
            timeline_signals,
            churning_companies,
            quotable_evidence,
            feature_gaps,
            price_rates,
            budget_signals,
            competitive_disp,
            signal_metadata,
            apollo_org_lookup,
            account_pool_lookup,
        ) = await asyncio.gather(
            _fetch_high_intent_companies(pool, int(min_urgency), window_days),
            _fetch_timeline_signals(pool, window_days),
            _fetch_churning_companies(pool, window_days),
            _fetch_quotable_evidence(pool, window_days, min_urgency=cfg.quotable_phrase_min_urgency),
            _fetch_feature_gaps(pool, window_days, min_mentions=cfg.feature_gap_min_mentions),
            _fetch_price_complaint_rates(pool, window_days),
            _fetch_budget_signals(pool, window_days),
            _fetch_competitive_displacement_source_of_truth(
                pool,
                as_of=today,
                analysis_window_days=window_days,
            ),
            _fetch_company_signal_metadata(pool, window_days),
            _fetch_apollo_org_lookup(pool),
            _fetch_latest_account_intelligence(
                pool,
                as_of=today,
                analysis_window_days=window_days,
            ),
        )
    except Exception:
        logger.exception("Accounts in motion data fetch failed")
        return {"_skip_synthesis": "Data fetch failed"}

    fallback_intent = _build_fallback_intent_rows(
        churning_companies,
        min_urgency=min_urgency,
    )
    signal_fallback_intent = _build_signal_metadata_fallback_rows(
        signal_metadata,
        min_urgency=min_urgency,
    )
    seed_rows = _build_account_pool_seed_rows(account_pool_lookup)
    combined_intent: list[dict[str, Any]] = list(high_intent)
    existing_keys = {
        (_normalize_company_key(row.get("company")), _canonicalize_vendor(row.get("vendor")))
        for row in high_intent
        if row.get("company") and row.get("vendor")
    }
    added_fallback = 0
    for row in [*fallback_intent, *signal_fallback_intent]:
        key = (
            _normalize_company_key(row.get("company")),
            _canonicalize_vendor(row.get("vendor")),
        )
        if not key[0] or not key[1] or key in existing_keys:
            continue
        existing_keys.add(key)
        combined_intent.append(row)
        added_fallback += 1
    if added_fallback:
        logger.info(
            "Accounts in motion: added %d fallback high-intent rows from churning companies",
            added_fallback,
        )

    if vendor_scope:
        raw_intent_count = len(combined_intent)
        raw_seed_count = len(seed_rows)
        combined_intent = [
            row for row in combined_intent
            if str(row.get("vendor") or "").strip().lower() in vendor_scope
        ]
        seed_rows = [
            row for row in seed_rows
            if str(row.get("vendor") or "").strip().lower() in vendor_scope
        ]
        logger.info(
            "Scoped accounts in motion to %d vendors: intent rows %d/%d, seed rows %d/%d",
            len(scoped_vendors), len(combined_intent), raw_intent_count,
            len(seed_rows), raw_seed_count,
        )

    if not combined_intent and not seed_rows:
        logger.info("No high-intent companies found, skipping")
        return {"_skip_synthesis": "No high-intent companies"}

    competitive_disp = _aggregate_competitive_disp(competitive_disp)
    try:
        evidence_vault_lookup = await _fetch_latest_evidence_vault(
            pool,
            as_of=today,
            analysis_window_days=window_days,
        )
    except Exception:
        logger.exception("Accounts in motion evidence-vault fetch failed")
        evidence_vault_lookup = {}

    # --- Phase 2: Reconstruct reasoning (synthesis-first) + cross-vendor ---
    from ._b2b_cross_vendor_synthesis import load_best_cross_vendor_lookup

    xv_lookup = await load_best_cross_vendor_lookup(
        pool,
        as_of=today,
        analysis_window_days=window_days,
    )
    try:
        raw_synthesis_views = await _fetch_latest_synthesis_views(
            pool,
            as_of=today,
            analysis_window_days=window_days,
        )
    except Exception:
        logger.exception("Accounts in motion synthesis lookup failed")
        raw_synthesis_views = {}
    # Build reasoning_lookup from synthesis views first, legacy fallback
    from ._b2b_synthesis_reader import build_reasoning_lookup_from_views
    legacy_lookup = await reconstruct_reasoning_lookup(pool, as_of=today)
    synth_lookup = build_reasoning_lookup_from_views(raw_synthesis_views)
    reasoning_lookup = {**legacy_lookup, **synth_lookup}
    synthesis_views = {
        _canonicalize_vendor(vendor): view
        for vendor, view in raw_synthesis_views.items()
        if _canonicalize_vendor(vendor)
    }

    # --- Phase 3: Merge company profiles ---
    await _update_execution_progress(
        task,
        stage=_STAGE_BUILDING_ACCOUNTS,
        progress_message="Building merged account profiles and vendor aggregates.",
        high_intent_companies=len(combined_intent),
    )
    merged = _merge_company_profiles(
        combined_intent,
        timeline_signals,
        churning_companies,
        quotable_evidence,
        signal_metadata,
        seed_rows=seed_rows,
        min_urgency=min_urgency,
        apollo_org_lookup=apollo_org_lookup,
        invalid_alternative_terms=cfg.accounts_in_motion_invalid_alternative_terms,
    )

    # --- Phase 4: Score each account ---
    for key, prof in merged.items():
        base_score, components = _compute_account_opportunity_score(prof)
        delta, quality_components, quality_flags = _apply_account_quality_adjustments(prof, cfg)
        prof["opportunity_score"] = max(0, min(100, base_score + delta))
        prof["score_components"] = {**components, **quality_components}
        prof["quality_flags"] = quality_flags

    # --- Phase 5: Build per-vendor aggregates ---
    # Group accounts by vendor
    vendor_accounts: dict[str, list[dict[str, Any]]] = {}
    vendor_category_counts: dict[str, Counter[str]] = {}
    for key, prof in merged.items():
        vendor = prof["vendor"]
        vendor_accounts.setdefault(vendor, []).append(prof)
        category = prof.get("category")
        if category:
            vendor_category_counts.setdefault(vendor, Counter())[category] += 1

    # Sort accounts per vendor by score DESC, limit to max_per_vendor
    for vendor in vendor_accounts:
        vendor_accounts[vendor].sort(key=lambda a: a.get("opportunity_score", 0), reverse=True)
        vendor_accounts[vendor] = vendor_accounts[vendor][:max_per_vendor]

    # Build lookups (canonicalize vendor keys to match merged profiles)
    feature_gap_lookup = _build_feature_gap_lookup(feature_gaps)
    price_lookup = {_canonicalize_vendor(r["vendor"]): r["price_complaint_rate"] for r in price_rates}
    budget_lookup = {
        _canonicalize_vendor(r["vendor"]): {k: v for k, v in r.items() if k != "vendor"}
        for r in budget_signals
    }
    competitor_lookup = _build_competitor_lookup(competitive_disp)

    aggregates: list[dict[str, Any]] = []
    for vendor, accounts in vendor_accounts.items():
        agg = _build_vendor_aggregate(
            vendor,
            accounts,
            category=vendor_category_counts.get(vendor, Counter()).most_common(1)[0][0]
            if vendor_category_counts.get(vendor)
            else None,
            reasoning_lookup=reasoning_lookup,
            xv_lookup=xv_lookup,
            feature_gap_lookup=feature_gap_lookup,
            price_lookup=price_lookup,
            budget_lookup=budget_lookup,
            competitor_lookup=competitor_lookup,
            evidence_vault_lookup=evidence_vault_lookup,
            account_pool_lookup=account_pool_lookup,
            synthesis_views=synthesis_views,
            requested_as_of=today,
        )
        aggregates.append(agg)

    # --- Phase 6: Persist ---
    total_accounts = sum(a["total_accounts_in_motion"] for a in aggregates)
    persisted = 0
    await _update_execution_progress(
        task,
        stage=_STAGE_PERSISTING_REPORTS,
        progress_current=0,
        progress_total=len(aggregates),
        progress_message="Persisting accounts-in-motion vendor reports.",
        vendors=len(aggregates),
        total_accounts=total_accounts,
        persisted=0,
    )
    for agg in aggregates:
        vendor = agg["vendor"]
        if not vendor:
            continue
        n_accounts = agg["total_accounts_in_motion"]
        top_score = agg["accounts"][0]["opportunity_score"] if agg["accounts"] else 0
        summary_parts = [
            (
                f"Accounts in motion for {vendor}: "
                f"{n_accounts} accounts, "
                f"top score {top_score}, "
                f"archetype {agg.get('archetype') or 'unknown'}."
            ),
        ]
        if agg.get("account_pressure_summary"):
            summary_parts.append(str(agg["account_pressure_summary"]))
        if agg.get("timing_summary"):
            summary_parts.append(str(agg["timing_summary"]))
        priority_names = agg.get("priority_account_names") or []
        if priority_names:
            summary_parts.append(
                "Priority accounts: %s." % ", ".join(priority_names[:3])
            )
        exec_summary = " ".join(summary_parts)
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
                "accounts_in_motion",
                vendor,
                json.dumps(agg, default=str),
                exec_summary,
                json.dumps({"vendors_analyzed": len(aggregates), "total_accounts": total_accounts}),
                "published" if agg.get("total_accounts_in_motion", 0) > 0 else "failed",
                "pipeline_deterministic",
                agg.get("source_review_count", 0),
                json.dumps(agg.get("source_distribution", {})),
            )
            persisted += 1
            await _update_execution_progress(
                task,
                stage=_STAGE_PERSISTING_REPORTS,
                progress_current=persisted,
                progress_total=len(aggregates),
                progress_message="Persisting accounts-in-motion vendor reports.",
                vendors=len(aggregates),
                total_accounts=total_accounts,
                persisted=persisted,
            )
        except Exception:
            logger.exception("Failed to persist accounts_in_motion for %s", vendor)

    logger.info(
        "Accounts in motion: %d vendors, %d total accounts",
        len(aggregates),
        total_accounts,
    )
    await _update_execution_progress(
        task,
        stage=_STAGE_FINALIZING,
        progress_current=len(aggregates),
        progress_total=len(aggregates),
        progress_message="Finalizing accounts-in-motion execution status.",
        vendors=len(aggregates),
        total_accounts=total_accounts,
        persisted=persisted,
    )

    return {
        "_skip_synthesis": "Accounts in motion complete",
        "vendors": len(aggregates),
        "total_accounts": total_accounts,
        "persisted": persisted,
    }
