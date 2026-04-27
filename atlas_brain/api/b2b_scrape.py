"""
REST API for B2B scrape target management.

CRUD for scrape targets, manual trigger, and log viewing.
"""

import asyncio
import json
import logging
import time as _time
from datetime import date
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import settings
from ..services.scraping.source_yield import (
    disable_persistently_blocked_targets,
    prune_low_yield_targets,
)
from ..services.scraping.source_fit import classify_source_fit
from ..services.scraping.target_planning import build_scrape_coverage_plan
from ..services.scraping.target_provisioning import (
    apply_missing_core_targets,
    fetch_coverage_inputs,
    provision_vendor_onboarding_targets,
)
from ..services.scraping.target_validation import validate_target_input
from ..services.scraping.sources import (
    filter_blocked_sources,
    filter_deprecated_sources,
    parse_source_allowlist,
    REQUIRED_SCRAPE_SOURCES,
    with_required_sources,
)
from ..services.vendor_registry import resolve_vendor_name
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_scrape")
_REVIEW_BASIS_RAW_TARGET_PROVENANCE = "raw_source_target_provenance"

router = APIRouter(prefix="/b2b/scrape", tags=["b2b-scrape"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ScrapeTargetCreate(BaseModel):
    source: str = Field(description="Source: g2, capterra, trustradius, gartner, peerspot, getapp, software_advice, trustpilot, reddit, hackernews, github, stackoverflow, slashdot, rss")
    vendor_name: str
    product_name: Optional[str] = None
    product_slug: str
    product_category: Optional[str] = None
    max_pages: int = Field(default=5, ge=1, le=100)
    enabled: bool = True
    priority: int = Field(default=0, ge=0, le=100)
    scrape_interval_hours: int = Field(default=168, ge=1, le=8760)
    scrape_mode: str = Field(default="incremental", pattern="^(incremental|exhaustive)$")
    metadata: dict = Field(default_factory=dict)


class ScrapeTargetUpdate(BaseModel):
    enabled: Optional[bool] = None
    priority: Optional[int] = Field(default=None, ge=0, le=100)
    max_pages: Optional[int] = Field(default=None, ge=1, le=100)
    scrape_interval_hours: Optional[int] = Field(default=None, ge=1, le=8760)
    scrape_mode: Optional[str] = Field(default=None, pattern="^(incremental|exhaustive)$")
    metadata: Optional[dict] = None


class SeedMissingCoreRequest(BaseModel):
    dry_run: bool = True
    limit: int = Field(default=200, ge=1, le=1000)
    vendors: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_tiers: list[str] = Field(default_factory=list)
    verticals: list[str] = Field(default_factory=list)


class VendorOnboardingTargetsRequest(BaseModel):
    vendor_name: str
    product_category: Optional[str] = None
    source_slug_overrides: dict[str, str] = Field(default_factory=dict)
    dry_run: bool = True
    limit: int = Field(default=200, ge=1, le=1000)


class DisablePoorFitRequest(BaseModel):
    dry_run: bool = True
    limit: int = Field(default=1000, ge=1, le=5000)
    vendors: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_tiers: list[str] = Field(default_factory=list)
    verticals: list[str] = Field(default_factory=list)
    include_overrides: bool = False


class SeedConditionalProbationRequest(BaseModel):
    dry_run: bool = True
    limit: int = Field(default=100, ge=1, le=1000)
    vendors: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_tiers: list[str] = Field(default_factory=list)
    verticals: list[str] = Field(default_factory=list)


class RunProbationBatchRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=200)
    source: Optional[str] = None
    due_only: bool = True


class DisableLowYieldProbationRequest(BaseModel):
    dry_run: bool = True
    limit: int = Field(default=500, ge=1, le=5000)
    lookback_days: Optional[int] = Field(default=None, ge=1, le=365)
    min_runs: int = Field(default=1, ge=1, le=100)
    max_reviews_inserted: int = Field(default=0, ge=0, le=10000)
    max_tracked_reviews: int = Field(default=0, ge=0, le=10000)
    max_named_company_reviews: Optional[int] = Field(default=None, ge=0, le=10000)
    max_actionable_reviews: Optional[int] = Field(default=None, ge=0, le=10000)
    max_company_signal_reviews: Optional[int] = Field(default=None, ge=0, le=10000)
    statuses: list[str] = Field(default_factory=lambda: ["failed", "blocked"])
    vendors: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_tiers: list[str] = Field(default_factory=list)
    verticals: list[str] = Field(default_factory=list)


class DisableLowYieldSourceRequest(BaseModel):
    dry_run: bool = True
    source: Optional[str] = None
    lookback_runs: Optional[int] = Field(default=None, ge=1, le=20)
    min_runs: Optional[int] = Field(default=None, ge=1, le=20)
    max_inserted_total: Optional[int] = Field(default=None, ge=0, le=10000)
    max_disable_per_run: Optional[int] = Field(default=None, ge=1, le=1000)
    enabled_only: bool = True


class DisableBlockedTargetsRequest(BaseModel):
    dry_run: bool = True
    min_blocked_age_hours: Optional[int] = Field(default=None, ge=1, le=2160)
    max_disable_per_run: Optional[int] = Field(default=None, ge=1, le=1000)
    vendors: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class PromoteProbationTargetsRequest(BaseModel):
    dry_run: bool = True
    limit: int = Field(default=500, ge=1, le=5000)
    lookback_days: Optional[int] = Field(default=None, ge=1, le=365)
    min_runs: int = Field(default=1, ge=1, le=100)
    min_reviews_inserted: int = Field(default=1, ge=0, le=10000)
    min_tracked_reviews: int = Field(default=1, ge=0, le=10000)
    min_actionable_reviews: int = Field(default=0, ge=0, le=10000)
    min_company_signal_reviews: int = Field(default=0, ge=0, le=10000)
    require_any_downstream_signal: bool = True
    statuses: list[str] = Field(default_factory=lambda: ["success", "partial"])
    vendors: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_tiers: list[str] = Field(default_factory=list)
    verticals: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _normalize_filters(values: list[str]) -> set[str]:
    return {value.strip().lower() for value in values if value and value.strip()}


def _current_allowed_sources() -> list[str]:
    deprecated_sources = set(
        parse_source_allowlist(getattr(settings.b2b_scrape, "deprecated_sources", ""))
    ) - set(REQUIRED_SCRAPE_SOURCES)
    sources = filter_deprecated_sources(
        with_required_sources(
            parse_source_allowlist(settings.b2b_scrape.source_allowlist),
            required=REQUIRED_SCRAPE_SOURCES,
        ),
        deprecated_sources,
    )
    return sorted(
        filter_blocked_sources(
            sources,
            getattr(settings.b2b_scrape, "infra_blocked_sources", ""),
        )
    )


def _matches_filters(
    item: dict[str, Any],
    vendors: set[str],
    sources: set[str],
    source_tiers: set[str],
    verticals: set[str],
) -> bool:
    vendor = str(item.get("vendor_name") or "").strip().lower()
    source = str(item.get("source") or "").strip().lower()
    source_tier = str(item.get("source_tier") or "").strip().lower()
    vertical = str(item.get("vertical") or "").strip().lower()
    if vendors and vendor not in vendors:
        return False
    if sources and source not in sources:
        return False
    if source_tiers and source_tier not in source_tiers:
        return False
    if verticals and vertical not in verticals:
        return False
    return True


async def _fetch_coverage_inputs(pool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return await fetch_coverage_inputs(pool)


async def _fetch_existing_targets(pool) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT id, source, vendor_name, product_name, product_category, product_slug,
               enabled, scrape_mode, priority, max_pages, scrape_interval_hours, metadata
        FROM b2b_scrape_targets
        """
    )
    targets: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["metadata"] = _safe_target_metadata(item.get("metadata"))
        targets.append(item)
    return targets


def _derive_seed_defaults(
    existing_targets: list[dict[str, Any]],
    source: str,
    product_category: str | None,
) -> dict[str, Any]:
    from ..services.scraping.target_provisioning import derive_seed_defaults

    return derive_seed_defaults(existing_targets, source, product_category)


def _derive_promotion_defaults(
    existing_targets: list[dict[str, Any]],
    source: str,
    product_category: str | None,
) -> dict[str, Any]:
    baseline_targets = [
        row
        for row in existing_targets
        if not bool(_safe_target_metadata(row.get("metadata")).get("source_fit_probation"))
    ]
    return _derive_seed_defaults(baseline_targets, source, product_category)


def _make_probation_metadata(reason: str) -> dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "source_fit_probation": True,
        "source_fit_probation_reason": reason,
        "source_fit_probation_seeded_at": datetime.now(timezone.utc).isoformat(),
    }


def _apply_probation_defaults(defaults: dict[str, Any], probation_cfg) -> dict[str, Any]:
    priority_value = int(defaults.get("priority") or 0)
    max_pages_value = int(defaults.get("max_pages") or 0)
    interval_value = int(defaults.get("scrape_interval_hours") or 0)

    if priority_value <= 0:
        priority_value = int(probation_cfg.source_fit_probation_priority)
    else:
        priority_value = min(priority_value, int(probation_cfg.source_fit_probation_priority))

    if max_pages_value <= 0:
        max_pages_value = int(probation_cfg.source_fit_probation_max_pages)
    else:
        max_pages_value = min(max_pages_value, int(probation_cfg.source_fit_probation_max_pages))

    if interval_value <= 0:
        interval_value = int(probation_cfg.source_fit_probation_scrape_interval_hours)
    else:
        interval_value = max(interval_value, int(probation_cfg.source_fit_probation_scrape_interval_hours))

    return {
        "priority": priority_value,
        "max_pages": max_pages_value,
        "scrape_interval_hours": interval_value,
        "scrape_mode": str(defaults.get("scrape_mode") or "incremental"),
    }


def _safe_target_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _api_iso_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _normalized_target_scrape_state(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    explicit = {
        "mode": row.get("scrape_mode"),
        "runtime_mode": row.get("last_scrape_runtime_mode"),
        "status": row.get("last_scrape_status"),
        "oldest_review": _api_iso_value(row.get("last_scrape_oldest_review")),
        "newest_review": _api_iso_value(row.get("last_scrape_newest_review")),
        "date_cutoff_used": _api_iso_value(row.get("last_scrape_date_cutoff")),
        "pages_scraped": row.get("last_scrape_pages_scraped"),
        "reviews_found": row.get("last_scrape_reviews_found"),
        "reviews_inserted": row.get("last_scrape_reviews"),
        "reviews_filtered": row.get("last_scrape_reviews_filtered"),
        "date_dropped": row.get("last_scrape_date_dropped"),
        "duration_ms": row.get("last_scrape_duration_ms"),
        "stop_reason": row.get("last_scrape_stop_reason"),
        "resume_page": row.get("last_scrape_resume_page"),
    }
    _ = metadata
    merged = {key: value for key, value in explicit.items() if value not in (None, "")}
    return merged or None


def _serialize_target_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    metadata = _safe_target_metadata(data.get("metadata"))
    metadata.pop("scrape_state", None)
    metadata.pop("scrape_mode", None)
    data["metadata"] = metadata
    data["scrape_state"] = _normalized_target_scrape_state(data, metadata)
    for key in (
        "last_scraped_at",
        "created_at",
        "updated_at",
        "last_scrape_oldest_review",
        "last_scrape_newest_review",
        "last_scrape_date_cutoff",
    ):
        if key in data:
            data[key] = _api_iso_value(data.get(key))
    return data


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _average(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 4)


async def _build_probation_telemetry(
    pool,
    *,
    limit: int,
    lookback_days: int,
    actionable_urgency_min: float,
) -> dict[str, Any]:
    target_rows = await pool.fetch(
        """
        SELECT t.id::text AS target_id, t.vendor_name, t.source, t.product_name,
               t.product_slug, t.product_category, t.priority, t.max_pages,
               t.scrape_interval_hours, t.last_scraped_at, t.last_scrape_status,
               t.last_scrape_reviews, t.metadata,
               COUNT(l.id) FILTER (
                   WHERE l.started_at >= NOW() - make_interval(days => $1)
               )::int AS runs_total,
               COALESCE(SUM(l.reviews_found) FILTER (
                   WHERE l.started_at >= NOW() - make_interval(days => $1)
               ), 0)::int AS reviews_found_total,
               COALESCE(SUM(l.reviews_inserted) FILTER (
                   WHERE l.started_at >= NOW() - make_interval(days => $1)
               ), 0)::int AS reviews_inserted_total,
               MAX(l.started_at) FILTER (
                   WHERE l.started_at >= NOW() - make_interval(days => $1)
               ) AS last_run_at
        FROM b2b_scrape_targets t
        LEFT JOIN b2b_scrape_log l ON l.target_id = t.id
        WHERE t.enabled = true
          AND COALESCE((t.metadata->>'source_fit_probation')::boolean, false) = true
        GROUP BY t.id
        ORDER BY COALESCE(MAX(l.started_at), t.last_scraped_at) DESC NULLS LAST,
                 t.priority DESC, t.vendor_name
        LIMIT $2
        """,
        lookback_days,
        limit,
    )

    target_ids = [str(row["target_id"]) for row in target_rows]
    review_rows = []
    company_signal_rows = []
    if target_ids:
        # APPROVED-ENRICHMENT-READ: churn_signals.intent_to_leave, urgency_score, competitors_mentioned
        # Reason: actionable review count aggregation
        review_rows = await pool.fetch(
            """
            SELECT raw_metadata->>'scrape_target_id' AS target_id,
                   COUNT(*)::int AS tracked_reviews,
                   COUNT(*) FILTER (
                       WHERE reviewer_company_norm IS NOT NULL
                   )::int AS named_company_reviews,
                   COUNT(*) FILTER (
                       WHERE enrichment_status = 'enriched'
                         AND (
                             COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
                             OR COALESCE((enrichment->>'urgency_score')::numeric, 0) >= $1
                             OR CASE
                                    WHEN jsonb_typeof(enrichment->'competitors_mentioned') = 'array'
                                    THEN jsonb_array_length(enrichment->'competitors_mentioned') > 0
                                    ELSE false
                                END
                         )
                   )::int AS actionable_reviews
            FROM b2b_reviews
            WHERE imported_at >= NOW() - make_interval(days => $2)
              AND raw_metadata ? 'scrape_target_id'
              AND (raw_metadata->>'scrape_target_id') = ANY($3::text[])
            GROUP BY 1
            """,
            actionable_urgency_min,
            lookback_days,
            target_ids,
        )
        company_signal_rows = await pool.fetch(
            """
            SELECT r.raw_metadata->>'scrape_target_id' AS target_id,
                   COUNT(DISTINCT cs.review_id)::int AS company_signal_reviews
            FROM b2b_company_signals cs
            JOIN b2b_reviews r ON r.id = cs.review_id
            WHERE r.imported_at >= NOW() - make_interval(days => $1)
              AND r.raw_metadata ? 'scrape_target_id'
              AND (r.raw_metadata->>'scrape_target_id') = ANY($2::text[])
            GROUP BY 1
            """,
            lookback_days,
            target_ids,
        )

    review_map = {str(row["target_id"]): dict(row) for row in review_rows}
    company_signal_map = {
        str(row["target_id"]): int(row["company_signal_reviews"] or 0)
        for row in company_signal_rows
    }

    targets: list[dict[str, Any]] = []
    for raw_row in target_rows:
        row = dict(raw_row)
        metadata = _safe_target_metadata(row.get("metadata"))
        target_id = str(row["target_id"])
        review_stats = review_map.get(target_id, {})
        tracked_reviews = int(review_stats.get("tracked_reviews") or 0)
        named_company_reviews = int(review_stats.get("named_company_reviews") or 0)
        actionable_reviews = int(review_stats.get("actionable_reviews") or 0)
        company_signal_reviews = int(company_signal_map.get(target_id) or 0)
        reviews_found_total = int(row.get("reviews_found_total") or 0)
        reviews_inserted_total = int(row.get("reviews_inserted_total") or 0)
        duplicate_noise_reviews = max(reviews_found_total - reviews_inserted_total, 0)
        fit = classify_source_fit(row.get("source"), row.get("product_category"))

        targets.append(
            {
                "target_id": target_id,
                "vendor_name": row.get("vendor_name"),
                "source": row.get("source"),
                "vertical": fit.vertical,
                "product_name": row.get("product_name"),
                "product_slug": row.get("product_slug"),
                "product_category": row.get("product_category"),
                "priority": row.get("priority"),
                "max_pages": row.get("max_pages"),
                "scrape_interval_hours": row.get("scrape_interval_hours"),
                "last_scraped_at": row.get("last_scraped_at"),
                "last_scrape_status": row.get("last_scrape_status"),
                "last_scrape_reviews": row.get("last_scrape_reviews"),
                "last_run_at": row.get("last_run_at"),
                "runs_total": int(row.get("runs_total") or 0),
                "reviews_found_total": reviews_found_total,
                "reviews_inserted_total": reviews_inserted_total,
                "scrape_yield_rate": _rate(reviews_inserted_total, reviews_found_total),
                "duplicate_noise_rate": _rate(duplicate_noise_reviews, reviews_found_total),
                "tracked_reviews": tracked_reviews,
                "named_company_reviews": named_company_reviews,
                "named_company_hit_rate": _rate(named_company_reviews, tracked_reviews),
                "actionable_reviews": actionable_reviews,
                "actionable_review_rate": _rate(actionable_reviews, tracked_reviews),
                "company_signal_reviews": company_signal_reviews,
                "company_signal_hit_rate": _rate(company_signal_reviews, tracked_reviews),
                "tracking_ready": tracked_reviews > 0,
                "probation_reason": metadata.get("source_fit_probation_reason"),
                "probation_seeded_at": metadata.get("source_fit_probation_seeded_at"),
            }
        )

    return {
        "basis": _REVIEW_BASIS_RAW_TARGET_PROVENANCE,
        "lookback_days": lookback_days,
        "actionable_urgency_min": actionable_urgency_min,
        "probation_targets": len(targets),
        "summary": {
            "targets_with_runs": sum(1 for item in targets if item["runs_total"] > 0),
            "targets_with_tracking": sum(1 for item in targets if item["tracking_ready"]),
            "avg_scrape_yield_rate": _average([item["scrape_yield_rate"] for item in targets]),
            "avg_duplicate_noise_rate": _average([item["duplicate_noise_rate"] for item in targets]),
            "avg_named_company_hit_rate": _average([item["named_company_hit_rate"] for item in targets]),
            "avg_actionable_review_rate": _average([item["actionable_review_rate"] for item in targets]),
            "avg_company_signal_hit_rate": _average([item["company_signal_hit_rate"] for item in targets]),
        },
        "targets": targets,
    }

@router.get("/targets")
async def list_targets(
    source: Optional[str] = None,
    enabled_only: bool = True,
) -> list[dict]:
    """List scrape targets."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    conditions = []
    args = []
    idx = 1

    if enabled_only:
        conditions.append("enabled = true")
    if source:
        conditions.append(f"source = ${idx}")
        args.append(source)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, source, vendor_name, product_name, product_slug,
               product_category, max_pages, enabled, priority, scrape_mode,
               last_scraped_at, last_scrape_status, last_scrape_reviews,
               last_scrape_runtime_mode, last_scrape_stop_reason,
               last_scrape_oldest_review, last_scrape_newest_review,
               last_scrape_date_cutoff, last_scrape_pages_scraped,
               last_scrape_reviews_found, last_scrape_reviews_filtered,
               last_scrape_date_dropped, last_scrape_duration_ms,
               last_scrape_resume_page,
               scrape_interval_hours, metadata, created_at, updated_at
        FROM b2b_scrape_targets
        {where}
        ORDER BY priority DESC, vendor_name
        LIMIT 500
        """,
        *args,
    )

    return [_serialize_target_row(r) for r in rows]


@router.get("/targets/probation-telemetry")
async def probation_telemetry(
    limit: int = Query(default=100, ge=1, le=500),
    lookback_days: Optional[int] = Query(default=None, ge=1, le=365),
) -> dict:
    """Summarize usefulness metrics for enabled probation targets."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    telemetry_window = int(
        lookback_days or settings.b2b_scrape.source_fit_probation_telemetry_lookback_days
    )
    actionable_urgency_min = float(settings.b2b_scrape.source_fit_probation_actionable_urgency_min)
    return await _build_probation_telemetry(
        pool,
        limit=limit,
        lookback_days=telemetry_window,
        actionable_urgency_min=actionable_urgency_min,
    )


@router.post("/targets/probation-telemetry/disable-low-yield")
async def disable_low_yield_probation_targets(body: DisableLowYieldProbationRequest) -> dict:
    """Disable enabled probation targets that have already shown no useful yield."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    telemetry_window = int(
        body.lookback_days or settings.b2b_scrape.source_fit_probation_telemetry_lookback_days
    )
    actionable_urgency_min = float(settings.b2b_scrape.source_fit_probation_actionable_urgency_min)
    telemetry = await _build_probation_telemetry(
        pool,
        limit=body.limit,
        lookback_days=telemetry_window,
        actionable_urgency_min=actionable_urgency_min,
    )

    vendors = _normalize_filters(body.vendors)
    sources = _normalize_filters(body.sources)
    source_tiers = _normalize_filters(body.source_tiers)
    verticals = _normalize_filters(body.verticals)
    statuses = {status.strip().lower() for status in body.statuses if status and status.strip()}

    candidates: list[dict[str, Any]] = []
    for item in telemetry["targets"]:
        status = str(item.get("last_scrape_status") or "").strip().lower()
        if item.get("runs_total", 0) < body.min_runs:
            continue
        if statuses and status not in statuses:
            continue
        if int(item.get("reviews_inserted_total") or 0) > body.max_reviews_inserted:
            continue
        if int(item.get("tracked_reviews") or 0) > body.max_tracked_reviews:
            continue
        if body.max_named_company_reviews is not None and (
            int(item.get("named_company_reviews") or 0) > body.max_named_company_reviews
        ):
            continue
        if body.max_actionable_reviews is not None and (
            int(item.get("actionable_reviews") or 0) > body.max_actionable_reviews
        ):
            continue
        if body.max_company_signal_reviews is not None and (
            int(item.get("company_signal_reviews") or 0) > body.max_company_signal_reviews
        ):
            continue
        if not _matches_filters(item, vendors, sources, source_tiers, verticals):
            continue
        candidates.append(item)
        if len(candidates) >= body.limit:
            break

    if not body.dry_run:
        disable_reason = (
            "low_downstream_signal_probation"
            if (
                body.max_named_company_reviews is not None
                or body.max_actionable_reviews is not None
                or body.max_company_signal_reviews is not None
            )
            else "low_yield_probation"
        )
        for item in candidates:
            await pool.execute(
                """
                UPDATE b2b_scrape_targets
                SET enabled = false,
                    metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                        'source_fit_probation_disabled', true,
                        'source_fit_probation_disabled_reason', $2::text,
                        'source_fit_probation_disabled_at', NOW()::text
                    ),
                    updated_at = NOW()
                WHERE id = $1
                """,
                item["target_id"],
                disable_reason,
            )

    return {
        "dry_run": body.dry_run,
        "lookback_days": telemetry_window,
        "requested": len(candidates),
        "disabled": len(candidates),
        "criteria": {
            "min_runs": body.min_runs,
            "max_reviews_inserted": body.max_reviews_inserted,
            "max_tracked_reviews": body.max_tracked_reviews,
            "max_named_company_reviews": body.max_named_company_reviews,
            "max_actionable_reviews": body.max_actionable_reviews,
            "max_company_signal_reviews": body.max_company_signal_reviews,
            "statuses": sorted(statuses),
        },
        "targets": candidates,
    }


@router.post("/targets/probation-telemetry/promote")
async def promote_probation_targets(body: PromoteProbationTargetsRequest) -> dict:
    """Promote useful probation targets onto normal scrape defaults."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    telemetry_window = int(
        body.lookback_days or settings.b2b_scrape.source_fit_probation_telemetry_lookback_days
    )
    actionable_urgency_min = float(settings.b2b_scrape.source_fit_probation_actionable_urgency_min)
    telemetry = await _build_probation_telemetry(
        pool,
        limit=body.limit,
        lookback_days=telemetry_window,
        actionable_urgency_min=actionable_urgency_min,
    )
    existing_targets = await _fetch_existing_targets(pool)

    vendors = _normalize_filters(body.vendors)
    sources = _normalize_filters(body.sources)
    source_tiers = _normalize_filters(body.source_tiers)
    verticals = _normalize_filters(body.verticals)
    statuses = {status.strip().lower() for status in body.statuses if status and status.strip()}

    candidates: list[dict[str, Any]] = []
    for item in telemetry["targets"]:
        status = str(item.get("last_scrape_status") or "").strip().lower()
        if item.get("runs_total", 0) < body.min_runs:
            continue
        if statuses and status not in statuses:
            continue
        if int(item.get("reviews_inserted_total") or 0) < body.min_reviews_inserted:
            continue
        if int(item.get("tracked_reviews") or 0) < body.min_tracked_reviews:
            continue
        if int(item.get("actionable_reviews") or 0) < body.min_actionable_reviews:
            continue
        if int(item.get("company_signal_reviews") or 0) < body.min_company_signal_reviews:
            continue
        if body.require_any_downstream_signal and (
            int(item.get("named_company_reviews") or 0) <= 0
            and int(item.get("actionable_reviews") or 0) <= 0
            and int(item.get("company_signal_reviews") or 0) <= 0
        ):
            continue
        if not _matches_filters(item, vendors, sources, source_tiers, verticals):
            continue
        defaults = _derive_promotion_defaults(
            existing_targets,
            str(item.get("source") or ""),
            item.get("product_category"),
        )
        candidate = dict(item)
        candidate["promotion_defaults"] = defaults
        candidates.append(candidate)
        if len(candidates) >= body.limit:
            break

    if not body.dry_run:
        for item in candidates:
            defaults = item["promotion_defaults"]
            await pool.execute(
                """
                UPDATE b2b_scrape_targets
                SET priority = $2,
                    max_pages = $3,
                    scrape_interval_hours = $4,
                    scrape_mode = $5,
                    metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                        'source_fit_probation', false,
                        'source_fit_promoted', true,
                        'source_fit_promoted_reason', 'probation_useful',
                        'source_fit_promoted_at', NOW()::text
                    ),
                    updated_at = NOW()
                WHERE id = $1
                """,
                item["target_id"],
                defaults["priority"],
                defaults["max_pages"],
                defaults["scrape_interval_hours"],
                defaults["scrape_mode"],
            )

    return {
        "dry_run": body.dry_run,
        "lookback_days": telemetry_window,
        "requested": len(candidates),
        "promoted": len(candidates),
        "criteria": {
            "min_runs": body.min_runs,
            "min_reviews_inserted": body.min_reviews_inserted,
            "min_tracked_reviews": body.min_tracked_reviews,
            "min_actionable_reviews": body.min_actionable_reviews,
            "min_company_signal_reviews": body.min_company_signal_reviews,
            "require_any_downstream_signal": body.require_any_downstream_signal,
            "statuses": sorted(statuses),
        },
        "targets": candidates,
    }


@router.post("/targets/source-yield/disable-low-yield")
async def disable_low_yield_source_targets(body: DisableLowYieldSourceRequest) -> dict:
    """Disable low-yield scrape targets for a source using recent scrape logs."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    cfg = settings.b2b_scrape
    source = (body.source or cfg.source_low_yield_pruning_source).strip().lower()
    lookback_runs = int(body.lookback_runs or cfg.source_low_yield_pruning_lookback_runs)
    min_runs = int(body.min_runs or cfg.source_low_yield_pruning_min_runs)
    max_inserted_total = int(
        body.max_inserted_total
        if body.max_inserted_total is not None
        else cfg.source_low_yield_pruning_max_inserted_total
    )
    max_disable_per_run = int(
        body.max_disable_per_run
        or cfg.source_low_yield_pruning_max_disable_per_run
    )

    try:
        return await prune_low_yield_targets(
            pool,
            source=source,
            lookback_runs=lookback_runs,
            min_runs=min_runs,
            max_inserted_total=max_inserted_total,
            max_disable_per_run=max_disable_per_run,
            dry_run=body.dry_run,
            enabled_only=body.enabled_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/targets/disable-blocked")
async def disable_blocked_targets(body: DisableBlockedTargetsRequest) -> dict:
    """Disable enabled scrape targets that have remained blocked past a cooldown."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    cfg = settings.b2b_scrape
    min_blocked_age_hours = int(
        body.min_blocked_age_hours
        or cfg.blocked_target_disable_min_age_hours
    )
    max_disable_per_run = int(
        body.max_disable_per_run
        or cfg.blocked_target_disable_max_disable_per_run
    )
    result = await disable_persistently_blocked_targets(
        pool,
        min_blocked_age_hours=min_blocked_age_hours,
        max_disable_per_run=max_disable_per_run,
        dry_run=body.dry_run,
        sources=body.sources,
        vendors=body.vendors,
    )
    return result


@router.get("/targets/coverage-plan")
async def coverage_plan(
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    """Audit scrape coverage gaps and poor-fit enabled targets."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    inventory_rows, existing_targets = await _fetch_coverage_inputs(pool)

    plan = build_scrape_coverage_plan(
        inventory_rows,
        existing_targets,
        allowed_sources=_current_allowed_sources(),
    )
    plan["missing_core_targets"] = plan["missing_core_targets"][:limit]
    plan["conditional_opportunities"] = plan["conditional_opportunities"][:limit]
    plan["poor_fit_enabled_targets"] = plan["poor_fit_enabled_targets"][:limit]
    return plan


@router.post("/targets/onboard-vendor")
async def onboard_vendor_targets(body: VendorOnboardingTargetsRequest) -> dict:
    """Bootstrap scrape targets for a net-new vendor."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    vendor_name = str(body.vendor_name or "").strip()
    if not vendor_name:
        raise HTTPException(status_code=422, detail="vendor_name is required")

    return await provision_vendor_onboarding_targets(
        pool,
        vendor_name,
        product_category=body.product_category,
        source_slug_overrides=body.source_slug_overrides,
        dry_run=body.dry_run,
        limit=body.limit,
    )


@router.post("/targets/coverage-plan/seed-missing-core")
async def seed_missing_core_targets(body: SeedMissingCoreRequest) -> dict:
    """Seed or re-enable verified missing core scrape targets."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    inventory_rows, existing_targets = await _fetch_coverage_inputs(pool)
    plan = build_scrape_coverage_plan(
        inventory_rows,
        existing_targets,
        allowed_sources=_current_allowed_sources(),
    )
    vendors = _normalize_filters(body.vendors)
    sources = _normalize_filters(body.sources)
    source_tiers = _normalize_filters(body.source_tiers)
    verticals = _normalize_filters(body.verticals)
    candidates = [
        item
        for item in plan["missing_core_targets"]
        if (
            item.get("existing_disabled_target_id")
            or item.get("verified_product_slug")
            or item.get("suggested_product_slug")
        ) and _matches_filters(item, vendors, sources, source_tiers, verticals)
    ][:body.limit]
    applied = await apply_missing_core_targets(
        pool,
        existing_targets,
        candidates,
        dry_run=body.dry_run,
    )

    return {
        "dry_run": body.dry_run,
        "requested": len(candidates),
        "applied": len(applied),
        "actions": applied,
    }


@router.post("/targets/coverage-plan/disable-poor-fit")
async def disable_poor_fit_targets(body: DisablePoorFitRequest) -> dict:
    """Disable currently enabled targets that violate source-fit policy."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    inventory_rows, existing_targets = await _fetch_coverage_inputs(pool)
    plan = build_scrape_coverage_plan(
        inventory_rows,
        existing_targets,
        allowed_sources=_current_allowed_sources(),
    )
    vendors = _normalize_filters(body.vendors)
    sources = _normalize_filters(body.sources)
    source_tiers = _normalize_filters(body.source_tiers)
    verticals = _normalize_filters(body.verticals)
    candidates = []
    skipped_overrides = []
    for item in plan["poor_fit_enabled_targets"]:
        if not _matches_filters(item, vendors, sources, source_tiers, verticals):
            continue
        if item.get("source_fit_override") == "allow" and not body.include_overrides:
            skipped_overrides.append(item)
            continue
        candidates.append(item)
        if len(candidates) >= body.limit:
            break

    if not body.dry_run:
        for item in candidates:
            await pool.execute(
                "UPDATE b2b_scrape_targets SET enabled = false, updated_at = NOW() WHERE id = $1",
                item["id"],
            )

    return {
        "dry_run": body.dry_run,
        "requested": len(candidates),
        "skipped_overrides": len(skipped_overrides),
        "disabled": len(candidates),
        "targets": candidates,
    }


@router.post("/targets/coverage-plan/seed-conditional-probation")
async def seed_conditional_probation_targets(body: SeedConditionalProbationRequest) -> dict:
    """Seed or re-enable conditional targets in probation mode."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    inventory_rows, existing_targets = await _fetch_coverage_inputs(pool)
    plan = build_scrape_coverage_plan(
        inventory_rows,
        existing_targets,
        allowed_sources=_current_allowed_sources(),
    )
    vendors = _normalize_filters(body.vendors)
    sources = _normalize_filters(body.sources)
    source_tiers = _normalize_filters(body.source_tiers)
    verticals = _normalize_filters(body.verticals)
    candidates = [
        item
        for item in plan["conditional_opportunities"]
        if item.get("can_probation_now") and _matches_filters(item, vendors, sources, source_tiers, verticals)
    ][:body.limit]

    applied: list[dict[str, Any]] = []
    probation_cfg = settings.b2b_scrape
    for item in candidates:
        metadata = _make_probation_metadata(item["reason"])
        if item.get("existing_disabled_target_id"):
            existing_row = next(
                (
                    row for row in existing_targets
                    if str(row.get("id") or "") == str(item["existing_disabled_target_id"])
                ),
                {},
            )
            defaults = _apply_probation_defaults(
                {
                    "priority": existing_row.get("priority"),
                    "max_pages": existing_row.get("max_pages"),
                    "scrape_interval_hours": existing_row.get("scrape_interval_hours"),
                    "scrape_mode": existing_row.get("scrape_mode"),
                },
                probation_cfg,
            )
            action = {
                "action": "enable_existing_probation",
                "target_id": item["existing_disabled_target_id"],
                "vendor_name": item["vendor_name"],
                "source": item["source"],
                "source_tier": item.get("source_tier"),
                "product_slug": item.get("existing_disabled_product_slug"),
                "priority": defaults["priority"],
                "max_pages": defaults["max_pages"],
                "scrape_interval_hours": defaults["scrape_interval_hours"],
                "scrape_mode": defaults["scrape_mode"],
                "metadata": metadata,
            }
            if not body.dry_run:
                await pool.execute(
                    """
                    UPDATE b2b_scrape_targets
                    SET enabled = true,
                        priority = $2,
                        max_pages = $3,
                        scrape_interval_hours = $4,
                        scrape_mode = $5,
                        metadata = COALESCE(metadata, '{}'::jsonb) || $6::jsonb,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    item["existing_disabled_target_id"],
                    defaults["priority"],
                    defaults["max_pages"],
                    defaults["scrape_interval_hours"],
                    defaults["scrape_mode"],
                    json.dumps(metadata),
                )
            applied.append(action)
            continue

        product_slug = item.get("verified_product_slug") or item.get("suggested_product_slug")
        if not product_slug:
            continue
        source, product_slug = validate_target_input(item["source"], product_slug)
        defaults = _apply_probation_defaults(
            _derive_seed_defaults(existing_targets, source, item.get("product_category")),
            probation_cfg,
        )
        action = {
            "action": "insert_probation_target",
            "vendor_name": item["vendor_name"],
            "source": source,
            "source_tier": item.get("source_tier"),
            "product_slug": product_slug,
            "product_name": item.get("verified_product_name") or item["vendor_name"],
            "product_category": item.get("product_category"),
            "priority": defaults["priority"],
            "max_pages": defaults["max_pages"],
            "scrape_interval_hours": defaults["scrape_interval_hours"],
            "scrape_mode": defaults["scrape_mode"],
            "metadata": metadata,
        }
        if not body.dry_run:
            vendor_name = await resolve_vendor_name(item["vendor_name"])
            row = await pool.fetchrow(
                """
                INSERT INTO b2b_scrape_targets
                    (source, vendor_name, product_name, product_slug, product_category,
                     max_pages, enabled, priority, scrape_interval_hours, scrape_mode, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, true, $7, $8, $9, $10::jsonb)
                RETURNING id
                """,
                source,
                vendor_name,
                action["product_name"],
                product_slug,
                item.get("product_category"),
                action["max_pages"],
                action["priority"],
                action["scrape_interval_hours"],
                action["scrape_mode"],
                json.dumps(metadata),
            )
            action["target_id"] = str(row["id"]) if row else None
        applied.append(action)

    return {
        "dry_run": body.dry_run,
        "requested": len(candidates),
        "applied": len(applied),
        "targets": applied,
    }


@router.post("/targets")
async def create_target(body: ScrapeTargetCreate) -> dict:
    """Create a new scrape target."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    from ..services.scraping.parsers import get_all_parsers

    source = body.source.strip().lower()
    allowed_sources = set(_current_allowed_sources())
    valid_sources = set(get_all_parsers().keys())
    if source not in valid_sources:
        raise HTTPException(status_code=400, detail=f"Invalid source. Must be one of: {valid_sources}")
    if source not in allowed_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Source '{source}' is currently disabled by B2B scrape source governance",
        )

    try:
        source, product_slug = validate_target_input(source, body.product_slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    vendor_name = await resolve_vendor_name(body.vendor_name)
    product_name = body.product_name.strip() if body.product_name else None
    product_category = body.product_category.strip() if body.product_category else None

    try:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_scrape_targets
                (source, vendor_name, product_name, product_slug,
                 product_category, max_pages, enabled, priority,
                 scrape_interval_hours, scrape_mode, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
            RETURNING id, source, vendor_name, product_slug, enabled, scrape_mode, created_at
            """,
            source, vendor_name, product_name, product_slug,
            product_category, body.max_pages, body.enabled, body.priority,
            body.scrape_interval_hours, body.scrape_mode, json.dumps(body.metadata),
        )
    except Exception as exc:
        if "idx_b2b_scrape_targets_dedup" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"Target already exists for {source}/{product_slug}/{body.scrape_mode}",
            )
        logger.error("Failed to create scrape target: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create target")

    return dict(row)


@router.patch("/targets/{target_id}")
async def update_target(target_id: UUID, body: ScrapeTargetUpdate) -> dict:
    """Update a scrape target."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    updates = []
    args = []
    idx = 2  # $1 is target_id

    if body.enabled is not None:
        updates.append(f"enabled = ${idx}")
        args.append(body.enabled)
        idx += 1
    if body.priority is not None:
        updates.append(f"priority = ${idx}")
        args.append(body.priority)
        idx += 1
    if body.max_pages is not None:
        updates.append(f"max_pages = ${idx}")
        args.append(body.max_pages)
        idx += 1
    if body.scrape_interval_hours is not None:
        updates.append(f"scrape_interval_hours = ${idx}")
        args.append(body.scrape_interval_hours)
        idx += 1
    if body.scrape_mode is not None:
        updates.append(f"scrape_mode = ${idx}")
        args.append(body.scrape_mode)
        idx += 1
    if body.metadata is not None:
        updates.append(f"metadata = ${idx}::jsonb")
        args.append(json.dumps(body.metadata))
        idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updates.append("updated_at = NOW()")

    try:
        row = await pool.fetchrow(
            f"""
            UPDATE b2b_scrape_targets
            SET {', '.join(updates)}
            WHERE id = $1
            RETURNING id, source, vendor_name, product_slug, enabled, priority, scrape_mode, updated_at
            """,
            target_id, *args,
        )
    except Exception as exc:
        if "idx_b2b_scrape_targets_dedup" in str(exc):
            raise HTTPException(
                status_code=409,
                detail="A target with that source/product_slug/scrape_mode already exists",
            )
        raise

    if not row:
        raise HTTPException(status_code=404, detail="Target not found")
    return dict(row)


@router.delete("/targets/{target_id}")
async def delete_target(target_id: UUID) -> dict:
    """Delete a scrape target and its logs."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    async with pool.transaction() as conn:
        await conn.execute("DELETE FROM b2b_scrape_log WHERE target_id = $1", target_id)
        result = await conn.execute("DELETE FROM b2b_scrape_targets WHERE id = $1", target_id)

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Target not found")
    return {"deleted": str(target_id)}


@router.post("/targets/{target_id}/run")
async def trigger_scrape(target_id: UUID) -> dict:
    """Manually trigger a scrape for a specific target."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    row = await pool.fetchrow(
        """
        SELECT id, source, vendor_name, product_name, product_slug,
               product_category, max_pages, metadata, scrape_mode,
               last_scrape_reviews, last_scrape_status,
               last_scrape_runtime_mode, last_scrape_stop_reason,
               last_scrape_oldest_review, last_scrape_newest_review,
               last_scrape_date_cutoff, last_scrape_pages_scraped,
               last_scrape_reviews_found, last_scrape_reviews_filtered,
               last_scrape_date_dropped, last_scrape_duration_ms,
               last_scrape_resume_page, last_scraped_at
        FROM b2b_scrape_targets
        WHERE id = $1
        """,
        target_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Target not found")
    if str(row["source"] or "").strip().lower() not in set(_current_allowed_sources()):
        raise HTTPException(
            status_code=400,
            detail=f"Source '{row['source']}' is currently disabled by B2B scrape source governance",
        )

    # Import and run inline
    from ..autonomous.tasks.b2b_scrape_intake import (
        _KNOWN_REVIEWS_PAGE_STOP_SOURCES,
        _build_scrape_state,
        _determine_stop_reason,
        _load_existing_source_review_ids,
        _prepare_scrape_target,
        _review_date_stats,
        _scrape_state_from_row,
        _update_target_after_scrape,
    )
    from ..services.scraping.client import get_scrape_client
    from ..services.scraping.parsers import get_parser

    target, scrape_mode, metadata = _prepare_scrape_target(row, settings.b2b_scrape)
    if target.source in _KNOWN_REVIEWS_PAGE_STOP_SOURCES:
        try:
            known_source_review_ids = await _load_existing_source_review_ids(
                pool,
                target.vendor_name,
                target.source,
            )
            if known_source_review_ids:
                target.metadata["known_source_review_ids"] = sorted(known_source_review_ids)
        except Exception:
            logger.debug(
                "Known review-id preload failed for %s/%s",
                target.source,
                target.vendor_name,
                exc_info=True,
            )
    previous_state = _scrape_state_from_row(dict(row), metadata)

    parser = get_parser(target.source)
    if not parser:
        raise HTTPException(status_code=400, detail=f"No parser for source: {target.source}")

    client = get_scrape_client()
    started_at = _time.monotonic()

    try:
        result = await asyncio.wait_for(parser.scrape(target, client), timeout=300)
    except asyncio.TimeoutError:
        duration_ms = int((_time.monotonic() - started_at) * 1000)
        await _write_scrape_log(pool, target_id, target.source, "failed", 0, 0, 0,
                                ["scrape timed out after 300s"], duration_ms, parser)
        await _update_target_after_scrape(pool, target_id, "failed", 0)
        raise HTTPException(status_code=504, detail="Scrape timed out after 300s")
    except Exception as exc:
        duration_ms = int((_time.monotonic() - started_at) * 1000)
        # Log failure
        await _write_scrape_log(pool, target_id, target.source, "failed", 0, 0, 0,
                                [str(exc)], duration_ms, parser)
        await _update_target_after_scrape(pool, target_id, "failed", 0)
        logger.error("Scrape failed for target %s: %s", target_id, exc)
        raise HTTPException(status_code=502, detail="Scrape failed")

    # Relevance filter: drop noise from social media sources
    filtered_count = 0
    if result.reviews:
        from ..services.scraping.relevance import STRUCTURED_SOURCES, filter_reviews
        if target.source not in STRUCTURED_SOURCES:
            original_count = len(result.reviews)
            result.reviews, filtered_count = filter_reviews(
                result.reviews, target.vendor_name, 0.55,
            )

    # Exhaustive mode: date filtering
    date_dropped = 0
    if scrape_mode == "exhaustive" and result.reviews and target.date_cutoff:
        from ..autonomous.tasks.b2b_scrape_intake import _filter_by_date
        cutoff = date.fromisoformat(target.date_cutoff)
        result.reviews, date_dropped = _filter_by_date(result.reviews, cutoff)

    # Insert reviews
    inserted = 0
    insert_stats = {
        "inserted": 0,
        "skipped_short": 0,
        "skipped_quality_gate": 0,
        "duplicate_or_existing": 0,
        "duplicate_same_batch": 0,
        "duplicate_existing": 0,
        "duplicate_db_conflict": 0,
        "named_company_reviews": 0,
        "eligible_rows": 0,
    }
    pv = getattr(parser, 'version', None)
    if result.reviews:
        from ..autonomous.tasks.b2b_scrape_intake import (
            _insert_reviews,
            _load_existing_review_fingerprints,
        )
        batch_id = f"manual_{target.source}_{target.product_slug}_{int(_time.time())}"
        known_keys, known_identities, known_content_hashes = await _load_existing_review_fingerprints(
            pool,
            target.vendor_name,
            target.source,
        )
        insert_stats = await _insert_reviews(
            pool,
            result.reviews,
            batch_id,
            parser_version=pv,
            known_keys=known_keys,
            known_identities=known_identities,
            known_content_hashes=known_content_hashes,
            target_context={"scrape_target_id": str(target_id)},
        )
        inserted = insert_stats["inserted"]

    duration_ms = int((_time.monotonic() - started_at) * 1000)

    # Log to scrape log
    cross_source_dupes_for_log = int(insert_stats.get("cross_source_duplicates", 0) or 0)
    if scrape_mode == "exhaustive":
        from ..autonomous.tasks.b2b_scrape_intake import _log_scrape_exhaustive
        date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
        stop_reason = _determine_stop_reason(result, target, date_dropped)
        await _log_scrape_exhaustive(
            pool, target, result.status,
            {
                "found": len(result.reviews) + filtered_count + date_dropped,
                "inserted": inserted,
                "date_dropped": date_dropped,
                "stop_reason": stop_reason,
                "oldest_review": date_info["oldest"],
                "newest_review": date_info["newest"],
                "status": result.status,
            },
            result, parser, duration_ms,
            cross_source_duplicates=cross_source_dupes_for_log,
        )
    else:
        date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
        stop_reason = _determine_stop_reason(result, target, date_dropped)
        await _write_scrape_log(
            pool, target_id, target.source, result.status,
            len(result.reviews) + filtered_count, inserted, result.pages_scraped,
            result.errors, duration_ms, parser,
            page_logs=getattr(result, "page_logs", []) or [],
            stop_reason=stop_reason,
            oldest_review=date_info["oldest"],
            newest_review=date_info["newest"],
            date_dropped=date_dropped,
            cross_source_duplicates=cross_source_dupes_for_log,
        )

    scrape_state = _build_scrape_state(
        metadata,
        target,
        scrape_mode,
        result,
        inserted=inserted,
        filtered_count=filtered_count,
        date_dropped=date_dropped,
        duration_ms=duration_ms,
        previous_state=previous_state,
    )
    await _update_target_after_scrape(
        pool,
        target_id,
        result.status,
        inserted,
        metadata=metadata,
        scrape_state=scrape_state,
    )

    return {
        "target_id": str(target_id),
        "source": target.source,
        "vendor": target.vendor_name,
        "status": result.status,
        "scrape_mode": scrape_mode,
        "reviews_found": len(result.reviews) + filtered_count + date_dropped,
        "reviews_inserted": inserted,
        "reviews_filtered": filtered_count,
        "date_dropped": date_dropped,
        "duplicate_or_existing": insert_stats["duplicate_or_existing"],
        "duplicate_same_batch": insert_stats["duplicate_same_batch"],
        "duplicate_existing": insert_stats["duplicate_existing"],
        "duplicate_db_conflict": insert_stats["duplicate_db_conflict"],
        "named_company_reviews": insert_stats["named_company_reviews"],
        "skipped_short": insert_stats["skipped_short"],
        "skipped_quality_gate": insert_stats["skipped_quality_gate"],
        "pages_scraped": result.pages_scraped,
        "duration_ms": duration_ms,
        "errors": result.errors,
    }


@router.post("/targets/run-probation-batch")
async def run_probation_batch(body: RunProbationBatchRequest) -> dict:
    """Manually run a small batch of enabled probation targets."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    allowed_sources = _current_allowed_sources()
    if body.source and body.source not in allowed_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Source '{body.source}' is currently disabled by ATLAS_B2B_SCRAPE_SOURCE_ALLOWLIST",
        )

    params: list[Any] = []
    where_clauses = [
        "enabled = true",
        "COALESCE((metadata->>'source_fit_probation')::boolean, false) = true",
    ]
    params.append(allowed_sources)
    where_clauses.append(f"source = ANY(${len(params)}::text[])")
    if body.source:
        params.append(body.source)
        where_clauses.append(f"source = ${len(params)}")
    if body.due_only:
        where_clauses.append(
            " (last_scraped_at IS NULL OR last_scraped_at < NOW() - make_interval(hours => scrape_interval_hours)) "
        )
    params.append(body.limit)
    rows = await pool.fetch(
        f"""
        SELECT id, source, vendor_name, last_scraped_at, priority
        FROM b2b_scrape_targets
        WHERE {' AND '.join(where_clauses)}
        ORDER BY CASE WHEN last_scraped_at IS NULL THEN 0 ELSE 1 END,
                 priority DESC,
                 last_scraped_at ASC NULLS FIRST,
                 source,
                 vendor_name
        LIMIT ${len(params)}
        """,
        *params,
    )

    candidate_rows = [row for row in rows if str(row.get("source") or "") in allowed_sources]
    skipped_disallowed = len(rows) - len(candidate_rows)

    results: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    for row in candidate_rows:
        try:
            result = await trigger_scrape(row["id"])
            succeeded += 1
            results.append(result)
        except HTTPException as exc:
            failed += 1
            results.append(
                {
                    "target_id": str(row["id"]),
                    "source": row["source"],
                    "vendor": row["vendor_name"],
                    "status": "failed",
                    "error": exc.detail,
                }
            )

    return {
        "selected": len(candidate_rows),
        "attempted": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "skipped_disallowed_sources": skipped_disallowed,
        "due_only": body.due_only,
        "results": results,
    }


@router.post("/run-all")
async def trigger_scrape_all(
    source: Optional[str] = None,
) -> dict:
    """Trigger scrape for all enabled targets (optionally filtered by source).

    Runs sequentially with a 1s delay between targets. Returns summary.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    allowed_sources = _current_allowed_sources()
    if source and source not in allowed_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Source '{source}' is currently disabled by ATLAS_B2B_SCRAPE_SOURCE_ALLOWLIST",
        )

    query = """
        SELECT id, source, vendor_name, product_name, product_slug,
               product_category, max_pages, metadata, scrape_mode,
               last_scrape_reviews, last_scrape_status,
               last_scrape_runtime_mode, last_scrape_stop_reason,
               last_scrape_oldest_review, last_scrape_newest_review,
               last_scrape_date_cutoff, last_scrape_pages_scraped,
               last_scrape_reviews_found, last_scrape_reviews_filtered,
               last_scrape_date_dropped, last_scrape_duration_ms,
               last_scrape_resume_page, last_scraped_at
        FROM b2b_scrape_targets
        WHERE enabled = true
          AND source = ANY($1::text[])
    """
    params: list[Any] = [allowed_sources]
    if source:
        query += " AND source = $2"
        params.append(source)
    query += " ORDER BY priority DESC, source, vendor_name"

    rows = await pool.fetch(query, *params)
    rows = [row for row in rows if str(row.get("source") or "") in allowed_sources]
    if not rows:
        return {"targets": 0, "message": "No enabled targets found"}

    from ..autonomous.tasks.b2b_scrape_intake import (
        _KNOWN_REVIEWS_PAGE_STOP_SOURCES,
        _build_scrape_state,
        _filter_by_date, _determine_stop_reason, _insert_reviews,
        _load_existing_source_review_ids,
        _load_existing_review_fingerprints,
        _log_scrape_exhaustive, _prepare_scrape_target, _review_date_stats,
        _scrape_state_from_row,
        _update_target_after_scrape,
    )
    from ..services.scraping.client import get_scrape_client
    from ..services.scraping.parsers import get_parser
    from ..services.scraping.relevance import STRUCTURED_SOURCES, filter_reviews

    import asyncio

    client = get_scrape_client()
    results = []
    total_inserted = 0
    total_filtered = 0

    for row in rows:
        target, row_mode, metadata = _prepare_scrape_target(row, settings.b2b_scrape)
        if target.source in _KNOWN_REVIEWS_PAGE_STOP_SOURCES:
            try:
                known_source_review_ids = await _load_existing_source_review_ids(
                    pool,
                    target.vendor_name,
                    target.source,
                )
                if known_source_review_ids:
                    target.metadata["known_source_review_ids"] = sorted(known_source_review_ids)
            except Exception:
                logger.debug(
                    "Known review-id preload failed for %s/%s",
                    target.source,
                    target.vendor_name,
                    exc_info=True,
                )
        previous_state = _scrape_state_from_row(dict(row), metadata)

        parser = get_parser(target.source)
        if not parser:
            results.append({"source": target.source, "vendor": target.vendor_name, "status": "no_parser"})
            continue

        started_at = _time.monotonic()
        try:
            result = await parser.scrape(target, client)
        except Exception as exc:
            duration_ms = int((_time.monotonic() - started_at) * 1000)
            await _write_scrape_log(pool, row["id"], target.source, "failed", 0, 0, 0,
                                    [str(exc)], duration_ms, parser)
            await _update_target_after_scrape(pool, row["id"], "failed", 0)
            results.append({"source": target.source, "vendor": target.vendor_name, "status": "failed", "error": str(exc)[:200]})
            logger.warning("Scrape failed for %s/%s: %s", target.source, target.vendor_name, exc)
            await asyncio.sleep(1)
            continue

        # Relevance filter
        filtered_count = 0
        if result.reviews and target.source not in STRUCTURED_SOURCES:
            result.reviews, filtered_count = filter_reviews(result.reviews, target.vendor_name, 0.55)
            total_filtered += filtered_count

        # Exhaustive mode: date filtering
        date_dropped = 0
        if row_mode == "exhaustive" and result.reviews and target.date_cutoff:
            cutoff = date.fromisoformat(target.date_cutoff)
            result.reviews, date_dropped = _filter_by_date(result.reviews, cutoff)

        inserted = 0
        insert_stats = {
            "inserted": 0,
            "skipped_short": 0,
            "skipped_quality_gate": 0,
            "duplicate_or_existing": 0,
            "duplicate_same_batch": 0,
            "duplicate_existing": 0,
            "duplicate_db_conflict": 0,
            "named_company_reviews": 0,
            "eligible_rows": 0,
        }
        pv = getattr(parser, 'version', None)
        if result.reviews:
            batch_id = f"bulk_{target.source}_{target.product_slug}_{int(_time.time())}"
            known_keys, known_identities, known_content_hashes = await _load_existing_review_fingerprints(
                pool,
                target.vendor_name,
                target.source,
            )
            insert_stats = await _insert_reviews(
                pool,
                result.reviews,
                batch_id,
                parser_version=pv,
                known_keys=known_keys,
                known_identities=known_identities,
                known_content_hashes=known_content_hashes,
                target_context={"scrape_target_id": str(row["id"])},
            )
            inserted = insert_stats["inserted"]
            total_inserted += inserted

        duration_ms = int((_time.monotonic() - started_at) * 1000)

        cross_source_dupes_for_log = int(insert_stats.get("cross_source_duplicates", 0) or 0)
        if row_mode == "exhaustive":
            date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
            stop_reason = _determine_stop_reason(result, target, date_dropped)
            await _log_scrape_exhaustive(
                pool, target, result.status,
                {
                    "found": len(result.reviews) + filtered_count + date_dropped,
                    "inserted": inserted,
                    "date_dropped": date_dropped,
                    "stop_reason": stop_reason,
                    "oldest_review": date_info["oldest"],
                    "newest_review": date_info["newest"],
                    "status": result.status,
                },
                result, parser, duration_ms,
                cross_source_duplicates=cross_source_dupes_for_log,
            )
        else:
            date_info = _review_date_stats(result.reviews) if result.reviews else {"oldest": None, "newest": None}
            stop_reason = _determine_stop_reason(result, target, date_dropped)
            scrape_errors = list(result.errors)
            if filtered_count:
                scrape_errors.append(f"relevance_filtered={filtered_count}")
            await _write_scrape_log(
                pool, row["id"], target.source, result.status,
                len(result.reviews) + filtered_count, inserted, result.pages_scraped,
                scrape_errors, duration_ms, parser,
                page_logs=getattr(result, "page_logs", []) or [],
                stop_reason=stop_reason,
                oldest_review=date_info["oldest"],
                newest_review=date_info["newest"],
                date_dropped=date_dropped,
                cross_source_duplicates=cross_source_dupes_for_log,
            )

        scrape_state = _build_scrape_state(
            metadata,
            target,
            row_mode,
            result,
            inserted=inserted,
            filtered_count=filtered_count,
            date_dropped=date_dropped,
            duration_ms=duration_ms,
            previous_state=previous_state,
        )
        await _update_target_after_scrape(
            pool,
            row["id"],
            result.status,
            inserted,
            metadata=metadata,
            scrape_state=scrape_state,
        )

        results.append({
            "source": target.source,
            "vendor": target.vendor_name,
            "status": result.status,
            "mode": row_mode,
            "found": len(result.reviews) + filtered_count + date_dropped,
            "inserted": inserted,
            "duplicate_or_existing": insert_stats["duplicate_or_existing"],
            "duplicate_same_batch": insert_stats["duplicate_same_batch"],
            "duplicate_existing": insert_stats["duplicate_existing"],
            "duplicate_db_conflict": insert_stats["duplicate_db_conflict"],
            "named_company_reviews": insert_stats["named_company_reviews"],
            "skipped_short": insert_stats["skipped_short"],
            "skipped_quality_gate": insert_stats["skipped_quality_gate"],
            "filtered": filtered_count,
            "date_dropped": date_dropped,
        })
        logger.info("Scraped %s/%s [%s]: found=%d inserted=%d filtered=%d date_dropped=%d",
                     target.source, target.vendor_name, row_mode,
                     len(result.reviews) + filtered_count + date_dropped, inserted, filtered_count, date_dropped)

        await asyncio.sleep(1)

    return {
        "targets_scraped": len(results),
        "total_inserted": total_inserted,
        "total_filtered": total_filtered,
        "total_duplicate_or_existing": sum(int(result.get("duplicate_or_existing") or 0) for result in results),
        "total_duplicate_same_batch": sum(int(result.get("duplicate_same_batch") or 0) for result in results),
        "total_duplicate_existing": sum(int(result.get("duplicate_existing") or 0) for result in results),
        "total_duplicate_db_conflict": sum(int(result.get("duplicate_db_conflict") or 0) for result in results),
        "total_skipped_quality_gate": sum(int(result.get("skipped_quality_gate") or 0) for result in results),
        "results": results,
    }


async def _write_scrape_log(
    pool, target_id: UUID, source: str, status: str,
    reviews_found: int, reviews_inserted: int, pages_scraped: int,
    errors: list[str], duration_ms: int, parser,
    *, captcha_attempts: int = 0, captcha_types: list[str] | None = None,
    captcha_solve_ms: int = 0,
    page_logs: list | None = None,
    stop_reason: str | None = None,
    oldest_review: str | None = None,
    newest_review: str | None = None,
    date_dropped: int = 0,
    cross_source_duplicates: int = 0,
) -> UUID | None:
    """Write a record to b2b_scrape_log for observability."""
    proxy_type = "residential" if parser.prefer_residential else "none"
    pv = getattr(parser, 'version', None)
    page_logs = page_logs or []
    duplicate_pages = sum(1 for pl in page_logs if pl.stop_reason == "duplicate_page")
    has_page_logs = bool(page_logs)
    oldest_d = None
    newest_d = None
    try:
        oldest_d = date.fromisoformat(oldest_review) if oldest_review else None
    except (TypeError, ValueError):
        pass
    try:
        newest_d = date.fromisoformat(newest_review) if newest_review else None
    except (TypeError, ValueError):
        pass
    # Classify block type from errors
    block_type = None
    if status in ("blocked", "failed"):
        error_text = " ".join(errors).lower()
        if "captcha" in error_text or "challenge" in error_text:
            block_type = "captcha"
        elif "403" in error_text and ("ban" in error_text or "forbidden" in error_text):
            block_type = "ip_ban"
        elif "429" in error_text or "rate" in error_text:
            block_type = "rate_limit"
        elif "403" in error_text or "blocked" in error_text:
            block_type = "waf"
        elif status == "blocked":
            block_type = "unknown"
    try:
        run_id = await pool.fetchval(
            """
            INSERT INTO b2b_scrape_log
                (target_id, source, status, reviews_found, reviews_inserted,
                 pages_scraped, errors, duration_ms, proxy_type, parser_version,
                 captcha_attempts, captcha_types, captcha_solve_ms, block_type,
                 stop_reason, oldest_review, newest_review,
                 date_dropped, duplicate_pages, has_page_logs,
                 cross_source_duplicates)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21)
            RETURNING id
            """,
            target_id, source, status, reviews_found, reviews_inserted,
            pages_scraped, json.dumps(errors), duration_ms, proxy_type, pv,
            captcha_attempts,
            captcha_types or [],
            captcha_solve_ms if captcha_solve_ms > 0 else None,
            block_type,
            stop_reason,
            oldest_d,
            newest_d,
            date_dropped,
            duplicate_pages,
            has_page_logs,
            int(cross_source_duplicates or 0),
        )
        if page_logs and run_id:
            from ..autonomous.tasks.b2b_scrape_intake import _persist_page_logs

            await _persist_page_logs(pool, run_id, page_logs)
        return run_id
    except Exception:
        logger.warning("Failed to write scrape log", exc_info=True)
        return None


@router.get("/logs")
async def list_logs(
    target_id: Optional[UUID] = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    """View scrape execution logs."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")

    if target_id:
        rows = await pool.fetch(
            """
            SELECT l.*, t.vendor_name, t.source as target_source
            FROM b2b_scrape_log l
            JOIN b2b_scrape_targets t ON t.id = l.target_id
            WHERE l.target_id = $1
            ORDER BY l.started_at DESC
            LIMIT $2
            """,
            target_id, limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT l.*, t.vendor_name, t.source as target_source
            FROM b2b_scrape_log l
            JOIN b2b_scrape_targets t ON t.id = l.target_id
            ORDER BY l.started_at DESC
            LIMIT $1
            """,
            limit,
        )

    return [dict(r) for r in rows]
