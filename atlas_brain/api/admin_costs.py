"""Admin cost analytics API.

Aggregates LLM usage from the local llm_usage table for the cost dashboard.
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timedelta, timezone

import psutil
from fastapi import APIRouter, HTTPException, Query

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.admin_costs")

router = APIRouter(prefix="/admin/costs", tags=["admin-costs"])


def _recent_metadata_value(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    business = metadata.get("business")
    if isinstance(business, dict):
        nested = business.get(key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _humanize_identifier(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("/", " ").replace(".", " ").replace("_", " ").strip().title()


def _describe_recent_call(span_name: str, metadata: dict) -> tuple[str, str | None]:
    vendor_name = _recent_metadata_value(metadata, "vendor_name")
    report_type = _recent_metadata_value(metadata, "report_type")
    reasoning_mode = _recent_metadata_value(metadata, "reasoning_mode")
    phase = _recent_metadata_value(metadata, "phase")
    skill = _recent_metadata_value(metadata, "skill")

    if span_name == "reasoning.stratified.reason":
        return "Stratified Reasoning", f"Full reason{f' for {vendor_name}' if vendor_name else ''}"
    if span_name == "reasoning.stratified.reason.challenge":
        return "Stratified Reasoning", f"Challenge{f' for {vendor_name}' if vendor_name else ''}"
    if span_name == "reasoning.stratified.reason.ground":
        return "Stratified Reasoning", f"Ground{f' for {vendor_name}' if vendor_name else ''}"
    if span_name == "reasoning.stratified.reconstitute":
        return "Stratified Reasoning", f"Reconstitute{f' for {vendor_name}' if vendor_name else ''}"
    if span_name == "reasoning.stratified.reconstitute.reason.ground":
        return "Stratified Reasoning", f"Reconstitute ground{f' for {vendor_name}' if vendor_name else ''}"
    if span_name == "b2b.churn_intelligence.exploratory_overview":
        return "Exploratory Overview", "Weekly churn feed synthesis"
    if span_name == "b2b.churn_intelligence.scorecard_narrative":
        return "Scorecard Narrative", vendor_name or "Vendor scorecard narrative"
    if span_name == "b2b.churn_intelligence.executive_summary":
        return "Executive Summary", _humanize_identifier(report_type) or "Report summary synthesis"
    if span_name == "b2b.churn_intelligence.battle_card_sales_copy":
        return "Battle Card Sales Copy", vendor_name or "Battle card enrichment"

    if span_name.startswith("pipeline."):
        base = skill or span_name.removeprefix("pipeline.")
        detail = vendor_name or _humanize_identifier(report_type or phase or reasoning_mode) or None
        return _humanize_identifier(base), detail

    detail = vendor_name or _humanize_identifier(report_type or phase or reasoning_mode) or None
    return span_name, detail


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


@router.get("/summary")
async def cost_summary(days: int = Query(default=30, ge=1, le=365)):
    """High-level cost summary for the dashboard header cards."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    row = await pool.fetchrow(
        """SELECT
             COALESCE(SUM(cost_usd), 0)         AS total_cost,
             COALESCE(SUM(input_tokens), 0)      AS total_input,
             COALESCE(SUM(output_tokens), 0)     AS total_output,
             COALESCE(SUM(total_tokens), 0)      AS total_tokens,
             COUNT(*)                             AS total_calls,
             COALESCE(AVG(duration_ms) FILTER (WHERE duration_ms > 0), 0) AS avg_duration_ms,
             COALESCE(
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tokens_per_second)
                 FILTER (WHERE duration_ms > 0 AND tokens_per_second IS NOT NULL),
               0
             ) AS avg_tps
           FROM llm_usage
           WHERE created_at >= $1""",
        since,
    )
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_row = await pool.fetchrow(
        "SELECT COALESCE(SUM(cost_usd), 0) AS today_cost, COUNT(*) AS today_calls FROM llm_usage WHERE created_at >= $1",
        today_start,
    )
    return {
        "period_days": days,
        "total_cost_usd": float(row["total_cost"]),
        "total_input_tokens": int(row["total_input"]),
        "total_output_tokens": int(row["total_output"]),
        "total_tokens": int(row["total_tokens"]),
        "total_calls": int(row["total_calls"]),
        "avg_duration_ms": round(float(row["avg_duration_ms"]), 1),
        "avg_tokens_per_second": round(float(row["avg_tps"]), 1),
        "today_cost_usd": float(today_row["today_cost"]),
        "today_calls": int(today_row["today_calls"]),
    }


@router.get("/by-provider")
async def cost_by_provider(days: int = Query(default=30, ge=1, le=365)):
    """Cost breakdown by provider."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = await pool.fetch(
        """SELECT
             COALESCE(model_provider, 'unknown') AS provider,
             COALESCE(SUM(cost_usd), 0)          AS cost,
             COALESCE(SUM(input_tokens), 0)       AS input_tokens,
             COALESCE(SUM(output_tokens), 0)      AS output_tokens,
             COUNT(*)                              AS calls
           FROM llm_usage
           WHERE created_at >= $1
           GROUP BY model_provider
           ORDER BY cost DESC""",
        since,
    )
    return {
        "period_days": days,
        "providers": [
            {
                "provider": r["provider"],
                "cost_usd": float(r["cost"]),
                "input_tokens": int(r["input_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "calls": int(r["calls"]),
            }
            for r in rows
        ],
    }


@router.get("/by-model")
async def cost_by_model(days: int = Query(default=30, ge=1, le=365)):
    """Cost breakdown by model."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = await pool.fetch(
        """SELECT
             COALESCE(model_name, 'unknown')      AS model,
             COALESCE(model_provider, 'unknown')   AS provider,
             COALESCE(SUM(cost_usd), 0)            AS cost,
             COALESCE(SUM(total_tokens), 0)        AS tokens,
             COUNT(*)                               AS calls,
             COALESCE(AVG(duration_ms) FILTER (WHERE duration_ms > 0), 0) AS avg_duration_ms,
             COALESCE(
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tokens_per_second)
                 FILTER (WHERE duration_ms > 0 AND tokens_per_second IS NOT NULL),
               0
             ) AS avg_tps
           FROM llm_usage
           WHERE created_at >= $1
           GROUP BY model_name, model_provider
           ORDER BY cost DESC""",
        since,
    )
    return {
        "period_days": days,
        "models": [
            {
                "model": r["model"],
                "provider": r["provider"],
                "cost_usd": float(r["cost"]),
                "total_tokens": int(r["tokens"]),
                "calls": int(r["calls"]),
                "avg_duration_ms": round(float(r["avg_duration_ms"]), 1),
                "avg_tokens_per_second": round(float(r["avg_tps"]), 1),
            }
            for r in rows
        ],
    }


@router.get("/by-workflow")
async def cost_by_workflow(days: int = Query(default=30, ge=1, le=365)):
    """Cost breakdown by workflow (span_name prefix)."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = await pool.fetch(
        """SELECT
             span_name,
             COALESCE(SUM(cost_usd), 0)        AS cost,
             COALESCE(SUM(total_tokens), 0)     AS tokens,
             COUNT(*)                            AS calls,
             COALESCE(AVG(duration_ms), 0)       AS avg_duration_ms
           FROM llm_usage
           WHERE created_at >= $1
           GROUP BY span_name
           ORDER BY cost DESC""",
        since,
    )
    return {
        "period_days": days,
        "workflows": [
            {
                "workflow": r["span_name"],
                "cost_usd": float(r["cost"]),
                "total_tokens": int(r["tokens"]),
                "calls": int(r["calls"]),
                "avg_duration_ms": round(float(r["avg_duration_ms"]), 1),
            }
            for r in rows
        ],
    }


@router.get("/reasoning-activity")
async def reasoning_activity(days: int = Query(default=30, ge=1, le=365)):
    """Per-pass breakdown of stratified reasoning activity (classify/challenge/ground)."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await pool.fetch(
        """SELECT
             span_name,
             COALESCE((metadata->>'pass_type'), 'single') AS pass_type,
             COALESCE((metadata->>'pass_number')::int, 1) AS pass_number,
             COALESCE(SUM(cost_usd), 0)                    AS cost,
             COALESCE(SUM(total_tokens), 0)                AS tokens,
             COUNT(*)                                       AS calls,
             COALESCE(AVG(duration_ms), 0)                  AS avg_duration_ms,
             COUNT(*) FILTER (
               WHERE (metadata->>'pass_changed')::boolean IS TRUE
             )                                              AS changed_count
           FROM llm_usage
           WHERE created_at >= $1
             AND span_name LIKE 'reasoning.stratified.%'
           GROUP BY span_name, pass_type, pass_number
           ORDER BY pass_number, span_name""",
        since,
    )
    phases = []
    total_cost = 0.0
    total_tokens = 0
    total_calls = 0
    for r in rows:
        cost = float(r["cost"])
        total_cost += cost
        total_tokens += int(r["tokens"])
        total_calls += int(r["calls"])
        phases.append({
            "span_name": r["span_name"],
            "pass_type": r["pass_type"],
            "pass_number": int(r["pass_number"]),
            "calls": int(r["calls"]),
            "cost_usd": cost,
            "total_tokens": int(r["tokens"]),
            "avg_duration_ms": round(float(r["avg_duration_ms"]), 1),
            "changed_count": int(r["changed_count"]),
        })
    return {
        "period_days": days,
        "phases": phases,
        "summary": {
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_calls": total_calls,
        },
    }


@router.get("/daily")
async def cost_daily(days: int = Query(default=30, ge=1, le=365)):
    """Daily cost time series for charting."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = await pool.fetch(
        """SELECT
             DATE(created_at AT TIME ZONE 'UTC') AS day,
             COALESCE(SUM(cost_usd), 0)          AS cost,
             COALESCE(SUM(total_tokens), 0)      AS tokens,
             COUNT(*)                             AS calls
           FROM llm_usage
           WHERE created_at >= $1
           GROUP BY day
           ORDER BY day""",
        since,
    )
    return {
        "period_days": days,
        "daily": [
            {
                "date": str(r["day"]),
                "cost_usd": float(r["cost"]),
                "total_tokens": int(r["tokens"]),
                "calls": int(r["calls"]),
            }
            for r in rows
        ],
    }


@router.get("/recent")
async def recent_calls(limit: int = Query(default=50, ge=1, le=200)):
    """Most recent LLM calls for the activity feed."""
    pool = _pool_or_503()
    rows = await pool.fetch(
        """SELECT span_name, model_name, model_provider,
                  input_tokens, output_tokens, cost_usd,
                  duration_ms, tokens_per_second, status,
                  metadata, created_at
           FROM llm_usage
           ORDER BY created_at DESC
           LIMIT $1""",
        limit,
    )
    calls = []
    for r in rows:
        metadata = r["metadata"] if isinstance(r["metadata"], dict) else {}
        title, detail = _describe_recent_call(r["span_name"], metadata)
        calls.append({
            "span_name": r["span_name"],
            "title": title,
            "detail": detail,
            "model": r["model_name"],
            "provider": r["model_provider"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd": float(r["cost_usd"]) if r["cost_usd"] else 0,
            "duration_ms": r["duration_ms"],
            "tokens_per_second": r["tokens_per_second"],
            "status": r["status"],
            "metadata": metadata,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        })
    return {
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# Task health
# ---------------------------------------------------------------------------

@router.get("/task-health")
async def task_health(days: int = Query(default=30, ge=1, le=365)):
    """Health overview for every scheduled task with latest execution status."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await pool.fetch(
        """
        SELECT
            t.id,
            t.name,
            t.task_type,
            t.schedule_type,
            t.cron_expression,
            t.interval_seconds,
            t.enabled,
            t.last_run_at,
            t.next_run_at,
            latest.status        AS last_status,
            latest.duration_ms   AS last_duration_ms,
            latest.error         AS last_error,
            stats.recent_runs,
            stats.recent_failures
        FROM scheduled_tasks t
        LEFT JOIN LATERAL (
            SELECT e.status, e.duration_ms, e.error
            FROM task_executions e
            WHERE e.task_id = t.id
            ORDER BY e.started_at DESC
            LIMIT 1
        ) latest ON true
        LEFT JOIN LATERAL (
            SELECT
                COUNT(*)                                   AS recent_runs,
                COUNT(*) FILTER (WHERE e2.status != 'completed') AS recent_failures
            FROM (
                SELECT e2.status
                FROM task_executions e2
                WHERE e2.task_id = t.id
                  AND e2.started_at >= $1
                ORDER BY e2.started_at DESC
                LIMIT 20
            ) e2
        ) stats ON true
        ORDER BY t.name
        """,
        since,
    )

    return {
        "tasks": [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "task_type": r["task_type"],
                "schedule_type": r["schedule_type"],
                "cron_expression": r["cron_expression"],
                "interval_seconds": r["interval_seconds"],
                "enabled": r["enabled"],
                "last_run_at": r["last_run_at"].isoformat() if r["last_run_at"] else None,
                "next_run_at": r["next_run_at"].isoformat() if r["next_run_at"] else None,
                "last_status": r["last_status"],
                "last_duration_ms": r["last_duration_ms"],
                "last_error": r["last_error"],
                "recent_failure_rate": round(
                    r["recent_failures"] / r["recent_runs"], 3
                ) if r["recent_runs"] else 0.0,
                "recent_runs": r["recent_runs"] or 0,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Error timeline
# ---------------------------------------------------------------------------

@router.get("/error-timeline")
async def error_timeline(days: int = Query(default=30, ge=1, le=365)):
    """Daily error counts alongside total LLM calls for charting."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await pool.fetch(
        """
        SELECT
            DATE(created_at AT TIME ZONE 'UTC') AS day,
            COUNT(*)                             AS total,
            COUNT(*) FILTER (WHERE status != 'completed') AS errors
        FROM llm_usage
        WHERE created_at >= $1
        GROUP BY day
        ORDER BY day
        """,
        since,
    )

    return {
        "period_days": days,
        "daily": [
            {
                "date": str(r["day"]),
                "total_calls": int(r["total"]),
                "error_calls": int(r["errors"]),
                "error_rate": round(r["errors"] / r["total"], 3) if r["total"] else 0.0,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Scraping observability
# ---------------------------------------------------------------------------


@router.get("/scraping/summary")
async def scraping_summary(days: int = Query(default=7, ge=1, le=30)):
    """
    Scraping throughput, signal quality, and useful-review rates.

    Throughput is grouped by source + vendor so you can see which targets
    are producing signal vs noise.  Quality metrics come from b2b_reviews
    using imported_at (when we scraped) not reviewed_at (when content was
    originally posted).
    """
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # -- Throughput per source x vendor --
    throughput_rows = await pool.fetch(
        """
        SELECT
            l.source,
            t.vendor_name,
            COUNT(*)                                                    AS total_runs,
            COUNT(*) FILTER (WHERE l.status = 'completed')             AS successes,
            COUNT(*) FILTER (WHERE l.status = 'failed')                AS failures,
            COUNT(*) FILTER (WHERE l.status = 'blocked')               AS blocked,
            COUNT(*) FILTER (WHERE l.status = 'partial')               AS partial,
            COALESCE(SUM(l.reviews_found), 0)                          AS reviews_found,
            COALESCE(SUM(l.reviews_inserted), 0)                       AS reviews_inserted,
            COALESCE(AVG(l.duration_ms), 0)                            AS avg_duration_ms,
            COALESCE(SUM(l.captcha_attempts), 0)                       AS captcha_attempts,
            COUNT(*) FILTER (WHERE l.block_type IS NOT NULL)           AS blocked_requests
        FROM b2b_scrape_log l
        JOIN b2b_scrape_targets t ON t.id = l.target_id
        WHERE l.started_at >= $1
        GROUP BY l.source, t.vendor_name
        ORDER BY reviews_inserted DESC
        """,
        since,
    )

    # -- Signal quality from b2b_reviews (imported in period) --
    quality_rows = await pool.fetch(
        """
        SELECT
            source,
            COUNT(*)                                                                           AS total_reviews,
            COUNT(*) FILTER (WHERE (raw_metadata->>'source_weight')::numeric > 0.7)           AS high_signal_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')                            AS enriched_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'failed')                              AS failed_enrichments,
            ROUND(AVG((raw_metadata->>'source_weight')::numeric)::numeric, 3)                 AS avg_source_weight,
            COUNT(*) FILTER (WHERE (raw_metadata->>'author_churn_score')::numeric >= 7)       AS high_value_authors
        FROM b2b_reviews
        WHERE imported_at >= $1
        GROUP BY source
        ORDER BY total_reviews DESC
        """,
        since,
    )

    # -- Today totals (quick headline numbers) --
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*)                                              AS runs_today,
            COALESCE(SUM(reviews_inserted), 0)                   AS inserted_today,
            COUNT(*) FILTER (WHERE status IN ('failed','blocked')) AS errors_today
        FROM b2b_scrape_log
        WHERE started_at >= $1
        """,
        today_start,
    )

    def _throughput(r: dict) -> dict:
        found = int(r["reviews_found"])
        inserted = int(r["reviews_inserted"])
        runs = int(r["total_runs"])
        return {
            "source": r["source"],
            "vendor_name": r["vendor_name"],
            "total_runs": runs,
            "successes": int(r["successes"]),
            "failures": int(r["failures"]),
            "blocked": int(r["blocked"]),
            "partial": int(r["partial"]),
            "reviews_found": found,
            "reviews_inserted": inserted,
            "insert_rate": round(inserted / found, 3) if found else 0.0,
            "avg_duration_ms": round(float(r["avg_duration_ms"]), 1),
            "captcha_attempts": int(r["captcha_attempts"]),
            "blocked_requests": int(r["blocked_requests"]),
        }

    def _quality(r: dict) -> dict:
        total = int(r["total_reviews"])
        high = int(r["high_signal_reviews"])
        enriched = int(r["enriched_reviews"])
        return {
            "source": r["source"],
            "total_reviews": total,
            "high_signal_reviews": high,
            "high_signal_rate": round(high / total, 3) if total else 0.0,
            "enriched_reviews": enriched,
            "enrichment_rate": round(enriched / total, 3) if total else 0.0,
            "failed_enrichments": int(r["failed_enrichments"]),
            "avg_source_weight": float(r["avg_source_weight"] or 0),
            "high_value_authors": int(r["high_value_authors"]),
        }

    return {
        "period_days": days,
        "today": {
            "runs": int(today_row["runs_today"]),
            "reviews_inserted": int(today_row["inserted_today"]),
            "errors": int(today_row["errors_today"]),
        },
        "throughput": [_throughput(dict(r)) for r in throughput_rows],
        "quality": [_quality(dict(r)) for r in quality_rows],
    }


@router.get("/scraping/details")
async def scraping_details(
    limit: int = Query(default=50, ge=1, le=200),
    source: str | None = Query(default=None),
    status: str | None = Query(default=None),
):
    """
    Recent scrape log entries with full debug detail: errors, duration,
    captcha telemetry, block types, and parser version.

    Filter by source (reddit, g2, ...) or status (completed, failed, blocked, partial).
    """
    pool = _pool_or_503()

    conditions = []
    params: list = [limit]
    idx = 2

    if source:
        conditions.append(f"l.source = ${idx}")
        params.append(source)
        idx += 1
    if status:
        conditions.append(f"l.status = ${idx}")
        params.append(status)
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    rows = await pool.fetch(
        f"""
        SELECT
            l.id,
            l.source,
            l.status,
            l.reviews_found,
            l.reviews_inserted,
            l.pages_scraped,
            l.duration_ms,
            l.errors,
            l.started_at,
            l.captcha_attempts,
            l.captcha_types,
            l.captcha_solve_ms,
            l.block_type,
            l.parser_version,
            l.proxy_type,
            l.stop_reason,
            l.oldest_review,
            l.newest_review,
            l.date_dropped,
            l.duplicate_pages,
            l.has_page_logs,
            jsonb_array_length(l.errors)            AS error_count,
            t.vendor_name,
            t.product_name,
            t.product_slug
        FROM b2b_scrape_log l
        JOIN b2b_scrape_targets t ON t.id = l.target_id
        {where}
        ORDER BY l.started_at DESC
        LIMIT $1
        """,
        *params,
    )

    return {
        "scrapes": [
            {
                "id": str(r["id"]),
                "source": r["source"],
                "status": r["status"],
                "vendor_name": r["vendor_name"],
                "product_name": r["product_name"],
                "product_slug": r["product_slug"],
                "reviews_found": r["reviews_found"],
                "reviews_inserted": r["reviews_inserted"],
                "insert_rate": (
                    round(r["reviews_inserted"] / r["reviews_found"], 3)
                    if r["reviews_found"] else 0.0
                ),
                "pages_scraped": r["pages_scraped"],
                "duration_ms": r["duration_ms"],
                "error_count": int(r["error_count"] or 0),
                "errors": r["errors"] if isinstance(r["errors"], list) else [],
                "captcha_attempts": r["captcha_attempts"] or 0,
                "captcha_types": r["captcha_types"] or [],
                "captcha_solve_ms": r["captcha_solve_ms"],
                "block_type": r["block_type"],
                "parser_version": r["parser_version"],
                "proxy_type": r["proxy_type"],
                "stop_reason": r["stop_reason"],
                "oldest_review": r["oldest_review"].isoformat() if r["oldest_review"] else None,
                "newest_review": r["newest_review"].isoformat() if r["newest_review"] else None,
                "date_dropped": r["date_dropped"] or 0,
                "duplicate_pages": r["duplicate_pages"] or 0,
                "has_page_logs": r["has_page_logs"] or False,
                "started_at": r["started_at"].isoformat() if r["started_at"] else None,
            }
            for r in rows
        ],
    }


@router.get("/scraping/runs/{run_id}/pages")
async def scraping_run_pages(run_id: str):
    """
    Page-level telemetry for a specific scrape run.

    Returns per-page diagnostics: URL requested, status code, review counts,
    date range, content hash, duplicate detection, and stop reason.
    Only available when the run has ``has_page_logs=true``.
    """
    pool = _pool_or_503()
    import uuid as _uuid

    try:
        rid = _uuid.UUID(run_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    # Verify run exists and get summary
    run_row = await pool.fetchrow(
        """
        SELECT l.id, l.source, l.status, l.stop_reason, l.pages_scraped,
               l.reviews_found, l.reviews_inserted, l.has_page_logs,
               l.oldest_review, l.newest_review, l.date_dropped,
               l.duplicate_pages, l.started_at, l.duration_ms,
               t.vendor_name
        FROM b2b_scrape_log l
        JOIN b2b_scrape_targets t ON t.id = l.target_id
        WHERE l.id = $1
        """,
        rid,
    )
    if not run_row:
        raise HTTPException(status_code=404, detail="Scrape run not found")

    pages = await pool.fetch(
        """
        SELECT page, url, requested_at, status_code, final_url,
               response_bytes, duration_ms,
               review_nodes_found, reviews_parsed,
               missing_date, missing_rating, missing_body, missing_author,
               oldest_review, newest_review,
               next_page_found, next_page_url, content_hash,
               duplicate_reviews, stop_reason, errors
        FROM b2b_scrape_page_logs
        WHERE run_id = $1
        ORDER BY page
        """,
        rid,
    )

    return {
        "run": {
            "id": str(run_row["id"]),
            "source": run_row["source"],
            "vendor_name": run_row["vendor_name"],
            "status": run_row["status"],
            "stop_reason": run_row["stop_reason"],
            "pages_scraped": run_row["pages_scraped"],
            "reviews_found": run_row["reviews_found"],
            "reviews_inserted": run_row["reviews_inserted"],
            "oldest_review": run_row["oldest_review"].isoformat() if run_row["oldest_review"] else None,
            "newest_review": run_row["newest_review"].isoformat() if run_row["newest_review"] else None,
            "date_dropped": run_row["date_dropped"] or 0,
            "duplicate_pages": run_row["duplicate_pages"] or 0,
            "duration_ms": run_row["duration_ms"],
            "started_at": run_row["started_at"].isoformat() if run_row["started_at"] else None,
        },
        "pages": [
            {
                "page": p["page"],
                "url": p["url"],
                "requested_at": p["requested_at"].isoformat() if p["requested_at"] else None,
                "status_code": p["status_code"],
                "final_url": p["final_url"],
                "response_bytes": p["response_bytes"],
                "duration_ms": p["duration_ms"],
                "review_nodes_found": p["review_nodes_found"],
                "reviews_parsed": p["reviews_parsed"],
                "missing_date": p["missing_date"],
                "missing_rating": p["missing_rating"],
                "missing_body": p["missing_body"],
                "missing_author": p["missing_author"],
                "oldest_review": p["oldest_review"].isoformat() if p["oldest_review"] else None,
                "newest_review": p["newest_review"].isoformat() if p["newest_review"] else None,
                "next_page_found": p["next_page_found"],
                "next_page_url": p["next_page_url"],
                "content_hash": p["content_hash"],
                "duplicate_reviews": p["duplicate_reviews"],
                "stop_reason": p["stop_reason"],
                "errors": p["errors"] if isinstance(p["errors"], list) else [],
            }
            for p in pages
        ],
        "page_count": len(pages),
    }


@router.get("/scraping/top-posts")
async def scraping_top_posts(
    limit: int = Query(default=25, ge=1, le=100),
    source: str = Query(default="reddit"),
    min_weight: float = Query(default=0.6, ge=0.0, le=1.0),
):
    """
    High-value scraped posts filtered by source_weight, trending score,
    or author churn score.  Useful for spot-checking signal quality and
    validating that the enrichment pipeline is working on the right posts.

    Ordered by source_weight DESC then imported_at DESC.
    """
    pool = _pool_or_503()

    rows = await pool.fetch(
        """
        SELECT
            id,
            source,
            vendor_name,
            summary,
            source_url,
            reviewed_at,
            imported_at,
            enrichment_status,
            (raw_metadata->>'source_weight')::numeric                        AS source_weight,
            raw_metadata->>'trending_score'                                  AS trending_score,
            (raw_metadata->>'author_churn_score')::numeric                   AS author_churn_score,
            raw_metadata->>'subreddit'                                       AS subreddit,
            (raw_metadata->>'score')::int                                    AS reddit_score,
            (raw_metadata->>'num_comments')::int                             AS num_comments,
            raw_metadata->>'post_flair'                                      AS post_flair,
            (raw_metadata->>'is_edited')::boolean                            AS is_edited,
            (raw_metadata->>'is_crosspost')::boolean                         AS is_crosspost,
            COALESCE(jsonb_array_length(raw_metadata->'comment_threads'), 0) AS comment_count
        FROM b2b_reviews
        WHERE source = $1
          AND (raw_metadata->>'source_weight')::numeric >= $2
        ORDER BY (raw_metadata->>'source_weight')::numeric DESC NULLS LAST,
                 imported_at DESC
        LIMIT $3
        """,
        source,
        min_weight,
        limit,
    )

    return {
        "source": source,
        "min_weight": min_weight,
        "posts": [
            {
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "summary": r["summary"],
                "source_url": r["source_url"],
                "reviewed_at": r["reviewed_at"].isoformat() if r["reviewed_at"] else None,
                "imported_at": r["imported_at"].isoformat() if r["imported_at"] else None,
                "enrichment_status": r["enrichment_status"],
                "source_weight": float(r["source_weight"] or 0),
                "trending_score": r["trending_score"],
                "author_churn_score": float(r["author_churn_score"] or 0),
                "subreddit": r["subreddit"],
                "reddit_score": r["reddit_score"],
                "num_comments": r["num_comments"],
                "post_flair": r["post_flair"],
                "is_edited": r["is_edited"] or False,
                "is_crosspost": r["is_crosspost"] or False,
                "comment_count": int(r["comment_count"]),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Reddit scraper -- deep monitoring sub-section
# ---------------------------------------------------------------------------


@router.get("/scraping/reddit/overview")
async def reddit_overview(days: int = Query(default=7, ge=1, le=30)):
    """
    Reddit scraper health dashboard.

    Covers auth mode, raw throughput, rate-limit events (parsed from the
    errors JSONB), the triage/enrichment signal funnel, and final actionable
    signal conversion (intent_to_leave + high_urgency).
    """
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # -- Scrape-log stats: runs, throughput, 429s --------------------------
    log_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*)                                                              AS total_runs,
            COUNT(*) FILTER (WHERE l.status = 'completed')                       AS completed,
            COUNT(*) FILTER (WHERE l.status = 'failed')                          AS failed,
            COUNT(*) FILTER (WHERE l.status = 'blocked')                         AS blocked,
            COUNT(*) FILTER (WHERE l.status = 'partial')                         AS partial,
            COALESCE(SUM(l.reviews_found), 0)                                    AS reviews_found,
            COALESCE(SUM(l.reviews_inserted), 0)                                 AS reviews_inserted,
            COALESCE(AVG(l.duration_ms), 0)                                      AS avg_duration_ms,
            COALESCE(SUM(l.pages_scraped), 0)                                    AS pages_scraped_total,
            -- Auth mode: reddit:3 = OAuth2 v3, reddit:2 = OAuth2 v2, reddit:1 = public
            -- MAX picks the newest version string that ran in this window
            MAX(l.parser_version)                                                 AS dominant_parser,
            -- Rate-limit events: count runs that had any 429 in their errors array
            COUNT(*) FILTER (
                WHERE EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(l.errors) e
                    WHERE e LIKE '%429%'
                )
            )                                                                     AS runs_with_429s,
            -- Total individual 429 occurrences across all runs
            COALESCE(SUM((
                SELECT COUNT(*) FROM jsonb_array_elements_text(l.errors) e
                WHERE e LIKE '%429%'
            )), 0)                                                                AS total_429_events
        FROM b2b_scrape_log l
        WHERE l.source = 'reddit'
          AND l.started_at >= $1
        """,
        since,
    )

    # -- Signal funnel: enrichment status breakdown -------------------------
    funnel_rows = await pool.fetch(
        """
        SELECT
            enrichment_status,
            COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        GROUP BY enrichment_status
        """,
        since,
    )
    funnel: dict[str, int] = {r["enrichment_status"]: int(r["cnt"]) for r in funnel_rows}
    enriched   = funnel.get("enriched", 0)
    no_signal  = funnel.get("no_signal", 0)
    failed_enr = funnel.get("failed", 0)
    pending    = funnel.get("pending", 0) + funnel.get("enriching", 0)
    inserted   = sum(funnel.values())
    triage_denominator = enriched + no_signal

    # -- Signal conversion: intent_to_leave + high urgency -----------------
    conversion_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE COALESCE(enrichment->'churn_signals'->>'intent_to_leave', 'false')::boolean
            )                                                                AS intent_to_leave,
            COUNT(*) FILTER (
                WHERE (enrichment->>'urgency_score')::numeric >= 7
            )                                                                AS high_urgency,
            COUNT(*) FILTER (
                WHERE COALESCE(enrichment->'churn_signals'->>'intent_to_leave', 'false')::boolean
                   OR (enrichment->>'urgency_score')::numeric >= 7
            )                                                                AS actionable
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
          AND enrichment_status = 'enriched'
        """,
        since,
    )

    # Derive auth mode from the most common parser version seen in this window
    dominant = log_row["dominant_parser"] or ""
    # reddit:2+ = OAuth2 (authenticated API); reddit:1 = public fallback
    auth_mode = "oauth2" if any(v in dominant for v in ("reddit:2", "reddit:3")) else ("public" if dominant else "unknown")

    reviews_found    = int(log_row["reviews_found"])
    reviews_inserted = int(log_row["reviews_inserted"])
    intent_to_leave  = int(conversion_row["intent_to_leave"])
    high_urgency     = int(conversion_row["high_urgency"])
    actionable       = int(conversion_row["actionable"])

    return {
        "period_days": days,
        "auth_mode": auth_mode,
        "runs": {
            "total":     int(log_row["total_runs"]),
            "completed": int(log_row["completed"]),
            "failed":    int(log_row["failed"]),
            "blocked":   int(log_row["blocked"]),
            "partial":   int(log_row["partial"]),
        },
        "throughput": {
            "reviews_found":       reviews_found,
            "reviews_inserted":    reviews_inserted,
            "insert_rate":         round(reviews_inserted / reviews_found, 3) if reviews_found else 0.0,
            "avg_duration_ms":     round(float(log_row["avg_duration_ms"]), 1),
            "pages_scraped_total": int(log_row["pages_scraped_total"]),
        },
        "rate_limits": {
            "runs_with_429s":   int(log_row["runs_with_429s"]),
            "total_429_events": int(log_row["total_429_events"]),
        },
        "signal_funnel": {
            "inserted":                  inserted,
            "enriched":                  enriched,
            "no_signal":                 no_signal,
            "failed":                    failed_enr,
            "pending":                   pending,
            "triage_pass_rate":          round(enriched / triage_denominator, 3) if triage_denominator else 0.0,
            "enrichment_completion_rate": round(enriched / inserted, 3) if inserted else 0.0,
        },
        "signal_conversion": {
            "intent_to_leave": intent_to_leave,
            "high_urgency":    high_urgency,
            "actionable":      actionable,
            "actionable_rate": round(actionable / inserted, 3) if inserted else 0.0,
        },
    }


@router.get("/scraping/reddit/by-subreddit")
async def reddit_by_subreddit(days: int = Query(default=30, ge=1, le=90)):
    """
    Per-subreddit signal yield.

    Shows which subreddits are producing actionable intelligence vs noise.
    Use this to tune the DEFAULT_SUBREDDITS list in the Reddit parser.
    """
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await pool.fetch(
        """
        SELECT
            raw_metadata->>'subreddit'                                                   AS subreddit,
            COUNT(*)                                                                      AS total_posts,
            COUNT(*) FILTER (WHERE (raw_metadata->>'source_weight')::numeric > 0.7)      AS high_signal_posts,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')                       AS enriched_posts,
            COUNT(*) FILTER (WHERE enrichment_status = 'no_signal')                      AS no_signal_posts,
            COUNT(*) FILTER (WHERE enrichment_status = 'failed')                         AS failed_posts,
            ROUND(AVG((raw_metadata->>'source_weight')::numeric)::numeric, 3)            AS avg_source_weight,
            ROUND(AVG(
                CASE WHEN enrichment_status = 'enriched'
                     THEN (enrichment->>'urgency_score')::numeric END
            )::numeric, 2)                                                                AS avg_urgency_score,
            ROUND(AVG(
                CASE WHEN (raw_metadata->>'score') ~ '^\-?[0-9]+$'
                     THEN (raw_metadata->>'score')::int END
            )::numeric, 1)                                                                AS avg_reddit_score,
            COUNT(*) FILTER (
                WHERE raw_metadata->>'trending_score' = 'high'
            )                                                                             AS trending_high_count,
            COUNT(*) FILTER (
                WHERE COALESCE(jsonb_array_length(raw_metadata->'comment_threads'), 0) > 0
            )                                                                             AS comment_harvested_count
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
          AND raw_metadata->>'subreddit' IS NOT NULL
          AND raw_metadata->>'subreddit' != ''
        GROUP BY raw_metadata->>'subreddit'
        ORDER BY enriched_posts DESC, total_posts DESC
        """,
        since,
    )

    def _sub(r: dict) -> dict:
        total    = int(r["total_posts"])
        enriched = int(r["enriched_posts"])
        no_sig   = int(r["no_signal_posts"])
        high_sig = int(r["high_signal_posts"])
        triage_d = enriched + no_sig
        return {
            "subreddit":               r["subreddit"],
            "total_posts":             total,
            "high_signal_posts":       high_sig,
            "signal_rate":             round(high_sig / total, 3) if total else 0.0,
            "enriched_posts":          enriched,
            "triage_pass_rate":        round(enriched / triage_d, 3) if triage_d else 0.0,
            "no_signal_posts":         no_sig,
            "failed_posts":            int(r["failed_posts"]),
            "avg_source_weight":       float(r["avg_source_weight"] or 0),
            "avg_urgency_score":       float(r["avg_urgency_score"] or 0),
            "avg_reddit_score":        float(r["avg_reddit_score"] or 0),
            "trending_high_count":     int(r["trending_high_count"]),
            "comment_harvested_count": int(r["comment_harvested_count"]),
        }

    return {
        "period_days": days,
        "subreddits": [_sub(dict(r)) for r in rows],
    }


@router.get("/scraping/reddit/signal-breakdown")
async def reddit_signal_breakdown(days: int = Query(default=30, ge=1, le=90)):
    """
    Post-quality characteristics for tuning the Reddit scraper.

    Covers flair vs signal correlation, edited/crosspost amplification,
    comment harvest stats, author churn score distribution, post age,
    and trending score spread.
    """
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # -- Flair analysis -------------------------------------------------------
    flair_rows = await pool.fetch(
        """
        SELECT
            COALESCE(NULLIF(raw_metadata->>'post_flair', ''), '(no flair)')  AS flair,
            COUNT(*)                                                           AS count,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')            AS enriched,
            ROUND(AVG((raw_metadata->>'source_weight')::numeric)::numeric, 3) AS avg_weight
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        GROUP BY raw_metadata->>'post_flair'
        ORDER BY count DESC
        LIMIT 15
        """,
        since,
    )

    # -- Edit stats -----------------------------------------------------------
    edit_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE (raw_metadata->>'is_edited')::boolean)    AS edited_posts,
            COUNT(*) FILTER (WHERE NOT (raw_metadata->>'is_edited')::boolean
                               OR raw_metadata->>'is_edited' IS NULL)        AS unedited_posts,
            COUNT(*) FILTER (WHERE (raw_metadata->>'is_edited')::boolean
                               AND enrichment_status = 'enriched')           AS edited_enriched,
            COUNT(*) FILTER (WHERE (NOT (raw_metadata->>'is_edited')::boolean
                               OR raw_metadata->>'is_edited' IS NULL)
                               AND enrichment_status = 'enriched')           AS unedited_enriched
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Crosspost stats ------------------------------------------------------
    crosspost_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE (raw_metadata->>'is_crosspost')::boolean)  AS crossposts,
            COUNT(*) FILTER (WHERE (raw_metadata->>'is_crosspost')::boolean
                               AND enrichment_status = 'enriched')            AS crosspost_enriched,
            -- Count total unique extra subreddits reached via crossposts
            COALESCE((
                SELECT COUNT(DISTINCT sub)
                FROM b2b_reviews r2,
                     jsonb_array_elements_text(r2.raw_metadata->'crosspost_subreddits') AS sub
                WHERE r2.source = 'reddit'
                  AND r2.imported_at >= $1
                  AND (r2.raw_metadata->>'is_crosspost')::boolean
            ), 0)                                                              AS crosspost_subreddits_reached
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Comment harvest stats ------------------------------------------------
    comment_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE COALESCE(jsonb_array_length(raw_metadata->'comment_threads'), 0) > 0
            )                                                                  AS posts_with_comments,
            ROUND(AVG(
                COALESCE(jsonb_array_length(raw_metadata->'comment_threads'), 0)
            )::numeric, 2)                                                     AS avg_comments_fetched,
            COUNT(*)                                                            AS total_posts
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Author churn score distribution + stats ------------------------------
    author_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE (raw_metadata->>'author_churn_score')::numeric >= 7
            )                                                                  AS high_score_authors,
            ROUND(AVG((raw_metadata->>'author_churn_score')::numeric)::numeric, 2)
                                                                               AS avg_churn_score,
            COUNT(*) FILTER (
                WHERE (raw_metadata->>'author_churn_score')::numeric < 3
            )                                                                  AS score_0_2,
            COUNT(*) FILTER (
                WHERE (raw_metadata->>'author_churn_score')::numeric >= 3
                  AND (raw_metadata->>'author_churn_score')::numeric < 5
            )                                                                  AS score_3_4,
            COUNT(*) FILTER (
                WHERE (raw_metadata->>'author_churn_score')::numeric >= 5
                  AND (raw_metadata->>'author_churn_score')::numeric < 7
            )                                                                  AS score_5_6,
            COUNT(*) FILTER (
                WHERE (raw_metadata->>'author_churn_score')::numeric >= 7
            )                                                                  AS score_7_10
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Post age distribution (reviewed_at = when content was written) -------
    age_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE reviewed_at >= NOW() - INTERVAL '7 days')   AS last_7d,
            COUNT(*) FILTER (WHERE reviewed_at >= NOW() - INTERVAL '30 days'
                               AND reviewed_at < NOW() - INTERVAL '7 days')    AS last_8_to_30d,
            COUNT(*) FILTER (WHERE reviewed_at >= NOW() - INTERVAL '90 days'
                               AND reviewed_at < NOW() - INTERVAL '30 days')   AS last_31_to_90d,
            COUNT(*) FILTER (WHERE reviewed_at < NOW() - INTERVAL '90 days'
                               OR reviewed_at IS NULL)                          AS older
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Trending distribution ------------------------------------------------
    trending_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE raw_metadata->>'trending_score' = 'high')   AS trending_high,
            COUNT(*) FILTER (WHERE raw_metadata->>'trending_score' = 'medium') AS trending_medium,
            COUNT(*) FILTER (WHERE raw_metadata->>'trending_score' = 'low'
                               OR raw_metadata->>'trending_score' IS NULL)     AS trending_low
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Assemble ---------------------------------------------------------
    def _flair(r: dict) -> dict:
        cnt      = int(r["count"])
        enriched = int(r["enriched"])
        return {
            "flair":       r["flair"],
            "count":       cnt,
            "signal_rate": round(enriched / cnt, 3) if cnt else 0.0,
            "avg_weight":  float(r["avg_weight"] or 0),
        }

    edited   = int(edit_row["edited_posts"])
    unedited = int(edit_row["unedited_posts"])
    ed_enr   = int(edit_row["edited_enriched"])
    un_enr   = int(edit_row["unedited_enriched"])

    crossposts     = int(crosspost_row["crossposts"])
    cp_enriched    = int(crosspost_row["crosspost_enriched"])
    total_posts    = int(comment_row["total_posts"])

    return {
        "period_days": days,
        "flair_analysis": [_flair(dict(r)) for r in flair_rows],
        "edit_stats": {
            "edited_posts":         edited,
            "edited_signal_rate":   round(ed_enr / edited, 3) if edited else 0.0,
            "unedited_signal_rate": round(un_enr / unedited, 3) if unedited else 0.0,
        },
        "crosspost_stats": {
            "crossposts":                   crossposts,
            "crosspost_signal_rate":        round(cp_enriched / crossposts, 3) if crossposts else 0.0,
            "crosspost_subreddits_reached": int(crosspost_row["crosspost_subreddits_reached"]),
        },
        "comment_harvest_stats": {
            "posts_with_comments":  int(comment_row["posts_with_comments"]),
            "avg_comments_fetched": float(comment_row["avg_comments_fetched"] or 0),
            "comment_trigger_rate": round(
                int(comment_row["posts_with_comments"]) / total_posts, 3
            ) if total_posts else 0.0,
        },
        "author_churn_stats": {
            "high_score_authors": int(author_row["high_score_authors"]),
            "avg_churn_score":    float(author_row["avg_churn_score"] or 0),
            "score_distribution": {
                "0-2":  int(author_row["score_0_2"]),
                "3-4":  int(author_row["score_3_4"]),
                "5-6":  int(author_row["score_5_6"]),
                "7-10": int(author_row["score_7_10"]),
            },
        },
        "post_age_distribution": {
            "last_7d":       int(age_row["last_7d"]),
            "last_8_to_30d": int(age_row["last_8_to_30d"]),
            "last_31_to_90d":int(age_row["last_31_to_90d"]),
            "older":         int(age_row["older"]),
        },
        "trending_distribution": {
            "high":   int(trending_row["trending_high"]),
            "medium": int(trending_row["trending_medium"]),
            "low":    int(trending_row["trending_low"]),
        },
    }


@router.get("/scraping/reddit/per-vendor")
async def reddit_per_vendor(
    days: int  = Query(default=30, ge=1, le=90),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Per-vendor Reddit signal breakdown.

    Identifies which vendor targets are worth scraping on Reddit vs which
    produce only noise.  Includes top subreddits and pain categories per
    vendor to guide targeting decisions.
    """
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Main per-vendor aggregation
    rows = await pool.fetch(
        """
        SELECT
            vendor_name,
            COUNT(*)                                                                     AS inserted,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')                      AS enriched,
            COUNT(*) FILTER (WHERE enrichment_status = 'no_signal')                     AS no_signal,
            COUNT(*) FILTER (WHERE enrichment_status = 'failed')                        AS failed,
            ROUND(AVG((raw_metadata->>'source_weight')::numeric)::numeric, 3)           AS avg_source_weight,
            ROUND(AVG(
                CASE WHEN enrichment_status = 'enriched'
                     THEN (enrichment->>'urgency_score')::numeric END
            )::numeric, 2)                                                               AS avg_urgency_score,
            COUNT(*) FILTER (
                WHERE COALESCE(enrichment->'churn_signals'->>'intent_to_leave', 'false')::boolean
            )                                                                            AS intent_to_leave_count,
            COUNT(*) FILTER (
                WHERE (enrichment->>'urgency_score')::numeric >= 7
            )                                                                            AS high_urgency_count,
            COUNT(*) FILTER (
                WHERE raw_metadata->>'trending_score' = 'high'
            )                                                                            AS trending_high_count
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
        GROUP BY vendor_name
        ORDER BY enriched DESC, inserted DESC
        LIMIT $2
        """,
        since,
        limit,
    )

    if not rows:
        return {"period_days": days, "vendors": []}

    vendor_names = [r["vendor_name"] for r in rows]

    # Top 3 subreddits per vendor (separate query to avoid heavy aggregation inline)
    sub_rows = await pool.fetch(
        """
        SELECT
            vendor_name,
            raw_metadata->>'subreddit'   AS subreddit,
            COUNT(*)                      AS cnt
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
          AND vendor_name = ANY($2)
          AND raw_metadata->>'subreddit' IS NOT NULL
          AND raw_metadata->>'subreddit' != ''
        GROUP BY vendor_name, raw_metadata->>'subreddit'
        ORDER BY vendor_name, cnt DESC
        """,
        since,
        vendor_names,
    )

    # Build vendor -> top subreddits map (top 3)
    top_subs: dict[str, list[str]] = {}
    for r in sub_rows:
        vn = r["vendor_name"]
        if vn not in top_subs:
            top_subs[vn] = []
        if len(top_subs[vn]) < 3:
            top_subs[vn].append(r["subreddit"])

    # Top 3 pain categories per vendor
    pain_rows = await pool.fetch(
        """
        SELECT
            vendor_name,
            enrichment->>'pain_category'  AS pain_category,
            COUNT(*)                       AS cnt
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND imported_at >= $1
          AND vendor_name = ANY($2)
          AND enrichment_status = 'enriched'
          AND enrichment->>'pain_category' IS NOT NULL
        GROUP BY vendor_name, enrichment->>'pain_category'
        ORDER BY vendor_name, cnt DESC
        """,
        since,
        vendor_names,
    )

    top_pain: dict[str, list[str]] = {}
    for r in pain_rows:
        vn = r["vendor_name"]
        if vn not in top_pain:
            top_pain[vn] = []
        if len(top_pain[vn]) < 3:
            top_pain[vn].append(r["pain_category"])

    def _vendor(r: dict) -> dict:
        vn       = r["vendor_name"]
        inserted = int(r["inserted"])
        enriched = int(r["enriched"])
        no_sig   = int(r["no_signal"])
        triage_d = enriched + no_sig
        return {
            "vendor_name":          vn,
            "inserted":             inserted,
            "enriched":             enriched,
            "no_signal":            no_sig,
            "failed":               int(r["failed"]),
            "triage_pass_rate":     round(enriched / triage_d, 3) if triage_d else 0.0,
            "avg_source_weight":    float(r["avg_source_weight"] or 0),
            "avg_urgency_score":    float(r["avg_urgency_score"] or 0),
            "intent_to_leave_count": int(r["intent_to_leave_count"]),
            "high_urgency_count":   int(r["high_urgency_count"]),
            "trending_high_count":  int(r["trending_high_count"]),
            "top_subreddits":       top_subs.get(vn, []),
            "top_pain_categories":  top_pain.get(vn, []),
        }

    return {
        "period_days": days,
        "vendors": [_vendor(dict(r)) for r in rows],
    }


# ---------------------------------------------------------------------------
# System resources (CPU + RAM + network + GPU)
# ---------------------------------------------------------------------------

# Module-level state for computing network throughput between polls
_last_net_io = psutil.net_io_counters()
_last_net_time = time.monotonic()


def _get_gpu_stats() -> dict | None:
    """Query nvidia-smi for GPU utilization, VRAM, and temperature."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return None

        util_pct = float(parts[0])
        vram_used_mb = float(parts[1])
        vram_total_mb = float(parts[2])
        temp_c = int(float(parts[3]))
        gpu_name = parts[4]

        vram_used_gb = round(vram_used_mb / 1024, 1)
        vram_total_gb = round(vram_total_mb / 1024, 1)

        return {
            "name": gpu_name,
            "utilization_percent": round(util_pct, 1),
            "vram_used_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
            "vram_percent": round(vram_used_mb / vram_total_mb * 100, 1) if vram_total_mb else 0.0,
            "temperature_c": temp_c,
        }
    except Exception:
        return None


@router.get("/system-resources")
async def system_resources():
    """CPU, RAM, network throughput, and GPU stats."""
    global _last_net_io, _last_net_time

    # CPU (non-blocking, kernel-cached)
    cpu = psutil.cpu_percent(interval=None)

    # Memory
    mem = psutil.virtual_memory()
    mem_used_gb = round(mem.used / (1024 ** 3), 1)
    mem_total_gb = round(mem.total / (1024 ** 3), 1)

    # Network throughput delta
    now = time.monotonic()
    cur_net = psutil.net_io_counters()
    elapsed = now - _last_net_time or 0.001
    bytes_delta = (
        (cur_net.bytes_sent - _last_net_io.bytes_sent)
        + (cur_net.bytes_recv - _last_net_io.bytes_recv)
    )
    mbps = round((bytes_delta * 8) / elapsed / 1_000_000, 1)
    _last_net_io = cur_net
    _last_net_time = now

    # GPU
    gpu = _get_gpu_stats()

    return {
        "cpu_percent": round(cpu, 1),
        "mem_percent": round(mem.percent, 1),
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": mem_total_gb,
        "net_mbps": max(0.0, mbps),
        "gpu": gpu,
    }
