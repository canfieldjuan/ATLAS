"""Admin cost analytics API.

Aggregates LLM usage from the local llm_usage table for the cost dashboard.
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import psutil
from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..services.b2b.cache_strategy import iter_core_b2b_cache_strategies
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


def _normalize_metadata(metadata: object) -> dict:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str) and metadata.strip():
        try:
            parsed = json.loads(metadata)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _humanize_identifier(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("/", " ").replace(".", " ").replace("_", " ").strip().title()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_task_result_payload(result_text: str | None) -> dict[str, Any]:
    text = str(result_text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _b2b_vendor_name(row_vendor: Any, metadata: dict) -> str:
    candidate = str(row_vendor or "").strip()
    if candidate:
        return candidate
    nested = _recent_metadata_value(metadata, "vendor_name")
    return str(nested or "").strip()


def _b2b_source_name(metadata: dict) -> str:
    for key in ("source", "source_name"):
        value = _recent_metadata_value(metadata, key)
        if value:
            return value
    return ""


def _classify_b2b_pass(span_name: str, metadata: dict) -> str | None:
    skill = str(metadata.get("skill") or "").strip().lower()
    stage = str(metadata.get("stage") or "").strip().lower()
    workflow = str(_recent_metadata_value(metadata, "workflow") or "").strip().lower()
    source_name = str(_recent_metadata_value(metadata, "source_name") or "").strip().lower()

    if span_name in {
        "task.b2b_enrichment_repair.extraction",
        "pipeline.digest/b2b_churn_repair_extraction",
    }:
        return "repair"
    if skill == "digest/b2b_churn_repair_extraction" or stage == "repair_extraction":
        return "repair"

    if span_name in {"task.b2b_enrichment.tier1", "task.b2b_enrichment.tier2"}:
        return "extraction"
    if skill in {
        "digest/b2b_churn_extraction_tier1",
        "digest/b2b_churn_extraction_tier2",
    }:
        return "extraction"

    if span_name.startswith("reasoning.stratified.") or span_name.startswith("reasoning.cross_vendor."):
        return "reasoning"
    if span_name in {"task.b2b_reasoning_synthesis", "task.b2b_reasoning_synthesis.cross_vendor"}:
        return "reasoning"
    if span_name == "reasoning.process" and source_name == "b2b_churn_intelligence":
        return "reasoning"
    if workflow in {"cross_vendor_reasoning"}:
        return "reasoning"

    return None


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
    if span_name == "reasoning.stratified.reconstitute.reason":
        return "Stratified Reasoning", f"Reconstitute classify{f' for {vendor_name}' if vendor_name else ''}"
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


def _build_recent_filters(
    *,
    days: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    span_name: str | None = None,
    operation_type: str | None = None,
    status: str | None = None,
    cache_only: bool | None = None,
) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    args: list[object] = []

    def _add(value: object) -> str:
        args.append(value)
        return f"${len(args)}"

    if days is not None:
        clauses.append(f"created_at >= {_add(datetime.now(timezone.utc) - timedelta(days=days))}")
    if provider:
        clauses.append(f"model_provider = {_add(provider)}")
    if model:
        clauses.append(f"model_name = {_add(model)}")
    if span_name:
        clauses.append(f"span_name = {_add(span_name)}")
    if operation_type:
        clauses.append(f"operation_type = {_add(operation_type)}")
    if status:
        clauses.append(f"status = {_add(status)}")
    if cache_only is True:
        clauses.append("(cached_tokens > 0 OR cache_write_tokens > 0)")
    elif cache_only is False:
        clauses.append("(cached_tokens = 0 AND cache_write_tokens = 0)")
    return clauses, args


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


async def _safe_fetchrow(pool, query: str, *args):
    try:
        return await pool.fetchrow(query, *args)
    except Exception:
        logger.exception("admin_costs.fetchrow_failed")
        return None


async def _safe_fetch(pool, query: str, *args):
    try:
        return await pool.fetch(query, *args)
    except Exception:
        logger.exception("admin_costs.fetch_failed")
        return []


@router.get("/summary")
async def cost_summary(days: int = Query(default=30, ge=1, le=365)):
    """High-level cost summary for the dashboard header cards."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    row = await pool.fetchrow(
        """SELECT
             COALESCE(SUM(cost_usd), 0)         AS total_cost,
             COALESCE(SUM(input_tokens), 0)      AS total_input,
             COALESCE(SUM(billable_input_tokens), 0) AS total_billable_input,
             COALESCE(SUM(cached_tokens), 0)    AS total_cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0) AS total_cache_write_tokens,
             COALESCE(SUM(output_tokens), 0)     AS total_output,
             COALESCE(SUM(total_tokens), 0)      AS total_tokens,
             COUNT(*)                             AS total_calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
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
        "total_billable_input_tokens": int(row["total_billable_input"]),
        "total_cached_tokens": int(row["total_cached_tokens"]),
        "total_cache_write_tokens": int(row["total_cache_write_tokens"]),
        "total_output_tokens": int(row["total_output"]),
        "total_tokens": int(row["total_tokens"]),
        "total_calls": int(row["total_calls"]),
        "cache_hit_calls": int(row["cache_hit_calls"]),
        "cache_write_calls": int(row["cache_write_calls"]),
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
             COALESCE(SUM(billable_input_tokens), 0) AS billable_input_tokens,
             COALESCE(SUM(cached_tokens), 0)      AS cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
             COALESCE(SUM(output_tokens), 0)      AS output_tokens,
             COUNT(*)                              AS calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls
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
                "billable_input_tokens": int(r["billable_input_tokens"]),
                "cached_tokens": int(r["cached_tokens"]),
                "cache_write_tokens": int(r["cache_write_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "calls": int(r["calls"]),
                "cache_hit_calls": int(r["cache_hit_calls"]),
                "cache_write_calls": int(r["cache_write_calls"]),
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
             COALESCE(SUM(input_tokens), 0)        AS input_tokens,
             COALESCE(SUM(billable_input_tokens), 0) AS billable_input_tokens,
             COALESCE(SUM(cached_tokens), 0)       AS cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0)  AS cache_write_tokens,
             COALESCE(SUM(output_tokens), 0)       AS output_tokens,
             COALESCE(SUM(total_tokens), 0)        AS tokens,
             COUNT(*)                               AS calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
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
                "input_tokens": int(r["input_tokens"]),
                "billable_input_tokens": int(r["billable_input_tokens"]),
                "cached_tokens": int(r["cached_tokens"]),
                "cache_write_tokens": int(r["cache_write_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "total_tokens": int(r["tokens"]),
                "calls": int(r["calls"]),
                "cache_hit_calls": int(r["cache_hit_calls"]),
                "cache_write_calls": int(r["cache_write_calls"]),
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
             operation_type,
             COALESCE(SUM(cost_usd), 0)        AS cost,
             COALESCE(SUM(input_tokens), 0)    AS input_tokens,
             COALESCE(SUM(billable_input_tokens), 0) AS billable_input_tokens,
             COALESCE(SUM(cached_tokens), 0)   AS cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
             COALESCE(SUM(output_tokens), 0)   AS output_tokens,
             COALESCE(SUM(total_tokens), 0)     AS tokens,
             COUNT(*)                            AS calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
             COALESCE(AVG(duration_ms), 0)       AS avg_duration_ms
           FROM llm_usage
           WHERE created_at >= $1
           GROUP BY span_name, operation_type
           ORDER BY cost DESC""",
        since,
    )
    return {
        "period_days": days,
        "workflows": [
            {
                "workflow": r["span_name"],
                "operation_type": r["operation_type"],
                "cost_usd": float(r["cost"]),
                "input_tokens": int(r["input_tokens"]),
                "billable_input_tokens": int(r["billable_input_tokens"]),
                "cached_tokens": int(r["cached_tokens"]),
                "cache_write_tokens": int(r["cache_write_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "total_tokens": int(r["tokens"]),
                "calls": int(r["calls"]),
                "cache_hit_calls": int(r["cache_hit_calls"]),
                "cache_write_calls": int(r["cache_write_calls"]),
                "avg_duration_ms": round(float(r["avg_duration_ms"]), 1),
            }
            for r in rows
        ],
    }


@router.get("/by-operation")
async def cost_by_operation(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=100, ge=1, le=500),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    span_name: str | None = Query(default=None),
    operation_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    cache_only: bool | None = Query(default=None),
):
    """Detailed cost rollup by operation + provider + model."""
    pool = _pool_or_503()
    clauses, args = _build_recent_filters(
        days=days,
        provider=provider,
        model=model,
        span_name=span_name,
        operation_type=operation_type,
        status=status,
        cache_only=cache_only,
    )
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    rows = await pool.fetch(
        f"""SELECT
             span_name,
             operation_type,
             COALESCE(model_name, 'unknown') AS model_name,
             COALESCE(model_provider, 'unknown') AS model_provider,
             COALESCE(SUM(cost_usd), 0) AS cost,
             COALESCE(SUM(input_tokens), 0) AS input_tokens,
             COALESCE(SUM(billable_input_tokens), 0) AS billable_input_tokens,
             COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
             COALESCE(SUM(output_tokens), 0) AS output_tokens,
             COALESCE(SUM(total_tokens), 0) AS total_tokens,
             COUNT(*) AS calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
             COALESCE(AVG(duration_ms) FILTER (WHERE duration_ms > 0), 0) AS avg_duration_ms,
             MAX(created_at) AS latest_created_at
           FROM llm_usage
           {where_sql}
           GROUP BY span_name, operation_type, model_name, model_provider
           ORDER BY cost DESC, calls DESC
           LIMIT ${len(args)}""",
        *args,
    )
    return {
        "period_days": days,
        "operations": [
            {
                "span_name": r["span_name"],
                "operation_type": r["operation_type"],
                "model": r["model_name"],
                "provider": r["model_provider"],
                "cost_usd": float(r["cost"]),
                "input_tokens": int(r["input_tokens"]),
                "billable_input_tokens": int(r["billable_input_tokens"]),
                "cached_tokens": int(r["cached_tokens"]),
                "cache_write_tokens": int(r["cache_write_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "total_tokens": int(r["total_tokens"]),
                "calls": int(r["calls"]),
                "cache_hit_calls": int(r["cache_hit_calls"]),
                "cache_write_calls": int(r["cache_write_calls"]),
                "avg_duration_ms": round(float(r["avg_duration_ms"]), 1),
                "latest_created_at": r["latest_created_at"].isoformat() if r["latest_created_at"] else None,
            }
            for r in rows
        ],
    }


@router.get("/by-vendor")
async def cost_by_vendor(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Cost breakdown by B2B vendor name (from trace metadata)."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = await pool.fetch(
        """SELECT
             vendor_name,
             COALESCE(SUM(cost_usd), 0)                AS cost,
             COALESCE(SUM(input_tokens), 0)             AS input_tokens,
             COALESCE(SUM(billable_input_tokens), 0)    AS billable_input_tokens,
             COALESCE(SUM(cached_tokens), 0)            AS cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0)       AS cache_write_tokens,
             COALESCE(SUM(output_tokens), 0)            AS output_tokens,
             COALESCE(SUM(total_tokens), 0)             AS total_tokens,
             COUNT(*)                                   AS calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0)  AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
             COALESCE(AVG(duration_ms) FILTER (WHERE duration_ms > 0), 0) AS avg_duration_ms
           FROM llm_usage
           WHERE created_at >= $1
             AND vendor_name IS NOT NULL
           GROUP BY vendor_name
           ORDER BY cost DESC
           LIMIT $2""",
        since,
        limit,
    )
    return {
        "period_days": days,
        "vendors": [
            {
                "vendor_name": r["vendor_name"],
                "cost_usd": float(r["cost"]),
                "input_tokens": int(r["input_tokens"]),
                "billable_input_tokens": int(r["billable_input_tokens"]),
                "cached_tokens": int(r["cached_tokens"]),
                "cache_write_tokens": int(r["cache_write_tokens"]),
                "output_tokens": int(r["output_tokens"]),
                "total_tokens": int(r["total_tokens"]),
                "calls": int(r["calls"]),
                "cache_hit_calls": int(r["cache_hit_calls"]),
                "cache_write_calls": int(r["cache_write_calls"]),
                "avg_duration_ms": round(float(r["avg_duration_ms"]), 1),
            }
            for r in rows
        ],
    }


@router.get("/b2b-efficiency")
async def b2b_efficiency(
    days: int = Query(default=30, ge=1, le=365),
    top_n: int = Query(default=25, ge=1, le=100),
    run_limit: int = Query(default=25, ge=1, le=100),
):
    """B2B-specific efficiency rollups for extraction, repair, and reasoning."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    usage_rows = await _safe_fetch(
        pool,
        """
        SELECT
          span_name,
          cost_usd,
          vendor_name,
          run_id,
          metadata,
          created_at
        FROM llm_usage
        WHERE created_at >= $1
          AND (
            vendor_name IS NOT NULL
            OR run_id IS NOT NULL
            OR span_name LIKE 'reasoning.%'
            OR span_name LIKE 'task.b2b_enrichment%'
            OR span_name = 'pipeline.digest/b2b_churn_repair_extraction'
            OR metadata::text ILIKE '%vendor_name%'
            OR metadata::text ILIKE '%source%'
          )
        """,
        since,
    )

    vendor_rollups: dict[str, dict[str, Any]] = {}
    source_rollups: dict[str, dict[str, Any]] = {}
    run_usage: dict[str, dict[str, Any]] = {}

    for row in usage_rows:
        metadata = _normalize_metadata(row.get("metadata"))
        pass_name = _classify_b2b_pass(str(row.get("span_name") or ""), metadata)
        if pass_name is None:
            continue
        cost_usd = _safe_float(row.get("cost_usd"))
        vendor_name = _b2b_vendor_name(row.get("vendor_name"), metadata)
        source_name = _b2b_source_name(metadata).lower()
        run_id = str(row.get("run_id") or "").strip()

        if vendor_name:
            vendor_bucket = vendor_rollups.setdefault(
                vendor_name,
                {
                    "vendor_name": vendor_name,
                    "extraction_cost_usd": 0.0,
                    "repair_cost_usd": 0.0,
                    "reasoning_cost_usd": 0.0,
                    "extraction_calls": 0,
                    "repair_calls": 0,
                    "reasoning_calls": 0,
                    "total_cost_usd": 0.0,
                },
            )
            vendor_bucket[f"{pass_name}_cost_usd"] += cost_usd
            vendor_bucket[f"{pass_name}_calls"] += 1
            vendor_bucket["total_cost_usd"] += cost_usd

        if pass_name in {"extraction", "repair"} and source_name:
            source_bucket = source_rollups.setdefault(
                source_name,
                {
                    "source": source_name,
                    "extraction_cost_usd": 0.0,
                    "repair_cost_usd": 0.0,
                    "extraction_calls": 0,
                    "repair_calls": 0,
                    "total_cost_usd": 0.0,
                    "enriched_rows": 0,
                    "repair_triggered_rows": 0,
                    "repair_promoted_rows": 0,
                    "rows_with_spans": 0,
                    "span_count": 0,
                    "witness_yield_rate": 0.0,
                    "repair_trigger_rate": 0.0,
                    "repair_promoted_rate": 0.0,
                    "cost_per_witness_usd": None,
                },
            )
            source_bucket[f"{pass_name}_cost_usd"] += cost_usd
            source_bucket[f"{pass_name}_calls"] += 1
            source_bucket["total_cost_usd"] += cost_usd

        if run_id:
            run_bucket = run_usage.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "total_cost_usd": 0.0,
                    "calls": 0,
                    "extraction_cost_usd": 0.0,
                    "repair_cost_usd": 0.0,
                    "reasoning_cost_usd": 0.0,
                },
            )
            run_bucket["total_cost_usd"] += cost_usd
            run_bucket["calls"] += 1
            run_bucket[f"{pass_name}_cost_usd"] += cost_usd

    source_quality_rows = await _safe_fetch(
        pool,
        """
        SELECT
          source,
          COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched_rows,
          COUNT(*) FILTER (WHERE enrichment_repair_attempts > 0) AS repair_triggered_rows,
          COUNT(*) FILTER (WHERE enrichment_repair_status = 'promoted') AS repair_promoted_rows,
          COUNT(*) FILTER (
            WHERE enrichment_status = 'enriched'
              AND enrichment->'evidence_spans' IS NOT NULL
              AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
              AND jsonb_array_length(enrichment->'evidence_spans') > 0
          ) AS rows_with_spans,
          COALESCE(
            SUM(
              CASE
                WHEN enrichment_status = 'enriched'
                  AND enrichment->'evidence_spans' IS NOT NULL
                  AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
                THEN jsonb_array_length(enrichment->'evidence_spans')
                ELSE 0
              END
            ),
            0
          ) AS span_count
        FROM b2b_reviews
        WHERE COALESCE(enriched_at, imported_at) >= $1
        GROUP BY source
        HAVING COUNT(*) FILTER (WHERE enrichment_status = 'enriched') > 0
        """,
        since,
    )
    for row in source_quality_rows:
        source_name = str(row.get("source") or "").strip().lower()
        bucket = source_rollups.setdefault(
            source_name,
            {
                "source": source_name,
                "extraction_cost_usd": 0.0,
                "repair_cost_usd": 0.0,
                "extraction_calls": 0,
                "repair_calls": 0,
                "total_cost_usd": 0.0,
                "enriched_rows": 0,
                "repair_triggered_rows": 0,
                "repair_promoted_rows": 0,
                "rows_with_spans": 0,
                "span_count": 0,
                "witness_yield_rate": 0.0,
                "repair_trigger_rate": 0.0,
                "repair_promoted_rate": 0.0,
                "cost_per_witness_usd": None,
            },
        )
        enriched_rows = _safe_int(row.get("enriched_rows"))
        span_count = _safe_int(row.get("span_count"))
        repair_triggered_rows = _safe_int(row.get("repair_triggered_rows"))
        repair_promoted_rows = _safe_int(row.get("repair_promoted_rows"))
        bucket["enriched_rows"] = enriched_rows
        bucket["repair_triggered_rows"] = repair_triggered_rows
        bucket["repair_promoted_rows"] = repair_promoted_rows
        bucket["rows_with_spans"] = _safe_int(row.get("rows_with_spans"))
        bucket["span_count"] = span_count
        bucket["witness_yield_rate"] = (
            round(span_count / enriched_rows, 4) if enriched_rows > 0 else 0.0
        )
        bucket["repair_trigger_rate"] = (
            round(repair_triggered_rows / enriched_rows, 4)
            if enriched_rows > 0
            else 0.0
        )
        bucket["repair_promoted_rate"] = (
            round(repair_promoted_rows / enriched_rows, 4)
            if enriched_rows > 0
            else 0.0
        )
        bucket["cost_per_witness_usd"] = (
            round(float(bucket["total_cost_usd"]) / span_count, 6)
            if span_count > 0 and float(bucket["total_cost_usd"]) > 0
            else None
        )

    run_rows = await _safe_fetch(
        pool,
        """
        SELECT
          e.id AS execution_id,
          t.name AS task_name,
          e.started_at,
          e.result_text
        FROM task_executions e
        JOIN scheduled_tasks t ON t.id = e.task_id
        WHERE e.started_at >= $1
          AND e.status = 'completed'
          AND t.name = ANY($2::text[])
        ORDER BY e.started_at DESC
        LIMIT $3
        """,
        since,
        ["b2b_enrichment", "b2b_enrichment_repair", "b2b_reasoning_synthesis"],
        run_limit,
    )

    recent_runs = []
    summary_cost = 0.0
    summary_witness_count = 0
    summary_measured_runs = 0
    for row in run_rows:
        run_id = str(row.get("execution_id"))
        payload = _parse_task_result_payload(row.get("result_text"))
        task_name = str(row.get("task_name") or "")
        reviews_processed = _safe_int(payload.get("reviews_processed"))
        if reviews_processed == 0:
            if task_name == "b2b_enrichment":
                reviews_processed = sum(
                    _safe_int(payload.get(key))
                    for key in ("enriched", "quarantined", "failed", "no_signal")
                )
            elif task_name == "b2b_enrichment_repair":
                reviews_processed = sum(
                    _safe_int(payload.get(key))
                    for key in ("promoted", "shadowed", "failed")
                )
            elif task_name == "b2b_reasoning_synthesis":
                reviews_processed = _safe_int(payload.get("vendors_reasoned"))
        witness_count = _safe_int(payload.get("witness_count"))
        total_cost_usd = _safe_float(run_usage.get(run_id, {}).get("total_cost_usd"))
        calls = _safe_int(run_usage.get(run_id, {}).get("calls"))
        if witness_count > 0:
            summary_cost += total_cost_usd
            summary_witness_count += witness_count
            summary_measured_runs += 1
        recent_runs.append(
            {
                "run_id": run_id,
                "task_name": task_name,
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "total_cost_usd": round(total_cost_usd, 6),
                "calls": calls,
                "reviews_processed": reviews_processed,
                "witness_rows": _safe_int(payload.get("witness_rows")),
                "witness_count": witness_count,
                "witness_yield_rate": (
                    round(witness_count / reviews_processed, 4)
                    if reviews_processed > 0
                    else 0.0
                ),
                "cost_per_witness_usd": (
                    round(total_cost_usd / witness_count, 6)
                    if witness_count > 0 and total_cost_usd > 0
                    else None
                ),
                "secondary_write_hits": _safe_int(payload.get("secondary_write_hits")),
                "exact_cache_hits": _safe_int(payload.get("exact_cache_hits")),
                "generated": _safe_int(payload.get("generated")),
                "extraction_cost_usd": round(_safe_float(run_usage.get(run_id, {}).get("extraction_cost_usd")), 6),
                "repair_cost_usd": round(_safe_float(run_usage.get(run_id, {}).get("repair_cost_usd")), 6),
                "reasoning_cost_usd": round(_safe_float(run_usage.get(run_id, {}).get("reasoning_cost_usd")), 6),
            }
        )

    vendor_passes = sorted(
        vendor_rollups.values(),
        key=lambda item: (-float(item["total_cost_usd"]), item["vendor_name"]),
    )[:top_n]
    source_efficiency = sorted(
        source_rollups.values(),
        key=lambda item: (-float(item["total_cost_usd"]), -int(item["enriched_rows"]), item["source"]),
    )[:top_n]

    return {
        "period_days": days,
        "top_n": top_n,
        "run_limit": run_limit,
        "summary": {
            "measured_runs": summary_measured_runs,
            "tracked_cost_usd": round(summary_cost, 6),
            "tracked_witness_count": summary_witness_count,
            "cost_per_witness_usd": (
                round(summary_cost / summary_witness_count, 6)
                if summary_witness_count > 0 and summary_cost > 0
                else None
            ),
        },
        "vendor_passes": vendor_passes,
        "source_efficiency": source_efficiency,
        "recent_runs": recent_runs,
    }


@router.get("/reasoning-activity")
async def reasoning_activity(days: int = Query(default=30, ge=1, le=365)):
    """Per-pass breakdown of legacy stratified reasoning activity."""
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
async def recent_calls(
    limit: int = Query(default=50, ge=1, le=500),
    days: int | None = Query(default=None, ge=1, le=365),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    span_name: str | None = Query(default=None),
    operation_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    cache_only: bool | None = Query(default=None),
):
    """Most recent LLM calls for the activity feed."""
    pool = _pool_or_503()
    clauses, args = _build_recent_filters(
        days=days,
        provider=provider,
        model=model,
        span_name=span_name,
        operation_type=operation_type,
        status=status,
        cache_only=cache_only,
    )
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    rows = await pool.fetch(
        f"""SELECT id, span_name, operation_type, model_name, model_provider,
                  input_tokens, billable_input_tokens, cached_tokens, cache_write_tokens,
                  output_tokens, total_tokens, cost_usd, duration_ms, ttft_ms,
                  inference_time_ms, queue_time_ms, tokens_per_second, status,
                  api_endpoint, provider_request_id, metadata, created_at
           FROM llm_usage
           {where_sql}
           ORDER BY created_at DESC
           LIMIT ${len(args)}""",
        *args,
    )
    calls = []
    for r in rows:
        metadata = _normalize_metadata(r["metadata"])
        title, detail = _describe_recent_call(r["span_name"], metadata)
        calls.append({
            "id": str(r["id"]),
            "span_name": r["span_name"],
            "operation_type": r["operation_type"],
            "title": title,
            "detail": detail,
            "model": r["model_name"],
            "provider": r["model_provider"],
            "input_tokens": r["input_tokens"],
            "billable_input_tokens": r["billable_input_tokens"],
            "cached_tokens": r["cached_tokens"],
            "cache_write_tokens": r["cache_write_tokens"],
            "output_tokens": r["output_tokens"],
            "total_tokens": r["total_tokens"],
            "cost_usd": float(r["cost_usd"]) if r["cost_usd"] else 0,
            "duration_ms": r["duration_ms"],
            "ttft_ms": r["ttft_ms"],
            "inference_time_ms": r["inference_time_ms"],
            "queue_time_ms": r["queue_time_ms"],
            "tokens_per_second": r["tokens_per_second"],
            "status": r["status"],
            "cache_hit": bool(r["cached_tokens"]),
            "cache_write": bool(r["cache_write_tokens"]),
            "api_endpoint": r["api_endpoint"],
            "provider_request_id": r["provider_request_id"],
            "metadata": metadata,
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        })
    return {
        "calls": calls,
    }


@router.get("/cache-health")
async def cache_health(
    days: int = Query(default=30, ge=1, le=365),
    top_n: int = Query(default=8, ge=1, le=25),
):
    """Atlas cache health across exact, provider, semantic, and reuse layers."""
    pool = _pool_or_503()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    exact_strategies = tuple(
        strategy for strategy in iter_core_b2b_cache_strategies() if strategy.mode == "exact"
    )

    exact_summary_row = await _safe_fetchrow(
        pool,
        """SELECT
             COUNT(*) AS total_rows,
             COALESCE(SUM(hit_count), 0) AS total_hits,
             COUNT(*) FILTER (WHERE created_at >= $1) AS writes_in_window,
             COUNT(*) FILTER (WHERE last_hit_at >= $1 AND hit_count > 0) AS rows_hit_in_window
           FROM b2b_llm_exact_cache""",
        since,
    ) or {}

    exact_stage_rows = await _safe_fetch(
        pool,
        """SELECT
             namespace,
             COUNT(*) AS rows,
             COALESCE(SUM(hit_count), 0) AS total_hits,
             COUNT(*) FILTER (WHERE created_at >= $1) AS writes_in_window,
             COUNT(*) FILTER (WHERE last_hit_at >= $1 AND hit_count > 0) AS rows_hit_in_window,
             MAX(created_at) AS last_write_at,
             MAX(last_hit_at) AS last_hit_at,
             COUNT(DISTINCT provider) AS provider_count,
             COUNT(DISTINCT model) AS model_count
           FROM b2b_llm_exact_cache
           GROUP BY namespace""",
        since,
    )
    exact_stage_map = {str(row["namespace"]): row for row in exact_stage_rows}

    prompt_cache_row = await _safe_fetchrow(
        pool,
        """SELECT
             COUNT(*) AS total_calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
             COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
             COALESCE(SUM(billable_input_tokens), 0) AS billable_input_tokens
           FROM llm_usage
           WHERE created_at >= $1""",
        since,
    ) or {}
    prompt_cache_spans = await _safe_fetch(
        pool,
        """SELECT
             span_name,
             COUNT(*) AS calls,
             COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
             COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
             COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
             COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens
           FROM llm_usage
           WHERE created_at >= $1
             AND (cached_tokens > 0 OR cache_write_tokens > 0)
           GROUP BY span_name
           ORDER BY (COALESCE(SUM(cached_tokens), 0) + COALESCE(SUM(cache_write_tokens), 0)) DESC,
                    COUNT(*) DESC
           LIMIT $2""",
        since,
        top_n,
    )

    batch_summary_row = await _safe_fetchrow(
        pool,
        """SELECT
             COUNT(*) AS total_jobs,
             COUNT(*) FILTER (WHERE provider_batch_id IS NOT NULL) AS submitted_jobs,
             COALESCE(SUM(total_items), 0) AS total_items,
             COALESCE(SUM(submitted_items), 0) AS submitted_items,
             COALESCE(SUM(cache_prefiltered_items), 0) AS cache_prefiltered_items,
             COALESCE(SUM(fallback_single_call_items), 0) AS fallback_single_call_items,
             COALESCE(SUM(completed_items), 0) AS completed_items,
             COALESCE(SUM(failed_items), 0) AS failed_items,
             COALESCE(SUM(estimated_sequential_cost_usd), 0) AS estimated_sequential_cost_usd,
             COALESCE(SUM(estimated_batch_cost_usd), 0) AS estimated_batch_cost_usd
           FROM anthropic_message_batches
           WHERE created_at >= $1""",
        since,
    ) or {}
    batch_stage_rows = await _safe_fetch(
        pool,
        """SELECT
             stage_id,
             task_name,
             COUNT(*) AS total_jobs,
             COUNT(*) FILTER (WHERE provider_batch_id IS NOT NULL) AS submitted_jobs,
             COALESCE(SUM(total_items), 0) AS total_items,
             COALESCE(SUM(submitted_items), 0) AS submitted_items,
             COALESCE(SUM(cache_prefiltered_items), 0) AS cache_prefiltered_items,
             COALESCE(SUM(fallback_single_call_items), 0) AS fallback_single_call_items,
             COALESCE(SUM(completed_items), 0) AS completed_items,
             COALESCE(SUM(failed_items), 0) AS failed_items,
             COALESCE(SUM(estimated_sequential_cost_usd), 0) AS estimated_sequential_cost_usd,
             COALESCE(SUM(estimated_batch_cost_usd), 0) AS estimated_batch_cost_usd,
             MAX(submitted_at) AS last_submitted_at,
             MAX(completed_at) AS last_completed_at
           FROM anthropic_message_batches
           WHERE created_at >= $1
           GROUP BY stage_id, task_name
           ORDER BY COALESCE(SUM(submitted_items), 0) DESC, stage_id
           LIMIT $2""",
        since,
        top_n,
    )

    semantic_summary_row = await _safe_fetchrow(
        pool,
        """SELECT
             COUNT(*) FILTER (WHERE invalidated_at IS NULL) AS active_entries,
             COUNT(*) FILTER (WHERE invalidated_at IS NOT NULL) AS invalidated_entries,
             COUNT(*) FILTER (
               WHERE invalidated_at IS NULL AND last_validated_at >= $1
             ) AS recent_validations
           FROM reasoning_semantic_cache""",
        since,
    ) or {}
    semantic_class_rows = await _safe_fetch(
        pool,
        """SELECT
             pattern_class,
             COUNT(*) FILTER (WHERE invalidated_at IS NULL) AS active_entries,
             COUNT(*) FILTER (
               WHERE invalidated_at IS NULL AND last_validated_at >= $1
             ) AS recent_validations
           FROM reasoning_semantic_cache
           GROUP BY pattern_class
           ORDER BY COUNT(*) FILTER (WHERE invalidated_at IS NULL) DESC, pattern_class
           LIMIT $2""",
        since,
        top_n,
    )

    vendor_packet_row = await _safe_fetchrow(
        pool,
        """SELECT
             COUNT(*) AS total_rows,
             COUNT(*) FILTER (WHERE created_at >= $1) AS writes_in_window,
             COUNT(DISTINCT vendor_name) AS unique_vendors,
             COUNT(DISTINCT evidence_hash) AS unique_hashes
           FROM b2b_vendor_reasoning_packets""",
        since,
    ) or {}
    cross_vendor_row = await _safe_fetchrow(
        pool,
        """SELECT
             COUNT(*) AS total_rows,
             COUNT(*) FILTER (WHERE cached IS TRUE) AS cached_rows,
             COUNT(*) FILTER (WHERE cached IS TRUE AND created_at >= $1) AS cached_rows_in_window
           FROM b2b_cross_vendor_conclusions""",
        since,
    ) or {}

    task_rows = await _safe_fetch(
        pool,
        """SELECT
             t.name,
             e.result_text,
             e.metadata
           FROM task_executions e
           JOIN scheduled_tasks t ON t.id = e.task_id
           WHERE e.started_at >= $1
             AND e.status = 'completed'
             AND t.name = ANY($2::text[])
           ORDER BY e.started_at DESC""",
        since,
        [
            "b2b_battle_cards",
            "b2b_churn_reports",
            "b2b_reasoning_synthesis",
            "b2b_enrichment",
            "b2b_enrichment_repair",
        ],
    )
    task_rollups: dict[str, dict[str, int | str]] = {
        "b2b_battle_cards": {
            "task_name": "b2b_battle_cards",
            "executions": 0,
            "reused": 0,
            "exact_cache_hits": 0,
            "semantic_cache_hits": 0,
            "evidence_hash_reuse": 0,
            "generated": 0,
        },
        "b2b_churn_reports": {
            "task_name": "b2b_churn_reports",
            "executions": 0,
            "reused": 0,
            "exact_cache_hits": 0,
            "semantic_cache_hits": 0,
            "evidence_hash_reuse": 0,
            "generated": 0,
        },
        "b2b_reasoning_synthesis": {
            "task_name": "b2b_reasoning_synthesis",
            "executions": 0,
            "reused": 0,
            "exact_cache_hits": 0,
            "semantic_cache_hits": 0,
            "evidence_hash_reuse": 0,
            "generated": 0,
        },
        "b2b_enrichment": {
            "task_name": "b2b_enrichment",
            "executions": 0,
            "reused": 0,
            "exact_cache_hits": 0,
            "semantic_cache_hits": 0,
            "evidence_hash_reuse": 0,
            "generated": 0,
        },
        "b2b_enrichment_repair": {
            "task_name": "b2b_enrichment_repair",
            "executions": 0,
            "reused": 0,
            "exact_cache_hits": 0,
            "semantic_cache_hits": 0,
            "evidence_hash_reuse": 0,
            "generated": 0,
        },
    }
    for row in task_rows:
        task_name = str(row["name"])
        bucket = task_rollups.get(task_name)
        if bucket is None:
            continue
        bucket["executions"] = _safe_int(bucket["executions"]) + 1
        metadata = _normalize_metadata(row["metadata"])
        payload = _parse_task_result_payload(row["result_text"])
        if task_name == "b2b_battle_cards":
            bucket["semantic_cache_hits"] = _safe_int(bucket["semantic_cache_hits"]) + _safe_int(metadata.get("cache_hits"))
            bucket["generated"] = _safe_int(bucket["generated"]) + _safe_int(metadata.get("cards_llm_updated"))
        elif task_name == "b2b_churn_reports":
            bucket["exact_cache_hits"] = _safe_int(bucket["exact_cache_hits"]) + _safe_int(payload.get("scorecard_cache_hits"))
            bucket["evidence_hash_reuse"] = _safe_int(bucket["evidence_hash_reuse"]) + _safe_int(payload.get("scorecard_reasoning_reused"))
            bucket["generated"] = _safe_int(bucket["generated"]) + _safe_int(payload.get("scorecard_llm_generated"))
        elif task_name == "b2b_reasoning_synthesis":
            bucket["evidence_hash_reuse"] = _safe_int(bucket["evidence_hash_reuse"]) + _safe_int(payload.get("vendors_skipped"))
            bucket["generated"] = _safe_int(bucket["generated"]) + _safe_int(payload.get("vendors_reasoned")) + _safe_int(payload.get("cross_vendor_succeeded"))
        elif task_name == "b2b_enrichment":
            bucket["exact_cache_hits"] = _safe_int(bucket["exact_cache_hits"]) + _safe_int(payload.get("exact_cache_hits"))
            bucket["generated"] = _safe_int(bucket["generated"]) + _safe_int(payload.get("generated"))
        elif task_name == "b2b_enrichment_repair":
            bucket["exact_cache_hits"] = _safe_int(bucket["exact_cache_hits"]) + _safe_int(payload.get("exact_cache_hits"))
            bucket["generated"] = _safe_int(bucket["generated"]) + _safe_int(payload.get("generated"))
        bucket["reused"] = (
            _safe_int(bucket["exact_cache_hits"])
            + _safe_int(bucket["semantic_cache_hits"])
            + _safe_int(bucket["evidence_hash_reuse"])
        )

    exact_stage_payload = []
    for strategy in exact_strategies:
        row = exact_stage_map.get(str(strategy.namespace or ""))
        exact_stage_payload.append({
            "stage_id": strategy.stage_id,
            "namespace": strategy.namespace,
            "file_path": strategy.file_path,
            "rationale": strategy.rationale,
            "rows": _safe_int(row["rows"]) if row else 0,
            "total_hits": _safe_int(row["total_hits"]) if row else 0,
            "writes_in_window": _safe_int(row["writes_in_window"]) if row else 0,
            "rows_hit_in_window": _safe_int(row["rows_hit_in_window"]) if row else 0,
            "provider_count": _safe_int(row["provider_count"]) if row else 0,
            "model_count": _safe_int(row["model_count"]) if row else 0,
            "last_write_at": row["last_write_at"].isoformat() if row and row["last_write_at"] else None,
            "last_hit_at": row["last_hit_at"].isoformat() if row and row["last_hit_at"] else None,
        })

    return {
        "period_days": days,
        "top_n": top_n,
        "exact_cache": {
            "enabled": bool(settings.b2b_churn.llm_exact_cache_enabled),
            "total_rows": _safe_int(exact_summary_row.get("total_rows")),
            "total_hits": _safe_int(exact_summary_row.get("total_hits")),
            "writes_in_window": _safe_int(exact_summary_row.get("writes_in_window")),
            "rows_hit_in_window": _safe_int(exact_summary_row.get("rows_hit_in_window")),
            "stages": exact_stage_payload,
        },
        "provider_prompt_cache": {
            "total_calls": _safe_int(prompt_cache_row.get("total_calls")),
            "cache_hit_calls": _safe_int(prompt_cache_row.get("cache_hit_calls")),
            "cache_write_calls": _safe_int(prompt_cache_row.get("cache_write_calls")),
            "cached_tokens": _safe_int(prompt_cache_row.get("cached_tokens")),
            "cache_write_tokens": _safe_int(prompt_cache_row.get("cache_write_tokens")),
            "billable_input_tokens": _safe_int(prompt_cache_row.get("billable_input_tokens")),
            "top_spans": [
                {
                    "span_name": row["span_name"],
                    "calls": _safe_int(row["calls"]),
                    "cache_hit_calls": _safe_int(row["cache_hit_calls"]),
                    "cache_write_calls": _safe_int(row["cache_write_calls"]),
                    "cached_tokens": _safe_int(row["cached_tokens"]),
                    "cache_write_tokens": _safe_int(row["cache_write_tokens"]),
                }
                for row in prompt_cache_spans
            ],
        },
        "anthropic_batching": {
            "enabled": bool(settings.b2b_churn.anthropic_batch_enabled),
            "total_jobs": _safe_int(batch_summary_row.get("total_jobs")),
            "submitted_jobs": _safe_int(batch_summary_row.get("submitted_jobs")),
            "total_items": _safe_int(batch_summary_row.get("total_items")),
            "submitted_items": _safe_int(batch_summary_row.get("submitted_items")),
            "cache_prefiltered_items": _safe_int(batch_summary_row.get("cache_prefiltered_items")),
            "fallback_single_call_items": _safe_int(batch_summary_row.get("fallback_single_call_items")),
            "completed_items": _safe_int(batch_summary_row.get("completed_items")),
            "failed_items": _safe_int(batch_summary_row.get("failed_items")),
            "estimated_sequential_cost_usd": round(
                _safe_float(batch_summary_row.get("estimated_sequential_cost_usd")),
                6,
            ),
            "estimated_batch_cost_usd": round(
                _safe_float(batch_summary_row.get("estimated_batch_cost_usd")),
                6,
            ),
            "estimated_savings_usd": round(
                max(
                    0.0,
                    _safe_float(batch_summary_row.get("estimated_sequential_cost_usd"))
                    - _safe_float(batch_summary_row.get("estimated_batch_cost_usd")),
                ),
                6,
            ),
            "stages": [
                {
                    "stage_id": str(row["stage_id"]),
                    "task_name": str(row["task_name"]),
                    "total_jobs": _safe_int(row["total_jobs"]),
                    "submitted_jobs": _safe_int(row["submitted_jobs"]),
                    "total_items": _safe_int(row["total_items"]),
                    "submitted_items": _safe_int(row["submitted_items"]),
                    "cache_prefiltered_items": _safe_int(row["cache_prefiltered_items"]),
                    "fallback_single_call_items": _safe_int(row["fallback_single_call_items"]),
                    "completed_items": _safe_int(row["completed_items"]),
                    "failed_items": _safe_int(row["failed_items"]),
                    "estimated_sequential_cost_usd": round(
                        _safe_float(row["estimated_sequential_cost_usd"]),
                        6,
                    ),
                    "estimated_batch_cost_usd": round(
                        _safe_float(row["estimated_batch_cost_usd"]),
                        6,
                    ),
                    "estimated_savings_usd": round(
                        max(
                            0.0,
                            _safe_float(row["estimated_sequential_cost_usd"])
                            - _safe_float(row["estimated_batch_cost_usd"]),
                        ),
                        6,
                    ),
                    "last_submitted_at": row["last_submitted_at"].isoformat() if row["last_submitted_at"] else None,
                    "last_completed_at": row["last_completed_at"].isoformat() if row["last_completed_at"] else None,
                }
                for row in batch_stage_rows
            ],
        },
        "semantic_cache": {
            "active_entries": _safe_int(semantic_summary_row.get("active_entries")),
            "invalidated_entries": _safe_int(semantic_summary_row.get("invalidated_entries")),
            "recent_validations": _safe_int(semantic_summary_row.get("recent_validations")),
            "pattern_classes": [
                {
                    "pattern_class": row["pattern_class"],
                    "active_entries": _safe_int(row["active_entries"]),
                    "recent_validations": _safe_int(row["recent_validations"]),
                }
                for row in semantic_class_rows
            ],
        },
        "evidence_hash_reuse": {
            "vendor_packet_rows": _safe_int(vendor_packet_row.get("total_rows")),
            "vendor_packet_writes_in_window": _safe_int(vendor_packet_row.get("writes_in_window")),
            "unique_vendors": _safe_int(vendor_packet_row.get("unique_vendors")),
            "unique_hashes": _safe_int(vendor_packet_row.get("unique_hashes")),
            "cross_vendor_rows": _safe_int(cross_vendor_row.get("total_rows")),
            "cross_vendor_cached_rows": _safe_int(cross_vendor_row.get("cached_rows")),
            "cross_vendor_cached_rows_in_window": _safe_int(cross_vendor_row.get("cached_rows_in_window")),
        },
        "task_reuse": {
            "tasks": list(task_rollups.values()),
        },
    }


@router.get("/runs/{run_id}")
async def cost_run_detail(
    run_id: str,
    call_limit: int = Query(default=25, ge=1, le=100),
    event_limit: int = Query(default=25, ge=1, le=100),
    attempt_limit: int = Query(default=25, ge=1, le=100),
    batch_item_limit: int = Query(default=100, ge=1, le=500),
):
    """Correlate one execution across task, llm, artifact, and visibility tables."""
    pool = _pool_or_503()
    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id:
        raise HTTPException(status_code=400, detail="run_id is required")

    execution_row = None
    try:
        import uuid as _uuid

        execution_row = await _safe_fetchrow(
            pool,
            """
            SELECT
                e.id,
                e.task_id,
                t.name AS task_name,
                e.status,
                e.started_at,
                e.completed_at,
                e.duration_ms,
                e.retry_count,
                e.result_text,
                e.error,
                e.metadata
            FROM task_executions e
            JOIN scheduled_tasks t ON t.id = e.task_id
            WHERE e.id = $1
            """,
            _uuid.UUID(normalized_run_id),
        )
    except (TypeError, ValueError):
        execution_row = None

    llm_summary_row = await _safe_fetchrow(
        pool,
        """
        SELECT
            COUNT(*) AS total_calls,
            COALESCE(SUM(cost_usd), 0) AS total_cost,
            COALESCE(SUM(input_tokens), 0) AS total_input,
            COALESCE(SUM(billable_input_tokens), 0) AS total_billable_input,
            COALESCE(SUM(cached_tokens), 0) AS total_cached_tokens,
            COALESCE(SUM(cache_write_tokens), 0) AS total_cache_write_tokens,
            COALESCE(SUM(output_tokens), 0) AS total_output,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
            COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
            MIN(created_at) AS first_call_at,
            MAX(created_at) AS last_call_at
        FROM llm_usage
        WHERE run_id = $1
        """,
        normalized_run_id,
    ) or {}
    operation_rows = await _safe_fetch(
        pool,
        """
        SELECT
            span_name,
            operation_type,
            model_name,
            model_provider,
            COALESCE(SUM(cost_usd), 0) AS cost,
            COALESCE(SUM(input_tokens), 0) AS input_tokens,
            COALESCE(SUM(billable_input_tokens), 0) AS billable_input_tokens,
            COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
            COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
            COALESCE(SUM(output_tokens), 0) AS output_tokens,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COUNT(*) AS calls,
            COUNT(*) FILTER (WHERE cached_tokens > 0) AS cache_hit_calls,
            COUNT(*) FILTER (WHERE cache_write_tokens > 0) AS cache_write_calls,
            COALESCE(AVG(duration_ms), 0) AS avg_duration_ms,
            MAX(created_at) AS latest_created_at
        FROM llm_usage
        WHERE run_id = $1
        GROUP BY span_name, operation_type, model_name, model_provider
        ORDER BY COALESCE(SUM(cost_usd), 0) DESC, COUNT(*) DESC
        """,
        normalized_run_id,
    )
    llm_call_rows = await _safe_fetch(
        pool,
        """
        SELECT
            id, span_name, operation_type, model_name, model_provider,
            input_tokens, billable_input_tokens, cached_tokens, cache_write_tokens,
            output_tokens, total_tokens, cost_usd, duration_ms, ttft_ms,
            inference_time_ms, queue_time_ms, tokens_per_second, status,
            api_endpoint, provider_request_id, metadata, created_at
        FROM llm_usage
        WHERE run_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        normalized_run_id,
        call_limit,
    )
    batch_summary_row = await _safe_fetchrow(
        pool,
        """
        SELECT
            COUNT(*) AS total_jobs,
            COUNT(*) FILTER (WHERE provider_batch_id IS NOT NULL) AS submitted_jobs,
            COALESCE(SUM(submitted_items), 0) AS submitted_items,
            COALESCE(SUM(cache_prefiltered_items), 0) AS cache_prefiltered_items,
            COALESCE(SUM(fallback_single_call_items), 0) AS fallback_single_call_items,
            COALESCE(SUM(completed_items), 0) AS completed_items,
            COALESCE(SUM(failed_items), 0) AS failed_items,
            COALESCE(SUM(estimated_sequential_cost_usd), 0) AS estimated_sequential_cost_usd,
            COALESCE(SUM(estimated_batch_cost_usd), 0) AS estimated_batch_cost_usd
        FROM anthropic_message_batches
        WHERE run_id = $1
        """,
        normalized_run_id,
    ) or {}
    batch_rows = await _safe_fetch(
        pool,
        """
        SELECT
            id,
            stage_id,
            task_name,
            status,
            provider_batch_id,
            total_items,
            submitted_items,
            cache_prefiltered_items,
            fallback_single_call_items,
            completed_items,
            failed_items,
            estimated_sequential_cost_usd,
            estimated_batch_cost_usd,
            submitted_at,
            completed_at
        FROM anthropic_message_batches
        WHERE run_id = $1
        ORDER BY created_at DESC
        """,
        normalized_run_id,
    )
    batch_item_rows = await _safe_fetch(
        pool,
        """
        SELECT
            i.id,
            i.batch_id,
            i.custom_id,
            i.stage_id,
            b.task_name,
            b.provider_batch_id,
            i.artifact_type,
            i.artifact_id,
            i.vendor_name,
            i.status,
            i.cache_prefiltered,
            i.fallback_single_call,
            i.input_tokens,
            i.billable_input_tokens,
            i.cached_tokens,
            i.cache_write_tokens,
            i.output_tokens,
            (i.input_tokens + i.output_tokens) AS total_tokens,
            i.cost_usd,
            i.provider_request_id,
            i.error_text,
            i.request_metadata,
            i.created_at,
            i.completed_at
        FROM anthropic_message_batch_items i
        JOIN anthropic_message_batches b ON b.id = i.batch_id
        WHERE b.run_id = $1
        ORDER BY COALESCE(i.completed_at, i.created_at) DESC, i.created_at DESC
        LIMIT $2
        """,
        normalized_run_id,
        batch_item_limit,
    )
    attempt_rows = await _safe_fetch(
        pool,
        """
        SELECT
            id, artifact_type, artifact_id, run_id, attempt_no, stage, status,
            score, threshold, blocker_count, warning_count,
            blocking_issues, warnings, failure_step, error_message,
            started_at, completed_at
        FROM artifact_attempts
        WHERE run_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        normalized_run_id,
        attempt_limit,
    )
    event_rows = await _safe_fetch(
        pool,
        """
        SELECT
            id, occurred_at, run_id, stage, event_type, severity, actionable,
            entity_type, entity_id, artifact_type, reason_code, rule_code,
            decision, summary, detail, fingerprint
        FROM pipeline_visibility_events
        WHERE run_id = $1
        ORDER BY occurred_at DESC
        LIMIT $2
        """,
        normalized_run_id,
        event_limit,
    )

    if (
        execution_row is None
        and _safe_int(llm_summary_row.get("total_calls")) == 0
        and _safe_int(batch_summary_row.get("total_jobs")) == 0
        and not attempt_rows
        and not event_rows
    ):
        raise HTTPException(status_code=404, detail="Run not found")

    calls = []
    for row in llm_call_rows:
        metadata = _normalize_metadata(row["metadata"])
        title, detail = _describe_recent_call(row["span_name"], metadata)
        calls.append({
            "id": str(row["id"]),
            "span_name": row["span_name"],
            "operation_type": row["operation_type"],
            "title": title,
            "detail": detail,
            "model": row["model_name"],
            "provider": row["model_provider"],
            "input_tokens": row["input_tokens"],
            "billable_input_tokens": row["billable_input_tokens"],
            "cached_tokens": row["cached_tokens"],
            "cache_write_tokens": row["cache_write_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "cost_usd": float(row["cost_usd"]) if row["cost_usd"] else 0,
            "duration_ms": row["duration_ms"],
            "ttft_ms": row["ttft_ms"],
            "inference_time_ms": row["inference_time_ms"],
            "queue_time_ms": row["queue_time_ms"],
            "tokens_per_second": row["tokens_per_second"],
            "status": row["status"],
            "cache_hit": bool(row["cached_tokens"]),
            "cache_write": bool(row["cache_write_tokens"]),
            "api_endpoint": row["api_endpoint"],
            "provider_request_id": row["provider_request_id"],
            "metadata": metadata,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        })

    return {
        "run_id": normalized_run_id,
        "task_execution": {
            "id": str(execution_row["id"]),
            "task_id": str(execution_row["task_id"]),
            "task_name": execution_row["task_name"],
            "status": execution_row["status"],
            "started_at": execution_row["started_at"].isoformat() if execution_row["started_at"] else None,
            "completed_at": execution_row["completed_at"].isoformat() if execution_row["completed_at"] else None,
            "duration_ms": execution_row["duration_ms"],
            "retry_count": execution_row["retry_count"],
            "result": _parse_task_result_payload(execution_row["result_text"]),
            "result_text": execution_row["result_text"],
            "error": execution_row["error"],
            "metadata": _normalize_metadata(execution_row["metadata"]),
        } if execution_row else None,
        "llm_summary": {
            "total_calls": _safe_int(llm_summary_row.get("total_calls")),
            "total_cost_usd": float(llm_summary_row.get("total_cost") or 0),
            "total_input_tokens": _safe_int(llm_summary_row.get("total_input")),
            "total_billable_input_tokens": _safe_int(llm_summary_row.get("total_billable_input")),
            "total_cached_tokens": _safe_int(llm_summary_row.get("total_cached_tokens")),
            "total_cache_write_tokens": _safe_int(llm_summary_row.get("total_cache_write_tokens")),
            "total_output_tokens": _safe_int(llm_summary_row.get("total_output")),
            "total_tokens": _safe_int(llm_summary_row.get("total_tokens")),
            "cache_hit_calls": _safe_int(llm_summary_row.get("cache_hit_calls")),
            "cache_write_calls": _safe_int(llm_summary_row.get("cache_write_calls")),
            "first_call_at": llm_summary_row.get("first_call_at").isoformat() if llm_summary_row.get("first_call_at") else None,
            "last_call_at": llm_summary_row.get("last_call_at").isoformat() if llm_summary_row.get("last_call_at") else None,
        },
        "batching_summary": {
            "total_jobs": _safe_int(batch_summary_row.get("total_jobs")),
            "submitted_jobs": _safe_int(batch_summary_row.get("submitted_jobs")),
            "submitted_items": _safe_int(batch_summary_row.get("submitted_items")),
            "cache_prefiltered_items": _safe_int(batch_summary_row.get("cache_prefiltered_items")),
            "fallback_single_call_items": _safe_int(batch_summary_row.get("fallback_single_call_items")),
            "completed_items": _safe_int(batch_summary_row.get("completed_items")),
            "failed_items": _safe_int(batch_summary_row.get("failed_items")),
            "estimated_sequential_cost_usd": round(
                _safe_float(batch_summary_row.get("estimated_sequential_cost_usd")),
                6,
            ),
            "estimated_batch_cost_usd": round(
                _safe_float(batch_summary_row.get("estimated_batch_cost_usd")),
                6,
            ),
            "estimated_savings_usd": round(
                max(
                    0.0,
                    _safe_float(batch_summary_row.get("estimated_sequential_cost_usd"))
                    - _safe_float(batch_summary_row.get("estimated_batch_cost_usd")),
                ),
                6,
            ),
        },
        "operations": [
            {
                "span_name": row["span_name"],
                "operation_type": row["operation_type"],
                "model": row["model_name"],
                "provider": row["model_provider"],
                "cost_usd": float(row["cost"]) if row["cost"] else 0,
                "input_tokens": _safe_int(row["input_tokens"]),
                "billable_input_tokens": _safe_int(row["billable_input_tokens"]),
                "cached_tokens": _safe_int(row["cached_tokens"]),
                "cache_write_tokens": _safe_int(row["cache_write_tokens"]),
                "output_tokens": _safe_int(row["output_tokens"]),
                "total_tokens": _safe_int(row["total_tokens"]),
                "calls": _safe_int(row["calls"]),
                "cache_hit_calls": _safe_int(row["cache_hit_calls"]),
                "cache_write_calls": _safe_int(row["cache_write_calls"]),
                "avg_duration_ms": round(float(row["avg_duration_ms"] or 0), 1),
                "latest_created_at": row["latest_created_at"].isoformat() if row["latest_created_at"] else None,
            }
            for row in operation_rows
        ],
        "batch_jobs": [
            {
                "id": str(row["id"]),
                "stage_id": str(row["stage_id"]),
                "task_name": str(row["task_name"]),
                "status": str(row["status"]),
                "provider_batch_id": row["provider_batch_id"],
                "total_items": _safe_int(row["total_items"]),
                "submitted_items": _safe_int(row["submitted_items"]),
                "cache_prefiltered_items": _safe_int(row["cache_prefiltered_items"]),
                "fallback_single_call_items": _safe_int(row["fallback_single_call_items"]),
                "completed_items": _safe_int(row["completed_items"]),
                "failed_items": _safe_int(row["failed_items"]),
                "estimated_sequential_cost_usd": round(_safe_float(row["estimated_sequential_cost_usd"]), 6),
                "estimated_batch_cost_usd": round(_safe_float(row["estimated_batch_cost_usd"]), 6),
                "estimated_savings_usd": round(
                    max(
                        0.0,
                        _safe_float(row["estimated_sequential_cost_usd"])
                        - _safe_float(row["estimated_batch_cost_usd"]),
                    ),
                    6,
                ),
                "submitted_at": row["submitted_at"].isoformat() if row["submitted_at"] else None,
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
            }
            for row in batch_rows
        ],
        "batch_items": [
            {
                "id": str(row["id"]),
                "batch_id": str(row["batch_id"]),
                "custom_id": str(row["custom_id"]),
                "stage_id": str(row["stage_id"]),
                "task_name": str(row["task_name"]),
                "provider_batch_id": row["provider_batch_id"],
                "artifact_type": str(row["artifact_type"]),
                "artifact_id": str(row["artifact_id"]),
                "vendor_name": str(row["vendor_name"]) if row["vendor_name"] else None,
                "status": str(row["status"]),
                "cache_prefiltered": bool(row["cache_prefiltered"]),
                "fallback_single_call": bool(row["fallback_single_call"]),
                "input_tokens": _safe_int(row["input_tokens"]),
                "billable_input_tokens": _safe_int(row["billable_input_tokens"]),
                "cached_tokens": _safe_int(row["cached_tokens"]),
                "cache_write_tokens": _safe_int(row["cache_write_tokens"]),
                "output_tokens": _safe_int(row["output_tokens"]),
                "total_tokens": _safe_int(row["total_tokens"]),
                "cost_usd": round(_safe_float(row["cost_usd"]), 6),
                "provider_request_id": row["provider_request_id"],
                "error_text": row["error_text"],
                "request_metadata": _normalize_metadata(row["request_metadata"]),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
            }
            for row in batch_item_rows
        ],
        "calls": calls,
        "artifact_attempts": [
            {
                "id": str(row["id"]),
                "artifact_type": row["artifact_type"],
                "artifact_id": row["artifact_id"],
                "run_id": row["run_id"],
                "attempt_no": _safe_int(row["attempt_no"]),
                "stage": row["stage"],
                "status": row["status"],
                "score": row["score"],
                "threshold": row["threshold"],
                "blocker_count": _safe_int(row["blocker_count"]),
                "warning_count": _safe_int(row["warning_count"]),
                "blocking_issues": row["blocking_issues"] if isinstance(row["blocking_issues"], list) else [],
                "warnings": row["warnings"] if isinstance(row["warnings"], list) else [],
                "failure_step": row["failure_step"],
                "error_message": row["error_message"],
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
            }
            for row in attempt_rows
        ],
        "visibility_events": [
            {
                "id": str(row["id"]),
                "occurred_at": row["occurred_at"].isoformat() if row["occurred_at"] else None,
                "run_id": row["run_id"],
                "stage": row["stage"],
                "event_type": row["event_type"],
                "severity": row["severity"],
                "actionable": bool(row["actionable"]),
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "artifact_type": row["artifact_type"],
                "reason_code": row["reason_code"],
                "rule_code": row["rule_code"],
                "decision": row["decision"],
                "summary": row["summary"],
                "detail": _normalize_metadata(row["detail"]),
                "fingerprint": row["fingerprint"],
            }
            for row in event_rows
        ],
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
            COUNT(*) FILTER (WHERE COALESCE(source_weight, 0.7) > 0.7)           AS high_signal_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')                            AS enriched_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'failed')                              AS failed_enrichments,
            ROUND(AVG(COALESCE(source_weight, 0.7))::numeric, 3)                 AS avg_source_weight,
            COUNT(*) FILTER (WHERE author_churn_score >= 7)       AS high_value_authors
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
            COALESCE(source_weight, 0.7)                        AS source_weight,
            reddit_trending                                  AS trending_score,
            author_churn_score                   AS author_churn_score,
            reddit_subreddit                                       AS subreddit,
            reddit_score                                        AS reddit_score,
            reddit_num_comments                             AS num_comments,
            reddit_flair                                      AS post_flair,
            reddit_is_edited                            AS is_edited,
            reddit_is_crosspost                         AS is_crosspost,
            COALESCE(reddit_comment_thread_count, 0)          AS comment_count
        FROM b2b_reviews
        WHERE source = $1
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND COALESCE(source_weight, 0.7) >= $2
        ORDER BY COALESCE(source_weight, 0.7) DESC NULLS LAST,
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
          AND COALESCE(enrichment_status, '') != 'filtered'
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
          AND COALESCE(enrichment_status, '') != 'filtered'
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
            reddit_subreddit                                                   AS subreddit,
            COUNT(*)                                                                      AS total_posts,
            COUNT(*) FILTER (WHERE COALESCE(source_weight, 0.7) > 0.7)      AS high_signal_posts,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')                       AS enriched_posts,
            COUNT(*) FILTER (WHERE enrichment_status = 'no_signal')                      AS no_signal_posts,
            COUNT(*) FILTER (WHERE enrichment_status = 'failed')                         AS failed_posts,
            ROUND(AVG(COALESCE(source_weight, 0.7))::numeric, 3)            AS avg_source_weight,
            ROUND(AVG(
                CASE WHEN enrichment_status = 'enriched'
                     THEN (enrichment->>'urgency_score')::numeric END
            )::numeric, 2)                                                                AS avg_urgency_score,
            ROUND(AVG(
                reddit_score
            )::numeric, 1)                                                                AS avg_reddit_score,
            COUNT(*) FILTER (
                WHERE reddit_trending = 'high'
            )                                                                             AS trending_high_count,
            COUNT(*) FILTER (
                WHERE COALESCE(reddit_comment_thread_count, 0) > 0
            )                                                                             AS comment_harvested_count
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND imported_at >= $1
          AND reddit_subreddit IS NOT NULL
          AND reddit_subreddit != ''
        GROUP BY reddit_subreddit
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
            COALESCE(NULLIF(reddit_flair, ''), '(no flair)')  AS flair,
            COUNT(*)                                                           AS count,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched')            AS enriched,
            ROUND(AVG(COALESCE(source_weight, 0.7))::numeric, 3) AS avg_weight
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND imported_at >= $1
        GROUP BY reddit_flair
        ORDER BY count DESC
        LIMIT 15
        """,
        since,
    )

    # -- Edit stats -----------------------------------------------------------
    edit_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE reddit_is_edited)    AS edited_posts,
            COUNT(*) FILTER (WHERE NOT reddit_is_edited
                               OR reddit_is_edited IS NULL)        AS unedited_posts,
            COUNT(*) FILTER (WHERE reddit_is_edited
                               AND enrichment_status = 'enriched')           AS edited_enriched,
            COUNT(*) FILTER (WHERE (NOT reddit_is_edited
                               OR reddit_is_edited IS NULL)
                               AND enrichment_status = 'enriched')           AS unedited_enriched
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Crosspost stats ------------------------------------------------------
    crosspost_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE reddit_is_crosspost)  AS crossposts,
            COUNT(*) FILTER (WHERE reddit_is_crosspost
                               AND enrichment_status = 'enriched')            AS crosspost_enriched,
            -- Count total unique extra subreddits reached via crossposts
            COALESCE((
                SELECT COUNT(DISTINCT sub)
                FROM b2b_reviews r2,
                     jsonb_array_elements_text(r2.reddit_crosspost_subreddits) AS sub
                WHERE r2.source = 'reddit'
                  AND COALESCE(r2.enrichment_status, '') != 'filtered'
                  AND r2.imported_at >= $1
                  AND r2.reddit_is_crosspost
            ), 0)                                                              AS crosspost_subreddits_reached
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Comment harvest stats ------------------------------------------------
    comment_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE COALESCE(reddit_comment_thread_count, 0) > 0
            )                                                                  AS posts_with_comments,
            ROUND(AVG(
                COALESCE(reddit_comment_thread_count, 0)
            )::numeric, 2)                                                     AS avg_comments_fetched,
            COUNT(*)                                                            AS total_posts
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Author churn score distribution + stats ------------------------------
    author_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE author_churn_score >= 7
            )                                                                  AS high_score_authors,
            ROUND(AVG(author_churn_score)::numeric, 2)
                                                                               AS avg_churn_score,
            COUNT(*) FILTER (
                WHERE author_churn_score < 3
            )                                                                  AS score_0_2,
            COUNT(*) FILTER (
                WHERE author_churn_score >= 3
                  AND author_churn_score < 5
            )                                                                  AS score_3_4,
            COUNT(*) FILTER (
                WHERE author_churn_score >= 5
                  AND author_churn_score < 7
            )                                                                  AS score_5_6,
            COUNT(*) FILTER (
                WHERE author_churn_score >= 7
            )                                                                  AS score_7_10
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
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
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND imported_at >= $1
        """,
        since,
    )

    # -- Trending distribution ------------------------------------------------
    trending_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE reddit_trending = 'high')   AS trending_high,
            COUNT(*) FILTER (WHERE reddit_trending = 'medium') AS trending_medium,
            COUNT(*) FILTER (WHERE reddit_trending = 'low'
                               OR reddit_trending IS NULL)     AS trending_low
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
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
            ROUND(AVG(COALESCE(source_weight, 0.7))::numeric, 3)           AS avg_source_weight,
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
                WHERE reddit_trending = 'high'
            )                                                                            AS trending_high_count
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
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
            reddit_subreddit   AS subreddit,
            COUNT(*)                      AS cnt
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND COALESCE(enrichment_status, '') != 'filtered'
          AND imported_at >= $1
          AND vendor_name = ANY($2)
          AND reddit_subreddit IS NOT NULL
          AND reddit_subreddit != ''
        GROUP BY vendor_name, reddit_subreddit
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
          AND COALESCE(enrichment_status, '') != 'filtered'
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
