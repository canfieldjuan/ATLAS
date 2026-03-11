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
             COALESCE(AVG(duration_ms), 0)        AS avg_duration_ms,
             COALESCE(AVG(tokens_per_second), 0)  AS avg_tps
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
             COALESCE(AVG(duration_ms), 0)          AS avg_duration_ms,
             COALESCE(AVG(tokens_per_second), 0)    AS avg_tps
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
    return {
        "calls": [
            {
                "span_name": r["span_name"],
                "model": r["model_name"],
                "provider": r["model_provider"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cost_usd": float(r["cost_usd"]) if r["cost_usd"] else 0,
                "duration_ms": r["duration_ms"],
                "tokens_per_second": r["tokens_per_second"],
                "status": r["status"],
                "metadata": r["metadata"] if isinstance(r["metadata"], dict) else {},
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
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
