"""Read-side usage rollups for AI Content Ops LLM calls."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

_CONTENT_OPS_SPAN_NAME = "content_ops.llm.complete"


async def summarize_content_ops_llm_usage(
    pool: Any,
    *,
    days: int = 7,
    asset_type: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Return a recent Content Ops cost/token summary from ``llm_usage``."""

    resolved_days = _bounded_days(days)
    filters = _UsageFilters(
        days=resolved_days,
        asset_type=_clean_filter(asset_type),
        run_id=_clean_filter(run_id),
        request_id=_clean_filter(request_id),
    )
    where_sql, args = _usage_where_clause(filters)
    summary = await pool.fetchrow(
        f"""
        SELECT
            COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
            COALESCE(SUM(input_tokens), 0)::BIGINT AS input_tokens,
            COALESCE(SUM(billable_input_tokens), 0)::BIGINT AS billable_input_tokens,
            COALESCE(SUM(output_tokens), 0)::BIGINT AS output_tokens,
            COALESCE(SUM(total_tokens), 0)::BIGINT AS total_tokens,
            COALESCE(SUM(cached_tokens), 0)::BIGINT AS cached_tokens,
            COALESCE(SUM(cache_write_tokens), 0)::BIGINT AS cache_write_tokens,
            COUNT(*)::BIGINT AS total_calls,
            COUNT(*) FILTER (WHERE status != 'completed')::BIGINT AS failed_calls,
            COUNT(*) FILTER (WHERE cached_tokens > 0)::BIGINT AS cache_hit_calls,
            COALESCE(AVG(duration_ms) FILTER (WHERE duration_ms > 0), 0) AS avg_duration_ms,
            MAX(created_at) AS latest_call_at
        FROM llm_usage
        WHERE {where_sql}
        """,
        *args,
    )
    by_model = await pool.fetch(
        f"""
        SELECT
            COALESCE(NULLIF(BTRIM(model_provider), ''), 'unknown') AS provider,
            COALESCE(NULLIF(BTRIM(model_name), ''), 'unknown') AS model,
            COALESCE(SUM(cost_usd), 0) AS cost_usd,
            COUNT(*)::BIGINT AS calls,
            COALESCE(SUM(input_tokens), 0)::BIGINT AS input_tokens,
            COALESCE(SUM(output_tokens), 0)::BIGINT AS output_tokens
        FROM llm_usage
        WHERE {where_sql}
        GROUP BY provider, model
        ORDER BY cost_usd DESC, calls DESC
        LIMIT 20
        """,
        *args,
    )
    by_asset_type = await pool.fetch(
        f"""
        SELECT
            COALESCE(NULLIF(BTRIM(metadata ->> 'asset_type'), ''), 'unknown') AS asset_type,
            COALESCE(SUM(cost_usd), 0) AS cost_usd,
            COUNT(*)::BIGINT AS calls,
            COALESCE(SUM(input_tokens), 0)::BIGINT AS input_tokens,
            COALESCE(SUM(output_tokens), 0)::BIGINT AS output_tokens
        FROM llm_usage
        WHERE {where_sql}
        GROUP BY asset_type
        ORDER BY cost_usd DESC, calls DESC
        LIMIT 20
        """,
        *args,
    )
    return {
        "period_days": resolved_days,
        "filters": filters.as_dict(),
        "summary": _summary_payload(summary),
        "by_model": [_breakdown_payload(row, label_keys=("provider", "model")) for row in by_model],
        "by_asset_type": [
            _breakdown_payload(row, label_keys=("asset_type",)) for row in by_asset_type
        ],
    }


class _UsageFilters:
    def __init__(
        self,
        *,
        days: int,
        asset_type: str | None,
        run_id: str | None,
        request_id: str | None,
    ) -> None:
        self.days = days
        self.asset_type = asset_type
        self.run_id = run_id
        self.request_id = request_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "run_id": self.run_id,
            "request_id": self.request_id,
        }


def _usage_where_clause(filters: _UsageFilters) -> tuple[str, list[Any]]:
    args: list[Any] = [filters.days]
    clauses = [
        "created_at >= NOW() - ($1::INT * INTERVAL '1 day')",
        (
            f"(span_name = '{_CONTENT_OPS_SPAN_NAME}' "
            "OR metadata ->> 'product' = 'content_ops')"
        ),
    ]
    if filters.asset_type:
        args.append(filters.asset_type)
        clauses.append(f"metadata ->> 'asset_type' = ${len(args)}")
    if filters.run_id:
        args.append(filters.run_id)
        clauses.append(f"(run_id = ${len(args)} OR metadata ->> 'run_id' = ${len(args)})")
    if filters.request_id:
        args.append(filters.request_id)
        clauses.append(f"metadata ->> 'request_id' = ${len(args)}")
    return "\n          AND ".join(clauses), args


def _summary_payload(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    return {
        "total_cost_usd": _float_value(row.get("total_cost_usd")),
        "total_calls": _int_value(row.get("total_calls")),
        "failed_calls": _int_value(row.get("failed_calls")),
        "input_tokens": _int_value(row.get("input_tokens")),
        "billable_input_tokens": _int_value(row.get("billable_input_tokens")),
        "output_tokens": _int_value(row.get("output_tokens")),
        "total_tokens": _int_value(row.get("total_tokens")),
        "cached_tokens": _int_value(row.get("cached_tokens")),
        "cache_write_tokens": _int_value(row.get("cache_write_tokens")),
        "cache_hit_calls": _int_value(row.get("cache_hit_calls")),
        "avg_duration_ms": round(_float_value(row.get("avg_duration_ms")), 1),
        "latest_call_at": _iso_or_none(row.get("latest_call_at")),
    }


def _breakdown_payload(
    row: Mapping[str, Any],
    *,
    label_keys: tuple[str, ...],
) -> dict[str, Any]:
    payload = {key: str(row.get(key) or "unknown") for key in label_keys}
    payload.update({
        "cost_usd": _float_value(row.get("cost_usd")),
        "calls": _int_value(row.get("calls")),
        "input_tokens": _int_value(row.get("input_tokens")),
        "output_tokens": _int_value(row.get("output_tokens")),
    })
    return payload


def _bounded_days(value: int) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return 7
    return min(max(days, 1), 90)


def _clean_filter(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:200]


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    if isinstance(value, Decimal):
        return round(float(value), 6)
    try:
        return round(float(value or 0), 6)
    except (TypeError, ValueError):
        return 0.0


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    text = str(value).strip()
    return text or None
