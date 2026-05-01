from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _json_object(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _can_write(pool: Any) -> bool:
    return pool is not None and hasattr(pool, "execute")


@dataclass(frozen=True)
class StageRunResolution:
    action: str
    stage_row: dict[str, Any] | None
    parsed_result: dict[str, Any] | None


async def get_stage_run(
    pool: Any,
    *,
    review_id: Any,
    stage_id: str,
    work_fingerprint: str,
) -> dict[str, Any] | None:
    if pool is None or not hasattr(pool, "fetchrow"):
        return None
    row = await pool.fetchrow(
        """
        SELECT *
        FROM b2b_enrichment_stage_runs
        WHERE review_id = $1
          AND stage_id = $2
          AND work_fingerprint = $3
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1
        """,
        review_id,
        stage_id,
        work_fingerprint,
    )
    if row is None:
        return None
    return dict(row)


async def resolve_stage_run(
    pool: Any,
    *,
    review_id: Any,
    stage_id: str,
    work_fingerprint: str,
    parse_response_text: Any,
    defer_on_submitted: bool = False,
) -> StageRunResolution:
    stage_row = await get_stage_run(
        pool,
        review_id=review_id,
        stage_id=stage_id,
        work_fingerprint=work_fingerprint,
    )
    parsed_result = parse_response_text(stage_row)
    state = str((stage_row or {}).get("state") or "").strip().lower()
    if parsed_result is not None and state == "succeeded":
        return StageRunResolution("reuse", stage_row, parsed_result)
    if defer_on_submitted and state == "submitted":
        return StageRunResolution("defer", stage_row, None)
    return StageRunResolution("execute", stage_row, None)


async def ensure_stage_run(
    pool: Any,
    *,
    review_id: Any,
    stage_id: str,
    work_fingerprint: str,
    request_fingerprint: str,
    provider: str,
    model: str,
    backend: str,
    run_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not _can_write(pool):
        return
    await pool.execute(
        """
        INSERT INTO b2b_enrichment_stage_runs (
            review_id,
            stage_id,
            work_fingerprint,
            request_fingerprint,
            provider,
            model,
            backend,
            state,
            result_source,
            run_id,
            metadata,
            updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, 'planned', NULL, $8, $9::jsonb, NOW())
        ON CONFLICT (review_id, stage_id, work_fingerprint)
        DO UPDATE SET
            request_fingerprint = EXCLUDED.request_fingerprint,
            provider = EXCLUDED.provider,
            model = EXCLUDED.model,
            backend = EXCLUDED.backend,
            run_id = COALESCE(EXCLUDED.run_id, b2b_enrichment_stage_runs.run_id),
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """,
        review_id,
        stage_id,
        work_fingerprint,
        request_fingerprint,
        provider,
        model,
        backend,
        run_id,
        _json_object(metadata),
    )


async def mark_stage_run(
    pool: Any,
    *,
    review_id: Any,
    stage_id: str,
    work_fingerprint: str,
    state: str,
    result_source: str | None = None,
    backend: str | None = None,
    batch_id: Any | None = None,
    batch_custom_id: str | None = None,
    usage: dict[str, Any] | None = None,
    response_text: str | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
    completed: bool = False,
) -> None:
    if not _can_write(pool):
        return
    await pool.execute(
        """
        UPDATE b2b_enrichment_stage_runs
        SET state = $4,
            result_source = COALESCE($5, result_source),
            backend = COALESCE($6, backend),
            batch_id = COALESCE($7, batch_id),
            batch_custom_id = COALESCE($8, batch_custom_id),
            usage_json = CASE
                WHEN $9::jsonb = '{}'::jsonb THEN usage_json
                ELSE $9::jsonb
            END,
            response_text = COALESCE($10, response_text),
            error_code = COALESCE($11, error_code),
            metadata = CASE
                WHEN $12::jsonb = '{}'::jsonb THEN metadata
                ELSE $12::jsonb
            END,
            updated_at = NOW(),
            completed_at = CASE
                WHEN $13 THEN NOW()
                ELSE completed_at
            END
        WHERE review_id = $1
          AND stage_id = $2
          AND work_fingerprint = $3
        """,
        review_id,
        stage_id,
        work_fingerprint,
        state,
        result_source,
        backend,
        batch_id,
        batch_custom_id,
        _json_object(usage),
        response_text,
        error_code,
        _json_object(metadata),
        completed,
    )
