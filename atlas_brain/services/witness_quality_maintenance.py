"""Operational maintenance for witness-quality field propagation.

Bundles:
  - The backfill orchestration that scripts/backfill_witness_quality_fields.py
    used to inline. Extracted so both the CLI script and the daily
    autonomous task can share one code path.
  - The combined "backfill + audit + alert decision" used by the daily
    maintenance task. After the backfill runs in --apply mode, we re-run
    summarize_witness_field_propagation and verify the invariant
    (fillable_missing_fields == 0 across every surface). Anything > 0
    indicates a write-path leak that backfill alone cannot keep up with;
    that's the alert condition.

The maintenance task itself lives at
``atlas_brain/autonomous/tasks/b2b_witness_quality_maintenance.py`` and
calls :func:`run_witness_quality_maintenance` here.
"""

from __future__ import annotations

import logging
from typing import Any

from atlas_brain.services.reasoning_delivery_audit import (
    summarize_witness_field_propagation,
)
from atlas_brain.services.witness_quality_propagation import (
    collect_quote_witness_ids,
    compact_json,
    decode_json_payload,
    decorate_witness_quality_fields,
    normalize_witness_quality_row,
)

logger = logging.getLogger("atlas.witness_quality_maintenance")


_REPORT_TYPES: tuple[str, ...] = (
    "battle_card",
    "challenger_brief",
    "accounts_in_motion",
    "weekly_churn_feed",
    "vendor_scorecard",
    "displacement_report",
    "category_overview",
)


def _limit_clause(limit: int | None, placeholder: str) -> str:
    return f"LIMIT {placeholder}" if limit is not None else ""


async def _quality_lookup(pool, witness_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not witness_ids:
        return {}
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (witness_id)
            witness_id,
            grounding_status,
            phrase_polarity,
            phrase_subject,
            phrase_role,
            phrase_verbatim,
            pain_confidence
        FROM b2b_vendor_witnesses
        WHERE witness_id = ANY($1::text[])
        ORDER BY witness_id, as_of_date DESC, created_at DESC
        """,
        sorted(witness_ids),
    )
    lookup: dict[str, dict[str, Any]] = {}
    for raw in rows:
        witness_id, quality = normalize_witness_quality_row(dict(raw))
        if witness_id and quality:
            lookup[witness_id] = quality
    return lookup


def _merge_counts(total: dict[str, int], delta: dict[str, int]) -> None:
    for key, value in delta.items():
        total[key] = int(total.get(key, 0) or 0) + int(value or 0)


async def _backfill_table(
    pool,
    *,
    table: str,
    payload_column: str,
    rows: list[Any],
    apply: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Decorate quote-bearing witnesses in one source table."""
    all_ids: set[str] = set()
    decoded_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        payload = decode_json_payload(row[payload_column])
        row["_decoded_payload"] = payload
        row["_witness_ids"] = collect_quote_witness_ids(payload)
        all_ids.update(row["_witness_ids"])
        decoded_rows.append(row)

    lookup = await _quality_lookup(pool, all_ids)
    summary: dict[str, Any] = {
        "scanned_rows": len(decoded_rows),
        "quote_witness_ids": len(all_ids),
        "quality_matches": len(lookup),
        "changed_rows": 0,
        "witness_objects_seen": 0,
        "witness_objects_matched": 0,
        "witness_objects_updated": 0,
        "fields_written": 0,
    }

    for row in decoded_rows:
        original = row["_decoded_payload"]
        decorated, stats = decorate_witness_quality_fields(
            original,
            lookup,
            overwrite=overwrite,
        )
        _merge_counts(summary, stats)
        if compact_json(decorated) == compact_json(original):
            continue
        summary["changed_rows"] += 1
        if not apply:
            continue
        if table == "b2b_reasoning_synthesis":
            await pool.execute(
                """
                UPDATE b2b_reasoning_synthesis
                   SET synthesis = $1::jsonb
                 WHERE vendor_name = $2
                   AND as_of_date = $3
                   AND analysis_window_days = $4
                   AND schema_version = $5
                """,
                compact_json(decorated),
                row["vendor_name"],
                row["as_of_date"],
                row["analysis_window_days"],
                row["schema_version"],
            )
        elif table == "b2b_cross_vendor_reasoning_synthesis":
            await pool.execute(
                """
                UPDATE b2b_cross_vendor_reasoning_synthesis
                   SET synthesis = $1::jsonb
                 WHERE id = $2
                """,
                compact_json(decorated),
                row["id"],
            )
        elif table == "b2b_intelligence":
            await pool.execute(
                """
                UPDATE b2b_intelligence
                   SET intelligence_data = $1::jsonb
                 WHERE id = $2
                """,
                compact_json(decorated),
                row["id"],
            )
        else:
            raise ValueError(f"unsupported table: {table}")
    return summary


async def run_backfill(
    pool,
    *,
    days: int = 30,
    apply: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the three-table backfill scan and (optionally) apply updates.

    Returns the same structure the CLI script prints: a top-level dict
    with ``apply``, ``days``, ``limit``, ``overwrite``, and a ``tables``
    map of per-table summaries.
    """
    vendor_args = (days, limit) if limit is not None else (days,)
    intelligence_args = (
        (days, list(_REPORT_TYPES), limit)
        if limit is not None
        else (days, list(_REPORT_TYPES))
    )
    vendor_rows = await pool.fetch(
        f"""
        WITH latest_vendor AS (
            SELECT DISTINCT ON (vendor_name)
                vendor_name,
                as_of_date,
                analysis_window_days,
                schema_version,
                synthesis,
                created_at
            FROM b2b_reasoning_synthesis
            WHERE created_at >= NOW() - make_interval(days => $1)
            ORDER BY vendor_name, created_at DESC
        )
        SELECT vendor_name, as_of_date, analysis_window_days, schema_version, synthesis
        FROM latest_vendor
        ORDER BY created_at DESC
        {_limit_clause(limit, '$2')}
        """,
        *vendor_args,
    )
    cross_vendor_rows = await pool.fetch(
        f"""
        SELECT id, synthesis
        FROM b2b_cross_vendor_reasoning_synthesis
        WHERE created_at >= NOW() - make_interval(days => $1)
        ORDER BY created_at DESC
        {_limit_clause(limit, '$2')}
        """,
        *vendor_args,
    )
    intelligence_rows = await pool.fetch(
        f"""
        SELECT id, intelligence_data
        FROM b2b_intelligence
        WHERE created_at >= NOW() - make_interval(days => $1)
          AND report_type = ANY($2::text[])
        ORDER BY created_at DESC
        {_limit_clause(limit, '$3')}
        """,
        *intelligence_args,
    )
    return {
        "apply": bool(apply),
        "days": days,
        "limit": limit,
        "overwrite": bool(overwrite),
        "tables": {
            "b2b_reasoning_synthesis": await _backfill_table(
                pool,
                table="b2b_reasoning_synthesis",
                payload_column="synthesis",
                rows=vendor_rows,
                apply=apply,
                overwrite=overwrite,
            ),
            "b2b_cross_vendor_reasoning_synthesis": await _backfill_table(
                pool,
                table="b2b_cross_vendor_reasoning_synthesis",
                payload_column="synthesis",
                rows=cross_vendor_rows,
                apply=apply,
                overwrite=overwrite,
            ),
            "b2b_intelligence": await _backfill_table(
                pool,
                table="b2b_intelligence",
                payload_column="intelligence_data",
                rows=intelligence_rows,
                apply=apply,
                overwrite=overwrite,
            ),
        },
    }


def _surface_fillable_summary(audit: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Return (total_fillable, leaking_surfaces) from an audit result."""
    total_fillable = 0
    leaking: list[dict[str, Any]] = []
    for surface in audit.get("surfaces", []):
        fillable = int(surface.get("fillable_missing_fields", 0) or 0)
        total_fillable += fillable
        if fillable > 0:
            leaking.append(
                {
                    "surface": surface.get("surface"),
                    "fillable_missing_fields": fillable,
                    "witness_objects": int(surface.get("witness_objects", 0) or 0),
                }
            )
    return total_fillable, leaking


async def run_witness_quality_maintenance(
    pool,
    *,
    days: int = 30,
    apply: bool = True,
    overwrite: bool = False,
    audit_row_limit: int | None = 500,
) -> dict[str, Any]:
    """Run the daily backfill + audit pass and return a structured result.

    The audit always runs on the post-backfill DB state. ``alert_triggered``
    is True when any surface still reports ``fillable_missing_fields > 0``
    after the backfill ran -- which means the propagation invariant we
    just established in Phase 8 has regressed and a write path is leaking
    quality fields that the lazy backfill cannot keep up with.

    The caller (the autonomous task handler) is responsible for fanning
    the alert out via ntfy / log / etc. This service does not touch the
    notification subsystem so it stays test-friendly.
    """
    backfill = await run_backfill(
        pool,
        days=days,
        apply=apply,
        overwrite=overwrite,
    )
    audit = await summarize_witness_field_propagation(
        pool,
        days=days,
        row_limit=audit_row_limit if audit_row_limit is not None else 500,
    )
    total_fillable, leaking_surfaces = _surface_fillable_summary(audit)
    return {
        "days": days,
        "apply": bool(apply),
        "backfill": backfill,
        "audit": audit,
        "fillable_missing_fields": total_fillable,
        "leaking_surfaces": leaking_surfaces,
        "alert_triggered": total_fillable > 0,
    }
