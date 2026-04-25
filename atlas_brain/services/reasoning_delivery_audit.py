"""Read-model helpers for reasoning delivery health and provenance coverage."""

from __future__ import annotations

import json
from typing import Any


WITNESS_QUALITY_FIELDS: tuple[str, ...] = (
    "grounding_status",
    "phrase_polarity",
    "phrase_subject",
    "phrase_role",
    "phrase_verbatim",
    "pain_confidence",
)

_WITNESS_ID_KEYS: tuple[str, ...] = ("witness_id", "_sid")
_WITNESS_CONTEXT_KEYS: tuple[str, ...] = (
    "witness_type",
    "selection_reason",
    "reviewer_company",
    "salience_score",
    "pain_category",
)


def _field_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _decode_json_payload(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _looks_like_witness(value: dict[str, Any]) -> bool:
    has_witness_text = _field_present(value.get("excerpt_text")) or _field_present(value.get("quote"))
    if not has_witness_text:
        return False
    return (
        any(_field_present(value.get(key)) for key in _WITNESS_ID_KEYS)
        or any(_field_present(value.get(key)) for key in _WITNESS_CONTEXT_KEYS)
    )


def iter_witness_objects(value: Any, *, path: str = "$"):
    """Yield ``(json_path, object)`` pairs for witness-like dicts in JSON."""
    if isinstance(value, dict):
        if _looks_like_witness(value):
            yield path, value
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield from iter_witness_objects(child, path=child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from iter_witness_objects(child, path=f"{path}[{idx}]")


def _empty_field_counts() -> dict[str, dict[str, int]]:
    return {
        field: {"present": 0, "missing": 0}
        for field in WITNESS_QUALITY_FIELDS
    }


def _empty_surface_stats(surface: str) -> dict[str, Any]:
    return {
        "surface": surface,
        "artifacts_scanned": 0,
        "witness_objects": 0,
        "full_quality_objects": 0,
        "field_counts": _empty_field_counts(),
        "drop_examples": [],
    }


def _record_witness_object(
    stats: dict[str, Any],
    witness: dict[str, Any],
    *,
    artifact_key: str,
    path: str,
) -> None:
    stats["witness_objects"] += 1
    missing: list[str] = []
    for field in WITNESS_QUALITY_FIELDS:
        present = _field_present(witness.get(field))
        bucket = "present" if present else "missing"
        stats["field_counts"][field][bucket] += 1
        if not present:
            missing.append(field)
    if not missing:
        stats["full_quality_objects"] += 1
    elif len(stats["drop_examples"]) < 10:
        stats["drop_examples"].append(
            {
                "artifact_key": artifact_key,
                "path": path,
                "missing_fields": missing,
                "witness_id": witness.get("witness_id") or witness.get("_sid"),
                "excerpt_preview": str(witness.get("excerpt_text") or "")[:120],
            }
        )


def _source_witness_stats(row: dict[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    total = int(row.get("witness_objects", 0) or 0)
    field_counts = _empty_field_counts()
    for field in WITNESS_QUALITY_FIELDS:
        present = int(row.get(f"{field}_present", 0) or 0)
        field_counts[field]["present"] = present
        field_counts[field]["missing"] = max(total - present, 0)
    return {
        "surface": "b2b_vendor_witnesses",
        "artifacts_scanned": total,
        "witness_objects": total,
        "full_quality_objects": int(row.get("full_quality_objects", 0) or 0),
        "field_counts": field_counts,
        "drop_examples": [],
    }


async def summarize_reasoning_delivery_health(
    pool,
    *,
    days: int = 7,
    top_n: int = 10,
) -> dict[str, Any]:
    vendor_summary_row = await pool.fetchrow(
        """
        WITH latest_vendor AS (
            SELECT DISTINCT ON (vendor_name)
                vendor_name,
                as_of_date,
                created_at,
                tokens_used,
                synthesis
            FROM b2b_reasoning_synthesis
            WHERE created_at >= NOW() - make_interval(days => $1)
            ORDER BY vendor_name, created_at DESC
        )
        SELECT
            COUNT(*) AS vendor_rows,
            MAX(created_at) AS latest_created_at,
            COALESCE(AVG(tokens_used), 0) AS avg_tokens_used,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.metric_ids[*]')
            ) AS metric_ref_rows,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.witness_ids[*]')
            ) AS witness_ref_rows
        FROM latest_vendor
        """,
        days,
    )

    cross_vendor_summary = await pool.fetch(
        """
        SELECT
            analysis_type,
            COUNT(*) AS row_count,
            MAX(created_at) AS latest_created_at,
            COALESCE(AVG(tokens_used), 0) AS avg_tokens_used,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.metric_ids[*]')
            ) AS metric_ref_rows,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.witness_ids[*]')
            ) AS witness_ref_rows
        FROM b2b_cross_vendor_reasoning_synthesis
        WHERE created_at >= NOW() - make_interval(days => $1)
        GROUP BY analysis_type
        ORDER BY row_count DESC, analysis_type ASC
        """,
        days,
    )

    validation_rules = await pool.fetch(
        """
        SELECT
            rule_code,
            severity,
            COUNT(*) AS finding_count,
            COUNT(DISTINCT vendor_name) AS vendor_count,
            MAX(created_at) AS latest_created_at
        FROM synthesis_validation_results
        WHERE created_at >= NOW() - make_interval(days => $1)
          AND passed = false
        GROUP BY rule_code, severity
        ORDER BY finding_count DESC, vendor_count DESC, rule_code ASC
        LIMIT $2
        """,
        days,
        top_n,
    )

    latest_attempt_summary = await pool.fetch(
        """
        WITH latest_attempt AS (
            SELECT DISTINCT ON (artifact_type, artifact_id)
                artifact_type,
                artifact_id,
                status,
                stage,
                created_at
            FROM artifact_attempts
            WHERE artifact_type IN ('reasoning_synthesis', 'cross_vendor_reasoning')
              AND created_at >= NOW() - make_interval(days => $1)
            ORDER BY artifact_type, artifact_id, created_at DESC
        )
        SELECT
            artifact_type,
            status,
            COUNT(*) AS row_count
        FROM latest_attempt
        GROUP BY artifact_type, status
        ORDER BY artifact_type ASC, row_count DESC, status ASC
        """,
        days,
    )

    downstream_coverage = await pool.fetch(
        """
        SELECT
            report_type,
            COUNT(*) AS row_count,
            MAX(created_at) AS latest_created_at,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(intelligence_data, '$.**.reference_ids')
                   OR jsonb_path_exists(intelligence_data, '$.**.reasoning_reference_ids')
            ) AS rows_with_reference_ids
        FROM b2b_intelligence
        WHERE report_type IN (
            'battle_card',
            'challenger_brief',
            'accounts_in_motion',
            'weekly_churn_feed',
            'vendor_scorecard',
            'displacement_report',
            'category_overview'
        )
          AND created_at >= NOW() - make_interval(days => $1)
        GROUP BY report_type
        ORDER BY row_count DESC, report_type ASC
        """,
        days,
    )

    vendor_summary = dict(vendor_summary_row or {})
    return {
        "days": days,
        "vendor_synthesis": {
            "rows": int(vendor_summary.get("vendor_rows", 0) or 0),
            "latest_created_at": vendor_summary.get("latest_created_at"),
            "avg_tokens_used": float(vendor_summary.get("avg_tokens_used", 0) or 0),
            "rows_with_metric_refs": int(vendor_summary.get("metric_ref_rows", 0) or 0),
            "rows_with_witness_refs": int(vendor_summary.get("witness_ref_rows", 0) or 0),
        },
        "cross_vendor_synthesis": [dict(row) for row in cross_vendor_summary],
        "top_validation_findings": [dict(row) for row in validation_rules],
        "latest_attempt_statuses": [dict(row) for row in latest_attempt_summary],
        "downstream_reference_coverage": [dict(row) for row in downstream_coverage],
    }


async def summarize_witness_field_propagation(
    pool,
    *,
    days: int = 7,
    row_limit: int = 250,
) -> dict[str, Any]:
    """Audit whether witness-quality fields survive downstream JSON surfaces."""
    source_row = await pool.fetchrow(
        """
        WITH latest AS (
            SELECT vendor_name, analysis_window_days, schema_version, MAX(as_of_date) AS as_of_date
            FROM b2b_vendor_witnesses
            GROUP BY vendor_name, analysis_window_days, schema_version
        )
        SELECT
            COUNT(*) AS witness_objects,
            COUNT(*) FILTER (WHERE w.grounding_status IS NOT NULL AND w.grounding_status <> '') AS grounding_status_present,
            COUNT(*) FILTER (WHERE w.phrase_polarity IS NOT NULL AND w.phrase_polarity <> '') AS phrase_polarity_present,
            COUNT(*) FILTER (WHERE w.phrase_subject IS NOT NULL AND w.phrase_subject <> '') AS phrase_subject_present,
            COUNT(*) FILTER (WHERE w.phrase_role IS NOT NULL AND w.phrase_role <> '') AS phrase_role_present,
            COUNT(*) FILTER (WHERE w.phrase_verbatim IS NOT NULL) AS phrase_verbatim_present,
            COUNT(*) FILTER (WHERE w.pain_confidence IS NOT NULL AND w.pain_confidence <> '') AS pain_confidence_present,
            COUNT(*) FILTER (
                WHERE w.grounding_status IS NOT NULL AND w.grounding_status <> ''
                  AND w.phrase_polarity IS NOT NULL AND w.phrase_polarity <> ''
                  AND w.phrase_subject IS NOT NULL AND w.phrase_subject <> ''
                  AND w.phrase_role IS NOT NULL AND w.phrase_role <> ''
                  AND w.phrase_verbatim IS NOT NULL
                  AND w.pain_confidence IS NOT NULL AND w.pain_confidence <> ''
            ) AS full_quality_objects
        FROM b2b_vendor_witnesses w
        JOIN latest l
          ON l.vendor_name = w.vendor_name
         AND l.analysis_window_days = w.analysis_window_days
         AND l.schema_version = w.schema_version
         AND l.as_of_date = w.as_of_date
        """
    )

    vendor_rows = await pool.fetch(
        """
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
        SELECT
            'b2b_reasoning_synthesis' AS surface,
            vendor_name || ':' || as_of_date::text || ':' || analysis_window_days::text || ':' || schema_version AS artifact_key,
            synthesis AS payload
        FROM latest_vendor
        ORDER BY created_at DESC
        LIMIT $2
        """,
        days,
        row_limit,
    )
    cross_vendor_rows = await pool.fetch(
        """
        SELECT
            'b2b_cross_vendor_reasoning_synthesis:' || analysis_type AS surface,
            id::text AS artifact_key,
            synthesis AS payload
        FROM b2b_cross_vendor_reasoning_synthesis
        WHERE created_at >= NOW() - make_interval(days => $1)
        ORDER BY created_at DESC
        LIMIT $2
        """,
        days,
        row_limit,
    )
    intelligence_rows = await pool.fetch(
        """
        SELECT
            'b2b_intelligence:' || report_type AS surface,
            id::text AS artifact_key,
            intelligence_data AS payload
        FROM b2b_intelligence
        WHERE created_at >= NOW() - make_interval(days => $1)
          AND report_type IN (
              'battle_card',
              'challenger_brief',
              'accounts_in_motion',
              'weekly_churn_feed',
              'vendor_scorecard',
              'displacement_report',
              'category_overview'
          )
        ORDER BY created_at DESC
        LIMIT $2
        """,
        days,
        row_limit,
    )

    surfaces: dict[str, dict[str, Any]] = {
        "b2b_vendor_witnesses": _source_witness_stats(dict(source_row or {})),
    }
    for raw_row in [*vendor_rows, *cross_vendor_rows, *intelligence_rows]:
        row = dict(raw_row)
        surface = str(row.get("surface") or "unknown")
        stats = surfaces.setdefault(surface, _empty_surface_stats(surface))
        stats["artifacts_scanned"] += 1
        artifact_key = str(row.get("artifact_key") or "")
        for path, witness in iter_witness_objects(_decode_json_payload(row.get("payload"))):
            _record_witness_object(
                stats,
                witness,
                artifact_key=artifact_key,
                path=path,
            )

    return {
        "days": days,
        "row_limit": row_limit,
        "quality_fields": list(WITNESS_QUALITY_FIELDS),
        "surfaces": list(surfaces.values()),
    }
