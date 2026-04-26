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


def _witness_id(value: dict[str, Any]) -> str:
    return str(value.get("witness_id") or value.get("_sid") or value.get("source_id") or "").strip()


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
        "source_matched_objects": 0,
        "full_quality_objects": 0,
        "field_counts": _empty_field_counts(),
        "fillable_missing_fields": 0,
        "source_unavailable_missing_fields": 0,
        "drop_examples": [],
        "fillable_drop_examples": [],
    }


def _record_witness_object(
    stats: dict[str, Any],
    witness: dict[str, Any],
    *,
    artifact_key: str,
    path: str,
    source_quality_by_witness_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    stats["witness_objects"] += 1
    witness_id = _witness_id(witness)
    source_quality = (source_quality_by_witness_id or {}).get(witness_id) or {}
    if source_quality:
        stats["source_matched_objects"] += 1
    missing: list[str] = []
    fillable_missing: list[str] = []
    for field in WITNESS_QUALITY_FIELDS:
        present = _field_present(witness.get(field))
        bucket = "present" if present else "missing"
        stats["field_counts"][field][bucket] += 1
        if not present:
            missing.append(field)
            if _field_present(source_quality.get(field)):
                fillable_missing.append(field)
                stats["fillable_missing_fields"] += 1
            else:
                stats["source_unavailable_missing_fields"] += 1
    if not missing:
        stats["full_quality_objects"] += 1
    elif len(stats["drop_examples"]) < 10:
        stats["drop_examples"].append(
            {
                "artifact_key": artifact_key,
                "path": path,
                "missing_fields": missing,
                "witness_id": witness_id or None,
                "excerpt_preview": str(witness.get("excerpt_text") or "")[:120],
            }
        )
    if fillable_missing and len(stats["fillable_drop_examples"]) < 10:
        stats["fillable_drop_examples"].append(
            {
                "artifact_key": artifact_key,
                "path": path,
                "missing_fields": fillable_missing,
                "witness_id": witness_id or None,
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
        "source_matched_objects": total,
        "full_quality_objects": int(row.get("full_quality_objects", 0) or 0),
        "field_counts": field_counts,
        "fillable_missing_fields": 0,
        "source_unavailable_missing_fields": sum(
            counts["missing"] for counts in field_counts.values()
        ),
        "drop_examples": [],
        "fillable_drop_examples": [],
    }


async def _source_quality_lookup(pool, witness_ids: set[str]) -> dict[str, dict[str, Any]]:
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
        row = dict(raw)
        witness_id = str(row.get("witness_id") or "").strip()
        if not witness_id:
            continue
        quality = {
            field: row.get(field)
            for field in WITNESS_QUALITY_FIELDS
            if _field_present(row.get(field))
        }
        if quality:
            lookup[witness_id] = quality
    return lookup


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

    artifact_rows = [
        {
            "surface": str(dict(raw_row).get("surface") or "unknown"),
            "artifact_key": str(dict(raw_row).get("artifact_key") or ""),
            "payload": _decode_json_payload(dict(raw_row).get("payload")),
        }
        for raw_row in [*vendor_rows, *cross_vendor_rows, *intelligence_rows]
    ]
    witness_ids: set[str] = set()
    for artifact_row in artifact_rows:
        for _path, witness in iter_witness_objects(artifact_row["payload"]):
            witness_id = _witness_id(witness)
            if witness_id:
                witness_ids.add(witness_id)
    source_quality_by_witness_id = await _source_quality_lookup(pool, witness_ids)

    surfaces: dict[str, dict[str, Any]] = {
        "b2b_vendor_witnesses": _source_witness_stats(dict(source_row or {})),
    }
    for row in artifact_rows:
        surface = row["surface"]
        stats = surfaces.setdefault(surface, _empty_surface_stats(surface))
        stats["artifacts_scanned"] += 1
        for path, witness in iter_witness_objects(row["payload"]):
            _record_witness_object(
                stats,
                witness,
                artifact_key=row["artifact_key"],
                path=path,
                source_quality_by_witness_id=source_quality_by_witness_id,
            )

    return {
        "days": days,
        "row_limit": row_limit,
        "quality_fields": list(WITNESS_QUALITY_FIELDS),
        "surfaces": list(surfaces.values()),
    }


# ----------------------------------------------------------------------------
# EvidenceClaim contract audit (Phase 9 step 6)
# ----------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


async def summarize_claim_validation(
    pool,
    *,
    as_of_date: Any,
    invalid_examples_per_reason: int = 3,
    rejection_reasons_per_claim_type: int = 10,
) -> dict[str, Any]:
    """Audit summary over rows in b2b_evidence_claims for one as_of_date.

    Used by:

      - the daily autonomous task (b2b_evidence_claim_audit) so the
        operator gets a digest after every nightly synthesis cycle.
      - the CLI script (scripts/audit_evidence_claims.py) for ad-hoc
        canary triage.

    Returns a dict with these keys:

      as_of_date                       (echoed back)
      scope                            total_rows + distinct vendors / artifacts
      totals                           valid / invalid / cannot_validate counts
      by_claim_type                    same totals broken out per claim_type
      rejection_reasons_by_claim_type  top-N reasons per claim_type for invalid
      cannot_validate_reasons_by_claim_type
                                       same for cannot_validate (so v3 / synthesized
                                       short-circuits don't get hidden)
      by_vendor                        per-vendor counts (canary single-vendor view)
      by_source                        joined to b2b_reviews.source platform
      by_pain_category                 derived from claim_payload.pain_category
      by_schema_version                joined to b2b_reviews.enrichment schema version
      invalid_examples                 small sample of invalid rows for hand audit

    Per-source / per-schema-version breakdowns LEFT JOIN b2b_reviews on
    source_review_id and gracefully bucket missing joins (synthesized
    spans, deleted reviews) under a 'unknown' / 'unjoined' label.
    """
    if as_of_date is None:
        raise ValueError("as_of_date is required")

    scope_row = await pool.fetchrow(
        """
        SELECT
            count(*) AS total_rows,
            count(DISTINCT vendor_name) AS distinct_vendors,
            count(DISTINCT artifact_id) AS distinct_artifacts
        FROM b2b_evidence_claims
        WHERE as_of_date = $1
        """,
        as_of_date,
    )

    totals_rows = await pool.fetch(
        """
        SELECT status, count(*) AS n
        FROM b2b_evidence_claims
        WHERE as_of_date = $1
        GROUP BY status
        """,
        as_of_date,
    )
    totals = {"valid": 0, "invalid": 0, "cannot_validate": 0}
    for row in totals_rows:
        if row["status"] in totals:
            totals[row["status"]] = int(row["n"] or 0)

    by_claim_type_rows = await pool.fetch(
        """
        SELECT claim_type, status, count(*) AS n
        FROM b2b_evidence_claims
        WHERE as_of_date = $1
        GROUP BY claim_type, status
        ORDER BY claim_type
        """,
        as_of_date,
    )
    by_claim_type: dict[str, dict[str, int]] = {}
    for row in by_claim_type_rows:
        bucket = by_claim_type.setdefault(
            row["claim_type"],
            {"total": 0, "valid": 0, "invalid": 0, "cannot_validate": 0},
        )
        n = int(row["n"] or 0)
        bucket["total"] += n
        if row["status"] in bucket:
            bucket[row["status"]] += n

    reason_rows = await pool.fetch(
        """
        SELECT * FROM (
            SELECT
                claim_type,
                status,
                rejection_reason,
                count(*) AS n,
                ROW_NUMBER() OVER (
                    PARTITION BY claim_type, status
                    ORDER BY count(*) DESC, rejection_reason
                ) AS rn
            FROM b2b_evidence_claims
            WHERE as_of_date = $1
              AND status IN ('invalid', 'cannot_validate')
              AND rejection_reason IS NOT NULL
            GROUP BY claim_type, status, rejection_reason
        ) t
        WHERE rn <= $2
        ORDER BY claim_type, status, rn
        """,
        as_of_date,
        int(rejection_reasons_per_claim_type),
    )
    rejection_reasons_by_claim_type: dict[str, list[dict[str, Any]]] = {}
    cannot_validate_reasons_by_claim_type: dict[str, list[dict[str, Any]]] = {}
    for row in reason_rows:
        entry = {
            "rejection_reason": row["rejection_reason"],
            "count": int(row["n"] or 0),
        }
        if row["status"] == "invalid":
            rejection_reasons_by_claim_type.setdefault(row["claim_type"], []).append(entry)
        else:
            cannot_validate_reasons_by_claim_type.setdefault(row["claim_type"], []).append(entry)

    by_vendor_rows = await pool.fetch(
        """
        SELECT
            vendor_name,
            count(*) FILTER (WHERE status = 'valid') AS valid_count,
            count(*) FILTER (WHERE status = 'invalid') AS invalid_count,
            count(*) FILTER (WHERE status = 'cannot_validate') AS cannot_validate_count,
            count(*) AS total
        FROM b2b_evidence_claims
        WHERE as_of_date = $1
        GROUP BY vendor_name
        ORDER BY total DESC, vendor_name
        """,
        as_of_date,
    )
    by_vendor = [
        {
            "vendor_name": row["vendor_name"],
            "valid": int(row["valid_count"] or 0),
            "invalid": int(row["invalid_count"] or 0),
            "cannot_validate": int(row["cannot_validate_count"] or 0),
            "total": int(row["total"] or 0),
        }
        for row in by_vendor_rows
    ]

    by_source_rows = await pool.fetch(
        """
        SELECT
            COALESCE(r.source, 'unjoined') AS source,
            ec.status,
            count(*) AS n
        FROM b2b_evidence_claims ec
        LEFT JOIN b2b_reviews r ON r.id = ec.source_review_id
        WHERE ec.as_of_date = $1
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        as_of_date,
    )
    by_source: dict[str, dict[str, int]] = {}
    for row in by_source_rows:
        bucket = by_source.setdefault(
            row["source"],
            {"total": 0, "valid": 0, "invalid": 0, "cannot_validate": 0},
        )
        n = int(row["n"] or 0)
        bucket["total"] += n
        if row["status"] in bucket:
            bucket[row["status"]] += n

    by_pain_category_rows = await pool.fetch(
        """
        SELECT
            COALESCE(NULLIF(claim_payload->>'pain_category', ''), 'unknown') AS pain_category,
            status,
            count(*) AS n
        FROM b2b_evidence_claims
        WHERE as_of_date = $1
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        as_of_date,
    )
    by_pain_category: dict[str, dict[str, int]] = {}
    for row in by_pain_category_rows:
        bucket = by_pain_category.setdefault(
            row["pain_category"],
            {"total": 0, "valid": 0, "invalid": 0, "cannot_validate": 0},
        )
        n = int(row["n"] or 0)
        bucket["total"] += n
        if row["status"] in bucket:
            bucket[row["status"]] += n

    by_schema_version_rows = await pool.fetch(
        """
        SELECT
            COALESCE(r.enrichment->>'enrichment_schema_version', 'unknown') AS schema_version,
            ec.status,
            count(*) AS n
        FROM b2b_evidence_claims ec
        LEFT JOIN b2b_reviews r ON r.id = ec.source_review_id
        WHERE ec.as_of_date = $1
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        as_of_date,
    )
    by_schema_version: dict[str, dict[str, int]] = {}
    for row in by_schema_version_rows:
        bucket = by_schema_version.setdefault(
            row["schema_version"],
            {"total": 0, "valid": 0, "invalid": 0, "cannot_validate": 0},
        )
        n = int(row["n"] or 0)
        bucket["total"] += n
        if row["status"] in bucket:
            bucket[row["status"]] += n

    invalid_examples_rows = await pool.fetch(
        """
        SELECT * FROM (
            SELECT
                claim_type,
                rejection_reason,
                witness_id,
                vendor_name,
                claim_payload->>'excerpt_text' AS excerpt_preview,
                ROW_NUMBER() OVER (
                    PARTITION BY claim_type, rejection_reason
                    ORDER BY validated_at DESC
                ) AS rn
            FROM b2b_evidence_claims
            WHERE as_of_date = $1
              AND status = 'invalid'
              AND rejection_reason IS NOT NULL
        ) t
        WHERE rn <= $2
        ORDER BY claim_type, rejection_reason, rn
        """,
        as_of_date,
        int(invalid_examples_per_reason),
    )
    invalid_examples = [
        {
            "claim_type": row["claim_type"],
            "rejection_reason": row["rejection_reason"],
            "witness_id": row["witness_id"],
            "vendor_name": row["vendor_name"],
            "excerpt_preview": (row["excerpt_preview"] or "")[:200],
        }
        for row in invalid_examples_rows
    ]

    return {
        "as_of_date": str(as_of_date),
        "scope": {
            "total_rows": int((scope_row or {}).get("total_rows", 0) or 0),
            "distinct_vendors": int((scope_row or {}).get("distinct_vendors", 0) or 0),
            "distinct_artifacts": int((scope_row or {}).get("distinct_artifacts", 0) or 0),
        },
        "totals": totals,
        "by_claim_type": by_claim_type,
        "rejection_reasons_by_claim_type": rejection_reasons_by_claim_type,
        "cannot_validate_reasons_by_claim_type": cannot_validate_reasons_by_claim_type,
        "by_vendor": by_vendor,
        "by_source": by_source,
        "by_pain_category": by_pain_category,
        "by_schema_version": by_schema_version,
        "invalid_examples": invalid_examples,
    }
