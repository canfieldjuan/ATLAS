"""B2B Churn MCP -- corrections tools."""

import json
import uuid as _uuid
from typing import Optional

from ._shared import ReviewSource, _is_uuid, _safe_json, get_pool, logger
from .server import mcp

_VALID_ENTITY_TYPES = {
    "review", "vendor", "displacement_edge", "pain_point",
    "churn_signal", "buyer_profile", "use_case", "integration",
    "source",
}
_VALID_CORRECTION_TYPES = {"suppress", "flag", "override_field", "merge_vendor", "reclassify", "suppress_source"}
_KNOWN_SOURCES = {s.value for s in ReviewSource}
_REVIEW_BASIS_RAW_PROVENANCE = "raw_source_provenance"


@mcp.tool()
async def create_data_correction(
    entity_type: str,
    entity_id: str,
    correction_type: str,
    reason: str,
    field_name: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    corrected_by: str = "mcp",
    metadata: Optional[str] = None,
) -> str:
    """
    Record an analyst correction for a data entity.

    entity_type: Type of entity being corrected (review, vendor, displacement_edge,
        pain_point, churn_signal, buyer_profile, use_case, integration, source)
    entity_id: UUID of the entity being corrected. For suppress_source corrections,
        this field is optional -- a deterministic sentinel UUID is generated server-side
        from the source_name in metadata.
    correction_type: Type of correction (suppress, flag, override_field, merge_vendor,
        reclassify, suppress_source)
    reason: Human explanation for the correction (required)
    field_name: Which field was changed (required for override_field; for suppress_source,
        optional vendor scope)
    old_value: Previous value (optional)
    new_value: New value (required for override_field)
    corrected_by: Who made the correction (default: mcp)
    metadata: Optional JSON string with extra data (e.g., '{"source_name": "reddit"}' for
        suppress_source)
    """
    if entity_type not in _VALID_ENTITY_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid entity_type. Must be one of: {sorted(_VALID_ENTITY_TYPES)}",
        })
    if correction_type not in _VALID_CORRECTION_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid correction_type. Must be one of: {sorted(_VALID_CORRECTION_TYPES)}",
        })
    if correction_type == "override_field":
        if not field_name:
            return json.dumps({"success": False, "error": "field_name required for override_field"})
        if new_value is None:
            return json.dumps({"success": False, "error": "new_value required for override_field"})
    if correction_type == "merge_vendor":
        if not old_value or not new_value:
            return json.dumps({
                "success": False,
                "error": "merge_vendor requires old_value (source vendor) and new_value (target vendor)",
            })
    if correction_type == "suppress_source":
        if entity_type != "source":
            return json.dumps({
                "success": False,
                "error": "suppress_source corrections must use entity_type='source'",
            })
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"success": False, "error": "metadata must be valid JSON"})
        if not meta_dict.get("source_name"):
            return json.dumps({
                "success": False,
                "error": "suppress_source requires metadata with source_name (e.g., '{\"source_name\": \"reddit\"}')",
            })
        if meta_dict["source_name"].lower() not in _KNOWN_SOURCES:
            return json.dumps({
                "success": False,
                "error": f"Unknown source '{meta_dict['source_name']}'. Known: {sorted(_KNOWN_SOURCES)}",
            })
        # Clients have no natural UUID for a source -- generate a deterministic sentinel
        # so the call succeeds without requiring a fake UUID in entity_id.
        import uuid as _uuid_mod
        entity_id = str(_uuid_mod.uuid5(
            _uuid_mod.NAMESPACE_DNS, f"source:{meta_dict['source_name'].lower()}"
        ))
    if not _is_uuid(entity_id):
        return json.dumps({"success": False, "error": "entity_id must be a valid UUID"})
    if not reason or not reason.strip():
        return json.dumps({"success": False, "error": "reason is required"})

    try:
        pool = get_pool()
        import uuid as _u
        entity_uuid = _u.UUID(entity_id)
        # Use the already-validated meta_dict (normalised, no extra whitespace/fields)
        # rather than the raw input string so stored JSONB is canonical.
        meta_json = json.dumps(meta_dict) if meta_dict else "{}"

        row = await pool.fetchrow(
            """
            INSERT INTO data_corrections
                (entity_type, entity_id, correction_type, field_name,
                 old_value, new_value, reason, corrected_by, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
            RETURNING id, entity_type, entity_id, correction_type, status, created_at
            """,
            entity_type,
            entity_uuid,
            correction_type,
            field_name,
            old_value,
            new_value,
            reason,
            corrected_by,
            meta_json,
        )

        correction_id = row["id"]

        # Execute vendor merge if applicable
        merge_info = None
        if correction_type == "merge_vendor":
            from atlas_brain.services.b2b.vendor_merge import execute_vendor_merge
            merge_result = await execute_vendor_merge(pool, old_value, new_value)
            await pool.execute(
                "UPDATE data_corrections SET affected_count = $1, metadata = $2::jsonb WHERE id = $3",
                merge_result["total_affected"], json.dumps(merge_result), correction_id,
            )
            merge_info = merge_result

        result = {
            "success": True,
            "correction": {
                "id": str(correction_id),
                "entity_type": row["entity_type"],
                "entity_id": str(row["entity_id"]),
                "correction_type": row["correction_type"],
                "status": row["status"],
                "created_at": str(row["created_at"]),
            },
        }
        if merge_info:
            result["merge"] = merge_info

        return json.dumps(result, default=str)
    except Exception:
        logger.exception("create_data_correction error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_data_corrections(
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    correction_type: Optional[str] = None,
    status: Optional[str] = None,
    corrected_by: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    List recorded data corrections with optional filters.

    entity_type: Filter by entity type (review, vendor, etc.)
    entity_id: Filter by entity UUID
    correction_type: Filter by correction type (suppress, flag, override_field, etc.)
    status: Filter by status (applied, reverted, pending_review)
    corrected_by: Filter by who made the correction (substring match)
    start_date: Corrections created on or after (ISO 8601)
    end_date: Corrections created before (ISO 8601)
    limit: Max results (1-200, default 50)
    """
    limit = max(1, min(limit, 200))

    if entity_type and entity_type not in _VALID_ENTITY_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid entity_type. Must be one of: {sorted(_VALID_ENTITY_TYPES)}",
        })
    if correction_type and correction_type not in _VALID_CORRECTION_TYPES:
        return json.dumps({
            "success": False,
            "error": f"Invalid correction_type. Must be one of: {sorted(_VALID_CORRECTION_TYPES)}",
        })
    if entity_id and not _is_uuid(entity_id):
        return json.dumps({"success": False, "error": "entity_id must be a valid UUID"})

    try:
        from datetime import datetime as _dt

        pool = get_pool()

        conditions: list[str] = []
        params: list = []
        idx = 1

        if entity_type:
            conditions.append(f"entity_type = ${idx}")
            params.append(entity_type)
            idx += 1
        if entity_id:
            import uuid as _u
            conditions.append(f"entity_id = ${idx}")
            params.append(_u.UUID(entity_id))
            idx += 1
        if correction_type:
            conditions.append(f"correction_type = ${idx}")
            params.append(correction_type)
            idx += 1
        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        if corrected_by:
            conditions.append(f"corrected_by ILIKE '%' || ${idx} || '%'")
            params.append(corrected_by)
            idx += 1
        if start_date:
            try:
                sd = _dt.fromisoformat(start_date.replace("Z", "+00:00"))
                conditions.append(f"created_at >= ${idx}")
                params.append(sd)
                idx += 1
            except ValueError:
                return json.dumps({"error": "Invalid start_date (ISO 8601 expected)"})
        if end_date:
            try:
                ed = _dt.fromisoformat(end_date.replace("Z", "+00:00"))
                conditions.append(f"created_at < ${idx}")
                params.append(ed)
                idx += 1
            except ValueError:
                return json.dumps({"error": "Invalid end_date (ISO 8601 expected)"})

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, entity_type, entity_id, correction_type, field_name,
                   old_value, new_value, reason, corrected_by, status,
                   affected_count, metadata, created_at, reverted_at, reverted_by
            FROM data_corrections
            {where}
            ORDER BY created_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        corrections = []
        for r in rows:
            corrections.append({
                "id": str(r["id"]),
                "entity_type": r["entity_type"],
                "entity_id": str(r["entity_id"]),
                "correction_type": r["correction_type"],
                "field_name": r["field_name"],
                "old_value": r["old_value"],
                "new_value": r["new_value"],
                "reason": r["reason"],
                "corrected_by": r["corrected_by"],
                "status": r["status"],
                "affected_count": r["affected_count"],
                "metadata": _safe_json(r["metadata"]),
                "created_at": str(r["created_at"]),
                "reverted_at": str(r["reverted_at"]) if r["reverted_at"] else None,
                "reverted_by": r["reverted_by"],
            })

        return json.dumps({
            "success": True,
            "corrections": corrections,
            "count": len(corrections),
        }, default=str)
    except Exception:
        logger.exception("list_data_corrections error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def revert_data_correction(
    correction_id: str,
    reason: Optional[str] = None,
    reverted_by: str = "mcp",
) -> str:
    """
    Revert a previously applied data correction.

    correction_id: UUID of the correction to revert
    reason: Optional explanation for the revert
    reverted_by: Who reverted the correction (default: mcp)
    """
    if not _is_uuid(correction_id):
        return json.dumps({"success": False, "error": "correction_id must be a valid UUID"})

    try:
        pool = get_pool()
        import uuid as _u
        cid = _u.UUID(correction_id)

        row = await pool.fetchrow(
            "SELECT id, status FROM data_corrections WHERE id = $1",
            cid,
        )
        if not row:
            return json.dumps({"success": False, "error": "Correction not found"})
        if row["status"] != "applied":
            return json.dumps({
                "success": False,
                "error": f"Cannot revert correction with status '{row['status']}' (must be 'applied')",
            })

        updated = await pool.fetchrow(
            """
            UPDATE data_corrections
            SET status = 'reverted', reverted_at = NOW(), reverted_by = $2
            WHERE id = $1
            RETURNING id, status, reverted_at
            """,
            cid,
            reverted_by,
        )

        return json.dumps({
            "success": True,
            "id": str(updated["id"]),
            "status": updated["status"],
            "reverted_at": str(updated["reverted_at"]),
        }, default=str)
    except Exception:
        logger.exception("revert_data_correction error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_data_correction(
    correction_id: str,
) -> str:
    """Fetch a single data correction by ID with full details.

    correction_id: UUID of the correction to fetch
    """
    if not _is_uuid(correction_id):
        return json.dumps({"error": "correction_id must be a valid UUID"})

    try:
        import uuid as _u

        pool = get_pool()
        row = await pool.fetchrow(
            """
            SELECT id, entity_type, entity_id, correction_type, field_name,
                   old_value, new_value, reason, corrected_by, status,
                   affected_count, metadata, created_at, reverted_at, reverted_by
            FROM data_corrections
            WHERE id = $1
            """,
            _u.UUID(correction_id),
        )
        if not row:
            return json.dumps({"error": "Correction not found"})

        return json.dumps({
            "success": True,
            "correction": {
                "id": str(row["id"]),
                "entity_type": row["entity_type"],
                "entity_id": str(row["entity_id"]),
                "correction_type": row["correction_type"],
                "field_name": row["field_name"],
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "reason": row["reason"],
                "corrected_by": row["corrected_by"],
                "status": row["status"],
                "affected_count": row["affected_count"],
                "metadata": _safe_json(row["metadata"]),
                "created_at": str(row["created_at"]),
                "reverted_at": str(row["reverted_at"]) if row["reverted_at"] else None,
                "reverted_by": row["reverted_by"],
            },
        }, default=str)
    except Exception:
        logger.exception("get_data_correction error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def get_correction_stats(
    days: int = 30,
) -> str:
    """Aggregate correction activity: counts by type, status, and top correctors.

    days: Window in days (default 30, max 365)
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        days = max(1, min(days, 365))

        row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_corrections,
                COUNT(*) FILTER (WHERE status = 'applied') AS applied,
                COUNT(*) FILTER (WHERE status = 'reverted') AS reverted,
                COUNT(*) FILTER (WHERE status = 'pending_review') AS pending_review,
                COUNT(*) FILTER (WHERE correction_type = 'suppress') AS suppress_count,
                COUNT(*) FILTER (WHERE correction_type = 'flag') AS flag_count,
                COUNT(*) FILTER (WHERE correction_type = 'override_field') AS override_count,
                COUNT(*) FILTER (WHERE correction_type = 'merge_vendor') AS merge_count,
                COUNT(*) FILTER (WHERE correction_type = 'reclassify') AS reclassify_count,
                COUNT(*) FILTER (WHERE correction_type = 'suppress_source') AS suppress_source_count,
                COALESCE(SUM(affected_count), 0) AS total_affected_records,
                MIN(created_at) AS first_correction_at,
                MAX(created_at) AS last_correction_at
            FROM data_corrections
            WHERE created_at > NOW() - ($1 || ' days')::interval
            """,
            str(days),
        )

        correctors = await pool.fetch(
            """
            SELECT corrected_by, COUNT(*) AS count
            FROM data_corrections
            WHERE created_at > NOW() - ($1 || ' days')::interval
            GROUP BY corrected_by
            ORDER BY count DESC
            LIMIT 10
            """,
            str(days),
        )

        by_entity = await pool.fetch(
            """
            SELECT entity_type, COUNT(*) AS count
            FROM data_corrections
            WHERE created_at > NOW() - ($1 || ' days')::interval
            GROUP BY entity_type
            ORDER BY count DESC
            """,
            str(days),
        )

        return json.dumps({
            "success": True,
            "window_days": days,
            "total_corrections": row["total_corrections"],
            "by_status": {
                "applied": row["applied"],
                "reverted": row["reverted"],
                "pending_review": row["pending_review"],
            },
            "by_type": {
                "suppress": row["suppress_count"],
                "flag": row["flag_count"],
                "override_field": row["override_count"],
                "merge_vendor": row["merge_count"],
                "reclassify": row["reclassify_count"],
                "suppress_source": row["suppress_source_count"],
            },
            "by_entity": [{"entity_type": r["entity_type"], "count": r["count"]} for r in by_entity],
            "total_affected_records": row["total_affected_records"],
            "top_correctors": [{"corrected_by": r["corrected_by"], "count": r["count"]} for r in correctors],
            "first_correction_at": str(row["first_correction_at"]) if row["first_correction_at"] else None,
            "last_correction_at": str(row["last_correction_at"]) if row["last_correction_at"] else None,
        }, default=str)
    except Exception:
        logger.exception("get_correction_stats error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_source_correction_impact() -> str:
    """
    Show impact of active source suppressions on review counts.

    Returns each active suppress_source correction with the number of enriched
    reviews that would be excluded by it (globally or vendor-scoped).
    """
    try:
        pool = get_pool()

        rows = await pool.fetch(
            """
            SELECT dc.metadata->>'source_name' AS source_name,
                   dc.field_name AS vendor_scope,
                   dc.reason,
                   dc.created_at,
                   (SELECT COUNT(*) FROM b2b_reviews r
                    WHERE LOWER(r.source) = LOWER(dc.metadata->>'source_name')
                      AND (
                            dc.field_name IS NULL
                            OR EXISTS (
                                SELECT 1
                                FROM b2b_review_vendor_mentions vm
                                WHERE vm.review_id = r.id
                                  AND LOWER(vm.vendor_name) = LOWER(dc.field_name)
                            )
                          )
                      AND r.enrichment_status = 'enriched'
                   ) AS affected_review_count
            FROM data_corrections dc
            WHERE dc.entity_type = 'source'
              AND dc.correction_type = 'suppress_source'
              AND dc.status = 'applied'
            ORDER BY dc.created_at DESC
            """,
        )

        return json.dumps({
            "success": True,
            "basis": _REVIEW_BASIS_RAW_PROVENANCE,
            "active_source_suppressions": [
                {
                    "source_name": r["source_name"],
                    "vendor_scope": r["vendor_scope"],
                    "reason": r["reason"],
                    "affected_review_count": r["affected_review_count"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
            "total": len(rows),
        }, default=str)
    except Exception:
        logger.exception("get_source_correction_impact error")
        return json.dumps({"success": False, "error": "Internal error"})
