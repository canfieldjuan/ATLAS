"""
Shared correction-aware query helpers for competitive intelligence.

These helpers are intentionally owned by the extracted product so API
and MCP surfaces do not need to import Atlas service modules for basic
suppression and override semantics.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("extracted_competitive_intelligence.services.b2b.corrections")


def suppress_predicate(
    entity_type: str,
    id_expr: str = "id",
    source_expr: str = "source",
    vendor_expr: str = "vendor_name",
) -> str:
    """Return a SQL predicate that excludes active suppress corrections."""
    pred = (
        f"NOT EXISTS (SELECT 1 FROM data_corrections dc"
        f" WHERE dc.entity_type = '{entity_type}' AND dc.entity_id = {id_expr}"
        f" AND dc.correction_type = 'suppress' AND dc.status = 'applied')"
    )
    if entity_type == "review":
        pred += " AND " + source_suppress_predicate(
            source_expr=source_expr,
            vendor_expr=vendor_expr,
        )
    return pred


def source_suppress_predicate(
    source_expr: str = "source",
    vendor_expr: str = "vendor_name",
) -> str:
    """Return a SQL predicate that excludes reviews from suppressed sources."""
    return (
        f"NOT EXISTS (SELECT 1 FROM data_corrections dc_ss"
        f" WHERE dc_ss.entity_type = 'source'"
        f" AND dc_ss.correction_type = 'suppress_source'"
        f" AND dc_ss.status = 'applied'"
        f" AND LOWER(dc_ss.metadata->>'source_name') = LOWER({source_expr})"
        f" AND (dc_ss.field_name IS NULL OR LOWER(dc_ss.field_name) = LOWER({vendor_expr}))"
        f")"
    )


async def apply_field_overrides(
    pool: Any,
    entity_type: str,
    entity_id: str,
    result: dict,
) -> dict:
    """Apply active field overrides to a fetched entity result dict."""
    rows = await pool.fetch(
        """
        SELECT field_name, new_value, old_value, reason, corrected_by
        FROM data_corrections
        WHERE entity_type = $1 AND entity_id = $2::uuid
          AND correction_type = 'override_field' AND status = 'applied'
        """,
        entity_type,
        entity_id,
    )
    if not rows:
        return result

    overrides: list[dict[str, str | None]] = []
    for row in rows:
        field = row["field_name"]
        if field in result:
            result[field] = row["new_value"]
            overrides.append(
                {
                    "field": field,
                    "old_value": row["old_value"],
                    "new_value": row["new_value"],
                    "reason": row["reason"],
                }
            )
    if overrides:
        result["_overrides_applied"] = overrides
    return result
