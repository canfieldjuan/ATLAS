"""
Consumer brand merge execution.

When an analyst creates a ``merge_brand`` correction, this module renames
the brand across all affected consumer tables in a single transaction,
refreshes the ``mv_brand_summary`` materialized view, adds the old name
as an alias on the target brand in consumer_brand_registry, and invalidates
the brand resolution cache.

Tables with UNIQUE constraints that include the brand column are handled
with a two-step approach: first DELETE source rows that would conflict
with existing target rows, then UPDATE the remaining source rows.
"""

import logging
from typing import Any

logger = logging.getLogger("atlas.services.brand_merge")

# Tables WITHOUT multi-column UNIQUE constraints involving the brand column.
_SIMPLE_TARGETS: list[tuple[str, str]] = [
    ("product_change_events", "brand"),
    ("product_metadata", "brand"),
]

# Tables WITH UNIQUE constraints involving the brand column.
# Each entry: (table, brand_column, list of OTHER columns in the UNIQUE).
_UNIQUE_TARGETS: list[tuple[str, str, list[str]]] = [
    ("brand_intelligence", "brand", ["source"]),
    ("brand_intelligence_snapshots", "brand", ["snapshot_date"]),
    ("product_displacement_edges", "from_brand", ["to_brand", "direction", "computed_date"]),
    ("product_displacement_edges", "to_brand", ["from_brand", "direction", "computed_date"]),
]


async def execute_brand_merge(pool: Any, source_name: str, target_name: str) -> dict:
    """Merge *source_name* into *target_name* across all consumer brand tables.

    Uses explicit BEGIN/COMMIT via pool.execute() to stay compatible with
    the DatabasePool wrapper (which does not support async-with on acquire).
    Returns a dict with per-table affected counts and a total.
    """
    if not source_name or not target_name:
        return {"error": "source_name and target_name are required", "total_affected": 0}
    if source_name.strip().lower() == target_name.strip().lower():
        return {"error": "source and target are the same brand", "total_affected": 0}

    source = source_name.strip()
    target = target_name.strip()

    table_counts: dict[str, int] = {}
    deleted_conflicts: dict[str, int] = {}
    total_updated = 0
    total_deleted = 0

    try:
        await pool.execute("BEGIN")

        # 1. Simple tables (no UNIQUE conflicts possible)
        for table, column in _SIMPLE_TARGETS:
            result = await pool.execute(
                f"UPDATE {table} SET {column} = $2 WHERE {column} = $1",
                source, target,
            )
            count = _parse_command_count(result)
            table_counts[f"{table}.{column}"] = count
            total_updated += count

        # 2. Tables with UNIQUE constraints -- delete conflicts first
        for table, column, key_cols in _UNIQUE_TARGETS:
            join_parts = []
            for kc in key_cols:
                join_parts.append(f"src.{kc} IS NOT DISTINCT FROM tgt.{kc}")
            join_pred = " AND ".join(join_parts)

            # Delete source rows that conflict with existing target rows
            del_sql = (
                f"DELETE FROM {table} src"
                f" USING {table} tgt"
                f" WHERE src.{column} = $1"
                f"   AND tgt.{column} = $2"
                f"   AND {join_pred}"
            )
            del_result = await pool.execute(del_sql, source, target)
            del_count = _parse_command_count(del_result)
            if del_count > 0:
                deleted_conflicts[f"{table}.{column}"] = del_count
                total_deleted += del_count

            # Now update remaining source rows (no conflicts left)
            upd_result = await pool.execute(
                f"UPDATE {table} SET {column} = $2 WHERE {column} = $1",
                source, target,
            )
            upd_count = _parse_command_count(upd_result)
            table_counts[f"{table}.{column}"] = upd_count
            total_updated += upd_count

        await pool.execute("COMMIT")
    except Exception:
        try:
            await pool.execute("ROLLBACK")
        except Exception:
            pass
        raise

    # 3. Refresh materialized view (outside transaction).
    try:
        await pool.execute(
            "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_brand_summary"
        )
    except Exception:
        logger.warning("Failed to refresh mv_brand_summary after merge")

    # 4. Add old name as alias on target brand in consumer_brand_registry.
    from .brand_registry import add_alias, add_brand, invalidate_cache
    await add_brand(target)
    await add_alias(target, source)
    invalidate_cache()

    logger.info(
        "Brand merge complete: '%s' -> '%s', %d updated, %d conflict-deleted",
        source, target, total_updated, total_deleted,
    )

    return {
        "source": source,
        "target": target,
        "total_affected": total_updated + total_deleted,
        "rows_updated": total_updated,
        "rows_deleted_conflicts": total_deleted,
        "table_counts": table_counts,
        "deleted_conflicts": deleted_conflicts,
    }


def _parse_command_count(result: str) -> int:
    """Extract row count from asyncpg command result like 'UPDATE 5'."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return 0
