"""
Vendor merge execution.

When an analyst creates a ``merge_vendor`` correction, this module renames
``vendor_name`` across all 20 affected table/column pairs in a single
transaction, refreshes the ``campaign_funnel_stats`` materialized view,
adds the old name as an alias on the target vendor, and invalidates the
vendor cache.

Tables with UNIQUE constraints that include vendor_name are handled with a
two-step approach: first DELETE source rows that would conflict with existing
target rows, then UPDATE the remaining source rows to the target name.
"""

import logging
from typing import Any

logger = logging.getLogger("atlas.services.b2b.vendor_merge")

# Tables WITHOUT multi-column UNIQUE constraints involving the vendor column.
# These can be updated with a simple SET column = target WHERE column = source.
_SIMPLE_TARGETS: list[tuple[str, str]] = [
    ("b2b_reviews", "vendor_name"),
    ("b2b_campaigns", "vendor_name"),
    ("b2b_keyword_signals", "vendor_name"),
    ("b2b_product_profiles", "vendor_name"),
    ("b2b_vendor_briefings", "vendor_name"),
    ("b2b_change_events", "vendor_name"),
    ("b2b_scrape_targets", "vendor_name"),
]

# Tables WITH UNIQUE constraints involving the vendor column.
# Each entry: (table, vendor_column, list of OTHER columns in the UNIQUE constraint).
# Source rows that share the same natural key as an existing target row are
# deleted (target has fresher data); the rest are renamed.
_UNIQUE_TARGETS: list[tuple[str, str, list[str]]] = [
    ("b2b_churn_signals", "vendor_name", ["product_category"]),
    ("b2b_vendor_pain_points", "vendor_name", ["pain_category"]),
    ("b2b_vendor_use_cases", "vendor_name", ["use_case_name"]),
    ("b2b_vendor_integrations", "vendor_name", ["integration_name"]),
    ("b2b_vendor_buyer_profiles", "vendor_name", ["role_type", "buying_stage"]),
    ("b2b_vendor_snapshots", "vendor_name", ["snapshot_date"]),
    ("b2b_company_signals", "vendor_name", ["company_name"]),
    ("b2b_displacement_edges", "from_vendor", ["to_vendor", "computed_date"]),
    ("b2b_displacement_edges", "to_vendor", ["from_vendor", "computed_date"]),
    ("tracked_vendors", "vendor_name", ["account_id"]),
    ("b2b_alert_baselines", "vendor_name", ["account_id", "metric"]),
    ("b2b_product_profile_snapshots", "vendor_name", ["snapshot_date"]),
]


async def execute_vendor_merge(pool: Any, source_name: str, target_name: str) -> dict:
    """Merge *source_name* into *target_name* across all B2B tables.

    Uses explicit BEGIN/COMMIT via pool.execute() to stay compatible with
    the DatabasePool wrapper. Returns a dict with per-table affected counts
    and a total.
    """
    if not source_name or not target_name:
        return {"error": "source_name and target_name are required", "total_affected": 0}
    if source_name.strip().lower() == target_name.strip().lower():
        return {"error": "source and target are the same vendor", "total_affected": 0}

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
            # Build join predicate: match source rows that would conflict
            # with existing target rows on the same natural key.
            # Use IS NOT DISTINCT FROM to handle NULL columns safely.
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

    # 3. Refresh materialized view that derives from b2b_campaigns.
    # Must be done outside the transaction (REFRESH cannot run inside one).
    try:
        await pool.execute(
            "REFRESH MATERIALIZED VIEW CONCURRENTLY campaign_funnel_stats"
        )
    except Exception:
        logger.warning("Failed to refresh campaign_funnel_stats after merge")

    # 4. Add old name as alias on target vendor in b2b_vendors.
    # Done outside the transaction since add_vendor/add_alias acquire their
    # own pool connections.  This is safe: an alias without a merge is harmless,
    # and the alias is idempotent.
    from ...services.vendor_registry import add_alias, add_vendor
    await add_vendor(target)
    await add_alias(target, source)

    logger.info(
        "Vendor merge complete: '%s' -> '%s', %d updated, %d conflict-deleted",
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
