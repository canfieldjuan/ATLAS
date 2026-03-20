"""Tracked vendor source reconciliation helpers."""

from __future__ import annotations

from typing import Any

MANUAL_SOURCE_TYPE = "manual"
VENDOR_TARGET_SOURCE_TYPE = "vendor_target"
MANUAL_DIRECT_SOURCE_KEY = "direct"
LEGACY_MANUAL_SOURCE_KEY = "legacy_import"


def _normalize_track_mode(value: str) -> str:
    return "own" if str(value or "").strip().lower() == "own" else "competitor"


def _normalize_vendor_name(value: str) -> str:
    return str(value or "").strip()


async def reconcile_tracked_vendor(pool, account_id: str, vendor_name: str) -> bool:
    normalized_vendor = _normalize_vendor_name(vendor_name)
    rows = await pool.fetch(
        """
        SELECT track_mode
        FROM tracked_vendor_sources
        WHERE account_id = $1
          AND vendor_name = $2
        """,
        account_id,
        normalized_vendor,
    )
    if not rows:
        await pool.execute(
            "DELETE FROM tracked_vendors WHERE account_id = $1 AND vendor_name = $2",
            account_id,
            normalized_vendor,
        )
        return False

    desired_track_mode = "own" if any(
        _normalize_track_mode(row.get("track_mode")) == "own" for row in rows
    ) else "competitor"
    await pool.execute(
        """
        INSERT INTO tracked_vendors (account_id, vendor_name, track_mode, label)
        VALUES ($1, $2, $3, NULL)
        ON CONFLICT (account_id, vendor_name) DO UPDATE
        SET track_mode = EXCLUDED.track_mode
        """,
        account_id,
        normalized_vendor,
        desired_track_mode,
    )
    return True


async def upsert_tracked_vendor_source(
    pool,
    account_id: str,
    vendor_name: str,
    *,
    source_type: str,
    source_key: str,
    track_mode: str,
) -> bool:
    normalized_vendor = _normalize_vendor_name(vendor_name)
    await pool.execute(
        """
        INSERT INTO tracked_vendor_sources
            (account_id, vendor_name, source_type, source_key, track_mode)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (account_id, vendor_name, source_type, source_key) DO UPDATE
        SET track_mode = EXCLUDED.track_mode,
            updated_at = NOW()
        """,
        account_id,
        normalized_vendor,
        source_type,
        source_key,
        _normalize_track_mode(track_mode),
    )
    return await reconcile_tracked_vendor(pool, account_id, normalized_vendor)


async def delete_tracked_vendor_source(
    pool,
    account_id: str,
    vendor_name: str,
    *,
    source_type: str,
    source_key: str,
) -> bool:
    normalized_vendor = _normalize_vendor_name(vendor_name)
    await pool.execute(
        """
        DELETE FROM tracked_vendor_sources
        WHERE account_id = $1
          AND vendor_name = $2
          AND source_type = $3
          AND source_key = $4
        """,
        account_id,
        normalized_vendor,
        source_type,
        source_key,
    )
    return await reconcile_tracked_vendor(pool, account_id, normalized_vendor)


async def purge_tracked_vendor_sources(
    pool,
    account_id: str,
    vendor_name: str,
) -> dict[str, Any]:
    normalized_vendor = _normalize_vendor_name(vendor_name)
    rows = await pool.fetch(
        """
        SELECT source_type, source_key
        FROM tracked_vendor_sources
        WHERE account_id = $1
          AND vendor_name = $2
        ORDER BY source_type, source_key
        """,
        account_id,
        normalized_vendor,
    )
    await pool.execute(
        """
        DELETE FROM tracked_vendor_sources
        WHERE account_id = $1
          AND vendor_name = $2
        """,
        account_id,
        normalized_vendor,
    )
    still_tracked = await reconcile_tracked_vendor(pool, account_id, normalized_vendor)
    return {
        "removed_sources": [
            {
                "source_type": str(row.get("source_type") or ""),
                "source_key": str(row.get("source_key") or ""),
            }
            for row in rows
        ],
        "still_tracked": still_tracked,
    }


async def replace_vendor_target_sources(
    pool,
    account_id: str,
    target_id: str,
    tracked_vendors: list[dict[str, str]],
) -> dict[str, Any]:
    existing_rows = await pool.fetch(
        """
        SELECT vendor_name
        FROM tracked_vendor_sources
        WHERE account_id = $1
          AND source_type = $3
          AND source_key = $2
        """,
        account_id,
        target_id,
        VENDOR_TARGET_SOURCE_TYPE,
    )
    existing_vendors = {
        str(row.get("vendor_name") or "").strip()
        for row in existing_rows
        if str(row.get("vendor_name") or "").strip()
    }
    await pool.execute(
        """
        DELETE FROM tracked_vendor_sources
        WHERE account_id = $1
          AND source_type = $3
          AND source_key = $2
        """,
        account_id,
        target_id,
        VENDOR_TARGET_SOURCE_TYPE,
    )

    synced: list[dict[str, str]] = []
    new_vendor_names: set[str] = set()
    for item in tracked_vendors:
        vendor_name = str(item.get("vendor_name") or "").strip()
        if not vendor_name or vendor_name in new_vendor_names:
            continue
        new_vendor_names.add(vendor_name)
        await upsert_tracked_vendor_source(
            pool,
            account_id,
            vendor_name,
            source_type=VENDOR_TARGET_SOURCE_TYPE,
            source_key=target_id,
            track_mode=str(item.get("track_mode") or "competitor"),
        )
        synced.append(
            {
                "vendor_name": vendor_name,
                "track_mode": _normalize_track_mode(str(item.get("track_mode") or "competitor")),
            }
        )

    for vendor_name in sorted(existing_vendors - new_vendor_names):
        await reconcile_tracked_vendor(pool, account_id, vendor_name)

    return {"synced_vendors": synced}


async def delete_vendor_target_sources(pool, account_id: str, target_id: str) -> dict[str, Any]:
    rows = await pool.fetch(
        """
        SELECT vendor_name
        FROM tracked_vendor_sources
        WHERE account_id = $1
          AND source_type = $3
          AND source_key = $2
        """,
        account_id,
        target_id,
        VENDOR_TARGET_SOURCE_TYPE,
    )
    removed_vendors = [
        str(row.get("vendor_name") or "").strip()
        for row in rows
        if str(row.get("vendor_name") or "").strip()
    ]
    await pool.execute(
        """
        DELETE FROM tracked_vendor_sources
        WHERE account_id = $1
          AND source_type = $3
          AND source_key = $2
        """,
        account_id,
        target_id,
        VENDOR_TARGET_SOURCE_TYPE,
    )
    for vendor_name in removed_vendors:
        await reconcile_tracked_vendor(pool, account_id, vendor_name)
    return {"removed_vendors": removed_vendors}


async def delete_vendor_target_sources_for_all_accounts(
    pool,
    target_id: str,
) -> dict[str, Any]:
    rows = await pool.fetch(
        """
        SELECT account_id, vendor_name
        FROM tracked_vendor_sources
        WHERE source_type = $2
          AND source_key = $1
        ORDER BY account_id, vendor_name
        """,
        target_id,
        VENDOR_TARGET_SOURCE_TYPE,
    )
    await pool.execute(
        """
        DELETE FROM tracked_vendor_sources
        WHERE source_type = $2
          AND source_key = $1
        """,
        target_id,
        VENDOR_TARGET_SOURCE_TYPE,
    )
    seen_pairs: set[tuple[str, str]] = set()
    removed_sources: list[dict[str, str]] = []
    for row in rows:
        account_id = str(row.get("account_id") or "").strip()
        vendor_name = _normalize_vendor_name(str(row.get("vendor_name") or ""))
        if not account_id or not vendor_name:
            continue
        pair = (account_id, vendor_name)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        await reconcile_tracked_vendor(pool, account_id, vendor_name)
        removed_sources.append(
            {
                "account_id": account_id,
                "vendor_name": vendor_name,
            }
        )
    return {"removed_sources": removed_sources}
