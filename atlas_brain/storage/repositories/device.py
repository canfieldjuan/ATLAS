"""
Device repository for managing discovered devices.

Provides CRUD operations for the discovered_devices table.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..models import DiscoveredDevice

logger = logging.getLogger("atlas.storage.device")


class DeviceRepository:
    """
    Repository for discovered device management.

    Handles persistence of auto-discovered network devices.
    """

    async def save_device(self, device: DiscoveredDevice) -> DiscoveredDevice:
        """
        Save or update a discovered device.

        Uses upsert logic - if device_id exists, updates it.

        Args:
            device: The device to save

        Returns:
            The saved device (with ID if newly created)
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            logger.debug("Database not initialized, skipping device save")
            return device

        metadata_json = json.dumps(device.metadata)

        try:
            # Upsert - insert or update on conflict
            row = await pool.fetchrow(
                """
                INSERT INTO discovered_devices (
                    id, device_id, name, device_type, protocol, host, port,
                    discovered_at, last_seen_at, is_active, auto_registered, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
                ON CONFLICT (device_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    device_type = EXCLUDED.device_type,
                    host = EXCLUDED.host,
                    port = EXCLUDED.port,
                    last_seen_at = EXCLUDED.last_seen_at,
                    is_active = EXCLUDED.is_active,
                    auto_registered = EXCLUDED.auto_registered,
                    metadata = EXCLUDED.metadata
                RETURNING id, discovered_at
                """,
                device.id,
                device.device_id,
                device.name,
                device.device_type,
                device.protocol,
                device.host,
                device.port,
                device.discovered_at,
                device.last_seen_at,
                device.is_active,
                device.auto_registered,
                metadata_json,
            )

            if row:
                device.id = row["id"]
                device.discovered_at = row["discovered_at"]
                logger.debug("Saved device: %s", device.device_id)

            return device

        except Exception as e:
            logger.error("Failed to save device %s: %s", device.device_id, e)
            raise

    async def get_device(self, device_id: str) -> Optional[DiscoveredDevice]:
        """Get a device by its device_id."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return None

        row = await pool.fetchrow(
            """
            SELECT id, device_id, name, device_type, protocol, host, port,
                   discovered_at, last_seen_at, is_active, auto_registered, metadata
            FROM discovered_devices
            WHERE device_id = $1
            """,
            device_id,
        )

        if not row:
            return None

        return self._row_to_device(row)

    async def get_device_by_host(self, host: str) -> Optional[DiscoveredDevice]:
        """Get a device by its IP address/hostname."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return None

        row = await pool.fetchrow(
            """
            SELECT id, device_id, name, device_type, protocol, host, port,
                   discovered_at, last_seen_at, is_active, auto_registered, metadata
            FROM discovered_devices
            WHERE host = $1
            ORDER BY last_seen_at DESC
            LIMIT 1
            """,
            host,
        )

        if not row:
            return None

        return self._row_to_device(row)

    async def get_all_devices(self) -> list[DiscoveredDevice]:
        """Get all discovered devices."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return []

        rows = await pool.fetch(
            """
            SELECT id, device_id, name, device_type, protocol, host, port,
                   discovered_at, last_seen_at, is_active, auto_registered, metadata
            FROM discovered_devices
            ORDER BY last_seen_at DESC
            """
        )

        return [self._row_to_device(row) for row in rows]

    async def get_active_devices(self) -> list[DiscoveredDevice]:
        """Get all active (reachable) devices."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return []

        rows = await pool.fetch(
            """
            SELECT id, device_id, name, device_type, protocol, host, port,
                   discovered_at, last_seen_at, is_active, auto_registered, metadata
            FROM discovered_devices
            WHERE is_active = TRUE
            ORDER BY last_seen_at DESC
            """
        )

        return [self._row_to_device(row) for row in rows]

    async def get_devices_by_type(self, device_type: str) -> list[DiscoveredDevice]:
        """Get devices of a specific type."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return []

        rows = await pool.fetch(
            """
            SELECT id, device_id, name, device_type, protocol, host, port,
                   discovered_at, last_seen_at, is_active, auto_registered, metadata
            FROM discovered_devices
            WHERE device_type = $1 AND is_active = TRUE
            ORDER BY last_seen_at DESC
            """,
            device_type,
        )

        return [self._row_to_device(row) for row in rows]

    async def update_last_seen(self, device_id: str) -> None:
        """Update the last_seen_at timestamp for a device."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return

        await pool.execute(
            """
            UPDATE discovered_devices
            SET last_seen_at = $2, is_active = TRUE
            WHERE device_id = $1
            """,
            device_id,
            datetime.utcnow(),
        )
        logger.debug("Updated last_seen for device: %s", device_id)

    async def mark_inactive(self, device_id: str) -> None:
        """Mark a device as inactive (not reachable)."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return

        await pool.execute(
            """
            UPDATE discovered_devices
            SET is_active = FALSE, last_seen_at = $2
            WHERE device_id = $1
            """,
            device_id,
            datetime.utcnow(),
        )
        logger.info("Marked device inactive: %s", device_id)

    async def mark_registered(self, device_id: str) -> None:
        """Mark a device as auto-registered to capability registry."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return

        await pool.execute(
            """
            UPDATE discovered_devices
            SET auto_registered = TRUE
            WHERE device_id = $1
            """,
            device_id,
        )
        logger.debug("Marked device registered: %s", device_id)

    async def delete_device(self, device_id: str) -> bool:
        """Delete a device from the database."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return False

        result = await pool.execute(
            "DELETE FROM discovered_devices WHERE device_id = $1",
            device_id,
        )

        deleted = result and "DELETE 1" in result
        if deleted:
            logger.info("Deleted device: %s", device_id)
        return deleted

    async def mark_all_inactive(self) -> int:
        """
        Mark all devices as inactive.

        Useful before a scan to detect devices that are no longer present.

        Returns:
            Number of devices marked inactive
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            return 0

        result = await pool.execute(
            "UPDATE discovered_devices SET is_active = FALSE WHERE is_active = TRUE"
        )

        count = int(result.split()[-1]) if result else 0
        logger.debug("Marked %d devices inactive", count)
        return count

    def _row_to_device(self, row) -> DiscoveredDevice:
        """Convert a database row to a DiscoveredDevice object."""
        return DiscoveredDevice(
            id=row["id"],
            device_id=row["device_id"],
            name=row["name"],
            device_type=row["device_type"],
            protocol=row["protocol"],
            host=row["host"],
            port=row["port"],
            discovered_at=row["discovered_at"],
            last_seen_at=row["last_seen_at"],
            is_active=row["is_active"],
            auto_registered=row["auto_registered"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


# Global repository instance
_device_repo: Optional[DeviceRepository] = None


def get_device_repo() -> DeviceRepository:
    """Get the global device repository."""
    global _device_repo
    if _device_repo is None:
        _device_repo = DeviceRepository()
    return _device_repo
