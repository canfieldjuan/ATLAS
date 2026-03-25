"""
REST API for network security monitor telemetry.

Provides runtime visibility into packet pipeline and detector health.
"""

import logging
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..config import settings

logger = logging.getLogger("atlas.api.security")

router = APIRouter(prefix="/security", tags=["security"])


def _security_enabled_config() -> dict[str, Any]:
    return {
        "network_monitor": settings.security.network_monitor_enabled,
        "network_ids": settings.security.network_ids_enabled,
        "asset_tracking": settings.security.asset_tracking_enabled,
        "arp_monitor": settings.security.arp_monitor_enabled,
        "traffic_analysis": settings.security.traffic_analysis_enabled,
        "pcap": settings.security.pcap_enabled,
    }


def _security_runtime_config() -> dict[str, Any]:
    return {
        "network_interface": settings.security.network_interface,
        "protocols_to_monitor": settings.security.protocols_to_monitor,
        "pcap_directory": settings.security.pcap_directory,
        "pcap_max_size_mb": settings.security.pcap_max_size_mb,
        "asset_stale_after_seconds": settings.security.asset_stale_after_seconds,
        "asset_max_tracked": settings.security.asset_max_tracked,
    }


def _get_security_monitor_or_none():
    try:
        from ..security import get_security_monitor
        return get_security_monitor()
    except ModuleNotFoundError as exc:
        if exc.name == "scapy":
            logger.warning("Security monitor unavailable: missing optional dependency 'scapy'")
            return None
        raise


class AssetObservationRequest(BaseModel):
    """Observation payload for security asset trackers."""

    asset_type: str = Field(..., pattern="^(drone|vehicle|sensor)$")
    identifier: str = Field(..., min_length=1, max_length=128)
    metadata: dict = Field(default_factory=dict)


@router.get("/status")
async def get_security_status():
    """Get live security monitor runtime and detector telemetry."""
    monitor = _get_security_monitor_or_none()
    if monitor is None:
        return {
            "enabled": _security_enabled_config(),
            "config": _security_runtime_config(),
            "monitor": {
                "is_running": False,
                "runtime": {},
                "assets": {},
                "unavailable_reason": "scapy_not_installed",
            },
            "detectors": {
                "port_scan": {"active_sources": 0, "sources": {}},
                "arp": {"tracked_ips": 0, "tracked_changes": 0},
                "traffic": {},
            },
        }

    port_detector = monitor.get_port_scan_detector()
    arp_monitor = monitor.get_arp_monitor()
    traffic_analyzer = monitor.get_traffic_analyzer()

    port_stats = port_detector.get_stats() if port_detector else {}
    arp_table = arp_monitor.get_arp_table() if arp_monitor else {}
    arp_history = arp_monitor.get_change_history() if arp_monitor else {}
    traffic_metrics = traffic_analyzer.get_metrics() if traffic_analyzer else {}

    return {
        "enabled": _security_enabled_config(),
        "config": _security_runtime_config(),
        "monitor": {
            "is_running": monitor.is_running,
            "runtime": monitor.get_runtime_stats(),
            "assets": monitor.get_asset_summary(),
        },
        "detectors": {
            "port_scan": {
                "active_sources": len(port_stats),
                "sources": port_stats,
            },
            "arp": {
                "tracked_ips": len(arp_table),
                "tracked_changes": sum(len(changes) for changes in arp_history.values()),
            },
            "traffic": traffic_metrics,
        },
    }


@router.get("/assets")
async def list_security_assets(
    asset_type: str | None = Query(default=None, pattern="^(drone|vehicle|sensor)$"),
    limit: int = Query(default=200, ge=1, le=5000),
):
    """List tracked assets from in-memory asset trackers."""
    monitor = _get_security_monitor_or_none()
    if monitor is None:
        return {
            "count": 0,
            "total": 0,
            "assets": [],
            "unavailable_reason": "scapy_not_installed",
        }

    assets = monitor.list_assets(asset_type=asset_type)
    return {
        "count": min(len(assets), limit),
        "total": len(assets),
        "assets": assets[:limit],
    }


@router.post("/assets/observe")
async def observe_security_asset(req: AssetObservationRequest):
    """Record an observed security asset into active asset trackers."""
    monitor = _get_security_monitor_or_none()
    if monitor is None:
        return {
            "recorded": False,
            "reason": "scapy_not_installed",
            "asset_type": req.asset_type,
            "identifier": req.identifier,
        }

    asset = monitor.observe_asset(
        asset_type=req.asset_type,
        identifier=req.identifier,
        metadata=req.metadata,
    )
    if asset is None:
        return {
            "recorded": False,
            "reason": "asset tracker not enabled",
            "asset_type": req.asset_type,
            "identifier": req.identifier,
        }

    await _persist_asset_observation(
        asset_type=req.asset_type,
        identifier=req.identifier,
        metadata=req.metadata,
    )

    return {
        "recorded": True,
        "asset": asset,
    }


@router.get("/assets/persisted")
async def list_persisted_security_assets(
    asset_type: str | None = Query(default=None, pattern="^(drone|vehicle|sensor)$"),
    status: str | None = Query(default=None, pattern="^(active|stale)$"),
    limit: int = Query(default=200, ge=1, le=5000),
):
    """List persisted security assets from the database registry."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"count": 0, "total": 0, "assets": []}

    conditions = []
    params: list[Any] = []
    idx = 1
    if asset_type:
        conditions.append(f"asset_type = ${idx}")
        params.append(asset_type)
        idx += 1
    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    total_row = await pool.fetchrow(
        f"SELECT count(*) AS cnt FROM security_assets {where_clause}",
        *params,
    )
    rows = await pool.fetch(
        f"""
        SELECT asset_type, identifier, name, status, first_seen, last_seen, metadata
        FROM security_assets
        {where_clause}
        ORDER BY last_seen DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )
    assets = [
        {
            "asset_type": row["asset_type"],
            "identifier": row["identifier"],
            "name": row["name"],
            "status": row["status"],
            "first_seen": row["first_seen"].isoformat() if row["first_seen"] else None,
            "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
            "metadata": row["metadata"] or {},
        }
        for row in rows
    ]
    return {
        "count": len(assets),
        "total": int(total_row["cnt"] if total_row else 0),
        "assets": assets,
    }


@router.get("/assets/telemetry")
async def get_security_asset_telemetry(
    asset_type: str | None = Query(default=None, pattern="^(drone|vehicle|sensor)$"),
    identifier: str | None = Query(default=None, min_length=1, max_length=128),
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=500, ge=1, le=5000),
):
    """Get persisted asset telemetry history from the database."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"hours": hours, "count": 0, "telemetry": []}

    conditions = [f"observed_at > NOW() - (${1}::int || ' hours')::interval"]
    params: list[Any] = [hours]
    idx = 2
    if asset_type:
        conditions.append(f"asset_type = ${idx}")
        params.append(asset_type)
        idx += 1
    if identifier:
        conditions.append(f"identifier = ${idx}")
        params.append(identifier)
        idx += 1
    where_clause = f"WHERE {' AND '.join(conditions)}"

    rows = await pool.fetch(
        f"""
        SELECT asset_type, identifier, observed_at, metadata
        FROM security_asset_telemetry
        {where_clause}
        ORDER BY observed_at DESC
        LIMIT ${idx}
        """,
        *params,
        limit,
    )
    telemetry = [
        {
            "asset_type": row["asset_type"],
            "identifier": row["identifier"],
            "observed_at": row["observed_at"].isoformat() if row["observed_at"] else None,
            "metadata": row["metadata"] or {},
        }
        for row in rows
    ]
    return {"hours": hours, "count": len(telemetry), "telemetry": telemetry}


async def _persist_asset_observation(
    asset_type: str,
    identifier: str,
    metadata: dict,
) -> None:
    """Persist asset registry and telemetry data when database is available."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return

    await pool.execute(
        """
        INSERT INTO security_assets
            (asset_type, identifier, status, metadata, first_seen, last_seen)
        VALUES
            ($1, $2, 'active', $3::jsonb, NOW(), NOW())
        ON CONFLICT (asset_type, identifier)
        DO UPDATE SET
            status = 'active',
            metadata = security_assets.metadata || EXCLUDED.metadata,
            last_seen = NOW()
        """,
        asset_type,
        identifier,
        metadata,
    )
    await pool.execute(
        """
        INSERT INTO security_asset_telemetry
            (asset_type, identifier, metadata, observed_at)
        VALUES
            ($1, $2, $3::jsonb, NOW())
        """,
        asset_type,
        identifier,
        metadata,
    )


@router.get("/threats/summary")
async def get_security_threat_summary(hours: int = Query(default=24, ge=1, le=168)):
    """Get threat summary from persisted security threat logs."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {
            "hours": hours,
            "total": 0,
            "by_type": {},
            "by_severity": {},
        }

    total_row = await pool.fetchrow(
        """
        SELECT count(*) AS cnt
        FROM security_threats
        WHERE timestamp > NOW() - ($1::int || ' hours')::interval
        """,
        hours,
    )
    type_rows = await pool.fetch(
        """
        SELECT threat_type, count(*) AS cnt
        FROM security_threats
        WHERE timestamp > NOW() - ($1::int || ' hours')::interval
        GROUP BY threat_type
        ORDER BY count(*) DESC
        """,
        hours,
    )
    severity_rows = await pool.fetch(
        """
        SELECT severity, count(*) AS cnt
        FROM security_threats
        WHERE timestamp > NOW() - ($1::int || ' hours')::interval
        GROUP BY severity
        ORDER BY count(*) DESC
        """,
        hours,
    )

    return {
        "hours": hours,
        "total": int(total_row["cnt"] if total_row else 0),
        "by_type": {row["threat_type"]: int(row["cnt"]) for row in type_rows},
        "by_severity": {row["severity"]: int(row["cnt"]) for row in severity_rows},
    }
