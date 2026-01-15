"""
Security zone management endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.constants import SecurityZoneStatus
from ..core.protocols import SecurityZone

logger = logging.getLogger("atlas.vision.api.security")

router = APIRouter()

# In-memory security zones (would be persisted in production)
_security_zones: dict[str, SecurityZone] = {
    "perimeter": SecurityZone(
        zone_id="perimeter",
        name="Perimeter",
        status=SecurityZoneStatus.DISARMED,
        cameras=["cam_front_door", "cam_backyard", "cam_driveway"],
    ),
    "interior": SecurityZone(
        zone_id="interior",
        name="Interior",
        status=SecurityZoneStatus.DISARMED,
        cameras=["cam_living_room", "cam_kitchen"],
    ),
    "garage": SecurityZone(
        zone_id="garage",
        name="Garage",
        status=SecurityZoneStatus.DISARMED,
        cameras=["cam_garage"],
    ),
}


class ZoneRequest(BaseModel):
    """Zone arm/disarm request."""
    zone: str


@router.get("")
async def list_zones():
    """List all security zones."""
    return {
        "zones": [zone.to_dict() for zone in _security_zones.values()]
    }


@router.post("/arm")
async def arm_zone(request: ZoneRequest):
    """Arm a security zone."""
    zone_id = request.zone.lower()

    if zone_id == "all":
        for zone in _security_zones.values():
            zone.status = SecurityZoneStatus.ARMED
        return {"success": True, "message": "All zones armed", "zones": list(_security_zones.keys())}

    if zone_id not in _security_zones:
        return {"success": False, "message": f"Unknown zone: {request.zone}"}

    _security_zones[zone_id].status = SecurityZoneStatus.ARMED
    logger.info("Armed zone: %s", zone_id)

    return {"success": True, "message": f"Zone '{zone_id}' armed", "zone": zone_id}


@router.post("/disarm")
async def disarm_zone(request: ZoneRequest):
    """Disarm a security zone."""
    zone_id = request.zone.lower()

    if zone_id == "all":
        for zone in _security_zones.values():
            zone.status = SecurityZoneStatus.DISARMED
        return {"success": True, "message": "All zones disarmed", "zones": list(_security_zones.keys())}

    if zone_id not in _security_zones:
        return {"success": False, "message": f"Unknown zone: {request.zone}"}

    _security_zones[zone_id].status = SecurityZoneStatus.DISARMED
    logger.info("Disarmed zone: %s", zone_id)

    return {"success": True, "message": f"Zone '{zone_id}' disarmed", "zone": zone_id}


@router.get("/{zone_id}")
async def get_zone_status(zone_id: str):
    """Get status of a specific zone."""
    if zone_id not in _security_zones:
        return {"success": False, "message": f"Unknown zone: {zone_id}"}

    return _security_zones[zone_id].to_dict()
