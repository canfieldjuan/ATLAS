"""
Health and info endpoints.
"""

from fastapi import APIRouter

from ..core.config import settings
from ..core.constants import DeviceType
from ..devices.registry import device_registry

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "atlas-vision",
        "version": "0.1.0",
    }


@router.get("/info")
async def node_info():
    """Node information endpoint."""
    cameras = device_registry.list_by_type(DeviceType.CAMERA)

    return {
        "node_id": settings.discovery.node_name,
        "version": "0.1.0",
        "capabilities": ["camera", "detection", "security"],
        "devices": {
            "cameras": len(cameras),
            "drones": 0,
            "sensors": 0,
        },
        "config": {
            "mqtt_enabled": settings.mqtt.enabled,
            "discovery_enabled": settings.discovery.enabled,
        },
    }
