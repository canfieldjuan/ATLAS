"""
Security tools for Atlas Brain.

Provides granular tools for camera management, detection queries,
and access control. Tools are designed for LLM tool calling.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from .base import Tool, ToolResult, ToolParameter
from ..config import settings

logger = logging.getLogger("atlas.tools.security")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Camera:
    """Camera information."""
    id: str
    name: str
    location: str
    status: str  # online, offline, recording
    capabilities: list[str] | None = None
    last_motion: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "status": self.status,
            "capabilities": self.capabilities or [],
            "last_motion": self.last_motion.isoformat() if self.last_motion else None,
        }


@dataclass
class Detection:
    """Detection event."""
    camera_id: str
    timestamp: datetime
    detection_type: str  # person, vehicle, animal, motion
    confidence: float
    label: str | None = None
    bbox: tuple | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "detection_type": self.detection_type,
            "confidence": self.confidence,
            "label": self.label,
            "bbox": self.bbox,
        }


@dataclass
class SecurityZone:
    """Security zone information."""
    id: str
    name: str
    status: str  # armed, disarmed
    cameras: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "cameras": self.cameras,
        }


# =============================================================================
# Security Client
# =============================================================================

class SecurityClient:
    """Client for security system operations."""

    def __init__(self):
        self._config = settings.security
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return self._config.video_processing_url

    @property
    def timeout(self) -> float:
        return self._config.timeout

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def resolve_camera_id(self, name_or_id: str) -> str:
        """Resolve camera name/alias to camera ID."""
        name_lower = name_or_id.lower().strip()
        return self._config.camera_aliases.get(name_lower, name_or_id)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # -------------------------------------------------------------------------
    # Camera Operations
    # -------------------------------------------------------------------------

    async def list_cameras(self) -> list[Camera]:
        """Get list of all cameras."""
        try:
            response = await self.client.get(f"{self.base_url}/cameras")
            response.raise_for_status()
            data = response.json()

            cameras = []
            for cam in data.get("cameras", []):
                cameras.append(Camera(
                    id=cam["id"],
                    name=cam.get("name", cam["id"]),
                    location=cam.get("location", "unknown"),
                    status=cam.get("status", "unknown"),
                    capabilities=cam.get("capabilities", []),
                    last_motion=datetime.fromisoformat(cam["last_motion"]) if cam.get("last_motion") else None,
                ))
            return cameras
        except Exception as e:
            logger.error("Failed to list cameras: %s", e)
            return []

    async def get_camera(self, camera_id: str) -> Camera | None:
        """Get specific camera information."""
        camera_id = self.resolve_camera_id(camera_id)
        try:
            response = await self.client.get(f"{self.base_url}/cameras/{camera_id}")
            response.raise_for_status()
            cam = response.json()

            return Camera(
                id=cam["id"],
                name=cam.get("name", cam["id"]),
                location=cam.get("location", "unknown"),
                status=cam.get("status", "unknown"),
                capabilities=cam.get("capabilities", []),
            )
        except Exception as e:
            logger.error("Failed to get camera %s: %s", camera_id, e)
            return None

    async def start_recording(self, camera_id: str) -> bool:
        """Start recording on a camera."""
        camera_id = self.resolve_camera_id(camera_id)
        try:
            response = await self.client.post(
                f"{self.base_url}/cameras/{camera_id}/record",
                json={"action": "start"}
            )
            response.raise_for_status()
            logger.info("Started recording on camera: %s", camera_id)
            return True
        except Exception as e:
            logger.error("Failed to start recording on %s: %s", camera_id, e)
            return False

    async def stop_recording(self, camera_id: str) -> bool:
        """Stop recording on a camera."""
        camera_id = self.resolve_camera_id(camera_id)
        try:
            response = await self.client.post(
                f"{self.base_url}/cameras/{camera_id}/record",
                json={"action": "stop"}
            )
            response.raise_for_status()
            logger.info("Stopped recording on camera: %s", camera_id)
            return True
        except Exception as e:
            logger.error("Failed to stop recording on %s: %s", camera_id, e)
            return False

    async def ptz_control(self, camera_id: str, action: str, value: float = 1.0) -> bool:
        """Control PTZ (pan/tilt/zoom) cameras."""
        camera_id = self.resolve_camera_id(camera_id)
        try:
            payload = {"action": action}
            if value != 1.0:
                payload["value"] = value

            response = await self.client.post(
                f"{self.base_url}/cameras/{camera_id}/ptz",
                json=payload
            )
            response.raise_for_status()
            logger.info("PTZ control on %s: %s", camera_id, action)
            return True
        except Exception as e:
            logger.error("Failed PTZ control on %s: %s", camera_id, e)
            return False

    # -------------------------------------------------------------------------
    # Detection Operations
    # -------------------------------------------------------------------------

    async def get_current_detections(
        self,
        camera_id: str | None = None,
        detection_type: str | None = None,
    ) -> list[Detection]:
        """Get current/recent detections (last 30 seconds)."""
        if camera_id:
            camera_id = self.resolve_camera_id(camera_id)
        try:
            params = {}
            if camera_id:
                params["camera_id"] = camera_id
            if detection_type:
                params["type"] = detection_type

            response = await self.client.get(
                f"{self.base_url}/detections/current",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            detections = []
            for d in data.get("detections", []):
                detections.append(Detection(
                    camera_id=d["camera_id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    detection_type=d["type"],
                    confidence=d.get("confidence", 0.0),
                    label=d.get("label"),
                    bbox=d.get("bbox"),
                ))
            return detections
        except Exception as e:
            logger.error("Failed to get current detections: %s", e)
            return []

    async def query_detections(
        self,
        camera_id: str | None = None,
        detection_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Detection]:
        """Query historical detections."""
        if camera_id:
            camera_id = self.resolve_camera_id(camera_id)
        try:
            params = {"limit": limit}
            if camera_id:
                params["camera_id"] = camera_id
            if detection_type:
                params["type"] = detection_type
            if since:
                params["since"] = since.isoformat()

            response = await self.client.get(
                f"{self.base_url}/detections",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            detections = []
            for d in data.get("detections", []):
                detections.append(Detection(
                    camera_id=d["camera_id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    detection_type=d["type"],
                    confidence=d.get("confidence", 0.0),
                    label=d.get("label"),
                    bbox=d.get("bbox"),
                ))
            return detections
        except Exception as e:
            logger.error("Failed to query detections: %s", e)
            return []

    async def get_motion_events(
        self,
        camera_id: str | None = None,
        hours: int = 1,
    ) -> list[Detection]:
        """Get motion events from the last N hours."""
        if camera_id:
            camera_id = self.resolve_camera_id(camera_id)
        try:
            params = {
                "since": (datetime.now() - timedelta(hours=hours)).isoformat(),
                "type": "motion",
            }
            if camera_id:
                params["camera_id"] = camera_id

            response = await self.client.get(
                f"{self.base_url}/events",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            events = []
            for e in data.get("events", []):
                events.append(Detection(
                    camera_id=e["camera_id"],
                    timestamp=datetime.fromisoformat(e["timestamp"]),
                    detection_type="motion",
                    confidence=e.get("confidence", 1.0),
                ))
            return events
        except Exception as e:
            logger.error("Failed to get motion events: %s", e)
            return []

    # -------------------------------------------------------------------------
    # Access Control Operations
    # -------------------------------------------------------------------------

    async def list_zones(self) -> list[SecurityZone]:
        """Get all security zones."""
        try:
            response = await self.client.get(f"{self.base_url}/security")
            response.raise_for_status()
            data = response.json()

            zones = []
            for z in data.get("zones", []):
                zones.append(SecurityZone(
                    id=z["zone_id"],
                    name=z["name"],
                    status=z["status"],
                    cameras=z.get("cameras", []),
                ))
            return zones
        except Exception as e:
            logger.error("Failed to list zones: %s", e)
            return []

    async def get_zone(self, zone_id: str) -> SecurityZone | None:
        """Get specific zone status."""
        try:
            response = await self.client.get(f"{self.base_url}/security/{zone_id}")
            response.raise_for_status()
            z = response.json()

            return SecurityZone(
                id=z["zone_id"],
                name=z["name"],
                status=z["status"],
                cameras=z.get("cameras", []),
            )
        except Exception as e:
            logger.error("Failed to get zone %s: %s", zone_id, e)
            return None

    async def arm_zone(self, zone_id: str) -> bool:
        """Arm a security zone."""
        try:
            response = await self.client.post(
                f"{self.base_url}/security/arm",
                json={"zone": zone_id},
            )
            response.raise_for_status()
            logger.info("Armed zone: %s", zone_id)
            return True
        except Exception as e:
            logger.error("Failed to arm zone %s: %s", zone_id, e)
            return False

    async def disarm_zone(self, zone_id: str) -> bool:
        """Disarm a security zone."""
        try:
            response = await self.client.post(
                f"{self.base_url}/security/disarm",
                json={"zone": zone_id},
            )
            response.raise_for_status()
            logger.info("Disarmed zone: %s", zone_id)
            return True
        except Exception as e:
            logger.error("Failed to disarm zone %s: %s", zone_id, e)
            return False


# Global client instance
_security_client: SecurityClient | None = None


def get_security_client() -> SecurityClient:
    """Get or create security client."""
    global _security_client
    if _security_client is None:
        _security_client = SecurityClient()
    return _security_client


# =============================================================================
# Camera Tools
# =============================================================================

class ListCamerasTool:
    """List all security cameras."""

    @property
    def name(self) -> str:
        return "list_cameras"

    @property
    def description(self) -> str:
        return "List all security cameras with status and location"

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    @property
    def aliases(self) -> list[str]:
        return ["cameras", "show cameras", "camera list"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        client = get_security_client()
        cameras = await client.list_cameras()

        if not cameras:
            return ToolResult(
                success=True,
                message="No cameras found in the system",
                data={"cameras": [], "total": 0, "online": 0},
            )

        camera_list = [cam.to_dict() for cam in cameras]
        online_count = sum(1 for c in cameras if c.status == "online")

        return ToolResult(
            success=True,
            message=f"Found {len(cameras)} cameras ({online_count} online)",
            data={"cameras": camera_list, "total": len(cameras), "online": online_count},
        )


class GetCameraStatusTool:
    """Get status of a specific camera."""

    @property
    def name(self) -> str:
        return "get_camera_status"

    @property
    def description(self) -> str:
        return "Get status of a specific camera"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Camera name or ID (e.g., 'front door', 'cam_front_door')",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["camera status", "check camera"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        camera_name = params.get("camera_name", "")
        if not camera_name:
            return ToolResult(success=False, message="camera_name is required")

        client = get_security_client()
        camera = await client.get_camera(camera_name)

        if not camera:
            return ToolResult(
                success=False,
                message=f"Camera '{camera_name}' not found",
            )

        return ToolResult(
            success=True,
            message=f"{camera.name} is {camera.status}",
            data={"camera": camera.to_dict()},
        )


class StartRecordingTool:
    """Start recording on a camera."""

    @property
    def name(self) -> str:
        return "start_recording"

    @property
    def description(self) -> str:
        return "Start recording on a camera"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Camera name or ID",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["record", "start recording", "record camera"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        camera_name = params.get("camera_name", "")
        if not camera_name:
            return ToolResult(success=False, message="camera_name is required")

        client = get_security_client()
        success = await client.start_recording(camera_name)

        if success:
            return ToolResult(
                success=True,
                message=f"Started recording on {camera_name}",
            )
        return ToolResult(
            success=False,
            message=f"Failed to start recording on {camera_name}",
        )


class StopRecordingTool:
    """Stop recording on a camera."""

    @property
    def name(self) -> str:
        return "stop_recording"

    @property
    def description(self) -> str:
        return "Stop recording on a camera"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Camera name or ID",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["stop recording", "stop record"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        camera_name = params.get("camera_name", "")
        if not camera_name:
            return ToolResult(success=False, message="camera_name is required")

        client = get_security_client()
        success = await client.stop_recording(camera_name)

        if success:
            return ToolResult(
                success=True,
                message=f"Stopped recording on {camera_name}",
            )
        return ToolResult(
            success=False,
            message=f"Failed to stop recording on {camera_name}",
        )


class PTZControlTool:
    """Control pan/tilt/zoom on a PTZ camera."""

    @property
    def name(self) -> str:
        return "ptz_control"

    @property
    def description(self) -> str:
        return "Control pan/tilt/zoom on a PTZ camera"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Camera name or ID",
                required=True,
            ),
            ToolParameter(
                name="action",
                param_type="string",
                description="PTZ action: pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out, home",
                required=True,
            ),
            ToolParameter(
                name="value",
                param_type="float",
                description="Movement amount (0.1-1.0), default 1.0",
                required=False,
                default=1.0,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["ptz", "pan camera", "tilt camera", "zoom camera"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        camera_name = params.get("camera_name", "")
        action = params.get("action", "")
        value = float(params.get("value", 1.0))

        if not camera_name:
            return ToolResult(success=False, message="camera_name is required")
        if not action:
            return ToolResult(success=False, message="action is required")

        valid_actions = [
            "pan_left", "pan_right", "tilt_up", "tilt_down",
            "zoom_in", "zoom_out", "home"
        ]
        if action not in valid_actions:
            return ToolResult(
                success=False,
                message=f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}",
            )

        client = get_security_client()
        success = await client.ptz_control(camera_name, action, value)

        if success:
            return ToolResult(
                success=True,
                message=f"PTZ {action} on {camera_name}",
            )
        return ToolResult(
            success=False,
            message=f"Failed PTZ control on {camera_name}. Camera may not support PTZ.",
        )


# =============================================================================
# Detection Tools
# =============================================================================

class GetCurrentDetectionsTool:
    """Get current detections on cameras."""

    @property
    def name(self) -> str:
        return "get_current_detections"

    @property
    def description(self) -> str:
        return "Get what's currently detected on cameras (people, vehicles, motion)"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Optional camera name/ID filter",
                required=False,
            ),
            ToolParameter(
                name="detection_type",
                param_type="string",
                description="Optional type filter: person, vehicle, motion",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["detections", "what do you see", "who is there", "check detections"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        camera_name = params.get("camera_name")
        detection_type = params.get("detection_type")

        client = get_security_client()
        detections = await client.get_current_detections(
            camera_id=camera_name,
            detection_type=detection_type,
        )

        if not detections:
            location = f" on {camera_name}" if camera_name else ""
            type_filter = f" ({detection_type})" if detection_type else ""
            return ToolResult(
                success=True,
                message=f"No detections{type_filter}{location}",
                data={"detections": [], "count": 0},
            )

        detection_list = [d.to_dict() for d in detections]

        # Group by camera for summary
        by_camera: dict[str, list[Detection]] = {}
        for d in detections:
            by_camera.setdefault(d.camera_id, []).append(d)

        summary = []
        for cam, dets in by_camera.items():
            types = {d.detection_type for d in dets}
            summary.append(f"{cam}: {len(dets)} detections ({', '.join(types)})")

        return ToolResult(
            success=True,
            message=f"Found {len(detections)} detections: {'; '.join(summary)}",
            data={
                "detections": detection_list,
                "count": len(detections),
                "by_camera": {k: len(v) for k, v in by_camera.items()},
            },
        )


class QueryDetectionsTool:
    """Query historical detections."""

    @property
    def name(self) -> str:
        return "query_detections"

    @property
    def description(self) -> str:
        return "Query historical detections from the last N hours"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="hours",
                param_type="int",
                description="How many hours back to search (default 1)",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Optional camera filter",
                required=False,
            ),
            ToolParameter(
                name="detection_type",
                param_type="string",
                description="Optional type filter",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["detection history", "past detections", "what was detected"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        hours = int(params.get("hours", 1))
        camera_name = params.get("camera_name")
        detection_type = params.get("detection_type")

        client = get_security_client()
        since = datetime.now() - timedelta(hours=hours)

        detections = await client.query_detections(
            camera_id=camera_name,
            detection_type=detection_type,
            since=since,
            limit=100,
        )

        if not detections:
            return ToolResult(
                success=True,
                message=f"No detections in the last {hours} hour(s)",
                data={"detections": [], "count": 0},
            )

        detection_list = [d.to_dict() for d in detections]

        # Stats
        by_type: dict[str, int] = {}
        by_camera: dict[str, int] = {}
        for d in detections:
            by_type[d.detection_type] = by_type.get(d.detection_type, 0) + 1
            by_camera[d.camera_id] = by_camera.get(d.camera_id, 0) + 1

        return ToolResult(
            success=True,
            message=f"Found {len(detections)} detections in last {hours}h",
            data={
                "detections": detection_list,
                "count": len(detections),
                "by_type": by_type,
                "by_camera": by_camera,
            },
        )


class GetPersonAtLocationTool:
    """Check who is at a specific location."""

    @property
    def name(self) -> str:
        return "get_person_at_location"

    @property
    def description(self) -> str:
        return "Check who is at a specific location (front door, backyard, etc)"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="location",
                param_type="string",
                description="Location name (front door, backyard, garage, etc)",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["who is at", "person at", "someone at door"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        location = params.get("location", "")
        if not location:
            return ToolResult(success=False, message="location is required")

        client = get_security_client()
        detections = await client.get_current_detections(
            camera_id=location,
            detection_type="person",
        )

        if not detections:
            return ToolResult(
                success=True,
                message=f"No one detected at {location}",
                data={"people": [], "count": 0},
            )

        people = []
        for d in detections:
            label = d.label or "unknown person"
            if "known:" in (d.label or ""):
                label = d.label.replace("known:", "")
            people.append({
                "label": label,
                "confidence": d.confidence,
                "timestamp": d.timestamp.isoformat(),
            })

        names = [p["label"] for p in people]

        return ToolResult(
            success=True,
            message=f"Detected {len(people)} person(s) at {location}: {', '.join(names)}",
            data={"people": people, "count": len(people)},
        )


class GetMotionEventsTool:
    """Get motion events from the last N hours."""

    @property
    def name(self) -> str:
        return "get_motion_events"

    @property
    def description(self) -> str:
        return "Get motion events from the last N hours"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="hours",
                param_type="int",
                description="Hours to look back (default 1)",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Optional camera filter",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["motion events", "motion history", "any motion"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        hours = int(params.get("hours", 1))
        camera_name = params.get("camera_name")

        client = get_security_client()
        events = await client.get_motion_events(
            camera_id=camera_name,
            hours=hours,
        )

        if not events:
            location = f" on {camera_name}" if camera_name else ""
            return ToolResult(
                success=True,
                message=f"No motion detected{location} in the last {hours} hour(s)",
                data={"events": [], "count": 0},
            )

        # Group by camera
        by_camera: dict[str, list[Detection]] = {}
        for e in events:
            by_camera.setdefault(e.camera_id, []).append(e)

        summary = [f"{cam}: {len(evts)} events" for cam, evts in by_camera.items()]

        return ToolResult(
            success=True,
            message=f"Motion detected in last {hours}h: {', '.join(summary)}",
            data={
                "events": [e.to_dict() for e in events],
                "count": len(events),
                "by_camera": {k: len(v) for k, v in by_camera.items()},
            },
        )


# =============================================================================
# Access Control Tools
# =============================================================================

class ListZonesTool:
    """List all security zones."""

    @property
    def name(self) -> str:
        return "list_zones"

    @property
    def description(self) -> str:
        return "List all security zones with their armed/disarmed status"

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    @property
    def aliases(self) -> list[str]:
        return ["zones", "security zones", "alarm zones"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        client = get_security_client()
        zones = await client.list_zones()

        if not zones:
            return ToolResult(
                success=True,
                message="No security zones configured",
                data={"zones": []},
            )

        zone_list = [z.to_dict() for z in zones]

        armed_zones = [z.name for z in zones if z.status == "armed"]
        disarmed_zones = [z.name for z in zones if z.status == "disarmed"]

        status_msg = []
        if armed_zones:
            status_msg.append(f"Armed: {', '.join(armed_zones)}")
        if disarmed_zones:
            status_msg.append(f"Disarmed: {', '.join(disarmed_zones)}")

        return ToolResult(
            success=True,
            message=f"{len(zones)} zones - {'; '.join(status_msg)}",
            data={"zones": zone_list, "count": len(zones)},
        )


class GetZoneStatusTool:
    """Get status of a specific security zone."""

    @property
    def name(self) -> str:
        return "get_zone_status"

    @property
    def description(self) -> str:
        return "Get the status of a specific security zone"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="zone_name",
                param_type="string",
                description="Zone name (perimeter, interior, garage, etc)",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["zone status", "is zone armed"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        zone_name = params.get("zone_name", "")
        if not zone_name:
            return ToolResult(success=False, message="zone_name is required")

        client = get_security_client()
        zone = await client.get_zone(zone_name.lower())

        if not zone:
            return ToolResult(
                success=False,
                message=f"Zone '{zone_name}' not found",
            )

        return ToolResult(
            success=True,
            message=f"{zone.name} zone is {zone.status}",
            data={"zone": zone.to_dict()},
        )


class ArmZoneTool:
    """Arm a security zone."""

    @property
    def name(self) -> str:
        return "arm_zone"

    @property
    def description(self) -> str:
        return "Arm a security zone to enable alerts"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="zone_name",
                param_type="string",
                description="Zone name or 'all' to arm all zones",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["arm", "arm alarm", "arm security"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        zone_name = params.get("zone_name", "")
        if not zone_name:
            return ToolResult(success=False, message="zone_name is required")

        client = get_security_client()
        zone_id = zone_name.lower()

        success = await client.arm_zone(zone_id)

        if success:
            return ToolResult(
                success=True,
                message=f"Armed {zone_name} zone",
                data={"zone": zone_id, "action": "armed"},
            )
        return ToolResult(
            success=False,
            message=f"Failed to arm {zone_name} zone. Zone may not exist.",
        )


class DisarmZoneTool:
    """Disarm a security zone."""

    @property
    def name(self) -> str:
        return "disarm_zone"

    @property
    def description(self) -> str:
        return "Disarm a security zone to disable alerts"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="zone_name",
                param_type="string",
                description="Zone name or 'all' to disarm all zones",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["disarm", "disarm alarm", "disarm security"]

    @property
    def category(self) -> str:
        return "security"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        zone_name = params.get("zone_name", "")
        if not zone_name:
            return ToolResult(success=False, message="zone_name is required")

        client = get_security_client()
        zone_id = zone_name.lower()

        success = await client.disarm_zone(zone_id)

        if success:
            return ToolResult(
                success=True,
                message=f"Disarmed {zone_name} zone",
                data={"zone": zone_id, "action": "disarmed"},
            )
        return ToolResult(
            success=False,
            message=f"Failed to disarm {zone_name} zone. Zone may not exist.",
        )


# =============================================================================
# Tool Instances for Registration
# =============================================================================

# Camera tools
list_cameras_tool = ListCamerasTool()
get_camera_status_tool = GetCameraStatusTool()
start_recording_tool = StartRecordingTool()
stop_recording_tool = StopRecordingTool()
ptz_control_tool = PTZControlTool()

# Detection tools
get_current_detections_tool = GetCurrentDetectionsTool()
query_detections_tool = QueryDetectionsTool()
get_person_at_location_tool = GetPersonAtLocationTool()
get_motion_events_tool = GetMotionEventsTool()

# Access control tools
list_zones_tool = ListZonesTool()
get_zone_status_tool = GetZoneStatusTool()
arm_zone_tool = ArmZoneTool()
disarm_zone_tool = DisarmZoneTool()
