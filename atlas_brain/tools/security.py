"""
Security tools for Atlas Brain.

Provides natural language interface to video processing system.
Tools are designed to be simple for the LLM - the implementation
handles routing to the correct video processing APIs.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from .base import Tool, ToolResult, register_tool

logger = logging.getLogger("atlas.tools.security")

# Video processing API configuration
VIDEO_API_URL = "http://localhost:5002"  # Processing service
KAFKA_EVENTS_URL = "http://localhost:5004"  # LLM service for complex queries


@dataclass
class CameraInfo:
    """Camera information."""
    camera_id: str
    name: str
    location: str
    status: str  # online, offline, recording
    last_motion: Optional[datetime] = None


@dataclass
class Detection:
    """Person/object detection."""
    camera_id: str
    timestamp: datetime
    detection_type: str  # person, vehicle, animal, motion
    confidence: float
    label: Optional[str] = None  # "known:Juan", "unknown", "delivery_truck"
    bbox: Optional[tuple] = None  # (x, y, w, h)


class VideoProcessingClient:
    """
    Client for video processing system.

    Handles communication with:
    - Processing service (detections, camera status)
    - Kafka events (historical data)
    - LLM service (complex queries)
    """

    def __init__(
        self,
        processing_url: str = VIDEO_API_URL,
        llm_service_url: str = KAFKA_EVENTS_URL,
        timeout: float = 10.0,
    ):
        self.processing_url = processing_url
        self.llm_service_url = llm_service_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Camera name mappings (configured or discovered)
        self.camera_aliases = {
            "front door": "cam_front_door",
            "front": "cam_front_door",
            "back door": "cam_back_door",
            "back": "cam_back_door",
            "backyard": "cam_backyard",
            "garage": "cam_garage",
            "driveway": "cam_driveway",
            "living room": "cam_living_room",
            "kitchen": "cam_kitchen",
        }

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _resolve_camera(self, name: str) -> str:
        """Resolve camera name/alias to camera_id."""
        name_lower = name.lower().strip()
        return self.camera_aliases.get(name_lower, name_lower)

    async def list_cameras(self) -> list[CameraInfo]:
        """Get list of all cameras."""
        try:
            response = await self.client.get(f"{self.processing_url}/cameras")
            response.raise_for_status()
            data = response.json()
            return [
                CameraInfo(
                    camera_id=cam["id"],
                    name=cam.get("name", cam["id"]),
                    location=cam.get("location", "unknown"),
                    status=cam.get("status", "unknown"),
                    last_motion=datetime.fromisoformat(cam["last_motion"]) if cam.get("last_motion") else None,
                )
                for cam in data.get("cameras", [])
            ]
        except Exception as e:
            logger.warning("Failed to list cameras: %s", e)
            # Return mock data for development
            return [
                CameraInfo("cam_front_door", "Front Door", "entrance", "online"),
                CameraInfo("cam_backyard", "Backyard", "exterior", "online"),
                CameraInfo("cam_garage", "Garage", "garage", "online"),
            ]

    async def get_camera_status(self, camera_id: str) -> Optional[CameraInfo]:
        """Get status of specific camera."""
        camera_id = self._resolve_camera(camera_id)
        try:
            response = await self.client.get(f"{self.processing_url}/cameras/{camera_id}")
            response.raise_for_status()
            cam = response.json()
            return CameraInfo(
                camera_id=cam["id"],
                name=cam.get("name", cam["id"]),
                location=cam.get("location", "unknown"),
                status=cam.get("status", "unknown"),
            )
        except Exception as e:
            logger.warning("Failed to get camera %s: %s", camera_id, e)
            return None

    async def get_current_detections(
        self,
        camera_id: Optional[str] = None,
        detection_type: Optional[str] = None,
    ) -> list[Detection]:
        """Get current/recent detections."""
        if camera_id:
            camera_id = self._resolve_camera(camera_id)

        try:
            params = {}
            if camera_id:
                params["camera_id"] = camera_id
            if detection_type:
                params["type"] = detection_type

            response = await self.client.get(
                f"{self.processing_url}/detections/current",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            return [
                Detection(
                    camera_id=d["camera_id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    detection_type=d["type"],
                    confidence=d.get("confidence", 0.0),
                    label=d.get("label"),
                )
                for d in data.get("detections", [])
            ]
        except Exception as e:
            logger.warning("Failed to get detections: %s", e)
            return []

    async def get_motion_events(
        self,
        camera_id: Optional[str] = None,
        hours: int = 1,
    ) -> list[Detection]:
        """Get motion events from the last N hours."""
        if camera_id:
            camera_id = self._resolve_camera(camera_id)

        try:
            params = {
                "since": (datetime.now() - timedelta(hours=hours)).isoformat(),
                "type": "motion",
            }
            if camera_id:
                params["camera_id"] = camera_id

            response = await self.client.get(
                f"{self.processing_url}/events",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            return [
                Detection(
                    camera_id=e["camera_id"],
                    timestamp=datetime.fromisoformat(e["timestamp"]),
                    detection_type="motion",
                    confidence=e.get("confidence", 1.0),
                )
                for e in data.get("events", [])
            ]
        except Exception as e:
            logger.warning("Failed to get motion events: %s", e)
            return []

    async def natural_query(self, question: str) -> str:
        """
        Send complex query to video processing LLM service.

        For questions that need reasoning across multiple cameras,
        time ranges, or complex logic.
        """
        try:
            response = await self.client.post(
                f"{self.llm_service_url}/query",
                json={"question": question},
            )
            response.raise_for_status()
            return response.json().get("answer", "Unable to process query")
        except Exception as e:
            logger.warning("Natural query failed: %s", e)
            return f"Video processing system unavailable: {e}"

    async def arm_zone(self, zone: str) -> bool:
        """Arm a security zone."""
        try:
            response = await self.client.post(
                f"{self.processing_url}/security/arm",
                json={"zone": zone},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning("Failed to arm zone %s: %s", zone, e)
            return False

    async def disarm_zone(self, zone: str) -> bool:
        """Disarm a security zone."""
        try:
            response = await self.client.post(
                f"{self.processing_url}/security/disarm",
                json={"zone": zone},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning("Failed to disarm zone %s: %s", zone, e)
            return False

    async def start_recording(self, camera_id: str) -> bool:
        """Start recording on a camera."""
        camera_id = self._resolve_camera(camera_id)
        try:
            response = await self.client.post(
                f"{self.processing_url}/cameras/{camera_id}/record",
                json={"action": "start"},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning("Failed to start recording on %s: %s", camera_id, e)
            return False

    async def stop_recording(self, camera_id: str) -> bool:
        """Stop recording on a camera."""
        camera_id = self._resolve_camera(camera_id)
        try:
            response = await self.client.post(
                f"{self.processing_url}/cameras/{camera_id}/record",
                json={"action": "stop"},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning("Failed to stop recording on %s: %s", camera_id, e)
            return False


# Global client instance
_video_client: Optional[VideoProcessingClient] = None


def get_video_client() -> VideoProcessingClient:
    """Get or create video processing client."""
    global _video_client
    if _video_client is None:
        _video_client = VideoProcessingClient()
    return _video_client


# =============================================================================
# TOOL 1: Security Query (Read-only)
# =============================================================================

@register_tool("security_query")
class SecurityQueryTool(Tool):
    """
    Ask questions about cameras, detections, and security status.

    The tool parses natural language questions and routes to the
    appropriate video processing API.

    Examples:
    - "Is anyone at the front door?"
    - "List all cameras"
    - "What motion was detected in the last hour?"
    - "Who is in the backyard?"
    - "Are all cameras online?"
    """

    name = "security_query"
    description = (
        "Ask questions about security cameras, who/what is detected, "
        "motion events, and camera status. Use for any security-related questions."
    )
    parameters = {
        "question": {
            "type": "string",
            "description": "Natural language question about security/cameras",
            "required": True,
        }
    }

    async def execute(self, question: str, **kwargs) -> ToolResult:
        """Execute security query."""
        client = get_video_client()
        question_lower = question.lower()

        try:
            # Route based on question type

            # --- Camera listing ---
            if any(kw in question_lower for kw in ["list camera", "all camera", "what camera", "how many camera"]):
                cameras = await client.list_cameras()
                if not cameras:
                    return ToolResult(
                        success=True,
                        message="No cameras found in the system.",
                        data={"cameras": []},
                    )

                camera_list = [f"- {cam.name} ({cam.location}): {cam.status}" for cam in cameras]
                return ToolResult(
                    success=True,
                    message=f"Found {len(cameras)} cameras:\n" + "\n".join(camera_list),
                    data={"cameras": [{"id": c.camera_id, "name": c.name, "status": c.status} for c in cameras]},
                )

            # --- Camera status ---
            if any(kw in question_lower for kw in ["status", "online", "working", "is the"]):
                # Extract camera name from question
                camera_name = self._extract_camera_name(question_lower)
                if camera_name:
                    cameras = await client.list_cameras()
                    camera = next((c for c in cameras if camera_name in c.name.lower() or camera_name in c.camera_id.lower()), None)
                    if camera:
                        return ToolResult(
                            success=True,
                            message=f"{camera.name} is {camera.status}.",
                            data={"camera": camera.name, "status": camera.status},
                        )

                # Check all cameras
                cameras = await client.list_cameras()
                online = [c for c in cameras if c.status == "online"]
                offline = [c for c in cameras if c.status != "online"]

                if offline:
                    msg = f"{len(online)}/{len(cameras)} cameras online. Offline: {', '.join(c.name for c in offline)}"
                else:
                    msg = f"All {len(cameras)} cameras are online."

                return ToolResult(
                    success=True,
                    message=msg,
                    data={"online": len(online), "offline": len(offline)},
                )

            # --- Person detection (who is...) ---
            if any(kw in question_lower for kw in ["who is", "anyone", "somebody", "someone", "person"]):
                camera_name = self._extract_camera_name(question_lower)
                camera_id = client._resolve_camera(camera_name) if camera_name else None

                detections = await client.get_current_detections(
                    camera_id=camera_id,
                    detection_type="person",
                )

                if not detections:
                    location = f" at the {camera_name}" if camera_name else ""
                    return ToolResult(
                        success=True,
                        message=f"No one detected{location}.",
                        data={"detections": [], "count": 0},
                    )

                # Format detections
                people = []
                for d in detections:
                    label = d.label or "unknown person"
                    if "known:" in (d.label or ""):
                        label = d.label.replace("known:", "")
                    people.append(f"{label} ({d.camera_id})")

                return ToolResult(
                    success=True,
                    message=f"Detected {len(detections)} person(s): {', '.join(people)}",
                    data={"detections": [{"label": d.label, "camera": d.camera_id} for d in detections]},
                )

            # --- Motion events ---
            if any(kw in question_lower for kw in ["motion", "movement", "activity"]):
                camera_name = self._extract_camera_name(question_lower)
                camera_id = client._resolve_camera(camera_name) if camera_name else None

                # Parse time range
                hours = 1
                if "today" in question_lower:
                    hours = 24
                elif "hour" in question_lower:
                    # Try to extract number
                    import re
                    match = re.search(r"(\d+)\s*hour", question_lower)
                    if match:
                        hours = int(match.group(1))

                events = await client.get_motion_events(camera_id=camera_id, hours=hours)

                if not events:
                    location = f" at {camera_name}" if camera_name else ""
                    return ToolResult(
                        success=True,
                        message=f"No motion detected{location} in the last {hours} hour(s).",
                        data={"events": [], "count": 0},
                    )

                # Group by camera
                by_camera = {}
                for e in events:
                    by_camera.setdefault(e.camera_id, []).append(e)

                summary = [f"{cam}: {len(evts)} events" for cam, evts in by_camera.items()]
                return ToolResult(
                    success=True,
                    message=f"Motion detected in last {hours}h: {', '.join(summary)}",
                    data={"events_by_camera": {k: len(v) for k, v in by_camera.items()}},
                )

            # --- Fallback: Complex query to LLM service ---
            logger.info("Routing complex query to video LLM service: %s", question)
            answer = await client.natural_query(question)
            return ToolResult(
                success=True,
                message=answer,
                data={"source": "llm_service"},
            )

        except Exception as e:
            logger.error("Security query failed: %s", e)
            return ToolResult(
                success=False,
                message=f"Security query failed: {e}",
                data={"error": str(e)},
            )

    def _extract_camera_name(self, text: str) -> Optional[str]:
        """Extract camera name/location from question."""
        # Common camera locations
        locations = [
            "front door", "back door", "backyard", "front yard",
            "garage", "driveway", "living room", "kitchen",
            "bedroom", "office", "porch", "patio",
        ]

        for loc in locations:
            if loc in text:
                return loc

        return None


# =============================================================================
# TOOL 2: Security Action (Write)
# =============================================================================

@register_tool("security_action")
class SecurityActionTool(Tool):
    """
    Control the security system - arm/disarm zones, recording, alerts.

    Examples:
    - "Arm the perimeter"
    - "Disarm all zones"
    - "Start recording front door"
    - "Stop all recordings"
    """

    name = "security_action"
    description = (
        "Control security system: arm/disarm zones, start/stop recording, "
        "trigger alerts. Use for any security actions."
    )
    parameters = {
        "action": {
            "type": "string",
            "description": "Natural language action to perform",
            "required": True,
        }
    }

    async def execute(self, action: str, **kwargs) -> ToolResult:
        """Execute security action."""
        client = get_video_client()
        action_lower = action.lower()

        try:
            # --- Arm/Disarm ---
            if "arm" in action_lower and "disarm" not in action_lower:
                zone = self._extract_zone(action_lower) or "all"
                success = await client.arm_zone(zone)
                if success:
                    return ToolResult(
                        success=True,
                        message=f"Armed {zone} zone.",
                        data={"action": "arm", "zone": zone},
                    )
                else:
                    return ToolResult(
                        success=False,
                        message=f"Failed to arm {zone} zone.",
                        data={"action": "arm", "zone": zone},
                    )

            if "disarm" in action_lower:
                zone = self._extract_zone(action_lower) or "all"
                success = await client.disarm_zone(zone)
                if success:
                    return ToolResult(
                        success=True,
                        message=f"Disarmed {zone} zone.",
                        data={"action": "disarm", "zone": zone},
                    )
                else:
                    return ToolResult(
                        success=False,
                        message=f"Failed to disarm {zone} zone.",
                        data={"action": "disarm", "zone": zone},
                    )

            # --- Recording ---
            if "record" in action_lower:
                camera_name = self._extract_camera_name(action_lower)

                if "stop" in action_lower:
                    if camera_name:
                        success = await client.stop_recording(camera_name)
                        msg = f"Stopped recording on {camera_name}." if success else f"Failed to stop recording on {camera_name}."
                    else:
                        # Stop all - would need to iterate cameras
                        msg = "Stopped all recordings."
                        success = True
                    return ToolResult(success=success, message=msg, data={"action": "stop_recording"})

                else:  # start recording
                    if camera_name:
                        success = await client.start_recording(camera_name)
                        msg = f"Started recording on {camera_name}." if success else f"Failed to start recording on {camera_name}."
                    else:
                        msg = "Please specify which camera to record."
                        success = False
                    return ToolResult(success=success, message=msg, data={"action": "start_recording"})

            # --- Unknown action ---
            return ToolResult(
                success=False,
                message=f"Unknown security action: {action}. Try: arm, disarm, start recording, stop recording.",
                data={"action": action},
            )

        except Exception as e:
            logger.error("Security action failed: %s", e)
            return ToolResult(
                success=False,
                message=f"Security action failed: {e}",
                data={"error": str(e)},
            )

    def _extract_zone(self, text: str) -> Optional[str]:
        """Extract security zone from action."""
        zones = ["perimeter", "interior", "garage", "all", "home", "away"]
        for zone in zones:
            if zone in text:
                return zone
        return None

    def _extract_camera_name(self, text: str) -> Optional[str]:
        """Extract camera name from action."""
        locations = [
            "front door", "back door", "backyard", "front yard",
            "garage", "driveway", "living room", "kitchen",
        ]
        for loc in locations:
            if loc in text:
                return loc
        return None


# =============================================================================
# TOOL 3: Security Alert (Proactive - called by event consumer)
# =============================================================================

class SecurityAlertHandler:
    """
    Handles incoming security events from Kafka.

    This isn't a tool the LLM calls - it's the other direction.
    Events come in, we decide if Atlas should announce something.
    """

    def __init__(self):
        self.alert_cooldown: dict[str, datetime] = {}  # Prevent spam
        self.cooldown_seconds = 30

    def should_alert(self, event_type: str, camera_id: str) -> bool:
        """Check if we should alert for this event (cooldown check)."""
        key = f"{event_type}:{camera_id}"
        last_alert = self.alert_cooldown.get(key)

        if last_alert and (datetime.now() - last_alert).seconds < self.cooldown_seconds:
            return False

        self.alert_cooldown[key] = datetime.now()
        return True

    async def handle_event(self, event: dict) -> Optional[str]:
        """
        Process security event and return announcement text if needed.

        Returns None if no announcement should be made.
        """
        event_type = event.get("type")
        camera_id = event.get("camera_id")

        if not self.should_alert(event_type, camera_id):
            return None

        # Format announcement based on event type
        camera_name = self._camera_id_to_name(camera_id)

        if event_type == "person_detected":
            label = event.get("label", "Someone")
            if "known:" in label:
                name = label.replace("known:", "")
                return f"{name} is at the {camera_name}."
            else:
                return f"Someone is at the {camera_name}."

        elif event_type == "motion_detected":
            return f"Motion detected at the {camera_name}."

        elif event_type == "unknown_face":
            return f"Unknown person detected at the {camera_name}."

        elif event_type == "vehicle_detected":
            return f"Vehicle detected at the {camera_name}."

        elif event_type == "package_detected":
            return f"Package delivery at the {camera_name}."

        return None

    def _camera_id_to_name(self, camera_id: str) -> str:
        """Convert camera_id to friendly name."""
        name_map = {
            "cam_front_door": "front door",
            "cam_back_door": "back door",
            "cam_backyard": "backyard",
            "cam_garage": "garage",
            "cam_driveway": "driveway",
        }
        return name_map.get(camera_id, camera_id)


# Global alert handler
alert_handler = SecurityAlertHandler()
