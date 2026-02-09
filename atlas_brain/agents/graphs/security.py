"""
Security workflow using LangGraph.

Consolidates 13 security tools into a single workflow:
- Camera: list, status, record start/stop, PTZ control
- Detection: current, query, person at location, motion events
- Zone: list, status, arm, disarm

The graph handles intent classification and routes to appropriate operations.
"""

import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any

from langgraph.graph import END, StateGraph

from .state import SecurityWorkflowState

logger = logging.getLogger("atlas.agents.graphs.security")

# Workflow type identifier for routing and state persistence
SECURITY_WORKFLOW_TYPE = "security"


# =============================================================================
# Tool Wrappers (Real vs Mock based on USE_REAL_TOOLS env var)
# =============================================================================

def _use_real_tools() -> bool:
    """Check if we should use real tools (configured via ATLAS_WORKFLOW_USE_REAL_TOOLS)."""
    from ...config import settings
    return settings.workflows.use_real_tools


async def tool_list_cameras() -> dict[str, Any]:
    """List all cameras."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            cameras = await client.list_cameras()
            return {
                "success": True,
                "cameras": [c.to_dict() for c in cameras],
                "count": len(cameras),
            }
        finally:
            await client.close()
    else:
        return {
            "success": True,
            "cameras": [
                {"id": "cam_front", "name": "Front Door", "location": "entrance", "status": "online"},
                {"id": "cam_back", "name": "Backyard", "location": "backyard", "status": "online"},
            ],
            "count": 2,
        }


async def tool_get_camera(camera_name: str) -> dict[str, Any]:
    """Get camera status by name."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            camera = await client.get_camera(camera_name)
            if camera:
                return {"success": True, "camera": camera.to_dict()}
            return {"success": False, "error": f"Camera '{camera_name}' not found"}
        finally:
            await client.close()
    else:
        return {
            "success": True,
            "camera": {
                "id": "cam_front",
                "name": camera_name or "Front Door",
                "location": "entrance",
                "status": "online",
                "capabilities": ["ptz", "record"],
            },
        }


async def tool_start_recording(camera_name: str) -> dict[str, Any]:
    """Start recording on a camera."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            success = await client.start_recording(camera_name)
            if success:
                return {"success": True, "message": f"Started recording on {camera_name}"}
            return {"success": False, "error": f"Failed to start recording on {camera_name}"}
        finally:
            await client.close()
    else:
        return {"success": True, "message": f"Started recording on {camera_name}"}


async def tool_stop_recording(camera_name: str) -> dict[str, Any]:
    """Stop recording on a camera."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            success = await client.stop_recording(camera_name)
            if success:
                return {"success": True, "message": f"Stopped recording on {camera_name}"}
            return {"success": False, "error": f"Failed to stop recording on {camera_name}"}
        finally:
            await client.close()
    else:
        return {"success": True, "message": f"Stopped recording on {camera_name}"}


async def tool_ptz_control(camera_name: str, action: str, value: float = 1.0) -> dict[str, Any]:
    """Control PTZ camera."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            success = await client.ptz_control(camera_name, action, value)
            if success:
                return {"success": True, "message": f"PTZ {action} on {camera_name}"}
            return {"success": False, "error": f"Failed PTZ {action} on {camera_name}"}
        finally:
            await client.close()
    else:
        return {"success": True, "message": f"PTZ {action} on {camera_name}"}


async def tool_get_current_detections(
    camera_name: str | None = None,
    detection_type: str | None = None,
) -> dict[str, Any]:
    """Get current/recent detections."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            detections = await client.get_current_detections(camera_name, detection_type)
            return {
                "success": True,
                "detections": [d.to_dict() for d in detections],
                "count": len(detections),
            }
        finally:
            await client.close()
    else:
        return {
            "success": True,
            "detections": [
                {"camera_id": "cam_front", "type": "person", "confidence": 0.95, "timestamp": datetime.now().isoformat()},
            ],
            "count": 1,
        }


async def tool_query_detections(
    camera_name: str | None = None,
    detection_type: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Query historical detections."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            detections = await client.query_detections(camera_name, detection_type, since, limit)
            return {
                "success": True,
                "detections": [d.to_dict() for d in detections],
                "count": len(detections),
            }
        finally:
            await client.close()
    else:
        return {
            "success": True,
            "detections": [
                {"camera_id": "cam_front", "type": "person", "confidence": 0.92, "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()},
                {"camera_id": "cam_back", "type": "vehicle", "confidence": 0.88, "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()},
            ],
            "count": 2,
        }


async def tool_get_motion_events(
    camera_name: str | None = None,
    hours: int = 1,
) -> dict[str, Any]:
    """Get motion events from the last N hours."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            events = await client.get_motion_events(camera_name, hours)
            return {
                "success": True,
                "events": [e.to_dict() for e in events],
                "count": len(events),
            }
        finally:
            await client.close()
    else:
        return {
            "success": True,
            "events": [
                {"camera_id": "cam_front", "type": "motion", "confidence": 1.0, "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()},
            ],
            "count": 1,
        }


async def tool_list_zones() -> dict[str, Any]:
    """List all security zones."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            zones = await client.list_zones()
            return {
                "success": True,
                "zones": [z.to_dict() for z in zones],
                "count": len(zones),
            }
        finally:
            await client.close()
    else:
        return {
            "success": True,
            "zones": [
                {"id": "zone_perimeter", "name": "Perimeter", "status": "armed", "cameras": ["cam_front", "cam_back"]},
                {"id": "zone_interior", "name": "Interior", "status": "disarmed", "cameras": []},
            ],
            "count": 2,
        }


async def tool_get_zone(zone_name: str) -> dict[str, Any]:
    """Get zone status by name."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            zone = await client.get_zone(zone_name)
            if zone:
                return {"success": True, "zone": zone.to_dict()}
            return {"success": False, "error": f"Zone '{zone_name}' not found"}
        finally:
            await client.close()
    else:
        return {
            "success": True,
            "zone": {"id": "zone_perimeter", "name": zone_name or "Perimeter", "status": "armed", "cameras": ["cam_front"]},
        }


async def tool_arm_zone(zone_name: str) -> dict[str, Any]:
    """Arm a security zone."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            success = await client.arm_zone(zone_name)
            if success:
                return {"success": True, "message": f"Zone '{zone_name}' armed"}
            return {"success": False, "error": f"Failed to arm zone '{zone_name}'"}
        finally:
            await client.close()
    else:
        return {"success": True, "message": f"Zone '{zone_name}' armed"}


async def tool_disarm_zone(zone_name: str) -> dict[str, Any]:
    """Disarm a security zone."""
    if _use_real_tools():
        from atlas_brain.tools.security import SecurityClient
        client = SecurityClient()
        try:
            success = await client.disarm_zone(zone_name)
            if success:
                return {"success": True, "message": f"Zone '{zone_name}' disarmed"}
            return {"success": False, "error": f"Failed to disarm zone '{zone_name}'"}
        finally:
            await client.close()
    else:
        return {"success": True, "message": f"Zone '{zone_name}' disarmed"}


# =============================================================================
# Intent Classification
# =============================================================================

# Intent patterns organized by category
CAMERA_PATTERNS = [
    # List cameras
    (r"(?:list|show|get|what are)\s+(?:all\s+)?(?:the\s+)?cameras?", "camera_list"),
    (r"(?:how many|which)\s+cameras?", "camera_list"),
    # Camera status
    (r"(?:status|state)\s+(?:of\s+)?(?:the\s+)?(.+?)\s+camera", "camera_status"),
    (r"(?:is|check)\s+(?:the\s+)?(.+?)\s+camera\s+(?:on|online|working)", "camera_status"),
    (r"(?:show|get)\s+(?:me\s+)?(?:the\s+)?(.+?)\s+camera(?:\s+status)?", "camera_status"),
    # Start recording
    (r"(?:start|begin)\s+recording\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+camera)?$", "camera_record_start"),
    (r"record\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+camera)?$", "camera_record_start"),
    # Stop recording
    (r"(?:stop|end)\s+recording\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+camera)?$", "camera_record_stop"),
    # PTZ control
    (r"(?:pan|tilt|zoom)\s+(?:the\s+)?(.+?)\s+camera\s+(left|right|up|down|in|out)", "camera_ptz"),
    (r"(?:move|turn)\s+(?:the\s+)?(.+?)\s+camera\s+(left|right|up|down)", "camera_ptz"),
]

DETECTION_PATTERNS = [
    # Current detections - check "cameras seeing" BEFORE camera patterns
    (r"(?:what|who)\s+(?:is|are)\s+(?:the\s+)?cameras?\s+seeing", "detection_current"),
    (r"(?:what|any)\s+(?:current\s+)?detections?", "detection_current"),
    (r"(?:is\s+)?(?:there\s+)?anyone\s+(?:on\s+)?(?:the\s+)?cameras?", "detection_current"),
    # Query detections
    (r"(?:show|get)\s+(?:me\s+)?(?:the\s+)?(?:recent\s+)?detections?", "detection_query"),
    (r"(?:detection|detections)\s+(?:from\s+)?(?:the\s+)?(?:last|past)\s+(\d+)\s+(hour|minute|day)", "detection_query"),
    # Person at location - include "someone"
    (r"(?:is\s+)?(?:there\s+)?(?:anyone|somebody|someone|a person)\s+(?:at|in|near)\s+(?:the\s+)?(.+)", "detection_person_location"),
    (r"(?:check|see)\s+(?:if\s+)?(?:anyone|somebody|someone)\s+(?:is\s+)?(?:at|in|near)\s+(?:the\s+)?(.+)", "detection_person_location"),
    # Motion events
    (r"(?:any|show|get)\s+motion\s+(?:events?)?", "detection_motion"),
    (r"(?:motion|movement)\s+(?:in|at)\s+(?:the\s+)?(?:last|past)\s+(\d+)\s+(hour|minute)", "detection_motion"),
    (r"(?:was|were)\s+there\s+(?:any\s+)?motion", "detection_motion"),
]

ZONE_PATTERNS = [
    # List zones
    (r"(?:list|show|get|what are)\s+(?:all\s+)?(?:the\s+)?(?:security\s+)?zones?", "zone_list"),
    (r"(?:how many|which)\s+(?:security\s+)?zones?", "zone_list"),
    # Zone status
    (r"(?:status|state)\s+(?:of\s+)?(?:the\s+)?(.+?)\s+zone", "zone_status"),
    (r"(?:is|check)\s+(?:the\s+)?(.+?)\s+zone\s+(?:armed|disarmed)", "zone_status"),
    (r"(?:show|get)\s+(?:me\s+)?(?:the\s+)?(.+?)\s+zone(?:\s+status)?", "zone_status"),
    # Disarm zone - MUST be before arm to avoid "disarm" matching "arm"
    (r"disarm\s+(?:the\s+)?(.+?)(?:\s+zone)?$", "zone_disarm"),
    (r"(?:disable|deactivate)\s+(?:the\s+)?(.+?)\s+(?:security\s+)?zone", "zone_disarm"),
    # Arm zone - use word boundary to not match "disarm"
    (r"(?<![a-z])arm\s+(?:the\s+)?(.+?)(?:\s+zone)?$", "zone_arm"),
    (r"(?:enable|activate)\s+(?:the\s+)?(.+?)\s+(?:security\s+)?zone", "zone_arm"),
]


def classify_security_intent(text: str) -> tuple[str, dict[str, Any]]:
    """
    Classify security intent from natural language.

    Returns:
        Tuple of (intent, extracted_params)
    """
    text_lower = text.lower().strip()
    params: dict[str, Any] = {}

    # Check detection patterns FIRST (e.g., "what are cameras seeing" before "what are cameras")
    for pattern, intent in DETECTION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if match.groups():
                if intent == "detection_person_location":
                    params["location"] = match.group(1).strip()
                elif intent in ("detection_query", "detection_motion"):
                    if len(match.groups()) >= 2:
                        params["time_value"] = int(match.group(1))
                        params["time_unit"] = match.group(2)
            return intent, params

    # Check camera patterns
    for pattern, intent in CAMERA_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if match.groups():
                if intent == "camera_ptz":
                    params["camera_name"] = match.group(1).strip()
                    params["ptz_action"] = match.group(2).strip()
                else:
                    params["camera_name"] = match.group(1).strip()
            return intent, params

    # Check zone patterns
    for pattern, intent in ZONE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if match.groups():
                params["zone_name"] = match.group(1).strip()
            return intent, params

    return "unknown", params


# =============================================================================
# Graph Nodes
# =============================================================================

def classify_intent(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Classify security intent from input text."""
    start = time.time()
    text = state.get("input_text", "")

    intent, params = classify_security_intent(text)

    updates: dict[str, Any] = {
        "intent": intent,
        "current_step": "execute",
        "step_timings": {**(state.get("step_timings") or {}), "classify": (time.time() - start) * 1000},
    }

    # Copy extracted params to state
    if "camera_name" in params:
        updates["camera_name"] = params["camera_name"]
    if "ptz_action" in params:
        updates["ptz_action"] = params["ptz_action"]
    if "zone_name" in params:
        updates["zone_name"] = params["zone_name"]
    if "location" in params:
        updates["location"] = params["location"]
    if "time_value" in params and "time_unit" in params:
        # Convert to hours for motion events
        hours = params["time_value"]
        if params["time_unit"] == "minute":
            hours = max(1, params["time_value"] // 60)
        elif params["time_unit"] == "day":
            hours = params["time_value"] * 24
        updates["limit"] = hours

    if intent == "unknown":
        updates["needs_clarification"] = True
        updates["clarification_prompt"] = "I'm not sure what security action you want. You can ask about cameras, detections, or security zones."

    return {**state, **updates}


# -----------------------------------------------------------------------------
# Camera Execution Nodes
# -----------------------------------------------------------------------------

async def execute_camera_list(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute list cameras."""
    start = time.time()

    result = await tool_list_cameras()

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["cameras_listed"] = True
        updates["camera_list"] = result.get("cameras", [])
        updates["camera_count"] = result.get("count", 0)
    else:
        updates["error"] = result.get("error", "Failed to list cameras")

    return {**state, **updates}


async def execute_camera_status(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute get camera status."""
    start = time.time()
    camera_name = state.get("camera_name")

    if not camera_name:
        return {
            **state,
            "error": "No camera name specified",
            "needs_clarification": True,
            "clarification_prompt": "Which camera do you want to check?",
        }

    result = await tool_get_camera(camera_name)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["camera_status"] = result.get("camera")
    else:
        updates["error"] = result.get("error", f"Failed to get camera '{camera_name}'")

    return {**state, **updates}


async def execute_camera_record_start(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute start recording."""
    start = time.time()
    camera_name = state.get("camera_name")

    if not camera_name:
        return {
            **state,
            "error": "No camera name specified",
            "needs_clarification": True,
            "clarification_prompt": "Which camera should I start recording on?",
        }

    result = await tool_start_recording(camera_name)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["recording_started"] = True
    else:
        updates["error"] = result.get("error", f"Failed to start recording on '{camera_name}'")

    return {**state, **updates}


async def execute_camera_record_stop(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute stop recording."""
    start = time.time()
    camera_name = state.get("camera_name")

    if not camera_name:
        return {
            **state,
            "error": "No camera name specified",
            "needs_clarification": True,
            "clarification_prompt": "Which camera should I stop recording on?",
        }

    result = await tool_stop_recording(camera_name)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["recording_stopped"] = True
    else:
        updates["error"] = result.get("error", f"Failed to stop recording on '{camera_name}'")

    return {**state, **updates}


async def execute_camera_ptz(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute PTZ control."""
    start = time.time()
    camera_name = state.get("camera_name")
    ptz_action = state.get("ptz_action")

    if not camera_name:
        return {
            **state,
            "error": "No camera name specified",
            "needs_clarification": True,
            "clarification_prompt": "Which camera do you want to control?",
        }

    if not ptz_action:
        return {
            **state,
            "error": "No PTZ action specified",
            "needs_clarification": True,
            "clarification_prompt": "What direction? (left, right, up, down, zoom in, zoom out)",
        }

    result = await tool_ptz_control(camera_name, ptz_action)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["response"] = f"Moved {camera_name} camera {ptz_action}"
    else:
        updates["error"] = result.get("error", f"Failed PTZ control on '{camera_name}'")

    return {**state, **updates}


# -----------------------------------------------------------------------------
# Detection Execution Nodes
# -----------------------------------------------------------------------------

async def execute_detection_current(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute get current detections."""
    start = time.time()
    camera_name = state.get("camera_name")
    detection_type = state.get("detection_type")

    result = await tool_get_current_detections(camera_name, detection_type)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["detections_retrieved"] = True
        updates["detection_list"] = result.get("detections", [])
        updates["detection_count"] = result.get("count", 0)
    else:
        updates["error"] = result.get("error", "Failed to get current detections")

    return {**state, **updates}


async def execute_detection_query(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute query detections."""
    start = time.time()
    camera_name = state.get("camera_name")
    detection_type = state.get("detection_type")
    limit = state.get("limit", 100)

    result = await tool_query_detections(camera_name, detection_type, None, limit)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["detections_retrieved"] = True
        updates["detection_list"] = result.get("detections", [])
        updates["detection_count"] = result.get("count", 0)
    else:
        updates["error"] = result.get("error", "Failed to query detections")

    return {**state, **updates}


async def execute_detection_person_location(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute check person at location."""
    start = time.time()
    location = state.get("location")

    if not location:
        return {
            **state,
            "error": "No location specified",
            "needs_clarification": True,
            "clarification_prompt": "Which location do you want to check?",
        }

    # Check current detections filtered by location (use as camera name)
    result = await tool_get_current_detections(location, "person")

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        detections = result.get("detections", [])
        updates["person_found"] = len(detections) > 0
        updates["person_location_result"] = {
            "location": location,
            "person_detected": len(detections) > 0,
            "detections": detections,
        }
    else:
        updates["error"] = result.get("error", f"Failed to check for person at '{location}'")

    return {**state, **updates}


async def execute_detection_motion(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute get motion events."""
    start = time.time()
    camera_name = state.get("camera_name")
    hours = state.get("limit", 1)

    result = await tool_get_motion_events(camera_name, hours)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["motion_events"] = result.get("events", [])
        updates["detection_count"] = result.get("count", 0)
    else:
        updates["error"] = result.get("error", "Failed to get motion events")

    return {**state, **updates}


# -----------------------------------------------------------------------------
# Zone Execution Nodes
# -----------------------------------------------------------------------------

async def execute_zone_list(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute list zones."""
    start = time.time()

    result = await tool_list_zones()

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["zones_listed"] = True
        updates["zone_list"] = result.get("zones", [])
        updates["zone_count"] = result.get("count", 0)
    else:
        updates["error"] = result.get("error", "Failed to list zones")

    return {**state, **updates}


async def execute_zone_status(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute get zone status."""
    start = time.time()
    zone_name = state.get("zone_name")

    if not zone_name:
        return {
            **state,
            "error": "No zone name specified",
            "needs_clarification": True,
            "clarification_prompt": "Which security zone do you want to check?",
        }

    result = await tool_get_zone(zone_name)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["zone_status_result"] = result.get("zone")
    else:
        updates["error"] = result.get("error", f"Failed to get zone '{zone_name}'")

    return {**state, **updates}


async def execute_zone_arm(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute arm zone."""
    start = time.time()
    zone_name = state.get("zone_name")

    if not zone_name:
        return {
            **state,
            "error": "No zone name specified",
            "needs_clarification": True,
            "clarification_prompt": "Which security zone do you want to arm?",
        }

    result = await tool_arm_zone(zone_name)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["zone_armed"] = True
    else:
        updates["error"] = result.get("error", f"Failed to arm zone '{zone_name}'")

    return {**state, **updates}


async def execute_zone_disarm(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Execute disarm zone."""
    start = time.time()
    zone_name = state.get("zone_name")

    if not zone_name:
        return {
            **state,
            "error": "No zone name specified",
            "needs_clarification": True,
            "clarification_prompt": "Which security zone do you want to disarm?",
        }

    result = await tool_disarm_zone(zone_name)

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["zone_disarmed"] = True
    else:
        updates["error"] = result.get("error", f"Failed to disarm zone '{zone_name}'")

    return {**state, **updates}


# -----------------------------------------------------------------------------
# Response Generation
# -----------------------------------------------------------------------------

def generate_response(state: SecurityWorkflowState) -> SecurityWorkflowState:
    """Generate human-readable response."""
    start = time.time()

    # Check for errors first
    if state.get("error"):
        response = state["error"]
    elif state.get("needs_clarification"):
        response = state.get("clarification_prompt", "I need more information.")
    elif state.get("response"):
        # Already have a response (e.g., from PTZ)
        response = state["response"]
    else:
        # Generate based on intent
        intent = state.get("intent", "unknown")
        response = _generate_intent_response(state, intent)

    total_ms = sum((state.get("step_timings") or {}).values()) + (time.time() - start) * 1000

    return {
        **state,
        "response": response,
        "current_step": "complete",
        "total_ms": total_ms,
    }


def _generate_intent_response(state: SecurityWorkflowState, intent: str) -> str:
    """Generate response based on intent and state."""
    # Camera responses
    if intent == "camera_list":
        cameras = state.get("camera_list", [])
        if not cameras:
            return "No cameras found."
        lines = [f"Found {len(cameras)} cameras:"]
        for cam in cameras:
            status = cam.get("status", "unknown")
            lines.append(f"  - {cam.get('name', cam.get('id'))}: {status}")
        return "\n".join(lines)

    if intent == "camera_status":
        cam = state.get("camera_status")
        if not cam:
            return "Could not get camera status."
        return f"{cam.get('name', 'Camera')} is {cam.get('status', 'unknown')}. Location: {cam.get('location', 'unknown')}"

    if intent == "camera_record_start":
        if state.get("recording_started"):
            return f"Started recording on {state.get('camera_name')} camera."
        return "Failed to start recording."

    if intent == "camera_record_stop":
        if state.get("recording_stopped"):
            return f"Stopped recording on {state.get('camera_name')} camera."
        return "Failed to stop recording."

    # Detection responses
    if intent == "detection_current":
        detections = state.get("detection_list", [])
        if not detections:
            return "No current detections."
        lines = [f"Found {len(detections)} current detections:"]
        for d in detections[:5]:  # Limit to 5
            lines.append(f"  - {d.get('type', 'unknown')} on {d.get('camera_id')} ({d.get('confidence', 0):.0%} confidence)")
        return "\n".join(lines)

    if intent == "detection_query":
        detections = state.get("detection_list", [])
        if not detections:
            return "No recent detections found."
        lines = [f"Found {len(detections)} detections:"]
        for d in detections[:5]:
            lines.append(f"  - {d.get('type', 'unknown')} on {d.get('camera_id')} at {d.get('timestamp', 'unknown')}")
        return "\n".join(lines)

    if intent == "detection_person_location":
        result = state.get("person_location_result", {})
        location = result.get("location", state.get("location", "unknown"))
        if result.get("person_detected"):
            return f"Yes, there is someone at {location}."
        return f"No one detected at {location}."

    if intent == "detection_motion":
        events = state.get("motion_events", [])
        if not events:
            return "No motion events in the specified time period."
        return f"Found {len(events)} motion events."

    # Zone responses
    if intent == "zone_list":
        zones = state.get("zone_list", [])
        if not zones:
            return "No security zones found."
        lines = [f"Found {len(zones)} security zones:"]
        for z in zones:
            lines.append(f"  - {z.get('name', z.get('id'))}: {z.get('status', 'unknown')}")
        return "\n".join(lines)

    if intent == "zone_status":
        zone = state.get("zone_status_result")
        if not zone:
            return "Could not get zone status."
        return f"{zone.get('name', 'Zone')} is {zone.get('status', 'unknown')}."

    if intent == "zone_arm":
        if state.get("zone_armed"):
            return f"{state.get('zone_name')} zone is now armed."
        return "Failed to arm zone."

    if intent == "zone_disarm":
        if state.get("zone_disarmed"):
            return f"{state.get('zone_name')} zone is now disarmed."
        return "Failed to disarm zone."

    return "I'm not sure how to respond to that security request."


# =============================================================================
# Graph Building
# =============================================================================

def route_by_intent(state: SecurityWorkflowState) -> str:
    """Route to appropriate execution node based on intent."""
    intent = state.get("intent", "unknown")

    # Handle clarification needed
    if state.get("needs_clarification"):
        return "respond"

    # Map intents to node names
    intent_routes = {
        "camera_list": "execute_camera_list",
        "camera_status": "execute_camera_status",
        "camera_record_start": "execute_camera_record_start",
        "camera_record_stop": "execute_camera_record_stop",
        "camera_ptz": "execute_camera_ptz",
        "detection_current": "execute_detection_current",
        "detection_query": "execute_detection_query",
        "detection_person_location": "execute_detection_person_location",
        "detection_motion": "execute_detection_motion",
        "zone_list": "execute_zone_list",
        "zone_status": "execute_zone_status",
        "zone_arm": "execute_zone_arm",
        "zone_disarm": "execute_zone_disarm",
    }

    return intent_routes.get(intent, "respond")


def build_security_graph() -> StateGraph:
    """Build the security workflow StateGraph."""
    graph = StateGraph(SecurityWorkflowState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)

    # Camera nodes
    graph.add_node("execute_camera_list", execute_camera_list)
    graph.add_node("execute_camera_status", execute_camera_status)
    graph.add_node("execute_camera_record_start", execute_camera_record_start)
    graph.add_node("execute_camera_record_stop", execute_camera_record_stop)
    graph.add_node("execute_camera_ptz", execute_camera_ptz)

    # Detection nodes
    graph.add_node("execute_detection_current", execute_detection_current)
    graph.add_node("execute_detection_query", execute_detection_query)
    graph.add_node("execute_detection_person_location", execute_detection_person_location)
    graph.add_node("execute_detection_motion", execute_detection_motion)

    # Zone nodes
    graph.add_node("execute_zone_list", execute_zone_list)
    graph.add_node("execute_zone_status", execute_zone_status)
    graph.add_node("execute_zone_arm", execute_zone_arm)
    graph.add_node("execute_zone_disarm", execute_zone_disarm)

    # Response node
    graph.add_node("respond", generate_response)

    # Set entry point
    graph.set_entry_point("classify_intent")

    # Add conditional routing from classify_intent
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "execute_camera_list": "execute_camera_list",
            "execute_camera_status": "execute_camera_status",
            "execute_camera_record_start": "execute_camera_record_start",
            "execute_camera_record_stop": "execute_camera_record_stop",
            "execute_camera_ptz": "execute_camera_ptz",
            "execute_detection_current": "execute_detection_current",
            "execute_detection_query": "execute_detection_query",
            "execute_detection_person_location": "execute_detection_person_location",
            "execute_detection_motion": "execute_detection_motion",
            "execute_zone_list": "execute_zone_list",
            "execute_zone_status": "execute_zone_status",
            "execute_zone_arm": "execute_zone_arm",
            "execute_zone_disarm": "execute_zone_disarm",
            "respond": "respond",
        },
    )

    # All execution nodes go to respond
    for node in [
        "execute_camera_list",
        "execute_camera_status",
        "execute_camera_record_start",
        "execute_camera_record_stop",
        "execute_camera_ptz",
        "execute_detection_current",
        "execute_detection_query",
        "execute_detection_person_location",
        "execute_detection_motion",
        "execute_zone_list",
        "execute_zone_status",
        "execute_zone_arm",
        "execute_zone_disarm",
    ]:
        graph.add_edge(node, "respond")

    # Respond goes to END
    graph.add_edge("respond", END)

    return graph


def compile_security_graph():
    """Compile the security workflow graph."""
    graph = build_security_graph()
    return graph.compile()


async def run_security_workflow(
    input_text: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the security workflow with the given input.

    Args:
        input_text: Natural language security request
        session_id: Optional session identifier

    Returns:
        Dict with response and workflow results
    """
    compiled = compile_security_graph()

    initial_state: SecurityWorkflowState = {
        "input_text": input_text,
        "session_id": session_id,
        "current_step": "classify",
        "step_timings": {},
    }

    result = await compiled.ainvoke(initial_state)

    return {
        "intent": result.get("intent"),
        "response": result.get("response"),
        "error": result.get("error"),
        "total_ms": result.get("total_ms", 0),
        # Camera results
        "cameras_listed": result.get("cameras_listed", False),
        "camera_list": result.get("camera_list", []),
        "camera_count": result.get("camera_count", 0),
        "camera_status": result.get("camera_status"),
        "recording_started": result.get("recording_started", False),
        "recording_stopped": result.get("recording_stopped", False),
        # Detection results
        "detections_retrieved": result.get("detections_retrieved", False),
        "detection_list": result.get("detection_list", []),
        "detection_count": result.get("detection_count", 0),
        "person_found": result.get("person_found", False),
        "motion_events": result.get("motion_events", []),
        # Zone results
        "zones_listed": result.get("zones_listed", False),
        "zone_list": result.get("zone_list", []),
        "zone_count": result.get("zone_count", 0),
        "zone_status": result.get("zone_status_result"),
        "zone_armed": result.get("zone_armed", False),
        "zone_disarmed": result.get("zone_disarmed", False),
    }
