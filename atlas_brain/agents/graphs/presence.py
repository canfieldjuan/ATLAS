"""
Presence-aware device control workflow using LangGraph.

Consolidates 4 presence tools into a single workflow:
- LightsNearUser: control lights near user
- MediaNearUser: control TV/media near user
- SceneNearUser: set lighting scene near user
- WhereAmI: report user location

The graph handles intent classification and routes to appropriate operations.
Location resolution happens via the PresenceProxyService.
"""

import logging
import os
import re
import time
from typing import Any

from langgraph.graph import END, StateGraph

from .state import PresenceWorkflowState

logger = logging.getLogger("atlas.agents.graphs.presence")

# Workflow type identifier for routing and state persistence
PRESENCE_WORKFLOW_TYPE = "presence"


# =============================================================================
# Tool Wrappers
# =============================================================================

def _use_real_tools() -> bool:
    """Check if we should use real tools (configured via ATLAS_WORKFLOW_USE_REAL_TOOLS)."""
    from ...config import settings
    return settings.workflows.use_real_tools


async def tool_get_presence_context(user_id: str = "primary") -> dict[str, Any]:
    """Get user's current room and presence info."""
    if _use_real_tools():
        from atlas_brain.presence import get_presence_proxy
        proxy = get_presence_proxy()
        try:
            user_presence = await proxy.get_user_presence(user_id)
            if user_presence and user_presence.current_room:
                return {
                    "success": True,
                    "room_id": user_presence.current_room,
                    "room_name": user_presence.current_room_name or user_presence.current_room,
                    "confidence": user_presence.confidence,
                    "source": user_presence.source.value if user_presence.source else None,
                }
            return {"success": False, "error": "Could not determine location"}
        except Exception as e:
            logger.error("Presence error: %s", e)
            return {"success": False, "error": str(e)}
    else:
        return {
            "success": True,
            "room_id": "living_room",
            "room_name": "Living Room",
            "confidence": 0.95,
            "source": "camera",
        }


async def tool_get_devices_near_user(
    device_type: str,
    user_id: str = "primary",
) -> dict[str, Any]:
    """Get device entity IDs near user."""
    if _use_real_tools():
        from atlas_brain.presence import get_presence_proxy
        proxy = get_presence_proxy()
        try:
            devices = await proxy.get_devices_near_user(user_id, device_type)
            return {"success": True, "devices": devices}
        except Exception as e:
            logger.error("Device lookup error: %s", e)
            return {"success": False, "error": str(e), "devices": []}
    else:
        mock_devices = {
            "lights": ["light.living_room_main", "light.living_room_lamp"],
            "media_players": ["media_player.living_room_tv"],
            "switches": ["switch.living_room_fan"],
        }
        return {"success": True, "devices": mock_devices.get(device_type, [])}


async def tool_control_lights(
    entity_ids: list[str],
    action: str,
    brightness: int | None = None,
) -> dict[str, Any]:
    """Control lights via Home Assistant."""
    if _use_real_tools():
        from atlas_brain.capabilities.backends.homeassistant import HomeAssistantBackend
        from atlas_brain.config import settings

        if not settings.homeassistant.enabled:
            return {"success": False, "error": "Home Assistant not configured"}

        try:
            backend = HomeAssistantBackend(
                base_url=settings.homeassistant.url,
                access_token=settings.homeassistant.token,
            )
            await backend.connect()

            if action == "on":
                service = "light/turn_on"
                payload = {"entity_id": entity_ids}
                if brightness is not None:
                    payload["brightness_pct"] = brightness
            elif action == "off":
                service = "light/turn_off"
                payload = {"entity_id": entity_ids}
            elif action == "toggle":
                service = "light/toggle"
                payload = {"entity_id": entity_ids}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}

            await backend.send_command(service, payload)
            await backend.disconnect()
            return {"success": True, "action": action, "entities": entity_ids}
        except Exception as e:
            logger.error("Light control error: %s", e)
            return {"success": False, "error": str(e)}
    else:
        return {"success": True, "action": action, "entities": entity_ids}


async def tool_control_media(
    entity_ids: list[str],
    action: str,
) -> dict[str, Any]:
    """Control media players via Home Assistant."""
    if _use_real_tools():
        from atlas_brain.capabilities.backends.homeassistant import HomeAssistantBackend
        from atlas_brain.config import settings

        if not settings.homeassistant.enabled:
            return {"success": False, "error": "Home Assistant not configured"}

        action_map = {
            "on": "media_player/turn_on",
            "off": "media_player/turn_off",
            "play": "media_player/media_play",
            "pause": "media_player/media_pause",
            "stop": "media_player/media_stop",
        }
        service = action_map.get(action)
        if not service:
            return {"success": False, "error": f"Unknown action: {action}"}

        try:
            backend = HomeAssistantBackend(
                base_url=settings.homeassistant.url,
                access_token=settings.homeassistant.token,
            )
            await backend.connect()
            await backend.send_command(service, {"entity_id": entity_ids})
            await backend.disconnect()
            return {"success": True, "action": action, "entities": entity_ids}
        except Exception as e:
            logger.error("Media control error: %s", e)
            return {"success": False, "error": str(e)}
    else:
        return {"success": True, "action": action, "entities": entity_ids}


# Scene definitions
SCENE_CONFIGS = {
    "bright": {"brightness_pct": 100, "color_temp_kelvin": 5000},
    "focus": {"brightness_pct": 100, "color_temp_kelvin": 5500},
    "dim": {"brightness_pct": 30, "color_temp_kelvin": 3000},
    "cozy": {"brightness_pct": 40, "color_temp_kelvin": 2700},
    "relax": {"brightness_pct": 50, "color_temp_kelvin": 2700},
    "movie": {"brightness_pct": 10, "color_temp_kelvin": 2700},
    "off": None,
}


async def tool_set_scene(
    entity_ids: list[str],
    scene_name: str,
) -> dict[str, Any]:
    """Set lighting scene via Home Assistant."""
    if scene_name not in SCENE_CONFIGS:
        return {"success": False, "error": f"Unknown scene: {scene_name}"}

    if _use_real_tools():
        from atlas_brain.capabilities.backends.homeassistant import HomeAssistantBackend
        from atlas_brain.config import settings

        if not settings.homeassistant.enabled:
            return {"success": False, "error": "Home Assistant not configured"}

        try:
            backend = HomeAssistantBackend(
                base_url=settings.homeassistant.url,
                access_token=settings.homeassistant.token,
            )
            await backend.connect()

            scene_config = SCENE_CONFIGS[scene_name]
            if scene_config is None:
                await backend.send_command("light/turn_off", {"entity_id": entity_ids})
            else:
                payload = {"entity_id": entity_ids, **scene_config}
                await backend.send_command("light/turn_on", payload)

            await backend.disconnect()
            return {"success": True, "scene": scene_name, "entities": entity_ids}
        except Exception as e:
            logger.error("Scene control error: %s", e)
            return {"success": False, "error": str(e)}
    else:
        return {"success": True, "scene": scene_name, "entities": entity_ids}


# =============================================================================
# Intent Classification
# =============================================================================

LIGHTS_PATTERNS = [
    # Turn on/off lights
    (r"turn\s+(on|off)\s+(?:the\s+)?lights?", "lights_control"),
    (r"(?:switch|flip)\s+(?:the\s+)?lights?\s+(on|off)", "lights_control"),
    (r"lights?\s+(on|off)", "lights_control"),
    # Toggle
    (r"toggle\s+(?:the\s+)?lights?", "lights_control"),
    # Brightness
    (r"(?:dim|brighten)\s+(?:the\s+)?lights?", "lights_control"),
    (r"(?:set|make)\s+(?:the\s+)?lights?\s+(?:to\s+)?(\d+)", "lights_control"),
]

MEDIA_PATTERNS = [
    # Turn on/off TV
    (r"turn\s+(on|off)\s+(?:the\s+)?(?:tv|television|media)", "media_control"),
    (r"(?:tv|television)\s+(on|off)", "media_control"),
    # Play/pause/stop - with optional context
    (r"(play|pause|stop)\s+(?:the\s+)?(?:tv|television|media|video)?", "media_control"),
    (r"(?:resume|continue)\s+(?:playing)?", "media_control"),
    # Standalone play/pause/stop (common voice commands)
    (r"^(play|pause|stop)$", "media_control"),
]

SCENE_PATTERNS = [
    # Explicit scene
    (r"(?:set|make|switch)\s+(?:it\s+)?(?:to\s+)?(bright|dim|cozy|movie|focus|relax)", "scene_set"),
    (r"(bright|dim|cozy|movie|focus|relax)\s+(?:mode|scene|lighting)", "scene_set"),
    # Context-based scenes
    (r"(?:i(?:'m|\s+am)\s+)?(?:going\s+to\s+)?watch(?:ing)?\s+(?:a\s+)?(?:movie|tv|film)", "scene_set"),
    (r"make\s+it\s+(cozy|bright|dim|relaxing)", "scene_set"),
    (r"(?:movie|cinema)\s+(?:time|mode)", "scene_set"),
]

LOCATION_PATTERNS = [
    (r"where\s+am\s+i", "where_am_i"),
    (r"what\s+room\s+am\s+i\s+in", "where_am_i"),
    (r"(?:which|what)\s+room", "where_am_i"),
    (r"my\s+(?:current\s+)?(?:location|room)", "where_am_i"),
]


def classify_presence_intent(text: str) -> tuple[str, dict[str, Any]]:
    """Classify presence intent from natural language."""
    text_lower = text.lower().strip()
    params: dict[str, Any] = {}

    # Check location patterns first (most specific)
    for pattern, intent in LOCATION_PATTERNS:
        if re.search(pattern, text_lower):
            return intent, params

    # Check scene patterns (before lights - "movie mode" should be scene)
    for pattern, intent in SCENE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            # Extract scene name if captured
            if match.groups():
                scene = match.group(1).lower()
                # Map context words to scenes
                scene_map = {"relaxing": "relax", "cinema": "movie"}
                params["scene_name"] = scene_map.get(scene, scene)
            # Default scene for movie watching
            if "movie" in text_lower or "watch" in text_lower:
                params["scene_name"] = params.get("scene_name", "movie")
            return intent, params

    # Check lights patterns
    for pattern, intent in LIGHTS_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if match.groups():
                action_or_value = match.group(1).lower()
                if action_or_value.isdigit():
                    params["brightness"] = int(action_or_value)
                    params["light_action"] = "on"
                else:
                    params["light_action"] = action_or_value
            # Handle dim/brighten
            if "dim" in text_lower:
                params["light_action"] = "on"
                params["brightness"] = params.get("brightness", 30)
            elif "brighten" in text_lower:
                params["light_action"] = "on"
                params["brightness"] = params.get("brightness", 100)
            elif "toggle" in text_lower:
                params["light_action"] = "toggle"
            return intent, params

    # Check media patterns
    for pattern, intent in MEDIA_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if match.groups():
                action = match.group(1).lower()
                if action in ("resume", "continue"):
                    params["media_action"] = "play"
                else:
                    params["media_action"] = action
            return intent, params

    return "unknown", params


# =============================================================================
# Graph Nodes
# =============================================================================

def classify_intent(state: PresenceWorkflowState) -> PresenceWorkflowState:
    """Classify presence intent from input text."""
    start = time.time()
    text = state.get("input_text", "")

    intent, params = classify_presence_intent(text)

    updates: dict[str, Any] = {
        "intent": intent,
        "current_step": "resolve_presence",
        "step_timings": {**(state.get("step_timings") or {}), "classify": (time.time() - start) * 1000},
    }

    # Copy extracted params
    if "light_action" in params:
        updates["light_action"] = params["light_action"]
    if "brightness" in params:
        updates["brightness"] = params["brightness"]
    if "media_action" in params:
        updates["media_action"] = params["media_action"]
    if "scene_name" in params:
        updates["scene_name"] = params["scene_name"]

    if intent == "unknown":
        updates["needs_clarification"] = True
        updates["clarification_prompt"] = "I can control lights, TV, or set a scene. What would you like?"

    return {**state, **updates}


async def resolve_presence(state: PresenceWorkflowState) -> PresenceWorkflowState:
    """Resolve user's current location."""
    start = time.time()
    user_id = state.get("user_id", "primary")

    result = await tool_get_presence_context(user_id)

    updates: dict[str, Any] = {
        "current_step": "execute",
        "step_timings": {**(state.get("step_timings") or {}), "resolve": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["current_room_id"] = result.get("room_id")
        updates["current_room_name"] = result.get("room_name")
        updates["presence_confidence"] = result.get("confidence", 0.0)
        updates["presence_source"] = result.get("source")
    else:
        updates["error"] = result.get("error", "Could not determine your location")
        updates["needs_clarification"] = True
        updates["clarification_prompt"] = "I could not determine which room you are in."

    return {**state, **updates}


async def execute_lights(state: PresenceWorkflowState) -> PresenceWorkflowState:
    """Execute lights control near user."""
    start = time.time()
    user_id = state.get("user_id", "primary")
    action = state.get("light_action", "on")
    brightness = state.get("brightness")

    # Get lights near user
    devices_result = await tool_get_devices_near_user("lights", user_id)
    lights = devices_result.get("devices", [])

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if not lights:
        updates["error"] = f"No lights found in {state.get('current_room_name', 'your location')}"
        return {**state, **updates}

    result = await tool_control_lights(lights, action, brightness)

    if result.get("success"):
        updates["action_executed"] = True
        updates["devices_controlled"] = lights
        updates["light_entities"] = lights
    else:
        updates["error"] = result.get("error", "Failed to control lights")

    return {**state, **updates}


async def execute_media(state: PresenceWorkflowState) -> PresenceWorkflowState:
    """Execute media control near user."""
    start = time.time()
    user_id = state.get("user_id", "primary")
    action = state.get("media_action", "on")

    # Get media players near user
    devices_result = await tool_get_devices_near_user("media_players", user_id)
    media_players = devices_result.get("devices", [])

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if not media_players:
        updates["error"] = f"No TV or media player found in {state.get('current_room_name', 'your location')}"
        return {**state, **updates}

    result = await tool_control_media(media_players, action)

    if result.get("success"):
        updates["action_executed"] = True
        updates["devices_controlled"] = media_players
        updates["media_entities"] = media_players
    else:
        updates["error"] = result.get("error", "Failed to control media")

    return {**state, **updates}


async def execute_scene(state: PresenceWorkflowState) -> PresenceWorkflowState:
    """Execute scene setting near user."""
    start = time.time()
    user_id = state.get("user_id", "primary")
    scene_name = state.get("scene_name", "cozy")

    # Get lights near user for scene
    devices_result = await tool_get_devices_near_user("lights", user_id)
    lights = devices_result.get("devices", [])

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if not lights:
        updates["error"] = f"No lights found in {state.get('current_room_name', 'your location')}"
        return {**state, **updates}

    result = await tool_set_scene(lights, scene_name)

    if result.get("success"):
        updates["action_executed"] = True
        updates["devices_controlled"] = lights
        updates["light_entities"] = lights
    else:
        updates["error"] = result.get("error", "Failed to set scene")

    return {**state, **updates}


async def execute_where_am_i(state: PresenceWorkflowState) -> PresenceWorkflowState:
    """Report user's current location."""
    start = time.time()

    updates: dict[str, Any] = {
        "current_step": "respond",
        "location_reported": True,
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    # Location already resolved in resolve_presence step
    if not state.get("current_room_id"):
        updates["error"] = "Could not determine your location"

    return {**state, **updates}


# -----------------------------------------------------------------------------
# Response Generation
# -----------------------------------------------------------------------------

def generate_response(state: PresenceWorkflowState) -> PresenceWorkflowState:
    """Generate human-readable response."""
    start = time.time()

    if state.get("error"):
        response = state["error"]
    elif state.get("needs_clarification"):
        response = state.get("clarification_prompt", "I need more information.")
    else:
        response = _generate_intent_response(state)

    total_ms = sum((state.get("step_timings") or {}).values()) + (time.time() - start) * 1000

    return {
        **state,
        "response": response,
        "current_step": "complete",
        "total_ms": total_ms,
    }


def _generate_intent_response(state: PresenceWorkflowState) -> str:
    """Generate response based on intent and state."""
    intent = state.get("intent", "unknown")
    room_name = state.get("current_room_name", "your location")

    if intent == "lights_control":
        if state.get("action_executed"):
            action = state.get("light_action", "on")
            action_text = {"on": "turned on", "off": "turned off", "toggle": "toggled"}.get(action, action)
            brightness = state.get("brightness")
            if brightness:
                return f"I have set the lights in the {room_name} to {brightness}%."
            return f"I have {action_text} the lights in the {room_name}."
        return "I could not control the lights."

    if intent == "media_control":
        if state.get("action_executed"):
            action = state.get("media_action", "on")
            action_text = {
                "on": "turned on",
                "off": "turned off",
                "play": "started playing",
                "pause": "paused",
                "stop": "stopped",
            }.get(action, action)
            return f"I have {action_text} the TV in the {room_name}."
        return "I could not control the TV."

    if intent == "scene_set":
        if state.get("action_executed"):
            scene = state.get("scene_name", "cozy")
            return f"I have set the {room_name} to {scene} mode."
        return "I could not set the scene."

    if intent == "where_am_i":
        if state.get("location_reported") and state.get("current_room_id"):
            confidence = state.get("presence_confidence", 0.0)
            source = state.get("presence_source", "unknown")
            return f"You are in the {room_name} (detected via {source}, {confidence:.0%} confidence)."
        return "I am not sure which room you are in right now."

    return "I am not sure how to help with that."


# =============================================================================
# Graph Building
# =============================================================================

def route_after_presence(state: PresenceWorkflowState) -> str:
    """Route to appropriate execution node after presence is resolved."""
    if state.get("error") or state.get("needs_clarification"):
        return "respond"

    intent = state.get("intent", "unknown")
    route_map = {
        "lights_control": "execute_lights",
        "media_control": "execute_media",
        "scene_set": "execute_scene",
        "where_am_i": "execute_where_am_i",
    }
    return route_map.get(intent, "respond")


def build_presence_graph() -> StateGraph:
    """Build the presence workflow StateGraph."""
    graph = StateGraph(PresenceWorkflowState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("resolve_presence", resolve_presence)
    graph.add_node("execute_lights", execute_lights)
    graph.add_node("execute_media", execute_media)
    graph.add_node("execute_scene", execute_scene)
    graph.add_node("execute_where_am_i", execute_where_am_i)
    graph.add_node("respond", generate_response)

    # Set entry point
    graph.set_entry_point("classify_intent")

    # Flow: classify -> resolve_presence -> execute_* -> respond
    graph.add_edge("classify_intent", "resolve_presence")

    # Conditional routing after presence resolution
    graph.add_conditional_edges(
        "resolve_presence",
        route_after_presence,
        {
            "execute_lights": "execute_lights",
            "execute_media": "execute_media",
            "execute_scene": "execute_scene",
            "execute_where_am_i": "execute_where_am_i",
            "respond": "respond",
        },
    )

    # All execution nodes go to respond
    graph.add_edge("execute_lights", "respond")
    graph.add_edge("execute_media", "respond")
    graph.add_edge("execute_scene", "respond")
    graph.add_edge("execute_where_am_i", "respond")

    # Respond goes to END
    graph.add_edge("respond", END)

    return graph


def compile_presence_graph():
    """Compile the presence workflow graph."""
    graph = build_presence_graph()
    return graph.compile()


async def run_presence_workflow(
    input_text: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the presence workflow with the given input.

    Args:
        input_text: Natural language presence request
        session_id: Optional session identifier
        user_id: Optional user identifier (default: "primary")

    Returns:
        Dict with response and workflow results
    """
    compiled = compile_presence_graph()

    initial_state: PresenceWorkflowState = {
        "input_text": input_text,
        "session_id": session_id,
        "user_id": user_id or "primary",
        "current_step": "classify",
        "step_timings": {},
    }

    result = await compiled.ainvoke(initial_state)

    return {
        "intent": result.get("intent"),
        "response": result.get("response"),
        "error": result.get("error"),
        "total_ms": result.get("total_ms", 0),
        # Presence context
        "room_id": result.get("current_room_id"),
        "room_name": result.get("current_room_name"),
        "presence_confidence": result.get("presence_confidence"),
        "presence_source": result.get("presence_source"),
        # Results
        "action_executed": result.get("action_executed", False),
        "devices_controlled": result.get("devices_controlled", []),
        "location_reported": result.get("location_reported", False),
    }
