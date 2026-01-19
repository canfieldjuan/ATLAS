"""
Mode configuration for Atlas.

Defines available modes and their tool groupings.
Each mode has a specific set of tools optimized for that use case.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModeType(str, Enum):
    """Available Atlas modes."""
    HOME = "home"
    RECEPTIONIST = "receptionist"
    COMMS = "comms"
    SECURITY = "security"
    CHAT = "chat"


@dataclass
class ModeConfig:
    """Configuration for a single mode."""
    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    model_preference: Optional[str] = None
    keywords: list[str] = field(default_factory=list)


# Shared tools available in all modes
SHARED_TOOLS = [
    "get_weather",
    "get_traffic",
    "get_location",
    "get_time",
]

# Mode-specific tool definitions
MODE_CONFIGS: dict[ModeType, ModeConfig] = {
    ModeType.HOME: ModeConfig(
        name="home",
        description="Smart home device control",
        tools=[
            "lights_near_user",
            "media_near_user",
            "scene_near_user",
            "where_am_i",
        ],
        model_preference="qwen3:8b",
        keywords=[
            "turn on", "turn off", "dim", "brightness",
            "light", "lights", "lamp",
            "tv", "television", "play", "pause", "volume",
            "scene", "mood",
        ],
    ),
    ModeType.RECEPTIONIST: ModeConfig(
        name="receptionist",
        description="Business scheduling, appointments, and client communications",
        tools=[
            # Calendar
            "get_calendar",
            # Reminders
            "set_reminder",
            "list_reminders",
            "complete_reminder",
            # Appointments
            "check_availability",
            "book_appointment",
            "cancel_appointment",
            "reschedule_appointment",
            "lookup_customer",
            # Business notifications
            "send_estimate_email",
            "send_proposal_email",
            "send_notification",
        ],
        model_preference="qwen3:14b",
        keywords=[
            "appointment", "estimate", "proposal", "client",
            "customer", "book", "cancel", "reschedule",
            "available", "availability", "business",
        ],
    ),
    ModeType.COMMS: ModeConfig(
        name="comms",
        description="Personal communications - phone, email, video chat, text",
        tools=[
            # Personal communications tools to be added:
            # - send_text (SMS to contacts)
            # - make_call (phone calls)
            # - send_personal_email
            # - video_call (initiate video chat)
            "send_email",
            "send_notification",
            # Reminders for personal use
            "set_reminder",
            "list_reminders",
            "complete_reminder",
            "get_calendar",
        ],
        model_preference="qwen3:14b",
        keywords=[
            "call", "text", "message", "email", "video",
            "phone", "contact", "friend", "family", "mom",
            "dad", "brother", "sister",
        ],
    ),
    ModeType.SECURITY: ModeConfig(
        name="security",
        description="Cameras, drones, recognition, and distributed monitoring",
        tools=[
            # Camera management
            "list_cameras",
            "get_camera_status",
            "start_recording",
            "stop_recording",
            "ptz_control",
            # Detection
            "get_current_detections",
            "query_detections",
            "get_person_at_location",
            "get_motion_events",
            # Access control
            "list_zones",
            "get_zone_status",
            "arm_zone",
            "disarm_zone",
            # Display
            "show_camera_feed",
            "close_camera_feed",
        ],
        model_preference="qwen3:14b",
        keywords=[
            "camera", "security", "motion", "alert",
            "person", "detect", "watch", "drone", "feed",
            "recognize", "face", "who", "intruder",
            "arm", "disarm", "recording",
        ],
    ),
    ModeType.CHAT: ModeConfig(
        name="chat",
        description="General conversation with cloud LLM",
        tools=[],
        model_preference="cloud",
        keywords=[],
    ),
}


def get_mode_config(mode: ModeType) -> ModeConfig:
    """Get configuration for a specific mode."""
    return MODE_CONFIGS.get(mode, MODE_CONFIGS[ModeType.CHAT])


def get_mode_tools(mode: ModeType, include_shared: bool = True) -> list[str]:
    """Get list of tools for a mode."""
    config = get_mode_config(mode)
    tools = list(config.tools)
    if include_shared:
        tools.extend(SHARED_TOOLS)
    return tools


def get_all_tools() -> list[str]:
    """Get list of all tools across all modes."""
    all_tools = set(SHARED_TOOLS)
    for config in MODE_CONFIGS.values():
        all_tools.update(config.tools)
    return list(all_tools)
