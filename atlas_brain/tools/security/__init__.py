"""
Security tools for camera management, detection, and access control.

Provides granular tools for LLM tool calling in security mode.
"""

from .security import (
    # Data classes
    Camera,
    Detection,
    SecurityZone,
    # Camera tools
    ListCamerasTool,
    list_cameras_tool,
    GetCameraStatusTool,
    get_camera_status_tool,
    StartRecordingTool,
    start_recording_tool,
    StopRecordingTool,
    stop_recording_tool,
    PTZControlTool,
    ptz_control_tool,
    # Detection tools
    GetCurrentDetectionsTool,
    get_current_detections_tool,
    QueryDetectionsTool,
    query_detections_tool,
    GetPersonAtLocationTool,
    get_person_at_location_tool,
    GetMotionEventsTool,
    get_motion_events_tool,
    # Access control tools
    ListZonesTool,
    list_zones_tool,
    GetZoneStatusTool,
    get_zone_status_tool,
    ArmZoneTool,
    arm_zone_tool,
    DisarmZoneTool,
    disarm_zone_tool,
)

__all__ = [
    # Data classes
    "Camera",
    "Detection",
    "SecurityZone",
    # Camera tools
    "ListCamerasTool",
    "list_cameras_tool",
    "GetCameraStatusTool",
    "get_camera_status_tool",
    "StartRecordingTool",
    "start_recording_tool",
    "StopRecordingTool",
    "stop_recording_tool",
    "PTZControlTool",
    "ptz_control_tool",
    # Detection tools
    "GetCurrentDetectionsTool",
    "get_current_detections_tool",
    "QueryDetectionsTool",
    "query_detections_tool",
    "GetPersonAtLocationTool",
    "get_person_at_location_tool",
    "GetMotionEventsTool",
    "get_motion_events_tool",
    # Access control tools
    "ListZonesTool",
    "list_zones_tool",
    "GetZoneStatusTool",
    "get_zone_status_tool",
    "ArmZoneTool",
    "arm_zone_tool",
    "DisarmZoneTool",
    "disarm_zone_tool",
]
