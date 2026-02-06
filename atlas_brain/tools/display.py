"""
Display tools for Atlas Brain.

Tools to display camera feeds and content on monitors.
"""

import asyncio
import logging
import subprocess
from typing import Any

from .base import Tool, ToolResult, ToolParameter
from ..config import settings

logger = logging.getLogger("atlas.tools.display")


# Monitor configuration - could be moved to config
MONITORS = {
    "left": {"name": "DP-2", "position": "0x0", "index": 0},
    "right": {"name": "HDMI-A-1-0", "position": "1920x0", "index": 1},
}


def get_video_processing_url() -> str:
    """Get the video processing service URL (atlas_vision)."""
    return settings.security.video_processing_url



class ShowCameraFeedTool:
    """Display a camera feed on a monitor."""

    @property
    def name(self) -> str:
        return "show_camera_feed"

    @property
    def description(self) -> str:
        return "Display a camera feed on a monitor (left or right). Use annotated=true for face/gait recognition overlays."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Camera name or ID (e.g., 'office webcam', 'front door')",
                required=True,
            ),
            ToolParameter(
                name="display",
                param_type="string",
                description="Display target: 'left' or 'right'",
                required=True,
            ),
            ToolParameter(
                name="annotated",
                param_type="boolean",
                description="If true, show face detection boxes and pose skeleton overlays",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["show camera", "display camera", "camera feed", "show feed"]

    @property
    def category(self) -> str:
        return "display"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        camera_name = params.get("camera_name", "").lower().strip()
        display = params.get("display", "").lower().strip()
        annotated = params.get("annotated", False)

        if not camera_name:
            return ToolResult(success=False, message="camera_name is required")
        if not display:
            return ToolResult(success=False, message="display is required")

        # Map camera names to IDs
        camera_aliases = {
            "office": "webcam_office",
            "office webcam": "webcam_office",
            "webcam": "webcam_office",
            "front door": "cam_front_door",
            "front": "cam_front_door",
            "backyard": "cam_backyard",
            "back": "cam_backyard",
            "garage": "cam_garage",
            "driveway": "cam_driveway",
            "living room": "cam_living_room",
            "kitchen": "cam_kitchen",
        }
        camera_id = camera_aliases.get(camera_name, camera_name)

        # Determine stream URL based on annotation request
        base_url = get_video_processing_url()
        if annotated:
            # Use atlas_vision's recognition stream with face/pose overlays
            stream_url = f"{base_url}/cameras/{camera_id}/recognition_stream"
        else:
            # Use atlas_vision's raw stream
            stream_url = f"{base_url}/cameras/{camera_id}/stream"

        if display in ("left", "right"):
            return await self._show_on_monitor(camera_id, stream_url, display)
        else:
            return ToolResult(
                success=False,
                message=f"Unknown display: {display}. Use 'left' or 'right'",
            )

    async def _show_on_monitor(
        self, camera_id: str, stream_url: str, monitor: str
    ) -> ToolResult:
        """Open camera stream using ffplay on specified monitor."""
        import os

        monitor_config = MONITORS.get(monitor)
        if not monitor_config:
            return ToolResult(success=False, message=f"Unknown monitor: {monitor}")

        position = monitor_config["position"]
        x, y = position.replace("x", ",").split(",") if "x" in position else (position, "0")

        try:
            # Use ffplay with window position
            # SDL_VIDEO_WINDOW_POS sets initial window position
            env = os.environ.copy()
            env["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"

            cmd = [
                "ffplay",
                "-window_title", f"Camera: {camera_id}",
                "-x", "960",
                "-y", "720",
                "-i", stream_url,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )

            # Store PID for later cleanup
            pid_file = f"/tmp/camera_viewer_{camera_id}.pid"
            with open(pid_file, "w") as f:
                f.write(str(proc.pid))

            logger.info("Opened camera %s on %s monitor (pid=%d)", camera_id, monitor, proc.pid)

            return ToolResult(
                success=True,
                message=f"Showing {camera_id} on {monitor} monitor",
                data={"camera_id": camera_id, "display": monitor, "pid": proc.pid},
            )

        except Exception as e:
            logger.error("Failed to show camera on monitor: %s", e)
            return ToolResult(success=False, message=f"Failed to open viewer: {e}")

class CloseCameraFeedTool:
    """Close a camera feed display."""

    @property
    def name(self) -> str:
        return "close_camera_feed"

    @property
    def description(self) -> str:
        return "Close a camera feed viewer window"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="camera_name",
                param_type="string",
                description="Camera name or 'all' to close all viewers",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["close camera", "hide camera", "close feed"]

    @property
    def category(self) -> str:
        return "display"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        import os
        import signal
        import glob

        camera_name = params.get("camera_name", "").lower().strip()

        if not camera_name:
            return ToolResult(success=False, message="camera_name is required")

        camera_aliases = {
            "office": "webcam_office",
            "office webcam": "webcam_office",
            "webcam": "webcam_office",
            "front door": "cam_front_door",
            "backyard": "cam_backyard",
        }

        try:
            closed = []
            if camera_name == "all":
                pid_files = glob.glob("/tmp/camera_viewer_*.pid")
            else:
                camera_id = camera_aliases.get(camera_name, camera_name)
                pid_files = [f"/tmp/camera_viewer_{camera_id}.pid"]

            for pid_file in pid_files:
                if os.path.exists(pid_file):
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())
                    try:
                        os.kill(pid, signal.SIGTERM)
                        closed.append(pid_file.split("_")[-1].replace(".pid", ""))
                    except ProcessLookupError:
                        pass
                    os.remove(pid_file)

            if closed:
                return ToolResult(
                    success=True,
                    message=f"Closed camera viewers: {', '.join(closed)}",
                )
            return ToolResult(
                success=True,
                message=f"No active viewer for {camera_name}",
            )

        except Exception as e:
            logger.error("Failed to close camera viewer: %s", e)
            return ToolResult(success=False, message=f"Failed: {e}")


# Tool instances
show_camera_feed_tool = ShowCameraFeedTool()
close_camera_feed_tool = CloseCameraFeedTool()
