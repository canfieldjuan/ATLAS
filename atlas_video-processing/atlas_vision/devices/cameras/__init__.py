"""
Camera devices module.
"""

from .base import BaseCameraCapability
from .mock import MockCamera, create_mock_cameras
from .rtsp import RTSPCamera
from .webcam import WebcamCamera, detect_webcams

__all__ = [
    "BaseCameraCapability",
    "MockCamera",
    "create_mock_cameras",
    "RTSPCamera",
    "WebcamCamera",
    "detect_webcams",
]
