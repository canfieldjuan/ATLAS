"""
Detection module - Motion and object detection.
"""

from .base import BaseDetector
from .motion import MotionDetector, get_motion_detector
from .yolo import YOLODetector, get_yolo_detector

__all__ = [
    "BaseDetector",
    "MotionDetector",
    "get_motion_detector",
    "YOLODetector",
    "get_yolo_detector",
]
