"""
Processing module - Frame handling and detection.
"""

from .frame_buffer import Frame, FrameBuffer, get_frame_buffer
from .tracking import TrackStore, get_track_store

__all__ = [
    "Frame",
    "FrameBuffer",
    "get_frame_buffer",
    "TrackStore",
    "get_track_store",
]
