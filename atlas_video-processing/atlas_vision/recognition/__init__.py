"""
Person recognition services for Atlas Vision.

Provides face and gait recognition with person tracking.
"""

from .face import FaceRecognitionService, get_face_service
from .gait import GaitRecognitionService, get_gait_service
from .repository import PersonRepository, get_person_repository
from .tracker import TrackManager, TrackedPerson, get_track_manager

__all__ = [
    "FaceRecognitionService",
    "get_face_service",
    "GaitRecognitionService",
    "get_gait_service",
    "PersonRepository",
    "get_person_repository",
    "TrackManager",
    "TrackedPerson",
    "get_track_manager",
]
