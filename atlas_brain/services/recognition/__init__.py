"""
Person recognition services for Atlas.

Provides face and gait recognition capabilities.
"""

from .face import FaceRecognitionService, get_face_service
from .gait import GaitRecognitionService, get_gait_service
from .repository import PersonRepository, get_person_repository

__all__ = [
    "FaceRecognitionService",
    "get_face_service",
    "GaitRecognitionService",
    "get_gait_service",
    "PersonRepository",
    "get_person_repository",
]
