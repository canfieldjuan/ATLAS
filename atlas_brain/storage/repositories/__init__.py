"""
Repository classes for database access.

Repositories provide a clean interface for data access,
hiding the SQL implementation details.
"""

from .conversation import ConversationRepository
from .device import DeviceRepository, get_device_repo
from .feedback import FeedbackRepository, get_feedback_repo
from .profile import ProfileRepository, get_profile_repo
from .session import SessionRepository
from .vector import VectorRepository, get_vector_repository

__all__ = [
    "ConversationRepository",
    "DeviceRepository",
    "FeedbackRepository",
    "ProfileRepository",
    "SessionRepository",
    "VectorRepository",
    "get_device_repo",
    "get_feedback_repo",
    "get_profile_repo",
    "get_vector_repository",
]
