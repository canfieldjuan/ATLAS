"""
Repository classes for database access.

Repositories provide a clean interface for data access,
hiding the SQL implementation details.
"""

from .appointment import AppointmentRepository, get_appointment_repo
from .conversation import ConversationRepository
from .device import DeviceRepository, get_device_repo
from .email import EmailRepository, get_email_repo
from .feedback import FeedbackRepository, get_feedback_repo
from .profile import ProfileRepository, get_profile_repo
from .reminder import ReminderRepository, get_reminder_repo
from .session import SessionRepository
from .unified_alerts import UnifiedAlertRepository, get_unified_alert_repo
from .vector import VectorRepository, get_vector_repository
from .vision import VisionEventRepository, get_vision_event_repo
from .speaker import SpeakerRepository, get_speaker_repo
from .identity import IdentityRepository, get_identity_repo
from .scheduled_task import ScheduledTaskRepository, get_scheduled_task_repo

__all__ = [
    "AppointmentRepository",
    "ConversationRepository",
    "DeviceRepository",
    "EmailRepository",
    "FeedbackRepository",
    "ProfileRepository",
    "ReminderRepository",
    "SessionRepository",
    "UnifiedAlertRepository",
    "VectorRepository",
    "VisionEventRepository",
    "get_appointment_repo",
    "get_device_repo",
    "get_email_repo",
    "get_feedback_repo",
    "get_profile_repo",
    "get_reminder_repo",
    "get_unified_alert_repo",
    "get_vector_repository",
    "get_vision_event_repo",
    "SpeakerRepository",
    "get_speaker_repo",
    "IdentityRepository",
    "get_identity_repo",
    "ScheduledTaskRepository",
    "get_scheduled_task_repo",
]
