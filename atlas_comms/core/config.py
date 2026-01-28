"""
Configuration for the external communications system.

Supports multiple business contexts, each with its own:
- Phone number(s)
- Operating hours
- Greeting/persona
- Services and pricing
- Scheduling rules
"""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BusinessHours(BaseModel):
    """Operating hours for a business context."""

    # 24-hour format, e.g., "09:00"
    monday_open: Optional[str] = "09:00"
    monday_close: Optional[str] = "17:00"
    tuesday_open: Optional[str] = "09:00"
    tuesday_close: Optional[str] = "17:00"
    wednesday_open: Optional[str] = "09:00"
    wednesday_close: Optional[str] = "17:00"
    thursday_open: Optional[str] = "09:00"
    thursday_close: Optional[str] = "17:00"
    friday_open: Optional[str] = "09:00"
    friday_close: Optional[str] = "17:00"
    saturday_open: Optional[str] = None  # None = closed
    saturday_close: Optional[str] = None
    sunday_open: Optional[str] = None
    sunday_close: Optional[str] = None

    timezone: str = "America/Chicago"


class SchedulingConfig(BaseModel):
    """Appointment scheduling configuration."""

    enabled: bool = True
    calendar_id: Optional[str] = None  # Google Calendar ID
    min_notice_hours: int = 24  # Minimum hours notice for booking
    max_advance_days: int = 30  # How far out can book
    default_duration_minutes: int = 60
    buffer_minutes: int = 15  # Buffer between appointments


class BusinessContext(BaseModel):
    """
    Configuration for a single business context.

    Each context represents a distinct phone identity (business or personal).
    """

    # Identity
    id: str  # Unique identifier, e.g., "effingham_maids", "personal"
    name: str  # Display name, e.g., "Effingham Office Maids"
    description: str = ""

    # Phone number(s) associated with this context (E.164 format)
    phone_numbers: list[str] = Field(default_factory=list)

    # Voice persona
    greeting: str = "Hello, how can I help you today?"
    voice_name: str = "Atlas"  # Name the AI uses for itself
    persona: str = ""  # Additional personality instructions for LLM

    # Business info (for LLM context)
    business_type: str = ""  # e.g., "cleaning service"
    services: list[str] = Field(default_factory=list)
    service_area: str = ""
    pricing_info: str = ""  # Free-form pricing description for LLM

    # Operating hours
    hours: BusinessHours = Field(default_factory=BusinessHours)
    after_hours_message: str = (
        "Thank you for calling. We're currently closed. "
        "Please leave a message or call back during business hours."
    )

    # Scheduling
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)

    # Call handling
    transfer_number: Optional[str] = None  # Number to transfer to owner
    take_messages: bool = True
    max_call_duration_minutes: int = 10

    # SMS settings
    sms_enabled: bool = True
    sms_auto_reply: bool = True


class ServerConfig(BaseSettings):
    """Server configuration."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_COMMS_SERVER_")

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5003, description="Server port")


class TwilioProviderConfig(BaseModel):
    """Twilio-specific configuration."""

    account_sid: str = ""
    auth_token: str = ""
    voice_url: str = ""  # Webhook URL for incoming calls
    sms_url: str = ""  # Webhook URL for incoming SMS


class SignalWireProviderConfig(BaseModel):
    """SignalWire-specific configuration."""

    project_id: str = ""
    api_token: str = ""
    space_name: str = ""  # Your SignalWire space, e.g., "mycompany"


class CalendarConfig(BaseSettings):
    """Google Calendar OAuth configuration for scheduling."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_COMMS_CALENDAR_")

    enabled: bool = Field(default=False, description="Enable calendar integration")
    client_id: Optional[str] = Field(default=None, description="Google OAuth client ID")
    client_secret: Optional[str] = Field(default=None, description="Google OAuth client secret")
    refresh_token: Optional[str] = Field(default=None, description="Google OAuth refresh token")


class AtlasBrainConfig(BaseSettings):
    """Configuration for connecting to atlas_brain for AI services."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_COMMS_BRAIN_")

    url: str = Field(default="http://localhost:8001", description="Atlas Brain API URL")
    timeout: float = Field(default=30.0, description="HTTP timeout seconds")


class CommsConfig(BaseSettings):
    """Main communications configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_COMMS_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable communications module")

    # Server config
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Atlas Brain connection (for AI services)
    brain: AtlasBrainConfig = Field(default_factory=AtlasBrainConfig)

    # Calendar integration for scheduling
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)

    # PersonaPlex speech-to-speech for phone calls
    personaplex_enabled: bool = Field(
        default=False,
        description="Use PersonaPlex for phone calls instead of STT+LLM+TTS"
    )

    # Provider selection
    provider: str = Field(
        default="twilio",
        description="Telephony provider: twilio, signalwire, telnyx"
    )

    # Provider configs (loaded from env)
    twilio_account_sid: str = Field(default="", description="Twilio Account SID")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token")

    signalwire_project_id: str = Field(default="", description="SignalWire Project ID")
    signalwire_api_token: str = Field(default="", description="SignalWire API Token")
    signalwire_space: str = Field(default="", description="SignalWire Space Name")

    # Webhook settings
    webhook_base_url: str = Field(
        default="",
        description="Base URL for webhooks (e.g., https://your-domain.com)"
    )

    # Audio settings for voice calls
    audio_sample_rate: int = Field(default=8000, description="Audio sample rate for calls")
    audio_encoding: str = Field(default="mulaw", description="Audio encoding: mulaw, pcm")

    # Recording
    record_calls: bool = Field(default=False, description="Record calls for review")
    recording_storage_path: str = Field(default="recordings/", description="Path to store recordings")

    # Default behavior
    default_context: str = Field(
        default="personal",
        description="Default context for unrecognized numbers"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")


# Singleton instance
comms_settings = CommsConfig()


# Business contexts are loaded from database or config file
# This is a placeholder for the default personal context
DEFAULT_PERSONAL_CONTEXT = BusinessContext(
    id="personal",
    name="Personal",
    description="Personal phone calls",
    greeting="Hello?",
    voice_name="Atlas",
    persona="You are a personal assistant. Be casual and helpful.",
    take_messages=True,
    sms_enabled=True,
)

# Effingham Office Maids business context
EFFINGHAM_MAIDS_CONTEXT = BusinessContext(
    id="effingham_maids",
    name="Effingham Office Maids",
    description="Professional commercial and office cleaning service in Effingham, IL",
    phone_numbers=["+16183683696"],  # SignalWire number
    greeting=(
        "Thank you for calling Effingham Office Maids, "
        "this is Atlas, your virtual assistant. How can I help you today?"
    ),
    voice_name="Atlas",
    persona=(
        "You are a friendly and professional virtual receptionist for Effingham Office Maids, "
        "a commercial cleaning company. Be helpful, courteous, and efficient. "
        "When discussing pricing, explain that prices depend on the size and condition of the space, "
        "cleaning frequency, and specific services needed. Always offer to schedule a free on-site estimate. "
        "Your goals are to: 1) Answer questions about services, 2) Schedule free estimates or cleaning appointments, "
        "3) Take messages for the owner if needed. Be warm but professional - this is a local family business."
    ),
    business_type="commercial cleaning service",
    services=[
        "Office cleaning - regular maintenance cleaning for offices",
        "Commercial cleaning - retail, medical offices, warehouses",
        "Move-in/move-out cleaning - thorough cleaning for vacated spaces",
        "Deep cleaning - one-time intensive cleaning",
        "Post-construction cleaning - debris and dust removal after renovations",
        "Floor care - stripping, waxing, carpet cleaning",
        "Window cleaning - interior and exterior",
    ],
    service_area="Effingham, Mattoon, Charleston, and surrounding areas within 30 miles",
    pricing_info=(
        "Pricing is customized based on square footage, cleaning frequency, and specific needs. "
        "We offer free on-site estimates with no obligation. "
        "General ranges: Small offices (under 2,000 sq ft) typically $100-200 per cleaning. "
        "Medium offices (2,000-5,000 sq ft) typically $200-400 per cleaning. "
        "Discounts available for weekly or bi-weekly recurring service. "
        "Deep cleaning and specialty services are quoted individually."
    ),
    hours=BusinessHours(
        monday_open="00:00",
        monday_close="23:59",
        tuesday_open="00:00",
        tuesday_close="23:59",
        wednesday_open="00:00",
        wednesday_close="23:59",
        thursday_open="00:00",
        thursday_close="23:59",
        friday_open="00:00",
        friday_close="23:59",
        saturday_open="00:00",
        saturday_close="23:59",
        sunday_open="00:00",
        sunday_close="23:59",
        timezone="America/Chicago",
    ),
    after_hours_message=(
        "Thank you for calling Effingham Office Maids. We're currently closed. "
        "Our office hours are Monday through Friday, 8 AM to 5 PM Central Time. "
        "Please leave your name and number, and we'll call you back on the next business day. "
        "You can also text us at this number, and we'll respond as soon as possible!"
    ),
    scheduling=SchedulingConfig(
        enabled=True,
        calendar_id=None,  # Set via ATLAS_COMMS_EFFINGHAM_MAIDS_CALENDAR_ID
        min_notice_hours=24,
        max_advance_days=60,
        default_duration_minutes=60,
        buffer_minutes=30,
    ),
    transfer_number=None,  # Set via ATLAS_COMMS_EFFINGHAM_MAIDS_TRANSFER_NUMBER
    max_call_duration_minutes=15,
    take_messages=True,
    sms_enabled=True,
    sms_auto_reply=True,
)


# Load calendar_id from environment variable if set
import os as _os

_effingham_calendar_id = _os.environ.get("ATLAS_COMMS_EFFINGHAM_MAIDS_CALENDAR_ID")
if _effingham_calendar_id:
    EFFINGHAM_MAIDS_CONTEXT.scheduling.calendar_id = _effingham_calendar_id

_effingham_transfer = _os.environ.get("ATLAS_COMMS_EFFINGHAM_MAIDS_TRANSFER_NUMBER")
if _effingham_transfer:
    EFFINGHAM_MAIDS_CONTEXT.transfer_number = _effingham_transfer
