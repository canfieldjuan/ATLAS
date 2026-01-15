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


class TwilioProviderConfig(BaseModel):
    """Twilio-specific configuration."""

    account_sid: str = ""
    auth_token: str = ""
    # Optional: specific settings
    voice_url: str = ""  # Webhook URL for incoming calls
    sms_url: str = ""  # Webhook URL for incoming SMS


class SignalWireProviderConfig(BaseModel):
    """SignalWire-specific configuration."""

    project_id: str = ""
    api_token: str = ""
    space_name: str = ""  # Your SignalWire space, e.g., "mycompany"


class CommsConfig(BaseSettings):
    """Main communications configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_COMMS_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable communications module")

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

# Example business context (to be loaded from DB)
EFFINGHAM_MAIDS_CONTEXT = BusinessContext(
    id="effingham_maids",
    name="Effingham Office Maids",
    description="Professional cleaning service",
    phone_numbers=[],  # To be filled with actual number
    greeting=(
        "Thank you for calling Effingham Office Maids, "
        "this is Atlas, your virtual assistant. How can I help you today?"
    ),
    voice_name="Atlas",
    persona=(
        "You are a friendly and professional virtual receptionist for a cleaning company. "
        "Be helpful, courteous, and efficient. If asked about pricing, explain that "
        "prices vary based on the size and condition of the space, and offer to schedule "
        "a free estimate. Always try to book an appointment or take a message."
    ),
    business_type="cleaning service",
    services=[
        "Office cleaning",
        "Commercial cleaning",
        "Move-in/move-out cleaning",
        "Deep cleaning",
        "Regular maintenance cleaning",
    ],
    service_area="Effingham and surrounding areas",
    pricing_info=(
        "Pricing varies based on square footage, frequency, and specific needs. "
        "We offer free estimates. Generally, regular office cleaning starts around "
        "$X per visit for small offices."  # To be filled in
    ),
    hours=BusinessHours(
        monday_open="08:00",
        monday_close="17:00",
        tuesday_open="08:00",
        tuesday_close="17:00",
        wednesday_open="08:00",
        wednesday_close="17:00",
        thursday_open="08:00",
        thursday_close="17:00",
        friday_open="08:00",
        friday_close="17:00",
        saturday_open=None,
        saturday_close=None,
        sunday_open=None,
        sunday_close=None,
        timezone="America/Chicago",
    ),
    after_hours_message=(
        "Thank you for calling Effingham Office Maids. We're currently closed. "
        "Our hours are Monday through Friday, 8 AM to 5 PM. "
        "Please leave your name and number, and we'll call you back on the next business day. "
        "Or feel free to send us a text message!"
    ),
    scheduling=SchedulingConfig(
        enabled=True,
        min_notice_hours=24,
        max_advance_days=60,
        default_duration_minutes=120,  # 2 hour default for cleaning
        buffer_minutes=30,
    ),
    transfer_number=None,  # Owner's number for transfers
    max_call_duration_minutes=15,
)
