"""
Protocols and data models for the communications system.

Defines provider-agnostic interfaces that all telephony providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Coroutine, Optional
from uuid import UUID, uuid4


class CallState(Enum):
    """State of a phone call."""
    INITIATED = "initiated"      # Outbound call started, not yet ringing
    RINGING = "ringing"          # Phone is ringing
    CONNECTED = "connected"      # Call answered, conversation active
    ON_HOLD = "on_hold"          # Call placed on hold
    TRANSFERRING = "transferring"  # Being transferred
    ENDED = "ended"              # Call has ended
    FAILED = "failed"            # Call failed to connect
    VOICEMAIL = "voicemail"      # Went to voicemail


class CallDirection(Enum):
    """Direction of a call."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class SMSDirection(Enum):
    """Direction of an SMS message."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class Call:
    """
    Represents a phone call.

    Tracks all metadata and state for a single call session.
    """
    id: UUID = field(default_factory=uuid4)
    provider_call_id: str = ""  # Provider's unique ID for the call

    # Phone numbers (E.164 format: +1234567890)
    from_number: str = ""
    to_number: str = ""
    direction: CallDirection = CallDirection.INBOUND

    # State
    state: CallState = CallState.INITIATED

    # Context
    context_id: Optional[str] = None  # Which business context handles this

    # Timing
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Call data
    caller_name: Optional[str] = None  # Caller ID name if available
    recording_url: Optional[str] = None
    voicemail_url: Optional[str] = None

    # Conversation tracking
    transcript: list[dict] = field(default_factory=list)

    # Provider-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get call duration in seconds."""
        if self.answered_at and self.ended_at:
            return (self.ended_at - self.answered_at).total_seconds()
        return None

    def add_transcript_entry(
        self,
        role: str,  # "caller" or "assistant"
        text: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add an entry to the call transcript."""
        self.transcript.append({
            "role": role,
            "text": text,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
        })


@dataclass
class SMSMessage:
    """
    Represents an SMS message.
    """
    id: UUID = field(default_factory=uuid4)
    provider_message_id: str = ""  # Provider's unique ID

    # Phone numbers (E.164 format)
    from_number: str = ""
    to_number: str = ""
    direction: SMSDirection = SMSDirection.INBOUND

    # Content
    body: str = ""
    media_urls: list[str] = field(default_factory=list)  # MMS attachments

    # Context
    context_id: Optional[str] = None
    conversation_id: Optional[UUID] = None  # For threading messages

    # Timing
    sent_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None

    # Status
    status: str = "pending"  # pending, sent, delivered, failed
    error_message: Optional[str] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# Type aliases for callbacks
AudioChunkCallback = Callable[[bytes], Coroutine[Any, Any, None]]
CallEventCallback = Callable[[Call, str], Coroutine[Any, Any, None]]
SMSCallback = Callable[[SMSMessage], Coroutine[Any, Any, None]]


class TelephonyProvider(ABC):
    """
    Abstract interface for telephony providers.

    Implementations handle the actual communication with services like
    Twilio, SignalWire, Telnyx, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'twilio', 'signalwire')."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        pass

    # === Lifecycle ===

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the provider."""
        pass

    # === Inbound Call Handling ===

    @abstractmethod
    async def answer_call(self, call: Call) -> bool:
        """
        Answer an incoming call.

        Returns True if successfully answered.
        """
        pass

    @abstractmethod
    async def reject_call(self, call: Call, reason: str = "") -> bool:
        """Reject an incoming call."""
        pass

    # === Outbound Calls ===

    @abstractmethod
    async def make_call(
        self,
        to_number: str,
        from_number: str,
        context_id: Optional[str] = None,
    ) -> Call:
        """
        Initiate an outbound call.

        Returns a Call object tracking the call state.
        """
        pass

    # === Call Control ===

    @abstractmethod
    async def hangup(self, call: Call) -> bool:
        """End an active call."""
        pass

    @abstractmethod
    async def hold(self, call: Call) -> bool:
        """Place a call on hold."""
        pass

    @abstractmethod
    async def unhold(self, call: Call) -> bool:
        """Take a call off hold."""
        pass

    @abstractmethod
    async def transfer(self, call: Call, to_number: str) -> bool:
        """Transfer a call to another number."""
        pass

    # === Audio Streaming ===

    @abstractmethod
    async def stream_audio_to_call(
        self,
        call: Call,
        audio_iterator: AsyncIterator[bytes],
    ) -> None:
        """
        Stream audio to an active call (for TTS output).

        Audio should be in the format expected by the provider
        (typically 8kHz mulaw or 16kHz PCM).
        """
        pass

    @abstractmethod
    def set_audio_callback(
        self,
        call: Call,
        callback: AudioChunkCallback,
    ) -> None:
        """
        Set callback for receiving audio from a call (for STT input).

        Callback receives audio chunks as they arrive.
        """
        pass

    # === SMS ===

    @abstractmethod
    async def send_sms(
        self,
        to_number: str,
        from_number: str,
        body: str,
        media_urls: Optional[list[str]] = None,
        context_id: Optional[str] = None,
    ) -> SMSMessage:
        """
        Send an SMS/MMS message.

        Returns an SMSMessage tracking the message status.
        """
        pass

    # === Event Callbacks ===

    @abstractmethod
    def set_call_event_callback(self, callback: CallEventCallback) -> None:
        """
        Set callback for call events (state changes, etc.).

        Events: 'ringing', 'answered', 'ended', 'failed', etc.
        """
        pass

    @abstractmethod
    def set_sms_callback(self, callback: SMSCallback) -> None:
        """
        Set callback for incoming SMS messages.
        """
        pass

    # === Utility ===

    @abstractmethod
    async def get_call(self, provider_call_id: str) -> Optional[Call]:
        """Get call by provider's call ID."""
        pass

    @abstractmethod
    async def lookup_caller_id(self, phone_number: str) -> Optional[str]:
        """
        Look up caller ID name for a phone number.

        Returns the name if available, None otherwise.
        """
        pass
