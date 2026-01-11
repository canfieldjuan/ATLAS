"""
Pipeline state machine for orchestration.

Defines the states and transitions for the audio processing pipeline.
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional


class PipelineState(Enum):
    """States of the audio processing pipeline."""

    IDLE = auto()              # Waiting, no activity
    LISTENING = auto()         # Listening for wake word
    WAKE_DETECTED = auto()     # Wake word heard, preparing to record
    RECORDING = auto()         # Recording user speech
    TRANSCRIBING = auto()      # Converting speech to text
    PROCESSING = auto()        # LLM/VLM processing the request
    EXECUTING = auto()         # Executing device actions
    RESPONDING = auto()        # Generating TTS response
    ERROR = auto()             # Error state


class PipelineEvent(Enum):
    """Events that trigger state transitions."""

    # Audio events
    WAKE_WORD_DETECTED = auto()
    SPEECH_START = auto()
    SPEECH_END = auto()
    SILENCE_TIMEOUT = auto()

    # Processing events
    TRANSCRIPTION_COMPLETE = auto()
    INTENT_PARSED = auto()
    ACTION_COMPLETE = auto()
    RESPONSE_READY = auto()

    # Control events
    CANCEL = auto()
    RESET = auto()
    ERROR_OCCURRED = auto()


# State transition table
STATE_TRANSITIONS: dict[tuple[PipelineState, PipelineEvent], PipelineState] = {
    # From IDLE
    (PipelineState.IDLE, PipelineEvent.WAKE_WORD_DETECTED): PipelineState.WAKE_DETECTED,
    (PipelineState.IDLE, PipelineEvent.SPEECH_START): PipelineState.RECORDING,  # No wake word mode

    # From LISTENING
    (PipelineState.LISTENING, PipelineEvent.WAKE_WORD_DETECTED): PipelineState.WAKE_DETECTED,
    (PipelineState.LISTENING, PipelineEvent.CANCEL): PipelineState.IDLE,

    # From WAKE_DETECTED
    (PipelineState.WAKE_DETECTED, PipelineEvent.SPEECH_START): PipelineState.RECORDING,
    (PipelineState.WAKE_DETECTED, PipelineEvent.SILENCE_TIMEOUT): PipelineState.IDLE,

    # From RECORDING
    (PipelineState.RECORDING, PipelineEvent.SPEECH_END): PipelineState.TRANSCRIBING,
    (PipelineState.RECORDING, PipelineEvent.SILENCE_TIMEOUT): PipelineState.TRANSCRIBING,
    (PipelineState.RECORDING, PipelineEvent.CANCEL): PipelineState.IDLE,

    # From TRANSCRIBING
    (PipelineState.TRANSCRIBING, PipelineEvent.TRANSCRIPTION_COMPLETE): PipelineState.PROCESSING,
    (PipelineState.TRANSCRIBING, PipelineEvent.ERROR_OCCURRED): PipelineState.ERROR,

    # From PROCESSING
    (PipelineState.PROCESSING, PipelineEvent.INTENT_PARSED): PipelineState.EXECUTING,
    (PipelineState.PROCESSING, PipelineEvent.RESPONSE_READY): PipelineState.RESPONDING,  # No action needed
    (PipelineState.PROCESSING, PipelineEvent.ERROR_OCCURRED): PipelineState.ERROR,

    # From EXECUTING
    (PipelineState.EXECUTING, PipelineEvent.ACTION_COMPLETE): PipelineState.RESPONDING,
    (PipelineState.EXECUTING, PipelineEvent.ERROR_OCCURRED): PipelineState.ERROR,

    # From RESPONDING
    (PipelineState.RESPONDING, PipelineEvent.RESPONSE_READY): PipelineState.IDLE,

    # From ERROR
    (PipelineState.ERROR, PipelineEvent.RESET): PipelineState.IDLE,

    # Global reset
    (PipelineState.LISTENING, PipelineEvent.RESET): PipelineState.IDLE,
    (PipelineState.WAKE_DETECTED, PipelineEvent.RESET): PipelineState.IDLE,
    (PipelineState.RECORDING, PipelineEvent.RESET): PipelineState.IDLE,
    (PipelineState.TRANSCRIBING, PipelineEvent.RESET): PipelineState.IDLE,
    (PipelineState.PROCESSING, PipelineEvent.RESET): PipelineState.IDLE,
    (PipelineState.EXECUTING, PipelineEvent.RESET): PipelineState.IDLE,
    (PipelineState.RESPONDING, PipelineEvent.RESET): PipelineState.IDLE,
}


@dataclass
class PipelineContext:
    """
    Context passed through the pipeline.

    Accumulates data as the request moves through stages.
    """

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    wake_detected_at: Optional[datetime] = None
    recording_started_at: Optional[datetime] = None
    recording_ended_at: Optional[datetime] = None

    # Audio data
    audio_bytes: Optional[bytes] = None
    audio_sample_rate: int = 16000

    # Transcription
    transcript: Optional[str] = None
    transcript_confidence: float = 0.0

    # Speaker identification
    speaker_id: Optional[str] = None
    speaker_confidence: float = 0.0

    # Intent / reasoning
    intent: Optional[Any] = None  # Intent dataclass
    llm_response: Optional[str] = None

    # Action results
    action_results: list[dict] = field(default_factory=list)

    # Response
    response_text: Optional[str] = None
    response_audio: Optional[bytes] = None

    # Error info
    error: Optional[str] = None

    def elapsed_ms(self) -> float:
        """Total time elapsed since pipeline started."""
        return (datetime.now() - self.started_at).total_seconds() * 1000


class PipelineStateMachine:
    """
    State machine for the audio processing pipeline.

    Manages transitions and validates state changes.
    """

    def __init__(self):
        self._state = PipelineState.IDLE
        self._context = PipelineContext()
        self._listeners: list[callable] = []

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def context(self) -> PipelineContext:
        return self._context

    def reset(self) -> None:
        """Reset to idle state with fresh context."""
        self._state = PipelineState.IDLE
        self._context = PipelineContext()
        self._notify(PipelineEvent.RESET)

    def transition(self, event: PipelineEvent) -> bool:
        """
        Attempt a state transition.

        Returns True if transition was valid, False otherwise.
        """
        key = (self._state, event)
        new_state = STATE_TRANSITIONS.get(key)

        if new_state is None:
            return False

        old_state = self._state
        self._state = new_state
        self._notify(event, old_state, new_state)
        return True

    def can_transition(self, event: PipelineEvent) -> bool:
        """Check if a transition is valid without performing it."""
        return (self._state, event) in STATE_TRANSITIONS

    def add_listener(self, callback: callable) -> None:
        """Add a listener for state changes."""
        self._listeners.append(callback)

    def remove_listener(self, callback: callable) -> None:
        """Remove a state change listener."""
        self._listeners.remove(callback)

    def _notify(
        self,
        event: PipelineEvent,
        old_state: Optional[PipelineState] = None,
        new_state: Optional[PipelineState] = None,
    ) -> None:
        """Notify listeners of state change."""
        for listener in self._listeners:
            try:
                result = listener(event, old_state, new_state, self._context)
                if inspect.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        pass
            except Exception:
                pass
