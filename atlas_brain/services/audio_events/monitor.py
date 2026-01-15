"""
Continuous audio monitoring for event detection.

Runs in the background, processing audio from microphone or stream,
detecting interesting events and updating the context.
"""

import asyncio
import logging
import struct
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

from ..protocols import AudioEvent

logger = logging.getLogger("atlas.services.audio_monitor")


@dataclass
class AudioMonitorConfig:
    """Configuration for audio monitoring."""

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 1000  # Process audio in 1-second chunks

    # Detection settings
    min_confidence: float = 0.3
    interesting_only: bool = True  # Only report interesting events

    # Cooldown to prevent spam
    event_cooldown_seconds: float = 5.0  # Don't repeat same event within this window

    # Performance
    processing_interval_ms: int = 500  # How often to run classification


@dataclass
class MonitoredEvent:
    """An event detected by the monitor with metadata."""

    event: AudioEvent
    timestamp: datetime = field(default_factory=datetime.now)
    location: Optional[str] = None
    priority: str = "low"


class AudioMonitor:
    """
    Continuous audio monitoring service.

    Processes audio chunks and detects events, with cooldown
    to prevent spam and integration with context aggregator.
    """

    def __init__(
        self,
        config: Optional[AudioMonitorConfig] = None,
        location: Optional[str] = None,
    ):
        self.config = config or AudioMonitorConfig()
        self.location = location

        self._classifier = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Audio buffer
        self._buffer: deque[bytes] = deque()
        self._buffer_size = int(
            self.config.sample_rate * self.config.chunk_duration_ms / 1000 * 2
        )

        # Event tracking
        self._recent_events: dict[str, datetime] = {}
        self._event_history: list[MonitoredEvent] = []

        # Callbacks
        self._on_event: Optional[Callable[[MonitoredEvent], None]] = None

    def _get_classifier(self):
        """Lazy load the audio classifier."""
        if self._classifier is None:
            from ..registry import audio_events_registry
            self._classifier = audio_events_registry.get_active()
        return self._classifier

    def set_event_callback(self, callback: Callable[[MonitoredEvent], None]) -> None:
        """Set callback for when events are detected."""
        self._on_event = callback

    def add_audio(self, audio_chunk: bytes) -> None:
        """Add audio data to the processing buffer."""
        self._buffer.append(audio_chunk)

    def _should_report_event(self, label: str) -> bool:
        """Check if an event should be reported (respecting cooldown)."""
        if label not in self._recent_events:
            return True

        last_time = self._recent_events[label]
        cooldown = timedelta(seconds=self.config.event_cooldown_seconds)

        return datetime.now() - last_time > cooldown

    def _mark_event_reported(self, label: str) -> None:
        """Mark an event as recently reported."""
        self._recent_events[label] = datetime.now()

    def process_buffer(self) -> list[MonitoredEvent]:
        """
        Process buffered audio and detect events.

        Returns list of detected events.
        """
        classifier = self._get_classifier()
        if classifier is None:
            return []

        # Collect buffered audio
        if not self._buffer:
            return []

        audio_data = b"".join(self._buffer)
        self._buffer.clear()

        if len(audio_data) < self._buffer_size // 2:
            return []

        # Convert to WAV for classifier
        wav_bytes = self._to_wav(audio_data)

        # Classify
        try:
            if self.config.interesting_only:
                events = classifier.get_interesting_events(
                    wav_bytes,
                    min_confidence=self.config.min_confidence,
                )
            else:
                events = classifier.classify(
                    wav_bytes,
                    min_confidence=self.config.min_confidence,
                )
        except Exception as e:
            logger.error("Classification error: %s", e)
            return []

        # Filter by cooldown and wrap in MonitoredEvent
        monitored = []
        for event in events:
            if self._should_report_event(event.label):
                priority = classifier.get_event_priority(event.label)
                monitored_event = MonitoredEvent(
                    event=event,
                    location=self.location,
                    priority=priority,
                )
                monitored.append(monitored_event)
                self._mark_event_reported(event.label)

                # Trigger callback
                if self._on_event:
                    try:
                        self._on_event(monitored_event)
                    except Exception as e:
                        logger.error("Event callback error: %s", e)

                # Send to centralized alert system
                asyncio.create_task(self._send_to_alerts(monitored_event))

                # Log interesting events
                if priority in ("high", "critical"):
                    logger.warning(
                        "Audio event [%s]: %s (%.2f) at %s",
                        priority.upper(),
                        event.label,
                        event.confidence,
                        self.location or "unknown",
                    )
                else:
                    logger.info(
                        "Audio event: %s (%.2f)",
                        event.label,
                        event.confidence,
                    )

        # Add to history
        self._event_history.extend(monitored)

        # Trim history (keep last 100 events)
        if len(self._event_history) > 100:
            self._event_history = self._event_history[-100:]

        return monitored

    async def _send_to_alerts(self, monitored_event: MonitoredEvent) -> None:
        """Send event to centralized alert system."""
        try:
            from ...alerts import AudioAlertEvent, get_alert_manager

            alert_event = AudioAlertEvent.from_monitored_event(
                monitored_event,
                source_id=self.location or "audio_monitor",
            )
            manager = get_alert_manager()
            await manager.process_event(alert_event)
        except Exception as e:
            logger.error("Failed to send audio event to alerts: %s", e)

    def _to_wav(self, audio_data: bytes) -> bytes:
        """Convert raw PCM to WAV format."""
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.config.channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(audio_data)
        return buffer.getvalue()

    async def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Audio monitor started (location=%s)", self.location)

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Audio monitor stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        interval = self.config.processing_interval_ms / 1000

        while self._running:
            try:
                self.process_buffer()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitor loop error: %s", e)
                await asyncio.sleep(1)

    def get_recent_events(
        self,
        seconds: int = 60,
        priority: Optional[str] = None,
    ) -> list[MonitoredEvent]:
        """Get recent events from history."""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        events = [e for e in self._event_history if e.timestamp >= cutoff]

        if priority:
            events = [e for e in events if e.priority == priority]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        self._recent_events.clear()


class MicrophoneMonitor(AudioMonitor):
    """
    Audio monitor that captures from the system microphone.

    Uses pyaudio for microphone access.
    """

    def __init__(
        self,
        config: Optional[AudioMonitorConfig] = None,
        location: Optional[str] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__(config, location)
        self._device_index = device_index
        self._audio_interface = None
        self._stream = None

    async def start(self) -> None:
        """Start microphone capture and monitoring."""
        if self._running:
            return

        try:
            import pyaudio

            self._audio_interface = pyaudio.PyAudio()
            self._stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=int(self.config.sample_rate * 0.1),  # 100ms chunks
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()

        except ImportError:
            raise ImportError(
                "pyaudio not installed. Install with: pip install pyaudio"
            )
        except Exception as e:
            logger.error("Failed to open microphone: %s", e)
            raise

        await super().start()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio."""
        import pyaudio

        self.add_audio(in_data)
        return (None, pyaudio.paContinue)

    async def stop(self) -> None:
        """Stop microphone capture and monitoring."""
        await super().stop()

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._audio_interface:
            self._audio_interface.terminate()
            self._audio_interface = None


# Global monitor instance
_audio_monitor: Optional[AudioMonitor] = None


def get_audio_monitor() -> Optional[AudioMonitor]:
    """Get the global audio monitor instance."""
    return _audio_monitor


def set_audio_monitor(monitor: AudioMonitor) -> None:
    """Set the global audio monitor instance."""
    global _audio_monitor
    _audio_monitor = monitor
