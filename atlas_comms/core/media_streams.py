"""
Media Stream Registry -- bridges WebSocket handlers and telephony providers.

When a Twilio/SignalWire media stream WebSocket connects, the handler registers
the connection here.  The TwilioProvider's stream_audio_to_call() and
set_audio_callback() methods use this registry to push/receive audio without
needing a direct reference to the WebSocket object.

Thread-safe: all access goes through asyncio primitives on the event loop.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any, Optional

logger = logging.getLogger("atlas.comms.media_streams")

# Type for audio callbacks: receives raw mulaw bytes
AudioCallback = Callable[[bytes], Coroutine[Any, Any, None]]

# Type for send functions: accepts base64-encoded audio payload
SendFunc = Callable[[str], Coroutine[Any, Any, None]]


@dataclass
class MediaStream:
    """Represents an active media stream for a single call."""

    call_sid: str
    stream_sid: Optional[str] = None

    # Function to send base64-encoded audio back to the caller
    _send_func: Optional[SendFunc] = None

    # Callback for inbound audio (caller → Atlas)
    _audio_callback: Optional[AudioCallback] = None

    # Queue for outbound audio when no send_func yet (stream not started)
    _outbound_buffer: list[str] = field(default_factory=list)

    def set_send_func(self, func: SendFunc) -> None:
        """Set the function used to send audio to the caller."""
        self._send_func = func

    def set_audio_callback(self, callback: AudioCallback) -> None:
        """Set callback for receiving inbound audio from the caller."""
        self._audio_callback = callback

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send audio to the caller (mulaw bytes → base64 → WebSocket)."""
        payload = base64.b64encode(audio_bytes).decode("ascii")
        if self._send_func is not None:
            await self._send_func(payload)
        else:
            self._outbound_buffer.append(payload)

    async def send_audio_b64(self, payload_b64: str) -> None:
        """Send pre-encoded base64 audio to the caller."""
        if self._send_func is not None:
            await self._send_func(payload_b64)
        else:
            self._outbound_buffer.append(payload_b64)

    async def flush_buffer(self) -> None:
        """Send any buffered outbound audio now that send_func is available."""
        if self._send_func is None:
            return
        for payload in self._outbound_buffer:
            await self._send_func(payload)
        self._outbound_buffer.clear()

    async def receive_audio(self, mulaw_bytes: bytes) -> None:
        """Called when inbound audio arrives from the caller."""
        if self._audio_callback is not None:
            try:
                await self._audio_callback(mulaw_bytes)
            except Exception as e:
                logger.error(
                    "Audio callback error for %s: %s", self.call_sid, e
                )


class MediaStreamRegistry:
    """Global registry of active media streams indexed by call_sid."""

    def __init__(self):
        self._streams: dict[str, MediaStream] = {}

    def register(self, call_sid: str) -> MediaStream:
        """Create and register a new media stream for a call."""
        stream = MediaStream(call_sid=call_sid)
        self._streams[call_sid] = stream
        logger.info("Registered media stream for call %s", call_sid)
        return stream

    def get(self, call_sid: str) -> Optional[MediaStream]:
        """Look up an active media stream by call SID."""
        return self._streams.get(call_sid)

    def unregister(self, call_sid: str) -> None:
        """Remove a media stream when the call ends."""
        removed = self._streams.pop(call_sid, None)
        if removed:
            logger.info("Unregistered media stream for call %s", call_sid)

    @property
    def active_count(self) -> int:
        return len(self._streams)


# Module-level singleton
_registry: Optional[MediaStreamRegistry] = None


def get_media_stream_registry() -> MediaStreamRegistry:
    """Get the global media stream registry."""
    global _registry
    if _registry is None:
        _registry = MediaStreamRegistry()
    return _registry
