"""
PersonaPlex phone call processor.

Handles bidirectional audio streaming using PersonaPlex for
speech-to-speech conversation with tool execution support.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from ..services.personaplex import PersonaPlexService, get_personaplex_config
from .tool_bridge import ToolBridge

logger = logging.getLogger("atlas.comms.personaplex_processor")


@dataclass
class PersonaPlexCallState:
    """State for an active PersonaPlex call."""

    call_sid: str
    from_number: str
    to_number: str
    context_id: str
    stream_sid: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    is_connected: bool = False
    accumulated_text: str = ""


class PersonaPlexProcessor:
    """
    Processes audio for phone calls using PersonaPlex.

    Audio flow:
    1. Receive mulaw 8kHz from SignalWire WebSocket
    2. Convert to Opus 24kHz for PersonaPlex
    3. Stream to PersonaPlex server
    4. Receive Opus 24kHz response from PersonaPlex
    5. Monitor text tokens for tool triggers
    6. Convert response to mulaw 8kHz for SignalWire
    """

    def __init__(
        self,
        call_state: PersonaPlexCallState,
        business_context: Any,
        on_audio_ready: Optional[Callable[[str], None]] = None,
    ):
        self._state = call_state
        self._business_context = business_context
        self._on_audio_ready = on_audio_ready
        self._service = PersonaPlexService()
        self._tool_bridge = ToolBridge(on_tool_result=self._handle_tool_result)
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._response_buffer: bytes = b""

    @property
    def state(self) -> PersonaPlexCallState:
        """Get current call state."""
        return self._state

    @property
    def tool_bridge(self) -> ToolBridge:
        """Get the tool bridge instance."""
        return self._tool_bridge

    async def connect(self) -> bool:
        """Connect to PersonaPlex server."""
        text_prompt = self._build_text_prompt()
        voice_prompt = self._get_voice_prompt()

        self._service.set_text_callback(self._on_text_token)
        self._service.set_audio_callback(self._on_audio_response)

        connected = await self._service.connect(
            text_prompt=text_prompt,
            voice_prompt=voice_prompt,
        )
        if connected:
            self._state.is_connected = True
            logger.info(
                "PersonaPlex connected for call %s (voice=%s)",
                self._state.call_sid,
                voice_prompt,
            )
        return connected

    async def disconnect(self) -> None:
        """Disconnect from PersonaPlex server."""
        self._state.is_connected = False
        await self._service.disconnect()
        logger.info("PersonaPlex disconnected for call %s", self._state.call_sid)

    def _build_text_prompt(self) -> str:
        """Build PersonaPlex text prompt from business context."""
        if not self._business_context:
            return "You are a helpful assistant."

        parts = []
        ctx = self._business_context

        if hasattr(ctx, "name"):
            parts.append(f"You are the receptionist for {ctx.name}.")

        if hasattr(ctx, "persona") and ctx.persona:
            parts.append(ctx.persona)

        if hasattr(ctx, "greeting"):
            parts.append(f"Greet callers with: {ctx.greeting}")

        if hasattr(ctx, "services") and ctx.services:
            services = ", ".join(ctx.services)
            parts.append(f"Services offered: {services}.")

        if hasattr(ctx, "pricing_info") and ctx.pricing_info:
            parts.append(f"Pricing: {ctx.pricing_info}")

        if hasattr(ctx, "service_area") and ctx.service_area:
            parts.append(f"Service area: {ctx.service_area}.")

        parts.append("Collect customer name, phone number, preferred date and time.")
        parts.append("When ready to book, say 'let me book that for you'.")

        return " ".join(parts) if parts else "You are a helpful assistant."

    def _get_voice_prompt(self) -> str:
        """Get PersonaPlex voice prompt based on business context."""
        config = get_personaplex_config()
        default_voice = config.voice_prompt

        if not self._business_context:
            return default_voice

        voice_name = getattr(self._business_context, "voice_name", "").lower()
        if "female" in voice_name or voice_name in ["sarah", "bella", "heart"]:
            return "NATF0"
        elif "male" in voice_name or voice_name in ["adam", "michael"]:
            return "NATM0"

        return default_voice

    def _on_text_token(self, text: str) -> None:
        """Handle text token from PersonaPlex."""
        self._state.accumulated_text += text
        self._tool_bridge.process_text(text)
        asyncio.create_task(self._check_tools())

    def _on_audio_response(self, opus_data: bytes) -> None:
        """Handle audio response from PersonaPlex."""
        logger.info(
            "PersonaPlex audio response: %d bytes opus",
            len(opus_data),
        )
        try:
            converter = self._service.audio_converter
            mulaw_data = converter.personaplex_to_signalwire(opus_data)
            logger.info(
                "Converted to mulaw: %d bytes",
                len(mulaw_data) if mulaw_data else 0,
            )
            if mulaw_data and self._on_audio_ready:
                mulaw_b64 = base64.b64encode(mulaw_data).decode("ascii")
                logger.info("Sending audio to caller via callback")
                self._on_audio_ready(mulaw_b64)
            elif not mulaw_data:
                logger.warning("No mulaw data after conversion")
            elif not self._on_audio_ready:
                logger.warning("No audio callback set")
        except RuntimeError as e:
            logger.error("Audio conversion failed: %s", e)

    async def _check_tools(self) -> None:
        """Check if tool execution is needed."""
        result = await self._tool_bridge.check_and_execute_tools()
        if result:
            logger.info("Tool executed: %s", result.message)

    def _handle_tool_result(self, tool_name: str, result: Any) -> None:
        """Handle result from tool execution."""
        message = self._tool_bridge.format_tool_result(tool_name, result)
        logger.info("Tool result for %s: %s", tool_name, message)

    async def process_audio_chunk(self, mulaw_b64: str) -> Optional[str]:
        """
        Process incoming audio chunk from SignalWire.

        Args:
            mulaw_b64: Base64-encoded mulaw audio from SignalWire

        Returns:
            None (audio responses are sent via callback)
        """
        if not self._state.is_connected:
            logger.warning("PersonaPlex not connected, dropping audio")
            return None

        mulaw_data = base64.b64decode(mulaw_b64)
        logger.debug(
            "Sending %d bytes mulaw to PersonaPlex",
            len(mulaw_data),
        )
        sent = await self._service.send_audio_mulaw(mulaw_data)
        if not sent:
            logger.warning("Failed to send audio to PersonaPlex")
        return None


_active_personaplex_calls: dict[str, PersonaPlexProcessor] = {}


def get_personaplex_processor(call_sid: str) -> Optional[PersonaPlexProcessor]:
    """Get PersonaPlex processor for an active call."""
    return _active_personaplex_calls.get(call_sid)


def create_personaplex_processor(
    call_sid: str,
    from_number: str,
    to_number: str,
    context_id: str,
    business_context: Any,
    on_audio_ready: Optional[Callable[[str], None]] = None,
) -> PersonaPlexProcessor:
    """Create a new PersonaPlex call processor."""
    state = PersonaPlexCallState(
        call_sid=call_sid,
        from_number=from_number,
        to_number=to_number,
        context_id=context_id,
    )
    processor = PersonaPlexProcessor(
        call_state=state,
        business_context=business_context,
        on_audio_ready=on_audio_ready,
    )
    _active_personaplex_calls[call_sid] = processor
    logger.info("Created PersonaPlex processor for %s", call_sid)
    return processor


async def remove_personaplex_processor(call_sid: str) -> None:
    """Remove and disconnect PersonaPlex processor when call ends."""
    processor = _active_personaplex_calls.get(call_sid)
    if processor:
        await processor.disconnect()
        del _active_personaplex_calls[call_sid]
        logger.info("Removed PersonaPlex processor for %s", call_sid)


def is_personaplex_enabled() -> bool:
    """Check if PersonaPlex is enabled for phone calls."""
    from . import comms_settings
    return comms_settings.personaplex_enabled
