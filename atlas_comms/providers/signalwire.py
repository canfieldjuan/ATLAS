"""
SignalWire telephony provider implementation.

SignalWire is a Twilio-compatible API with lower pricing.
Uses RELAY for real-time communication and REST API for SMS.

Requirements:
    pip install signalwire

SignalWire Concepts:
- RELAY: Real-time WebSocket protocol for voice
- LaML: SignalWire's markup language (Twilio TwiML compatible)
- Spaces: Your SignalWire project namespace
"""

import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from ..core.protocols import (
    TelephonyProvider,
    Call,
    CallState,
    CallDirection,
    SMSMessage,
    SMSDirection,
    AudioChunkCallback,
    CallEventCallback,
    SMSCallback,
)
from ..core.config import comms_settings
from . import register_provider

logger = logging.getLogger("atlas.comms.providers.signalwire")


@register_provider("signalwire")
class SignalWireProvider(TelephonyProvider):
    """
    SignalWire implementation of the TelephonyProvider protocol.

    Uses SignalWire's REST API for call/SMS control and
    RELAY protocol for real-time audio streaming.
    """

    def __init__(self):
        self._client = None
        self._connected = False
        self._space_url = ""

        # Active calls by provider call ID
        self._calls: dict[str, Call] = {}

        # WebSocket connections for audio streaming
        self._audio_streams: dict[str, asyncio.Task] = {}

        # Callbacks
        self._call_event_callback: Optional[CallEventCallback] = None
        self._sms_callback: Optional[SMSCallback] = None
        self._audio_callbacks: dict[str, AudioChunkCallback] = {}

    @property
    def name(self) -> str:
        return "signalwire"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Initialize SignalWire client."""
        try:
            from signalwire.rest import Client

            project_id = comms_settings.signalwire_project_id
            api_token = comms_settings.signalwire_api_token
            space = comms_settings.signalwire_space

            if not project_id or not api_token or not space:
                raise ValueError(
                    "SignalWire credentials not configured. Set "
                    "ATLAS_COMMS_SIGNALWIRE_PROJECT_ID, "
                    "ATLAS_COMMS_SIGNALWIRE_API_TOKEN, and "
                    "ATLAS_COMMS_SIGNALWIRE_SPACE"
                )

            self._space_url = f"https://{space}.signalwire.com"
            self._client = Client(
                project_id,
                api_token,
                signalwire_space_url=self._space_url,
            )
            self._connected = True
            logger.info("SignalWire provider connected to %s", self._space_url)

        except ImportError:
            raise ImportError(
                "SignalWire package not installed. Run: pip install signalwire"
            )

    async def disconnect(self) -> None:
        """Disconnect from SignalWire."""
        # Cancel any active audio streams
        for task in self._audio_streams.values():
            task.cancel()
        self._audio_streams.clear()

        self._client = None
        self._connected = False
        self._calls.clear()
        logger.info("SignalWire provider disconnected")

    async def answer_call(self, call: Call) -> bool:
        """
        Answer an incoming call.

        In SignalWire (like Twilio), answering is implicit when
        webhook returns LaML. This updates our tracking.
        """
        if call.provider_call_id not in self._calls:
            self._calls[call.provider_call_id] = call

        call.state = CallState.CONNECTED
        call.answered_at = datetime.now(timezone.utc)

        if self._call_event_callback:
            await self._call_event_callback(call, "answered")

        return True

    async def reject_call(self, call: Call, reason: str = "") -> bool:
        """Reject an incoming call."""
        if not self._client:
            return False

        try:
            self._client.calls(call.provider_call_id).update(
                status="completed"
            )
            call.state = CallState.ENDED

            if self._call_event_callback:
                await self._call_event_callback(call, "rejected")

            return True

        except Exception as e:
            logger.error("Failed to reject call: %s", e)
            return False

    async def make_call(
        self,
        to_number: str,
        from_number: str,
        context_id: Optional[str] = None,
    ) -> Call:
        """Initiate an outbound call."""
        if not self._client:
            raise RuntimeError("Provider not connected")

        call = Call(
            from_number=from_number,
            to_number=to_number,
            direction=CallDirection.OUTBOUND,
            state=CallState.INITIATED,
            context_id=context_id,
        )

        try:
            webhook_url = f"{comms_settings.webhook_base_url}/api/v1/comms/voice/outbound"
            status_url = f"{comms_settings.webhook_base_url}/api/v1/comms/voice/status"

            sw_call = self._client.calls.create(
                to=to_number,
                from_=from_number,
                url=webhook_url,
                status_callback=status_url,
                status_callback_event=["initiated", "ringing", "answered", "completed"],
            )

            call.provider_call_id = sw_call.sid
            self._calls[sw_call.sid] = call

            logger.info(
                "Initiated outbound call %s: %s -> %s",
                sw_call.sid,
                from_number,
                to_number,
            )

            return call

        except Exception as e:
            logger.error("Failed to make call: %s", e)
            call.state = CallState.FAILED
            raise

    async def hangup(self, call: Call) -> bool:
        """End an active call."""
        if not self._client:
            return False

        try:
            self._client.calls(call.provider_call_id).update(
                status="completed"
            )
            call.state = CallState.ENDED
            call.ended_at = datetime.now(timezone.utc)

            # Clean up audio stream if active
            if call.provider_call_id in self._audio_streams:
                self._audio_streams[call.provider_call_id].cancel()
                del self._audio_streams[call.provider_call_id]

            if self._call_event_callback:
                await self._call_event_callback(call, "ended")

            return True

        except Exception as e:
            logger.error("Failed to hangup call: %s", e)
            return False

    async def hold(self, call: Call) -> bool:
        """Place call on hold."""
        call.state = CallState.ON_HOLD
        logger.info("Call %s placed on hold", call.provider_call_id)
        return True

    async def unhold(self, call: Call) -> bool:
        """Take call off hold."""
        call.state = CallState.CONNECTED
        logger.info("Call %s taken off hold", call.provider_call_id)
        return True

    async def transfer(self, call: Call, to_number: str) -> bool:
        """Transfer call to another number."""
        if not self._client:
            return False

        try:
            # LaML (compatible with TwiML) for transfer
            transfer_laml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Please hold while I transfer your call.</Say>
    <Dial>{to_number}</Dial>
</Response>"""

            self._client.calls(call.provider_call_id).update(
                twiml=transfer_laml
            )

            call.state = CallState.TRANSFERRING

            if self._call_event_callback:
                await self._call_event_callback(call, "transferring")

            logger.info(
                "Transferring call %s to %s",
                call.provider_call_id,
                to_number,
            )
            return True

        except Exception as e:
            logger.error("Failed to transfer call: %s", e)
            return False

    async def stream_audio_to_call(
        self,
        call: Call,
        audio_iterator: AsyncIterator[bytes],
    ) -> None:
        """
        Stream TTS audio to the call.

        SignalWire supports media streaming via WebSocket.
        Audio should be base64-encoded mulaw at 8kHz.
        """
        # TODO: Implement via SignalWire's Stream noun
        # This requires bidirectional WebSocket to media stream
        logger.warning("Audio streaming implementation pending")
        async for chunk in audio_iterator:
            # Will send to WebSocket when implemented
            pass

    def set_audio_callback(
        self,
        call: Call,
        callback: AudioChunkCallback,
    ) -> None:
        """Set callback for receiving audio from the call."""
        self._audio_callbacks[call.provider_call_id] = callback

    async def send_sms(
        self,
        to_number: str,
        from_number: str,
        body: str,
        media_urls: Optional[list[str]] = None,
        context_id: Optional[str] = None,
    ) -> SMSMessage:
        """Send an SMS message."""
        if not self._client:
            raise RuntimeError("Provider not connected")

        message = SMSMessage(
            from_number=from_number,
            to_number=to_number,
            direction=SMSDirection.OUTBOUND,
            body=body,
            media_urls=media_urls or [],
            context_id=context_id,
            status="pending",
        )

        try:
            params = {
                "to": to_number,
                "from_": from_number,
                "body": body,
            }

            if media_urls:
                params["media_url"] = media_urls

            if comms_settings.webhook_base_url:
                params["status_callback"] = (
                    f"{comms_settings.webhook_base_url}/api/v1/comms/sms/status"
                )

            sw_message = self._client.messages.create(**params)

            message.provider_message_id = sw_message.sid
            message.status = "sent"

            logger.info(
                "Sent SMS %s: %s -> %s",
                sw_message.sid,
                from_number,
                to_number,
            )

            return message

        except Exception as e:
            logger.error("Failed to send SMS: %s", e)
            message.status = "failed"
            message.error_message = str(e)
            raise

    def set_call_event_callback(self, callback: CallEventCallback) -> None:
        """Set callback for call events."""
        self._call_event_callback = callback

    def set_sms_callback(self, callback: SMSCallback) -> None:
        """Set callback for incoming SMS."""
        self._sms_callback = callback

    async def get_call(self, provider_call_id: str) -> Optional[Call]:
        """Get a call by provider ID."""
        return self._calls.get(provider_call_id)

    async def lookup_caller_id(self, phone_number: str) -> Optional[str]:
        """
        Look up caller ID name.

        SignalWire provides CNAM lookup as a separate service.
        """
        # SignalWire CNAM lookup would go here
        # For now, return None
        return None

    # === Webhook Handlers ===

    async def handle_incoming_call(
        self,
        call_sid: str,
        from_number: str,
        to_number: str,
        caller_name: Optional[str] = None,
    ) -> Call:
        """Handle an incoming call webhook."""
        call = Call(
            provider_call_id=call_sid,
            from_number=from_number,
            to_number=to_number,
            direction=CallDirection.INBOUND,
            state=CallState.RINGING,
            caller_name=caller_name,
        )

        self._calls[call_sid] = call

        if self._call_event_callback:
            await self._call_event_callback(call, "ringing")

        logger.info(
            "Incoming call %s: %s -> %s",
            call_sid,
            from_number,
            to_number,
        )

        return call

    async def handle_call_status(
        self,
        call_sid: str,
        status: str,
    ) -> None:
        """Handle a call status webhook."""
        call = self._calls.get(call_sid)
        if not call:
            logger.warning("Status update for unknown call: %s", call_sid)
            return

        status_map = {
            "initiated": CallState.INITIATED,
            "ringing": CallState.RINGING,
            "in-progress": CallState.CONNECTED,
            "completed": CallState.ENDED,
            "busy": CallState.FAILED,
            "failed": CallState.FAILED,
            "no-answer": CallState.FAILED,
            "canceled": CallState.ENDED,
        }

        new_state = status_map.get(status, call.state)

        if new_state == CallState.ENDED and call.state != CallState.ENDED:
            call.ended_at = datetime.now(timezone.utc)

        call.state = new_state

        if self._call_event_callback:
            await self._call_event_callback(call, status)

        logger.debug("Call %s status: %s", call_sid, status)

    async def handle_incoming_sms(
        self,
        message_sid: str,
        from_number: str,
        to_number: str,
        body: str,
        media_urls: Optional[list[str]] = None,
    ) -> SMSMessage:
        """Handle an incoming SMS webhook."""
        message = SMSMessage(
            provider_message_id=message_sid,
            from_number=from_number,
            to_number=to_number,
            direction=SMSDirection.INBOUND,
            body=body,
            media_urls=media_urls or [],
            status="received",
        )

        if self._sms_callback:
            await self._sms_callback(message)

        logger.info(
            "Incoming SMS %s: %s -> %s: %s",
            message_sid,
            from_number,
            to_number,
            body[:50] + "..." if len(body) > 50 else body,
        )

        return message

    async def handle_media_stream(
        self,
        call_sid: str,
        stream_data: dict,
    ) -> None:
        """
        Handle incoming media stream data.

        Called when audio data arrives via WebSocket.
        """
        call = self._calls.get(call_sid)
        if not call:
            return

        callback = self._audio_callbacks.get(call_sid)
        if not callback:
            return

        event = stream_data.get("event")

        if event == "media":
            # Decode audio payload
            payload = stream_data.get("media", {}).get("payload", "")
            if payload:
                audio_bytes = base64.b64decode(payload)
                await callback(audio_bytes)

        elif event == "start":
            logger.debug("Media stream started for call %s", call_sid)

        elif event == "stop":
            logger.debug("Media stream stopped for call %s", call_sid)

    def generate_stream_laml(self, call: Call) -> str:
        """
        Generate LaML to start bidirectional audio streaming.

        Returns LaML that tells SignalWire to connect audio to our WebSocket.
        """
        ws_url = comms_settings.webhook_base_url.replace("https://", "wss://").replace("http://", "ws://")
        stream_url = f"{ws_url}/api/v1/comms/voice/stream/{call.provider_call_id}"

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
