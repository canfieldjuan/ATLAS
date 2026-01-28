"""
Twilio telephony provider implementation.

Handles voice calls and SMS through Twilio's API.

Requirements:
    pip install twilio

Twilio Concepts:
- Media Streams: WebSocket connection for real-time audio
- TwiML: XML-based instructions for call handling
- Webhooks: HTTP callbacks for call/SMS events
"""

import asyncio
import logging
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

logger = logging.getLogger("atlas.comms.providers.twilio")


@register_provider("twilio")
class TwilioProvider(TelephonyProvider):
    """
    Twilio implementation of the TelephonyProvider protocol.

    Uses Twilio's REST API for call control and Media Streams
    for real-time audio streaming.
    """

    def __init__(self):
        self._client = None
        self._connected = False

        # Active calls by provider call ID
        self._calls: dict[str, Call] = {}

        # Callbacks
        self._call_event_callback: Optional[CallEventCallback] = None
        self._sms_callback: Optional[SMSCallback] = None
        self._audio_callbacks: dict[str, AudioChunkCallback] = {}

    @property
    def name(self) -> str:
        return "twilio"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Initialize Twilio client."""
        try:
            from twilio.rest import Client

            account_sid = comms_settings.twilio_account_sid
            auth_token = comms_settings.twilio_auth_token

            if not account_sid or not auth_token:
                raise ValueError(
                    "Twilio credentials not configured. "
                    "Set ATLAS_COMMS_TWILIO_ACCOUNT_SID and ATLAS_COMMS_TWILIO_AUTH_TOKEN"
                )

            self._client = Client(account_sid, auth_token)
            self._connected = True
            logger.info("Twilio provider connected")

        except ImportError:
            raise ImportError(
                "Twilio package not installed. Run: pip install twilio"
            )

    async def disconnect(self) -> None:
        """Disconnect from Twilio."""
        self._client = None
        self._connected = False
        self._calls.clear()
        logger.info("Twilio provider disconnected")

    async def answer_call(self, call: Call) -> bool:
        """
        Answer an incoming call.

        Note: In Twilio, answering is implicit - when the webhook returns
        TwiML, the call is answered. This method updates our tracking.
        """
        if call.provider_call_id not in self._calls:
            self._calls[call.provider_call_id] = call

        call.state = CallState.CONNECTED
        call.answered_at = asyncio.get_event_loop().time()

        if self._call_event_callback:
            await self._call_event_callback(call, "answered")

        return True

    async def reject_call(self, call: Call, reason: str = "") -> bool:
        """Reject an incoming call."""
        if not self._client:
            return False

        try:
            # Update the call to reject it
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
        """
        Initiate an outbound call.

        The webhook_base_url must be configured to handle the call flow.
        """
        if not self._client:
            raise RuntimeError("Provider not connected")

        # Create call tracking object
        call = Call(
            from_number=from_number,
            to_number=to_number,
            direction=CallDirection.OUTBOUND,
            state=CallState.INITIATED,
            context_id=context_id,
        )

        try:
            # Make the call via Twilio
            webhook_url = f"{comms_settings.webhook_base_url}/api/v1/comms/voice/outbound"

            twilio_call = self._client.calls.create(
                to=to_number,
                from_=from_number,
                url=webhook_url,
                status_callback=f"{comms_settings.webhook_base_url}/api/v1/comms/voice/status",
                status_callback_event=["initiated", "ringing", "answered", "completed"],
            )

            call.provider_call_id = twilio_call.sid
            self._calls[twilio_call.sid] = call

            logger.info(
                "Initiated outbound call %s: %s -> %s",
                twilio_call.sid,
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

            if self._call_event_callback:
                await self._call_event_callback(call, "ended")

            return True

        except Exception as e:
            logger.error("Failed to hangup call: %s", e)
            return False

    async def hold(self, call: Call) -> bool:
        """Place call on hold (plays hold music)."""
        # Twilio hold is done via TwiML <Play> or <Say> in a loop
        # This would require updating the call's TwiML
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
            # Update call with new TwiML to dial the transfer number
            transfer_twiml = f"""
            <Response>
                <Say>Please hold while I transfer your call.</Say>
                <Dial>{to_number}</Dial>
            </Response>
            """

            self._client.calls(call.provider_call_id).update(
                twiml=transfer_twiml
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
        Stream audio to the call.

        This is used to play TTS responses. In Twilio, this is done
        through Media Streams WebSocket.
        """
        # TODO: Implement Media Streams integration
        # This requires a WebSocket connection to the media stream
        logger.warning("Audio streaming not yet implemented")
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
            # Build message params
            params = {
                "to": to_number,
                "from_": from_number,
                "body": body,
            }

            if media_urls:
                params["media_url"] = media_urls

            # Add status callback
            if comms_settings.webhook_base_url:
                params["status_callback"] = (
                    f"{comms_settings.webhook_base_url}/api/v1/comms/sms/status"
                )

            twilio_message = self._client.messages.create(**params)

            message.provider_message_id = twilio_message.sid
            message.status = "sent"

            logger.info(
                "Sent SMS %s: %s -> %s",
                twilio_message.sid,
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
        """Look up caller ID name."""
        if not self._client:
            return None

        try:
            # Twilio Lookup API
            lookup = self._client.lookups.v2.phone_numbers(phone_number).fetch(
                fields="caller_name"
            )
            return lookup.caller_name.get("caller_name") if lookup.caller_name else None

        except Exception as e:
            logger.debug("Caller ID lookup failed for %s: %s", phone_number, e)
            return None

    # === Webhook Handlers ===
    # These are called by the API endpoints when Twilio sends webhooks

    async def handle_incoming_call(
        self,
        call_sid: str,
        from_number: str,
        to_number: str,
        caller_name: Optional[str] = None,
    ) -> Call:
        """
        Handle an incoming call webhook.

        Returns the Call object for further processing.
        """
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

        # Map Twilio status to our CallState
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
