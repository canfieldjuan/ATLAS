"""
Main Communications Service.

Orchestrates telephony providers, context routing, and voice pipeline integration.
Manages the lifecycle of calls and SMS conversations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable

from .core.config import comms_settings, BusinessContext, EFFINGHAM_MAIDS_CONTEXT
from .context import get_context_router, ContextRouter
from .core.protocols import (
    TelephonyProvider,
    Call,
    CallState,
    SMSMessage,
)
from .providers import get_provider

logger = logging.getLogger("atlas.comms.service")

# Type for message handlers
MessageHandler = Callable[[str, BusinessContext], Awaitable[str]]


class CommsService:
    """
    Main communications service.

    Manages:
    - Provider lifecycle (connect/disconnect)
    - Context routing for incoming calls/SMS
    - Voice pipeline integration (via atlas_brain HTTP)
    - Appointment scheduling
    """

    def __init__(self):
        self._provider: Optional[TelephonyProvider] = None
        self._context_router: Optional[ContextRouter] = None
        self._connected = False

        # Message handler for SMS auto-replies (LLM integration)
        self._sms_handler: Optional[MessageHandler] = None

        # Active calls tracking
        self._active_calls: dict[str, Call] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._provider is not None

    @property
    def provider(self) -> Optional[TelephonyProvider]:
        return self._provider

    @property
    def context_router(self) -> ContextRouter:
        if self._context_router is None:
            self._context_router = get_context_router()
        return self._context_router

    async def start(self) -> bool:
        """
        Start the communications service.

        Initializes provider, registers contexts, and sets up callbacks.
        """
        if not comms_settings.enabled:
            logger.info("Communications service disabled")
            return False

        try:
            # Get and connect provider
            self._provider = get_provider()
            await self._provider.connect()

            # Register business contexts
            self._register_contexts()

            # Set up callbacks
            self._provider.set_call_event_callback(self._on_call_event)
            self._provider.set_sms_callback(self._on_sms_received)

            self._connected = True
            logger.info(
                "Communications service started with provider: %s",
                self._provider.name,
            )
            return True

        except Exception as e:
            logger.error("Failed to start communications service: %s", e)
            return False

    async def stop(self) -> None:
        """Stop the communications service."""
        if self._provider:
            try:
                await self._provider.disconnect()
            except Exception as e:
                logger.error("Error disconnecting provider: %s", e)

        self._provider = None
        self._connected = False
        self._active_calls.clear()

        logger.info("Communications service stopped")

    def _register_contexts(self) -> None:
        """Register business contexts from configuration."""
        router = self.context_router

        # Register Effingham Office Maids if phone numbers are configured
        if EFFINGHAM_MAIDS_CONTEXT.phone_numbers:
            router.register_context(EFFINGHAM_MAIDS_CONTEXT)
            logger.info(
                "Registered context: %s with numbers %s",
                EFFINGHAM_MAIDS_CONTEXT.id,
                EFFINGHAM_MAIDS_CONTEXT.phone_numbers,
            )

        # TODO: Load additional contexts from database

    def set_sms_handler(self, handler: MessageHandler) -> None:
        """
        Set handler for generating SMS responses.

        Handler receives message text and context, returns response text.
        """
        self._sms_handler = handler

    async def _on_call_event(self, call: Call, event: str) -> None:
        """Handle call state changes."""
        logger.info("Call event: %s - %s", call.provider_call_id, event)

        if event in ("ringing", "answered"):
            self._active_calls[call.provider_call_id] = call
        elif event in ("ended", "failed", "completed"):
            self._active_calls.pop(call.provider_call_id, None)

            # Log call summary
            if call.duration_seconds:
                logger.info(
                    "Call %s ended after %.1f seconds",
                    call.provider_call_id,
                    call.duration_seconds,
                )

    async def _on_sms_received(self, message: SMSMessage) -> None:
        """Handle incoming SMS messages."""
        logger.info(
            "SMS received from %s: %s",
            message.from_number,
            message.body[:50],
        )

        # Get context for this number
        context = self.context_router.get_context_for_number(message.to_number)
        message.context_id = context.id

        # Generate auto-reply if enabled
        if context.sms_auto_reply and context.sms_enabled and self._sms_handler:
            try:
                response_text = await self._sms_handler(message.body, context)

                if response_text and self._provider:
                    await self._provider.send_sms(
                        to_number=message.from_number,
                        from_number=message.to_number,
                        body=response_text,
                        context_id=context.id,
                    )
                    logger.info("Sent auto-reply to %s", message.from_number)

            except Exception as e:
                logger.error("Failed to send SMS auto-reply: %s", e)

    async def make_call(
        self,
        to_number: str,
        context_id: Optional[str] = None,
    ) -> Optional[Call]:
        """
        Make an outbound call.

        Args:
            to_number: Phone number to call
            context_id: Business context (determines from number and persona)

        Returns:
            Call object if initiated successfully
        """
        if not self.is_connected:
            logger.error("Cannot make call: service not connected")
            return None

        # Get context and from number
        context = None
        from_number = None

        if context_id:
            context = self.context_router.get_context(context_id)
            if context and context.phone_numbers:
                from_number = context.phone_numbers[0]

        if not from_number:
            logger.error("No from_number available for call")
            return None

        try:
            call = await self._provider.make_call(
                to_number=to_number,
                from_number=from_number,
                context_id=context_id,
            )
            self._active_calls[call.provider_call_id] = call
            return call

        except Exception as e:
            logger.error("Failed to make call: %s", e)
            return None

    async def send_sms(
        self,
        to_number: str,
        body: str,
        context_id: Optional[str] = None,
        media_urls: Optional[list[str]] = None,
    ) -> Optional[SMSMessage]:
        """
        Send an SMS message.

        Args:
            to_number: Recipient phone number
            body: Message text
            context_id: Business context (determines from number)
            media_urls: Optional MMS attachments

        Returns:
            SMSMessage object if sent successfully
        """
        if not self.is_connected:
            logger.error("Cannot send SMS: service not connected")
            return None

        # Get context and from number
        from_number = None

        if context_id:
            context = self.context_router.get_context(context_id)
            if context and context.phone_numbers:
                from_number = context.phone_numbers[0]

        if not from_number:
            logger.error("No from_number available for SMS")
            return None

        try:
            message = await self._provider.send_sms(
                to_number=to_number,
                from_number=from_number,
                body=body,
                media_urls=media_urls,
                context_id=context_id,
            )
            return message

        except Exception as e:
            logger.error("Failed to send SMS: %s", e)
            return None

    async def hangup_call(self, call_id: str) -> bool:
        """Hang up an active call."""
        if not self.is_connected:
            return False

        call = self._active_calls.get(call_id)
        if not call:
            logger.warning("Call not found: %s", call_id)
            return False

        try:
            return await self._provider.hangup(call)
        except Exception as e:
            logger.error("Failed to hangup call: %s", e)
            return False

    async def transfer_call(self, call_id: str, to_number: str) -> bool:
        """Transfer an active call to another number."""
        if not self.is_connected:
            return False

        call = self._active_calls.get(call_id)
        if not call:
            logger.warning("Call not found: %s", call_id)
            return False

        try:
            return await self._provider.transfer(call, to_number)
        except Exception as e:
            logger.error("Failed to transfer call: %s", e)
            return False

    def get_active_calls(self) -> list[Call]:
        """Get list of currently active calls."""
        return list(self._active_calls.values())

    def get_call(self, call_id: str) -> Optional[Call]:
        """Get a specific call by ID."""
        return self._active_calls.get(call_id)


# Module-level service instance
_comms_service: Optional[CommsService] = None


def get_comms_service() -> CommsService:
    """Get or create the global comms service."""
    global _comms_service
    if _comms_service is None:
        _comms_service = CommsService()
    return _comms_service


async def init_comms_service() -> Optional[CommsService]:
    """Initialize and start the communications service."""
    service = get_comms_service()
    if await service.start():
        return service
    return None


async def shutdown_comms_service() -> None:
    """Shutdown the communications service."""
    global _comms_service
    if _comms_service:
        await _comms_service.stop()
        _comms_service = None
