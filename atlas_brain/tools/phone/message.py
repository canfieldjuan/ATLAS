"""
Take Message tool - Save caller messages for callback.

Stores caller messages and optionally sends notifications
to the business owner for follow-up.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from ..base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.phone.message")


class TakeMessageTool:
    """Tool to take and store caller messages."""

    def __init__(self) -> None:
        # In-memory storage for messages (replace with DB in production)
        self._messages: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "take_message"

    @property
    def description(self) -> str:
        return "Take a message from the caller for callback"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="caller_name",
                param_type="string",
                description="Caller's name",
                required=True,
            ),
            ToolParameter(
                name="caller_phone",
                param_type="string",
                description="Callback phone number",
                required=True,
            ),
            ToolParameter(
                name="message",
                param_type="string",
                description="Message content",
                required=True,
            ),
            ToolParameter(
                name="urgency",
                param_type="string",
                description="Urgency level: normal, urgent",
                required=False,
                default="normal",
            ),
            ToolParameter(
                name="context_id",
                param_type="string",
                description="Business context ID",
                required=False,
            ),
            ToolParameter(
                name="send_notification",
                param_type="bool",
                description="Send notification to owner",
                required=False,
                default=True,
            ),
        ]

    async def _send_notification(
        self,
        caller_name: str,
        caller_phone: str,
        message: str,
        urgency: str,
    ) -> bool:
        """Send notification to business owner."""
        try:
            from ..notify import notify_tool

            notification_msg = (
                f"Message from {caller_name} ({caller_phone}): {message}"
            )

            if urgency == "urgent":
                notification_msg = f"URGENT: {notification_msg}"

            result = await notify_tool.execute({
                "message": notification_msg,
                "title": f"Callback Request - {caller_name}",
            })

            return result.success

        except Exception as e:
            logger.warning("Failed to send notification: %s", e)
            return False

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Take a message and optionally notify owner."""
        caller_name = params.get("caller_name", "Unknown")
        caller_phone = params.get("caller_phone", "")
        message = params.get("message", "")
        urgency = params.get("urgency", "normal")
        context_id = params.get("context_id")
        send_notification = params.get("send_notification", True)

        # Validate required fields
        if not caller_phone:
            return ToolResult(
                success=False,
                error="MISSING_PHONE",
                message="A callback number is required to take a message.",
            )

        if not message:
            return ToolResult(
                success=False,
                error="MISSING_MESSAGE",
                message="Please provide a message to leave.",
            )

        try:
            # Create message record
            timestamp = datetime.now().isoformat()
            message_record = {
                "id": len(self._messages) + 1,
                "timestamp": timestamp,
                "caller_name": caller_name,
                "caller_phone": caller_phone,
                "message": message,
                "urgency": urgency,
                "context_id": context_id,
                "status": "pending",
            }

            # Store message
            self._messages.append(message_record)
            logger.info(
                "Message saved: %s from %s (%s)",
                message[:50],
                caller_name,
                caller_phone,
            )

            # Send notification if enabled
            notification_sent = False
            if send_notification:
                notification_sent = await self._send_notification(
                    caller_name,
                    caller_phone,
                    message,
                    urgency,
                )

            # Build response
            data = {
                "message_id": message_record["id"],
                "timestamp": timestamp,
                "notification_sent": notification_sent,
            }

            response_msg = (
                f"Message received from {caller_name}. "
                f"We will call back at {caller_phone} as soon as possible."
            )

            if urgency == "urgent":
                response_msg = (
                    f"Urgent message received from {caller_name}. "
                    f"We will prioritize your callback at {caller_phone}."
                )

            return ToolResult(
                success=True,
                data=data,
                message=response_msg,
            )

        except Exception as e:
            logger.exception("Error taking message")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    def get_pending_messages(
        self,
        context_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get pending messages for a context."""
        messages = [
            m for m in self._messages
            if m.get("status") == "pending"
        ]

        if context_id:
            messages = [
                m for m in messages
                if m.get("context_id") == context_id
            ]

        return messages


# Module instance
take_message_tool = TakeMessageTool()
