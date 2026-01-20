"""
Notification tool for sending push notifications via ntfy.

Allows the LLM to send notifications to the user's phone on demand.
"""

import logging
from typing import Any

import httpx

from ..config import settings
from .base import Tool, ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.notify")


class NotifyTool:
    """Send push notifications to user's phone via ntfy."""

    def __init__(self) -> None:
        self._config = settings.alerts
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "send_notification"

    @property
    def description(self) -> str:
        return (
            "Send a push notification to the user's phone. "
            "Use this when the user asks you to send them something, "
            "remind them about something, or share information like a list."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                param_type="string",
                description="The notification message to send",
                required=True,
            ),
            ToolParameter(
                name="title",
                param_type="string",
                description="Optional title for the notification",
                required=False,
                default="Atlas",
            ),
            ToolParameter(
                name="priority",
                param_type="string",
                description="Priority: low, default, high, or urgent",
                required=False,
                default="default",
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["notify", "notification", "push notification", "send to phone"]

    @property
    def category(self) -> str:
        return "utility"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Send notification via ntfy."""
        if not self._config.ntfy_enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Notifications are disabled. ntfy is not configured.",
            )

        message = params.get("message")
        if not message:
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="Message is required",
            )

        title = params.get("title", "Atlas")
        priority = params.get("priority", "default")

        # Validate priority
        valid_priorities = ["min", "low", "default", "high", "urgent"]
        if priority not in valid_priorities:
            priority = "default"

        try:
            await self._send_notification(message, title, priority)
            return ToolResult(
                success=True,
                data={"message": message, "title": title, "priority": priority},
                message=f"Notification sent: {message[:50]}{'...' if len(message) > 50 else ''}",
            )
        except httpx.HTTPStatusError as e:
            logger.error("ntfy HTTP error: %s", e)
            return ToolResult(
                success=False,
                error="API_ERROR",
                message=f"Failed to send notification: {e.response.status_code}",
            )
        except Exception as e:
            logger.exception("Notification tool error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def _send_notification(
        self, message: str, title: str, priority: str
    ) -> None:
        """Send notification to ntfy server."""
        client = await self._ensure_client()

        url = f"{self._config.ntfy_url.rstrip('/')}/{self._config.ntfy_topic}"

        headers = {
            "Title": title,
            "Priority": priority,
        }

        response = await client.post(url, content=message, headers=headers)
        response.raise_for_status()
        logger.info("Notification sent via ntfy: %s", message[:50])


# Module-level instance
notify_tool = NotifyTool()
