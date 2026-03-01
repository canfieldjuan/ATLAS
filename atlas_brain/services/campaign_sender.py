"""
Provider-agnostic campaign email sender.

Defines a CampaignSender protocol and ships a Resend implementation.
Swap providers by implementing the protocol and updating the factory.
"""

import logging
import re
from typing import Any, Protocol, runtime_checkable

import httpx

from ..config import settings

logger = logging.getLogger("atlas.services.campaign_sender")

_RESEND_API_URL = "https://api.resend.com/emails"
_TAG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")


@runtime_checkable
class CampaignSender(Protocol):
    """Protocol for sending campaign emails via an ESP."""

    async def send(
        self,
        *,
        to: str,
        from_email: str,
        subject: str,
        body: str,
        reply_to: str | None = None,
        headers: dict[str, str] | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Send a campaign email. Returns at least ``{"id": "<esp_message_id>"}``."""
        ...


class ResendCampaignSender:
    """Send campaign emails via the Resend REST API.

    Resend automatically injects an open-tracking pixel into HTML emails
    and rewrites links for click tracking when tracking is enabled on
    the domain.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def send(
        self,
        *,
        to: str,
        from_email: str,
        subject: str,
        body: str,
        reply_to: str | None = None,
        headers: dict[str, str] | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "from": from_email,
            "to": [to],
            "subject": subject,
            "html": body,
        }
        if reply_to:
            payload["reply_to"] = reply_to
        if headers:
            payload["headers"] = headers
        if tags:
            payload["tags"] = [
                {"name": t["name"], "value": _TAG_SANITIZE_RE.sub("_", t["value"])}
                for t in tags
            ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                _RESEND_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code >= 400:
                logger.error(
                    "Resend API error %s: %s", resp.status_code, resp.text,
                )
            resp.raise_for_status()
            data = resp.json()
            logger.info("Resend email sent: id=%s to=%s", data.get("id"), to)
            return {"id": data["id"]}


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_sender_instance: CampaignSender | None = None


def get_campaign_sender() -> CampaignSender:
    """Return the configured campaign sender (singleton)."""
    global _sender_instance
    if _sender_instance is None:
        api_key = settings.campaign_sequence.resend_api_key
        if not api_key:
            raise RuntimeError(
                "ATLAS_CAMPAIGN_SEQ_RESEND_API_KEY is required when campaign sequences are enabled"
            )
        _sender_instance = ResendCampaignSender(api_key)
    return _sender_instance
