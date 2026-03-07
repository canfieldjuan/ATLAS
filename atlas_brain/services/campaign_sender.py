"""
Provider-agnostic campaign email sender.

Defines a CampaignSender protocol and ships Resend + Amazon SES implementations.
Swap providers by implementing the protocol and updating the factory.
"""

from __future__ import annotations

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
                    "Resend API error %s: %s", resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()
            data = resp.json()
            logger.info("Resend email sent: id=%s to=%s", data.get("id"), to)
            return {"id": data["id"]}


class SESCampaignSender:
    """Send campaign emails via Amazon SES v2 API."""

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        configuration_set: str | None = None,
        from_email: str,
    ) -> None:
        import boto3

        kwargs: dict[str, Any] = {"region_name": region}
        if access_key_id and secret_access_key:
            kwargs["aws_access_key_id"] = access_key_id
            kwargs["aws_secret_access_key"] = secret_access_key
        self._client = boto3.client("sesv2", **kwargs)
        self._configuration_set = configuration_set
        self._from_email = from_email

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
        import asyncio

        destination: dict[str, Any] = {"ToAddresses": [to]}
        content: dict[str, Any] = {
            "Simple": {
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Html": {"Data": body, "Charset": "UTF-8"}},
            }
        }
        if headers:
            content["Simple"]["Headers"] = [
                {"Name": k, "Value": v} for k, v in headers.items()
            ]

        send_kwargs: dict[str, Any] = {
            "FromEmailAddress": from_email or self._from_email,
            "Destination": destination,
            "Content": content,
        }
        if reply_to:
            send_kwargs["ReplyToAddresses"] = [reply_to]
        if self._configuration_set:
            send_kwargs["ConfigurationSetName"] = self._configuration_set
        if tags:
            send_kwargs["EmailTags"] = [
                {"Name": t["name"], "Value": _TAG_SANITIZE_RE.sub("_", t["value"])}
                for t in tags
            ]

        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: self._client.send_email(**send_kwargs)
        )
        message_id = resp.get("MessageId", "")
        logger.info("SES email sent: id=%s to=%s", message_id, to)
        return {"id": message_id}


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_sender_instance: CampaignSender | None = None


def get_campaign_sender() -> CampaignSender:
    """Return the configured campaign sender (singleton)."""
    global _sender_instance
    if _sender_instance is None:
        cfg = settings.campaign_sequence
        if cfg.sender_type == "ses":
            _sender_instance = SESCampaignSender(
                region=cfg.ses_region,
                access_key_id=cfg.ses_access_key_id or None,
                secret_access_key=cfg.ses_secret_access_key or None,
                configuration_set=cfg.ses_configuration_set or None,
                from_email=cfg.ses_from_email,
            )
        else:
            api_key = cfg.resend_api_key
            if not api_key:
                raise RuntimeError(
                    "ATLAS_CAMPAIGN_SEQ_RESEND_API_KEY is required when sender_type=resend"
                )
            _sender_instance = ResendCampaignSender(api_key)
    return _sender_instance
