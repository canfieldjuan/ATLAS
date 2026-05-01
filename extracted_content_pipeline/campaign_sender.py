"""Standalone campaign email sender adapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Any, Mapping

import httpx

from .campaign_ports import SendRequest, SendResult


RESEND_API_URL = "https://api.resend.com/emails"
_TAG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")


def sanitize_tag_value(value: str) -> str:
    return _TAG_SANITIZE_RE.sub("_", str(value))


def normalize_tags(tags: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in tags or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if not name or not value:
            continue
        normalized.append({"name": name, "value": sanitize_tag_value(value)})
    return normalized


@dataclass(frozen=True)
class ResendSenderConfig:
    api_key: str
    api_url: str = RESEND_API_URL
    timeout_seconds: float = 30.0


class ResendCampaignSender:
    """Send campaign emails through Resend without Atlas settings."""

    def __init__(
        self,
        config: ResendSenderConfig,
        *,
        http_client: Any | None = None,
    ) -> None:
        if not config.api_key:
            raise ValueError("Resend api_key is required")
        self._config = config
        self._http_client = http_client

    async def send(self, request: SendRequest) -> SendResult:
        payload: dict[str, Any] = {
            "from": request.from_email,
            "to": [request.to_email],
            "subject": request.subject,
            "html": request.html_body,
        }
        if request.text_body:
            payload["text"] = request.text_body
        if request.reply_to:
            payload["reply_to"] = request.reply_to
        if request.headers:
            payload["headers"] = dict(request.headers)
        tags = normalize_tags(request.tags)
        if tags:
            payload["tags"] = tags

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        if self._http_client is not None:
            response = await self._http_client.post(
                self._config.api_url,
                json=payload,
                headers=headers,
            )
        else:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await client.post(
                    self._config.api_url,
                    json=payload,
                    headers=headers,
                )
        response.raise_for_status()
        data = response.json()
        return SendResult(provider="resend", message_id=str(data["id"]), raw=data)


@dataclass(frozen=True)
class SESSenderConfig:
    from_email: str
    region: str = "us-east-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None
    configuration_set: str | None = None


class SESCampaignSender:
    """Send campaign emails through Amazon SES without Atlas settings."""

    def __init__(
        self,
        config: SESSenderConfig,
        *,
        client: Any | None = None,
    ) -> None:
        if not config.from_email:
            raise ValueError("SES from_email is required")
        self._config = config
        if client is not None:
            self._client = client
            return

        import boto3

        kwargs: dict[str, Any] = {"region_name": config.region}
        if config.access_key_id and config.secret_access_key:
            kwargs["aws_access_key_id"] = config.access_key_id
            kwargs["aws_secret_access_key"] = config.secret_access_key
        self._client = boto3.client("sesv2", **kwargs)

    async def send(self, request: SendRequest) -> SendResult:
        body: dict[str, Any] = {
            "Html": {"Data": request.html_body, "Charset": "UTF-8"},
        }
        if request.text_body:
            body["Text"] = {"Data": request.text_body, "Charset": "UTF-8"}

        simple: dict[str, Any] = {
            "Subject": {"Data": request.subject, "Charset": "UTF-8"},
            "Body": body,
        }
        if request.headers:
            simple["Headers"] = [
                {"Name": key, "Value": value}
                for key, value in request.headers.items()
            ]

        send_kwargs: dict[str, Any] = {
            "FromEmailAddress": request.from_email or self._config.from_email,
            "Destination": {"ToAddresses": [request.to_email]},
            "Content": {"Simple": simple},
        }
        if request.reply_to:
            send_kwargs["ReplyToAddresses"] = [request.reply_to]
        if self._config.configuration_set:
            send_kwargs["ConfigurationSetName"] = self._config.configuration_set
        tags = normalize_tags(request.tags)
        if tags:
            send_kwargs["EmailTags"] = [
                {"Name": item["name"], "Value": item["value"]}
                for item in tags
            ]

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.send_email(**send_kwargs),
        )
        return SendResult(
            provider="ses",
            message_id=str(response.get("MessageId", "")),
            raw=response,
        )


def create_campaign_sender(
    provider: str,
    config: Mapping[str, Any],
    *,
    http_client: Any | None = None,
    ses_client: Any | None = None,
) -> ResendCampaignSender | SESCampaignSender:
    """Create a sender from product-owned provider config."""
    normalized = str(provider or "").strip().lower()
    if normalized == "resend":
        return ResendCampaignSender(
            ResendSenderConfig(
                api_key=str(config.get("api_key") or ""),
                api_url=str(config.get("api_url") or RESEND_API_URL),
                timeout_seconds=float(config.get("timeout_seconds") or 30.0),
            ),
            http_client=http_client,
        )
    if normalized == "ses":
        return SESCampaignSender(
            SESSenderConfig(
                from_email=str(config.get("from_email") or ""),
                region=str(config.get("region") or "us-east-1"),
                access_key_id=config.get("access_key_id"),
                secret_access_key=config.get("secret_access_key"),
                configuration_set=config.get("configuration_set"),
            ),
            client=ses_client,
        )
    raise ValueError(f"Unsupported campaign sender provider: {provider!r}")
