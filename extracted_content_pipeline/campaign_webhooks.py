"""Standalone webhook verification and event normalization for campaign ESPs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from typing import Any, Mapping

from .campaign_ports import AuditSink, CampaignRepository, Clock, WebhookEvent, WebhookVerifier
from .campaign_suppression import CampaignSuppressionService


class WebhookVerificationError(ValueError):
    """Raised when a webhook signature is invalid."""


class WebhookPayloadError(ValueError):
    """Raised when a webhook payload cannot be parsed."""


_RESEND_EVENT_MAP = {
    "email.opened": "opened",
    "email.clicked": "clicked",
    "email.bounced": "bounced",
    "email.complained": "complained",
    "email.delivered": "delivered",
    "email.unsubscribed": "unsubscribed",
}

_HANDLED_EVENT_TYPES = {
    "opened",
    "clicked",
    "bounced",
    "complained",
    "delivered",
    "unsubscribed",
}


@dataclass(frozen=True)
class ResendWebhookConfig:
    signing_secret: str = ""
    verify_signatures: bool = True


@dataclass(frozen=True)
class CampaignWebhookIngestionConfig:
    soft_bounce_suppression_days: int = 30
    record_unknown_events: bool = False


@dataclass(frozen=True)
class CampaignWebhookIngestionResult:
    status: str
    event_type: str | None = None
    message_id: str | None = None
    reason: str | None = None
    suppressed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "event_type": self.event_type,
            "message_id": self.message_id,
            "reason": self.reason,
            "suppressed": self.suppressed,
        }


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def _header(headers: Mapping[str, str], name: str) -> str:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value or "")
    return ""


def verify_svix_signature(
    payload_bytes: bytes,
    headers: Mapping[str, str],
    secret: str,
    *,
    verify_signatures: bool = True,
) -> bool:
    """Verify a Svix-format Resend webhook signature."""
    if not verify_signatures or not secret:
        return True

    msg_id = _header(headers, "svix-id")
    timestamp = _header(headers, "svix-timestamp")
    signature_header = _header(headers, "svix-signature")
    if not msg_id or not timestamp or not signature_header:
        return False

    raw_secret = str(secret)
    if raw_secret.startswith("whsec_"):
        raw_secret = raw_secret[6:]

    try:
        secret_bytes = base64.b64decode(raw_secret)
    except Exception:
        return False

    to_sign = f"{msg_id}.{timestamp}.".encode("utf-8") + payload_bytes
    expected = base64.b64encode(
        hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()
    ).decode("utf-8")

    for signature in signature_header.split(" "):
        version, sep, value = signature.partition(",")
        if sep and version == "v1" and hmac.compare_digest(value, expected):
            return True
    return False


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def normalize_resend_payload(payload: Mapping[str, Any]) -> WebhookEvent:
    raw_type = str(payload.get("type") or "").strip()
    data = payload.get("data")
    data = data if isinstance(data, Mapping) else {}
    event_type = _RESEND_EVENT_MAP.get(raw_type, raw_type or "unknown")
    click = data.get("click") if isinstance(data.get("click"), Mapping) else {}
    bounce = data.get("bounce") if isinstance(data.get("bounce"), Mapping) else {}
    normalized_payload = dict(payload)
    normalized_payload["normalized"] = {
        "raw_event_type": raw_type,
        "click_url": click.get("link"),
        "bounce_type": bounce.get("type"),
    }
    return WebhookEvent(
        provider="resend",
        event_type=event_type,
        message_id=str(data.get("email_id") or "").strip() or None,
        email=str(data.get("to") or data.get("email") or "").strip() or None,
        occurred_at=_parse_datetime(
            data.get("created_at")
            or data.get("createdAt")
            or payload.get("created_at")
            or payload.get("createdAt")
        ),
        payload=normalized_payload,
    )


class ResendWebhookVerifier:
    """Verify and normalize Resend/Svix campaign webhook payloads."""

    def __init__(self, config: ResendWebhookConfig):
        self._config = config

    def verify_and_parse(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
    ) -> WebhookEvent:
        if not verify_svix_signature(
            body,
            headers,
            self._config.signing_secret,
            verify_signatures=self._config.verify_signatures,
        ):
            raise WebhookVerificationError("Invalid webhook signature")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WebhookPayloadError("Invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise WebhookPayloadError("Webhook payload must be a JSON object")
        return normalize_resend_payload(payload)


def _bounce_type(event: WebhookEvent) -> str:
    normalized = event.payload.get("normalized")
    if isinstance(normalized, Mapping):
        value = str(normalized.get("bounce_type") or "").strip().lower()
        if value:
            return value
    data = event.payload.get("data")
    data = data if isinstance(data, Mapping) else {}
    bounce = data.get("bounce")
    bounce = bounce if isinstance(bounce, Mapping) else {}
    return str(bounce.get("type") or "hard").strip().lower() or "hard"


class CampaignWebhookIngestionService:
    """Process normalized campaign webhooks through product-owned ports."""

    def __init__(
        self,
        *,
        verifier: WebhookVerifier,
        campaigns: CampaignRepository,
        suppression: CampaignSuppressionService | None = None,
        audit: AuditSink | None = None,
        clock: Clock | None = None,
        config: CampaignWebhookIngestionConfig | None = None,
    ):
        self._verifier = verifier
        self._campaigns = campaigns
        self._suppression = suppression
        self._audit = audit
        self._clock = clock or SystemClock()
        self._config = config or CampaignWebhookIngestionConfig()

    async def ingest(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
    ) -> CampaignWebhookIngestionResult:
        event = self._verifier.verify_and_parse(body=body, headers=headers)
        if not event.message_id:
            return CampaignWebhookIngestionResult(
                status="ignored",
                event_type=event.event_type,
                reason="no_message_id",
            )
        if event.event_type not in _HANDLED_EVENT_TYPES and not self._config.record_unknown_events:
            return CampaignWebhookIngestionResult(
                status="ignored",
                event_type=event.event_type,
                message_id=event.message_id,
                reason="unhandled_event_type",
            )

        await self._campaigns.record_webhook_event(event)
        suppressed = await self._apply_suppression(event)
        if self._audit:
            await self._audit.record(
                f"webhook_{event.event_type}",
                metadata={
                    "provider": event.provider,
                    "message_id": event.message_id,
                    "email": event.email,
                    "suppressed": suppressed,
                },
            )
        return CampaignWebhookIngestionResult(
            status="ok",
            event_type=event.event_type,
            message_id=event.message_id,
            suppressed=suppressed,
        )

    async def _apply_suppression(self, event: WebhookEvent) -> bool:
        if not self._suppression:
            return False
        if event.event_type == "complained":
            return await self._suppression.add_suppression(
                email=event.email,
                reason="complaint",
                source="webhook",
                metadata={"provider_message_id": event.message_id},
            )
        if event.event_type == "unsubscribed":
            return await self._suppression.add_suppression(
                email=event.email,
                reason="unsubscribe",
                source="webhook",
                metadata={"provider_message_id": event.message_id},
            )
        if event.event_type == "bounced":
            bounce_type = _bounce_type(event)
            expires_at = None
            if bounce_type != "hard":
                expires_at = self._clock.now() + timedelta(
                    days=self._config.soft_bounce_suppression_days
                )
            return await self._suppression.add_suppression(
                email=event.email,
                reason="bounce_hard" if bounce_type == "hard" else "bounce_soft",
                source="webhook",
                expires_at=expires_at,
                metadata={
                    "provider_message_id": event.message_id,
                    "bounce_type": bounce_type,
                },
            )
        return False
