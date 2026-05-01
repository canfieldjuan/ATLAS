"""Resend webhook provider.

Extracted verbatim from atlas_brain/api/campaign_webhooks.py (pre-Gap-1).
Resend uses Svix for signing: HMAC-SHA256 over ``msg_id.timestamp.body``
with a base64-encoded shared secret optionally prefixed with ``whsec_``.

Event types we map: email.{delivered, opened, clicked, bounced, complained}.
Each Resend POST contains exactly one event.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from typing import Mapping

from ...schemas.campaigns import CanonicalEvent
from . import register

logger = logging.getLogger("atlas.services.email_webhooks.resend")


_RESEND_TO_CANONICAL = {
    "email.delivered": "delivered",
    "email.opened": "opened",
    "email.clicked": "clicked",
    "email.bounced": "bounced",
    "email.complained": "complained",
}


class ResendProvider:
    name = "resend"

    def verify_signature(
        self,
        payload_bytes: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> bool:
        if not secret:
            logger.warning(
                "Resend webhook signing secret not configured -- signature verification disabled",
            )
            return True

        msg_id = headers.get("svix-id", "")
        timestamp = headers.get("svix-timestamp", "")
        signature_header = headers.get("svix-signature", "")
        if not msg_id or not timestamp or not signature_header:
            return False

        to_sign = f"{msg_id}.{timestamp}.".encode() + payload_bytes
        raw_secret = secret[6:] if secret.startswith("whsec_") else secret
        try:
            secret_bytes = base64.b64decode(raw_secret)
        except (ValueError, TypeError):
            return False
        expected = base64.b64encode(
            hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()
        ).decode()

        for sig in signature_header.split(" "):
            parts = sig.split(",", 1)
            if (
                len(parts) == 2
                and parts[0] == "v1"
                and hmac.compare_digest(parts[1], expected)
            ):
                return True
        return False

    def normalize_event(self, payload_bytes: bytes) -> list[CanonicalEvent]:
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError:
            return []
        raw_type = str(payload.get("type") or "")
        canonical_type = _RESEND_TO_CANONICAL.get(raw_type)
        if not canonical_type:
            return []
        data = payload.get("data") or {}
        message_id = str(data.get("email_id") or "").strip()
        recipient = str(data.get("to") or data.get("email_to") or "").strip()
        if isinstance(data.get("to"), list) and data["to"]:
            recipient = str(data["to"][0]).strip()
        timestamp = str(payload.get("created_at") or data.get("created_at") or "").strip()
        if not message_id:
            return []
        bounce = data.get("bounce") or {}
        click = data.get("click") or {}
        return [
            CanonicalEvent(
                provider="resend",
                event_type=canonical_type,
                message_id=message_id,
                recipient_email=recipient,
                timestamp=timestamp,
                bounce_type=str(bounce.get("type") or "").strip() or None,
                bounce_subtype=str(bounce.get("subType") or "").strip() or None,
                click_url=str(click.get("link") or "").strip() or None,
                user_agent=str(click.get("userAgent") or data.get("user_agent") or "").strip() or None,
                ip=str(click.get("ipAddress") or data.get("ip") or "").strip() or None,
                raw=data if isinstance(data, dict) else None,
            )
        ]


register(ResendProvider())
