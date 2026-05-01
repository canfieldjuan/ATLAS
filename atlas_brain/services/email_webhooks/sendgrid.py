"""SendGrid webhook provider (stubbed).

SendGrid delivers events as a JSON array (one POST may carry many events).
Signature verification uses Ed25519: the public key is configured on the
SendGrid event-webhook settings page, and the signature lives in the
``X-Twilio-Email-Event-Webhook-Signature`` header. The signed payload is
``X-Twilio-Email-Event-Webhook-Timestamp + body``.

Implementing this requires the ``cryptography`` package for Ed25519
verification. Stubbed here so the provider registry resolves cleanly; a
follow-up branch will land the implementation when a customer brings
SendGrid into scope.
"""

from __future__ import annotations

from typing import Mapping

from ...schemas.campaigns import CanonicalEvent
from . import register


class SendGridProvider:
    name = "sendgrid"

    def verify_signature(
        self,
        payload_bytes: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> bool:
        raise NotImplementedError(
            "SendGrid webhook provider not yet implemented. Track follow-up "
            "in the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )

    def normalize_event(self, payload_bytes: bytes) -> list[CanonicalEvent]:
        raise NotImplementedError(
            "SendGrid webhook provider not yet implemented. Track follow-up "
            "in the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )


register(SendGridProvider())
