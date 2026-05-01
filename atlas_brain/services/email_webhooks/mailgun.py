"""Mailgun webhook provider (stubbed).

Mailgun signs each event with HMAC-SHA256 over ``timestamp + token`` using
the API signing key. Payloads can arrive as JSON or as form-encoded data
depending on the route configuration. ``event-data`` carries the canonical
event with fields like ``event``, ``recipient``, ``message.headers
.message-id``, and ``severity`` for bounces.

Stubbed here so the provider registry resolves cleanly; a follow-up branch
will land the implementation when a customer brings Mailgun into scope.
"""

from __future__ import annotations

from typing import Mapping

from ...schemas.campaigns import CanonicalEvent
from . import register


class MailgunProvider:
    name = "mailgun"

    def verify_signature(
        self,
        payload_bytes: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> bool:
        raise NotImplementedError(
            "Mailgun webhook provider not yet implemented. Track follow-up "
            "in the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )

    def normalize_event(self, payload_bytes: bytes) -> list[CanonicalEvent]:
        raise NotImplementedError(
            "Mailgun webhook provider not yet implemented. Track follow-up "
            "in the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )


register(MailgunProvider())
