"""Postmark webhook provider (stubbed).

Postmark delivers events as flat JSON, one event per POST. Auth is handled
via HTTPS Basic Auth (configured in the Postmark webhook settings) or a
shared bearer token; there is no signature scheme like Resend's Svix.

Stubbed here so the provider registry resolves cleanly; a follow-up branch
will land the implementation when a customer brings Postmark into scope.
"""

from __future__ import annotations

from typing import Mapping

from ...schemas.campaigns import CanonicalEvent
from . import register


class PostmarkProvider:
    name = "postmark"

    def verify_signature(
        self,
        payload_bytes: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> bool:
        raise NotImplementedError(
            "Postmark webhook provider not yet implemented. Track follow-up "
            "in the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )

    def normalize_event(self, payload_bytes: bytes) -> list[CanonicalEvent]:
        raise NotImplementedError(
            "Postmark webhook provider not yet implemented. Track follow-up "
            "in the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )


register(PostmarkProvider())
