"""AWS SES webhook provider (stubbed).

SES delivers events via SNS notifications. A complete implementation must:

  1. Handle the SNS SubscriptionConfirmation handshake (auto-confirm by
     fetching the SubscribeURL) before any Notification can flow.
  2. Verify SNS signatures using the X.509 certificate at
     ``SigningCertURL`` (must be a *.amazonaws.com host) and the documented
     canonical-string format.
  3. Decode the inner ``Message`` JSON whose ``eventType`` is one of
     ``Delivery|Bounce|Complaint|Open|Click``.

Stubbed here so the provider registry resolves cleanly; a follow-up branch
will land the implementation when a customer brings SES into scope.
"""

from __future__ import annotations

from typing import Mapping

from ...schemas.campaigns import CanonicalEvent
from . import register


class SESProvider:
    name = "ses"

    def verify_signature(
        self,
        payload_bytes: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> bool:
        raise NotImplementedError(
            "SES webhook provider not yet implemented. Track follow-up in "
            "the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )

    def normalize_event(self, payload_bytes: bytes) -> list[CanonicalEvent]:
        raise NotImplementedError(
            "SES webhook provider not yet implemented. Track follow-up in "
            "the email-campaign-pilot-readiness plan, Gap 1 phase 2."
        )


register(SESProvider())
