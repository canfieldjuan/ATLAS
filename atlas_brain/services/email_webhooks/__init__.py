"""Multi-ESP webhook provider abstraction.

The route handler in atlas_brain/api/campaign_webhooks.py is provider-
agnostic: it operates on a CanonicalEvent (atlas_brain/schemas/campaigns.py)
and updates b2b_campaigns / campaign_sequences uniformly. Each ESP supplies
a WebhookProvider that:

  1. Verifies the inbound HTTP request's signature against a per-provider
     shared secret (raises if invalid).
  2. Normalizes the raw payload into one or more CanonicalEvent objects.

A provider may emit zero, one, or many events per HTTP body. Postmark and
Resend send one; SendGrid batches; SES wraps in SNS envelopes. The route
handler iterates the returned list and updates state per event.

Resolve(name) returns the provider implementation for a given ESP name.
Unknown names raise UnknownProviderError so the route can return 400.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from ...schemas.campaigns import CanonicalEvent


class UnknownProviderError(Exception):
    """Raised when ?provider= references an ESP we don't have a plugin for."""


class SignatureVerificationError(Exception):
    """Raised when the inbound webhook fails signature verification.

    Distinct from generic exceptions so the route handler can map this to
    HTTP 401 deterministically.
    """


@runtime_checkable
class WebhookProvider(Protocol):
    """Contract every ESP plugin honors.

    Implementations live in sibling modules (resend.py, ses.py, etc.) and
    register themselves via the resolve() registry below.
    """

    name: str

    def verify_signature(
        self,
        payload_bytes: bytes,
        headers: Mapping[str, str],
        secret: str,
    ) -> bool:
        """Return True iff the request signature is valid (or skipped).

        Implementations should return True when ``secret`` is empty so that
        local-dev environments without a configured signing key still
        function. Signature verification failure is reported by returning
        False; raising should be reserved for non-recoverable parse errors.
        """
        ...

    def normalize_event(self, payload_bytes: bytes) -> list[CanonicalEvent]:
        """Parse the raw body into zero-or-more CanonicalEvent objects.

        Returns an empty list when the payload is a non-event message
        (e.g. SES SNS SubscriptionConfirmation, which the implementation
        handles out-of-band).
        """
        ...


_PROVIDERS: dict[str, WebhookProvider] = {}


def register(provider: WebhookProvider) -> WebhookProvider:
    """Register a provider implementation. Idempotent for testing."""
    _PROVIDERS[provider.name] = provider
    return provider


def resolve(name: str) -> WebhookProvider:
    """Look up a provider by canonical name (lowercased).

    Raises UnknownProviderError when no plugin matches; the route handler
    maps this to HTTP 400.
    """
    key = (name or "").strip().lower()
    if not key:
        key = "resend"
    provider = _PROVIDERS.get(key)
    if provider is None:
        raise UnknownProviderError(
            f"unknown email webhook provider: {key!r} "
            f"(registered: {sorted(_PROVIDERS)})"
        )
    return provider


# Eagerly register the bundled providers. Each module has side-effects
# (calls register()) on import; importing here makes resolve() work without
# the route handler caring about plugin discovery.
from . import resend  # noqa: F401,E402
from . import ses  # noqa: F401,E402
from . import sendgrid  # noqa: F401,E402
from . import postmark  # noqa: F401,E402
from . import mailgun  # noqa: F401,E402


__all__ = [
    "WebhookProvider",
    "UnknownProviderError",
    "SignatureVerificationError",
    "register",
    "resolve",
]
