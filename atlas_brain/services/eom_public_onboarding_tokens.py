"""Opaque, HMAC-signed authority for one EOM public-onboarding link.

Atlas owns this format because it owns the lead lifecycle the link may finish.
The value is deliberately not a JWT: the durable database row decides whether a
validly signed bearer is issued, revoked, or redeemed. The random UUID supplies
the bearer entropy; the HMAC prevents an attacker from modifying or inventing
the identifier without the Atlas-only secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from uuid import UUID


EOM_PUBLIC_ONBOARDING_TOKEN_VERSION = "eomob1"
EOM_PUBLIC_ONBOARDING_TOKEN_STATUSES = ("issued", "redeemed", "revoked")
_HMAC_DIGEST_BYTES = hashlib.sha256().digest_size
_HMAC_SIGNATURE_LENGTH = len(
    base64.urlsafe_b64encode(b"\0" * _HMAC_DIGEST_BYTES).decode("ascii").rstrip("=")
)
_MAX_TOKEN_LENGTH = (
    len(EOM_PUBLIC_ONBOARDING_TOKEN_VERSION) + 1 + 36 + 1 + _HMAC_SIGNATURE_LENGTH
)
_TOKEN_PATTERN = re.compile(
    rf"^{EOM_PUBLIC_ONBOARDING_TOKEN_VERSION}\."
    r"(?P<token_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\."
    rf"(?P<signature>[A-Za-z0-9_-]{{{_HMAC_SIGNATURE_LENGTH}}})$"
)


class EOMPublicOnboardingTokenError(ValueError):
    """A caller presented a token outside the one admitted bearer grammar."""


def validate_eom_public_onboarding_hmac_secret(secret: str) -> str:
    """Return a usable secret or reject before any bearer can be minted.

    A fixed minimum avoids accepting an accidental short placeholder. It is not
    an entropy oracle, so provisioning still uses an independently generated
    secret; this is a fail-closed configuration floor rather than a claim that
    arbitrary human text has sufficient entropy.
    """

    if not isinstance(secret, str):
        raise ValueError("public onboarding HMAC secret must be text")
    normalized = secret.strip()
    if len(normalized.encode("utf-8")) < 32:
        raise ValueError(
            "public onboarding HMAC secret must be at least 32 bytes"
        )
    return normalized


def _signature(*, token_id: UUID, secret: str) -> str:
    message = f"{EOM_PUBLIC_ONBOARDING_TOKEN_VERSION}.{token_id}".encode("ascii")
    digest = hmac.new(
        validate_eom_public_onboarding_hmac_secret(secret).encode("utf-8"),
        message,
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def format_eom_public_onboarding_token(*, token_id: UUID, secret: str) -> str:
    """Build the canonical bearer without persisting its raw value."""

    return (
        f"{EOM_PUBLIC_ONBOARDING_TOKEN_VERSION}.{token_id}."
        f"{_signature(token_id=token_id, secret=secret)}"
    )


def parse_eom_public_onboarding_token(*, token: object, secret: str) -> UUID:
    """Authenticate the closed token grammar before any database lookup.

    Invalid formatting and a bad MAC intentionally share one exception. The
    service route turns that into the same generic unavailable response as an
    unknown/revoked durable token, so it does not become a token-validity oracle.
    """

    if not isinstance(token, str) or len(token) > _MAX_TOKEN_LENGTH:
        raise EOMPublicOnboardingTokenError("invalid public onboarding token")
    match = _TOKEN_PATTERN.fullmatch(token)
    if match is None:
        raise EOMPublicOnboardingTokenError("invalid public onboarding token")
    try:
        token_id = UUID(match.group("token_id"))
    except ValueError as exc:  # pattern is deliberately strict, retain a safe edge.
        raise EOMPublicOnboardingTokenError(
            "invalid public onboarding token"
        ) from exc
    expected = _signature(token_id=token_id, secret=secret)
    if not hmac.compare_digest(expected, match.group("signature")):
        raise EOMPublicOnboardingTokenError("invalid public onboarding token")
    return token_id


def build_eom_public_onboarding_link(*, base_url: str, token: str) -> str:
    """Put the bearer in a fragment so normal HTTP referrers omit it."""

    normalized_base_url = base_url.strip().rstrip("#")
    if not normalized_base_url:
        raise ValueError("public onboarding URL is required")
    return f"{normalized_base_url}#token={token}"


def append_eom_public_onboarding_invitation(*, body: str, link: str) -> str:
    """Create the transport-only invitation without changing the draft snapshot."""

    return (
        f"{body}\n\n"
        "Please complete your onboarding details before your first visit.\n"
        "Complete sus datos de incorporación antes de su primera visita:\n"
        f"{link}\n"
    )
