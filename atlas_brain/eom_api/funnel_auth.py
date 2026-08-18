"""Fail-closed service authentication for EOM office lead conversion."""

from __future__ import annotations

import hmac
import hashlib
import secrets
from dataclasses import dataclass, field

from fastapi import Depends, Header, HTTPException

from .auth import (
    validate_generated_service_token,
    validate_generated_service_token_digest,
)
from .config import EOMFunnelConfig, funnel_settings
from ..services.eom_public_onboarding_tokens import (
    validate_eom_public_onboarding_hmac_secret,
)

_MAX_SIGNED_BIGINT = 2**63 - 1
_MAX_LIFECYCLE_ACTOR_LENGTH = 128
_ACTOR_PREFIX = "employee:"
_MAX_ACTOR_NAME_LENGTH = (
    _MAX_LIFECYCLE_ACTOR_LENGTH
    - len(f"{_ACTOR_PREFIX}{_MAX_SIGNED_BIGINT}:")
)
_GENERATED_TOKEN_PREFIX = "eomf_v1_"
_MIN_GENERATED_TOKEN_UNIQUE_CHARACTERS = 16
_FUNNEL_TOKEN_GENERATOR_REFERENCE = (
    "atlas_brain.eom_api.funnel_auth.generate_eom_funnel_service_token()"
)


def _validate_generated_funnel_token(token: str) -> None:
    validate_generated_service_token(
        token,
        token_prefix=_GENERATED_TOKEN_PREFIX,
        service_name="EOM funnel",
        generator_reference=_FUNNEL_TOKEN_GENERATOR_REFERENCE,
        minimum_unique_characters=_MIN_GENERATED_TOKEN_UNIQUE_CHARACTERS,
    )


@dataclass(frozen=True)
class GeneratedEOMFunnelServiceToken:
    """Generated tracker bearer plus the Atlas-only trust anchor."""

    token: str
    sha256: str


@dataclass(frozen=True)
class EOMPublicOnboardingConfig:
    """Validated Atlas-only ingredients for one public onboarding authority."""

    base_url: str
    hmac_secret: str = field(repr=False)
    previous_hmac_secret: str | None = field(default=None, repr=False)


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def eom_funnel_service_token_sha256(token: str) -> str:
    """Return the Atlas trust anchor for a freshly generated tracker bearer."""
    _validate_generated_funnel_token(token)
    return _token_sha256(token)


def generate_eom_funnel_service_token() -> GeneratedEOMFunnelServiceToken:
    """Generate the tracker bearer and its Atlas-only SHA-256 trust anchor."""
    token = f"{_GENERATED_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    return GeneratedEOMFunnelServiceToken(
        token=token,
        sha256=eom_funnel_service_token_sha256(token),
    )


def validate_eom_funnel_api_config(config: EOMFunnelConfig | None = None) -> None:
    """Fail startup when an enabled private API has unsafe credentials."""
    resolved = config or funnel_settings
    if not resolved.api_enabled:
        return
    token_digest = resolved.service_token_sha256.strip()
    if not token_digest:
        raise RuntimeError(
            "ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256 is required when "
            "ATLAS_EOM_FUNNEL_API_ENABLED=true"
        )
    validate_generated_service_token_digest(
        token_digest,
        service_name="EOM funnel",
    )


def get_eom_funnel_api_config() -> EOMFunnelConfig:
    """Expose the config through an overrideable request boundary."""
    return funnel_settings


def require_eom_public_onboarding_config(
    config: EOMFunnelConfig = Depends(get_eom_funnel_api_config),
) -> EOMPublicOnboardingConfig:
    """Require the separately enabled public-link authority, never a fallback.

    The parent service bearer dependency stays on every route that uses this.
    Keeping the feature gate separate lets the base private funnel continue its
    current office behavior while callers roll out ahead of the manual Atlas
    deployment or before the operator explicitly enables public links.
    """

    if not config.public_onboarding_enabled:
        raise HTTPException(
            status_code=503,
            detail="Public onboarding is not enabled",
        )
    try:
        secret = validate_eom_public_onboarding_hmac_secret(
            config.public_onboarding_hmac_secret.get_secret_value()
        )
        previous_secret_raw = (
            config.public_onboarding_previous_hmac_secret.get_secret_value().strip()
        )
        previous_secret = (
            validate_eom_public_onboarding_hmac_secret(previous_secret_raw)
            if previous_secret_raw
            else None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Public onboarding configuration is unavailable",
        ) from exc
    base_url = config.public_onboarding_url.strip()
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail="Public onboarding configuration is unavailable",
        )
    return EOMPublicOnboardingConfig(
        base_url=base_url,
        hmac_secret=secret,
        previous_hmac_secret=previous_secret,
    )


async def require_eom_funnel_api(
    authorization: str = Header(default="", alias="Authorization"),
    config: EOMFunnelConfig = Depends(get_eom_funnel_api_config),
) -> None:
    """Require the tracker-only bearer token; disabled never means public."""
    if not config.api_enabled:
        raise HTTPException(status_code=503, detail="EOM funnel API is disabled")
    validate_eom_funnel_api_config(config)
    scheme, separator, provided = authorization.partition(" ")
    expected_digest = config.service_token_sha256.strip()
    if separator != " " or scheme.lower() != "bearer" or not provided.strip():
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        provided_digest = _token_sha256(provided.strip())
    except UnicodeEncodeError as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc
    if not hmac.compare_digest(provided_digest, expected_digest):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def require_eom_funnel_actor(
    x_eom_actor: str = Header(default="", alias="X-EOM-Actor"),
    x_eom_actor_id: str = Header(default="", alias="X-EOM-Actor-ID"),
) -> dict[str, object]:
    """Accept actor evidence only after the dedicated service is authenticated."""
    actor = x_eom_actor.strip()
    if not actor:
        raise HTTPException(status_code=422, detail="X-EOM-Actor is required")
    if len(actor) > _MAX_ACTOR_NAME_LENGTH:
        raise HTTPException(status_code=422, detail="X-EOM-Actor is too long")
    try:
        actor_id = int(x_eom_actor_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="X-EOM-Actor-ID must be a positive integer") from exc
    if not 0 < actor_id <= _MAX_SIGNED_BIGINT:
        raise HTTPException(status_code=422, detail="X-EOM-Actor-ID must fit in a signed 64-bit integer")
    return {"id": actor_id, "name": actor}
