"""Fail-closed service authentication for EOM office lead conversion."""

from __future__ import annotations

import hmac
import secrets

from fastapi import Depends, Header, HTTPException

from .auth import validate_generated_service_token
from .config import EOMFunnelConfig, funnel_settings

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


def generate_eom_funnel_service_token() -> str:
    """Generate the server-only credential for the EOM tracker-to-Atlas route."""
    token = f"{_GENERATED_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    _validate_generated_funnel_token(token)
    return token


def validate_eom_funnel_api_config(config: EOMFunnelConfig | None = None) -> None:
    """Fail startup when an enabled private API has unsafe credentials."""
    resolved = config or funnel_settings
    if not resolved.api_enabled:
        return
    token = resolved.service_token.strip()
    if not token:
        raise RuntimeError(
            "ATLAS_EOM_FUNNEL_SERVICE_TOKEN is required when "
            "ATLAS_EOM_FUNNEL_API_ENABLED=true"
        )
    _validate_generated_funnel_token(token)


def get_eom_funnel_api_config() -> EOMFunnelConfig:
    """Expose the config through an overrideable request boundary."""
    return funnel_settings


async def require_eom_funnel_api(
    authorization: str = Header(default="", alias="Authorization"),
    config: EOMFunnelConfig = Depends(get_eom_funnel_api_config),
) -> None:
    """Require the tracker-only bearer token; disabled never means public."""
    if not config.api_enabled:
        raise HTTPException(status_code=503, detail="EOM funnel API is disabled")
    validate_eom_funnel_api_config(config)
    scheme, separator, provided = authorization.partition(" ")
    expected = config.service_token.strip()
    if separator != " " or scheme.lower() != "bearer" or not provided.strip():
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        provided_bytes = provided.strip().encode("ascii")
        expected_bytes = expected.encode("ascii")
    except UnicodeEncodeError as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc
    if not hmac.compare_digest(provided_bytes, expected_bytes):
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
