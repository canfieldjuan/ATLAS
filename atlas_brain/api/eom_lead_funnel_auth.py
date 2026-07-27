"""Fail-closed service authentication for EOM office lead conversion."""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException

from ..config import EOMFunnelConfig, settings

_MIN_TOKEN_LENGTH = 24
_PLACEHOLDER_TOKENS = {
    "<token>",
    "change-me",
    "changeme",
    "password",
    "secret",
    "test-token",
    "token",
}


def validate_eom_funnel_api_config(config: EOMFunnelConfig | None = None) -> None:
    """Fail startup when an enabled private API has unsafe credentials."""
    resolved = config or settings.eom_funnel
    if not resolved.api_enabled:
        return
    token = resolved.service_token.strip()
    if not token:
        raise RuntimeError(
            "ATLAS_EOM_FUNNEL_SERVICE_TOKEN is required when "
            "ATLAS_EOM_FUNNEL_API_ENABLED=true"
        )
    if token.lower() in _PLACEHOLDER_TOKENS:
        raise RuntimeError("EOM funnel service token must not be a placeholder")
    if len(token) < _MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"EOM funnel service token must be at least {_MIN_TOKEN_LENGTH} characters"
        )


def get_eom_funnel_api_config() -> EOMFunnelConfig:
    """Expose the config through an overrideable request boundary."""
    return settings.eom_funnel


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
    if not hmac.compare_digest(provided.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def require_eom_funnel_actor(
    x_eom_actor: str = Header(default="", alias="X-EOM-Actor"),
    x_eom_actor_id: str = Header(default="", alias="X-EOM-Actor-ID"),
) -> dict[str, object]:
    """Accept actor evidence only after the dedicated service is authenticated."""
    actor = x_eom_actor.strip()
    if not actor:
        raise HTTPException(status_code=422, detail="X-EOM-Actor is required")
    if len(actor) > 128:
        raise HTTPException(status_code=422, detail="X-EOM-Actor is too long")
    try:
        actor_id = int(x_eom_actor_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="X-EOM-Actor-ID must be a positive integer") from exc
    if actor_id <= 0:
        raise HTTPException(status_code=422, detail="X-EOM-Actor-ID must be a positive integer")
    return {"id": actor_id, "name": actor}
