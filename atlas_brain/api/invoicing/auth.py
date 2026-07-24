"""Dedicated fail-closed authentication for invoicing and receivables routes."""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException

from ...config import InvoicingConfig, settings

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


def validate_receivables_api_config(
    config: InvoicingConfig | None = None,
) -> None:
    """Fail startup when an enabled finance API has an unsafe token."""
    resolved = config or settings.invoicing
    if not resolved.receivables_api_enabled:
        return
    token = resolved.receivables_service_token.strip()
    if not token:
        raise RuntimeError(
            "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN is required when "
            "ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true"
        )
    if token.lower() in _PLACEHOLDER_TOKENS:
        raise RuntimeError("Receivables service token must not be a placeholder")
    if len(token) < _MIN_TOKEN_LENGTH:
        raise RuntimeError(
            f"Receivables service token must be at least {_MIN_TOKEN_LENGTH} characters"
        )


def get_receivables_api_config() -> InvoicingConfig:
    """Return the runtime config through an overrideable request boundary."""
    return settings.invoicing


async def require_receivables_api(
    authorization: str = Header(default="", alias="Authorization"),
    config: InvoicingConfig = Depends(get_receivables_api_config),
) -> None:
    """Require the configured bearer token; disabled is unavailable, not open."""
    if not config.receivables_api_enabled:
        raise HTTPException(status_code=503, detail="Receivables API is disabled")
    validate_receivables_api_config(config)
    scheme, separator, provided = authorization.partition(" ")
    expected = config.receivables_service_token.strip()
    if separator != " " or scheme.lower() != "bearer" or not provided.strip():
        raise HTTPException(status_code=401, detail="Bearer token required")
    if not hmac.compare_digest(provided.strip(), expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def require_actor(
    x_eom_actor: str = Header(default="", alias="X-EOM-Actor"),
) -> str:
    actor = x_eom_actor.strip()
    if not actor:
        raise HTTPException(status_code=422, detail="X-EOM-Actor is required")
    if len(actor) > 128:
        raise HTTPException(status_code=422, detail="X-EOM-Actor is too long")
    return actor
