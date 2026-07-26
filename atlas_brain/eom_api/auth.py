"""Fail-closed service authentication for the slim EOM API profile."""

from __future__ import annotations

import hmac
import secrets
from collections import Counter
from re import fullmatch

from fastapi import Depends, Header, HTTPException

from .config import EOMInvoicingConfig, invoicing_settings

_GENERATED_TOKEN_PREFIX = "eomrx_"
_GENERATED_TOKEN_RANDOM_LENGTH = 43
_MIN_TOKEN_UNIQUE_CHARS = 12
_MAX_TOKEN_CHAR_FREQUENCY_RATIO = 0.30
_TOKEN_RANDOM_PATTERN = r"[A-Za-z0-9_-]+"
_PLACEHOLDER_TOKENS = {
    "<token>",
    "change-me",
    "changeme",
    "password",
    "secret",
    "test-token",
    "token",
}


def generate_receivables_service_token() -> str:
    """Generate an EOM receivables service token suitable for Render secrets."""
    return f"{_GENERATED_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"


def _validate_generated_token(token: str) -> None:
    if not token.startswith(_GENERATED_TOKEN_PREFIX):
        raise RuntimeError(
            "Receivables service token must be generated with the eomrx_ prefix; "
            "use atlas_brain.eom_api.auth.generate_receivables_service_token()"
        )
    random_part = token.removeprefix(_GENERATED_TOKEN_PREFIX)
    if len(random_part) < _GENERATED_TOKEN_RANDOM_LENGTH:
        raise RuntimeError(
            "Receivables service token random payload is too short; generate a "
            "new token with atlas_brain.eom_api.auth.generate_receivables_service_token()"
        )
    if fullmatch(_TOKEN_RANDOM_PATTERN, random_part) is None:
        raise RuntimeError(
            "Receivables service token contains invalid characters; generate a "
            "new token with atlas_brain.eom_api.auth.generate_receivables_service_token()"
        )
    counts = Counter(random_part)
    if len(counts) < _MIN_TOKEN_UNIQUE_CHARS:
        raise RuntimeError(
            "Receivables service token is too predictable; generate a new token "
            "with atlas_brain.eom_api.auth.generate_receivables_service_token()"
        )
    if max(counts.values()) / len(random_part) > _MAX_TOKEN_CHAR_FREQUENCY_RATIO:
        raise RuntimeError(
            "Receivables service token has a weak repeated-character pattern; "
            "generate a new token with atlas_brain.eom_api.auth.generate_receivables_service_token()"
        )


def validate_receivables_api_config(
    config: EOMInvoicingConfig | None = None,
) -> None:
    """Fail startup when an enabled finance API has an unsafe token."""
    resolved = config or invoicing_settings
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
    _validate_generated_token(token)


def get_receivables_api_config() -> EOMInvoicingConfig:
    """Return the runtime config through an overrideable request boundary."""
    return invoicing_settings


async def require_receivables_api(
    authorization: str = Header(default="", alias="Authorization"),
    config: EOMInvoicingConfig = Depends(get_receivables_api_config),
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
