"""Dedicated fail-closed authentication for invoicing and receivables routes."""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException

from ...config import InvoicingConfig, settings
from ...eom_api.auth import (
    receivables_service_token_sha256,
    validate_generated_service_token_digest,
)
from ...eom_api.config import RAW_RECEIVABLES_SERVICE_TOKEN_ENV


def validate_receivables_api_config(
    config: InvoicingConfig | None = None,
) -> None:
    """Fail startup unless enabled auth has digest-only generated-token config."""
    resolved = config or settings.invoicing
    raw_token = getattr(resolved, "receivables_service_token", "")
    if raw_token is not None and str(raw_token).strip():
        raise RuntimeError(
            "Raw EOM receivables bearer token material must not be configured "
            f"in {RAW_RECEIVABLES_SERVICE_TOKEN_ENV}; provision only "
            "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256 here and keep "
            "the raw token on the caller side."
        )
    if not resolved.receivables_api_enabled:
        return
    token_digest = getattr(resolved, "receivables_service_token_sha256", "")
    token_digest = token_digest.strip() if isinstance(token_digest, str) else ""
    validate_generated_service_token_digest(
        token_digest,
        service_name="Receivables",
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
    if separator != " " or scheme.lower() != "bearer" or not provided.strip():
        raise HTTPException(status_code=401, detail="Bearer token required")
    provided_token = provided.strip()
    try:
        provided_digest = receivables_service_token_sha256(provided_token)
    except (RuntimeError, UnicodeEncodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc
    expected_digest = config.receivables_service_token_sha256.strip()
    if not hmac.compare_digest(provided_digest, expected_digest):
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
