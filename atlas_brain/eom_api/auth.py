"""Fail-closed service authentication for the slim EOM API profile."""

from __future__ import annotations

import hmac
import hashlib
import secrets
from dataclasses import dataclass
from re import fullmatch
from typing import Protocol

from fastapi import Depends, Header, HTTPException

from .config import (
    EOMInvoicingConfig,
    RAW_RECEIVABLES_SERVICE_TOKEN_ENV,
    invoicing_settings,
    raw_receivables_service_token_env_value,
)

_GENERATED_TOKEN_PREFIX = "eomrx_v1_"
_GENERATED_TOKEN_RANDOM_LENGTH = 43
_TOKEN_RANDOM_PATTERN = r"[A-Za-z0-9_-]+"
_TOKEN_SHA256_PATTERN = r"[0-9a-f]{64}"
_PLACEHOLDER_TOKENS = {
    "<token>",
    "change-me",
    "changeme",
    "password",
    "secret",
    "test-token",
    "token",
}


@dataclass(frozen=True)
class GeneratedReceivablesServiceToken:
    """Generated service-token material for future Slice C provisioning."""

    token: str
    sha256: str


@dataclass(frozen=True)
class TrustedReceivablesApiConfig:
    """In-process config for transport tests before external provisioning exists."""

    receivables_api_enabled: bool
    receivables_service_token_sha256: str


class ReceivablesApiAuthConfig(Protocol):
    receivables_api_enabled: bool
    receivables_service_token_sha256: str


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


_PLACEHOLDER_TOKEN_DIGESTS = {
    _token_sha256(token) for token in _PLACEHOLDER_TOKENS
}
_PLACEHOLDER_TOKEN_DIGESTS.add("0" * 64)


def _validate_generated_token(token: str) -> None:
    if not token.startswith(_GENERATED_TOKEN_PREFIX):
        raise RuntimeError(
            "Receivables service token must be generated with the eomrx_v1_ prefix; "
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


def receivables_service_token_sha256(token: str) -> str:
    """Return the digest accepted by the EOM API for a generated token."""
    _validate_generated_token(token)
    return _token_sha256(token)


def generate_receivables_service_token() -> GeneratedReceivablesServiceToken:
    """Generate EOM receivables service-token material for secret provisioning."""
    token = f"{_GENERATED_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    token_digest = receivables_service_token_sha256(token)
    return GeneratedReceivablesServiceToken(
        token=token,
        sha256=token_digest,
    )


def _validate_generated_token_digest(token_digest: str) -> None:
    if not token_digest:
        raise RuntimeError(
            "Receivables service token digest is required for trusted "
            "receivables API config"
        )
    if fullmatch(_TOKEN_SHA256_PATTERN, token_digest) is None:
        raise RuntimeError(
            "Receivables service token digest must be a lowercase SHA-256 hex "
            "digest produced from a generated token"
        )
    if token_digest in _PLACEHOLDER_TOKEN_DIGESTS:
        raise RuntimeError(
            "Receivables service token digest must not come from a placeholder"
        )


def validate_receivables_api_config(
    config: ReceivablesApiAuthConfig | EOMInvoicingConfig | None = None,
) -> None:
    """Fail startup unless enabled auth has digest-only generated-token config."""
    resolved = config or invoicing_settings
    if not resolved.receivables_api_enabled:
        return
    if raw_receivables_service_token_env_value():
        raise RuntimeError(
            "Raw EOM receivables bearer token material must not be configured "
            f"in {RAW_RECEIVABLES_SERVICE_TOKEN_ENV}; provision only "
            "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256 on the Atlas API "
            "service and keep the raw token on the caller side."
        )
    raw_token = getattr(resolved, "receivables_service_token", "")
    if raw_token is not None and str(raw_token).strip():
        raise RuntimeError(
            "Raw EOM receivables bearer token material must not be configured "
            "on the Atlas API service; provision only "
            "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256 here and keep "
            "the raw token on the caller side."
        )
    token_digest = getattr(resolved, "receivables_service_token_sha256", "")
    token_digest = token_digest.strip() if isinstance(token_digest, str) else ""
    _validate_generated_token_digest(token_digest)


def trusted_receivables_api_config(
    generated: GeneratedReceivablesServiceToken,
) -> TrustedReceivablesApiConfig:
    """Build trusted in-process auth config for tests before Slice C provisioning."""
    return TrustedReceivablesApiConfig(
        receivables_api_enabled=True,
        receivables_service_token_sha256=generated.sha256,
    )


def get_receivables_api_config() -> EOMInvoicingConfig:
    """Return the runtime config through an overrideable request boundary."""
    return invoicing_settings


async def require_receivables_api(
    authorization: str = Header(default="", alias="Authorization"),
    config: ReceivablesApiAuthConfig = Depends(get_receivables_api_config),
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
        _validate_generated_token(provided_token)
    except RuntimeError as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc
    try:
        provided_digest = _token_sha256(provided_token)
    except UnicodeEncodeError as exc:
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
