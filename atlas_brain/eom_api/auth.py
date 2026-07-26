"""Fail-closed service authentication for the slim EOM API profile."""

from __future__ import annotations

import hmac
import hashlib
import secrets
from dataclasses import dataclass
from re import fullmatch

from fastapi import Depends, Header, HTTPException

from .config import EOMInvoicingConfig, invoicing_settings

_GENERATED_TOKEN_PREFIX = "eomrx_v1_"
_GENERATED_TOKEN_PROVENANCE_KEY_PREFIX = "eomrxk_v1_"
_GENERATED_TOKEN_PROVENANCE_PREFIX = "eomrxp_v1_"
_GENERATED_TOKEN_RANDOM_LENGTH = 43
_TOKEN_RANDOM_PATTERN = r"[A-Za-z0-9_-]+"
_TOKEN_SHA256_PATTERN = r"[0-9a-f]{64}"
_TOKEN_PROVENANCE_PATTERN = rf"{_GENERATED_TOKEN_PROVENANCE_PREFIX}[0-9a-f]{{64}}"
_PROVENANCE_CONTEXT = b"atlas-eom-receivables-token-v1"
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
    """Generated service-token material for provisioning two systems."""

    token: str
    sha256: str
    provenance_key: str
    provenance: str


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


_PLACEHOLDER_TOKEN_DIGESTS = {
    _token_sha256(token) for token in _PLACEHOLDER_TOKENS
}


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


def _validate_generated_provenance_key(provenance_key: str) -> None:
    if not provenance_key.startswith(_GENERATED_TOKEN_PROVENANCE_KEY_PREFIX):
        raise RuntimeError(
            "Receivables service token provenance key must be generated with "
            "the eomrxk_v1_ prefix"
        )
    random_part = provenance_key.removeprefix(_GENERATED_TOKEN_PROVENANCE_KEY_PREFIX)
    if len(random_part) < _GENERATED_TOKEN_RANDOM_LENGTH:
        raise RuntimeError(
            "Receivables service token provenance key random payload is too short"
        )
    if fullmatch(_TOKEN_RANDOM_PATTERN, random_part) is None:
        raise RuntimeError(
            "Receivables service token provenance key contains invalid characters"
        )


def _generated_token_provenance(token_digest: str, provenance_key: str) -> str:
    _validate_generated_token_digest(token_digest)
    _validate_generated_provenance_key(provenance_key)
    message = _PROVENANCE_CONTEXT + b":" + token_digest.encode("ascii")
    digest = hmac.new(
        provenance_key.encode("ascii"),
        message,
        hashlib.sha256,
    ).hexdigest()
    return f"{_GENERATED_TOKEN_PROVENANCE_PREFIX}{digest}"


def receivables_service_token_sha256(token: str) -> str:
    """Return the digest accepted by the EOM API for a generated token."""
    _validate_generated_token(token)
    return _token_sha256(token)


def generate_receivables_service_token() -> GeneratedReceivablesServiceToken:
    """Generate EOM receivables service-token material for secret provisioning."""
    token = f"{_GENERATED_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    token_digest = receivables_service_token_sha256(token)
    provenance_key = (
        f"{_GENERATED_TOKEN_PROVENANCE_KEY_PREFIX}{secrets.token_urlsafe(32)}"
    )
    return GeneratedReceivablesServiceToken(
        token=token,
        sha256=token_digest,
        provenance_key=provenance_key,
        provenance=_generated_token_provenance(token_digest, provenance_key),
    )


def _validate_generated_token_digest(token_digest: str) -> None:
    if not token_digest:
        raise RuntimeError(
            "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256 is required when "
            "ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true"
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


def _validate_generated_token_provenance(
    token_digest: str,
    provenance_key: str,
    provenance: str,
) -> None:
    if not provenance_key.strip():
        raise RuntimeError(
            "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_PROVENANCE_KEY is "
            "required when ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true"
        )
    if not provenance.strip():
        raise RuntimeError(
            "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_PROVENANCE is required "
            "when ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true"
        )
    if fullmatch(_TOKEN_PROVENANCE_PATTERN, provenance) is None:
        raise RuntimeError(
            "Receivables service token provenance must be generated with the "
            "eomrxp_v1_ prefix"
        )
    expected = _generated_token_provenance(token_digest, provenance_key)
    if not hmac.compare_digest(provenance, expected):
        raise RuntimeError(
            "Receivables service token digest is not bound to generated "
            "provenance; regenerate the EOM receivables service-token material"
        )


def validate_receivables_api_config(
    config: EOMInvoicingConfig | None = None,
) -> None:
    """Fail startup when an enabled finance API has an unsafe token."""
    resolved = config or invoicing_settings
    if not resolved.receivables_api_enabled:
        return
    token_digest = resolved.receivables_service_token_sha256.strip()
    _validate_generated_token_digest(token_digest)
    _validate_generated_token_provenance(
        token_digest,
        resolved.receivables_service_token_provenance_key.strip(),
        resolved.receivables_service_token_provenance.strip(),
    )


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
    if separator != " " or scheme.lower() != "bearer" or not provided.strip():
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        provided_digest = _token_sha256(provided.strip())
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
