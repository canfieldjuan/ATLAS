"""Fail-closed auth for the public /api/v1/settings router (#2335).

The settings router mutates alert/email/LLM/integration destinations and persists
them to .env.local, and every route is reachable from the public Tailscale Funnel
(the same surface as the lead-intake endpoint). Without auth an unauthenticated
caller could repoint outbound destinations (ntfy_url, imap_host, ollama_url,
asr_url, ha_url, mqtt_host, default_from) at a host they control — leaking
env-held credentials / in-flight data — or flip toggles like alerts_enabled to
silence the operator.

Two credential shapes are accepted, both proving possession of the deploy-time
admin token:

- ``Authorization: Bearer <token>`` — for CLI/API callers.
- an HttpOnly signed **session cookie** — for the browser admin UI, which is
  served from ``atlas-ui/dist`` on the SAME public origin (atlas_brain/main.py).
  A build-time bearer would be inlined into publicly-fetchable JS, so the browser
  instead exchanges the token for an opaque cookie it cannot read (see
  ``settings_session.py``) and carries that.

Digest-only, fail-closed: only the SHA-256 digest is configured on the server via
``validate_generated_service_token_digest`` (the established service-token
validator, which also rejects placeholder-derived digests). When no VALID digest
is provisioned the router is UNAVAILABLE (503), NEVER open — a missing / blank /
malformed / placeholder token can never degrade into unauthenticated access.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from fastapi import Cookie, Depends, Header, HTTPException

from ..config import SettingsAdminConfig, settings
from ..eom_api.auth import (
    validate_generated_service_token,
    validate_generated_service_token_digest,
)

# The admin token must be a generated service token (prefix + entropy), not any
# string whose digest happens to be provisioned — so a weak/guessable token can
# never authenticate even if its digest is configured. Mirrors the receivables
# gate's request-time strength enforcement.
_SETTINGS_TOKEN_PREFIX = "eomset_v1_"
# Entropy floor on the random payload, matching the sibling funnel gate
# (atlas_brain/eom_api/funnel_auth.py) — rejects correctly-shaped but low-entropy
# tokens (e.g. a repeated-character payload) even if their digest is provisioned.
_MIN_TOKEN_UNIQUE_CHARACTERS = 16
_GENERATOR_REFERENCE = (
    "atlas_brain.api.settings_auth.generate_settings_admin_service_token()"
)

# The browser carries this HttpOnly cookie instead of a bearer the JS could read.
SESSION_COOKIE_NAME = "atlas_settings_admin_session"
# Scope the cookie to the settings API so it is sent to both the login endpoint
# (/api/v1/settings/session) and the gated routes (/api/v1/settings/*) only.
SESSION_COOKIE_PATH = "/api/v1/settings"
SESSION_TTL_SECONDS = 12 * 60 * 60  # 12h operator session
_SESSION_VERSION = "v1"


def get_settings_admin_config() -> SettingsAdminConfig:
    """Return the runtime config through an overrideable request boundary."""
    return settings.settings_admin


def resolve_expected_digest(config: SettingsAdminConfig) -> str:
    """Return the validated lowercase digest, or raise 503 (fail-closed).

    Delegates to the established service-token validator, so an empty, malformed,
    OR placeholder-derived digest (e.g. sha256("change-me")) makes the admin API
    unavailable rather than open.
    """
    expected_digest = (config.token_sha256 or "").strip().lower()
    try:
        validate_generated_service_token_digest(
            expected_digest, service_name="Settings admin"
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Settings admin API is not configured"
        ) from exc
    return expected_digest


def bearer_token(authorization: str) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header, or None."""
    scheme, separator, provided = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not provided.strip():
        return None
    return provided.strip()


def generate_settings_admin_service_token() -> tuple[str, str]:
    """Return ``(raw_token, sha256_digest)`` for provisioning the admin token.

    Provision the digest as ``ATLAS_SETTINGS_ADMIN_TOKEN_SHA256`` and keep the raw
    token on the caller/operator side.
    """
    token = f"{_SETTINGS_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    return token, hashlib.sha256(token.encode("ascii")).hexdigest()


def token_matches(provided: str, expected_digest: str) -> bool:
    """Constant-time check that a GENERATED admin token hashes to the digest.

    The presented token must first pass the generated-service-token format check
    (prefix + fixed-length url-safe random payload) — so a weak, guessable, or
    non-generated token can never match even if its digest were provisioned. A
    non-ascii token fails the format check (charset), so we never reach the
    ascii encode.
    """
    try:
        validate_generated_service_token(
            provided,
            token_prefix=_SETTINGS_TOKEN_PREFIX,
            service_name="Settings admin",
            generator_reference=_GENERATOR_REFERENCE,
            exact_random_length=True,
            minimum_unique_characters=_MIN_TOKEN_UNIQUE_CHARACTERS,
        )
    except RuntimeError:
        return False
    provided_digest = hashlib.sha256(provided.encode("ascii")).hexdigest()
    return hmac.compare_digest(provided_digest, expected_digest)


def _session_signing_key(expected_digest: str) -> bytes:
    """Domain-separated signing key derived from the server-only digest.

    The digest never leaves the server, so it is a safe HMAC key; deriving a
    dedicated key keeps the session signature distinct from the raw digest.
    """
    return hmac.new(
        expected_digest.encode("ascii"),
        b"atlas-settings-admin-session-v1",
        hashlib.sha256,
    ).digest()


def _sign_session(expiry_epoch: int, expected_digest: str) -> str:
    payload = f"{_SESSION_VERSION}.{expiry_epoch}"
    mac = hmac.new(
        _session_signing_key(expected_digest), payload.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return f"{payload}.{mac}"


def mint_settings_session(
    expected_digest: str,
    *,
    ttl_seconds: int = SESSION_TTL_SECONDS,
    now: float | None = None,
) -> str:
    """Return a signed session cookie value valid for ``ttl_seconds`` from now."""
    current = int(now if now is not None else time.time())
    return _sign_session(current + ttl_seconds, expected_digest)


def verify_settings_session(
    cookie_value: str, expected_digest: str, *, now: float | None = None
) -> bool:
    """Return True iff ``cookie_value`` is a well-formed, unexpired, valid signature."""
    if not cookie_value:
        return False
    parts = cookie_value.split(".")
    if len(parts) != 3:
        return False
    version, exp_str, _mac = parts
    if version != _SESSION_VERSION:
        return False
    # Require ASCII decimal digits, bounded length, BEFORE int(). str.isdigit()
    # alone accepts non-ASCII digit characters (e.g. "²") that int() rejects with
    # ValueError, and a many-thousand-digit string overflows CPython's conversion
    # limit — both would 500 instead of failing closed. isascii()+isdigit() admits
    # only [0-9]; a valid epoch is ~10 digits, so cap at 20. Malformed => 401.
    if not (exp_str.isascii() and exp_str.isdigit()) or len(exp_str) > 20:
        return False
    expiry_epoch = int(exp_str)
    expected_value = _sign_session(expiry_epoch, expected_digest)
    if not hmac.compare_digest(cookie_value, expected_value):
        return False
    current = now if now is not None else time.time()
    return expiry_epoch > current


async def require_settings_admin(
    authorization: str = Header(default="", alias="Authorization"),
    session_cookie: str = Cookie(default="", alias=SESSION_COOKIE_NAME),
    config: SettingsAdminConfig = Depends(get_settings_admin_config),
) -> None:
    """Require a valid session cookie OR bearer token. Fail-closed on misconfig."""
    expected_digest = resolve_expected_digest(config)  # 503 if unconfigured/placeholder
    if verify_settings_session(session_cookie, expected_digest):
        return
    token = bearer_token(authorization)
    if token is not None and token_matches(token, expected_digest):
        return
    raise HTTPException(
        status_code=401, detail="Settings admin authentication required"
    )
