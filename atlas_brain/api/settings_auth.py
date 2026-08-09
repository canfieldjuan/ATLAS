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
import time

from fastapi import Cookie, Depends, Header, HTTPException

from ..config import SettingsAdminConfig, settings
from ..eom_api.auth import validate_generated_service_token_digest

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


def token_matches(provided: str, expected_digest: str) -> bool:
    """Constant-time check that ``sha256(provided) == expected_digest``.

    A non-ascii token can never match an ascii-hex digest, so reject it up front
    rather than letting ``encode('ascii')`` raise.
    """
    if not provided.isascii():
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
    if not exp_str.isdigit():  # non-numeric expiry is malformed, not an error to raise
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
