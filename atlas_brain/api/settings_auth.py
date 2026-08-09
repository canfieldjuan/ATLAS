"""Fail-closed service-token auth for the public /api/v1/settings router (#2335).

The settings router mutates alert/email/LLM/integration destinations and persists
them to .env.local, and every route is reachable from the public Tailscale Funnel
(the same surface as the lead-intake endpoint). Without auth an unauthenticated
caller could repoint outbound destinations (ntfy_url, imap_host, ollama_url,
asr_url, ha_url, mqtt_host, default_from) at a host they control — leaking
env-held credentials / in-flight data — or flip toggles like alerts_enabled to
silence the operator. This dependency requires a deploy-time bearer service
token, mirroring the receivables API (atlas_brain/api/invoicing/auth.py).

Digest-only, fail-closed: only the SHA-256 digest is configured on the server;
the raw token stays with the caller. When no valid digest is provisioned the
router is UNAVAILABLE (503), NEVER open — a missing/blank/malformed token can
never degrade into unauthenticated access.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from fastapi import Depends, Header, HTTPException

from ..config import SettingsAdminConfig, settings

# Lowercase SHA-256 hex digest, matching the service-token provisioning format.
_SHA256_HEX_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def get_settings_admin_config() -> SettingsAdminConfig:
    """Return the runtime config through an overrideable request boundary."""
    return settings.settings_admin


async def require_settings_admin(
    authorization: str = Header(default="", alias="Authorization"),
    config: SettingsAdminConfig = Depends(get_settings_admin_config),
) -> None:
    """Require the configured bearer token. Fail-closed on every misconfiguration."""
    expected_digest = (config.token_sha256 or "").strip().lower()
    # No/blank/malformed digest => the admin API is unavailable, not open.
    if not expected_digest or _SHA256_HEX_RE.fullmatch(expected_digest) is None:
        raise HTTPException(status_code=503, detail="Settings admin API is not configured")

    scheme, separator, provided = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not provided.strip():
        raise HTTPException(status_code=401, detail="Bearer token required")

    try:
        provided_digest = hashlib.sha256(provided.strip().encode("ascii")).hexdigest()
    except UnicodeEncodeError as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc
    if not hmac.compare_digest(provided_digest, expected_digest):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
