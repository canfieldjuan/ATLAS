"""Login/logout for the settings admin UI (#2335).

The browser cannot hold the raw admin bearer (it is served from the same public
origin, so any build-time secret leaks into the JS bundle). Instead the operator
POSTs the token ONCE to this endpoint, which verifies it and returns an HttpOnly,
Secure, SameSite=Strict session cookie the JS cannot read. Subsequent
``/api/v1/settings/*`` calls authenticate with that cookie
(see ``settings_auth.require_settings_admin``).

This router is deliberately NOT behind ``require_settings_admin`` — you must be
able to log in before you hold a session — but it performs the identical
fail-closed token check, so it is not an unauthenticated surface: 503 until a
valid digest is provisioned, 401 unless the correct token is presented.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Response

from ..config import SettingsAdminConfig
from .settings_auth import (
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_PATH,
    SESSION_TTL_SECONDS,
    bearer_token,
    get_settings_admin_config,
    mint_settings_session,
    resolve_expected_digest,
    token_matches,
)

# Shares the "/settings" prefix with the gated router but is a SEPARATE router
# with NO admin dependency, so only /settings/session is reachable pre-login.
router = APIRouter(prefix="/settings", tags=["Settings"])


@router.post("/session")
async def create_settings_session(
    authorization: str = Header(default="", alias="Authorization"),
    token: str = Body(default="", embed=True),
    config: SettingsAdminConfig = Depends(get_settings_admin_config),
) -> Response:
    """Exchange the admin token (bearer or JSON ``{token}``) for a session cookie."""
    expected_digest = resolve_expected_digest(config)  # 503 if unconfigured/placeholder
    presented = bearer_token(authorization) or (token.strip() if token else "")
    if not presented or not token_matches(presented, expected_digest):
        raise HTTPException(status_code=401, detail="Invalid settings admin token")

    response = Response(
        content='{"authenticated": true, "expires_in": %d}' % SESSION_TTL_SECONDS,
        media_type="application/json",
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=mint_settings_session(expected_digest),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="strict",
        path=SESSION_COOKIE_PATH,
    )
    return response


@router.delete("/session", status_code=204)
async def delete_settings_session() -> Response:
    """Clear the session cookie. Idempotent; requires no auth to log out."""
    response = Response(status_code=204)
    response.delete_cookie(key=SESSION_COOKIE_NAME, path=SESSION_COOKIE_PATH)
    return response
