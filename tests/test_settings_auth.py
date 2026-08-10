"""Auth gate for the public /api/v1/settings router + session login (#2335).

The router is reachable from the public Tailscale Funnel, so every route must
require the deploy-time admin token, presented EITHER as a bearer (CLI/API) OR as
an HttpOnly signed session cookie (browser). It must FAIL CLOSED: a missing /
blank / malformed / placeholder-derived server digest yields 503 (unavailable),
never open.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from unittest.mock import MagicMock

import pytest

# Match the leads-intake test's import backstop so the api package imports.
_asyncpg = MagicMock()
_asyncpg_ex = MagicMock()
_asyncpg_ex.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg.exceptions = _asyncpg_ex
sys.modules.setdefault("asyncpg", _asyncpg)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_ex)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# `atlas_brain.api.settings` (the submodule) is shadowed by the config `settings`
# object bound in the api package __init__, so import the module unambiguously.
settings_mod = importlib.import_module("atlas_brain.api.settings")
session_mod = importlib.import_module("atlas_brain.api.settings_session")
from atlas_brain.api.settings_auth import (  # noqa: E402
    SESSION_COOKIE_NAME,
    generate_settings_admin_service_token,
    get_settings_admin_config,
    mint_settings_session,
)
from atlas_brain.config import SettingsAdminConfig  # noqa: E402

# A properly generated admin token (prefix + entropy) and its digest.
RAW_TOKEN, DIGEST = generate_settings_admin_service_token()
PLACEHOLDER_DIGEST = "0" * 64  # in the established _PLACEHOLDER_TOKEN_DIGESTS set
NOTIF = "/api/v1/settings/notifications"
SESSION = "/api/v1/settings/session"


def _client(digest: str) -> TestClient:
    app = FastAPI()
    app.include_router(settings_mod.router, prefix="/api/v1")
    app.include_router(session_mod.router, prefix="/api/v1")
    app.dependency_overrides[get_settings_admin_config] = (
        lambda: SettingsAdminConfig(token_sha256=digest)
    )
    return TestClient(app)


def _cookie_from_login(resp) -> str:
    """Extract the raw session cookie value from a login Set-Cookie header.

    (httpx will not resend a Secure cookie over http://testserver, so tests carry
    it explicitly via a Cookie header.)
    """
    set_cookie = resp.headers.get("set-cookie", "")
    assert set_cookie.startswith(f"{SESSION_COOKIE_NAME}=")
    return set_cookie.split(f"{SESSION_COOKIE_NAME}=", 1)[1].split(";", 1)[0]


# ── fail-closed on server misconfiguration ──────────────────────────────────

@pytest.mark.parametrize("digest", ["", PLACEHOLDER_DIGEST, "not-a-valid-sha256-digest"])
def test_unconfigured_or_placeholder_or_malformed_digest_returns_503(digest):
    """No/placeholder/malformed digest => admin API UNAVAILABLE, never open —
    even with a syntactically valid bearer, on both the gate and the login."""
    c = _client(digest)
    assert c.get(NOTIF, headers={"Authorization": f"Bearer {RAW_TOKEN}"}).status_code == 503
    assert c.post(SESSION, headers={"Authorization": f"Bearer {RAW_TOKEN}"}).status_code == 503


# ── bearer path ─────────────────────────────────────────────────────────────

def test_configured_missing_credential_returns_401():
    assert _client(DIGEST).get(NOTIF).status_code == 401


@pytest.mark.parametrize(
    "header",
    ["Basic abc", RAW_TOKEN, "Bearer", "Bearer ", "token " + RAW_TOKEN, "Bearer wrong-token"],
)
def test_configured_bad_authorization_returns_401(header):
    assert _client(DIGEST).get(NOTIF, headers={"Authorization": header}).status_code == 401


def test_correct_bearer_passes_the_gate():
    r = _client(DIGEST).get(NOTIF, headers={"Authorization": f"Bearer {RAW_TOKEN}"})
    assert r.status_code == 200


def test_nongenerated_token_is_invalid_even_if_its_digest_is_configured():
    """Strength enforcement: a weak/non-generated token is rejected even when an
    operator provisions its digest — it lacks the generated prefix/entropy, so it
    can never match (401), and the same holds at the login endpoint."""
    weak = "password123"
    weak_digest = hashlib.sha256(weak.encode("ascii")).hexdigest()  # not a placeholder
    c = _client(weak_digest)
    assert c.get(NOTIF, headers={"Authorization": f"Bearer {weak}"}).status_code == 401
    assert c.post(SESSION, json={"token": weak}).status_code == 401


def test_low_entropy_but_correctly_shaped_token_is_invalid():
    """Entropy floor: a token with the right prefix + length but a low-unique-char
    payload (e.g. a repeated character) is rejected even if its digest is provisioned."""
    low_entropy = "eomset_v1_" + ("A" * 43)  # correct prefix + 43-char payload, 1 unique
    low_digest = hashlib.sha256(low_entropy.encode("ascii")).hexdigest()
    c = _client(low_digest)
    assert c.get(NOTIF, headers={"Authorization": f"Bearer {low_entropy}"}).status_code == 401
    assert c.post(SESSION, json={"token": low_entropy}).status_code == 401


def test_patch_mutation_blocked_when_credential_missing():
    r = _client(DIGEST).patch(NOTIF, json={"ntfy_url": "https://attacker.example"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_non_ascii_bearer_is_invalid_returns_401():
    """A non-ascii bearer can't be sent over HTTP (the client rejects it), but the
    server guard must still 401 rather than 500 if one reaches it directly."""
    from fastapi import HTTPException
    from atlas_brain.api.settings_auth import require_settings_admin
    with pytest.raises(HTTPException) as exc_info:
        await require_settings_admin(
            authorization="Bearer tökén",
            session_cookie="",
            config=SettingsAdminConfig(token_sha256=DIGEST),
        )
    assert exc_info.value.status_code == 401


# ── session login / cookie path ─────────────────────────────────────────────

def test_login_with_bearer_sets_hardened_cookie():
    r = _client(DIGEST).post(SESSION, headers={"Authorization": f"Bearer {RAW_TOKEN}"})
    assert r.status_code == 200
    set_cookie = r.headers.get("set-cookie", "").lower()
    assert "httponly" in set_cookie
    assert "secure" in set_cookie
    assert "samesite=strict" in set_cookie
    assert "path=/api/v1/settings" in set_cookie


def test_login_with_json_body_token_succeeds():
    r = _client(DIGEST).post(SESSION, json={"token": RAW_TOKEN})
    assert r.status_code == 200
    assert r.headers.get("set-cookie", "").startswith(f"{SESSION_COOKIE_NAME}=")


def test_login_with_invalid_token_returns_401_and_sets_no_cookie():
    r = _client(DIGEST).post(SESSION, json={"token": "wrong-token"})
    assert r.status_code == 401
    assert "set-cookie" not in {k.lower() for k in r.headers}


def test_valid_session_cookie_passes_the_gate():
    c = _client(DIGEST)
    cookie = _cookie_from_login(c.post(SESSION, headers={"Authorization": f"Bearer {RAW_TOKEN}"}))
    r = c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={cookie}"})
    assert r.status_code == 200


def test_tampered_session_cookie_is_invalid_returns_401():
    c = _client(DIGEST)
    cookie = _cookie_from_login(c.post(SESSION, headers={"Authorization": f"Bearer {RAW_TOKEN}"}))
    tampered = cookie[:-1] + ("0" if cookie[-1] != "0" else "1")
    r = c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={tampered}"})
    assert r.status_code == 401


def test_expired_session_cookie_is_invalid_returns_401():
    c = _client(DIGEST)
    expired = mint_settings_session(DIGEST, ttl_seconds=-10)  # exp in the past
    r = c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={expired}"})
    assert r.status_code == 401


@pytest.mark.parametrize("exp", ["9" * 5000, "12.3", " 12", "0x1f"])
def test_session_cookie_with_ascii_malformed_expiry_returns_401(exp):
    """An ASCII-but-non-decimal or oversized expiry must 401 end-to-end, not 500."""
    c = _client(DIGEST)
    r = c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}=v1.{exp}.deadbeef"})
    assert r.status_code == 401


@pytest.mark.parametrize("exp", ["9" * 5000, "²", "1²3", "٣", "12.3", " 12", "0x1f", "", "-1"])
def test_verify_settings_session_rejects_malformed_expiry(exp):
    """Guard the numeric-parse directly (httpx cannot send a non-ASCII cookie
    header): str.isdigit() accepts non-ASCII digits like '²'/'٣' that int()
    rejects, and huge values overflow int() — all must be rejected (False), so
    the endpoint fails closed with 401 rather than raising a 500."""
    from atlas_brain.api.settings_auth import verify_settings_session
    assert verify_settings_session(f"v1.{exp}.deadbeef", DIGEST) is False


def test_session_cookie_with_invalid_signature_is_rejected():
    """A cookie minted against a different digest must not authenticate here."""
    c = _client(DIGEST)
    foreign = mint_settings_session("f" * 64)
    r = c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={foreign}"})
    assert r.status_code == 401


def test_logout_clears_the_cookie():
    r = _client(DIGEST).delete(SESSION)
    assert r.status_code == 204
    set_cookie = r.headers.get("set-cookie", "").lower()
    assert set_cookie.startswith(f"{SESSION_COOKIE_NAME}=")
    assert "max-age=0" in set_cookie or "expires=" in set_cookie


# ── router-level coverage ───────────────────────────────────────────────────

def test_production_aggregate_router_serves_the_session_route():
    """Regression guard: the REAL `atlas_brain.api` aggregate router — the one
    `atlas_brain.main` mounts under /api/v1 — must actually register the session
    route at runtime (a dropped or commented-out `include_router(
    settings_session_router)` leaves it absent). Path membership on the live route
    table, so a commented-out include cannot pass it; no dependency-identity
    assertion (that was order-fragile in the full suite). The gate's behavior on
    the settings routes is covered by the tests above."""
    import importlib
    from starlette.routing import Route

    api = importlib.import_module("atlas_brain.api")
    session_mod = importlib.import_module("atlas_brain.api.settings_session")
    session_paths = {r.path for r in session_mod.router.routes if isinstance(r, Route)}
    aggregate_paths = {r.path for r in api.router.routes if isinstance(r, Route)}
    assert session_paths, "settings_session router registers no routes"
    assert session_paths <= aggregate_paths, (
        f"session routes {session_paths} not included in the aggregate"
    )


def test_every_settings_route_is_gated():
    """Router-level dependency covers ALL /settings/* routes, not just notifications."""
    from starlette.routing import Route
    c = _client("")  # unconfigured => every gated route should 503 (never 200) unauth
    paths = sorted({r.path for r in settings_mod.router.routes if isinstance(r, Route)})
    assert len(paths) >= 7  # voice/email/daily/intelligence/llm/notifications/integrations
    for p in paths:
        resp = c.get("/api/v1" + p)
        assert resp.status_code in (503, 405), f"{p} -> {resp.status_code}"
