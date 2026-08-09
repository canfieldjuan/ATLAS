"""Auth gate for the public /api/v1/settings router (#2335).

The router is reachable from the public Tailscale Funnel, so every route must
require the deploy-time bearer service token, and it must FAIL CLOSED: a
missing/blank/malformed server-side digest yields 503 (unavailable), never open.
"""

from __future__ import annotations

import hashlib
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

import importlib  # noqa: E402

# `atlas_brain.api.settings` (the submodule) is shadowed by the config `settings`
# object bound in the api package __init__, so import the module unambiguously.
settings_mod = importlib.import_module("atlas_brain.api.settings")
from atlas_brain.api.settings_auth import get_settings_admin_config  # noqa: E402
from atlas_brain.config import SettingsAdminConfig  # noqa: E402

RAW_TOKEN = "s3cret-admin-token-0123456789abcdef"
DIGEST = hashlib.sha256(RAW_TOKEN.encode("ascii")).hexdigest()
PATH = "/api/v1/settings/notifications"


def _client(digest: str) -> TestClient:
    app = FastAPI()
    app.include_router(settings_mod.router, prefix="/api/v1")
    app.dependency_overrides[get_settings_admin_config] = (
        lambda: SettingsAdminConfig(token_sha256=digest)
    )
    return TestClient(app)


def test_unconfigured_digest_returns_503_even_with_a_bearer():
    """Fail-closed: no server-side token => the admin API is UNAVAILABLE, not open."""
    c = _client("")
    assert c.get(PATH).status_code == 503
    # even a syntactically valid bearer cannot open an unconfigured router
    assert c.get(PATH, headers={"Authorization": f"Bearer {RAW_TOKEN}"}).status_code == 503


def test_malformed_server_digest_returns_503():
    """A non-SHA-256 configured digest is a deploy error => fail closed (503)."""
    c = _client("not-a-valid-sha256-digest")
    assert c.get(PATH, headers={"Authorization": f"Bearer {RAW_TOKEN}"}).status_code == 503


def test_configured_but_no_bearer_returns_401():
    assert _client(DIGEST).get(PATH).status_code == 401


@pytest.mark.parametrize(
    "header",
    ["Basic abc", RAW_TOKEN, "Bearer", "Bearer ", "token " + RAW_TOKEN, "Bearer wrong-token"],
)
def test_configured_bad_authorization_returns_401(header):
    assert _client(DIGEST).get(PATH, headers={"Authorization": header}).status_code == 401


@pytest.mark.asyncio
async def test_non_ascii_bearer_does_not_crash_returns_401():
    """A non-ascii bearer can't be sent over HTTP (the client rejects it), but the
    server guard must still 401 rather than 500 if one reaches it directly."""
    from fastapi import HTTPException
    from atlas_brain.api.settings_auth import require_settings_admin
    with pytest.raises(HTTPException) as exc_info:
        await require_settings_admin(
            authorization="Bearer tökén",
            config=SettingsAdminConfig(token_sha256=DIGEST),
        )
    assert exc_info.value.status_code == 401


def test_correct_bearer_passes_the_gate():
    """The right token reaches the handler (GET returns the current settings)."""
    r = _client(DIGEST).get(PATH, headers={"Authorization": f"Bearer {RAW_TOKEN}"})
    assert r.status_code == 200


def test_patch_mutation_is_blocked_without_a_bearer():
    """The mutation path (repoint a destination) is gated too — 401, no write."""
    r = _client(DIGEST).patch(PATH, json={"ntfy_url": "https://attacker.example"})
    assert r.status_code == 401


def test_every_settings_route_is_gated():
    """Router-level dependency covers ALL /settings/* routes, not just notifications."""
    from starlette.routing import Route
    c = _client("")  # unconfigured => every route should 503, proving the gate is on all
    paths = sorted({r.path for r in settings_mod.router.routes if isinstance(r, Route)})
    assert len(paths) >= 7  # voice/email/daily/intelligence/llm/notifications/integrations
    for p in paths:
        resp = c.get("/api/v1" + p)
        # GET-only routes 503; PATCH-only routes 405 for GET but STILL never 200 unauth
        assert resp.status_code in (503, 405), f"{p} -> {resp.status_code}"
