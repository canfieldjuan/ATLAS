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
    generate_settings_admin_session_secret,
    get_settings_admin_config,
    mint_settings_session,
)
from atlas_brain.config import SettingsAdminConfig  # noqa: E402

# A properly generated admin token (prefix + entropy) and its digest.
RAW_TOKEN, DIGEST = generate_settings_admin_service_token()
# An INDEPENDENT signing secret — deliberately NOT derived from DIGEST.
SIGNING_SECRET = generate_settings_admin_session_secret()
PLACEHOLDER_DIGEST = "0" * 64  # in the established _PLACEHOLDER_TOKEN_DIGESTS set
NOTIF = "/api/v1/settings/notifications"
SESSION = "/api/v1/settings/session"


def _client(digest: str, secret: str = SIGNING_SECRET) -> TestClient:
    app = FastAPI()
    app.include_router(settings_mod.router, prefix="/api/v1")
    app.include_router(session_mod.router, prefix="/api/v1")
    app.dependency_overrides[get_settings_admin_config] = (
        lambda: SettingsAdminConfig(token_sha256=digest, session_secret=secret)
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
    expired = mint_settings_session(SIGNING_SECRET, DIGEST, ttl_seconds=-10)  # exp in the past
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
    assert verify_settings_session(f"v1.{exp}.deadbeef", SIGNING_SECRET, DIGEST) is False


@pytest.mark.parametrize("cookie", ["v1.1234567890.²", "v1.1234567890.dead²eef", "v1.1234567890.déadbeef"])
def test_verify_rejects_non_ascii_mac_without_crashing(cookie):
    """A non-ASCII MAC must be rejected (False), not raise TypeError in
    hmac.compare_digest (which would surface as a 500 instead of fail-closed 401)."""
    from atlas_brain.api.settings_auth import verify_settings_session
    assert verify_settings_session(cookie, SIGNING_SECRET, DIGEST) is False


def test_session_cookie_with_invalid_signature_is_rejected():
    """A cookie minted with a different signing secret must not authenticate here."""
    c = _client(DIGEST)
    foreign = mint_settings_session("z" * 40, DIGEST)  # a different (valid-length) secret
    r = c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={foreign}"})
    assert r.status_code == 401


def test_cookie_forged_from_the_digest_alone_is_rejected():
    """A read-only disclosure of the token DIGEST must not enable cookie forgery:
    the signing key is an INDEPENDENT secret, so a cookie signed with the digest
    (all an attacker learns from a config/env leak) is rejected (401)."""
    c = _client(DIGEST)
    forged = mint_settings_session(DIGEST, DIGEST)  # attacker signs with the leaked digest
    r = c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={forged}"})
    assert r.status_code == 401


def test_rotating_the_admin_token_invalidates_existing_cookies():
    """Rotating the bearer (new digest) invalidates cookies minted under the old one,
    even if the independent signing secret is retained — the signature is bound to the
    current digest. A fresh cookie under the new digest still authenticates."""
    new_token, new_digest = generate_settings_admin_service_token()
    old_cookie = mint_settings_session(SIGNING_SECRET, DIGEST)  # minted under the OLD digest
    c = _client(new_digest, secret=SIGNING_SECRET)  # server rotated to the NEW digest
    assert c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={old_cookie}"}).status_code == 401
    fresh = _cookie_from_login(c.post(SESSION, headers={"Authorization": f"Bearer {new_token}"}))
    assert c.get(NOTIF, headers={"Cookie": f"{SESSION_COOKIE_NAME}={fresh}"}).status_code == 200


def test_login_returns_503_when_session_signing_secret_missing():
    """Without an independent signing secret the login cannot mint a verifiable
    cookie, so it is unavailable (503) rather than issuing a forgeable one."""
    c = _client(DIGEST, secret="")
    r = c.post(SESSION, headers={"Authorization": f"Bearer {RAW_TOKEN}"})
    assert r.status_code == 503


def test_bearer_still_works_without_session_signing_secret():
    """The cookie path being unavailable (no signing secret) must not break the
    bearer path — a valid bearer still authenticates."""
    c = _client(DIGEST, secret="")
    assert c.get(NOTIF, headers={"Authorization": f"Bearer {RAW_TOKEN}"}).status_code == 200


def test_weak_same_length_session_secret_disables_the_cookie_path():
    """A length-compliant but low-entropy signing secret (e.g. a repeated character)
    is rejected by the entropy floor: the cookie/login path is unavailable (login 503,
    cookies never verify) rather than usable with a guessable key — the bearer path
    still works."""
    c = _client(DIGEST, secret="a" * 40)  # 40 chars but only 1 unique
    assert c.post(SESSION, headers={"Authorization": f"Bearer {RAW_TOKEN}"}).status_code == 503
    assert c.get(NOTIF, headers={"Authorization": f"Bearer {RAW_TOKEN}"}).status_code == 200


def test_config_loads_secrets_from_the_advertised_env_vars(monkeypatch):
    """Load through the REAL environment boundary: the field names + env_prefix must
    derive exactly the advertised env vars, so following the deploy docs populates the
    config (regression for the SESSION_SECRET vs SESSION_SIGNING_SECRET mismatch)."""
    monkeypatch.setenv("ATLAS_SETTINGS_ADMIN_TOKEN_SHA256", "a" * 64)
    monkeypatch.setenv("ATLAS_SETTINGS_ADMIN_SESSION_SECRET", "e" * 40)
    cfg = SettingsAdminConfig()
    assert cfg.token_sha256 == "a" * 64
    assert cfg.session_secret == "e" * 40


def test_logout_clears_the_cookie():
    r = _client(DIGEST).delete(SESSION)
    assert r.status_code == 204
    set_cookie = r.headers.get("set-cookie", "").lower()
    assert set_cookie.startswith(f"{SESSION_COOKIE_NAME}=")
    assert "max-age=0" in set_cookie or "expires=" in set_cookie


# ── router-level coverage ───────────────────────────────────────────────────

def test_init_ast_includes_the_session_router():
    """Regression guard: the api package __init__ must import AND `include_router`
    the session router into the aggregate. Verified by parsing the __init__ AST —
    a commented-out or absent include is NOT an AST call node, so it cannot pass
    (addresses the source-text-substring weakness); and because it parses the file
    rather than reading the shared runtime `atlas_brain.api.router`, it is immune to
    the full-suite import-order/reload pollution that made runtime assertions flaky.
    The gate's runtime behavior on the settings routes is covered by the tests above."""
    import ast
    import importlib
    from pathlib import Path

    api = importlib.import_module("atlas_brain.api")
    tree = ast.parse(Path(api.__file__).read_text(encoding="utf-8"))
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "settings_session"
        and any(alias.asname == "settings_session_router" for alias in node.names)
        for node in ast.walk(tree)
    )
    included = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "include_router"
        and any(isinstance(a, ast.Name) and a.id == "settings_session_router" for a in node.args)
        for node in ast.walk(tree)
    )
    assert imported, "__init__ does not import settings_session_router"
    assert included, "__init__ does not include_router(settings_session_router)"


def test_session_route_reachable_and_gated_via_fresh_app_subprocess():
    """Exercise the REAL application wiring at runtime, but in a FRESH interpreter
    (subprocess) so sibling tests cannot pollute the shared `atlas_brain.api.router`
    (the in-process runtime/mount variants regressed the CI unit-gate for exactly
    that reason). Imports the production aggregate, mounts it, and asserts over real
    HTTP that the session route is served (not 404) AND the gate is active (503 when
    unconfigured) — production reachability + gated behavior, deterministically."""
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys
        from unittest.mock import MagicMock
        _a = MagicMock(); _e = MagicMock()
        _e.UndefinedTableError = type("U", (Exception,), {}); _a.exceptions = _e
        sys.modules.setdefault("asyncpg", _a); sys.modules.setdefault("asyncpg.exceptions", _e)
        import importlib
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        api = importlib.import_module("atlas_brain.api")
        from atlas_brain.api.settings_auth import get_settings_admin_config
        from atlas_brain.config import SettingsAdminConfig
        app = FastAPI(); app.include_router(api.router, prefix="/api/v1")
        app.dependency_overrides[get_settings_admin_config] = lambda: SettingsAdminConfig(token_sha256="")
        c = TestClient(app)
        assert c.get("/api/v1/settings/session").status_code != 404, "session route not mounted"
        assert c.get("/api/v1/settings/notifications").status_code == 503, "gate not active"
        print("WIRING_OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=str(repo_root), capture_output=True, text=True
    )
    assert proc.returncode == 0 and "WIRING_OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
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
