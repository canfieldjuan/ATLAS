"""Tests for BYOK key encryption + storage + management (PR-D5).

Pure unit + structural tests:
  - Encryption roundtrip with KEK rotation
  - Resolver fallback order (DB -> env)
  - Router signatures + plan-gating dep
  - Migration content (sentinel + composite indexes)

DB-bound integration tests live with other auth integration tests
and are gated on a running Postgres -- not in this file.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest


_MIG_DIR = Path(__file__).resolve().parent.parent / "atlas_brain" / "storage" / "migrations"


def _read_migration(filename: str) -> str:
    return (_MIG_DIR / filename).read_text(encoding="utf-8")


# ---- Migration ----------------------------------------------------------


def test_migration_316_creates_byok_keys_table():
    sql = _read_migration("316_byok_keys.sql")
    assert "CREATE TABLE IF NOT EXISTS byok_keys" in sql
    # FK CASCADE so a deleted account drops its keys.
    assert "REFERENCES saas_accounts(id) ON DELETE CASCADE" in sql
    # Encrypted at rest -- never raw plaintext.
    assert "encrypted_key   BYTEA NOT NULL" in sql
    # Per-row KEK ID for rotation.
    assert "encryption_kid" in sql


def test_migration_316_has_unique_active_constraint():
    """One active row per (account, provider). Customers rotate by
    revoking + adding; the partial unique index enforces this."""
    sql = _read_migration("316_byok_keys.sql")
    assert "uq_byok_keys_one_active_per_provider" in sql
    assert "WHERE revoked_at IS NULL" in sql


def test_migration_316_has_lookup_index():
    """Gateway resolver queries on (account_id, provider) WHERE
    revoked_at IS NULL -- the partial index covers this."""
    sql = _read_migration("316_byok_keys.sql")
    assert "idx_byok_keys_account_provider_active" in sql


# ---- Encryption module --------------------------------------------------


def _setup_kek(monkeypatch, *, kid: str = "v1") -> str:
    """Helper: install a fresh KEK in env + reload config so tests
    that touch encryption see it."""
    from atlas_brain.auth.encryption import generate_kek

    raw_kek = generate_kek()
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"{kid}:{raw_kek}")

    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    return raw_kek


def test_encrypt_then_decrypt_roundtrip(monkeypatch):
    _setup_kek(monkeypatch)
    from atlas_brain.auth.encryption import decrypt_secret, encrypt_secret

    plaintext = "sk-ant-api03-fakekey-for-roundtrip"
    ciphertext, kid = encrypt_secret(plaintext)

    assert kid == "v1"
    # Ciphertext is bytes; never the plaintext.
    assert isinstance(ciphertext, bytes)
    assert plaintext.encode() not in ciphertext

    decoded = decrypt_secret(ciphertext, kid)
    assert decoded == plaintext


def test_encrypt_refuses_empty_plaintext(monkeypatch):
    _setup_kek(monkeypatch)
    from atlas_brain.auth.encryption import encrypt_secret

    with pytest.raises(ValueError):
        encrypt_secret("")


def test_decrypt_returns_none_on_unknown_kid(monkeypatch):
    """When a row's kid isn't in the configured KEK list (e.g.,
    rotated out), decrypt returns None and logs -- caller treats
    None as 'BYOK not available' rather than crashing."""
    _setup_kek(monkeypatch, kid="v1")
    from atlas_brain.auth.encryption import decrypt_secret, encrypt_secret

    ciphertext, _kid = encrypt_secret("sk-ant-fake")
    decoded = decrypt_secret(ciphertext, "v999-not-configured")
    assert decoded is None


def test_kek_rotation_old_rows_still_decrypt(monkeypatch):
    """Add a v2 KEK while keeping v1; v1-encrypted rows still decrypt
    via the multi-key list."""
    from atlas_brain.auth.encryption import generate_kek

    v1 = generate_kek()
    v2 = generate_kek()

    # Start with only v1 -- write a row.
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"v1:{v1}")
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    from atlas_brain.auth.encryption import decrypt_secret, encrypt_secret

    ciphertext, kid_at_write = encrypt_secret("sk-rotation-test")
    assert kid_at_write == "v1"

    # Rotate: prepend v2 (new write key); keep v1 for decrypt.
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"v2:{v2},v1:{v1}")
    importlib.reload(config_mod)

    # Old row still decrypts under v1.
    assert decrypt_secret(ciphertext, "v1") == "sk-rotation-test"

    # New writes go under v2.
    new_ct, new_kid = encrypt_secret("sk-after-rotation")
    assert new_kid == "v2"
    assert decrypt_secret(new_ct, "v2") == "sk-after-rotation"


def test_load_keks_rejects_default_sentinel(monkeypatch):
    """A deployment that ships with the default sentinel KEK fails
    fast -- ``_parse_kek_string`` raises ValueError so the encryption
    module can't be used to encrypt against a known-public key."""
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", "byok-kek-change-me")
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    from atlas_brain.auth.encryption import encrypt_secret

    with pytest.raises(ValueError, match="not configured"):
        encrypt_secret("anything")


def test_load_keks_rejects_malformed_entry(monkeypatch):
    """Malformed entries (no colon, empty kid, bad base64) raise so
    operators catch typos at deploy time."""
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", "this-has-no-colon")
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    from atlas_brain.auth.encryption import encrypt_secret

    with pytest.raises(ValueError, match="Malformed BYOK KEK"):
        encrypt_secret("anything")


def test_load_keks_rejects_short_key(monkeypatch):
    import base64

    short_key = base64.urlsafe_b64encode(b"only-16-bytes!!!").decode()
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"v1:{short_key}")
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    from atlas_brain.auth.encryption import encrypt_secret

    with pytest.raises(ValueError, match="32 raw bytes"):
        encrypt_secret("anything")


# ---- Resolver -----------------------------------------------------------


def test_supported_providers_contract():
    from atlas_brain.services.byok_keys import SUPPORTED_PROVIDERS

    # PR-D5 ships these four providers; new providers extend the
    # tuple without changing the resolver shape.
    assert SUPPORTED_PROVIDERS == ("anthropic", "openrouter", "together", "groq")


def test_lookup_provider_key_sync_uses_env_var_only():
    """The sync helper is the legacy PR-D4 surface -- env-var only.
    The async resolver is the new path that hits the DB."""
    from atlas_brain.services import byok_keys

    src = inspect.getsource(byok_keys.lookup_provider_key)
    # No DB query on the sync path.
    assert "fetchrow" not in src
    assert "_env_var_fallback" in src


def test_lookup_provider_key_async_queries_db_first():
    """The async resolver MUST hit the DB before falling back to env
    var so production deployments use customer-configured keys."""
    from atlas_brain.services import byok_keys

    src = inspect.getsource(byok_keys.lookup_provider_key_async)
    # SQL filter on the (account_id, provider, revoked_at) tuple.
    assert "WHERE account_id = $1 AND provider = $2" in src
    assert "AND revoked_at IS NULL" in src
    # Env fallback only after the DB path.
    assert "_env_var_fallback" in src


def test_lookup_provider_key_async_decrypts_via_kid():
    """The decrypt call must use the row's stored kid (not a hard-
    coded default) so KEK rotation works."""
    from atlas_brain.services import byok_keys

    src = inspect.getsource(byok_keys.lookup_provider_key_async)
    assert "decrypt_secret(bytes(row[\"encrypted_key\"]), row[\"encryption_kid\"])" in src


def test_lookup_provider_key_async_rejects_unsupported(monkeypatch):
    """Non-supported providers short-circuit before any DB call --
    avoids leaking info via DB-error timing."""
    monkeypatch.delenv(
        "ATLAS_BYOK_OPENAI_00000000_0000_0000_0000_000000000000", raising=False
    )

    import asyncio
    from atlas_brain.services.byok_keys import lookup_provider_key_async

    result = asyncio.run(
        lookup_provider_key_async(None, "openai", "00000000-0000-0000-0000-000000000000")
    )
    assert result is None


def test_lookup_provider_key_async_env_fallback(monkeypatch):
    """When the DB has no row (and the pool is uninitialized in this
    test), the env-var fallback resolves the key."""
    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-ant-fake-from-env",
    )

    import asyncio
    from atlas_brain.services.byok_keys import lookup_provider_key_async

    # Pool is None -> resolver skips DB and goes straight to env.
    result = asyncio.run(
        lookup_provider_key_async(None, "anthropic", "00000000-0000-0000-0000-000000000000")
    )
    assert result == "sk-ant-fake-from-env"


# ---- Router routes registered ------------------------------------------


def test_byok_router_exposes_crud_and_providers():
    from atlas_brain.api.byok_keys import router

    paths = sorted({route.path for route in router.routes if hasattr(route, "path")})
    assert "/byok-keys" in paths
    assert "/byok-keys/{key_id}" in paths
    assert "/byok-keys/providers" in paths


def test_byok_router_post_uses_dual_auth():
    """add_key must use require_auth_or_api_key (PR-D4) so customers
    can manage keys from the dashboard (JWT) OR via script (API key)."""
    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.add_key)
    assert "require_auth_or_api_key" in src


def test_byok_router_get_uses_dual_auth():
    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.list_keys)
    assert "require_auth_or_api_key" in src


def test_byok_router_delete_uses_dual_auth():
    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.revoke_key)
    assert "require_auth_or_api_key" in src


def test_byok_router_registered_in_aggregator():
    """``api/__init__.py`` must include the byok_keys router so it
    actually mounts at ``/api/v1/byok-keys/*``."""
    import sys
    sys.modules.pop("atlas_brain.api", None)
    api_pkg = importlib.import_module("atlas_brain.api")
    paths = {getattr(route, "path", "") for route in api_pkg.router.routes}
    assert any(p.startswith("/byok-keys") for p in paths)


# ---- Schema shape ------------------------------------------------------


def test_add_request_validates_provider():
    from atlas_brain.api.byok_keys import AddBYOKKeyRequest

    req = AddBYOKKeyRequest(provider="anthropic", raw_key="sk-ant-fake-key-1234")
    assert req.provider == "anthropic"
    assert req.label == ""


def test_add_request_rejects_short_key():
    from atlas_brain.api.byok_keys import AddBYOKKeyRequest

    with pytest.raises(Exception):
        AddBYOKKeyRequest(provider="anthropic", raw_key="short")


def test_view_response_omits_sensitive_fields():
    """BYOKKeyView is the customer-facing display shape. It MUST NOT
    expose encrypted_key, encryption_kid, or any plaintext field."""
    from atlas_brain.api.byok_keys import BYOKKeyView

    fields = set(BYOKKeyView.model_fields.keys())
    forbidden = {"encrypted_key", "encryption_kid", "raw_key", "plaintext", "key"}
    assert not (fields & forbidden), f"BYOKKeyView leaks sensitive field(s): {fields & forbidden}"


# ---- Config validator --------------------------------------------------


def test_saas_auth_rejects_default_byok_kek_when_enabled(monkeypatch):
    """When SaaS auth is enabled, the default sentinel KEK raises
    ValueError so prod can't ship with a known KEK."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    # Leave KEK at default sentinel.
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", "byok-kek-change-me")

    import atlas_brain.config as config_mod

    with pytest.raises(Exception, match="BYOK_ENCRYPTION_KEK"):
        importlib.reload(config_mod)

    # Restore valid env so subsequent tests don't inherit a half-loaded
    # module.
    monkeypatch.setenv(
        "ATLAS_SAAS_BYOK_ENCRYPTION_KEK",
        "v1:" + ("A" * 43) + "=",
    )
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    importlib.reload(config_mod)


def test_saas_auth_rejects_empty_byok_kek_when_enabled(monkeypatch):
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", "   ")

    import atlas_brain.config as config_mod

    with pytest.raises(Exception, match="BYOK_ENCRYPTION_KEK"):
        importlib.reload(config_mod)

    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    importlib.reload(config_mod)


def test_saas_auth_rejects_malformed_byok_kek_when_enabled(monkeypatch):
    """Codex P2 fix: a malformed KEK like ``v1:not-base64`` used to
    pass startup and crash on first encrypt/decrypt. Validate at
    config load so ops catch typos at boot, not runtime."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    # Malformed: not valid base64 for a 32-byte Fernet key.
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", "v1:not-base64-at-all!!")

    import atlas_brain.config as config_mod

    with pytest.raises(Exception, match="BYOK_ENCRYPTION_KEK invalid"):
        importlib.reload(config_mod)

    # Restore valid env so other tests don't inherit a half-loaded module.
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    importlib.reload(config_mod)


def test_saas_auth_rejects_kek_missing_colon_when_enabled(monkeypatch):
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", "no-colon-here-just-key")

    import atlas_brain.config as config_mod

    with pytest.raises(Exception, match="BYOK_ENCRYPTION_KEK invalid"):
        importlib.reload(config_mod)

    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    importlib.reload(config_mod)


# ---- Codex/Copilot review (P1: fail-closed + duplicate-kid + plan-limit + race) ----


def test_parse_kek_string_rejects_duplicate_kids(monkeypatch):
    """Copilot fix: ``v1:new,v1:old`` must raise -- decrypt_secret
    matches the FIRST entry by kid, so an older key under the same
    kid never gets tried, leaving rows undecryptable."""
    from atlas_brain.auth.encryption import generate_kek, parse_kek_string

    k1 = generate_kek()
    k2 = generate_kek()
    with pytest.raises(ValueError, match="duplicate kid"):
        parse_kek_string(f"v1:{k1},v1:{k2}")


def test_parse_kek_string_accepts_distinct_kids(monkeypatch):
    """Sanity: distinct kids are the rotation pattern; must not
    raise."""
    from atlas_brain.auth.encryption import generate_kek, parse_kek_string

    k1 = generate_kek()
    k2 = generate_kek()
    entries = parse_kek_string(f"v2:{k2},v1:{k1}")
    assert len(entries) == 2


def test_lookup_async_fails_closed_on_db_error(monkeypatch):
    """Copilot P1: a DB error must NOT silently fall back to env
    var -- doing so routes customer traffic through atlas's
    process-level keys during an outage and bills the wrong
    credentials."""
    import asyncio

    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-fake-from-env",
    )

    class _BrokenPool:
        is_initialized = True

        async def fetchrow(self, *_args, **_kwargs):
            raise RuntimeError("DB connection lost")

    from atlas_brain.services.byok_keys import lookup_provider_key_async

    result = asyncio.run(
        lookup_provider_key_async(_BrokenPool(), "anthropic", "00000000-0000-0000-0000-000000000000")
    )
    # Even though env-var is set, the DB error must NOT fall back.
    assert result is None


def test_lookup_async_fails_closed_on_decrypt_failure(monkeypatch):
    """Copilot P1: KEK rotation drift -- a row encrypted under a
    kid no longer in the configured KEK list -- must NOT fall back
    to env. Hides real config breakage otherwise."""
    import asyncio
    from atlas_brain.auth.encryption import generate_kek

    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-fake-from-env",
    )
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"v1:{generate_kek()}")
    importlib.reload(__import__("atlas_brain.config", fromlist=["settings"]))

    class _RotationDriftPool:
        is_initialized = True

        async def fetchrow(self, *_args, **_kwargs):
            # Row exists with a kid that's NOT in the configured KEK list.
            return {
                "id": "00000000-0000-0000-0000-000000000001",
                "encrypted_key": b"some-ciphertext-bytes",
                "encryption_kid": "v999-not-configured",
            }

        async def execute(self, *_args, **_kwargs):
            return None

    from atlas_brain.services.byok_keys import lookup_provider_key_async

    result = asyncio.run(
        lookup_provider_key_async(_RotationDriftPool(), "anthropic", "00000000-0000-0000-0000-000000000000")
    )
    # Decrypt failed -> None, not env-fallback.
    assert result is None


def test_lookup_async_falls_back_when_no_db_row(monkeypatch):
    """Sanity: legitimate "no key configured" case (no DB row) DOES
    use env-var fallback so local dev keeps working."""
    import asyncio

    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-fake-from-env",
    )

    class _EmptyPool:
        is_initialized = True

        async def fetchrow(self, *_args, **_kwargs):
            return None

    from atlas_brain.services.byok_keys import lookup_provider_key_async

    result = asyncio.run(
        lookup_provider_key_async(_EmptyPool(), "anthropic", "00000000-0000-0000-0000-000000000000")
    )
    assert result == "sk-fake-from-env"


def test_byok_router_imports_plan_limits():
    """Copilot fix: ``add_key`` enforces per-plan ``byok_keys_max``.
    The handler must reference LLM_PLAN_LIMITS, otherwise plan-tier
    limits are advertised but never gated."""
    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.add_key)
    assert "LLM_PLAN_LIMITS" in src
    assert "byok_keys_max" in src


def test_byok_router_returns_403_when_at_plan_limit():
    """Copilot fix: when active count >= max_keys, the handler
    raises 403 (not just lets the insert happen). The actual count
    check moved into ``insert_provider_key`` (FOR UPDATE on
    saas_accounts row) to close the COUNT-then-INSERT race; the
    router catches the resulting BYOKKeyLimitExceeded and translates
    it to 403."""
    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.add_key)
    assert "status_code=403" in src
    assert "BYOKKeyLimitExceeded" in src


# ---- Codex/Copilot third pass (race + DB-skip + stale doc) ------------


def test_env_fallback_returns_none_when_saas_auth_enabled(monkeypatch):
    """Copilot fix: when SaaS auth is enabled (prod), the env-var
    fallback must return None -- otherwise a DB outage with
    pool.is_initialized=False silently routes through atlas's
    process-level keys."""
    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-ant-from-env",
    )
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    from atlas_brain.auth.encryption import generate_kek

    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"v1:{generate_kek()}")

    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)

    from atlas_brain.services.byok_keys import _env_var_fallback

    # Env var is set, but SaaS auth is enabled -> fallback must NOT fire.
    assert _env_var_fallback("anthropic", "00000000-0000-0000-0000-000000000000") is None

    # Cleanup
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    importlib.reload(config_mod)


def test_env_fallback_works_when_saas_auth_disabled(monkeypatch):
    """Sanity: local-dev path (SaaS auth off) keeps env fallback."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-ant-from-env",
    )

    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)

    from atlas_brain.services.byok_keys import _env_var_fallback

    assert _env_var_fallback("anthropic", "00000000-0000-0000-0000-000000000000") == "sk-ant-from-env"


def test_lookup_async_skips_db_when_pool_uninitialized(monkeypatch):
    """When ``pool.is_initialized=False`` (DB outage / startup fail),
    ``lookup_provider_key_async`` skips the DB path and falls through
    to ``_env_var_fallback``. Combined with the SaaS-auth gate above,
    this means prod fails closed (env returns None) while dev still
    works (env returns the configured key)."""
    import asyncio

    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    from atlas_brain.auth.encryption import generate_kek

    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"v1:{generate_kek()}")
    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-from-env",
    )

    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)

    class _UninitializedPool:
        is_initialized = False

    from atlas_brain.services.byok_keys import lookup_provider_key_async

    result = asyncio.run(
        lookup_provider_key_async(
            _UninitializedPool(), "anthropic", "00000000-0000-0000-0000-000000000000"
        )
    )
    # Prod (SaaS enabled) + uninitialized pool -> None (fail closed).
    assert result is None

    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    importlib.reload(config_mod)


def test_byok_key_limit_exceeded_exception_exported():
    """The router catches BYOKKeyLimitExceeded specifically. Pin
    that the public name exists."""
    from atlas_brain.services.byok_keys import BYOKKeyLimitExceeded

    assert issubclass(BYOKKeyLimitExceeded, Exception)


def test_insert_provider_key_signature_accepts_max_keys():
    """The plan-limit cap moved INTO the insert transaction (FOR
    UPDATE on saas_accounts) to close the COUNT-then-INSERT race.
    Pin the new ``max_keys`` kwarg."""
    import inspect

    from atlas_brain.services.byok_keys import insert_provider_key

    sig = inspect.signature(insert_provider_key)
    assert "max_keys" in sig.parameters
    assert sig.parameters["max_keys"].default is None


def test_insert_provider_key_locks_saas_accounts_row():
    """Source-text inspection: insert_provider_key must lock the
    saas_accounts row before counting+inserting so concurrent
    submissions for the same account serialize."""
    import inspect

    from atlas_brain.services import byok_keys

    src = inspect.getsource(byok_keys.insert_provider_key)
    assert "FOR UPDATE" in src
    assert "saas_accounts WHERE id = $1" in src


def test_byok_router_passes_max_keys_to_service():
    """Router pulls ``byok_keys_max`` from LLM_PLAN_LIMITS and
    passes it through; the service does the count check inside the
    transaction. The router must NOT do its own pre-count (that
    was the racy pattern)."""
    import inspect

    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.add_key)
    # Must pass max_keys to the service.
    assert "max_keys=max_keys" in src
    # Must NOT do the racy COUNT + insert pattern in the route.
    assert "active_count_row = await pool.fetchrow" not in src


def test_byok_router_translates_limit_exception_to_403():
    """``BYOKKeyLimitExceeded`` is caught and translated to 403 so
    the customer sees a deterministic plan-limit response."""
    import inspect

    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.add_key)
    assert "BYOKKeyLimitExceeded" in src
    assert "status_code=403" in src


def test_byok_router_handles_unique_violation_as_409():
    """Copilot fix: concurrent rotate/add for the same
    (account_id, provider) races on the partial UNIQUE index. The
    losing INSERT raises asyncpg.UniqueViolationError; the handler
    must translate to 409, not let it become a 500."""
    from atlas_brain.api import byok_keys

    src = inspect.getsource(byok_keys.add_key)
    assert "UniqueViolationError" in src
    assert "status_code=409" in src


def test_saas_auth_accepts_valid_byok_kek_when_enabled(monkeypatch):
    """Sanity: a real Fernet key passes the validator and the
    config object exposes it."""
    from atlas_brain.auth.encryption import generate_kek

    valid_kek = generate_kek()
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", f"v1:{valid_kek}")

    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    assert config_mod.settings.saas_auth.byok_encryption_kek == f"v1:{valid_kek}"

    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    importlib.reload(config_mod)
