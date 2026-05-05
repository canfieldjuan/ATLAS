"""Tests for the LLM Gateway API key auth substrate (PR-D1).

Covers the pure (non-DB) helpers in ``atlas_brain.auth.api_keys``:
generation format, hashing one-wayness, prefix extraction, constant-
time compare. The DB-backed paths (insert/list/lookup/revoke/touch)
plus ``require_api_key`` dependency are integration tests gated on a
running Postgres -- they live alongside other auth integration tests
and are skipped when no pool is available.
"""

from __future__ import annotations

import os
import re

import pytest

from atlas_brain.auth.api_keys import (
    DEFAULT_SCOPES,
    KEY_BODY_LENGTH,
    KEY_LIVE_PREFIX,
    KEY_PREFIX_LEN,
    constant_time_equals,
    generate_api_key,
    hash_api_key,
    split_prefix,
)


# ---- Generation format ---------------------------------------------------


def test_generated_key_has_live_prefix():
    raw, _prefix = generate_api_key()
    assert raw.startswith(KEY_LIVE_PREFIX)


def test_generated_key_body_length():
    raw, _prefix = generate_api_key()
    body = raw[len(KEY_LIVE_PREFIX) :]
    assert len(body) == KEY_BODY_LENGTH


def test_generated_key_body_uses_lowercase_base32_alphabet():
    raw, _prefix = generate_api_key()
    body = raw[len(KEY_LIVE_PREFIX) :]
    assert re.fullmatch(r"[a-z2-7]+", body), body


def test_generated_key_prefix_matches_expected_length():
    _raw, prefix = generate_api_key()
    assert len(prefix) == KEY_PREFIX_LEN


def test_generated_keys_are_unique_across_calls():
    keys = {generate_api_key()[0] for _ in range(50)}
    assert len(keys) == 50


# ---- Hashing one-wayness + determinism -----------------------------------


def test_hash_is_deterministic_for_same_input(monkeypatch):
    """With a fixed pepper, the same raw key always hashes to the
    same value. Deterministic hashing is required for the prefix-
    narrow + HMAC-compare lookup pattern in ``lookup_api_key``."""
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "test-pepper-deterministic")

    # Force settings to re-read by clearing the cached module
    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)

    raw = "atls_live_examplekey1234567890abcdefgh"
    h1 = hash_api_key(raw)
    h2 = hash_api_key(raw)
    assert h1 == h2


def test_hash_is_different_for_different_inputs(monkeypatch):
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "test-pepper-distinct")
    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)

    raw_a = "atls_live_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    raw_b = "atls_live_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert hash_api_key(raw_a) != hash_api_key(raw_b)


def test_hash_changes_when_pepper_changes(monkeypatch):
    """Same raw key + different pepper = different hash. Prevents
    cross-environment hash reuse if the pepper rotates."""
    raw = "atls_live_pepperrotationtestkey1234567890ab"
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "test-pepper-one")
    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    h1 = hash_api_key(raw)

    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "test-pepper-two")
    importlib.reload(config_mod)
    h2 = hash_api_key(raw)

    assert h1 != h2


def test_hash_output_format_is_hex_sha256(monkeypatch):
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "test-pepper-format")
    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)

    h = hash_api_key("atls_live_anykey")
    assert len(h) == 64
    assert re.fullmatch(r"[0-9a-f]+", h)


# ---- Prefix extraction --------------------------------------------------


def test_split_prefix_returns_lookup_hint_for_well_formed_key():
    raw, prefix = generate_api_key()
    assert split_prefix(raw) == prefix


def test_split_prefix_rejects_non_atlas_token():
    """JWT-shaped tokens fall through to the JWT extractor, not the
    API-key path. ``split_prefix`` must return the empty string for
    them so ``lookup_api_key`` short-circuits."""
    assert split_prefix("eyJhbGciOiJIUzI1NiJ9.fake.token") == ""


def test_split_prefix_rejects_too_short_input():
    assert split_prefix("atls_live_short") == ""


def test_split_prefix_rejects_empty_input():
    assert split_prefix("") == ""


# ---- Constant-time compare -----------------------------------------------


def test_constant_time_equals_matches_identical_strings():
    assert constant_time_equals("abcdef", "abcdef") is True


def test_constant_time_equals_rejects_different_strings():
    assert constant_time_equals("abcdef", "abcdee") is False


def test_constant_time_equals_rejects_different_length():
    assert constant_time_equals("abc", "abcdef") is False


# ---- Defaults ------------------------------------------------------------


def test_default_scopes_is_llm_wildcard():
    """v1 scope model. PR-D4 expands this when fine-grained scopes
    land. The wildcard locks the default so future restriction is
    additive."""
    assert DEFAULT_SCOPES == ("llm:*",)


# ---- Pepper validator (config-level) -------------------------------------


# Valid Fernet KEK (base64 of 32 bytes) used by tests that enable SaaS
# auth and want to isolate validation to the pepper -- without this, the
# new BYOK_ENCRYPTION_KEK validator (PR-D5) would also fire and the
# pepper-specific assertion would be ambiguous.
_VALID_TEST_KEK = "v1:" + ("A" * 43) + "="


def test_saas_auth_rejects_default_pepper_when_enabled(monkeypatch):
    """When ``ATLAS_SAAS_ENABLED=true``, the default sentinel pepper
    raises a ValueError -- prevents shipping prod with a known pepper."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", _VALID_TEST_KEK)
    # Leave pepper unset -> defaults to sentinel.
    monkeypatch.delenv("ATLAS_SAAS_API_KEY_PEPPER", raising=False)

    import importlib
    import atlas_brain.config as config_mod

    with pytest.raises(Exception):
        importlib.reload(config_mod)

    # Cleanup -- restore a valid environment so subsequent tests don't
    # inherit a half-loaded module.
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "post-test-pepper")
    importlib.reload(config_mod)


def test_saas_auth_accepts_non_default_pepper_when_enabled(monkeypatch):
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "non-default-pepper")
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", _VALID_TEST_KEK)

    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    assert config_mod.settings.saas_auth.api_key_pepper == "non-default-pepper"


def test_saas_auth_rejects_empty_pepper_when_enabled(monkeypatch):
    """An empty pepper HMACs every key with a known empty secret --
    functionally identical to the sentinel default. Reject all
    blank/whitespace values too (Codex review on PR-D1)."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "")
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", _VALID_TEST_KEK)

    import importlib
    import atlas_brain.config as config_mod

    with pytest.raises(Exception):
        importlib.reload(config_mod)

    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "post-test-pepper")
    importlib.reload(config_mod)


def test_saas_auth_rejects_whitespace_only_pepper_when_enabled(monkeypatch):
    """A whitespace-only pepper is also a known secret -- ``.strip()``
    on the validator catches both `"   "` and `"\\t\\n"`."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "true")
    monkeypatch.setenv("ATLAS_SAAS_JWT_SECRET", "non-default-jwt")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "   ")
    monkeypatch.setenv("ATLAS_SAAS_BYOK_ENCRYPTION_KEK", _VALID_TEST_KEK)

    import importlib
    import atlas_brain.config as config_mod

    with pytest.raises(Exception):
        importlib.reload(config_mod)

    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "post-test-pepper")
    importlib.reload(config_mod)


def test_saas_auth_allows_empty_pepper_when_disabled(monkeypatch):
    """Local-dev mode (SaaS auth disabled) does not need a real
    pepper -- the validator only fires when ``enabled=True``."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    monkeypatch.setenv("ATLAS_SAAS_API_KEY_PEPPER", "")

    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)
    assert config_mod.settings.saas_auth.enabled is False
