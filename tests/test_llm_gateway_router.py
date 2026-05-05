"""Tests for the LLM Gateway router (PR-D4).

Pure structural tests: route registration, schema shape, BYOK
resolver fallback, and source-text inspection of the chat /
usage endpoint contract. DB-bound integration tests (live chat
call, usage rollup against a populated llm_usage) live alongside
other auth integration fixtures and are gated on a running
Postgres -- not in this file.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import pytest


# ---- BYOK resolver -------------------------------------------------------


def test_byok_supported_providers_includes_anthropic():
    from atlas_brain.services.byok_keys import SUPPORTED_PROVIDERS

    assert "anthropic" in SUPPORTED_PROVIDERS


def test_byok_lookup_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        raising=False,
    )
    from atlas_brain.services.byok_keys import lookup_provider_key

    assert (
        lookup_provider_key("anthropic", "00000000-0000-0000-0000-000000000000")
        is None
    )


def test_byok_lookup_reads_env_fallback(monkeypatch):
    """The PR-D4 stub uses
    ``ATLAS_BYOK_<PROVIDER>_<UNDERSCORED_UUID>`` so the dev
    environment can supply a key without DB-backed BYOK storage
    (which lands in PR-D5)."""
    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-ant-fake-test-key",
    )
    from atlas_brain.services.byok_keys import lookup_provider_key

    assert (
        lookup_provider_key("anthropic", "00000000-0000-0000-0000-000000000000")
        == "sk-ant-fake-test-key"
    )


def test_byok_lookup_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv(
        "ATLAS_BYOK_OPENAI_00000000_0000_0000_0000_000000000000",
        "sk-fake",
    )
    from atlas_brain.services.byok_keys import lookup_provider_key

    # OpenAI isn't in the SUPPORTED_PROVIDERS tuple for v1.
    assert (
        lookup_provider_key("openai", "00000000-0000-0000-0000-000000000000")
        is None
    )


def test_byok_lookup_strips_whitespace(monkeypatch):
    """Customers occasionally paste keys with leading/trailing
    whitespace; treat those the same as empty so we 503 cleanly
    instead of failing the provider call with a malformed key."""
    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "   ",
    )
    from atlas_brain.services.byok_keys import lookup_provider_key

    assert (
        lookup_provider_key("anthropic", "00000000-0000-0000-0000-000000000000")
        is None
    )


# ---- Router routes registered -------------------------------------------


def test_router_exposes_chat_and_usage():
    from atlas_brain.api.llm_gateway import router

    paths = sorted({route.path for route in router.routes if hasattr(route, "path")})
    assert "/llm/chat" in paths
    assert "/llm/usage" in paths


def test_router_chat_is_post():
    from atlas_brain.api.llm_gateway import router

    chat_route = next(
        r for r in router.routes if hasattr(r, "path") and r.path == "/llm/chat"
    )
    assert "POST" in chat_route.methods


def test_router_usage_is_get():
    from atlas_brain.api.llm_gateway import router

    usage_route = next(
        r for r in router.routes if hasattr(r, "path") and r.path == "/llm/usage"
    )
    assert "GET" in usage_route.methods


def test_router_registered_in_api_aggregator():
    """``api/__init__.py`` must include the gateway router so it
    actually mounts at ``/api/v1/llm/*`` via main.py's prefix."""
    # Reset any cached import so the assertion sees the latest state.
    sys.modules.pop("atlas_brain.api", None)
    api_pkg = importlib.import_module("atlas_brain.api")
    paths = {getattr(route, "path", "") for route in api_pkg.router.routes}
    assert any(p.startswith("/llm/") for p in paths), (
        "api/__init__.py is missing include_router(llm_gateway_router)"
    )


# ---- Plan gating contract ------------------------------------------------


def test_chat_endpoint_requires_llm_plan_dependency():
    """``/chat`` must use ``require_llm_plan('llm_trial')`` so the
    endpoint is gated to llm_gateway-product accounts. Pinning this
    via source-text inspection because the actual dependency
    triggers async DB lookup."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert "require_llm_plan(\"llm_trial\")" in src or "require_llm_plan('llm_trial')" in src


def test_usage_endpoint_requires_llm_plan_dependency():
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.usage)
    assert "require_llm_plan(\"llm_trial\")" in src or "require_llm_plan('llm_trial')" in src


# ---- Provider validation ------------------------------------------------


def test_validate_chat_provider_rejects_non_anthropic():
    from atlas_brain.api.llm_gateway import _validate_chat_provider
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        _validate_chat_provider("openai")
    assert exc_info.value.status_code == 400


def test_validate_chat_provider_accepts_anthropic():
    from atlas_brain.api.llm_gateway import _validate_chat_provider

    # Returns None on success (the chat handler proceeds).
    assert _validate_chat_provider("anthropic") is None


# ---- BYOK 503 path ------------------------------------------------------


def test_resolve_byok_raises_503_when_no_key(monkeypatch):
    monkeypatch.delenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        raising=False,
    )
    from atlas_brain.api.llm_gateway import _resolve_byok_or_503
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        _resolve_byok_or_503("anthropic", "00000000-0000-0000-0000-000000000000")
    assert exc_info.value.status_code == 503
    assert "BYOK key" in exc_info.value.detail


def test_resolve_byok_returns_key_when_set(monkeypatch):
    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-ant-fake",
    )
    from atlas_brain.api.llm_gateway import _resolve_byok_or_503

    assert (
        _resolve_byok_or_503("anthropic", "00000000-0000-0000-0000-000000000000")
        == "sk-ant-fake"
    )


# ---- Schema shape -------------------------------------------------------


def test_chat_request_schema_validates():
    from atlas_brain.api.llm_gateway import ChatRequest

    req = ChatRequest(
        provider="anthropic",
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert req.max_tokens == 1024
    assert req.temperature == 0.7


def test_chat_request_rejects_empty_messages():
    from atlas_brain.api.llm_gateway import ChatRequest

    with pytest.raises(Exception):
        ChatRequest(provider="anthropic", model="claude-haiku-4-5", messages=[])


def test_chat_request_caps_max_tokens():
    from atlas_brain.api.llm_gateway import ChatRequest

    with pytest.raises(Exception):
        ChatRequest(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=999_999_999,
        )


def test_usage_response_schema_shape():
    from atlas_brain.api.llm_gateway import UsageResponse, UsageBreakdownRow

    resp = UsageResponse(
        account_id="00000000-0000-0000-0000-000000000000",
        period_start="2026-04-01T00:00:00",
        period_end="2026-05-01T00:00:00",
        total_input_tokens=1000,
        total_output_tokens=500,
        total_cost_usd=1.23,
        by_provider=[
            UsageBreakdownRow(
                provider="anthropic", model="claude-haiku-4-5",
                input_tokens=1000, output_tokens=500, total_tokens=1500,
                cost_usd=1.23, call_count=5,
            )
        ],
    )
    assert len(resp.by_provider) == 1
    assert resp.by_provider[0].cost_usd == 1.23


# ---- Account threading into trace_llm_call -----------------------------


def test_chat_threads_account_id_into_trace_metadata():
    """The /chat handler must set ``metadata={"account_id": ...}`` so
    the FTL tracer (PR-D3) writes the right account_id to llm_usage.
    Without this, usage rollups attribute the call to the SENTINEL."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert '"account_id": user.account_id' in src


# ---- Usage SQL scopes by account_id -----------------------------------


def test_usage_sql_scopes_by_account_id():
    """The /usage handler MUST filter llm_usage on account_id, else
    we'd return atlas's internal pipeline rows (SENTINEL account)
    to customers."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.usage)
    assert "WHERE account_id = $1" in src


# ---- Codex review fixes (post-review on PR-D4) -------------------------


def test_chat_handler_does_not_await_synchronous_load():
    """Codex P0 fix: ``AnthropicLLM.load()`` is synchronous (returns
    None). Awaiting it raises TypeError at runtime and breaks every
    /chat call. Pin via source-text inspection that the handler
    calls ``llm.load()`` (not ``await llm.load()``)."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert "await llm.load()" not in src
    assert "llm.load()" in src


def test_resolve_byok_helper_renamed_to_503():
    """Codex naming fix: helper was ``_resolve_byok_or_403`` but
    raises 503. Rename keeps the name aligned with behavior."""
    from atlas_brain.api import llm_gateway

    assert hasattr(llm_gateway, "_resolve_byok_or_503")
    assert not hasattr(llm_gateway, "_resolve_byok_or_403")


def test_unused_helper_cache_enabled_for_plan_removed():
    """Codex dead-code fix: ``_cache_enabled_for_plan`` was unused.
    Cache gates land in PR-D4b when /chat reaches the cache path;
    until then the helper is dead code."""
    from atlas_brain.api import llm_gateway

    assert not hasattr(llm_gateway, "_cache_enabled_for_plan")
