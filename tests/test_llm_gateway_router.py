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
    """When neither DB nor env-var fallback resolves a key, the
    helper raises 503. PR-D5 review fix: helper is now async + takes
    pool so DB-stored keys (added via /byok-keys) are honored."""
    import asyncio

    monkeypatch.delenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        raising=False,
    )
    from atlas_brain.api.llm_gateway import _resolve_byok_or_503
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        # pool=None -> resolver skips DB lookup, env is unset -> 503.
        asyncio.run(
            _resolve_byok_or_503(None, "anthropic", "00000000-0000-0000-0000-000000000000")
        )
    assert exc_info.value.status_code == 503
    assert "BYOK key" in exc_info.value.detail


def test_resolve_byok_returns_key_when_set(monkeypatch):
    """The async helper falls back to env var when no DB row exists,
    keeping local-dev workflows functional."""
    import asyncio

    monkeypatch.setenv(
        "ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000",
        "sk-ant-fake",
    )
    from atlas_brain.api.llm_gateway import _resolve_byok_or_503

    result = asyncio.run(
        _resolve_byok_or_503(None, "anthropic", "00000000-0000-0000-0000-000000000000")
    )
    assert result == "sk-ant-fake"


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
    """Codex P0 on PR-D4 pinned that ``llm.load()`` (sync) must not
    be awaited. PR-D4b refactored the chat handler to use
    ``AsyncAnthropic`` directly (in ``async with``) and dropped the
    ``llm.load()`` call entirely -- closes the P0 more thoroughly
    since there's no client to load. Both variants satisfy the
    original concern."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    # The original P0 was "awaiting a sync method"; that pattern
    # must never reappear regardless of how the handler constructs
    # its provider client.
    assert "await llm.load()" not in src


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


# ---- Codex second-pass review (P1 fixes) -------------------------------


def test_require_auth_or_api_key_dispatches_by_token_shape():
    """Codex P1: ``require_llm_plan`` previously chained through
    ``require_auth`` (JWT-only), so ``Authorization: Bearer atls_live_*``
    rejected. The new helper inspects token shape and dispatches to
    either auth method, returning the same AuthUser."""
    from atlas_brain.auth.dependencies import require_auth_or_api_key

    assert callable(require_auth_or_api_key)
    src = inspect.getsource(require_auth_or_api_key)
    # Token shape check: atls_live_ prefix routes to API-key path.
    assert 'startswith("atls_live_")' in src
    assert "require_api_key(request)" in src
    assert "require_auth(request)" in src


def test_require_llm_plan_accepts_api_keys():
    """The plan-tier dependency must use the dual-auth helper so
    customer scripts (API keys) can hit /api/v1/llm/* routes."""
    from atlas_brain.auth import dependencies as deps_mod

    src = inspect.getsource(deps_mod.require_llm_plan)
    assert "Depends(require_auth_or_api_key)" in src
    # The old JWT-only dep must NOT be the user resolver anymore.
    assert "user: AuthUser = Depends(require_auth)" not in src


def test_chat_handler_captures_real_token_usage():
    """Codex P1: chat used to hard-code input_tokens=output_tokens=0,
    so _store_local short-circuited (all token fields falsy) and no
    llm_usage row was written. Pin via source-text inspection that
    the handler reads response.usage."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert "response.usage" in src or "getattr(response, \"usage\"" in src
    # Token counts must reach trace_llm_call (not hardcoded zero).
    assert "input_tokens=input_tokens" in src
    assert "output_tokens=output_tokens" in src


def test_chat_handler_returns_usage_in_response():
    """Customers see token counts in the response so they can
    pre-validate against their own quotas without hitting /usage."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert "input_tokens=input_tokens" in src
    assert "total_tokens=input_tokens + output_tokens" in src


def test_chat_handler_threads_provider_request_id_to_trace():
    """Anthropic returns ``response.id`` -- propagating it to
    trace_llm_call lets ops correlate llm_usage rows with provider
    billing dashboards (PR-A4c openai_billing logic)."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert "provider_request_id=" in src


# ---- Codex PR-D5 review (P1: gateway must use DB resolver) ----------


def test_chat_handler_uses_async_db_resolver():
    """Codex P1 fix on PR-D5: the gateway used the SYNC env-only
    resolver, so keys added via /api/v1/byok-keys were silently
    ignored. Switch to the async resolver that hits the DB first."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    # Pool must be acquired and passed to the resolver.
    assert "_resolve_byok_or_503(pool" in src
    # The await is critical -- coroutine must be awaited.
    assert "await _resolve_byok_or_503" in src


def test_resolve_byok_helper_is_async():
    from atlas_brain.api import llm_gateway
    import inspect as _inspect

    assert _inspect.iscoroutinefunction(llm_gateway._resolve_byok_or_503)


def test_resolve_byok_helper_calls_db_aware_resolver():
    """The helper must call the DB-aware async resolver, not the
    legacy sync env-only one."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway._resolve_byok_or_503)
    assert "lookup_provider_key_async" in src
    assert "lookup_provider_key(" not in src.replace("lookup_provider_key_async", "")


# ---- Exact cache wiring (PR-D6b) ----------------------------------------


def test_llm_gateway_config_has_exact_cache_flag():
    """PR-D6b: the customer-facing /chat surface gets its own
    feature flag, decoupled from b2b_churn.llm_exact_cache_enabled
    so the two products toggle independently."""
    from atlas_brain.config import settings

    assert hasattr(settings, "llm_gateway")
    assert hasattr(settings.llm_gateway, "exact_cache_enabled")
    # Default OFF -- caching is opt-in.
    assert settings.llm_gateway.exact_cache_enabled is False


def test_chat_namespace_constant_routes_to_gateway_flag():
    """The cache namespace must start with the prefix the cache
    module's _is_cache_enabled_for_namespace() dispatcher checks.
    Otherwise enablement falls through to the B2B flag and the
    products couple."""
    from atlas_brain.api import llm_gateway
    from atlas_brain.services.b2b.llm_exact_cache import (
        LLM_GATEWAY_NAMESPACE_PREFIX,
    )

    assert llm_gateway._LLM_CHAT_CACHE_NAMESPACE.startswith(
        LLM_GATEWAY_NAMESPACE_PREFIX
    )


def test_chat_route_imports_cache_helpers():
    """Source-text pin: the chat route must import the cache
    primitives. Without the imports the wiring below silently
    falls back to the un-cached path."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway)
    assert "from ..services.b2b.llm_exact_cache import" in src
    assert "build_request_envelope" in src
    assert "is_llm_gateway_exact_cache_enabled" in src
    assert "lookup_cached_text" in src
    assert "store_cached_text" in src


def test_chat_lookup_runs_before_anthropic_call():
    """Cache lookup must happen BEFORE the Anthropic call --
    otherwise we pay for a request whose answer is already cached.
    Pin the source ordering."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    lookup_idx = src.find("await lookup_cached_text(")
    create_idx = src.find("await client.messages.create(")
    assert lookup_idx > 0 and create_idx > 0
    assert lookup_idx < create_idx


def test_chat_lookup_is_account_scoped():
    """Cross-account leak guard: the lookup must thread the
    customer's account_id (not the sentinel) so PR-D3's
    composite (cache_key, account_id) PK isolates tenants."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    # Find the lookup block specifically (not the store block).
    lookup_block = src.split("await lookup_cached_text(")[1].split(")", 1)[0]
    assert "account_id=user.account_id" in lookup_block


def test_chat_lookup_gated_by_feature_flag():
    """Lookup only runs when settings.llm_gateway.exact_cache_enabled
    is True. With the flag off (the default), the cache module
    isn't even called -- no DB round-trip."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    flag_idx = src.find("if is_llm_gateway_exact_cache_enabled():")
    lookup_idx = src.find("await lookup_cached_text(")
    assert flag_idx > 0 and lookup_idx > 0
    # Flag check is the immediate guard for the lookup.
    assert flag_idx < lookup_idx


def test_chat_cache_lookup_failures_do_not_fail_request():
    """If the cache lookup raises (DB transient, schema drift),
    the chat request must NOT 500 -- fall through to a normal
    Anthropic call. Pin the try/except wrap and the empty
    fall-through."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    # try/except around the lookup call.
    assert "try:\n            cache_hit = await lookup_cached_text(" in src
    assert "llm_gateway.chat cache lookup failed" in src


def test_chat_cache_hit_returns_zero_token_usage():
    """On cache hit: ChatResponse.usage = 0/0/0 (no tokens consumed
    this call). Customer can infer cache via the zero usage; a
    follow-up PR adds a `cached: bool` field for explicit signaling."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    # The cache-hit ChatResponse path uses ChatUsage zeros.
    assert "if cache_hit is not None:" in src
    hit_block = src.split("if cache_hit is not None:")[1].split("# Capture full response")[0]
    assert "input_tokens=0" in hit_block
    assert "output_tokens=0" in hit_block
    assert "total_tokens=0" in hit_block
    # And the response_text comes from the cache hit.
    assert 'response=cache_hit["response_text"]' in hit_block


def test_chat_cache_hit_writes_zero_token_usage_row_with_cache_hit_metadata():
    """Cache savings analytics depend on a zero-token llm_usage
    row tagged ``cache_hit: true`` -- without it, dashboard can't
    distinguish "didn't call provider" from "didn't track".

    Codex P1 on PR-D6b: trace_llm_call -> _store_local
    short-circuits when every token field is falsy, so routing
    through it would drop the row silently. Direct INSERT to
    pool.execute(_CACHE_HIT_USAGE_INSERT_SQL, ...) instead --
    same pattern _persist_batch_usage uses for batch items."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    hit_block = src.split("if cache_hit is not None:")[1].split("# Capture full response")[0]
    # Direct INSERT, NOT trace_llm_call (which would silently drop).
    assert "await pool.execute(\n                _CACHE_HIT_USAGE_INSERT_SQL," in hit_block
    assert "trace_llm_call(" not in hit_block
    # cache_hit metadata flag is in the JSON payload.
    assert '"cache_hit": True' in hit_block

    # The constant SQL is shaped correctly: zero token counts
    # baked in (legitimate -- they didn't consume tokens) and
    # the cache_hit row is identifiable by api_endpoint
    # 'llm_gateway.chat' + metadata.cache_hit=True.
    sql = llm_gateway._CACHE_HIT_USAGE_INSERT_SQL
    assert "INSERT INTO llm_usage" in sql
    assert "input_tokens, output_tokens, total_tokens" in sql
    assert "0, 0, 0" in sql  # token zeros baked in


def test_cache_hit_insert_sql_does_not_route_through_tracer_drop_filter():
    """Pin the design rationale: the tracer's _store_local returns
    early when every token field is falsy. A cache-hit row has
    legitimately-zero tokens and would be dropped if we routed
    through trace_llm_call. Direct INSERT bypasses the filter."""
    from atlas_brain.api import llm_gateway
    from atlas_brain.services import tracing

    # Confirm the tracer DOES drop zero-token payloads (the bug
    # this test guards against).
    store_local_src = inspect.getsource(tracing.FTLTracingClient._store_local)
    assert "if not any(" in store_local_src
    assert '"input_tokens"' in store_local_src
    assert '"output_tokens"' in store_local_src

    # The cache-hit constant is the bypass.
    assert hasattr(llm_gateway, "_CACHE_HIT_USAGE_INSERT_SQL")
    # And the chat handler uses it in the hit branch.
    src = inspect.getsource(llm_gateway.chat)
    hit_block = src.split("if cache_hit is not None:")[1].split("# Capture full response")[0]
    assert "_CACHE_HIT_USAGE_INSERT_SQL" in hit_block


def test_chat_cache_hit_skips_anthropic_call():
    """The cache-hit branch must return BEFORE the Anthropic
    create call -- otherwise the customer pays for a request
    whose answer was just served from cache."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    hit_block = src.split("if cache_hit is not None:")[1].split("# Capture full response")[0]
    # The hit branch must return.
    assert "return ChatResponse(" in hit_block


def test_chat_store_runs_after_successful_anthropic_call():
    """Cache store must happen AFTER the Anthropic call so we
    only persist responses that actually came back -- and after
    the trace_llm_call usage row is written so the success
    accounting precedes the cache write."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    create_idx = src.find("await client.messages.create(")
    store_idx = src.find("await store_cached_text(")
    assert create_idx > 0 and store_idx > 0
    assert create_idx < store_idx


def test_chat_store_is_account_scoped():
    """Same isolation guard as lookup -- store must thread the
    customer's account_id so the row lands in the right tenant
    namespace."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    store_block = src.split("await store_cached_text(")[1].split(")", 1)[0]
    assert "account_id=user.account_id" in store_block


def test_chat_cache_store_failures_do_not_fail_request():
    """Same posture as lookup: cache store errors get logged but
    do not raise -- the chat response was already produced."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert "try:\n            await store_cached_text(" in src
    assert "llm_gateway.chat cache store failed" in src


def test_chat_envelope_includes_system_prompt_in_extra():
    """The system prompt is part of the request's identity and
    must be in the cache key envelope -- otherwise two calls with
    the same messages but different system prompts collide."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    assert 'extra={"system": system_prompt}' in src


# ---- Cache module: namespace dispatch (PR-D6b) -------------------------


def test_cache_module_dispatches_enablement_by_namespace():
    """The B2B and LLM Gateway products own separate feature flags;
    the cache module dispatches based on namespace prefix so
    callers don't accidentally couple."""
    from atlas_brain.services.b2b import llm_exact_cache

    assert hasattr(llm_exact_cache, "is_llm_gateway_exact_cache_enabled")
    assert hasattr(llm_exact_cache, "_is_cache_enabled_for_namespace")
    assert llm_exact_cache.LLM_GATEWAY_NAMESPACE_PREFIX == "llm_gateway."

    # Dispatcher branches.
    src = inspect.getsource(llm_exact_cache._is_cache_enabled_for_namespace)
    assert "LLM_GATEWAY_NAMESPACE_PREFIX" in src
    assert "is_llm_gateway_exact_cache_enabled" in src
    assert "is_b2b_llm_exact_cache_enabled" in src


def test_lookup_cached_text_uses_namespace_dispatch_not_b2b_flag():
    """The lookup must check the namespace-aware dispatcher, not
    the bare B2B flag -- otherwise gateway-namespaced calls fail
    when the B2B flag is off."""
    from atlas_brain.services.b2b import llm_exact_cache

    src = inspect.getsource(llm_exact_cache.lookup_cached_text)
    assert "_is_cache_enabled_for_namespace(namespace)" in src
    # The bare B2B flag check shouldn't appear in this function.
    assert "is_b2b_llm_exact_cache_enabled()" not in src


def test_store_cached_text_uses_namespace_dispatch_not_b2b_flag():
    from atlas_brain.services.b2b import llm_exact_cache

    src = inspect.getsource(llm_exact_cache.store_cached_text)
    assert "_is_cache_enabled_for_namespace(namespace)" in src
    assert "is_b2b_llm_exact_cache_enabled()" not in src


# ---- Cache savings rollup (PR-D6c) -------------------------------------


def test_cache_savings_metadata_key_is_module_constant():
    """Write site (cache-hit handler) and read site (/usage SQL)
    must agree on the metadata key. Defining once as a module
    constant prevents drift."""
    from atlas_brain.api import llm_gateway

    assert hasattr(llm_gateway, "_CACHE_SAVINGS_METADATA_KEY")
    assert llm_gateway._CACHE_SAVINGS_METADATA_KEY == "cache_savings_usd"
    # Read site: the SQL clause that pulls it out for the rollup.
    src = inspect.getsource(llm_gateway.usage)
    assert "metadata->>'cache_savings_usd'" in src


def test_cache_hit_stamps_savings_metadata():
    """The cache-hit branch must stamp the would-have-paid USD on
    the llm_usage row's metadata so /usage can sum it later. The
    value comes from _estimate_cost_usd applied to the cached
    entry's stored token counts."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    hit_block = src.split("if cache_hit is not None:")[1].split("# Capture full response")[0]
    # _estimate_cost_usd is called on the cached usage.
    assert "_estimate_cost_usd(" in hit_block
    assert 'cache_hit.get("usage")' in hit_block
    # Result is stamped into the metadata payload using the constant.
    assert "_CACHE_SAVINGS_METADATA_KEY: cache_savings_usd" in hit_block


def test_cache_hit_savings_estimate_failure_falls_back_to_zero():
    """A model missing from the pricing config (or any other
    estimate failure) must NOT break the chat response. Catch
    the exception, log, default the field to 0.0 -- the row
    still gets written, the savings just don't show up for that
    call."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.chat)
    hit_block = src.split("if cache_hit is not None:")[1].split("# Capture full response")[0]
    # try/except wraps the estimate call.
    assert "try:\n                cache_savings_usd = _estimate_cost_usd(" in hit_block
    assert "cache_savings_usd = 0.0" in hit_block
    # Distinct log line so ops can spot pricing-config holes.
    assert "cache-hit savings estimate failed" in hit_block


def test_estimate_cost_usd_imported_from_batch_module():
    """Reuse the existing helper rather than duplicating the
    pricing math. The function lives in llm_gateway_batch.py
    (PR-D4d) -- import it from there. Underscore-private name is
    deliberate (no public alias yet); revisit if a third caller
    appears."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway)
    assert "_estimate_cost_usd," in src


def test_usage_response_exposes_total_cache_savings_usd():
    """New top-level field on UsageResponse so customers see
    aggregate cache savings for the period in a single number."""
    from atlas_brain.api.llm_gateway import UsageResponse

    fields = UsageResponse.model_fields
    assert "total_cache_savings_usd" in fields
    # Default 0.0 -- back-compat with clients that don't know
    # about the field, and correct value for periods with no hits.
    assert fields["total_cache_savings_usd"].default == 0.0


def test_usage_breakdown_row_exposes_per_group_savings():
    """Customers want savings broken out by (provider, model) too
    so they can see "cache hit-rate on Opus" vs "Haiku"."""
    from atlas_brain.api.llm_gateway import UsageBreakdownRow

    fields = UsageBreakdownRow.model_fields
    assert "cache_savings_usd" in fields
    assert fields["cache_savings_usd"].default == 0.0


def test_usage_sql_aggregates_cache_savings_per_group():
    """Source-text pin on the rollup SQL: SUM the metadata field
    per (provider, model) group. The jsonb_typeof guard (Codex
    P2 fix on PR-D6c) ensures only actual JSON numbers get cast,
    so a malformed string value like {"cache_savings_usd": "n/a"}
    contributes 0 instead of raising ``invalid input syntax for
    type double precision`` and breaking /usage for the period."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.usage)
    # Cast only when the JSONB value is actually a number.
    assert "jsonb_typeof(metadata->'cache_savings_usd') = 'number'" in src
    assert "(metadata->>'cache_savings_usd')::float" in src
    # The racy NULLIF-only pattern from the initial commit must
    # not return -- it doesn't catch malformed strings.
    assert "NULLIF(metadata->>'cache_savings_usd'" not in src
    assert "AS cache_savings_usd" in src


def test_usage_handler_threads_savings_into_response():
    """The query result feeds both the per-row breakdown and
    the top-level total. Accumulator pattern matches the existing
    cost_usd handling."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.usage)
    # Per-row breakdown carries it.
    assert "cache_savings_usd=row_cache_savings" in src
    # Total accumulator.
    assert "total_cache_savings = 0.0" in src
    assert "total_cache_savings += row_cache_savings" in src
    # And the response object surfaces the total.
    assert "total_cache_savings_usd=total_cache_savings" in src


# ---- Reconciliation endpoint (PR-D6d, layer 3) -------------------------


def test_reconciliation_endpoint_registered():
    """The new layer-3 endpoint must be a POST registered under
    /llm/reconciliation (prefix from the router)."""
    from atlas_brain.api.llm_gateway import router

    matches = [
        r for r in router.routes
        if hasattr(r, "path") and r.path == "/llm/reconciliation"
    ]
    assert matches, "/llm/reconciliation route not registered"
    assert "POST" in matches[0].methods


def test_reconciliation_endpoint_plan_gated_to_starter():
    """Reconciliation is a paying-customer feature -- llm_starter
    or above. Trial users see /usage but not this."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert 'require_llm_plan("llm_starter")' in src


def test_reconciliation_request_schema_shape():
    """Schema captures everything the customer needs to reconcile
    against atlas-routed traffic: period bounds, invoice totals,
    per-model breakdown, currency. Provider defaults to anthropic
    (the only one currently supported) and currency defaults to
    USD; the customer rarely needs to set either explicitly."""
    from atlas_brain.api.llm_gateway import (
        ReconciliationByModelRequest,
        ReconciliationRequest,
    )

    req_fields = ReconciliationRequest.model_fields
    assert "provider" in req_fields
    assert req_fields["provider"].default == "anthropic"
    assert "period_start" in req_fields
    assert "period_end" in req_fields
    assert "invoice_total_usd" in req_fields
    assert "currency" in req_fields
    assert req_fields["currency"].default == "USD"
    assert "by_model" in req_fields

    item_fields = ReconciliationByModelRequest.model_fields
    assert "model" in item_fields
    assert "invoice_cost_usd" in item_fields


def test_reconciliation_response_schema_shape():
    """Response surfaces both sides plus the delta and a
    human-readable explanation. The customer compares numbers;
    the explanation tells them whether the diff is actionable."""
    from atlas_brain.api.llm_gateway import (
        ReconciliationByModelRow,
        ReconciliationResponse,
    )

    resp_fields = ReconciliationResponse.model_fields
    for field in (
        "period_start",
        "period_end",
        "provider",
        "currency",
        "atlas_total_usd",
        "invoice_total_usd",
        "delta_usd",
        "delta_explanation",
        "by_model",
    ):
        assert field in resp_fields

    row_fields = ReconciliationByModelRow.model_fields
    for field in ("model", "atlas_usd", "invoice_usd", "delta_usd"):
        assert field in row_fields


def test_reconciliation_sql_filters_by_account_period_provider():
    """SQL must scope by account_id (PR-D3 isolation), provider
    (only anthropic for now), and the date range. The full-day
    semantics on period_end use ``< period_end::date + INTERVAL
    '1 day'`` so calls made on the period_end day are included."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert "WHERE account_id = $1" in src
    assert "AND model_provider = $2" in src
    assert "AND created_at >= $3::date" in src
    assert "AND created_at <  ($4::date + INTERVAL '1 day')" in src


def test_reconciliation_excludes_cache_hits_via_jsonb_containment():
    """Cache-hit rows have cost_usd=0 so they don't affect the
    sum, but excluding them is honest -- they don't appear on
    the Anthropic invoice either. Use JSONB containment (``@>``)
    rather than ``->>`` cast so malformed metadata can't trigger
    the same DoS PR-D6c P2 fixed for cache_savings_usd."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert (
        "AND NOT (metadata @> '{\"cache_hit\": true}'::jsonb)"
        in src
    )


def test_reconciliation_unions_atlas_and_invoice_models():
    """The breakdown must include EVERY model from either side
    so the customer can see (a) atlas-routed traffic for a
    model not on the invoice (= our drift), and (b) invoice
    line items for a model with no atlas-routed traffic
    (= calls made outside atlas)."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert "set(atlas_by_model.keys()) | set(invoice_by_model.keys())" in src


def test_reconciliation_delta_explanation_has_three_signed_branches():
    """Helper composes user-facing text for: (1) match within
    tolerance, (2) Anthropic > atlas (positive delta, expected
    case), (3) atlas > Anthropic (negative delta, possible
    drift to investigate)."""
    from atlas_brain.api.llm_gateway import (
        _RECONCILIATION_MATCH_TOLERANCE_USD,
        _reconciliation_delta_explanation,
    )

    # Positive branch.
    pos = _reconciliation_delta_explanation(5.00, "USD")
    assert "exceeds atlas-routed" in pos.lower()
    # Negative branch.
    neg = _reconciliation_delta_explanation(-5.00, "USD")
    assert "more than the invoice" in neg.lower() or "drift" in neg.lower()
    # Match-within-tolerance branch.
    match = _reconciliation_delta_explanation(
        _RECONCILIATION_MATCH_TOLERANCE_USD / 2, "USD"
    )
    assert "matches" in match.lower()
    # Non-USD branch returns a currency caveat instead of a delta sign.
    eur = _reconciliation_delta_explanation(5.00, "EUR")
    assert "EUR" in eur and "convert" in eur.lower()


def test_reconciliation_validates_date_format():
    """Malformed dates 400, not 500. Pin the helper that
    centralizes the parse + raise so error messages stay
    consistent and one helper is the single source of truth."""
    from fastapi import HTTPException
    from atlas_brain.api.llm_gateway import _parse_reconciliation_date

    # Valid passes through.
    parsed = _parse_reconciliation_date("period_start", "2026-04-01")
    assert parsed.year == 2026

    # Malformed raises HTTPException 400.
    try:
        _parse_reconciliation_date("period_start", "not-a-date")
        raise AssertionError("expected HTTPException")
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "period_start" in str(exc.detail)


def test_reconciliation_validates_period_ordering():
    """period_start > period_end must 400 -- otherwise the SQL
    range produces empty results silently and the customer can't
    tell why their reconciliation looks blank."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert "period_start must be on or before period_end" in src


def test_reconciliation_validates_period_span():
    """Periods longer than the configured max (365d) must 400.
    Anthropic invoices customers monthly; reconciling a year of
    traffic in one POST is a wrong-tool signal, and the SQL
    grouping loses meaning at that scale."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert "_RECONCILIATION_MAX_PERIOD_DAYS" in src
    # Default value reasonable for a SaaS billing-cycle product.
    assert llm_gateway._RECONCILIATION_MAX_PERIOD_DAYS == 365


def test_reconciliation_match_tolerance_is_a_module_constant():
    """The match-within threshold lives as a module constant so
    it can be tuned without a code-path change. Defaults to
    $0.01 (penny rounding)."""
    from atlas_brain.api import llm_gateway

    assert hasattr(llm_gateway, "_RECONCILIATION_MATCH_TOLERANCE_USD")
    assert llm_gateway._RECONCILIATION_MATCH_TOLERANCE_USD == 0.01


# ---- Reconciliation audit fixes (PR-D6d) -------------------------------


def test_reconciliation_validates_provider_against_allowlist():
    """Audit fix on PR-D6d: same provider allowlist as /chat
    (_PROVIDERS_THIS_PR). Prevents a customer from POSTing
    provider='openai', getting empty atlas data, and concluding
    atlas didn't track any of their calls."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert "if body.provider not in _PROVIDERS_THIS_PR:" in src
    assert "is not supported for" in src
    assert "reconciliation in this release" in src


def test_reconciliation_dedupes_by_model_via_sum_not_overwrite():
    """Audit fix on PR-D6d: customers can legitimately have
    multi-row line items per model on their invoice (different
    price tiers, mid-cycle pricing changes). The dict-comp
    overwrite pattern would silently drop all but the last and
    understate the invoice total. Sum instead so duplicates
    aggregate correctly. After the Codex P2 fix the dict key is
    the normalized model id (so two aliases collapsing to the
    same canonical model also aggregate cleanly)."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    # The summing pattern keyed on the normalized model id.
    assert "invoice_by_model[normalized] = (" in src
    assert "invoice_by_model.get(normalized, 0.0)" in src
    assert "+ float(item.invoice_cost_usd)" in src
    # The naive dict-comp must not return.
    assert "invoice_by_model: dict[str, float] = {\n        item.model:" not in src


def test_reconciliation_request_caps_by_model_length():
    """Audit fix on PR-D6d: unbounded by_model list lets a
    malformed payload chew memory in the dict comprehension.
    128 is well above any provider's model count so legitimate
    customers never hit the cap."""
    from atlas_brain.api.llm_gateway import ReconciliationRequest

    fields = ReconciliationRequest.model_fields
    by_model_field = fields["by_model"]
    # max_length sits in the field metadata.
    constraints = getattr(by_model_field, "metadata", []) or []
    max_lens = [
        getattr(c, "max_length", None) for c in constraints
    ]
    assert 128 in max_lens, f"max_length=128 missing from by_model constraints: {max_lens}"


def test_reconciliation_rounds_dollar_values_to_4_decimals():
    """Audit fix on PR-D6d: floating-point arithmetic produces
    representation artifacts (5.0 - 4.99 ~ 1e-10) that are ugly
    in the response. Round to 4 decimals -- enough to keep
    sub-cent drift visible (Anthropic invoices show 2 decimals)
    while hiding the artifacts."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    # Per-row rounding.
    assert "atlas_usd=round(atlas_usd, 4)" in src
    assert "invoice_usd=round(invoice_usd, 4)" in src
    assert "delta_usd=round(invoice_usd - atlas_usd, 4)" in src
    # Top-level rounding.
    assert "delta = round(body.invoice_total_usd - atlas_total, 4)" in src
    assert "atlas_total_usd=round(atlas_total, 4)" in src
    assert "invoice_total_usd=round(body.invoice_total_usd, 4)" in src


def test_reconciliation_sql_anchors_period_to_utc():
    """Audit fix on PR-D6d: ``$::date`` casts resolve in the
    session timezone. A non-UTC Postgres deployment would
    silently misframe periods (calls late in a day might land
    in the next day's bucket). Explicit ``AT TIME ZONE 'UTC'``
    matches Anthropic's Cost Report API (UTC-day buckets) and
    makes the intent legible."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    assert "AND created_at >= $3::date AT TIME ZONE 'UTC'" in src
    assert "AND created_at <  ($4::date + INTERVAL '1 day') AT TIME ZONE 'UTC'" in src


def test_reconciliation_normalizes_model_aliases_on_both_sides():
    """Codex P2 on PR-D6d: /chat persists body.model (customer
    input) to llm_usage.model_name while Anthropic invoices use
    the canonical model id AnthropicLLM normalized to. Without
    normalizing on read, an alias would split into atlas-only +
    invoice-only rows in the breakdown -- falsely suggesting
    non-atlas traffic or pricing drift.

    Both sides go through ``_normalize_anthropic_model`` so the
    breakdown groups by canonical name. Sum-aggregation handles
    the case where multiple aliases collapse to the same
    canonical model."""
    from atlas_brain.api import llm_gateway

    src = inspect.getsource(llm_gateway.reconciliation)
    # Normalization helper imported.
    assert "from ..services.llm.anthropic import _normalize_anthropic_model" in inspect.getsource(llm_gateway)
    # Atlas-side normalization (with sum-aggregation in case
    # multiple aliases collapse to the same canonical key).
    assert "model = _normalize_anthropic_model(row[\"model_name\"] or \"\")" in src
    assert 'atlas_by_model[model] = (' in src
    assert "atlas_by_model.get(model, 0.0)" in src
    # Invoice-side normalization.
    assert "normalized = _normalize_anthropic_model(item.model)" in src
    assert "invoice_by_model[normalized] = (" in src
    # Helper itself collapses aliases (smoke test the round-trip).
    from atlas_brain.services.llm.anthropic import _normalize_anthropic_model
    # Empty input gets the default model -- harmless for our path
    # (empty model strings shouldn't make it past Pydantic min_length
    # validation on invoice items, and llm_usage rows with NULL
    # model_name are an edge case worth bucketing somewhere).
    assert _normalize_anthropic_model("") != ""
