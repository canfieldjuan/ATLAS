from __future__ import annotations

import asyncio
import threading

import pytest

from extracted_content_pipeline.campaign_llm_client import (
    LLMUnavailableError,
    PipelineLLMClient,
    PipelineLLMClientConfig,
    create_pipeline_llm_client,
    reset_content_ops_llm_trace_context,
    set_content_ops_llm_trace_context,
)
from extracted_content_pipeline.campaign_ports import LLMMessage
from extracted_content_pipeline.content_ops_cache_policy import (
    ContentOpsExactCachePolicy,
)
from extracted_content_pipeline.settings import build_settings


class _ChatLLM:
    model = "chat-model"

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, max_tokens, temperature):
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return {
            "response": "chat response",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }


class _AsyncChatLLM:
    name = "async-chat-model"

    async def chat(self, messages, *, max_tokens, temperature):
        return {
            "message": {"content": "async chat response"},
            "model": "explicit-model",
            "usage": {"input_tokens": 3},
        }


class _GenerateLLM:
    model_id = "generate-model"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, system_prompt=None, max_tokens, temperature):
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return {"message": {"content": "generated response"}}


class _GenerateStringLLM:
    def generate(self, prompt, *, system_prompt=None, max_tokens, temperature):
        return "plain generated response"


class _TraceLLM:
    name = "openrouter"
    model = "anthropic/claude-haiku-4-5"

    def chat(self, messages, *, max_tokens, temperature):
        return {
            "response": "traced response",
            "model": "anthropic/claude-haiku-4-5",
            "usage": {
                "input_tokens": "100",
                "output_tokens": 25,
                "total_tokens": 125,
            },
            "_trace_meta": {
                "cache_read_tokens": 7,
                "cache_creation_tokens": 5,
                "billable_input_tokens": 93,
                "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "provider_request_id": "req_trace_123",
                "ttft_ms": "12.5",
                "inference_time_ms": 55,
                "queue_time_ms": None,
            },
        }


class _FailingChatLLM:
    name = "openrouter"
    model = "anthropic/claude-haiku-4-5"

    def chat(self, messages, *, max_tokens, temperature):
        raise RuntimeError("provider timeout")


class _FakeExactCache:
    def __init__(
        self,
        *,
        hit=None,
        lookup_exc: Exception | None = None,
        store_exc: Exception | None = None,
        store_result: bool = True,
    ):
        self.hit = hit
        self.lookup_exc = lookup_exc
        self.store_exc = store_exc
        self.store_result = store_result
        self.envelopes = []
        self.lookups = []
        self.stores = []

    def build_request_envelope(
        self,
        *,
        provider,
        model,
        messages,
        max_tokens,
        temperature,
    ):
        envelope = {
            "provider": provider,
            "model": model,
            "messages": [
                {"role": item.role, "content": item.content}
                for item in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        self.envelopes.append(envelope)
        return envelope

    async def lookup(self, decision, request_envelope):
        self.lookups.append((decision, request_envelope))
        if self.lookup_exc is not None:
            raise self.lookup_exc
        return self.hit

    async def store(
        self,
        decision,
        request_envelope,
        *,
        provider,
        model,
        response,
    ):
        self.stores.append({
            "decision": decision,
            "request_envelope": request_envelope,
            "provider": provider,
            "model": model,
            "response": response,
        })
        if self.store_exc is not None:
            raise self.store_exc
        return self.store_result


@pytest.mark.asyncio
async def test_pipeline_llm_client_resolves_and_normalizes_chat_response():
    llm = _ChatLLM()
    resolver_calls = []

    def resolver(**kwargs):
        resolver_calls.append(kwargs)
        return llm

    client = PipelineLLMClient(workload="draft", resolver=resolver)

    response = await client.complete(
        [LLMMessage(role="user", content="Write the email")],
        max_tokens=200,
        temperature=0.2,
    )

    assert response.content == "chat response"
    assert response.model == "chat-model"
    assert response.usage == {"input_tokens": 10, "output_tokens": 4}
    assert llm.calls[0]["max_tokens"] == 200
    assert resolver_calls[0]["workload"] == "draft"


@pytest.mark.asyncio
async def test_pipeline_llm_client_offloads_sync_chat_without_blocking_event_loop():
    started = threading.Event()
    release = threading.Event()

    class _BlockingChatLLM:
        model = "blocking-chat-model"

        def chat(self, messages, *, max_tokens, temperature):
            del messages, max_tokens, temperature
            started.set()
            release.wait(timeout=0.25)
            return {"response": "released"}

    client = PipelineLLMClient(resolver=lambda **_: _BlockingChatLLM())
    task = asyncio.create_task(
        client.complete(
            [LLMMessage(role="user", content="Write the email")],
            max_tokens=200,
            temperature=0.2,
        )
    )

    for _ in range(50):
        if started.is_set():
            break
        await asyncio.sleep(0.01)

    assert started.is_set()
    await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
    assert task.done() is False
    release.set()
    response = await asyncio.wait_for(task, timeout=1)

    assert response.content == "released"


@pytest.mark.asyncio
async def test_pipeline_llm_client_chat_messages_include_tool_fields():
    llm = _ChatLLM()
    client = PipelineLLMClient(resolver=lambda **_: llm)

    await client.complete(
        [LLMMessage(role="user", content="Write the email")],
        max_tokens=200,
        temperature=0.2,
    )

    message = llm.calls[0]["messages"][0]
    assert message.role == "user"
    assert message.content == "Write the email"
    assert message.tool_calls is None
    assert message.tool_call_id is None


@pytest.mark.asyncio
async def test_pipeline_llm_client_accepts_async_chat_result():
    client = PipelineLLMClient(resolver=lambda **_: _AsyncChatLLM())

    response = await client.complete(
        [LLMMessage(role="user", content="Write the email")],
        max_tokens=200,
        temperature=0.2,
    )

    assert response.content == "async chat response"
    assert response.model == "explicit-model"
    assert response.usage == {"input_tokens": 3}


@pytest.mark.asyncio
async def test_pipeline_llm_client_falls_back_to_generate_shape():
    llm = _GenerateLLM()
    client = PipelineLLMClient(resolver=lambda **_: llm)

    response = await client.complete(
        [
            LLMMessage(role="system", content="System instructions"),
            LLMMessage(role="user", content="Prompt body"),
        ],
        max_tokens=100,
        temperature=0.4,
    )

    assert response.content == "generated response"
    assert response.model == "generate-model"
    assert llm.calls[0]["system_prompt"] == "System instructions"
    assert llm.calls[0]["prompt"] == "Prompt body"


@pytest.mark.asyncio
async def test_pipeline_llm_client_normalizes_string_generate_response():
    client = PipelineLLMClient(resolver=lambda **_: _GenerateStringLLM())

    response = await client.complete(
        [LLMMessage(role="user", content="Prompt body")],
        max_tokens=100,
        temperature=0.4,
    )

    assert response.content == "plain generated response"
    assert response.model is None
    assert response.raw == "plain generated response"


@pytest.mark.asyncio
async def test_pipeline_llm_client_traces_successful_provider_usage_without_io_capture():
    trace_calls = []
    client = PipelineLLMClient(
        workload="draft",
        resolver=lambda **_: _TraceLLM(),
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
    )

    response = await client.complete(
        [LLMMessage(role="user", content="customer ticket text")],
        max_tokens=200,
        temperature=0.2,
        metadata={"asset_type": "blog_post", "request_id": "req_123"},
    )

    assert response.content == "traced response"
    assert len(trace_calls) == 1
    span_name, trace = trace_calls[0]
    assert span_name == "content_ops.llm.complete"
    assert trace["status"] == "completed"
    assert trace["input_tokens"] == 100
    assert trace["output_tokens"] == 25
    assert trace["cached_tokens"] == 7
    assert trace["cache_write_tokens"] == 5
    assert trace["billable_input_tokens"] == 93
    assert trace["model"] == "anthropic/claude-haiku-4-5"
    assert trace["provider"] == "openrouter"
    assert trace["provider_request_id"] == "req_trace_123"
    assert trace["api_endpoint"] == "https://openrouter.ai/api/v1/chat/completions"
    assert trace["ttft_ms"] == 12.5
    assert trace["inference_time_ms"] == 55
    assert trace["queue_time_ms"] is None
    assert trace["metadata"] == {
        "product": "content_ops",
        "workload": "draft",
        "llm_adapter": "pipeline",
        "asset_type": "blog_post",
        "request_id": "req_123",
        "cache_mode": "no_store",
        "cache_reason": "exact_cache_disabled",
    }
    assert "input_data" not in trace
    assert "output_data" not in trace


@pytest.mark.asyncio
async def test_pipeline_llm_client_merges_scoped_trace_metadata_and_resets():
    trace_calls = []
    client = PipelineLLMClient(
        workload="draft",
        resolver=lambda **_: _TraceLLM(),
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
    )

    token = set_content_ops_llm_trace_context({
        "account_id": "acct-1",
        "user_id": "user-1",
        "asset_type": "scope-default",
    })
    try:
        await client.complete(
            [LLMMessage(role="user", content="customer ticket text")],
            max_tokens=200,
            temperature=0.2,
            metadata={
                "account_id": "spoofed-account",
                "user_id": "spoofed-user",
                "asset_type": "blog_post",
                "request_id": "req_123",
            },
        )
    finally:
        reset_content_ops_llm_trace_context(token)

    await client.complete(
        [LLMMessage(role="user", content="customer ticket text")],
        max_tokens=200,
        temperature=0.2,
        metadata={"asset_type": "landing_page"},
    )

    assert trace_calls[0][1]["metadata"] == {
        "product": "content_ops",
        "workload": "draft",
        "llm_adapter": "pipeline",
        "account_id": "acct-1",
        "user_id": "user-1",
        "asset_type": "blog_post",
        "request_id": "req_123",
        "cache_mode": "no_store",
        "cache_reason": "exact_cache_disabled",
    }
    assert trace_calls[1][1]["metadata"] == {
        "product": "content_ops",
        "workload": "draft",
        "llm_adapter": "pipeline",
        "asset_type": "landing_page",
        "cache_mode": "no_store",
        "cache_reason": "exact_cache_disabled",
    }


@pytest.mark.asyncio
async def test_pipeline_llm_client_traces_exact_cache_policy_decision_from_scope():
    trace_calls = []
    client = PipelineLLMClient(
        workload="draft",
        resolver=lambda **_: _TraceLLM(),
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
        cache_policy=ContentOpsExactCachePolicy(exact_cache_enabled=True),
    )

    token = set_content_ops_llm_trace_context({
        "account_id": "acct-cache",
        "user_id": "user-cache",
    })
    try:
        await client.complete(
            [LLMMessage(role="user", content="non-customer brief")],
            max_tokens=200,
            temperature=0.2,
            metadata={
                "asset_type": "landing_page",
                "cache_policy": "exact",
                "cache_mode": "spoofed",
                "cache_reason": "spoofed",
                "cache_namespace": "spoofed",
                "cache_account_id": "spoofed",
            },
        )
    finally:
        reset_content_ops_llm_trace_context(token)

    trace_metadata = trace_calls[0][1]["metadata"]
    assert trace_metadata["cache_mode"] == "exact"
    assert trace_metadata["cache_reason"] == "eligible"
    assert trace_metadata["cache_namespace"] == "content_ops.landing_page"
    assert trace_metadata["cache_account_id"] == "acct-cache"
    assert trace_metadata["account_id"] == "acct-cache"


@pytest.mark.asyncio
async def test_pipeline_llm_client_returns_exact_cache_hit_without_provider_call():
    llm = _ChatLLM()
    trace_calls = []
    exact_cache = _FakeExactCache(hit={
        "namespace": "content_ops.landing_page",
        "model": "cached-model",
        "response_text": "cached response",
        "usage": {"input_tokens": 8, "output_tokens": 3},
        "metadata": {"cache_version": "v1"},
    })
    client = PipelineLLMClient(
        workload="draft",
        resolver=lambda **_: llm,
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
        cache_policy=ContentOpsExactCachePolicy(exact_cache_enabled=True),
        exact_cache=exact_cache,
    )

    token = set_content_ops_llm_trace_context({"account_id": "acct-cache"})
    try:
        response = await client.complete(
            [LLMMessage(role="user", content="stable non-customer prompt")],
            max_tokens=200,
            temperature=0.2,
            metadata={"asset_type": "landing_page", "cache_policy": "exact"},
        )
    finally:
        reset_content_ops_llm_trace_context(token)

    assert response.content == "cached response"
    assert response.model == "cached-model"
    assert response.usage == {"input_tokens": 0, "output_tokens": 0}
    assert llm.calls == []
    assert len(exact_cache.lookups) == 1
    assert exact_cache.stores == []
    trace_metadata = trace_calls[0][1]["metadata"]
    assert trace_metadata["cache_mode"] == "exact"
    assert trace_metadata["cache_result"] == "hit"
    assert trace_metadata["cached_input_tokens"] == "8"
    assert trace_metadata["cached_output_tokens"] == "3"
    assert trace_metadata["billable_output_tokens"] == "0"
    assert trace_calls[0][1]["input_tokens"] == 0
    assert trace_calls[0][1]["output_tokens"] == 0
    assert trace_calls[0][1]["cached_tokens"] == 8
    assert trace_calls[0][1]["billable_input_tokens"] == 0


@pytest.mark.asyncio
async def test_pipeline_llm_client_stores_exact_cache_miss_after_provider_call():
    llm = _ChatLLM()
    trace_calls = []
    exact_cache = _FakeExactCache()
    client = PipelineLLMClient(
        workload="draft",
        resolver=lambda **_: llm,
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
        cache_policy=ContentOpsExactCachePolicy(exact_cache_enabled=True),
        exact_cache=exact_cache,
    )

    token = set_content_ops_llm_trace_context({"account_id": "acct-cache"})
    try:
        response = await client.complete(
            [LLMMessage(role="user", content="stable non-customer prompt")],
            max_tokens=200,
            temperature=0.2,
            metadata={"asset_type": "landing_page", "cache_policy": "exact"},
        )
    finally:
        reset_content_ops_llm_trace_context(token)

    assert response.content == "chat response"
    assert len(llm.calls) == 1
    assert len(exact_cache.lookups) == 1
    assert len(exact_cache.stores) == 1
    stored = exact_cache.stores[0]
    assert stored["provider"] == "_ChatLLM"
    assert stored["model"] == "chat-model"
    assert stored["response"].content == "chat response"
    trace_metadata = trace_calls[0][1]["metadata"]
    assert trace_metadata["cache_result"] == "miss"
    assert trace_metadata["cache_store_result"] == "stored"


@pytest.mark.asyncio
async def test_pipeline_llm_client_skips_adapter_when_policy_is_no_store():
    llm = _ChatLLM()
    exact_cache = _FakeExactCache()
    client = PipelineLLMClient(
        resolver=lambda **_: llm,
        cache_policy=ContentOpsExactCachePolicy(exact_cache_enabled=False),
        exact_cache=exact_cache,
    )

    response = await client.complete(
        [LLMMessage(role="user", content="prompt")],
        max_tokens=200,
        temperature=0.2,
        metadata={"asset_type": "landing_page", "cache_policy": "exact"},
    )

    assert response.content == "chat response"
    assert len(llm.calls) == 1
    assert exact_cache.envelopes == []
    assert exact_cache.lookups == []
    assert exact_cache.stores == []


@pytest.mark.asyncio
async def test_pipeline_llm_client_continues_when_cache_lookup_fails():
    llm = _ChatLLM()
    trace_calls = []
    exact_cache = _FakeExactCache(lookup_exc=RuntimeError("cache unavailable"))
    client = PipelineLLMClient(
        resolver=lambda **_: llm,
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
        cache_policy=ContentOpsExactCachePolicy(exact_cache_enabled=True),
        exact_cache=exact_cache,
    )

    token = set_content_ops_llm_trace_context({"account_id": "acct-cache"})
    try:
        response = await client.complete(
            [LLMMessage(role="user", content="stable non-customer prompt")],
            max_tokens=200,
            temperature=0.2,
            metadata={"asset_type": "landing_page", "cache_policy": "exact"},
        )
    finally:
        reset_content_ops_llm_trace_context(token)

    assert response.content == "chat response"
    assert len(llm.calls) == 1
    assert len(exact_cache.lookups) == 1
    assert exact_cache.stores == []
    trace_metadata = trace_calls[0][1]["metadata"]
    assert trace_metadata["cache_result"] == "lookup_error"
    assert trace_metadata["cache_error_type"] == "RuntimeError"
    assert trace_metadata["cache_error_message"] == "cache unavailable"


@pytest.mark.asyncio
async def test_pipeline_llm_client_traces_failed_provider_calls_without_io_capture():
    trace_calls = []
    client = PipelineLLMClient(
        resolver=lambda **_: _FailingChatLLM(),
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
    )

    token = set_content_ops_llm_trace_context({"account_id": "acct-failed"})
    try:
        with pytest.raises(RuntimeError, match="provider timeout"):
            await client.complete(
                [LLMMessage(role="user", content="customer ticket text")],
                max_tokens=200,
                temperature=0.2,
                metadata={"asset_type": "landing_page"},
            )
    finally:
        reset_content_ops_llm_trace_context(token)

    assert len(trace_calls) == 1
    span_name, trace = trace_calls[0]
    assert span_name == "content_ops.llm.complete"
    assert trace["status"] == "failed"
    assert trace["input_tokens"] == 0
    assert trace["output_tokens"] == 0
    assert trace["model"] == "anthropic/claude-haiku-4-5"
    assert trace["provider"] == "openrouter"
    assert trace["error_type"] == "RuntimeError"
    assert trace["error_message"] == "provider timeout"
    assert trace["metadata"]["product"] == "content_ops"
    assert trace["metadata"]["account_id"] == "acct-failed"
    assert trace["metadata"]["asset_type"] == "landing_page"
    assert trace["metadata"]["cache_mode"] == "no_store"
    assert trace["metadata"]["cache_reason"] == "exact_cache_disabled"
    assert "input_data" not in trace
    assert "output_data" not in trace


@pytest.mark.asyncio
async def test_pipeline_llm_client_does_not_fail_generation_when_tracing_fails():
    def broken_tracer(span_name, **kwargs):
        raise RuntimeError("trace table unavailable")

    client = PipelineLLMClient(
        resolver=lambda **_: _TraceLLM(),
        tracer=broken_tracer,
    )

    response = await client.complete(
        [LLMMessage(role="user", content="Write the email")],
        max_tokens=200,
        temperature=0.2,
    )

    assert response.content == "traced response"


def test_llm_client_config_from_mapping_parses_provider_routing_fields():
    config = PipelineLLMClientConfig.from_mapping({
        "workload": "campaign",
        "prefer_cloud": "false",
        "try_openrouter": "0",
        "auto_activate_ollama": "yes",
        "openrouter_model": "anthropic/claude-haiku-4-5",
        "exact_cache_enabled": "true",
        "customer_data_exact_cache_enabled": "false",
        "exact_cache_namespace_prefix": "tenant_content_ops",
    })

    assert config == PipelineLLMClientConfig(
        workload="campaign",
        prefer_cloud=False,
        try_openrouter=False,
        auto_activate_ollama=True,
        openrouter_model="anthropic/claude-haiku-4-5",
        exact_cache_enabled=True,
        customer_data_exact_cache_enabled=False,
        exact_cache_namespace_prefix="tenant_content_ops",
    )


def test_llm_client_config_from_env_accepts_custom_prefix_and_blank_model():
    config = PipelineLLMClientConfig.from_env(
        {
            "PIPE_LLM_WORKLOAD": "draft",
            "PIPE_LLM_PREFER_CLOUD": "off",
            "PIPE_LLM_TRY_OPENROUTER": "true",
            "PIPE_LLM_AUTO_ACTIVATE_OLLAMA": "false",
            "PIPE_LLM_OPENROUTER_MODEL": "  ",
        },
        prefix="PIPE_LLM_",
    )

    assert config == PipelineLLMClientConfig(
        workload="draft",
        prefer_cloud=False,
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model=None,
    )


def test_llm_client_config_from_settings_namespace():
    class _Settings:
        workload = "campaign"
        prefer_cloud = False
        try_openrouter = True
        auto_activate_ollama = False
        openrouter_model = "openai/gpt-4o-mini"
        exact_cache_enabled = True
        customer_data_exact_cache_enabled = False
        exact_cache_namespace_prefix = "tenant_content_ops"

    assert PipelineLLMClientConfig.from_settings(_Settings()) == PipelineLLMClientConfig(
        workload="campaign",
        prefer_cloud=False,
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model="openai/gpt-4o-mini",
        exact_cache_enabled=True,
        customer_data_exact_cache_enabled=False,
        exact_cache_namespace_prefix="tenant_content_ops",
    )


def test_build_settings_exposes_campaign_llm_provider_config(monkeypatch):
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_LLM_WORKLOAD", "campaign")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_LLM_PREFER_CLOUD", "false")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_LLM_TRY_OPENROUTER", "true")
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA", "false")
    monkeypatch.setenv(
        "EXTRACTED_CAMPAIGN_LLM_OPENROUTER_MODEL",
        "anthropic/claude-haiku-4-5",
    )
    monkeypatch.setenv("EXTRACTED_CAMPAIGN_LLM_EXACT_CACHE_ENABLED", "true")
    monkeypatch.setenv(
        "EXTRACTED_CAMPAIGN_LLM_CUSTOMER_DATA_EXACT_CACHE_ENABLED",
        "false",
    )
    monkeypatch.setenv(
        "EXTRACTED_CAMPAIGN_LLM_EXACT_CACHE_NAMESPACE_PREFIX",
        "tenant_content_ops",
    )

    config = PipelineLLMClientConfig.from_settings(build_settings().campaign_llm)

    assert config == PipelineLLMClientConfig(
        workload="campaign",
        prefer_cloud=False,
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model="anthropic/claude-haiku-4-5",
        exact_cache_enabled=True,
        customer_data_exact_cache_enabled=False,
        exact_cache_namespace_prefix="tenant_content_ops",
    )


@pytest.mark.asyncio
async def test_create_pipeline_llm_client_applies_config_to_resolver():
    llm = _ChatLLM()
    resolver_calls = []

    def resolver(**kwargs):
        resolver_calls.append(kwargs)
        return llm

    client = create_pipeline_llm_client(
        {
            "workload": "campaign",
            "prefer_cloud": False,
            "try_openrouter": False,
            "auto_activate_ollama": False,
            "openrouter_model": "anthropic/claude-haiku-4-5",
        },
        resolver=resolver,
    )

    await client.complete(
        [LLMMessage(role="user", content="Write")],
        max_tokens=50,
        temperature=0.1,
    )

    assert resolver_calls == [{
        "workload": "campaign",
        "prefer_cloud": False,
        "try_openrouter": False,
        "auto_activate_ollama": False,
        "openrouter_model": "anthropic/claude-haiku-4-5",
    }]


@pytest.mark.asyncio
async def test_pipeline_llm_client_raises_when_no_llm_is_configured():
    client = PipelineLLMClient(resolver=lambda **_: None)

    with pytest.raises(LLMUnavailableError):
        await client.complete(
            [LLMMessage(role="user", content="Draft")],
            max_tokens=100,
            temperature=0.3,
        )


@pytest.mark.asyncio
async def test_pipeline_llm_client_raises_when_resolved_llm_has_no_supported_method():
    client = PipelineLLMClient(resolver=lambda **_: object())

    with pytest.raises(LLMUnavailableError):
        await client.complete(
            [LLMMessage(role="user", content="Draft")],
            max_tokens=100,
            temperature=0.3,
        )
