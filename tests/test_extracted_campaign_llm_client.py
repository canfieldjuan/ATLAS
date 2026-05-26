from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_llm_client import (
    LLMUnavailableError,
    PipelineLLMClient,
    PipelineLLMClientConfig,
    create_pipeline_llm_client,
)
from extracted_content_pipeline.campaign_ports import LLMMessage
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
    }
    assert "input_data" not in trace
    assert "output_data" not in trace


@pytest.mark.asyncio
async def test_pipeline_llm_client_traces_failed_provider_calls_without_io_capture():
    trace_calls = []
    client = PipelineLLMClient(
        resolver=lambda **_: _FailingChatLLM(),
        tracer=lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
    )

    with pytest.raises(RuntimeError, match="provider timeout"):
        await client.complete(
            [LLMMessage(role="user", content="customer ticket text")],
            max_tokens=200,
            temperature=0.2,
            metadata={"asset_type": "landing_page"},
        )

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
    assert trace["metadata"]["asset_type"] == "landing_page"
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
    })

    assert config == PipelineLLMClientConfig(
        workload="campaign",
        prefer_cloud=False,
        try_openrouter=False,
        auto_activate_ollama=True,
        openrouter_model="anthropic/claude-haiku-4-5",
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

    assert PipelineLLMClientConfig.from_settings(_Settings()) == PipelineLLMClientConfig(
        workload="campaign",
        prefer_cloud=False,
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model="openai/gpt-4o-mini",
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

    config = PipelineLLMClientConfig.from_settings(build_settings().campaign_llm)

    assert config == PipelineLLMClientConfig(
        workload="campaign",
        prefer_cloud=False,
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model="anthropic/claude-haiku-4-5",
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
