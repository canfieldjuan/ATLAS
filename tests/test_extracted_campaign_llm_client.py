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
