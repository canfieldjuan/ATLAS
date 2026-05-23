"""Pin the host's Content Ops LLM + Skill adapters.

`atlas_brain/_content_ops_infrastructure.py` bridges the host's
sync `LLMService` and packaged `LocalSkillRegistry` to the
extracted package's async `LLMClient` / sync `SkillStore`
ports. Nine regression tests covering both adapters and both
factories:

Skill side:
1. Real-disk lookup of a host-shipped skill (`digest/blog_post_generation`).
2. Codex P2 fallback canary: skills that live only in the
   extracted package (`digest/landing_page_generation` etc.)
   resolve through the packaged-fallback root.
3. Missing-skill name returns None.

LLM side:
4. Full happy-path message translation + response wrap.
5. Codex P1 canary: each duck-typed message exposes
   `.tool_calls` / `.tool_call_id` so cloud backends that read
   them during payload conversion don't AttributeError.
6. Response-shape `content` field alias.

Factories:
7. `build_content_ops_llm_client()` returns `None` when no
   host LLM is routable.
8. `build_content_ops_llm_client()` wraps a pipeline-routed
   OpenRouter service.
9. `build_content_ops_llm_client()` falls back to an active
   registry service.
10. `build_content_ops_skill_store(registry=stub)` delegates
   through the injected stub.

Tests use the dependency-injection kwargs on both factories so
the test harness doesn't trigger
`atlas_brain.services.__init__`'s torch / ollama eager-load
chain.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from atlas_brain._content_ops_infrastructure import (
    _HostLLMClient,
    _HostSkillStore,
    build_content_ops_llm_client,
    build_content_ops_skill_store,
)
from extracted_content_pipeline.campaign_ports import LLMMessage


def test_host_skill_store_returns_existing_skill_content() -> None:
    """Skill store resolves a host-shipped skill via the host
    overrides root."""

    store = build_content_ops_skill_store()
    prompt = store.get_prompt("digest/blog_post_generation")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_host_skill_store_falls_back_to_packaged_skills() -> None:
    """Codex P2 fix: skills that the extracted services depend on
    by default (e.g. `digest/landing_page_generation`,
    `digest/report_generation`, `digest/sales_brief_generation`,
    `digest/b2b_campaign_reasoning_context`) live in the
    extracted package, not in `atlas_brain/skills/`. The factory
    must fall back to the packaged copies so the next slice's
    services don't immediately fail with "skill not found"."""

    store = build_content_ops_skill_store()
    for name in (
        "digest/landing_page_generation",
        "digest/report_generation",
        "digest/sales_brief_generation",
        "digest/b2b_campaign_reasoning_context",
    ):
        prompt = store.get_prompt(name)
        assert isinstance(prompt, str), name
        assert len(prompt) > 0, name


def test_host_skill_store_returns_none_for_missing_skill() -> None:
    """Skill store doesn't raise on lookup miss."""

    store = build_content_ops_skill_store()
    assert store.get_prompt("digest/__definitely_not_a_real_skill__") is None


@pytest.mark.asyncio
async def test_host_llm_client_translates_messages_and_response() -> None:
    """LLM adapter bridges sync host -> async extracted with the
    right message + response shape."""

    received: dict[str, Any] = {}

    class _FakeHostLLM:
        def __init__(self) -> None:
            self.model_info = SimpleNamespace(name="test-model")

        def chat(
            self,
            messages: list[Any],
            *,
            max_tokens: int,
            temperature: float,
        ) -> dict[str, Any]:
            received["messages"] = messages
            received["max_tokens"] = max_tokens
            received["temperature"] = temperature
            return {
                "response": "Hello, world.",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }

    client = _HostLLMClient(_FakeHostLLM())
    response = await client.complete(
        [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Say hello."),
        ],
        max_tokens=64,
        temperature=0.5,
        metadata={"asset_type": "blog_post"},  # silently dropped
    )

    # Translation: the host saw two `Message`-shaped objects.
    assert len(received["messages"]) == 2
    assert received["messages"][0].role == "system"
    assert received["messages"][0].content == "You are helpful."
    assert received["messages"][1].role == "user"
    assert received["messages"][1].content == "Say hello."
    assert received["max_tokens"] == 64
    assert received["temperature"] == 0.5

    # Response wrap.
    assert response.content == "Hello, world."
    assert response.model == "test-model"
    assert response.usage == {"input_tokens": 5, "output_tokens": 3}
    assert response.raw == {
        "response": "Hello, world.",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


@pytest.mark.asyncio
async def test_host_llm_client_messages_carry_tool_call_attributes() -> None:
    """Codex P1 fix: cloud backends (OpenRouter / Groq / Together /
    Ollama) read `msg.tool_calls` and `msg.tool_call_id` while
    converting messages to provider payloads. Without these
    attributes the adapter would AttributeError before any
    Content Ops generation request reached the backend. Pin the
    duck-typed shape includes them with `None` defaults."""

    captured: dict[str, Any] = {}

    class _ToolReadingFakeLLM:
        model_info = None

        def chat(self, messages, *, max_tokens, temperature):
            del max_tokens, temperature
            # Simulates how OpenRouter / Groq / Ollama backends
            # touch every message during payload conversion.
            for msg in messages:
                _ = msg.role
                _ = msg.content
                _ = msg.tool_calls
                _ = msg.tool_call_id
            captured["read_succeeded"] = True
            return {"response": "ok"}

    client = _HostLLMClient(_ToolReadingFakeLLM())
    response = await client.complete(
        [LLMMessage(role="user", content="hi")],
        max_tokens=8,
        temperature=0.0,
    )
    assert captured["read_succeeded"] is True
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_host_llm_client_handles_content_field_alias() -> None:
    """Some host backends return `{"content": ...}` instead of
    `{"response": ...}`. The adapter accepts either."""

    class _FakeHostLLM:
        model_info = None

        def chat(self, messages, *, max_tokens, temperature):
            del messages, max_tokens, temperature
            return {"content": "Body via content field."}

    client = _HostLLMClient(_FakeHostLLM())
    response = await client.complete(
        [LLMMessage(role="user", content="hi")],
        max_tokens=8,
        temperature=0.0,
    )
    assert response.content == "Body via content field."
    assert response.model is None


def test_build_content_ops_llm_client_returns_none_when_no_llm_routable() -> None:
    """Factory short-circuits when neither pipeline routing nor
    `llm_registry.get_active()` returns a provider -- the bundle
    factory's signal to leave LLM-needing slots `None`."""

    stub_registry = SimpleNamespace(get_active=lambda: None)
    assert (
        build_content_ops_llm_client(
            llm_registry=stub_registry,
            pipeline_llm_resolver=lambda **_kwargs: None,
        )
        is None
    )


def test_build_content_ops_llm_client_wraps_pipeline_routed_service() -> None:
    """Factory resolves through Atlas pipeline routing before
    looking at the active registry slot. This is the production
    path for configured OpenRouter Claude models that are not
    pre-activated globally."""

    calls: list[dict[str, Any]] = []
    fake_service = SimpleNamespace(model_info=None, chat=lambda *a, **k: {})

    def _resolver(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return fake_service

    client = build_content_ops_llm_client(
        llm_registry=SimpleNamespace(get_active=lambda: None),
        pipeline_llm_resolver=_resolver,
    )

    assert isinstance(client, _HostLLMClient)
    # The wrapped service is the one we patched.
    assert client._host is fake_service  # type: ignore[attr-defined]
    assert calls == [{
        "workload": "openrouter",
        "try_openrouter": True,
        "auto_activate_ollama": False,
    }]


def test_build_content_ops_llm_client_falls_back_to_active_registry_service() -> None:
    """If pipeline routing is unavailable, preserve the previous
    active-registry behavior."""

    fake_service = SimpleNamespace(model_info=None, chat=lambda *a, **k: {})
    stub_registry = SimpleNamespace(get_active=lambda: fake_service)
    client = build_content_ops_llm_client(
        llm_registry=stub_registry,
        pipeline_llm_resolver=lambda **_kwargs: None,
    )
    assert isinstance(client, _HostLLMClient)
    assert client._host is fake_service  # type: ignore[attr-defined]


def test_build_content_ops_skill_store_uses_injected_registry() -> None:
    """Skill-store factory accepts an injected registry so tests
    can verify the wiring without depending on the host's
    on-disk skills directory."""

    sentinel_skill = SimpleNamespace(content="injected body")
    stub_registry = SimpleNamespace(
        get=lambda name: sentinel_skill if name == "x/y" else None,
    )
    store = build_content_ops_skill_store(registry=stub_registry)
    assert store.get_prompt("x/y") == "injected body"
    assert store.get_prompt("missing") is None
