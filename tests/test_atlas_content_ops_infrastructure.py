"""Pin the host's Content Ops LLM + Skill adapters.

`atlas_brain/_content_ops_infrastructure.py` bridges the host's
sync `LLMService` and `SkillRegistry` to the extracted package's
async `LLMClient` / sync `SkillStore` ports. Four regression
tests:

1. Skill adapter resolves an existing skill name to its markdown
   body.
2. Skill adapter returns `None` for missing skill names (canary
   for the "not loaded yet" branch).
3. LLM adapter translates `LLMMessage` -> host `Message`,
   bridges sync host `chat()` to async via `asyncio.to_thread`,
   and wraps the dict response into an `LLMResponse`.
4. `build_content_ops_llm_client()` returns `None` when no host
   LLM is active -- the bundle factory's signal to skip wiring
   LLM-needing services.
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
    """Skill adapter resolves a real host skill via `get_skill_registry()`."""

    store = build_content_ops_skill_store()
    # The blog-post skill ships with the repo at
    # atlas_brain/skills/digest/blog_post_generation.md.
    prompt = store.get_prompt("digest/blog_post_generation")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_host_skill_store_returns_none_for_missing_skill() -> None:
    """Skill adapter doesn't raise on lookup miss."""

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


def test_build_content_ops_llm_client_returns_none_when_no_active() -> None:
    """Factory short-circuits when `llm_registry.get_active()`
    returns None -- the bundle factory's signal to leave LLM-
    needing slots `None`."""

    stub_registry = SimpleNamespace(get_active=lambda: None)
    assert build_content_ops_llm_client(llm_registry=stub_registry) is None


def test_build_content_ops_llm_client_wraps_active_service() -> None:
    """Factory returns a `_HostLLMClient` adapter wrapping the
    currently active host service."""

    fake_service = SimpleNamespace(model_info=None, chat=lambda *a, **k: {})
    stub_registry = SimpleNamespace(get_active=lambda: fake_service)
    client = build_content_ops_llm_client(llm_registry=stub_registry)
    assert isinstance(client, _HostLLMClient)
    # The wrapped service is the one we patched.
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
