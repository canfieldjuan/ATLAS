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

from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import pytest

pytest.importorskip("torch")

from atlas_brain._content_ops_infrastructure import (
    CONTENT_OPS_FAQ_EMBEDDING_DEVICE,
    CONTENT_OPS_FAQ_EMBEDDING_MODEL,
    CONTENT_OPS_FAQ_EMBEDDING_REVISION,
    _HostEmbeddingPort,
    _HostLLMClient,
    _HostSkillStore,
    build_content_ops_faq_embedding_port,
    build_content_ops_llm_client,
    build_content_ops_skill_store,
)
from atlas_brain.services.embedding.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from extracted_content_pipeline.campaign_llm_client import PipelineLLMClient
from extracted_content_pipeline.campaign_ports import LLMMessage
import extracted_content_pipeline.pipelines.llm as pipeline_llm_module

_ROOT = Path(__file__).resolve().parents[1]


def test_host_skill_store_returns_existing_skill_content() -> None:
    """Skill store resolves a host-shipped skill via the host
    overrides root."""

    store = build_content_ops_skill_store()
    prompt = store.get_prompt("digest/blog_post_generation")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_blog_generation_prompt_requires_early_entity_clarity() -> None:
    """Blog prompt stays aligned with the GEO entity-clarity gate."""

    store = build_content_ops_skill_store()
    prompt = store.get_prompt("digest/blog_post_generation")
    assert isinstance(prompt, str)
    assert "include that exact phrase in the display `title`" in prompt
    assert "repeat it naturally in the first answer paragraph" in prompt
    assert "At least two H2 sections must start with a 40-120 word answer paragraph" in prompt
    assert "opening 40-60 words" in prompt


def test_blog_generation_prompt_trims_small_support_ticket_uploads() -> None:
    """Host and extracted prompts keep the small-upload support-ticket
    article shape compact."""

    prompt_paths = (
        _ROOT / "atlas_brain/skills/digest/blog_post_generation.md",
        _ROOT / "extracted_content_pipeline/skills/digest/blog_post_generation.md",
    )

    for path in prompt_paths:
        prompt = path.read_text(encoding="utf-8")
        assert "data_context.source_row_count" in prompt
        assert "data_context.included_ticket_row_count" in prompt
        assert "50 or fewer" in prompt
        assert "short FAQ review brief instead of a full article" in prompt
        assert "700-1100 words" in prompt
        assert "3-4 H2 sections" in prompt
        assert "no H3 subsections" in prompt
        assert "no broad scaling/process section" in prompt
        assert "first two H2 sections must open with 40-80 word answer paragraphs" in prompt
        assert "exact `target_keyword`" in prompt
        assert "customers will or could find answers" in prompt
        assert "no repeated sections that explain the same cluster" in prompt
        assert "required_section_outline" in prompt
        assert "draft_faq_shells" in prompt
        assert "Do not add extra benefit, impact, search, or self-service sections" in prompt


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


def test_host_embedding_port_converts_batch_output_to_float_vectors() -> None:
    """Host embedding service output is normalized to the extracted port shape."""

    class _ArrayLike:
        def tolist(self):
            return [["1.0", 2], [3.5, "4.25"]]

    class _FakeEmbeddingService:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_batch(self, texts: list[str]) -> _ArrayLike:
            self.calls.append(texts)
            return _ArrayLike()

    service = _FakeEmbeddingService()
    port = _HostEmbeddingPort(service)

    assert port.embed_texts(["refund help", "password reset"]) == (
        (1.0, 2.0),
        (3.5, 4.25),
    )
    assert service.calls == [["refund help", "password reset"]]


def test_content_ops_faq_embedding_factory_pins_mxbai_offline_model() -> None:
    """Factory builds the mxbai service with the pinned offline contract."""

    received: dict[str, Any] = {}

    class _FakeEmbeddingService:
        def __init__(self, **kwargs: Any) -> None:
            received.update(kwargs)

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0] for text in texts]

    port = build_content_ops_faq_embedding_port(
        embedding_service_factory=_FakeEmbeddingService
    )

    assert received == {
        "model_name": CONTENT_OPS_FAQ_EMBEDDING_MODEL,
        "device": CONTENT_OPS_FAQ_EMBEDDING_DEVICE,
        "revision": CONTENT_OPS_FAQ_EMBEDDING_REVISION,
        "local_files_only": True,
    }
    assert port.embed_texts(["a", "abcd"]) == ((1.0, 1.0), (4.0, 1.0))


def test_sentence_transformer_embedding_load_uses_revision_and_offline_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host embedding service passes revision/offline args to the loader."""

    received: dict[str, Any] = {}

    class _FakeSentenceTransformer:
        device = "cpu"

        def __init__(self, model_name: str, **kwargs: Any) -> None:
            received["model_name"] = model_name
            received["kwargs"] = kwargs

        def get_sentence_embedding_dimension(self) -> int:
            return 1024

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )

    service = SentenceTransformerEmbedding(
        model_name="example/model",
        device="cpu",
        revision="abc123",
        local_files_only=True,
    )
    service.load()

    assert received == {
        "model_name": "example/model",
        "kwargs": {
            "device": "cpu",
            "local_files_only": True,
            "revision": "abc123",
        },
    }
    assert service.dimension == 1024


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


@pytest.mark.asyncio
async def test_build_content_ops_llm_client_uses_pipeline_tracing_client(
    monkeypatch,
) -> None:
    """Factory resolves through Atlas pipeline routing before
    looking at the active registry slot. This is the production
    path for configured OpenRouter Claude models that are not
    pre-activated globally."""

    calls: list[dict[str, Any]] = []
    trace_calls: list[tuple[str, dict[str, Any]]] = []

    class _TracingFakeService:
        model = "anthropic/claude-haiku-4-5"
        name = "openrouter"

        def chat(self, messages, *, max_tokens, temperature):
            del messages, max_tokens, temperature
            return {
                "response": "ok",
                "model": "anthropic/claude-haiku-4-5",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }

    fake_service = _TracingFakeService()

    def _resolver(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return fake_service

    monkeypatch.setattr(
        pipeline_llm_module,
        "trace_llm_call",
        lambda span_name, **kwargs: trace_calls.append((span_name, kwargs)),
    )

    client = build_content_ops_llm_client(
        llm_registry=SimpleNamespace(get_active=lambda: None),
        pipeline_llm_resolver=_resolver,
    )

    assert isinstance(client, PipelineLLMClient)
    assert calls == [{
        "workload": "openrouter",
        "prefer_cloud": True,
        "try_openrouter": True,
        "auto_activate_ollama": False,
        "openrouter_model": None,
    }]

    response = await client.complete(
        [LLMMessage(role="user", content="Draft")],
        max_tokens=32,
        temperature=0.2,
        metadata={"asset_type": "blog_post", "request_id": "req-hosted"},
    )

    assert response.content == "ok"
    assert calls == [
        {
            "workload": "openrouter",
            "prefer_cloud": True,
            "try_openrouter": True,
            "auto_activate_ollama": False,
            "openrouter_model": None,
        },
        {
            "workload": "openrouter",
            "prefer_cloud": True,
            "try_openrouter": True,
            "auto_activate_ollama": False,
            "openrouter_model": None,
        },
    ]
    assert len(trace_calls) == 1
    span_name, trace = trace_calls[0]
    assert span_name == "content_ops.llm.complete"
    assert trace["input_tokens"] == 11
    assert trace["output_tokens"] == 7
    assert trace["metadata"] == {
        "product": "content_ops",
        "workload": "openrouter",
        "llm_adapter": "pipeline",
        "asset_type": "blog_post",
        "request_id": "req-hosted",
        "cache_mode": "no_store",
        "cache_reason": "exact_cache_disabled",
    }


def test_build_content_ops_llm_client_falls_back_to_active_registry_service() -> None:
    """If pipeline routing is unavailable, preserve the previous
    active-registry behavior."""

    fake_service = SimpleNamespace(model_info=None, chat=lambda *a, **k: {})
    stub_registry = SimpleNamespace(get_active=lambda: fake_service)
    client = build_content_ops_llm_client(
        llm_registry=stub_registry,
        pipeline_llm_resolver=lambda **_kwargs: None,
    )
    assert isinstance(client, PipelineLLMClient)
    assert client.resolver() is fake_service


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
