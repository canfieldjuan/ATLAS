"""Tests for the AnthropicBatchableLLM Protocol (PR-A5d).

Pins the structural-typing semantics that replaced 9
``isinstance(llm, AnthropicLLM)`` checks across the batch path:

  * ``AnthropicLLM`` MUST satisfy the Protocol (would otherwise break
    9 dispatch + guard sites at runtime).
  * Other LLM providers (Ollama, OpenRouter, etc.) MUST NOT satisfy
    the Protocol -- they would otherwise be wrongly dispatched into
    the Anthropic batch API.
  * The Protocol is ``runtime_checkable``; structural duck-typing
    works without nominal subclass.
  * The companion ``getattr(llm, "_async_client", None) is None``
    check still gates on whether ``load()`` has been called -- the
    Protocol covers attribute *presence*, not loaded-ness.
"""

from __future__ import annotations

from atlas_brain.services.llm.anthropic import (
    AnthropicBatchableLLM,
    AnthropicLLM,
)


# ---- Protocol satisfaction (positive cases) ----


def test_anthropic_llm_class_satisfies_protocol():
    """The whole point of the Protocol -- the production class
    keeps satisfying it without nominal subclass declarations.
    """
    llm = AnthropicLLM()
    assert isinstance(llm, AnthropicBatchableLLM)


def test_duck_typed_class_satisfies_protocol():
    """A future Anthropic-via-proxy adapter that exposes the same
    attribute surface should pass without inheriting from
    ``AnthropicLLM``.
    """
    class FakeBatchable:
        def __init__(self) -> None:
            self.name = "anthropic-via-proxy"
            self.model = "claude-haiku-4-5"
            self._async_client = object()

    assert isinstance(FakeBatchable(), AnthropicBatchableLLM)


def test_subclass_satisfies_protocol():
    """``AnthropicLLM`` subclasses (e.g. test fakes that extend the
    real class) must keep satisfying the Protocol.
    """
    class SubAnthropic(AnthropicLLM):
        pass

    assert isinstance(SubAnthropic(), AnthropicBatchableLLM)


# ---- Protocol satisfaction (negative cases) ----


def test_class_missing_async_client_does_not_satisfy():
    """The Protocol REQUIRES ``_async_client`` so that other LLM
    providers (Ollama / OpenRouter / Together / Groq) -- which expose
    only ``name`` + ``model`` from the shared base class -- do not
    accidentally satisfy the structural check and get routed into
    the Anthropic batch API.
    """
    class Half:
        def __init__(self) -> None:
            self.name = "x"
            self.model = "y"
            # missing _async_client

    assert not isinstance(Half(), AnthropicBatchableLLM)


def test_class_missing_model_does_not_satisfy():
    class Half:
        def __init__(self) -> None:
            self.name = "x"
            self._async_client = object()
            # missing model

    assert not isinstance(Half(), AnthropicBatchableLLM)


def test_class_missing_name_does_not_satisfy():
    class Half:
        def __init__(self) -> None:
            self.model = "x"
            self._async_client = object()
            # missing name

    assert not isinstance(Half(), AnthropicBatchableLLM)


def test_other_provider_classes_do_not_satisfy():
    """The non-Anthropic LLM providers in the codebase must NOT
    satisfy this Protocol; if any did, the dispatch gates would
    incorrectly route them into the Anthropic batch API.
    """
    from atlas_brain.services.llm.ollama import OllamaLLM
    from atlas_brain.services.llm.openrouter import OpenRouterLLM
    from atlas_brain.services.llm.together import TogetherLLM
    from atlas_brain.services.llm.groq import GroqLLM

    for ProviderCls in (OllamaLLM, OpenRouterLLM, TogetherLLM, GroqLLM):
        instance = ProviderCls()
        assert not isinstance(instance, AnthropicBatchableLLM), (
            f"{ProviderCls.__name__} should NOT satisfy AnthropicBatchableLLM"
        )


# ---- Companion guard semantics ----


def test_unloaded_client_passes_protocol_but_fails_companion_guard():
    """The Protocol passes because the attribute exists (its value
    is None pre-load). The dispatch sites use a companion
    ``getattr(llm, "_async_client", None) is None`` check to gate
    on actually-loaded; this test pins that semantic.
    """
    llm = AnthropicLLM()
    assert llm._async_client is None
    assert isinstance(llm, AnthropicBatchableLLM)
    # Replicate the production guard expression
    is_dispatch_eligible = (
        isinstance(llm, AnthropicBatchableLLM)
        and getattr(llm, "_async_client", None) is not None
    )
    assert is_dispatch_eligible is False


def test_loaded_client_passes_companion_guard():
    """Once ``load()`` is called, the companion guard flips True.
    Use a stand-in client object (no real API call needed)."""
    llm = AnthropicLLM()
    llm._async_client = object()  # simulate load() effect
    is_dispatch_eligible = (
        isinstance(llm, AnthropicBatchableLLM)
        and getattr(llm, "_async_client", None) is not None
    )
    assert is_dispatch_eligible is True


# ---- runtime_checkable nature ----


def test_protocol_is_runtime_checkable():
    """``isinstance`` against a non-runtime_checkable Protocol raises
    ``TypeError`` at the call site. Pin the decorator so a future
    refactor that drops it would fail this test instead of breaking
    9 production isinstance sites silently.
    """
    # Try and succeed (no TypeError) -- if runtime_checkable was
    # removed, isinstance(_, AnthropicBatchableLLM) would raise.
    isinstance(object(), AnthropicBatchableLLM)


# ---- Backwards-compat: AnthropicLLM still publicly importable ----


def test_anthropic_llm_class_still_publicly_importable():
    """Rule 14 guard: callers may still import AnthropicLLM directly
    (the Protocol is additive, not replacing).
    """
    from atlas_brain.services.llm.anthropic import AnthropicLLM as cls
    assert cls is AnthropicLLM


def test_protocol_attribute_surface_matches_documented_contract():
    """The Protocol declares ``name``, ``model``, ``_async_client``.
    Pinning the field names so an accidental rename in the Protocol
    declaration would be caught immediately rather than only at
    runtime via a missing-attribute false-negative.
    """
    annotations = AnthropicBatchableLLM.__annotations__
    assert "name" in annotations
    assert "model" in annotations
    assert "_async_client" in annotations
