"""Tier 1 / Tier 2 OpenRouter callers in b2b_enrichment.py bypass the
OpenRouterLLM wrapper and POST directly via httpx. They were sending the
system prompt as a plain string, which means OpenRouter never marked it
cacheable -- 0 cached tokens across all enrichment calls.

These tests cover the _maybe_anthropic_cache helper that converts the
system message into Anthropic's content-block array form with
cache_control: ephemeral when the model is an Anthropic OpenRouter
model and the system content is large enough.
"""

from __future__ import annotations

from atlas_brain.autonomous.tasks.b2b_enrichment import (
    _ANTHROPIC_CACHE_MIN_CHARS,
    _maybe_anthropic_cache,
)


_LARGE_SYSTEM = "x" * (_ANTHROPIC_CACHE_MIN_CHARS + 100)
_SMALL_SYSTEM = "x" * (_ANTHROPIC_CACHE_MIN_CHARS - 10)


def test_anthropic_model_large_system_gets_cache_control():
    messages = [
        {"role": "system", "content": _LARGE_SYSTEM},
        {"role": "user", "content": "payload"},
    ]
    out = _maybe_anthropic_cache("anthropic/claude-haiku-4-5", messages)
    assert out[0]["role"] == "system"
    assert isinstance(out[0]["content"], list)
    assert len(out[0]["content"]) == 1
    block = out[0]["content"][0]
    assert block["type"] == "text"
    assert block["text"] == _LARGE_SYSTEM
    assert block["cache_control"] == {"type": "ephemeral"}
    # User message untouched
    assert out[1] == {"role": "user", "content": "payload"}


def test_non_anthropic_model_passes_through_unchanged():
    messages = [
        {"role": "system", "content": _LARGE_SYSTEM},
        {"role": "user", "content": "payload"},
    ]
    out = _maybe_anthropic_cache("openai/gpt-4o-mini", messages)
    # Same list, same string content, no cache_control wrapping
    assert out is messages or out == messages
    assert isinstance(out[0]["content"], str)
    assert out[0]["content"] == _LARGE_SYSTEM


def test_anthropic_model_small_system_does_not_get_cache_control():
    """Anthropic's prompt cache has a minimum size (~1024 tokens). Wrapping
    a small system prompt would still cost the cache_write but never pay
    back, and OpenRouter may reject undersized cache_control entirely."""
    messages = [
        {"role": "system", "content": _SMALL_SYSTEM},
        {"role": "user", "content": "payload"},
    ]
    out = _maybe_anthropic_cache("anthropic/claude-haiku-4-5", messages)
    assert isinstance(out[0]["content"], str)
    assert out[0]["content"] == _SMALL_SYSTEM


def test_user_messages_not_touched_for_anthropic():
    """Cache_control on user messages is risky -- payloads vary per call,
    so caching them only writes and never reads. Helper must only touch
    system messages."""
    big_user = "u" * (_ANTHROPIC_CACHE_MIN_CHARS + 500)
    messages = [
        {"role": "system", "content": _LARGE_SYSTEM},
        {"role": "user", "content": big_user},
    ]
    out = _maybe_anthropic_cache("anthropic/claude-haiku-4-5", messages)
    assert isinstance(out[1]["content"], str)
    assert out[1]["content"] == big_user


def test_empty_messages_pass_through():
    assert _maybe_anthropic_cache("anthropic/claude-haiku-4-5", []) == []


def test_no_system_message_passes_through():
    messages = [{"role": "user", "content": "hello"}]
    out = _maybe_anthropic_cache("anthropic/claude-haiku-4-5", messages)
    assert out[0]["role"] == "user"
    assert isinstance(out[0]["content"], str)


def test_already_cache_controlled_system_left_alone():
    """If a caller already supplied content as a list (e.g. another helper
    upstream), we should not double-wrap."""
    pre_wrapped = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": _LARGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "payload"},
    ]
    out = _maybe_anthropic_cache("anthropic/claude-haiku-4-5", pre_wrapped)
    assert out[0]["content"] == pre_wrapped[0]["content"]
    # No nested wrapping
    assert isinstance(out[0]["content"], list)
    assert len(out[0]["content"]) == 1


def test_empty_model_id_passes_through():
    messages = [
        {"role": "system", "content": _LARGE_SYSTEM},
        {"role": "user", "content": "payload"},
    ]
    out = _maybe_anthropic_cache("", messages)
    assert isinstance(out[0]["content"], str)
    out = _maybe_anthropic_cache(None, messages)  # type: ignore[arg-type]
    assert isinstance(out[0]["content"], str)
