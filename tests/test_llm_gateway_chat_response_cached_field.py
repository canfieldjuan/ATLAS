"""Regression tests for PR-D6e: ChatResponse.cached field.

Pins three contracts:

1. The ``ChatResponse`` Pydantic model declares ``cached: bool = False``
   in source.
2. The cache-hit branch of ``chat()`` constructs the response with
   ``cached=True``.
3. The provider-call (miss) branch of ``chat()`` constructs the
   response with ``cached=False`` (explicit, not relying on the
   default).

Tests use file-text inspection so they don't require the gateway
module's full settings stack (which depends on
``ATLAS_SAAS_API_KEY_PEPPER`` / ``ATLAS_SAAS_BYOK_ENCRYPTION_KEK``).
This matches the source-text style of the existing
``test_llm_gateway_router.py`` cache tests.

See plans/PR-D6e-chat-response-cached-field.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATEWAY_PATH = ROOT / "atlas_brain" / "api" / "llm_gateway.py"


@pytest.fixture(scope="module")
def gateway_source() -> str:
    return GATEWAY_PATH.read_text(encoding="utf-8")


def test_chat_response_model_declares_cached_field(gateway_source):
    """ChatResponse must expose `cached: bool = False`. Default-False
    keeps the contract additive: existing clients that don't
    deserialize the field aren't affected."""
    # Locate the ChatResponse class block.
    assert "class ChatResponse(BaseModel):" in gateway_source
    block = gateway_source.split("class ChatResponse(BaseModel):", 1)[1]
    # Take just the class body up to the next class definition.
    block = block.split("\nclass ", 1)[0]
    assert "cached: bool = False" in block, (
        "ChatResponse must declare `cached: bool = False` -- explicit "
        "type annotation and default-False for additive contract."
    )


def test_cache_hit_path_returns_cached_true(gateway_source):
    """The cache-hit branch in chat() returns ChatResponse(..., cached=True)."""
    # The cache-hit block is bounded by `if cache_hit is not None:`
    # at the top and the comment that introduces the provider-call
    # path at the bottom.
    assert "if cache_hit is not None:" in gateway_source
    hit_block = gateway_source.split("if cache_hit is not None:", 1)[1].split(
        "# Capture full response", 1
    )[0]
    assert "cached=True" in hit_block, (
        "Cache-hit return path must set cached=True -- the explicit "
        "signal customers can read on every chat response."
    )


def test_cache_miss_path_returns_cached_false_explicitly(gateway_source):
    """Total occurrence count: exactly one cached=True (hit path)
    and exactly one cached=False (miss path). Pinning the explicit
    miss-path value rather than relying on the default keeps the
    contract visible at the call site."""
    assert gateway_source.count("cached=True") == 1, (
        "Expected exactly one cached=True return (cache-hit path)."
    )
    assert gateway_source.count("cached=False") == 1, (
        "Expected exactly one cached=False return (cache-miss path) "
        "-- explicit value preferred over relying on the model default."
    )


def test_chat_response_cached_field_appears_after_usage_in_class_body(gateway_source):
    """Field declaration order: cached comes after usage in the class
    body. Stable order keeps generated OpenAPI schemas / docstrings
    consistent."""
    block = gateway_source.split("class ChatResponse(BaseModel):", 1)[1].split(
        "\nclass ", 1
    )[0]
    usage_idx = block.find("usage: ChatUsage")
    cached_idx = block.find("cached: bool")
    assert usage_idx >= 0 and cached_idx >= 0
    assert cached_idx > usage_idx, (
        "cached field should be declared after usage so the "
        "generated schema lists usage first (existing clients) "
        "and cached after (new field)."
    )
