"""
Unit tests for HybridLLM backend routing, fallback, and kwarg handling.

Tests use mocked local/cloud backends -- no real APIs or GPU needed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.services.llm.hybrid import HybridLLM
from atlas_brain.services.protocols import Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _mock_chat_result(text: str = "hello") -> dict:
    return {"response": text, "message": {"role": "assistant", "content": text}}


def _mock_tool_result(text: str = "", tool_calls: list | None = None) -> dict:
    return {
        "response": text,
        "tool_calls": tool_calls or [],
        "message": {"role": "assistant", "content": text},
    }


@pytest.fixture
def hybrid():
    """Create a HybridLLM with mocked sub-backends."""
    llm = HybridLLM(local_kwargs={}, cloud_kwargs={})

    # Mock local backend
    local = MagicMock()
    local.model_info = MagicMock()
    local.model_info.model_id = "qwen3:30b"
    local.chat.return_value = _mock_chat_result("local-response")
    local.chat_with_tools.return_value = _mock_tool_result("local-tool-response")
    local.chat_async = AsyncMock(return_value="local-async-response")
    local.prefill_async = AsyncMock(return_value={"prompt_tokens": 10, "prefill_time_ms": 5.0})

    async def _stream(*args, **kwargs):
        for token in ["local", "-", "stream"]:
            yield token

    local.chat_stream_async = _stream

    # Mock cloud backend
    cloud = MagicMock()
    cloud.model_info = MagicMock()
    cloud.model_info.model_id = "groq:llama-70b|together:llama-70b"
    cloud.chat.return_value = _mock_chat_result("cloud-response")
    cloud.chat_with_tools.return_value = _mock_tool_result("cloud-tool-response")
    cloud.chat_async = AsyncMock(return_value="cloud-async-response")

    llm._local = local
    llm._cloud = cloud
    llm._loaded = True

    return llm


@pytest.fixture
def hybrid_local_only():
    """HybridLLM with only local backend available."""
    llm = HybridLLM(local_kwargs={}, cloud_kwargs={})

    local = MagicMock()
    local.model_info = MagicMock()
    local.model_info.model_id = "qwen3:30b"
    local.chat.return_value = _mock_chat_result("local-response")
    local.chat_with_tools.return_value = _mock_tool_result("local-tool-response")
    local.chat_async = AsyncMock(return_value="local-async-response")
    local.prefill_async = AsyncMock(return_value={"prompt_tokens": 10, "prefill_time_ms": 5.0})

    llm._local = local
    llm._cloud = None
    llm._loaded = True

    return llm


@pytest.fixture
def hybrid_cloud_only():
    """HybridLLM with only cloud backend available."""
    llm = HybridLLM(local_kwargs={}, cloud_kwargs={})

    cloud = MagicMock()
    cloud.model_info = MagicMock()
    cloud.model_info.model_id = "groq:llama-70b"
    cloud.chat.return_value = _mock_chat_result("cloud-response")
    cloud.chat_with_tools.return_value = _mock_tool_result("cloud-tool-response")
    cloud.chat_async = AsyncMock(return_value="cloud-async-response")

    llm._local = None
    llm._cloud = cloud
    llm._loaded = True

    return llm


MESSAGES = [_make_msg("user", "hello")]
TOOLS = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]


# ---------------------------------------------------------------------------
# Routing: default method -> backend mapping
# ---------------------------------------------------------------------------

class TestDefaultRouting:
    """Verify each method routes to the correct default backend."""

    def test_chat_routes_to_local(self, hybrid):
        result = hybrid.chat(MESSAGES)
        hybrid._local.chat.assert_called_once()
        hybrid._cloud.chat.assert_not_called()
        assert result["routed_to"] == "local"

    def test_chat_with_tools_routes_to_cloud(self, hybrid):
        result = hybrid.chat_with_tools(MESSAGES, tools=TOOLS)
        hybrid._cloud.chat_with_tools.assert_called_once()
        hybrid._local.chat_with_tools.assert_not_called()
        assert result["routed_to"] == "cloud"

    @pytest.mark.asyncio
    async def test_chat_async_routes_to_local(self, hybrid):
        result = await hybrid.chat_async(MESSAGES)
        hybrid._local.chat_async.assert_awaited_once()
        hybrid._cloud.chat_async.assert_not_awaited()
        assert result == "local-async-response"

    @pytest.mark.asyncio
    async def test_chat_stream_routes_to_local(self, hybrid):
        tokens = []
        async for token in hybrid.chat_stream_async(MESSAGES):
            tokens.append(token)
        assert tokens == ["local", "-", "stream"]

    @pytest.mark.asyncio
    async def test_prefill_routes_to_local(self, hybrid):
        result = await hybrid.prefill_async(MESSAGES)
        hybrid._local.prefill_async.assert_awaited_once()
        assert result["prompt_tokens"] == 10

    def test_generate_routes_to_local(self, hybrid):
        result = hybrid.generate("say hello")
        hybrid._local.chat.assert_called_once()
        hybrid._cloud.chat.assert_not_called()
        assert result["routed_to"] == "local"


# ---------------------------------------------------------------------------
# Explicit backend override
# ---------------------------------------------------------------------------

class TestBackendOverride:
    """Verify backend= kwarg overrides default routing."""

    def test_chat_override_to_cloud(self, hybrid):
        result = hybrid.chat(MESSAGES, backend="cloud")
        hybrid._cloud.chat.assert_called_once()
        hybrid._local.chat.assert_not_called()
        assert result["routed_to"] == "cloud"

    def test_chat_with_tools_override_to_local(self, hybrid):
        result = hybrid.chat_with_tools(MESSAGES, tools=TOOLS, backend="local")
        hybrid._local.chat_with_tools.assert_called_once()
        hybrid._cloud.chat_with_tools.assert_not_called()
        assert result["routed_to"] == "local"

    @pytest.mark.asyncio
    async def test_chat_async_override_to_cloud(self, hybrid):
        result = await hybrid.chat_async(MESSAGES, backend="cloud")
        hybrid._cloud.chat_async.assert_awaited_once()
        hybrid._local.chat_async.assert_not_awaited()
        assert result == "cloud-async-response"

    def test_generate_override_to_cloud(self, hybrid):
        result = hybrid.generate("say hello", backend="cloud")
        hybrid._cloud.chat.assert_called_once()
        hybrid._local.chat.assert_not_called()
        assert result["routed_to"] == "cloud"


# ---------------------------------------------------------------------------
# backend= kwarg must NOT leak to sub-backends
# ---------------------------------------------------------------------------

class TestBackendKwargConsumed:
    """Verify backend= is popped and not passed to underlying backends."""

    def test_chat_does_not_leak_backend(self, hybrid):
        hybrid.chat(MESSAGES, backend="local")
        _, kwargs = hybrid._local.chat.call_args
        assert "backend" not in kwargs

    def test_chat_with_tools_does_not_leak_backend(self, hybrid):
        hybrid.chat_with_tools(MESSAGES, tools=TOOLS, backend="cloud")
        _, kwargs = hybrid._cloud.chat_with_tools.call_args
        assert "backend" not in kwargs

    @pytest.mark.asyncio
    async def test_chat_async_does_not_leak_backend(self, hybrid):
        await hybrid.chat_async(MESSAGES, backend="local")
        _, kwargs = hybrid._local.chat_async.call_args
        assert "backend" not in kwargs


# ---------------------------------------------------------------------------
# Fallback: unavailable backend -> try the other
# ---------------------------------------------------------------------------

class TestFallbackOnUnavailable:
    """When preferred backend is None, fall back to the available one."""

    def test_chat_falls_back_to_cloud(self, hybrid_cloud_only):
        result = hybrid_cloud_only.chat(MESSAGES)
        hybrid_cloud_only._cloud.chat.assert_called_once()
        assert result["routed_to"] == "cloud"

    def test_chat_with_tools_falls_back_to_local(self, hybrid_local_only):
        result = hybrid_local_only.chat_with_tools(MESSAGES, tools=TOOLS)
        hybrid_local_only._local.chat_with_tools.assert_called_once()
        assert result["routed_to"] == "local"

    @pytest.mark.asyncio
    async def test_chat_async_falls_back_to_cloud(self, hybrid_cloud_only):
        result = await hybrid_cloud_only.chat_async(MESSAGES)
        hybrid_cloud_only._cloud.chat_async.assert_awaited_once()
        assert result == "cloud-async-response"


# ---------------------------------------------------------------------------
# Fallback: runtime exception -> retry on other backend
# ---------------------------------------------------------------------------

class TestFallbackOnException:
    """When preferred backend raises, retry on the other."""

    def test_chat_retries_on_cloud_after_local_failure(self, hybrid):
        hybrid._local.chat.side_effect = RuntimeError("ollama down")
        result = hybrid.chat(MESSAGES)
        hybrid._local.chat.assert_called_once()
        hybrid._cloud.chat.assert_called_once()
        assert result["routed_to"] == "cloud"

    def test_chat_with_tools_retries_on_local_after_cloud_failure(self, hybrid):
        hybrid._cloud.chat_with_tools.side_effect = RuntimeError("groq down")
        result = hybrid.chat_with_tools(MESSAGES, tools=TOOLS)
        hybrid._cloud.chat_with_tools.assert_called_once()
        hybrid._local.chat_with_tools.assert_called_once()
        assert result["routed_to"] == "local"

    @pytest.mark.asyncio
    async def test_chat_async_retries_on_cloud_after_local_failure(self, hybrid):
        hybrid._local.chat_async.side_effect = RuntimeError("ollama down")
        result = await hybrid.chat_async(MESSAGES)
        hybrid._local.chat_async.assert_awaited_once()
        hybrid._cloud.chat_async.assert_awaited_once()
        assert result == "cloud-async-response"

    def test_chat_raises_when_both_fail(self, hybrid):
        hybrid._local.chat.side_effect = RuntimeError("ollama down")
        hybrid._cloud.chat.side_effect = RuntimeError("groq down")
        with pytest.raises(RuntimeError, match="groq down"):
            hybrid.chat(MESSAGES)

    def test_chat_raises_when_only_backend_fails(self, hybrid_local_only):
        hybrid_local_only._local.chat.side_effect = RuntimeError("ollama down")
        with pytest.raises(RuntimeError, match="ollama down"):
            hybrid_local_only.chat(MESSAGES)


# ---------------------------------------------------------------------------
# Streaming/prefill require local
# ---------------------------------------------------------------------------

class TestLocalOnlyMethods:
    """chat_stream_async and prefill_async require local backend."""

    @pytest.mark.asyncio
    async def test_stream_raises_without_local(self, hybrid_cloud_only):
        with pytest.raises(RuntimeError, match="streaming requires local"):
            async for _ in hybrid_cloud_only.chat_stream_async(MESSAGES):
                pass

    @pytest.mark.asyncio
    async def test_prefill_raises_without_local(self, hybrid_cloud_only):
        with pytest.raises(RuntimeError, match="prefill requires local"):
            await hybrid_cloud_only.prefill_async(MESSAGES)


# ---------------------------------------------------------------------------
# model_info
# ---------------------------------------------------------------------------

class TestModelInfo:

    def test_model_info_both_backends(self, hybrid):
        info = hybrid.model_info
        assert "local:" in info.model_id
        assert "cloud:" in info.model_id
        assert info.device == "hybrid"
        assert info.is_loaded is True

    def test_model_info_local_only(self, hybrid_local_only):
        info = hybrid_local_only.model_info
        assert "local:" in info.model_id
        assert "cloud:" not in info.model_id

    def test_model_info_cloud_only(self, hybrid_cloud_only):
        info = hybrid_cloud_only.model_info
        assert "cloud:" in info.model_id
        assert "local:" not in info.model_id

    def test_model_info_not_loaded(self):
        llm = HybridLLM()
        info = llm.model_info
        assert info.model_id == "hybrid:not-loaded"
        assert info.is_loaded is False


# ---------------------------------------------------------------------------
# Load / unload
# ---------------------------------------------------------------------------

class TestLoadUnload:

    @patch("atlas_brain.services.llm.cloud.CloudLLM")
    @patch("atlas_brain.services.llm.ollama.OllamaLLM")
    def test_load_creates_both_backends(self, MockOllama, MockCloud):
        MockOllama.return_value.load.return_value = None
        MockCloud.return_value.load.return_value = None

        llm = HybridLLM(
            local_kwargs={"model": "qwen3:30b", "base_url": "http://localhost:11434"},
            cloud_kwargs={"groq_model": "llama-70b"},
        )
        llm.load()

        MockOllama.assert_called_once_with(model="qwen3:30b", base_url="http://localhost:11434")
        MockCloud.assert_called_once_with(groq_model="llama-70b")
        assert llm.is_loaded is True

    @patch("atlas_brain.services.llm.cloud.CloudLLM")
    @patch("atlas_brain.services.llm.ollama.OllamaLLM")
    def test_load_survives_cloud_failure(self, MockOllama, MockCloud):
        MockOllama.return_value.load.return_value = None
        MockCloud.return_value.load.side_effect = ValueError("no API keys")

        llm = HybridLLM()
        llm.load()

        assert llm._local is not None
        assert llm._cloud is None
        assert llm.is_loaded is True

    @patch("atlas_brain.services.llm.cloud.CloudLLM")
    @patch("atlas_brain.services.llm.ollama.OllamaLLM")
    def test_load_survives_local_failure(self, MockOllama, MockCloud):
        MockOllama.return_value.load.side_effect = RuntimeError("no ollama")
        MockCloud.return_value.load.return_value = None

        llm = HybridLLM()
        llm.load()

        assert llm._local is None
        assert llm._cloud is not None
        assert llm.is_loaded is True

    @patch("atlas_brain.services.llm.cloud.CloudLLM")
    @patch("atlas_brain.services.llm.ollama.OllamaLLM")
    def test_load_fails_when_both_fail(self, MockOllama, MockCloud):
        MockOllama.return_value.load.side_effect = RuntimeError("no ollama")
        MockCloud.return_value.load.side_effect = ValueError("no API keys")

        llm = HybridLLM()
        with pytest.raises(RuntimeError, match="both local and cloud"):
            llm.load()

    def test_unload_cleans_up(self, hybrid):
        hybrid.unload()
        assert hybrid._local is None
        assert hybrid._cloud is None
        assert hybrid.is_loaded is False
