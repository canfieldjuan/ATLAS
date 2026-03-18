from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from atlas_brain.reasoning.graph import _llm_generate
from atlas_brain.services.llm.anthropic import AnthropicLLM
from atlas_brain.services.protocols import Message


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [SimpleNamespace(type="text", text=text)]
        self.usage = SimpleNamespace(input_tokens=12, output_tokens=7)
        self.id = "req_test"


def test_anthropic_chat_uses_request_timeout_override():
    base_client = MagicMock()
    timed_client = MagicMock()
    base_client.with_options.return_value = timed_client
    timed_client.messages.create.return_value = _FakeAnthropicResponse('{"ok":true}')

    llm = AnthropicLLM(model="claude-sonnet-4-5-20250929", api_key="test-key")
    llm._sync_client = base_client

    result = llm.chat(
        [Message(role="user", content="test")],
        max_tokens=256,
        temperature=0.1,
        timeout=120.0,
    )

    base_client.with_options.assert_called_once_with(timeout=120.0)
    timed_client.messages.create.assert_called_once()
    assert result["response"] == '{"ok":true}'


@pytest.mark.asyncio
async def test_graph_llm_generate_passes_timeout(monkeypatch):
    async def _direct_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("atlas_brain.reasoning.graph.asyncio.to_thread", _direct_to_thread)

    class FakeLLM:
        def __init__(self):
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {
                "response": '{"ok":true}',
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }

    llm = FakeLLM()
    result = await _llm_generate(
        llm,
        prompt="hello",
        system_prompt="system",
        max_tokens=64,
        temperature=0.2,
        timeout=120.0,
    )

    assert llm.kwargs["timeout"] == 120.0
    assert result["response"] == '{"ok":true}'
