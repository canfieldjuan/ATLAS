from unittest.mock import MagicMock

from atlas_brain.services.llm.openrouter import OpenRouterLLM
from atlas_brain.services.protocols import Message


class _FakeResponse:
    def __init__(self, data):
        self._data = data
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _sample_messages():
    return [Message(role="user", content='{"vendor":"ClickUp"}')]


def test_openrouter_uses_json_schema_when_guided_json_present():
    client = MagicMock()
    client.post.return_value = _FakeResponse(
        {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {},
        }
    )
    llm = OpenRouterLLM(model="openai/o4-mini", api_key="test-key")
    llm._sync_client = client

    llm.chat(
        _sample_messages(),
        max_tokens=256,
        guided_json={"title": "battle_card", "type": "object"},
        response_format={"type": "json_object"},
    )

    payload = client.post.call_args.kwargs["json"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "battle_card"
    assert payload["response_format"]["json_schema"]["schema"]["type"] == "object"
    assert payload["reasoning"] == {"exclude": True}
    assert payload["plugins"] == [{"id": "response-healing"}]


def test_openrouter_structured_request_ignores_non_json_reasoning_fallback():
    client = MagicMock()
    client.post.return_value = _FakeResponse(
        {
            "choices": [{
                "message": {
                    "content": None,
                    "reasoning": "I think the answer should mention pricing pressure first.",
                }
            }],
            "usage": {},
        }
    )
    llm = OpenRouterLLM(model="openai/o4-mini", api_key="test-key")
    llm._sync_client = client

    result = llm.chat(
        _sample_messages(),
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    assert result["response"] == ""


def test_openrouter_structured_request_can_use_json_reasoning_fallback():
    client = MagicMock()
    client.post.return_value = _FakeResponse(
        {
            "choices": [{
                "message": {
                    "content": None,
                    "reasoning": '{"executive_summary":"Pricing pressure is rising."}',
                }
            }],
            "usage": {},
        }
    )
    llm = OpenRouterLLM(model="openai/o4-mini", api_key="test-key")
    llm._sync_client = client

    result = llm.chat(
        _sample_messages(),
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    assert result["response"] == '{"executive_summary":"Pricing pressure is rising."}'


def test_openrouter_unstructured_request_can_use_reasoning_fallback():
    client = MagicMock()
    client.post.return_value = _FakeResponse(
        {
            "choices": [{
                "message": {
                    "content": None,
                    "reasoning": "Pricing pressure is rising.",
                }
            }],
            "usage": {},
        }
    )
    llm = OpenRouterLLM(model="openai/o4-mini", api_key="test-key")
    llm._sync_client = client

    result = llm.chat(_sample_messages(), max_tokens=256)

    assert result["response"] == "Pricing pressure is rising."
