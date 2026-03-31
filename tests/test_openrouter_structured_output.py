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


def test_openrouter_anthropic_caches_stable_system_and_initial_user_prefix():
    llm = OpenRouterLLM(model="anthropic/claude-sonnet-4-6", api_key="test-key")

    payload = llm._convert_messages([
        Message(role="system", content="s" * 1500),
        Message(role="user", content="u" * 3000),
        Message(role="assistant", content='{"draft":true}'),
        Message(role="user", content="fix this"),
    ])

    assert payload[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload[2]["content"] == '{"draft":true}'
    assert payload[3]["content"] == "fix this"


def test_openrouter_reports_cache_usage_fields():
    client = MagicMock()
    client.post.return_value = _FakeResponse(
        {
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {
                "prompt_tokens": 2500,
                "completion_tokens": 400,
                "completion_tokens_details": {"reasoning_tokens": 120},
                "prompt_tokens_details": {
                    "cached_tokens": 2200,
                    "cache_write_tokens": 2400,
                },
            },
        }
    )
    llm = OpenRouterLLM(model="anthropic/claude-sonnet-4-6", api_key="test-key")
    llm._sync_client = client

    result = llm.chat(_sample_messages(), max_tokens=256)

    assert result["usage"]["input_tokens"] == 2500
    assert result["usage"]["output_tokens"] == 400
    assert result["usage"]["reasoning_tokens"] == 120
    assert result["usage"]["cached_tokens"] == 2200
    assert result["usage"]["cache_write_tokens"] == 2400
    assert result["_trace_meta"]["cached_tokens"] == 2200
    assert result["_trace_meta"]["cache_write_tokens"] == 2400
