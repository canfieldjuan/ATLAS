"""Tests for atlas_brain.services.llm.anthropic.convert_messages.

The function under test is pure: takes a list of Message objects and
returns the Anthropic API shape. No I/O, no SDK dependency. Pure unit
tests, no fixtures, no mocks.
"""

from __future__ import annotations

import pytest

from atlas_brain.services.llm.anthropic import (
    _DEFAULT_CACHE_THRESHOLD_CHARS,
    AnthropicLLM,
    convert_messages,
)
from atlas_brain.services.protocols import Message


# ---- Plain message conversion ----


def test_user_message_passes_through():
    sys, api = convert_messages([Message(role="user", content="hi")])
    assert sys == ""
    assert api == [{"role": "user", "content": "hi"}]


def test_assistant_message_passes_through():
    sys, api = convert_messages([Message(role="assistant", content="ok")])
    assert sys == ""
    assert api == [{"role": "assistant", "content": "ok"}]


def test_empty_input_returns_empty_outputs():
    sys, api = convert_messages([])
    assert sys == ""
    assert api == []


# ---- System prompt extraction ----


def test_single_system_message_returns_plain_string():
    sys, api = convert_messages([Message(role="system", content="be helpful")])
    assert sys == "be helpful"
    assert api == []


def test_multiple_system_messages_join_with_double_newline():
    sys, api = convert_messages(
        [
            Message(role="system", content="be helpful"),
            Message(role="system", content="be concise"),
        ]
    )
    assert sys == "be helpful\n\nbe concise"


def test_system_separated_from_messages():
    sys, api = convert_messages(
        [
            Message(role="system", content="be helpful"),
            Message(role="user", content="hello"),
        ]
    )
    assert sys == "be helpful"
    assert api == [{"role": "user", "content": "hello"}]


# ---- cache_threshold_chars policy ----


def test_default_threshold_constant_is_1024():
    # The legacy hardcoded threshold; verify the parametric default
    # matches so behaviour is preserved when no threshold is passed.
    assert _DEFAULT_CACHE_THRESHOLD_CHARS == 1024


def test_short_system_below_threshold_stays_string():
    short = "x" * 500
    sys, _ = convert_messages([Message(role="system", content=short)])
    assert isinstance(sys, str)
    assert sys == short


def test_at_threshold_stays_string():
    # Strict ">" comparison: exactly threshold-many chars stays a string.
    text = "x" * _DEFAULT_CACHE_THRESHOLD_CHARS
    sys, _ = convert_messages([Message(role="system", content=text)])
    assert isinstance(sys, str)


def test_above_threshold_wraps_with_cache_control():
    text = "x" * (_DEFAULT_CACHE_THRESHOLD_CHARS + 1)
    sys, _ = convert_messages([Message(role="system", content=text)])
    assert isinstance(sys, list)
    assert len(sys) == 1
    assert sys[0]["type"] == "text"
    assert sys[0]["text"] == text
    assert sys[0]["cache_control"] == {"type": "ephemeral"}


def test_custom_threshold_lowers_cache_cutoff():
    text = "x" * 100
    sys, _ = convert_messages(
        [Message(role="system", content=text)],
        cache_threshold_chars=50,
    )
    assert isinstance(sys, list)


def test_custom_threshold_raises_cache_cutoff():
    text = "x" * 2000
    sys, _ = convert_messages(
        [Message(role="system", content=text)],
        cache_threshold_chars=4096,
    )
    assert isinstance(sys, str)


# ---- Tool call handling ----


def test_assistant_with_tool_calls_emits_content_blocks():
    msg = Message(
        role="assistant",
        content="thinking",
        tool_calls=[
            {
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": {"city": "SF"}},
            }
        ],
    )
    _, api = convert_messages([msg])
    assert len(api) == 1
    blocks = api[0]["content"]
    assert len(blocks) == 2
    assert blocks[0] == {"type": "text", "text": "thinking"}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {"city": "SF"},
    }


def test_assistant_tool_calls_without_text_skips_text_block():
    msg = Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "function": {"name": "fn", "arguments": {}},
            }
        ],
    )
    _, api = convert_messages([msg])
    blocks = api[0]["content"]
    # No text block when content was empty
    assert all(b["type"] != "text" for b in blocks)
    assert blocks[0]["type"] == "tool_use"


def test_assistant_without_tool_calls_uses_plain_content():
    msg = Message(role="assistant", content="hello", tool_calls=None)
    _, api = convert_messages([msg])
    assert api == [{"role": "assistant", "content": "hello"}]


# ---- Tool result handling ----


def test_tool_result_becomes_user_message_with_tool_result_block():
    msg = Message(role="tool", content="42", tool_call_id="call_1")
    _, api = convert_messages([msg])
    assert len(api) == 1
    assert api[0]["role"] == "user"
    block = api[0]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "42"


def test_consecutive_tool_results_coalesce_into_one_user_message():
    msgs = [
        Message(role="tool", content="r1", tool_call_id="t1"),
        Message(role="tool", content="r2", tool_call_id="t2"),
    ]
    _, api = convert_messages(msgs)
    assert len(api) == 1
    assert api[0]["role"] == "user"
    assert len(api[0]["content"]) == 2


def test_tool_result_without_call_id_uses_empty_string():
    msg = Message(role="tool", content="r1", tool_call_id=None)
    _, api = convert_messages([msg])
    assert api[0]["content"][0]["tool_use_id"] == ""


def test_tool_results_split_when_assistant_message_intervenes():
    msgs = [
        Message(role="tool", content="r1", tool_call_id="t1"),
        Message(role="assistant", content="thanks"),
        Message(role="tool", content="r2", tool_call_id="t2"),
    ]
    _, api = convert_messages(msgs)
    assert len(api) == 3
    assert api[0]["role"] == "user"
    assert api[1]["role"] == "assistant"
    assert api[2]["role"] == "user"


# ---- Backwards-compat alias ----


def test_method_delegates_to_module_function():
    """The legacy `_convert_messages` method must produce the same
    result as the public `convert_messages` function.
    """
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="u"),
        Message(role="assistant", content="a"),
        Message(role="tool", content="t", tool_call_id="t1"),
    ]
    llm = AnthropicLLM()
    method_result = llm._convert_messages(msgs)
    function_result = convert_messages(msgs)
    assert method_result == function_result


def test_anthropic_batch_uses_module_function():
    """``anthropic_batch.convert_messages`` is the lifted public symbol,
    not a re-binding of the method.
    """
    from atlas_brain.services.b2b.anthropic_batch import (
        convert_messages as batch_convert,
    )
    assert batch_convert is convert_messages


# ---- Frozenness of output structure ----


def test_returned_message_dicts_are_independent_per_call():
    """Two consecutive calls must not share aliased dicts (regression
    guard: a future refactor that caches a list could quietly let
    callers mutate each others' output).
    """
    msgs = [Message(role="user", content="hi")]
    _, api1 = convert_messages(msgs)
    _, api2 = convert_messages(msgs)
    api1[0]["content"] = "MUTATED"
    assert api2[0]["content"] == "hi"
