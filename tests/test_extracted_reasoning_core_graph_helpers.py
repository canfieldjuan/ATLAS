"""Unit tests for ``extracted_reasoning_core.graph_helpers``.

PR-C4e1 promoted the host-agnostic helpers from atlas's reasoning
graph into core. These tests exercise each function directly so a
later sub-slice (PR-C4e2/e3) that wires them through the bigger
graph engine has a stable contract to lean on.

All tests are pure -- no atlas imports, no I/O -- so they wire into
the standalone-CI suite cleanly.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from extracted_reasoning_core.graph_helpers import (
    SAFE_ACTIONS,
    build_notification_fallback,
    clean_summary_text,
    complete_with_json,
    extract_completion_text,
    has_suspicious_trailing_fragment,
    make_chat_messages,
    parse_llm_json,
    plan_actions,
    sanitize_notification_summary,
    valid_uuid_str,
)


# ----------------------------------------------------------------------
# parse_llm_json
# ----------------------------------------------------------------------


def test_parse_llm_json_raw_object() -> None:
    assert parse_llm_json('{"k": "v"}') == {"k": "v"}


def test_parse_llm_json_strips_markdown_fence() -> None:
    text = '```json\n{"priority": "high"}\n```'
    assert parse_llm_json(text) == {"priority": "high"}


def test_parse_llm_json_strips_unlabeled_fence() -> None:
    text = '```\n{"priority": "high"}\n```'
    assert parse_llm_json(text) == {"priority": "high"}


def test_parse_llm_json_extracts_object_embedded_in_prose() -> None:
    text = 'Here is the result:\n{"answer": 42}\nThanks.'
    assert parse_llm_json(text) == {"answer": 42}


def test_parse_llm_json_raises_on_empty() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("")


def test_parse_llm_json_raises_on_no_object() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("just prose, no JSON here")


def test_parse_llm_json_raises_on_non_dict_result() -> None:
    # Downstream graph nodes always call ``.get(...)`` on the parsed
    # result -- a JSON array would AttributeError. Pin the dict-only
    # contract so a future refactor can't broaden the return type
    # without an explicit test update.
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("[1, 2, 3]")
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json('"plain string"')
    # Still raises when the array is fenced or embedded.
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("```json\n[1, 2]\n```")


# ----------------------------------------------------------------------
# valid_uuid_str
# ----------------------------------------------------------------------


def test_valid_uuid_str_canonicalizes_valid_uuid() -> None:
    raw = "11111111222233334444555555555555"
    canonical = str(UUID(raw))
    assert valid_uuid_str(raw) == canonical


def test_valid_uuid_str_returns_none_on_invalid() -> None:
    assert valid_uuid_str("not-a-uuid") is None
    assert valid_uuid_str("") is None
    assert valid_uuid_str(None) is None
    assert valid_uuid_str(123) is None  # ints aren't UUIDs even when castable


# ----------------------------------------------------------------------
# clean_summary_text
# ----------------------------------------------------------------------


def test_clean_summary_text_strips_markdown_and_collapses_whitespace() -> None:
    # The helper collapses runs of ``[ \t]+`` to a single space and runs
    # of 3+ newlines to a paragraph break, but does NOT strip line-edge
    # whitespace inside the body -- only the final ``.strip()`` trims
    # the outer ends. So mid-content single spaces survive.
    raw = "**bold** title\n\n\n  with    spaces"
    assert clean_summary_text(raw) == "bold title\n\n with spaces"


def test_clean_summary_text_unwraps_json_fence() -> None:
    raw = "```json\n{\"a\": 1}\n```"
    cleaned = clean_summary_text(raw)
    assert "```" not in cleaned
    assert "{a: 1}" in cleaned or '{"a": 1}' in cleaned


def test_clean_summary_text_handles_none_input() -> None:
    # graph.py passes raw LLM output, which can be None on some failure
    # paths -- the helper must not raise.
    assert clean_summary_text(None) == ""  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# has_suspicious_trailing_fragment
# ----------------------------------------------------------------------


def test_has_suspicious_trailing_fragment_flags_truncated() -> None:
    assert has_suspicious_trailing_fragment("This was cut off bef.")
    assert has_suspicious_trailing_fragment("Sentence ends mid wo")


def test_has_suspicious_trailing_fragment_allows_known_short_words() -> None:
    # Words on the allowlist (and, the, you, etc.) are common sentence
    # endings -- not truncations.
    assert not has_suspicious_trailing_fragment("Atlas you and the.")
    assert not has_suspicious_trailing_fragment("Stack: aws and gcp.")


def test_has_suspicious_trailing_fragment_allows_capitalized_short_words() -> None:
    # Capitalized short words read as proper nouns/acronyms, not
    # truncations. Only lowercase short fragments are suspicious.
    assert not has_suspicious_trailing_fragment("Sent to AWS.")


# ----------------------------------------------------------------------
# build_notification_fallback
# ----------------------------------------------------------------------


def test_build_notification_fallback_uses_top_connection_when_present() -> None:
    state = {
        "connections_found": ["Vendor X showing churn signals"],
        "action_results": [
            {"tool": "log_interaction", "success": True},
        ],
        "rationale": "ignored when connection is present",
    }
    summary = build_notification_fallback(state)
    assert "Vendor X showing churn signals" in summary
    assert "log interaction" in summary  # underscore->space


def test_build_notification_fallback_falls_back_to_rationale() -> None:
    state = {
        "connections_found": [],
        "action_results": [],
        "rationale": "Pattern detected across emails",
    }
    summary = build_notification_fallback(state)
    assert "Pattern detected across emails" in summary


def test_build_notification_fallback_actions_only() -> None:
    state = {
        "connections_found": [],
        "action_results": [
            {"tool": "generate_draft", "success": True},
        ],
        "rationale": "",
    }
    summary = build_notification_fallback(state)
    assert "Atlas completed follow-up actions" in summary
    assert "generate draft" in summary


def test_build_notification_fallback_default_when_state_empty() -> None:
    summary = build_notification_fallback({})
    assert "Atlas completed reasoning" in summary


# ----------------------------------------------------------------------
# sanitize_notification_summary
# ----------------------------------------------------------------------


def test_sanitize_notification_summary_keeps_useful_sentences() -> None:
    state = {"action_results": [], "connections_found": []}
    text = "Vendor X is at risk. Follow-up has been scheduled for tomorrow."
    summary = sanitize_notification_summary(text, state)
    # Both sentences should survive the sanitizer. (Avoid sentences
    # ending in short lowercase tokens like ``.com``, which the
    # truncation detector flags as likely cutoffs.)
    assert "Vendor X" in summary
    assert "Follow-up" in summary


def test_sanitize_notification_summary_drops_meta_narration() -> None:
    state = {
        "action_results": [],
        "connections_found": ["fallback connection"],
        "rationale": "fallback rationale",
    }
    # Every sentence is meta narration; the helper should fall back to
    # build_notification_fallback rather than emit "Drafting a summary..."
    text = (
        "Drafting a summary of the event.\n"
        "I'm thinking about what to say.\n"
        "The user wants a notification.\n"
    )
    summary = sanitize_notification_summary(text, state)
    assert "Drafting" not in summary
    assert "I'm" not in summary
    assert "The user wants" not in summary
    # Falls back to deterministic content from state.
    assert "fallback connection" in summary


def test_sanitize_notification_summary_truncates_long_output() -> None:
    state = {"action_results": [], "connections_found": []}
    text = "A" * 500 + ". " + "B" * 500 + "."
    summary = sanitize_notification_summary(text, state)
    assert len(summary) <= 320


# ----------------------------------------------------------------------
# plan_actions
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_actions_keeps_safe_high_confidence_actions() -> None:
    state = {
        "recommended_actions": [
            {"tool": "generate_draft", "confidence": 0.9, "params": {"x": 1}},
            {"tool": "log_interaction", "confidence": 0.7, "params": {}},
        ],
    }
    await plan_actions(state)
    assert len(state["planned_actions"]) == 2
    assert state["planned_actions"][0]["tool"] == "generate_draft"
    assert state["planned_actions"][1]["tool"] == "log_interaction"


@pytest.mark.asyncio
async def test_plan_actions_drops_unsafe_tool_names() -> None:
    state = {
        "recommended_actions": [
            {"tool": "delete_contact", "confidence": 0.99, "params": {}},
            {"tool": "send_raw_email", "confidence": 0.95, "params": {}},
        ],
    }
    await plan_actions(state)
    assert state["planned_actions"] == []


@pytest.mark.asyncio
async def test_plan_actions_drops_low_confidence_actions() -> None:
    state = {
        "recommended_actions": [
            {"tool": "generate_draft", "confidence": 0.4, "params": {}},
            {"tool": "log_interaction", "confidence": 0.49, "params": {}},
        ],
    }
    await plan_actions(state)
    assert state["planned_actions"] == []


@pytest.mark.asyncio
async def test_plan_actions_handles_missing_recommended_actions_key() -> None:
    state: dict = {}  # state["recommended_actions"] absent
    await plan_actions(state)
    assert state["planned_actions"] == []


def test_safe_actions_set_contents() -> None:
    # Pin the SAFE_ACTIONS allowlist explicitly so a future PR can't
    # silently widen the set without test acknowledgement.
    assert SAFE_ACTIONS == frozenset({
        "generate_draft",
        "show_slots",
        "log_interaction",
        "create_reminder",
        "send_notification",
    })


# ----------------------------------------------------------------------
# make_chat_messages
# ----------------------------------------------------------------------


def test_make_chat_messages_returns_role_content_pairs() -> None:
    msgs = make_chat_messages("system text", "user text")
    assert msgs == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]


def test_make_chat_messages_returns_fresh_list_each_call() -> None:
    a = make_chat_messages("a", "b")
    b = make_chat_messages("a", "b")
    assert a is not b
    a.append({"role": "user", "content": "extra"})
    assert len(b) == 2  # unaffected


# ----------------------------------------------------------------------
# extract_completion_text
# ----------------------------------------------------------------------


def test_extract_completion_text_canonical_shape() -> None:
    result = {"response": "hello", "usage": {"input_tokens": 5, "output_tokens": 2}}
    text, usage = extract_completion_text(result)
    assert text == "hello"
    assert usage == {"input_tokens": 5, "output_tokens": 2}


def test_extract_completion_text_openai_choices_shape() -> None:
    result = {
        "choices": [{"message": {"content": "from openai"}}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    text, usage = extract_completion_text(result)
    assert text == "from openai"
    assert usage == {"input_tokens": 1, "output_tokens": 1}


def test_extract_completion_text_content_only_shape() -> None:
    text, usage = extract_completion_text({"content": "plain content"})
    assert text == "plain content"
    assert usage == {}


def test_extract_completion_text_handles_non_dict_input() -> None:
    text, usage = extract_completion_text("not a dict")  # type: ignore[arg-type]
    assert text == ""
    assert usage == {}


def test_extract_completion_text_handles_missing_response() -> None:
    text, usage = extract_completion_text({})
    assert text == ""
    assert usage == {}


def test_extract_completion_text_coerces_non_string_response() -> None:
    # Some Provider clients return an integer or model object as
    # ``response``. Coerce to string so callers don't have to.
    text, _ = extract_completion_text({"response": 42})
    assert text == "42"


# ----------------------------------------------------------------------
# complete_with_json
# ----------------------------------------------------------------------


class _FakeJSONClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "metadata": dict(metadata) if metadata else None,
            }
        )
        return {"response": self._response, "usage": {"input_tokens": 3, "output_tokens": 1}}


@pytest.mark.asyncio
async def test_complete_with_json_round_trip() -> None:
    fake = _FakeJSONClient('{"answer": 42}')
    result = await complete_with_json(
        fake, "system", "user", max_tokens=128, temperature=0.2,
    )
    assert result["response"] == '{"answer": 42}'
    assert result["usage"] == {"input_tokens": 3, "output_tokens": 1}
    assert result["parsed"] == {"answer": 42}
    assert result["parse_ok"] is True

    # Metadata carries json_mode + response_format so the atlas adapter
    # can lift them into chat()'s typed kwargs.
    assert fake.calls[0]["metadata"]["json_mode"] is True
    assert fake.calls[0]["metadata"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_complete_with_json_signals_parse_failure() -> None:
    fake = _FakeJSONClient("not valid JSON at all")
    result = await complete_with_json(
        fake, "system", "user", max_tokens=64, temperature=0.0,
    )
    # Raw response is preserved -- caller can fall back to it.
    assert result["response"] == "not valid JSON at all"
    # parsed is empty rather than raising, so call sites can check
    # parse_ok rather than wrap in try/except.
    assert result["parsed"] == {}
    assert result["parse_ok"] is False


@pytest.mark.asyncio
async def test_complete_with_json_distinguishes_empty_object_from_parse_failure() -> None:
    # A model returning a *valid* empty JSON object ``{}`` should be
    # distinguishable from a parse failure -- the graph nodes' "force
    # notify on parse failure" semantics require it. parse_ok is the
    # source of truth, not parsed-truthiness.
    fake = _FakeJSONClient("{}")
    result = await complete_with_json(
        fake, "system", "user", max_tokens=64, temperature=0.0,
    )
    assert result["parsed"] == {}
    assert result["parse_ok"] is True


@pytest.mark.asyncio
async def test_complete_with_json_signals_parse_failure_on_non_object() -> None:
    # parse_llm_json now raises on JSON arrays/scalars; complete_with_json
    # must surface that as parse_ok=False rather than letting the array
    # leak through and crash the caller's ``.get(...)``.
    fake = _FakeJSONClient("[1, 2, 3]")
    result = await complete_with_json(
        fake, "system", "user", max_tokens=64, temperature=0.0,
    )
    assert result["parsed"] == {}
    assert result["parse_ok"] is False


@pytest.mark.asyncio
async def test_complete_with_json_omits_json_mode_when_disabled() -> None:
    fake = _FakeJSONClient("plain text")
    await complete_with_json(
        fake, "system", "user", max_tokens=64, temperature=0.5, json_mode=False,
    )
    assert fake.calls[0]["metadata"] is None  # no metadata when no flags set


@pytest.mark.asyncio
async def test_complete_with_json_threads_timeout_through_metadata() -> None:
    fake = _FakeJSONClient('{"x": 1}')
    await complete_with_json(
        fake, "sys", "user", max_tokens=64, temperature=0.0, timeout=30.0,
    )
    assert fake.calls[0]["metadata"]["timeout"] == 30.0
