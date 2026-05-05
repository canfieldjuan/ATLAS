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
    has_suspicious_trailing_fragment,
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
