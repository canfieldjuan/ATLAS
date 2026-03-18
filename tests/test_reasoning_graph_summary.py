from unittest.mock import AsyncMock

import pytest

from atlas_brain.reasoning.graph import (
    _build_notification_fallback,
    _node_synthesize,
    _sanitize_notification_summary,
)


def test_sanitize_notification_summary_removes_meta_narration():
    text = (
        "**Summarizing Atlas's results**\n\n"
        "We alerted you about LARKI's high interest in AWS reliability.\n\n"
        "**Updating contact details**\n\n"
        "I should remind myself to update the contact details and reach out accordingly."
    )
    state = {"rationale": "Reliability concerns are rising.", "action_results": []}

    summary = _sanitize_notification_summary(text, state)

    assert "Summarizing" not in summary
    assert "Updating contact details" not in summary
    assert "I should remind myself" not in summary
    assert summary == "We alerted you about LARKI's high interest in AWS reliability."


def test_build_notification_fallback_uses_rationale_and_actions():
    state = {
        "rationale": "LARKI raised repeated AWS reliability concerns.",
        "action_results": [
            {"tool": "send_notification", "success": True},
            {"tool": "create_reminder", "success": True},
        ],
    }

    summary = _build_notification_fallback(state)

    assert "LARKI raised repeated AWS reliability concerns." in summary
    assert "send notification" in summary
    assert "create reminder" in summary


def test_build_notification_fallback_prefers_connections():
    state = {
        "connections_found": [
            "This is the third high-intent churn signal from LARKI about AWS reliability"
        ],
        "action_results": [{"tool": "send_notification", "success": True}],
        "rationale": "Generic rationale",
    }

    summary = _build_notification_fallback(state)

    assert "third high-intent churn signal" in summary
    assert "send notification" in summary


@pytest.mark.asyncio
async def test_node_synthesize_sanitizes_graph_summary(monkeypatch):
    monkeypatch.setattr(
        "atlas_brain.reasoning.graph._resolve_graph_llm",
        lambda workload: object(),
    )
    monkeypatch.setattr(
        "atlas_brain.reasoning.graph._llm_generate",
        AsyncMock(
            return_value={
                "response": (
                    "**Summarizing Atlas's results**\n\n"
                    "We alerted you about LARKI's reliability concerns and queued a reminder.\n\n"
                    "I'm thinking the owner should follow up soon."
                ),
                "usage": {"input_tokens": 5, "output_tokens": 7},
            }
        ),
    )

    state = {
        "should_notify": True,
        "event_type": "b2b.high_intent_detected",
        "action_results": [{"tool": "send_notification", "success": True}],
        "rationale": "LARKI raised repeated AWS reliability concerns.",
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    result = await _node_synthesize(state)

    assert result["summary"] == "We alerted you about LARKI's reliability concerns and queued a reminder."
    assert result["total_input_tokens"] == 5
    assert result["total_output_tokens"] == 7


def test_sanitize_notification_summary_rejects_task_talk():
    text = "The user wants a clear summary of the push notification's details. The event is about detecting high intent in B2B."
    state = {
        "connections_found": ["Repeated high-intent churn signals point to AWS reliability risk."],
        "action_results": [{"tool": "send_notification", "success": True}],
        "rationale": "Generic rationale",
    }

    summary = _sanitize_notification_summary(text, state)

    assert "The user wants" not in summary
    assert "push notification" not in summary
    assert "Repeated high-intent churn signals" in summary


def test_sanitize_notification_summary_drops_truncated_tail_sentence():
    text = (
        "LARKI has repeatedly signaled high-urgency reliability pain with AWS and is actively evaluating GCP. "
        "Immediate notification, CRM logging, and proactive outreach will help retain the account and address their concerns bef."
    )
    state = {
        "connections_found": [],
        "action_results": [{"tool": "send_notification", "success": True}],
        "rationale": "Generic rationale",
    }

    summary = _sanitize_notification_summary(text, state)

    assert summary == (
        "LARKI has repeatedly signaled high-urgency reliability pain with AWS and is actively evaluating GCP."
    )


def test_sanitize_notification_summary_rejects_generic_filler():
    text = "Assessing churn risk."
    state = {
        "connections_found": ["Repeated high-intent churn signals from LARKI indicate an escalating issue."],
        "action_results": [{"tool": "send_notification", "success": True}],
        "rationale": "Generic rationale",
    }

    summary = _sanitize_notification_summary(text, state)

    assert summary == (
        "Repeated high-intent churn signals from LARKI indicate an escalating issue. "
        "Actions completed: send notification."
    )
