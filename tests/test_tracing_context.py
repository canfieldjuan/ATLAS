from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.reasoning.agent import ReasoningAgentGraph
from atlas_brain.reasoning.events import AtlasEvent
from atlas_brain.services.tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)


def test_build_business_trace_context_filters_empty_values():
    context = build_business_trace_context(
        account_id=" acct_123 ",
        workflow="tenant_report",
        vendor_name="",
        company_name=None,
    )

    assert context == {
        "account_id": "acct_123",
        "workflow": "tenant_report",
    }


def test_build_reasoning_trace_context_respects_raw_reasoning_flag():
    with patch.object(tracer, "_enabled", False), patch(
        "atlas_brain.services.tracing.settings.ftl_tracing.capture_raw_reasoning",
        False,
    ):
        context = build_reasoning_trace_context(
            decision={"action_type": "conversation"},
            rationale="Short summary",
            raw_reasoning="internal scratchpad",
        )

    assert context["summary"] == "Short summary"
    assert "raw_preview" not in context


@pytest.mark.asyncio
async def test_reasoning_agent_emits_trace_with_reasoning_summary():
    captured = []
    event = AtlasEvent(
        id=uuid4(),
        event_type="b2b.intelligence_generated",
        source="b2b_churn_intelligence",
        entity_type="vendor",
        entity_id="Acme",
        payload={"vendor_name": "Acme"},
    )
    result_state = {
        "triage_priority": "high",
        "needs_reasoning": True,
        "queued": False,
        "connections_found": ["pricing pressure"],
        "planned_actions": [{"tool": "send_notification", "confidence": 0.9}],
        "action_results": [{"tool": "send_notification", "success": True}],
        "notification_sent": True,
        "summary": "Acme needs follow-up",
        "triage_reasoning": "High-signal churn event",
        "rationale": "Pricing pressure and negative sentiment are increasing",
        "reasoning_output": "Longer reasoning output",
    }

    with (
        patch.object(tracer, "_enabled", True),
        patch.object(tracer, "_dispatch", side_effect=captured.append),
        patch("atlas_brain.reasoning.graph.run_reasoning_graph", new=AsyncMock(return_value=result_state)),
    ):
        graph = ReasoningAgentGraph()
        result = await graph.process_event(event)

    assert result["status"] == "completed"
    assert captured

    payload = captured[-1]
    assert payload["span_name"] == "reasoning.process"
    assert payload["reasoning"] == "Pricing pressure and negative sentiment are increasing"
    assert payload["metadata"]["business"]["workflow"] == "reasoning_agent"
    assert payload["metadata"]["reasoning"]["triage"] == "High-signal churn event"
    assert payload["metadata"]["reasoning"]["summary"] == "Pricing pressure and negative sentiment are increasing"
