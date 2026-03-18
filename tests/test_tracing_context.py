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


@pytest.mark.asyncio
async def test_store_local_uses_standalone_connection_when_shared_pool_disabled():
    raw_conn = AsyncMock()
    pool = AsyncMock()
    pool.is_initialized = True
    pool.acquire_raw = AsyncMock(return_value=raw_conn)
    pool.execute = AsyncMock()
    payload = {
        "span_name": "pipeline.digest/battle_card_sales_copy",
        "input_tokens": 100,
        "output_tokens": 50,
        "metadata": {"vendor": "WooCommerce"},
    }

    with patch("atlas_brain.storage.database.get_db_pool", return_value=pool):
        await tracer._store_local(payload, use_shared_pool=False)

    pool.execute.assert_not_awaited()
    pool.acquire_raw.assert_awaited_once()
    raw_conn.execute.assert_awaited_once()
    raw_conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_uses_ephemeral_client_when_shared_resources_disabled():
    client = AsyncMock()
    client.post = AsyncMock(return_value=type("Resp", (), {"status_code": 200})())
    client_cm = AsyncMock()
    client_cm.__aenter__.return_value = client
    client_cm.__aexit__.return_value = None
    payload = {
        "span_id": "span_123",
        "span_name": "pipeline.digest/battle_card_sales_copy",
        "input_tokens": 100,
        "output_tokens": 50,
        "status": "completed",
    }

    with (
        patch.object(tracer, "_enabled", True),
        patch.object(tracer, "_base_url", "https://example.com"),
        patch.object(tracer, "_api_key", "test-key"),
        patch.object(tracer, "_store_local", new=AsyncMock()),
        patch.object(tracer, "_ensure_client", new=AsyncMock()) as ensure_client,
        patch("atlas_brain.services.tracing.httpx.AsyncClient", return_value=client_cm),
    ):
        await tracer._send(payload, use_shared_resources=False)

    ensure_client.assert_not_called()
    client.post.assert_awaited_once()
