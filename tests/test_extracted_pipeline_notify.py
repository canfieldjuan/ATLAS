from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from extracted_content_pipeline.pipelines.notify import (
    configure_pipeline_notification_sink,
    get_pipeline_notification_sink,
    send_pipeline_notification,
)


@dataclass
class _Task:
    name: str = "b2b_campaign_generation"
    id: str = "task-1"
    metadata: dict[str, Any] = field(default_factory=dict)


class _Sink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[dict[str, Any]] = []

    async def emit(self, event_type, payload):
        if self.fail:
            raise RuntimeError("visibility down")
        self.events.append({"event_type": event_type, "payload": payload})


@pytest.fixture(autouse=True)
def _reset_sink():
    previous = configure_pipeline_notification_sink(None)
    yield
    configure_pipeline_notification_sink(previous)


@pytest.mark.asyncio
async def test_notification_is_noop_without_configured_sink():
    await send_pipeline_notification("hello", _Task())

    assert get_pipeline_notification_sink() is None


@pytest.mark.asyncio
async def test_notification_emits_to_configured_visibility_sink():
    sink = _Sink()
    configure_pipeline_notification_sink(sink)

    await send_pipeline_notification(
        "generated campaigns",
        _Task(metadata={"notify_priority": "high", "notify_tags": "sales,campaign"}),
        max_chars=9,
    )

    assert sink.events == [{
        "event_type": "pipeline_notification",
        "payload": {
            "message": "generated",
            "title": "Atlas: B2B Campaign Generation",
            "task_name": "b2b_campaign_generation",
            "task_id": "task-1",
            "priority": "high",
            "tags": "sales,campaign",
            "markdown": True,
            "metadata": {
                "source": "extracted_content_pipeline",
                "parsed": False,
            },
        },
    }]


@pytest.mark.asyncio
async def test_notification_can_use_call_scoped_visibility_sink():
    global_sink = _Sink()
    scoped_sink = _Sink()
    configure_pipeline_notification_sink(global_sink)

    await send_pipeline_notification("hello", _Task(), visibility=scoped_sink)

    assert global_sink.events == []
    assert scoped_sink.events[0]["payload"]["message"] == "hello"


@pytest.mark.asyncio
async def test_notification_respects_task_notify_opt_out():
    sink = _Sink()
    configure_pipeline_notification_sink(sink)

    await send_pipeline_notification("hello", _Task(metadata={"notify": False}))

    assert sink.events == []


@pytest.mark.asyncio
async def test_notification_formats_structured_payload_fields():
    sink = _Sink()
    configure_pipeline_notification_sink(sink)

    await send_pipeline_notification(
        "fallback",
        _Task(name="competitive_intelligence"),
        title="Custom title",
        parsed={
            "analysis_text": "Summary line.",
            "top_pain_points": [{
                "asin": "B001",
                "primary_issue": "Battery complaints",
                "avg_pain_score": 8,
            }],
            "competitive_flows": [{
                "from_brand": "A",
                "to_brand": "B",
                "count": 4,
                "primary_reason": "price",
            }],
            "recommendations": [{"action": "Inspect pricing", "urgency": "high"}],
        },
    )

    payload = sink.events[0]["payload"]
    assert payload["title"] == "Custom title"
    assert payload["metadata"]["parsed"] is True
    assert "Summary line." in payload["message"]
    assert "**Top Pain Points**" in payload["message"]
    assert "- **B001**: Battery complaints (pain: 8)" in payload["message"]
    assert "**Competitive Flows**" in payload["message"]
    assert "- A -> B (4 mentions): price" in payload["message"]
    assert "**Recommendations**" in payload["message"]


@pytest.mark.asyncio
async def test_notification_sink_failures_do_not_break_pipeline():
    configure_pipeline_notification_sink(_Sink(fail=True))

    await send_pipeline_notification("hello", _Task())
