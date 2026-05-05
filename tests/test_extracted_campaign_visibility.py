from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_visibility import (
    InMemoryVisibilitySink,
    JsonlVisibilitySink,
    read_jsonl_visibility_events,
)


def _clock() -> datetime:
    return datetime(2026, 5, 5, 19, 20, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_in_memory_visibility_sink_records_events() -> None:
    sink = InMemoryVisibilitySink(clock=_clock)

    await sink.emit("campaign_operation_started", {"operation": "send_queued"})

    assert len(sink.events) == 1
    assert sink.events[0].event_type == "campaign_operation_started"
    assert sink.events[0].payload == {"operation": "send_queued"}
    assert sink.as_dicts() == [
        {
            "event_type": "campaign_operation_started",
            "payload": {"operation": "send_queued"},
            "emitted_at": "2026-05-05T19:20:00+00:00",
        }
    ]


@pytest.mark.asyncio
async def test_in_memory_visibility_sink_respects_max_events() -> None:
    sink = InMemoryVisibilitySink(max_events=2, clock=_clock)

    await sink.emit("one", {"index": 1})
    await sink.emit("two", {"index": 2})
    await sink.emit("three", {"index": 3})

    assert [event.event_type for event in sink.events] == ["two", "three"]


@pytest.mark.asyncio
async def test_jsonl_visibility_sink_appends_events(tmp_path: Path) -> None:
    path = tmp_path / "visibility" / "events.jsonl"
    sink = JsonlVisibilitySink(path, clock=_clock)

    await sink.emit(
        "campaign_operation_failed",
        {
            "operation": "draft_generation",
            "error_type": "reported_failures",
            "result": {"error_count": 1},
        },
    )

    assert read_jsonl_visibility_events(path) == [
        {
            "emitted_at": "2026-05-05T19:20:00+00:00",
            "event_type": "campaign_operation_failed",
            "payload": {
                "error_type": "reported_failures",
                "operation": "draft_generation",
                "result": {"error_count": 1},
            },
        }
    ]


@pytest.mark.asyncio
async def test_jsonl_visibility_sink_serializes_non_json_values(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonlVisibilitySink(path, clock=_clock)

    await sink.emit("pipeline_notification", {"path": Path("artifact.json")})

    rows = read_jsonl_visibility_events(path)

    assert rows[0]["payload"] == {"path": "artifact.json"}


@pytest.mark.asyncio
async def test_read_jsonl_visibility_events_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = JsonlVisibilitySink(path, clock=_clock)

    await sink.emit("one", {"index": 1})
    await sink.emit("two", {"index": 2})
    await sink.emit("three", {"index": 3})

    assert [
        row["event_type"]
        for row in read_jsonl_visibility_events(path, limit=2)
    ] == ["two", "three"]
