from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_visibility import JsonlVisibilitySink


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/read_extracted_campaign_visibility.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "read_extracted_campaign_visibility",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clock() -> datetime:
    return datetime(2026, 5, 5, 20, 0, tzinfo=timezone.utc)


async def _write_events(path: Path) -> None:
    sink = JsonlVisibilitySink(path, clock=_clock)
    await sink.emit(
        "campaign_operation_started",
        {"operation": "draft_generation", "limit": 2},
    )
    await sink.emit(
        "campaign_operation_failed",
        {
            "operation": "send_queued",
            "error_type": "reported_failures",
            "result": {"failed": 1, "sent": 0},
        },
    )
    await sink.emit(
        "campaign_operation_completed",
        {
            "operation": "send_queued",
            "result": {"failed": 0, "sent": 2},
        },
    )


@pytest.mark.asyncio
async def test_visibility_reader_cli_filters_operation_and_emits_json(
    tmp_path,
    capsys,
) -> None:
    cli = _load_cli_module()
    path = tmp_path / "events.jsonl"
    await _write_events(path)

    exit_code = cli.main([
        str(path),
        "--operation",
        "send_queued",
        "--json",
    ])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["count"] == 2
    assert [row["event_type"] for row in output["events"]] == [
        "campaign_operation_failed",
        "campaign_operation_completed",
    ]


@pytest.mark.asyncio
async def test_visibility_reader_cli_filters_event_type_and_limit(
    tmp_path,
    capsys,
) -> None:
    cli = _load_cli_module()
    path = tmp_path / "events.jsonl"
    await _write_events(path)

    exit_code = cli.main([
        str(path),
        "--event-type",
        "campaign_operation_completed",
        "--limit",
        "1",
    ])

    output = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert "campaign_operation_completed" in output
    assert "operation=send_queued" in output
    assert 'result={"failed": 0, "sent": 2}' in output


@pytest.mark.asyncio
async def test_visibility_reader_cli_writes_output_file(tmp_path, capsys) -> None:
    cli = _load_cli_module()
    path = tmp_path / "events.jsonl"
    output_path = tmp_path / "events-summary.json"
    await _write_events(path)

    exit_code = cli.main([
        str(path),
        "--event-type",
        "campaign_operation_failed",
        "--json",
        "--output",
        str(output_path),
    ])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["count"] == 1
    assert output["events"][0]["payload"]["error_type"] == "reported_failures"


def test_visibility_reader_cli_rejects_non_positive_limit(tmp_path) -> None:
    cli = _load_cli_module()
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="Invalid --limit"):
        cli.main([str(path), "--limit", "0"])
