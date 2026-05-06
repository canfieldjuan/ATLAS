#!/usr/bin/env python3
"""Read campaign operation visibility events from a JSONL audit trail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_visibility import (  # noqa: E402
    read_jsonl_visibility_events,
)


DEFAULT_LIMIT = 20


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read campaign operation visibility events from a JSONL file."
    )
    parser.add_argument("path", type=Path, help="Visibility JSONL file to read.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum matching events to return.",
    )
    parser.add_argument("--operation", help="Filter by payload.operation.")
    parser.add_argument("--event-type", help="Filter by event_type.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of concise text lines.",
    )
    parser.add_argument("--output", type=Path, help="Optional output path.")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.limit) <= 0:
        raise SystemExit("Invalid --limit: must be greater than 0")


def _event_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, dict) else {}


def _matches_event(
    row: dict[str, Any],
    *,
    operation: str | None,
    event_type: str | None,
) -> bool:
    if event_type and row.get("event_type") != event_type:
        return False
    payload = _event_payload(row)
    if operation and payload.get("operation") != operation:
        return False
    return True


def _filtered_events(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl_visibility_events(args.path)
    matched = [
        row
        for row in rows
        if _matches_event(
            row,
            operation=args.operation,
            event_type=args.event_type,
        )
    ]
    return matched[-int(args.limit) :]


def _format_event(row: dict[str, Any]) -> str:
    payload = _event_payload(row)
    parts = [
        str(row.get("emitted_at") or ""),
        str(row.get("event_type") or ""),
        f"operation={payload.get('operation') or '-'}",
    ]
    if payload.get("error_type"):
        parts.append(f"error_type={payload['error_type']}")
    result = payload.get("result")
    if isinstance(result, dict):
        parts.append(f"result={json.dumps(result, sort_keys=True)}")
    return " ".join(part for part in parts if part)


def _render_output(rows: list[dict[str, Any]], *, as_json: bool) -> str:
    if as_json:
        return json.dumps(
            {"count": len(rows), "events": rows},
            indent=2,
            sort_keys=True,
        )
    return "\n".join(_format_event(row) for row in rows)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    rows = _filtered_events(args)
    output = _render_output(rows, as_json=bool(args.json))
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    elif output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
