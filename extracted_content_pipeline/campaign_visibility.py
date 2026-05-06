"""Reference visibility sinks for host-mounted campaign operations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .campaign_ports import VisibilitySink


Clock = Callable[[], datetime]
OPERATION_STARTED_EVENT = "campaign_operation_started"
OPERATION_COMPLETED_EVENT = "campaign_operation_completed"
OPERATION_FAILED_EVENT = "campaign_operation_failed"
REPORTED_ERROR_TYPE = "reported_error"
REPORTED_FAILURES_ERROR_TYPE = "reported_failures"


@dataclass(frozen=True)
class VisibilityEvent:
    """A host-visible campaign operation or pipeline event."""

    event_type: str
    payload: Mapping[str, Any]
    emitted_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "emitted_at": self.emitted_at.isoformat(),
        }


class InMemoryVisibilitySink:
    """Visibility sink for tests, local demos, and lightweight hosts."""

    def __init__(
        self,
        *,
        max_events: int | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._max_events = max_events
        self._clock = clock or _utcnow
        self._events: list[VisibilityEvent] = []

    @property
    def events(self) -> tuple[VisibilityEvent, ...]:
        return tuple(self._events)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self._events]

    async def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self._events.append(
            VisibilityEvent(
                event_type=str(event_type),
                payload=dict(payload),
                emitted_at=self._clock(),
            )
        )
        if self._max_events is not None:
            self._events = self._events[-max(0, int(self._max_events)) :]


class JsonlVisibilitySink:
    """Append visibility events to a host-owned JSONL file."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Clock | None = None,
        ensure_parent: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self._clock = clock or _utcnow
        self._ensure_parent = ensure_parent
        self._encoding = encoding

    async def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = VisibilityEvent(
            event_type=str(event_type),
            payload=dict(payload),
            emitted_at=self._clock(),
        )
        if self._ensure_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.as_dict(), default=_json_default, sort_keys=True)
        with self.path.open("a", encoding=self._encoding) as handle:
            handle.write(line + "\n")


def read_jsonl_visibility_events(
    path: str | Path,
    *,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> list[dict[str, Any]]:
    """Read visibility events written by ``JsonlVisibilitySink``."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding=encoding) as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    if limit is None:
        return rows
    return rows[-max(0, int(limit)) :]


def visibility_result_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact visibility-safe summary for operation result payloads."""

    summary: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (bool, int, float)) or value is None:
            summary[key] = value
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            if key == "errors":
                summary["error_count"] = len(value)
            elif key.endswith("_ids"):
                summary[f"{key}_count"] = len(value)
    return summary


async def emit_operation_event(
    visibility: VisibilitySink | None,
    event_type: str,
    operation: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    """Best effort operation event emission for host-facing runners."""

    if visibility is None:
        return
    event_payload = {"operation": str(operation)}
    event_payload.update(dict(payload or {}))
    try:
        await visibility.emit(event_type, event_payload)
    except Exception:
        return


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = [
    "OPERATION_COMPLETED_EVENT",
    "OPERATION_FAILED_EVENT",
    "OPERATION_STARTED_EVENT",
    "REPORTED_ERROR_TYPE",
    "REPORTED_FAILURES_ERROR_TYPE",
    "InMemoryVisibilitySink",
    "JsonlVisibilitySink",
    "VisibilityEvent",
    "emit_operation_event",
    "read_jsonl_visibility_events",
    "visibility_result_summary",
]
