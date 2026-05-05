"""Reference visibility sinks for host-mounted campaign operations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Clock = Callable[[], datetime]


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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = [
    "InMemoryVisibilitySink",
    "JsonlVisibilitySink",
    "VisibilityEvent",
    "read_jsonl_visibility_events",
]
