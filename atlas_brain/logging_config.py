"""Logging configuration helpers for Atlas Brain."""

from __future__ import annotations

import json
import logging
import math
import traceback as traceback_module
from collections.abc import Mapping, Sequence, Set
from datetime import datetime, timezone
from typing import Any


TEXT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_STANDARD_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class AtlasJsonFormatter(logging.Formatter):
    """Format log records as stable JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = _exception_from_info(
                record.exc_info,
                self.formatException(record.exc_info),
            )
        elif record.exc_text:
            payload["exception"] = _exception_from_text(record.exc_text)

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        extra: dict[str, Any] = {}
        for key, value in sorted(record.__dict__.items()):
            if key in _STANDARD_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            extra[key] = _json_safe(value)
        if extra:
            payload["extra"] = extra

        return json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def build_log_formatter(log_format: str) -> logging.Formatter:
    normalized = str(log_format or "text").strip().lower()
    if normalized == "text":
        return logging.Formatter(TEXT_LOG_FORMAT)
    if normalized == "json":
        return AtlasJsonFormatter()
    raise ValueError("ATLAS_LOG_FORMAT must be one of: text, json")


def configure_logging(*, level: str, log_format: str) -> None:
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    formatter = build_log_formatter(log_format)
    normalized = str(log_format or "text").strip().lower()

    if normalized == "text":
        logging.basicConfig(level=resolved_level, format=TEXT_LOG_FORMAT)
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler())
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(logger_name).handlers:
            handler.setFormatter(formatter)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
            "traceback": "".join(
                traceback_module.format_exception(type(value), value, value.__traceback__)
            ),
        }
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
    return value


def _exception_from_info(
    exc_info: tuple[type[BaseException], BaseException, Any] | tuple[None, None, None],
    traceback_text: str,
) -> dict[str, str]:
    exc_type, exc_value, _traceback = exc_info
    return {
        "type": getattr(exc_type, "__name__", str(exc_type)),
        "message": str(exc_value),
        "traceback": traceback_text,
    }


def _exception_from_text(traceback_text: str) -> dict[str, str]:
    lines = [line for line in traceback_text.strip().splitlines() if line]
    return {
        "type": "unknown",
        "message": lines[-1] if lines else "",
        "traceback": traceback_text,
    }
