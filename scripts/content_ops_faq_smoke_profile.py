"""Shared input profiling helpers for FAQ smoke scripts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
from typing import Any

from extracted_content_pipeline.campaign_customer_data import (
    CampaignOpportunityLoadResult,
)
from extracted_content_pipeline.campaign_source_adapters import (
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)


JSON_ROW_KEYS = (
    "support_tickets",
    "tickets",
    "cases",
    "conversations",
    "complaints",
    "reviews",
    "feedback",
    "sources",
    "rows",
    "data",
)
SKIP_WARNING_CODES = {"empty_row", "missing_source_text", "row_not_object"}


def empty_input_profile(*, status: str = "ok") -> dict[str, Any]:
    return {
        "status": status,
        "raw_row_count": None,
        "raw_row_count_source": None,
        "usable_source_count": None,
        "warning_count": None,
        "warnings_by_code": {},
        "skipped_row_count": None,
        "missing_source_text_count": None,
        "warning_sample": [],
    }


def load_source_input_profile(
    path: str | Path,
    *,
    source_format: str,
    max_text_chars: int,
    default_field: Sequence[str] | None = None,
) -> dict[str, Any]:
    profile = empty_input_profile()
    profile.update(raw_row_profile_or_error(Path(path), source_format))
    try:
        loaded = load_source_campaign_opportunities_from_file(
            path,
            file_format=source_format,
            max_text_chars=max_text_chars,
            default_fields=parse_default_fields_or_exit(default_field or []),
        )
    except (Exception, SystemExit) as exc:
        return input_profile_error(exc, raw_profile=profile)
    return input_profile_from_loaded(loaded, raw_profile=profile)


def input_profile_error(
    exc: BaseException,
    *,
    raw_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = empty_input_profile(status="error")
    if raw_profile:
        profile.update(dict(raw_profile))
        profile["status"] = "error"
    profile["error"] = f"{type(exc).__name__}: {exc}"
    return profile


def input_profile_from_loaded(
    loaded: CampaignOpportunityLoadResult,
    *,
    raw_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = empty_input_profile()
    if raw_profile:
        profile.update(dict(raw_profile))
    warnings = loaded.warning_dicts()
    warnings_by_code = Counter(str(warning.get("code") or "unknown") for warning in warnings)
    skipped_rows = {
        int(warning["row_index"])
        for warning in warnings
        if warning.get("code") in SKIP_WARNING_CODES
        and isinstance(warning.get("row_index"), int)
    }
    raw_count = profile.get("raw_row_count")
    usable_count = len(loaded.opportunities)
    profile.update({
        "usable_source_count": usable_count,
        "warning_count": len(warnings),
        "warnings_by_code": dict(sorted(warnings_by_code.items())),
        "skipped_row_count": len(skipped_rows),
        "missing_source_text_count": warnings_by_code.get("missing_source_text", 0),
        "warning_sample": warnings[:10],
    })
    if isinstance(raw_count, int) and raw_count > 0:
        profile["usable_source_ratio"] = round(usable_count / raw_count, 6)
    return profile


def raw_row_profile_or_error(path: Path, source_format: str) -> dict[str, Any]:
    try:
        return raw_row_profile(path, source_format)
    except Exception as exc:  # pragma: no cover - exact host filesystem errors vary.
        return {
            "raw_row_count": None,
            "raw_row_count_source": None,
            "raw_row_count_error": f"{type(exc).__name__}: {exc}",
        }


def raw_row_profile(path: Path, source_format: str) -> dict[str, Any]:
    resolved = resolve_source_format(path, source_format)
    if resolved == "csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return {
                "raw_row_count": sum(1 for _row in csv.DictReader(handle)),
                "raw_row_count_source": "csv_rows",
            }
    if resolved == "jsonl":
        return {
            "raw_row_count": sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            "raw_row_count_source": "jsonl_lines",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"raw_row_count": len(data), "raw_row_count_source": "json_array"}
    if isinstance(data, Mapping):
        bundle_counts = []
        for key in JSON_ROW_KEYS:
            value = data.get(key)
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                bundle_counts.append((key, len(value)))
        if bundle_counts:
            keys = ",".join(key for key, _count in bundle_counts)
            return {
                "raw_row_count": sum(count for _key, count in bundle_counts),
                "raw_row_count_source": f"json_bundle.{keys}",
            }
    return {"raw_row_count": None, "raw_row_count_source": None}


def resolve_source_format(path: Path, source_format: str) -> str:
    if source_format != "auto":
        return source_format
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "json"


def console_input_profile(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "source_rows=unknown"
    parts = [f"input_status={value.get('status') or 'unknown'}"]
    usable = value.get("usable_source_count")
    raw = value.get("raw_row_count")
    if usable is not None or raw is not None:
        parts.append(f"source_rows={_console_value(usable)}/{_console_value(raw)}")
    for key, label in (
        ("skipped_row_count", "skipped_rows"),
        ("missing_source_text_count", "missing_source_text"),
        ("warning_count", "warnings"),
    ):
        item = value.get(key)
        if item not in (None, 0):
            parts.append(f"{label}={item}")
    return " ".join(parts)


def _console_value(value: Any) -> str:
    return str(value) if value is not None else "unknown"
