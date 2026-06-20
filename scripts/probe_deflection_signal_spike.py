#!/usr/bin/env python3
"""Probe deflection report S1 signal availability from a local ticket export."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    source_material_to_source_rows,
)
from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    scrub_deflection_report_payload,
)
from extracted_content_pipeline.support_ticket_input_package import (  # noqa: E402
    build_support_ticket_input_package,
)
from extracted_content_pipeline.support_ticket_zendesk_thread import (  # noqa: E402
    rows_from_zendesk_full_thread,
)


SCHEMA_VERSION = "deflection_signal_spike.v1"
_READINESS_READY = "ready"
_READINESS_PARTIAL = "partial"
_READINESS_INSUFFICIENT = "insufficient_data"

_RESOLUTION_KEYS = frozenset({
    "resolution",
    "resolutiontext",
    "resolutionnotes",
    "resolvednotes",
    "solution",
    "solutiontext",
    "answer",
    "answertext",
    "supportanswer",
    "supportresponse",
    "supportreply",
    "agentanswer",
    "agentresponse",
    "agentreply",
    "adminreply",
    "latestagentreply",
    "lastagentreply",
    "publicagentreply",
    "staffreply",
    "fixsummary",
    "workaround",
})
_STATUS_KEYS = frozenset({
    "ticketstatus",
    "issuestatus",
    "casestatus",
    "status",
    "ticketstate",
    "state",
})
_CSAT_KEYS = frozenset({
    "csat",
    "csatscore",
    "satisfactionscore",
    "satisfactionrating",
    "customersatisfactionrating",
    "customersatisfactionscore",
    "customersatisfaction",
    "satisfaction",
    "rating",
})
_STRUCTURED_CONTEXT_KEYS = frozenset({
    "product",
    "productname",
    "subproduct",
    "subproductname",
    "issue",
    "issuetype",
    "subissue",
    "subissuetype",
    "category",
    "paincategory",
    "intent",
    "topic",
})
_SOURCE_COST_KEYS = frozenset({
    "agentcost",
    "assistedcontactcost",
    "cost",
    "costusd",
    "handleminutes",
    "handletime",
    "loadedcost",
    "supportcost",
    "supportcostusd",
    "timespent",
    "timespentminutes",
})
_TEXT_KEYS_TO_SCRUB = ("source_title", "text", "resolution_text")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = probe_signal_availability(
        args.path,
        source_format=args.source_format,
        zendesk_thread=args.zendesk_thread,
        max_rows=args.max_rows,
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.require_s1_ready:
        insufficient = _insufficient_dependencies(summary)
        if insufficient:
            print(
                "Signal spike has insufficient S1 dependencies: "
                + ", ".join(insufficient),
                file=sys.stderr,
            )
            return 1
    return 0


def probe_signal_availability(
    path: Path,
    *,
    source_format: str = "auto",
    zendesk_thread: str = "auto",
    max_rows: int = 1000,
) -> dict[str, Any]:
    material, input_format = _load_source_material(path, source_format=source_format)
    rows, import_warning_codes, used_zendesk_import = _source_rows(
        material,
        zendesk_thread=zendesk_thread,
    )
    package = build_support_ticket_input_package(
        rows,
        provider="deflection_signal_spike",
        outputs=("faq_deflection_report",),
        max_rows=max_rows,
    )
    normalized_rows = _normalized_rows(package)
    included_count = int(package.metadata.get("included_row_count") or 0)
    raw_rows = [row for row in rows if isinstance(row, Mapping)]

    resolution = _resolution_signal(package, included_count)
    status = _status_signal(package, included_count)
    csat = _csat_signal(package, included_count)
    cost = _cost_signal(raw_rows, included_count)
    owner_lane = _owner_lane_signal(normalized_rows, included_count)
    snippet_safety = _snippet_safety_signal(normalized_rows, included_count)
    readiness = {
        "support_resolution_evidence": {
            "status": resolution["readiness"],
            "reason": resolution["reason"],
        },
        "csat_prioritization": {
            "status": csat["readiness"],
            "reason": csat["reason"],
        },
        "cost_basis": {
            "status": cost["readiness"],
            "reason": cost["reason"],
        },
        "owner_lane_fix_type": {
            "status": owner_lane["readiness"],
            "reason": owner_lane["reason"],
        },
        "snippet_projection_safety": {
            "status": snippet_safety["readiness"],
            "reason": snippet_safety["reason"],
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "source_format": input_format,
            "zendesk_thread_import": used_zendesk_import,
            "source_path_recorded": False,
            "raw_source_values_recorded": False,
        },
        "rows": {
            "source_row_count": int(package.metadata.get("source_row_count") or 0),
            "included_row_count": included_count,
            "skipped_row_count": int(package.metadata.get("skipped_row_count") or 0),
            "truncated_row_count": int(package.metadata.get("truncated_row_count") or 0),
        },
        "warnings": {
            "import_warning_codes": import_warning_codes,
            "package_warning_codes": _package_warning_codes(package.warnings),
        },
        "signals": {
            "support_resolution_evidence": resolution,
            "ticket_status": status,
            "csat": csat,
            "cost_basis": cost,
            "owner_lane_fix_type": owner_lane,
            "snippet_projection_safety": snippet_safety,
        },
        "s1_readiness": readiness,
        "safety": {
            "summary_only": True,
            "raw_source_values_recorded": False,
            "source_ids_recorded": False,
            "snippets_recorded": False,
        },
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local CSV, JSON, or JSONL export.")
    parser.add_argument(
        "--source-format",
        choices=("auto", "csv", "json", "jsonl"),
        default="auto",
    )
    parser.add_argument(
        "--zendesk-thread",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Parse JSON as Zendesk full-thread export before support-ticket packaging.",
    )
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-s1-ready",
        action="store_true",
        help="Exit non-zero when any S1 dependency remains insufficient_data.",
    )
    return parser.parse_args(argv)


def _load_source_material(path: Path, *, source_format: str) -> tuple[Any, str]:
    resolved = _resolve_format(path, source_format)
    if resolved == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle)), resolved
    if resolved == "jsonl":
        rows: list[Any] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"malformed JSONL at line {line_number}: {exc.msg}"
                ) from exc
        return rows, resolved
    try:
        return json.loads(path.read_text(encoding="utf-8")), resolved
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed JSON: {exc.msg}") from exc


def _resolve_format(path: Path, source_format: str) -> str:
    if source_format != "auto":
        return source_format
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "json"


def _source_rows(
    material: Any,
    *,
    zendesk_thread: str,
) -> tuple[list[Any], dict[str, int], bool]:
    use_zendesk = (
        zendesk_thread == "yes"
        or (zendesk_thread == "auto" and _looks_like_zendesk_thread(material))
    )
    if use_zendesk:
        imported = rows_from_zendesk_full_thread(material)
        return list(imported.rows), _warning_code_counts(imported.warnings), True
    return source_material_to_source_rows(material), {}, False


def _looks_like_zendesk_thread(material: Any) -> bool:
    entries: Sequence[Any]
    if isinstance(material, Mapping):
        tickets = material.get("tickets")
        if not isinstance(tickets, Sequence) or isinstance(tickets, (str, bytes, bytearray)):
            return False
        entries = tickets
    elif isinstance(material, Sequence) and not isinstance(material, (str, bytes, bytearray)):
        entries = material
    else:
        return False
    first = next((entry for entry in entries if isinstance(entry, Mapping)), None)
    if not isinstance(first, Mapping):
        return False
    ticket = first.get("ticket") if isinstance(first.get("ticket"), Mapping) else first
    comments = first.get("comments")
    has_comments = isinstance(comments, Mapping) or (
        isinstance(comments, Sequence)
        and not isinstance(comments, (str, bytes, bytearray))
    )
    has_ticket_wrapper = isinstance(first.get("ticket"), Mapping)
    return isinstance(ticket, Mapping) and (
        has_ticket_wrapper or (has_comments and "requester_id" in ticket)
    )


def _normalized_rows(package: Any) -> list[Mapping[str, Any]]:
    rows = package.inputs.get("source_material")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _resolution_signal(package: Any, included_count: int) -> dict[str, Any]:
    count = int(package.metadata.get("support_ticket_resolution_evidence_count") or 0)
    readiness = _READINESS_READY if count > 0 else _READINESS_INSUFFICIENT
    return {
        "readiness": readiness,
        "reason": (
            "resolution evidence is present"
            if count > 0
            else "no support-resolution evidence found"
        ),
        "rows_with_resolution_evidence": count,
        "coverage": _ratio(count, included_count),
    }


def _status_signal(package: Any, included_count: int) -> dict[str, Any]:
    count = int(package.metadata.get("ticket_status_present_count") or 0)
    readiness = _coverage_readiness(count, included_count)
    return {
        "readiness": readiness,
        "reason": _coverage_reason(count, included_count, "ticket status"),
        "rows_with_status": count,
        "coverage": _ratio(count, included_count),
        "summary": _safe_count_mapping(package.metadata.get("ticket_status_summary")),
    }


def _csat_signal(package: Any, included_count: int) -> dict[str, Any]:
    present_count = int(package.metadata.get("csat_present_count") or 0)
    score_count = int(package.metadata.get("csat_score_count") or 0)
    textual_count = max(present_count - score_count, 0)
    if present_count == 0:
        readiness = _READINESS_INSUFFICIENT
        basis = "absent"
        reason = "no CSAT signal found"
    elif present_count < included_count or textual_count:
        readiness = _READINESS_PARTIAL
        basis = "mixed" if score_count and textual_count else ("numeric" if score_count else "textual")
        reason = "CSAT is present but sparse or non-numeric"
    else:
        readiness = _READINESS_READY
        basis = "numeric"
        reason = "numeric CSAT is present for all included rows"
    return {
        "readiness": readiness,
        "reason": reason,
        "basis": basis,
        "rows_with_csat": present_count,
        "numeric_score_count": score_count,
        "textual_rating_count": textual_count,
        "coverage": _ratio(present_count, included_count),
        "numeric_average": package.metadata.get("csat_score_average"),
    }


def _cost_signal(rows: Sequence[Mapping[str, Any]], included_count: int) -> dict[str, Any]:
    source_cost_rows = _rows_with_matching_fields(rows, _SOURCE_COST_KEYS)
    if source_cost_rows:
        readiness = _coverage_readiness(source_cost_rows, included_count)
        reason = "source cost or handle-time fields are present"
        basis = "source_fields_present"
    else:
        readiness = _READINESS_PARTIAL if included_count else _READINESS_INSUFFICIENT
        reason = (
            "report can use the benchmark assisted-contact formula only"
            if included_count
            else "no included rows to estimate cost"
        )
        basis = "benchmark_only"
    return {
        "readiness": readiness,
        "reason": reason,
        "basis": basis,
        "rows_with_source_cost_fields": source_cost_rows,
        "coverage": _ratio(source_cost_rows, included_count),
        "requires_formula_source_label": True,
    }


def _owner_lane_signal(rows: Sequence[Mapping[str, Any]], included_count: int) -> dict[str, Any]:
    structured_rows = _rows_with_matching_fields(rows, _STRUCTURED_CONTEXT_KEYS)
    cluster_rows = sum(1 for row in rows if _clean(row.get("support_ticket_cluster")))
    if included_count == 0:
        readiness = _READINESS_INSUFFICIENT
        reason = "no included rows to classify"
    elif structured_rows == included_count:
        readiness = _READINESS_READY
        reason = "structured product/issue/category fields cover all rows"
    elif structured_rows:
        readiness = _READINESS_PARTIAL
        reason = "deterministic context exists, but Unknown fallback is required"
    elif cluster_rows:
        readiness = _READINESS_PARTIAL
        reason = "cluster labels exist, but owner lane/fix type require Unknown fallback"
    else:
        readiness = _READINESS_INSUFFICIENT
        reason = "no deterministic context fields found for owner lane or fix type"
    return {
        "readiness": readiness,
        "reason": reason,
        "rows_with_structured_context": structured_rows,
        "rows_with_cluster_label": cluster_rows,
        "structured_context_coverage": _ratio(structured_rows, included_count),
        "unknown_fallback_required": structured_rows < included_count,
    }


def _snippet_safety_signal(rows: Sequence[Mapping[str, Any]], included_count: int) -> dict[str, Any]:
    checked = 0
    changed = 0
    for row in rows:
        for key in _TEXT_KEYS_TO_SCRUB:
            text = _clean(row.get(key))
            if not text:
                continue
            checked += 1
            scrubbed = scrub_deflection_report_payload({"value": text}).get("value")
            if scrubbed != text:
                changed += 1
    if included_count == 0:
        readiness = _READINESS_INSUFFICIENT
        reason = "no included rows to scrub-check"
    else:
        readiness = _READINESS_PARTIAL
        reason = "scrub ran on text fields, but unlabeled data cannot prove open-set recall"
    return {
        "readiness": readiness,
        "reason": reason,
        "text_fields_checked": checked,
        "text_fields_changed_by_scrub": changed,
        "requires_fail_closed_snapshot_allowlist": True,
        "requires_shared_detector_contract": True,
        "raw_snippets_recorded": False,
    }


def _rows_with_matching_fields(
    rows: Sequence[Mapping[str, Any]],
    wanted_keys: frozenset[str],
) -> int:
    count = 0
    for row in rows:
        if any(_key(key) in wanted_keys and _clean(value) for key, value in row.items()):
            count += 1
    return count


def _coverage_readiness(count: int, total: int) -> str:
    if total <= 0 or count <= 0:
        return _READINESS_INSUFFICIENT
    if count < total:
        return _READINESS_PARTIAL
    return _READINESS_READY


def _coverage_reason(count: int, total: int, label: str) -> str:
    if total <= 0 or count <= 0:
        return f"no {label} signal found"
    if count < total:
        return f"{label} signal is present but sparse"
    return f"{label} signal covers all included rows"


def _warning_code_counts(warnings: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(
        _clean(warning.get("code"))
        for warning in warnings
        if isinstance(warning, Mapping) and _clean(warning.get("code"))
    )
    return dict(sorted(counter.items()))


def _package_warning_codes(warnings: Sequence[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for warning in warnings:
        code = ""
        if isinstance(warning, Mapping):
            code = _clean(warning.get("code"))
        else:
            code = _clean(getattr(warning, "code", ""))
        if code:
            counter[code] += 1
    return dict(sorted(counter.items()))


def _safe_count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        _clean(key): int(count)
        for key, count in sorted(value.items())
        if _clean(key) and _intable(count)
    }


def _insufficient_dependencies(summary: Mapping[str, Any]) -> list[str]:
    readiness = summary.get("s1_readiness")
    if not isinstance(readiness, Mapping):
        return []
    return [
        _clean(name)
        for name, value in readiness.items()
        if isinstance(value, Mapping)
        and value.get("status") == _READINESS_INSUFFICIENT
        and _clean(name)
    ]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _intable(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
