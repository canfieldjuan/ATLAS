#!/usr/bin/env python3
"""Run a repeatable FAQ Markdown scale smoke with standard artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FAQ_CLI = ROOT / "scripts/build_extracted_ticket_faq_markdown.py"

from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)


_JSON_ROW_KEYS = (
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
_SKIP_WARNING_CODES = {"empty_row", "missing_source_text", "row_not_object"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FAQ Markdown generation and capture scale-smoke artifacts."
    )
    parser.add_argument("path", type=Path, help="Source JSON, JSONL, or CSV file.")
    parser.add_argument("--source-format", choices=("auto", "json", "jsonl", "csv"), default="auto")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--title", default="Customer Ticket FAQ Scale Smoke")
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--max-evidence-per-item", type=int, default=5)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--window-days", type=int)
    parser.add_argument("--as-of-date")
    parser.add_argument("--support-contact")
    parser.add_argument("--default-field", action="append", default=[])
    parser.add_argument("--allow-output-check-failures", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    code, summary = run_scale_smoke(args)
    _print_scale_summary(summary)
    return code


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_items < 1:
        raise SystemExit("--max-items must be positive")
    if args.max_evidence_per_item < 1:
        raise SystemExit("--max-evidence-per-item must be positive")
    if args.max_text_chars < 1:
        raise SystemExit("--max-text-chars must be positive")
    if args.window_days is not None and args.window_days < 1:
        raise SystemExit("--window-days must be positive")
    if args.as_of_date and args.window_days is None:
        raise SystemExit("--as-of-date requires --window-days")


def run_scale_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = artifact_dir / "faq.md"
    result_path = artifact_dir / "faq_result.json"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    summary_path = artifact_dir / "run_summary.json"
    command = _build_command(args, markdown_path=markdown_path, result_path=result_path)
    input_profile = _input_profile(args)
    started = time.monotonic()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed_seconds = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result_payload = _read_json(result_path)
    faq_run_summary = _faq_run_summary(result_payload)
    artifact_paths = {
        "markdown": markdown_path,
        "result": result_path,
        "stdout": stdout_path,
        "stderr": stderr_path,
        "summary": summary_path,
    }
    summary = {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "source": str(args.path),
        "source_format": args.source_format,
        "command": command,
        "input_profile": input_profile,
        "timing": {"elapsed_seconds": round(elapsed_seconds, 6)},
        "artifacts": {
            "markdown": _artifact_path(markdown_path),
            "result": _artifact_path(result_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
        },
        "artifact_details": _artifact_details(artifact_paths),
        "faq_run_summary": faq_run_summary,
        "failure": _failure_summary(
            returncode=completed.returncode,
            result_payload=result_payload,
            stderr=completed.stderr,
        ),
        "result": result_payload,
    }
    _write_summary(summary_path, summary, artifact_paths)
    if (
        completed.returncode != 0
        and bool(args.allow_output_check_failures)
        and _is_output_check_failure(result_payload)
    ):
        return 0, summary
    return completed.returncode, summary


def _faq_run_summary(result_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result_payload, dict):
        return None
    diagnostics = result_payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    run_summary = diagnostics.get("run_summary")
    return dict(run_summary) if isinstance(run_summary, dict) else None


def _build_command(
    args: argparse.Namespace,
    *,
    markdown_path: Path,
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(FAQ_CLI),
        str(args.path),
        "--source-format",
        str(args.source_format),
        "--title",
        str(args.title),
        "--max-items",
        str(args.max_items),
        "--max-evidence-per-item",
        str(args.max_evidence_per_item),
        "--max-text-chars",
        str(args.max_text_chars),
        "--output",
        str(markdown_path),
        "--result-output",
        str(result_path),
        "--require-output-checks",
    ]
    if args.window_days is not None:
        command.extend(["--window-days", str(args.window_days)])
    if args.as_of_date:
        command.extend(["--as-of-date", str(args.as_of_date)])
    if args.support_contact:
        command.extend(["--support-contact", str(args.support_contact)])
    for value in args.default_field:
        command.extend(["--default-field", str(value)])
    return command


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _input_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = {
        "status": "ok",
        "raw_row_count": None,
        "raw_row_count_source": None,
        "usable_source_count": None,
        "warning_count": None,
        "warnings_by_code": {},
        "skipped_row_count": None,
        "missing_source_text_count": None,
        "warning_sample": [],
    }
    try:
        profile.update(_raw_row_profile(Path(args.path), str(args.source_format)))
    except Exception as exc:  # pragma: no cover - exact host filesystem errors vary.
        profile["raw_row_count_error"] = f"{type(exc).__name__}: {exc}"
    try:
        loaded = load_source_campaign_opportunities_from_file(
            args.path,
            file_format=args.source_format,
            max_text_chars=args.max_text_chars,
            default_fields=parse_default_fields_or_exit(args.default_field),
        )
    except (Exception, SystemExit) as exc:
        profile["status"] = "error"
        profile["error"] = f"{type(exc).__name__}: {exc}"
        return profile
    warnings = loaded.warning_dicts()
    warnings_by_code = Counter(str(warning.get("code") or "unknown") for warning in warnings)
    skipped_rows = {
        int(warning["row_index"])
        for warning in warnings
        if warning.get("code") in _SKIP_WARNING_CODES and isinstance(warning.get("row_index"), int)
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


def _raw_row_profile(path: Path, source_format: str) -> dict[str, Any]:
    resolved = _resolve_source_format(path, source_format)
    if resolved == "csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return {
                "raw_row_count": sum(1 for _row in csv.DictReader(handle)),
                "raw_row_count_source": "csv_rows",
            }
    if resolved == "jsonl":
        return {
            "raw_row_count": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "raw_row_count_source": "jsonl_lines",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"raw_row_count": len(data), "raw_row_count_source": "json_array"}
    if isinstance(data, Mapping):
        bundle_counts = []
        for key in _JSON_ROW_KEYS:
            value = data.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                bundle_counts.append((key, len(value)))
        if bundle_counts:
            keys = ",".join(key for key, _count in bundle_counts)
            return {
                "raw_row_count": sum(count for _key, count in bundle_counts),
                "raw_row_count_source": f"json_bundle.{keys}",
            }
    return {"raw_row_count": None, "raw_row_count_source": None}


def _resolve_source_format(path: Path, source_format: str) -> str:
    if source_format != "auto":
        return source_format
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "json"


def _artifact_path(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _artifact_details(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    details = {}
    for name, path in paths.items():
        exists = path.exists()
        details[name] = {
            "path": str(path),
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
        }
    return details


def _write_summary(
    summary_path: Path,
    summary: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    details = _artifact_details(artifact_paths)
    if "summary" in details:
        details["summary"]["exists"] = True
        details["summary"]["bytes"] = None
    summary["artifact_details"] = details
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_scale_summary(summary: Mapping[str, Any]) -> None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
    summary_path = artifacts.get("summary") if isinstance(artifacts, Mapping) else None
    profile = _console_input_profile(summary.get("input_profile"))
    faq_profile = _console_faq_run_summary(summary.get("faq_run_summary"))
    if summary.get("ok") is True:
        print(
            "Content Ops FAQ scale smoke passed: "
            f"{profile} {faq_profile} summary={summary_path}"
        )
        return
    failure = summary.get("failure") if isinstance(summary.get("failure"), Mapping) else {}
    failure_type = failure.get("type") if isinstance(failure, Mapping) else None
    print(
        "Content Ops FAQ scale smoke failed: "
        f"{profile} {faq_profile} failure={failure_type or 'unknown'} summary={summary_path}",
        file=sys.stderr,
    )


def _console_input_profile(value: Any) -> str:
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


def _console_faq_run_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "faq=unavailable"
    parts = ["faq=available"]
    for key, label in (
        ("generated", "generated"),
        ("weighted_source_volume", "weighted_volume"),
    ):
        item = value.get(key)
        if item is not None:
            parts.append(f"{label}={item}")
    output_checks = value.get("output_checks")
    if isinstance(output_checks, Mapping):
        failed = output_checks.get("failed")
        total = output_checks.get("total")
        if failed is not None or total is not None:
            parts.append(
                f"checks_failed={_console_value(failed)}/{_console_value(total)}"
            )
    score_distribution = value.get("item_score_distribution")
    if (
        isinstance(score_distribution, Mapping)
        and score_distribution.get("max") is not None
    ):
        parts.append(f"score_max={score_distribution.get('max')}")
    return " ".join(parts)


def _failure_summary(
    *,
    returncode: int,
    result_payload: dict[str, Any] | None,
    stderr: str,
) -> dict[str, Any] | None:
    if returncode == 0:
        return None
    failed_checks: list[str] = []
    result_status = None
    output_check_details: list[dict[str, Any]] = []
    if isinstance(result_payload, dict):
        result_status = result_payload.get("status")
        failed_checks = [
            str(check)
            for check in result_payload.get("failed_output_checks") or []
        ]
        diagnostics = result_payload.get("diagnostics")
        if isinstance(diagnostics, dict):
            details = diagnostics.get("output_check_details")
            if isinstance(details, list):
                output_check_details = [
                    dict(detail)
                    for detail in details
                    if isinstance(detail, dict) and detail.get("passed") is not True
                ]
    return {
        "type": "output_checks" if failed_checks else "cli_error",
        "exit_code": returncode,
        "result_status": result_status,
        "failed_output_checks": failed_checks,
        "output_check_details": output_check_details,
        "stderr_tail": _text_tail(stderr),
    }


def _text_tail(text: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    tail = "\n".join(text.splitlines()[-max_lines:])
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def _is_output_check_failure(result_payload: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(result_payload, dict)
        and result_payload.get("status") == "failed_output_checks"
    )


if __name__ == "__main__":
    raise SystemExit(main())
