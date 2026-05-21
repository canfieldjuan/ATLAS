#!/usr/bin/env python3
"""Run a repeatable FAQ Markdown scale smoke with standard artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FAQ_CLI = ROOT / "scripts/build_extracted_ticket_faq_markdown.py"


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
    code, _summary = run_scale_smoke(args)
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
    started = time.monotonic()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed_seconds = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result_payload = _read_json(result_path)
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
        "timing": {"elapsed_seconds": round(elapsed_seconds, 6)},
        "artifacts": {
            "markdown": _artifact_path(markdown_path),
            "result": _artifact_path(result_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
        },
        "artifact_details": _artifact_details(artifact_paths),
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
