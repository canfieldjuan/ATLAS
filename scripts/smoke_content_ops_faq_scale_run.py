#!/usr/bin/env python3
"""Run a repeatable FAQ Markdown scale smoke with standard artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
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
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result_payload = _read_json(result_path)
    summary = {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "source": str(args.path),
        "source_format": args.source_format,
        "command": command,
        "artifacts": {
            "markdown": _artifact_path(markdown_path),
            "result": _artifact_path(result_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
        },
        "result": result_payload,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _is_output_check_failure(result_payload: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(result_payload, dict)
        and result_payload.get("status") == "failed_output_checks"
    )


if __name__ == "__main__":
    raise SystemExit(main())
