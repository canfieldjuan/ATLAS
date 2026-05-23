#!/usr/bin/env python3
"""Run the FAQ lifecycle smoke and capture standard artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_CLI = ROOT / "scripts/smoke_content_ops_faq_lifecycle.py"
DEFAULT_SOURCE_PATH = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"
RESERVED_LIFECYCLE_FLAGS = ("--output-result", "--json", "--summary-json")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FAQ lifecycle smoke with artifact capture. Lifecycle flags such "
            "as --account-id, --source-format, and --min-source-rows pass through."
        )
    )
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args, lifecycle_args = parser.parse_known_args(argv)
    args.lifecycle_args = lifecycle_args
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    code, summary = run_lifecycle_artifact_smoke(args)
    _print_run_summary(summary)
    return code


def _validate_args(args: argparse.Namespace) -> None:
    reserved = [
        value
        for value in args.lifecycle_args
        if any(value == flag or value.startswith(f"{flag}=") for flag in RESERVED_LIFECYCLE_FLAGS)
    ]
    if reserved:
        raise SystemExit(
            "lifecycle artifact runner owns these flags: "
            + ", ".join(sorted(set(reserved)))
        )


def run_lifecycle_artifact_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "lifecycle_result.json"
    summary_stdout_path = artifact_dir / "summary_stdout.json"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    run_summary_path = artifact_dir / "run_summary.json"
    command = [
        sys.executable,
        str(LIFECYCLE_CLI),
        str(args.path),
        *[str(value) for value in args.lifecycle_args],
        "--output-result",
        str(result_path),
        "--summary-json",
    ]
    started = time.monotonic()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed_seconds = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result_payload = _read_json(result_path)
    summary_stdout = _parse_stdout_summary(completed.stdout)
    if summary_stdout is not None:
        summary_stdout_path.write_text(
            json.dumps(summary_stdout, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    artifacts = {
        "result": result_path,
        "summary_stdout": summary_stdout_path,
        "stdout": stdout_path,
        "stderr": stderr_path,
        "summary": run_summary_path,
    }
    summary = {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "source": str(args.path),
        "command": command,
        "timing": {"elapsed_seconds": round(elapsed_seconds, 6)},
        "artifacts": _artifact_paths(artifacts),
        "lifecycle_summary": _lifecycle_summary(result_payload, summary_stdout),
        "failure": _failure_summary(
            returncode=completed.returncode,
            result_payload=result_payload,
            summary_stdout=summary_stdout,
            stderr=completed.stderr,
        ),
        "result": result_payload,
    }
    _write_summary(run_summary_path, summary, artifacts)
    return completed.returncode, summary


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_stdout_summary(stdout: str) -> dict[str, Any] | None:
    try:
        data = json.loads(stdout.strip())
    except json.JSONDecodeError:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _lifecycle_summary(
    result_payload: dict[str, Any] | None,
    summary_stdout: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(summary_stdout, dict):
        return summary_stdout
    if isinstance(result_payload, dict) and isinstance(result_payload.get("lifecycle_summary"), Mapping):
        return dict(result_payload["lifecycle_summary"])
    return None


def _artifact_paths(paths: Mapping[str, Path]) -> dict[str, str | None]:
    return {
        name: str(path) if path.exists() or name == "summary" else None
        for name, path in paths.items()
    }


def _artifact_details(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
        }
        for name, path in paths.items()
    }


def _write_summary(
    summary_path: Path,
    summary: dict[str, Any],
    artifact_paths: Mapping[str, Path],
) -> None:
    details = _artifact_details(artifact_paths)
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
    summary_stdout: dict[str, Any] | None,
    stderr: str,
) -> dict[str, Any] | None:
    if returncode == 0:
        return None
    errors = _payload_errors(result_payload) or _payload_errors(summary_stdout)
    return {
        "type": "lifecycle_error" if errors else "missing_result",
        "errors": errors,
        "stderr_tail": _text_tail(stderr),
        "summary_status": summary_stdout.get("status") if isinstance(summary_stdout, dict) else None,
    }


def _payload_errors(value: dict[str, Any] | None) -> list[str]:
    errors = value.get("errors") if isinstance(value, dict) else None
    return [str(error) for error in errors] if isinstance(errors, list) else []


def _text_tail(value: str, *, max_chars: int = 4000, max_lines: int = 20) -> str:
    lines = value.strip().splitlines()
    text = "\n".join(lines[-max_lines:])
    return text[-max_chars:] if len(text) > max_chars else text


def _print_run_summary(summary: Mapping[str, Any]) -> None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), Mapping) else {}
    lifecycle_summary = summary.get("lifecycle_summary")
    status = lifecycle_summary.get("status") if isinstance(lifecycle_summary, Mapping) else "unknown"
    source_rows = lifecycle_summary.get("source_rows") if isinstance(lifecycle_summary, Mapping) else "unknown"
    saved_faqs = lifecycle_summary.get("saved_faq_count") if isinstance(lifecycle_summary, Mapping) else "unknown"
    message = (
        f"status={status} source_rows={source_rows} saved_faqs={saved_faqs} "
        f"summary={artifacts.get('summary')}"
    )
    if summary.get("ok") is True:
        print(f"Content Ops FAQ lifecycle artifact smoke passed: {message}")
        return
    failure = summary.get("failure") if isinstance(summary.get("failure"), Mapping) else {}
    print(
        "Content Ops FAQ lifecycle artifact smoke failed: "
        f"{message} failure={failure.get('type') or 'unknown'}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
