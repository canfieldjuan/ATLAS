#!/usr/bin/env python3
"""Validate and optionally re-render a saved claim/evidence result artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.claim_evidence_benchmark import (
    load_claim_evidence_result_artifact_text,
    render_claim_evidence_result_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a saved Content Ops claim/evidence result artifact."
    )
    parser.add_argument("result_path", type=Path)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional path to write the rendered Markdown report.",
    )
    return parser


def _error_payload(
    errors: Sequence[str],
    *,
    result_path: Path,
    markdown_output: Path | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "go_no_go": "no_go",
        "result_path": str(result_path),
        "markdown_output": str(markdown_output) if markdown_output is not None else "",
        "markdown_written": False,
        "errors": list(errors),
        "artifact_errors": [],
        "verdict_failures": [],
    }


def _read_result_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"result artifact file not found: {path}"
    except IsADirectoryError:
        return None, f"result artifact path is a directory: {path}"
    except OSError as error:
        return (
            None,
            f"result artifact file could not be read: {path}: {error.strerror or error}",
        )
    except UnicodeDecodeError as error:
        return None, f"result artifact file could not be read: {path}: {error}"


def _write_markdown(path: Path, content: str) -> str | None:
    if path.is_symlink():
        return f"markdown output path is a symlink: {path}"
    if path.exists() and path.is_dir():
        return f"markdown output path is a directory: {path}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        return f"markdown output could not be written: {path}: {error.strerror or error}"
    return None


def validate_claim_evidence_result_file(
    result_path: Path,
    *,
    markdown_output: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    text, read_error = _read_result_text(result_path)
    if read_error:
        return 2, _error_payload(
            (read_error,),
            result_path=result_path,
            markdown_output=markdown_output,
        )

    artifact = load_claim_evidence_result_artifact_text(text)
    payload = {
        "ok": artifact.ok,
        "go_no_go": artifact.go_no_go,
        "result_path": str(result_path),
        "markdown_output": str(markdown_output) if markdown_output is not None else "",
        "markdown_written": False,
        "errors": [],
        "artifact_errors": list(artifact.errors),
        "verdict_failures": list(artifact.verdict.failure_reasons),
    }

    if markdown_output is not None:
        markdown_error = _write_markdown(
            markdown_output,
            render_claim_evidence_result_markdown(artifact),
        )
        if markdown_error:
            payload["errors"] = [markdown_error]
            return 2, payload
        payload["markdown_written"] = True

    return (0 if artifact.ok else 1), payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, payload = validate_claim_evidence_result_file(
        args.result_path,
        markdown_output=args.markdown_output,
    )
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
