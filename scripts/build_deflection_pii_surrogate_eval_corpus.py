#!/usr/bin/env python3
"""Build a surrogate-only deflection PII eval corpus from labeled local JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.deflection_pii_eval_corpus import (  # noqa: E402
    SCHEMA_VERSION,
    format_labeled_source_intake_summary_markdown,
    summarize_labeled_source,
    build_surrogate_eval_corpus,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _write_error({"ok": False, "schema_version": SCHEMA_VERSION, "errors": [{
            "code": "input_json_unreadable",
            "message": str(exc.__class__.__name__),
        }]})
        return 1
    result = build_surrogate_eval_corpus(source) if args.output else None
    summary: dict[str, Any] | None = None
    if args.summary_output or args.summary_markdown_output:
        summary = summarize_labeled_source(source, build_result=result)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.summary_markdown_output:
        args.summary_markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_markdown_output.write_text(
            format_labeled_source_intake_summary_markdown(summary),
            encoding="utf-8",
        )
    if summary is not None:
        if not summary["ok"]:
            _write_error({
                "ok": False,
                "schema_version": summary["schema_version"],
                "errors": list(summary["errors"]),
            })
            return 1

    artifact: dict[str, Any] | None = None
    if args.output:
        if result is None:
            result = build_surrogate_eval_corpus(source)
        if not result.ok:
            _write_error({
                "ok": False,
                "schema_version": SCHEMA_VERSION,
                "errors": list(result.errors),
            })
            return 1
        artifact = result.artifact
        if artifact is None:
            _write_error({
                "ok": False,
                "schema_version": SCHEMA_VERSION,
                "errors": [{"code": "surrogate_artifact_missing"}],
            })
            return 1
        text = json.dumps(artifact, indent=2 if args.pretty else None, sort_keys=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    payload = {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
    }
    if artifact is not None:
        payload.update({
            "output": str(args.output),
            "ticket_count": artifact["summary"]["ticket_count"],
            "label_count": artifact["summary"]["label_count"],
        })
    if args.summary_output and summary is not None:
        payload["summary_output"] = str(args.summary_output)
        payload["summary_schema_version"] = str(summary["schema_version"])
    if args.summary_markdown_output and summary is not None:
        payload["summary_markdown_output"] = str(args.summary_markdown_output)
        payload["summary_schema_version"] = str(summary["schema_version"])
    print(json.dumps(payload, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local labeled-source JSON.")
    parser.add_argument("--output", type=Path, help="Surrogate artifact path.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Sanitized labeled-source intake summary path.",
    )
    parser.add_argument(
        "--summary-markdown-output",
        type=Path,
        help="Sanitized Markdown intake summary path.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write pretty JSON.")
    args = parser.parse_args(argv)
    outputs = (
        ("--output", args.output),
        ("--summary-output", args.summary_output),
        ("--summary-markdown-output", args.summary_markdown_output),
    )
    if not any(path is not None for _, path in outputs):
        parser.error(
            "at least one of --output, --summary-output, or "
            "--summary-markdown-output is required"
        )
    _reject_duplicate_output_paths(parser, outputs)
    return args


def _reject_duplicate_output_paths(
    parser: argparse.ArgumentParser,
    outputs: tuple[tuple[str, Path | None], ...],
) -> None:
    seen: dict[Path, str] = {}
    for flag, path in outputs:
        if path is None:
            continue
        resolved = path.expanduser().resolve()
        existing = seen.get(resolved)
        if existing is not None:
            parser.error(f"{existing} and {flag} must be different paths")
        seen[resolved] = flag


def _write_error(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
