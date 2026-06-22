#!/usr/bin/env python3
"""Build a surrogate-only deflection PII eval corpus from labeled local JSON."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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

REVIEW_BUNDLE_ARTIFACT_NAME = "deflection-pii-surrogate-eval-corpus.json"
REVIEW_BUNDLE_SUMMARY_NAME = "source-intake-summary.json"
REVIEW_BUNDLE_MARKDOWN_NAME = "source-intake-summary.md"
REVIEW_BUNDLE_MANIFEST_NAME = "review-bundle-manifest.json"
REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION = "deflection_pii_review_bundle_manifest.v1"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = args.output
    summary_output = args.summary_output
    summary_markdown_output = args.summary_markdown_output
    manifest_output: Path | None = None
    if args.review_bundle_dir is not None:
        bundle_paths = _review_bundle_paths(args.review_bundle_dir)
        if bundle_paths is None:
            return 1
        output_path = bundle_paths["artifact"]
        summary_output = bundle_paths["summary"]
        summary_markdown_output = bundle_paths["markdown"]
        manifest_output = bundle_paths["manifest"]

    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        error_summary = _blocked_summary("input_json_unreadable", exc.__class__.__name__)
        if manifest_output is not None:
            if not _write_review_bundle_manifest(
                manifest_output,
                summary=error_summary,
                artifact=None,
                pretty=args.pretty,
                source_summary_present=False,
                source_markdown_present=False,
            ):
                return 1
        _write_error(error_summary)
        return 1

    result = build_surrogate_eval_corpus(source) if output_path else None
    summary: dict[str, Any] | None = None
    if summary_output or summary_markdown_output:
        summary = summarize_labeled_source(source, build_result=result)
    if (
        args.review_bundle_dir is not None
        and summary is not None
        and not summary["ok"]
        and output_path is not None
    ):
        if not _remove_existing_review_bundle_artifact(output_path):
            return 1
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if summary_markdown_output:
        summary_markdown_output.parent.mkdir(parents=True, exist_ok=True)
        summary_markdown_output.write_text(
            format_labeled_source_intake_summary_markdown(summary),
            encoding="utf-8",
        )
    if summary is not None:
        if not summary["ok"]:
            if manifest_output is not None:
                if not _write_review_bundle_manifest(
                    manifest_output,
                    summary=summary,
                    artifact=None,
                    pretty=args.pretty,
                ):
                    return 1
            _write_error({
                "ok": False,
                "schema_version": summary["schema_version"],
                "errors": list(summary["errors"]),
            })
            return 1

    artifact: dict[str, Any] | None = None
    if output_path:
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if manifest_output is not None:
        if not _write_review_bundle_manifest(
            manifest_output,
            summary=summary,
            artifact=artifact,
            pretty=args.pretty,
        ):
            return 1
    payload = {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
    }
    if artifact is not None:
        payload.update({
            "output": str(output_path),
            "ticket_count": artifact["summary"]["ticket_count"],
            "label_count": artifact["summary"]["label_count"],
        })
    if summary_output and summary is not None:
        payload["summary_output"] = str(summary_output)
        payload["summary_schema_version"] = str(summary["schema_version"])
    if summary_markdown_output and summary is not None:
        payload["summary_markdown_output"] = str(summary_markdown_output)
        payload["summary_schema_version"] = str(summary["schema_version"])
    if args.review_bundle_dir is not None:
        payload["review_bundle_dir"] = str(args.review_bundle_dir)
        payload["review_bundle_manifest"] = str(manifest_output)
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
    parser.add_argument(
        "--review-bundle-dir",
        type=Path,
        help=(
            "Directory for a sanitized review bundle containing the source "
            "intake JSON summary, Markdown summary, and surrogate artifact."
        ),
    )
    parser.add_argument("--pretty", action="store_true", help="Write pretty JSON.")
    args = parser.parse_args(argv)
    outputs = (
        ("--output", args.output),
        ("--summary-output", args.summary_output),
        ("--summary-markdown-output", args.summary_markdown_output),
    )
    if args.review_bundle_dir is not None and any(path is not None for _, path in outputs):
        parser.error(
            "--review-bundle-dir cannot be combined with --output, "
            "--summary-output, or --summary-markdown-output"
        )
    if args.review_bundle_dir is None and not any(path is not None for _, path in outputs):
        parser.error(
            "at least one of --output, --summary-output, or "
            "--summary-markdown-output is required unless --review-bundle-dir is set"
        )
    _reject_duplicate_output_paths(parser, outputs)
    return args


def _review_bundle_paths(bundle_dir: Path) -> dict[str, Path] | None:
    if bundle_dir.exists() and not bundle_dir.is_dir():
        _write_error({
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "errors": [{"code": "review_bundle_dir_not_directory"}],
        })
        return None
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return {
        "artifact": bundle_dir / REVIEW_BUNDLE_ARTIFACT_NAME,
        "summary": bundle_dir / REVIEW_BUNDLE_SUMMARY_NAME,
        "markdown": bundle_dir / REVIEW_BUNDLE_MARKDOWN_NAME,
        "manifest": bundle_dir / REVIEW_BUNDLE_MANIFEST_NAME,
    }


def _remove_existing_review_bundle_artifact(output_path: Path) -> bool:
    try:
        output_path.unlink(missing_ok=True)
    except OSError as exc:
        _write_error({
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "errors": [{
                "code": "review_bundle_artifact_unremovable",
                "message": exc.__class__.__name__,
            }],
        })
        return False
    return True


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


def _write_review_bundle_manifest(
    path: Path,
    *,
    summary: Mapping[str, Any] | None,
    artifact: Mapping[str, Any] | None,
    pretty: bool,
    source_summary_present: bool | None = None,
    source_markdown_present: bool | None = None,
) -> bool:
    try:
        path.write_text(
            json.dumps(
                _review_bundle_manifest(
                    summary=summary,
                    artifact=artifact,
                    source_summary_present=source_summary_present,
                    source_markdown_present=source_markdown_present,
                ),
                indent=2 if pretty else None,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        _write_error({
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "errors": [{
                "code": "review_bundle_manifest_unwritable",
                "message": exc.__class__.__name__,
            }],
        })
        return False
    return True


def _review_bundle_manifest(
    *,
    summary: Mapping[str, Any] | None,
    artifact: Mapping[str, Any] | None,
    source_summary_present: bool | None = None,
    source_markdown_present: bool | None = None,
) -> dict[str, Any]:
    summary_ok = bool(summary and summary.get("ok") is True)
    summary_present = summary is not None if source_summary_present is None else source_summary_present
    markdown_present = summary is not None if source_markdown_present is None else source_markdown_present
    summary_file: dict[str, Any] = {
        "path": REVIEW_BUNDLE_SUMMARY_NAME,
        "present": summary_present,
        "ok": summary_ok,
    }
    if summary_present and summary is not None:
        summary_file["schema_version"] = _clean(summary.get("schema_version"))
    manifest: dict[str, Any] = {
        "schema_version": REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "status": "ok" if summary_ok and artifact is not None else "blocked",
        "files": {
            "source_intake_summary": summary_file,
            "source_intake_markdown": {
                "path": REVIEW_BUNDLE_MARKDOWN_NAME,
                "present": markdown_present,
            },
            "surrogate_eval_corpus": {
                "path": REVIEW_BUNDLE_ARTIFACT_NAME,
                "present": artifact is not None,
            },
        },
    }
    if summary is not None:
        manifest["blocking_error_codes"] = [
            _clean(error.get("code"))
            for error in _sequence(summary.get("errors"))
            if isinstance(error, Mapping) and _clean(error.get("code"))
        ]
    if artifact is not None:
        corpus_file = manifest["files"]["surrogate_eval_corpus"]
        artifact_summary = artifact.get("summary") if isinstance(artifact.get("summary"), Mapping) else {}
        corpus_file.update({
            "schema_version": _clean(artifact.get("schema_version")),
            "ticket_count": int(artifact_summary.get("ticket_count", 0)),
            "label_count": int(artifact_summary.get("label_count", 0)),
        })
    return manifest


def _blocked_summary(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": SCHEMA_VERSION,
        "errors": [{
            "code": code,
            "message": message,
        }],
    }


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _write_error(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
