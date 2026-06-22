#!/usr/bin/env python3
"""Validate and export a scored deflection PII review-bundle corpus candidate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for candidate in (ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_deflection_pii_surrogate_eval_corpus import (  # noqa: E402
    REVIEW_BUNDLE_ARTIFACT_NAME,
    REVIEW_BUNDLE_MARKDOWN_NAME,
    REVIEW_BUNDLE_MANIFEST_NAME,
    REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION,
    REVIEW_BUNDLE_SUMMARY_NAME,
)
from extracted_content_pipeline.deflection_pii_eval_corpus import (  # noqa: E402
    SCHEMA_VERSION as CORPUS_SCHEMA_VERSION,
)
from score_deflection_pii_recall import (  # noqa: E402
    REVIEW_BUNDLE_SCORE_MARKDOWN_NAME,
    REVIEW_BUNDLE_SCORE_NAME,
    SCORE_SCHEMA_VERSION,
    _corpus_errors as score_corpus_errors,
)


EXPORT_SCHEMA_VERSION = "deflection_pii_review_bundle_candidate.v1"
HEADLINE_KEYS = (
    "free_high_severity_gate_eligible_leak_count",
    "free_high_severity_leak_count",
    "deferred_open_set_name_leak_count",
)
RESERVED_BUNDLE_NAMES = frozenset({
    REVIEW_BUNDLE_ARTIFACT_NAME,
    REVIEW_BUNDLE_SUMMARY_NAME,
    REVIEW_BUNDLE_MARKDOWN_NAME,
    REVIEW_BUNDLE_MANIFEST_NAME,
    REVIEW_BUNDLE_SCORE_NAME,
    REVIEW_BUNDLE_SCORE_MARKDOWN_NAME,
})


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    candidate, errors = _validated_candidate(args.review_bundle_dir)
    if candidate is None:
        _write_errors(errors)
        return 1

    output_errors = _output_errors(
        bundle_dir=args.review_bundle_dir,
        source_path=candidate["source_path"],
        output_path=args.output,
        force=args.force,
    )
    if output_errors:
        _write_errors(output_errors)
        return 1

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate["source_path"], args.output)
    except OSError as exc:
        _write_errors([_error("output_unwritable", message=exc.__class__.__name__)])
        return 1

    payload = {
        "ok": True,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "output": str(args.output),
        "ticket_count": candidate["ticket_count"],
        "label_count": candidate["label_count"],
        "score_status": "ok",
        "headline": candidate["headline"],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_bundle_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validated_candidate(bundle_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    manifest, errors = _read_object(
        bundle_dir / REVIEW_BUNDLE_MANIFEST_NAME,
        "manifest_load_failed",
    )
    if errors:
        return None, errors

    corpus_path = bundle_dir / REVIEW_BUNDLE_ARTIFACT_NAME
    score_path = bundle_dir / REVIEW_BUNDLE_SCORE_NAME
    corpus, corpus_errors = _read_object(corpus_path, "corpus_load_failed")
    score, score_errors = _read_object(score_path, "score_load_failed")
    errors.extend(corpus_errors)
    errors.extend(score_errors)
    corpus_counts = _corpus_counts(corpus)
    if not corpus_errors:
        errors.extend(_corpus_errors(corpus))
    if not score_errors:
        errors.extend(_score_errors(score, expected_counts=corpus_counts))
    errors.extend(_manifest_errors(manifest, expected_counts=corpus_counts))
    if errors:
        return None, errors

    return {
        "source_path": corpus_path,
        "ticket_count": corpus_counts["ticket_count"],
        "label_count": corpus_counts["label_count"],
        "headline": _headline(score),
    }, []


def _manifest_errors(
    manifest: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if _clean(manifest.get("schema_version")) != REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION:
        errors.append(_error("manifest_schema_version_mismatch"))
    if _clean(manifest.get("status")) != "ok":
        errors.append(_error("manifest_status_not_ok"))
    if _clean(manifest.get("score_status")) != "ok":
        errors.append(_error("manifest_score_status_not_ok"))

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        errors.append(_error("manifest_files_missing"))
        return errors
    errors.extend(_manifest_corpus_file_errors(files, expected_counts=expected_counts))
    errors.extend(_manifest_file_errors(files, "recall_score", REVIEW_BUNDLE_SCORE_NAME))
    return errors


def _manifest_corpus_file_errors(
    files: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
) -> list[dict[str, str]]:
    errors = _manifest_file_errors(
        files,
        "surrogate_eval_corpus",
        REVIEW_BUNDLE_ARTIFACT_NAME,
    )
    entry = files.get("surrogate_eval_corpus")
    if not isinstance(entry, Mapping):
        return errors
    if _clean(entry.get("schema_version")) != CORPUS_SCHEMA_VERSION:
        errors.append(_error("manifest_corpus_schema_version_mismatch"))
    for key in ("ticket_count", "label_count"):
        if _safe_int(entry.get(key)) != expected_counts[key]:
            errors.append(_error("manifest_corpus_count_mismatch", field=key))
    return errors


def _manifest_file_errors(
    files: Mapping[str, Any],
    key: str,
    expected_path: str,
) -> list[dict[str, str]]:
    entry = files.get(key)
    if not isinstance(entry, Mapping):
        return [_error("manifest_file_missing", file=key)]
    errors: list[dict[str, str]] = []
    if entry.get("present") is not True:
        errors.append(_error("manifest_file_not_present", file=key))
    if _clean(entry.get("path")) != expected_path:
        errors.append(_error("manifest_file_path_mismatch", file=key))
    return errors


def _corpus_errors(corpus: Mapping[str, Any]) -> list[dict[str, str]]:
    errors = [
        _error(_clean(error.get("code")) or "corpus_invalid")
        for error in score_corpus_errors(corpus)
    ]
    source = corpus.get("source")
    if not isinstance(source, Mapping) or not (
        _clean(source.get("kind")) == "surrogated_eval"
        and source.get("raw_label_spans_persisted") is False
        and source.get("raw_source_persisted") is False
        and source.get("surrogate_positions_are_recall_labels") is True
    ):
        errors.append(_error("corpus_not_surrogate_only"))
    errors.extend(_corpus_summary_errors(corpus, expected_counts=_corpus_counts(corpus)))
    return errors


def _score_errors(
    score: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if _clean(score.get("schema_version")) != SCORE_SCHEMA_VERSION:
        errors.append(_error("score_schema_version_mismatch"))
    if _clean(score.get("status")) != "ok":
        errors.append(_error("score_status_not_ok"))
    input_summary = score.get("input")
    if not isinstance(input_summary, Mapping):
        errors.append(_error("score_input_missing"))
    else:
        if _clean(input_summary.get("schema_version")) != CORPUS_SCHEMA_VERSION:
            errors.append(_error("score_input_schema_version_mismatch"))
        for key in ("ticket_count", "label_count", "must_survive_count"):
            if _safe_int(input_summary.get(key)) != expected_counts[key]:
                errors.append(_error("score_input_count_mismatch", field=key))
    errors.extend(_headline_errors(score.get("headline")))
    return errors


def _output_errors(
    *,
    bundle_dir: Path,
    source_path: Path,
    output_path: Path,
    force: bool,
) -> list[dict[str, str]]:
    try:
        resolved_output = output_path.resolve()
        if source_path.resolve() == resolved_output:
            return [_error("output_same_as_source")]
        reserved_paths = {
            (bundle_dir / name).resolve()
            for name in RESERVED_BUNDLE_NAMES
        }
        if resolved_output in reserved_paths:
            return [_error("output_reserved_bundle_artifact")]
    except OSError:
        return [_error("output_path_unresolvable")]
    if output_path.exists() and not force:
        return [_error("output_exists")]
    return []


def _read_object(path: Path, code: str) -> tuple[Mapping[str, Any], list[dict[str, str]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, [_error(code)]
    if not isinstance(raw, Mapping):
        return {}, [_error(code)]
    return raw, []


def _corpus_summary_errors(
    corpus: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
) -> list[dict[str, str]]:
    summary = corpus.get("summary")
    if not isinstance(summary, Mapping):
        return [_error("corpus_summary_missing")]
    errors: list[dict[str, str]] = []
    for key in ("ticket_count", "label_count", "must_survive_count"):
        if _safe_int(summary.get(key)) != expected_counts[key]:
            errors.append(_error("corpus_summary_count_mismatch", field=key))
    return errors


def _corpus_counts(corpus: Mapping[str, Any]) -> dict[str, int]:
    tickets = corpus.get("tickets")
    if not isinstance(tickets, list):
        return {"ticket_count": 0, "label_count": 0, "must_survive_count": 0}
    label_count = 0
    must_survive_count = 0
    ticket_count = 0
    for ticket in tickets:
        if not isinstance(ticket, Mapping):
            continue
        ticket_count += 1
        labels = ticket.get("labels")
        if isinstance(labels, list):
            label_count += len(labels)
        must_survive = ticket.get("must_survive")
        if isinstance(must_survive, list):
            must_survive_count += len(must_survive)
    return {
        "ticket_count": ticket_count,
        "label_count": label_count,
        "must_survive_count": must_survive_count,
    }


def _headline_errors(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Mapping):
        return [_error("score_headline_missing")]
    errors: list[dict[str, str]] = []
    for key in HEADLINE_KEYS:
        if key not in value:
            errors.append(_error("score_headline_metric_missing", field=key))
        elif not isinstance(value.get(key), int) or isinstance(value.get(key), bool):
            errors.append(_error("score_headline_metric_not_integer", field=key))
    return errors


def _headline(score: Mapping[str, Any]) -> dict[str, int]:
    headline = score.get("headline")
    if not isinstance(headline, Mapping):
        return {key: 0 for key in HEADLINE_KEYS}
    return {key: _safe_int(headline.get(key)) for key in HEADLINE_KEYS}


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _error(code: str, **details: str) -> dict[str, str]:
    return {"code": code, **details}


def _write_errors(errors: list[dict[str, str]]) -> None:
    print(
        json.dumps(
            {"ok": False, "schema_version": EXPORT_SCHEMA_VERSION, "errors": errors},
            sort_keys=True,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
