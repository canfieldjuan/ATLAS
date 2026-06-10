#!/usr/bin/env python3
"""Validate Content Ops claim/evidence benchmark fixture files."""

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
    FIXTURE_FORMAT_JSON,
    FIXTURE_FORMAT_JSONL,
    BenchmarkFixture,
    load_claim_evidence_fixture_text,
)


FORMAT_AUTO = "auto"
VALID_FORMATS = (FORMAT_AUTO, FIXTURE_FORMAT_JSON, FIXTURE_FORMAT_JSONL)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Content Ops claim/evidence benchmark fixture files."
    )
    parser.add_argument("fixture_path", type=Path)
    parser.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default=FORMAT_AUTO,
        help="Fixture format. Auto accepts only .json and .jsonl suffixes.",
    )
    parser.add_argument(
        "--require-final-shape",
        action="store_true",
        help="Require the final #1435 15/15/10 benchmark composition.",
    )
    return parser


def _infer_format(path: Path, requested_format: str) -> tuple[str | None, str | None]:
    if requested_format != FORMAT_AUTO:
        return requested_format, None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return FIXTURE_FORMAT_JSON, None
    if suffix == ".jsonl":
        return FIXTURE_FORMAT_JSONL, None
    return None, "fixture format auto-detection requires .json or .jsonl suffix"


def _result_payload(
    fixture: BenchmarkFixture,
    *,
    source_format: str | None,
) -> dict[str, Any]:
    return {
        "ok": fixture.ok,
        "source_format": source_format,
        "errors": list(fixture.errors),
        "triple_count": len(fixture.triples),
        "easy_supports_count": fixture.easy_supports_count,
        "easy_not_supports_count": fixture.easy_not_supports_count,
        "hard_count": fixture.hard_count,
    }


def _error_payload(error: str, *, source_format: str | None = None) -> dict[str, Any]:
    return _result_payload(
        BenchmarkFixture((), (error,)),
        source_format=source_format,
    )


def validate_fixture_file(
    path: Path,
    *,
    requested_format: str = FORMAT_AUTO,
    require_final_shape: bool = False,
) -> tuple[int, dict[str, Any]]:
    source_format, format_error = _infer_format(path, requested_format)
    if format_error:
        return 2, _error_payload(format_error, source_format=source_format)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 2, _error_payload(f"fixture file not found: {path}", source_format=source_format)
    except OSError as error:
        return 2, _error_payload(
            f"fixture file could not be read: {path}: {error.strerror or error}",
            source_format=source_format,
        )

    fixture = load_claim_evidence_fixture_text(
        text,
        source_format=source_format,
        require_final_shape=require_final_shape,
    )
    return (0 if fixture.ok else 1), _result_payload(
        fixture,
        source_format=source_format,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, payload = validate_fixture_file(
        args.fixture_path,
        requested_format=args.format,
        require_final_shape=args.require_final_shape,
    )
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
