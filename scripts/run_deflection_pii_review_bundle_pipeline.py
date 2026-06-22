#!/usr/bin/env python3
"""Build, score, and promote a deflection PII review-bundle candidate."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for candidate in (ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import build_deflection_pii_surrogate_eval_corpus as build_cli  # noqa: E402
import promote_deflection_pii_review_bundle as promote_cli  # noqa: E402
import score_deflection_pii_recall as score_cli  # noqa: E402


PIPELINE_SCHEMA_VERSION = "deflection_pii_review_bundle_pipeline.v1"
RESERVED_BUNDLE_NAMES = frozenset({
    build_cli.REVIEW_BUNDLE_ARTIFACT_NAME,
    build_cli.REVIEW_BUNDLE_SUMMARY_NAME,
    build_cli.REVIEW_BUNDLE_MARKDOWN_NAME,
    build_cli.REVIEW_BUNDLE_MANIFEST_NAME,
    score_cli.REVIEW_BUNDLE_SCORE_NAME,
    score_cli.REVIEW_BUNDLE_SCORE_MARKDOWN_NAME,
})


@dataclass(frozen=True)
class StepResult:
    name: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    preflight_errors = _preflight_errors(args)
    if preflight_errors:
        return _write_failure(
            StepResult(
                name="preflight",
                exit_code=1,
                stdout="",
                stderr=json.dumps({
                    "errors": [{"code": code} for code in preflight_errors],
                }),
            )
        )
    _clear_reserved_bundle_outputs(args.review_bundle_dir)

    build_args = [
        str(args.input),
        "--review-bundle-dir",
        str(args.review_bundle_dir),
    ]
    if args.pretty:
        build_args.append("--pretty")
    build = _run_step("build", build_cli.main, build_args)
    if not build.ok:
        return _write_failure(build)

    score = _run_step(
        "score",
        score_cli.main,
        ["--review-bundle-dir", str(args.review_bundle_dir)],
    )
    if not score.ok:
        return _write_failure(
            score,
            fallback_errors=_score_error_codes(args.review_bundle_dir),
        )

    promote_args = [
        str(args.review_bundle_dir),
        "--output",
        str(args.candidate_output),
    ]
    if args.force:
        promote_args.append("--force")
    promote = _run_step("promote", promote_cli.main, promote_args)
    if not promote.ok:
        return _write_failure(promote)

    promote_payload = _json_object(promote.stdout)
    if not promote_payload:
        return _write_failure(
            StepResult(
                name="promote",
                exit_code=1,
                stdout="",
                stderr=json.dumps({
                    "errors": [{"code": "promote_payload_unreadable"}],
                }),
            )
        )

    payload = {
        "ok": True,
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "review_bundle_dir": str(args.review_bundle_dir),
        "candidate_output": str(args.candidate_output),
        "steps": {
            "build": "ok",
            "score": "ok",
            "promote": "ok",
        },
        "ticket_count": _safe_int(promote_payload.get("ticket_count")),
        "label_count": _safe_int(promote_payload.get("label_count")),
        "score_status": "ok",
        "headline": _safe_mapping(promote_payload.get("headline")),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local labeled-source JSON.")
    parser.add_argument(
        "--review-bundle-dir",
        type=Path,
        required=True,
        help="Directory to write the sanitized review bundle and score artifacts.",
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        required=True,
        help="Validated surrogate corpus candidate output path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing candidate output after validation.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write pretty JSON bundle artifacts during the build step.",
    )
    return parser.parse_args(argv)


def _preflight_errors(args: argparse.Namespace) -> tuple[str, ...]:
    input_path = args.input.resolve()
    candidate_path = args.candidate_output.resolve()
    reserved_bundle_paths = {
        path.resolve()
        for path in _reserved_bundle_paths(args.review_bundle_dir)
    }
    errors: list[str] = []
    if input_path in reserved_bundle_paths:
        errors.append("input_reserved_bundle_artifact")
    if candidate_path == input_path:
        errors.append("candidate_output_same_as_input")
    if candidate_path in reserved_bundle_paths:
        errors.append("candidate_output_reserved_bundle_artifact")
    return tuple(errors)


def _reserved_bundle_paths(bundle_dir: Path) -> tuple[Path, ...]:
    return tuple(bundle_dir / name for name in sorted(RESERVED_BUNDLE_NAMES))


def _clear_reserved_bundle_outputs(bundle_dir: Path) -> None:
    for path in _reserved_bundle_paths(bundle_dir):
        path.unlink(missing_ok=True)


def _run_step(
    name: str,
    func: Callable[[list[str] | None], int],
    argv: list[str],
) -> StepResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exit_code = func(argv)
        except SystemExit as exc:
            exit_code = _system_exit_code(exc)
    return StepResult(
        name=name,
        exit_code=exit_code,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _system_exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    return 1


def _write_failure(
    result: StepResult,
    *,
    fallback_errors: Sequence[str] = (),
) -> int:
    errors = _error_codes(result.stderr) or _error_codes(result.stdout)
    if not errors and fallback_errors:
        errors = tuple(fallback_errors)
    if not errors:
        errors = (f"{result.name}_failed",)
    print(
        json.dumps(
            {
                "ok": False,
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "failed_step": result.name,
                "exit_code": result.exit_code,
                "errors": [{"code": code} for code in errors],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return result.exit_code if result.exit_code else 1


def _score_error_codes(bundle_dir: Path) -> tuple[str, ...]:
    payload = _read_json(bundle_dir / score_cli.REVIEW_BUNDLE_SCORE_NAME)
    codes = payload.get("blocking_error_codes") if isinstance(payload, Mapping) else None
    if isinstance(codes, Sequence) and not isinstance(codes, (str, bytes)):
        return tuple(str(code) for code in codes if isinstance(code, str))
    return ()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "blocking_error_codes": [
                f"score_summary_{exc.__class__.__name__}",
            ],
        }
    return value if isinstance(value, Mapping) else {}


def _error_codes(text: str) -> tuple[str, ...]:
    payload = _json_object(text)
    if not payload:
        return ()
    errors = payload.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        return tuple(
            str(error.get("code"))
            for error in errors
            if isinstance(error, Mapping) and isinstance(error.get("code"), str)
        )
    codes = payload.get("blocking_error_codes")
    if isinstance(codes, Sequence) and not isinstance(codes, (str, bytes)):
        return tuple(str(code) for code in codes if isinstance(code, str))
    return ()


def _json_object(text: str) -> Mapping[str, Any]:
    decode_error_code = ""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            decode_error_code = f"child_json_{exc.__class__.__name__}"
            continue
        if isinstance(value, Mapping):
            return value
    if decode_error_code:
        return {"errors": [{"code": decode_error_code}]}
    return {}


def _safe_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): value[key] for key in sorted(value)}


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


if __name__ == "__main__":
    raise SystemExit(main())
