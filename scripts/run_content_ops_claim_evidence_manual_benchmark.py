#!/usr/bin/env python3
"""Run the manual claim/evidence benchmark from returned prompt responses."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from import_content_ops_claim_evidence_prompt_responses import (  # noqa: E402
    FORMAT_AUTO as ROW_FORMAT_AUTO,
    ROW_FORMAT_JSON,
    ROW_FORMAT_JSONL,
    import_prompt_responses_from_files,
)
from run_content_ops_claim_evidence_artifact import (  # noqa: E402
    VALID_FORMATS as FIXTURE_FORMATS,
    build_claim_evidence_artifact_from_files,
)


RECORDED_RESPONSES_FILENAME = "recorded_responses.json"
RESULT_FILENAMES = frozenset(
    {
        "claim_evidence_result.json",
        "claim_evidence_result.md",
    }
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the manual Content Ops claim/evidence benchmark."
    )
    parser.add_argument("fixture_path", type=Path)
    parser.add_argument("packets_path", type=Path)
    parser.add_argument("responses_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--recorded-responses-output",
        type=Path,
        default=None,
        help="Optional path for the intermediate recorded-response JSON.",
    )
    parser.add_argument(
        "--fixture-format",
        choices=FIXTURE_FORMATS,
        default=ROW_FORMAT_AUTO,
    )
    parser.add_argument(
        "--packets-format",
        choices=(ROW_FORMAT_AUTO, ROW_FORMAT_JSON, ROW_FORMAT_JSONL),
        default=ROW_FORMAT_AUTO,
    )
    parser.add_argument(
        "--responses-format",
        choices=(ROW_FORMAT_AUTO, ROW_FORMAT_JSON, ROW_FORMAT_JSONL),
        default=ROW_FORMAT_AUTO,
    )
    parser.add_argument(
        "--require-final-shape",
        action="store_true",
        help="Require the final #1435 15/15/10 benchmark composition.",
    )
    return parser


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left.absolute() == right.absolute()


def _error_payload(
    errors: Sequence[str],
    *,
    output_dir: Path,
    recorded_output: Path,
    import_payload: Mapping[str, object] | None = None,
    artifact_payload: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "go_no_go": "no_go",
        "output_dir": str(output_dir),
        "recorded_responses_output": str(recorded_output),
        "import_payload": dict(import_payload or {}),
        "artifact_payload": dict(artifact_payload or {}),
        "errors": list(errors),
    }


def _path_errors(
    fixture_path: Path,
    packets_path: Path,
    responses_path: Path,
    output_dir: Path,
    recorded_output: Path,
) -> tuple[str, ...]:
    errors: list[str] = []
    if output_dir.is_symlink():
        errors.append(f"output_dir is a symlink: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        errors.append(f"output_dir is not a directory: {output_dir}")
    if recorded_output.is_symlink():
        errors.append(f"recorded responses output is a symlink: {recorded_output}")
    if _same_path(recorded_output, fixture_path):
        errors.append("recorded responses output must differ from fixture path")
    if _same_path(recorded_output, packets_path):
        errors.append("recorded responses output must differ from packets path")
    if _same_path(recorded_output, responses_path):
        errors.append("recorded responses output must differ from responses path")
    for filename in RESULT_FILENAMES:
        artifact_path = output_dir / filename
        if artifact_path.exists() or artifact_path.is_symlink():
            errors.append(
                f"output_dir already contains benchmark artifact file: {artifact_path}"
            )
        if _same_path(recorded_output, artifact_path):
            errors.append(
                f"recorded responses output must not overwrite artifact file: {artifact_path}"
            )
    return tuple(errors)


def run_manual_benchmark_from_files(
    fixture_path: Path,
    packets_path: Path,
    responses_path: Path,
    output_dir: Path,
    *,
    recorded_responses_output: Path | None = None,
    fixture_format: str = ROW_FORMAT_AUTO,
    packets_format: str = ROW_FORMAT_AUTO,
    responses_format: str = ROW_FORMAT_AUTO,
    require_final_shape: bool = False,
) -> tuple[int, dict[str, Any]]:
    recorded_output = recorded_responses_output or (
        output_dir / RECORDED_RESPONSES_FILENAME
    )
    path_errors = _path_errors(
        fixture_path,
        packets_path,
        responses_path,
        output_dir,
        recorded_output,
    )
    if path_errors:
        return 2, _error_payload(
            path_errors,
            output_dir=output_dir,
            recorded_output=recorded_output,
        )

    import_exit, import_payload = import_prompt_responses_from_files(
        packets_path,
        responses_path,
        recorded_output,
        packets_format=packets_format,
        responses_format=responses_format,
    )
    if import_exit != 0:
        return import_exit, _error_payload(
            ("response import failed",),
            output_dir=output_dir,
            recorded_output=recorded_output,
            import_payload=import_payload,
        )

    artifact_exit, artifact_payload = build_claim_evidence_artifact_from_files(
        fixture_path,
        recorded_output,
        output_dir,
        fixture_format=fixture_format,
        require_final_shape=require_final_shape,
    )
    ok = artifact_exit == 0 and bool(artifact_payload.get("ok"))
    payload = {
        "ok": ok,
        "go_no_go": artifact_payload.get("go_no_go", "no_go"),
        "output_dir": str(output_dir),
        "recorded_responses_output": str(recorded_output),
        "import_payload": import_payload,
        "artifact_payload": artifact_payload,
        "errors": [] if ok else ["benchmark artifact failed"],
    }
    return artifact_exit, payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, payload = run_manual_benchmark_from_files(
        args.fixture_path,
        args.packets_path,
        args.responses_path,
        args.output_dir,
        recorded_responses_output=args.recorded_responses_output,
        fixture_format=args.fixture_format,
        packets_format=args.packets_format,
        responses_format=args.responses_format,
        require_final_shape=args.require_final_shape,
    )
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
