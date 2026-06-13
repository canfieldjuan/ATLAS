#!/usr/bin/env python3
"""Build claim/evidence benchmark artifacts from recorded responses."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.claim_evidence_benchmark import (
    FIXTURE_FORMAT_JSON,
    FIXTURE_FORMAT_JSONL,
    VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION,
    ClaimEvidenceModelRun,
    ClaimEvidenceResponse,
    ClaimEvidenceRunRow,
    build_claim_evidence_result_artifact,
    load_claim_evidence_fixture_text,
    write_claim_evidence_result_artifact_files,
)


FORMAT_AUTO = "auto"
VALID_FORMATS = (FORMAT_AUTO, FIXTURE_FORMAT_JSON, FIXTURE_FORMAT_JSONL)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write Content Ops claim/evidence benchmark result artifacts."
    )
    parser.add_argument("fixture_path", type=Path)
    parser.add_argument("responses_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--fixture-format",
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


def _infer_fixture_format(
    path: Path,
    requested_format: str,
) -> tuple[str | None, str | None]:
    if requested_format != FORMAT_AUTO:
        return requested_format, None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return FIXTURE_FORMAT_JSON, None
    if suffix == ".jsonl":
        return FIXTURE_FORMAT_JSONL, None
    return None, "fixture format auto-detection requires .json or .jsonl suffix"


def _read_text(path: Path, label: str) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"{label} file not found: {path}"
    except OSError as error:
        return None, f"{label} file could not be read: {path}: {error.strerror or error}"


def _decode_response_runs_text(
    text: object,
) -> tuple[
    tuple[ClaimEvidenceModelRun, ...],
    dict[str, tuple[ClaimEvidenceModelRun, ...]],
    tuple[str, ...],
]:
    if not isinstance(text, str):
        return (), {}, ("responses text must be a string",)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        return (), {}, (f"responses json is malformed: {error.msg}",)
    if not isinstance(decoded, Mapping):
        return (), {}, ("responses json must decode to an object",)

    errors: list[str] = []
    model_runs = _decode_model_runs(decoded.get("model_runs"), "model_runs", errors)
    stability = _decode_stability_runs(
        decoded.get("stability_runs_by_model_id", {}),
        errors,
    )
    return model_runs, stability, tuple(errors)


def _decode_model_runs(
    value: object,
    label: str,
    errors: list[str],
) -> tuple[ClaimEvidenceModelRun, ...]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a non-empty list")
        return ()
    runs: list[ClaimEvidenceModelRun] = []
    for index, item in enumerate(value, start=1):
        run = _decode_model_run(item, f"{label}[{index}]", errors)
        if run is not None:
            runs.append(run)
    return tuple(runs)


def _decode_stability_runs(
    value: object,
    errors: list[str],
) -> dict[str, tuple[ClaimEvidenceModelRun, ...]]:
    if not isinstance(value, Mapping):
        errors.append("stability_runs_by_model_id must be an object")
        return {}
    result: dict[str, tuple[ClaimEvidenceModelRun, ...]] = {}
    for raw_model_id, raw_runs in value.items():
        if not isinstance(raw_model_id, str) or not raw_model_id.strip():
            errors.append("stability model_id missing")
            continue
        normalized_model_id = raw_model_id.strip()
        if normalized_model_id in result:
            errors.append(f"stability model_id duplicated: {normalized_model_id}")
            continue
        result[normalized_model_id] = _decode_model_runs(
            raw_runs,
            f"stability_runs_by_model_id[{raw_model_id}]",
            errors,
        )
    return result


def _decode_model_run(
    value: object,
    label: str,
    errors: list[str],
) -> ClaimEvidenceModelRun | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return None
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append(f"{label}.model_id missing")
        return None
    responses = value.get("responses")
    if not isinstance(responses, Mapping) or not responses:
        errors.append(f"{label}.responses must be a non-empty object")
        return None

    rows: list[ClaimEvidenceRunRow] = []
    run_errors: list[str] = []
    for raw_triple_id, raw_response in responses.items():
        if not isinstance(raw_triple_id, str) or not raw_triple_id.strip():
            run_errors.append("response triple_id missing")
            continue
        response, response_errors = ClaimEvidenceResponse.from_mapping(raw_response)
        rows.append(
            ClaimEvidenceRunRow(
                model_id=model_id.strip(),
                triple_id=raw_triple_id.strip(),
                contract_version=VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION,
                response=response,
                errors=response_errors,
            )
        )
    return ClaimEvidenceModelRun(model_id.strip(), tuple(rows), tuple(run_errors))


def _error_payload(errors: Sequence[str], *, output_dir: Path | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "go_no_go": "no_go",
        "output_dir": str(output_dir) if output_dir is not None else "",
        "files": [],
        "errors": list(errors),
        "artifact_errors": [],
        "verdict_failures": [],
    }


def build_claim_evidence_artifact_from_files(
    fixture_path: Path,
    responses_path: Path,
    output_dir: Path,
    *,
    fixture_format: str = FORMAT_AUTO,
    require_final_shape: bool = False,
) -> tuple[int, dict[str, Any]]:
    source_format, format_error = _infer_fixture_format(fixture_path, fixture_format)
    if format_error:
        return 2, _error_payload((format_error,), output_dir=output_dir)

    fixture_text, fixture_error = _read_text(fixture_path, "fixture")
    if fixture_error:
        return 2, _error_payload((fixture_error,), output_dir=output_dir)
    responses_text, responses_error = _read_text(responses_path, "responses")
    if responses_error:
        return 2, _error_payload((responses_error,), output_dir=output_dir)

    fixture = load_claim_evidence_fixture_text(
        fixture_text,
        source_format=source_format,
        require_final_shape=require_final_shape,
    )
    if not fixture.ok:
        return 1, _error_payload(fixture.errors, output_dir=output_dir)

    model_runs, stability_runs, response_errors = _decode_response_runs_text(responses_text)
    if response_errors:
        return 1, _error_payload(response_errors, output_dir=output_dir)

    artifact = build_claim_evidence_result_artifact(
        fixture.triples,
        model_runs,
        stability_runs_by_model_id=stability_runs,
    )
    write_result = write_claim_evidence_result_artifact_files(artifact, output_dir)
    payload = {
        "ok": artifact.ok and write_result.ok,
        "go_no_go": artifact.go_no_go,
        "output_dir": write_result.output_dir,
        "files": [asdict(file) for file in write_result.files],
        "errors": list(write_result.errors),
        "artifact_errors": list(artifact.errors),
        "verdict_failures": list(artifact.verdict.failure_reasons),
    }
    if write_result.errors:
        return 2, payload
    return (0 if artifact.ok else 1), payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, payload = build_claim_evidence_artifact_from_files(
        args.fixture_path,
        args.responses_path,
        args.output_dir,
        fixture_format=args.fixture_format,
        require_final_shape=args.require_final_shape,
    )
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
