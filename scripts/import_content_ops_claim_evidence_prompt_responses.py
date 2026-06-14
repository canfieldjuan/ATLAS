#!/usr/bin/env python3
"""Import returned claim/evidence prompt responses into recorded-response JSON."""

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
    VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION,
    ClaimEvidenceResponse,
)


FORMAT_AUTO = "auto"
ROW_FORMAT_JSON = "json"
ROW_FORMAT_JSONL = "jsonl"
VALID_ROW_FORMATS = (FORMAT_AUTO, ROW_FORMAT_JSON, ROW_FORMAT_JSONL)
RUN_TYPE_MAIN = "main"
RUN_TYPE_STABILITY = "stability"
VALID_RUN_TYPES = (RUN_TYPE_MAIN, RUN_TYPE_STABILITY)
FORBIDDEN_RESPONSE_ROW_FIELDS = frozenset(
    {
        "claim_id",
        "claim_text",
        "difficulty",
        "evidence_quote",
        "expected_supports",
        "source_id",
    }
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import Content Ops claim/evidence prompt responses."
    )
    parser.add_argument("packets_path", type=Path)
    parser.add_argument("responses_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--packets-format",
        choices=VALID_ROW_FORMATS,
        default=FORMAT_AUTO,
        help="Prompt packet format. Auto accepts only .json and .jsonl suffixes.",
    )
    parser.add_argument(
        "--responses-format",
        choices=VALID_ROW_FORMATS,
        default=FORMAT_AUTO,
        help="Returned response-row format. Auto accepts only .json and .jsonl suffixes.",
    )
    return parser


def _infer_row_format(path: Path, requested_format: str, label: str) -> tuple[str | None, str | None]:
    if requested_format not in VALID_ROW_FORMATS:
        return None, f"{label} format must be auto, json, or jsonl"
    if requested_format != FORMAT_AUTO:
        return requested_format, None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return ROW_FORMAT_JSON, None
    if suffix == ".jsonl":
        return ROW_FORMAT_JSONL, None
    return None, f"{label} format auto-detection requires .json or .jsonl suffix"


def _read_text(path: Path, label: str) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"{label} file not found: {path}"
    except IsADirectoryError:
        return None, f"{label} path is a directory: {path}"
    except UnicodeDecodeError as error:
        return None, f"{label} file could not be read: {path}: {error}"
    except OSError as error:
        return None, f"{label} file could not be read: {path}: {error.strerror or error}"


def _decode_row_text(text: object, row_format: str, label: str) -> tuple[tuple[Mapping[str, object], ...], tuple[str, ...]]:
    if not isinstance(text, str):
        return (), (f"{label} text must be a string",)
    if row_format == ROW_FORMAT_JSON:
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            return (), (f"{label} json is malformed: {error.msg}",)
        if not isinstance(decoded, list):
            return (), (f"{label} json must decode to an array",)
        return _typed_rows(decoded, label)
    if row_format == ROW_FORMAT_JSONL:
        rows: list[object] = []
        errors: list[str] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"{label} jsonl line {line_number} is malformed: {error.msg}")
        if errors:
            return (), tuple(errors)
        return _typed_rows(rows, label)
    return (), (f"{label} format must be json or jsonl",)


def _typed_rows(rows: Sequence[object], label: str) -> tuple[tuple[Mapping[str, object], ...], tuple[str, ...]]:
    typed_rows: list[Mapping[str, object]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            errors.append(f"{label} row {index} must be an object")
            continue
        typed_rows.append(row)
    return tuple(typed_rows), tuple(errors)


def _required_text(row: Mapping[str, object], field: str, label: str, errors: list[str]) -> str | None:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label}.{field} missing")
        return None
    return value.strip()


def _packet_keys(
    rows: Sequence[Mapping[str, object]]
) -> tuple[tuple[tuple[str, str, str], ...], dict[str, tuple[str, ...]], tuple[str, ...]]:
    keys: list[tuple[str, str, str]] = []
    triples_by_model: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        label = f"packet row {index}"
        model_id = _required_text(row, "model_id", label, errors)
        triple_id = _required_text(row, "triple_id", label, errors)
        contract_version = _required_text(row, "contract_version", label, errors)
        if None in (model_id, triple_id, contract_version):
            continue
        key = (model_id or "", triple_id or "", contract_version or "")
        if key in seen:
            errors.append(f"{label}: duplicate packet key: {model_id}/{triple_id}/{contract_version}")
            continue
        if contract_version != VERIFY_CLAIM_EVIDENCE_CONTRACT_VERSION:
            errors.append(
                f"{label}: unsupported contract_version: {contract_version}"
            )
            continue
        seen.add(key)
        keys.append(key)
        triples_by_model.setdefault(model_id or "", []).append(triple_id or "")
    if not keys:
        errors.append("packets must contain at least one packet")
    return tuple(keys), {key: tuple(value) for key, value in triples_by_model.items()}, tuple(errors)


def _decode_response_row(
    row: Mapping[str, object],
    index: int,
    allowed_keys: set[tuple[str, str, str]],
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    label = f"response row {index}"
    errors: list[str] = []
    model_id = _required_text(row, "model_id", label, errors)
    triple_id = _required_text(row, "triple_id", label, errors)
    contract_version = _required_text(row, "contract_version", label, errors)
    leaked_fields = sorted(FORBIDDEN_RESPONSE_ROW_FIELDS & set(row))
    if leaked_fields:
        errors.append(
            f"{label}: response row carries fixture/label fields: "
            f"{', '.join(leaked_fields)}"
        )
    raw_run_type = row.get("run_type", RUN_TYPE_MAIN)
    if not isinstance(raw_run_type, str) or not raw_run_type.strip():
        errors.append(f"{label}.run_type missing")
        run_type = ""
    else:
        run_type = raw_run_type.strip()
        if run_type not in VALID_RUN_TYPES:
            errors.append(f"{label}.run_type must be main or stability")

    run_id = ""
    raw_run_id = row.get("run_id", "")
    if run_type == RUN_TYPE_STABILITY:
        if not isinstance(raw_run_id, str) or not raw_run_id.strip():
            errors.append(f"{label}.run_id missing")
        else:
            run_id = raw_run_id.strip()
    elif raw_run_id not in ("", None):
        errors.append(f"{label}.run_id only applies to stability rows")

    if None not in (model_id, triple_id, contract_version):
        key = (model_id or "", triple_id or "", contract_version or "")
        if key not in allowed_keys:
            errors.append(
                f"{label}: no exported packet for {model_id}/{triple_id}/{contract_version}"
            )

    response, response_errors = ClaimEvidenceResponse.from_mapping(row.get("response"))
    for response_error in response_errors:
        errors.append(f"{label}.response {response_error}")
    if errors:
        return None, tuple(errors)
    if response is None:
        return None, (f"{label}.response missing",)
    return (
        {
            "model_id": model_id or "",
            "triple_id": triple_id or "",
            "contract_version": contract_version or "",
            "run_type": run_type,
            "run_id": run_id,
            "response": asdict(response),
        },
        (),
    )


def _recorded_payload(
    packet_keys: Sequence[tuple[str, str, str]],
    triples_by_model: Mapping[str, Sequence[str]],
    response_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    allowed_keys = set(packet_keys)
    main: dict[str, dict[str, dict[str, object]]] = {}
    stability: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    errors: list[str] = []
    for index, row in enumerate(response_rows, start=1):
        decoded, row_errors = _decode_response_row(row, index, allowed_keys)
        errors.extend(row_errors)
        if decoded is None:
            continue
        model_id = str(decoded["model_id"])
        triple_id = str(decoded["triple_id"])
        if decoded["run_type"] == RUN_TYPE_MAIN:
            model_rows = main.setdefault(model_id, {})
            if triple_id in model_rows:
                errors.append(f"response row {index}: duplicate main row: {model_id}/{triple_id}")
                continue
            model_rows[triple_id] = decoded["response"]  # type: ignore[assignment]
            continue
        run_id = str(decoded["run_id"])
        run_rows = stability.setdefault(model_id, {}).setdefault(run_id, {})
        if triple_id in run_rows:
            errors.append(
                f"response row {index}: duplicate stability row: {model_id}/{run_id}/{triple_id}"
            )
            continue
        run_rows[triple_id] = decoded["response"]  # type: ignore[assignment]

    for model_id, triple_ids in triples_by_model.items():
        model_main = main.get(model_id, {})
        for triple_id in triple_ids:
            if triple_id not in model_main:
                errors.append(f"missing main response for {model_id}/{triple_id}")

    if errors:
        return None, tuple(errors)

    model_runs = [
        {
            "model_id": model_id,
            "responses": {triple_id: main[model_id][triple_id] for triple_id in triples_by_model[model_id]},
        }
        for model_id in triples_by_model
    ]
    stability_runs_by_model_id: dict[str, list[dict[str, object]]] = {}
    for model_id in sorted(stability):
        stability_runs_by_model_id[model_id] = [
            {
                "model_id": model_id,
                "responses": {
                    triple_id: stability[model_id][run_id][triple_id]
                    for triple_id in triples_by_model.get(model_id, ())
                    if triple_id in stability[model_id][run_id]
                },
            }
            for run_id in sorted(stability[model_id])
        ]
    return {
        "model_runs": model_runs,
        "stability_runs_by_model_id": stability_runs_by_model_id,
    }, ()


def _write_json(path: Path, payload: Mapping[str, object]) -> str | None:
    if path.is_symlink():
        return f"output path is a symlink: {path}"
    if path.exists() and path.is_dir():
        return f"output path is a directory: {path}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as error:
        return f"output file could not be written: {path}: {error.strerror or error}"
    return None


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left.absolute() == right.absolute()


def _error_payload(errors: Sequence[str], *, output_path: Path | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "output_path": str(output_path) if output_path is not None else "",
        "model_run_count": 0,
        "stability_model_count": 0,
        "response_count": 0,
        "errors": list(errors),
    }


def import_prompt_responses_from_files(
    packets_path: Path,
    responses_path: Path,
    output_path: Path,
    *,
    packets_format: str = FORMAT_AUTO,
    responses_format: str = FORMAT_AUTO,
) -> tuple[int, dict[str, Any]]:
    packet_format, packet_format_error = _infer_row_format(
        packets_path,
        packets_format,
        "packets",
    )
    if packet_format_error:
        return 2, _error_payload((packet_format_error,), output_path=output_path)
    response_format, response_format_error = _infer_row_format(
        responses_path,
        responses_format,
        "responses",
    )
    if response_format_error:
        return 2, _error_payload((response_format_error,), output_path=output_path)
    if _same_path(output_path, packets_path):
        return 2, _error_payload(
            ("output path must differ from packets path",),
            output_path=output_path,
        )
    if _same_path(output_path, responses_path):
        return 2, _error_payload(
            ("output path must differ from responses path",),
            output_path=output_path,
        )

    packets_text, packets_error = _read_text(packets_path, "packets")
    if packets_error:
        return 2, _error_payload((packets_error,), output_path=output_path)
    responses_text, responses_error = _read_text(responses_path, "responses")
    if responses_error:
        return 2, _error_payload((responses_error,), output_path=output_path)

    packet_rows, packet_decode_errors = _decode_row_text(
        packets_text,
        packet_format or "",
        "packets",
    )
    if packet_decode_errors:
        return 1, _error_payload(packet_decode_errors, output_path=output_path)
    response_rows, response_decode_errors = _decode_row_text(
        responses_text,
        response_format or "",
        "responses",
    )
    if response_decode_errors:
        return 1, _error_payload(response_decode_errors, output_path=output_path)

    packet_keys, triples_by_model, packet_errors = _packet_keys(packet_rows)
    if packet_errors:
        return 1, _error_payload(packet_errors, output_path=output_path)
    recorded_payload, response_errors = _recorded_payload(
        packet_keys,
        triples_by_model,
        response_rows,
    )
    if response_errors or recorded_payload is None:
        return 1, _error_payload(response_errors, output_path=output_path)

    write_error = _write_json(output_path, recorded_payload)
    if write_error:
        return 2, _error_payload((write_error,), output_path=output_path)

    model_runs = recorded_payload["model_runs"]
    stability = recorded_payload["stability_runs_by_model_id"]
    response_count = sum(
        len(run["responses"]) for run in model_runs if isinstance(run, Mapping)
    )
    if isinstance(stability, Mapping):
        for runs in stability.values():
            if isinstance(runs, list):
                response_count += sum(
                    len(run["responses"]) for run in runs if isinstance(run, Mapping)
                )
    return (
        0,
        {
            "ok": True,
            "output_path": str(output_path),
            "model_run_count": len(model_runs) if isinstance(model_runs, list) else 0,
            "stability_model_count": len(stability) if isinstance(stability, Mapping) else 0,
            "response_count": response_count,
            "errors": [],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, payload = import_prompt_responses_from_files(
        args.packets_path,
        args.responses_path,
        args.output_path,
        packets_format=args.packets_format,
        responses_format=args.responses_format,
    )
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
