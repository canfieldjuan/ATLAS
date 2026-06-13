#!/usr/bin/env python3
"""Export claim/evidence benchmark prompt packets for model witnesses."""

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
    ClaimEvidenceTriple,
    build_claim_evidence_prompt_contract,
    load_claim_evidence_fixture_text,
)


FORMAT_AUTO = "auto"
PACKET_FORMAT_JSON = "json"
PACKET_FORMAT_JSONL = "jsonl"
VALID_FIXTURE_FORMATS = (FORMAT_AUTO, FIXTURE_FORMAT_JSON, FIXTURE_FORMAT_JSONL)
VALID_PACKET_FORMATS = (FORMAT_AUTO, PACKET_FORMAT_JSON, PACKET_FORMAT_JSONL)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Content Ops claim/evidence prompt packets."
    )
    parser.add_argument("fixture_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Model id to include in the prompt packet export. Repeat per model.",
    )
    parser.add_argument(
        "--fixture-format",
        choices=VALID_FIXTURE_FORMATS,
        default=FORMAT_AUTO,
        help="Fixture format. Auto accepts only .json and .jsonl suffixes.",
    )
    parser.add_argument(
        "--output-format",
        choices=VALID_PACKET_FORMATS,
        default=FORMAT_AUTO,
        help="Prompt packet format. Auto accepts only .json and .jsonl suffixes.",
    )
    parser.add_argument(
        "--require-final-shape",
        action="store_true",
        help="Require the final #1435 15/15/10 benchmark composition.",
    )
    return parser


def _infer_fixture_format(path: Path, requested_format: str) -> tuple[str | None, str | None]:
    if requested_format not in VALID_FIXTURE_FORMATS:
        return None, "fixture format must be auto, json, or jsonl"
    if requested_format != FORMAT_AUTO:
        return requested_format, None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return FIXTURE_FORMAT_JSON, None
    if suffix == ".jsonl":
        return FIXTURE_FORMAT_JSONL, None
    return None, "fixture format auto-detection requires .json or .jsonl suffix"


def _infer_packet_format(path: Path, requested_format: str) -> tuple[str | None, str | None]:
    if requested_format not in VALID_PACKET_FORMATS:
        return None, "output format must be auto, json, or jsonl"
    if requested_format != FORMAT_AUTO:
        return requested_format, None
    suffix = path.suffix.lower()
    if suffix == ".json":
        return PACKET_FORMAT_JSON, None
    if suffix == ".jsonl":
        return PACKET_FORMAT_JSONL, None
    return None, "output format auto-detection requires .json or .jsonl suffix"


def _read_text(path: Path, label: str) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"{label} file not found: {path}"
    except IsADirectoryError:
        return None, f"{label} path is a directory: {path}"
    except OSError as error:
        return None, f"{label} file could not be read: {path}: {error.strerror or error}"
    except UnicodeDecodeError as error:
        return None, f"{label} file could not be read: {path}: {error}"


def _normalize_model_ids(model_ids: Sequence[object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(model_ids, Sequence) or isinstance(model_ids, (str, bytes)):
        return (), ("model_ids must be a sequence",)
    errors: list[str] = []
    normalized: list[str] = []
    seen: dict[str, int] = {}
    for index, model_id in enumerate(model_ids, start=1):
        if not isinstance(model_id, str) or not model_id.strip():
            errors.append(f"model_id {index} missing")
            continue
        value = model_id.strip()
        if value in seen:
            errors.append(f"model_id duplicated from position {seen[value]}: {value}")
            continue
        seen[value] = index
        normalized.append(value)
    if not normalized:
        errors.append("at least one model_id is required")
    return tuple(normalized), tuple(errors)


def _packet_for_triple(model_id: str, triple: ClaimEvidenceTriple) -> dict[str, object]:
    contract = build_claim_evidence_prompt_contract(triple)
    return {
        "model_id": model_id,
        "triple_id": triple.triple_id,
        "claim_id": triple.claim_id,
        "source_id": triple.source_id,
        "difficulty": triple.difficulty,
        "contract_version": contract.contract_version,
        "prompt": contract.prompt,
        "response_schema": contract.response_schema,
    }


def _render_packets(packets: Sequence[dict[str, object]], output_format: str) -> str:
    if output_format == PACKET_FORMAT_JSONL:
        return "\n".join(json.dumps(packet, sort_keys=True) for packet in packets) + "\n"
    return json.dumps(list(packets), indent=2, sort_keys=True) + "\n"


def _write_packets(path: Path, content: str) -> str | None:
    if path.is_symlink():
        return f"output path is a symlink: {path}"
    if path.exists() and path.is_dir():
        return f"output path is a directory: {path}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        return f"output file could not be written: {path}: {error.strerror or error}"
    return None


def _error_payload(
    errors: Sequence[str],
    *,
    output_path: Path | None = None,
    output_format: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "output_path": str(output_path) if output_path is not None else "",
        "output_format": output_format or "",
        "packet_count": 0,
        "triple_count": 0,
        "model_count": 0,
        "model_ids": [],
        "errors": list(errors),
    }


def export_prompt_packets_from_fixture_file(
    fixture_path: Path,
    output_path: Path,
    *,
    model_ids: Sequence[object],
    fixture_format: str = FORMAT_AUTO,
    output_format: str = FORMAT_AUTO,
    require_final_shape: bool = False,
) -> tuple[int, dict[str, Any]]:
    source_format, fixture_format_error = _infer_fixture_format(
        fixture_path,
        fixture_format,
    )
    if fixture_format_error:
        return 2, _error_payload((fixture_format_error,), output_path=output_path)

    packet_format, output_format_error = _infer_packet_format(output_path, output_format)
    if output_format_error:
        return 2, _error_payload((output_format_error,), output_path=output_path)

    normalized_model_ids, model_id_errors = _normalize_model_ids(model_ids)
    if model_id_errors:
        return 2, _error_payload(
            model_id_errors,
            output_path=output_path,
            output_format=packet_format,
        )

    fixture_text, read_error = _read_text(fixture_path, "fixture")
    if read_error:
        return 2, _error_payload(
            (read_error,),
            output_path=output_path,
            output_format=packet_format,
        )

    fixture = load_claim_evidence_fixture_text(
        fixture_text,
        source_format=source_format,
        require_final_shape=require_final_shape,
    )
    if not fixture.ok:
        return 1, _error_payload(
            fixture.errors,
            output_path=output_path,
            output_format=packet_format,
        )

    packets = tuple(
        _packet_for_triple(model_id, triple)
        for model_id in normalized_model_ids
        for triple in fixture.triples
    )
    write_error = _write_packets(output_path, _render_packets(packets, packet_format or ""))
    if write_error:
        return 2, _error_payload(
            (write_error,),
            output_path=output_path,
            output_format=packet_format,
        )

    return (
        0,
        {
            "ok": True,
            "output_path": str(output_path),
            "output_format": packet_format,
            "packet_count": len(packets),
            "triple_count": len(fixture.triples),
            "model_count": len(normalized_model_ids),
            "model_ids": list(normalized_model_ids),
            "errors": [],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    exit_code, payload = export_prompt_packets_from_fixture_file(
        args.fixture_path,
        args.output_path,
        model_ids=args.model_id,
        fixture_format=args.fixture_format,
        output_format=args.output_format,
        require_final_shape=args.require_final_shape,
    )
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
