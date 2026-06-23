#!/usr/bin/env python3
"""Generate or check frontend TypeScript types from the deflection contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    deflection_report_model_contract_shape,
)


DEFAULT_OUTPUT = ROOT / "portfolio-ui/src/types/deflectionSnapshot.ts"
DEFAULT_API_OUTPUT = ROOT / "portfolio-ui/api/content-ops/deflection/snapshot-contract.js"

TYPE_BY_FIELD = {
    "answer": "string",
    "answer_evidence_status": "string",
    "body_withheld": "boolean",
    "customer_wording": "string",
    "drafted_answer_count": "number",
    "full_answer": "DeflectionSnapshotTeaserFullAnswer | null",
    "generated": "number",
    "no_proven_answer_count": "number",
    "non_repeat_ticket_count": "number",
    "previews": "DeflectionSnapshotTeaserPreview[]",
    "question": "string",
    "rank": "number",
    "repeat_ticket_count": "number",
    "resolution_evidence_scope": "string",
    "source_count": "number",
    "source_date_end": "string | null",
    "source_date_start": "string | null",
    "source_window_days": "number | null",
    "step_count": "number",
    "steps": "string[]",
    "support_ticket_resolution_evidence_count": "number",
    "support_ticket_resolution_evidence_present": "boolean",
    "ticket_count": "number",
    "weighted_frequency": "number",
}

TYPE_NAME_BY_SNAPSHOT_FIELD = {
    "summary": "DeflectionSnapshotSummary",
    "top_questions": "DeflectionSnapshotTopQuestion",
    "locked_questions": "DeflectionSnapshotLockedQuestion",
    "top_blind_spots": "DeflectionSnapshotTopBlindSpot",
}

RESULT_PAGE_SNAPSHOT_FIELDS = ("summary", "top_questions", "top_blind_spots")


def _indent(lines: Sequence[str], spaces: int = 2) -> list[str]:
    prefix = " " * spaces
    return [f"{prefix}{line}" if line else "" for line in lines]


def _field_type(field: str) -> str:
    try:
        return TYPE_BY_FIELD[field]
    except KeyError as exc:
        raise ValueError(f"missing TypeScript type mapping for snapshot field: {field}") from exc


def _object_type(
    name: str,
    fields: Sequence[str],
    *,
    optional_fields: Sequence[str] = (),
    export: bool = True,
) -> list[str]:
    optional = set(optional_fields)
    prefix = "export " if export else ""
    lines = [f"{prefix}type {name} = {{"]
    lines.extend(
        _indent([
            f"{field}{'?' if field in optional else ''}: {_field_type(field)};"
            for field in fields
        ])
    )
    lines.append("};")
    return lines


def _projection_entry(
    projection: Mapping[str, Any],
    field: str,
) -> Mapping[str, Any]:
    entries = projection.get("fields")
    if not isinstance(entries, list):
        raise ValueError("snapshot_projection.fields must be a list")
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("field") == field:
            return entry
    raise ValueError(f"snapshot_projection missing field entry: {field}")


def _projected_fields(entry: Mapping[str, Any], key: str = "projected_fields") -> list[str]:
    fields = entry.get(key)
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise ValueError(f"snapshot projection entry {entry.get('field')!r} has invalid {key}")
    return fields


def _optional_projected_fields(entry: Mapping[str, Any]) -> list[str]:
    fields = entry.get("optional_projected_fields", [])
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise ValueError(
            f"snapshot projection entry {entry.get('field')!r} has invalid optional_projected_fields"
        )
    projected = set(_projected_fields(entry))
    unknown = [field for field in fields if field not in projected]
    if unknown:
        raise ValueError(
            f"optional projected fields are not projected for {entry.get('field')!r}: "
            + ", ".join(unknown)
        )
    return fields


def _render_field_tuple(name: str, fields: Sequence[str]) -> list[str]:
    quoted = ", ".join(f'"{field}"' for field in fields)
    return [f"export const {name} = [{quoted}] as const;"]


def _render_field_array(name: str, fields: Sequence[str]) -> list[str]:
    quoted = ", ".join(f'"{field}"' for field in fields)
    return [f"export const {name} = Object.freeze([{quoted}]);"]


def _snapshot_projection_metadata(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    shape = contract if contract is not None else deflection_report_model_contract_shape()
    projection = shape.get("snapshot_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("contract missing snapshot_projection")
    schema_version = projection.get("schema_version")
    if not isinstance(schema_version, str):
        raise ValueError("snapshot_projection.schema_version must be a string")

    top_level_fields = projection.get("top_level_fields")
    if not isinstance(top_level_fields, list) or not all(
        isinstance(field, str) for field in top_level_fields
    ):
        raise ValueError("snapshot_projection.top_level_fields must be a string list")

    summary_entry = _projection_entry(projection, "summary")
    summary = _projected_fields(summary_entry)
    summary_optional = _optional_projected_fields(summary_entry)
    top_questions = _projected_fields(_projection_entry(projection, "top_questions"))
    locked_questions = _projected_fields(_projection_entry(projection, "locked_questions"))
    top_blind_spots = _projected_fields(_projection_entry(projection, "top_blind_spots"))
    teaser = _projection_entry(projection, "teaser")
    teaser_fields = _projected_fields(teaser)
    teaser_full_answer = _projected_fields(teaser, "full_answer_fields")
    teaser_previews = _projected_fields(teaser, "preview_fields")
    return {
        "schema_version": schema_version,
        "top_level_fields": top_level_fields,
        "summary": summary,
        "summary_optional": summary_optional,
        "top_questions": top_questions,
        "locked_questions": locked_questions,
        "top_blind_spots": top_blind_spots,
        "teaser_fields": teaser_fields,
        "teaser_full_answer": teaser_full_answer,
        "teaser_previews": teaser_previews,
    }


def render_types(contract: Mapping[str, Any] | None = None) -> str:
    metadata = _snapshot_projection_metadata(contract)
    schema_version = metadata["schema_version"]
    top_level_fields = metadata["top_level_fields"]
    summary = metadata["summary"]
    summary_optional = metadata["summary_optional"]
    top_questions = metadata["top_questions"]
    locked_questions = metadata["locked_questions"]
    top_blind_spots = metadata["top_blind_spots"]
    teaser_fields = metadata["teaser_fields"]
    teaser_full_answer = metadata["teaser_full_answer"]
    teaser_previews = metadata["teaser_previews"]

    generated: list[str] = [
        "/*",
        " * Generated by scripts/generate_deflection_frontend_contract_types.py.",
        " * Do not edit by hand; run the generator after backend contract changes.",
        " */",
        "",
        f'export const DEFLECTION_SNAPSHOT_SCHEMA_VERSION = "{schema_version}" as const;',
        "",
    ]
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_TOP_LEVEL_FIELDS", top_level_fields))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_RESULT_PAGE_SNAPSHOT_FIELDS", RESULT_PAGE_SNAPSHOT_FIELDS))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_SUMMARY_FIELDS", summary))
    generated.append("")
    generated.extend(
        _render_field_tuple("DEFLECTION_SNAPSHOT_SUMMARY_OPTIONAL_FIELDS", summary_optional)
    )
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_TOP_QUESTION_FIELDS", top_questions))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_LOCKED_QUESTION_FIELDS", locked_questions))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_TOP_BLIND_SPOT_FIELDS", top_blind_spots))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_TEASER_FIELDS", teaser_fields))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_TEASER_FULL_ANSWER_FIELDS", teaser_full_answer))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_SNAPSHOT_TEASER_PREVIEW_FIELDS", teaser_previews))
    generated.append("")
    generated.extend(
        _object_type(
            "DeflectionSnapshotSummary",
            summary,
            optional_fields=summary_optional,
        )
    )
    generated.append("")
    generated.extend(_object_type("DeflectionSnapshotTopQuestion", top_questions))
    generated.append("")
    generated.extend(_object_type("DeflectionSnapshotLockedQuestion", locked_questions))
    generated.append("")
    generated.extend(_object_type("DeflectionSnapshotTopBlindSpot", top_blind_spots))
    generated.append("")
    generated.extend(_object_type("DeflectionSnapshotTeaserFullAnswer", teaser_full_answer))
    generated.append("")
    generated.extend(_object_type("DeflectionSnapshotTeaserPreview", teaser_previews))
    generated.append("")
    generated.extend(_object_type("DeflectionSnapshotTeaser", teaser_fields))
    generated.append("")
    generated.append("export type DeflectionSnapshot = {")
    for field in top_level_fields:
        if field == "teaser":
            generated.extend(_indent(["teaser: DeflectionSnapshotTeaser;"]))
            continue
        type_name = TYPE_NAME_BY_SNAPSHOT_FIELD.get(field)
        if type_name is None:
            raise ValueError(f"missing TypeScript object type for snapshot field: {field}")
        suffix = "" if field == "summary" else "[]"
        generated.extend(_indent([f"{field}: {type_name}{suffix};"]))
    generated.append("};")
    generated.append("")
    generated.append(
        "export type DeflectionResultPageSnapshot = Pick<"
        'DeflectionSnapshot, "summary" | "top_questions" | "top_blind_spots">;'
    )
    generated.append("")
    return "\n".join(generated)


def render_api_contract(contract: Mapping[str, Any] | None = None) -> str:
    metadata = _snapshot_projection_metadata(contract)
    schema_version = metadata["schema_version"]
    summary = metadata["summary"]
    summary_optional = metadata["summary_optional"]
    top_questions = metadata["top_questions"]
    top_blind_spots = metadata["top_blind_spots"]

    generated: list[str] = [
        "/*",
        " * Generated by scripts/generate_deflection_frontend_contract_types.py.",
        " * Do not edit by hand; run the generator after backend contract changes.",
        " */",
        "",
        f'export const DEFLECTION_SNAPSHOT_SCHEMA_VERSION = "{schema_version}";',
        "",
    ]
    generated.extend(_render_field_array("DEFLECTION_RESULT_PAGE_SNAPSHOT_FIELDS", RESULT_PAGE_SNAPSHOT_FIELDS))
    generated.append("")
    generated.extend(_render_field_array("DEFLECTION_SNAPSHOT_SUMMARY_FIELDS", summary))
    generated.append("")
    generated.extend(
        _render_field_array("DEFLECTION_SNAPSHOT_SUMMARY_OPTIONAL_FIELDS", summary_optional)
    )
    generated.append("")
    generated.extend(_render_field_array("DEFLECTION_SNAPSHOT_TOP_QUESTION_FIELDS", top_questions))
    generated.append("")
    generated.extend(_render_field_array("DEFLECTION_SNAPSHOT_TOP_BLIND_SPOT_FIELDS", top_blind_spots))
    generated.append("")
    return "\n".join(generated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-output", type=Path, default=DEFAULT_API_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        outputs = [
            (args.output, render_types()),
            (args.api_output, render_api_contract()),
        ]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.check:
        ok = True
        for output, expected in outputs:
            try:
                current = output.read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"{output} is missing; run this generator to create it", file=sys.stderr)
                ok = False
                continue
            if current != expected:
                print(f"{output} is stale; run this generator to refresh it", file=sys.stderr)
                ok = False
            else:
                print(f"{output} is current")
        return 0 if ok else 1

    for output, expected in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected, encoding="utf-8")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
