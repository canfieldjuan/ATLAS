#!/usr/bin/env python3
"""Generate or check frontend TypeScript types from the deflection contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
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
DEFAULT_REPORT_MODEL_OUTPUT = ROOT / "portfolio-ui/src/types/deflectionReportModel.ts"
DEFAULT_REPORT_MODEL_API_OUTPUT = (
    ROOT / "portfolio-ui/api/content-ops/deflection/report-model-contract.js"
)

TYPE_BY_FIELD = {
    "action_label": "string",
    "answer": "string",
    "answer_linkage": "string",
    "answer_evidence_status": "string",
    "answer_status": "string",
    "annualized_run_rate_support_cost": "number",
    "annualized_support_cost": "number",
    "assisted_contact_cost": "number",
    "backlog_limit": "number",
    "body_withheld": "boolean",
    "cluster_id": "string",
    "confidence": "string",
    "cost_confidence": "string",
    "cost_period": "string",
    "customer_term": "string",
    "customer_vocabulary": "string[]",
    "customer_wording": "string",
    "csat_present_count": "number",
    "default_limit": "number",
    "displayed_phrase_count": "number",
    "documentation_term": "string",
    "drafted_answer_count": "number",
    "estimated_support_cost": "number",
    "evidence_tier": "string",
    "evidence_row_count": "number",
    "evidence_quote": "string",
    "evidence_quotes": "string[]",
    "fix_type": "string",
    "formula": "string",
    "group": "string[]",
    "full_answer": "DeflectionSnapshotTeaserFullAnswer | null",
    "generated": "number",
    "generated_question_count": "number",
    "guidance": "string",
    "identity_basis": "string",
    "identity_confidence": "string",
    "jira_template": "DeflectionReportActionJiraTemplate",
    "assignee": "string[]",
    "brand": "string[]",
    "custom_product_area": "string[]",
    "limit": "number",
    "negative_csat_ticket_count": "number",
    "no_proven_answer_count": "number",
    "non_repeat_ticket_count": "number",
    "numeric_average": "number | null",
    "omitted_phrase_count": "number",
    "opportunity_score": "number",
    "outcome_diagnostic_ticket_count": "number",
    "outcome_diagnostics": "DeflectionReportQuestionOutcomeDiagnostics | null",
    "outcome_risk_ticket_count": "number",
    "owner_category": "string",
    "owner_lane": "string",
    "pdf_limit": "number",
    "phrases": "string[]",
    "previews": "DeflectionSnapshotTeaserPreview[]",
    "priority_drivers": "string[]",
    "priority_score": "number",
    "question": "string",
    "question_count": "number",
    "rank": "number",
    "recommended_action": "string",
    "recommended_title": "string",
    "reopened_ticket_count": "number",
    "repeat_ticket_count": "number",
    "repeat_key": "string",
    "representative_phrasing": "string",
    "reason_counts": "Record<string, number>",
    "review_key": "string",
    "resolution_evidence_scope": "string",
    "result_page_limit": "number",
    "routing_signals": "DeflectionReportActionRoutingSignals",
    "source": "string",
    "source_count": "number",
    "source_date_end": "string | null",
    "source_date_start": "string | null",
    "source_date_window": "DeflectionReportSourceDateWindow | null",
    "source_id": "string",
    "source_id_count": "number",
    "source_ids": "string[]",
    "source_label": "string",
    "source_proof": "string",
    "source_window_days": "number | null",
    "status": "string",
    "status_counts": "Record<string, number>",
    "status_mix": "string",
    "step_count": "number",
    "steps": "string[]",
    "suppression_reason": "string",
    "suppression_reason_label": "string",
    "support_cost_formula": "string",
    "support_cost_source": "string",
    "support_ticket_resolution_evidence_count": "number",
    "support_ticket_resolution_evidence_present": "boolean",
    "suggestion": "string",
    "surfaces": "string[]",
    "term_mappings": "DeflectionReportTermMapping[]",
    "ticket_count": "number",
    "title": "string",
    "ticket_source_count": "number",
    "top_item_count": "number",
    "total_item_count": "number",
    "total_phrase_count": "number",
    "topic": "string",
    "organization": "string[]",
    "product_area": "string[]",
    "product_gap_summary": "string",
    "tags": "string[]",
    "weighted_frequency": "number",
}

TYPE_NAME_BY_SNAPSHOT_FIELD = {
    "summary": "DeflectionSnapshotSummary",
    "top_questions": "DeflectionSnapshotTopQuestion",
    "locked_questions": "DeflectionSnapshotLockedQuestion",
    "top_blind_spots": "DeflectionSnapshotTopBlindSpot",
}

RESULT_PAGE_SNAPSHOT_FIELDS = ("title", "summary", "top_questions", "top_blind_spots")


def _indent(lines: Sequence[str], spaces: int = 2) -> list[str]:
    prefix = " " * spaces
    return [f"{prefix}{line}" if line else "" for line in lines]


def _field_type(field: str) -> str:
    try:
        return TYPE_BY_FIELD[field]
    except KeyError as exc:
        raise ValueError(f"missing TypeScript type mapping for contract field: {field}") from exc


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


def _hosted_consumer_safe_fields(entry: Mapping[str, Any], owner: str) -> list[str]:
    fields = entry.get("hosted_consumer_safe_fields")
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise ValueError(
            f"report_projection {owner} has invalid hosted_consumer_safe_fields"
        )
    projected = set(_projected_fields(entry))
    unknown = [field for field in fields if field not in projected]
    if unknown:
        raise ValueError(
            f"hosted consumer safe fields are not projected for {owner}: "
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
        if field == "title":
            generated.extend(_indent(["title: string;"]))
            continue
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
        'DeflectionSnapshot, "title" | "summary" | "top_questions" | "top_blind_spots">;'
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


def _pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\W]+", value) if part)


def _singular_pascal(field: str) -> str:
    if field == "rows":
        return "Row"
    if field == "items":
        return "Item"
    if field.endswith("ies"):
        return f"{_pascal_case(field[:-3])}y"
    if field.endswith("s"):
        return _pascal_case(field[:-1])
    return _pascal_case(field)


def _section_data_type_name(section_id: str) -> str:
    return f"DeflectionReport{_pascal_case(section_id)}Data"


def _section_type_name(section_id: str) -> str:
    return f"DeflectionReport{_pascal_case(section_id)}Section"


def _collection_item_type_name(section_id: str, field: str) -> str:
    return f"DeflectionReport{_pascal_case(section_id)}{_singular_pascal(field)}"


def _nested_type_name(section_id: str, field: str) -> str:
    return f"DeflectionReport{_pascal_case(section_id)}{_pascal_case(field)}"


def _report_projection_metadata(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    shape = contract if contract is not None else deflection_report_model_contract_shape()
    projection = shape.get("report_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("contract missing report_projection")

    schema_version = projection.get("schema_version")
    if not isinstance(schema_version, str):
        raise ValueError("report_projection.schema_version must be a string")

    model_fields = projection.get("model_fields")
    if not isinstance(model_fields, list) or not all(
        isinstance(field, str) for field in model_fields
    ):
        raise ValueError("report_projection.model_fields must be a string list")

    section_fields = projection.get("section_fields")
    if not isinstance(section_fields, list) or not all(
        isinstance(field, str) for field in section_fields
    ):
        raise ValueError("report_projection.section_fields must be a string list")

    sections = projection.get("sections")
    if not isinstance(sections, list) or not all(
        isinstance(section, Mapping) for section in sections
    ):
        raise ValueError("report_projection.sections must be a list of objects")

    metadata_sections: list[Mapping[str, Any]] = []
    for section in sections:
        section_id = section.get("id")
        if not isinstance(section_id, str):
            raise ValueError("report_projection section missing string id")
        for required in ("title", "priority", "surfaces", "default_limit", "required_data", "snapshot_safe_fields"):
            if required not in section:
                raise ValueError(f"report_projection section {section_id!r} missing {required}")

        projected_fields = _projected_fields(section)
        optional_fields = _optional_projected_fields(section)
        record_fields = _record_fields(section, section_id, projected_fields)

        structural_fields: list[str] = []
        collection = section.get("collection")
        if isinstance(collection, Mapping) and isinstance(collection.get("field"), str):
            structural_fields.append(str(collection["field"]))
        for nested in section.get("nested_object_fields", []):
            if isinstance(nested, Mapping) and isinstance(nested.get("field"), str):
                structural_fields.append(str(nested["field"]))
        for nested in section.get("nested_collection_fields", []):
            if isinstance(nested, Mapping) and isinstance(nested.get("field"), str):
                structural_fields.append(str(nested["field"]))

        _validate_report_fields(
            section_id,
            projected_fields,
            optional_fields,
            record_fields,
            structural_fields,
        )
        _hosted_consumer_safe_fields(section, section_id)

        if collection is not None:
            if not isinstance(collection, Mapping):
                raise ValueError(f"report_projection section {section_id!r} has invalid collection")
            collection_field = collection.get("field")
            if not isinstance(collection_field, str) or collection_field not in projected_fields:
                raise ValueError(
                    f"report_projection section {section_id!r} collection field must be projected"
                )
            item_type = collection.get("item_type")
            if item_type not in {"object", "string"}:
                raise ValueError(
                    f"report_projection section {section_id!r} collection item_type is invalid"
                )
            if item_type == "object":
                item_fields = _projected_fields(collection)
                item_optional_fields = _optional_projected_fields(collection)
                collection_record_fields = _record_fields(
                    collection,
                    f"{section_id}.{collection_field}",
                    item_fields,
                )
                collection_structural_fields: list[str] = []
                for nested in collection.get("nested_object_fields", []):
                    if isinstance(nested, Mapping) and isinstance(nested.get("field"), str):
                        collection_structural_fields.append(str(nested["field"]))
                for nested in collection.get("nested_collection_fields", []):
                    if isinstance(nested, Mapping) and isinstance(nested.get("field"), str):
                        collection_structural_fields.append(str(nested["field"]))
                _validate_report_fields(
                    section_id,
                    item_fields,
                    item_optional_fields,
                    collection_record_fields,
                    collection_structural_fields,
                )
                _hosted_consumer_safe_fields(
                    collection,
                    f"{section_id}.{collection_field}",
                )
                _validate_nested_fields(section_id, collection, item_fields)

        _validate_nested_fields(section_id, section, projected_fields)
        metadata_sections.append(section)

    return {
        "schema_version": schema_version,
        "model_fields": model_fields,
        "section_fields": section_fields,
        "sections": metadata_sections,
    }


def _validate_report_fields(
    section_id: str,
    projected_fields: Sequence[str],
    optional_fields: Sequence[str],
    record_fields: Sequence[str],
    structural_fields: Sequence[str] = (),
) -> None:
    projected = set(projected_fields)
    unknown_optional = [field for field in optional_fields if field not in projected]
    if unknown_optional:
        raise ValueError(
            f"optional projected fields are not projected for {section_id!r}: "
            + ", ".join(unknown_optional)
        )
    record = set(record_fields)
    structural = set(structural_fields)
    for field in projected_fields:
        if field in record or field in structural:
            continue
        _field_type(field)


def _record_fields(
    owner: Mapping[str, Any],
    owner_id: str,
    projected_fields: Sequence[str],
) -> list[str]:
    fields = owner.get("record_fields", [])
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise ValueError(f"report_projection {owner_id!r} has invalid record_fields")
    projected = set(projected_fields)
    unknown_records = [field for field in fields if field not in projected]
    if unknown_records:
        raise ValueError(
            f"record fields are not projected for {owner_id!r}: "
            + ", ".join(unknown_records)
        )
    return list(fields)


def _validate_nested_fields(
    section_id: str,
    owner: Mapping[str, Any],
    owner_fields: Sequence[str],
) -> None:
    for key in ("nested_object_fields", "nested_collection_fields"):
        nested_entries = owner.get(key, [])
        if not isinstance(nested_entries, list) or not all(
            isinstance(entry, Mapping) for entry in nested_entries
        ):
            raise ValueError(f"report_projection section {section_id!r} has invalid {key}")
        for entry in nested_entries:
            field = entry.get("field")
            if not isinstance(field, str) or field not in owner_fields:
                raise ValueError(
                    f"report_projection section {section_id!r} nested field must be projected"
                )
            nested_fields = _projected_fields(entry)
            record_fields = _record_fields(
                entry,
                f"{section_id}.{field}",
                nested_fields,
            )
            _validate_report_fields(section_id, nested_fields, [], record_fields)
            _hosted_consumer_safe_fields(entry, f"{section_id}.{field}")


def _field_const_token(field: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", field.upper()).strip("_")


def _scalar_hosted_shape(field: str) -> str:
    field_type = _field_type(field)
    if field_type in {"string", "number", "boolean", "string | null", "number | null"}:
        return "scalar"
    if field_type == "string[]":
        return "scalar_array"
    raise ValueError(
        f"hosted-safe field {field!r} has non-scalar type {field_type!r} "
        "but no record/nested shape metadata"
    )


def _hosted_shape_paths(section: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    section_id = str(section["id"])
    paths: dict[str, dict[str, str]] = {}

    def add_owner(owner: Mapping[str, Any], owner_path: str, owner_label: str) -> None:
        hosted_fields = _hosted_consumer_safe_fields(owner, owner_label)
        if not hosted_fields:
            return

        record_fields = set(str(field) for field in owner.get("record_fields", []))
        nested_objects = {
            str(entry["field"]): entry
            for entry in owner.get("nested_object_fields", [])
            if isinstance(entry, Mapping) and isinstance(entry.get("field"), str)
        }
        nested_collections = {
            str(entry["field"]): entry
            for entry in owner.get("nested_collection_fields", [])
            if isinstance(entry, Mapping) and isinstance(entry.get("field"), str)
        }
        collection = owner.get("collection")
        collection_field = (
            str(collection["field"])
            if isinstance(collection, Mapping) and isinstance(collection.get("field"), str)
            else None
        )

        field_shapes: dict[str, str] = {}
        for field in hosted_fields:
            if field in record_fields:
                field_shapes[field] = "record"
            elif field in nested_objects:
                field_shapes[field] = "object"
            elif field in nested_collections:
                nested = nested_collections[field]
                field_shapes[field] = (
                    "object_array" if nested.get("item_type", "object") == "object" else "scalar_array"
                )
            elif field == collection_field and isinstance(collection, Mapping):
                field_shapes[field] = (
                    "object_array" if collection.get("item_type", "object") == "object" else "scalar_array"
                )
            else:
                field_shapes[field] = _scalar_hosted_shape(field)
        paths[owner_path] = field_shapes

        for field, nested in nested_objects.items():
            if field in hosted_fields:
                add_owner(nested, f"{owner_path}.{field}", f"{owner_label}.{field}")
        for field, nested in nested_collections.items():
            if field in hosted_fields and nested.get("item_type", "object") == "object":
                add_owner(nested, f"{owner_path}.{field}", f"{owner_label}.{field}")
        if (
            collection_field
            and collection_field in hosted_fields
            and isinstance(collection, Mapping)
            and collection.get("item_type", "object") == "object"
        ):
            add_owner(
                collection,
                f"{owner_path}.{collection_field}",
                f"{owner_label}.{collection_field}",
            )

    add_owner(section, section_id, section_id)
    return paths


def _render_report_hosted_shape_map(
    name: str,
    sections: Sequence[Mapping[str, Any]],
    *,
    typescript: bool,
) -> list[str]:
    shape_paths: dict[str, dict[str, str]] = {}
    for section in sections:
        shape_paths.update(_hosted_shape_paths(section))

    opener = "{" if typescript else "Object.freeze({"
    lines = [f"export const {name} = {opener}"]
    for owner_path in sorted(shape_paths):
        owner_opener = "{" if typescript else "Object.freeze({"
        lines.append(f'  "{owner_path}": {owner_opener}')
        for field, shape in shape_paths[owner_path].items():
            lines.append(f'    "{field}": "{shape}",')
        owner_suffix = "}," if typescript else "}),"
        lines.append(f"  {owner_suffix}")
    suffix = " as const;" if typescript else ";"
    closer = "}" if typescript else "})"
    lines.append(f"{closer}{suffix}")
    return lines


def _render_report_hosted_safe_constants(
    section: Mapping[str, Any],
    render_fields,
) -> list[str]:
    section_id = str(section["id"])
    const_prefix = f"DEFLECTION_REPORT_{section_id.upper()}"
    generated: list[str] = []

    generated.extend(
        render_fields(
            f"{const_prefix}_HOSTED_CONSUMER_SAFE_FIELDS",
            _hosted_consumer_safe_fields(section, section_id),
        )
    )
    generated.append("")

    def add_nested_constants(
        owner: Mapping[str, Any],
        owner_prefix: str,
    ) -> None:
        for nested in owner.get("nested_object_fields", []):
            field = str(nested["field"])
            generated.extend(
                render_fields(
                    f"{owner_prefix}_{_field_const_token(field)}_HOSTED_CONSUMER_SAFE_FIELDS",
                    _hosted_consumer_safe_fields(nested, f"{section_id}.{field}"),
                )
            )
            generated.append("")
        for nested in owner.get("nested_collection_fields", []):
            field = str(nested["field"])
            generated.extend(
                render_fields(
                    f"{owner_prefix}_{_field_const_token(field)}_HOSTED_CONSUMER_SAFE_FIELDS",
                    _hosted_consumer_safe_fields(nested, f"{section_id}.{field}"),
                )
            )
            generated.append("")

    add_nested_constants(section, const_prefix)

    collection = section.get("collection")
    if isinstance(collection, Mapping) and collection.get("item_type") == "object":
        collection_field = str(collection["field"])
        collection_prefix = f"{const_prefix}_{_field_const_token(collection_field)}"
        generated.extend(
            render_fields(
                f"{collection_prefix}_HOSTED_CONSUMER_SAFE_FIELDS",
                _hosted_consumer_safe_fields(
                    collection,
                    f"{section_id}.{collection_field}",
                ),
            )
        )
        generated.append("")
        add_nested_constants(collection, collection_prefix)

    return generated


def _object_type_with_overrides(
    name: str,
    fields: Sequence[str],
    *,
    optional_fields: Sequence[str] = (),
    type_overrides: Mapping[str, str] | None = None,
    export: bool = True,
) -> list[str]:
    optional = set(optional_fields)
    overrides = type_overrides or {}
    prefix = "export " if export else ""
    lines = [f"{prefix}type {name} = {{"]
    rendered_fields = []
    for field in fields:
        field_type = overrides[field] if field in overrides else _field_type(field)
        rendered_fields.append(f"{field}{'?' if field in optional else ''}: {field_type};")
    lines.extend(_indent(rendered_fields))
    lines.append("};")
    return lines


def _render_report_support_types() -> list[str]:
    return [
        "export type DeflectionReportSourceDateWindow = {",
        "  source_date_start: string | null;",
        "  source_date_end: string | null;",
        "  source_window_days: number | null;",
        "};",
        "",
        "export type DeflectionReportTermMapping = {",
        "  customer_term: string;",
        "  documentation_term: string;",
        "  suggestion: string;",
        "  source_id_count: number;",
        "};",
        "",
        "export type DeflectionReportQuestionOutcomeDiagnostics = {",
        "  ticket_status_summary?: Record<string, number>;",
        "  diagnostic_ticket_count?: number;",
        "  outcome_risk_ticket_count?: number;",
        "  reopened_ticket_count?: number;",
        "  negative_csat_ticket_count?: number;",
        "  csat_present_count?: number;",
        "  csat_score_average?: number | null;",
        "};",
    ]


def _render_report_nested_types(section: Mapping[str, Any]) -> list[str]:
    section_id = str(section["id"])
    generated: list[str] = []

    def add_nested(owner: Mapping[str, Any]) -> None:
        for nested in owner.get("nested_object_fields", []):
            nested_type = _nested_type_name(section_id, str(nested["field"]))
            generated.extend(
                _object_type_with_overrides(
                    nested_type,
                    _projected_fields(nested),
                    type_overrides=_record_type_overrides(nested),
                )
            )
            generated.append("")
        for nested in owner.get("nested_collection_fields", []):
            nested_type = _nested_type_name(section_id, str(nested["field"]))
            generated.extend(
                _object_type_with_overrides(
                    nested_type,
                    _projected_fields(nested),
                    type_overrides=_record_type_overrides(nested),
                )
            )
            generated.append("")

    add_nested(section)
    collection = section.get("collection")
    if isinstance(collection, Mapping):
        add_nested(collection)
        if collection.get("item_type") == "object":
            collection_overrides: dict[str, str] = _record_type_overrides(collection)
            for nested in collection.get("nested_object_fields", []):
                field = str(nested["field"])
                collection_overrides[field] = _nested_type_name(section_id, field)
            for nested in collection.get("nested_collection_fields", []):
                field = str(nested["field"])
                collection_overrides[field] = f"{_nested_type_name(section_id, field)}[]"
            item_type = _collection_item_type_name(section_id, str(collection["field"]))
            generated.extend(
                _object_type_with_overrides(
                    item_type,
                    _projected_fields(collection),
                    optional_fields=_optional_projected_fields(collection),
                    type_overrides=collection_overrides,
                )
            )
            generated.append("")
    return generated


def _record_type_overrides(owner: Mapping[str, Any]) -> dict[str, str]:
    return {str(field): "Record<string, number>" for field in owner.get("record_fields", [])}


def _nested_override_type(section_id: str, field: str) -> str:
    nested_type = _nested_type_name(section_id, field)
    field_type = TYPE_BY_FIELD.get(field, "")
    return f"{nested_type} | null" if "| null" in field_type else nested_type


def _report_section_type_overrides(section: Mapping[str, Any]) -> dict[str, str]:
    section_id = str(section["id"])
    overrides: dict[str, str] = {}
    for field in section.get("record_fields", []):
        overrides[str(field)] = "Record<string, number>"
    for nested in section.get("nested_object_fields", []):
        field = str(nested["field"])
        overrides[field] = _nested_override_type(section_id, field)
    collection = section.get("collection")
    if isinstance(collection, Mapping):
        field = str(collection["field"])
        if collection.get("item_type") == "string":
            overrides[field] = "string[]"
        else:
            overrides[field] = f"{_collection_item_type_name(section_id, field)}[]"
    return overrides


def render_report_model_types(contract: Mapping[str, Any] | None = None) -> str:
    metadata = _report_projection_metadata(contract)
    schema_version = metadata["schema_version"]
    model_fields = metadata["model_fields"]
    section_fields = metadata["section_fields"]
    sections = list(metadata["sections"])
    section_ids = [str(section["id"]) for section in sections]
    conditional_section_ids = [
        str(section["id"])
        for section in sections
        if isinstance(section.get("presence"), Mapping)
        and section["presence"].get("mode") == "conditional"
    ]

    generated: list[str] = [
        "/*",
        " * Generated by scripts/generate_deflection_frontend_contract_types.py.",
        " * Do not edit by hand; run the generator after backend contract changes.",
        " */",
        "",
        f'export const DEFLECTION_REPORT_MODEL_SCHEMA_VERSION = "{schema_version}" as const;',
        "",
    ]
    generated.extend(_render_field_tuple("DEFLECTION_REPORT_MODEL_FIELDS", model_fields))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_REPORT_SECTION_FIELDS", section_fields))
    generated.append("")
    generated.extend(_render_field_tuple("DEFLECTION_REPORT_SECTION_IDS", section_ids))
    generated.append("")
    generated.extend(
        _render_field_tuple("DEFLECTION_REPORT_CONDITIONAL_SECTION_IDS", conditional_section_ids)
    )
    generated.append("")

    for section in sections:
        section_id = str(section["id"])
        const_prefix = f"DEFLECTION_REPORT_{section_id.upper()}"
        projected_fields = _projected_fields(section)
        generated.extend(_render_field_tuple(f"{const_prefix}_FIELDS", projected_fields))
        generated.append("")
        optional_fields = _optional_projected_fields(section)
        if optional_fields:
            generated.extend(_render_field_tuple(f"{const_prefix}_OPTIONAL_FIELDS", optional_fields))
            generated.append("")
        generated.extend(_render_field_tuple(f"{const_prefix}_REQUIRED_DATA", section["required_data"]))
        generated.append("")
        generated.extend(
            _render_field_tuple(f"{const_prefix}_SNAPSHOT_SAFE_FIELDS", section["snapshot_safe_fields"])
        )
        generated.append("")
        generated.extend(_render_report_hosted_safe_constants(section, _render_field_tuple))

        collection = section.get("collection")
        if isinstance(collection, Mapping) and collection.get("item_type") == "object":
            generated.extend(
                _render_field_tuple(
                    f"{const_prefix}_{str(collection['field']).upper()}_FIELDS",
                    _projected_fields(collection),
                )
            )
            generated.append("")

    generated.extend(
        _render_report_hosted_shape_map(
            "DEFLECTION_REPORT_HOSTED_FIELD_SHAPES",
            sections,
            typescript=True,
        )
    )
    generated.append("")

    generated.extend(_render_report_support_types())
    generated.append("")

    section_type_names: list[str] = []
    data_map_lines: list[str] = []
    for section in sections:
        section_id = str(section["id"])
        data_type = _section_data_type_name(section_id)
        section_type = _section_type_name(section_id)
        section_type_names.append(section_type)
        optional_key = (
            "?"
            if isinstance(section.get("presence"), Mapping)
            and section["presence"].get("mode") == "conditional"
            else ""
        )
        data_map_lines.append(f"{section_id}{optional_key}: {data_type};")

        generated.extend(_render_report_nested_types(section))
        generated.extend(
            _object_type_with_overrides(
                data_type,
                _projected_fields(section),
                optional_fields=_optional_projected_fields(section),
                type_overrides=_report_section_type_overrides(section),
            )
        )
        generated.append("")
        generated.extend(
            [
                f'export type {section_type} = {{',
                f'  id: "{section_id}";',
                "  title: string;",
                "  priority: number;",
                "  surfaces: string[];",
                "  default_limit: number | null;",
                "  required_data: string[];",
                "  snapshot_safe_fields: string[];",
                f"  data: {data_type};",
                "};",
                "",
            ]
        )

    generated.append("export type DeflectionReportSectionDataById = {")
    generated.extend(_indent(data_map_lines))
    generated.append("};")
    generated.append("")
    generated.append("export type DeflectionReportSection =")
    section_union = [f"| {name}" for name in section_type_names]
    section_union[-1] = f"{section_union[-1]};"
    generated.extend(_indent(section_union))
    generated.append("")
    generated.append("export type DeflectionStructuredReport = {")
    generated.extend(
        _indent(
            [
                "schema_version: typeof DEFLECTION_REPORT_MODEL_SCHEMA_VERSION;",
                "title: string;",
                "summary: Record<string, unknown>;",
                "sections: DeflectionReportSection[];",
            ]
        )
    )
    generated.append("};")
    generated.append("")
    return "\n".join(generated)


def render_report_model_api_contract(contract: Mapping[str, Any] | None = None) -> str:
    metadata = _report_projection_metadata(contract)
    schema_version = metadata["schema_version"]
    model_fields = metadata["model_fields"]
    section_fields = metadata["section_fields"]
    sections = list(metadata["sections"])
    section_ids = [str(section["id"]) for section in sections]
    conditional_section_ids = [
        str(section["id"])
        for section in sections
        if isinstance(section.get("presence"), Mapping)
        and section["presence"].get("mode") == "conditional"
    ]

    generated: list[str] = [
        "/*",
        " * Generated by scripts/generate_deflection_frontend_contract_types.py.",
        " * Do not edit by hand; run the generator after backend contract changes.",
        " */",
        "",
        f'export const DEFLECTION_REPORT_MODEL_SCHEMA_VERSION = "{schema_version}";',
        "",
    ]
    generated.extend(_render_field_array("DEFLECTION_REPORT_MODEL_FIELDS", model_fields))
    generated.append("")
    generated.extend(_render_field_array("DEFLECTION_REPORT_SECTION_FIELDS", section_fields))
    generated.append("")
    generated.extend(_render_field_array("DEFLECTION_REPORT_SECTION_IDS", section_ids))
    generated.append("")
    generated.extend(
        _render_field_array("DEFLECTION_REPORT_CONDITIONAL_SECTION_IDS", conditional_section_ids)
    )
    generated.append("")

    for section in sections:
        section_id = str(section["id"])
        const_prefix = f"DEFLECTION_REPORT_{section_id.upper()}"
        generated.extend(_render_field_array(f"{const_prefix}_FIELDS", _projected_fields(section)))
        generated.append("")
        optional_fields = _optional_projected_fields(section)
        if optional_fields:
            generated.extend(_render_field_array(f"{const_prefix}_OPTIONAL_FIELDS", optional_fields))
            generated.append("")
        generated.extend(_render_field_array(f"{const_prefix}_REQUIRED_DATA", section["required_data"]))
        generated.append("")
        generated.extend(
            _render_field_array(f"{const_prefix}_SNAPSHOT_SAFE_FIELDS", section["snapshot_safe_fields"])
        )
        generated.append("")
        generated.extend(_render_report_hosted_safe_constants(section, _render_field_array))
        collection = section.get("collection")
        if isinstance(collection, Mapping) and collection.get("item_type") == "object":
            generated.extend(
                _render_field_array(
                    f"{const_prefix}_{str(collection['field']).upper()}_FIELDS",
                    _projected_fields(collection),
                )
            )
            generated.append("")

    generated.extend(
        _render_report_hosted_shape_map(
            "DEFLECTION_REPORT_HOSTED_FIELD_SHAPES",
            sections,
            typescript=False,
        )
    )
    generated.append("")

    return "\n".join(generated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-output", type=Path, default=DEFAULT_API_OUTPUT)
    parser.add_argument("--report-model-output", type=Path, default=DEFAULT_REPORT_MODEL_OUTPUT)
    parser.add_argument(
        "--report-model-api-output",
        type=Path,
        default=DEFAULT_REPORT_MODEL_API_OUTPUT,
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    try:
        outputs = [
            (args.output, render_types()),
            (args.api_output, render_api_contract()),
            (args.report_model_output, render_report_model_types()),
            (args.report_model_api_output, render_report_model_api_contract()),
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
