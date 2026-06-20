"""Build surrogate-only eval corpora for deflection PII recall scoring.

The real-labeling step happens upstream and may touch raw customer PII. This
module starts after that step: it accepts already-labeled local records and
emits a versionable artifact whose text and labels contain only synthetic
surrogates.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "deflection_pii_eval_corpus.v1"
INPUT_SCHEMA_VERSION = "deflection_pii_labeled_source.v1"

FIELD_KEYS = (
    "subject",
    "customer_message",
    "agent_reply",
    "private_note",
    "source_id",
)
PII_CLASSES = frozenset({
    "email",
    "phone",
    "ssn",
    "payment_card",
    "person_name",
    "street_address",
    "account_id",
    "order_id",
    "opaque_id",
    "dob",
    "other",
})
NAME_SUBTYPES = frozenset({"cue_prefixed", "cue_less"})
SEVERITY_BY_CLASS = {
    "email": "high",
    "phone": "high",
    "ssn": "high",
    "payment_card": "high",
    "person_name": "high",
    "dob": "high",
    "street_address": "medium",
    "account_id": "medium",
    "order_id": "medium",
    "opaque_id": "low",
    "other": "low",
}

_SURROGATES = {
    "email": ("alex.rivera@example.test", "maya.chen@example.test"),
    "phone": ("555-010-4301", "555-010-4302"),
    "ssn": ("123-45-6789", "987-65-4321"),
    "payment_card": ("4111 1111 1111 1111", "4242 4242 4242 4242"),
    "person_name": ("Maya Chen", "Jordan Lee", "Taylor Brooks"),
    "street_address": ("1842 Pine Street", "77 Market Avenue"),
    "account_id": ("acct_SYNTH_1001", "acct_SYNTH_1002"),
    "order_id": ("order_SYNTH_2001", "order_SYNTH_2002"),
    "opaque_id": ("tok_SYNTH_3001", "tok_SYNTH_3002"),
    "dob": ("1990-04-17", "1984-11-02"),
    "other": ("SYNTHETIC-PII-1", "SYNTHETIC-PII-2"),
}

_BOUNDARY_LEFT = r"(?<![A-Za-z0-9])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9])"
_LEAK_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(
        rf"{_BOUNDARY_LEFT}[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{{2,}}{_BOUNDARY_RIGHT}"
    )),
    ("phone", re.compile(
        rf"{_BOUNDARY_LEFT}(?:\+?1[\s.-]?)?(?:\(?\d{{3}}\)?[\s.-]?)\d{{3}}[\s.-]?\d{{4}}{_BOUNDARY_RIGHT}"
    )),
    ("ssn", re.compile(rf"{_BOUNDARY_LEFT}\d{{3}}-\d{{2}}-\d{{4}}{_BOUNDARY_RIGHT}")),
    ("payment_card", re.compile(rf"{_BOUNDARY_LEFT}(?:\d[ -]?){{13,19}}{_BOUNDARY_RIGHT}")),
    ("street_address", re.compile(
        rf"{_BOUNDARY_LEFT}"
        r"\d{1,6}\s+"
        r"(?:[NSEW]\.?\s+)?"
        r"(?:[A-Z0-9][A-Za-z0-9'.-]*\s+){0,6}"
        r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
        r"Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Way|Place|Pl\.?|Terrace|Ter\.?|"
        r"Circle|Cir\.?|Highway|Hwy\.?)"
        r"(?:\s+(?:Apt|Apartment|Suite|Ste\.?|Unit|#)\s*[A-Za-z0-9-]+)?"
        rf"{_BOUNDARY_RIGHT}"
    )),
    ("person_name", re.compile(
        rf"{_BOUNDARY_LEFT}"
        r"(?i:(?:customer|requester|contact|client|user|agent|member|person|name)"
        r"(?:\s+name)?\s*(?:is|:|=|-)\s*)"
        r"(?P<name>[A-Za-z][A-Za-z'-]+\s+[A-Za-z][A-Za-z'-]+"
        r"(?:\s+[A-Z][A-Za-z'-]+)?)"
        r"(?=$|[\s,.;:!?)]|</)"
    )),
)


@dataclass(frozen=True)
class SurrogateEvalCorpusBuildResult:
    """Result envelope for tolerant decoded-input validation."""

    artifact: dict[str, Any] | None
    errors: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.artifact is not None and not self.errors


def build_surrogate_eval_corpus(source: Any) -> SurrogateEvalCorpusBuildResult:
    """Return a surrogate-only eval artifact from already-labeled records."""

    records, top_errors = _records_from_source(source)
    errors: list[dict[str, Any]] = list(top_errors)
    tickets: list[dict[str, Any]] = []
    surrogate_counts: Counter[str] = Counter()
    if not top_errors and not records:
        errors.append(_error("source_empty_records"))
    for record_index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            errors.append(_error("record_not_object", record_index=record_index))
            continue
        built = _build_ticket(record, record_index=record_index, surrogate_counts=surrogate_counts)
        errors.extend(built.errors)
        if built.ticket is not None:
            tickets.append(built.ticket)
    if errors:
        return SurrogateEvalCorpusBuildResult(
            artifact=None,
            errors=tuple(errors),
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "kind": "surrogated_eval",
            "raw_source_persisted": False,
            "raw_label_spans_persisted": False,
            "surrogate_positions_are_recall_labels": True,
        },
        "summary": _summary(tickets),
        "tickets": tickets,
    }
    return SurrogateEvalCorpusBuildResult(artifact=artifact)


@dataclass(frozen=True)
class _TicketBuild:
    ticket: dict[str, Any] | None
    errors: tuple[dict[str, Any], ...]


def _build_ticket(
    record: Mapping[str, Any],
    *,
    record_index: int,
    surrogate_counts: Counter[str],
) -> _TicketBuild:
    fields = _fields(record)
    labels = _sequence(record.get("labels"))
    must_survive = _sequence(record.get("must_survive"))
    errors: list[dict[str, Any]] = []
    if not any(fields.values()):
        errors.append(_error("record_missing_text", record_index=record_index))
    if not labels:
        errors.append(_error("record_missing_labels", record_index=record_index))
    replacements_by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_spans_by_field: dict[str, list[str]] = defaultdict(list)
    occurrence_counts: Counter[tuple[str, str]] = Counter()
    for label_index, label in enumerate(labels, start=1):
        parsed, label_errors = _parse_label(
            label,
            fields,
            record_index=record_index,
            label_index=label_index,
            occurrence_counts=occurrence_counts,
            surrogate_counts=surrogate_counts,
        )
        errors.extend(label_errors)
        if parsed is not None:
            replacements_by_field[parsed["origin_field"]].append(parsed)
            raw_spans_by_field[parsed["origin_field"]].append(parsed["raw_span"])
    if errors:
        return _TicketBuild(ticket=None, errors=tuple(errors))

    output_fields = dict(fields)
    output_labels: list[dict[str, Any]] = []
    for field in FIELD_KEYS:
        replacements = replacements_by_field.get(field, [])
        if not replacements:
            continue
        field_text, field_labels, field_errors = _replace_field(
            output_fields[field],
            replacements,
            record_index=record_index,
            origin_field=field,
        )
        errors.extend(field_errors)
        output_fields[field] = field_text
        output_labels.extend(field_labels)
    if errors:
        return _TicketBuild(ticket=None, errors=tuple(errors))

    errors.extend(_raw_span_residue_errors(
        raw_spans_by_field,
        output_fields,
        output_labels,
        record_index=record_index,
    ))
    errors.extend(_output_leak_errors(
        output_fields,
        output_labels,
        record_index=record_index,
    ))
    if errors:
        return _TicketBuild(ticket=None, errors=tuple(errors))

    output_must_survive, must_errors = _must_survive_records(
        must_survive,
        fields,
        output_fields,
        replacements_by_field,
        record_index=record_index,
    )
    errors.extend(must_errors)
    if errors:
        return _TicketBuild(ticket=None, errors=tuple(errors))
    return _TicketBuild(
        ticket={
            "ticket_id": f"pii-eval-{record_index:03d}",
            "fields": {key: value for key, value in output_fields.items() if value},
            "labels": sorted(output_labels, key=lambda item: (item["origin_field"], item["start"])),
            "must_survive": output_must_survive,
        },
        errors=(),
    )


def _parse_label(
    label: Any,
    fields: Mapping[str, str],
    *,
    record_index: int,
    label_index: int,
    occurrence_counts: Counter[tuple[str, str]],
    surrogate_counts: Counter[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(label, Mapping):
        return None, [_error("label_not_object", record_index=record_index, label_index=label_index)]
    pii_class = _clean(label.get("class"))
    origin_field = _clean(label.get("origin_field"))
    span = _clean(label.get("span"))
    if pii_class not in PII_CLASSES:
        errors.append(_error("label_invalid_class", record_index=record_index, label_index=label_index))
    if origin_field not in FIELD_KEYS:
        errors.append(_error("label_invalid_origin_field", record_index=record_index, label_index=label_index))
    if not span:
        errors.append(_error("label_missing_span", record_index=record_index, label_index=label_index))
    name_subtype = _clean(label.get("name_subtype"))
    if pii_class == "person_name" and name_subtype not in NAME_SUBTYPES:
        errors.append(_error("person_name_missing_subtype", record_index=record_index, label_index=label_index))
    if errors:
        return None, errors
    field_text = fields.get(origin_field, "")
    occurrence = _positive_int(label.get("occurrence"))
    if occurrence is None:
        key = (origin_field, span)
        occurrence_counts[key] += 1
        occurrence = occurrence_counts[key]
    match = _find_occurrence(field_text, span, occurrence)
    if match is None:
        return None, [_error(
            "label_span_not_found",
            record_index=record_index,
            label_index=label_index,
            origin_field=origin_field,
        )]
    surrogate_counts[pii_class] += 1
    surrogate = _surrogate_for(pii_class, surrogate_counts[pii_class], raw_span=span)
    if surrogate is None:
        return None, [_error(
            "surrogate_matches_raw_span",
            record_index=record_index,
            label_index=label_index,
            origin_field=origin_field,
        )]
    severity = _clean(label.get("severity")) or SEVERITY_BY_CLASS[pii_class]
    if severity not in {"high", "medium", "low"}:
        severity = SEVERITY_BY_CLASS[pii_class]
    parsed = {
        "origin_field": origin_field,
        "raw_start": match[0],
        "raw_end": match[1],
        "raw_span": span,
        "surrogate": surrogate,
        "class": pii_class,
        "severity": severity,
        "surrogate_id": f"{pii_class}-{surrogate_counts[pii_class]:03d}",
    }
    if pii_class == "person_name":
        parsed["name_subtype"] = name_subtype
    return parsed, []


def _replace_field(
    text: str,
    replacements: Sequence[Mapping[str, Any]],
    *,
    record_index: int,
    origin_field: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(replacements, key=lambda item: int(item["raw_start"]))
    errors: list[dict[str, Any]] = []
    out: list[str] = []
    labels: list[dict[str, Any]] = []
    cursor = 0
    for item in ordered:
        raw_start = int(item["raw_start"])
        raw_end = int(item["raw_end"])
        if raw_start < cursor:
            errors.append(_error("label_spans_overlap", record_index=record_index, origin_field=origin_field))
            continue
        out.append(text[cursor:raw_start])
        start = sum(len(part) for part in out)
        surrogate = _clean(item.get("surrogate"))
        out.append(surrogate)
        end = start + len(surrogate)
        label = {
            "span": surrogate,
            "class": item["class"],
            "severity": item["severity"],
            "origin_field": origin_field,
            "start": start,
            "end": end,
            "surrogate_id": item["surrogate_id"],
        }
        if item.get("name_subtype"):
            label["name_subtype"] = item["name_subtype"]
        labels.append(label)
        cursor = raw_end
    out.append(text[cursor:])
    return "".join(out), labels, errors


def _must_survive_records(
    values: Sequence[Any],
    raw_fields: Mapping[str, str],
    output_fields: Mapping[str, str],
    replacements_by_field: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    record_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    occurrence_counts: Counter[tuple[str, str]] = Counter()
    for index, item in enumerate(values, start=1):
        if not isinstance(item, Mapping):
            errors.append(_error("must_survive_not_object", record_index=record_index, must_survive_index=index))
            continue
        origin_field = _clean(item.get("origin_field"))
        span = _clean(item.get("span"))
        if origin_field not in FIELD_KEYS or not span:
            errors.append(_error("must_survive_missing_field_or_span", record_index=record_index, must_survive_index=index))
            continue
        occurrence = _positive_int(item.get("occurrence"))
        if occurrence is None:
            key = (origin_field, span)
            occurrence_counts[key] += 1
            occurrence = occurrence_counts[key]
        raw_match = _find_occurrence(raw_fields.get(origin_field, ""), span, occurrence)
        if raw_match is None:
            errors.append(_error("must_survive_span_not_found", record_index=record_index, must_survive_index=index, origin_field=origin_field))
            continue
        raw_start, raw_end = raw_match
        if _overlaps_replacement(raw_start, raw_end, replacements_by_field.get(origin_field, ())):
            errors.append(_error("must_survive_overlaps_label", record_index=record_index, must_survive_index=index, origin_field=origin_field))
            continue
        start = _map_raw_start_to_output(raw_start, replacements_by_field.get(origin_field, ()))
        output_text = output_fields.get(origin_field, "")
        if output_text[start:start + len(span)] != span:
            errors.append(_error("must_survive_offset_drift", record_index=record_index, must_survive_index=index, origin_field=origin_field))
            continue
        out.append({
            "span": span,
            "origin_field": origin_field,
            "start": start,
            "end": start + len(span),
            "reason": _clean(item.get("reason")) or "must_survive",
        })
    return out, errors


def _raw_span_residue_errors(
    raw_spans_by_field: Mapping[str, Sequence[str]],
    output_fields: Mapping[str, str],
    output_labels: Sequence[Mapping[str, Any]],
    *,
    record_index: int,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for origin_field, raw_spans in raw_spans_by_field.items():
        text = output_fields.get(origin_field, "")
        for raw_span in set(raw_spans):
            start = text.find(raw_span)
            while start >= 0:
                end = start + len(raw_span)
                if not _is_recorded_surrogate_span(output_labels, origin_field, start, end):
                    errors.append(_error(
                        "raw_label_span_remaining",
                        record_index=record_index,
                        origin_field=origin_field,
                    ))
                    break
                start = text.find(raw_span, start + 1)
    return errors


def _output_leak_errors(
    output_fields: Mapping[str, str],
    output_labels: Sequence[Mapping[str, Any]],
    *,
    record_index: int,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for origin_field, text in output_fields.items():
        if not text:
            continue
        for detector_name, pattern in _LEAK_DETECTORS:
            for match in pattern.finditer(text):
                start, end = match.span("name") if detector_name == "person_name" else match.span()
                if _is_recorded_surrogate_span(output_labels, origin_field, start, end):
                    continue
                errors.append(_error(
                    "unlabeled_pii_detected",
                    record_index=record_index,
                    origin_field=origin_field,
                    detector=detector_name,
                ))
                break
    return errors


def _is_recorded_surrogate_span(
    labels: Sequence[Mapping[str, Any]],
    origin_field: str,
    start: int,
    end: int,
) -> bool:
    return any(
        label.get("origin_field") == origin_field
        and label.get("start") == start
        and label.get("end") == end
        for label in labels
    )


def _map_raw_start_to_output(
    raw_start: int,
    replacements: Sequence[Mapping[str, Any]],
) -> int:
    offset = 0
    for item in replacements:
        raw_end = int(item["raw_end"])
        if raw_end <= raw_start:
            offset += len(_clean(item.get("surrogate"))) - (raw_end - int(item["raw_start"]))
    return raw_start + offset


def _overlaps_replacement(
    raw_start: int,
    raw_end: int,
    replacements: Sequence[Mapping[str, Any]],
) -> bool:
    return any(
        raw_start < int(item["raw_end"]) and raw_end > int(item["raw_start"])
        for item in replacements
    )


def _records_from_source(source: Any) -> tuple[list[Any], tuple[dict[str, Any], ...]]:
    if isinstance(source, Mapping):
        records = source.get("records")
        if records is None:
            records = source.get("tickets")
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
            return list(records), ()
        return [], (_error("source_missing_records"),)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return list(source), ()
    return [], (_error("source_invalid_shape"),)


def _fields(record: Mapping[str, Any]) -> dict[str, str]:
    raw_fields = record.get("fields")
    values = raw_fields if isinstance(raw_fields, Mapping) else record
    return {key: _clean(values.get(key)) if isinstance(values, Mapping) else "" for key in FIELD_KEYS}


def _summary(tickets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [label for ticket in tickets for label in ticket.get("labels", ())]
    must_survive_count = sum(len(ticket.get("must_survive", ())) for ticket in tickets)
    by_class = Counter(_clean(label.get("class")) for label in labels)
    by_severity = Counter(_clean(label.get("severity")) for label in labels)
    return {
        "ticket_count": len(tickets),
        "label_count": len(labels),
        "must_survive_count": must_survive_count,
        "labels_by_class": dict(sorted(by_class.items())),
        "labels_by_severity": dict(sorted(by_severity.items())),
        "cue_prefixed_person_name_count": sum(
            1 for label in labels if label.get("name_subtype") == "cue_prefixed"
        ),
        "cue_less_person_name_count": sum(
            1 for label in labels if label.get("name_subtype") == "cue_less"
        ),
    }


def _find_occurrence(text: str, span: str, occurrence: int) -> tuple[int, int] | None:
    start = -1
    for _ in range(max(1, occurrence)):
        start = text.find(span, start + 1)
        if start < 0:
            return None
    return start, start + len(span)


def _surrogate_for(pii_class: str, index: int, *, raw_span: str = "") -> str | None:
    values = _SURROGATES[pii_class]
    for offset in range(len(values)):
        candidate = values[(index - 1 + offset) % len(values)]
        if candidate.casefold() != raw_span.casefold():
            return candidate
    return None


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _error(code: str, **details: Any) -> dict[str, Any]:
    return {
        "code": code,
        **{
            key: value
            for key, value in details.items()
            if isinstance(value, (int, str)) and value not in ("", 0)
        },
    }


__all__ = [
    "FIELD_KEYS",
    "INPUT_SCHEMA_VERSION",
    "PII_CLASSES",
    "SCHEMA_VERSION",
    "SEVERITY_BY_CLASS",
    "SurrogateEvalCorpusBuildResult",
    "build_surrogate_eval_corpus",
]
