#!/usr/bin/env python3
"""Validate the operator source-decision handoff for deflection PII measurement."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "deflection_pii_source_decision.v1"
REQUIRED_HIGH_SEVERITY_CLASSES = frozenset({
    "dob",
    "email",
    "payment_card",
    "person_name",
    "phone",
    "ssn",
})
REQUIRED_PERSON_NAME_SUBTYPES = frozenset({"cue_less", "cue_prefixed"})
ALLOWED_SOURCE_KINDS = frozenset({
    "intercom_export",
    "operator_curated_sample",
    "support_ticket_export",
    "zendesk_export",
})
ALLOWED_SOURCE_SUPPLIES = frozenset({
    "secure_upstream_export",
    "transient_local_file",
})
COMPLETED_QUALITY_REVIEW = "completed"

SAFE_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,78}[A-Za-z0-9]$")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DIGIT_RE = re.compile(r"\d")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = json.loads(args.decision_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _write_payload(
            _failure_payload([_error("decision_json_unreadable", detail=exc.__class__.__name__)]),
            pretty=args.pretty,
            stream=sys.stderr,
        )
        return 1

    errors = validate_source_decision(decision)
    if errors:
        _write_payload(_failure_payload(errors), pretty=args.pretty, stream=sys.stderr)
        return 1

    _write_payload(_success_payload(decision), pretty=args.pretty, stream=sys.stdout)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision_json", type=Path)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


def validate_source_decision(decision: Any) -> list[dict[str, Any]]:
    if not isinstance(decision, Mapping):
        return [_error("decision_not_object")]

    errors: list[dict[str, Any]] = []
    if _clean(decision.get("schema_version")) != SCHEMA_VERSION:
        errors.append(_error("schema_version_mismatch", field="schema_version"))

    source = _object_field(decision, "source", errors)
    corpus = _object_field(decision, "corpus", errors)
    labeling = _object_field(decision, "labeling", errors)

    if source is not None:
        errors.extend(_source_errors(source))
    if corpus is not None:
        errors.extend(_corpus_errors(corpus))
    if labeling is not None:
        errors.extend(_labeling_errors(labeling))
    return errors


def _source_errors(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    kind = _clean(source.get("kind"))
    supply = _clean(source.get("supply"))
    reference = _clean(source.get("reference"))
    if not kind:
        errors.append(_error("source_kind_missing", field="source.kind"))
    elif kind not in ALLOWED_SOURCE_KINDS:
        errors.append(_error("source_kind_unsupported", field="source.kind"))
    if not supply:
        errors.append(_error("source_supply_missing", field="source.supply"))
    elif supply not in ALLOWED_SOURCE_SUPPLIES:
        errors.append(_error("source_supply_unsupported", field="source.supply"))
    if not reference:
        errors.append(_error("source_reference_missing", field="source.reference"))
    elif _unsafe_reference(reference):
        errors.append(_error("source_reference_unsafe", field="source.reference"))
    return errors


def _corpus_errors(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    target = corpus.get("target_ticket_count")
    minimum = corpus.get("minimum_ticket_count")
    if not _positive_int(target):
        errors.append(_error("target_ticket_count_invalid", field="corpus.target_ticket_count"))
    if not _positive_int(minimum):
        errors.append(_error("minimum_ticket_count_invalid", field="corpus.minimum_ticket_count"))
    if _positive_int(target) and _positive_int(minimum) and int(minimum) > int(target):
        errors.append(_error("minimum_ticket_count_exceeds_target", field="corpus.minimum_ticket_count"))

    class_targets = _string_set(corpus.get("pii_class_targets"))
    if class_targets is None:
        errors.append(_error("pii_class_targets_invalid", field="corpus.pii_class_targets"))
    else:
        missing = sorted(REQUIRED_HIGH_SEVERITY_CLASSES - class_targets)
        if missing:
            errors.append(
                _error(
                    "pii_class_targets_missing_required",
                    field="corpus.pii_class_targets",
                    missing=missing,
                )
            )

    name_subtypes = _string_set(corpus.get("person_name_subtypes"))
    if name_subtypes is None:
        errors.append(_error("person_name_subtypes_invalid", field="corpus.person_name_subtypes"))
    else:
        missing = sorted(REQUIRED_PERSON_NAME_SUBTYPES - name_subtypes)
        if missing:
            errors.append(
                _error(
                    "person_name_subtypes_missing_required",
                    field="corpus.person_name_subtypes",
                    missing=missing,
                )
            )
    return errors


def _labeling_errors(labeling: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    owner = _clean(labeling.get("owner"))
    reviewer = _clean(labeling.get("reviewer"))
    quality_review = _clean(labeling.get("quality_review"))
    if not owner:
        errors.append(_error("labeling_owner_missing", field="labeling.owner"))
    elif _unsafe_reference(owner):
        errors.append(_error("labeling_owner_unsafe", field="labeling.owner"))
    if not reviewer:
        errors.append(_error("labeling_reviewer_missing", field="labeling.reviewer"))
    elif _unsafe_reference(reviewer):
        errors.append(_error("labeling_reviewer_unsafe", field="labeling.reviewer"))
    if owner and reviewer and owner == reviewer:
        errors.append(_error("labeling_reviewer_matches_owner", field="labeling.reviewer"))
    if quality_review != COMPLETED_QUALITY_REVIEW:
        errors.append(_error("quality_review_not_completed", field="labeling.quality_review"))
    return errors


def _object_field(
    decision: Mapping[str, Any],
    field: str,
    errors: list[dict[str, Any]],
) -> Mapping[str, Any] | None:
    value = decision.get(field)
    if not isinstance(value, Mapping):
        errors.append(_error(f"{field}_missing", field=field))
        return None
    return value


def _unsafe_reference(value: str) -> bool:
    if (
        len(value) > 80
        or "/" in value
        or "\\" in value
        or ".." in value
        or "://" in value
        or EMAIL_RE.search(value)
        or PHONE_RE.search(value)
        or SSN_RE.search(value)
        or _looks_like_card(value)
    ):
        return True
    return SAFE_SLUG_RE.fullmatch(value) is None


def _looks_like_card(value: str) -> bool:
    digits = "".join(DIGIT_RE.findall(value))
    return len(digits) >= 13


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _string_set(value: Any) -> set[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    normalized: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        normalized.add(item.strip())
    return normalized


def _success_payload(decision: Mapping[str, Any]) -> dict[str, Any]:
    source = _as_mapping(decision.get("source"))
    corpus = _as_mapping(decision.get("corpus"))
    labeling = _as_mapping(decision.get("labeling"))
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "kind": _clean(source.get("kind")),
            "supply": _clean(source.get("supply")),
            "reference": _clean(source.get("reference")),
        },
        "corpus": {
            "target_ticket_count": int(corpus["target_ticket_count"]),
            "minimum_ticket_count": int(corpus["minimum_ticket_count"]),
            "pii_class_targets": sorted(_string_set(corpus["pii_class_targets"]) or []),
            "person_name_subtypes": sorted(_string_set(corpus["person_name_subtypes"]) or []),
        },
        "labeling": {
            "owner": _clean(labeling.get("owner")),
            "reviewer": _clean(labeling.get("reviewer")),
            "quality_review": _clean(labeling.get("quality_review")),
        },
        "raw_source_persisted": False,
        "raw_label_spans_persisted": False,
    }


def _failure_payload(errors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": SCHEMA_VERSION,
        "errors": list(errors),
    }


def _error(code: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, **{key: value for key, value in extra.items() if value}}


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _write_payload(payload: Mapping[str, Any], *, pretty: bool, stream: Any) -> None:
    print(
        json.dumps(payload, indent=2 if pretty else None, sort_keys=True),
        file=stream,
    )


if __name__ == "__main__":
    raise SystemExit(main())
