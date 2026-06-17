#!/usr/bin/env python3
"""Validate deflection full-report PDF/export artifacts against the QA scorecard."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS,
    build_deflection_full_report_qa_deterministic_harness,
)


DEFAULT_REQUIRED_SURFACES = ("pdf", "evidence_export")
PDF_COUNT_KEYS = (
    "repeat_ticket_count",
    "generated_question_count",
    "ranked_question_count",
    "drafted_answer_count",
    "no_proven_answer_count",
    "ticket_source_count",
    "estimated_support_cost",
)
REQUIRED_PDF_MARKERS = (
    "Support Tax Confirmation",
    "Ranked Question Opportunities",
    "Question Details and Evidence",
    "complete evidence export",
)
PDF_COUNT_PATTERNS = {
    "repeat_ticket_count": (
        r"\b{value}\s+question-level\s+repeat tickets\b",
        r"\b{value}\s+repeat tickets\b",
    ),
    "generated_question_count": (
        r"\bacross\s+{value}\s+ranked questions\b",
    ),
    "ranked_question_count": (
        r"\b{value}\s+ranked questions\b",
        r"\b{value}\s+ranked question opportunities\b",
    ),
    "drafted_answer_count": (
        r"\bpublishable answers drafted(?: from proven resolutions)?\s*:\s*{value}\b",
        r"\b{value}\s+publishable answers drafted\b",
        r"\b{value}\s+questions?\s+have publishable answers\b",
    ),
    "no_proven_answer_count": (
        r"\bquestions still needing an approved resolution\s*:\s*{value}\b",
        r"\b{value}\s+questions?\s+(?:still\s+)?have no proven answer\b",
        r"\bno proven answers?\s*:\s*{value}\b",
    ),
    "ticket_source_count": (
        r"\bticket sources represented\s*:\s*{value}\b",
        r"\bgrounded in\s+{value}\s+source tickets\b",
        r"\b{value}\s+source tickets\b",
        r"\b{value}\s+source rows\b",
    ),
    "estimated_support_cost": (
        r"\bsizes to about\s+{value}\b",
        r"(?<!\w){value}\s+of assisted-contact handling\b",
        r"(?<!\w){value}\s+estimated assisted-contact handling\b",
        r"\bsupport cost\s*:\s*{value}\b",
    ),
}
LEAK_PATTERNS = (
    (
        "request_id",
        re.compile(
            r"\b(?:request[_ -]?id|content-ops-[A-Za-z0-9_-]{8,})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "result_url",
        re.compile(
            r"https?://[^\s\"'<>]+/"
            r"(?:"
            r"(?:systems/support-ticket-deflection|services/faq-deflection)/results"
            r"|content-ops/deflection-reports"
            r")/"
            r"[^\s\"'<>]+",
            re.IGNORECASE,
        ),
    ),
    ("customer_email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("absolute_local_path", re.compile(r"(?<!\w)(?:/home/|/tmp/)")),
    ("stripe_checkout_session_id", re.compile(r"\bcs_(?:test|live)_[A-Za-z0-9_]+\b")),
    ("stripe_payment_intent_id", re.compile(r"\bpi_[A-Za-z0-9_]+\b")),
    ("private_note", re.compile(r"\bprivate note\b|\binternal note\b", re.IGNORECASE)),
)
RAW_EVIDENCE_QUOTE_MARKER_PATTERN = re.compile(
    r"\b(?:evidence_quote|raw quoted evidence)\b",
    re.IGNORECASE,
)
SOURCE_ID_TOKEN_PATTERN = re.compile(r"\bticket[-_][A-Za-z0-9][A-Za-z0-9_.:-]*\b")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} must be readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return payload


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"{label} must be readable text: {exc}") from exc


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"{label} must be readable bytes: {exc}") from exc


def _section_data(model: Mapping[str, Any], section_id: str) -> Mapping[str, Any]:
    sections = model.get("sections")
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes, bytearray)):
        return {}
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        if str(section.get("id") or "") != section_id:
            continue
        data = section.get("data")
        return data if isinstance(data, Mapping) else {}
    return {}


def _rows_count(data: Mapping[str, Any]) -> int:
    rows = data.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return 0
    return len([row for row in rows if isinstance(row, Mapping)])


def _rows(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = data.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _sequence(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(value)


def _number(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric >= 0 else 0.0


def _int(value: Any) -> int:
    return int(_number(value))


def _model_counts(model: Mapping[str, Any]) -> dict[str, Any]:
    support_tax = _section_data(model, "support_tax")
    ranked_questions = _section_data(model, "ranked_questions")
    return {
        "repeat_ticket_count": _int(support_tax.get("repeat_ticket_count")),
        "generated_question_count": _int(support_tax.get("generated_question_count")),
        "ranked_question_count": _rows_count(ranked_questions),
        "drafted_answer_count": _int(support_tax.get("drafted_answer_count")),
        "no_proven_answer_count": _int(support_tax.get("no_proven_answer_count")),
        "ticket_source_count": _int(support_tax.get("ticket_source_count")),
        "estimated_support_cost": _number(support_tax.get("estimated_support_cost")),
    }


def _evidence_export_observation(evidence_export: Mapping[str, Any]) -> dict[str, Any]:
    summary = evidence_export.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    return {
        "counts": {
            "evidence_question_count": _int(summary.get("question_count")),
            "evidence_row_count": _int(summary.get("evidence_row_count")),
            "source_id_count": _int(summary.get("source_id_count")),
            "drafted_answer_count": _int(summary.get("drafted_answer_count")),
            "no_proven_answer_count": _int(summary.get("no_proven_answer_count")),
        },
    }


def _assertion(assertion_id: str, ok: bool, *, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "id": assertion_id,
        "ok": bool(ok),
        "expected": expected,
        "actual": actual,
    }


def _normal_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _number_forms(value: Any, *, money: bool = False) -> tuple[str, ...]:
    numeric = _number(value)
    whole = int(numeric)
    if money:
        rounded = int(numeric + 0.5)
        forms = {f"${rounded:,}", f"${numeric:,.2f}"}
        if numeric.is_integer():
            forms.add(f"${whole:,}")
    else:
        forms = {str(whole), f"{whole:,}"}
        if not numeric.is_integer():
            forms.add(f"{numeric:.1f}")
            forms.add(f"{numeric:.2f}")
    return tuple(sorted(forms, key=len, reverse=True))


def _count_line_candidates(text: str) -> list[str]:
    lines = [_normal_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    candidates = list(lines)
    for index, line in enumerate(lines[:-1]):
        candidates.append(_normal_text(f"{line} {lines[index + 1]}"))
    return candidates


def _line_has_labeled_value(line: str, *, patterns: Sequence[str], forms: Sequence[str]) -> bool:
    for pattern in patterns:
        for form in forms:
            value_pattern = pattern.format(value=re.escape(form))
            if re.search(value_pattern, line, re.IGNORECASE):
                return True
    return False


def _text_has_labeled_value(text: str, key: str, value: Any, *, money: bool = False) -> bool:
    patterns = PDF_COUNT_PATTERNS.get(key, ())
    if not patterns:
        return False
    forms = _number_forms(value, money=money)
    return any(
        _line_has_labeled_value(candidate, patterns=patterns, forms=forms)
        for candidate in _count_line_candidates(text)
    )


def _source_ids_from_export(evidence_export: Mapping[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    for row in _sequence(evidence_export.get("evidence_rows")):
        if isinstance(row, Mapping):
            source_id = str(row.get("source_id") or "").strip()
            if _source_id_candidate(source_id):
                source_ids.add(source_id)
    for question in _sequence(evidence_export.get("questions")):
        if not isinstance(question, Mapping):
            continue
        for raw_source_id in _sequence(question.get("source_ids")):
            source_id = str(raw_source_id or "").strip()
            if _source_id_candidate(source_id):
                source_ids.add(source_id)
    return source_ids


def _source_ids_from_model(report_model: Mapping[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    for section_id in ("ranked_questions", "question_details"):
        for row in _rows(_section_data(report_model, section_id)):
            for raw_source_id in _sequence(row.get("source_ids")):
                source_id = str(raw_source_id or "").strip()
                if _source_id_candidate(source_id):
                    source_ids.add(source_id)
    return source_ids


def _source_id_candidate(value: str) -> bool:
    return len(value) >= 3


def _text_contains_source_id(text: str, source_id: str) -> bool:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_.:-]){re.escape(source_id)}(?![A-Za-z0-9_.:-])"
    )
    return bool(pattern.search(text))


def _source_id_leak_present(text: str, source_ids: set[str]) -> bool:
    if SOURCE_ID_TOKEN_PATTERN.search(text):
        return True
    return any(_text_contains_source_id(text, source_id) for source_id in source_ids)


def _evidence_quotes_from_export(evidence_export: Mapping[str, Any]) -> set[str]:
    quotes: set[str] = set()
    for row in _sequence(evidence_export.get("evidence_rows")):
        if not isinstance(row, Mapping):
            continue
        quote = _normal_text(str(row.get("evidence_quote") or ""))
        if len(quote) >= 24:
            quotes.add(quote)
    return quotes


def _raw_evidence_quote_leak_present(
    text: str,
    evidence_export: Mapping[str, Any],
) -> bool:
    if RAW_EVIDENCE_QUOTE_MARKER_PATTERN.search(text):
        return True
    haystack = _normal_text(text).casefold()
    return any(
        quote.casefold() in haystack
        for quote in _evidence_quotes_from_export(evidence_export)
    )


def _section_text(text: str, marker: str, end_markers: Sequence[str]) -> str:
    haystack = text.casefold()
    start = haystack.find(marker.casefold())
    if start < 0:
        return ""
    section_start = start + len(marker)
    ends = [
        index
        for end_marker in end_markers
        if (index := haystack.find(end_marker.casefold(), section_start)) >= 0
    ]
    section_end = min(ends) if ends else len(text)
    return text[section_start:section_end]


def _visible_question_rows(section_text: str, rows: Sequence[Mapping[str, Any]]) -> int:
    haystack = _normal_text(section_text).casefold()
    visible = 0
    for row in rows:
        question = _normal_text(str(row.get("question") or ""))
        if question and question.casefold() in haystack:
            visible += 1
    return visible


def _pdf_displayed_rows(report_model: Mapping[str, Any], pdf_text: str) -> dict[str, int]:
    ranked_section = _section_text(
        pdf_text,
        "Ranked Question Opportunities",
        ("Resolution Outcome Diagnostics", "Question Details and Evidence"),
    )
    detail_section = _section_text(
        pdf_text,
        "Question Details and Evidence",
        ("Complete Evidence",),
    )
    ranked_rows = _rows(_section_data(report_model, "ranked_questions"))
    detail_rows = _rows(_section_data(report_model, "question_details"))
    caps = DEFAULT_DEFLECTION_FULL_REPORT_SURFACE_CAPS.get("pdf", {})
    ranked_cap = _int(caps.get("ranked_questions")) if isinstance(caps, Mapping) else 0
    detail_cap = _int(caps.get("question_details")) if isinstance(caps, Mapping) else 0
    return {
        "ranked_questions": _visible_question_rows(
            ranked_section,
            ranked_rows[:ranked_cap] if ranked_cap else ranked_rows,
        ),
        "question_details": _visible_question_rows(
            detail_section,
            detail_rows[:detail_cap] if detail_cap else detail_rows,
        ),
    }

def _pdf_artifact_assertions(
    *,
    pdf_bytes: bytes,
    pdf_text: str,
    counts: Mapping[str, Any],
    evidence_export: Mapping[str, Any],
    source_ids: set[str],
) -> list[dict[str, Any]]:
    assertions = [
        _assertion(
            "artifact.pdf.bytes.header",
            pdf_bytes.startswith(b"%PDF-"),
            expected="%PDF-",
            actual="present" if pdf_bytes.startswith(b"%PDF-") else "missing",
        ),
        _assertion(
            "artifact.pdf.bytes.minimum_size",
            len(pdf_bytes) >= 32,
            expected=">=32 bytes",
            actual=len(pdf_bytes),
        ),
    ]
    normalized_text = _normal_text(pdf_text).casefold()
    for marker in REQUIRED_PDF_MARKERS:
        marker_id = re.sub(r"[^a-z0-9]+", "_", marker.casefold()).strip("_")
        assertions.append(_assertion(
            f"artifact.pdf.text.marker.{marker_id}",
            marker.casefold() in normalized_text,
            expected="present",
            actual="present" if marker.casefold() in normalized_text else "missing",
        ))
    for key in PDF_COUNT_KEYS:
        visible = _text_has_labeled_value(
            pdf_text,
            key,
            counts.get(key),
            money=key == "estimated_support_cost",
        )
        assertions.append(_assertion(
            f"artifact.pdf.text.count.{key}",
            visible,
            expected=counts.get(key),
            actual="visible" if visible else "missing",
        ))
    for leak_id, pattern in LEAK_PATTERNS:
        matched = bool(pattern.search(pdf_text))
        assertions.append(_assertion(
            f"artifact.pdf.leak.{leak_id}",
            not matched,
            expected="absent",
            actual="matched" if matched else "absent",
        ))
    quote_matched = _raw_evidence_quote_leak_present(pdf_text, evidence_export)
    assertions.append(_assertion(
        "artifact.pdf.leak.raw_evidence_quote",
        not quote_matched,
        expected="absent",
        actual="matched" if quote_matched else "absent",
    ))
    source_id_matched = _source_id_leak_present(pdf_text, source_ids)
    assertions.append(_assertion(
        "artifact.pdf.leak.source_id_list",
        not source_id_matched,
        expected="absent",
        actual="matched" if source_id_matched else "absent",
    ))
    return assertions


def _pdf_observation(
    *,
    report_model: Mapping[str, Any],
    pdf_text: str,
    counts: Mapping[str, Any],
    artifact_assertions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    hard_failures = {
        "artifact.pdf.bytes.header",
        "artifact.pdf.bytes.minimum_size",
    }
    for assertion in artifact_assertions:
        assertion_id = str(assertion.get("id") or "")
        if assertion_id in hard_failures and assertion.get("ok") is not True:
            return {"counts": {}}
        if (
            assertion_id.startswith("artifact.pdf.text.marker.")
            and assertion.get("ok") is not True
        ):
            return {"counts": {}}
        if assertion_id.startswith("artifact.pdf.leak.") and assertion.get("ok") is not True:
            return {"counts": {}}

    visible_counts: dict[str, Any] = {}
    for key in PDF_COUNT_KEYS:
        if _text_has_labeled_value(
            pdf_text,
            key,
            counts.get(key),
            money=key == "estimated_support_cost",
        ):
            visible_counts[key] = counts.get(key)
    return {
        "counts": visible_counts,
        "displayed_rows": _pdf_displayed_rows(report_model, pdf_text),
    }


def _safe_sequence_len(value: Any) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return 0
    return len(value)


def build_pdf_export_scorecard(
    *,
    report_model: Any,
    evidence_export: Any,
    pdf_bytes: bytes,
    pdf_text: str,
    required_surfaces: tuple[str, ...] = DEFAULT_REQUIRED_SURFACES,
) -> dict[str, Any]:
    """Return a sanitized scorecard for PDF/export artifacts."""

    report_payload = report_model if isinstance(report_model, Mapping) else {}
    export_payload = evidence_export if isinstance(evidence_export, Mapping) else {}
    counts = _model_counts(report_payload)
    source_ids = (
        _source_ids_from_export(export_payload)
        | _source_ids_from_model(report_payload)
    )
    artifact_assertions = _pdf_artifact_assertions(
        pdf_bytes=pdf_bytes,
        pdf_text=pdf_text,
        counts=counts,
        evidence_export=export_payload,
        source_ids=source_ids,
    )
    observations = {
        "pdf": _pdf_observation(
            report_model=report_payload,
            pdf_text=pdf_text,
            counts=counts,
            artifact_assertions=artifact_assertions,
        ),
        "evidence_export": _evidence_export_observation(export_payload),
    }
    scorecard = build_deflection_full_report_qa_deterministic_harness(
        report_payload,
        evidence_export=export_payload,
        surface_observations=observations,
        required_surfaces=required_surfaces,
    )
    assertions = [dict(assertion) for assertion in scorecard["assertions"]]
    assertions.extend(artifact_assertions)
    result = dict(scorecard)
    result["ok"] = all(assertion["ok"] for assertion in assertions)
    result["assertions"] = assertions
    result["surfaces"] = dict(scorecard.get("surfaces", {}))
    result["artifacts"] = {
        "pdf": {
            "bytes": len(pdf_bytes),
            "text_chars": len(pdf_text),
        },
        "evidence_export": {
            "questions": _safe_sequence_len(export_payload.get("questions")),
            "evidence_rows": _safe_sequence_len(export_payload.get("evidence_rows")),
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-model", type=Path, required=True)
    parser.add_argument("--evidence-export", type=Path, required=True)
    parser.add_argument("--pdf-bytes", type=Path, required=True)
    parser.add_argument("--pdf-text", type=Path, required=True)
    parser.add_argument(
        "--required-surface",
        action="append",
        default=[],
        help="Required artifact surface. Repeat for multiple surfaces.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    scorecard = build_pdf_export_scorecard(
        report_model=_load_object(args.report_model, "--report-model"),
        evidence_export=_load_object(args.evidence_export, "--evidence-export"),
        pdf_bytes=_read_bytes(args.pdf_bytes, "--pdf-bytes"),
        pdf_text=_read_text(args.pdf_text, "--pdf-text"),
        required_surfaces=tuple(args.required_surface) or DEFAULT_REQUIRED_SURFACES,
    )
    text = json.dumps(scorecard, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if scorecard.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
