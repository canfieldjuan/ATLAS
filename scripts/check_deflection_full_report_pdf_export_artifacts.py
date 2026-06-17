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
LEAK_PATTERNS = (
    ("request_id", re.compile(r"\brequest[_ -]?id\b", re.IGNORECASE)),
    ("result_url", re.compile(r"https?://\S+/services/faq-deflection/results/\S+", re.IGNORECASE)),
    ("customer_email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("absolute_local_path", re.compile(r"(?<!\w)(?:/home/|/tmp/)")),
    ("stripe_checkout_session_id", re.compile(r"\bcs_(?:test|live)_[A-Za-z0-9_]+\b")),
    ("stripe_payment_intent_id", re.compile(r"\bpi_[A-Za-z0-9_]+\b")),
    ("raw_evidence_quote", re.compile(r"\b(?:evidence_quote|raw quoted evidence)\b", re.IGNORECASE)),
    ("source_id_list", re.compile(r"\bticket[-_][A-Za-z0-9][A-Za-z0-9_.:-]*\b")),
    ("private_note", re.compile(r"\bprivate note\b|\binternal note\b", re.IGNORECASE)),
)


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
    forms = {str(whole)}
    if not numeric.is_integer():
        forms.add(f"{numeric:.1f}")
        forms.add(f"{numeric:.2f}")
    if money:
        forms.add(f"${whole:,}")
        forms.add(f"${numeric:,.2f}")
        forms.add(f"{whole:,}")
    return tuple(sorted(forms, key=len, reverse=True))


def _text_has_value(text: str, value: Any, *, money: bool = False) -> bool:
    haystack = _normal_text(text)
    for form in _number_forms(value, money=money):
        if re.search(rf"(?<![\w$]){re.escape(form)}(?!\w)", haystack):
            return True
    return False


def _pdf_artifact_assertions(
    *,
    pdf_bytes: bytes,
    pdf_text: str,
    counts: Mapping[str, Any],
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
        assertions.append(_assertion(
            f"artifact.pdf.text.count.{key}",
            _text_has_value(
                pdf_text,
                counts.get(key),
                money=key == "estimated_support_cost",
            ),
            expected=counts.get(key),
            actual="visible" if _text_has_value(
                pdf_text,
                counts.get(key),
                money=key == "estimated_support_cost",
            ) else "missing",
        ))
    for leak_id, pattern in LEAK_PATTERNS:
        matched = bool(pattern.search(pdf_text))
        assertions.append(_assertion(
            f"artifact.pdf.leak.{leak_id}",
            not matched,
            expected="absent",
            actual="matched" if matched else "absent",
        ))
    return assertions


def _pdf_observation(
    *,
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
        if assertion_id.startswith("artifact.pdf.text.marker.") and assertion.get("ok") is not True:
            return {"counts": {}}
        if assertion_id.startswith("artifact.pdf.leak.") and assertion.get("ok") is not True:
            return {"counts": {}}

    visible_counts: dict[str, Any] = {}
    for key in PDF_COUNT_KEYS:
        if _text_has_value(pdf_text, counts.get(key), money=key == "estimated_support_cost"):
            visible_counts[key] = counts.get(key)
    return {"counts": visible_counts}


def build_pdf_export_scorecard(
    *,
    report_model: dict[str, Any],
    evidence_export: dict[str, Any],
    pdf_bytes: bytes,
    pdf_text: str,
    required_surfaces: tuple[str, ...] = DEFAULT_REQUIRED_SURFACES,
) -> dict[str, Any]:
    """Return a sanitized scorecard for PDF/export artifacts."""

    counts = _model_counts(report_model)
    artifact_assertions = _pdf_artifact_assertions(
        pdf_bytes=pdf_bytes,
        pdf_text=pdf_text,
        counts=counts,
    )
    observations = {
        "pdf": _pdf_observation(
            pdf_text=pdf_text,
            counts=counts,
            artifact_assertions=artifact_assertions,
        ),
        "evidence_export": _evidence_export_observation(evidence_export),
    }
    scorecard = build_deflection_full_report_qa_deterministic_harness(
        report_model,
        evidence_export=evidence_export,
        surface_observations=observations,
        surface_caps={"pdf": {}},
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
            "questions": len(evidence_export.get("questions") or []),
            "evidence_rows": len(evidence_export.get("evidence_rows") or []),
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
