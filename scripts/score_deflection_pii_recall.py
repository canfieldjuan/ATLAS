#!/usr/bin/env python3
"""Score deflection PII recall and must-survive precision on a surrogate corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from atlas_brain.deflection_pdf_renderer import (  # noqa: E402
        _artifact_report_model_pdf_markdown,
        render_deflection_full_report_pdf,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by monkeypatch.
    if exc.name != "fpdf":
        raise
    _PDF_RENDERER_IMPORT_ERROR: ModuleNotFoundError | None = exc
    _artifact_report_model_pdf_markdown = None  # type: ignore[assignment]
    render_deflection_full_report_pdf = None  # type: ignore[assignment]
else:
    _PDF_RENDERER_IMPORT_ERROR = None
from extracted_content_pipeline.deflection_pii_eval_corpus import (  # noqa: E402
    SCHEMA_VERSION as CORPUS_SCHEMA_VERSION,
)
from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    build_deflection_report_artifact,
    build_deflection_snapshot,
    scrub_deflection_report_payload,
)
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
    TicketFAQMarkdownResult,
)


SCORE_SCHEMA_VERSION = "deflection_pii_recall_score.v1"
DEFAULT_CORPUS = (
    ROOT
    / "docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json"
)
SURFACES = ("free_snapshot", "free_teaser", "paid_artifact", "paid_pdf")
FREE_SURFACES = frozenset({"free_snapshot", "free_teaser"})
HIGH_SEVERITY_CLASSES = frozenset({
    "email",
    "phone",
    "ssn",
    "payment_card",
    "person_name",
    "dob",
})
# Keep the curated surrogate corpus on 3+ character name tokens; two-letter
# surnames need an explicit scorer/corpus follow-up before they are reliable.
PERSON_NAME_TOKEN_MIN_LENGTH = 3
PERSON_NAME_TOKEN_STOPWORDS = frozenset({
    "agent",
    "client",
    "contact",
    "customer",
    "member",
    "name",
    "person",
    "requester",
    "user",
})


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus, errors = _load_corpus(args.corpus)
    if errors:
        summary = _error_summary(errors)
    else:
        summary = score_corpus(corpus)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _markdown_summary(summary),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "ok":
        print(
            "Deflection PII recall scoring failed: "
            + ", ".join(summary["blocking_error_codes"]),
            file=sys.stderr,
        )
        return 1
    return 0


def score_corpus(corpus: Mapping[str, Any]) -> dict[str, Any]:
    errors = _corpus_errors(corpus)
    if errors:
        return _error_summary(errors)

    tickets = tuple(_ticket for _ticket in _sequence(corpus.get("tickets")) if isinstance(_ticket, Mapping))
    artifact = build_deflection_report_artifact(
        _faq_result_from_corpus(tickets),
        source_label="deflection_pii_eval_corpus",
    )
    baseline_artifact = artifact.as_dict()
    scrubbed_artifact = scrub_deflection_report_payload(baseline_artifact)
    baseline_snapshot = build_deflection_snapshot(baseline_artifact).as_dict()
    scrubbed_snapshot = build_deflection_snapshot(scrubbed_artifact).as_dict()

    baseline_pdf_text, scrubbed_pdf_text, paid_pdf_generation = _paid_pdf_surfaces(
        baseline_artifact=baseline_artifact,
        scrubbed_artifact=scrubbed_artifact,
    )
    baseline_surfaces = _surface_texts(
        artifact=baseline_artifact,
        snapshot=baseline_snapshot,
        pdf_text=baseline_pdf_text,
    )
    scrubbed_surfaces = _surface_texts(
        artifact=scrubbed_artifact,
        snapshot=scrubbed_snapshot,
        pdf_text=scrubbed_pdf_text,
    )
    scrubbed_surfaces_by_ticket = _scrubbed_surface_texts_by_ticket(tickets)

    labels = _labels(tickets)
    must_survive = _must_survive(tickets)
    label_scores, leak_samples = _score_labels(
        labels=labels,
        baseline_surfaces=baseline_surfaces,
        scrubbed_surfaces=scrubbed_surfaces,
        scrubbed_surfaces_by_ticket=scrubbed_surfaces_by_ticket,
    )
    must_survive_result = _score_must_survive(
        records=must_survive,
        baseline_surfaces=baseline_surfaces,
        scrubbed_surfaces=scrubbed_surfaces,
    )
    surface_summary = _surface_summary(label_scores)
    person_name_summary = _person_name_summary(label_scores)
    headline_summary = _headline_summary(label_scores)

    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "ok",
        "blocking_error_codes": [],
        "input": {
            "schema_version": _clean(corpus.get("schema_version")),
            "ticket_count": len(tickets),
            "label_count": len(labels),
            "must_survive_count": len(must_survive),
        },
        "headline": headline_summary,
        "surfaces": surface_summary,
        "person_name": person_name_summary,
        "must_survive": must_survive_result,
        "leak_samples": leak_samples[:20],
        "surface_generation": {
            "paid_pdf": paid_pdf_generation,
        },
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Deflection PII Recall Advisory",
        "",
        _markdown_table(
            ("Field", "Value"),
            (
                ("Status", _clean(summary.get("status"))),
                ("Schema", _clean(summary.get("schema_version"))),
            ),
        ),
        "",
    ]
    if _clean(summary.get("status")) != "ok":
        lines.extend([
            "## Blocking Errors",
            "",
            _markdown_table(
                ("Code",),
                ((code,) for code in _sequence(summary.get("blocking_error_codes"))),
            ),
            "",
        ])
        return "\n".join(lines).rstrip() + "\n"

    input_summary = summary.get("input")
    if not isinstance(input_summary, Mapping):
        input_summary = {}
    headline = summary.get("headline")
    if not isinstance(headline, Mapping):
        headline = {}
    lines.extend([
        "## Input",
        "",
        _markdown_table(
            ("Metric", "Value"),
            (
                ("Corpus schema", _clean(input_summary.get("schema_version"))),
                ("Tickets", input_summary.get("ticket_count", 0)),
                ("Labels", input_summary.get("label_count", 0)),
                ("Must-survive records", input_summary.get("must_survive_count", 0)),
            ),
        ),
        "",
        "## Headline",
        "",
        _markdown_table(
            ("Metric", "Value"),
            (
                ("Free high-severity leaks (all-in)", headline.get("free_high_severity_leak_count", 0)),
                ("Free high-severity pass (all-in)", headline.get("free_high_severity_pass", False)),
                ("Gate-eligible free high-severity leaks", headline.get("free_high_severity_gate_eligible_leak_count", 0)),
                ("Gate-eligible pass", headline.get("free_high_severity_gate_eligible_pass", False)),
                ("Deferred open-set name leaks", headline.get("deferred_open_set_name_leak_count", 0)),
            ),
        ),
        "",
        "## Person Name Split",
        "",
        _class_rows_table(summary.get("person_name"), first_header="Subtype"),
        "",
        "## Surface Recall",
        "",
        _surface_rows_table(summary.get("surfaces")),
        "",
        "## Must-Survive Precision",
        "",
        _must_survive_table(summary.get("must_survive")),
        "",
        "## Leak Samples",
        "",
        _leak_samples_table(summary.get("leak_samples")),
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _class_rows_table(value: Any, *, first_header: str) -> str:
    if not isinstance(value, Mapping):
        return _markdown_table((first_header, "Expected", "Redacted", "Leaks", "Recall"), ())
    return _markdown_table(
        (first_header, "Expected", "Redacted", "Leaks", "Recall"),
        (
            (
                key,
                counts.get("expected", 0),
                counts.get("redacted", 0),
                counts.get("leaks", 0),
                _format_recall(counts.get("recall")),
            )
            for key, counts in sorted(value.items())
            if isinstance(counts, Mapping)
        ),
    )


def _surface_rows_table(value: Any) -> str:
    rows: list[tuple[Any, ...]] = []
    if isinstance(value, Mapping):
        for surface, class_counts in sorted(value.items()):
            if not isinstance(class_counts, Mapping):
                continue
            for pii_class, counts in sorted(class_counts.items()):
                if not isinstance(counts, Mapping):
                    continue
                rows.append((
                    surface,
                    pii_class,
                    counts.get("expected", 0),
                    counts.get("redacted", 0),
                    counts.get("leaks", 0),
                    _format_recall(counts.get("recall")),
                ))
    return _markdown_table(
        ("Surface", "Class", "Expected", "Redacted", "Leaks", "Recall"),
        rows,
    )


def _must_survive_table(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _markdown_table(("Metric", "Value"), ())
    rows: list[tuple[Any, ...]] = [
        ("Violation count", value.get("violation_count", 0)),
    ]
    surface_counts = value.get("surface_counts")
    if isinstance(surface_counts, Mapping):
        for surface, counts in sorted(surface_counts.items()):
            if not isinstance(counts, Mapping):
                continue
            rows.append((
                f"{surface} reached/total",
                f"{counts.get('reached_surface', 0)}/{counts.get('total_tokens', 0)}",
            ))
    return _markdown_table(("Metric", "Value"), rows)


def _leak_samples_table(value: Any) -> str:
    rows: list[tuple[Any, ...]] = []
    for sample in _sequence(value):
        if not isinstance(sample, Mapping):
            continue
        rows.append((
            _clean(sample.get("surrogate_id")),
            _clean(sample.get("class")),
            _clean(sample.get("name_subtype")),
            _clean(sample.get("severity")),
            _clean(sample.get("surface")),
            _clean(sample.get("leak_kind")),
        ))
    return _markdown_table(
        ("Surrogate ID", "Class", "Name subtype", "Severity", "Surface", "Leak kind"),
        rows,
    )


def _markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    row_list = list(rows)
    header_line = "| " + " | ".join(_markdown_cell(header) for header in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(cell) for cell in row) + " |"
        for row in row_list
    ]
    if not body:
        body = ["| " + " | ".join("None" for _ in headers) + " |"]
    return "\n".join([header_line, separator, *body])


def _markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_recall(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return "n/a"


def _load_corpus(path: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, [_error("corpus_load_failed")]
    if not isinstance(raw, Mapping):
        return {}, [_error("corpus_not_object")]
    return raw, []


def _corpus_errors(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if _clean(corpus.get("schema_version")) != CORPUS_SCHEMA_VERSION:
        errors.append(_error("corpus_schema_version_mismatch"))
    tickets = _sequence(corpus.get("tickets"))
    if not tickets:
        errors.append(_error("corpus_empty_tickets"))
    for index, ticket in enumerate(tickets, start=1):
        if not isinstance(ticket, Mapping):
            errors.append(_error("ticket_not_object", ticket_index=index))
            continue
        labels = _sequence(ticket.get("labels"))
        if not labels:
            errors.append(_error("ticket_missing_labels", ticket_index=index))
        for label_index, label in enumerate(labels, start=1):
            if not isinstance(label, Mapping):
                errors.append(
                    _error(
                        "label_not_object",
                        ticket_index=index,
                        label_index=label_index,
                    )
                )
                continue
            if not _clean(label.get("span")):
                errors.append(
                    _error(
                        "label_missing_span",
                        ticket_index=index,
                        label_index=label_index,
                    )
                )
        for record_index, record in enumerate(
            _sequence(ticket.get("must_survive")),
            start=1,
        ):
            if not isinstance(record, Mapping):
                errors.append(
                    _error(
                        "must_survive_not_object",
                        ticket_index=index,
                        record_index=record_index,
                    )
                )
                continue
            if not _clean(record.get("span")):
                errors.append(
                    _error(
                        "must_survive_missing_span",
                        ticket_index=index,
                        record_index=record_index,
                    )
                )
    return errors


def _paid_pdf_surfaces(
    *,
    baseline_artifact: Mapping[str, Any],
    scrubbed_artifact: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    if (
        _PDF_RENDERER_IMPORT_ERROR is not None
        or _artifact_report_model_pdf_markdown is None
        or render_deflection_full_report_pdf is None
    ):
        return "", "", {
            "rendered": False,
            "skipped": True,
            "skip_reason": _pdf_skip_reason(),
            "byte_count": 0,
            "scored_text_source": None,
        }

    pdf_rendered_bytes = render_deflection_full_report_pdf(scrubbed_artifact)
    return (
        _artifact_report_model_pdf_markdown(baseline_artifact),
        _artifact_report_model_pdf_markdown(scrubbed_artifact),
        {
            "rendered": pdf_rendered_bytes.startswith(b"%PDF-"),
            "skipped": False,
            "byte_count": len(pdf_rendered_bytes),
            "scored_text_source": "renderer_curated_pdf_markdown",
        },
    )


def _pdf_skip_reason() -> str:
    if _PDF_RENDERER_IMPORT_ERROR is None:
        return ""
    missing_name = _clean(getattr(_PDF_RENDERER_IMPORT_ERROR, "name", ""))
    if missing_name:
        return f"missing_optional_dependency:{missing_name}"
    return "pdf_renderer_unavailable"


def _faq_result_from_corpus(tickets: Sequence[Mapping[str, Any]]) -> TicketFAQMarkdownResult:
    items: list[dict[str, Any]] = []
    markdown_lines = ["# Deflection PII Recall Corpus", ""]
    for index, ticket in enumerate(tickets, start=1):
        fields = ticket.get("fields") if isinstance(ticket.get("fields"), Mapping) else {}
        question = _first_text(
            fields.get("subject"),
            fields.get("customer_message"),
            f"PII eval ticket {index}",
        )
        customer_wording = _first_text(fields.get("customer_message"), question)
        answer = _clean(fields.get("agent_reply")) or "Resolution evidence was present."
        source_id = _first_text(fields.get("source_id"), ticket.get("ticket_id"), f"pii-eval-{index:03d}")
        evidence_quotes = [
            value
            for value in (
                _clean(fields.get("customer_message")),
                _clean(fields.get("agent_reply")),
                _clean(fields.get("private_note")),
            )
            if value
        ]
        item = {
            "question": question,
            "customer_wording": customer_wording,
            "topic": "pii recall eval",
            "weighted_frequency": 1,
            "ticket_count": 1,
            "opportunity_score": 1,
            "answer": answer,
            "answer_evidence_status": "resolution_evidence",
            "resolution_evidence_scope": "scoped",
            "steps": (answer,),
            "source_ids": (source_id,),
            "evidence_quotes": tuple(evidence_quotes),
        }
        items.append(item)
        markdown_lines.extend([
            f"## {index}. {question}",
            "",
            answer,
            "",
        ])
    return TicketFAQMarkdownResult(
        markdown="\n".join(markdown_lines).strip() + "\n",
        items=tuple(items),
        source_count=len(tickets),
        ticket_source_count=len(tickets),
        output_checks={"pii_recall_eval": True},
    )


def _surface_texts(
    *,
    artifact: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    pdf_text: str,
) -> dict[str, str]:
    snapshot_without_teaser = {
        key: value
        for key, value in snapshot.items()
        if key != "teaser"
    }
    return {
        "free_snapshot": _surface_text(snapshot_without_teaser),
        "free_teaser": _surface_text(snapshot.get("teaser")),
        "paid_artifact": _surface_text(artifact),
        "paid_pdf": _surface_text(pdf_text),
    }


def _scrubbed_surface_texts_by_ticket(
    tickets: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    surfaces: dict[str, dict[str, str]] = {}
    for index, ticket in enumerate(tickets, start=1):
        artifact = build_deflection_report_artifact(
            _faq_result_from_corpus((ticket,)),
            source_label="deflection_pii_eval_corpus",
        )
        scrubbed_artifact = scrub_deflection_report_payload(artifact.as_dict())
        scrubbed_snapshot = build_deflection_snapshot(scrubbed_artifact).as_dict()
        _, scrubbed_pdf_text, _ = _paid_pdf_surfaces(
            baseline_artifact=artifact.as_dict(),
            scrubbed_artifact=scrubbed_artifact,
        )
        surfaces[_ticket_key(ticket, index)] = _surface_texts(
            artifact=scrubbed_artifact,
            snapshot=scrubbed_snapshot,
            pdf_text=scrubbed_pdf_text,
        )
    return surfaces


def _score_labels(
    *,
    labels: Sequence[Mapping[str, Any]],
    baseline_surfaces: Mapping[str, str],
    scrubbed_surfaces: Mapping[str, str],
    scrubbed_surfaces_by_ticket: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    leaks: list[dict[str, Any]] = []
    for label in labels:
        span = _clean(label.get("span"))
        if not span:
            continue
        for surface in SURFACES:
            baseline_text = baseline_surfaces.get(surface, "")
            scrubbed_text = scrubbed_surfaces.get(surface, "")
            reaches = span in baseline_text
            label_scrubbed_text = scrubbed_surfaces_by_ticket.get(
                _clean(label.get("_ticket_key")),
                {},
            ).get(surface, scrubbed_text)
            leak_kind = _label_leak_kind(
                label=label,
                baseline_text=baseline_text,
                scrubbed_text=label_scrubbed_text,
            )
            leaked = bool(leak_kind)
            score = {
                "surrogate_id": _clean(label.get("surrogate_id")),
                "class": _clean(label.get("class")),
                "severity": _clean(label.get("severity")),
                "name_subtype": _clean(label.get("name_subtype")),
                "surface": surface,
                "reaches_surface": reaches,
                "leaked": leaked,
                "leak_kind": leak_kind,
            }
            scores.append(score)
            if leaked:
                leaks.append({
                    "surrogate_id": score["surrogate_id"],
                    "class": score["class"],
                    "severity": score["severity"],
                    "surface": surface,
                    "name_subtype": score["name_subtype"],
                    "leak_kind": leak_kind,
                })
    return scores, leaks


def _label_leak_kind(
    *,
    label: Mapping[str, Any],
    baseline_text: str,
    scrubbed_text: str,
) -> str:
    span = _clean(label.get("span"))
    if not span or span not in baseline_text:
        return ""
    if span in scrubbed_text:
        return "full_span"
    if _clean(label.get("class")) != "person_name":
        return ""
    if _person_name_token_survives(span, scrubbed_text):
        return "partial_name_token"
    return ""


def _person_name_token_survives(span: str, scrubbed_text: str) -> bool:
    for token in _person_name_tokens(span):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", scrubbed_text):
            return True
    return False


def _person_name_tokens(span: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", span)
        if len(token) >= PERSON_NAME_TOKEN_MIN_LENGTH
        and token.casefold() not in PERSON_NAME_TOKEN_STOPWORDS
    )


def _score_must_survive(
    *,
    records: Sequence[Mapping[str, Any]],
    baseline_surfaces: Mapping[str, str],
    scrubbed_surfaces: Mapping[str, str],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    reached: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        span = _clean(record.get("span"))
        if not span:
            continue
        for surface in SURFACES:
            totals[surface] += 1
            if span not in baseline_surfaces.get(surface, ""):
                continue
            reached[surface] += 1
            if span not in scrubbed_surfaces.get(surface, ""):
                violations.append({
                    "must_survive_id": f"must-survive-{index:03d}",
                    "reason": _clean(record.get("reason")),
                    "surface": surface,
                })
    return {
        "violation_count": len(violations),
        "violations": violations,
        "surface_counts": {
            surface: {
                "total_tokens": totals[surface],
                "reached_surface": reached[surface],
            }
            for surface in SURFACES
        },
    }


def _surface_summary(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Counter[str]]] = {
        surface: defaultdict(Counter)
        for surface in SURFACES
    }
    for score in scores:
        surface = _clean(score.get("surface"))
        pii_class = _clean(score.get("class"))
        if surface not in buckets or not pii_class:
            continue
        if score.get("reaches_surface") is True:
            buckets[surface][pii_class]["expected"] += 1
        if score.get("leaked") is True:
            buckets[surface][pii_class]["leaks"] += 1
    return {
        surface: {
            pii_class: _class_summary(counts)
            for pii_class, counts in sorted(class_counts.items())
        }
        for surface, class_counts in buckets.items()
    }


def _person_name_summary(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for score in scores:
        if _clean(score.get("class")) != "person_name":
            continue
        subtype = _clean(score.get("name_subtype")) or "unspecified"
        if score.get("reaches_surface") is True:
            buckets[subtype]["expected"] += 1
        if score.get("leaked") is True:
            buckets[subtype]["leaks"] += 1
    return {
        subtype: _class_summary(counts)
        for subtype, counts in sorted(buckets.items())
    }


def _headline_summary(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    leaked_ids = _free_high_severity_leaked_ids(scores)
    deferred_open_set_name_ids = {
        _score_surrogate_id(score)
        for score in scores
        if _score_surrogate_id(score) in leaked_ids
        and _clean(score.get("class")) == "person_name"
        and _clean(score.get("name_subtype")) == "cue_less"
    }
    gate_eligible_ids = leaked_ids - deferred_open_set_name_ids
    return {
        "free_high_severity_leak_count": len(leaked_ids),
        "free_high_severity_pass": len(leaked_ids) == 0,
        "free_high_severity_gate_eligible_leak_count": len(gate_eligible_ids),
        "free_high_severity_gate_eligible_pass": len(gate_eligible_ids) == 0,
        "deferred_open_set_name_leak_count": len(deferred_open_set_name_ids),
    }


def _free_high_severity_leaked_ids(scores: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        leaked_id
        for score in scores
        if score.get("leaked") is True
        and _clean(score.get("surface")) in FREE_SURFACES
        and _score_is_high_severity(score)
        for leaked_id in (_score_surrogate_id(score),)
        if leaked_id
    }


def _score_is_high_severity(score: Mapping[str, Any]) -> bool:
    return (
        _clean(score.get("severity")) == "high"
        or _clean(score.get("class")) in HIGH_SEVERITY_CLASSES
    )


def _score_surrogate_id(score: Mapping[str, Any]) -> str:
    return _clean(score.get("surrogate_id"))


def _class_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    expected = int(counts.get("expected", 0))
    leaks = int(counts.get("leaks", 0))
    redacted = max(expected - leaks, 0)
    return {
        "expected": expected,
        "redacted": redacted,
        "leaks": leaks,
        "recall": _ratio(redacted, expected),
    }


def _labels(tickets: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    labels: list[Mapping[str, Any]] = []
    for index, ticket in enumerate(tickets, start=1):
        ticket_key = _ticket_key(ticket, index)
        for label in _sequence(ticket.get("labels")):
            if not isinstance(label, Mapping):
                continue
            labels.append({**label, "_ticket_key": ticket_key})
    return labels


def _must_survive(tickets: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        record
        for ticket in tickets
        for record in _sequence(ticket.get("must_survive"))
        if isinstance(record, Mapping)
    ]


def _ticket_key(ticket: Mapping[str, Any], index: int) -> str:
    return _clean(ticket.get("ticket_id")) or f"ticket-{index:03d}"


def _surface_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _error_summary(errors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "failed",
        "blocking_error_codes": [
            _clean(error.get("code"))
            for error in errors
            if _clean(error.get("code"))
        ],
        "errors": [dict(error) for error in errors],
    }


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _first_text(*values: Any) -> str:
    for value in values:
        text = _clean(value)
        if text:
            return text
    return ""


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _error(code: str, **details: Any) -> dict[str, Any]:
    return {
        "code": code,
        **{
            key: value
            for key, value in details.items()
            if isinstance(value, (int, str)) and value not in ("", 0)
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
