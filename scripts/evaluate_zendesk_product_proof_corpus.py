#!/usr/bin/env python3
"""Evaluate the committed Zendesk product-proof corpus through deflection."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    build_deflection_report_artifact,
)
from extracted_content_pipeline.support_ticket_input_package import (  # noqa: E402
    build_support_ticket_input_package,
)
from extracted_content_pipeline.support_ticket_zendesk_thread import (  # noqa: E402
    rows_from_zendesk_full_thread,
)
from extracted_content_pipeline.support_ticket_clustering import (  # noqa: E402
    support_ticket_plain_text,
)
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
    DEFAULT_TITLE,
    build_ticket_faq_markdown,
)


DEFAULT_CORPUS = (
    ROOT / "docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "docs/extraction/validation/fixtures/"
    / "deflection_zendesk_product_proof_eval_20260614"
)
DEFAULT_DOC = (
    ROOT
    / "docs/extraction/validation/"
    / "deflection_zendesk_product_proof_eval_2026-06-14.md"
)
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_PRIVATE_NOTE_LEAK_WINDOW_TOKENS = 6


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = _load_corpus(args.corpus)
    artifact, import_warnings, package_warnings = _build_artifact(corpus)
    labels = _expected_labels(corpus)
    private_texts = _private_comment_texts(corpus)
    summary = evaluate_items(
        labels=labels,
        items=artifact.faq_result.items,
        markdown=artifact.markdown,
        private_texts=private_texts,
        import_warning_count=len(import_warnings),
        package_warning_count=len(package_warnings),
        faq_warnings=artifact.faq_result.warnings,
        output_checks=artifact.faq_result.output_checks,
    )
    summary["corpus"] = _corpus_summary(corpus)
    summary["artifact_summary"] = dict(artifact.summary)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "report_excerpt.md").write_text(
            _report_excerpt(artifact.markdown),
            encoding="utf-8",
        )
    if args.doc:
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(_proof_doc(summary), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "ok":
        print(
            "Zendesk product-proof evaluation failed: "
            + ", ".join(summary["blocking_violation_codes"]),
            file=sys.stderr,
        )
        return 1
    return 0


def evaluate_items(
    *,
    labels: Mapping[str, Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    markdown: str,
    private_texts: Mapping[str, Sequence[str]],
    import_warning_count: int = 0,
    package_warning_count: int = 0,
    faq_warnings: Sequence[Mapping[str, Any]] = (),
    output_checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    publishable_items = [
        item for item in items if item.get("answer_evidence_status") == "resolution_evidence"
    ]
    publishable_source_ids = _source_ids(publishable_items)
    expected_publishable_ids = {
        ticket_id
        for ticket_id, expected in labels.items()
        if expected.get("should_publish_answer") is True
    }
    false_positive_ids = sorted(
        ticket_id
        for ticket_id in publishable_source_ids
        if labels.get(ticket_id, {}).get("should_publish_answer") is not True
    )
    unresolved_ids = sorted(
        ticket_id
        for ticket_id in publishable_source_ids
        if labels.get(ticket_id, {}).get("unresolved") is True
    )
    reopened_ids = sorted(
        ticket_id
        for ticket_id in publishable_source_ids
        if labels.get(ticket_id, {}).get("reopened") is True
    )
    private_leaks = _private_note_leaks(markdown, private_texts)
    bad_questions = _degraded_questions(items)
    draft_bad_questions = _degraded_draft_questions(items)
    failed_output_checks = _failed_output_checks(output_checks or {})
    covered_expected_ids = sorted(expected_publishable_ids & publishable_source_ids)
    blocking_codes: list[str] = []
    if false_positive_ids:
        blocking_codes.append("publishable_false_positive")
    if unresolved_ids:
        blocking_codes.append("unresolved_source_published")
    if reopened_ids:
        blocking_codes.append("reopened_source_published")
    if private_leaks:
        blocking_codes.append("private_note_leak")
    if bad_questions:
        blocking_codes.append("degraded_question_label")
    if failed_output_checks:
        blocking_codes.append("failed_output_checks")

    return {
        "status": "ok" if not blocking_codes else "failed",
        "blocking_violation_codes": blocking_codes,
        "generated_item_count": len(items),
        "publishable_answer_item_count": len(publishable_items),
        "expected_publishable_source_count": len(expected_publishable_ids),
        "covered_publishable_source_count": len(covered_expected_ids),
        "publishable_source_coverage": _ratio(
            len(covered_expected_ids), len(expected_publishable_ids)
        ),
        "publishable_false_positive_source_ids": false_positive_ids,
        "unresolved_publishable_source_ids": unresolved_ids,
        "reopened_publishable_source_ids": reopened_ids,
        "private_note_leaks": private_leaks,
        "degraded_question_labels": bad_questions,
        "degraded_draft_question_labels": draft_bad_questions,
        "output_checks": dict(output_checks or {}),
        "failed_output_checks": failed_output_checks,
        "import_warning_count": import_warning_count,
        "package_warning_count": package_warning_count,
        "faq_warning_count": len(faq_warnings),
        "faq_warning_codes": [
            _text(warning.get("code"))
            for warning in faq_warnings
            if _text(warning.get("code"))
        ],
        "faq_warnings": [dict(warning) for warning in faq_warnings],
        "top_questions": [
            {
                "rank": index,
                "question": _text(item.get("question")),
                "answer_evidence_status": _text(item.get("answer_evidence_status")),
                "ticket_count": _int(item.get("ticket_count")),
                "source_ids": sorted(_item_source_ids(item)),
            }
            for index, item in enumerate(items, start=1)
        ],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _load_corpus(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not load Zendesk product-proof corpus: {path}") from exc
    if not isinstance(value, Mapping):
        raise SystemExit("Zendesk product-proof corpus must be a JSON object")
    return value


def _build_artifact(corpus: Mapping[str, Any]) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    imported = rows_from_zendesk_full_thread(corpus)
    package = build_support_ticket_input_package(
        imported.rows,
        provider="zendesk_product_proof_corpus",
        outputs=("faq_deflection_report",),
    )
    faq_result = build_ticket_faq_markdown(
        package.inputs["source_material"],
        title=DEFAULT_TITLE,
        max_items=0,
        max_evidence_per_item=3,
    )
    return (
        build_deflection_report_artifact(
            faq_result,
            source_label="zendesk_product_proof_corpus",
        ),
        tuple(imported.warnings),
        tuple(package.warnings),
    )


def _expected_labels(corpus: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    labels: dict[str, Mapping[str, Any]] = {}
    for ticket in _tickets(corpus):
        ticket_id = _text(ticket.get("id"))
        expected = ticket.get("expected")
        if ticket_id and isinstance(expected, Mapping):
            labels[ticket_id] = expected
    return labels


def _private_comment_texts(corpus: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    private: dict[str, tuple[str, ...]] = {}
    for ticket in _tickets(corpus):
        ticket_id = _text(ticket.get("id"))
        values = []
        comments = ticket.get("comments")
        if not isinstance(comments, Sequence) or isinstance(comments, (str, bytes)):
            continue
        for comment in comments:
            if not isinstance(comment, Mapping) or comment.get("public") is not False:
                continue
            text = _normalize_for_match(comment.get("body"))
            if len(text) >= 16:
                values.append(text)
        if ticket_id and values:
            private[ticket_id] = tuple(values)
    return private


def _private_note_leaks(
    markdown: str,
    private_texts: Mapping[str, Sequence[str]],
) -> list[dict[str, str]]:
    normalized_markdown = _normalize_for_match(markdown)
    markdown_windows = _token_windows(
        normalized_markdown,
        _PRIVATE_NOTE_LEAK_WINDOW_TOKENS,
    )
    leaks: list[dict[str, str]] = []
    for ticket_id, texts in private_texts.items():
        for text in texts:
            if text and _private_text_matches_markdown(
                text,
                normalized_markdown=normalized_markdown,
                markdown_windows=markdown_windows,
            ):
                leaks.append({"source_id": ticket_id, "text": text[:120]})
    return leaks


def _private_text_matches_markdown(
    text: str,
    *,
    normalized_markdown: str,
    markdown_windows: set[str],
) -> bool:
    if text in normalized_markdown:
        return True
    tokens = _tokens_for_window_match(text)
    if len(tokens) < _PRIVATE_NOTE_LEAK_WINDOW_TOKENS:
        return False
    for index in range(len(tokens) - _PRIVATE_NOTE_LEAK_WINDOW_TOKENS + 1):
        window = " ".join(tokens[index : index + _PRIVATE_NOTE_LEAK_WINDOW_TOKENS])
        if window in markdown_windows:
            return True
    return False


def _tokens_for_window_match(value: str) -> list[str]:
    return _WORD_RE.findall(_normalize_for_match(value))


def _token_windows(value: str, size: int) -> set[str]:
    tokens = _tokens_for_window_match(value)
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _degraded_questions(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        question
        for item in items
        if item.get("answer_evidence_status") == "resolution_evidence"
        for question in [_text(item.get("question"))]
        if _is_degraded_question_label(question)
    ]


def _degraded_draft_questions(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        question
        for item in items
        if item.get("answer_evidence_status") != "resolution_evidence"
        for question in [_text(item.get("question"))]
        if _is_degraded_question_label(question)
    ]


def _is_degraded_question_label(question: str) -> bool:
    lowered = question.lower()
    if not lowered:
        return False
    if (
        "[atlas seed" in lowered
        or "what should i do about atla?" in lowered
        or lowered.startswith("localized support question ")
    ):
        return True
    if lowered in {"what should i do about reset?", "what should i do about duplicate?"}:
        return True
    return False


def _failed_output_checks(output_checks: Mapping[str, Any]) -> list[str]:
    return [
        name
        for name, passed in sorted(output_checks.items())
        if passed is not True
    ]


def _source_ids(items: Sequence[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for item in items:
        values.update(_item_source_ids(item))
    return values


def _item_source_ids(item: Mapping[str, Any]) -> set[str]:
    raw = item.get("source_ids")
    if isinstance(raw, str):
        return {_text(raw)} if _text(raw) else set()
    if isinstance(raw, Sequence):
        return {_text(value) for value in raw if _text(value)}
    return set()


def _tickets(corpus: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    tickets = corpus.get("tickets")
    if not isinstance(tickets, Sequence) or isinstance(tickets, (str, bytes)):
        return ()
    return tuple(ticket for ticket in tickets if isinstance(ticket, Mapping))


def _corpus_summary(corpus: Mapping[str, Any]) -> dict[str, Any]:
    labels = _expected_labels(corpus)
    return {
        "ticket_count": len(labels),
        "should_publish_answer_true": sum(
            1 for expected in labels.values() if expected.get("should_publish_answer") is True
        ),
        "unresolved_true": sum(
            1 for expected in labels.values() if expected.get("unresolved") is True
        ),
        "reopened_true": sum(
            1 for expected in labels.values() if expected.get("reopened") is True
        ),
        "has_private_note_true": sum(
            1 for expected in labels.values() if expected.get("has_private_note") is True
        ),
    }


def _report_excerpt(markdown: str, *, max_lines: int = 120) -> str:
    lines = markdown.splitlines()
    excerpt = "\n".join(lines[:max_lines]).rstrip()
    return excerpt + "\n"


def _proof_doc(summary: Mapping[str, Any]) -> str:
    corpus = summary["corpus"]
    warning_codes = ", ".join(summary["faq_warning_codes"]) or "none"
    return f"""# Deflection Zendesk Product-Proof Eval

Date: 2026-06-14

Issues: #1419, #1440

## What Ran

This validation feeds the committed sanitized Zendesk product-proof corpus
through the real full-thread importer, support-ticket input package, FAQ
deflection report builder, and product-proof evaluator after the question-label
cleanup in #1568.

## Result

| Metric | Value |
|---|---:|
| Tickets evaluated | {corpus["ticket_count"]} |
| Expected publishable-answer tickets | {summary["expected_publishable_source_count"]} |
| Covered publishable-answer tickets | {summary["covered_publishable_source_count"]} |
| Generated ranked questions | {summary["generated_item_count"]} |
| Publishable-answer items | {summary["publishable_answer_item_count"]} |
| Publishable false-positive sources | {len(summary["publishable_false_positive_source_ids"])} |
| Unresolved sources published | {len(summary["unresolved_publishable_source_ids"])} |
| Reopened sources published | {len(summary["reopened_publishable_source_ids"])} |
| Private note leaks | {len(summary["private_note_leaks"])} |
| Degraded published question labels | {len(summary["degraded_question_labels"])} |
| Degraded draft question labels recorded | {len(summary["degraded_draft_question_labels"])} |
| Failed artifact output checks | {len(summary["failed_output_checks"])} |
| FAQ warnings | {summary["faq_warning_count"]} |

Status: `{summary["status"]}`

The run is safety-clean for published answers. Remaining draft-only weak labels
are recorded in the summary JSON, not hidden, so follow-up slices can decide
whether they are launch-blocking polish or expected diagnostics.

FAQ warning codes: `{warning_codes}`

## Artifact Links

- Summary JSON:
  `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/summary.json`
- Report excerpt:
  `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/report_excerpt.md`

## Boundary

This is an offline product-shaped validation run. It does not call live
Zendesk, mutate Stripe state, unlock a paid report, send email, or exercise the
hosted portfolio result page. It pairs with the CFPB full-volume proof: CFPB
proves stress and this corpus proves Zendesk ticket-shape quality.
"""


def _normalize_for_match(value: Any) -> str:
    text = support_ticket_plain_text(value)
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
