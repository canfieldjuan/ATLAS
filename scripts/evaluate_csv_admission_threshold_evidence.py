#!/usr/bin/env python3
"""Generate CSV source-row admission evidence from product-shaped fixtures."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.ingestion_diagnostics import (  # noqa: E402
    inspect_ingestion_file,
)


DEFAULT_CORPUS = (
    ROOT / "docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "docs/extraction/validation/fixtures/"
    / "deflection_csv_admission_threshold_evidence_20260615"
)
DEFAULT_DOC = (
    ROOT
    / "docs/extraction/validation/"
    / "deflection_csv_admission_threshold_evidence_2026-06-15.md"
)


@dataclass(frozen=True)
class CsvAdmissionCase:
    """One CSV projection to inspect through source-row admission diagnostics."""

    name: str
    description: str
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    expected_full_coverage: bool


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = _load_corpus(args.corpus)
    summary = evaluate_zendesk_corpus(corpus)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.doc:
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(_proof_doc(summary), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "ok":
        print(
            "CSV admission threshold evidence failed: "
            + ", ".join(summary["blocking_violation_codes"]),
            file=sys.stderr,
        )
        return 1
    return 0


def evaluate_zendesk_corpus(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect product-shaped Zendesk CSV projections and summarize evidence."""

    cases = _zendesk_cases(corpus)
    case_results = [evaluate_csv_case(case) for case in cases]
    blocking = _blocking_codes(case_results)
    ratios = [
        float(result["usable_source_ratio"])
        for result in case_results
        if isinstance(result.get("usable_source_ratio"), (int, float))
    ]
    return {
        "status": "ok" if not blocking else "failed",
        "blocking_violation_codes": blocking,
        "corpus": _corpus_summary(corpus),
        "case_count": len(case_results),
        "observed_min_usable_source_ratio": min(ratios) if ratios else None,
        "observed_max_usable_source_ratio": max(ratios) if ratios else None,
        "threshold_conclusion": (
            "The committed product-shaped Zendesk CSV projections are accepted "
            "at full coverage. This supports the clean ACCEPT path, but does "
            "not justify a hard low-coverage reject threshold by itself."
        ),
        "cases": case_results,
    }


def evaluate_csv_case(case: CsvAdmissionCase) -> dict[str, Any]:
    """Write a projection to a temporary CSV and inspect the real diagnostics path."""

    with TemporaryDirectory(prefix="atlas-csv-admission-") as tmp_dir:
        path = Path(tmp_dir) / f"{case.name}.csv"
        _write_csv(path, fieldnames=case.fieldnames, rows=case.rows)
        payload = inspect_ingestion_file(
            path,
            source_rows=True,
            source_format="csv",
            sample_limit=0,
        ).as_dict()

    admission = dict(payload.get("source_row_admission") or {})
    decision = dict(admission.get("admission_decision") or {})
    warnings = list(admission.get("coverage_warnings") or [])
    usable_ratio = admission.get("usable_source_ratio")
    result = {
        "name": case.name,
        "description": case.description,
        "expected_full_coverage": case.expected_full_coverage,
        "ok": payload.get("ok") is True,
        "warning_counts": dict(payload.get("warning_counts") or {}),
        "admission_status": decision.get("status"),
        "raw_source_row_count": _int(admission.get("raw_source_row_count")),
        "usable_source_row_count": _int(admission.get("usable_source_row_count")),
        "usable_source_ratio": usable_ratio,
        "mapped_fields": dict(admission.get("mapped_fields") or {}),
        "ignored_private_fields": list(admission.get("ignored_private_fields") or []),
        "populated_unmapped_fields": list(
            admission.get("populated_unmapped_fields") or []
        ),
        "coverage_warnings": warnings,
    }
    result["case_status"] = _case_status(result)
    return result


def _case_status(result: Mapping[str, Any]) -> str:
    if result.get("expected_full_coverage") is not True:
        return "observed"
    if result.get("admission_status") != "ACCEPT":
        return "failed"
    if result.get("raw_source_row_count") != result.get("usable_source_row_count"):
        return "failed"
    if result.get("coverage_warnings"):
        return "failed"
    return "ok"


def _blocking_codes(case_results: Sequence[Mapping[str, Any]]) -> list[str]:
    codes: list[str] = []
    for result in case_results:
        if result.get("case_status") != "ok":
            codes.append(f"{result.get('name')}:unexpected_admission_result")
    return codes


def _zendesk_cases(corpus: Mapping[str, Any]) -> tuple[CsvAdmissionCase, ...]:
    tickets = _tickets(corpus)
    public_rows = tuple(
        {
            "Ticket ID": _text(ticket.get("id")),
            "Subject": _text(ticket.get("subject")),
            "Public Comments": _public_comments(ticket),
            "Internal Notes": _private_comments(ticket),
        }
        for ticket in tickets
    )
    description_rows = tuple(
        {
            "Ticket ID": _text(ticket.get("id")),
            "Subject": _text(ticket.get("subject")),
            "Description": _text(ticket.get("description")),
            "Internal Notes": _private_comments(ticket),
        }
        for ticket in tickets
    )
    return (
        CsvAdmissionCase(
            name="zendesk_public_comments_csv",
            description=(
                "Zendesk-shaped CSV with ticket id, subject, public comments, "
                "and internal notes."
            ),
            fieldnames=("Ticket ID", "Subject", "Public Comments", "Internal Notes"),
            rows=public_rows,
            expected_full_coverage=True,
        ),
        CsvAdmissionCase(
            name="zendesk_description_csv",
            description=(
                "Zendesk-shaped CSV with ticket id, subject, requester "
                "description, and internal notes."
            ),
            fieldnames=("Ticket ID", "Subject", "Description", "Internal Notes"),
            rows=description_rows,
            expected_full_coverage=True,
        ),
    )


def _load_corpus(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not load Zendesk corpus: {path}") from exc
    if not isinstance(value, Mapping):
        raise SystemExit("Zendesk corpus must be a JSON object")
    return value


def _tickets(corpus: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    tickets = corpus.get("tickets")
    if not isinstance(tickets, Sequence) or isinstance(tickets, (str, bytes)):
        return ()
    return tuple(ticket for ticket in tickets if isinstance(ticket, Mapping))


def _comments(ticket: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    comments = ticket.get("comments")
    if not isinstance(comments, Sequence) or isinstance(comments, (str, bytes)):
        return ()
    return tuple(comment for comment in comments if isinstance(comment, Mapping))


def _public_comments(ticket: Mapping[str, Any]) -> str:
    return "\n".join(
        _text(comment.get("body"))
        for comment in _comments(ticket)
        if comment.get("public") is not False and _text(comment.get("body"))
    )


def _private_comments(ticket: Mapping[str, Any]) -> str:
    return "\n".join(
        _text(comment.get("body"))
        for comment in _comments(ticket)
        if comment.get("public") is False and _text(comment.get("body"))
    )


def _corpus_summary(corpus: Mapping[str, Any]) -> dict[str, Any]:
    tickets = _tickets(corpus)
    private_count = sum(1 for ticket in tickets if _private_comments(ticket))
    return {
        "source": _text(corpus.get("source")),
        "run_tag": _text(corpus.get("run_tag")),
        "ticket_count": len(tickets),
        "tickets_with_private_notes": private_count,
    }


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _text(row.get(field)) for field in fieldnames})


def _proof_doc(summary: Mapping[str, Any]) -> str:
    rows = "\n".join(
        "| {name} | {raw} | {usable} | {ratio} | {status} | {warnings} |".format(
            name=case["name"],
            raw=case["raw_source_row_count"],
            usable=case["usable_source_row_count"],
            ratio=case["usable_source_ratio"],
            status=case["admission_status"],
            warnings=len(case["coverage_warnings"]),
        )
        for case in summary["cases"]
    )
    return f"""# Deflection CSV Admission Threshold Evidence

Date: 2026-06-15

Issue: #1467

## What Ran

This validation projects the committed sanitized Zendesk product-proof corpus
into source-row CSV shapes and inspects each CSV through the real ingestion
diagnostics path with source-row mode enabled.

## Result

| Case | Raw rows | Usable rows | Usable ratio | Admission | Coverage warnings |
|---|---:|---:|---:|---|---:|
{rows}

Status: `{summary["status"]}`

Observed minimum usable source ratio: `{summary["observed_min_usable_source_ratio"]}`

## Interpretation

{summary["threshold_conclusion"]}

This artifact therefore supports keeping clean Zendesk-shaped CSV uploads on
the ACCEPT path. It does not choose a low non-zero reject threshold. That
threshold still needs observed partial-coverage exports, not a synthetic ratio
derived from a clean corpus.

## Artifact Links

- Summary JSON:
  `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json`

## Boundary

This is offline validation. It does not call live Zendesk, upload a customer
file, mutate data, charge Stripe, or change parser policy.
"""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
