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


@dataclass(frozen=True)
class CsvBreakageCase:
    """One synthetic parser-admission edge case to score."""

    name: str
    description: str
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    expected_outcome: str
    expected_decision_reason: str | None = None
    expected_decision_location: str | None = None
    known_gap: bool = False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = _load_corpus(args.corpus)
    observed_csvs = _parse_observed_csvs(args.observed_csv)
    summary = evaluate_zendesk_corpus(corpus, observed_csvs=observed_csvs)

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


def evaluate_zendesk_corpus(
    corpus: Mapping[str, Any],
    *,
    observed_csvs: Sequence[tuple[str, Path]] = (),
) -> dict[str, Any]:
    """Inspect product-shaped Zendesk CSV projections and summarize evidence."""

    cases = _zendesk_cases(corpus)
    case_results = [evaluate_csv_case(case) for case in cases]
    case_results.extend(
        evaluate_observed_csv(name=name, path=path)
        for name, path in observed_csvs
    )
    breakage_results = evaluate_breakage_matrix()
    blocking = _blocking_codes(case_results) + _breakage_blocking_codes(
        breakage_results
    )
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
        "strict_case_count": sum(
            1 for result in case_results if result.get("expected_full_coverage") is True
        ),
        "observed_case_count": sum(
            1 for result in case_results if result.get("case_status") == "observed"
        ),
        "blocking_case_count": sum(
            1 for result in case_results if result.get("case_status") == "failed"
        ),
        "observed_min_usable_source_ratio": min(ratios) if ratios else None,
        "observed_max_usable_source_ratio": max(ratios) if ratios else None,
        "threshold_conclusion": (
            "The committed product-shaped Zendesk CSV projections are accepted "
            "at full coverage. This supports the clean ACCEPT path, but does "
            "not justify a hard low-coverage reject threshold by itself. "
            "Operator-supplied observed CSVs are evidence only until a later "
            "policy slice promotes a threshold."
        ),
        "breakage_matrix": {
            "case_count": len(breakage_results),
            "blocking_case_count": sum(
                1 for result in breakage_results if result.get("case_status") == "failed"
            ),
            "known_gap_count": sum(
                1 for result in breakage_results if result.get("case_status") == "known_gap"
            ),
            "cases": breakage_results,
            "conclusion": (
                "Synthetic breakage cases prove parser mechanics only. "
                "Fail-closed and warning expectations are blocking; known "
                "fail-open cases are recorded as explicit gaps and do not set "
                "low-coverage reject policy."
            ),
        },
        "cases": case_results,
    }


def evaluate_csv_case(case: CsvAdmissionCase) -> dict[str, Any]:
    """Write a projection to a temporary CSV and inspect the real diagnostics path."""

    with TemporaryDirectory(prefix="atlas-csv-admission-") as tmp_dir:
        path = Path(tmp_dir) / f"{case.name}.csv"
        _write_csv(path, fieldnames=case.fieldnames, rows=case.rows)
        return _evaluate_csv_path(
            name=case.name,
            description=case.description,
            path=path,
            expected_full_coverage=case.expected_full_coverage,
        )


def evaluate_observed_csv(*, name: str, path: Path) -> dict[str, Any]:
    """Inspect an operator-supplied CSV as non-gating threshold evidence."""

    return _evaluate_csv_path(
        name=name,
        description=f"Observed source-row CSV evidence from {path.name}.",
        path=path,
        expected_full_coverage=False,
    )


def evaluate_breakage_matrix() -> list[dict[str, Any]]:
    """Run synthetic parser breakage cases through the real CSV diagnostics."""

    return [evaluate_breakage_case(case) for case in _breakage_cases()]


def evaluate_breakage_case(case: CsvBreakageCase) -> dict[str, Any]:
    """Write one breakage case to CSV and score the observed admission outcome."""

    with TemporaryDirectory(prefix="atlas-csv-breakage-") as tmp_dir:
        path = Path(tmp_dir) / f"{case.name}.csv"
        _write_csv(path, fieldnames=case.fieldnames, rows=case.rows)
        result = _evaluate_csv_path(
            name=case.name,
            description=case.description,
            path=path,
            expected_full_coverage=False,
        )
    observed = _observed_admission_outcome(result)
    decision_matched = _breakage_decision_matches(case, result)
    matched = observed == case.expected_outcome and decision_matched
    status = "ok" if matched else "failed"
    if case.known_gap and matched:
        status = "known_gap"
    return {
        **result,
        "expected_outcome": case.expected_outcome,
        "observed_outcome": observed,
        "expected_decision_reason": case.expected_decision_reason,
        "expected_decision_location": case.expected_decision_location,
        "decision_matched": decision_matched,
        "known_gap": case.known_gap,
        "case_status": status,
    }


def _evaluate_csv_path(
    *,
    name: str,
    description: str,
    path: Path,
    expected_full_coverage: bool,
) -> dict[str, Any]:
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
        "name": name,
        "description": description,
        "expected_full_coverage": expected_full_coverage,
        "ok": payload.get("ok") is True,
        "warning_counts": dict(payload.get("warning_counts") or {}),
        "admission_status": decision.get("status"),
        "admission_decision": decision,
        "admission_decision_reason": decision.get("reason"),
        "admission_decision_location": decision.get("location"),
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
        if result.get("case_status") == "failed":
            codes.append(f"{result.get('name')}:unexpected_admission_result")
    return codes


def _breakage_blocking_codes(case_results: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"{result.get('name')}:unexpected_breakage_outcome"
        for result in case_results
        if result.get("case_status") == "failed"
    ]


def _observed_admission_outcome(result: Mapping[str, Any]) -> str:
    if result.get("admission_status") == "REJECT":
        return "REJECT"
    if result.get("coverage_warnings"):
        return "ACCEPT_WITH_WARNING"
    if result.get("admission_status") == "ACCEPT":
        return "ACCEPT_CLEAN"
    return "NO_POLICY_DECISION"


def _breakage_decision_matches(
    case: CsvBreakageCase,
    result: Mapping[str, Any],
) -> bool:
    if case.expected_decision_reason is None and case.expected_decision_location is None:
        return True
    return (
        result.get("admission_decision_reason") == case.expected_decision_reason
        and result.get("admission_decision_location") == case.expected_decision_location
    )


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


def _breakage_cases() -> tuple[CsvBreakageCase, ...]:
    return (
        CsvBreakageCase(
            name="unknown_body_like_column_rejects_zero_usable",
            description="Populated body-like column is not mapped and must reject.",
            fieldnames=("Ticket ID", "Conversation Text"),
            rows=({"Ticket ID": "T-1", "Conversation Text": "Cannot export reports."},),
            expected_outcome="REJECT",
            expected_decision_reason="no_usable_source_rows",
            expected_decision_location="source_row_csv",
        ),
        CsvBreakageCase(
            name="private_note_only_rejects_zero_usable",
            description="Private/internal-only rows must not count as usable text.",
            fieldnames=("Ticket ID", "Internal Notes"),
            rows=({"Ticket ID": "T-1", "Internal Notes": "Private workaround."},),
            expected_outcome="REJECT",
            expected_decision_reason="no_usable_source_rows",
            expected_decision_location="source_row_csv",
        ),
        CsvBreakageCase(
            name="status_timestamp_only_rejects_zero_usable",
            description="Status/SLA timestamp columns alone are not customer wording.",
            fieldnames=("Ticket ID", "Status", "First Response", "Last Response"),
            rows=({
                "Ticket ID": "T-1",
                "Status": "open",
                "First Response": "2026-01-01",
                "Last Response": "2026-01-02",
            },),
            expected_outcome="REJECT",
            expected_decision_reason="no_usable_source_rows",
            expected_decision_location="source_row_csv",
        ),
        CsvBreakageCase(
            name="partial_blank_rows_warns_without_rejecting",
            description="Some usable rows plus blanks should accept with coverage warning.",
            fieldnames=("Ticket ID", "Message"),
            rows=(
                {"Ticket ID": "T-1", "Message": "Customer cannot export reports."},
                {"Ticket ID": "T-2", "Message": ""},
            ),
            expected_outcome="ACCEPT_WITH_WARNING",
        ),
        CsvBreakageCase(
            name="header_only_csv_has_no_policy_decision",
            description="Header-only CSV has no hard source-row admission decision.",
            fieldnames=("Ticket ID", "Message"),
            rows=(),
            expected_outcome="NO_POLICY_DECISION",
        ),
        CsvBreakageCase(
            name="json_blob_message_rejects_zero_usable",
            description="Machine JSON in a mapped text field must not count as usable text.",
            fieldnames=("Ticket ID", "Message"),
            rows=({"Ticket ID": "T-1", "Message": '{"event":"ticket_created","id":123}'},),
            expected_outcome="REJECT",
            expected_decision_reason="no_usable_source_rows",
            expected_decision_location="source_row_csv",
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
        "| {name} | {case_status} | {raw} | {usable} | {ratio} | {status} | {warnings} |".format(
            name=case["name"],
            case_status=case["case_status"],
            raw=case["raw_source_row_count"],
            usable=case["usable_source_row_count"],
            ratio=case["usable_source_ratio"],
            status=case["admission_status"],
            warnings=len(case["coverage_warnings"]),
        )
        for case in summary["cases"]
    )
    breakage_rows = "\n".join(
        "| {name} | {case_status} | {expected} | {observed} | {raw} | {usable} | {status} | {reason} | {location} | {warnings} |".format(
            name=case["name"],
            case_status=case["case_status"],
            expected=case["expected_outcome"],
            observed=case["observed_outcome"],
            raw=case["raw_source_row_count"],
            usable=case["usable_source_row_count"],
            status=case["admission_status"],
            reason=case["admission_decision_reason"] or "",
            location=case["admission_decision_location"] or "",
            warnings=len(case["coverage_warnings"]),
        )
        for case in summary["breakage_matrix"]["cases"]
    )
    return f"""# Deflection CSV Admission Threshold Evidence

Date: 2026-06-15

Issue: #1467

## What Ran

This validation projects the committed sanitized Zendesk product-proof corpus
into source-row CSV shapes and inspects each CSV through the real ingestion
diagnostics path with source-row mode enabled.

## Result

| Case | Case status | Raw rows | Usable rows | Usable ratio | Admission | Coverage warnings |
|---|---|---:|---:|---:|---|---:|
{rows}

Status: `{summary["status"]}`

Observed minimum usable source ratio: `{summary["observed_min_usable_source_ratio"]}`

Observed evidence cases: `{summary["observed_case_count"]}`

## Interpretation

{summary["threshold_conclusion"]}

This artifact therefore supports keeping clean Zendesk-shaped CSV uploads on
the ACCEPT path. It does not choose a low non-zero reject threshold. That
threshold still needs observed partial-coverage exports, not a synthetic ratio
derived from a clean corpus. Operator-supplied observed CSVs are recorded as
evidence and do not block this runner until a later policy slice promotes a
threshold.

## Parser Breakage Matrix

These synthetic cases break parser mechanics through the same CSV diagnostics
path. They prove whether current guards fail closed, warn, or expose a known
fail-open gap. They do not justify a low-coverage threshold.

| Case | Case status | Expected outcome | Observed outcome | Raw rows | Usable rows | Admission | Decision reason | Decision location | Coverage warnings |
|---|---|---|---|---:|---:|---|---|---|---:|
{breakage_rows}

Known fail-open gaps: `{summary["breakage_matrix"]["known_gap_count"]}`

{summary["breakage_matrix"]["conclusion"]}

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
    parser.add_argument(
        "--observed-csv",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional observed source-row CSV evidence to summarize without "
            "failing the run. May be supplied more than once."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _parse_observed_csvs(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    observed: list[tuple[str, Path]] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit("--observed-csv values must use NAME=PATH")
        name, path_text = raw.split("=", 1)
        name = name.strip()
        path = Path(path_text.strip())
        if not name:
            raise SystemExit("--observed-csv name must be non-empty")
        if not path.exists():
            raise SystemExit(f"--observed-csv path does not exist: {path}")
        observed.append((name, path))
    return tuple(observed)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
