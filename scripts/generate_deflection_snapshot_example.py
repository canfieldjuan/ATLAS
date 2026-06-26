#!/usr/bin/env python3
"""Generate or check the frontend deflection snapshot example JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.faq_deflection_report import (
    build_deflection_report_artifact,
    build_deflection_snapshot,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


DEFAULT_REPORT_OUTPUT = ROOT / "docs/frontend/content_ops_faq_deflection_report_example.json"
DEFAULT_SNAPSHOT_OUTPUT = (
    ROOT / "docs/frontend/content_ops_faq_deflection_snapshot_example.json"
)
DEFAULT_OUTPUT = DEFAULT_SNAPSHOT_OUTPUT
SNAPSHOT_TOP_N = 5


_SYNTHETIC_REPEAT_COHORTS: tuple[dict[str, Any], ...] = (
    {
        "key": "export",
        "ticket_count": 100,
        "source_title": "Export attribution",
        "support_ticket_cluster": "reporting friction",
        "product_area": "Reporting",
        "tags": "report-export, analytics",
        "resolution_text": (
            "Open Analytics, choose Attribution, then click Download report."
        ),
        "texts": (
            "How do I export attribution reports?",
        ),
        "statuses": ("solved", "solved", "closed", "closed"),
        "csat_scores": (5, 4, 4, 5),
    },
    {
        "key": "sso",
        "ticket_count": 85,
        "source_title": "SSO setup",
        "support_ticket_cluster": "auth setup",
        "product_area": "Auth",
        "tags": "sso, admin-setup",
        "texts": (
            "How do I enable SSO for my team?",
        ),
        "statuses": ("open", "reopened", "pending", "reopened"),
        "csat_scores": (2, 1, 3, 2),
    },
    {
        "key": "billing-pause",
        "ticket_count": 70,
        "source_title": "Subscription pause",
        "support_ticket_cluster": "billing pause requests",
        "product_area": "Billing",
        "tags": "billing, subscription",
        "texts": (
            "How do I pause my subscription for one month?",
        ),
        "statuses": ("open", "pending", "reopened", "open"),
        "csat_scores": (3, 2, 1, 2),
    },
    {
        "key": "invoice-admin",
        "ticket_count": 55,
        "source_title": "Invoice recipients",
        "support_ticket_cluster": "billing invoice administration",
        "product_area": "Billing",
        "tags": "billing, invoice-settings",
        "resolution_text": (
            "Open Billing, choose Invoice settings, then add the finance "
            "recipient under Invoice contacts."
        ),
        "texts": (
            "How do I add an invoice recipient?",
        ),
        "statuses": ("solved", "closed", "solved", "closed"),
        "csat_scores": (5, 4, 5, 4),
    },
    {
        "key": "roles",
        "ticket_count": 50,
        "source_title": "Role permissions",
        "support_ticket_cluster": "workspace permission setup",
        "product_area": "Workspace Admin",
        "tags": "roles, permissions",
        "texts": (
            "Which role can invite teammates?",
        ),
        "statuses": ("open", "pending", "reopened", "pending"),
        "csat_scores": (3, 3, 2, 2),
    },
)


def synthetic_support_ticket_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cohort in _SYNTHETIC_REPEAT_COHORTS:
        texts = tuple(str(text) for text in cohort["texts"])
        statuses = tuple(str(status) for status in cohort["statuses"])
        csat_scores = tuple(int(score) for score in cohort["csat_scores"])
        for index in range(int(cohort["ticket_count"])):
            day = (index % 30) + 1
            text = texts[index % len(texts)]
            row: dict[str, object] = {
                "source_id": f"synthetic-{cohort['key']}-{index + 1:04d}",
                "source_type": "support_ticket",
                "source_title": cohort["source_title"],
                "support_ticket_cluster": cohort["support_ticket_cluster"],
                "text": text,
                "created_at": f"2026-05-{day:02d}T12:00:00Z",
                "status": statuses[index % len(statuses)],
                "csat_score": csat_scores[index % len(csat_scores)],
                "group": "Synthetic Support",
                "product_area": cohort["product_area"],
                "tags": cohort["tags"],
            }
            resolution_text = cohort.get("resolution_text")
            if resolution_text:
                row["resolution_text"] = resolution_text
            rows.append(row)
    return rows


def producer_deflection_report_payload() -> dict[str, object]:
    result = build_ticket_faq_markdown(
        synthetic_support_ticket_rows(),
        title="Support Ticket FAQ Source",
        max_items=8,
        max_evidence_per_item=1,
        support_contact="https://example.com/support",
        documentation_terms=(
            "Download report",
            "Single sign-on setup",
            "Subscription pause",
            "Invoice contacts",
            "Workspace roles",
        ),
        vocabulary_gap_rules=(
            ("export", "Download report"),
            ("SSO", "Single sign-on setup"),
            ("report download", "Download report"),
            ("pause subscription", "Subscription pause"),
            ("invoice recipient", "Invoice contacts"),
            ("invite permissions", "Workspace roles"),
        ),
    )

    return build_deflection_report_artifact(result).as_dict()


def build_snapshot_example_payload() -> dict[str, Any]:
    return build_deflection_snapshot(
        producer_deflection_report_payload(),
        top_n=SNAPSHOT_TOP_N,
    ).as_dict()


def render_report_example(payload: dict[str, Any] | None = None) -> str:
    return json.dumps(
        payload if payload is not None else producer_deflection_report_payload(),
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_snapshot_example(payload: dict[str, Any] | None = None) -> str:
    return json.dumps(
        payload if payload is not None else build_snapshot_example_payload(),
        indent=2,
        sort_keys=True,
    ) + "\n"


def _check_file(path: Path, expected: str) -> int:
    try:
        current = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"{path} is missing; run this generator to create it", file=sys.stderr)
        return 1
    if current != expected:
        print(
            f"{path} is stale; run this generator to refresh it",
            file=sys.stderr,
        )
        return 1
    print(f"{path} is current")
    return 0


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path}")


def _sibling_report_output(snapshot_output: Path) -> Path:
    suffix = snapshot_output.suffix or ".json"
    return snapshot_output.with_name(f"{snapshot_output.stem}.report{suffix}")


def _resolve_example_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    snapshot_output = args.snapshot_output or args.output or DEFAULT_SNAPSHOT_OUTPUT
    if args.report_output is not None:
        return args.report_output, snapshot_output
    if args.output is None and args.snapshot_output is None:
        return DEFAULT_REPORT_OUTPUT, snapshot_output
    return _sibling_report_output(snapshot_output), snapshot_output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Deprecated alias for --snapshot-output.",
    )
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--snapshot-output", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report_output, snapshot_output = _resolve_example_outputs(args)
    report_payload = producer_deflection_report_payload()
    expected_report = render_report_example(report_payload)
    expected_snapshot = render_snapshot_example(
        build_deflection_snapshot(
            report_payload,
            top_n=SNAPSHOT_TOP_N,
        ).as_dict()
    )

    if args.check:
        return max(
            _check_file(report_output, expected_report),
            _check_file(snapshot_output, expected_snapshot),
        )

    _write_file(report_output, expected_report)
    _write_file(snapshot_output, expected_snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
