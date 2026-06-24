#!/usr/bin/env python3
"""Generate or check the frontend deflection report/snapshot example JSON."""
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


DEFAULT_REPORT_OUTPUT = (
    ROOT / "docs/frontend/content_ops_faq_deflection_report_example.json"
)
DEFAULT_SNAPSHOT_OUTPUT = (
    ROOT / "docs/frontend/content_ops_faq_deflection_snapshot_example.json"
)
SNAPSHOT_TOP_N = 2


def synthetic_deflection_demo_records() -> tuple[dict[str, str], ...]:
    """Return fabricated support-ticket rows for the public demo example."""

    return (
        {
            "source_id": "ticket-export-1",
            "source_type": "support_ticket",
            "source_title": "Export attribution",
            "text": "How do I export attribution reports?",
            "created_at": "2026-05-01T12:00:00Z",
            "resolution_text": (
                "Open Analytics, choose Attribution, then click Download report"
            ),
        },
        {
            "source_id": "ticket-export-2",
            "source_type": "support_ticket",
            "source_title": "Report download",
            "text": "Where is the report download for attribution exports?",
            "created_at": "2026-05-03",
            "resolution_text": (
                "Open Analytics, choose Attribution, then click Download report"
            ),
        },
        {
            "source_id": "ticket-sso-1",
            "source_type": "support_ticket",
            "source_title": "SSO setup",
            "text": "How do I enable SSO for my team?",
            "created_at": "2026-05-10",
        },
        {
            "source_id": "ticket-sso-2",
            "source_type": "support_ticket",
            "source_title": "SSO setup",
            "text": "Can we enable SSO for our team?",
            "created_at": "2026-05-15",
        },
    )


def producer_deflection_report_payload() -> dict[str, object]:
    result = build_ticket_faq_markdown(
        synthetic_deflection_demo_records(),
        title="Support Ticket FAQ Source",
        max_items=2,
        max_evidence_per_item=1,
        support_contact="https://example.com/support",
        documentation_terms=("Download report", "Single sign-on setup"),
        vocabulary_gap_rules=(
            ("export", "Download report"),
            ("SSO", "Single sign-on setup"),
            ("report download", "Download report"),
        ),
    )

    return build_deflection_report_artifact(result).as_dict()


def build_report_example_payload() -> dict[str, object]:
    return producer_deflection_report_payload()


def build_snapshot_example_payload(
    report_payload: dict[str, object] | None = None,
) -> dict[str, Any]:
    source_payload = (
        report_payload if report_payload is not None else build_report_example_payload()
    )
    return build_deflection_snapshot(
        source_payload,
        top_n=SNAPSHOT_TOP_N,
    ).as_dict()


def _render_json(payload: dict[str, Any] | dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_report_example(payload: dict[str, object] | None = None) -> str:
    return _render_json(
        payload if payload is not None else build_report_example_payload()
    )


def render_snapshot_example(payload: dict[str, Any] | None = None) -> str:
    return _render_json(
        payload if payload is not None else build_snapshot_example_payload()
    )


def _check_output(output: Path, expected: str) -> int:
    try:
        current = output.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"{output} is missing; run this generator to create it", file=sys.stderr)
        return 1
    if current != expected:
        print(f"{output} is stale; run this generator to refresh it", file=sys.stderr)
        return 1
    print(f"{output} is current")
    return 0


def _write_output(output: Path, expected: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"wrote {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "--snapshot-output",
        dest="snapshot_output",
        type=Path,
        default=DEFAULT_SNAPSHOT_OUTPUT,
    )
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report_payload = build_report_example_payload()
    expected = (
        (args.report_output, render_report_example(report_payload)),
        (
            args.snapshot_output,
            render_snapshot_example(build_snapshot_example_payload(report_payload)),
        ),
    )

    if args.check:
        return max(_check_output(output, text) for output, text in expected)

    for output, text in expected:
        _write_output(output, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
