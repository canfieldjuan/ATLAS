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
SNAPSHOT_TOP_N = 2


def producer_deflection_report_payload() -> dict[str, object]:
    result = build_ticket_faq_markdown(
        [
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
        ],
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
