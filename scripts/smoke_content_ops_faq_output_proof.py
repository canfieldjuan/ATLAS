#!/usr/bin/env python3
"""Run a compact end-to-end proof for generated FAQ Markdown output."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.faq_output_ingestion import (
    FAQ_OUTPUT_SOURCE_TYPE,
    faq_output_to_source_rows,
)
from extracted_content_pipeline.campaign_source_adapters import (
    load_source_campaign_opportunities_from_file,
)
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


FAQ_CLI = ROOT / "scripts" / "build_extracted_ticket_faq_markdown.py"
DEFAULT_SUPPORT_CONTACT = "support@example.com"
DEFAULT_DOCUMENTATION_TERMS = (
    "Download report",
    "Analytics",
    "Single sign-on setup",
    "Invoice settings",
)
DEFAULT_VOCABULARY_RULES = (
    "SSO,single sign-on",
)
DEFAULT_ROWS: tuple[dict[str, str], ...] = (
    {
        "source_type": "support_ticket",
        "source_id": "ticket-export-1",
        "source_title": "Export attribution report",
        "text": "How do I export the attribution dashboard before renewal?",
        "resolution_text": "Open Analytics, choose Attribution, then select Download report.",
        "pain_category": "reporting friction",
    },
    {
        "source_type": "support_ticket",
        "source_id": "ticket-export-2",
        "source_title": "Reporting dashboard missing export",
        "text": "The reporting dashboard export is missing for my analyst role.",
        "resolution_text": "Enable Report Downloads for the analyst role before exporting.",
        "pain_category": "reporting friction",
    },
    {
        "source_type": "search_log",
        "source_id": "search-export-1",
        "source_title": "Zero-result export search",
        "text": "export attribution report",
        "search_count": "18",
        "results_count": "0",
    },
    {
        "source_type": "support_ticket",
        "source_id": "ticket-sso-1",
        "source_title": "Integration SSO setup",
        "text": "How do I configure SSO for my team integration?",
        "pain_category": "integration setup",
    },
    {
        "source_type": "sales_objection",
        "source_id": "objection-sso-1",
        "source_title": "Security setup objection",
        "text": "Prospects ask whether SSO setup works before they connect the integration.",
    },
    {
        "source_type": "support_ticket",
        "source_id": "ticket-billing-1",
        "source_title": "Invoice charge dispute",
        "text": "Why was I charged a fee after the invoice was already paid?",
        "pain_category": "billing and payments",
    },
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_output_proof(args)
    _print_summary(summary)
    return 0 if summary["ok"] else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--support-contact", default=DEFAULT_SUPPORT_CONTACT)
    return parser.parse_args(argv)


def run_output_proof(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_path = artifact_dir / "faq_source_rows.csv"
    _write_source_csv(source_path, DEFAULT_ROWS)

    markdown_path = artifact_dir / "faq_output.md"
    result_path = artifact_dir / "faq_result.json"
    full_result_path = artifact_dir / "faq_full_result.json"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    summary_path = artifact_dir / "proof_summary.json"
    command = _faq_command(
        source_path=source_path,
        markdown_path=markdown_path,
        result_path=result_path,
        support_contact=args.support_contact,
    )
    started = time.monotonic()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed_seconds = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    result_payload = _read_json(result_path)
    full_result_payload = _full_faq_result_payload(
        source_path=source_path,
        support_contact=args.support_contact,
    )
    full_result_path.write_text(
        json.dumps(full_result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
    proof = _proof_payload(
        result_payload=result_payload,
        full_result_payload=full_result_payload,
        markdown=markdown,
        support_contact=args.support_contact,
    )
    failures = _proof_failures(
        returncode=completed.returncode,
        result_payload=result_payload,
        proof=proof,
        markdown=markdown,
        support_contact=args.support_contact,
    )
    summary = {
        "ok": not failures,
        "exit_code": completed.returncode,
        "source": str(source_path),
        "command": command,
        "timing": {"elapsed_seconds": round(elapsed_seconds, 6)},
        "artifacts": {
            "source": str(source_path),
            "markdown": str(markdown_path) if markdown_path.exists() else None,
            "result": str(result_path) if result_path.exists() else None,
            "full_result": str(full_result_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(summary_path),
        },
        "proof": proof,
        "failures": failures,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _faq_command(
    *,
    source_path: Path,
    markdown_path: Path,
    result_path: Path,
    support_contact: str,
) -> list[str]:
    command = [
        sys.executable,
        str(FAQ_CLI),
        str(source_path),
        "--source-format",
        "csv",
        "--title",
        "Support Ticket FAQ Output Proof",
        "--max-items",
        "6",
        "--output",
        str(markdown_path),
        "--result-output",
        str(result_path),
        "--require-output-checks",
        "--support-contact",
        support_contact,
    ]
    for term in DEFAULT_DOCUMENTATION_TERMS:
        command.extend(("--documentation-term", term))
    for rule in DEFAULT_VOCABULARY_RULES:
        command.extend(("--vocabulary-gap-rule", rule))
    return command


def _full_faq_result_payload(*, source_path: Path, support_contact: str) -> dict[str, Any]:
    loaded = load_source_campaign_opportunities_from_file(
        source_path,
        file_format="csv",
    )
    result = build_ticket_faq_markdown(
        loaded.opportunities,
        title="Support Ticket FAQ Output Proof",
        max_items=6,
        support_contact=support_contact,
        documentation_terms=DEFAULT_DOCUMENTATION_TERMS,
        vocabulary_gap_rules=tuple(
            tuple(part.strip() for part in rule.split(",") if part.strip())
            for rule in DEFAULT_VOCABULARY_RULES
        ),
    )
    return result.as_dict()


def _proof_payload(
    *,
    result_payload: Mapping[str, Any] | None,
    full_result_payload: Mapping[str, Any] | None,
    markdown: str,
    support_contact: str,
) -> dict[str, Any]:
    diagnostics = _mapping(result_payload.get("diagnostics")) if result_payload else {}
    run_summary = _mapping(diagnostics.get("run_summary"))
    vocabulary_gaps = _mapping(run_summary.get("vocabulary_gaps"))
    items = [
        dict(item)
        for item in _sequence(diagnostics.get("items"))
        if isinstance(item, Mapping)
    ]
    topics = [str(item.get("topic") or "") for item in items if item.get("topic")]
    source_id_counts = [
        _integer(item.get("source_id_count")) for item in items
    ]
    step_counts = [_integer(item.get("step_count")) for item in items]
    output_checks = _mapping(result_payload.get("output_checks")) if result_payload else {}
    bridge = _ingestion_bridge_proof(full_result_payload)
    return {
        "status": str(result_payload.get("status") or "") if result_payload else "",
        "generated": _integer(result_payload.get("generated")) if result_payload else 0,
        "source_count": _integer(result_payload.get("source_count")) if result_payload else 0,
        "ticket_source_count": (
            _integer(result_payload.get("ticket_source_count")) if result_payload else 0
        ),
        "output_checks": dict(output_checks),
        "failed_output_checks": list(_sequence(result_payload.get("failed_output_checks")))
        if result_payload
        else [],
        "topics": topics,
        "first_source_ids": [
            item.get("first_source_id") for item in items if item.get("first_source_id")
        ],
        "min_source_id_count": min(source_id_counts) if source_id_counts else 0,
        "min_step_count": min(step_counts) if step_counts else 0,
        "support_contact_present": support_contact in markdown,
        "source_ids_present": all(
            source_id in markdown
            for source_id in ("ticket-export-1", "search-export-1", "ticket-sso-1")
        ),
        "vocabulary_gaps": dict(vocabulary_gaps),
        "ingestion_bridge": bridge,
        "markdown_bytes": len(markdown.encode("utf-8")),
    }


def _ingestion_bridge_proof(result_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result_payload, Mapping):
        return {
            "adapted_source_row_count": 0,
            "resolution_text_row_count": 0,
            "support_ticket_resolution_evidence_present": False,
            "support_ticket_resolution_evidence_count": 0,
            "support_ticket_resolution_example_count": 0,
        }
    source_rows = faq_output_to_source_rows(result_payload)
    resolution_text_rows = [
        row for row in source_rows
        if isinstance(row, Mapping) and row.get("resolution_text")
    ]
    package = build_support_ticket_input_package(
        result_payload,
        outputs=("blog_post",),
    )
    return {
        "adapted_source_row_count": len(source_rows),
        "adapted_source_types": sorted({
            str(row.get("source_type") or "")
            for row in source_rows
            if isinstance(row, Mapping) and row.get("source_type")
        }),
        "resolution_text_row_count": len(resolution_text_rows),
        "support_ticket_resolution_evidence_present": (
            package.inputs.get("support_ticket_resolution_evidence_present") is True
        ),
        "support_ticket_resolution_evidence_count": _integer(
            package.inputs.get("support_ticket_resolution_evidence_count")
        ),
        "support_ticket_resolution_example_count": len(_sequence(
            package.inputs.get("support_ticket_resolution_examples")
        )),
    }


def _proof_failures(
    *,
    returncode: int,
    result_payload: Mapping[str, Any] | None,
    proof: Mapping[str, Any],
    markdown: str,
    support_contact: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if returncode != 0:
        failures.append({"check": "cli_exit_zero", "detail": returncode})
    if not isinstance(result_payload, Mapping):
        failures.append({"check": "result_json_present", "detail": "missing_or_invalid"})
        return failures
    if proof.get("status") != "ok":
        failures.append({"check": "result_status_ok", "detail": proof.get("status")})
    failed_output_checks = list(_sequence(proof.get("failed_output_checks")))
    if failed_output_checks:
        failures.append({
            "check": "output_checks_pass",
            "detail": failed_output_checks,
        })
    output_checks = _mapping(proof.get("output_checks"))
    for name in ("uses_user_vocabulary", "condensed", "has_action_items"):
        if output_checks.get(name) is not True:
            failures.append({"check": f"output_check_{name}", "detail": output_checks.get(name)})
    topics = set(str(topic) for topic in _sequence(proof.get("topics")))
    for topic in ("reporting friction", "integration setup", "billing and payments"):
        if topic not in topics:
            failures.append({"check": "topic_present", "detail": topic})
    if _integer(proof.get("generated")) < 3:
        failures.append({"check": "generated_items", "detail": proof.get("generated")})
    if _integer(proof.get("min_source_id_count")) < 1:
        failures.append({"check": "source_id_coverage", "detail": proof.get("min_source_id_count")})
    if _integer(proof.get("min_step_count")) < 2:
        failures.append({"check": "action_step_coverage", "detail": proof.get("min_step_count")})
    bridge = _mapping(proof.get("ingestion_bridge"))
    if _integer(bridge.get("adapted_source_row_count")) < _integer(proof.get("generated")):
        failures.append({
            "check": "faq_output_adapter_row_coverage",
            "detail": bridge.get("adapted_source_row_count"),
        })
    if FAQ_OUTPUT_SOURCE_TYPE not in set(_sequence(bridge.get("adapted_source_types"))):
        failures.append({
            "check": "faq_output_adapter_source_type",
            "detail": bridge.get("adapted_source_types"),
        })
    if _integer(bridge.get("resolution_text_row_count")) < 1:
        failures.append({
            "check": "faq_output_resolution_text_bridge",
            "detail": bridge.get("resolution_text_row_count"),
        })
    if bridge.get("support_ticket_resolution_evidence_present") is not True:
        failures.append({
            "check": "support_ticket_resolution_evidence_present",
            "detail": bridge.get("support_ticket_resolution_evidence_present"),
        })
    if _integer(bridge.get("support_ticket_resolution_evidence_count")) < 1:
        failures.append({
            "check": "support_ticket_resolution_evidence_count",
            "detail": bridge.get("support_ticket_resolution_evidence_count"),
        })
    if proof.get("support_contact_present") is not True:
        failures.append({"check": "support_contact_present", "detail": support_contact})
    if proof.get("source_ids_present") is not True:
        failures.append({"check": "source_ids_rendered", "detail": "expected source ids missing"})
    vocabulary_gaps = _mapping(proof.get("vocabulary_gaps"))
    if _integer(vocabulary_gaps.get("term_mapping_count")) < 2:
        failures.append({
            "check": "vocabulary_gap_mappings",
            "detail": vocabulary_gaps.get("term_mapping_count"),
        })
    top_customer_terms = {
        str(term).lower() for term in _sequence(vocabulary_gaps.get("top_customer_terms"))
    }
    if not {"export", "sso"}.issubset(top_customer_terms):
        failures.append({
            "check": "vocabulary_gap_terms",
            "detail": sorted(top_customer_terms),
        })
    if "What to do next" not in markdown:
        failures.append({"check": "markdown_action_section", "detail": "missing"})
    return failures


def _write_source_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)):
        return []
    if isinstance(value, Sequence):
        return list(value)
    return []


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _print_summary(summary: Mapping[str, Any]) -> None:
    proof = _mapping(summary.get("proof"))
    message = (
        f"generated={proof.get('generated', 0)} "
        f"topics={len(_sequence(proof.get('topics')))} "
        f"vocab_mappings={_mapping(proof.get('vocabulary_gaps')).get('term_mapping_count', 0)} "
        f"summary={_mapping(summary.get('artifacts')).get('summary')}"
    )
    if summary.get("ok") is True:
        print(f"Content Ops FAQ output proof passed: {message}")
        return
    print(
        "Content Ops FAQ output proof failed: "
        f"{message} failures={summary.get('failures')}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
