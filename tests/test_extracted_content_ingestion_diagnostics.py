from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import extracted_content_pipeline.ingestion_diagnostics as ingestion_diagnostics
from extracted_content_pipeline.ingestion_diagnostics import inspect_ingestion_file


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/inspect_extracted_content_ingestion.py"
ZERO_USABLE_DECISION = {
    "status": "REJECT",
    "reason": "no_usable_source_rows",
    "location": "source_row_csv",
    "message": "No usable support-ticket text was found in this file.",
    "how_to_fix": (
        "Re-export your tickets with the customer message or description "
        "column included, then upload again. Rows marked private or internal "
        "are skipped on purpose and do not count toward usable text."
    ),
}


def test_inspect_opportunity_json_reports_ready_rows(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(json.dumps([
        {
            "id": "opp-1",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "email": "buyer@example.com",
            "evidence": [{"text": "Renewal pricing is under review."}],
        }
    ]))

    report = inspect_ingestion_file(path, sample_limit=1)
    payload = report.as_dict()

    assert payload["ok"] is True
    assert payload["mode"] == "opportunities"
    assert payload["opportunity_count"] == 1
    assert payload["warning_count"] == 0
    assert payload["missing_field_counts"] == {}
    assert payload["samples"][0]["target_id"] == "opp-1"
    assert "source_row_admission" not in payload


def test_inspect_source_rows_counts_source_types_and_warnings(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "ticket_id": "ticket-1",
                "company": "Acme Logistics",
                "vendor": "HubSpot",
                "message": "Renewal quote jumped before export work finished.",
            }),
            json.dumps({
                "id": "empty-1",
                "company": "Beta",
                "vendor": "Zendesk",
            }),
        ])
    )

    report = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="jsonl",
        sample_limit=2,
    )
    payload = report.as_dict()

    assert payload["ok"] is True
    assert payload["mode"] == "source_rows"
    assert payload["opportunity_count"] == 1
    assert payload["warning_counts"] == {
        "missing_contact_email": 1,
        "missing_source_text": 1,
    }
    assert payload["source_type_counts"] == {"support_ticket": 1}
    assert payload["samples"][0]["evidence"][0]["source_type"] == "support_ticket"


def test_inspect_source_jsonl_reports_malformed_line_without_aborting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "ticket_id": "ticket-1",
                "company": "Acme Logistics",
                "vendor": "HubSpot",
                "message": "Renewal quote jumped before export work finished.",
            }),
            '{"ticket_id": "bad", "message": ',
            json.dumps({
                "ticket_id": "ticket-2",
                "company": "Beta",
                "vendor": "Zendesk",
                "message": "Ticket exports timeout after filtering.",
            }),
        ]),
        encoding="utf-8",
    )

    report = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="jsonl",
        sample_limit=2,
    )
    payload = report.as_dict()

    assert payload["ok"] is True
    assert payload["opportunity_count"] == 2
    assert payload["warning_counts"] == {
        "malformed_jsonl_line": 1,
        "missing_contact_email": 2,
    }
    assert payload["warnings"][0] == {
        "code": "malformed_jsonl_line",
        "message": (
            "Skipped JSONL source row because it is not valid JSON "
            "(Expecting value at column 32)."
        ),
        "row_index": 2,
        "field": "jsonl",
    }
    assert "bad" not in payload["warnings"][0]["message"]
    assert [
        warning["row_index"]
        for warning in payload["warnings"]
        if warning["code"] == "missing_contact_email"
    ] == [1, 3]


def test_inspect_source_rows_applies_default_fields(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(json.dumps({
        "id": "g2-review-1",
        "vendor": "Slack",
        "review_text": "Search gets slow once message history grows.",
    }))

    report = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="jsonl",
        default_fields={
            "company_name": "Acme Logistics",
            "contact_email": "ops@example.com",
        },
    )

    assert report.ok is True
    assert report.opportunities[0]["company_name"] == "Acme Logistics"
    assert report.opportunities[0]["contact_email"] == "ops@example.com"


def test_inspect_source_csv_reports_unmapped_populated_text_column(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    path.write_text(
        "Ticket ID,Conversation Text\n"
        "T-1,Customer cannot export the weekly report.\n"
    )

    report = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
    )
    payload = report.as_dict()

    assert payload["ok"] is False
    assert payload["warning_counts"] == {"missing_source_text": 1}
    admission = payload["source_row_admission"]
    assert admission["input_format"] == "csv"
    assert admission["raw_source_row_count"] == 1
    assert admission["usable_source_row_count"] == 0
    assert admission["usable_source_ratio"] == 0.0
    assert admission["mapped_fields"] == {"source_id": ["Ticket ID"]}
    assert admission["populated_unmapped_fields"] == ["Conversation Text"]
    assert admission["admission_decision"] == ZERO_USABLE_DECISION
    assert "Customer cannot export" not in admission["admission_decision"]["message"]
    assert (
        "Customer cannot export"
        not in admission["admission_decision"]["how_to_fix"]
    )
    assert "coverage_warnings" not in admission


def test_inspect_source_csv_classifies_zendesk_public_and_private_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "zendesk.csv"
    path.write_text(
        "Ticket ID,Subject,Public Comments,Internal Notes\n"
        "T-1,Export failed,Customer cannot export the weekly report.,"
        "Use the private workaround.\n"
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
    ).as_dict()

    assert payload["ok"] is True
    admission = payload["source_row_admission"]
    assert admission["mapped_fields"] == {
        "source_id": ["Ticket ID"],
        "source_title": ["Subject"],
        "thread_text": ["Public Comments"],
    }
    assert admission["ignored_private_fields"] == ["Internal Notes"]
    assert admission["populated_unmapped_fields"] == []
    assert admission["admission_decision"] == {"status": "ACCEPT"}
    assert "coverage_warnings" not in admission


def test_inspect_source_csv_warns_when_accepted_upload_has_skipped_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.csv"
    path.write_text(
        "Ticket ID,Message,Conversation Text\n"
        "T-1,Customer cannot export the weekly report.,\n"
        "T-2,,Customer cannot invite a teammate.\n"
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
        default_fields={
            "company_name": "Acme Logistics",
            "vendor_name": "Atlas",
            "contact_email": "ops@example.com",
        },
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is True
    assert payload["warning_counts"] == {"missing_source_text": 1}
    assert admission["admission_decision"] == {"status": "ACCEPT"}
    assert admission["coverage_warnings"] == [{
        "code": "partial_source_row_coverage",
        "location": "source_row_csv",
        "raw_source_row_count": 2,
        "usable_source_row_count": 1,
        "skipped_source_row_count": 1,
        "usable_source_ratio": 0.5,
    }]


def test_inspect_source_csv_rejects_machine_json_message_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "machine-json.csv"
    path.write_text(
        'Ticket ID,Message\n'
        'T-1,"{""event"":""ticket_created"",""id"":123}"\n',
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
        default_fields={
            "company_name": "Acme Logistics",
            "vendor_name": "Atlas",
            "contact_email": "ops@example.com",
        },
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is False
    assert payload["warning_counts"] == {"machine_source_payload_text": 1}
    assert admission["raw_source_row_count"] == 1
    assert admission["usable_source_row_count"] == 0
    assert admission["usable_source_ratio"] == 0.0
    assert admission["mapped_fields"] == {
        "source_id": ["Ticket ID"],
        "source_text": ["Message"],
    }
    assert admission["admission_decision"] == ZERO_USABLE_DECISION


def test_inspect_source_csv_warns_when_partial_upload_has_machine_json_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial-machine-json.csv"
    path.write_text(
        'Ticket ID,Message\n'
        'T-1,Customer cannot export the weekly report.\n'
        'T-2,"{""event"":""ticket_created"",""id"":123}"\n',
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
        default_fields={
            "company_name": "Acme Logistics",
            "vendor_name": "Atlas",
            "contact_email": "ops@example.com",
        },
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is True
    assert payload["warning_counts"] == {"machine_source_payload_text": 1}
    assert admission["admission_decision"] == {"status": "ACCEPT"}
    assert admission["coverage_warnings"] == [{
        "code": "partial_source_row_coverage",
        "location": "source_row_csv",
        "raw_source_row_count": 2,
        "usable_source_row_count": 1,
        "skipped_source_row_count": 1,
        "usable_source_ratio": 0.5,
    }]


def test_partial_coverage_csv_still_produces_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "see-something.csv"
    path.write_text(
        "Ticket ID,Message,Internal Notes,Unused Column\n"
        "T-1,Customer cannot export the weekly report.,,\n"
        "T-2,,secret_ticket_text_private_note,\n"
        "T-3,,,metadata only\n",
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=2,
        default_fields={
            "company_name": "Acme Logistics",
            "vendor_name": "Atlas",
            "contact_email": "ops@example.com",
        },
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is True
    assert payload["opportunity_count"] >= 1
    assert admission["admission_decision"] == {"status": "ACCEPT"}
    assert admission["coverage_warnings"] == [{
        "code": "partial_source_row_coverage",
        "location": "source_row_csv",
        "raw_source_row_count": 3,
        "usable_source_row_count": 1,
        "skipped_source_row_count": 2,
        "usable_source_ratio": 0.333333,
    }]


def test_zero_usable_csv_rejects_with_guidance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "zero-usable.csv"
    path.write_text(
        "Ticket ID,Internal Notes,Unused Column\n"
        "T-1,secret_ticket_text_private_note,metadata only\n",
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
    ).as_dict()

    decision = payload["source_row_admission"]["admission_decision"]
    assert payload["ok"] is False
    assert decision == ZERO_USABLE_DECISION
    assert decision["message"]
    assert decision["how_to_fix"]
    assert "secret_ticket_text_private_note" not in decision["message"]
    assert "secret_ticket_text_private_note" not in decision["how_to_fix"]


def test_inspect_source_csv_accepts_long_ticket_body_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long-ticket-body.csv"
    long_body = "Customer cannot export the weekly report. " + ("x" * 200_000)
    path.write_text(
        "Ticket ID,Message\n"
        f"T-1,{long_body}\n",
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
        default_fields={
            "company_name": "Acme Logistics",
            "vendor_name": "Atlas",
            "contact_email": "ops@example.com",
        },
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is True
    assert payload["warning_counts"] == {}
    assert admission["raw_source_row_count"] == 1
    assert admission["usable_source_row_count"] == 1
    assert admission["usable_source_ratio"] == 1.0
    assert admission["mapped_fields"] == {
        "source_id": ["Ticket ID"],
        "source_text": ["Message"],
    }
    assert admission["admission_decision"] == {"status": "ACCEPT"}
    assert "coverage_warnings" not in admission


@pytest.mark.parametrize(
    ("header", "text"),
    (
        ("actionbody", "Customer cannot download the invoice PDF."),
        ("Ticket Description", "The customer cannot export the weekly report."),
    ),
)
def test_inspect_source_csv_accepts_observed_export_body_aliases(
    tmp_path: Path,
    header: str,
    text: str,
) -> None:
    path = tmp_path / "observed-body-alias.csv"
    path.write_text(
        f"Ticket ID,{header}\n"
        f"T-1,{text}\n",
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
        default_fields={
            "company_name": "Acme Logistics",
            "vendor_name": "Atlas",
            "contact_email": "ops@example.com",
        },
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is True
    assert payload["warning_counts"] == {}
    assert admission["raw_source_row_count"] == 1
    assert admission["usable_source_row_count"] == 1
    assert admission["usable_source_ratio"] == 1.0
    assert admission["mapped_fields"] == {
        "source_id": ["Ticket ID"],
        "source_text": [header],
    }
    assert admission["admission_decision"] == {"status": "ACCEPT"}
    assert admission.get("coverage_warnings", []) == []


def test_inspect_source_csv_rejects_observed_body_alias_when_row_is_private(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private-observed-body-alias.csv"
    path.write_text(
        "Ticket ID,actionbody,is_private\n"
        "T-1,Do not publish this internal workaround.,1.0\n",
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
        default_fields={
            "company_name": "Acme Logistics",
            "vendor_name": "Atlas",
            "contact_email": "ops@example.com",
        },
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is False
    assert payload["warning_counts"] == {"private_source_text": 1}
    assert admission["raw_source_row_count"] == 1
    assert admission["usable_source_row_count"] == 0
    assert admission["usable_source_ratio"] == 0.0
    assert admission["mapped_fields"] == {
        "source_id": ["Ticket ID"],
        "source_text": ["actionbody"],
    }
    assert admission["admission_decision"] == ZERO_USABLE_DECISION


def test_inspect_source_csv_sample_limit_does_not_make_known_aliases_unmapped(
    tmp_path: Path,
) -> None:
    id_fields = [
        "Source ID",
        "ID",
        "Chat ID",
        "Sales Objection ID",
        "Objection ID",
        "Review ID",
        "Transcript ID",
        "Call ID",
        "Meeting ID",
        "Recording ID",
        "Deal ID",
        "Opportunity ID",
        "Note ID",
        "Activity ID",
        "Renewal ID",
        "Contract ID",
        "Subscription ID",
        "Document ID",
        "Search ID",
        "Search Log ID",
        "Search Query ID",
        "Query ID",
        "Zero Result Query ID",
        "Complaint ID",
        "Ticket ID",
        "Ticket Number",
    ]
    path = tmp_path / "wide.csv"
    path.write_text(
        ",".join([*id_fields, "Message"]) + "\n"
        + ",".join([f"src-{index}" for index in range(len(id_fields))])
        + ",Customer cannot export the weekly report.\n"
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
        sample_limit=1,
    ).as_dict()

    admission = payload["source_row_admission"]
    assert len(admission["mapped_fields"]["source_id"]) == 25
    assert "Ticket Number" not in admission["mapped_fields"]["source_id"]
    assert admission["populated_unmapped_fields"] == []


def test_inspect_source_jsonl_does_not_emit_csv_admission_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(json.dumps({
        "ticket_id": "ticket-1",
        "message": "Customer cannot export the weekly report.",
    }))

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="jsonl",
    ).as_dict()

    assert payload["ok"] is True
    assert "source_row_admission" not in payload


def test_inspect_source_csv_with_header_only_has_no_hard_policy_decision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("Ticket ID,Message\n")

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
    ).as_dict()

    admission = payload["source_row_admission"]
    assert payload["ok"] is False
    assert admission["raw_source_row_count"] == 0
    assert admission["usable_source_row_count"] == 0
    assert admission["usable_source_ratio"] is None
    assert "admission_decision" not in admission
    assert "coverage_warnings" not in admission


def test_inspect_source_csv_reports_missing_header_parse_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-header.csv"
    path.write_text(
        "export generated by support system\n"
        "not an actual header row\n",
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
    ).as_dict()

    assert payload["ok"] is False
    assert payload["mode"] == "source_rows"
    assert payload["opportunity_count"] == 0
    assert "source_row_admission" not in payload
    assert payload["parse_error"] == {
        "code": "csv_missing_header",
        "message": "CSV customer data must include a header row.",
        "how_to_fix": (
            "Add a header row with column names such as ticket_id, subject, "
            "and message, then export the CSV again."
        ),
        "location": "source_row_csv",
    }


def test_inspect_source_csv_reports_inconsistent_columns_parse_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-columns.csv"
    path.write_text(
        "Ticket ID,Message\n"
        "T-1,Customer cannot export the weekly report.\n"
        "T-2,Customer cannot invite a teammate.,extra cell\n",
        encoding="utf-8",
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
    ).as_dict()

    assert payload["ok"] is False
    assert payload["parse_error"]["code"] == "csv_inconsistent_columns"
    assert payload["parse_error"]["row_index"] == 3
    assert payload["parse_error"]["location"] == "source_row_csv"
    assert "Customer cannot invite" not in payload["parse_error"]["message"]


def test_inspect_opportunity_json_reports_parse_error_without_raw_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"secret_ticket_text": ', encoding="utf-8")

    payload = inspect_ingestion_file(path, file_format="json").as_dict()

    assert payload["ok"] is False
    assert payload["mode"] == "opportunities"
    assert payload["opportunity_count"] == 0
    assert payload["parse_error"] == {
        "code": "json_parse_error",
        "message": (
            "JSON customer data could not be parsed "
            "(Expecting value at line 1, column 24)."
        ),
        "how_to_fix": (
            "Export the file again as valid UTF-8 JSON, then upload the new export."
        ),
        "location": "customer_data_json",
        "line": 1,
        "column": 24,
    }
    assert "secret_ticket_text" not in payload["parse_error"]["message"]


def test_inspect_opportunity_json_reports_decode_error_without_raw_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-encoding.json"
    path.write_bytes(b'{"secret_ticket_text": "\xff"}')

    payload = inspect_ingestion_file(path, file_format="json").as_dict()

    assert payload["ok"] is False
    assert payload["parse_error"] == {
        "code": "file_decode_error",
        "message": (
            "Ingestion file could not be decoded as utf-8 text at byte 24."
        ),
        "how_to_fix": (
            "Export the file again as valid UTF-8 CSV or JSON, then upload "
            "the new export."
        ),
        "location": "customer_data_json",
        "encoding": "utf-8",
        "byte": 24,
    }
    assert "secret_ticket_text" not in payload["parse_error"]["message"]
    assert "\\xff" not in payload["parse_error"]["message"]


def test_inspect_source_csv_reports_decode_error_without_raw_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bad-encoding.csv"
    path.write_text("Ticket ID,Message\nT-1,placeholder\n", encoding="utf-8")

    def raise_decode_error(*args: object, **kwargs: object) -> None:
        raise UnicodeDecodeError(
            "utf-8",
            b"secret-private-ticket-\xff",
            22,
            23,
            "bad byte",
        )

    monkeypatch.setattr(
        ingestion_diagnostics,
        "load_source_rows_with_warnings_from_file",
        raise_decode_error,
    )

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
    ).as_dict()

    assert payload["ok"] is False
    assert payload["parse_error"] == {
        "code": "file_decode_error",
        "message": (
            "Ingestion file could not be decoded as utf-8 text at byte 22."
        ),
        "how_to_fix": (
            "Export the file again as valid UTF-8 CSV or JSON, then upload "
            "the new export."
        ),
        "location": "source_row_csv",
        "encoding": "utf-8",
        "byte": 22,
    }
    assert "secret-private-ticket" not in payload["parse_error"]["message"]
    assert "\\xff" not in payload["parse_error"]["message"]


def test_inspect_source_csv_reports_raw_csv_parser_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oversized-cell.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Ticket ID,Message\nT-1,")
        handle.write("x" * (16 * 1024 * 1024 + 1))
        handle.write("\n")

    payload = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="csv",
    ).as_dict()

    assert payload["ok"] is False
    assert payload["parse_error"]["code"] == "csv_parse_error"
    assert payload["parse_error"]["location"] == "source_row_csv"
    assert "field larger than field limit" in payload["parse_error"]["message"]
    assert "xxxxxxxx" not in payload["parse_error"]["message"]


def test_inspect_reports_missing_generation_fields(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(json.dumps([
        {
            "id": "opp-1",
            "company": "Acme Logistics",
        }
    ]))

    payload = inspect_ingestion_file(path).as_dict()

    assert payload["ok"] is True
    assert payload["missing_field_counts"] == {
        "evidence": 1,
        "vendor_name": 1,
    }


def test_inspect_sample_limit_bounds_output(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(json.dumps([
        {
            "id": f"opp-{index}",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "evidence": [{"text": "Renewal pricing is under review."}],
        }
        for index in range(3)
    ]))

    payload = inspect_ingestion_file(path, sample_limit=2).as_dict()

    assert payload["opportunity_count"] == 3
    assert [sample["target_id"] for sample in payload["samples"]] == ["opp-0", "opp-1"]


def test_inspection_cli_emits_valid_json_for_source_rows(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(json.dumps({
        "review_id": "review-1",
        "reviewer_company": "Acme Logistics",
        "vendor": "HubSpot",
        "review_text": "Pricing is hard to justify.",
    }))

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--source-rows",
            "--source-format",
            "jsonl",
            "--sample-limit",
            "1",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["source_type_counts"] == {"review": 1}
    assert len(payload["samples"]) == 1


def test_inspection_cli_emits_csv_source_row_admission(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    path.write_text(
        "Ticket ID,Conversation Text\n"
        "T-1,Customer cannot export the weekly report.\n"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--source-rows",
            "--source-format",
            "csv",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert proc.returncode == 1
    assert payload["source_row_admission"]["raw_source_row_count"] == 1
    assert payload["source_row_admission"]["usable_source_row_count"] == 0
    assert payload["source_row_admission"]["populated_unmapped_fields"] == [
        "Conversation Text"
    ]
    assert payload["source_row_admission"]["admission_decision"] == (
        ZERO_USABLE_DECISION
    )


def test_inspection_cli_emits_structured_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "bad-columns.csv"
    path.write_text(
        "Ticket ID,Message\n"
        "T-1,Customer cannot export the weekly report.,extra cell\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--source-rows",
            "--source-format",
            "csv",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert payload["ok"] is False
    assert payload["parse_error"]["code"] == "csv_inconsistent_columns"
    assert payload["parse_error"]["location"] == "source_row_csv"


def test_inspection_cli_rejects_negative_sample_limit(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text("[]")

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--sample-limit",
            "-1",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "--sample-limit must be non-negative" in proc.stderr
