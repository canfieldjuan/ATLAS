from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from extracted_content_pipeline import campaign_source_adapters as adapters
from extracted_content_pipeline.campaign_customer_data import CsvCustomerDataParseError
from extracted_content_pipeline.campaign_source_adapters import (
    load_csv_source_rows_result_from_file,
    load_source_campaign_opportunities_from_file,
    load_source_rows_from_file,
    load_source_rows_with_warnings_from_file,
    parse_default_fields,
    parse_default_fields_with_booking_url_or_exit,
    parse_default_fields_or_exit,
    source_material_to_source_rows,
    source_row_to_campaign_opportunity,
    source_rows_to_campaign_opportunities,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/build_extracted_campaign_opportunities_from_sources.py"
EXAMPLE_SOURCE_ROWS = (
    ROOT / "extracted_content_pipeline/examples/campaign_source_rows.jsonl"
)
EXAMPLE_SOURCE_BUNDLE = (
    ROOT / "extracted_content_pipeline/examples/campaign_source_bundle.json"
)
EXAMPLE_SUPPORT_TICKET_CSV = (
    ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"
)
EXAMPLE_SUPPORT_TICKET_BUNDLE = (
    ROOT / "extracted_content_pipeline/examples/support_ticket_bundle.json"
)


def test_source_type_precedence_table_matches_current_contract() -> None:
    # This intentionally locks a private table: source-type order is product behavior.
    assert adapters._SOURCE_TYPE_PRECEDENCE == (
        (("review_text",), "review"),
        ((
            "objection",
            "objection_text",
            "buyer_objection",
            "sales_objection",
            "sales_objection_id",
            "objection_id",
        ), "sales_objection"),
        (("transcript",), "transcript"),
        ((
            "search_query",
            "query",
            "query_text",
            "search_term",
            "search_terms",
            "search_phrase",
            "zero_result_query",
            "search_id",
            "search_log_id",
            "search_query_id",
            "query_id",
            "zero_result_query_id",
        ), "search_log"),
        (("chat_id",), "chat"),
        (("call_id", "recording_id"), "sales_call"),
        (("meeting_id",), "meeting"),
        (("deal_id", "opportunity_id"), "crm_deal"),
        (("note_id", "activity_id"), "crm_note"),
        (("renewal_id",), "renewal"),
        (("contract_id",), "contract"),
        (("subscription_id",), "subscription"),
        (("complaint", "complaint_id", "complaint_narrative", "consumer_complaint_narrative"), "complaint"),
        (("ticket_id", "ticket_number", "request_id"), "support_ticket"),
        (("case_id", "case_number"), "case"),
        (("conversation_id", "conversation_number"), "conversation"),
        (("nps_score", "nps"), "nps_response"),
        (("csat_score", "csat"), "csat_response"),
        (("survey_id", "response_id"), "survey_response"),
    )


def test_source_type_precedence_table_controls_ambiguous_rows() -> None:
    row = {
        keys[0]: f"{source_type}-value"
        for keys, source_type in adapters._SOURCE_TYPE_PRECEDENCE
    }
    row["summary"] = "Ambiguous source row with every source-type key."

    opportunity, warnings = source_row_to_campaign_opportunity(row)

    assert warnings == ()
    assert opportunity["source_type"] == "review"
    assert opportunity["target_id"] == "chat-value"


def test_message_id_does_not_relabel_support_ticket_as_chat() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "request_id": "ticket-1",
        "message_id": "message-1",
        "message": "The attribution export is not working.",
    })

    assert warnings == ()
    assert opportunity["source_type"] == "support_ticket"
    assert opportunity["target_id"] == "ticket-1"


def test_source_field_lookup_preserves_exact_key_precedence() -> None:
    lookup = adapters._SourceFieldLookup({
        "ticket_id": "exact-ticket",
        "Ticket ID": "provider-ticket",
    })

    assert adapters._field_value(lookup, "ticket_id") == "exact-ticket"


def test_source_field_lookup_caches_provider_style_aliases() -> None:
    lookup = adapters._SourceFieldLookup({
        "Ticket ID": "ticket-1",
        "Vendor Name": "HubSpot",
    })

    assert adapters._field_value(lookup, "ticket_id") == "ticket-1"
    # Private cache assertion is intentional: this slice exists to avoid repeat scans.
    assert lookup._cache["ticket_id"] == "ticket-1"
    assert adapters._field_value(lookup, "ticket_id") == "ticket-1"
    assert adapters._field_value(lookup, "vendor_name") == "HubSpot"


def test_source_field_lookup_reuses_cached_alias_results(monkeypatch) -> None:
    calls = 0
    original = adapters._normalized_field_key

    def spy(key: str) -> str:
        nonlocal calls
        calls += 1
        return original(key)

    monkeypatch.setattr(adapters, "_normalized_field_key", spy)
    lookup = adapters._SourceFieldLookup({
        "Ticket ID": "ticket-1",
        "Vendor Name": "HubSpot",
    })

    for _ in range(5):
        assert adapters._field_value(lookup, "ticket_id") == "ticket-1"
    assert adapters._field_value(lookup, "vendor_name") == "HubSpot"
    assert adapters._field_value(lookup, "missing_key") is None
    assert adapters._field_value(lookup, "missing_key") is None
    assert calls == 5


def test_source_material_to_source_rows_expands_bundle_keys() -> None:
    rows = source_material_to_source_rows({
        "support_tickets": [],
        "sales_objections": [{
            "objection_id": "obj-1",
            "objection_text": "Exports are blocked.",
        }],
        "opportunities": [{
            "source_id": "opp-1",
            "text": "Opportunity row stays in the bundle.",
        }],
    })

    assert rows == [
        {
            "source_id": "opp-1",
            "text": "Opportunity row stays in the bundle.",
        },
        {
            "objection_id": "obj-1",
            "objection_text": "Exports are blocked.",
        },
    ]


def test_source_row_maps_review_text_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "review_id": "review-1",
        "reviewer_company": "Acme Logistics",
        "vendor": "HubSpot",
        "review_text": "Pricing became hard to justify after renewal.",
        "pain_category": "pricing pressure",
        "contact_email": "buyer@example.com",
    })

    assert warnings == ()
    assert opportunity["id"] == "review-1"
    assert opportunity["target_id"] == "review-1"
    assert opportunity["source_id"] == "review-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["vendor"] == "HubSpot"
    assert opportunity["pain_points"] == ["pricing pressure"]
    assert opportunity["evidence"] == [{
        "text": "Pricing became hard to justify after renewal.",
        "source_id": "review-1",
        "source_type": "review",
    }]


def test_source_rows_normalize_and_warn_for_missing_text() -> None:
    loaded = source_rows_to_campaign_opportunities(
        [
            {
                "id": "call-1",
                "company": "Acme",
                "vendor": "HubSpot",
                "transcript": "We are reviewing Salesforce before renewal.",
                "pain_points": ["renewal pressure"],
            },
            {"id": "empty-1", "company": "Beta", "vendor": "Zendesk"},
        ],
        target_mode="vendor_retention",
    )

    assert len(loaded.opportunities) == 1
    first = loaded.opportunities[0]
    assert first["target_id"] == "call-1"
    assert first["target_mode"] == "vendor_retention"
    assert first["evidence"][0]["source_type"] == "transcript"
    assert "missing_source_text" in [warning.code for warning in loaded.warnings]


def test_load_source_rows_from_semicolon_csv_detects_support_ticket_export(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets.csv"
    path.write_text(
        "\n".join([
            "ticket_id;subject;message",
            "ticket-1;Export help;How do I export attribution reports?",
        ]),
        encoding="utf-8",
    )

    rows = load_source_rows_from_file(path, file_format="csv")

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": "How do I export attribution reports?",
    }]


def test_load_source_rows_streams_csv_without_whole_file_byte_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "support_tickets.csv"
    path.write_text(
        "\n".join([
            "ticket_id,subject,message",
            "ticket-1,Export help,How do I export attribution reports?",
            "ticket-2,Billing help,Where can I download invoices?",
        ]),
        encoding="utf-8",
    )

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"unexpected whole-file byte read for {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    rows = load_source_rows_from_file(path, file_format="csv")

    assert rows == [
        {
            "ticket_id": "ticket-1",
            "subject": "Export help",
            "message": "How do I export attribution reports?",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Billing help",
            "message": "Where can I download invoices?",
        },
    ]


def test_load_source_rows_finds_header_after_large_provider_preamble(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_large_preamble.csv"
    path.write_text(
        "\n".join([
            "Zendesk export metadata " + ("x" * 70_000),
            "ticket_id;subject;message",
            "ticket-1;Export help;How do I export attribution reports?",
            "ticket-2;Billing help;Where can I download invoices?",
        ]),
        encoding="utf-8",
    )

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert rows == [
        {
            "ticket_id": "ticket-1",
            "subject": "Export help",
            "message": "How do I export attribution reports?",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Billing help",
            "message": "Where can I download invoices?",
        },
    ]
    assert [warning.code for warning in warnings] == ["csv_leading_rows_skipped"]
    assert warnings[0].row_index == 1
    assert "CSV header on row 2" in warnings[0].message


def test_load_csv_source_rows_result_reports_counts_and_preserves_warnings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_capped.csv"
    path.write_text(
        "\n".join([
            "Zendesk export metadata",
            "ticket_id,subject,message",
            "ticket-1,Export help,How do I export reports?",
            "ticket-2,Billing help,Where can I download invoices?",
            "ticket-3,Password help,How do I reset my password?",
        ]),
        encoding="utf-8",
    )

    result = load_csv_source_rows_result_from_file(path, max_rows=2)

    assert result.source_row_count == 3
    assert result.included_row_count == 2
    assert result.truncated_row_count == 1
    assert [row["ticket_id"] for row in result.rows] == ["ticket-1", "ticket-2"]
    assert [warning.code for warning in result.warnings] == ["csv_leading_rows_skipped"]


def test_load_source_rows_with_warnings_keeps_tuple_contract_after_result_api(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets.csv"
    path.write_text(
        "\n".join([
            "ticket_id,subject,message",
            "ticket-1,Export help,How do I export reports?",
        ]),
        encoding="utf-8",
    )

    rows, warnings = load_source_rows_with_warnings_from_file(path, file_format="csv")

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": "How do I export reports?",
    }]
    assert warnings == ()


def test_load_source_rows_keeps_comma_delimiter_when_body_has_tabs_semicolons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets.csv"
    path.write_text(
        "\n".join([
            "ticket_id,subject,message",
            (
                'ticket-1,Export help,"Clicked reports; exports; analytics\t'
                'but the download still fails"'
            ),
            (
                'ticket-2,Billing help,"Invoice tab\tshows paid; portal says due"'
            ),
        ]),
        encoding="utf-8",
    )

    rows = load_source_rows_from_file(path, file_format="csv")

    assert rows == [
        {
            "ticket_id": "ticket-1",
            "subject": "Export help",
            "message": (
                "Clicked reports; exports; analytics\t"
                "but the download still fails"
            ),
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Billing help",
            "message": "Invoice tab\tshows paid; portal says due",
        },
    ]


def test_load_source_rows_rejects_inconsistent_mixed_delimiters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_mixed_delimiters.csv"
    path.write_text(
        "\n".join([
            "ticket_id,subject,message",
            "ticket-1,Export help,How do I export reports?",
            "ticket-2;Billing help;Invoice portal is wrong",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inconsistent column counts"):
        load_source_rows_from_file(path, file_format="csv")


def test_load_source_rows_structures_missing_header_parse_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_header.csv"
    path.write_text(
        "\n".join([
            "I need help exporting reports",
            "Where can I download invoices?",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(CsvCustomerDataParseError) as exc:
        load_source_rows_from_file(path, file_format="csv")

    assert exc.value.code == "csv_missing_header"
    assert "header row" in exc.value.message
    assert "ticket_id" in exc.value.how_to_fix


def test_load_source_rows_structures_encoding_parse_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nul_heavy.csv"
    path.write_bytes(b"\x00" * 128)

    with pytest.raises(CsvCustomerDataParseError) as exc:
        load_source_rows_from_file(path, file_format="csv")

    assert exc.value.code == "csv_encoding_error"
    assert "NUL-heavy" in exc.value.message
    assert "UTF-8 or UTF-16" in exc.value.how_to_fix


@pytest.mark.parametrize(
    ("payload", "encoding_name"),
    [
        (b"\xff\xfeticket_id\x00\xff", "utf-16"),
        (b"\x00\x00\xfe\xffnot-a-valid-utf32", "utf-32"),
    ],
)
def test_load_source_rows_structures_malformed_bom_encoding_errors(
    tmp_path: Path,
    payload: bytes,
    encoding_name: str,
) -> None:
    path = tmp_path / "malformed_bom.csv"
    path.write_bytes(payload)

    with pytest.raises(CsvCustomerDataParseError) as exc:
        load_source_rows_from_file(path, file_format="csv")

    assert exc.value.code == "csv_encoding_error"
    assert encoding_name in exc.value.message
    assert "UTF-8 or UTF-16" in exc.value.how_to_fix


def test_load_source_rows_structures_inconsistent_columns_without_raw_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_extra_cell.csv"
    path.write_text(
        "\n".join([
            "ticket_id,subject,message",
            "ticket-1,Export help,How do I export reports?",
            "ticket-2,Billing help,Invoice portal is wrong,lead@example.com",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(CsvCustomerDataParseError) as exc:
        load_source_rows_from_file(path, file_format="csv")

    detail = exc.value.as_dict()
    assert detail["code"] == "csv_inconsistent_columns"
    assert detail["row_index"] == 3
    assert "more cells than the header" in detail["message"]
    assert "lead@example.com" not in json.dumps(detail)


def test_load_source_rows_rejects_collapsed_mixed_delimiter_at_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_mixed_delimiters.csv"
    lines = ["ticket_id,subject,message"]
    lines.extend(
        f"ticket-{index},Export help,How do I export report {index}?"
        for index in range(1, 10)
    )
    lines.append("ticket-bad;Billing help;Invoice portal is wrong for lead@example.com")
    path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(CsvCustomerDataParseError) as exc:
        load_source_rows_from_file(path, file_format="csv")

    detail = exc.value.as_dict()
    assert detail["code"] == "csv_inconsistent_columns"
    assert detail["row_index"] == 11
    assert "first collapsed mixed-delimiter row 11" in detail["message"]
    assert "lead@example.com" not in json.dumps(detail)


def test_load_source_rows_allows_single_column_message_with_delimiters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_ticket_messages.csv"
    path.write_text(
        "\n".join([
            "message",
            "I tried A; then B failed, and setup still loops",
        ]),
        encoding="utf-8",
    )

    rows = load_source_rows_from_file(path, file_format="csv")

    assert rows == [{
        "message": "I tried A; then B failed, and setup still loops",
    }]


def test_load_source_rows_preserves_single_quote_quoted_commas(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_single_quote.csv"
    path.write_text(
        "\n".join([
            "ticket_id,subject,message",
            "ticket-1,Export help,'How do I export, then download?'",
            "ticket-2,Billing help,'Invoice portal says paid, then due'",
        ]),
        encoding="utf-8",
    )

    rows = load_source_rows_from_file(path, file_format="csv")

    assert rows == [
        {
            "ticket_id": "ticket-1",
            "subject": "Export help",
            "message": "How do I export, then download?",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Billing help",
            "message": "Invoice portal says paid, then due",
        },
    ]


def test_load_source_rows_preserves_quoted_commas_and_multiline_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets.csv"
    path.write_text(
        "\n".join([
            "ticket_id,subject,message",
            'ticket-1,Export help,"How do I export, then download',
            'the attribution report?"',
        ]),
        encoding="utf-8",
    )

    rows = load_source_rows_from_file(path, file_format="csv")

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": "How do I export, then download\nthe attribution report?",
    }]


def test_load_source_rows_from_utf16_bom_csv_decodes_support_ticket_export(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_utf16.csv"
    path.write_bytes(
        (
            "ticket_id,subject,message\n"
            "ticket-1,Export help,How do I export attribution reports?\n"
        ).encode("utf-16")
    )

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert warnings == ()
    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": "How do I export attribution reports?",
    }]


@pytest.mark.parametrize(
    ("header", "body"),
    (
        ("Ticket Comments", "How do I see the duplicate charge refund status?"),
        ("Comment Body", "Can I export all invoices before month-end?"),
        ("Public Comments", "Where do I update the billing contact?"),
        ("Ticket History", "How do I reset MFA for a locked-out admin?"),
        ("History", "Can I cancel before the annual renewal?"),
        ("Conversation", "Where can I download last quarter's receipt?"),
    ),
)
def test_source_rows_admit_zendesk_public_comment_history_scalar_aliases(
    tmp_path: Path,
    header: str,
    body: str,
) -> None:
    path = tmp_path / "zendesk_public_comment_aliases.csv"
    path.write_text(
        "\n".join([
            f"Ticket ID,Subject,Description,{header},Internal Notes",
            f"zd-1,Support question,,{body},Do not publish this workaround",
        ]),
        encoding="utf-8",
    )

    rows = load_source_rows_from_file(path, file_format="csv")
    loaded = source_rows_to_campaign_opportunities(rows)

    assert len(loaded.opportunities) == 1
    opportunity = loaded.opportunities[0]
    assert "missing_source_text" not in [warning.code for warning in loaded.warnings]
    assert opportunity["source_type"] == "support_ticket"
    assert opportunity["target_id"] == "zd-1"
    assert body in opportunity["evidence"][0]["text"]
    assert "Do not publish" not in opportunity["evidence"][0]["text"]
    assert "Internal Notes" not in opportunity


@pytest.mark.parametrize("history_key", ("comments", "ticket_history"))
def test_source_rows_skip_private_comment_objects_in_public_history_aliases(
    history_key: str,
) -> None:
    loaded = source_rows_to_campaign_opportunities([{
        "ticket_id": "zd-1",
        "subject": "Refund status",
        history_key: [
            {"body": "How do I see the duplicate charge refund?", "public": True},
            {"body": "INTERNAL refund quietly; do not publish", "public": False},
            {"body": "Where can I download the refund receipt?"},
        ],
    }])

    assert len(loaded.opportunities) == 1
    opportunity = loaded.opportunities[0]
    text = opportunity["evidence"][0]["text"]
    assert "How do I see the duplicate charge refund?" in text
    assert "Where can I download the refund receipt?" in text
    assert "INTERNAL refund quietly" not in text
    preserved_payload = json.dumps(dict(opportunity), default=str)
    assert "INTERNAL refund quietly" not in preserved_payload
    assert len(opportunity[history_key]) == 2
    assert all(item.get("public") is not False for item in opportunity[history_key])
    assert "missing_source_text" not in [warning.code for warning in loaded.warnings]


@pytest.mark.parametrize("header", ("Internal Notes", "Private Notes"))
def test_source_rows_do_not_admit_private_note_aliases_as_customer_text(
    tmp_path: Path,
    header: str,
) -> None:
    path = tmp_path / "zendesk_private_notes_only.csv"
    path.write_text(
        "\n".join([
            f"Ticket ID,Subject,Description,{header}",
            "zd-1,Support question,,Do not publish this private note",
        ]),
        encoding="utf-8",
    )

    rows = load_source_rows_from_file(path, file_format="csv")
    loaded = source_rows_to_campaign_opportunities(rows)

    assert loaded.opportunities == ()
    assert "missing_source_text" in [warning.code for warning in loaded.warnings]


@pytest.mark.parametrize("encoding", ("utf-16-le", "utf-16-be"))
def test_load_source_rows_infers_bomless_utf16_after_utf8_failure(
    tmp_path: Path,
    encoding: str,
) -> None:
    path = tmp_path / f"support_tickets_{encoding}.csv"
    message = "Le caf\u00e9 export failed."
    path.write_bytes((
        "ticket_id,subject,message\n"
        f"ticket-1,Export help,{message}\n"
    ).encode(encoding))

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": message,
    }]
    assert [warning.code for warning in warnings] == ["csv_encoding_inferred"]
    assert encoding in warnings[0].message


@pytest.mark.parametrize(
    ("encoding", "message"),
    (
        ("cp1252", "Customer can\u2019t export reports."),
        ("latin-1", "Le caf\u00e9 export failed."),
    ),
)
def test_load_source_rows_from_legacy_csv_decodes_clean_text_without_warning(
    tmp_path: Path,
    encoding: str,
    message: str,
) -> None:
    path = tmp_path / f"support_tickets_{encoding}.csv"
    path.write_bytes((
        "ticket_id,subject,message\n"
        f"ticket-1,Export help,{message}\n"
    ).encode(encoding))

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert warnings == ()
    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": message,
    }]


@pytest.mark.parametrize(
    "message",
    (
        "\u00c3lvaro can\u2019t export reports.",
        "\u00c2ngela can\u2019t export reports.",
    ),
)
def test_load_source_rows_prefers_clean_cp1252_marker_text_over_utf8_recovery(
    tmp_path: Path,
    message: str,
) -> None:
    path = tmp_path / "support_tickets_cp1252_marker.csv"
    path.write_bytes((
        "ticket_id,subject,message\n"
        f"ticket-1,Export help,{message}\n"
    ).encode("cp1252"))

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert warnings == ()
    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": message,
    }]
    assert "\ufffd" not in rows[0]["message"]


def test_load_source_rows_warns_on_high_replacement_character_ratio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_corrupt.csv"
    corrupt_text = "\ufffd" * 16
    path.write_bytes((
        "ticket_id,subject,message\n"
        f"ticket-1,Corrupt export,{corrupt_text}\n"
    ).encode("utf-8"))

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Corrupt export",
        "message": corrupt_text,
    }]
    assert [warning.code for warning in warnings] == ["csv_replacement_characters"]
    assert warnings[0].field == "encoding"
    assert "Unicode replacement character" in warnings[0].message


def test_load_source_rows_prefers_warned_utf8_recovery_over_cp1252_mojibake(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_truncated_utf8.csv"
    message = (
        "Caf\u00e9 export failed \u2014 cannot save "
        + "the monthly attribution report " * 20
    )
    path.write_bytes(
        (
            "ticket_id,subject,message\n"
            f"ticket-1,Export help,{message}"
        ).encode("utf-8")
        + b"\xff\n"
    )

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": f"{message}\ufffd",
    }]
    assert [warning.code for warning in warnings] == ["csv_replacement_characters"]
    assert "CafÃ" not in rows[0]["message"]
    assert "â" not in rows[0]["message"]


@pytest.mark.parametrize("corrupt_byte", (0x81, 0x8D, 0x8F, 0x90, 0x9D, 0xFE, 0xFF))
@pytest.mark.parametrize("suffix", (b"\n", b"cd\n"))
def test_load_source_rows_warns_on_implausible_legacy_fallback_artifacts(
    tmp_path: Path,
    corrupt_byte: int,
    suffix: bytes,
) -> None:
    path = tmp_path / "support_tickets_ascii_corrupt_utf8.csv"
    message = "Cannot export the monthly attribution report at all today"
    path.write_bytes(
        (
            "ticket_id,subject,message\n"
            f"ticket-1,Export help,{message}"
        ).encode("utf-8")
        + bytes([corrupt_byte])
        + suffix
    )

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )
    try:
        legacy_char = bytes([corrupt_byte]).decode("cp1252")
    except UnicodeDecodeError:
        legacy_char = bytes([corrupt_byte]).decode("latin-1")

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": f"{message}{legacy_char}{suffix.decode('ascii').strip()}",
    }]
    assert [warning.code for warning in warnings] == ["csv_encoding_ambiguous"]
    assert warnings[0].field == "encoding"
    assert "failed strict UTF-8" in warnings[0].message


def test_load_source_rows_keeps_legacy_fallback_warning_position_correlated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "support_tickets_legacy_warning_alignment.csv"
    path.write_bytes(b"ticket_id,subject,message\nA,B,\xc4\x9dqqqq\xc1\n")

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert rows == [{
        "ticket_id": "A",
        "subject": "B",
        "message": "\xc4\x9dqqqq\xc1",
    }]
    assert warnings == ()


@pytest.mark.parametrize(
    ("prefix", "corrupt_byte", "suffix"),
    (
        ("oops", 0x81, "cd"),
        ("aa", 0x90, "bb"),
    ),
)
def test_load_source_rows_warns_on_mid_field_implausible_fallback_artifacts(
    tmp_path: Path,
    prefix: str,
    corrupt_byte: int,
    suffix: str,
) -> None:
    path = tmp_path / "support_tickets_mid_field_corrupt_utf8.csv"
    path.write_bytes(
        b"ticket_id,subject,message\n"
        + f"ticket-1,Export help,{prefix}".encode("utf-8")
        + bytes([corrupt_byte])
        + f"{suffix}\n".encode("utf-8")
    )

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )
    legacy_char = bytes([corrupt_byte]).decode("latin-1")

    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": f"{prefix}{legacy_char}{suffix}",
    }]
    assert [warning.code for warning in warnings] == ["csv_encoding_ambiguous"]


@pytest.mark.parametrize(
    "message",
    (
        "Le caf\u00e9",
        "5\u20ac",
        "(c)\u00a9",
        "caf\u00e9,done",
        "Se\u00f1or \u00fcber",
    ),
)
def test_load_source_rows_does_not_warn_on_clean_cp1252_field_endings(
    tmp_path: Path,
    message: str,
) -> None:
    path = tmp_path / "support_tickets_cp1252_clean_field_endings.csv"
    path.write_bytes((
        "ticket_id,subject,message\n"
        f'ticket-1,Export help,"{message}"\n'
    ).encode("cp1252"))

    rows, warnings = load_source_rows_with_warnings_from_file(
        path,
        file_format="csv",
    )

    assert warnings == ()
    assert rows == [{
        "ticket_id": "ticket-1",
        "subject": "Export help",
        "message": message,
    }]


def test_source_row_maps_provider_complaint_narrative_aliases() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "Complaint ID": "3182593",
        "Consumer complaint narrative": "I received a collection letter for a debt I do not owe.",
        "Issue": "Attempts to collect debt not owed",
        "Product": "Debt collection",
        "Company": "Example Collector",
        "Date received": "03/01/2019",
    })

    assert warnings == ()
    assert opportunity["id"] == "3182593"
    assert opportunity["target_id"] == "3182593"
    assert opportunity["source_id"] == "3182593"
    assert opportunity["source_type"] == "complaint"
    assert opportunity["company_name"] == "Example Collector"
    assert opportunity["pain_points"] == ["Attempts to collect debt not owed"]
    assert opportunity["evidence"] == [{
        "text": "I received a collection letter for a debt I do not owe.",
        "source_id": "3182593",
        "source_type": "complaint",
    }]


def test_source_row_maps_search_query_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "Query ID": "query-1",
        "Search Query": "export attribution report",
        "Results Count": "0",
        "Zero Results": "true",
        "Page URL": "https://example.com/help",
        "Topic": "Analytics",
    })

    assert warnings == ()
    assert opportunity["id"] == "query-1"
    assert opportunity["target_id"] == "query-1"
    assert opportunity["source_id"] == "query-1"
    assert opportunity["source_type"] == "search_log"
    assert opportunity["Results Count"] == "0"
    assert opportunity["Zero Results"] == "true"
    assert opportunity["Page URL"] == "https://example.com/help"
    assert opportunity["pain_points"] == ["Analytics"]
    assert opportunity["evidence"] == [{
        "text": "export attribution report",
        "source_id": "query-1",
        "source_type": "search_log",
    }]


def test_source_bundle_extracts_zero_result_search_logs(tmp_path: Path) -> None:
    source = tmp_path / "search_bundle.json"
    source.write_text(
        json.dumps({
            "account_name": "Acme Logistics",
            "search_logs": [
                {
                    "zero_result_query_id": "zero-1",
                    "zero_result_query": "change billing email",
                    "results_count": 0,
                    "zero_results": True,
                },
                {
                    "query_id": "search-2",
                    "query": "download invoice",
                    "results_count": 4,
                },
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(source, file_format="json")

    assert [item["source_id"] for item in loaded.opportunities] == ["zero-1", "search-2"]
    assert [item["source_type"] for item in loaded.opportunities] == ["search_log", "search_log"]
    assert [item["company_name"] for item in loaded.opportunities] == [
        "Acme Logistics",
        "Acme Logistics",
    ]
    assert loaded.opportunities[0]["evidence"][0]["text"] == "change billing email"
    assert loaded.opportunities[0]["zero_results"] is True


def test_source_rows_apply_default_fields_without_overriding_row_values() -> None:
    loaded = source_rows_to_campaign_opportunities(
        [
            {
                "id": "review-1",
                "vendor": "HubSpot",
                "review_text": "Pricing is a problem.",
            },
            {
                "id": "review-2",
                "company_name": "",
                "contact_email": None,
                "vendor": "HubSpot",
                "review_text": "Reporting exports are blocked.",
            },
            {
                "id": "review-3",
                "company_name": "Row Company",
                "contact_email": "row@example.com",
                "vendor": "HubSpot",
                "review_text": "Notifications are noisy.",
            },
            {
                "id": "review-4",
                "company": "Alias Company",
                "email": "alias@example.com",
                "vendor": "HubSpot",
                "review_text": "Admin controls are confusing.",
            },
        ],
        default_fields={
            "company_name": "Acme Logistics",
            "contact_email": "ops@example.com",
        },
    )

    assert loaded.warnings == ()
    assert loaded.opportunities[0]["company_name"] == "Acme Logistics"
    assert loaded.opportunities[0]["contact_email"] == "ops@example.com"
    assert loaded.opportunities[1]["company_name"] == "Acme Logistics"
    assert loaded.opportunities[1]["contact_email"] == "ops@example.com"
    assert loaded.opportunities[2]["company_name"] == "Row Company"
    assert loaded.opportunities[2]["contact_email"] == "row@example.com"
    assert loaded.opportunities[3]["company_name"] == "Alias Company"
    assert loaded.opportunities[3]["contact_email"] == "alias@example.com"


def test_source_rows_row_aliases_override_default_aliases() -> None:
    loaded = source_rows_to_campaign_opportunities(
        [
            {
                "id": "review-1",
                "account_name": "Row Company",
                "recipient_email": "row@example.com",
                "vendor": "HubSpot",
                "review_text": "The renewal flow is hard to manage.",
            },
        ],
        default_fields={
            "company": "Default Company",
            "email": "default@example.com",
        },
    )

    assert loaded.warnings == ()
    assert loaded.opportunities[0]["company_name"] == "Row Company"
    assert loaded.opportunities[0]["contact_email"] == "row@example.com"


def test_parse_default_fields_rejects_invalid_values() -> None:
    assert parse_default_fields(["company_name=Acme", "contact_email=ops@example.com"]) == {
        "company_name": "Acme",
        "contact_email": "ops@example.com",
    }

    with pytest.raises(ValueError, match="key=value"):
        parse_default_fields(["company_name"])


def test_parse_default_fields_or_exit_raises_clean_system_exit() -> None:
    with pytest.raises(SystemExit, match="key=value"):
        parse_default_fields_or_exit(["company_name"])


def test_parse_default_fields_with_booking_url_nests_selling_context() -> None:
    defaults = parse_default_fields_with_booking_url_or_exit(
        ["company_name=Acme Logistics"],
        booking_url=" https://app.example.test/book ",
    )

    assert defaults == {
        "company_name": "Acme Logistics",
        "selling": {"booking_url": "https://app.example.test/book"},
    }


def test_parse_default_fields_with_booking_url_ignores_blank_url() -> None:
    defaults = parse_default_fields_with_booking_url_or_exit(
        ["company_name=Acme Logistics"],
        booking_url="   ",
    )

    assert defaults == {"company_name": "Acme Logistics"}


def test_source_document_title_does_not_become_buyer_fields() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "id": "doc-1",
        "name": "Q1 churn transcript",
        "title": "Discovery call notes",
        "vendor": "HubSpot",
        "text": "The account is comparing HubSpot with Salesforce.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "doc-1"
    assert opportunity["source_title"] == "Discovery call notes"
    assert opportunity["evidence"][0]["source_title"] == "Discovery call notes"
    assert "company_name" not in opportunity
    assert "contact_name" not in opportunity
    assert "contact_title" not in opportunity
    assert "name" not in opportunity
    assert "title" not in opportunity


def test_source_row_maps_support_ticket_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "subject": "Renewal pricing escalation",
        "message": "The renewal quote jumped and support has not explained the new tier.",
        "pain_category": "pricing pressure",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "ticket-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["vendor"] == "HubSpot"
    assert opportunity["source_title"] == "Renewal pricing escalation"
    assert "subject" not in opportunity
    assert opportunity["evidence"] == [{
        "text": "The renewal quote jumped and support has not explained the new tier.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Renewal pricing escalation",
    }]


def test_source_row_accepts_provider_style_ticket_field_labels() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "Ticket ID": "ticket-1",
        "Account Name": "Acme Logistics",
        "Vendor Name": "HubSpot",
        "Subject": "Renewal pricing escalation",
        "Description": "The renewal quote jumped before the team could export reports.",
        "Pain Category": "pricing pressure",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "ticket-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["vendor_name"] == "HubSpot"
    assert opportunity["source_title"] == "Renewal pricing escalation"
    assert opportunity["pain_points"] == ["pricing pressure"]
    assert opportunity["evidence"] == [{
        "text": "The renewal quote jumped before the team could export reports.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Renewal pricing escalation",
    }]


def test_source_row_accepts_common_helpdesk_ticket_aliases() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "Ticket Number": "ZD-1001",
        "Organization Name": "Acme Logistics",
        "Vendor Name": "Zendesk",
        "Requester Email": "ops@example.com",
        "Requester Name": "Jordan Lee",
        "Requester Title": "VP Revenue Operations",
        "Ticket Title": "Attribution export blocked",
        "Issue Description": (
            "The support team cannot export campaign attribution data before renewal."
        ),
        "Pain Category": "reporting limits",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "ZD-1001"
    assert opportunity["source_type"] == "support_ticket"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["vendor_name"] == "Zendesk"
    assert opportunity["contact_email"] == "ops@example.com"
    assert opportunity["contact_name"] == "Jordan Lee"
    assert opportunity["contact_title"] == "VP Revenue Operations"
    assert opportunity["source_title"] == "Attribution export blocked"
    assert "Ticket Title" not in opportunity
    assert opportunity["pain_points"] == ["reporting limits"]
    assert opportunity["evidence"] == [{
        "text": (
            "The support team cannot export campaign attribution data before renewal."
        ),
        "source_id": "ZD-1001",
        "source_type": "support_ticket",
        "source_title": "Attribution export blocked",
    }]


def test_source_row_accepts_case_aliases_and_latest_comment_text() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "Case Number": "CASE-42",
        "Customer Company": "Northstar Finance",
        "Customer Email": "admin@northstar.example",
        "Customer Contact Name": "Maya Chen",
        "Customer Title": "Director of Support",
        "Current Vendor": "Intercom",
        "Case Subject": "SLA export delays",
        "Latest Comment": "The team cannot prove SLA history before renewal.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "CASE-42"
    assert opportunity["source_type"] == "case"
    assert opportunity["company_name"] == "Northstar Finance"
    assert opportunity["vendor_name"] == "Intercom"
    assert opportunity["contact_email"] == "admin@northstar.example"
    assert opportunity["contact_name"] == "Maya Chen"
    assert opportunity["contact_title"] == "Director of Support"
    assert opportunity["source_title"] == "SLA export delays"
    assert "Case Subject" not in opportunity
    assert opportunity["evidence"] == [{
        "text": "The team cannot prove SLA history before renewal.",
        "source_id": "CASE-42",
        "source_type": "case",
        "source_title": "SLA export delays",
    }]


def test_source_row_prefers_body_over_ticket_message() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "body": "Use this longer exported body as the evidence text.",
        "message": "Do not use this short ticket message when body exists.",
    })

    assert warnings == ()
    assert opportunity["evidence"][0]["text"] == (
        "Use this longer exported body as the evidence text."
    )


def test_source_row_maps_ticket_comments_to_evidence_text() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-thread-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "subject": "Reporting export blocked",
        "comments": [
            {
                "author": "Customer",
                "body": "We cannot export attribution data before renewal.",
            },
            {
                "role": "Agent",
                "message": "The export requires a higher plan.",
            },
        ],
    })

    assert warnings == ()
    assert opportunity["target_id"] == "ticket-thread-1"
    assert opportunity["evidence"][0] == {
        "text": (
            "Customer: We cannot export attribution data before renewal.\n"
            "Agent: The export requires a higher plan."
        ),
        "source_id": "ticket-thread-1",
        "source_type": "support_ticket",
        "source_title": "Reporting export blocked",
    }


def test_source_row_maps_call_transcript_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "call_id": "call-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "title": "Renewal discovery call",
        "transcript": "Buyer: We are comparing Salesforce before the renewal.",
        "pain_category": "renewal pressure",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "call-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["source_title"] == "Renewal discovery call"
    assert opportunity["source_type"] == "transcript"
    assert opportunity["pain_points"] == ["renewal pressure"]
    assert opportunity["evidence"][0] == {
        "text": "Buyer: We are comparing Salesforce before the renewal.",
        "source_id": "call-1",
        "source_type": "transcript",
        "source_title": "Renewal discovery call",
    }


def test_source_row_maps_recording_notes_to_sales_call() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "recording_id": "rec-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "notes": "RevOps asked for attribution exports before renewal.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "rec-1"
    assert opportunity["source_type"] == "sales_call"
    assert opportunity["evidence"][0] == {
        "text": "RevOps asked for attribution exports before renewal.",
        "source_id": "rec-1",
        "source_type": "sales_call",
    }


def test_source_row_maps_call_speaker_turns_to_evidence_text() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "call_id": "call-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "turns": [
            {"speaker": "Buyer", "text": "Attribution exports are blocked."},
            {"speaker": "AE", "text": "That requires the next plan."},
        ],
    })

    assert warnings == ()
    assert opportunity["source_type"] == "sales_call"
    assert opportunity["evidence"][0] == {
        "text": (
            "Buyer: Attribution exports are blocked.\n"
            "AE: That requires the next plan."
        ),
        "source_id": "call-1",
        "source_type": "sales_call",
    }


def test_source_row_with_call_and_meeting_ids_prefers_sales_call() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "call_id": "call-1",
        "meeting_id": "meeting-1",
        "summary": "The buyer asked about Salesforce migration risk.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "call-1"
    assert opportunity["source_type"] == "sales_call"


def test_source_row_maps_crm_deal_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "deal_id": "deal-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "stage": "renewal_review",
        "summary": "The account is evaluating Salesforce before renewal.",
        "pain_category": "renewal pressure",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "deal-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["stage"] == "renewal_review"
    assert opportunity["source_type"] == "crm_deal"
    assert opportunity["pain_points"] == ["renewal pressure"]
    assert opportunity["evidence"][0] == {
        "text": "The account is evaluating Salesforce before renewal.",
        "source_id": "deal-1",
        "source_type": "crm_deal",
    }


def test_source_row_maps_crm_note_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "note_id": "note-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "notes": "Champion asked for pricing proof before the renewal call.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "note-1"
    assert opportunity["source_type"] == "crm_note"
    assert opportunity["evidence"][0] == {
        "text": "Champion asked for pricing proof before the renewal call.",
        "source_id": "note-1",
        "source_type": "crm_note",
    }


def test_source_row_prefers_review_type_over_crm_deal_id() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "deal_id": "deal-1",
        "review_text": "The reviewer called out reporting limits.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "deal-1"
    assert opportunity["source_type"] == "review"


def test_source_row_maps_contract_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "contract_id": "contract-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "contract_end_date": "2026-07-01",
        "summary": "The contract is up for renewal while Salesforce is in evaluation.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "contract-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["contract_end_date"] == "2026-07-01"
    assert opportunity["source_type"] == "contract"
    assert opportunity["evidence"][0] == {
        "text": "The contract is up for renewal while Salesforce is in evaluation.",
        "source_id": "contract-1",
        "source_type": "contract",
    }


def test_source_row_maps_renewal_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "renewal_id": "renewal-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "notes": "Renewal is blocked until attribution exports are resolved.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "renewal-1"
    assert opportunity["source_type"] == "renewal"
    assert opportunity["evidence"][0] == {
        "text": "Renewal is blocked until attribution exports are resolved.",
        "source_id": "renewal-1",
        "source_type": "renewal",
    }


def test_source_row_maps_subscription_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "subscription_id": "subscription-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "message": "The team reduced seats after pricing changed.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "subscription-1"
    assert opportunity["source_type"] == "subscription"
    assert opportunity["evidence"][0] == {
        "text": "The team reduced seats after pricing changed.",
        "source_id": "subscription-1",
        "source_type": "subscription",
    }


def test_source_row_prefers_review_type_over_contract_id() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "contract_id": "contract-1",
        "review_text": "The reviewer called out implementation delays.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "contract-1"
    assert opportunity["source_type"] == "review"


def test_source_row_prefers_renewal_over_contract_id() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "renewal_id": "renewal-1",
        "contract_id": "contract-1",
        "summary": "Renewal approval is blocked on contract pricing proof.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "renewal-1"
    assert opportunity["source_id"] == "renewal-1"
    assert opportunity["source_type"] == "renewal"
    assert opportunity["evidence"][0] == {
        "text": "Renewal approval is blocked on contract pricing proof.",
        "source_id": "renewal-1",
        "source_type": "renewal",
    }


def test_source_row_prefers_renewal_over_subscription_id() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "renewal_id": "renewal-1",
        "subscription_id": "subscription-1",
        "message": "Renewal is at risk after the subscription seat reduction.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "renewal-1"
    assert opportunity["source_type"] == "renewal"


def test_source_row_prefers_note_type_over_subscription_context() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "note_id": "note-1",
        "subscription_id": "subscription-1",
        "notes": "CSM logged a subscription downgrade note.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "note-1"
    assert opportunity["source_type"] == "crm_note"


def test_source_row_prefers_scalar_text_over_thread_messages() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-thread-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "message": "Use this exported ticket summary.",
        "comments": [{"body": "Do not use this comment when scalar text exists."}],
    })

    assert warnings == ()
    assert opportunity["evidence"][0]["text"] == "Use this exported ticket summary."


def test_source_row_maps_nps_feedback_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "response_id": "nps-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "nps_score": 3,
        "feedback_text": "Renewal pricing is hard to defend against Salesforce.",
        "pain_category": "pricing pressure",
    })

    assert warnings == ()
    assert opportunity["id"] == "nps-1"
    assert opportunity["target_id"] == "nps-1"
    assert opportunity["nps_score"] == 3
    assert opportunity["source_type"] == "nps_response"
    assert opportunity["pain_points"] == ["pricing pressure"]
    assert opportunity["evidence"] == [{
        "text": "Renewal pricing is hard to defend against Salesforce.",
        "source_id": "nps-1",
        "source_type": "nps_response",
    }]


def test_source_row_accepts_provider_style_survey_field_labels() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "Response ID": "survey-1",
        "Company": "Beta Retail",
        "Vendor": "Zendesk",
        "NPS Score": "4",
        "Open Ended Response": "Support takes too long and we are comparing Intercom.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "survey-1"
    assert opportunity["company_name"] == "Beta Retail"
    assert opportunity["vendor_name"] == "Zendesk"
    assert opportunity["nps_score"] == "4"
    assert opportunity["source_type"] == "nps_response"
    assert opportunity["evidence"] == [{
        "text": "Support takes too long and we are comparing Intercom.",
        "source_id": "survey-1",
        "source_type": "nps_response",
    }]


def test_source_row_prefers_nps_over_csat_when_both_scores_exist() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "response_id": "survey-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "nps_score": 5,
        "csat_score": 4,
        "feedback": "The team is still evaluating alternatives.",
    })

    assert warnings == ()
    assert opportunity["source_type"] == "nps_response"


def test_load_source_campaign_opportunities_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "id": "review-1",
                "company": "Acme",
                "vendor": "HubSpot",
                "review_text": "Pricing is a problem.",
            }),
            json.dumps({
                "id": "complaint-1",
                "company": "Beta",
                "vendor": "Zendesk",
                "complaint": "Support tickets are taking too long.",
            }),
        ]),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.source == str(path)
    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-1",
        "complaint-1",
    ]
    assert loaded.opportunities[1]["evidence"][0]["source_type"] == "complaint"


def test_load_source_campaign_opportunities_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text(
        "\n".join([
            "id,company,vendor,review_text,pain_category",
            "review-1,Acme,HubSpot,Pricing is a problem,pricing",
        ]),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "review-1"
    assert loaded.opportunities[0]["company_name"] == "Acme"
    assert loaded.opportunities[0]["pain_points"] == ["pricing"]
    assert loaded.opportunities[0]["evidence"] == [{
        "text": "Pricing is a problem",
        "source_id": "review-1",
        "source_type": "review",
    }]


def test_load_source_campaign_opportunities_applies_default_fields(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        json.dumps({
            "id": "g2-review-1",
            "vendor": "Slack",
            "review_text": "Search gets slow once message history grows.",
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(
        path,
        default_fields={
            "company_name": "Acme Logistics",
            "contact_email": "ops@example.com",
        },
    )

    assert loaded.warnings == ()
    assert loaded.opportunities[0]["company_name"] == "Acme Logistics"
    assert loaded.opportunities[0]["contact_email"] == "ops@example.com"


def test_packaged_source_rows_example_loads() -> None:
    loaded = load_source_campaign_opportunities_from_file(EXAMPLE_SOURCE_ROWS)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-acme-1",
        "transcript-northstar-1",
        "ticket-riverbend-1",
    ]
    assert loaded.opportunities[0]["evidence"][0]["source_type"] == "review"
    assert loaded.opportunities[1]["evidence"][0]["source_type"] == "transcript"
    assert loaded.opportunities[2]["evidence"][0]["source_type"] == "support_ticket"
    assert loaded.warnings == ()


def test_packaged_source_bundle_example_loads_all_collections() -> None:
    loaded = load_source_campaign_opportunities_from_file(EXAMPLE_SOURCE_BUNDLE)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "bundle-review-acme-1",
        "bundle-ticket-acme-1",
        "bundle-survey-acme-1",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Acme Logistics",
        "Acme Logistics",
        "Acme Logistics",
    ]
    assert [row["source_type"] for row in loaded.opportunities] == [
        "review",
        "support_ticket",
        "nps_response",
    ]
    assert loaded.warnings == ()


def test_packaged_support_ticket_csv_example_loads_provider_labels() -> None:
    loaded = load_source_campaign_opportunities_from_file(
        EXAMPLE_SUPPORT_TICKET_CSV,
        file_format="csv",
    )

    assert [row["target_id"] for row in loaded.opportunities] == [
        "ticket-acme-1",
        "ticket-acme-2",
        "ticket-northstar-1",
        "ticket-northstar-2",
    ]
    assert [row["source_type"] for row in loaded.opportunities] == [
        "support_ticket",
        "support_ticket",
        "support_ticket",
        "support_ticket",
    ]
    assert loaded.opportunities[0]["company_name"] == "Acme Logistics"
    assert loaded.opportunities[0]["vendor_name"] == "HubSpot"
    assert loaded.opportunities[0]["source_title"] == "Change login email"
    assert loaded.opportunities[0]["evidence"][0] == {
        "text": "How do I change my login email?",
        "source_id": "ticket-acme-1",
        "source_type": "support_ticket",
        "source_title": "Change login email",
    }
    assert loaded.warnings == ()


def test_packaged_support_ticket_bundle_inherits_account_metadata() -> None:
    loaded = load_source_campaign_opportunities_from_file(EXAMPLE_SUPPORT_TICKET_BUNDLE)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "support-riverbend-1",
        "support-riverbend-2",
        "support-riverbend-3",
        "support-riverbend-4",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Riverbend Supply",
        "Riverbend Supply",
        "Riverbend Supply",
        "Riverbend Supply",
    ]
    assert [row["source_type"] for row in loaded.opportunities] == [
        "support_ticket",
        "support_ticket",
        "support_ticket",
        "support_ticket",
    ]
    assert loaded.opportunities[1]["evidence"][0] == {
        "text": (
            "Customer: Every demo follow-up still has to be rebuilt by hand.\n"
            "Agent: The workflow automation feature is not available on the current plan."
        ),
        "source_id": "support-riverbend-2",
        "source_type": "support_ticket",
        "source_title": "Manual sequence cleanup after demos",
    }
    assert loaded.warnings == ()


def test_build_sources_cli_converts_packaged_support_ticket_csv(tmp_path: Path) -> None:
    output = tmp_path / "support_ticket_opportunities.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(EXAMPLE_SUPPORT_TICKET_CSV),
            "--format",
            "csv",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.stdout == ""
    assert [row["target_id"] for row in payload["opportunities"]] == [
        "ticket-acme-1",
        "ticket-acme-2",
        "ticket-northstar-1",
        "ticket-northstar-2",
    ]
    assert payload["opportunities"][0]["source_type"] == "support_ticket"


def test_load_source_campaign_opportunities_from_support_ticket_object(tmp_path: Path) -> None:
    path = tmp_path / "support_tickets.json"
    path.write_text(
        json.dumps({
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "company": "Acme",
                    "vendor": "HubSpot",
                    "subject": "Reporting export issue",
                    "description": "The team cannot export attribution data before renewal.",
                }
            ]
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "ticket-1"
    assert loaded.opportunities[0]["source_type"] == "support_ticket"
    assert loaded.opportunities[0]["source_title"] == "Reporting export issue"
    assert loaded.opportunities[0]["evidence"][0] == {
        "text": "The team cannot export attribution data before renewal.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Reporting export issue",
    }


def test_load_source_campaign_opportunities_from_provider_style_bundle_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provider_bundle.json"
    path.write_text(
        json.dumps({
            "Account Name": "Acme Logistics",
            "Vendor Name": "HubSpot",
            "Support Tickets": [
                {
                    "Ticket ID": "ticket-1",
                    "Subject": "Reporting export issue",
                    "Description": "Exports require a higher tier before renewal.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [warning.code for warning in loaded.warnings] == ["missing_contact_email"]
    assert loaded.opportunities[0]["target_id"] == "ticket-1"
    assert loaded.opportunities[0]["company_name"] == "Acme Logistics"
    assert loaded.opportunities[0]["vendor_name"] == "HubSpot"
    assert loaded.opportunities[0]["evidence"][0] == {
        "text": "Exports require a higher tier before renewal.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Reporting export issue",
    }


def test_load_source_campaign_opportunities_from_survey_object(tmp_path: Path) -> None:
    path = tmp_path / "survey_responses.json"
    path.write_text(
        json.dumps({
            "survey_responses": [
                {
                    "survey_id": "survey-1",
                    "company": "Beta Retail",
                    "vendor": "Zendesk",
                    "csat_score": "2",
                    "open_ended_response": (
                        "Support responses take too long and the team is comparing Intercom."
                    ),
                }
            ]
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "survey-1"
    assert loaded.opportunities[0]["csat_score"] == "2"
    assert loaded.opportunities[0]["source_type"] == "csat_response"
    assert loaded.opportunities[0]["evidence"][0] == {
        "text": "Support responses take too long and the team is comparing Intercom.",
        "source_id": "survey-1",
        "source_type": "csat_response",
    }


def test_load_source_campaign_opportunities_from_multi_collection_bundle(tmp_path: Path) -> None:
    path = tmp_path / "customer_bundle.json"
    path.write_text(
        json.dumps({
            "account_id": "acct-1",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "reviews": [
                {
                    "review_id": "review-1",
                    "review_text": "Renewal pricing is hard to defend.",
                }
            ],
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "subject": "Reporting exports blocked",
                    "description": "Exports require a higher tier before renewal.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-1",
        "ticket-1",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Acme Logistics",
        "Acme Logistics",
    ]
    assert [row["vendor"] for row in loaded.opportunities] == ["HubSpot", "HubSpot"]
    assert loaded.opportunities[0]["account_id"] == "acct-1"
    assert loaded.opportunities[0]["evidence"][0]["source_type"] == "review"
    assert loaded.opportunities[1]["evidence"][0] == {
        "text": "Exports require a higher tier before renewal.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Reporting exports blocked",
    }


def test_load_source_campaign_opportunities_from_meeting_bundle(tmp_path: Path) -> None:
    path = tmp_path / "meeting_bundle.json"
    path.write_text(
        json.dumps({
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "meetings": [
                {
                    "meeting_id": "meeting-1",
                    "subject": "Renewal checkpoint",
                    "summary": "The buying team asked for Salesforce migration risk.",
                }
            ],
            "call_transcripts": [
                {
                    "recording_id": "recording-1",
                    "transcript": "Buyer: Attribution exports are blocked.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "recording-1",
        "meeting-1",
    ]
    assert [row["source_type"] for row in loaded.opportunities] == [
        "transcript",
        "meeting",
    ]
    assert loaded.opportunities[0]["company_name"] == "Acme Logistics"
    assert loaded.opportunities[1]["evidence"][0] == {
        "text": "The buying team asked for Salesforce migration risk.",
        "source_id": "meeting-1",
        "source_type": "meeting",
        "source_title": "Renewal checkpoint",
    }


def test_load_source_campaign_opportunities_from_chat_and_objection_bundle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat_objection_bundle.json"
    path.write_text(
        json.dumps({
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "chats": [
                {
                    "chat_id": "chat-1",
                    "subject": "Dashboard export",
                    "messages": [
                        {
                            "speaker": "customer",
                            "message": "How do I export the attribution dashboard?",
                        },
                        {"speaker": "agent", "message": "I can send the export steps."},
                    ],
                }
            ],
            "sales_objections": [
                {
                    "objection_id": "obj-1",
                    "objection_text": "We cannot export attribution reports before renewal.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == ["chat-1", "obj-1"]
    assert [row["source_type"] for row in loaded.opportunities] == ["chat", "sales_objection"]
    assert loaded.opportunities[0]["evidence"][0]["text"] == (
        "customer: How do I export the attribution dashboard?\n"
        "agent: I can send the export steps."
    )
    assert loaded.opportunities[1]["evidence"][0] == {
        "text": "We cannot export attribution reports before renewal.",
        "source_id": "obj-1",
        "source_type": "sales_objection",
    }


def test_load_source_campaign_opportunities_from_crm_bundle(tmp_path: Path) -> None:
    path = tmp_path / "crm_bundle.json"
    path.write_text(
        json.dumps({
            "account_id": "acct-1",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "deals": [
                {
                    "deal_id": "deal-1",
                    "stage": "renewal_review",
                    "summary": "The buying committee is comparing Salesforce.",
                }
            ],
            "account_notes": [
                {
                    "note_id": "note-1",
                    "notes": "RevOps asked for proof on attribution exports.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "deal-1",
        "note-1",
    ]
    assert [row["source_type"] for row in loaded.opportunities] == [
        "crm_deal",
        "crm_note",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Acme Logistics",
        "Acme Logistics",
    ]
    assert loaded.opportunities[0]["account_id"] == "acct-1"
    assert loaded.opportunities[1]["evidence"][0] == {
        "text": "RevOps asked for proof on attribution exports.",
        "source_id": "note-1",
        "source_type": "crm_note",
    }


def test_load_source_campaign_opportunities_from_renewal_bundle(tmp_path: Path) -> None:
    path = tmp_path / "renewal_bundle.json"
    path.write_text(
        json.dumps({
            "account_id": "acct-1",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "contracts": [
                {
                    "contract_id": "contract-1",
                    "summary": "Contract renewal is due this quarter.",
                }
            ],
            "renewal_notes": [
                {
                    "renewal_id": "renewal-1",
                    "notes": "Procurement asked for proof before renewal approval.",
                }
            ],
            "subscriptions": [
                {
                    "subscription_id": "subscription-1",
                    "message": "Seat count dropped after the new pricing tier.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "contract-1",
        "renewal-1",
        "subscription-1",
    ]
    assert [row["source_type"] for row in loaded.opportunities] == [
        "contract",
        "renewal",
        "subscription",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Acme Logistics",
        "Acme Logistics",
        "Acme Logistics",
    ]
    assert loaded.opportunities[0]["account_id"] == "acct-1"
    assert loaded.opportunities[2]["evidence"][0] == {
        "text": "Seat count dropped after the new pricing tier.",
        "source_id": "subscription-1",
        "source_type": "subscription",
    }


def test_load_source_campaign_opportunities_from_nested_sources_bundle(tmp_path: Path) -> None:
    path = tmp_path / "nested_sources.json"
    path.write_text(
        json.dumps({
            "company": "Beta Retail",
            "vendor": "Zendesk",
            "sources": {
                "reviews": [
                    {
                        "id": "review-1",
                        "review_text": "The team is comparing Intercom.",
                    }
                ],
                "surveys": [
                    {
                        "response_id": "survey-1",
                        "csat_score": 2,
                        "open_ended_response": "Support takes too long.",
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-1",
        "survey-1",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Beta Retail",
        "Beta Retail",
    ]
    assert loaded.opportunities[0]["source_type"] == "review"
    assert loaded.opportunities[1]["source_type"] == "csat_response"


def test_load_source_campaign_opportunities_from_deep_nested_sources_bundle(tmp_path: Path) -> None:
    path = tmp_path / "deep_nested_sources.json"
    path.write_text(
        json.dumps({
            "company": "Beta Retail",
            "sources": {
                "sources": {
                    "reviews": [
                        {
                            "id": "review-1",
                            "review_text": "The team is comparing Intercom.",
                        }
                    ]
                }
            },
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "review-1"
    assert loaded.opportunities[0]["company_name"] == "Beta Retail"
    assert loaded.opportunities[0]["source_type"] == "review"


def test_source_bundle_nested_mapping_without_collection_loads_as_row(tmp_path: Path) -> None:
    path = tmp_path / "nested_single_source.json"
    path.write_text(
        json.dumps({
            "company": "Parent Co",
            "vendor": "HubSpot",
            "sources": {
                "id": "review-1",
                "review_text": "The account is comparing alternatives.",
            },
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "review-1"
    assert loaded.opportunities[0]["company_name"] == "Parent Co"
    assert loaded.opportunities[0]["vendor"] == "HubSpot"
    assert loaded.opportunities[0]["evidence"][0] == {
        "text": "The account is comparing alternatives.",
        "source_id": "review-1",
        "source_type": "review",
    }


def test_source_bundle_child_fields_override_parent_metadata(tmp_path: Path) -> None:
    path = tmp_path / "child_override.json"
    path.write_text(
        json.dumps({
            "company": "Parent Co",
            "vendor": "HubSpot",
            "reviews": [
                {
                    "id": "review-1",
                    "company": "Child Co",
                    "vendor": "Salesforce",
                    "review_text": "The team is switching platforms.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["company_name"] == "Child Co"
    assert loaded.opportunities[0]["vendor"] == "Salesforce"


def test_source_adapter_cli_outputs_generation_payload(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps({
            "reviews": [
                {
                    "id": "review-1",
                    "company": "Acme",
                    "vendor": "HubSpot",
                    "review_text": "Pricing is a problem.",
                }
            ]
        }),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--target-mode",
            "vendor_retention",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["target_mode"] == "vendor_retention"
    assert payload["limit"] == 1
    assert payload["opportunities"][0]["target_id"] == "review-1"
    assert payload["opportunities"][0]["evidence"][0]["text"] == "Pricing is a problem."


def test_source_adapter_cli_outputs_csv_generation_payload(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text(
        "\n".join([
            "id,company,vendor,transcript,pain_points",
            "call-1,Acme,HubSpot,Renewal review is active,renewal pressure",
        ]),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--format",
            "csv",
            "--target-mode",
            "vendor_retention",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["opportunities"][0]["target_id"] == "call-1"
    assert payload["opportunities"][0]["evidence"][0]["source_type"] == "transcript"


def test_source_adapter_cli_applies_default_fields(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        json.dumps({
            "id": "g2-review-1",
            "vendor": "Slack",
            "review_text": "Search gets slow once message history grows.",
        }),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--format",
            "jsonl",
            "--default-field",
            "company_name=Acme Logistics",
            "--default-field",
            "contact_email=ops@example.com",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    opportunity = json.loads(completed.stdout)["opportunities"][0]
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["contact_email"] == "ops@example.com"


def test_source_adapter_cli_applies_booking_url_to_selling_context(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        json.dumps({
            "id": "g2-review-1",
            "vendor": "Slack",
            "review_text": "Search gets slow once message history grows.",
        }),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--format",
            "jsonl",
            "--booking-url",
            "https://app.example.test/book",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    opportunity = json.loads(completed.stdout)["opportunities"][0]
    assert opportunity["selling"] == {"booking_url": "https://app.example.test/book"}


def test_source_adapter_cli_rejects_non_positive_text_limit(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text("[]", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--max-text-chars",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "--max-text-chars must be positive" in completed.stderr


def test_source_adapter_cli_rejects_invalid_default_field_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        json.dumps({
            "id": "review-1",
            "vendor": "HubSpot",
            "review_text": "Pricing is a problem.",
        }),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--format",
            "jsonl",
            "--default-field",
            "company_name",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "--default-field values must use key=value" in completed.stderr
    assert "Traceback" not in completed.stderr
