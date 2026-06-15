from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_source_adapters import (
    load_source_rows_from_file,
)
from extracted_content_pipeline.faq_deflection_report import (
    deflection_report_summary,
)
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)
from extracted_content_pipeline.ticket_faq_markdown import (
    build_ticket_faq_markdown,
)


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke_content_ops_support_ticket_package.py"
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROVIDER_FIXTURE_DIR = (
    _REPO_ROOT
    / "extracted_content_pipeline"
    / "examples"
    / "support_ticket_provider_exports"
)
_FULL_THREAD_PROVIDER_FIXTURES = (
    (
        "zendesk_full_thread_export.csv",
        "How do I reset MFA?",
        "Open Profile, Security",
    ),
    (
        "freshdesk_full_thread_export.csv",
        "How do I export the attribution report before our board meeting?",
        "Open Analytics, choose Attribution",
    ),
    (
        "help_scout_full_thread_export.csv",
        "Can I download a receipt for last month's invoice?",
        "Open Billing, select Invoices",
    ),
    (
        "intercom_conversation_export.csv",
        "How do I change the account owner email?",
        "Open Settings, choose Team",
    ),
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_support_ticket_package",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
assert _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)

build_support_ticket_package_smoke_summary = (
    _SCRIPT_MODULE.build_support_ticket_package_smoke_summary
)
main = _SCRIPT_MODULE.main


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(header, "")) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_cell(value: str) -> str:
    escaped = str(value).replace('"', '""')
    return f'"{escaped}"'


def test_support_ticket_package_smoke_summarizes_undated_csv_without_window_filter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tickets.csv"
    _write_csv(path, [
        {
            "ticket_id": "ticket-1",
            "subject": "How do I change my login email?",
            "description": "I cannot find where to update the email on my account.",
            "pain_category": "profile updates",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Export dashboard",
            "description": "Where do I export the dashboard before renewal?",
        },
        {"ticket_id": "ticket-3"},
    ])

    summary = build_support_ticket_package_smoke_summary(path)

    assert summary["source_row_count"] == 3
    assert summary["included_ticket_row_count"] == 2
    assert summary["skipped_ticket_row_count"] == 1
    assert summary["truncated_ticket_row_count"] == 0
    assert summary["source_period"] == "Uploaded support tickets"
    assert summary["has_dated_window"] is False
    assert summary["has_window_filter"] is False
    assert summary["faq_question_count"] == 2
    assert summary["question_like_ticket_count"] == 2
    assert summary["contact_email_count"] == 0
    assert summary["top_ticket_clusters"] == [
        {"label": "profile updates", "count": 1},
        {"label": "dashboard export renewal", "count": 1},
    ]
    assert summary["customer_wording_examples"][0] == {
        "source_id": "ticket-1",
        "source_title": "How do I change my login email?",
        "pain_category": "profile updates",
        "text": (
            "How do I change my login email? I cannot find where to update the "
            "email on my account."
        ),
    }
    assert summary["support_ticket_resolution_evidence_present"] is False
    assert summary["support_ticket_resolution_evidence_count"] == 0
    assert summary["support_ticket_resolution_examples"] == []
    assert summary["has_measured_outcomes"] is False
    assert summary["measured_outcome_count"] == 0
    assert summary["measured_outcome_examples"] == []
    assert summary["warnings"] == [
        {
            "code": "ticket_row_missing_text",
            "row_index": 3,
            "message": "Skipped ticket row because it did not include customer wording.",
        }
    ]


def test_support_ticket_package_uses_zendesk_public_comments_not_internal_notes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "zendesk_public_comments.csv"
    _write_csv(path, [
        {
            "Ticket ID": "zd-1",
            "Subject": "Refund status",
            "Description": "",
            "Ticket Comments": "How do I see when the duplicate charge was refunded?",
            "Ticket History": "I still cannot find the refund receipt.",
            "Internal Notes": "Known billing bug; do not publish this detail.",
        },
        {
            "Ticket ID": "zd-2",
            "Subject": "",
            "Description": "",
            "Private Notes": "Customer is escalated internally.",
        },
    ])

    summary = build_support_ticket_package_smoke_summary(path)

    assert summary["source_row_count"] == 2
    assert summary["included_ticket_row_count"] == 1
    assert summary["skipped_ticket_row_count"] == 1
    assert summary["customer_wording_examples"] == [
        {
            "source_id": "zd-1",
            "source_title": "Refund status",
            "text": (
                "Refund status How do I see when the duplicate charge was "
                "refunded? I still cannot find the refund receipt."
            ),
        }
    ]
    customer_wording = str(summary["customer_wording_examples"])
    assert "Known billing bug" not in customer_wording
    assert "Customer is escalated internally" not in customer_wording
    assert summary["warnings"] == [
        {
            "code": "ticket_row_missing_text",
            "row_index": 2,
            "message": "Skipped ticket row because it did not include customer wording.",
        }
    ]


def test_support_ticket_package_skips_private_comment_objects_in_history() -> None:
    package = build_support_ticket_input_package([{
        "ticket_id": "zd-1",
        "subject": "Refund status",
        "ticket_history": [
            {"body": "How do I see when the duplicate charge was refunded?", "public": True},
            {"body": "INTERNAL refund quietly; do not publish", "public": False},
            {"body": "Where can I download the refund receipt?"},
        ],
    }])

    source_material = package.inputs["source_material"]
    assert len(source_material) == 1
    text = source_material[0]["text"]
    assert "How do I see when the duplicate charge was refunded?" in text
    assert "Where can I download the refund receipt?" in text
    assert "INTERNAL refund quietly" not in text
    assert package.warnings == ()


def test_support_ticket_package_smoke_keeps_date_window_for_dated_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tickets.csv"
    _write_csv(path, [
        {
            "ticket_id": "ticket-1",
            "description": "How do I export reports?",
            "created_at": "2026-05-01",
        },
        {
            "ticket_id": "ticket-2",
            "description": "Where do I update billing?",
            "created_at": "2026-05-02T12:00:00Z",
        },
    ])

    summary = build_support_ticket_package_smoke_summary(path, window_days=45)

    assert summary["source_period"] == "Last 45 days of support tickets"
    assert summary["has_dated_window"] is True
    assert summary["has_window_filter"] is True
    assert summary["faq_window_days"] == 45


def test_support_ticket_package_smoke_reports_resolution_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tickets.csv"
    _write_csv(path, [
        {
            "ticket_id": "ticket-1",
            "subject": "How do I export reports?",
            "description": "Where do I export the dashboard?",
            "resolution_text": "Open Reports, choose Export, then select CSV.",
        },
    ])

    summary = build_support_ticket_package_smoke_summary(path)

    assert summary["support_ticket_resolution_evidence_present"] is True
    assert summary["support_ticket_resolution_evidence_count"] == 1
    assert summary["support_ticket_resolution_example_count"] == 1
    assert summary["support_ticket_resolution_examples"] == [
        {
            "source_id": "ticket-1",
            "source_title": "How do I export reports?",
            "text": "Open Reports, choose Export, then select CSV.",
        }
    ]


def test_support_ticket_package_smoke_accepts_platform_export_fixture() -> None:
    fixture = (
        _REPO_ROOT
        / "extracted_content_pipeline"
        / "examples"
        / "support_ticket_platform_export_shapes.csv"
    )

    summary = build_support_ticket_package_smoke_summary(
        fixture,
        require_included_rows=True,
    )

    assert summary["source_row_count"] == 3
    assert summary["included_ticket_row_count"] == 3
    assert summary["skipped_ticket_row_count"] == 0
    assert summary["source_period"] == "Last 90 days of support tickets"
    assert summary["has_dated_window"] is True
    assert summary["contact_email_count"] == 3
    assert summary["faq_questions"] == [
        "How do I reset MFA?",
        "Where do I export my invoice?",
        "Can I cancel my account before it renews?",
    ]
    assert summary["customer_wording_examples"][1] == {
        "source_id": "fd-200",
        "source_title": "Where do I export my invoice?",
        "pain_category": "billing export",
        "text": (
            "Where do I export my invoice? "
            "Where do I export my invoice before month-end?"
        ),
    }


@pytest.mark.parametrize(
    ("filename", "expected_question", "expected_resolution"),
    _FULL_THREAD_PROVIDER_FIXTURES,
)
def test_support_ticket_package_smoke_accepts_provider_full_thread_exports(
    filename: str,
    expected_question: str,
    expected_resolution: str,
) -> None:
    fixture = _PROVIDER_FIXTURE_DIR / filename

    summary = build_support_ticket_package_smoke_summary(
        fixture,
        require_included_rows=True,
    )

    assert summary["source_row_count"] == 3
    assert summary["included_ticket_row_count"] == 3
    assert summary["skipped_ticket_row_count"] == 0
    assert summary["contact_email_count"] == 3
    assert summary["question_like_ticket_count"] >= 3
    assert summary["support_ticket_resolution_evidence_present"] is True
    assert summary["support_ticket_resolution_evidence_count"] == 2
    assert summary["support_ticket_resolution_example_count"] == 2
    assert summary["warning_count"] == 0
    assert expected_question in summary["faq_questions"]
    assert expected_resolution in str(summary["support_ticket_resolution_examples"])
    assert "<p>" not in str(summary["customer_wording_examples"])
    assert "&amp;" not in str(summary["customer_wording_examples"])
    assert summary["metadata"]["cluster_quality"]["largest_cluster_count"] >= 2


@pytest.mark.parametrize(
    ("filename", "_expected_question", "expected_resolution"),
    _FULL_THREAD_PROVIDER_FIXTURES,
)
def test_provider_full_thread_exports_generate_publishable_deflection_items(
    filename: str,
    _expected_question: str,
    expected_resolution: str,
) -> None:
    rows = load_source_rows_from_file(
        _PROVIDER_FIXTURE_DIR / filename,
        file_format="csv",
    )
    package = build_support_ticket_input_package(rows)

    result = build_ticket_faq_markdown(
        package.inputs["source_material"],
        max_items=4,
    )
    summary = deflection_report_summary(result)

    assert summary["support_ticket_resolution_evidence_present"] is True
    assert summary["drafted_answer_count"] >= 1
    assert "resolution_evidence" in {
        item["answer_evidence_status"] for item in result.items
    }
    assert expected_resolution in result.markdown
    assert "<p>" not in result.markdown
    assert "example.test" not in result.markdown


def test_support_ticket_package_smoke_marks_ticket_index_only_export_gap_list_only() -> None:
    fixture = _PROVIDER_FIXTURE_DIR / "zendesk_ticket_index_only.csv"

    summary = build_support_ticket_package_smoke_summary(
        fixture,
        require_included_rows=True,
    )

    assert summary["source_row_count"] == 2
    assert summary["included_ticket_row_count"] == 2
    assert summary["question_like_ticket_count"] == 0
    assert summary["support_ticket_resolution_evidence_present"] is False
    assert summary["support_ticket_resolution_evidence_count"] == 0
    assert summary["faq_questions"] == []


def test_support_ticket_package_smoke_reports_measured_outcome_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tickets.csv"
    _write_csv(path, [
        {
            "ticket_id": "ticket-1",
            "subject": "Do FAQs reduce repeat tickets?",
            "description": "Can we tell if the billing FAQ helped?",
            "measured_outcome": "Repeat billing tickets fell from 18 to 11.",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Zero deflections",
            "description": "What happened after the trial FAQ update?",
            "deflection_rate": "0",
        },
    ])

    summary = build_support_ticket_package_smoke_summary(path)

    assert summary["has_measured_outcomes"] is True
    assert summary["measured_outcome_count"] == 2
    assert summary["measured_outcome_example_count"] == 2
    assert summary["measured_outcome_examples"] == [
        {
            "source_id": "ticket-1",
            "source_title": "Do FAQs reduce repeat tickets?",
            "text": "Repeat billing tickets fell from 18 to 11.",
        },
        {
            "source_id": "ticket-2",
            "source_title": "Zero deflections",
            "text": "0",
        },
    ]


def test_support_ticket_package_smoke_reports_cluster_rollup_and_truncation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tickets.csv"
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "description": f"How do I fix issue {index}?",
            "pain_category": f"category-{index}",
        }
        for index in range(1, 9)
    ]
    rows.append({"ticket_id": "ticket-9", "description": "Can I change my plan?"})
    rows.append({"ticket_id": "ticket-10", "description": "Why was I charged?"})
    _write_csv(path, rows)

    summary = build_support_ticket_package_smoke_summary(path, max_rows=9)

    assert summary["source_row_count"] == 10
    assert summary["included_ticket_row_count"] == 9
    assert summary["truncated_ticket_row_count"] == 1
    assert summary["top_ticket_clusters"] == [
        {"label": "category-1", "count": 1},
        {"label": "category-2", "count": 1},
        {"label": "category-3", "count": 1},
        {"label": "category-4", "count": 1},
        {"label": "category-5", "count": 1},
        {"label": "category-6", "count": 1},
        {"label": "category-7", "count": 1},
        {"label": "category-8", "count": 1},
        {"label": "plan update", "count": 1},
    ]
    assert summary["warning_count"] == 1
    assert summary["warnings"][0]["code"] == "ticket_rows_truncated"


def test_support_ticket_package_smoke_rolls_up_only_after_twelve_clusters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tickets.csv"
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "description": f"How do I fix issue {index}?",
            "pain_category": f"category-{index}",
        }
        for index in range(1, 15)
    ]
    _write_csv(path, rows)

    summary = build_support_ticket_package_smoke_summary(path)

    assert summary["top_ticket_clusters"] == [
        *[
            {"label": f"category-{index}", "count": 1}
            for index in range(1, 13)
        ],
        {"label": "remaining", "count": 2},
    ]


def test_support_ticket_package_smoke_main_writes_json(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "tickets.csv"
    _write_csv(path, [
        {
            "ticket_id": "ticket-1",
            "subject": "How do I export reports?",
            "created_at": "2026-05-01",
        }
    ])

    assert main([str(path), "--outputs", "landing_page", "--pretty"]) == 0

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["outputs"] == ["landing_page"]
    assert summary["included_ticket_row_count"] == 1
    assert captured.err == ""


def test_support_ticket_package_smoke_can_fail_when_no_rows_survive(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "tickets.csv"
    _write_csv(path, [{"ticket_id": "ticket-1"}])

    assert main([str(path), "--require-included-rows"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No usable support-ticket rows survived packaging." in captured.err
