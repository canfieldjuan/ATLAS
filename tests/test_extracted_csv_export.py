from __future__ import annotations

import csv
from io import StringIO

import pytest

from extracted_content_pipeline.ad_copy_export import AdCopyDraftExportResult
from extracted_content_pipeline.csv_export import csv_cell_value


@pytest.mark.parametrize(
    "value",
    (
        "=HYPERLINK(\"https://example.invalid\")",
        "+SUM(A1:A2)",
        "-IMPORTXML(\"https://example.invalid\", \"//a\")",
        "@NOW()",
        " \t=HYPERLINK(\"https://example.invalid\")",
    ),
)
def test_csv_cell_value_neutralizes_formula_like_strings(value: str) -> None:
    escaped = csv_cell_value(value)

    assert escaped == "'" + value


def test_csv_cell_value_preserves_normal_cells() -> None:
    assert csv_cell_value("normal copy") == "normal copy"
    assert csv_cell_value("") == ""
    assert csv_cell_value(None) == ""
    assert csv_cell_value(-42) == -42
    assert csv_cell_value(3.5) == 3.5


def test_csv_cell_value_preserves_json_serialization_contract() -> None:
    assert csv_cell_value({"scope": {"account_id": "acct_1"}}) == (
        '{"scope":{"account_id":"acct_1"}}'
    )
    assert csv_cell_value(["slow support", "renewal risk"]) == (
        '["slow support","renewal risk"]'
    )


def test_ad_copy_csv_export_applies_formula_guard_to_cells() -> None:
    result = AdCopyDraftExportResult(
        rows=({
            "target_id": "target-1",
            "target_mode": "review",
            "channel": "paid_social",
            "format": "single_image",
            "company_name": "Acme",
            "vendor_name": "Zendesk",
            "source_id": "review-1",
            "source_type": "review",
            "pain_point_count": 1,
            "headline": "=HYPERLINK(\"https://example.invalid\")",
            "primary_text": "Open the proof.",
            "cta": "+SUM(A1:A2)",
            "pain_points": ["slow support"],
            "metadata": {"scope": {"account_id": "acct_1"}},
            "id": "ad-copy-1",
            "status": "draft",
        },),
        limit=1,
        filters={"status": "draft"},
    )

    row = next(csv.DictReader(StringIO(result.as_csv())))

    assert row["headline"] == "'=HYPERLINK(\"https://example.invalid\")"
    assert row["cta"] == "'+SUM(A1:A2)"
    assert row["pain_points"] == '["slow support"]'
    assert row["metadata"] == '{"scope":{"account_id":"acct_1"}}'
