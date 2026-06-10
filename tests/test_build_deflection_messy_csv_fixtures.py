from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_source_adapters import (
    load_source_campaign_opportunities_from_file,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_deflection_messy_csv_fixtures.py"
SPEC = importlib.util.spec_from_file_location("build_deflection_messy_csv_fixtures", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def _write_source_rows(tmp_path: Path) -> Path:
    path = tmp_path / "source_rows.jsonl"
    rows = [
        {
            "source_id": "cfpb:1",
            "source_title": "Checking account - Managing an account",
            "text": "The bank charged a fee after I closed the account.",
            "pain_category": "Managing an account",
            "company_response": "Closed with explanation",
        },
        {
            "source_id": "cfpb:2",
            "source_title": "Credit card - Billing dispute",
            "text": "My payment was applied to the wrong balance.",
            "pain_category": "Billing dispute",
            "company_response": "Closed with non-monetary relief",
        },
        {
            "source_id": "cfpb:3",
            "source_title": "Debt collection - Communication tactics",
            "text": "The collector called repeatedly after I requested email only.",
            "pain_category": "Communication tactics",
        },
        {
            "source_id": "cfpb:4",
            "source_title": "Mortgage - Escrow shortage",
            "text": "My escrow payment changed and the notice did not explain why.",
            "pain_category": "Escrow shortage",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_build_deflection_messy_csv_fixtures_writes_manifest_and_cases(tmp_path: Path) -> None:
    source_rows = _write_source_rows(tmp_path)
    output_dir = tmp_path / "fixtures"

    manifest = generator.write_messy_csv_fixtures(
        generator.support_ticket_rows(generator.load_source_rows(source_rows)),
        output_dir,
    )

    assert (output_dir / "manifest.json").exists()
    assert [case["name"] for case in manifest["cases"]] == [
        "bom_utf8.csv",
        "cp1252_semicolon.csv",
        "tab_delimited.csv",
        "html_bodies.csv",
        "leading_metadata_row.csv",
        "ragged_extra_cells.csv",
        "ragged_short_rows.csv",
        "quoted_multiline.csv",
    ]
    assert {case["expected"] for case in manifest["cases"]} == {
        "fail_loud",
        "parsed",
        "parsed_partial",
    }
    assert (output_dir / "bom_utf8.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert b"Buyer\x92s wording kept intact" in (output_dir / "cp1252_semicolon.csv").read_bytes()


@pytest.mark.parametrize(
    ("filename", "minimum_count"),
    [
        ("bom_utf8.csv", 4),
        ("cp1252_semicolon.csv", 4),
        ("tab_delimited.csv", 4),
        ("html_bodies.csv", 4),
        ("leading_metadata_row.csv", 3),
        ("quoted_multiline.csv", 3),
        ("ragged_short_rows.csv", 1),
    ],
)
def test_generated_messy_csv_cases_parse_through_real_source_adapter(
    tmp_path: Path,
    filename: str,
    minimum_count: int,
) -> None:
    output_dir = _build_fixtures(tmp_path)

    loaded = load_source_campaign_opportunities_from_file(
        output_dir / filename,
        file_format="csv",
    )

    assert len(loaded.opportunities) >= minimum_count
    assert loaded.opportunities[0]["source_type"] == "support_ticket"
    assert loaded.opportunities[0]["evidence"][0]["text"]


def test_generated_html_case_produces_non_empty_deflection_report_items(tmp_path: Path) -> None:
    output_dir = _build_fixtures(tmp_path)
    loaded = load_source_campaign_opportunities_from_file(
        output_dir / "html_bodies.csv",
        file_format="csv",
    )

    result = build_ticket_faq_markdown(loaded.opportunities, max_items=0)

    assert result.items
    assert "<p>" not in result.markdown
    assert "Needs review & policy check." in result.markdown


@pytest.mark.parametrize("filename", ["ragged_extra_cells.csv"])
def test_generated_bad_messy_csv_cases_fail_loud(tmp_path: Path, filename: str) -> None:
    output_dir = _build_fixtures(tmp_path)

    with pytest.raises(ValueError, match="more cells than the header"):
        load_source_campaign_opportunities_from_file(output_dir / filename, file_format="csv")


def test_generator_cli_outputs_json_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source_rows = _write_source_rows(tmp_path)
    output_dir = tmp_path / "fixtures"

    assert generator.main([str(source_rows), "--output-dir", str(output_dir), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "deflection_messy_csv_fixtures"
    assert len(payload["cases"]) == 8


def _build_fixtures(tmp_path: Path) -> Path:
    source_rows = _write_source_rows(tmp_path)
    output_dir = tmp_path / "fixtures"
    generator.write_messy_csv_fixtures(
        generator.support_ticket_rows(generator.load_source_rows(source_rows)),
        output_dir,
    )
    return output_dir
