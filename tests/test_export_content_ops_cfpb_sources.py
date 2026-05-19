from __future__ import annotations

import importlib.util
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_content_ops_cfpb_sources.py"
SPEC = importlib.util.spec_from_file_location(
    "export_content_ops_cfpb_sources",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def _cfpb_row(**overrides):
    row = {
        "Complaint ID": "12345",
        "Company": "Example Bank",
        "Product": "Checking or savings account",
        "Sub-product": "Checking account",
        "Issue": "Managing an account",
        "Sub-issue": "Problem using a debit or ATM card",
        "Consumer complaint narrative": "The bank kept charging fees after I closed the account.",
        "Date received": "2026-01-15",
        "Date sent to company": "2026-01-16",
        "Company public response": "Company believes it acted appropriately.",
        "Company response to consumer": "Closed with explanation",
        "Timely response?": "Yes",
        "Consumer disputed?": "N/A",
        "Consumer consent provided?": "Consent provided",
        "Submitted via": "Web",
        "State": "TX",
        "Tags": "Older American",
    }
    row.update(overrides)
    return row


def test_cfpb_row_to_source_row_maps_public_complaint_fields():
    source_row = exporter.cfpb_row_to_source_row(_cfpb_row())

    assert source_row["id"] == "12345"
    assert source_row["source_id"] == "12345"
    assert source_row["source"] == "cfpb"
    assert source_row["source_system"] == "cfpb"
    assert source_row["source_type"] == "support_ticket"
    assert source_row["vendor_name"] == "Example Bank"
    assert source_row["text"].startswith("The bank kept charging fees")
    assert source_row["pain_category"] == "Managing an account"
    assert source_row["source_title"] == "Checking or savings account - Managing an account"
    assert source_row["source_url"].endswith("/12345")
    assert source_row["category"] == "Checking or savings account"
    assert source_row["submitted_via"] == "Web"


def test_cfpb_row_to_source_row_drops_rows_without_narrative_or_id():
    assert exporter.cfpb_row_to_source_row(_cfpb_row(**{"Consumer complaint narrative": ""})) == {}
    assert exporter.cfpb_row_to_source_row(_cfpb_row(**{"Complaint ID": ""})) == {}


def test_build_cfpb_query_keeps_filters_parametric():
    query = exporter.build_cfpb_query(
        company="Example Bank",
        product="Credit card",
        issue="Fees",
        search_term="late fee",
        date_received_min="2026-01-01",
        date_received_max="2026-02-01",
        limit=3,
    )

    assert query["format"] == "csv"
    assert query["field"] == "all"
    assert query["no_aggs"] == "true"
    assert query["size"] == 3
    assert query["company"] == "Example Bank"
    assert query["product"] == "Credit card"
    assert query["issue"] == "Fees"
    assert query["search_term"] == "late fee"
    assert query["date_received_min"] == "2026-01-01"
    assert query["date_received_max"] == "2026-02-01"


def test_build_cfpb_url_preserves_existing_query_string():
    url = exporter.build_cfpb_url(
        "https://example.test/api?existing=1",
        {"format": "csv", "search_term": "late fee"},
    )

    assert url == "https://example.test/api?existing=1&format=csv&search_term=late+fee"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_fetch_cfpb_source_rows_streams_until_limit(monkeypatch):
    csv_payload = (
        "Complaint ID,Company,Product,Issue,Consumer complaint narrative\n"
        "1,Example Bank,Checking,Fees,\n"
        "2,Example Bank,Checking,Fees,First usable complaint.\n"
        "3,Example Bank,Checking,Access,Second usable complaint.\n"
        "4,Example Bank,Checking,Access,Third usable complaint.\n"
    ).encode("utf-8")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append({"url": request.full_url, "timeout": timeout, "headers": request.headers})
        return _Response(csv_payload)

    monkeypatch.setattr(exporter, "urlopen", fake_urlopen)

    rows = exporter.fetch_cfpb_source_rows(
        api_url="https://example.test/cfpb",
        company="Example Bank",
        search_term="fees",
        limit=2,
        max_rows_scanned=4,
        timeout=7.5,
    )

    assert [row["id"] for row in rows] == ["2", "3"]
    assert calls[0]["timeout"] == 7.5
    assert "company=Example+Bank" in calls[0]["url"]
    assert "search_term=fees" in calls[0]["url"]
    assert calls[0]["headers"]["User-agent"] == "Atlas-Content-Ops-CFPB-Source/1.0"


def test_render_jsonl_outputs_one_sorted_json_object_per_row():
    payload = exporter.render_jsonl([
        {"source": "cfpb", "id": "2"},
        {"source": "cfpb", "id": "3"},
    ])

    assert payload.splitlines() == [
        '{"id": "2", "source": "cfpb"}',
        '{"id": "3", "source": "cfpb"}',
    ]
