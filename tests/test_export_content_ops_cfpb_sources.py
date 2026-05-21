from __future__ import annotations

import importlib.util
import io
import json
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

    assert source_row["id"] == "cfpb:12345"
    assert source_row["source_id"] == "cfpb:12345"
    assert source_row["source"] == "cfpb"
    assert source_row["source_system"] == "cfpb"
    assert source_row["source_type"] == "support_ticket"
    assert source_row["complaint_id"] == "12345"
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
    assert query["has_narrative"] == "true"
    assert query["size"] == 3
    assert query["company"] == "Example Bank"
    assert query["product"] == "Credit card"
    assert query["issue"] == "Fees"
    assert query["search_term"] == "late fee"
    assert query["date_received_min"] == "2026-01-01"
    assert query["date_received_max"] == "2026-02-01"


def test_build_cfpb_query_can_disable_narrative_filter():
    query = exporter.build_cfpb_query(search_term="fees", require_narrative=False)

    assert "has_narrative" not in query


def test_build_cfpb_url_preserves_existing_query_string():
    url = exporter.build_cfpb_url(
        "https://example.test/api?existing=1",
        {"format": "csv", "search_term": "late fee"},
    )

    assert url == "https://example.test/api?existing=1&format=csv&search_term=late+fee"


def test_build_cfpb_headers_uses_browser_compatible_defaults():
    headers = exporter.build_cfpb_headers()

    assert headers["User-Agent"].startswith("Mozilla/5.0")
    assert "AtlasContentOps/1.0" in headers["User-Agent"]
    assert headers["Accept"] == "text/csv,*/*"
    assert headers["Referer"] == exporter.DEFAULT_REFERER


def test_build_cfpb_headers_accepts_host_overrides():
    headers = exporter.build_cfpb_headers(
        user_agent="HostFetcher/2.0",
        referer="https://host.example/source-export",
    )

    assert headers["User-Agent"] == "HostFetcher/2.0"
    assert headers["Referer"] == "https://host.example/source-export"


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

    assert [row["id"] for row in rows] == ["cfpb:2", "cfpb:3"]
    assert calls[0]["timeout"] == 7.5
    assert "company=Example+Bank" in calls[0]["url"]
    assert "search_term=fees" in calls[0]["url"]
    assert "has_narrative=true" in calls[0]["url"]
    assert calls[0]["headers"]["User-agent"].startswith("Mozilla/5.0")
    assert calls[0]["headers"]["Accept"] == "text/csv,*/*"


def test_fetch_cfpb_source_rows_with_profile_counts_skipped_rows(monkeypatch):
    csv_payload = (
        "Complaint ID,Company,Product,Issue,Consumer complaint narrative\n"
        ",Example Bank,Checking,Fees,Missing id complaint.\n"
        "2,Example Bank,Checking,Fees,\n"
        "3,Example Bank,Checking,Fees,First usable complaint.\n"
        "4,Example Bank,Checking,Access,Second usable complaint.\n"
        "5,Example Bank,Checking,Access,Third usable complaint.\n"
    ).encode("utf-8")

    def fake_urlopen(_request, timeout):
        assert timeout == exporter.DEFAULT_TIMEOUT_SECONDS
        return _Response(csv_payload)

    monkeypatch.setattr(exporter, "urlopen", fake_urlopen)

    rows, profile = exporter.fetch_cfpb_source_rows_with_profile(
        api_url="https://example.test/cfpb",
        limit=2,
        max_rows_scanned=5,
    )

    assert [row["id"] for row in rows] == ["cfpb:3", "cfpb:4"]
    assert profile == {
        "status": "ok",
        "raw_row_count": 4,
        "raw_row_count_source": "cfpb_csv_rows_scanned",
        "usable_source_count": 2,
        "skipped_row_count": 2,
        "missing_complaint_id_count": 1,
        "missing_narrative_count": 1,
        "skipped_other_count": 0,
        "usable_source_ratio": 0.5,
        "requested_source_count": 2,
        "max_rows_scanned": 5,
        "stop_reason": "limit",
        "require_narrative": True,
    }


def test_fetch_cfpb_source_rows_threads_request_overrides(monkeypatch):
    csv_payload = (
        "Complaint ID,Company,Product,Issue,Consumer complaint narrative\n"
        "2,Example Bank,Checking,Fees,First usable complaint.\n"
    ).encode("utf-8")
    calls = []

    def fake_urlopen(request, timeout):
        calls.append({"url": request.full_url, "timeout": timeout, "headers": request.headers})
        return _Response(csv_payload)

    monkeypatch.setattr(exporter, "urlopen", fake_urlopen)

    rows = exporter.fetch_cfpb_source_rows(
        api_url="https://example.test/cfpb",
        search_term="fees",
        limit=1,
        max_rows_scanned=1,
        user_agent="HostFetcher/2.0",
        referer="https://host.example/source-export",
        require_narrative=False,
    )

    assert rows[0]["id"] == "cfpb:2"
    assert "has_narrative=true" not in calls[0]["url"]
    assert calls[0]["headers"]["User-agent"] == "HostFetcher/2.0"
    assert calls[0]["headers"]["Referer"] == "https://host.example/source-export"


def test_main_threads_cli_fetch_options(monkeypatch, tmp_path: Path):
    output = tmp_path / "cfpb.jsonl"
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return [{"id": "cfpb:2", "source_type": "support_ticket"}]

    monkeypatch.setattr(exporter, "fetch_cfpb_source_rows", fake_fetch)

    result = exporter._main([
        "--search-term",
        "fees",
        "--limit",
        "1",
        "--max-rows-scanned",
        "2",
        "--user-agent",
        "HostFetcher/2.0",
        "--referer",
        "https://host.example/source-export",
        "--include-rows-without-narrative",
        "--output",
        str(output),
    ])

    assert result == 0
    assert calls[0]["search_term"] == "fees"
    assert calls[0]["user_agent"] == "HostFetcher/2.0"
    assert calls[0]["referer"] == "https://host.example/source-export"
    assert calls[0]["require_narrative"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "id": "cfpb:2",
        "source_type": "support_ticket",
    }


def test_render_jsonl_outputs_one_sorted_json_object_per_row():
    payload = exporter.render_jsonl([
        {"source": "cfpb", "id": "2"},
        {"source": "cfpb", "id": "3"},
    ])

    assert payload.splitlines() == [
        '{"id": "2", "source": "cfpb"}',
        '{"id": "3", "source": "cfpb"}',
    ]
