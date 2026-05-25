from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_route_concurrency.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_search_route_concurrency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _args(**overrides):
    values = {
        "base_url": "https://atlas.example.com",
        "token": "token-123",
        "query": "mortgage dispute",
        "corpus_id": "",
        "status": "",
        "limit": 5,
        "route": "/api/v1/content-ops/faq-deflection-search",
        "timeout": 10.0,
        "requests": 4,
        "concurrency": 2,
        "max_error_rate": 0.0,
        "max_p95_ms": None,
        "max_single_request_ms": None,
        "require_results": True,
        "output_result": None,
        "json": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _valid_payload():
    return {
        "query": "mortgage dispute",
        "results": [
            {
                "question": "How do I dispute a mortgage payment error?",
                "answer_summary": "Gather records and contact support.",
                "topic": "Mortgage servicing issues",
                "source_ids": ["CFPB-1"],
                "ticket_count": 4,
                "score": 10,
            }
        ],
        "count": 1,
    }


def test_validate_args_reports_preflight_errors():
    errors = smoke._validate_args(
        _args(base_url="", token="", requests=0, concurrency=0, max_error_rate=1.5)
    )

    assert errors == [
        "ATLAS_API_BASE_URL or --base-url is required",
        "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
        "--requests must be positive",
        "--concurrency must be positive",
        "--max-error-rate must be between 0 and 1",
    ]


def test_latency_and_error_summaries_are_compact():
    results = [
        {"index": 0, "elapsed_ms": 5.0, "errors": []},
        {"index": 1, "elapsed_ms": 2.0, "errors": ["bad envelope"]},
        {"index": 2, "elapsed_ms": 4.0, "errors": []},
        {"index": 3, "elapsed_ms": 3.0, "errors": []},
    ]

    assert smoke._latency_summary(results) == {
        "count": 4,
        "p50_ms": 3.5,
        "p95_ms": 5.0,
        "max_ms": 5.0,
    }
    assert smoke._error_summary(results) == {
        "count": 1,
        "rate": 0.25,
        "items": [{"index": 1, "errors": ["bad envelope"]}],
        "truncated": False,
    }


def test_budget_summary_reports_error_and_latency_failures():
    summary = smoke._budget_summary(
        latency={"p95_ms": 50.0, "max_ms": 75.0},
        errors={"rate": 0.25},
        max_error_rate=0.0,
        max_p95_ms=40.0,
        max_single_request_ms=100.0,
    )

    assert summary == {
        "ok": False,
        "checks": [
            {"metric": "error_rate", "actual": 0.25, "max": 0.0, "ok": False},
            {"metric": "p95_ms", "actual": 50.0, "max": 40.0, "ok": False},
            {"metric": "max_ms", "actual": 75.0, "max": 100.0, "ok": True},
        ],
        "failures": [
            "error_rate exceeded 0.0",
            "p95_ms exceeded 40.0",
        ],
    }


def test_run_one_validates_contract_and_records_count(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.012]).__next__)
    monkeypatch.setattr(smoke.contract, "_fetch_json", lambda *_args, **_kwargs: _valid_payload())

    result = smoke._run_one(3, _args())

    assert result == {
        "index": 3,
        "ok": True,
        "count": 1,
        "elapsed_ms": 12.0,
        "errors": [],
    }


def test_run_one_captures_fetch_and_contract_errors(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.001, 2.0, 2.002]).__next__)
    calls = iter([
        RuntimeError("network failed"),
        {"query": "mortgage dispute", "results": [], "count": 0},
    ])

    def _fake_fetch(*_args, **_kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(smoke.contract, "_fetch_json", _fake_fetch)

    failed_fetch = smoke._run_one(0, _args(require_results=False))
    bad_contract = smoke._run_one(1, _args(require_results=True))

    assert failed_fetch["ok"] is False
    assert failed_fetch["errors"] == ["RuntimeError: network failed"]
    assert bad_contract["ok"] is False
    assert bad_contract["errors"] == ["results must include at least one item"]


def test_run_concurrent_sorts_results(monkeypatch):
    def _fake_run_one(index, _args):
        return {"index": index, "ok": True, "count": 1, "elapsed_ms": float(3 - index), "errors": []}

    monkeypatch.setattr(smoke, "_run_one", _fake_run_one)

    assert [row["index"] for row in smoke._run_concurrent(_args(requests=3, concurrency=3))] == [0, 1, 2]


def test_main_writes_preflight_result(tmp_path, capsys):
    result_path = tmp_path / "hosted-concurrency.json"

    code = smoke.main([
        "--base-url",
        "",
        "--token",
        "token-123",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["ok"] is False
    assert payload["phase"] == "preflight"
    assert payload["preflight_errors"] == ["ATLAS_API_BASE_URL or --base-url is required"]
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight"
