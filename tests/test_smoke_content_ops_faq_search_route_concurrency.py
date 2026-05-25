from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_route_concurrency.py"
SPEC = importlib.util.spec_from_file_location("smoke_content_ops_faq_search_route_concurrency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


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
        "case_file": None,
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


def _json_response(payload):
    return _Response(json.dumps(payload).encode("utf-8"))


def _write_case_file(tmp_path, payload):
    path = tmp_path / "cases.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def test_parser_requires_results_by_default_and_allows_explicit_liveness_probe():
    required = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
    ])
    liveness_only = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--allow-empty-results",
    ])

    assert required.require_results is True
    assert liveness_only.require_results is False


def test_load_cases_defaults_to_cli_case():
    cases, errors = smoke._load_cases(
        _args(query="credit report", corpus_id="corp-1", status="published", limit=7)
    )

    assert errors == []
    assert cases == [
        {
            "query": "credit report",
            "corpus_id": "corp-1",
            "status": "published",
            "limit": 7,
            "require_results": True,
        }
    ]


def test_load_cases_inherits_optional_cli_defaults(tmp_path):
    case_file = _write_case_file(
        tmp_path,
        [
            {"query": "credit report", "require_results": False},
            {
                "query": "mortgage issue",
                "corpus_id": "corp-b",
                "status": "draft",
                "limit": 2,
            },
        ],
    )

    cases, errors = smoke._load_cases(
        _args(case_file=case_file, corpus_id="corp-a", status="published", limit=9)
    )

    assert errors == []
    assert cases == [
        {
            "query": "credit report",
            "corpus_id": "corp-a",
            "status": "published",
            "limit": 9,
            "require_results": False,
        },
        {
            "query": "mortgage issue",
            "corpus_id": "corp-b",
            "status": "draft",
            "limit": 2,
            "require_results": True,
        },
    ]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ("{bad json", "--case-file must contain JSON: Expecting property name enclosed in double quotes"),
        ({}, "--case-file must contain a non-empty JSON list"),
        ([], "--case-file must contain a non-empty JSON list"),
        ([[]], "case[0] must be an object"),
        ([{}], "case[0].query must be a non-empty string"),
        ([{"query": ""}], "case[0].query must be a non-empty string"),
        ([{"query": "x", "corpus_id": 1}], "case[0].corpus_id must be a string"),
        ([{"query": "x", "status": 1}], "case[0].status must be a string"),
        ([{"query": "x", "limit": "5"}], "case[0].limit must be a positive integer"),
        ([{"query": "x", "limit": True}], "case[0].limit must be a positive integer"),
        ([{"query": "x", "limit": 0}], "case[0].limit must be a positive integer"),
        ([{"query": "x", "require_results": "yes"}], "case[0].require_results must be a boolean"),
    ],
)
def test_load_cases_rejects_bad_case_file_shapes(tmp_path, payload, expected_error):
    case_file = _write_case_file(tmp_path, payload)

    cases, errors = smoke._load_cases(_args(case_file=case_file))

    assert cases == []
    assert expected_error in errors


def test_load_cases_reports_unreadable_case_file():
    cases, errors = smoke._load_cases(_args(case_file=Path("/tmp/atlas-missing-case-file.json")))

    assert cases == []
    assert errors
    assert errors[0].startswith("--case-file could not be read:")


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
    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _json_response(_valid_payload()),
    )

    result = smoke._run_one(3, _args())

    assert result == {
        "index": 3,
        "ok": True,
        "count": 1,
        "elapsed_ms": 12.0,
        "errors": [],
        "case_index": 0,
        "case": {
            "query": "mortgage dispute",
            "corpus_id": "",
            "status": "",
            "limit": 5,
            "require_results": True,
        },
    }


def test_run_one_captures_fetch_and_contract_errors(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.001, 2.0, 2.002]).__next__)
    calls = iter([
        smoke.contract.urllib.error.URLError("network failed"),
        _json_response({"query": "mortgage dispute", "results": [], "count": 0}),
    ])

    def _fake_urlopen(*_args, **_kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    failed_fetch = smoke._run_one(0, _args(require_results=False))
    bad_contract = smoke._run_one(1, _args(require_results=True))

    assert failed_fetch["ok"] is False
    assert failed_fetch["errors"] == ["RuntimeError: route request failed: network failed"]
    assert bad_contract["ok"] is False
    assert bad_contract["errors"] == ["results must include at least one item"]


def test_run_one_rejects_malformed_and_non_object_transport_payloads(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.001, 2.0, 2.002]).__next__)
    calls = iter([
        _Response(b"{bad json"),
        _Response(json.dumps(["not", "an", "object"]).encode("utf-8")),
    ])
    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", lambda *_args, **_kwargs: next(calls))

    bad_json = smoke._run_one(0, _args())
    non_object = smoke._run_one(1, _args())

    assert bad_json["ok"] is False
    assert bad_json["errors"] == ["RuntimeError: route did not return JSON"]
    assert non_object["ok"] is False
    assert non_object["errors"] == ["RuntimeError: route returned non-object JSON"]


def test_run_one_counts_raw_transport_timeout_as_request_failure(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.001]).__next__)

    def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        _raise_timeout,
    )

    result = smoke._run_one(0, _args())

    assert result["ok"] is False
    assert result["errors"] == ["TimeoutError: read timed out"]


def test_run_one_rejects_result_envelope_drift_at_transport_boundary(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.001, 2.0, 2.002]).__next__)
    calls = iter([
        _json_response({"query": "mortgage dispute", "results": "not-a-list", "count": 0}),
        _json_response({"query": "mortgage dispute", "results": [], "count": "0"}),
    ])
    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", lambda *_args, **_kwargs: next(calls))

    bad_results = smoke._run_one(0, _args())
    bad_count = smoke._run_one(1, _args())

    assert bad_results["ok"] is False
    assert bad_results["errors"] == [
        "results must be a list",
        "results must include at least one item",
    ]
    assert bad_count["ok"] is False
    assert bad_count["errors"] == [
        "count must be an integer",
        "results must include at least one item",
    ]


def test_run_concurrent_sorts_results(monkeypatch):
    def _fake_run_one(index, _args, _case):
        return {"index": index, "ok": True, "count": 1, "elapsed_ms": float(3 - index), "errors": []}

    monkeypatch.setattr(smoke, "_run_one", _fake_run_one)

    assert [row["index"] for row in smoke._run_concurrent(_args(requests=3, concurrency=3))] == [0, 1, 2]


def test_run_concurrent_uses_real_worker_threads(monkeypatch):
    barrier = threading.Barrier(2, timeout=2)
    lock = threading.Lock()
    thread_ids = set()

    def _fake_urlopen(*_args, **_kwargs):
        with lock:
            thread_ids.add(threading.get_ident())
        barrier.wait()
        return _json_response(_valid_payload())

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    results = smoke._run_concurrent(_args(requests=2, concurrency=2))

    assert [row["ok"] for row in results] == [True, True]
    assert len(thread_ids) == 2


def test_run_concurrent_round_robins_case_file_requests(monkeypatch, tmp_path):
    case_file = _write_case_file(
        tmp_path,
        [
            {"query": "credit report", "corpus_id": "corp-a", "limit": 2},
            {
                "query": "mortgage dispute",
                "status": "published",
                "limit": 3,
                "require_results": False,
            },
        ],
    )
    args = _args(case_file=case_file, requests=3, concurrency=2)
    cases, errors = smoke._load_cases(args)
    urls = []
    lock = threading.Lock()

    def _fake_urlopen(request, **_kwargs):
        with lock:
            urls.append(request.full_url)
        return _json_response(_valid_payload())

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    results = smoke._run_concurrent(args, cases)

    parsed = [
        smoke.contract.urllib.parse.parse_qs(smoke.contract.urllib.parse.urlsplit(url).query)
        for url in sorted(urls)
    ]
    assert [row["case_index"] for row in results] == [0, 1, 0]
    assert results[1]["case"]["require_results"] is False
    assert parsed == [
        {"corpus_id": ["corp-a"], "limit": ["2"], "q": ["credit report"]},
        {"corpus_id": ["corp-a"], "limit": ["2"], "q": ["credit report"]},
        {"limit": ["3"], "q": ["mortgage dispute"], "status": ["published"]},
    ]


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


def test_main_writes_case_file_preflight_result(tmp_path, capsys):
    case_file = _write_case_file(tmp_path, [{"query": "x", "limit": "5"}])
    result_path = tmp_path / "hosted-concurrency.json"

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--case-file",
        str(case_file),
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["ok"] is False
    assert payload["phase"] == "preflight"
    assert payload["cases"]["case_file"] == str(case_file)
    assert payload["cases"]["total"] == 0
    assert payload["preflight_errors"] == ["case[0].limit must be a positive integer"]
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight"


def test_main_returns_exit_1_and_writes_result_for_contract_failures(tmp_path, monkeypatch):
    result_path = tmp_path / "hosted-concurrency.json"
    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _json_response(
            {"query": "mortgage dispute", "results": [], "count": 0}
        ),
    )

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--requests",
        "1",
        "--concurrency",
        "1",
        "--output-result",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["ok"] is False
    assert payload["phase"] == "complete"
    assert payload["require_results"] is True
    assert payload["errors"]["count"] == 1
    assert payload["errors"]["items"] == [
        {
            "index": 0,
            "case_index": 0,
            "case": {
                "query": "mortgage payment dispute",
                "corpus_id": "",
                "status": "",
                "limit": 5,
                "require_results": True,
            },
            "errors": ["results must include at least one item"],
        }
    ]
