from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_route_concurrency.py"
RUNBOOK = ROOT / "docs/extraction/validation/content_ops_faq_route_concurrency_runbook.md"
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
        "detail_route": "",
        "timeout": 10.0,
        "requests": 4,
        "concurrency": 2,
        "max_error_rate": 0.0,
        "max_p95_ms": None,
        "max_single_request_ms": None,
        "max_detail_ms": None,
        "max_case_error_rate": None,
        "max_case_p95_ms": None,
        "max_case_single_request_ms": None,
        "require_results": True,
        "require_detail": False,
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
                "account_id": "acct-1",
                "corpus_id": "corpus-1",
                "faq_id": "11111111-1111-1111-1111-111111111111",
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


def _valid_detail_payload(faq_id="11111111-1111-1111-1111-111111111111"):
    return {
        "account_id": "acct-1",
        "id": faq_id,
        "target_id": "corpus-1",
        "target_mode": "faq_report",
        "title": "Mortgage servicing issues",
        "markdown": "# Mortgage servicing issues\n\nDraft answer.",
        "items": [{
            "topic": "Mortgage servicing issues",
            "question": "How do I dispute a mortgage payment error?",
            "question_source": "customer_wording",
            "summary": "Customers ask how to dispute mortgage payment errors.",
            "frequency": 4,
            "weighted_frequency": 4,
            "ticket_count": 4,
            "opportunity_score": 8,
            "failure_risk_score": 1,
            "failure_risk_signals": ["repeat_question"],
            "answer": "Customers mention mortgage payment disputes.",
            "steps": ["Review the statement.", "Contact support with records."],
            "action_items": ["Review the statement.", "Contact support with records."],
            "answer_evidence_status": "draft_needs_review",
            "resolution_source_count": 0,
            "when_to_contact_support": "Contact support if the payment still looks wrong.",
            "evidence_quotes": ["`CFPB-1`: payment dispute"],
            "source_ids": ["CFPB-1"],
            "source_labels": ["`CFPB-1`"],
            "source_type_counts": {"support_ticket": 1},
            "weighted_source_volume_by_type": {"support_ticket": 4},
            "term_mappings": [],
            "evidence_count": 1,
            "displayed_evidence_count": 1,
        }],
        "source_count": 1,
        "ticket_source_count": 4,
        "output_checks": {},
        "warnings": [],
        "metadata": {},
        "status": "published",
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


def test_validate_args_rejects_detail_without_required_results():
    errors = smoke._validate_args(_args(require_detail=True, require_results=False))

    assert errors == [
        "--require-detail requires result rows; remove --allow-empty-results"
    ]


def test_validate_args_rejects_detail_budget_without_detail_check():
    errors = smoke._validate_args(_args(max_detail_ms=100.0, require_detail=False))

    assert errors == ["--max-detail-ms requires --require-detail"]


def test_validate_args_rejects_non_positive_detail_budget():
    errors = smoke._validate_args(_args(max_detail_ms=0.0, require_detail=True))

    assert errors == ["--max-detail-ms must be positive"]


def test_validate_args_rejects_case_error_budget_outside_rate_range():
    assert smoke._validate_args(_args(max_case_error_rate=-0.1)) == [
        "--max-case-error-rate must be between 0 and 1"
    ]
    assert smoke._validate_args(_args(max_case_error_rate=1.1)) == [
        "--max-case-error-rate must be between 0 and 1"
    ]


def test_validate_args_rejects_non_positive_case_latency_budgets():
    assert smoke._validate_args(_args(max_case_p95_ms=0.0)) == [
        "--max-case-p95-ms must be positive"
    ]
    assert smoke._validate_args(_args(max_case_single_request_ms=-1.0)) == [
        "--max-case-single-request-ms must be positive"
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


def test_parser_accepts_opt_in_detail_route_check():
    parsed = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--require-detail",
        "--detail-route",
        "/api/v1/content-ops/faq-deflection-reports/{faq_id}",
    ])

    assert parsed.require_detail is True
    assert parsed.detail_route == "/api/v1/content-ops/faq-deflection-reports/{faq_id}"


def test_parser_accepts_detail_latency_budget_with_detail_check():
    parsed = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--require-detail",
        "--max-detail-ms",
        "250",
    ])

    assert parsed.require_detail is True
    assert parsed.max_detail_ms == 250.0


def test_parser_accepts_case_error_rate_budget():
    parsed = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--max-case-error-rate",
        "0.25",
    ])

    assert parsed.max_case_error_rate == 0.25


def test_parser_accepts_case_latency_budgets():
    parsed = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--max-case-p95-ms",
        "1500",
        "--max-case-single-request-ms",
        "3000",
    ])

    assert parsed.max_case_p95_ms == 1500.0
    assert parsed.max_case_single_request_ms == 3000.0


def test_route_concurrency_runbook_documents_current_budget_flags():
    text = RUNBOOK.read_text(encoding="utf-8")
    expected_flags = [
        "--require-detail",
        "--max-error-rate",
        "--max-case-error-rate",
        "--max-p95-ms",
        "--max-single-request-ms",
        "--max-case-p95-ms",
        "--max-case-single-request-ms",
        "--max-detail-ms",
        "--output-result",
    ]

    for flag in expected_flags:
        assert flag in text

    assert '--token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}"' in text
    assert '--token "$ATLAS_B2B_JWT"' not in text
    assert "Use hit-only case files with `--require-detail`" in text
    assert "Run known miss cases separately without\n`--require-detail`" in text

    parsed = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}",
        "--case-file",
        "/tmp/faq-route-cases.json",
        "--requests",
        "40",
        "--concurrency",
        "8",
        "--require-detail",
        "--max-error-rate",
        "0",
        "--max-case-error-rate",
        "0",
        "--max-p95-ms",
        "1500",
        "--max-single-request-ms",
        "3000",
        "--max-case-p95-ms",
        "1500",
        "--max-case-single-request-ms",
        "3000",
        "--max-detail-ms",
        "1000",
        "--output-result",
        "/tmp/faq-route-concurrency-result.json",
    ])

    assert parsed.require_detail is True
    assert parsed.max_case_error_rate == 0.0
    assert parsed.max_case_p95_ms == 1500.0
    assert parsed.max_case_single_request_ms == 3000.0
    assert parsed.max_detail_ms == 1000.0


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


def test_summary_payload_uses_single_case_file_query(tmp_path):
    case_file = tmp_path / "cases.json"
    cases = [
        {
            "query": "cancel enterprise export",
            "corpus_id": "corp-a",
            "status": "published",
            "limit": 3,
            "require_results": True,
        }
    ]

    payload = smoke._summary_payload(
        args=_args(query="mortgage dispute", case_file=case_file),
        cases=cases,
        results=[],
        elapsed_seconds=0.0,
    )

    assert payload["query"] == "cancel enterprise export"
    assert payload["query_mode"] == "case_file"
    assert payload["query_count"] == 1


def test_summary_payload_marks_mixed_case_file_queries(tmp_path):
    case_file = tmp_path / "cases.json"
    cases = [
        {
            "query": "cancel enterprise export",
            "corpus_id": "corp-a",
            "status": "published",
            "limit": 3,
            "require_results": True,
        },
        {
            "query": "invite seat limit",
            "corpus_id": "corp-b",
            "status": "published",
            "limit": 3,
            "require_results": True,
        },
    ]

    payload = smoke._summary_payload(
        args=_args(query="mortgage dispute", case_file=case_file),
        cases=cases,
        results=[],
        elapsed_seconds=0.0,
    )

    assert payload["query"] == "multiple case-file queries"
    assert payload["query_mode"] == "case_file_mixed"
    assert payload["query_count"] == 2


def test_summary_payload_marks_invalid_case_file_without_default_query(tmp_path):
    case_file = tmp_path / "cases.json"

    payload = smoke._summary_payload(
        args=_args(query="mortgage dispute", case_file=case_file),
        cases=[],
        results=[],
        elapsed_seconds=0.0,
        preflight_errors=["case[0].query must be a non-empty string"],
    )

    assert payload["query"] == "no valid case queries"
    assert payload["query_mode"] == "case_file_invalid"
    assert payload["query_count"] == 0


def test_load_cases_rejects_case_empty_results_with_detail_required(tmp_path):
    case_file = _write_case_file(
        tmp_path,
        [{"query": "unknown billing workflow", "require_results": False}],
    )

    cases, errors = smoke._load_cases(_args(case_file=case_file, require_detail=True))

    assert cases == []
    assert errors == [
        "case[0].require_results cannot be false when --require-detail is set"
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
        ([{"query": "x", "expected_count": "1"}], "case[0].expected_count must be a non-negative integer"),
        ([{"query": "x", "expected_count": True}], "case[0].expected_count must be a non-negative integer"),
        ([{"query": "x", "expected_count": -1}], "case[0].expected_count must be a non-negative integer"),
        (
            [{"query": "x", "expected_first_account_id": ""}],
            "case[0].expected_first_account_id must be a non-empty string",
        ),
        (
            [{"query": "x", "expected_first_corpus_id": 1}],
            "case[0].expected_first_corpus_id must be a non-empty string",
        ),
        (
            [{"query": "x", "expected_first_faq_id": []}],
            "case[0].expected_first_faq_id must be a non-empty string",
        ),
        (
            [{"query": "x", "expected_detail_account_id": ""}],
            "case[0].expected_detail_account_id must be a non-empty string",
        ),
        (
            [{"query": "x", "expected_detail_target_id": 1}],
            "case[0].expected_detail_target_id must be a non-empty string",
        ),
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


def test_load_cases_accepts_seeded_expectations(tmp_path):
    case_file = _write_case_file(
        tmp_path,
        [{
            "query": "reset password",
            "expected_count": 1,
            "expected_first_account_id": "acct-1",
            "expected_first_corpus_id": "corpus-1",
            "expected_first_faq_id": "11111111-1111-1111-1111-111111111111",
            "expected_detail_account_id": "acct-1",
            "expected_detail_target_id": "support-corpus-1",
            "expected_detail_target_mode": "support_account",
            "expected_detail_title": "Seeded FAQ",
            "expected_detail_status": "approved",
        }],
    )

    cases, errors = smoke._load_cases(_args(case_file=case_file))

    assert errors == []
    assert cases[0]["expected_count"] == 1
    assert cases[0]["expected_first_account_id"] == "acct-1"
    assert cases[0]["expected_first_corpus_id"] == "corpus-1"
    assert cases[0]["expected_first_faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert cases[0]["expected_detail_account_id"] == "acct-1"
    assert cases[0]["expected_detail_target_id"] == "support-corpus-1"
    assert cases[0]["expected_detail_target_mode"] == "support_account"
    assert cases[0]["expected_detail_title"] == "Seeded FAQ"
    assert cases[0]["expected_detail_status"] == "approved"


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


def test_detail_summary_reports_required_detail_failures():
    results = [
        {
            "index": 0,
            "detail_checked": True,
            "detail_faq_id": "faq-1",
            "detail_elapsed_ms": 12.0,
            "detail_errors": [],
        },
        {
            "index": 1,
            "detail_checked": False,
            "detail_faq_id": "faq-2",
            "detail_elapsed_ms": None,
            "detail_errors": ["RuntimeError: route request failed"],
        },
    ]

    assert smoke._detail_summary(results, required=True) == {
        "required": True,
        "checked": 1,
        "failures": 1,
        "items": [
            {
                "index": 1,
                "faq_id": "faq-2",
                "errors": ["RuntimeError: route request failed"],
            }
        ],
        "truncated": False,
        "latency": {"count": 1, "p50_ms": 12.0, "p95_ms": 12.0, "max_ms": 12.0},
    }
    assert smoke._detail_summary(results, required=False) == {
        "required": False,
        "checked": 0,
        "failures": 0,
        "items": [],
        "latency": {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0},
    }


def test_case_result_summaries_group_mixed_case_visibility():
    cases = [
        {
            "query": "credit report",
            "corpus_id": "corp-a",
            "status": "published",
            "limit": 2,
            "require_results": True,
        },
        {
            "query": "mortgage dispute",
            "corpus_id": "corp-b",
            "status": "",
            "limit": 3,
            "require_results": True,
        },
    ]
    results = [
        {
            "index": 0,
            "case_index": 0,
            "elapsed_ms": 10.0,
            "errors": [],
            "detail_checked": True,
            "detail_elapsed_ms": 4.0,
            "detail_errors": [],
        },
        {
            "index": 1,
            "case_index": 1,
            "elapsed_ms": 30.0,
            "errors": ["results must include at least one item"],
            "detail_checked": False,
            "detail_elapsed_ms": None,
            "detail_errors": ["results[0].faq_id is required when --require-detail is set"],
        },
        {
            "index": 2,
            "case_index": 0,
            "elapsed_ms": 20.0,
            "errors": [],
            "detail_checked": True,
            "detail_elapsed_ms": 6.0,
            "detail_errors": [],
        },
    ]

    assert smoke._case_result_summaries(cases, results, detail_required=True) == [
        {
            "case_index": 0,
            "case": {
                "query": "credit report",
                "corpus_id": "corp-a",
                "status": "published",
                "limit": 2,
                "require_results": True,
            },
            "requests": 2,
            "errors": {"count": 0, "rate": 0.0},
            "latency": {"count": 2, "p50_ms": 15.0, "p95_ms": 20.0, "max_ms": 20.0},
            "detail": {
                "checked": 2,
                "failures": 0,
                "latency": {"count": 2, "p50_ms": 5.0, "p95_ms": 6.0, "max_ms": 6.0},
            },
        },
        {
            "case_index": 1,
            "case": {
                "query": "mortgage dispute",
                "corpus_id": "corp-b",
                "status": "",
                "limit": 3,
                "require_results": True,
            },
            "requests": 1,
            "errors": {"count": 1, "rate": 1.0},
            "latency": {"count": 1, "p50_ms": 30.0, "p95_ms": 30.0, "max_ms": 30.0},
            "detail": {
                "checked": 0,
                "failures": 1,
                "latency": {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0},
            },
        },
    ]


def test_worst_case_summary_prefers_errors_then_latency():
    summary = {
        "cases": {
            "summaries": [
                {
                    "case_index": 0,
                    "errors": {"count": 0, "rate": 0.0},
                    "latency": {"p95_ms": 200.0, "max_ms": 250.0},
                },
                {
                    "case_index": 1,
                    "errors": {"count": 1, "rate": 0.5},
                    "latency": {"p95_ms": 100.0, "max_ms": 120.0},
                },
                {
                    "case_index": 2,
                    "errors": {"count": 1, "rate": 0.5},
                    "latency": {"p95_ms": 100.0, "max_ms": 120.0},
                },
                {
                    "case_index": 3,
                    "errors": {"count": 1, "rate": 0.5},
                    "latency": {"p95_ms": 150.0, "max_ms": 160.0},
                },
            ]
        }
    }

    tie_summary = {
        "cases": {
            "summaries": [
                {
                    "case_index": 1,
                    "errors": {"count": 1, "rate": 0.5},
                    "latency": {"p95_ms": 100.0, "max_ms": 120.0},
                },
                {
                    "case_index": 2,
                    "errors": {"count": 1, "rate": 0.5},
                    "latency": {"p95_ms": 100.0, "max_ms": 120.0},
                },
            ]
        }
    }

    assert smoke._worst_case_summary(summary)["case_index"] == 3
    assert smoke._worst_case_summary(tie_summary)["case_index"] == 1
    assert smoke._worst_case_summary({"cases": {"summaries": []}}) is None
    assert smoke._worst_case_summary({"cases": {"summaries": "not-a-list"}}) is None


def test_print_summary_includes_worst_case_signal(capsys):
    summary = {
        "ok": False,
        "requests": {"total": 3},
        "errors": {"count": 1},
        "latency": {"p95_ms": 50.0, "max_ms": 75.0},
        "detail": {"checked": 2, "failures": 1},
        "budgets": {"failures": ["error_rate exceeded 0.0"]},
        "cases": {
            "summaries": [
                {
                    "case_index": 0,
                    "errors": {"count": 0, "rate": 0.0},
                    "latency": {"p95_ms": 15.0, "max_ms": 20.0},
                },
                {
                    "case_index": 1,
                    "errors": {"count": 1, "rate": 1.0},
                    "latency": {"p95_ms": 30.0, "max_ms": 30.0},
                },
            ]
        },
    }

    smoke._print_summary(summary, as_json=False)

    assert capsys.readouterr().out.strip() == (
        "FAQ search hosted concurrency smoke: ok=False requests=3 errors=1 "
        "p95_ms=50.0 max_ms=75.0 detail_checked=2 detail_failures=1 "
        "budget_failures=1 worst_case_index=1 worst_case_errors=1 "
        "worst_case_p95_ms=30.0 worst_case_max_ms=30.0"
    )


def test_budget_summary_reports_error_and_latency_failures():
    summary = smoke._budget_summary(
        latency={"p95_ms": 50.0, "max_ms": 75.0},
        detail_latency={"count": 2, "p95_ms": 25.0, "max_ms": 30.0},
        case_summaries=[
            {"case_index": 0, "requests": 2, "errors": {"rate": 0.0}},
            {"case_index": 1, "requests": 2, "errors": {"rate": 0.5}},
        ],
        errors={"rate": 0.25},
        max_error_rate=0.0,
        max_case_error_rate=0.25,
        max_p95_ms=40.0,
        max_single_request_ms=100.0,
        max_detail_ms=20.0,
        max_case_p95_ms=None,
        max_case_single_request_ms=None,
    )

    assert summary == {
        "ok": False,
        "checks": [
            {"metric": "error_rate", "actual": 0.25, "max": 0.0, "ok": False},
            {"metric": "p95_ms", "actual": 50.0, "max": 40.0, "ok": False},
            {"metric": "max_ms", "actual": 75.0, "max": 100.0, "ok": True},
            {"metric": "detail_max_ms", "actual": 30.0, "max": 20.0, "ok": False},
            {
                "metric": "case_error_rate",
                "case_index": 0,
                "actual": 0.0,
                "max": 0.25,
                "ok": True,
            },
            {
                "metric": "case_error_rate",
                "case_index": 1,
                "actual": 0.5,
                "max": 0.25,
                "ok": False,
            },
        ],
        "failures": [
            "error_rate exceeded 0.0",
            "p95_ms exceeded 40.0",
            "detail_max_ms exceeded 20.0",
            "case_error_rate exceeded 0.25 for case 1",
        ],
    }


def test_budget_summary_fails_closed_when_detail_budget_has_no_detail_rows():
    summary = smoke._budget_summary(
        latency={"p95_ms": 50.0, "max_ms": 75.0},
        detail_latency={"count": 0, "p95_ms": 0.0, "max_ms": 0.0},
        case_summaries=[],
        errors={"rate": 0.0},
        max_error_rate=0.0,
        max_case_error_rate=None,
        max_p95_ms=None,
        max_single_request_ms=None,
        max_detail_ms=20.0,
        max_case_p95_ms=None,
        max_case_single_request_ms=None,
    )

    assert summary == {
        "ok": False,
        "checks": [
            {"metric": "error_rate", "actual": 0.0, "max": 0.0, "ok": True},
            {"metric": "detail_max_ms", "actual": None, "max": 20.0, "ok": False},
        ],
        "failures": ["detail_max_ms had no checked detail rows"],
    }


def test_budget_summary_fails_case_error_budget_without_request_samples():
    summary = smoke._budget_summary(
        latency={"p95_ms": 50.0, "max_ms": 75.0},
        detail_latency={"count": 0, "p95_ms": 0.0, "max_ms": 0.0},
        case_summaries=[
            {"case_index": 0, "requests": 1, "errors": {"rate": 0.0}},
            {"case_index": 1, "requests": 0, "errors": {"rate": 0.0}},
        ],
        errors={"rate": 0.0},
        max_error_rate=0.0,
        max_case_error_rate=0.0,
        max_p95_ms=None,
        max_single_request_ms=None,
        max_detail_ms=None,
        max_case_p95_ms=None,
        max_case_single_request_ms=None,
    )

    assert summary == {
        "ok": False,
        "checks": [
            {"metric": "error_rate", "actual": 0.0, "max": 0.0, "ok": True},
            {
                "metric": "case_error_rate",
                "case_index": 0,
                "actual": 0.0,
                "max": 0.0,
                "ok": True,
            },
            {
                "metric": "case_error_rate",
                "case_index": 1,
                "actual": None,
                "max": 0.0,
                "ok": False,
            },
        ],
        "failures": ["case_error_rate had no request samples for case 1"],
    }


def test_budget_summary_reports_case_latency_failures_when_aggregate_passes():
    summary = smoke._budget_summary(
        latency={"p95_ms": 80.0, "max_ms": 100.0},
        detail_latency={"count": 0, "p95_ms": 0.0, "max_ms": 0.0},
        case_summaries=[
            {
                "case_index": 0,
                "errors": {"rate": 0.0},
                "latency": {"count": 1, "p95_ms": 50.0, "max_ms": 60.0},
            },
            {
                "case_index": 1,
                "errors": {"rate": 0.0},
                "latency": {"count": 1, "p95_ms": 120.0, "max_ms": 180.0},
            },
        ],
        errors={"rate": 0.0},
        max_error_rate=0.0,
        max_case_error_rate=None,
        max_p95_ms=200.0,
        max_single_request_ms=200.0,
        max_detail_ms=None,
        max_case_p95_ms=100.0,
        max_case_single_request_ms=150.0,
    )

    assert summary == {
        "ok": False,
        "checks": [
            {"metric": "error_rate", "actual": 0.0, "max": 0.0, "ok": True},
            {"metric": "p95_ms", "actual": 80.0, "max": 200.0, "ok": True},
            {"metric": "max_ms", "actual": 100.0, "max": 200.0, "ok": True},
            {
                "metric": "case_p95_ms",
                "case_index": 0,
                "actual": 50.0,
                "max": 100.0,
                "ok": True,
            },
            {
                "metric": "case_p95_ms",
                "case_index": 1,
                "actual": 120.0,
                "max": 100.0,
                "ok": False,
            },
            {
                "metric": "case_max_ms",
                "case_index": 0,
                "actual": 60.0,
                "max": 150.0,
                "ok": True,
            },
            {
                "metric": "case_max_ms",
                "case_index": 1,
                "actual": 180.0,
                "max": 150.0,
                "ok": False,
            },
        ],
        "failures": [
            "case_p95_ms exceeded 100.0 for case 1",
            "case_max_ms exceeded 150.0 for case 1",
        ],
    }


def test_budget_summary_fails_case_latency_budget_without_samples():
    summary = smoke._budget_summary(
        latency={"p95_ms": 50.0, "max_ms": 75.0},
        detail_latency={"count": 0, "p95_ms": 0.0, "max_ms": 0.0},
        case_summaries=[
            {
                "case_index": 0,
                "errors": {"rate": 0.0},
                "latency": {"count": 1, "p95_ms": 50.0, "max_ms": 75.0},
            },
            {
                "case_index": 1,
                "errors": {"rate": 0.0},
                "latency": {"count": 0, "p95_ms": 0.0, "max_ms": 0.0},
            },
        ],
        errors={"rate": 0.0},
        max_error_rate=0.0,
        max_case_error_rate=None,
        max_p95_ms=None,
        max_single_request_ms=None,
        max_detail_ms=None,
        max_case_p95_ms=100.0,
        max_case_single_request_ms=None,
    )

    assert summary == {
        "ok": False,
        "checks": [
            {"metric": "error_rate", "actual": 0.0, "max": 0.0, "ok": True},
            {
                "metric": "case_p95_ms",
                "case_index": 0,
                "actual": 50.0,
                "max": 100.0,
                "ok": True,
            },
            {
                "metric": "case_p95_ms",
                "case_index": 1,
                "actual": None,
                "max": 100.0,
                "ok": False,
            },
        ],
        "failures": ["case_p95_ms had no latency samples for case 1"],
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


def test_run_one_fetches_and_validates_detail_when_required(monkeypatch):
    responses = iter([
        _json_response(_valid_payload()),
        _json_response(_valid_detail_payload()),
    ])
    urls = []

    def _fake_urlopen(request, **_kwargs):
        urls.append(request.full_url)
        return next(responses)

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    result = smoke._run_one(0, _args(require_detail=True))

    assert result["ok"] is True
    assert result["detail_checked"] is True
    assert result["detail_faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert result["detail_errors"] == []
    assert urls == [
        (
            "https://atlas.example.com/api/v1/content-ops/faq-deflection-search"
            "?q=mortgage+dispute&limit=5"
        ),
        (
            "https://atlas.example.com/api/v1/content-ops/faq-deflection-search/"
            "11111111-1111-1111-1111-111111111111"
        ),
    ]


def test_run_one_uses_detail_route_template_when_required(monkeypatch):
    responses = iter([
        _json_response(_valid_payload()),
        _json_response(_valid_detail_payload()),
    ])
    urls = []

    def _fake_urlopen(request, **_kwargs):
        urls.append(request.full_url)
        return next(responses)

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    result = smoke._run_one(
        0,
        _args(
            require_detail=True,
            detail_route="/api/v1/content-ops/faq-deflection-reports/{faq_id}",
        ),
    )

    assert result["ok"] is True
    assert urls[-1] == (
        "https://atlas.example.com/api/v1/content-ops/faq-deflection-reports/"
        "11111111-1111-1111-1111-111111111111"
    )


def test_run_one_requires_faq_id_for_detail_check(monkeypatch):
    payload = _valid_payload()
    del payload["results"][0]["faq_id"]
    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _json_response(payload),
    )

    result = smoke._run_one(0, _args(require_detail=True))

    assert result["ok"] is False
    assert result["detail_checked"] is False
    assert result["detail_faq_id"] is None
    assert result["detail_errors"] == [
        "results[0].faq_id is required when --require-detail is set"
    ]
    assert result["errors"] == result["detail_errors"]


def test_run_one_rejects_malformed_detail_envelope(monkeypatch):
    responses = iter([
        _json_response(_valid_payload()),
        _json_response(_valid_detail_payload("22222222-2222-2222-2222-222222222222")),
    ])
    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: next(responses),
    )

    result = smoke._run_one(0, _args(require_detail=True))

    assert result["ok"] is False
    assert result["detail_checked"] is True
    assert result["detail_errors"] == ["detail.id must match results[0].faq_id"]
    assert result["errors"] == ["detail.id must match results[0].faq_id"]


def test_run_one_records_detail_transport_failures(monkeypatch):
    calls = iter([
        _json_response(_valid_payload()),
        smoke.contract.urllib.error.URLError("detail unavailable"),
    ])

    def _fake_urlopen(*_args, **_kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    result = smoke._run_one(0, _args(require_detail=True))

    assert result["ok"] is False
    assert result["detail_checked"] is False
    assert result["detail_faq_id"] == "11111111-1111-1111-1111-111111111111"
    assert result["detail_errors"] == [
        "RuntimeError: route request failed: detail unavailable"
    ]
    assert result["errors"] == result["detail_errors"]


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


def test_run_one_rejects_seeded_expectation_drift(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.001]).__next__)
    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _json_response(_valid_payload()),
    )

    result = smoke._run_one(
        0,
        _args(),
        {
            "query": "mortgage dispute",
            "corpus_id": "corpus-1",
            "status": "",
            "limit": 5,
            "require_results": True,
            "expected_count": 2,
            "expected_first_account_id": "acct-2",
            "expected_first_corpus_id": "corpus-2",
            "expected_first_faq_id": "22222222-2222-2222-2222-222222222222",
        },
    )

    assert result["ok"] is False
    assert result["errors"] == [
        "expected count 2 but got 1",
        "expected first account_id 'acct-2' but got 'acct-1'",
        "expected first corpus_id 'corpus-2' but got 'corpus-1'",
        (
            "expected first faq_id '22222222-2222-2222-2222-222222222222' "
            "but got '11111111-1111-1111-1111-111111111111'"
        ),
    ]


def test_run_one_rejects_missing_expected_first_result(monkeypatch):
    monkeypatch.setattr(smoke.time, "perf_counter", iter([1.0, 1.001]).__next__)
    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _json_response(
            {"query": "mortgage dispute", "results": [], "count": 0}
        ),
    )

    result = smoke._run_one(
        0,
        _args(require_results=False),
        {
            "query": "mortgage dispute",
            "corpus_id": "corpus-1",
            "status": "",
            "limit": 5,
            "require_results": False,
            "expected_count": 0,
            "expected_first_faq_id": "11111111-1111-1111-1111-111111111111",
        },
    )

    assert result["ok"] is False
    assert result["errors"] == ["expected first result but none was returned"]


def test_run_one_rejects_seeded_detail_expectation_drift(monkeypatch):
    responses = iter([
        _json_response(_valid_payload()),
        _json_response(_valid_detail_payload()),
    ])
    monkeypatch.setattr(
        smoke.time,
        "perf_counter",
        iter([1.0, 1.001, 1.002, 1.003, 1.004]).__next__,
    )
    monkeypatch.setattr(
        smoke.contract.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: next(responses),
    )

    result = smoke._run_one(
        0,
        _args(require_detail=True),
        {
            "query": "mortgage dispute",
            "corpus_id": "corpus-1",
            "status": "",
            "limit": 5,
            "require_results": True,
            "expected_first_faq_id": "11111111-1111-1111-1111-111111111111",
            "expected_detail_account_id": "acct-2",
            "expected_detail_target_id": "support-corpus-1",
            "expected_detail_target_mode": "support_account",
            "expected_detail_title": "Expected SaaS FAQ",
            "expected_detail_status": "approved",
        },
    )

    assert result["ok"] is False
    assert result["detail_checked"] is True
    assert result["errors"] == [
        "detail.account_id expected 'acct-2' but got 'acct-1'",
        "detail.target_id expected 'support-corpus-1' but got 'corpus-1'",
        "detail.target_mode expected 'support_account' but got 'faq_report'",
        "detail.title expected 'Expected SaaS FAQ' but got 'Mortgage servicing issues'",
        "detail.status expected 'approved' but got 'published'",
    ]
    assert result["case"]["expected_detail_title"] == "Expected SaaS FAQ"


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


def test_main_rejects_detail_with_allowed_empty_results_before_network(tmp_path, monkeypatch, capsys):
    result_path = tmp_path / "hosted-concurrency.json"

    def _unexpected_urlopen(*_args, **_kwargs):
        raise AssertionError("preflight failures must not issue route requests")

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _unexpected_urlopen)

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--require-detail",
        "--allow-empty-results",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["ok"] is False
    assert payload["phase"] == "preflight"
    assert payload["require_detail"] is True
    assert payload["require_results"] is False
    assert payload["requests"]["total"] == 0
    assert payload["preflight_errors"] == [
        "--require-detail requires result rows; remove --allow-empty-results"
    ]
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight"


def test_main_rejects_case_empty_results_with_detail_before_network(tmp_path, monkeypatch, capsys):
    case_file = _write_case_file(
        tmp_path,
        [{"query": "unknown billing workflow", "require_results": False}],
    )
    result_path = tmp_path / "hosted-concurrency.json"

    def _unexpected_urlopen(*_args, **_kwargs):
        raise AssertionError("preflight failures must not issue route requests")

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _unexpected_urlopen)

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--case-file",
        str(case_file),
        "--require-detail",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["ok"] is False
    assert payload["phase"] == "preflight"
    assert payload["require_detail"] is True
    assert payload["cases"]["total"] == 0
    assert payload["preflight_errors"] == [
        "case[0].require_results cannot be false when --require-detail is set"
    ]
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight"


def test_main_rejects_detail_budget_without_detail_check_before_network(tmp_path, monkeypatch, capsys):
    result_path = tmp_path / "hosted-concurrency.json"

    def _unexpected_urlopen(*_args, **_kwargs):
        raise AssertionError("preflight failures must not issue route requests")

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _unexpected_urlopen)

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--max-detail-ms",
        "250",
        "--output-result",
        str(result_path),
        "--json",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["ok"] is False
    assert payload["phase"] == "preflight"
    assert payload["require_detail"] is False
    assert payload["detail"]["latency"] == {
        "count": 0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "max_ms": 0.0,
    }
    assert payload["preflight_errors"] == ["--max-detail-ms requires --require-detail"]
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight"


def test_main_result_includes_case_summaries_for_mixed_cases(tmp_path, monkeypatch):
    case_file = _write_case_file(
        tmp_path,
        [
            {"query": "credit report", "corpus_id": "corp-a", "limit": 2},
            {"query": "mortgage dispute", "corpus_id": "corp-b", "limit": 3},
        ],
    )
    result_path = tmp_path / "hosted-concurrency.json"

    def _fake_urlopen(request, **_kwargs):
        query = smoke.contract.urllib.parse.parse_qs(
            smoke.contract.urllib.parse.urlsplit(request.full_url).query
        )["q"][0]
        if query == "credit report":
            return _json_response(_valid_payload())
        return _json_response({"query": query, "results": [], "count": 0})

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--case-file",
        str(case_file),
        "--requests",
        "2",
        "--concurrency",
        "2",
        "--output-result",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["query"] == "multiple case-file queries"
    assert payload["query_mode"] == "case_file_mixed"
    assert payload["query_count"] == 2
    assert payload["cases"]["summaries"][0]["requests"] == 1
    assert payload["cases"]["summaries"][0]["errors"] == {"count": 0, "rate": 0.0}
    assert payload["cases"]["summaries"][0]["latency"]["count"] == 1
    assert payload["cases"]["summaries"][1]["requests"] == 1
    assert payload["cases"]["summaries"][1]["errors"] == {"count": 1, "rate": 1.0}
    assert payload["cases"]["summaries"][1]["latency"]["count"] == 1


def test_main_fails_case_error_budget_when_aggregate_error_budget_passes(tmp_path, monkeypatch):
    case_file = _write_case_file(
        tmp_path,
        [
            {"query": "credit report", "corpus_id": "corp-a", "limit": 2},
            {"query": "mortgage dispute", "corpus_id": "corp-b", "limit": 3},
        ],
    )
    result_path = tmp_path / "hosted-concurrency.json"

    def _fake_urlopen(request, **_kwargs):
        query = smoke.contract.urllib.parse.parse_qs(
            smoke.contract.urllib.parse.urlsplit(request.full_url).query
        )["q"][0]
        if query == "credit report":
            return _json_response(_valid_payload())
        return _json_response({"query": query, "results": [], "count": 0})

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--case-file",
        str(case_file),
        "--requests",
        "2",
        "--concurrency",
        "2",
        "--max-error-rate",
        "1",
        "--max-case-error-rate",
        "0",
        "--output-result",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["errors"]["rate"] == 0.5
    assert payload["budgets"]["checks"][-2:] == [
        {
            "metric": "case_error_rate",
            "case_index": 0,
            "actual": 0.0,
            "max": 0.0,
            "ok": True,
        },
        {
            "metric": "case_error_rate",
            "case_index": 1,
            "actual": 1.0,
            "max": 0.0,
            "ok": False,
        },
    ]
    assert payload["budgets"]["failures"] == [
        "case_error_rate exceeded 0.0 for case 1"
    ]


def test_main_case_summaries_cover_cases_beyond_preview_cap(tmp_path, monkeypatch):
    case_file = _write_case_file(
        tmp_path,
        [
            {
                "query": f"query {index}",
                "corpus_id": f"corp-{index}",
                "limit": 1,
            }
            for index in range(21)
        ],
    )
    result_path = tmp_path / "hosted-concurrency.json"

    def _fake_urlopen(request, **_kwargs):
        query = smoke.contract.urllib.parse.parse_qs(
            smoke.contract.urllib.parse.urlsplit(request.full_url).query
        )["q"][0]
        if query == "query 20":
            return _json_response({"query": query, "results": [], "count": 0})
        return _json_response(_valid_payload())

    monkeypatch.setattr(smoke.contract.urllib.request, "urlopen", _fake_urlopen)

    code = smoke.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--case-file",
        str(case_file),
        "--requests",
        "21",
        "--concurrency",
        "3",
        "--output-result",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["cases"]["total"] == 21
    assert len(payload["cases"]["items"]) == 20
    assert payload["cases"]["truncated"] is True
    assert len(payload["cases"]["summaries"]) == 21
    assert payload["cases"]["summaries"][20]["case"]["query"] == "query 20"
    assert payload["cases"]["summaries"][20]["errors"] == {"count": 1, "rate": 1.0}
    assert smoke._worst_case_summary(payload)["case_index"] == 20


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
