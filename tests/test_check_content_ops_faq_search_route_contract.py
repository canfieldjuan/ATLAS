import importlib.util
from io import BytesIO
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "check_content_ops_faq_search_route_contract.py"
)
_HANDOFF_DOC_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "extraction"
    / "validation"
    / "content_ops_faq_search_route_contract_handoff.md"
)
_SPEC = importlib.util.spec_from_file_location(
    "check_content_ops_faq_search_route_contract",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["check_content_ops_faq_search_route_contract.py", *args])


def _set_clock(monkeypatch, *values):
    readings = iter(values)
    monkeypatch.setattr(_MODULE, "_now_ms", lambda: next(readings))


def _valid_payload():
    return {
        "query": "mortgage payment dispute",
        "results": [
            {
                "faq_id": "11111111-1111-1111-1111-111111111111",
                "question": "How do I dispute a mortgage payment error?",
                "answer_summary": "Check the statement, gather records, then contact support.",
                "topic": "Mortgage servicing issues",
                "source_ids": ["CFPB-1"],
                "ticket_count": 12,
                "score": 42,
            }
        ],
        "count": 1,
    }


def _valid_detail_payload():
    return {
        "account_id": "acct-1",
        "id": "11111111-1111-1111-1111-111111111111",
        "target_id": "support-account-1",
        "target_mode": "support_account",
        "title": "Support FAQ",
        "markdown": "# Support FAQ",
        "items": [_valid_detail_item()],
        "source_count": 1,
        "ticket_source_count": 1,
        "output_checks": {},
        "warnings": [],
        "metadata": {"corpus_id": "corpus-1"},
        "status": "approved",
    }


def _valid_detail_item():
    return {
        "topic": "Mortgage servicing issues",
        "question": "How do I dispute a mortgage payment error?",
        "question_source": "customer_wording",
        "summary": "Customers ask how to dispute mortgage payment errors.",
        "frequency": 12,
        "weighted_frequency": 12,
        "ticket_count": 12,
        "opportunity_score": 42,
        "failure_risk_score": 1,
        "failure_risk_signals": ["repeat_question"],
        "answer": "Customers mention mortgage payment disputes.",
        "steps": ["Review the statement.", "Contact support with records."],
        "action_items": ["Review the statement.", "Contact support with records."],
        "answer_evidence_status": "resolution_evidence",
        "resolution_source_count": 1,
        "when_to_contact_support": "Contact support if the payment still looks wrong.",
        "evidence_quotes": ["`CFPB-1`: payment dispute"],
        "source_ids": ["CFPB-1"],
        "source_labels": ["`CFPB-1`"],
        "source_type_counts": {"support_ticket": 1},
        "weighted_source_volume_by_type": {"support_ticket": 12},
        "term_mappings": [_valid_term_mapping()],
        "evidence_count": 1,
        "displayed_evidence_count": 1,
    }


def _valid_term_mapping():
    return {
        "customer_term": "payment error",
        "documentation_term": "payment dispute",
        "suggestion": "Use payment error as alternate phrasing.",
        "source_id_count": 1,
        "zero_result_source_count": 0,
        "failure_risk_score": 1,
        "failure_risk_signals": ["repeat_question"],
        "opportunity_score": 42,
        "first_source_id": "CFPB-1",
    }


def _expected_checker_item_keys():
    return (
        set(_MODULE.DETAIL_ITEM_STRING_FIELDS)
        | set(_MODULE.DETAIL_ITEM_INT_FIELDS)
        | set(_MODULE.DETAIL_ITEM_STRING_LIST_FIELDS)
        | set(_MODULE.DETAIL_ITEM_COUNT_MAP_FIELDS)
        | {"term_mappings"}
    )


def _expected_checker_term_mapping_keys():
    return (
        set(_MODULE.DETAIL_TERM_MAPPING_STRING_FIELDS)
        | set(_MODULE.DETAIL_TERM_MAPPING_INT_FIELDS)
        | {"failure_risk_signals"}
    )


def _producer_detail_item():
    result = build_ticket_faq_markdown(
        [
            {
                "source_id": "search-export-1",
                "source_type": "search_log",
                "text": "How do I export attribution report?",
                "zero_results": "true",
                "source_weight": "20",
            },
            {
                "source_id": "ticket-email-1",
                "source_type": "support_ticket",
                "source_title": "Email update",
                "text": "How do I change my email?",
                "resolution_text": (
                    "Open account settings, choose Profile, update the email "
                    "address, then confirm the verification email"
                ),
            },
        ],
        title="Support Ticket FAQ Report",
        max_items=2,
        max_evidence_per_item=2,
        support_contact="https://example.com/support",
        documentation_terms=("Download report",),
        vocabulary_gap_rules=(("export", "download report"),),
    )
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["items"]
    return payload["items"][0]


def test_build_url_encodes_query_and_optional_filters():
    url = _MODULE._build_url(
        base_url="https://atlas.example.com/",
        route="/api/v1/content-ops/faq-deflection-search",
        query="mortgage payment dispute",
        corpus_id="corpus 1",
        status="approved",
        limit=5,
    )

    assert url == (
        "https://atlas.example.com/api/v1/content-ops/faq-deflection-search"
        "?q=mortgage+payment+dispute&limit=5&corpus_id=corpus+1&status=approved"
    )


def test_build_detail_url_defaults_to_search_route_child():
    url = _MODULE._build_detail_url(
        base_url="https://atlas.example.com/",
        route="/api/v1/content-ops/faq-deflection-search",
        detail_route="",
        faq_id="11111111-1111-1111-1111-111111111111",
    )

    assert url == (
        "https://atlas.example.com/api/v1/content-ops/faq-deflection-search/"
        "11111111-1111-1111-1111-111111111111"
    )


def test_build_detail_url_allows_template_override():
    url = _MODULE._build_detail_url(
        base_url="https://atlas.example.com",
        route="/ignored",
        detail_route="/api/v2/faqs/{faq_id}/full",
        faq_id="id with space",
    )

    assert url == "https://atlas.example.com/api/v2/faqs/id%20with%20space/full"


def test_contract_handoff_doc_matches_checker_fields_and_semantics():
    text = _HANDOFF_DOC_PATH.read_text(encoding="utf-8")

    assert '{"query": "<query>", "results": [], "count": 0}' in text
    assert "No-match is not an error." in text
    assert "not a percentage" in text
    assert "not the FAQ opportunity score" in text
    assert "count is the number of returned rows" in text

    for field in ("faq_id", *list(_MODULE.RESULT_FIELDS)):
        assert f"`{field}`" in text
    for field in _MODULE.DETAIL_FIELDS:
        assert f"`{field}`" in text
    for field in (
        *_MODULE.DETAIL_ITEM_STRING_FIELDS,
        *_MODULE.DETAIL_ITEM_INT_FIELDS,
        *_MODULE.DETAIL_ITEM_STRING_LIST_FIELDS,
        *_MODULE.DETAIL_ITEM_COUNT_MAP_FIELDS,
        "term_mappings",
    ):
        assert f"`{field}`" in text
    for field in (
        *_MODULE.DETAIL_TERM_MAPPING_STRING_FIELDS,
        *_MODULE.DETAIL_TERM_MAPPING_INT_FIELDS,
        "failure_risk_signals",
    ):
        assert f"`{field}`" in text


def test_validate_envelope_rejects_bool_count():
    payload = _valid_payload()
    payload["count"] = True

    assert _MODULE._validate_envelope(payload, require_results=False) == [
        "count must be an integer"
    ]


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"query": 1}, "query must be a string"),
        ({"results": {}}, "results must be a list"),
        ({"count": 2}, "count must match len(results)"),
    ],
)
def test_validate_envelope_rejects_envelope_contract_drift(patch, expected):
    payload = _valid_payload()
    payload.update(patch)

    assert expected in _MODULE._validate_envelope(payload, require_results=False)


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([], "results must include at least one item"),
        (["bad"], "results[0] must be an object"),
        ([{**_valid_payload()["results"][0], "source_ids": "bad"}], "results[0].source_ids must be a list"),
        ([{**_valid_payload()["results"][0], "ticket_count": True}], "results[0].ticket_count must be an integer"),
        ([{**_valid_payload()["results"][0], "score": False}], "results[0].score must be an integer"),
        ([{**_valid_payload()["results"][0], "question": 1}], "results[0].question must be a string"),
        ([{**_valid_payload()["results"][0], "answer_summary": 2}], "results[0].answer_summary must be a string"),
        ([{**_valid_payload()["results"][0], "topic": 3}], "results[0].topic must be a string"),
    ],
)
def test_validate_envelope_rejects_required_result_contract_drift(results, expected):
    payload = {"query": "reset", "results": results, "count": len(results)}

    assert expected in _MODULE._validate_envelope(payload, require_results=True)


def test_validate_envelope_requires_demo_fields_when_requested():
    payload = {
        "query": "reset",
        "results": [{"question": "How do I reset?"}],
        "count": 1,
    }

    assert _MODULE._validate_envelope(payload, require_results=True) == [
        "results[0].answer_summary is required",
        "results[0].topic is required",
        "results[0].source_ids is required",
        "results[0].ticket_count is required",
        "results[0].score is required",
    ]


@pytest.mark.parametrize("field", _MODULE.RESULT_FIELDS)
def test_validate_envelope_requires_each_demo_field_when_requested(field):
    result = dict(_valid_payload()["results"][0])
    result.pop(field)
    payload = {"query": "reset", "results": [result], "count": 1}

    assert (
        f"results[0].{field} is required"
        in _MODULE._validate_envelope(payload, require_results=True)
    )


def test_main_requires_base_url(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("ATLAS_API_BASE_URL", raising=False)
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "",
        "--token",
        "token-123",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 2
    assert "ATLAS_API_BASE_URL or --base-url is required" in capsys.readouterr().out
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "preflight"
    assert payload["errors"] == ["ATLAS_API_BASE_URL or --base-url is required"]


def test_validate_detail_accepts_full_generated_faq_payload():
    assert _MODULE._validate_detail(
        _valid_detail_payload(),
        faq_id="11111111-1111-1111-1111-111111111111",
    ) == []


def test_validate_detail_item_matches_generated_faq_producer_shape():
    item = _producer_detail_item()

    assert set(item) == _expected_checker_item_keys()
    assert item["term_mappings"]
    assert set(item["term_mappings"][0]) == _expected_checker_term_mapping_keys()
    assert _MODULE._validate_detail_item(item, index=0) == []


def test_validate_detail_accepts_expected_seed_fields():
    assert _MODULE._validate_detail(
        _valid_detail_payload(),
        faq_id="11111111-1111-1111-1111-111111111111",
        expected={
            "account_id": "acct-1",
            "target_id": "support-account-1",
            "target_mode": "support_account",
            "title": "Support FAQ",
            "status": "approved",
        },
    ) == []


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"id": "22222222-2222-2222-2222-222222222222"}, "detail.id must match results[0].faq_id"),
        ({"markdown": 1}, "detail.markdown must be a string"),
        ({"items": {}}, "detail.items must be a list"),
        ({"items": []}, "detail.items must include at least one item"),
        ({"source_count": True}, "detail.source_count must be an integer"),
        ({"output_checks": []}, "detail.output_checks must be an object"),
    ],
)
def test_validate_detail_rejects_contract_drift(patch, expected):
    payload = _valid_detail_payload()
    payload.update(patch)

    assert expected in _MODULE._validate_detail(
        payload,
        faq_id="11111111-1111-1111-1111-111111111111",
    )


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"account_id": "acct-2"}, "detail.account_id expected 'acct-2' but got 'acct-1'"),
        (
            {"target_id": "support-account-2"},
            "detail.target_id expected 'support-account-2' but got 'support-account-1'",
        ),
        (
            {"target_mode": "org"},
            "detail.target_mode expected 'org' but got 'support_account'",
        ),
        ({"title": "Seed FAQ"}, "detail.title expected 'Seed FAQ' but got 'Support FAQ'"),
        ({"status": "draft"}, "detail.status expected 'draft' but got 'approved'"),
    ],
)
def test_validate_detail_rejects_expected_seed_field_mismatches(expected, message):
    assert message in _MODULE._validate_detail(
        _valid_detail_payload(),
        faq_id="11111111-1111-1111-1111-111111111111",
        expected=expected,
    )


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"question": 1}, "detail.items[0].question must be a string"),
        ({"summary": None}, "detail.items[0].summary must be a string"),
        ({"ticket_count": True}, "detail.items[0].ticket_count must be an integer"),
        ({"steps": "read docs"}, "detail.items[0].steps must be a list"),
        ({"source_ids": [1]}, "detail.items[0].source_ids[0] must be a string"),
        ({"source_type_counts": []}, "detail.items[0].source_type_counts must be an object"),
        ({"source_type_counts": {"support_ticket": True}}, "detail.items[0].source_type_counts.support_ticket must be an integer"),
        ({"question_source": "internal_taxonomy"}, "detail.items[0].question_source must be one of ['customer_wording', 'source_policy']"),
        ({"answer_evidence_status": "guessed"}, "detail.items[0].answer_evidence_status must be one of ['draft_needs_review', 'resolution_evidence']"),
        ({"term_mappings": {}}, "detail.items[0].term_mappings must be a list"),
    ],
)
def test_validate_detail_rejects_generated_item_contract_drift(patch, expected):
    payload = _valid_detail_payload()
    payload["items"][0].update(patch)

    assert expected in _MODULE._validate_detail(
        payload,
        faq_id="11111111-1111-1111-1111-111111111111",
    )


def test_validate_detail_requires_generated_item_fields():
    payload = _valid_detail_payload()
    del payload["items"][0]["action_items"]

    assert "detail.items[0].action_items is required" in _MODULE._validate_detail(
        payload,
        faq_id="11111111-1111-1111-1111-111111111111",
    )


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        ("bad", "detail.items[0].term_mappings[0] must be an object"),
        ({"customer_term": 1}, "detail.items[0].term_mappings[0].customer_term must be a string"),
        ({"source_id_count": True}, "detail.items[0].term_mappings[0].source_id_count must be an integer"),
        ({"failure_risk_signals": [1]}, "detail.items[0].term_mappings[0].failure_risk_signals[0] must be a string"),
    ],
)
def test_validate_detail_rejects_term_mapping_contract_drift(mapping, expected):
    payload = _valid_detail_payload()
    if isinstance(mapping, dict):
        payload["items"][0]["term_mappings"][0].update(mapping)
    else:
        payload["items"][0]["term_mappings"][0] = mapping

    assert expected in _MODULE._validate_detail(
        payload,
        faq_id="11111111-1111-1111-1111-111111111111",
    )


def test_main_requires_token(monkeypatch, capsys):
    monkeypatch.delenv("ATLAS_B2B_JWT", raising=False)
    monkeypatch.delenv("ATLAS_TOKEN", raising=False)
    _set_argv(monkeypatch, "--base-url", "https://atlas.example.com")

    assert _MODULE.main() == 2
    assert "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required" in capsys.readouterr().out


def test_main_requires_query(monkeypatch, capsys, tmp_path):
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--query",
        "",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 2
    assert "ATLAS_FAQ_SEARCH_QUERY or --query is required" in capsys.readouterr().out
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "preflight"
    assert payload["errors"] == ["ATLAS_FAQ_SEARCH_QUERY or --query is required"]


def test_main_requires_positive_limit(monkeypatch, capsys, tmp_path):
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--limit",
        "0",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 2
    assert "--limit must be positive" in capsys.readouterr().out
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "preflight"
    assert payload["errors"] == ["--limit must be positive"]


def test_main_returns_failure_for_bad_contract(monkeypatch, capsys):
    monkeypatch.setattr(
        _MODULE,
        "_fetch_json",
        lambda url, *, token, timeout: {"query": "reset", "results": [], "count": 1},
    )
    _set_argv(monkeypatch, "--base-url", "https://atlas.example.com", "--token", "token-123")

    assert _MODULE.main() == 1
    assert "count must match len(results)" in capsys.readouterr().out


def test_main_reports_invalid_env_limit_through_argparse(monkeypatch):
    monkeypatch.setenv("ATLAS_API_BASE_URL", "https://atlas.example.com")
    monkeypatch.setenv("ATLAS_B2B_JWT", "token-123")
    monkeypatch.setenv("ATLAS_FAQ_SEARCH_LIMIT", "bad")
    _set_argv(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _MODULE.main()

    assert exc.value.code == 2


def test_main_reports_invalid_env_latency_budget_through_argparse(monkeypatch):
    monkeypatch.setenv("ATLAS_API_BASE_URL", "https://atlas.example.com")
    monkeypatch.setenv("ATLAS_B2B_JWT", "token-123")
    monkeypatch.setenv("ATLAS_FAQ_SEARCH_MAX_SEARCH_MS", "bad")
    _set_argv(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        _MODULE.main()

    assert exc.value.code == 2


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def test_fetch_json_uses_urlopen_and_returns_object(monkeypatch):
    calls = []

    def _urlopen(request, *, timeout):
        calls.append((request, timeout))
        return _Response(b'{"query": "reset", "results": [], "count": 0}')

    monkeypatch.setattr(_MODULE.urllib.request, "urlopen", _urlopen)

    assert _MODULE._fetch_json("https://atlas.example.com/search", token="tok", timeout=3) == {
        "query": "reset",
        "results": [],
        "count": 0,
    }
    assert calls[0][0].headers["Authorization"] == "Bearer tok"
    assert calls[0][1] == 3


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            urllib.error.HTTPError(
                "https://atlas.example.com/search",
                401,
                "Unauthorized",
                {},
                BytesIO(b"unauthorized"),
            ),
            "route returned HTTP 401: unauthorized",
        ),
        (urllib.error.URLError("connection refused"), "route request failed: connection refused"),
        (_Response(b"<html>bad</html>"), "route did not return JSON"),
        (_Response(b'["not", "object"]'), "route returned non-object JSON"),
    ],
)
def test_fetch_json_reports_route_failures(monkeypatch, failure, expected):
    def _urlopen(request, *, timeout):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(_MODULE.urllib.request, "urlopen", _urlopen)

    with pytest.raises(RuntimeError, match=expected):
        _MODULE._fetch_json("https://atlas.example.com/search", token="tok", timeout=3)


def test_main_uses_urlopen_transport_for_success(monkeypatch, capsys):
    calls = []

    def _urlopen(request, *, timeout):
        calls.append((request, timeout))
        return _Response(json.dumps(_valid_payload()).encode("utf-8"))

    monkeypatch.setattr(_MODULE.urllib.request, "urlopen", _urlopen)
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--query",
        "mortgage payment dispute",
        "--require-results",
    )

    assert _MODULE.main() == 0
    assert len(calls) == 1
    assert calls[0][0].full_url == (
        "https://atlas.example.com/api/v1/content-ops/faq-deflection-search"
        "?q=mortgage+payment+dispute&limit=5"
    )
    assert calls[0][0].headers["Authorization"] == "Bearer token-123"
    assert calls[0][1] == 10.0
    assert "FAQ search route contract passed" in capsys.readouterr().out


def test_main_uses_urlopen_transport_for_http_failure(monkeypatch, capsys, tmp_path):
    def _urlopen(request, *, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "Server Error",
            {},
            BytesIO(b"<html>server error</html>"),
        )

    monkeypatch.setattr(_MODULE.urllib.request, "urlopen", _urlopen)
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 1
    output = capsys.readouterr().out
    assert "route returned HTTP 500: <html>server error</html>" in output
    assert "secret-token" not in output
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "request"
    assert payload["errors"] == ["route returned HTTP 500: <html>server error</html>"]
    assert "secret-token" not in result_path.read_text(encoding="utf-8")


def test_main_checks_route_and_prints_summary(monkeypatch, capsys):
    calls = []

    def _fake_fetch_json(url, *, token, timeout):
        calls.append((url, token, timeout))
        return _valid_payload()

    monkeypatch.setattr(_MODULE, "_fetch_json", _fake_fetch_json)
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--query",
        "mortgage payment dispute",
        "--require-results",
    )

    assert _MODULE.main() == 0
    assert calls == [
        (
            "https://atlas.example.com/api/v1/content-ops/faq-deflection-search"
            "?q=mortgage+payment+dispute&limit=5",
            "token-123",
            10.0,
        )
    ]
    assert "FAQ search route contract passed" in capsys.readouterr().out


def test_main_passes_when_latency_budgets_are_met(monkeypatch, capsys):
    _set_clock(monkeypatch, 100.0, 112.5)
    monkeypatch.setattr(
        _MODULE,
        "_fetch_json",
        lambda url, *, token, timeout: _valid_payload(),
    )
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--max-search-ms",
        "20",
        "--max-total-ms",
        "25",
    )

    assert _MODULE.main() == 0
    output = capsys.readouterr().out
    assert "total_elapsed_ms=12.500" in output


def test_main_fails_when_search_latency_budget_is_exceeded(monkeypatch, capsys):
    _set_clock(monkeypatch, 100.0, 140.0)
    monkeypatch.setattr(
        _MODULE,
        "_fetch_json",
        lambda url, *, token, timeout: _valid_payload(),
    )
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--max-search-ms",
        "25",
    )

    assert _MODULE.main() == 1
    assert (
        "search latency 40.000 ms exceeds --max-search-ms 25.000 ms"
        in capsys.readouterr().out
    )


def test_main_checks_detail_when_requested(monkeypatch, capsys):
    calls = []

    def _fake_fetch_json(url, *, token, timeout):
        calls.append((url, token, timeout))
        if url.endswith("/11111111-1111-1111-1111-111111111111"):
            return _valid_detail_payload()
        return _valid_payload()

    monkeypatch.setattr(_MODULE, "_fetch_json", _fake_fetch_json)
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--query",
        "mortgage payment dispute",
        "--require-results",
        "--require-detail",
    )

    assert _MODULE.main() == 0
    assert calls == [
        (
            "https://atlas.example.com/api/v1/content-ops/faq-deflection-search"
            "?q=mortgage+payment+dispute&limit=5",
            "token-123",
            10.0,
        ),
        (
            "https://atlas.example.com/api/v1/content-ops/faq-deflection-search/"
            "11111111-1111-1111-1111-111111111111",
            "token-123",
            10.0,
        ),
    ]
    assert "detail_checked=True" in capsys.readouterr().out


def test_main_fails_when_detail_or_total_latency_budget_is_exceeded(monkeypatch, capsys):
    _set_clock(monkeypatch, 0.0, 10.0, 10.0, 45.0)

    def _fake_fetch_json(url, *, token, timeout):
        if url.endswith("/11111111-1111-1111-1111-111111111111"):
            return _valid_detail_payload()
        return _valid_payload()

    monkeypatch.setattr(_MODULE, "_fetch_json", _fake_fetch_json)
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--require-results",
        "--require-detail",
        "--max-detail-ms",
        "30",
        "--max-total-ms",
        "40",
    )

    assert _MODULE.main() == 1
    output = capsys.readouterr().out
    assert "detail latency 35.000 ms exceeds --max-detail-ms 30.000 ms" in output
    assert "total latency 45.000 ms exceeds --max-total-ms 40.000 ms" in output


def test_main_rejects_detail_latency_budget_without_detail_check(monkeypatch, capsys):
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--max-detail-ms",
        "10",
    )

    assert _MODULE.main() == 2
    assert "--max-detail-ms requires --require-detail" in capsys.readouterr().out


def test_main_rejects_expected_detail_without_detail_check(monkeypatch, capsys):
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--expected-detail-account-id",
        "acct-1",
    )

    assert _MODULE.main() == 2
    assert "expected detail checks require --require-detail" in capsys.readouterr().out


def test_main_rejects_non_finite_latency_budget(monkeypatch, capsys):
    monkeypatch.setattr(
        _MODULE,
        "_fetch_json",
        lambda url, *, token, timeout: pytest.fail("preflight should stop before fetch"),
    )
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--max-search-ms",
        "nan",
    )

    assert _MODULE.main() == 2
    assert "--max-search-ms must be finite and positive" in capsys.readouterr().out


def test_main_requires_faq_id_for_detail_check(monkeypatch, capsys):
    payload = _valid_payload()
    del payload["results"][0]["faq_id"]
    monkeypatch.setattr(_MODULE, "_fetch_json", lambda url, *, token, timeout: payload)
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--require-results",
        "--require-detail",
    )

    assert _MODULE.main() == 1
    assert "results[0].faq_id is required for detail check" in capsys.readouterr().out


def test_main_reports_detail_contract_failure(monkeypatch, capsys):
    def _fake_fetch_json(url, *, token, timeout):
        if url.endswith("/11111111-1111-1111-1111-111111111111"):
            return {**_valid_detail_payload(), "items": {}}
        return _valid_payload()

    monkeypatch.setattr(_MODULE, "_fetch_json", _fake_fetch_json)
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--require-results",
        "--require-detail",
    )

    assert _MODULE.main() == 1
    assert "detail.items must be a list" in capsys.readouterr().out


def test_main_reports_expected_detail_mismatch(monkeypatch, capsys):
    def _fake_fetch_json(url, *, token, timeout):
        if url.endswith("/11111111-1111-1111-1111-111111111111"):
            return _valid_detail_payload()
        return _valid_payload()

    monkeypatch.setattr(_MODULE, "_fetch_json", _fake_fetch_json)
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "token-123",
        "--require-results",
        "--require-detail",
        "--expected-detail-account-id",
        "acct-2",
    )

    assert _MODULE.main() == 1
    assert "detail.account_id expected 'acct-2' but got 'acct-1'" in capsys.readouterr().out


def test_main_writes_success_result_without_token(monkeypatch, tmp_path):
    _set_clock(monkeypatch, 100.0, 112.5)
    monkeypatch.setattr(
        _MODULE,
        "_fetch_json",
        lambda url, *, token, timeout: _valid_payload(),
    )
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--query",
        "mortgage payment dispute",
        "--corpus-id",
        "corpus-1",
        "--require-results",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload == {
        "base_url": "https://atlas.example.com",
        "corpus_id": "corpus-1",
        "count": 1,
        "detail_checked": False,
        "detail_route": "/api/v1/content-ops/faq-deflection-search/{faq_id}",
        "errors": [],
        "limit": 5,
        "ok": True,
        "phase": "contract",
        "query": "mortgage payment dispute",
        "require_detail": False,
        "require_results": True,
        "route": "/api/v1/content-ops/faq-deflection-search",
        "search_elapsed_ms": 12.5,
        "status": "",
        "total_elapsed_ms": 12.5,
    }
    assert "secret-token" not in result_path.read_text(encoding="utf-8")


def test_main_writes_contract_failure_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _MODULE,
        "_fetch_json",
        lambda url, *, token, timeout: {"query": "reset", "results": [], "count": 1},
    )
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 1
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["phase"] == "contract"
    assert payload["errors"] == ["count must match len(results)"]
    assert payload["detail_checked"] is False
    assert "secret-token" not in result_path.read_text(encoding="utf-8")


def test_main_writes_preflight_failure_result(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ATLAS_B2B_JWT", raising=False)
    monkeypatch.delenv("ATLAS_TOKEN", raising=False)
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 2
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["phase"] == "preflight"
    assert payload["errors"] == ["ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required"]
    assert "token is required" in capsys.readouterr().out


def test_main_writes_request_failure_result_without_token(monkeypatch, tmp_path):
    def _fake_fetch_json(url, *, token, timeout):
        raise RuntimeError("route returned HTTP 401: unauthorized")

    monkeypatch.setattr(_MODULE, "_fetch_json", _fake_fetch_json)
    result_path = tmp_path / "faq-search-route-result.json"
    _set_argv(
        monkeypatch,
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--output-result",
        str(result_path),
    )

    assert _MODULE.main() == 1
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["phase"] == "request"
    assert payload["errors"] == ["route returned HTTP 401: unauthorized"]
    assert "secret-token" not in result_path.read_text(encoding="utf-8")


def test_main_does_not_print_token_on_http_failure(monkeypatch, capsys):
    def _fake_fetch_json(url, *, token, timeout):
        raise RuntimeError("route returned HTTP 401: unauthorized")

    monkeypatch.setattr(_MODULE, "_fetch_json", _fake_fetch_json)
    _set_argv(monkeypatch, "--base-url", "https://atlas.example.com", "--token", "secret-token")

    assert _MODULE.main() == 1
    output = capsys.readouterr().out
    assert "route returned HTTP 401" in output
    assert "secret-token" not in output
