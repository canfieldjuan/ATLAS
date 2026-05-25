import importlib.util
from io import BytesIO
import json
import sys
import urllib.error
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "check_content_ops_faq_search_route_contract.py"
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


def _valid_payload():
    return {
        "query": "mortgage payment dispute",
        "results": [
            {
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


def test_main_requires_token(monkeypatch, capsys):
    monkeypatch.delenv("ATLAS_B2B_JWT", raising=False)
    monkeypatch.delenv("ATLAS_TOKEN", raising=False)
    _set_argv(monkeypatch, "--base-url", "https://atlas.example.com")

    assert _MODULE.main() == 2
    assert "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required" in capsys.readouterr().out


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


def test_main_writes_success_result_without_token(monkeypatch, tmp_path):
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
        "errors": [],
        "limit": 5,
        "ok": True,
        "phase": "contract",
        "query": "mortgage payment dispute",
        "require_results": True,
        "route": "/api/v1/content-ops/faq-deflection-search",
        "status": "",
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
