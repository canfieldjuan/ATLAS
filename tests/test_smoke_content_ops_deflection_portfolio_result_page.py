from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_deflection_portfolio_result_page.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_deflection_portfolio_result_page",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


SNAPSHOT = {
    "summary": {
        "generated": 2,
        "drafted_answer_count": 1,
        "no_proven_answer_count": 1,
    },
    "top_questions": [
        {
            "rank": 1,
            "question": "How do I export reports?",
            "weighted_frequency": 3,
            "customer_wording": "How do I export reports?",
        }
    ],
}


@pytest.fixture(autouse=True)
def _isolate_deflection_result_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL",
        "ATLAS_API_BASE_URL",
        "ATLAS_B2B_JWT",
        "ATLAS_TOKEN",
        "ATLAS_ACCOUNT_ID",
        "ATLAS_FAQ_SEARCH_ACCOUNT_ID",
        "ATLAS_DEFLECTION_REQUEST_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ATLAS_DISABLE_DOTENV", "1")


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--result-url",
        "https://portfolio.example.com/deflection/result/content-ops-123",
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--account-id",
        "acct-123",
        "--request-id",
        "content-ops-123",
        "--output-result",
        str(tmp_path / "result.json"),
    ]


def _page_html() -> str:
    return """
    <main data-atlas-deflection-result>
      <section data-atlas-deflection-request-id="content-ops-123"></section>
      <a data-atlas-deflection-unlock
         data-checkout-source="content_ops_deflection_report"
         data-checkout-request_id="content-ops-123">Unlock full report</a>
    </main>
    """


def _paid_page_html() -> str:
    return """
    <main>
      <div>FULL BACKLOG REPORT</div>
      <h1>Your complete Support Tax report is ready.</h1>
      <section>Paid report contents for content-ops-123</section>
    </main>
    """


PAID_ARTIFACT = {
    "markdown": "# Paid report",
    "faq_result": {"items": []},
}


def test_load_dotenv_files_honors_disable_flag(monkeypatch) -> None:
    calls: list[Path] = []
    monkeypatch.setenv("ATLAS_DISABLE_DOTENV", "1")
    monkeypatch.setattr(smoke, "load_dotenv", lambda path, **_kwargs: calls.append(path))

    smoke._load_dotenv_files()

    assert calls == []


class FakeResponse:
    def __init__(self, status: int, body: str | dict[str, Any]):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        if isinstance(self._body, str):
            return self._body.encode("utf-8")
        return json.dumps(self._body).encode("utf-8")


def _fake_open(sequence: list[tuple[int, str | dict[str, Any]]], calls: list[dict[str, Any]]):
    def _open(request, *, timeout):
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": dict(request.header_items()),
            "timeout": timeout,
        })
        status, body = sequence.pop(0)
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "error", {}, None)
        return FakeResponse(status, body)

    return _open


def test_validate_args_fails_closed_for_missing_and_unsafe_inputs() -> None:
    args = smoke._build_parser().parse_args([
        "--result-url",
        "http://localhost:3000/result",
        "--base-url",
        "http://127.0.0.1:8000",
        "--token",
        "",
        "--account-id",
        "",
        "--request-id",
        "",
        "--timeout",
        "0",
        "--snapshot-path-template",
        "snapshot",
        "--artifact-path-template",
        "/artifact",
    ])

    assert smoke._validate_args(args) == [
        "--result-url must be an absolute HTTPS URL",
        "--base-url must be an absolute HTTPS URL",
        "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
        "ATLAS_ACCOUNT_ID, ATLAS_FAQ_SEARCH_ACCOUNT_ID, or --account-id is required",
        "ATLAS_DEFLECTION_REQUEST_ID or --request-id is required",
        "--timeout must be a positive finite number",
        "--snapshot-path-template must start with /",
        "--snapshot-path-template must include {request_id}",
        "--artifact-path-template must include {request_id}",
    ]


def test_validate_args_rejects_account_id_in_public_result_url(tmp_path) -> None:
    args = smoke._build_parser().parse_args([
        *_base_args(tmp_path),
        "--result-url",
        "https://portfolio.example.com/deflection/result/content-ops-123?account_id=acct-123",
    ])

    assert smoke._validate_args(args) == [
        "--result-url must not include account_id",
        "--result-url must not include the account_id value",
    ]


def test_preflight_only_writes_missing_inputs_without_network(monkeypatch, tmp_path, capsys) -> None:
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("preflight must not call network")

    monkeypatch.setattr(smoke, "_open_http_request", _unexpected)
    result_path = tmp_path / "preflight.json"

    code = smoke.main(["--preflight-only", "--output-result", str(result_path), "--json"])

    assert code == 2
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert printed == payload
    assert "ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL or --result-url is required" in payload["preflight_errors"]
    assert payload["inputs"]["token_present"] is False


def test_page_errors_require_portfolio_hooks_and_metadata_values() -> None:
    errors = smoke._page_errors(
        "<main data-atlas-deflection-result>content-ops-123</main>",
        request_id="content-ops-123",
        account_id="acct-123",
    )

    assert "portfolio result page missing marker: data-atlas-deflection-unlock" in errors
    assert "portfolio result page missing marker: content_ops_deflection_report" in errors


def test_page_errors_bind_checkout_request_to_unlock_cta() -> None:
    html = """
    <main data-atlas-deflection-result>
      <section data-atlas-deflection-request-id="content-ops-123">
        content-ops-123 content_ops_deflection_report request_id
      </section>
      <a data-atlas-deflection-unlock
         data-checkout-source="content_ops_deflection_report"
         data-checkout-request_id="stale-request">Unlock full report</a>
    </main>
    """

    errors = smoke._page_errors(
        html,
        request_id="content-ops-123",
        account_id="acct-123",
    )

    assert errors == [
        "portfolio result page unlock CTA data-checkout-request_id must be content-ops-123",
    ]


def test_paid_page_errors_require_paid_markers_and_reject_locked_state() -> None:
    errors = smoke._paid_page_errors(
        """
        <main>
          SNAPSHOT TEMPORARILY UNAVAILABLE
          <a data-atlas-deflection-unlock>Unlock</a>
        </main>
        """,
        request_id="content-ops-123",
        account_id="acct-123",
    )

    assert errors == [
        "portfolio paid result page missing marker: FULL BACKLOG REPORT",
        "portfolio paid result page missing marker: Your complete Support Tax report is ready",
        "portfolio paid result page missing marker: Paid report contents",
        "portfolio paid result page rendered unavailable state",
        "portfolio paid result page must not render unlock CTA",
        "portfolio paid result page missing request_id value",
    ]


def test_paid_page_errors_allow_inert_unlock_selector_text() -> None:
    errors = smoke._paid_page_errors(
        """
        <main>
          <div>FULL BACKLOG REPORT</div>
          <h1>Your complete Support Tax report is ready.</h1>
          <section>Paid report contents for content-ops-123</section>
          <script>
            document.querySelector("[data-atlas-deflection-unlock]");
          </script>
        </main>
        """,
        request_id="content-ops-123",
        account_id="acct-123",
    )

    assert errors == []


def test_page_errors_reject_browser_exposed_account_id() -> None:
    html = """
    <main data-atlas-deflection-result>
      <section data-atlas-deflection-request-id="content-ops-123">
        content-ops-123 content_ops_deflection_report request_id acct-123
      </section>
      <a data-atlas-deflection-unlock
         data-checkout-source="content_ops_deflection_report"
         data-checkout-request_id="content-ops-123"
         data-checkout-account_id="acct-123">Unlock full report</a>
    </main>
    """

    errors = smoke._page_errors(
        html,
        request_id="content-ops-123",
        account_id="acct-123",
    )

    assert "portfolio result page unlock CTA must not expose account_id" in errors
    assert "portfolio result page must not expose account_id marker" in errors
    assert "portfolio result page must not expose account_id value" in errors


def test_snapshot_errors_reject_paid_report_leaks() -> None:
    snapshot = {
        **SNAPSHOT,
        "top_questions": [
            {
                **SNAPSHOT["top_questions"][0],
                "answer": "Open Reports.",
                "source_ids": ["ticket-1"],
            }
        ],
    }

    assert smoke._snapshot_errors(snapshot) == [
        "snapshot leaked paid-report fields: answer, source_ids"
    ]


def test_main_validates_page_snapshot_and_locked_artifact(monkeypatch, tmp_path, capsys) -> None:
    calls: list[dict[str, Any]] = []

    def _open(request, *, timeout):
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": dict(request.header_items()),
            "timeout": timeout,
        })
        if request.full_url.endswith("/artifact"):
            raise urllib.error.HTTPError(request.full_url, 403, "locked", {}, None)
        if "/snapshot" in request.full_url:
            return FakeResponse(200, SNAPSHOT)
        return FakeResponse(200, _page_html())

    monkeypatch.setattr(smoke, "_open_http_request", _open)
    result_path = tmp_path / "result.json"

    code = smoke.main([*_base_args(tmp_path), "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert printed == payload
    assert payload["ok"] is True
    assert payload["page"]["status"] == 200
    assert payload["page"]["markers"]["locked_markers_present"] is True
    assert payload["page"]["markers"]["paid_markers_present"] is False
    assert payload["page"]["markers"]["unlock_cta_present"] is True
    assert payload["snapshot"]["status"] == 200
    assert payload["artifact"]["status"] == 403
    assert calls == [
        {
            "url": "https://portfolio.example.com/deflection/result/content-ops-123",
            "method": "GET",
            "headers": {"Accept": "text/html"},
            "timeout": 30.0,
        },
        {
            "url": "https://atlas.example.com/api/v1/content-ops/deflection-reports/content-ops-123/snapshot",
            "method": "GET",
            "headers": {"Accept": "application/json", "Authorization": "Bearer secret-token"},
            "timeout": 30.0,
        },
        {
            "url": "https://atlas.example.com/api/v1/content-ops/deflection-reports/content-ops-123/artifact",
            "method": "GET",
            "headers": {"Accept": "application/json", "Authorization": "Bearer secret-token"},
            "timeout": 30.0,
        },
    ]


def test_main_validates_paid_page_and_unlocked_artifact(monkeypatch, tmp_path, capsys) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open([
            (200, _paid_page_html()),
            (200, SNAPSHOT),
            (200, PAID_ARTIFACT),
        ], calls),
    )
    result_path = tmp_path / "result.json"

    code = smoke.main([*_base_args(tmp_path), "--artifact-state", "unlocked", "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert printed == payload
    assert payload["ok"] is True
    assert payload["inputs"]["artifact_state"] == "unlocked"
    assert payload["page"]["status"] == 200
    assert payload["page"]["markers"]["paid_markers_present"] is True
    assert payload["page"]["markers"]["unavailable_state_present"] is False
    assert payload["page"]["markers"]["unlock_cta_present"] is False
    assert payload["snapshot"]["status"] == 200
    assert payload["artifact"]["status"] == 200


def test_main_rejects_malformed_paid_artifact_in_unlocked_mode(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open([
            (200, _paid_page_html()),
            (200, SNAPSHOT),
            (200, {"markdown": ""}),
        ], calls),
    )

    code = smoke.main([*_base_args(tmp_path), "--artifact-state", "unlocked"])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "paid artifact markdown must be a non-empty string",
        "paid artifact faq_result must be an object",
    ]


def test_main_reports_unlock_regression_when_artifact_is_200(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        _fake_open([
            (200, _page_html()),
            (200, SNAPSHOT),
            (200, {"markdown": "# paid"}),
        ], calls),
    )

    code = smoke.main(_base_args(tmp_path))
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["artifact endpoint must return 403 before payment; got 200"]


def test_fetch_json_rejects_malformed_json(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke,
        "_open_http_request",
        lambda _request, *, timeout: FakeResponse(200, "{not-json"),
    )

    result = smoke._fetch_json("https://atlas.example.com/snapshot", token="token", timeout=1)

    assert result.status is None
    assert result.payload is None
    assert result.errors
