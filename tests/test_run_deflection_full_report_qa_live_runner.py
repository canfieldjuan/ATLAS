from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_deflection_full_report_qa_live_runner.py"
SPEC = importlib.util.spec_from_file_location(
    "run_deflection_full_report_qa_live_runner",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


REQUEST_ID = "content-ops-1234567890"
TOKEN = "secret-token-1234567890"


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


def _http_error(url: str, status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, status, "error", {}, None)


def _report_model() -> dict[str, object]:
    return {
        "schema_version": "deflection.v1",
        "title": "Support Ticket Deflection Report",
        "summary": {"generated": 2},
        "sections": [
            {
                "id": "support_tax",
                "priority": 10,
                "data": {
                    "repeat_ticket_count": 8,
                    "non_repeat_ticket_count": 0,
                    "generated_question_count": 2,
                    "assisted_contact_cost": 13.5,
                    "estimated_support_cost": 108.0,
                    "source_date_window": {},
                    "drafted_answer_count": 1,
                    "no_proven_answer_count": 1,
                    "ticket_source_count": 8,
                },
            },
            {
                "id": "seo_targets",
                "priority": 20,
                "data": {
                    "phrases": ["export attribution reports", "report download"],
                    "total_phrase_count": 2,
                    "displayed_phrase_count": 2,
                    "omitted_phrase_count": 0,
                    "limit": 50,
                },
            },
            {
                "id": "ranked_questions",
                "priority": 30,
                "data": {
                    "rows": [
                        {"rank": 1, "question": "How do I export attribution reports?"},
                        {"rank": 2, "question": "How do I invite teammates?"},
                    ],
                },
            },
            {
                "id": "question_details",
                "priority": 50,
                "data": {
                    "rows": [
                        {
                            "rank": 1,
                            "question": "How do I export attribution reports?",
                            "source_ids": ["zd-100"],
                        },
                        {
                            "rank": 2,
                            "question": "How do I invite teammates?",
                            "source_ids": ["fd-200"],
                        },
                    ],
                },
            },
            {
                "id": "complete_evidence",
                "priority": 90,
                "data": {
                    "question_count": 2,
                    "evidence_row_count": 8,
                    "source_id_count": 8,
                    "surfaces": ["export"],
                },
            },
        ],
    }


def _evidence_export() -> dict[str, object]:
    return {
        "schema_version": "deflection_evidence.v1",
        "summary": {
            "question_count": 2,
            "evidence_row_count": 8,
            "source_id_count": 8,
            "drafted_answer_count": 1,
            "no_proven_answer_count": 1,
        },
        "questions": [{}, {}],
        "evidence_rows": [
            {"source_id": "zd-100"},
            {"source_id": "fd-200"},
            {},
            {},
            {},
            {},
            {},
            {},
        ],
    }


def _process_contract(**contract_overrides: Any) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "report_model_schema_version": "deflection.v1",
        "report_model_contract": runner.EXPECTED_REPORT_MODEL_CONTRACT,
        "evidence_export_schema_version": "deflection_evidence.v1",
        "paid_artifact_requires": {
            "report_model": "object",
            "evidence_export": "object",
        },
    }
    contract.update(contract_overrides)
    return {
        "schema_version": "deflection_report_process.v1",
        "service": "content_ops_deflection_reports",
        "contract": contract,
        "routes": {
            "process_contract": "/api/v1/content-ops/deflection-reports/process-contract",
            "snapshot": "/api/v1/content-ops/deflection-reports/{request_id}/snapshot",
            "artifact": "/api/v1/content-ops/deflection-reports/{request_id}/artifact",
            "report_model": "/api/v1/content-ops/deflection-reports/{request_id}/report-model",
        },
    }


def _pdf_text() -> str:
    return """
    Support Ticket Deflection Report

    Support Tax Confirmation
    This report found 8 question-level repeat tickets across 2 ranked questions.
    At the benchmark, that repeated-question work sizes to about $108.
    1 questions have publishable answers and 1 questions still have no proven answer.
    The report is grounded in 8 source tickets.

    Ranked Question Opportunities
    2 ranked questions appear in the curated PDF.
    How do I export attribution reports?
    How do I invite teammates?

    Question Details and Evidence
    How do I export attribution reports?
    How do I invite teammates?
    The PDF is curated for sharing. Use the complete evidence export for the
    uncapped audit trail.
    """


def _write_pdf_inputs(
    tmp_path: Path,
    *,
    text: str | None = None,
    pdf_bytes_payload: bytes | None = None,
) -> tuple[Path, Path]:
    pdf_bytes = tmp_path / "report.pdf"
    pdf_text = tmp_path / "report_pdf_text.txt"
    pdf_bytes.write_bytes(
        pdf_bytes_payload
        or b"%PDF-1.7\n% synthetic live-runner bytes\n%%EOF\n"
    )
    pdf_text.write_text(_pdf_text() if text is None else text, encoding="utf-8")
    return pdf_bytes, pdf_text


def _base_args(tmp_path: Path) -> list[str]:
    pdf_bytes, pdf_text = _write_pdf_inputs(tmp_path)
    return [
        "--base-url",
        "https://atlas.example.com",
        "--token",
        TOKEN,
        "--request-id",
        REQUEST_ID,
        "--pdf-bytes",
        str(pdf_bytes),
        "--pdf-text",
        str(pdf_text),
        "--output-result",
        str(tmp_path / "result.json"),
    ]


def _artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "report_model": _report_model(),
        "evidence_export": _evidence_export(),
        "markdown": "# Paid report",
        "faq_result": {"items": []},
    }
    payload.update(overrides)
    return payload


def _pdf_renderable_artifact(**overrides: Any) -> dict[str, Any]:
    payload = _artifact(**overrides)
    for section in payload["report_model"]["sections"]:
        section["surfaces"] = ["pdf"]
    return payload


def _pdf_report_model() -> dict[str, Any]:
    return _pdf_renderable_artifact()["report_model"]


def _render_pdf_bytes(artifact: dict[str, Any]) -> bytes:
    pytest.importorskip("fpdf")
    from atlas_brain.deflection_pdf_renderer import (  # noqa: PLC0415
        render_deflection_full_report_pdf,
    )

    return render_deflection_full_report_pdf(artifact)


def _payload(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))


def test_live_runner_extracts_pdf_text_from_renderer_bytes_by_default(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    pdf_bytes = tmp_path / "report.pdf"
    pdf_bytes.write_bytes(_render_pdf_bytes(_pdf_renderable_artifact()))
    calls: list[str] = []

    def _urlopen(request, *, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith(f"/{REQUEST_ID}/report-model"):
            return FakeResponse(200, _pdf_report_model())
        if request.full_url.endswith(f"/{REQUEST_ID}/artifact"):
            return FakeResponse(200, _pdf_renderable_artifact())
        raise AssertionError(request.full_url)

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        TOKEN,
        "--request-id",
        REQUEST_ID,
        "--pdf-bytes",
        str(pdf_bytes),
        "--output-result",
        str(tmp_path / "result.json"),
        "--json",
    ])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = _payload(tmp_path)
    assert printed == payload
    assert payload["ok"] is True
    assert payload["pdf_text"] == {
        "source": "extracted_from_pdf_bytes",
        "verified_from_pdf_bytes": True,
    }
    assert payload["inputs"]["pdf_text_present"] is False
    assert payload["scorecard"]["artifacts"]["pdf"]["text_chars"] > 0
    assert calls == [
        "https://atlas.example.com/api/v1/content-ops/deflection-reports/process-contract",
        f"https://atlas.example.com/api/v1/content-ops/deflection-reports/{REQUEST_ID}/report-model",
        f"https://atlas.example.com/api/v1/content-ops/deflection-reports/{REQUEST_ID}/artifact",
    ]


def test_live_runner_fails_real_renderer_pdf_leaks_before_network(
    monkeypatch,
    tmp_path,
) -> None:
    artifact = _pdf_renderable_artifact()
    rows = artifact["report_model"]["sections"][2]["data"]["rows"]
    rows[0]["question"] = (
        "How do I export content-ops-deadbeefLEAK reports for buyer@example.com?"
    )
    pdf_bytes = tmp_path / "report.pdf"
    pdf_bytes.write_bytes(_render_pdf_bytes(artifact))

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("rendered PDF leaks must fail before network")

    monkeypatch.setattr(runner.urllib.request, "urlopen", _unexpected)

    code = runner.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        TOKEN,
        "--request-id",
        REQUEST_ID,
        "--pdf-bytes",
        str(pdf_bytes),
        "--output-result",
        str(tmp_path / "result.json"),
    ])

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["ok"] is False
    assert payload["errors"] == [
        "pdf bytes contain sensitive pattern: request_id",
        "pdf bytes contain sensitive pattern: customer_email",
    ]
    encoded = json.dumps(payload, sort_keys=True)
    assert "content-ops-deadbeefLEAK" not in encoded
    assert "buyer@example.com" not in encoded


def test_live_runner_fetches_live_json_and_writes_redacted_scorecard(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    calls: list[dict[str, Any]] = []

    def _urlopen(request, *, timeout):
        calls.append({
            "url": request.full_url,
            "headers": dict(request.header_items()),
            "timeout": timeout,
        })
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith(f"/{REQUEST_ID}/report-model"):
            return FakeResponse(200, _report_model())
        if request.full_url.endswith(f"/{REQUEST_ID}/artifact"):
            return FakeResponse(200, _artifact())
        raise AssertionError(request.full_url)

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main([*_base_args(tmp_path), "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = _payload(tmp_path)
    assert printed == payload
    assert payload["ok"] is True
    assert payload["fetches"]["process_contract"] == {"status": 200, "ok": True}
    assert payload["fetches"]["report_model"] == {"status": 200}
    assert payload["fetches"]["artifact"] == {"status": 200}
    assert payload["scorecard"]["ok"] is True
    assert payload["pdf_text"] == {
        "source": "operator_asserted",
        "verified_from_pdf_bytes": False,
    }
    assert payload["scorecard"]["artifacts"]["pdf"]["text_chars"] > 0
    assert [call["url"] for call in calls] == [
        "https://atlas.example.com/api/v1/content-ops/deflection-reports/process-contract",
        f"https://atlas.example.com/api/v1/content-ops/deflection-reports/{REQUEST_ID}/report-model",
        f"https://atlas.example.com/api/v1/content-ops/deflection-reports/{REQUEST_ID}/artifact",
    ]
    assert calls[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    encoded = json.dumps(payload, sort_keys=True)
    assert TOKEN not in encoded
    assert REQUEST_ID not in encoded
    assert str(tmp_path) not in encoded
    assert "https://atlas.example.com" not in encoded


def test_pdf_text_extraction_failure_fails_before_network(monkeypatch, tmp_path) -> None:
    pdf_bytes, _pdf_text = _write_pdf_inputs(
        tmp_path,
        pdf_bytes_payload=b"not a pdf",
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("PDF text extraction failure must fail before network")

    monkeypatch.setattr(runner.urllib.request, "urlopen", _unexpected)

    code = runner.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        TOKEN,
        "--request-id",
        REQUEST_ID,
        "--pdf-bytes",
        str(pdf_bytes),
        "--output-result",
        str(tmp_path / "result.json"),
    ])

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["ok"] is False
    assert payload["inputs"]["pdf_text_present"] is False
    assert payload["errors"] == [
        "pdf text extraction requires PDF bytes",
        "pdf text extraction must be non-empty",
    ]
    assert str(tmp_path) not in json.dumps(payload, sort_keys=True)


def test_preflight_only_reports_validation_errors_without_network(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("preflight must not call network")

    monkeypatch.setattr(runner.urllib.request, "urlopen", _unexpected)

    code = runner.main([
        "--preflight-only",
        "--base-url",
        "http://127.0.0.1:8000",
        "--token",
        "",
        "--request-id",
        "",
        "--pdf-bytes",
        str(tmp_path / "missing.pdf"),
        "--pdf-text",
        str(tmp_path / "missing.txt"),
        "--report-model-path-template",
        "report-model",
        "--artifact-path-template",
        "/artifact",
        "--output-result",
        str(tmp_path / "result.json"),
        "--json",
    ])

    assert code == 2
    printed = json.loads(capsys.readouterr().out)
    payload = _payload(tmp_path)
    assert printed == payload
    assert payload["ok"] is False
    assert payload["inputs"]["token_present"] is False
    assert "--base-url must be an absolute HTTPS URL" in payload["errors"]
    assert "--request-id is required" in payload["errors"]
    assert "--pdf-bytes must be a readable file" in payload["errors"]
    assert "--pdf-text must be a readable file" in payload["errors"]
    encoded = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in encoded


def test_empty_pdf_text_fails_before_network(monkeypatch, tmp_path) -> None:
    pdf_bytes, pdf_text = _write_pdf_inputs(tmp_path, text="  \n")

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("empty PDF text must fail before network")

    monkeypatch.setattr(runner.urllib.request, "urlopen", _unexpected)

    code = runner.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        TOKEN,
        "--request-id",
        REQUEST_ID,
        "--pdf-bytes",
        str(pdf_bytes),
        "--pdf-text",
        str(pdf_text),
        "--output-result",
        str(tmp_path / "result.json"),
    ])

    assert code == 1
    assert _payload(tmp_path)["errors"] == ["pdf text extraction must be non-empty"]


def test_pdf_byte_leak_fails_before_network(monkeypatch, tmp_path) -> None:
    pdf_bytes, pdf_text = _write_pdf_inputs(
        tmp_path,
        pdf_bytes_payload=(
            b"%PDF-1.7\ncontent-ops-deadbeefLEAK buyer@example.com\n%%EOF\n"
        ),
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("PDF byte leaks must fail before network")

    monkeypatch.setattr(runner.urllib.request, "urlopen", _unexpected)

    code = runner.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        TOKEN,
        "--request-id",
        REQUEST_ID,
        "--pdf-bytes",
        str(pdf_bytes),
        "--pdf-text",
        str(pdf_text),
        "--output-result",
        str(tmp_path / "result.json"),
    ])

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["ok"] is False
    assert payload["errors"] == [
        "pdf bytes contain sensitive pattern: request_id",
        "pdf bytes contain sensitive pattern: customer_email",
    ]
    encoded = json.dumps(payload, sort_keys=True)
    assert "content-ops-deadbeefLEAK" not in encoded
    assert "buyer@example.com" not in encoded


def test_unsupported_pdf_stream_filter_fails_before_network(monkeypatch, tmp_path) -> None:
    pdf_bytes, _pdf_text = _write_pdf_inputs(
        tmp_path,
        pdf_bytes_payload=(
            b"%PDF-1.7\n"
            b"1 0 obj\n<< /Length 35 >>\nstream\n"
            b"(Support Ticket Deflection Report) Tj\n"
            b"endstream\nendobj\n"
            b"2 0 obj\n<< /Filter /ASCIIHexDecode /Length 20 >>\nstream\n"
            b"636f6e74656e742d6f70732d6465616462656566>\n"
            b"endstream\nendobj\n%%EOF\n"
        ),
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("unsupported PDF streams must fail before network")

    monkeypatch.setattr(runner.urllib.request, "urlopen", _unexpected)

    code = runner.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        TOKEN,
        "--request-id",
        REQUEST_ID,
        "--pdf-bytes",
        str(pdf_bytes),
        "--output-result",
        str(tmp_path / "result.json"),
    ])

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["ok"] is False
    assert payload["errors"] == [
        "pdf stream uses unsupported filter: ASCIIHexDecode",
        "pdf text extraction must be non-empty",
    ]


def test_artifact_missing_evidence_export_fails_closed(monkeypatch, tmp_path) -> None:
    def _urlopen(request, *, timeout):
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith("/report-model"):
            return FakeResponse(200, _report_model())
        return FakeResponse(200, {"report_model": _report_model()})

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main(_base_args(tmp_path))

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["ok"] is False
    assert payload["fetches"]["artifact"] == {"status": 200}
    assert payload["errors"] == ["artifact.evidence_export must be an object"]
    assert "evidence_rows" not in json.dumps(payload)


def test_process_contract_drift_fails_before_paid_artifact_fetches(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    def _urlopen(request, *, timeout):
        del timeout
        calls.append(request.full_url)
        if request.full_url.endswith("/process-contract"):
            stale_shape = dict(runner.EXPECTED_REPORT_MODEL_CONTRACT)
            stale_shape["sections"] = [
                section
                for section in runner.EXPECTED_REPORT_MODEL_CONTRACT["sections"]
                if section["id"] != "complete_evidence"
            ]
            return FakeResponse(
                200,
                _process_contract(report_model_contract=stale_shape),
            )
        raise AssertionError("paid artifact endpoints must not be fetched")

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main(_base_args(tmp_path))

    assert code == 1
    payload = _payload(tmp_path)
    assert calls == [
        "https://atlas.example.com/api/v1/content-ops/deflection-reports/process-contract"
    ]
    assert payload["fetches"] == {
        "process_contract": {
            "status": 200,
            "ok": False,
            "errors": [
                "contract.report_model_contract must match current deflection.v1 shape"
            ],
        }
    }
    assert payload["errors"] == [
        "process contract preflight failed: "
        "contract.report_model_contract must match current deflection.v1 shape"
    ]


def test_artifact_report_model_drift_fails_closed(monkeypatch, tmp_path) -> None:
    drifted_model = dict(_report_model())
    drifted_model["title"] = "Different report"

    def _urlopen(request, *, timeout):
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith("/report-model"):
            return FakeResponse(200, _report_model())
        return FakeResponse(200, _artifact(report_model=drifted_model))

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main(_base_args(tmp_path))

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["errors"] == ["artifact.report_model must match the report-model route"]


@pytest.mark.parametrize(
    "raw_report_model",
    [
        None,
        "not-an-object",
        ["not", "an", "object"],
    ],
)
def test_artifact_report_model_must_be_object_when_present(
    monkeypatch,
    tmp_path,
    raw_report_model,
) -> None:
    def _urlopen(request, *, timeout):
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith("/report-model"):
            return FakeResponse(200, _report_model())
        return FakeResponse(200, _artifact(report_model=raw_report_model))

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main(_base_args(tmp_path))

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["errors"] == ["artifact.report_model must be an object when present"]


def test_invalid_json_response_fails_without_raw_body(monkeypatch, tmp_path) -> None:
    def _urlopen(request, *, timeout):
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith("/report-model"):
            return FakeResponse(200, "{not json")
        return FakeResponse(200, _artifact())

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main(_base_args(tmp_path))

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["fetches"]["report_model"] == {
        "status": 200,
        "errors": ["invalid_json"],
    }
    assert payload["errors"] == ["report-model endpoint response must be a JSON object"]
    assert "{not json" not in json.dumps(payload)


def test_http_error_status_is_sanitized(monkeypatch, tmp_path) -> None:
    def _urlopen(request, *, timeout):
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith("/report-model"):
            raise _http_error(request.full_url, 403)
        return FakeResponse(200, _artifact())

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)

    code = runner.main(_base_args(tmp_path))

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["fetches"]["report_model"] == {
        "status": 403,
        "errors": ["http_error"],
    }
    assert payload["errors"] == ["report-model endpoint must return 200, got 403"]
    encoded = json.dumps(payload, sort_keys=True)
    assert TOKEN not in encoded
    assert REQUEST_ID not in encoded


def test_output_sanitizer_rewrites_sensitive_scorecard_and_returns_one(
    monkeypatch,
    tmp_path,
) -> None:
    def _urlopen(request, *, timeout):
        if request.full_url.endswith("/process-contract"):
            return FakeResponse(200, _process_contract())
        if request.full_url.endswith("/report-model"):
            return FakeResponse(200, _report_model())
        return FakeResponse(200, _artifact())

    def _leaky_scorecard(**_kwargs):
        return {
            "ok": True,
            "leaked": (
                "Bearer leaked-token content-ops-outputleak "
                "cs_live_outputleak ticket-outputleak"
            ),
        }

    monkeypatch.setattr(runner.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(runner, "build_pdf_export_scorecard", _leaky_scorecard)

    code = runner.main(_base_args(tmp_path))

    assert code == 1
    payload = _payload(tmp_path)
    assert payload["ok"] is False
    assert payload["inputs"] == {
        "base_url_present": True,
        "token_present": True,
        "request_id_present": True,
        "pdf_bytes_present": True,
        "pdf_text_present": True,
        "preflight_only": False,
    }
    assert "runner output contains sensitive pattern: bearer_token" in payload["errors"]
    assert "runner output contains sensitive pattern: stripe_checkout_session_id" in payload["errors"]
    assert "runner output contains sensitive pattern: source_id" in payload["errors"]
    encoded = json.dumps(payload, sort_keys=True)
    assert "Bearer leaked-token" not in encoded
    assert "content-ops-outputleak" not in encoded
    assert "cs_live_outputleak" not in encoded
    assert "ticket-outputleak" not in encoded
