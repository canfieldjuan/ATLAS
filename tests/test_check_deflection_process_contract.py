from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_deflection_process_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "check_deflection_process_contract",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class FakeResponse:
    def __init__(self, status: int, body: str | dict[str, Any] | list[Any]):
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


def _contract(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "deflection_report_process.v1",
        "service": "content_ops_deflection_reports",
        "contract": {
            "report_model_schema_version": "deflection.v1",
            "report_model_contract": checker.EXPECTED_REPORT_MODEL_CONTRACT,
            "evidence_export_schema_version": "deflection_evidence.v1",
            "paid_artifact_requires": {
                "report_model": "object",
                "evidence_export": "object",
            },
        },
        "routes": _routes(),
    }
    payload.update(overrides)
    return payload


def _routes(**overrides: str) -> dict[str, str]:
    routes = {
        "process_contract": "/api/v1/content-ops/deflection-reports/process-contract",
        "snapshot": "/api/v1/content-ops/deflection-reports/{request_id}/snapshot",
        "artifact": "/api/v1/content-ops/deflection-reports/{request_id}/artifact",
        "report_model": "/api/v1/content-ops/deflection-reports/{request_id}/report-model",
        "delete": "/api/v1/content-ops/deflection-reports/{request_id}",
    }
    routes.update(overrides)
    return routes


def _run(monkeypatch, body: Any, tmp_path: Path, *, status: int = 200):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["authorization"] = request.headers.get("Authorization")
        return FakeResponse(status, body)

    monkeypatch.setattr(checker.urllib.request, "urlopen", fake_urlopen)
    output = tmp_path / "result.json"
    code = checker.main([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--output-result",
        str(output),
    ])
    return code, json.loads(output.read_text(encoding="utf-8")), seen


def test_process_contract_checker_accepts_current_contract(monkeypatch, tmp_path):
    code, payload, seen = _run(monkeypatch, _contract(), tmp_path)

    assert code == 0
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["observed"] == {
        "schema_version": "deflection_report_process.v1",
        "service": "content_ops_deflection_reports",
        "report_model_schema_version": "deflection.v1",
        "report_model_contract": checker.EXPECTED_REPORT_MODEL_CONTRACT,
        "evidence_export_schema_version": "deflection_evidence.v1",
        "paid_artifact_requires": {
            "report_model": "object",
            "evidence_export": "object",
        },
        "routes": _routes(),
    }
    assert seen["url"] == (
        "https://atlas.example.com/api/v1/content-ops/"
        "deflection-reports/process-contract"
    )
    assert seen["authorization"] == "Bearer secret-token"


def test_process_contract_checker_fails_on_missing_route(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout):
        del timeout
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(checker.urllib.request, "urlopen", fake_urlopen)
    output = tmp_path / "result.json"
    code = checker.main([
        "--base-url",
        "https://atlas.example.com",
        "--output-result",
        str(output),
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "process contract endpoint must return 200, got 404"
    ]


def test_process_contract_checker_fails_on_missing_delete_route(
    monkeypatch,
    tmp_path,
):
    routes = _routes()
    del routes["delete"]
    code, payload, _seen = _run(
        monkeypatch,
        _contract(routes=routes),
        tmp_path,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "routes.delete must be /api/v1/content-ops/deflection-reports/{request_id}"
    ]
    assert payload["observed"]["routes"] == routes


def test_process_contract_checker_fails_on_stale_delete_route(
    monkeypatch,
    tmp_path,
):
    code, payload, _seen = _run(
        monkeypatch,
        _contract(routes=_routes(delete="/api/v1/content-ops/deflection-reports")),
        tmp_path,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "routes.delete must be /api/v1/content-ops/deflection-reports/{request_id}"
    ]


def test_process_contract_checker_fails_on_non_200_success_status(monkeypatch, tmp_path):
    code, payload, _seen = _run(monkeypatch, _contract(), tmp_path, status=201)

    assert code == 1
    assert payload["ok"] is False
    assert payload["endpoint"]["status"] == 201
    assert payload["errors"] == [
        "process contract endpoint must return 200, got 201"
    ]


def test_process_contract_checker_fails_on_non_object_json(monkeypatch, tmp_path):
    code, payload, _seen = _run(monkeypatch, ["not", "an", "object"], tmp_path)

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "process contract endpoint response must be a JSON object"
    ]


def test_process_contract_checker_fails_on_wrong_schema(monkeypatch, tmp_path):
    code, payload, _seen = _run(
        monkeypatch,
        _contract(schema_version="legacy.v1"),
        tmp_path,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "schema_version must be deflection_report_process.v1"
    ]


def test_process_contract_checker_fails_on_partial_artifact_requirements(
    monkeypatch,
    tmp_path,
):
    code, payload, _seen = _run(
        monkeypatch,
        _contract(
            contract={
                "report_model_schema_version": "deflection.v1",
                "report_model_contract": checker.EXPECTED_REPORT_MODEL_CONTRACT,
                "evidence_export_schema_version": "deflection_evidence.v1",
                "paid_artifact_requires": {"report_model": "object"},
            },
        ),
        tmp_path,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "contract.paid_artifact_requires.evidence_export must be object"
    ]


def test_process_contract_checker_fails_on_report_model_shape_drift(
    monkeypatch,
    tmp_path,
):
    stale_shape = dict(checker.EXPECTED_REPORT_MODEL_CONTRACT)
    stale_shape["sections"] = [
        section
        for section in checker.EXPECTED_REPORT_MODEL_CONTRACT["sections"]
        if section["id"] != "complete_evidence"
    ]
    code, payload, _seen = _run(
        monkeypatch,
        _contract(
            contract={
                "report_model_schema_version": "deflection.v1",
                "report_model_contract": stale_shape,
                "evidence_export_schema_version": "deflection_evidence.v1",
                "paid_artifact_requires": {
                    "report_model": "object",
                    "evidence_export": "object",
                },
            },
        ),
        tmp_path,
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "contract.report_model_contract must match current deflection.v1 shape"
    ]


def test_process_contract_checker_rejects_localhost_before_network(monkeypatch, tmp_path):
    def fake_urlopen(request, timeout):
        del request, timeout
        raise AssertionError("network should not be touched for localhost")

    monkeypatch.setattr(checker.urllib.request, "urlopen", fake_urlopen)
    output = tmp_path / "result.json"
    code = checker.main([
        "--base-url",
        "https://localhost",
        "--output-result",
        str(output),
    ])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "base-url must point to a hosted URL; local hosts are not accepted"
    ]
