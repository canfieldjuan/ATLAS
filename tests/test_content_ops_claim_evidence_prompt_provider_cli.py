from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import httpx

from extracted_content_pipeline.claim_evidence_benchmark import (
    claim_evidence_response_json_schema,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_content_ops_claim_evidence_prompt_provider.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_content_ops_claim_evidence_prompt_provider",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packet(model_id: str = "gpt-4.1-mini", triple_id: str = "easy") -> dict[str, object]:
    return {
        "model_id": model_id,
        "triple_id": triple_id,
        "contract_version": "verify_claim_evidence.v1",
        "prompt": f"Prompt for {triple_id}",
        "response_schema": claim_evidence_response_json_schema(),
    }


def _completion(response: dict[str, object]) -> dict[str, object]:
    return {"choices": [{"message": {"content": json.dumps(response)}}]}


def _response(supports: bool = True, reason: str = "quote supports it") -> dict[str, object]:
    return {"supports": supports, "confidence": 5, "reason": reason}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prompt_provider_cli_writes_importer_compatible_json_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    output_path = tmp_path / "responses.json"
    seen_requests: list[dict[str, object]] = []
    _write_json(packets_path, [_packet("model-a", "easy"), _packet("model-b", "hard")])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_requests.append(
            {
                "url": str(request.url),
                "authorization": request.headers.get("authorization"),
                "payload": payload,
            }
        )
        return httpx.Response(
            200,
            json=_completion(_response(payload["model"] == "model-a")),
        )

    exit_code, payload = cli.run_prompt_packets_with_openai_compatible_provider(
        packets_path,
        output_path,
        api_base_url="https://provider.example/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload == {
        "ok": True,
        "output_path": str(output_path),
        "output_format": "json",
        "response_count": 2,
        "errors": [],
    }
    assert [row["model_id"] for row in rows] == ["model-a", "model-b"]
    assert rows[0]["response"] == _response(True)
    assert rows[1]["response"] == _response(False)
    assert [request["url"] for request in seen_requests] == [
        "https://provider.example/v1/chat/completions",
        "https://provider.example/v1/chat/completions",
    ]
    assert seen_requests[0]["authorization"] == "Bearer sk-test-not-real"
    request_payload = seen_requests[0]["payload"]
    assert request_payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "claim_evidence_response",
            "strict": True,
            "schema": claim_evidence_response_json_schema(),
        },
    }
    assert request_payload["messages"][1]["content"] == "Prompt for easy"
    assert "expected_supports" not in output_path.read_text(encoding="utf-8")


def test_prompt_provider_cli_writes_jsonl_stability_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.jsonl"
    output_path = tmp_path / "responses.jsonl"
    packets_path.write_text(json.dumps(_packet()) + "\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    def handler(request: httpx.Request) -> httpx.Response:
        metadata = json.loads(request.content)["metadata"]
        return httpx.Response(
            200,
            json=_completion(_response(reason=f"{metadata['run_type']}:{metadata.get('run_id', '')}")),
        )

    exit_code, payload = cli.run_prompt_packets_with_openai_compatible_provider(
        packets_path,
        output_path,
        stability_run_count=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert payload["output_format"] == "jsonl"
    assert [row.get("run_type", "main") for row in rows] == [
        "main",
        "stability",
        "stability",
    ]
    assert [row.get("run_id", "") for row in rows] == ["", "rerun-1", "rerun-2"]


def test_prompt_provider_cli_requires_api_key_before_provider_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    output_path = tmp_path / "responses.json"
    _write_json(packets_path, [_packet()])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code, payload = cli.run_prompt_packets_with_openai_compatible_provider(
        packets_path,
        output_path,
    )

    assert exit_code == 2
    assert payload["errors"] == ["OPENAI_API_KEY missing or provider options invalid"]
    assert output_path.exists() is False


def test_prompt_provider_cli_rejects_bad_packets_before_http_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    output_path = tmp_path / "responses.json"
    calls = 0
    _write_json(
        packets_path,
        [_packet(), _packet(), {**_packet("gpt-4.1-mini", "bad"), "response_schema": {}}],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_completion(_response()))

    exit_code, payload = cli.run_prompt_packets_with_openai_compatible_provider(
        packets_path,
        output_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert exit_code == 1
    assert payload["errors"] == [
        "packet row 2: duplicate packet key: gpt-4.1-mini/easy/verify_claim_evidence.v1",
        "packet row 3: response_schema does not match the canonical claim/evidence schema",
    ]
    assert calls == 0
    assert output_path.exists() is False


def test_prompt_provider_cli_sanitizes_provider_failure_details(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli_module()
    packets_path = tmp_path / "packets.json"
    output_path = tmp_path / "responses.json"
    _write_json(packets_path, [_packet("model-a", "easy")])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": "sk-test-secret and provider internal details"},
        )

    exit_code, payload = cli.run_prompt_packets_with_openai_compatible_provider(
        packets_path,
        output_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert exit_code == 1
    assert payload["errors"] == ["main model-a/easy: provider error: ProviderHTTPStatusError"]
    assert "sk-test-secret" not in json.dumps(payload)
    assert "internal details" not in json.dumps(payload)
    assert output_path.exists() is False


def test_direct_script_invocation_prints_missing_key_envelope(tmp_path: Path, monkeypatch) -> None:
    packets_path = tmp_path / "packets.json"
    output_path = tmp_path / "responses.json"
    _write_json(packets_path, [_packet()])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(packets_path), str(output_path)],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert result.stderr == ""
    assert payload["ok"] is False
    assert payload["errors"] == ["OPENAI_API_KEY missing or provider options invalid"]
