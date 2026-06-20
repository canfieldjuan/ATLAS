from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/_deflection_http.py"
SPEC = importlib.util.spec_from_file_location("_deflection_http", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
http = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = http
SPEC.loader.exec_module(http)


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


def _http_error(url: str, status: int, body: dict[str, Any] | None = None):
    fp = FakeResponse(status, body or {})
    return urllib.error.HTTPError(url, status, "error", {}, fp)


def test_json_request_preserves_http_error_status(monkeypatch) -> None:
    def _urlopen(request, *, timeout):
        assert timeout == 7
        raise _http_error(request.full_url, 409, {"detail": "already paid"})

    monkeypatch.setattr(http.urllib.request, "urlopen", _urlopen)

    result = http.json_request(
        "POST",
        "https://atlas.example.com/proof",
        timeout=7,
        http_error_template="HTTP {status}",
    )

    assert result.status == 409
    assert result.payload == {"detail": "already paid"}
    assert result.errors == ("HTTP 409",)


def test_json_request_redacts_http_error_body_before_payload_parse(monkeypatch) -> None:
    def _urlopen(request, *, timeout):
        assert timeout == 7
        raise _http_error(
            request.full_url,
            422,
            {
                "detail": (
                    "Bearer sk_test_secret for content-ops-proof-123 "
                    "buyer@example.com at https://blob.example.com/file.csv?sig=secret"
                )
            },
        )

    def _redact_body(raw: str) -> str:
        return (
            raw.replace("sk_test_secret", "[stripe-redacted]")
            .replace("content-ops-proof-123", "content-ops-[redacted]")
            .replace("buyer@example.com", "[email-redacted]")
            .replace("sig=secret", "sig=[redacted]")
        )

    monkeypatch.setattr(http.urllib.request, "urlopen", _urlopen)

    result = http.json_request(
        "POST",
        "https://atlas.example.com/proof",
        timeout=7,
        http_error_template="HTTP {status}",
        error_body_redactor=_redact_body,
    )

    assert result.status == 422
    assert result.errors == ("HTTP 422",)
    serialized = json.dumps(result.payload, sort_keys=True) + result.raw_text
    assert "sk_test_secret" not in serialized
    assert "content-ops-proof-123" not in serialized
    assert "buyer@example.com" not in serialized
    assert "sig=secret" not in serialized
    assert "[stripe-redacted]" in serialized


def test_json_request_structural_redaction_preserves_numeric_json_fields(monkeypatch) -> None:
    def _urlopen(request, *, timeout):
        assert timeout == 7
        raise _http_error(
            request.full_url,
            422,
            {
                "detail": "Bearer sk_test_secret for batch 1500000",
                "received_count": 1500000,
            },
        )

    def _redact_body(raw: str) -> str:
        return raw.replace("sk_test_secret", "[stripe-redacted]").replace("1500000", "[id-redacted]")

    monkeypatch.setattr(http.urllib.request, "urlopen", _urlopen)

    result = http.json_request(
        "POST",
        "https://atlas.example.com/proof",
        timeout=7,
        http_error_template="HTTP {status}",
        error_body_redactor=_redact_body,
    )

    assert result.status == 422
    assert result.payload == {
        "detail": "Bearer [stripe-redacted] for batch [id-redacted]",
        "received_count": 1500000,
    }
    assert result.errors == ("HTTP 422",)


def test_json_request_keyed_error_redaction_preserves_unrelated_opaque_strings(monkeypatch) -> None:
    def _urlopen(request, *, timeout):
        assert timeout == 7
        raise _http_error(
            request.full_url,
            422,
            {
                "source_id": "opaque-source-A",
                "trace_id": "opaque-trace-B",
                "source_ids": ["shortA", 12345],
                "received_count": 1500000,
            },
        )

    def _redact_keyed_value(path: tuple[str, ...], value: Any) -> Any | None:
        if path and path[-1] in {"source_id", "source_ids"} and isinstance(value, (str, int)):
            return "[source-id-redacted]"
        return None

    monkeypatch.setattr(http.urllib.request, "urlopen", _urlopen)

    result = http.json_request(
        "POST",
        "https://atlas.example.com/proof",
        timeout=7,
        http_error_template="HTTP {status}",
        error_body_redactor=lambda raw: raw,
        error_body_json_value_redactor=_redact_keyed_value,
    )

    assert result.status == 422
    assert result.payload == {
        "source_id": "[source-id-redacted]",
        "trace_id": "opaque-trace-B",
        "source_ids": ["[source-id-redacted]", "[source-id-redacted]"],
        "received_count": 1500000,
    }
    assert "opaque-source-A" not in result.raw_text
    assert "shortA" not in result.raw_text
    assert "opaque-trace-B" in result.raw_text


def test_json_request_redacts_transport_error(monkeypatch) -> None:
    def _urlopen(_request, *, timeout):
        assert timeout == 3
        raise urllib.error.URLError("failed for sk_test_secret and content-ops-proof-123")

    monkeypatch.setattr(http.urllib.request, "urlopen", _urlopen)

    result = http.json_request(
        "GET",
        "https://atlas.example.com/proof",
        timeout=3,
        redactor=lambda value: str(value).replace("sk_test_secret", "[redacted]"),
        transport_error_template="{error}",
    )

    assert result.status is None
    assert result.payload is None
    assert "sk_test_secret" not in result.errors[0]
    assert "[redacted]" in result.errors[0]


def test_json_request_encodes_mapping_body_and_bearer_header(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _urlopen(request, *, timeout):
        calls.append({
            "timeout": timeout,
            "headers": dict(request.header_items()),
            "data": request.data,
            "method": request.get_method(),
        })
        return FakeResponse(200, {"ok": True})

    monkeypatch.setattr(http.urllib.request, "urlopen", _urlopen)

    result = http.json_request(
        "POST",
        "https://atlas.example.com/login",
        timeout=5,
        token="atlas-token",
        body={"email": "growth@example.com", "password": "secret"},
    )

    assert result.status == 200
    assert result.payload == {"ok": True}
    assert calls == [{
        "timeout": 5,
        "headers": {
            "Accept": "application/json",
            "Authorization": "Bearer atlas-token",
            "Content-type": "application/json",
        },
        "data": b'{"email":"growth@example.com","password":"secret"}',
        "method": "POST",
    }]


def test_json_request_preserves_raw_webhook_bytes_and_signature(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _urlopen(request, *, timeout):
        calls.append({
            "timeout": timeout,
            "headers": dict(request.header_items()),
            "data": request.data,
        })
        return FakeResponse(200, {"status": "ok"})

    monkeypatch.setattr(http.urllib.request, "urlopen", _urlopen)
    body = b'{"id":"evt_test","object":"event"}'

    result = http.json_request(
        "POST",
        "https://atlas.example.com/webhooks/stripe",
        timeout=11,
        data=body,
        stripe_signature="t=123,v1=sig",
    )

    assert result.payload == {"status": "ok"}
    assert calls[0]["data"] == body
    assert calls[0]["headers"]["Stripe-signature"] == "t=123,v1=sig"
    assert calls[0]["headers"]["Content-type"] == "application/json"


def test_json_request_rejects_ambiguous_body_inputs() -> None:
    with pytest.raises(ValueError, match="either body or data"):
        http.json_request(
            "POST",
            "https://atlas.example.com/proof",
            timeout=1,
            body={"ok": True},
            data=b"raw",
        )
