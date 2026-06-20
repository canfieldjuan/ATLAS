from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import urllib.error
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_deflection_test_mode_checkout_proof.py"
SPEC = importlib.util.spec_from_file_location(
    "run_deflection_test_mode_checkout_proof",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
proof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proof
SPEC.loader.exec_module(proof)

ACCOUNT_ID = "11111111-1111-4111-8111-111111111111"
REQUEST_ID = "content-ops-proof-123"
STRIPE_KEY = "sk_test_0000000000000000"
CHECKOUT_URL = "https://checkout.stripe.com/c/pay/cs_test_sanitized"
SESSION_ID = "cs_test_sanitized"


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


def _http_error(url: str, status: int, body: dict[str, Any] | None = None) -> urllib.error.HTTPError:
    fp = None
    if body is not None:
        fp = FakeResponse(status, body)
    return urllib.error.HTTPError(url, status, "error", {}, fp)


class FakeCheckoutSessionApi:
    def __init__(self):
        self.create_kwargs: dict[str, Any] | None = None
        self.retrieve_calls: list[tuple[str, dict[str, Any]]] = []
        self.retrieve_status = SimpleNamespace(status="open", payment_status="unpaid")

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(
            id=SESSION_ID,
            url=CHECKOUT_URL,
            status="open",
            payment_status="unpaid",
        )

    def retrieve(self, session_id: str, **kwargs):
        self.retrieve_calls.append((session_id, kwargs))
        return self.retrieve_status


def _stripe_module(session_api: FakeCheckoutSessionApi):
    return SimpleNamespace(
        api_key="",
        checkout=SimpleNamespace(Session=session_api),
    )


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "atlas-token",
        "--database-url",
        "postgres://user:pass@example/db",
        "--stripe-key",
        STRIPE_KEY,
        "--request-id",
        REQUEST_ID,
        "--success-url",
        f"https://juancanfield.com/systems/support-ticket-deflection/results/{REQUEST_ID}?checkout=success",
        "--cancel-url",
        f"https://juancanfield.com/systems/support-ticket-deflection/results/{REQUEST_ID}?checkout=cancel",
        "--output-result",
        str(tmp_path / "result.json"),
    ]


def _authorization_payload() -> dict[str, Any]:
    return {
        "request_id": REQUEST_ID,
        "status": "authorized",
        "checkout": {
            "amount_cents": 150000,
            "currency": "usd",
            "price_id": "price_test_deflection",
        },
    }


def _payload(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))


def test_validate_args_rejects_live_or_non_test_stripe_keys(tmp_path: Path) -> None:
    args = proof._build_parser().parse_args(_base_args(tmp_path))
    args.stripe_key = "sk_live_abc"

    assert proof._validate_args(args) == [
        "--stripe-key must be a test-mode key for the no-live-charge proof"
    ]

    args.stripe_key = "not-a-key"

    assert proof._validate_args(args) == ["--stripe-key must start with sk_test_ or rk_test_"]

    args.stripe_key = "rk_live_abc"

    assert proof._validate_args(args) == [
        "--stripe-key must be a test-mode key for the no-live-charge proof"
    ]

    args.stripe_key = "rk_test_0000000000000000"

    assert proof._validate_args(args) == []


def test_invalid_authorization_terms_fail_closed() -> None:
    terms, errors = proof._checkout_terms_from_authorization({
        "status": "authorized",
        "checkout": {"amount_cents": 0, "currency": "eur", "price_id": ""},
    })

    assert terms is None
    assert errors == [
        "checkout amount must be positive",
        "checkout currency must be usd",
        "checkout price_id is required",
    ]


def test_preflight_writes_sanitized_presence_only_payload(tmp_path: Path, capsys) -> None:
    code = proof.main([*_base_args(tmp_path), "--preflight-only", "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = _payload(tmp_path)
    assert printed == payload
    assert payload["ok"] is True
    assert payload["inputs"] == {
        "base_url_present": True,
        "token_present": True,
        "database_url_present": True,
        "stripe_key_present": True,
        "request_id_present": True,
        "success_url_present": True,
        "cancel_url_present": True,
        "wait_for_unlock": False,
    }
    encoded = json.dumps(payload, sort_keys=True)
    assert REQUEST_ID not in encoded
    assert ACCOUNT_ID not in encoded
    assert STRIPE_KEY not in encoded
    assert "postgres://" not in encoded
    assert "atlas-token" not in encoded


def test_authorization_header_does_not_double_prefix_bearer_token() -> None:
    assert proof._authorization_header("atlas-token") == "Bearer atlas-token"
    assert proof._authorization_header("Bearer atlas-token") == "Bearer atlas-token"


def test_json_request_preserves_http_error_status_through_real_wrapper(monkeypatch) -> None:
    def _urlopen(request, *, timeout):
        assert timeout == 30.0
        raise _http_error(request.full_url, 409, {"detail": "already paid"})

    monkeypatch.setattr(proof.urllib.request, "urlopen", _urlopen)

    result = proof._json_request(
        "POST",
        "https://atlas.example.com/proof",
        token="atlas-token",
        timeout=30.0,
    )

    assert result.status == 409
    assert result.payload == {"detail": "already paid"}
    assert result.errors == ("HTTP 409",)


def test_json_request_redacts_transport_errors_through_real_wrapper(monkeypatch) -> None:
    def _urlopen(_request, *, timeout):
        assert timeout == 30.0
        raise urllib.error.URLError(
            f"failed {STRIPE_KEY} for "
            f"https://juancanfield.com/systems/support-ticket-deflection/results/{REQUEST_ID}"
        )

    monkeypatch.setattr(proof.urllib.request, "urlopen", _urlopen)

    result = proof._json_request(
        "GET",
        "https://atlas.example.com/proof",
        token="atlas-token",
        timeout=30.0,
    )

    assert result.status is None
    assert len(result.errors) == 1
    assert STRIPE_KEY not in result.errors[0]
    assert REQUEST_ID not in result.errors[0]
    assert "support-ticket-deflection/results" not in result.errors[0]


def test_sanitizer_rewrites_constructed_result_urls() -> None:
    payload = proof._finalize_payload({
        "ok": True,
        "errors": [],
        "url": f"https://juancanfield.com/systems/support-ticket-deflection/results/{REQUEST_ID}?checkout=success",
    })

    encoded = json.dumps(payload, sort_keys=True)
    assert REQUEST_ID not in encoded
    assert "support-ticket-deflection/results" not in encoded
    assert payload["url"] == "[redacted]"


def test_main_creates_test_checkout_with_persisted_metadata_and_sanitized_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    requests: list[tuple[str, str]] = []
    session_api = FakeCheckoutSessionApi()
    fake_stripe = _stripe_module(session_api)

    def _lookup(database_url: str, request_id: str) -> str:
        assert database_url == "postgres://user:pass@example/db"
        assert request_id == REQUEST_ID
        return ACCOUNT_ID

    def _open(request, *, timeout):
        requests.append((request.get_method(), request.full_url))
        assert request.headers["Authorization"] == "Bearer atlas-token"
        assert timeout == 30.0
        return FakeResponse(200, _authorization_payload())

    monkeypatch.setattr(proof.paid_unlock, "_lookup_report_account_id", _lookup)
    monkeypatch.setattr(proof, "_open_http_request", _open)
    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)

    code = proof.main([*_base_args(tmp_path), "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = _payload(tmp_path)
    assert printed == payload
    assert requests == [
        (
            "POST",
            f"https://atlas.example.com/api/v1/content-ops/deflection-reports/{REQUEST_ID}/checkout-authorization",
        )
    ]
    assert fake_stripe.api_key == STRIPE_KEY
    assert session_api.create_kwargs is not None
    kwargs = session_api.create_kwargs
    assert kwargs["mode"] == "payment"
    assert kwargs["line_items"] == [{"price": "price_test_deflection", "quantity": 1}]
    assert kwargs["metadata"] == {
        "source": "content_ops_deflection_report",
        "account_id": ACCOUNT_ID,
        "request_id": REQUEST_ID,
    }
    assert kwargs["success_url"].startswith("https://juancanfield.com/")
    assert kwargs["cancel_url"].startswith("https://juancanfield.com/")
    assert "payment_method_types" not in kwargs
    assert kwargs["idempotency_key"].startswith("deflection-test-checkout:")

    assert payload["ok"] is True
    assert payload["checkout_authorization"] == {
        "status": 200,
        "authorized": True,
        "amount_cents": 150000,
        "currency": "usd",
        "price_id_present": True,
    }
    assert payload["checkout_session"] == {
        "created": True,
        "checkout_url_printed": False,
        "session_id_present": True,
        "url_present": True,
        "status": "open",
        "payment_status": "unpaid",
        "retrieved_status": "",
        "retrieved_payment_status": "",
    }
    encoded = json.dumps(payload, sort_keys=True)
    assert REQUEST_ID not in encoded
    assert ACCOUNT_ID not in encoded
    assert SESSION_ID not in encoded
    assert CHECKOUT_URL not in encoded
    assert STRIPE_KEY not in encoded
    assert "postgres://" not in encoded
    assert "atlas-token" not in encoded


def test_main_stops_before_stripe_when_report_lookup_fails(monkeypatch, tmp_path: Path) -> None:
    session_api = FakeCheckoutSessionApi()

    def _missing(*_args):
        raise RuntimeError("persisted deflection report row was not found")

    def _unexpected_network(*_args, **_kwargs):
        raise AssertionError("missing report must stop before HTTP")

    monkeypatch.setattr(proof.paid_unlock, "_lookup_report_account_id", _missing)
    monkeypatch.setattr(proof, "_open_http_request", _unexpected_network)
    monkeypatch.setitem(sys.modules, "stripe", _stripe_module(session_api))

    code = proof.main(_base_args(tmp_path))
    payload = _payload(tmp_path)

    assert code == 1
    assert session_api.create_kwargs is None
    assert payload["ok"] is False
    assert payload["errors"] == ["persisted deflection report row was not found"]
    assert payload["checkout_session"]["created"] is False


def test_main_stops_before_stripe_when_authorization_is_not_usable(monkeypatch, tmp_path: Path) -> None:
    session_api = FakeCheckoutSessionApi()

    def _open(request, *, timeout):
        raise _http_error(request.full_url, 409, {"detail": "already paid"})

    monkeypatch.setattr(proof.paid_unlock, "_lookup_report_account_id", lambda *_args: ACCOUNT_ID)
    monkeypatch.setattr(proof, "_open_http_request", _open)
    monkeypatch.setitem(sys.modules, "stripe", _stripe_module(session_api))

    code = proof.main(_base_args(tmp_path))
    payload = _payload(tmp_path)

    assert code == 1
    assert session_api.create_kwargs is None
    assert payload["errors"] == ["checkout authorization status must be 200, got 409"]
    assert payload["metadata_resolution"] == {
        "account_id_source": "persisted_report",
        "report_row_checked": True,
    }
    assert payload["checkout_authorization"] == {
        "status": 409,
        "authorized": False,
        "amount_cents": None,
        "currency": None,
        "price_id_present": False,
    }


def test_main_polls_locked_then_unlocked_artifact_without_leaking_ids(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    session_api = FakeCheckoutSessionApi()

    def _open(request, *, timeout):
        calls.append((request.get_method(), request.full_url))
        if request.full_url.endswith("/checkout-authorization"):
            return FakeResponse(200, _authorization_payload())
        artifact_calls = [call for call in calls if call[1].endswith("/artifact")]
        if len(artifact_calls) == 1:
            raise _http_error(request.full_url, 403, {"detail": "payment required"})
        return FakeResponse(200, {"markdown": "# paid", "faq_result": {}, "summary": {}})

    monkeypatch.setattr(proof.paid_unlock, "_lookup_report_account_id", lambda *_args: ACCOUNT_ID)
    monkeypatch.setattr(proof, "_open_http_request", _open)
    monkeypatch.setitem(sys.modules, "stripe", _stripe_module(session_api))

    code = proof.main([
        *_base_args(tmp_path),
        "--wait-for-unlock",
        "--poll-timeout",
        "1",
        "--poll-interval",
        "0",
    ])
    payload = _payload(tmp_path)

    assert code == 0
    assert payload["ok"] is True
    assert payload["artifact_poll"] == {
        "enabled": True,
        "statuses": [403, 200],
        "unlocked": True,
    }
    encoded = json.dumps(payload, sort_keys=True)
    assert REQUEST_ID not in encoded
    assert ACCOUNT_ID not in encoded
    assert SESSION_ID not in encoded


def test_main_records_unpaid_timeout_with_sanitized_session_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session_api = FakeCheckoutSessionApi()

    def _open(request, *, timeout):
        if request.full_url.endswith("/checkout-authorization"):
            return FakeResponse(200, _authorization_payload())
        raise _http_error(request.full_url, 403, {"detail": "payment required"})

    monkeypatch.setattr(proof.paid_unlock, "_lookup_report_account_id", lambda *_args: ACCOUNT_ID)
    monkeypatch.setattr(proof, "_open_http_request", _open)
    monkeypatch.setitem(sys.modules, "stripe", _stripe_module(session_api))

    code = proof.main([
        *_base_args(tmp_path),
        "--wait-for-unlock",
        "--poll-timeout",
        "0",
        "--poll-interval",
        "0",
    ])
    payload = _payload(tmp_path)

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["timed out waiting for paid artifact unlock"]
    assert payload["artifact_poll"] == {
        "enabled": True,
        "statuses": [403],
        "unlocked": False,
    }
    assert payload["checkout_session"]["retrieved_status"] == "open"
    assert payload["checkout_session"]["retrieved_payment_status"] == "unpaid"
    assert session_api.retrieve_calls == [(SESSION_ID, {"timeout": 30.0})]
    encoded = json.dumps(payload, sort_keys=True)
    assert REQUEST_ID not in encoded
    assert SESSION_ID not in encoded


def test_output_leak_guard_rewrites_sensitive_strings(monkeypatch, tmp_path: Path) -> None:
    session_api = FakeCheckoutSessionApi()

    def _open(request, *, timeout):
        return FakeResponse(200, _authorization_payload())

    def _leaky_create(**_kwargs):
        return SimpleNamespace(
            id="cs_test_outputleak",
            url="https://checkout.stripe.com/c/pay/cs_test_outputleak",
            status="open",
            payment_status="unpaid",
        )

    session_api.create = _leaky_create  # type: ignore[method-assign]
    monkeypatch.setattr(proof.paid_unlock, "_lookup_report_account_id", lambda *_args: ACCOUNT_ID)
    monkeypatch.setattr(proof, "_open_http_request", _open)
    monkeypatch.setitem(sys.modules, "stripe", _stripe_module(session_api))

    code = proof.main([*_base_args(tmp_path), "--json"])
    payload = _payload(tmp_path)

    assert code == 0
    encoded = json.dumps(payload, sort_keys=True)
    assert "cs_test_outputleak" not in encoded
    assert "checkout.stripe.com" not in encoded
    assert payload["checkout_session"]["session_id_present"] is True
    assert payload["checkout_session"]["url_present"] is True


def test_create_checkout_session_raises_sanitized_error_on_stripe_failure(
    monkeypatch,
) -> None:
    session_api = FakeCheckoutSessionApi()

    def _fail(**_kwargs):
        raise RuntimeError("Stripe failed for sk_test_0000000000000000")

    session_api.create = _fail  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "stripe", _stripe_module(session_api))

    with pytest.raises(RuntimeError) as excinfo:
        proof._create_checkout_session(
            stripe_key=STRIPE_KEY,
            terms=proof.CheckoutTerms(
                amount_cents=150000,
                currency="usd",
                price_id="price_test_deflection",
            ),
            account_id=ACCOUNT_ID,
            request_id=REQUEST_ID,
            success_url="https://juancanfield.com/success",
            cancel_url="https://juancanfield.com/cancel",
            timeout=30.0,
        )

    message = str(excinfo.value)
    assert "stripe checkout creation failed" in message
    assert STRIPE_KEY not in message
