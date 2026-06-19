from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_deflection_stripe_paid_unlock.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_deflection_stripe_paid_unlock",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


ACCOUNT_ID = "11111111-1111-4111-8111-111111111111"
OTHER_ACCOUNT_ID = "22222222-2222-4222-8222-222222222222"
ARTIFACT = {
    "summary": {"source_count": 2},
    "markdown": "# Paid report",
    "faq_result": {"markdown": "# FAQ"},
}


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


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--webhook-secret",
        "whsec_test",
        "--account-id",
        ACCOUNT_ID,
        "--request-id",
        "content-ops-123",
        "--event-id",
        "evt_paid_unlock",
        "--session-id",
        "cs_paid_unlock",
        "--output-result",
        str(tmp_path / "result.json"),
    ]


def _without_account_id(args: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for value in args:
        if skip_next:
            skip_next = False
            continue
        if value == "--account-id":
            skip_next = True
            continue
        out.append(value)
    return out


def _http_error(url: str, status: int, body: dict[str, Any] | None = None) -> urllib.error.HTTPError:
    fp = None
    if body is not None:
        fp = FakeResponse(status, body)
    return urllib.error.HTTPError(url, status, "error", {}, fp)


def test_validate_args_fails_closed_for_missing_and_unsafe_inputs() -> None:
    args = smoke._build_parser().parse_args([
        "--base-url",
        "http://127.0.0.1:8000",
        "--token",
        "",
        "--webhook-secret",
        "",
        "--account-id",
        "not-a-uuid",
        "--request-id",
        "",
        "--amount-total",
        "0",
        "--currency",
        "eur",
        "--timeout",
        "0",
        "--webhook-path",
        "webhooks/stripe",
        "--artifact-path-template",
        "/artifact",
    ])

    assert smoke._validate_args(args) == [
        "--base-url must be an absolute HTTPS URL",
        "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
        "ATLAS_SAAS_STRIPE_WEBHOOK_SECRET, STRIPE_WEBHOOK_SECRET, or --webhook-secret is required",
        "--account-id must be a UUID for the Stripe metadata contract",
        "ATLAS_DEFLECTION_REQUEST_ID or --request-id is required",
        "--amount-total must be positive",
        "--currency must be usd",
        "--timeout must be a positive finite number",
        "--webhook-path must start with /",
        "--artifact-path-template must include {request_id}",
    ]


def test_validate_args_requires_database_url_when_deriving_account_id() -> None:
    args = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--webhook-secret",
        "whsec_test",
        "--account-id",
        "",
        "--request-id",
        "content-ops-123",
        "--derive-account-id-from-report",
    ])

    assert smoke._validate_args(args) == [
        "--database-url is required with --derive-account-id-from-report",
    ]


def test_validate_args_allows_missing_account_id_when_report_lookup_is_enabled() -> None:
    args = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--webhook-secret",
        "whsec_test",
        "--account-id",
        "",
        "--request-id",
        "content-ops-123",
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
    ])

    assert smoke._validate_args(args) == []


def test_metadata_resolution_ignores_env_default_account_when_deriving(
    monkeypatch,
) -> None:
    monkeypatch.setattr(smoke, "_lookup_report_account_id", lambda *_args: ACCOUNT_ID)
    args = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--webhook-secret",
        "whsec_test",
        "--account-id",
        OTHER_ACCOUNT_ID,
        "--request-id",
        "content-ops-123",
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
    ])
    args.account_id_explicit = False

    metadata_resolution, errors = smoke._resolve_metadata(args)

    assert errors == []
    assert metadata_resolution == smoke.MetadataResolution(
        account_id=ACCOUNT_ID,
        account_id_source="persisted_report",
        account_id_supplied=False,
        report_row_checked=True,
    )


def test_validate_args_ignores_non_explicit_env_account_when_deriving() -> None:
    args = smoke._build_parser().parse_args([
        "--base-url",
        "https://atlas.example.com",
        "--token",
        "secret-token",
        "--webhook-secret",
        "whsec_test",
        "--account-id",
        "acct-123",
        "--request-id",
        "content-ops-123",
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
    ])
    args.account_id_explicit = False

    assert smoke._validate_args(args) == []

    args.account_id_explicit = True

    assert smoke._validate_args(args) == [
        "--account-id must be a UUID for the Stripe metadata contract"
    ]


def test_preflight_only_writes_redacted_inputs_without_network(monkeypatch, tmp_path, capsys) -> None:
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("preflight must not call network")

    monkeypatch.setattr(smoke, "_open_http_request", _unexpected)

    code = smoke.main([*_base_args(tmp_path), "--preflight-only", "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert printed == payload
    assert payload["inputs"]["token_present"] is True
    assert payload["inputs"]["webhook_secret_present"] is True
    assert "secret-token" not in json.dumps(payload)
    assert "whsec_test" not in json.dumps(payload)


def test_stripe_signature_header_matches_expected_digest() -> None:
    body = b'{"id":"evt_test"}'
    header = smoke._stripe_signature_header(body, secret="whsec_test", timestamp=1700000000)
    digest = hmac.new(
        b"whsec_test",
        b'1700000000.{"id":"evt_test"}',
        hashlib.sha256,
    ).hexdigest()

    assert header == f"t=1700000000,v1={digest}"


def test_main_posts_signed_webhook_and_unlocks_artifact(monkeypatch, tmp_path, capsys) -> None:
    calls: list[dict[str, Any]] = []

    def _open(request, *, timeout):
        body = request.data or b""
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": dict(request.header_items()),
            "body": body.decode("utf-8"),
            "timeout": timeout,
        })
        if request.get_method() == "GET" and len([c for c in calls if c["method"] == "GET"]) == 1:
            raise _http_error(request.full_url, 403, {"detail": "payment_required"})
        if request.get_method() == "POST":
            return FakeResponse(200, {"status": "ok"})
        return FakeResponse(200, ARTIFACT)

    monkeypatch.setattr(smoke, "_open_http_request", _open)
    monkeypatch.setattr(smoke.time, "time", lambda: 1700000000)

    code = smoke.main([*_base_args(tmp_path), "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert printed == payload
    assert payload["ok"] is True
    assert payload["before_artifact"] == {"status": 403}
    assert payload["webhook"]["status"] == 200
    assert payload["after_artifact"]["status"] == 200

    webhook_call = calls[1]
    assert webhook_call["url"] == "https://atlas.example.com/webhooks/stripe"
    assert webhook_call["method"] == "POST"
    assert webhook_call["headers"]["Stripe-signature"].startswith("t=1700000000,v1=")
    event = json.loads(webhook_call["body"])
    session = event["data"]["object"]
    assert event["id"] == "evt_paid_unlock"
    assert session["id"] == "cs_paid_unlock"
    assert session["mode"] == "payment"
    assert session["payment_status"] == "paid"
    assert session["amount_total"] == 150000
    assert session["currency"] == "usd"
    assert session["metadata"] == {
        "source": "content_ops_deflection_report",
        "account_id": ACCOUNT_ID,
        "request_id": "content-ops-123",
    }


def test_main_derives_account_id_from_persisted_report_before_webhook(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    calls: list[dict[str, Any]] = []

    def _lookup(database_url: str, request_id: str) -> str:
        assert database_url == "postgres://example"
        assert request_id == "content-ops-123"
        return ACCOUNT_ID

    def _open(request, *, timeout):
        body = request.data or b""
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": dict(request.header_items()),
            "body": body.decode("utf-8"),
            "timeout": timeout,
        })
        if request.get_method() == "GET" and len([c for c in calls if c["method"] == "GET"]) == 1:
            raise _http_error(request.full_url, 403, {"detail": "payment_required"})
        if request.get_method() == "POST":
            return FakeResponse(200, {"status": "ok"})
        return FakeResponse(200, ARTIFACT)

    monkeypatch.setattr(smoke, "_lookup_report_account_id", _lookup)
    monkeypatch.setattr(smoke, "_open_http_request", _open)
    monkeypatch.setattr(smoke.time, "time", lambda: 1700000000)

    code = smoke.main([
        *_without_account_id(_base_args(tmp_path)),
        "--account-id",
        "",
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
        "--json",
    ])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert printed == payload
    assert payload["ok"] is True
    assert payload["inputs"]["account_id"] == ACCOUNT_ID
    assert payload["inputs"]["database_url_present"] is True
    assert payload["inputs"]["metadata_resolution"] == {
        "account_id_source": "persisted_report",
        "account_id_supplied": False,
        "report_row_checked": True,
    }
    assert "postgres://example" not in json.dumps(payload)

    event = json.loads(calls[1]["body"])
    assert event["data"]["object"]["metadata"] == {
        "source": "content_ops_deflection_report",
        "account_id": ACCOUNT_ID,
        "request_id": "content-ops-123",
    }


def test_main_rejects_mismatched_persisted_account_before_network(
    monkeypatch,
    tmp_path,
) -> None:
    def _unexpected_network(*_args, **_kwargs):
        raise AssertionError("mismatched account must stop before network")

    monkeypatch.setattr(smoke, "_lookup_report_account_id", lambda *_args: ACCOUNT_ID)
    monkeypatch.setattr(smoke, "_open_http_request", _unexpected_network)

    args = _base_args(tmp_path)
    args[args.index("--account-id") + 1] = OTHER_ACCOUNT_ID
    code = smoke.main([
        *args,
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
    ])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "--account-id does not match the persisted deflection report row"
    ]
    assert payload["inputs"]["account_id"] == ACCOUNT_ID
    assert payload["inputs"]["metadata_resolution"] == {
        "account_id_source": "persisted_report",
        "account_id_supplied": True,
        "report_row_checked": True,
    }


def test_main_treats_equals_form_account_id_as_explicit(
    monkeypatch,
    tmp_path,
) -> None:
    def _unexpected_network(*_args, **_kwargs):
        raise AssertionError("mismatched account must stop before network")

    monkeypatch.setattr(smoke, "_lookup_report_account_id", lambda *_args: ACCOUNT_ID)
    monkeypatch.setattr(smoke, "_open_http_request", _unexpected_network)

    code = smoke.main([
        *_without_account_id(_base_args(tmp_path)),
        f"--account-id={OTHER_ACCOUNT_ID}",
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
    ])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "--account-id does not match the persisted deflection report row"
    ]
    assert payload["inputs"]["metadata_resolution"] == {
        "account_id_source": "persisted_report",
        "account_id_supplied": True,
        "report_row_checked": True,
    }


def test_main_reports_missing_persisted_report_before_network(
    monkeypatch,
    tmp_path,
) -> None:
    def _missing(*_args):
        raise RuntimeError("persisted deflection report row was not found")

    def _unexpected_network(*_args, **_kwargs):
        raise AssertionError("missing report must stop before network")

    monkeypatch.setattr(smoke, "_lookup_report_account_id", _missing)
    monkeypatch.setattr(smoke, "_open_http_request", _unexpected_network)

    code = smoke.main([
        *_without_account_id(_base_args(tmp_path)),
        "--account-id",
        "",
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
    ])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["persisted deflection report row was not found"]
    assert payload["inputs"]["metadata_resolution"] == {
        "account_id_source": "unresolved",
        "account_id_supplied": False,
        "report_row_checked": False,
    }
    assert "postgres://example" not in json.dumps(payload)


def test_main_reports_database_lookup_failure_before_network(
    monkeypatch,
    tmp_path,
) -> None:
    async def _lookup_failure(*_args):
        raise OSError("database connection failed")

    def _unexpected_network(*_args, **_kwargs):
        raise AssertionError("database lookup failure must stop before network")

    monkeypatch.setattr(smoke, "_fetch_report_account_ids", _lookup_failure)
    monkeypatch.setattr(smoke, "_open_http_request", _unexpected_network)

    code = smoke.main([
        *_without_account_id(_base_args(tmp_path)),
        "--account-id",
        "",
        "--database-url",
        "postgres://example",
        "--derive-account-id-from-report",
    ])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert payload["ok"] is False
    assert payload["errors"] == [
        "persisted report lookup failed: database connection failed"
    ]
    assert payload["inputs"]["metadata_resolution"] == {
        "account_id_source": "unresolved",
        "account_id_supplied": False,
        "report_row_checked": False,
    }
    assert "postgres://example" not in json.dumps(payload)


def test_main_replays_signed_webhook_and_requires_idempotent_response(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    calls: list[dict[str, Any]] = []

    def _open(request, *, timeout):
        body = request.data or b""
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": dict(request.header_items()),
            "body": body.decode("utf-8"),
            "timeout": timeout,
        })
        get_count = len([c for c in calls if c["method"] == "GET"])
        post_count = len([c for c in calls if c["method"] == "POST"])
        if request.get_method() == "GET" and get_count == 1:
            raise _http_error(request.full_url, 403, {"detail": "payment_required"})
        if request.get_method() == "POST" and post_count == 1:
            return FakeResponse(200, {"status": "ok"})
        if request.get_method() == "POST":
            return FakeResponse(200, {"status": "already_processed"})
        return FakeResponse(200, ARTIFACT)

    monkeypatch.setattr(smoke, "_open_http_request", _open)
    monkeypatch.setattr(smoke.time, "time", lambda: 1700000000)

    code = smoke.main([*_base_args(tmp_path), "--replay-webhook", "--json"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert printed == payload
    assert payload["ok"] is True
    assert payload["before_artifact"] == {"status": 403}
    assert payload["webhook"] == {"status": 200}
    assert payload["replay_webhook"] == {
        "status": 200,
        "payload_status": "already_processed",
    }
    assert payload["after_artifact"]["status"] == 200

    assert [call["method"] for call in calls] == ["GET", "POST", "POST", "GET"]
    first_webhook = calls[1]
    replay_webhook = calls[2]
    assert replay_webhook["url"] == first_webhook["url"]
    assert replay_webhook["body"] == first_webhook["body"]
    assert (
        replay_webhook["headers"]["Stripe-signature"]
        == first_webhook["headers"]["Stripe-signature"]
    )


def test_main_rejects_replay_webhook_without_already_processed(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    def _open(request, *, timeout):
        calls.append(request.get_method())
        if request.get_method() == "GET":
            raise _http_error(request.full_url, 403, {"detail": "payment_required"})
        return FakeResponse(200, {"status": "ok"})

    monkeypatch.setattr(smoke, "_open_http_request", _open)

    code = smoke.main([*_base_args(tmp_path), "--replay-webhook"])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert calls == ["GET", "POST", "POST"]
    assert payload["replay_webhook"] == {"status": 200, "payload_status": "ok"}
    assert payload["after_artifact"]["status"] is None
    assert payload["errors"] == ["stripe webhook replay status must be already_processed"]


def test_main_stops_before_webhook_when_artifact_is_not_locked(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def _open(request, *, timeout):
        calls.append(request.get_method())
        return FakeResponse(200, ARTIFACT)

    monkeypatch.setattr(smoke, "_open_http_request", _open)

    code = smoke.main(_base_args(tmp_path))
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert calls == ["GET"]
    assert payload["errors"] == ["pre-webhook artifact status must be 403, got 200"]


def test_main_reports_webhook_failure_without_fetching_unlocked_artifact(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def _open(request, *, timeout):
        calls.append(request.get_method())
        if request.get_method() == "GET":
            raise _http_error(request.full_url, 403, {"detail": "payment_required"})
        raise _http_error(request.full_url, 409, {"detail": "Deflection report not found"})

    monkeypatch.setattr(smoke, "_open_http_request", _open)

    code = smoke.main(_base_args(tmp_path))
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert calls == ["GET", "POST"]
    assert payload["errors"] == ["stripe webhook status must be 200, got 409"]
    assert payload["webhook"] == {
        "status": 409,
        "error_detail": "Deflection report not found",
    }
    assert payload["after_artifact"]["status"] is None


def test_main_redacts_reflected_secret_values_from_error_detail(monkeypatch, tmp_path) -> None:
    secret_values = {
        "whsec_reflectedsecret",
        "sk_live_reflectedsecret",
        "Bearer reflected-token",
        "ATLAS_SAAS_STRIPE_WEBHOOK_SECRET=reflectedsecret",
    }

    def _open(request, *, timeout):
        if request.get_method() == "GET":
            raise _http_error(request.full_url, 403, {"detail": "payment_required"})
        raise _http_error(
            request.full_url,
            400,
            {
                "detail": (
                    "Invalid signature for whsec_reflectedsecret using "
                    "sk_live_reflectedsecret and Bearer reflected-token; "
                    "ATLAS_SAAS_STRIPE_WEBHOOK_SECRET=reflectedsecret"
                )
            },
        )

    monkeypatch.setattr(smoke, "_open_http_request", _open)

    code = smoke.main(_base_args(tmp_path))
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    serialized = json.dumps(payload)

    assert code == 1
    assert payload["webhook"] == {
        "status": 400,
        "error_detail": (
            "Invalid signature for [redacted] using [redacted] and [redacted]; "
            "[redacted]"
        ),
    }
    assert payload["before_artifact"] == {"status": 403}
    for value in secret_values:
        assert value not in serialized


def test_main_rejects_unlocked_artifact_without_paid_report_fields(monkeypatch, tmp_path) -> None:
    def _open(request, *, timeout):
        if request.get_method() == "GET" and request.full_url.endswith("/artifact"):
            if not hasattr(_open, "seen_locked"):
                _open.seen_locked = True
                raise _http_error(request.full_url, 403, {"detail": "payment_required"})
            return FakeResponse(200, {"summary": {}})
        return FakeResponse(200, {"status": "ok"})

    monkeypatch.setattr(smoke, "_open_http_request", _open)

    code = smoke.main(_base_args(tmp_path))
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert code == 1
    assert payload["errors"] == [
        "paid artifact markdown must be a non-empty string",
        "paid artifact faq_result must be an object",
    ]
