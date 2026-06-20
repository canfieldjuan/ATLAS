#!/usr/bin/env python3
"""Create a sanitized Stripe test-mode Checkout proof for deflection reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _deflection_http as deflection_http  # noqa: E402
import smoke_content_ops_deflection_stripe_paid_unlock as paid_unlock

HttpResult = deflection_http.HttpResponse

AUTHORIZATION_PATH_TEMPLATE = (
    "/api/v1/content-ops/deflection-reports/{request_id}/checkout-authorization"
)
ARTIFACT_PATH_TEMPLATE = "/api/v1/content-ops/deflection-reports/{request_id}/artifact"
DEFAULT_CURRENCY = "usd"
MAX_ERROR_DETAIL_CHARS = 300
LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("stripe_secret_key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9_=-]+")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)),
    ("database_url", re.compile(r"\bpostgres(?:ql)?://[^\s\"']+", re.IGNORECASE)),
    ("checkout_session_id", re.compile(r"\bcs_(?:test|live)_[A-Za-z0-9_=-]+")),
    ("checkout_url", re.compile(r"https://checkout\.stripe\.com/[^\s\"']+")),
    (
        "result_url",
        re.compile(
            r"https://[^\s\"']*/systems/support-ticket-deflection/results/[^\s\"']+"
        ),
    ),
    ("request_id", re.compile(r"\bcontent-ops-[A-Za-z0-9_-]+\b")),
    (
        "uuid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
            r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
        ),
    ),
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


@dataclass(frozen=True)
class CheckoutTerms:
    amount_cents: int
    currency: str
    price_id: str


@dataclass(frozen=True)
class CheckoutSessionResult:
    session_id: str
    url: str
    status: str
    payment_status: str


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _build_parser() -> argparse.ArgumentParser:
    _load_dotenv_files()
    parser = argparse.ArgumentParser(
        description="Create a sanitized test-mode Stripe Checkout proof for a deflection report."
    )
    parser.add_argument("--base-url", default=_env("ATLAS_API_BASE_URL"))
    parser.add_argument("--token", default=_env("ATLAS_B2B_JWT", "ATLAS_TOKEN"))
    parser.add_argument("--database-url", default=_env("DATABASE_URL", "ATLAS_DATABASE_URL"))
    parser.add_argument("--stripe-key", default=_env("STRIPE_SECRET_KEY", "ATLAS_STRIPE_SECRET_KEY"))
    parser.add_argument("--request-id", default=_env("ATLAS_DEFLECTION_REQUEST_ID"))
    parser.add_argument("--success-url", default="")
    parser.add_argument("--cancel-url", default="")
    parser.add_argument("--authorization-path-template", default=AUTHORIZATION_PATH_TEMPLATE)
    parser.add_argument("--artifact-path-template", default=ARTIFACT_PATH_TEMPLATE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--wait-for-unlock", action="store_true")
    parser.add_argument("--poll-timeout", type=float, default=600.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--print-checkout-url", action="store_true")
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _hosted_url_errors(value: str, *, label: str) -> list[str]:
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return [f"{label} must be an absolute HTTPS URL"]
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return [f"{label} must be an absolute HTTPS URL"]
    host = (parsed.hostname or "").lower()
    if not host:
        return [f"{label} must include a host"]
    if host in LOCAL_HOSTS or host.startswith("127."):
        return [f"{label} must point to a hosted URL; local hosts are not accepted"]
    return []


def _stripe_key_errors(value: str) -> list[str]:
    if not value:
        return ["STRIPE_SECRET_KEY, ATLAS_STRIPE_SECRET_KEY, or --stripe-key is required"]
    if value.startswith(("sk_live_", "rk_live_")):
        return ["--stripe-key must be a test-mode key for the no-live-charge proof"]
    if not value.startswith(("sk_test_", "rk_test_")):
        return ["--stripe-key must start with sk_test_ or rk_test_"]
    return []


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    for attr, label in (
        ("base_url", "--base-url"),
        ("success_url", "--success-url"),
        ("cancel_url", "--cancel-url"),
    ):
        value = _clean(getattr(args, attr))
        if not value:
            errors.append(f"{label} is required")
        else:
            errors.extend(_hosted_url_errors(value, label=label))
    if not _clean(args.token):
        errors.append("ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required")
    if not _clean(args.database_url):
        errors.append("DATABASE_URL, ATLAS_DATABASE_URL, or --database-url is required")
    errors.extend(_stripe_key_errors(_clean(args.stripe_key)))
    if not _clean(args.request_id):
        errors.append("ATLAS_DEFLECTION_REQUEST_ID or --request-id is required")
    for attr, label in (
        ("authorization_path_template", "--authorization-path-template"),
        ("artifact_path_template", "--artifact-path-template"),
    ):
        value = _clean(getattr(args, attr))
        if not value.startswith("/"):
            errors.append(f"{label} must start with /")
        if "{request_id}" not in value:
            errors.append(f"{label} must include {{request_id}}")
    for attr, label in (
        ("timeout", "--timeout"),
        ("poll_timeout", "--poll-timeout"),
    ):
        value = float(getattr(args, attr))
        if value < 0 or value == float("inf") or value != value:
            errors.append(f"{label} must be a finite non-negative number")
    if float(args.timeout) == 0:
        errors.append("--timeout must be greater than zero")
    if float(args.poll_interval) < 0 or float(args.poll_interval) == float("inf") or float(args.poll_interval) != float(args.poll_interval):
        errors.append("--poll-interval must be a finite non-negative number")
    return errors


def _open_http_request(request: urllib.request.Request, *, timeout: float) -> Any:
    return deflection_http.open_http_request(request, timeout=timeout)


def _authorization_header(token: str) -> str:
    return deflection_http.bearer_header(token)


def _read_http_error(exc: deflection_http.urllib.error.HTTPError) -> str:
    return deflection_http.read_http_error(exc)


def _json_request(method: str, url: str, *, token: str, timeout: float) -> HttpResult:
    return deflection_http.json_request(
        method,
        url,
        authorization=_authorization_header(token),
        timeout=timeout,
        redactor=_redact_text,
        opener=_open_http_request,
        http_error_template="HTTP {status}",
        transport_error_template="{error}",
        invalid_json_template="{error}",
        invalid_json_status=None,
    )


def _authorization_url(args: argparse.Namespace) -> str:
    request_id = urllib.parse.quote(_clean(args.request_id), safe="")
    path = _clean(args.authorization_path_template).format(request_id=request_id)
    return _join_url(_clean(args.base_url), path)


def _artifact_url(args: argparse.Namespace) -> str:
    request_id = urllib.parse.quote(_clean(args.request_id), safe="")
    path = _clean(args.artifact_path_template).format(request_id=request_id)
    return _join_url(_clean(args.base_url), path)


def _checkout_terms_from_authorization(payload: Any) -> tuple[CheckoutTerms | None, list[str]]:
    if not isinstance(payload, Mapping):
        return None, ["checkout authorization response must be an object"]
    if payload.get("status") != "authorized":
        return None, ["checkout authorization status must be authorized"]
    checkout = payload.get("checkout")
    if not isinstance(checkout, Mapping):
        return None, ["checkout authorization payload must include checkout terms"]
    errors: list[str] = []
    try:
        amount_cents = int(checkout.get("amount_cents"))
    except (TypeError, ValueError):
        amount_cents = 0
    currency = _clean(checkout.get("currency")).lower()
    price_id = _clean(checkout.get("price_id"))
    if amount_cents <= 0:
        errors.append("checkout amount must be positive")
    if currency != DEFAULT_CURRENCY:
        errors.append(f"checkout currency must be {DEFAULT_CURRENCY}")
    if not price_id:
        errors.append("checkout price_id is required")
    if errors:
        return None, errors
    return CheckoutTerms(amount_cents=amount_cents, currency=currency, price_id=price_id), []


def _load_stripe_module() -> Any:
    try:
        import stripe  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError("stripe is required to create a test-mode Checkout Session") from exc
    return stripe


def _idempotency_key(*, account_id: str, request_id: str, price_id: str, success_url: str, cancel_url: str) -> str:
    digest = hashlib.sha256(
        "\0".join((account_id, request_id, price_id, success_url, cancel_url)).encode("utf-8")
    ).hexdigest()[:32]
    return f"deflection-test-checkout:{digest}"


def _safe_session_field(value: Any) -> str:
    text = _clean(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{0,80}", text):
        return ""
    return text


def _create_checkout_session(
    *,
    stripe_key: str,
    terms: CheckoutTerms,
    account_id: str,
    request_id: str,
    success_url: str,
    cancel_url: str,
    timeout: float,
) -> CheckoutSessionResult:
    stripe = _load_stripe_module()
    stripe.api_key = stripe_key
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{"price": terms.price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "source": "content_ops_deflection_report",
                "account_id": account_id,
                "request_id": request_id,
            },
            idempotency_key=_idempotency_key(
                account_id=account_id,
                request_id=request_id,
                price_id=terms.price_id,
                success_url=success_url,
                cancel_url=cancel_url,
            ),
            timeout=timeout,
        )
    except Exception as exc:  # Stripe's concrete exception type is optional in tests.
        raise RuntimeError(f"stripe checkout creation failed: {_redact_text(exc)}") from exc
    return CheckoutSessionResult(
        session_id=_clean(getattr(session, "id", "")),
        url=_clean(getattr(session, "url", "")),
        status=_safe_session_field(getattr(session, "status", "")),
        payment_status=_safe_session_field(getattr(session, "payment_status", "")),
    )


def _retrieve_checkout_session(session_id: str, *, timeout: float) -> tuple[str, str]:
    if not session_id:
        return "", ""
    stripe = _load_stripe_module()
    try:
        session = stripe.checkout.Session.retrieve(session_id, timeout=timeout)
    except Exception as exc:
        return "retrieve_failed", _redact_text(exc)
    return (
        _safe_session_field(getattr(session, "status", "")),
        _safe_session_field(getattr(session, "payment_status", "")),
    )


def _poll_artifact(args: argparse.Namespace) -> tuple[HttpResult | None, list[int | None], list[str]]:
    statuses: list[int | None] = []
    deadline = time.monotonic() + float(args.poll_timeout)
    while True:
        result = _json_request(
            "GET",
            _artifact_url(args),
            token=_clean(args.token),
            timeout=float(args.timeout),
        )
        statuses.append(result.status)
        if result.status == 200:
            return result, statuses, []
        if result.status not in (403, 404, None):
            return result, statuses, [f"paid artifact polling got unexpected status {result.status}"]
        if time.monotonic() >= deadline:
            return result, statuses, ["timed out waiting for paid artifact unlock"]
        if float(args.poll_interval) > 0:
            time.sleep(float(args.poll_interval))


def _redact_text(value: Any) -> str:
    text = _clean(value)
    for _name, pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[redacted]", text)
    if len(text) > MAX_ERROR_DETAIL_CHARS:
        text = f"{text[:MAX_ERROR_DETAIL_CHARS]}..."
    return text


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _sensitive_findings(payload: Mapping[str, Any]) -> list[str]:
    encoded = json.dumps(payload, sort_keys=True)
    return [f"proof output contains sensitive pattern: {name}" for name, pattern in SENSITIVE_PATTERNS if pattern.search(encoded)]


def _result_payload(
    *,
    args: argparse.Namespace,
    preflight_errors: Sequence[str],
    errors: Sequence[str] = (),
    terms: CheckoutTerms | None = None,
    authorization: HttpResult | None = None,
    checkout_session: CheckoutSessionResult | None = None,
    account_resolved: bool = False,
    artifact_statuses: Sequence[int | None] = (),
    retrieved_session_status: str = "",
    retrieved_payment_status: str = "",
) -> dict[str, Any]:
    return {
        "ok": not preflight_errors and not errors,
        "preflight_errors": list(preflight_errors),
        "errors": [_redact_text(error) for error in errors],
        "inputs": {
            "base_url_present": bool(_clean(args.base_url)),
            "token_present": bool(_clean(args.token)),
            "database_url_present": bool(_clean(args.database_url)),
            "stripe_key_present": bool(_clean(args.stripe_key)),
            "request_id_present": bool(_clean(args.request_id)),
            "success_url_present": bool(_clean(args.success_url)),
            "cancel_url_present": bool(_clean(args.cancel_url)),
            "wait_for_unlock": bool(args.wait_for_unlock),
        },
        "metadata_resolution": {
            "account_id_source": "persisted_report" if account_resolved else "unresolved",
            "report_row_checked": account_resolved,
        },
        "checkout_authorization": {
            "status": authorization.status if authorization else None,
            "authorized": bool(terms),
            "amount_cents": terms.amount_cents if terms else None,
            "currency": terms.currency if terms else None,
            "price_id_present": bool(terms and terms.price_id),
        },
        "checkout_session": {
            "created": bool(checkout_session),
            "checkout_url_printed": bool(checkout_session and args.print_checkout_url and not args.json),
            "session_id_present": bool(checkout_session and checkout_session.session_id),
            "url_present": bool(checkout_session and checkout_session.url),
            "status": checkout_session.status if checkout_session else "",
            "payment_status": checkout_session.payment_status if checkout_session else "",
            "retrieved_status": retrieved_session_status,
            "retrieved_payment_status": retrieved_payment_status,
        },
        "artifact_poll": {
            "enabled": bool(args.wait_for_unlock),
            "statuses": list(artifact_statuses),
            "unlocked": 200 in artifact_statuses,
        },
    }


def _finalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_payload(payload)
    findings = _sensitive_findings(sanitized)
    if findings:
        sanitized["ok"] = False
        sanitized.setdefault("errors", [])
        sanitized["errors"] = [*sanitized["errors"], *findings]
    return sanitized


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _emit(args: argparse.Namespace, payload: Mapping[str, Any], *, checkout_url: str = "") -> None:
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    elif checkout_url and args.print_checkout_url:
        print(f"Checkout URL: {checkout_url}")
    elif payload.get("ok"):
        print("test-mode checkout proof step passed")
    else:
        print("test-mode checkout proof step failed")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    preflight_errors = _validate_args(args)
    if preflight_errors or args.preflight_only:
        payload = _finalize_payload(
            _result_payload(args=args, preflight_errors=preflight_errors)
        )
        _write_result(args.output_result, payload)
        _emit(args, payload)
        return 2 if preflight_errors else 0

    errors: list[str] = []
    terms: CheckoutTerms | None = None
    authorization: HttpResult | None = None
    checkout_session: CheckoutSessionResult | None = None
    account_resolved = False
    artifact_statuses: list[int | None] = []
    retrieved_status = ""
    retrieved_payment_status = ""

    try:
        account_id = paid_unlock._lookup_report_account_id(  # noqa: SLF001 - shared operator helper.
            _clean(args.database_url),
            _clean(args.request_id),
        )
        account_resolved = True
    except RuntimeError as exc:
        errors.append(str(exc))
        account_id = ""

    if not errors:
        authorization = _json_request(
            "POST",
            _authorization_url(args),
            token=_clean(args.token),
            timeout=float(args.timeout),
        )
        if authorization.status != 200:
            errors.append(f"checkout authorization status must be 200, got {authorization.status}")
        else:
            terms, term_errors = _checkout_terms_from_authorization(authorization.payload)
            errors.extend(term_errors)

    if not errors and terms is not None:
        try:
            checkout_session = _create_checkout_session(
                stripe_key=_clean(args.stripe_key),
                terms=terms,
                account_id=account_id,
                request_id=_clean(args.request_id),
                success_url=_clean(args.success_url),
                cancel_url=_clean(args.cancel_url),
                timeout=float(args.timeout),
            )
        except RuntimeError as exc:
            errors.append(str(exc))

    if not errors and args.wait_for_unlock:
        _artifact, artifact_statuses, poll_errors = _poll_artifact(args)
        errors.extend(poll_errors)
        if poll_errors and checkout_session is not None:
            retrieved_status, retrieved_payment_status = _retrieve_checkout_session(
                checkout_session.session_id,
                timeout=float(args.timeout),
            )

    payload = _finalize_payload(
        _result_payload(
            args=args,
            preflight_errors=(),
            errors=errors,
            terms=terms,
            authorization=authorization,
            checkout_session=checkout_session,
            account_resolved=account_resolved,
            artifact_statuses=artifact_statuses,
            retrieved_session_status=retrieved_status,
            retrieved_payment_status=retrieved_payment_status,
        )
    )
    _write_result(args.output_result, payload)
    _emit(args, payload, checkout_url=checkout_session.url if checkout_session else "")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
