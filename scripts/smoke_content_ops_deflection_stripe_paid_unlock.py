#!/usr/bin/env python3
"""Validate hosted FAQ deflection Stripe paid-unlock handoff."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
WEBHOOK_PATH = "/webhooks/stripe"
ARTIFACT_PATH_TEMPLATE = "/api/v1/content-ops/deflection-reports/{request_id}/artifact"
DEFAULT_AMOUNT_CENTS = 150000
DEFAULT_CURRENCY = "usd"
MAX_ERROR_DETAIL_CHARS = 300
SECRET_TOKEN_PATTERNS = (
    re.compile(r"\bwhsec_[A-Za-z0-9_=-]+"),
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9_=-]+"),
    re.compile(r"\brk_(?:live|test)_[A-Za-z0-9_=-]+"),
    re.compile(r"\bATLAS_[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|KEY)[A-Z0-9_]*=[^\s,;]+"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    text: str
    payload: Any = None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetadataResolution:
    account_id: str
    account_id_source: str
    account_id_supplied: bool
    report_row_checked: bool


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
        description="Validate the hosted deflection Stripe paid-unlock flow."
    )
    parser.add_argument("--base-url", default=_env("ATLAS_API_BASE_URL"))
    parser.add_argument("--token", default=_env("ATLAS_B2B_JWT", "ATLAS_TOKEN"))
    parser.add_argument("--webhook-secret", default=_env("ATLAS_SAAS_STRIPE_WEBHOOK_SECRET", "STRIPE_WEBHOOK_SECRET"))
    parser.add_argument("--account-id", default=_env("ATLAS_ACCOUNT_ID", "ATLAS_FAQ_SEARCH_ACCOUNT_ID"))
    parser.add_argument("--database-url", default="")
    parser.add_argument(
        "--derive-account-id-from-report",
        action="store_true",
        help="Look up the report row and use its account_id for Stripe metadata.",
    )
    parser.add_argument("--request-id", default=_env("ATLAS_DEFLECTION_REQUEST_ID"))
    parser.add_argument("--event-id", default=_env("ATLAS_DEFLECTION_STRIPE_EVENT_ID"))
    parser.add_argument("--session-id", default=_env("ATLAS_DEFLECTION_STRIPE_SESSION_ID"))
    parser.add_argument("--amount-total", type=int, default=DEFAULT_AMOUNT_CENTS)
    parser.add_argument("--currency", default=DEFAULT_CURRENCY)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--webhook-path", default=WEBHOOK_PATH)
    parser.add_argument("--artifact-path-template", default=ARTIFACT_PATH_TEMPLATE)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--replay-webhook", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not _clean(args.base_url):
        errors.append("ATLAS_API_BASE_URL or --base-url is required")
    else:
        errors.extend(_hosted_url_errors(_clean(args.base_url), label="--base-url"))
    if not _clean(args.token):
        errors.append("ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required")
    if not _clean(args.webhook_secret):
        errors.append("ATLAS_SAAS_STRIPE_WEBHOOK_SECRET, STRIPE_WEBHOOK_SECRET, or --webhook-secret is required")
    derive_account_id = bool(getattr(args, "derive_account_id_from_report", False))
    account_id = _clean(args.account_id)
    account_id_explicit = bool(
        getattr(args, "account_id_explicit", bool(account_id))
    )
    if not account_id:
        if not derive_account_id:
            errors.append("ATLAS_ACCOUNT_ID, ATLAS_FAQ_SEARCH_ACCOUNT_ID, or --account-id is required")
    elif not derive_account_id or account_id_explicit:
        try:
            uuid.UUID(account_id)
        except ValueError:
            errors.append("--account-id must be a UUID for the Stripe metadata contract")
    if derive_account_id and not _clean(getattr(args, "database_url", "")):
        errors.append("--database-url is required with --derive-account-id-from-report")
    if not _clean(args.request_id):
        errors.append("ATLAS_DEFLECTION_REQUEST_ID or --request-id is required")
    if int(args.amount_total) <= 0:
        errors.append("--amount-total must be positive")
    if _clean(args.currency).lower() != DEFAULT_CURRENCY:
        errors.append(f"--currency must be {DEFAULT_CURRENCY}")
    if not math.isfinite(float(args.timeout)) or float(args.timeout) <= 0:
        errors.append("--timeout must be a positive finite number")
    for attr, label in (
        ("webhook_path", "--webhook-path"),
        ("artifact_path_template", "--artifact-path-template"),
    ):
        value = _clean(getattr(args, attr))
        if not value.startswith("/"):
            errors.append(f"{label} must start with /")
    if "{request_id}" not in _clean(args.artifact_path_template):
        errors.append("--artifact-path-template must include {request_id}")
    return errors


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _open_http_request(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _report_lookup_error(exc: BaseException) -> RuntimeError:
    detail = _redact_error_detail(exc)
    message = "persisted report lookup failed"
    if detail:
        message = f"{message}: {detail}"
    return RuntimeError(message)


async def _fetch_report_account_ids(database_url: str, request_id: str) -> list[str]:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError("asyncpg is required to derive account_id from the report row") from exc

    try:
        pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=1)
        try:
            rows = await pool.fetch(
                """
                SELECT account_id::text AS account_id
                FROM content_ops_deflection_reports
                WHERE request_id = $1
                ORDER BY created_at DESC
                LIMIT 2
                """,
                request_id,
            )
        finally:
            await pool.close()
    except (
        OSError,
        TimeoutError,
        ValueError,
        asyncio.TimeoutError,
        asyncpg.PostgresError,
    ) as exc:
        raise _report_lookup_error(exc) from exc
    return [_clean(row["account_id"]) for row in rows if _clean(row["account_id"])]


def _lookup_report_account_id(database_url: str, request_id: str) -> str:
    try:
        account_ids = asyncio.run(_fetch_report_account_ids(database_url, request_id))
    except (OSError, TimeoutError, ValueError, asyncio.TimeoutError) as exc:
        raise _report_lookup_error(exc) from exc
    if not account_ids:
        raise RuntimeError("persisted deflection report row was not found")
    if len(account_ids) > 1:
        raise RuntimeError("persisted deflection report request_id is ambiguous")
    account_id = account_ids[0]
    try:
        uuid.UUID(account_id)
    except ValueError as exc:
        raise RuntimeError("persisted report account_id is not a UUID") from exc
    return account_id


def _resolve_metadata(args: argparse.Namespace) -> tuple[MetadataResolution | None, list[str]]:
    supplied_account_id = _clean(args.account_id)
    account_id_explicit = bool(
        getattr(args, "account_id_explicit", bool(supplied_account_id))
    )
    if not bool(getattr(args, "derive_account_id_from_report", False)):
        return (
            MetadataResolution(
                account_id=supplied_account_id,
                account_id_source="supplied",
                account_id_supplied=bool(supplied_account_id),
                report_row_checked=False,
            ),
            [],
        )

    try:
        persisted_account_id = _lookup_report_account_id(
            _clean(args.database_url),
            _clean(args.request_id),
        )
    except RuntimeError as exc:
        return (None, [str(exc)])

    if account_id_explicit and supplied_account_id and supplied_account_id != persisted_account_id:
        return (
            MetadataResolution(
                account_id=persisted_account_id,
                account_id_source="persisted_report",
                account_id_supplied=True,
                report_row_checked=True,
            ),
            ["--account-id does not match the persisted deflection report row"],
        )

    return (
        MetadataResolution(
            account_id=persisted_account_id,
            account_id_source="persisted_report",
            account_id_supplied=account_id_explicit and bool(supplied_account_id),
            report_row_checked=True,
        ),
        [],
    )


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    if not exc.fp:
        return ""
    return exc.read().decode("utf-8", errors="replace")


def _json_request(
    method: str,
    url: str,
    *,
    timeout: float,
    token: str = "",
    body: bytes | None = None,
    stripe_signature: str = "",
) -> HttpResult:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        headers["Content-Type"] = "application/json"
    if stripe_signature:
        headers["Stripe-Signature"] = stripe_signature
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with _open_http_request(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return HttpResult(
                status=int(response.getcode()),
                text=text,
                payload=json.loads(text) if text else None,
            )
    except urllib.error.HTTPError as exc:
        text = _read_http_error(exc)
        payload = None
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
        return HttpResult(status=int(exc.code), text=text, payload=payload, errors=(f"HTTP {exc.code}",))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return HttpResult(status=None, text="", errors=(str(exc),))


def _stripe_event_payload(
    *,
    event_id: str,
    session_id: str,
    account_id: str,
    request_id: str,
    amount_total: int,
    currency: str,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "object": "event",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": session_id,
                "object": "checkout.session",
                "mode": "payment",
                "payment_status": "paid",
                "amount_total": int(amount_total),
                "currency": currency,
                "metadata": {
                    "source": "content_ops_deflection_report",
                    "account_id": account_id,
                    "request_id": request_id,
                },
            }
        },
    }


def _event_body(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _stripe_signature_header(body: bytes, *, secret: str, timestamp: int) -> str:
    signed_payload = f"{timestamp}.{body.decode('utf-8')}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def _artifact_errors(artifact: Any) -> list[str]:
    if not isinstance(artifact, Mapping):
        return ["paid artifact response must be an object"]
    errors: list[str] = []
    if not isinstance(artifact.get("markdown"), str) or not artifact.get("markdown"):
        errors.append("paid artifact markdown must be a non-empty string")
    if not isinstance(artifact.get("faq_result"), Mapping):
        errors.append("paid artifact faq_result must be an object")
    return errors


def _redact_error_detail(value: Any) -> str:
    detail = _clean(value)
    for pattern in SECRET_TOKEN_PATTERNS:
        detail = pattern.sub("[redacted]", detail)
    if len(detail) > MAX_ERROR_DETAIL_CHARS:
        detail = f"{detail[:MAX_ERROR_DETAIL_CHARS]}..."
    return detail


def _safe_error_detail(result: HttpResult | None) -> str:
    if result is None or not isinstance(result.payload, Mapping):
        return ""
    detail = result.payload.get("detail")
    if detail is None or isinstance(detail, (Mapping, Sequence)) and not isinstance(detail, str):
        return ""
    return _redact_error_detail(detail)


def _http_summary(
    result: HttpResult | None,
    *,
    include_error_detail: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"status": result.status if result else None}
    detail = _safe_error_detail(result) if include_error_detail else ""
    if include_error_detail and detail:
        summary["error_detail"] = detail
    return summary


def _generated_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _result_payload(
    *,
    args: argparse.Namespace,
    preflight_errors: Sequence[str],
    event_id: str = "",
    session_id: str = "",
    before_artifact: HttpResult | None = None,
    webhook: HttpResult | None = None,
    replay_webhook: HttpResult | None = None,
    after_artifact: HttpResult | None = None,
    errors: Sequence[str] = (),
    metadata_resolution: MetadataResolution | None = None,
) -> dict[str, Any]:
    resolved_account_id = (
        metadata_resolution.account_id
        if metadata_resolution is not None
        else _clean(args.account_id)
    )
    return {
        "ok": not preflight_errors and not errors,
        "preflight_errors": list(preflight_errors),
        "errors": list(errors),
        "inputs": {
            "base_url": _clean(args.base_url),
            "request_id": _clean(args.request_id),
            "account_id": resolved_account_id,
            "event_id": event_id,
            "session_id": session_id,
            "token_present": bool(_clean(args.token)),
            "webhook_secret_present": bool(_clean(args.webhook_secret)),
            "database_url_present": bool(_clean(getattr(args, "database_url", ""))),
            "metadata_resolution": (
                {
                    "account_id_source": metadata_resolution.account_id_source,
                    "account_id_supplied": metadata_resolution.account_id_supplied,
                    "report_row_checked": metadata_resolution.report_row_checked,
                }
                if metadata_resolution is not None
                else {
                    "account_id_source": "unresolved",
                    "account_id_supplied": bool(_clean(args.account_id)),
                    "report_row_checked": False,
                }
            ),
        },
        "before_artifact": _http_summary(
            before_artifact,
            include_error_detail=before_artifact is not None and before_artifact.status != 403,
        ),
        "webhook": _http_summary(webhook),
        "replay_webhook": {
            **_http_summary(replay_webhook),
            "payload_status": (
                replay_webhook.payload.get("status")
                if replay_webhook and isinstance(replay_webhook.payload, Mapping)
                else None
            ),
        },
        "after_artifact": _http_summary(after_artifact),
    }


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_argv)
    args.account_id_explicit = any(
        value == "--account-id" or value.startswith("--account-id=")
        for value in raw_argv
    )
    preflight_errors = _validate_args(args)
    event_id = _clean(args.event_id) or _generated_id("evt_content_ops_deflection_paid")
    session_id = _clean(args.session_id) or _generated_id("cs_content_ops_deflection")
    if preflight_errors or args.preflight_only:
        payload = _result_payload(
            args=args,
            preflight_errors=preflight_errors,
            event_id=event_id,
            session_id=session_id,
        )
        _write_result(args.output_result, payload)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print("preflight failed" if preflight_errors else "preflight passed")
        return 2 if preflight_errors else 0

    metadata_resolution, metadata_errors = _resolve_metadata(args)
    if metadata_errors:
        payload = _result_payload(
            args=args,
            preflight_errors=(),
            event_id=event_id,
            session_id=session_id,
            metadata_resolution=metadata_resolution,
            errors=metadata_errors,
        )
        _write_result(args.output_result, payload)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print("stripe paid-unlock smoke failed")
        return 1
    if metadata_resolution is None:
        raise AssertionError("metadata resolution unexpectedly missing")

    request_id = _clean(args.request_id)
    artifact_path = _clean(args.artifact_path_template).format(
        request_id=urllib.parse.quote(request_id, safe="")
    )
    artifact_url = _join_url(_clean(args.base_url), artifact_path)
    before_artifact = _json_request(
        "GET",
        artifact_url,
        token=_clean(args.token),
        timeout=float(args.timeout),
    )
    errors: list[str] = []
    if before_artifact.status != 403:
        errors.append(f"pre-webhook artifact status must be 403, got {before_artifact.status}")
        payload = _result_payload(
            args=args,
            preflight_errors=(),
            event_id=event_id,
            session_id=session_id,
            before_artifact=before_artifact,
            errors=errors,
            metadata_resolution=metadata_resolution,
        )
        _write_result(args.output_result, payload)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print("stripe paid-unlock smoke failed")
        return 1

    event = _stripe_event_payload(
        event_id=event_id,
        session_id=session_id,
        account_id=metadata_resolution.account_id,
        request_id=request_id,
        amount_total=int(args.amount_total),
        currency=_clean(args.currency).lower(),
    )
    body = _event_body(event)
    signature = _stripe_signature_header(
        body,
        secret=_clean(args.webhook_secret),
        timestamp=int(time.time()),
    )
    webhook = _json_request(
        "POST",
        _join_url(_clean(args.base_url), _clean(args.webhook_path)),
        body=body,
        stripe_signature=signature,
        timeout=float(args.timeout),
    )
    if webhook.status != 200:
        errors.append(f"stripe webhook status must be 200, got {webhook.status}")
    elif not isinstance(webhook.payload, Mapping) or webhook.payload.get("status") != "ok":
        errors.append("stripe webhook response status must be ok")

    replay_webhook = None
    if not errors and args.replay_webhook:
        replay_webhook = _json_request(
            "POST",
            _join_url(_clean(args.base_url), _clean(args.webhook_path)),
            body=body,
            stripe_signature=signature,
            timeout=float(args.timeout),
        )
        if replay_webhook.status != 200:
            errors.append(
                f"stripe webhook replay status must be 200, got {replay_webhook.status}"
            )
        elif (
            not isinstance(replay_webhook.payload, Mapping)
            or replay_webhook.payload.get("status") != "already_processed"
        ):
            errors.append("stripe webhook replay status must be already_processed")

    after_artifact = None
    if not errors:
        after_artifact = _json_request(
            "GET",
            artifact_url,
            token=_clean(args.token),
            timeout=float(args.timeout),
        )
        if after_artifact.status != 200:
            errors.append(f"post-webhook artifact status must be 200, got {after_artifact.status}")
        else:
            errors.extend(_artifact_errors(after_artifact.payload))

    payload = _result_payload(
        args=args,
        preflight_errors=(),
        event_id=event_id,
        session_id=session_id,
        before_artifact=before_artifact,
        webhook=webhook,
        replay_webhook=replay_webhook,
        after_artifact=after_artifact,
        errors=errors,
        metadata_resolution=metadata_resolution,
    )
    _write_result(args.output_result, payload)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("stripe paid-unlock smoke passed" if payload["ok"] else "stripe paid-unlock smoke failed")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
