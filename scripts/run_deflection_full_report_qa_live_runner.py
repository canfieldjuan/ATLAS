#!/usr/bin/env python3
"""Run the live deflection full-report PDF/export QA proof."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from check_deflection_full_report_pdf_export_artifacts import (  # noqa: E402
    DEFAULT_REQUIRED_SURFACES,
    LEAK_PATTERNS as PDF_LEAK_PATTERNS,
    build_pdf_export_scorecard,
)


SCHEMA_VERSION = "deflection_full_report_qa_live_runner.v1"
REPORT_MODEL_PATH_TEMPLATE = (
    "/api/v1/content-ops/deflection-reports/{request_id}/report-model"
)
ARTIFACT_PATH_TEMPLATE = (
    "/api/v1/content-ops/deflection-reports/{request_id}/artifact"
)
LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
SENSITIVE_PATTERNS = (
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)),
    ("secret_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9_=-]+")),
    ("webhook_secret", re.compile(r"\bwhsec_[A-Za-z0-9_=-]+")),
    ("stripe_checkout_session_id", re.compile(r"\bcs_(?:test|live)_[A-Za-z0-9_=-]+")),
    ("stripe_payment_intent_id", re.compile(r"\bpi_[A-Za-z0-9_=-]+")),
    (
        "result_url",
        re.compile(
            r"https?://[^\s\"'<>]+/"
            r"(?:systems/support-ticket-deflection/results|content-ops/deflection-reports)/"
            r"[^\s\"'<>]+",
            re.IGNORECASE,
        ),
    ),
    ("request_id", re.compile(r"\bcontent-ops-[A-Za-z0-9_-]{6,}\b")),
    ("absolute_local_path", re.compile(r"(?<!\w)(?:/home/|/tmp/)")),
    ("customer_email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    (
        "source_id",
        re.compile(r"\bticket[-_](?!source(?:_|$))[A-Za-z0-9][A-Za-z0-9_.:-]*\b"),
    ),
    ("private_note", re.compile(r"\b(?:private|internal)\s+note\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    payload: Any = None
    errors: tuple[str, ...] = ()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--pdf-bytes", type=Path)
    parser.add_argument("--pdf-text", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--report-model-path-template", default=REPORT_MODEL_PATH_TEMPLATE)
    parser.add_argument("--artifact-path-template", default=ARTIFACT_PATH_TEMPLATE)
    parser.add_argument(
        "--required-surface",
        action="append",
        default=[],
        help="Required artifact surface. Repeat for multiple surfaces.",
    )
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--pretty", action="store_true")
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


def _validate_file(path: Path | None, *, label: str) -> list[str]:
    if path is None:
        return [f"{label} is required"]
    try:
        if not path.is_file():
            return [f"{label} must be a readable file"]
    except OSError:
        return [f"{label} must be a readable file"]
    return []


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not _clean(args.base_url):
        errors.append("--base-url is required")
    else:
        errors.extend(_hosted_url_errors(_clean(args.base_url), label="--base-url"))
    if not _clean(args.token):
        errors.append("--token is required")
    if not _clean(args.request_id):
        errors.append("--request-id is required")
    if not math.isfinite(float(args.timeout)) or float(args.timeout) <= 0:
        errors.append("--timeout must be a positive finite number")
    for attr, label in (
        ("report_model_path_template", "--report-model-path-template"),
        ("artifact_path_template", "--artifact-path-template"),
    ):
        value = _clean(getattr(args, attr))
        if not value.startswith("/"):
            errors.append(f"{label} must start with /")
        if "{request_id}" not in value:
            errors.append(f"{label} must include {{request_id}}")
    errors.extend(_validate_file(args.pdf_bytes, label="--pdf-bytes"))
    errors.extend(_validate_file(args.pdf_text, label="--pdf-text"))
    return errors


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _open_http_request(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _fetch_json(url: str, *, token: str, timeout: float) -> HttpResult:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with _open_http_request(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body) if body else None
            except json.JSONDecodeError:
                return HttpResult(status=int(response.getcode()), errors=("invalid_json",))
            return HttpResult(status=int(response.getcode()), payload=payload)
    except urllib.error.HTTPError as exc:
        return HttpResult(status=int(exc.code), errors=("http_error",))
    except (OSError, urllib.error.URLError):
        return HttpResult(status=None, errors=("network_error",))


def _read_bytes(path: Path, *, label: str) -> tuple[bytes, list[str]]:
    try:
        return path.read_bytes(), []
    except OSError:
        return b"", [f"{label} could not be read"]


def _read_text(path: Path, *, label: str) -> tuple[str, list[str]]:
    try:
        return path.read_text(encoding="utf-8"), []
    except OSError:
        return "", [f"{label} could not be read"]


def _status_summary(result: HttpResult | None) -> dict[str, Any]:
    if result is None:
        return {"status": None}
    summary: dict[str, Any] = {"status": result.status}
    if result.errors:
        summary["errors"] = list(result.errors)
    return summary


def _input_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "base_url_present": bool(_clean(args.base_url)),
        "token_present": bool(_clean(args.token)),
        "request_id_present": bool(_clean(args.request_id)),
        "pdf_bytes_present": args.pdf_bytes is not None,
        "pdf_text_present": args.pdf_text is not None,
        "preflight_only": bool(args.preflight_only),
    }


def _forbidden_values(args: argparse.Namespace) -> tuple[str, ...]:
    values = (
        _clean(args.base_url),
        _clean(args.token),
        _clean(args.request_id),
        str(args.pdf_bytes or ""),
        str(args.pdf_text or ""),
    )
    return tuple(value for value in values if value)


def _pdf_byte_leak_errors(pdf_bytes: bytes) -> list[str]:
    text = pdf_bytes.decode("utf-8", errors="replace")
    errors: list[str] = []
    for label, pattern in PDF_LEAK_PATTERNS:
        if pattern.search(text):
            errors.append(f"pdf bytes contain sensitive pattern: {label}")
    for label, pattern in SENSITIVE_PATTERNS:
        if pattern.search(text) and label not in {
            error.rsplit(": ", 1)[-1]
            for error in errors
        }:
            errors.append(f"pdf bytes contain sensitive pattern: {label}")
    return errors


def _sanitizer_errors(payload: Mapping[str, Any], forbidden_values: Sequence[str]) -> list[str]:
    encoded = json.dumps(payload, sort_keys=True)
    errors: list[str] = []
    for value in forbidden_values:
        if value and value in encoded:
            errors.append("runner output contains a forbidden live input value")
            break
    for label, pattern in SENSITIVE_PATTERNS:
        if pattern.search(encoded):
            errors.append(f"runner output contains sensitive pattern: {label}")
    return errors


def _safe_failure_payload(args: argparse.Namespace, errors: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "inputs": _input_summary(args),
        "errors": list(errors),
    }


def _write_result(
    path: Path | None,
    payload: Mapping[str, Any],
    *,
    pretty: bool,
    forbidden_values: Sequence[str],
    safe_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    output = dict(payload)
    sanitizer_errors = _sanitizer_errors(output, forbidden_values)
    sanitizer_failed = bool(sanitizer_errors)
    if sanitizer_failed:
        output = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "inputs": dict(safe_inputs),
            "errors": sanitizer_errors,
        }
        if _sanitizer_errors(output, ()):
            output = {
                "schema_version": SCHEMA_VERSION,
                "ok": False,
                "errors": ["runner output sanitizer failed closed"],
            }
    text = json.dumps(output, indent=2 if pretty else None, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return output, sanitizer_failed


def _emit_result(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    *,
    exit_code: int,
) -> int:
    output, sanitizer_failed = _write_result(
        args.output_result,
        payload,
        pretty=bool(args.pretty),
        forbidden_values=_forbidden_values(args),
        safe_inputs=_input_summary(args),
    )
    if args.json:
        print(json.dumps(output, sort_keys=True))
    else:
        print("live full-report QA passed" if output.get("ok") is True else "live full-report QA failed")
    if sanitizer_failed:
        return 1
    return exit_code


def _request_path(template: str, request_id: str) -> str:
    return template.format(request_id=urllib.parse.quote(request_id, safe=""))


def _fetch_live_inputs(args: argparse.Namespace) -> tuple[HttpResult, HttpResult]:
    request_id = _clean(args.request_id)
    report_model_url = _join_url(
        _clean(args.base_url),
        _request_path(_clean(args.report_model_path_template), request_id),
    )
    artifact_url = _join_url(
        _clean(args.base_url),
        _request_path(_clean(args.artifact_path_template), request_id),
    )
    return (
        _fetch_json(report_model_url, token=_clean(args.token), timeout=float(args.timeout)),
        _fetch_json(artifact_url, token=_clean(args.token), timeout=float(args.timeout)),
    )


def _live_payload_errors(
    *,
    report_model: HttpResult,
    artifact: HttpResult,
) -> tuple[list[str], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    errors: list[str] = []
    model_payload: Mapping[str, Any] = {}
    artifact_payload: Mapping[str, Any] = {}
    evidence_export: Mapping[str, Any] = {}

    if report_model.status != 200:
        errors.append(f"report-model endpoint must return 200, got {report_model.status}")
    elif not isinstance(report_model.payload, Mapping):
        errors.append("report-model endpoint response must be a JSON object")
    else:
        model_payload = report_model.payload

    if artifact.status != 200:
        errors.append(f"artifact endpoint must return 200, got {artifact.status}")
    elif not isinstance(artifact.payload, Mapping):
        errors.append("artifact endpoint response must be a JSON object")
    else:
        artifact_payload = artifact.payload
        raw_export = artifact_payload.get("evidence_export")
        if not isinstance(raw_export, Mapping):
            errors.append("artifact.evidence_export must be an object")
        else:
            evidence_export = raw_export
        if "report_model" in artifact_payload:
            raw_model = artifact_payload.get("report_model")
            if not isinstance(raw_model, Mapping):
                errors.append("artifact.report_model must be an object when present")
            elif model_payload and raw_model != model_payload:
                errors.append("artifact.report_model must match the report-model route")

    return errors, model_payload, artifact_payload, evidence_export


def _run(args: argparse.Namespace) -> dict[str, Any]:
    pdf_bytes, byte_errors = _read_bytes(args.pdf_bytes, label="--pdf-bytes")
    pdf_text, text_errors = _read_text(args.pdf_text, label="--pdf-text")
    read_errors = [*byte_errors, *text_errors]
    if pdf_text.strip() == "":
        read_errors.append("pdf text extraction must be non-empty")
    read_errors.extend(_pdf_byte_leak_errors(pdf_bytes))
    if read_errors:
        return _safe_failure_payload(args, read_errors)

    report_model_result, artifact_result = _fetch_live_inputs(args)
    payload_errors, report_model, _artifact, evidence_export = _live_payload_errors(
        report_model=report_model_result,
        artifact=artifact_result,
    )
    fetches = {
        "report_model": _status_summary(report_model_result),
        "artifact": _status_summary(artifact_result),
    }
    if payload_errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "inputs": _input_summary(args),
            "fetches": fetches,
            "errors": payload_errors,
        }

    scorecard = build_pdf_export_scorecard(
        report_model=report_model,
        evidence_export=evidence_export,
        pdf_bytes=pdf_bytes,
        pdf_text=pdf_text,
        required_surfaces=tuple(args.required_surface) or DEFAULT_REQUIRED_SURFACES,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": scorecard.get("ok") is True,
        "inputs": _input_summary(args),
        "fetches": fetches,
        "pdf_text": {
            "source": "operator_asserted",
            "verified_from_pdf_bytes": False,
        },
        "scorecard": scorecard,
        "errors": [] if scorecard.get("ok") is True else ["scorecard failed"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    preflight_errors = _validate_args(args)
    if preflight_errors or args.preflight_only:
        payload = _safe_failure_payload(args, preflight_errors)
        if args.preflight_only and not preflight_errors:
            payload["ok"] = True
        return _emit_result(args, payload, exit_code=2 if preflight_errors else 0)

    payload = _run(args)
    return _emit_result(args, payload, exit_code=0 if payload.get("ok") is True else 1)


if __name__ == "__main__":
    raise SystemExit(main())
