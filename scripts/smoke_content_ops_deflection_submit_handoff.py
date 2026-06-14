#!/usr/bin/env python3
"""Validate the hosted portfolio deflection submit handoff."""

from __future__ import annotations

import argparse
import json
import math
import os
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
DEFAULT_CSV_FILE = (
    ROOT
    / "docs"
    / "extraction"
    / "validation"
    / "fixtures"
    / "faq_deflection_live_upload_sample.csv"
)
LOCAL_BASE_URL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
SUPPORT_PLATFORMS = frozenset({"zendesk", "intercom", "help_scout", "other"})
CSV_UPLOAD_MAX_BYTES = 50 * 1024 * 1024
SUBMIT_ROW_LIMIT_MAX = CSV_UPLOAD_MAX_BYTES
SUBMIT_PATH = "/api/v1/content-ops/deflection-reports/submit"
SNAPSHOT_PATH_TEMPLATE = "/api/v1/content-ops/deflection-reports/{request_id}/snapshot"
ARTIFACT_PATH_TEMPLATE = "/api/v1/content-ops/deflection-reports/{request_id}/artifact"
FORBIDDEN_SNAPSHOT_KEYS = frozenset({
    "answer",
    "answers",
    "draft",
    "evidence",
    "faq_result",
    "markdown",
    "source_id",
    "source_ids",
    "steps",
    "term_mappings",
})
OPTIONAL_CUSTOMER_WORDING_QUESTION_SOURCES = frozenset({
    "source_policy",
    "topic_fallback",
})

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


@dataclass(frozen=True)
class HttpJsonResponse:
    status: int | None
    payload: Any
    raw_text: str
    errors: tuple[str, ...] = ()


def _load_dotenv_files() -> None:
    if os.getenv("ATLAS_DISABLE_DOTENV") == "1":
        return
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _env_path(*names: str) -> Path | None:
    value = _env(*names)
    return Path(value) if value else None


def _build_parser() -> argparse.ArgumentParser:
    _load_dotenv_files()
    parser = argparse.ArgumentParser(
        description="Run the hosted FAQ deflection portfolio submit handoff smoke."
    )
    parser.add_argument("--base-url", default=_env("ATLAS_API_BASE_URL"))
    parser.add_argument("--token", default=_env("ATLAS_B2B_JWT", "ATLAS_TOKEN"))
    parser.add_argument("--account-id", default=_env("ATLAS_ACCOUNT_ID", "ATLAS_FAQ_SEARCH_ACCOUNT_ID"))
    parser.add_argument("--csv-file", type=Path, default=_env_path("ATLAS_DEFLECTION_SUBMIT_CSV_FILE"))
    parser.add_argument("--blob-url", default=_env("ATLAS_DEFLECTION_SUBMIT_BLOB_URL"))
    parser.add_argument("--support-platform", default=_env("ATLAS_DEFLECTION_SUPPORT_PLATFORM") or "zendesk")
    parser.add_argument("--company-name", default=_env("ATLAS_DEFLECTION_COMPANY_NAME"))
    parser.add_argument("--contact-email", default=_env("ATLAS_DEFLECTION_CONTACT_EMAIL"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-uploaded-bytes", type=int, default=0)
    parser.add_argument("--min-source-row-count", type=int, default=0)
    parser.add_argument("--min-submitted-row-count", type=int, default=0)
    parser.add_argument("--min-generated-questions", type=int, default=0)
    parser.add_argument("--min-repeat-ticket-count", type=int, default=0)
    parser.add_argument("--min-top-question-count", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--submit-path", default=SUBMIT_PATH)
    parser.add_argument("--snapshot-path-template", default=SNAPSHOT_PATH_TEMPLATE)
    parser.add_argument("--artifact-path-template", default=ARTIFACT_PATH_TEMPLATE)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _apply_default_csv_file(args: argparse.Namespace) -> None:
    if not _clean(args.csv_file) and not _clean(args.blob_url):
        args.csv_file = DEFAULT_CSV_FILE


def _validate_args(args: argparse.Namespace) -> list[str]:
    _apply_default_csv_file(args)
    errors: list[str] = []
    if not _clean(args.base_url):
        errors.append("ATLAS_API_BASE_URL or --base-url is required")
    else:
        errors.extend(_base_url_errors(_clean(args.base_url)))
    if not _clean(args.token):
        errors.append("ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required")
    if not _clean(args.account_id):
        errors.append("ATLAS_ACCOUNT_ID, ATLAS_FAQ_SEARCH_ACCOUNT_ID, or --account-id is required")
    if _clean(args.csv_file):
        errors.extend(_csv_file_errors(Path(args.csv_file)))
    elif not _clean(args.blob_url):
        errors.append(
            "ATLAS_DEFLECTION_SUBMIT_CSV_FILE/--csv-file or "
            "ATLAS_DEFLECTION_SUBMIT_BLOB_URL/--blob-url is required"
        )
    else:
        errors.extend(_blob_url_errors(_clean(args.blob_url)))
    if _clean(args.support_platform) not in SUPPORT_PLATFORMS:
        errors.append("--support-platform must be one of: help_scout, intercom, other, zendesk")
    if not _clean(args.company_name):
        errors.append("ATLAS_DEFLECTION_COMPANY_NAME or --company-name is required")
    if not _clean(args.contact_email):
        errors.append("ATLAS_DEFLECTION_CONTACT_EMAIL or --contact-email is required")
    if args.limit is not None and (
        int(args.limit) <= 0 or int(args.limit) > SUBMIT_ROW_LIMIT_MAX
    ):
        errors.append(f"--limit must be between 1 and {SUBMIT_ROW_LIMIT_MAX}")
    for attr, label in (
        ("min_uploaded_bytes", "--min-uploaded-bytes"),
        ("min_source_row_count", "--min-source-row-count"),
        ("min_submitted_row_count", "--min-submitted-row-count"),
        ("min_generated_questions", "--min-generated-questions"),
        ("min_repeat_ticket_count", "--min-repeat-ticket-count"),
        ("min_top_question_count", "--min-top-question-count"),
    ):
        if int(getattr(args, attr)) < 0:
            errors.append(f"{label} must be zero or greater")
    if not math.isfinite(float(args.timeout)) or float(args.timeout) <= 0:
        errors.append("--timeout must be a positive finite number")
    for attr, label in (
        ("submit_path", "--submit-path"),
        ("snapshot_path_template", "--snapshot-path-template"),
        ("artifact_path_template", "--artifact-path-template"),
    ):
        value = _clean(getattr(args, attr))
        if not value.startswith("/"):
            errors.append(f"{label} must start with /")
    for attr, label in (
        ("snapshot_path_template", "--snapshot-path-template"),
        ("artifact_path_template", "--artifact-path-template"),
    ):
        if "{request_id}" not in _clean(getattr(args, attr)):
            errors.append(f"{label} must include {{request_id}}")
    return errors


def _base_url_errors(base_url: str) -> list[str]:
    try:
        parsed = urllib.parse.urlparse(base_url)
    except ValueError:
        return ["--base-url must be an absolute HTTPS URL for hosted proof"]
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return ["--base-url must be an absolute HTTPS URL for hosted proof"]
    host = (parsed.hostname or "").lower()
    if not host:
        return ["--base-url must include a host for hosted proof"]
    if host in LOCAL_BASE_URL_HOSTS or host.startswith("127."):
        return ["--base-url must point to a deployed host; local hosts are not accepted"]
    return []


def _blob_url_errors(blob_url: str) -> list[str]:
    try:
        parsed = urllib.parse.urlparse(blob_url)
    except ValueError:
        return ["--blob-url must be an absolute HTTPS URL"]
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return ["--blob-url must be an absolute HTTPS URL"]
    if parsed.username or parsed.password:
        return ["--blob-url must not include credentials"]
    return []


def _csv_file_errors(csv_file: Path) -> list[str]:
    try:
        path = csv_file.expanduser()
        stat = path.stat()
    except OSError:
        return ["--csv-file must point to a readable support-ticket CSV file"]
    if not path.is_file():
        return ["--csv-file must point to a file"]
    if stat.st_size <= 0:
        return ["--csv-file must not be empty"]
    if stat.st_size > CSV_UPLOAD_MAX_BYTES:
        return ["--csv-file must be 50 MB or smaller"]
    return []


def _submit_mode(args: argparse.Namespace) -> str:
    return "multipart" if _clean(args.csv_file) else "json_blob_url"


def _csv_file_size(args: argparse.Namespace) -> int | None:
    if not _clean(args.csv_file):
        return None
    try:
        return Path(args.csv_file).expanduser().stat().st_size
    except OSError:
        return None


def _required_input_status(args: argparse.Namespace) -> dict[str, dict[str, bool]]:
    return {
        "base_url": {"present": bool(_clean(args.base_url))},
        "token": {"present": bool(_clean(args.token))},
        "account_id": {"present": bool(_clean(args.account_id))},
        "csv_file": {"present": bool(_clean(args.csv_file))},
        "blob_url": {"present": bool(_clean(args.blob_url))},
        "company_name": {"present": bool(_clean(args.company_name))},
        "contact_email": {"present": bool(_clean(args.contact_email))},
    }


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _redacted_blob_host(blob_url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(blob_url)
    except ValueError:
        return ""
    return parsed.hostname or ""


def _submit_body(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {
        "blob_url": _clean(args.blob_url),
        "support_platform": _clean(args.support_platform),
        "company_name": _clean(args.company_name),
        "contact_email": _clean(args.contact_email),
    }
    if args.limit is not None:
        body["limit"] = int(args.limit)
    return body


def _submit_fields(args: argparse.Namespace) -> dict[str, str]:
    fields = {
        "support_platform": _clean(args.support_platform),
        "company_name": _clean(args.company_name),
        "contact_email": _clean(args.contact_email),
    }
    if args.limit is not None:
        fields["limit"] = str(int(args.limit))
    return fields


def _open_http_request(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _parse_http_json_response(
    method: str,
    url: str,
    *,
    request: urllib.request.Request,
    timeout: float,
) -> HttpJsonResponse:
    try:
        with _open_http_request(request, timeout=timeout) as response:
            status = int(getattr(response, "status", None) or response.getcode())
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return HttpJsonResponse(
            status=None,
            payload=None,
            raw_text="",
            errors=(f"{method} {url} transport failed: {exc}",),
        )
    if not raw.strip():
        return HttpJsonResponse(status=status, payload=None, raw_text=raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return HttpJsonResponse(
            status=status,
            payload=None,
            raw_text=raw[:2000],
            errors=(f"{method} {url} returned invalid JSON: {exc.msg}",),
        )
    return HttpJsonResponse(status=status, payload=payload, raw_text=raw[:2000])


def _json_request(
    method: str,
    url: str,
    *,
    token: str,
    timeout: float,
    body: Mapping[str, Any] | None = None,
) -> HttpJsonResponse:
    encoded_body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if body is not None:
        encoded_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=encoded_body, headers=headers, method=method)
    return _parse_http_json_response(method, url, request=request, timeout=timeout)


def _multipart_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def _encode_multipart_body(
    fields: Mapping[str, str],
    *,
    file_field: str,
    file_path: Path,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    boundary = f"atlas-deflection-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend((
            f"--{boundary}\r\n".encode("ascii"),
            (
                'Content-Disposition: form-data; '
                f'name="{_multipart_escape(name)}"\r\n\r\n'
            ).encode("utf-8"),
            value.encode("utf-8"),
            b"\r\n",
        ))
    filename = _multipart_escape(file_path.name or "support-tickets.csv")
    chunks.extend((
        f"--{boundary}\r\n".encode("ascii"),
        (
            'Content-Disposition: form-data; '
            f'name="{_multipart_escape(file_field)}"; filename="{filename}"\r\n'
            "Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("ascii"),
    ))
    return b"".join(chunks), boundary


def _multipart_submit_request(
    url: str,
    *,
    token: str,
    timeout: float,
    fields: Mapping[str, str],
    csv_file: Path,
) -> HttpJsonResponse:
    file_path = csv_file.expanduser()
    body, boundary = _encode_multipart_body(
        fields,
        file_field="csv_file",
        file_path=file_path,
        file_bytes=file_path.read_bytes(),
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    return _parse_http_json_response("POST", url, request=request, timeout=timeout)


def _forbidden_key_paths(value: Any, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}"
            if key_text in FORBIDDEN_SNAPSHOT_KEYS:
                paths.append(child_prefix)
            paths.extend(_forbidden_key_paths(child, prefix=child_prefix))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_key_paths(child, prefix=f"{prefix}[{index}]"))
    return paths


def _validate_snapshot(snapshot: Any, *, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(snapshot, Mapping):
        return [f"{label} must be an object"]
    forbidden_paths = _forbidden_key_paths(snapshot)
    if forbidden_paths:
        errors.append(f"{label} leaked forbidden fields: {', '.join(forbidden_paths[:5])}")
    summary = snapshot.get("summary")
    if not isinstance(summary, Mapping):
        errors.append(f"{label}.summary must be an object")
    else:
        for key in ("generated", "drafted_answer_count", "no_proven_answer_count"):
            if not isinstance(summary.get(key), int):
                errors.append(f"{label}.summary.{key} must be an integer")
    top_questions = snapshot.get("top_questions")
    if not isinstance(top_questions, Sequence) or isinstance(top_questions, (str, bytes, bytearray)):
        errors.append(f"{label}.top_questions must be a list")
    else:
        for index, item in enumerate(top_questions):
            if not isinstance(item, Mapping):
                errors.append(f"{label}.top_questions[{index}] must be an object")
                continue
            if not isinstance(item.get("rank"), int):
                errors.append(f"{label}.top_questions[{index}].rank must be an integer")
            if not _clean(item.get("question")):
                errors.append(f"{label}.top_questions[{index}].question must be non-empty")
            question_source = _clean(item.get("question_source"))
            if (
                not _clean(item.get("customer_wording"))
                and question_source
                and question_source not in OPTIONAL_CUSTOMER_WORDING_QUESTION_SOURCES
            ):
                errors.append(
                    f"{label}.top_questions[{index}].customer_wording must be non-empty "
                    "for customer-wording questions"
                )
            if not isinstance(item.get("weighted_frequency"), (int, float)):
                errors.append(
                    f"{label}.top_questions[{index}].weighted_frequency must be numeric"
                )
    return errors


def _validate_submit_envelope(
    payload: Any,
    *,
    submit_mode: str = "json_blob_url",
) -> tuple[str, Mapping[str, Any] | None, dict[str, Any], list[str]]:
    errors: list[str] = []
    diagnostics: dict[str, Any] = {}
    if not isinstance(payload, Mapping):
        return "", None, diagnostics, ["submit response must be a JSON object"]
    if payload.get("status") != "completed":
        errors.append("submit response status must be completed")
    request_id = _clean(payload.get("request_id"))
    if not request_id:
        errors.append("submit response request_id must be non-empty")
    steps = payload.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
        errors.append("submit response steps must be a list")
        steps = []
    step = next(
        (
            item
            for item in steps
            if isinstance(item, Mapping) and item.get("output") == "faq_deflection_report"
        ),
        None,
    )
    if not isinstance(step, Mapping):
        errors.append("submit response must include faq_deflection_report step")
        return request_id, None, diagnostics, errors
    if step.get("status") != "completed":
        errors.append("faq_deflection_report step status must be completed")
    result = step.get("result")
    if not isinstance(result, Mapping):
        errors.append("faq_deflection_report step result must be an object")
        return request_id, None, diagnostics, errors
    nested_request_id = _clean(result.get("request_id"))
    if not nested_request_id:
        errors.append("faq_deflection_report result request_id must be non-empty")
    elif request_id and nested_request_id != request_id:
        errors.append("faq_deflection_report result request_id must match top-level request_id")
    snapshot = result.get("snapshot")
    errors.extend(_validate_snapshot(snapshot, label="faq_deflection_report.result.snapshot"))
    full_report = result.get("full_report")
    if not isinstance(full_report, Mapping):
        errors.append("faq_deflection_report result full_report must be an object")
    else:
        if full_report.get("status") != "locked":
            errors.append("faq_deflection_report full_report.status must be locked")
        if full_report.get("reason") != "payment_required":
            errors.append("faq_deflection_report full_report.reason must be payment_required")
    input_provider = payload.get("input_provider")
    if not isinstance(input_provider, Mapping):
        errors.append("submit response input_provider must be an object")
    else:
        diagnostics["provider"] = input_provider.get("provider")
        if input_provider.get("provider") != "portfolio_deflection_submit":
            errors.append("input_provider.provider must be portfolio_deflection_submit")
        metadata = input_provider.get("metadata")
        if not isinstance(metadata, Mapping):
            errors.append("input_provider.metadata must be an object")
        else:
            diagnostics["metadata"] = {
                key: metadata.get(key)
                for key in (
                    "source_row_count",
                    "submitted_row_count",
                    "truncated_row_count",
                    "max_source_material_rows",
                    "uploaded_bytes",
                    "blob_bytes",
                    "support_platform",
                )
                if key in metadata
            }
            if metadata.get("source") != "portfolio_deflection_submit":
                errors.append("input_provider.metadata.source must be portfolio_deflection_submit")
            expected_byte_key = "uploaded_bytes" if submit_mode == "multipart" else "blob_bytes"
            expected_byte_value = metadata.get(expected_byte_key)
            if not isinstance(expected_byte_value, int) or expected_byte_value <= 0:
                errors.append(
                    f"input_provider.metadata.{expected_byte_key} must be a positive integer "
                    f"for {submit_mode} submit"
                )
    return request_id, snapshot if isinstance(snapshot, Mapping) else None, diagnostics, errors


def _stale_multipart_submit_route_errors(response: HttpJsonResponse, *, submit_mode: str) -> list[str]:
    if submit_mode != "multipart" or response.status != 422:
        return []
    payload = response.payload
    if not isinstance(payload, Mapping):
        return []
    detail = payload.get("detail")
    if not isinstance(detail, Sequence) or isinstance(detail, (str, bytes, bytearray)):
        return []
    for item in detail:
        if not isinstance(item, Mapping):
            continue
        loc = item.get("loc")
        if (
            item.get("type") == "model_attributes_type"
            and isinstance(loc, Sequence)
            and not isinstance(loc, (str, bytes, bytearray))
            and tuple(loc) == ("body",)
        ):
            return [
                "deployed submit route rejected multipart as a JSON body; "
                "expected multipart CSV Request route, so the host is likely "
                "serving stale route code or importing a stale extracted_content_pipeline"
            ]
    return []


def _status_summary(response: HttpJsonResponse) -> dict[str, Any]:
    return {
        "status": response.status,
        "errors": list(response.errors),
    }


def _configured_volume_gates(args: argparse.Namespace) -> dict[str, int]:
    return {
        "uploaded_bytes": int(args.min_uploaded_bytes),
        "source_row_count": int(args.min_source_row_count),
        "submitted_row_count": int(args.min_submitted_row_count),
        "generated_questions": int(args.min_generated_questions),
        "repeat_ticket_count": int(args.min_repeat_ticket_count),
        "top_question_count": int(args.min_top_question_count),
    }


def _strict_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _volume_gate_actuals(
    *,
    submit_mode: str,
    diagnostics: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
) -> dict[str, int | None]:
    metadata = diagnostics.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    summary = snapshot.get("summary") if isinstance(snapshot, Mapping) else None
    if not isinstance(summary, Mapping):
        summary = {}
    top_questions = snapshot.get("top_questions") if isinstance(snapshot, Mapping) else None
    byte_key = "uploaded_bytes" if submit_mode == "multipart" else "blob_bytes"
    return {
        "uploaded_bytes": _strict_count(metadata.get(byte_key)),
        "source_row_count": _strict_count(metadata.get("source_row_count")),
        "submitted_row_count": _strict_count(metadata.get("submitted_row_count")),
        "generated_questions": _strict_count(summary.get("generated")),
        "repeat_ticket_count": _strict_count(summary.get("repeat_ticket_count")),
        "top_question_count": (
            len(top_questions)
            if isinstance(top_questions, Sequence)
            and not isinstance(top_questions, (str, bytes, bytearray))
            else None
        ),
    }


def _volume_gate_errors(
    args: argparse.Namespace,
    *,
    submit_mode: str,
    diagnostics: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    expected = _configured_volume_gates(args)
    actual = _volume_gate_actuals(
        submit_mode=submit_mode,
        diagnostics=diagnostics,
        snapshot=snapshot,
    )
    errors: list[str] = []
    for gate, minimum in expected.items():
        if minimum <= 0:
            continue
        value = actual.get(gate)
        if value is None:
            errors.append(f"volume gate {gate} expected >= {minimum}, got missing")
        elif value < minimum:
            errors.append(f"volume gate {gate} expected >= {minimum}, got {value}")
    configured = {key: value for key, value in expected.items() if value > 0}
    return {
        "configured": configured,
        "actual": actual,
        "ok": not errors,
        "errors": errors,
    }, errors


def _skipped_volume_gates(
    args: argparse.Namespace,
    *,
    reason: str,
) -> dict[str, Any]:
    configured = {
        key: value
        for key, value in _configured_volume_gates(args).items()
        if value > 0
    }
    if not configured:
        return {"configured": {}, "actual": {}, "ok": True, "errors": []}
    return {
        "configured": configured,
        "actual": {},
        "ok": False,
        "skipped": True,
        "not_run_reason": reason,
        "errors": [],
    }


def _preflight_summary(
    args: argparse.Namespace,
    errors: Sequence[str],
    elapsed: float,
    *,
    reason: str = "preflight_failed",
) -> dict[str, Any]:
    return {
        "ok": not errors,
        "phase": "preflight",
        "preflight_errors": list(errors),
        "required_inputs": _required_input_status(args),
        "submit_mode": _submit_mode(args),
        "csv_file_size": _csv_file_size(args),
        "blob_host": _redacted_blob_host(_clean(args.blob_url)),
        "volume_gates": _skipped_volume_gates(args, reason=reason),
        "submit": {"ok": False, "skipped": True, "not_run_reason": reason},
        "snapshot": {"ok": False, "skipped": True, "not_run_reason": reason},
        "artifact": {"ok": False, "skipped": True, "not_run_reason": reason},
        "elapsed_seconds": elapsed,
    }


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    request_id = summary.get("request_id") or ""
    print(
        "FAQ deflection submit handoff: "
        f"ok={summary['ok']} phase={summary['phase']} request_id={request_id}"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    errors: list[str] = []
    submit_url = _join_url(_clean(args.base_url), _clean(args.submit_path))
    submit_mode = _submit_mode(args)
    if submit_mode == "multipart":
        submit_response = _multipart_submit_request(
            submit_url,
            token=_clean(args.token),
            timeout=float(args.timeout),
            fields=_submit_fields(args),
            csv_file=Path(args.csv_file),
        )
    else:
        submit_response = _json_request(
            "POST",
            submit_url,
            token=_clean(args.token),
            timeout=float(args.timeout),
            body=_submit_body(args),
        )
    submit_errors = list(submit_response.errors)
    request_id = ""
    submit_snapshot: Mapping[str, Any] | None = None
    diagnostics: dict[str, Any] = {}
    volume_gates = _skipped_volume_gates(args, reason="submit_failed")
    submit_errors.extend(
        _stale_multipart_submit_route_errors(submit_response, submit_mode=submit_mode)
    )
    if submit_response.status != 200:
        submit_errors.append(f"submit status must be 200, got {submit_response.status}")
    else:
        request_id, submit_snapshot, diagnostics, envelope_errors = _validate_submit_envelope(
            submit_response.payload,
            submit_mode=submit_mode,
        )
        submit_errors.extend(envelope_errors)
        if not envelope_errors:
            volume_gates, volume_gate_errors = _volume_gate_errors(
                args,
                submit_mode=submit_mode,
                diagnostics=diagnostics,
                snapshot=submit_snapshot,
            )
            errors.extend(volume_gate_errors)
    errors.extend(submit_errors)
    snapshot_summary: dict[str, Any] = {
        "ok": False,
        "skipped": True,
        "not_run_reason": "submit_failed",
    }
    artifact_summary: dict[str, Any] = {
        "ok": False,
        "skipped": True,
        "not_run_reason": "submit_failed",
    }
    if request_id and not submit_errors:
        snapshot_path = _clean(args.snapshot_path_template).format(
            request_id=urllib.parse.quote(request_id, safe="")
        )
        snapshot_response = _json_request(
            "GET",
            _join_url(_clean(args.base_url), snapshot_path),
            token=_clean(args.token),
            timeout=float(args.timeout),
        )
        snapshot_errors = list(snapshot_response.errors)
        if snapshot_response.status != 200:
            snapshot_errors.append(f"snapshot status must be 200, got {snapshot_response.status}")
        snapshot_errors.extend(_validate_snapshot(snapshot_response.payload, label="snapshot response"))
        if submit_snapshot is not None and snapshot_response.payload != submit_snapshot:
            snapshot_errors.append("snapshot response must match submit snapshot")
        errors.extend(snapshot_errors)
        snapshot_summary = {
            **_status_summary(snapshot_response),
            "ok": not snapshot_errors,
            "errors": snapshot_errors,
        }

        artifact_path = _clean(args.artifact_path_template).format(
            request_id=urllib.parse.quote(request_id, safe="")
        )
        artifact_response = _json_request(
            "GET",
            _join_url(_clean(args.base_url), artifact_path),
            token=_clean(args.token),
            timeout=float(args.timeout),
        )
        artifact_errors = list(artifact_response.errors)
        if artifact_response.status != 403:
            artifact_errors.append(f"unpaid artifact status must be 403, got {artifact_response.status}")
        errors.extend(artifact_errors)
        artifact_summary = {
            **_status_summary(artifact_response),
            "ok": not artifact_errors,
            "expected_status": 403,
            "errors": artifact_errors,
        }
    summary = {
        "ok": not errors,
        "phase": "complete",
        "request_id": request_id or None,
        "submit_mode": submit_mode,
        "csv_file_size": _csv_file_size(args),
        "blob_host": _redacted_blob_host(_clean(args.blob_url)),
        "volume_gates": volume_gates,
        "submit": {
            **_status_summary(submit_response),
            "ok": not submit_errors,
            "errors": submit_errors,
            "diagnostics": diagnostics,
        },
        "snapshot": snapshot_summary,
        "artifact": artifact_summary,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    start = time.perf_counter()
    args = _build_parser().parse_args(argv)
    errors = _validate_args(args)
    if errors:
        summary = _preflight_summary(args, errors, time.perf_counter() - start)
        _write_result(args.output_result, summary)
        _print_summary(summary, as_json=args.json)
        return 2
    if args.preflight_only:
        summary = _preflight_summary(args, (), time.perf_counter() - start, reason="preflight_only")
        _write_result(args.output_result, summary)
        _print_summary(summary, as_json=args.json)
        return 0
    summary = run(args)
    _write_result(args.output_result, summary)
    _print_summary(summary, as_json=args.json)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
