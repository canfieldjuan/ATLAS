#!/usr/bin/env python3
"""Run the live deflection full-report PDF/export QA proof."""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
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
from check_deflection_process_contract import (  # noqa: E402
    DEFAULT_CONTRACT_PATH as PROCESS_CONTRACT_PATH,
    EXPECTED_REPORT_MODEL_CONTRACT,
    check_process_contract,
)


SCHEMA_VERSION = "deflection_full_report_qa_live_runner.v1"
REPORT_MODEL_PATH_TEMPLATE = (
    "/api/v1/content-ops/deflection-reports/{request_id}/report-model"
)
ARTIFACT_PATH_TEMPLATE = (
    "/api/v1/content-ops/deflection-reports/{request_id}/artifact"
)
LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
PDF_STREAM_RE = re.compile(
    rb"<<(?P<dict>.*?)>>\s*stream\r?\n(?P<body>.*?)\r?\nendstream",
    re.DOTALL,
)
PDF_FILTER_RE = re.compile(rb"/([A-Za-z0-9]+Decode)\b")
PDF_TJ_RE = re.compile(rb"\((?:\\.|[^\\()])*\)\s*Tj", re.DOTALL)
PDF_TJ_ARRAY_RE = re.compile(rb"\[(.*?)\]\s*TJ", re.DOTALL)
PDF_LITERAL_RE = re.compile(rb"\((?:\\.|[^\\()])*\)", re.DOTALL)
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
    parser.add_argument(
        "--pdf-text",
        type=Path,
        help="Optional extracted PDF text override. Omit to extract from --pdf-bytes.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--process-contract-path", default=PROCESS_CONTRACT_PATH)
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
        ("process_contract_path", "--process-contract-path"),
        ("report_model_path_template", "--report-model-path-template"),
        ("artifact_path_template", "--artifact-path-template"),
    ):
        value = _clean(getattr(args, attr))
        if not value.startswith("/"):
            errors.append(f"{label} must start with /")
        if label != "--process-contract-path" and "{request_id}" not in value:
            errors.append(f"{label} must include {{request_id}}")
    errors.extend(_validate_file(args.pdf_bytes, label="--pdf-bytes"))
    if args.pdf_text is not None:
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


def _pdf_streams(pdf_bytes: bytes) -> list[tuple[bytes, bytes]]:
    return [
        (match.group("dict"), match.group("body"))
        for match in PDF_STREAM_RE.finditer(pdf_bytes)
    ]


def _decode_pdf_stream(stream_dict: bytes, stream: bytes) -> tuple[bytes, str | None]:
    if b"/Filter" not in stream_dict:
        return stream, None
    filters = PDF_FILTER_RE.findall(stream_dict)
    if filters != [b"FlateDecode"]:
        names = ", ".join(
            f.decode("ascii", errors="replace") for f in filters
        ) or "unknown"
        return b"", f"pdf stream uses unsupported filter: {names}"
    try:
        return zlib.decompress(stream), None
    except zlib.error:
        return b"", "pdf stream FlateDecode decompression failed"


def _decoded_pdf_streams(pdf_bytes: bytes) -> tuple[list[bytes], list[str]]:
    decoded: list[bytes] = []
    errors: list[str] = []
    for stream_dict, stream in _pdf_streams(pdf_bytes):
        content, error = _decode_pdf_stream(stream_dict, stream)
        if error:
            errors.append(error)
        else:
            decoded.append(content)
    return decoded, errors


def _decode_pdf_literal(value: bytes) -> str:
    if value.startswith(b"(") and value.endswith(b")"):
        value = value[1:-1]
    out = bytearray()
    index = 0
    while index < len(value):
        char = value[index]
        if char != 0x5C:
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= len(value):
            break
        escaped = value[index]
        index += 1
        if escaped in b"nrtbf":
            out.append({
                ord("n"): 0x0A,
                ord("r"): 0x0D,
                ord("t"): 0x09,
                ord("b"): 0x08,
                ord("f"): 0x0C,
            }[escaped])
        elif escaped in b"()\\":
            out.append(escaped)
        elif escaped in b"\r\n":
            if escaped == 0x0D and index < len(value) and value[index] == 0x0A:
                index += 1
        elif 0x30 <= escaped <= 0x37:
            digits = bytes([escaped])
            while (
                len(digits) < 3
                and index < len(value)
                and 0x30 <= value[index] <= 0x37
            ):
                digits += bytes([value[index]])
                index += 1
            out.append(int(digits, 8) & 0xFF)
        else:
            out.append(escaped)
    return out.decode("latin-1", errors="replace")


def _pdf_text_operands(content: bytes) -> list[str]:
    text: list[str] = []
    for match in PDF_TJ_RE.finditer(content):
        literal = match.group(0).rsplit(b")", 1)[0] + b")"
        decoded = _decode_pdf_literal(literal).strip()
        if decoded:
            text.append(decoded)
    for match in PDF_TJ_ARRAY_RE.finditer(content):
        for literal in PDF_LITERAL_RE.findall(match.group(1)):
            decoded = _decode_pdf_literal(literal).strip()
            if decoded:
                text.append(decoded)
    return text


def _drop_extracted_table_of_contents(lines: Sequence[str]) -> list[str]:
    output: list[str] = []
    in_toc = False
    for line in lines:
        cleaned = line.strip()
        if cleaned.casefold() == "table of contents":
            in_toc = True
            continue
        if in_toc and cleaned.startswith("-"):
            continue
        in_toc = False
        output.append(line)
    return output


def _extract_pdf_text(pdf_bytes: bytes) -> tuple[str, list[str]]:
    if not pdf_bytes.startswith(b"%PDF-"):
        return "", ["pdf text extraction requires PDF bytes"]
    streams = _pdf_streams(pdf_bytes)
    if not streams:
        return "", ["pdf text extraction found no PDF streams"]
    decoded_streams, decode_errors = _decoded_pdf_streams(pdf_bytes)
    if decode_errors:
        return "", decode_errors
    text: list[str] = []
    for stream in decoded_streams:
        text.extend(_pdf_text_operands(stream))
    extracted = "\n".join(part for part in _drop_extracted_table_of_contents(text) if part)
    if not extracted.strip():
        return "", ["pdf text extraction produced no text"]
    return extracted, []


def _pdf_text_input(args: argparse.Namespace, pdf_bytes: bytes) -> tuple[str, list[str], dict[str, Any]]:
    if args.pdf_text is not None:
        pdf_text, errors = _read_text(args.pdf_text, label="--pdf-text")
        return pdf_text, errors, {
            "source": "operator_asserted",
            "verified_from_pdf_bytes": False,
        }
    pdf_text, errors = _extract_pdf_text(pdf_bytes)
    return pdf_text, errors, {
        "source": "extracted_from_pdf_bytes",
        "verified_from_pdf_bytes": True,
    }


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
    errors: list[str] = []
    seen: set[str] = set()
    decoded_streams, decode_errors = _decoded_pdf_streams(pdf_bytes)
    errors.extend(decode_errors)
    texts = [
        pdf_bytes.decode("utf-8", errors="replace"),
        *[
            stream.decode("utf-8", errors="replace")
            for stream in decoded_streams
        ],
    ]
    for label, pattern in PDF_LEAK_PATTERNS:
        if any(pattern.search(text) for text in texts):
            errors.append(f"pdf bytes contain sensitive pattern: {label}")
            seen.add(label)
    for label, pattern in SENSITIVE_PATTERNS:
        if label not in seen and any(pattern.search(text) for text in texts):
            errors.append(f"pdf bytes contain sensitive pattern: {label}")
            seen.add(label)
    return errors


def _dedupe_errors(errors: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for error in errors:
        if error in seen:
            continue
        deduped.append(error)
        seen.add(error)
    return deduped


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


def _process_contract_preflight(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    code, payload = check_process_contract(argparse.Namespace(
        base_url=_clean(args.base_url),
        token=_clean(args.token),
        timeout=float(args.timeout),
        path=_clean(args.process_contract_path),
        output_result=None,
        json=False,
        pretty=False,
    ))
    if not isinstance(payload, Mapping):
        return ["process contract preflight failed"], {"status": None}
    endpoint = payload.get("endpoint")
    errors = payload.get("errors")
    error_list = [
        str(error)
        for error in (errors if isinstance(errors, Sequence) and not isinstance(errors, str) else ())
    ]
    summary = {
        "status": endpoint.get("status") if isinstance(endpoint, Mapping) else None,
        "ok": code == 0 and payload.get("ok") is True,
    }
    if error_list:
        summary["errors"] = error_list
    return (
        [f"process contract preflight failed: {error}" for error in error_list]
        or ([] if summary["ok"] else ["process contract preflight failed"]),
        summary,
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
    pdf_text, text_errors, pdf_text_meta = _pdf_text_input(args, pdf_bytes)
    read_errors = [*byte_errors, *text_errors]
    if pdf_text.strip() == "":
        read_errors.append("pdf text extraction must be non-empty")
    read_errors.extend(_pdf_byte_leak_errors(pdf_bytes))
    read_errors = _dedupe_errors(read_errors)
    if read_errors:
        return _safe_failure_payload(args, read_errors)

    process_contract_errors, process_contract_summary = _process_contract_preflight(args)
    if process_contract_errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "inputs": _input_summary(args),
            "fetches": {"process_contract": process_contract_summary},
            "errors": process_contract_errors,
        }

    report_model_result, artifact_result = _fetch_live_inputs(args)
    payload_errors, report_model, _artifact, evidence_export = _live_payload_errors(
        report_model=report_model_result,
        artifact=artifact_result,
    )
    fetches = {
        "process_contract": process_contract_summary,
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
        "pdf_text": pdf_text_meta,
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
