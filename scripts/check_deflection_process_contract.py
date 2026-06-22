#!/usr/bin/env python3
"""Check that a hosted deflection API process serves the current contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from extracted_content_pipeline.faq_deflection_report import (
    deflection_report_model_contract_shape,
)


SCHEMA_VERSION = "deflection_process_contract_check.v1"
EXPECTED_PROCESS_SCHEMA_VERSION = "deflection_report_process.v1"
EXPECTED_REPORT_MODEL_SCHEMA_VERSION = "deflection.v1"
EXPECTED_EVIDENCE_EXPORT_SCHEMA_VERSION = "deflection_evidence.v1"
EXPECTED_REPORT_MODEL_CONTRACT = deflection_report_model_contract_shape()
DEFAULT_CONTRACT_PATH = (
    "/api/v1/content-ops/deflection-reports/process-contract"
)
LOCAL_HOSTS = frozenset({"localhost", "0.0.0.0", "::1"})
REQUIRED_ROUTE_SUFFIXES = {
    "process_contract": "/process-contract",
    "snapshot": "/{request_id}/snapshot",
    "artifact": "/{request_id}/artifact",
    "report_model": "/{request_id}/report-model",
    "delete": "/{request_id}",
}


@dataclass(frozen=True)
class HttpResult:
    status: int | None
    payload: Any = None
    errors: tuple[str, ...] = ()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--path", default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-result", type=Path)
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


def _join_url(base_url: str, path: str) -> str:
    base = _clean(base_url).rstrip("/")
    suffix = "/" + _clean(path).lstrip("/")
    return f"{base}{suffix}"


def _fetch_json(url: str, *, token: str, timeout: float) -> HttpResult:
    headers = {"Accept": "application/json"}
    if _clean(token):
        headers["Authorization"] = f"Bearer {_clean(token)}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.getcode())
            body = response.read()
    except urllib.error.HTTPError as exc:
        return HttpResult(
            status=int(exc.code),
            errors=(f"process contract endpoint must return 200, got {exc.code}",),
        )
    except urllib.error.URLError as exc:
        return HttpResult(status=None, errors=(f"process contract fetch failed: {exc.reason}",))
    except OSError as exc:
        return HttpResult(status=None, errors=(f"process contract fetch failed: {exc}",))

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return HttpResult(
            status=status,
            errors=("process contract endpoint response must be a JSON object",),
        )
    if status != 200:
        return HttpResult(
            status=status,
            payload=payload,
            errors=(f"process contract endpoint must return 200, got {status}",),
        )
    if not isinstance(payload, Mapping):
        return HttpResult(
            status=status,
            payload=payload,
            errors=("process contract endpoint response must be a JSON object",),
        )
    return HttpResult(status=status, payload=dict(payload))


def _expected_routes(process_contract_path: str) -> dict[str, str]:
    base = _clean(process_contract_path)
    if base.endswith("/process-contract"):
        base = base[: -len("/process-contract")]
    else:
        base = base.rstrip("/")
    return {
        key: f"{base}{suffix}"
        for key, suffix in REQUIRED_ROUTE_SUFFIXES.items()
    }


def _validate_contract(payload: Any, *, process_contract_path: str) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["process contract endpoint response must be a JSON object"]
    errors: list[str] = []
    if _clean(payload.get("schema_version")) != EXPECTED_PROCESS_SCHEMA_VERSION:
        errors.append(
            "schema_version must be "
            f"{EXPECTED_PROCESS_SCHEMA_VERSION}"
        )
    if _clean(payload.get("service")) != "content_ops_deflection_reports":
        errors.append("service must be content_ops_deflection_reports")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        return [*errors, "contract must be an object"]
    routes = payload.get("routes")
    if not isinstance(routes, Mapping):
        errors.append("routes must be an object")
    else:
        for key, expected in _expected_routes(process_contract_path).items():
            if _clean(routes.get(key)) != expected:
                errors.append(f"routes.{key} must be {expected}")
    if (
        _clean(contract.get("report_model_schema_version"))
        != EXPECTED_REPORT_MODEL_SCHEMA_VERSION
    ):
        errors.append(
            "contract.report_model_schema_version must be "
            f"{EXPECTED_REPORT_MODEL_SCHEMA_VERSION}"
        )
    if contract.get("report_model_contract") != EXPECTED_REPORT_MODEL_CONTRACT:
        errors.append(
            "contract.report_model_contract must match current deflection.v1 shape"
        )
    if (
        _clean(contract.get("evidence_export_schema_version"))
        != EXPECTED_EVIDENCE_EXPORT_SCHEMA_VERSION
    ):
        errors.append(
            "contract.evidence_export_schema_version must be "
            f"{EXPECTED_EVIDENCE_EXPORT_SCHEMA_VERSION}"
        )
    required = contract.get("paid_artifact_requires")
    if not isinstance(required, Mapping):
        errors.append("contract.paid_artifact_requires must be an object")
        return errors
    for key in ("report_model", "evidence_export"):
        if _clean(required.get(key)) != "object":
            errors.append(f"contract.paid_artifact_requires.{key} must be object")
    return errors


def check_process_contract(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    url = _join_url(args.base_url, args.path)
    errors = _hosted_url_errors(_clean(args.base_url), label="base-url")
    payload: Any = None
    status: int | None = None
    if not errors:
        result = _fetch_json(url, token=_clean(args.token), timeout=float(args.timeout))
        status = result.status
        payload = result.payload
        errors.extend(result.errors)
        if not result.errors:
            errors.extend(_validate_contract(payload, process_contract_path=args.path))
    output = {
        "schema_version": SCHEMA_VERSION,
        "ok": not errors,
        "endpoint": {
            "path": _clean(args.path),
            "status": status,
        },
        "observed": _safe_observed_contract(payload),
        "errors": errors,
    }
    return (0 if output["ok"] else 1), output


def _safe_observed_contract(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    contract = payload.get("contract")
    required = contract.get("paid_artifact_requires") if isinstance(contract, Mapping) else {}
    routes = payload.get("routes")
    return {
        "schema_version": _clean(payload.get("schema_version")),
        "service": _clean(payload.get("service")),
        "report_model_schema_version": (
            _clean(contract.get("report_model_schema_version"))
            if isinstance(contract, Mapping)
            else ""
        ),
        "report_model_contract": (
            dict(contract.get("report_model_contract"))
            if (
                isinstance(contract, Mapping)
                and isinstance(contract.get("report_model_contract"), Mapping)
            )
            else {}
        ),
        "evidence_export_schema_version": (
            _clean(contract.get("evidence_export_schema_version"))
            if isinstance(contract, Mapping)
            else ""
        ),
        "paid_artifact_requires": dict(required) if isinstance(required, Mapping) else {},
        "routes": dict(routes) if isinstance(routes, Mapping) else {},
    }


def _emit(payload: Mapping[str, Any], *, pretty: bool) -> str:
    return json.dumps(payload, indent=2 if pretty else None, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    code, payload = check_process_contract(args)
    if args.output_result:
        args.output_result.parent.mkdir(parents=True, exist_ok=True)
        args.output_result.write_text(_emit(payload, pretty=True) + "\n", encoding="utf-8")
    if args.json or not args.output_result:
        print(_emit(payload, pretty=args.pretty))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
