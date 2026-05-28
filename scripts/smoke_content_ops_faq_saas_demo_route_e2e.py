#!/usr/bin/env python3
"""Seed the SaaS FAQ demo, hit the hosted route, then clean up."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEED_SCRIPT = ROOT / "scripts/seed_content_ops_faq_saas_demo.py"
ROUTE_SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_route_concurrency.py"

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


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
        description="Run the SaaS demo FAQ seed plus hosted route/detail smoke."
    )
    parser.add_argument("--database-url", default=_env("EXTRACTED_DATABASE_URL", "DATABASE_URL"))
    parser.add_argument("--base-url", default=_env("ATLAS_API_BASE_URL"))
    parser.add_argument("--token", default=_env("ATLAS_B2B_JWT", "ATLAS_TOKEN"))
    parser.add_argument("--account-id", default=_env("ATLAS_FAQ_SEARCH_ACCOUNT_ID", "ATLAS_ACCOUNT_ID"))
    parser.add_argument("--route", default="/api/v1/content-ops/faq-deflection-search")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--route-requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-case-error-rate", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-single-request-ms", type=float)
    parser.add_argument("--max-case-p95-ms", type=float)
    parser.add_argument("--max-case-single-request-ms", type=float)
    parser.add_argument("--max-detail-ms", type=float, default=2500.0)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    for name, message in (
        ("database_url", "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL"),
        ("base_url", "ATLAS_API_BASE_URL or --base-url is required"),
        ("token", "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required"),
        ("account_id", "ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required"),
    ):
        if not str(getattr(args, name) or "").strip():
            errors.append(message)
    for name in ("route_requests", "concurrency"):
        if int(getattr(args, name)) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "timeout",
        "max_error_rate",
        "max_case_error_rate",
        "max_p95_ms",
        "max_single_request_ms",
        "max_case_p95_ms",
        "max_case_single_request_ms",
        "max_detail_ms",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(float(value)):
            errors.append(f"--{name.replace('_', '-')} must be finite")
    if float(args.timeout) <= 0:
        errors.append("--timeout must be positive")
    for name in ("max_error_rate", "max_case_error_rate"):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            errors.append(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in (
        "max_p95_ms",
        "max_single_request_ms",
        "max_case_p95_ms",
        "max_case_single_request_ms",
        "max_detail_ms",
    ):
        value = getattr(args, name)
        if value is not None and float(value) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
    return errors


def _seed_command(args: argparse.Namespace, *, case_file: Path, seed_result: Path) -> list[str]:
    return [
        sys.executable,
        str(SEED_SCRIPT),
        "--database-url",
        str(args.database_url),
        "--account-id",
        str(args.account_id),
        "--route-case-file-output",
        str(case_file),
        "--output-result",
        str(seed_result),
        "--json",
    ]


def _route_command(args: argparse.Namespace, *, case_file: Path, route_result: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROUTE_SCRIPT),
        "--base-url",
        str(args.base_url),
        "--token",
        str(args.token),
        "--route",
        str(args.route),
        "--timeout",
        str(args.timeout),
        "--requests",
        str(args.route_requests),
        "--concurrency",
        str(args.concurrency),
        "--case-file",
        str(case_file),
        "--require-detail",
        "--max-error-rate",
        str(args.max_error_rate),
        "--max-case-error-rate",
        str(args.max_case_error_rate),
        "--max-detail-ms",
        str(args.max_detail_ms),
        "--output-result",
        str(route_result),
        "--json",
    ]
    for attr, flag in (
        ("max_p95_ms", "--max-p95-ms"),
        ("max_single_request_ms", "--max-single-request-ms"),
        ("max_case_p95_ms", "--max-case-p95-ms"),
        ("max_case_single_request_ms", "--max-case-single-request-ms"),
    ):
        value = getattr(args, attr)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _cleanup_command(args: argparse.Namespace, *, faq_id: str, cleanup_result: Path) -> list[str]:
    return [
        sys.executable,
        str(SEED_SCRIPT),
        "--database-url",
        str(args.database_url),
        "--account-id",
        str(args.account_id),
        "--cleanup-faq-id",
        faq_id,
        "--output-result",
        str(cleanup_result),
        "--json",
    ]


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, [f"{label} result could not be read: {exc}"]
    except json.JSONDecodeError as exc:
        return None, [f"{label} result must contain JSON: {exc.msg}"]
    if not isinstance(payload, Mapping):
        return None, [f"{label} result must contain a JSON object"]
    return dict(payload), []


def _compact_artifact(path: Path, *, label: str) -> dict[str, Any]:
    payload, errors = _read_json_object(path, label=label)
    if payload is None:
        return {"available": False, "ok": False, "path": str(path), "errors": errors}
    summary = {
        "available": True,
        "ok": payload.get("ok") is True,
        "path": str(path),
    }
    for key in (
        "phase",
        "account_id",
        "faq_id",
        "corpus_id",
        "target_id",
        "status",
        "source_count",
        "generated_items",
        "deleted_faq_ids",
        "delete_status",
    ):
        if key in payload:
            summary[key] = payload[key]
    if isinstance(payload.get("search"), Mapping):
        search = payload["search"]
        summary["search"] = {
            "query": search.get("query"),
            "count": search.get("count"),
            "matched_seeded_faq": search.get("matched_seeded_faq"),
        }
    if isinstance(payload.get("route_case_file"), Mapping):
        route_case_file = payload["route_case_file"]
        summary["route_case_file"] = {
            "ok": route_case_file.get("ok"),
            "path": route_case_file.get("path"),
            "cases": route_case_file.get("cases"),
            "error": route_case_file.get("error"),
        }
    if isinstance(payload.get("requests"), Mapping):
        summary["requests"] = payload["requests"]
    if isinstance(payload.get("cases"), Mapping):
        summary["cases"] = payload["cases"]
    if isinstance(payload.get("detail"), Mapping):
        summary["detail"] = payload["detail"]
    if isinstance(payload.get("budgets"), Mapping):
        summary["budgets"] = payload["budgets"]
    if isinstance(payload.get("errors"), list):
        summary["errors"] = payload["errors"][:10]
    return summary


def _faq_id_from_seed_result(seed_result: Path) -> tuple[str, list[str]]:
    payload, errors = _read_json_object(seed_result, label="seed")
    if payload is None:
        return "", errors
    faq_id = payload.get("faq_id")
    if not isinstance(faq_id, str) or not faq_id.strip():
        return "", ["seed result faq_id must be a non-empty string"]
    return faq_id.strip(), []


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _not_run(reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "skipped": True,
        "not_run_reason": reason,
    }


def _preflight_summary(errors: Sequence[str], elapsed: float) -> dict[str, Any]:
    return {
        "ok": False,
        "phase": "preflight",
        "preflight_errors": list(errors),
        "seed": _not_run("preflight_failed"),
        "route": _not_run("preflight_failed"),
        "cleanup": {"ok": True, "skipped": True, "not_run_reason": "preflight_failed"},
        "elapsed_seconds": elapsed,
    }


def _print_summary(summary: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(
        "SaaS FAQ route e2e: "
        f"ok={summary['ok']} seed={summary['seed']['ok']} "
        f"route={summary['route']['ok']} cleanup={summary['cleanup']['ok']}"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="faq-saas-demo-route-e2e-") as temp_root:
        artifact_dir = args.artifact_dir or Path(temp_root)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        seed_result = artifact_dir / "seed-result.json"
        route_result = artifact_dir / "route-result.json"
        cleanup_result = artifact_dir / "cleanup-result.json"
        case_file = artifact_dir / "route-cases.json"

        seed = _run_command(_seed_command(args, case_file=case_file, seed_result=seed_result))
        route = _not_run("seed_failed")
        cleanup: dict[str, Any] = {"ok": True, "skipped": True, "not_run_reason": "seed_failed"}
        errors: list[str] = []

        if seed["ok"]:
            route = _run_command(_route_command(args, case_file=case_file, route_result=route_result))
            faq_id, faq_errors = _faq_id_from_seed_result(seed_result)
            errors.extend(faq_errors)
            if args.keep_data:
                cleanup = {"ok": True, "skipped": True, "not_run_reason": "keep_data"}
            elif faq_id:
                cleanup = _run_command(
                    _cleanup_command(args, faq_id=faq_id, cleanup_result=cleanup_result)
                )
            else:
                cleanup = {
                    "ok": False,
                    "skipped": True,
                    "not_run_reason": "missing_seed_faq_id",
                }

        seed_artifact = _compact_artifact(seed_result, label="seed")
        route_artifact = _compact_artifact(route_result, label="route") if not route.get("skipped") else None
        cleanup_artifact = (
            _compact_artifact(cleanup_result, label="cleanup")
            if not cleanup.get("skipped")
            else None
        )
        summary = {
            "ok": bool(seed["ok"]) and bool(route["ok"]) and bool(cleanup["ok"]) and not errors,
            "phase": "complete",
            "artifacts": {
                "directory": str(artifact_dir),
                "seed_result": str(seed_result),
                "route_case_file": str(case_file),
                "route_result": str(route_result),
                "cleanup_result": str(cleanup_result),
            },
            "seed": {**seed, "result_artifact": seed_artifact},
            "route": (
                {**route, "result_artifact": route_artifact}
                if route_artifact is not None
                else route
            ),
            "cleanup": (
                {**cleanup, "result_artifact": cleanup_artifact}
                if cleanup_artifact is not None
                else cleanup
            ),
            "errors": errors,
            "elapsed_seconds": time.perf_counter() - start,
        }
        return summary


def main(argv: Sequence[str] | None = None) -> int:
    start = time.perf_counter()
    args = _build_parser().parse_args(argv)
    errors = _validate_args(args)
    if errors:
        summary = _preflight_summary(errors, time.perf_counter() - start)
        _write_result(args.output_result, summary)
        _print_summary(summary, as_json=args.json)
        return 2

    summary = run(args)
    _write_result(args.output_result, summary)
    _print_summary(summary, as_json=args.json)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
