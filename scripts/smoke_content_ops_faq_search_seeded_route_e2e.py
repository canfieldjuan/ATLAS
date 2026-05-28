#!/usr/bin/env python3
"""Seed FAQ search rows, hit the hosted route, then clean up."""

from __future__ import annotations

import argparse
import asyncio
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
SEED_SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_concurrency.py"
ROUTE_SCRIPT = ROOT / "scripts/smoke_content_ops_faq_search_route_concurrency.py"
CONTRACT_SCRIPT = ROOT / "scripts/check_content_ops_faq_search_route_contract.py"

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - host dependency
    load_dotenv = None


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _default_database_url() -> str:
    raw = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if raw:
        return raw
    try:
        from atlas_brain.storage.config import db_settings
    except Exception:
        return ""
    return str(getattr(db_settings, "dsn", "") or "").strip()


def _build_parser() -> argparse.ArgumentParser:
    _load_dotenv_files()
    parser = argparse.ArgumentParser(
        description="Run seeded DB plus hosted FAQ search route smoke."
    )
    parser.add_argument("--database-url", default=_default_database_url())
    parser.add_argument("--base-url", default=os.getenv("ATLAS_API_BASE_URL", ""))
    parser.add_argument(
        "--token",
        default=os.getenv("ATLAS_B2B_JWT") or os.getenv("ATLAS_TOKEN", ""),
    )
    parser.add_argument("--account-id", default=os.getenv("ATLAS_FAQ_SEARCH_ACCOUNT_ID", ""))
    parser.add_argument("--corpora-per-account", type=int, default=2)
    parser.add_argument("--documents-per-corpus", type=int, default=3)
    parser.add_argument("--seed-iterations", type=int, default=12)
    parser.add_argument("--route-requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--route", default="/api/v1/content-ops/faq-deflection-search")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-single-request-ms", type=float)
    parser.add_argument("--max-case-error-rate", type=float)
    parser.add_argument("--max-case-p95-ms", type=float)
    parser.add_argument("--max-case-single-request-ms", type=float)
    parser.add_argument("--max-detail-ms", type=float)
    parser.add_argument("--detail-route", default=os.getenv("ATLAS_FAQ_DETAIL_ROUTE", ""))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--skip-detail-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    for name, message in (
        ("database_url", "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL"),
        ("base_url", "ATLAS_API_BASE_URL or --base-url is required"),
        ("token", "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required"),
        ("account_id", "ATLAS_FAQ_SEARCH_ACCOUNT_ID or --account-id is required"),
    ):
        if not str(getattr(args, name) or "").strip():
            errors.append(message)
    for name in (
        "corpora_per_account",
        "documents_per_corpus",
        "seed_iterations",
        "route_requests",
        "concurrency",
        "pool_size",
    ):
        if int(getattr(args, name)) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "timeout",
        "max_error_rate",
        "max_p95_ms",
        "max_single_request_ms",
        "max_case_error_rate",
        "max_case_p95_ms",
        "max_case_single_request_ms",
        "max_detail_ms",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(float(value)):
            errors.append(f"--{name.replace('_', '-')} must be finite")
    if float(args.timeout) <= 0:
        errors.append("--timeout must be positive")
    if not 0 <= float(args.max_error_rate) <= 1:
        errors.append("--max-error-rate must be between 0 and 1")
    if args.max_case_error_rate is not None and not 0 <= float(args.max_case_error_rate) <= 1:
        errors.append("--max-case-error-rate must be between 0 and 1")
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
    if args.max_detail_ms is not None and bool(args.skip_detail_check):
        errors.append("--max-detail-ms requires detail checks; remove --skip-detail-check")
    return errors


def _seed_command(
    args: argparse.Namespace,
    *,
    case_file: Path,
    cleanup_manifest: Path,
    seed_result: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SEED_SCRIPT),
        "--database-url",
        str(args.database_url),
        "--account-count",
        "1",
        "--account-id",
        str(args.account_id),
        "--corpora-per-account",
        str(args.corpora_per_account),
        "--documents-per-corpus",
        str(args.documents_per_corpus),
        "--iterations",
        str(args.seed_iterations),
        "--concurrency",
        str(args.concurrency),
        "--pool-size",
        str(args.pool_size),
        "--keep-data",
        "--route-case-file-output",
        str(case_file),
        "--cleanup-manifest-output",
        str(cleanup_manifest),
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
        "--max-error-rate",
        str(args.max_error_rate),
        "--case-file",
        str(case_file),
        "--output-result",
        str(route_result),
        "--json",
    ]
    for arg_name, flag in (
        ("max_p95_ms", "--max-p95-ms"),
        ("max_single_request_ms", "--max-single-request-ms"),
        ("max_case_error_rate", "--max-case-error-rate"),
        ("max_case_p95_ms", "--max-case-p95-ms"),
        ("max_case_single_request_ms", "--max-case-single-request-ms"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _detail_case_from_route_cases(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, [f"route case file could not be read: {exc}"]
    except json.JSONDecodeError as exc:
        return None, [f"route case file must contain JSON: {exc.msg}"]
    if not isinstance(data, list) or not data:
        return None, ["route case file must contain a non-empty JSON list"]

    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            return None, [f"route case[{index}] must be an object"]
        require_results = item.get("require_results")
        if "require_results" in item and type(require_results) is not bool:
            return None, [f"route case[{index}].require_results must be a boolean"]
        if require_results is not True:
            continue
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            return None, [f"route case[{index}].query must be a non-empty string"]
        account_id = item.get("expected_first_account_id")
        if not isinstance(account_id, str) or not account_id.strip():
            return None, [f"route case[{index}].expected_first_account_id must be a non-empty string"]
        corpus_id = item.get("corpus_id", "")
        if not isinstance(corpus_id, str):
            return None, [f"route case[{index}].corpus_id must be a string"]
        status = item.get("status", "")
        if not isinstance(status, str):
            return None, [f"route case[{index}].status must be a string"]
        limit = item.get("limit", 5)
        if type(limit) is not int or limit <= 0:
            return None, [f"route case[{index}].limit must be a positive integer"]
        expected_detail: dict[str, str] = {}
        for key in (
            "expected_detail_account_id",
            "expected_detail_target_id",
            "expected_detail_target_mode",
            "expected_detail_title",
            "expected_detail_status",
        ):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                return None, [f"route case[{index}].{key} must be a non-empty string"]
            expected_detail[key] = value.strip()
        return {
            "query": query.strip(),
            "corpus_id": corpus_id.strip(),
            "status": status.strip(),
            "limit": limit,
            "expected_detail_account_id": account_id.strip(),
            **expected_detail,
        }, []
    return None, ["route case file must include a require_results case for detail check"]


def _detail_command(
    args: argparse.Namespace,
    *,
    detail_case: Mapping[str, Any],
    detail_result: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(CONTRACT_SCRIPT),
        "--base-url",
        str(args.base_url),
        "--token",
        str(args.token),
        "--route",
        str(args.route),
        "--query",
        str(detail_case["query"]),
        "--limit",
        str(detail_case["limit"]),
        "--require-results",
        "--require-detail",
        "--output-result",
        str(detail_result),
    ]
    if str(detail_case.get("corpus_id") or "").strip():
        command.extend(["--corpus-id", str(detail_case["corpus_id"])])
    if str(detail_case.get("status") or "").strip():
        command.extend(["--status", str(detail_case["status"])])
    if str(args.detail_route or "").strip():
        command.extend(["--detail-route", str(args.detail_route)])
    if args.max_detail_ms is not None:
        command.extend(["--max-detail-ms", str(args.max_detail_ms)])
    for key, flag in (
        ("expected_detail_account_id", "--expected-detail-account-id"),
        ("expected_detail_target_id", "--expected-detail-target-id"),
        ("expected_detail_target_mode", "--expected-detail-target-mode"),
        ("expected_detail_title", "--expected-detail-title"),
        ("expected_detail_status", "--expected-detail-status"),
    ):
        value = str(detail_case.get(key) or "").strip()
        if value:
            command.extend([flag, value])
    return command


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _compact_error_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"count": 0, "items": [], "truncated": False}
    raw_items = value.get("items")
    items = raw_items if isinstance(raw_items, list) else []
    count = value.get("count")
    return {
        "count": count if type(count) is int else 0,
        **({"rate": value.get("rate")} if "rate" in value else {}),
        "items": items[:5],
        "truncated": bool(value.get("truncated")) or len(items) > 5,
    }


def _child_result_artifact_error(path: Path, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "available": False,
        "path": str(path),
        "errors": [message],
    }


def _compact_child_result_artifact(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return _child_result_artifact_error(path, f"result artifact could not be read: {exc}")
    except json.JSONDecodeError as exc:
        return _child_result_artifact_error(path, f"result artifact must contain JSON: {exc.msg}")
    if not isinstance(payload, Mapping):
        return _child_result_artifact_error(path, "result artifact must contain a JSON object")

    raw_ok = payload.get("ok")
    artifact_errors = []
    if type(raw_ok) is not bool:
        artifact_errors.append("result artifact ok must be a boolean")
    summary: dict[str, Any] = {
        "ok": raw_ok is True and not artifact_errors,
        "available": True,
        "path": str(path),
    }
    if artifact_errors:
        summary["artifact_errors"] = artifact_errors
    if kind == "seed":
        for key in (
            "run_id",
            "requests",
            "seed",
            "setup",
            "cleanup",
            "pool_close",
            "latency",
            "latency_budget",
            "elapsed_seconds",
        ):
            if key in payload:
                summary[key] = payload[key]
        if isinstance(payload.get("isolation"), Mapping):
            summary["isolation"] = _compact_error_summary(payload["isolation"])
        return summary
    if kind == "route":
        for key in (
            "phase",
            "requests",
            "latency",
            "budgets",
            "preflight_errors",
            "elapsed_seconds",
        ):
            if key in payload:
                summary[key] = payload[key]
        if isinstance(payload.get("cases"), Mapping):
            cases = payload["cases"]
            total = cases.get("total")
            summary["cases"] = {
                "total": total if type(total) is int else 0,
                "case_file": str(cases.get("case_file") or ""),
                "truncated": bool(cases.get("truncated")),
            }
        if isinstance(payload.get("errors"), Mapping):
            summary["errors"] = _compact_error_summary(payload["errors"])
        return summary
    if kind == "detail":
        for key in (
            "phase",
            "count",
            "detail_checked",
            "detail_faq_id",
            "search_elapsed_ms",
            "detail_elapsed_ms",
            "total_elapsed_ms",
            "max_detail_ms",
            "errors",
        ):
            if key in payload:
                summary[key] = payload[key]
        if isinstance(summary.get("errors"), list):
            summary["errors"] = summary["errors"][:10]
        return summary
    summary["errors"] = [f"unknown child result kind: {kind}"]
    summary["ok"] = False
    return summary


def _with_result_artifact(
    phase: dict[str, Any],
    path: Path,
    *,
    kind: str,
) -> dict[str, Any]:
    result_artifact = _compact_child_result_artifact(path, kind=kind)
    updated = dict(phase)
    updated["result_artifact"] = result_artifact
    if bool(phase.get("ok")) and not bool(result_artifact["ok"]):
        updated["ok"] = False
    return updated


def _faq_ids_from_cleanup_manifest(path: Path) -> tuple[list[str], list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [], [f"cleanup manifest could not be read: {exc}"]
    except json.JSONDecodeError as exc:
        return [], [f"cleanup manifest must contain JSON: {exc.msg}"]
    if not isinstance(data, Mapping):
        return [], ["cleanup manifest must contain a JSON object"]

    faq_ids: list[str] = []
    errors: list[str] = []
    raw_faq_ids = data.get("faq_ids")
    if not isinstance(raw_faq_ids, list):
        return [], ["cleanup manifest faq_ids must be a list"]
    for index, faq_id in enumerate(raw_faq_ids):
        if not isinstance(faq_id, str) or not faq_id.strip():
            errors.append(f"cleanup manifest faq_ids[{index}] must be a non-empty string")
            continue
        if faq_id.strip() not in faq_ids:
            faq_ids.append(faq_id.strip())
    return faq_ids, errors


def _deleted_row_count(delete_status: object) -> int | None:
    if not isinstance(delete_status, str):
        return None
    parts = delete_status.strip().split()
    if len(parts) != 2 or parts[0] != "DELETE":
        return None
    try:
        row_count = int(parts[1])
    except ValueError:
        return None
    return row_count if row_count >= 0 else None


async def _cleanup_seeded_faqs(database_url: str, faq_ids: Sequence[str]) -> dict[str, Any]:
    requested_faq_ids = len(faq_ids)
    if not faq_ids:
        return _cleanup_result(
            requested_faq_ids=0,
            deleted_faq_ids=0,
            delete_status=None,
        )
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        return _cleanup_result(
            requested_faq_ids=requested_faq_ids,
            deleted_faq_ids=0,
            delete_status=None,
            errors=[f"ImportError: {exc}"],
        )

    delete_status = None
    try:
        pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)
        try:
            delete_status = await pool.execute(
                "DELETE FROM ticket_faq_markdown WHERE id = ANY($1::uuid[])",
                list(faq_ids),
            )
        finally:
            await pool.close()
    except Exception as exc:
        return _cleanup_result(
            requested_faq_ids=requested_faq_ids,
            deleted_faq_ids=0,
            delete_status=delete_status,
            errors=[f"{type(exc).__name__}: {exc}"],
        )
    deleted_faq_ids = _deleted_row_count(delete_status)
    if deleted_faq_ids is None:
        return _cleanup_result(
            requested_faq_ids=requested_faq_ids,
            deleted_faq_ids=None,
            delete_status=delete_status,
            errors=[f"cleanup delete status is not parseable: {delete_status!r}"],
        )
    if deleted_faq_ids != requested_faq_ids:
        return _cleanup_result(
            requested_faq_ids=requested_faq_ids,
            deleted_faq_ids=deleted_faq_ids,
            delete_status=delete_status,
            errors=[
                "cleanup deleted "
                f"{deleted_faq_ids} FAQ rows but requested {requested_faq_ids}"
            ],
        )
    return _cleanup_result(
        requested_faq_ids=requested_faq_ids,
        deleted_faq_ids=deleted_faq_ids,
        delete_status=delete_status,
    )


def _cleanup_result(
    *,
    requested_faq_ids: int,
    deleted_faq_ids: int | None,
    delete_status: object,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "ok": not errors,
        "requested_faq_ids": requested_faq_ids,
        "deleted_faq_ids": deleted_faq_ids,
        "delete_status": delete_status,
        "errors": list(errors),
    }


def _lifecycle_result(*, attempted: bool, error: BaseException | None = None) -> dict[str, Any]:
    return {
        "ok": error is None,
        "attempted": attempted,
        "error": None
        if error is None
        else {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _with_lifecycle_result(
    summary: dict[str, Any],
    name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(summary)
    updated[name] = result
    updated["ok"] = bool(summary["ok"]) and bool(result["ok"])
    return updated


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    artifact_cleanup = summary["artifact_cleanup"]
    artifact_cleanup_error = artifact_cleanup.get("error") or {}
    artifact_cleanup_suffix = ""
    if not artifact_cleanup["ok"]:
        artifact_cleanup_suffix = (
            " artifact_cleanup_error="
            f"{artifact_cleanup_error.get('type', 'Error')}: "
            f"{artifact_cleanup_error.get('message', '')}"
        )
    print(
        "FAQ search seeded route e2e: "
        f"ok={summary['ok']} seed={summary['seed']['ok']} "
        f"route={summary['route']['ok']} detail={summary['detail']['ok']} "
        f"cleanup={summary['cleanup']['ok']} "
        f"artifact_cleanup={artifact_cleanup['ok']}"
        f"{artifact_cleanup_suffix}"
    )


def _preflight_summary(args: argparse.Namespace, errors: Sequence[str], elapsed: float) -> dict[str, Any]:
    return {
        "ok": False,
        "phase": "preflight",
        "artifacts": {},
        "seed": _seed_not_run("preflight_failed", ok=False),
        "route": _route_not_run("preflight_failed", ok=False),
        "detail": _detail_not_run(
            "skip_detail_check" if args.skip_detail_check else "preflight_failed",
            ok=bool(args.skip_detail_check),
        ),
        "cleanup": {
            "ok": False,
            "requested_faq_ids": 0,
            "deleted_faq_ids": 0,
            "delete_status": None,
            "errors": [],
        },
        "artifact_cleanup": _lifecycle_result(attempted=False),
        "preflight_errors": list(errors),
        "keep_data": bool(args.keep_data),
        "elapsed_seconds": round(elapsed, 6),
    }


async def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    errors = _validate_args(args)
    if errors:
        return 2, _preflight_summary(args, errors, time.perf_counter() - started)

    temp_context = (
        tempfile.TemporaryDirectory(prefix="atlas-faq-search-e2e-")
        if args.artifact_dir is None
        else None
    )
    artifact_dir = Path(temp_context.name) if temp_context else Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    case_file = artifact_dir / "route-cases.json"
    cleanup_manifest = artifact_dir / "cleanup-manifest.json"
    seed_result = artifact_dir / "seed-result.json"
    route_result = artifact_dir / "route-result.json"
    detail_result = artifact_dir / "detail-result.json"

    detail = _detail_not_run(
        "skip_detail_check" if args.skip_detail_check else "waiting_for_seed_and_route",
        ok=bool(args.skip_detail_check),
    )
    cleanup = {
        "ok": True,
        "requested_faq_ids": 0,
        "deleted_faq_ids": 0,
        "delete_status": None,
        "errors": [],
    }
    artifact_cleanup = _lifecycle_result(attempted=False)
    summary: dict[str, Any] | None = None
    try:
        seed = _run_command(
            _seed_command(
                args,
                case_file=case_file,
                cleanup_manifest=cleanup_manifest,
                seed_result=seed_result,
            )
        )
        seed = _with_result_artifact(seed, seed_result, kind="seed")
        route = _route_not_run("waiting_for_seed", ok=False)
        if seed["ok"]:
            route = _run_command(_route_command(args, case_file=case_file, route_result=route_result))
            route = _with_result_artifact(route, route_result, kind="route")
        else:
            route = _route_not_run("seed_failed", ok=False)
        if not bool(seed["ok"]) and not bool(args.skip_detail_check):
            detail = _detail_not_run("seed_failed", ok=False)
        if bool(seed["ok"]) and not bool(route["ok"]) and not bool(args.skip_detail_check):
            detail = _detail_not_run("route_failed", ok=False)
        if bool(seed["ok"] and route["ok"]) and not bool(args.skip_detail_check):
            detail_case, detail_errors = _detail_case_from_route_cases(case_file)
            if detail_errors:
                detail = {
                    "ok": False,
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "; ".join(detail_errors),
                    "skipped": False,
                }
            elif detail_case is not None:
                detail = _run_command(
                    _detail_command(args, detail_case=detail_case, detail_result=detail_result)
                )
                detail = _with_result_artifact(detail, detail_result, kind="detail")
                detail["skipped"] = False
        faq_ids, manifest_errors = (
            _faq_ids_from_cleanup_manifest(cleanup_manifest)
            if cleanup_manifest.exists()
            else ([], [])
        )
        if manifest_errors:
            cleanup = _cleanup_result(
                requested_faq_ids=len(faq_ids),
                deleted_faq_ids=0,
                delete_status=None,
                errors=manifest_errors,
            )
        elif not bool(args.keep_data):
            cleanup = await _cleanup_seeded_faqs(str(args.database_url), faq_ids)
        ok = bool(seed["ok"] and route["ok"] and detail["ok"] and cleanup["ok"])
        summary = {
            "ok": ok,
            "phase": "complete",
            "artifacts": {
                "dir": str(artifact_dir),
                "case_file": str(case_file),
                "cleanup_manifest": str(cleanup_manifest),
                "seed_result": str(seed_result),
                "route_result": str(route_result),
                "detail_result": str(detail_result),
            },
            "seed": seed,
            "route": route,
            "detail": detail,
            "cleanup": cleanup,
            "preflight_errors": [],
            "keep_data": bool(args.keep_data),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    finally:
        if temp_context is not None:
            try:
                temp_context.cleanup()
                artifact_cleanup = _lifecycle_result(attempted=True)
            except Exception as exc:
                artifact_cleanup = _lifecycle_result(attempted=True, error=exc)
    if summary is None:  # pragma: no cover - defensive guard for unexpected control flow.
        raise RuntimeError("FAQ search seeded route e2e did not produce a summary")
    summary = _with_lifecycle_result(summary, "artifact_cleanup", artifact_cleanup)
    return (0 if summary["ok"] else 1), summary


def _detail_not_run(reason: str, *, ok: bool) -> dict[str, Any]:
    return {
        "ok": ok,
        "returncode": None,
        "skipped": True,
        "not_run_reason": reason,
    }


def _route_not_run(reason: str, *, ok: bool) -> dict[str, Any]:
    return {
        "ok": ok,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "skipped": True,
        "not_run_reason": reason,
    }


def _seed_not_run(reason: str, *, ok: bool) -> dict[str, Any]:
    return {
        "ok": ok,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "skipped": True,
        "not_run_reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    code, summary = asyncio.run(_run(args))
    _write_result(args.output_result, summary)
    _print_summary(summary, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
