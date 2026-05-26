#!/usr/bin/env python3
"""Seed FAQ search rows, hit the hosted route, then clean up."""

from __future__ import annotations

import argparse
import asyncio
import json
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
    if float(args.timeout) <= 0:
        errors.append("--timeout must be positive")
    if not 0 <= float(args.max_error_rate) <= 1:
        errors.append("--max-error-rate must be between 0 and 1")
    for name in ("max_p95_ms", "max_single_request_ms"):
        value = getattr(args, name)
        if value is not None and float(value) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
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
    ):
        value = getattr(args, arg_name)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


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
        return {
            "ok": True,
            "requested_faq_ids": 0,
            "deleted_faq_ids": 0,
            "delete_status": None,
            "error": None,
        }
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        return {
            "ok": False,
            "requested_faq_ids": requested_faq_ids,
            "deleted_faq_ids": 0,
            "delete_status": None,
            "error": f"ImportError: {exc}",
        }

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
        return {
            "ok": False,
            "requested_faq_ids": requested_faq_ids,
            "deleted_faq_ids": 0,
            "delete_status": delete_status,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": True,
        "requested_faq_ids": requested_faq_ids,
        "deleted_faq_ids": _deleted_row_count(delete_status),
        "delete_status": delete_status,
        "error": None,
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
    print(
        "FAQ search seeded route e2e: "
        f"ok={summary['ok']} seed={summary['seed']['ok']} "
        f"route={summary['route']['ok']} cleanup={summary['cleanup']['ok']}"
    )


def _preflight_summary(args: argparse.Namespace, errors: Sequence[str], elapsed: float) -> dict[str, Any]:
    return {
        "ok": False,
        "phase": "preflight",
        "artifacts": {},
        "seed": {"ok": False, "returncode": None},
        "route": {"ok": False, "returncode": None},
        "cleanup": {
            "ok": False,
            "requested_faq_ids": 0,
            "deleted_faq_ids": 0,
            "delete_status": None,
            "error": None,
        },
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

    cleanup = {
        "ok": True,
        "requested_faq_ids": 0,
        "deleted_faq_ids": 0,
        "delete_status": None,
        "error": None,
    }
    try:
        seed = _run_command(
            _seed_command(
                args,
                case_file=case_file,
                cleanup_manifest=cleanup_manifest,
                seed_result=seed_result,
            )
        )
        route = {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": ""}
        if seed["ok"]:
            route = _run_command(_route_command(args, case_file=case_file, route_result=route_result))
        faq_ids, manifest_errors = (
            _faq_ids_from_cleanup_manifest(cleanup_manifest)
            if cleanup_manifest.exists()
            else ([], [])
        )
        if manifest_errors:
            cleanup = {
                "ok": False,
                "requested_faq_ids": len(faq_ids),
                "deleted_faq_ids": 0,
                "delete_status": None,
                "error": "; ".join(manifest_errors),
            }
        elif not bool(args.keep_data):
            cleanup = await _cleanup_seeded_faqs(str(args.database_url), faq_ids)
        ok = bool(seed["ok"] and route["ok"] and cleanup["ok"])
        summary = {
            "ok": ok,
            "phase": "complete",
            "artifacts": {
                "dir": str(artifact_dir),
                "case_file": str(case_file),
                "cleanup_manifest": str(cleanup_manifest),
                "seed_result": str(seed_result),
                "route_result": str(route_result),
            },
            "seed": seed,
            "route": route,
            "cleanup": cleanup,
            "preflight_errors": [],
            "keep_data": bool(args.keep_data),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        return (0 if ok else 1), summary
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    code, summary = asyncio.run(_run(args))
    _write_result(args.output_result, summary)
    _print_summary(summary, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
