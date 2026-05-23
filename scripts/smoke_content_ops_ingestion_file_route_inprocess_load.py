#!/usr/bin/env python3
"""Run concurrent uploaded-file imports through one shared route pool."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extracted_content_pipeline.api.control_surfaces import (  # noqa: E402
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    parse_default_fields_or_exit,
)
from smoke_content_ops_ingestion_file_route import (  # noqa: E402
    _LocalUploadFile,
    _close_pool,
    _compact_detail,
    _create_pool,
    _default_database_url,
    _diagnostics_summary,
    _import_summary,
)


_EXPECTED_AT_CAPACITY_REASON = "content_ops_ingestion_import_at_capacity"
_REQUIRED_DEFAULT_FIELDS = ("company_name", "vendor_name", "contact_email")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run concurrent /content-ops/ingestion/files/import writes inside "
            "one process using one shared import pool."
        )
    )
    parser.add_argument("path", type=Path, help="Source-row JSON, JSONL, or CSV file.")
    parser.add_argument("--source-format", choices=("auto", "json", "jsonl", "csv"), default="auto")
    parser.add_argument("--source", default="content-ops-file-route-inprocess-load")
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--min-source-rows", type=int, default=1)
    parser.add_argument("--max-source-text-chars", type=int, default=1200)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help=(
            "Fallback metadata for public/source-row files. Repeat as key=value. "
            "The load runner requires company_name, vendor_name, and contact_email."
        ),
    )
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--opportunity-table", default="campaign_opportunities")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--database-url", default=_default_database_url())
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--import-max-concurrency", type=int, default=2)
    parser.add_argument("--min-successes", type=int, default=1)
    parser.add_argument("--expect-at-capacity-min", type=int, default=1)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    code, payload = asyncio.run(run_inprocess_load(args))
    if args.output_result is not None:
        args.output_result.parent.mkdir(parents=True, exist_ok=True)
        args.output_result.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload["summary"]
        print(
            "ok={ok} successes={successes} at_capacity={at_capacity} "
            "failures={failures} inserted={inserted}".format(
                ok=str(payload["ok"]).lower(),
                successes=summary["successes"],
                at_capacity=summary["at_capacity"],
                failures=summary["unexpected_failures"],
                inserted=summary["inserted"],
            )
        )
    return code


def _validate_args(args: argparse.Namespace) -> None:
    if not args.path.exists():
        raise SystemExit(f"source file not found: {args.path}")
    if int(args.min_source_rows) < 1:
        raise SystemExit("--min-source-rows must be positive")
    if int(args.max_source_text_chars) < 1:
        raise SystemExit("--max-source-text-chars must be positive")
    if int(args.sample_limit) < 0:
        raise SystemExit("--sample-limit must be non-negative")
    if int(args.concurrency) < 1:
        raise SystemExit("--concurrency must be positive")
    if int(args.import_max_concurrency) < 1:
        raise SystemExit("--import-max-concurrency must be positive")
    if int(args.min_successes) < 0:
        raise SystemExit("--min-successes must be non-negative")
    if int(args.expect_at_capacity_min) < 0:
        raise SystemExit("--expect-at-capacity-min must be non-negative")
    if not str(args.account_id or "").strip():
        raise SystemExit("--account-id is required")
    if not str(args.database_url or "").strip():
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    if not str(args.opportunity_table or "").strip():
        raise SystemExit("--opportunity-table is required")
    defaults = parse_default_fields_or_exit(args.default_field)
    missing = [field for field in _REQUIRED_DEFAULT_FIELDS if not str(defaults.get(field) or "").strip()]
    if missing:
        raise SystemExit(
            "uploaded source-row load runner requires default fields: "
            + ", ".join(missing)
        )


async def run_inprocess_load(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    pool = None
    route_path = "/ops/ingestion/files/import"
    source_bytes = args.path.read_bytes()
    defaults = parse_default_fields_or_exit(args.default_field)
    base_payload = {
        "ok": False,
        "route": route_path,
        "source_path": str(args.path),
        "source_format": str(args.source_format),
        "source": str(args.source),
        "account_id": str(args.account_id or "").strip(),
        "user_id": str(args.user_id or "").strip() or None,
        "opportunity_table": str(args.opportunity_table),
        "replace_existing": bool(args.replace_existing),
        "concurrency": int(args.concurrency),
        "import_max_concurrency": int(args.import_max_concurrency),
        "min_source_rows": int(args.min_source_rows),
        "min_successes": int(args.min_successes),
        "expect_at_capacity_min": int(args.expect_at_capacity_min),
        "elapsed_seconds": None,
        "summary": None,
        "results": [],
        "errors": [],
    }
    try:
        pool = await _create_pool(str(args.database_url))
        router = create_content_ops_control_surface_router(
            config=ContentOpsControlSurfaceApiConfig(
                prefix="/ops",
                tags=("ops",),
                ingestion_opportunity_table=str(args.opportunity_table),
                ingestion_import_max_concurrency=int(args.import_max_concurrency),
            ),
            opportunity_import_pool_provider=lambda: pool,
            scope_provider=_scope_provider(args),
        )
        route = _route(router, route_path, "POST")
        start_event = asyncio.Event()
        tasks = [
            asyncio.create_task(
                _run_one_import(
                    index=index,
                    route=route,
                    start_event=start_event,
                    source_bytes=source_bytes,
                    filename=args.path.name,
                    defaults=defaults,
                    args=args,
                )
            )
            for index in range(int(args.concurrency))
        ]
        start_event.set()
        results = await asyncio.gather(*tasks)
    except Exception as exc:
        payload = {
            **base_payload,
            "summary": _empty_summary(),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        payload["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return 1, payload
    finally:
        if pool is not None:
            await _close_pool(pool)

    summary = _summarize_results(results)
    errors = _load_errors(
        summary,
        min_source_rows=int(args.min_source_rows),
        min_successes=int(args.min_successes),
        expect_at_capacity_min=int(args.expect_at_capacity_min),
    )
    payload = {
        **base_payload,
        "ok": not errors,
        "summary": summary,
        "results": results,
        "errors": errors,
    }
    payload["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return (0 if payload["ok"] else 1), payload


async def _run_one_import(
    *,
    index: int,
    route: Any,
    start_event: asyncio.Event,
    source_bytes: bytes,
    filename: str,
    defaults: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    await start_event.wait()
    started = time.monotonic()
    try:
        response = await route.endpoint(
            file=_LocalUploadFile(filename, source_bytes),
            source_rows=True,
            source=str(args.source),
            target_mode=str(args.target_mode),
            file_format=str(args.source_format),
            max_source_text_chars=int(args.max_source_text_chars),
            sample_limit=int(args.sample_limit),
            default_fields=json.dumps(dict(defaults), sort_keys=True),
            replace_existing=bool(args.replace_existing),
            dry_run=False,
        )
    except Exception as exc:
        status_code = int(getattr(exc, "status_code", 500) or 500)
        detail = _compact_detail(getattr(exc, "detail", str(exc)))
        reason = _detail_reason(detail)
        return {
            "index": index,
            "ok": False,
            "status_code": status_code,
            "reason": reason,
            "detail": detail,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }

    diagnostics = _diagnostics_summary(response.get("diagnostics"))
    import_summary = _import_summary(response.get("import"))
    inserted = int((import_summary or {}).get("inserted") or 0)
    return {
        "index": index,
        "ok": True,
        "status_code": 200,
        "reason": None,
        "inserted": inserted,
        "diagnostics": diagnostics,
        "import": import_summary,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _route(router: Any, path: str, method: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set()):
            return route
    raise RuntimeError(f"route not found: {method} {path}")


def _scope_provider(args: argparse.Namespace):
    scope = TenantScope(
        account_id=str(args.account_id or "").strip() or None,
        user_id=str(args.user_id or "").strip() or None,
    )
    return lambda: scope


def _summarize_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    successes = [item for item in results if item.get("ok")]
    at_capacity = [
        item
        for item in results
        if int(item.get("status_code") or 0) == 429
        and item.get("reason") == _EXPECTED_AT_CAPACITY_REASON
    ]
    unexpected_failures = [
        item
        for item in results
        if not item.get("ok") and item not in at_capacity
    ]
    return {
        "successes": len(successes),
        "at_capacity": len(at_capacity),
        "unexpected_failures": len(unexpected_failures),
        "inserted": sum(int(item.get("inserted") or 0) for item in successes),
        "status_counts": _count_by(results, "status_code"),
        "reason_counts": _count_by(
            [item for item in results if item.get("reason")],
            "reason",
        ),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "successes": 0,
        "at_capacity": 0,
        "unexpected_failures": 0,
        "inserted": 0,
        "status_counts": {},
        "reason_counts": {},
    }


def _load_errors(
    summary: Mapping[str, Any],
    *,
    min_source_rows: int,
    min_successes: int,
    expect_at_capacity_min: int,
) -> list[str]:
    errors: list[str] = []
    successes = int(summary.get("successes") or 0)
    at_capacity = int(summary.get("at_capacity") or 0)
    inserted = int(summary.get("inserted") or 0)
    unexpected_failures = int(summary.get("unexpected_failures") or 0)
    if inserted < min_source_rows:
        errors.append(
            "expected at least "
            f"{min_source_rows} inserted source row(s), got {inserted}"
        )
    if successes < min_successes:
        errors.append(f"expected at least {min_successes} success(es), got {successes}")
    if at_capacity < expect_at_capacity_min:
        errors.append(
            "expected at least "
            f"{expect_at_capacity_min} admission 429 response(s), got {at_capacity}"
        )
    if unexpected_failures:
        errors.append(f"unexpected failure count: {unexpected_failures}")
    return errors


def _count_by(items: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "").strip()
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _detail_reason(detail: Any) -> str | None:
    if isinstance(detail, Mapping):
        text = str(detail.get("reason") or "").strip()
        return text or None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
