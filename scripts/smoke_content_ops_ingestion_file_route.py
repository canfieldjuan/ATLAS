#!/usr/bin/env python3
"""Smoke-test the Content Ops uploaded-file ingestion route."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.api.control_surfaces import (  # noqa: E402
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    parse_default_fields_or_exit,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency.
    load_dotenv = None


_REQUIRED_DEFAULT_FIELDS = ("company_name", "vendor_name", "contact_email")


class _LocalUploadFile:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._content) - self._offset
        start = self._offset
        end = min(len(self._content), start + int(size))
        self._offset = end
        return self._content[start:end]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run uploaded source rows through /content-ops/ingestion/files/import."
    )
    parser.add_argument("path", type=Path, help="Source-row JSON, JSONL, or CSV file.")
    parser.add_argument("--source-format", choices=("auto", "json", "jsonl", "csv"), default="auto")
    parser.add_argument("--source", default="content-ops-file-route-smoke")
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--min-source-rows", type=int, default=1)
    parser.add_argument("--max-source-text-chars", type=int, default=1200)
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help=(
            "Fallback metadata for public/source-row files. Repeat as key=value. "
            "The smoke requires company_name, vendor_name, and contact_email."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist rows through the route instead of using dry_run=True.",
    )
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--opportunity-table", default="campaign_opportunities")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--database-url", default=_default_database_url())
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    code, payload = asyncio.run(run_route_smoke(args))
    if args.output_result is not None:
        args.output_result.parent.mkdir(parents=True, exist_ok=True)
        args.output_result.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "ok={ok} status={status} opportunities={opportunities} errors={errors}".format(
                ok=str(payload["ok"]).lower(),
                status=payload.get("status_code"),
                opportunities=(payload.get("diagnostics") or {}).get("opportunity_count"),
                errors=len(payload.get("errors") or ()),
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
    defaults = parse_default_fields_or_exit(args.default_field)
    missing = [field for field in _REQUIRED_DEFAULT_FIELDS if not str(defaults.get(field) or "").strip()]
    if missing:
        raise SystemExit(
            "uploaded source-row smoke requires default fields: " + ", ".join(missing)
        )
    if bool(args.write):
        if not str(args.account_id or "").strip():
            raise SystemExit("--account-id is required when --write is selected")
        if not str(args.database_url or "").strip():
            raise SystemExit(
                "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL"
            )
    if not str(args.opportunity_table or "").strip():
        raise SystemExit("--opportunity-table is required")


async def run_route_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    defaults = parse_default_fields_or_exit(args.default_field)
    route_path = "/ops/ingestion/files/import"
    base_payload = {
        "ok": False,
        "route": route_path,
        "source_path": str(args.path),
        "source_format": str(args.source_format),
        "source": str(args.source),
        "dry_run": not bool(args.write),
        "account_id": str(args.account_id or "").strip() or None,
        "user_id": str(args.user_id or "").strip() or None,
        "opportunity_table": str(args.opportunity_table),
        "replace_existing": bool(args.replace_existing),
        "min_source_rows": int(args.min_source_rows),
        "elapsed_seconds": None,
        "status_code": None,
        "diagnostics": None,
        "import": None,
        "errors": [],
    }
    pool = None
    try:
        if args.write:
            pool = await _create_pool(str(args.database_url))
        router = create_content_ops_control_surface_router(
            config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
            opportunity_import_pool_provider=(lambda: pool) if pool is not None else None,
            scope_provider=_scope_provider(args) if args.write else None,
        )
        route = _route(router, route_path, "POST")
        response = await route.endpoint(
            file=_LocalUploadFile(args.path.name, args.path.read_bytes()),
            source_rows=True,
            source=str(args.source),
            target_mode=str(args.target_mode),
            file_format=str(args.source_format),
            max_source_text_chars=int(args.max_source_text_chars),
            sample_limit=int(args.sample_limit),
            default_fields=json.dumps(defaults, sort_keys=True),
            replace_existing=bool(args.replace_existing),
            dry_run=not bool(args.write),
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        payload = {
            **base_payload,
            "status_code": status_code,
            "detail": _compact_detail(getattr(exc, "detail", str(exc))),
            "errors": [f"{type(exc).__name__}: {getattr(exc, 'detail', exc)}"],
        }
        payload["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return 1, payload
    finally:
        if pool is not None:
            await _close_pool(pool)

    diagnostics = _diagnostics_summary(response.get("diagnostics"))
    import_summary = _import_summary(response.get("import"))
    errors = _result_errors(
        diagnostics=diagnostics,
        import_summary=import_summary,
        min_source_rows=int(args.min_source_rows),
        expected_dry_run=not bool(args.write),
    )
    payload = {
        **base_payload,
        "ok": not errors,
        "status_code": 200,
        "diagnostics": diagnostics,
        "import": import_summary,
        "errors": errors,
    }
    payload["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return (0 if payload["ok"] else 1), payload


def _route(router: Any, path: str, method: str) -> Any:
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(route, "methods", set()):
            return route
    raise RuntimeError(f"route not found: {method} {path}")


def _diagnostics_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    samples = value.get("samples")
    return {
        "ok": bool(value.get("ok")),
        "mode": value.get("mode"),
        "source": value.get("source"),
        "ingestion_path": value.get("ingestion_path"),
        "opportunity_count": int(value.get("opportunity_count") or 0),
        "warning_count": int(value.get("warning_count") or 0),
        "warning_counts": dict(value.get("warning_counts") or {}),
        "missing_field_counts": dict(value.get("missing_field_counts") or {}),
        "source_type_counts": dict(value.get("source_type_counts") or {}),
        "sample_count": len(samples) if isinstance(samples, list) else 0,
        "limits": dict(value.get("limits") or {}),
    }


def _import_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    target_ids = value.get("target_ids")
    return {
        "dry_run": bool(value.get("dry_run")),
        "inserted": int(value.get("inserted") or 0),
        "skipped": int(value.get("skipped") or 0),
        "replace_existing": bool(value.get("replace_existing")),
        "source": value.get("source"),
        "target_id_count": len(target_ids) if isinstance(target_ids, list) else 0,
        "first_target_id": target_ids[0] if isinstance(target_ids, list) and target_ids else None,
    }


def _result_errors(
    *,
    diagnostics: Mapping[str, Any] | None,
    import_summary: Mapping[str, Any] | None,
    min_source_rows: int,
    expected_dry_run: bool,
) -> list[str]:
    errors: list[str] = []
    if not diagnostics or not diagnostics.get("ok"):
        errors.append("ingestion diagnostics did not report ok=true")
    opportunity_count = int((diagnostics or {}).get("opportunity_count") or 0)
    if opportunity_count < min_source_rows:
        errors.append(f"expected at least {min_source_rows} source row(s), got {opportunity_count}")
    actual_dry_run = bool((import_summary or {}).get("dry_run"))
    if actual_dry_run is not bool(expected_dry_run):
        errors.append(
            "file import route returned dry_run="
            f"{str(actual_dry_run).lower()}, expected {str(expected_dry_run).lower()}"
        )
    inserted = int((import_summary or {}).get("inserted") or 0)
    if inserted < min_source_rows:
        errors.append(f"expected at least {min_source_rows} inserted row(s), got {inserted}")
    return errors


def _scope_provider(args: argparse.Namespace):
    scope = TenantScope(
        account_id=str(args.account_id or "").strip() or None,
        user_id=str(args.user_id or "").strip() or None,
    )
    return lambda: scope


def _default_database_url() -> str | None:
    _load_dotenv_files()
    raw = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if raw:
        return raw
    try:
        from atlas_brain.storage.config import db_settings
    except Exception:
        return None
    dsn = str(getattr(db_settings, "dsn", "") or "").strip()
    return dsn or None


async def _create_pool(database_url: str):
    _load_dotenv_files()
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency.
        raise RuntimeError(
            "asyncpg is required for --write; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _close_pool(pool: Any) -> None:
    close = getattr(pool, "close", None)
    if close is None:
        return
    maybe_awaitable = close()
    if hasattr(maybe_awaitable, "__await__"):
        await maybe_awaitable


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _compact_detail(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    detail = dict(value)
    diagnostics = detail.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        detail["diagnostics"] = _diagnostics_summary(diagnostics)
    return detail


if __name__ == "__main__":
    raise SystemExit(main())
