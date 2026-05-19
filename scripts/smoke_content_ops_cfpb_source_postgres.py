#!/usr/bin/env python3
"""Smoke-test CFPB source rows through Postgres-backed draft persistence."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_llm_client import (  # noqa: E402
    create_pipeline_llm_client,
)
from extracted_content_pipeline.campaign_postgres_import import (  # noqa: E402
    import_campaign_opportunities,
)
from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)
from extracted_content_pipeline.ingestion_diagnostics import (  # noqa: E402
    inspect_ingestion_file,
)
from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: E402

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - host dependency
    load_dotenv = None


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cfpb_exporter_module():
    path = ROOT / "scripts" / "export_content_ops_cfpb_sources.py"
    spec = importlib.util.spec_from_file_location("content_ops_cfpb_exporter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CFPB source exporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_source_postgres_helpers = _load_script_module(
    "content_ops_source_postgres_smoke_helpers",
    ROOT / "scripts" / "content_ops_source_postgres_smoke_helpers.py",
)
_cfpb_exporter = _load_cfpb_exporter_module()
DEFAULT_CHANNELS = ("email_cold", "email_followup")
DEFAULT_FORBIDDEN_PHRASES = ("appears to be weighing",)
DEFAULT_API_URL = _cfpb_exporter.DEFAULT_API_URL
DEFAULT_MAX_ROWS_SCANNED = _cfpb_exporter.DEFAULT_MAX_ROWS_SCANNED
DEFAULT_SOURCE_SYSTEM = _cfpb_exporter.DEFAULT_SOURCE_SYSTEM
DEFAULT_SOURCE_TYPE = _cfpb_exporter.DEFAULT_SOURCE_TYPE
DEFAULT_TIMEOUT_SECONDS = _cfpb_exporter.DEFAULT_TIMEOUT_SECONDS
fetch_cfpb_source_rows = _cfpb_exporter.fetch_cfpb_source_rows
render_jsonl = _cfpb_exporter.render_jsonl
draft_errors = _source_postgres_helpers.draft_errors
fetch_saved_drafts = _source_postgres_helpers.fetch_saved_drafts
generate_imported_target_drafts = _source_postgres_helpers.generate_imported_target_drafts
generation_errors = _source_postgres_helpers.generation_errors
saved_draft_target_errors = _source_postgres_helpers.saved_draft_target_errors
schema_readiness_errors = _source_postgres_helpers.schema_readiness_errors


def _default_database_url() -> str | None:
    raw = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if raw:
        return raw
    try:
        from atlas_brain.storage.config import db_settings
    except Exception:
        return None
    dsn = str(getattr(db_settings, "dsn", "") or "").strip()
    return dsn or None


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for the CFPB Postgres smoke; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


def _csv_list(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    return tuple(str(item or "").strip() for item in raw if str(item or "").strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test CFPB complaint rows through Content Ops source export, "
            "Postgres import, and DB-backed draft generation."
        )
    )
    parser.add_argument("--company", default=None)
    parser.add_argument("--product", default=None)
    parser.add_argument("--issue", default=None)
    parser.add_argument("--search-term", default=None)
    parser.add_argument("--date-received-min", default=None)
    parser.add_argument("--date-received-max", default=None)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-rows-scanned", type=int, default=DEFAULT_MAX_ROWS_SCANNED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--source-system", default=DEFAULT_SOURCE_SYSTEM)
    parser.add_argument("--source-type", default=DEFAULT_SOURCE_TYPE)
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    parser.add_argument(
        "--llm",
        choices=("offline", "pipeline"),
        default="offline",
        help="Use deterministic offline generation or the product PipelineLLMClient.",
    )
    parser.add_argument("--min-drafts", type=int, default=None)
    parser.add_argument("--allow-ingestion-warnings", action="store_true")
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help="Fallback metadata applied to every CFPB source row. Repeat as key=value.",
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="Tenant/account id used to scope imported opportunities and drafts.",
    )
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--opportunity-table", default="campaign_opportunities")
    parser.add_argument("--keep-existing-opportunities", action="store_true")
    parser.add_argument(
        "--forbidden-phrase",
        action="append",
        default=list(DEFAULT_FORBIDDEN_PHRASES),
    )
    parser.add_argument("--output-source-rows", type=Path, default=None)
    parser.add_argument("--output-result", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--database-url", default=_default_database_url())
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.limit) < 1:
        raise SystemExit("--limit must be positive")
    if int(args.max_rows_scanned) < int(args.limit):
        raise SystemExit("--max-rows-scanned must be >= --limit")
    if float(args.timeout) <= 0:
        raise SystemExit("--timeout must be positive")
    if args.min_drafts is not None and int(args.min_drafts) < 1:
        raise SystemExit("--min-drafts must be positive")
    if not _csv_list(args.channels):
        raise SystemExit("--channels must include at least one channel")
    if not str(args.account_id or "").strip():
        raise SystemExit("--account-id is required")
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")


def _generation_ports_for_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.llm == "offline":
        return {}
    return {
        "llm": create_pipeline_llm_client(),
        "skills": get_skill_registry(),
    }


async def run_cfpb_source_postgres_smoke(
    args: argparse.Namespace,
    *,
    source_rows_path: Path,
) -> tuple[int, dict[str, Any]]:
    default_fields = parse_default_fields_or_exit(args.default_field)
    channels = _csv_list(args.channels)
    min_drafts = (
        int(args.min_drafts)
        if args.min_drafts is not None
        else int(args.limit) * len(channels)
    )
    pool = await _create_pool(str(args.database_url))
    source_rows: list[dict[str, Any]] = []
    ingestion_report: dict[str, Any] | None = None
    import_result: dict[str, Any] | None = None
    drafts_result: dict[str, Any] | None = None
    saved_drafts: list[dict[str, Any]] = []
    imported_target_ids: list[str] = []
    errors: list[str] = []
    try:
        try:
            source_rows = fetch_cfpb_source_rows(
                api_url=str(args.api_url),
                company=args.company,
                product=args.product,
                issue=args.issue,
                search_term=args.search_term,
                date_received_min=args.date_received_min,
                date_received_max=args.date_received_max,
                limit=int(args.limit),
                max_rows_scanned=int(args.max_rows_scanned),
                timeout=float(args.timeout),
                source_system=str(args.source_system),
                source_type=str(args.source_type),
            )
            if len(source_rows) < int(args.limit):
                errors.append(
                    f"expected {int(args.limit)} CFPB source row(s), got {len(source_rows)}"
                )
            if not errors:
                _write_jsonl(source_rows, source_rows_path)
                report = inspect_ingestion_file(
                    source_rows_path,
                    source_rows=True,
                    source_format="jsonl",
                    target_mode=str(args.target_mode),
                    default_fields=default_fields,
                )
                ingestion_report = report.as_dict()
                if not report.ok:
                    errors.append("ingestion inspection failed")
                if report.warnings and not bool(args.allow_ingestion_warnings):
                    errors.append(
                        "ingestion inspection produced warnings; provide --default-field "
                        "bindings or pass --allow-ingestion-warnings"
                    )
            if not errors:
                errors.extend(
                    await schema_readiness_errors(
                        pool,
                        opportunity_table=str(args.opportunity_table),
                    )
                )
            if not errors:
                loaded = load_source_campaign_opportunities_from_file(
                    source_rows_path,
                    file_format="jsonl",
                    target_mode=str(args.target_mode),
                    default_fields=default_fields,
                )
                imported = await import_campaign_opportunities(
                    pool,
                    loaded.opportunities,
                    scope=TenantScope(account_id=str(args.account_id), user_id=args.user_id),
                    target_mode=str(args.target_mode),
                    opportunity_table=str(args.opportunity_table),
                    replace_existing=not bool(args.keep_existing_opportunities),
                    normalize=False,
                    warnings=loaded.warnings,
                    source=loaded.source,
                )
                import_result = imported.as_dict()
                imported_target_ids = list(imported.target_ids)
                if imported.skipped:
                    errors.append(f"import skipped {imported.skipped} row(s)")
            if not errors:
                drafts_result = await generate_imported_target_drafts(
                    pool=pool,
                    account_id=str(args.account_id),
                    user_id=args.user_id,
                    target_mode=str(args.target_mode),
                    channels=channels,
                    target_ids=imported_target_ids,
                    opportunity_table=str(args.opportunity_table),
                    **_generation_ports_for_args(args),
                )
                saved_drafts = await fetch_saved_drafts(pool, drafts_result.get("saved_ids") or [])
                errors.extend(generation_errors(drafts_result))
                errors.extend(draft_errors(
                    {"drafts": saved_drafts},
                    min_drafts=min_drafts,
                    forbidden_phrases=args.forbidden_phrase,
                ))
                errors.extend(saved_draft_target_errors(saved_drafts, imported_target_ids))
        except Exception as exc:  # pragma: no cover - exercised by live hosts
            errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    payload = {
        "ok": not errors,
        "source": "cfpb",
        "source_rows": len(source_rows),
        "source_rows_path": str(source_rows_path),
        "ingestion": ingestion_report,
        "import": import_result,
        "drafts": drafts_result,
        "saved_drafts": saved_drafts,
        "errors": errors,
    }
    if args.output_result:
        args.output_result.parent.mkdir(parents=True, exist_ok=True)
        args.output_result.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return (0 if not errors else 1), payload


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_jsonl(rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
        return
    if payload.get("ok"):
        imported = payload.get("import") if isinstance(payload.get("import"), Mapping) else {}
        drafts = payload.get("drafts") if isinstance(payload.get("drafts"), Mapping) else {}
        print(
            "Content Ops CFPB Postgres smoke passed: "
            f"source_rows={payload.get('source_rows')} "
            f"imported={imported.get('inserted', 0)} "
            f"generated={drafts.get('generated', 0)}"
        )
        return
    print("Content Ops CFPB Postgres smoke failed:", file=sys.stderr)
    for error in payload.get("errors") or []:
        print(f"- {error}", file=sys.stderr)


async def _main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()
    args = _parse_args(argv)
    _validate_args(args)
    if args.output_source_rows:
        code, payload = await run_cfpb_source_postgres_smoke(
            args,
            source_rows_path=args.output_source_rows,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="content-ops-cfpb-source-db-") as tmpdir:
            code, payload = await run_cfpb_source_postgres_smoke(
                args,
                source_rows_path=Path(tmpdir) / "cfpb_sources.jsonl",
            )
    _print_payload(payload, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
