#!/usr/bin/env python3
"""Reconcile pending Content Ops FAQ macro writeback mappings."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain._content_ops_macro_writeback import (  # noqa: E402
    ConfigZendeskMacroCredentialsProvider,
)
from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.faq_macro_writeback import (  # noqa: E402
    MacroWritebackMapping,
)
from extracted_content_pipeline.faq_macro_writeback_postgres import (  # noqa: E402
    PostgresFAQMacroWritebackMappingRepository,
)
from extracted_content_pipeline.faq_macro_writeback_zendesk import (  # noqa: E402
    ZENDESK_PLATFORM,
    ZendeskMacroPublishProvider,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", required=True, help="Tenant/account id.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum pending rows.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform reconciliation. Without this flag the command only previews.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to reconcile macro writebacks; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    pool = await _create_pool(args.database_url)
    try:
        payload = await reconcile_pending_macro_writebacks(args, pool)
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    if not str(args.account_id or "").strip():
        raise SystemExit("--account-id is required")
    if args.limit < 1:
        raise SystemExit("--limit must be positive")


async def reconcile_pending_macro_writebacks(
    args: argparse.Namespace,
    pool: Any,
    *,
    provider: Any | None = None,
) -> dict[str, Any]:
    scope = TenantScope(account_id=str(args.account_id).strip())
    repository = PostgresFAQMacroWritebackMappingRepository(pool)
    pending = await repository.list_pending_mappings(
        platform=ZENDESK_PLATFORM,
        scope=scope,
        limit=args.limit,
    )
    if not args.execute:
        results = [_dry_run_result(mapping) for mapping in pending]
    else:
        reconciler = provider or _build_provider(pool)
        results = [
            (
                await reconciler.reconcile_pending_mapping(mapping, scope=scope)
            ).as_dict()
            for mapping in pending
        ]
    return _summary(
        account_id=scope.account_id or "",
        execute=bool(args.execute),
        limit=args.limit,
        pending=pending,
        results=results,
    )


def _build_provider(pool: Any) -> ZendeskMacroPublishProvider:
    return ZendeskMacroPublishProvider(
        credentials_provider=ConfigZendeskMacroCredentialsProvider(_host_config()),
        mapping_repository=PostgresFAQMacroWritebackMappingRepository(pool),
    )


def _host_config() -> Any:
    from atlas_brain.config import settings

    return settings.b2b_campaign


def _dry_run_result(mapping: MacroWritebackMapping) -> dict[str, Any]:
    return {
        "mapping": mapping.as_dict(),
        "status": "dry_run",
        "external_id": "",
        "error": "",
    }


def _summary(
    *,
    account_id: str,
    execute: bool,
    limit: int,
    pending: Sequence[MacroWritebackMapping],
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(str(result.get("status") or "") for result in results)
    return {
        "account_id": account_id,
        "platform": ZENDESK_PLATFORM,
        "execute": execute,
        "limit": limit,
        "pending_count": len(pending),
        "result_counts": dict(sorted(counts.items())),
        "results": list(results),
    }


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
