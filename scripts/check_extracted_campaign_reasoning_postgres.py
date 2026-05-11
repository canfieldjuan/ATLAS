#!/usr/bin/env python3
"""Check DB-backed Content Ops reasoning context lookup."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_reasoning_postgres import (  # noqa: E402
    PostgresCampaignReasoningContextRepository,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a campaign_reasoning_contexts row can be read for a "
            "target through the DB-backed Content Ops reasoning provider."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", default="", help="Tenant/account id.")
    parser.add_argument("--target-id", required=True, help="Primary target id.")
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--company-name", default="")
    parser.add_argument("--company", default="")
    parser.add_argument("--contact-email", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--vendor-name", default="")
    parser.add_argument("--vendor", default="")
    parser.add_argument(
        "--table",
        default="campaign_reasoning_contexts",
        help="Reasoning context table name.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to check campaign reasoning context; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _check_reasoning_context(
    repository: Any,
    *,
    account_id: str,
    target_id: str,
    target_mode: str,
    opportunity: Mapping[str, Any],
) -> dict[str, Any]:
    context = await repository.read_campaign_reasoning_context(
        scope=TenantScope(account_id=account_id),
        target_id=target_id,
        target_mode=target_mode,
        opportunity=opportunity,
    )
    if context is None:
        return {
            "status": "missing",
            "target_id": target_id,
            "target_mode": target_mode,
        }
    payload = context.as_dict() if hasattr(context, "as_dict") else dict(context)
    return {
        "status": "ok",
        "target_id": target_id,
        "target_mode": target_mode,
        "context": payload,
    }


def _opportunity_from_args(args: argparse.Namespace) -> dict[str, Any]:
    pairs = {
        "company_name": args.company_name,
        "company": args.company,
        "contact_email": args.contact_email,
        "email": args.email,
        "vendor_name": args.vendor_name,
        "vendor": args.vendor,
    }
    return {
        key: str(value).strip()
        for key, value in pairs.items()
        if str(value or "").strip()
    }


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        repository = PostgresCampaignReasoningContextRepository(
            pool=pool,
            table=args.table,
        )
        result = await _check_reasoning_context(
            repository,
            account_id=str(args.account_id or ""),
            target_id=str(args.target_id or ""),
            target_mode=str(args.target_mode or ""),
            opportunity=_opportunity_from_args(args),
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["status"] == "ok":
        print(
            "found reasoning context for "
            f"target_id={result['target_id']} target_mode={result['target_mode']}"
        )
    else:
        print(
            "missing reasoning context for "
            f"target_id={result['target_id']} target_mode={result['target_mode']}",
            file=sys.stderr,
        )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
