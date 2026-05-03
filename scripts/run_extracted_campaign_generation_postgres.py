#!/usr/bin/env python3
"""Generate campaign drafts from the product Postgres opportunity table."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_example import (  # noqa: E402
    DeterministicCampaignLLM,
    StaticCampaignSkillStore,
)
from extracted_content_pipeline.campaign_postgres_generation import (  # noqa: E402
    generate_campaign_drafts_from_postgres,
)


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate campaign drafts from campaign_opportunities and persist "
            "drafts into b2b_campaigns."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", help="Tenant/account scope filter.")
    parser.add_argument("--user-id", help="User id to attach to generated draft metadata.")
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--channel", default="email")
    parser.add_argument(
        "--channels",
        help=(
            "Comma-separated draft channels to generate per opportunity, "
            "for example email_cold,email_followup."
        ),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--filters-json",
        help="Optional JSON object of opportunity filters.",
    )
    parser.add_argument(
        "--llm",
        choices=("pipeline", "offline"),
        default="pipeline",
        help="Use configured PipelineLLMClient or deterministic offline LLM.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write result JSON to this path instead of stdout.",
    )
    return parser.parse_args()


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError(
            "asyncpg is required for the Postgres runner; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


def _dependency_overrides(args: argparse.Namespace) -> dict[str, Any]:
    if args.llm == "pipeline":
        return {}
    return {
        "llm": DeterministicCampaignLLM(),
        "skills": StaticCampaignSkillStore(),
    }


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await generate_campaign_drafts_from_postgres(
            pool,
            scope={"account_id": args.account_id, "user_id": args.user_id},
            target_mode=args.target_mode,
            channel=args.channel,
            channels=tuple(
                item.strip()
                for item in str(args.channels or "").split(",")
                if item.strip()
            ),
            limit=args.limit,
            filters=_json_object(args.filters_json),
            **_dependency_overrides(args),
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    output = json.dumps(result.as_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
