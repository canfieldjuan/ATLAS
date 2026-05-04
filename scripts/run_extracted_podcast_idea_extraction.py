#!/usr/bin/env python3
"""Extract ranked ideas from podcast transcripts and persist them."""

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

from extracted_content_pipeline.podcast_example import (  # noqa: E402
    DeterministicPodcastLLM,
)
from extracted_content_pipeline.podcast_idea_data import (  # noqa: E402
    load_podcast_idea_provider,
)
from extracted_content_pipeline.podcast_postgres_extraction import (  # noqa: E402
    extract_podcast_ideas_from_postgres,
)
from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ranked ideas from podcast_transcripts and persist into "
            "podcast_extracted_ideas."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", help="Tenant/account scope filter.")
    parser.add_argument("--user-id", help="User id to attach to extraction metadata.")
    parser.add_argument(
        "--episode-id",
        help="If set, extract just this episode; otherwise process up to --limit.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--target-idea-count", type=int, default=3)
    parser.add_argument(
        "--ideas-file",
        type=Path,
        help=(
            "Optional JSON file of pre-baked ideas keyed by episode_id. "
            "When set, skips the LLM and uses FilePodcastIdeaProvider as the "
            "BYO extractor."
        ),
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        help=(
            "Optional directory of host-provided markdown skill prompts. "
            "Custom prompts override packaged prompts with the same name."
        ),
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
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for the Postgres runner; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


def _dependency_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.ideas_file:
        overrides["extractor"] = load_podcast_idea_provider(args.ideas_file)
    if args.skills_root:
        overrides["skills"] = get_skill_registry(root=args.skills_root)
    if args.llm == "offline":
        overrides["llm"] = DeterministicPodcastLLM()
    return overrides


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await extract_podcast_ideas_from_postgres(
            pool,
            scope={"account_id": args.account_id, "user_id": args.user_id},
            episode_id=args.episode_id,
            limit=args.limit,
            target_idea_count=args.target_idea_count,
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
