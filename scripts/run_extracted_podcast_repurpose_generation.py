#!/usr/bin/env python3
"""Repurpose extracted podcast ideas into multi-format drafts."""

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
from extracted_content_pipeline.podcast_postgres_repurpose import (  # noqa: E402
    repurpose_podcast_episode_from_postgres,
)
from extracted_content_pipeline.podcast_repurpose_generation import (  # noqa: E402
    SUPPORTED_FORMATS,
)
from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repurpose stored podcast ideas into per-format drafts and persist "
            "them into podcast_format_drafts."
        )
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", help="Tenant/account scope filter.")
    parser.add_argument("--user-id", help="User id to attach to repurpose metadata.")
    parser.add_argument("--episode-id", required=True, help="Episode id to repurpose.")
    parser.add_argument(
        "--formats",
        default=",".join(SUPPORTED_FORMATS),
        help=(
            "Comma-separated list of formats to generate. Defaults to all five: "
            f"{','.join(SUPPORTED_FORMATS)}."
        ),
    )
    parser.add_argument("--idea-limit", type=int, default=3)
    parser.add_argument(
        "--voice-anchors",
        type=Path,
        help=(
            "Optional JSON file with voice_anchors {tone_descriptors, "
            "banned_phrases, style_examples}."
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
    if args.skills_root:
        overrides["skills"] = get_skill_registry(root=args.skills_root)
    if args.llm == "offline":
        overrides["llm"] = DeterministicPodcastLLM()
    return overrides


def _load_voice_anchors(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--voice-anchors file must contain a JSON object")
    return payload


def _resolve_formats(raw: str) -> tuple[str, ...]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        return SUPPORTED_FORMATS
    valid = [item for item in items if item in SUPPORTED_FORMATS]
    if not valid:
        invalid = ", ".join(items)
        supported = ", ".join(SUPPORTED_FORMATS)
        raise SystemExit(
            f"--formats had no valid entries (got: {invalid}). "
            f"Supported formats: {supported}."
        )
    return tuple(valid)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    pool = await _create_pool(args.database_url)
    try:
        result = await repurpose_podcast_episode_from_postgres(
            pool,
            scope={"account_id": args.account_id, "user_id": args.user_id},
            episode_id=args.episode_id,
            formats=_resolve_formats(args.formats),
            idea_limit=args.idea_limit,
            voice_anchors=_load_voice_anchors(args.voice_anchors),
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
