#!/usr/bin/env python3
"""Generate queued follow-up steps for due extracted campaign sequences."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_example import (  # noqa: E402
    DeterministicCampaignLLM,
)
from extracted_content_pipeline.campaign_llm_client import (  # noqa: E402
    create_pipeline_llm_client,
)
from extracted_content_pipeline.campaign_postgres_sequence_progression import (  # noqa: E402
    progress_campaign_sequences_from_postgres,
)
from extracted_content_pipeline.campaign_sequence_progression import (  # noqa: E402
    CampaignSequenceProgressionConfig,
)
from extracted_content_pipeline.campaign_visibility import (  # noqa: E402
    JsonlVisibilitySink,
    OPERATION_COMPLETED_EVENT,
    OPERATION_FAILED_EVENT,
    OPERATION_STARTED_EVENT,
    emit_operation_event,
    visibility_result_summary,
)
from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: E402


DATABASE_URL_ENV = ("EXTRACTED_DATABASE_URL", "DATABASE_URL")
FROM_EMAIL_ENV = (
    "EXTRACTED_CAMPAIGN_SEQUENCE_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_FROM_EMAIL",
)
LLM_MODES = ("pipeline", "offline")


def _env(*names: str, default: str | None = None) -> str | None:
    return _env_match(*names, default=default)[1]


def _env_match(*names: str, default: str | None = None) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return name, value
    return None, default


def _env_int(names: tuple[str, ...], default: int) -> int:
    name, raw = _env_match(*names)
    if raw in (None, ""):
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        source = name or "default"
        raise SystemExit(f"Invalid integer for {source}: {raw!r}") from exc


def _env_float(names: tuple[str, ...], default: float) -> float:
    name, raw = _env_match(*names)
    if raw in (None, ""):
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        source = name or "default"
        raise SystemExit(f"Invalid float for {source}: {raw!r}") from exc


def _llm_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    if mode not in LLM_MODES:
        choices = ", ".join(LLM_MODES)
        raise SystemExit(f"Invalid --llm: {value!r}; expected one of {choices}")
    return mode


def _llm_mode_arg(value: str) -> str:
    try:
        return _llm_mode(value)
    except SystemExit as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = CampaignSequenceProgressionConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Generate queued follow-up campaign steps for due sequence rows "
            "in the extracted product database."
        )
    )
    parser.add_argument(
        "--database-url",
        default=_env(*DATABASE_URL_ENV),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int(("EXTRACTED_CAMPAIGN_SEQUENCE_LIMIT",), defaults.batch_limit),
        help="Maximum due sequences to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_env_int(
            ("EXTRACTED_CAMPAIGN_SEQUENCE_MAX_STEPS",),
            defaults.max_steps,
        ),
        help="Fallback maximum steps per sequence.",
    )
    parser.add_argument(
        "--from-email",
        default=_env(*FROM_EMAIL_ENV, default=defaults.from_email),
        help="From email stamped on queued follow-up rows.",
    )
    parser.add_argument(
        "--onboarding-product-name",
        default=_env(
            "EXTRACTED_CAMPAIGN_ONBOARDING_PRODUCT_NAME",
            default=defaults.onboarding_product_name,
        ),
        help="Fallback product name for onboarding sequences.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_env_float(
            ("EXTRACTED_CAMPAIGN_SEQUENCE_TEMPERATURE",),
            defaults.temperature,
        ),
        help="LLM temperature for follow-up generation.",
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        help="Optional directory of host-provided markdown sequence prompts.",
    )
    parser.add_argument(
        "--llm",
        choices=LLM_MODES,
        type=_llm_mode_arg,
        default=_llm_mode(_env("EXTRACTED_CAMPAIGN_SEQUENCE_LLM", default="pipeline")),
        help="Use configured PipelineLLMClient or deterministic offline LLM.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of a concise text summary.",
    )
    parser.add_argument(
        "--visibility-jsonl",
        type=Path,
        help="Append campaign operation visibility events to this JSONL file.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.limit) <= 0:
        raise SystemExit("Invalid --limit: must be greater than 0")
    if int(args.max_steps) <= 0:
        raise SystemExit("Invalid --max-steps: must be greater than 0")
    if not str(args.from_email or "").strip():
        raise SystemExit(
            "Missing --from-email, EXTRACTED_CAMPAIGN_SEQUENCE_FROM_EMAIL, "
            "or EXTRACTED_CAMPAIGN_FROM_EMAIL"
        )


def _config_from_args(args: argparse.Namespace) -> CampaignSequenceProgressionConfig:
    return CampaignSequenceProgressionConfig(
        batch_limit=int(args.limit),
        max_steps=int(args.max_steps),
        from_email=str(args.from_email or "").strip(),
        onboarding_product_name=str(args.onboarding_product_name or ""),
        temperature=float(args.temperature),
    )


def _llm_from_args(args: argparse.Namespace):
    if args.llm == "offline":
        return DeterministicCampaignLLM()
    return create_pipeline_llm_client()


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to progress campaign sequences; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    _validate_args(args)
    visibility = JsonlVisibilitySink(args.visibility_jsonl) if args.visibility_jsonl else None
    operation_payload = {"limit": args.limit, "max_steps": args.max_steps}
    await emit_operation_event(
        visibility,
        OPERATION_STARTED_EVENT,
        "sequence_progression",
        operation_payload,
    )
    pool = None
    try:
        pool = await _create_pool(args.database_url)
        result = await progress_campaign_sequences_from_postgres(
            pool,
            llm=_llm_from_args(args),
            skills=get_skill_registry(root=args.skills_root),
            config=_config_from_args(args),
        )
    except Exception as exc:
        await emit_operation_event(
            visibility,
            OPERATION_FAILED_EVENT,
            "sequence_progression",
            {**operation_payload, "error_type": type(exc).__name__},
        )
        raise
    finally:
        if pool is not None and (close := getattr(pool, "close", None)) is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

    summary = result.as_dict()
    await emit_operation_event(
        visibility,
        OPERATION_COMPLETED_EVENT,
        "sequence_progression",
        {**operation_payload, "result": visibility_result_summary(summary)},
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "due={due_sequences} progressed={progressed} skipped={skipped} "
            "disabled={disabled}".format(**summary)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
