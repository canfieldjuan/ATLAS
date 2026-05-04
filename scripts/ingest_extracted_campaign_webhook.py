#!/usr/bin/env python3
"""Ingest extracted campaign ESP webhooks into Postgres."""

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

from extracted_content_pipeline.campaign_postgres_webhooks import (  # noqa: E402
    ingest_resend_webhook_from_postgres,
)
from extracted_content_pipeline.campaign_webhooks import (  # noqa: E402
    CampaignWebhookIngestionConfig,
)


DATABASE_URL_ENV = ("EXTRACTED_DATABASE_URL", "DATABASE_URL")
RESEND_WEBHOOK_SECRET_ENV = (
    "EXTRACTED_RESEND_WEBHOOK_SECRET",
    "EXTRACTED_CAMPAIGN_RESEND_WEBHOOK_SECRET",
)
WEBHOOK_PROVIDERS = ("resend",)


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {raw!r}") from exc


def _parse_header_pair(value: str) -> tuple[str, str]:
    name, sep, header_value = str(value or "").partition(":")
    name = name.strip()
    if not sep or not name:
        raise argparse.ArgumentTypeError(
            "Headers must use 'Name: value' format"
        )
    return name, header_value.strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = CampaignWebhookIngestionConfig()
    parser = argparse.ArgumentParser(
        description="Ingest campaign ESP webhook payloads into the extracted product database."
    )
    parser.add_argument(
        "--database-url",
        default=_env(*DATABASE_URL_ENV),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--provider",
        choices=WEBHOOK_PROVIDERS,
        default="resend",
        help="Webhook provider to ingest.",
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        help="Path to the raw webhook body. Defaults to stdin.",
    )
    parser.add_argument(
        "--headers-json",
        type=Path,
        help="Path to a JSON object containing webhook request headers.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        type=_parse_header_pair,
        help="Additional webhook header in 'Name: value' format. May repeat.",
    )
    parser.add_argument(
        "--signing-secret",
        default=_env(*RESEND_WEBHOOK_SECRET_ENV),
        help=(
            "Resend webhook signing secret. Defaults to "
            "EXTRACTED_RESEND_WEBHOOK_SECRET or "
            "EXTRACTED_CAMPAIGN_RESEND_WEBHOOK_SECRET."
        ),
    )
    parser.add_argument(
        "--skip-signature-verification",
        action="store_true",
        help="Skip webhook signature verification for trusted local replays.",
    )
    parser.add_argument(
        "--record-unknown-events",
        action="store_true",
        help="Record provider events the extracted product does not handle yet.",
    )
    parser.add_argument(
        "--soft-bounce-suppression-days",
        type=int,
        default=_env_int(
            "EXTRACTED_CAMPAIGN_SOFT_BOUNCE_SUPPRESSION_DAYS",
            defaults.soft_bounce_suppression_days,
        ),
        help="Temporary suppression length for soft bounces.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of a concise text summary.",
    )
    return parser.parse_args(argv)


def _read_body(path: Path | None) -> bytes:
    if path is None:
        return sys.stdin.buffer.read()
    return path.read_bytes()


def _read_headers(
    headers_json: Path | None,
    header_pairs: list[tuple[str, str]],
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if headers_json is not None:
        try:
            parsed = json.loads(headers_json.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --headers-json file: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("--headers-json must contain a JSON object")
        headers.update({str(key): str(value) for key, value in parsed.items()})
    for name, value in header_pairs:
        headers[name] = value
    return headers


def _validate_args(args: argparse.Namespace) -> None:
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    if args.provider != "resend":
        raise SystemExit(f"Unsupported --provider: {args.provider!r}")
    if args.soft_bounce_suppression_days <= 0:
        raise SystemExit("--soft-bounce-suppression-days must be positive")
    if (
        not args.skip_signature_verification
        and not str(args.signing_secret or "").strip()
    ):
        raise SystemExit(
            "Missing --signing-secret or EXTRACTED_RESEND_WEBHOOK_SECRET; "
            "use --skip-signature-verification only for trusted local replays"
        )


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to ingest campaign webhooks; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    _validate_args(args)
    body = _read_body(args.body_file)
    headers = _read_headers(args.headers_json, args.header)
    pool = await _create_pool(args.database_url)
    try:
        result = await ingest_resend_webhook_from_postgres(
            pool,
            body=body,
            headers=headers,
            signing_secret=args.signing_secret or "",
            verify_signatures=not args.skip_signature_verification,
            config=CampaignWebhookIngestionConfig(
                soft_bounce_suppression_days=args.soft_bounce_suppression_days,
                record_unknown_events=args.record_unknown_events,
            ),
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

    summary = result.as_dict()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        text = "status={status}".format(**summary)
        if summary.get("event_type"):
            text = f"{text} event_type={summary['event_type']}"
        if summary.get("message_id"):
            text = f"{text} message_id={summary['message_id']}"
        if summary.get("reason"):
            text = f"{text} reason={summary['reason']}"
        text = f"{text} suppressed={str(summary.get('suppressed')).lower()}"
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
