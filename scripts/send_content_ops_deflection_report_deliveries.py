#!/usr/bin/env python3
"""Send pending paid FAQ deflection report delivery emails."""

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

from atlas_brain.content_ops_deflection_delivery import (  # noqa: E402
    DEFAULT_DEFLECTION_DELIVERY_SUBJECT,
    DeflectionReportDeliveryConfig,
    send_pending_deflection_report_deliveries,
)
from extracted_content_pipeline.campaign_ports import SendRequest, SendResult  # noqa: E402
from extracted_content_pipeline.campaign_sender import (  # noqa: E402
    DEFAULT_RESEND_TIMEOUT_SECONDS,
    RESEND_API_URL,
    create_campaign_sender,
)


DATABASE_URL_ENV = ("EXTRACTED_DATABASE_URL", "DATABASE_URL")
FROM_EMAIL_ENV = (
    "ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_FROM_EMAIL",
    "EXTRACTED_RESEND_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_RESEND_FROM_EMAIL",
)
RESEND_API_KEY_ENV = (
    "ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY",
    "EXTRACTED_RESEND_API_KEY",
    "EXTRACTED_CAMPAIGN_RESEND_API_KEY",
    "EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY",
)


class _DryRunSender:
    async def send(self, _request: SendRequest) -> SendResult:
        raise RuntimeError("dry-run sender should not be called")


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid float for {name}: {raw!r}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send pending paid FAQ deflection report delivery emails."
    )
    parser.add_argument(
        "--database-url",
        default=_env(*DATABASE_URL_ENV),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int("ATLAS_DEFLECTION_DELIVERY_LIMIT", 20),
        help="Maximum pending delivery rows to process.",
    )
    parser.add_argument(
        "--from-email",
        default=_env(*FROM_EMAIL_ENV),
        help="Transactional From email.",
    )
    parser.add_argument(
        "--reply-to",
        default=_env("ATLAS_DEFLECTION_DELIVERY_REPLY_TO"),
        help="Optional Reply-To email.",
    )
    parser.add_argument(
        "--result-base-url",
        default=_env("ATLAS_DEFLECTION_PORTFOLIO_BASE_URL"),
        help="Portfolio origin; production result path is appended.",
    )
    parser.add_argument(
        "--result-url-template",
        default=_env("ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL_TEMPLATE"),
        help="Full result URL template containing {request_id}.",
    )
    parser.add_argument(
        "--subject",
        default=_env(
            "ATLAS_DEFLECTION_DELIVERY_SUBJECT",
            default=DEFAULT_DEFLECTION_DELIVERY_SUBJECT,
        ),
        help="Email subject line.",
    )
    parser.add_argument(
        "--resend-api-key",
        default=_env(*RESEND_API_KEY_ENV),
        help="Resend API key. Required only with --send.",
    )
    parser.add_argument(
        "--resend-api-url",
        default=_env("ATLAS_DEFLECTION_DELIVERY_RESEND_API_URL", default=RESEND_API_URL),
        help="Resend email API URL.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_env_float(
            "ATLAS_DEFLECTION_DELIVERY_RESEND_TIMEOUT_SECONDS",
            DEFAULT_RESEND_TIMEOUT_SECONDS,
        ),
        help="HTTP timeout for Resend calls.",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Perform live sends. Omit for dry-run mode.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not _configured(args.database_url):
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    if not _configured(args.from_email):
        raise SystemExit("Missing --from-email or ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL")
    if not _configured(args.result_base_url) and not _configured(args.result_url_template):
        raise SystemExit(
            "Missing --result-base-url, --result-url-template, "
            "ATLAS_DEFLECTION_PORTFOLIO_BASE_URL, or "
            "ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL_TEMPLATE"
        )
    if int(args.limit) <= 0:
        raise SystemExit("Invalid --limit: must be greater than 0")
    if args.send and not _configured(args.resend_api_key):
        raise SystemExit("Missing --resend-api-key or ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY")


def _delivery_config(args: argparse.Namespace) -> DeflectionReportDeliveryConfig:
    return DeflectionReportDeliveryConfig(
        from_email=str(args.from_email or ""),
        result_base_url=str(args.result_base_url or ""),
        result_url_template=str(args.result_url_template or ""),
        reply_to=args.reply_to,
        subject=str(args.subject or DEFAULT_DEFLECTION_DELIVERY_SUBJECT),
        limit=int(args.limit),
        dry_run=not bool(args.send),
    )


def _sender(args: argparse.Namespace) -> Any:
    if not args.send:
        return _DryRunSender()
    return create_campaign_sender(
        "resend",
        {
            "api_key": args.resend_api_key,
            "api_url": args.resend_api_url,
            "timeout_seconds": args.timeout_seconds,
        },
    )


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to send deflection report deliveries"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


def _summary_payload(summary: Any) -> dict[str, int]:
    return {
        "scanned": int(summary.scanned),
        "sent": int(summary.sent),
        "failed": int(summary.failed),
        "dry_run": int(summary.dry_run),
    }


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    pool = await _create_pool(str(args.database_url))
    try:
        summary = await send_pending_deflection_report_deliveries(
            pool,
            sender=_sender(args),
            config=_delivery_config(args),
        )
    finally:
        await pool.close()

    payload = _summary_payload(summary)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        mode = "send" if args.send else "dry-run"
        print(
            "FAQ deflection deliveries "
            f"mode={mode} scanned={summary.scanned} sent={summary.sent} "
            f"failed={summary.failed} dry_run={summary.dry_run}"
        )
    return 0 if summary.failed == 0 else 1


def _configured(value: Any) -> bool:
    return bool(str(value or "").strip())


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
