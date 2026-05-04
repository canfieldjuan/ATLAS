#!/usr/bin/env python3
"""Send queued extracted campaign emails from Postgres."""

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

from extracted_content_pipeline.campaign_postgres_send import (  # noqa: E402
    send_due_campaigns_from_postgres,
)
from extracted_content_pipeline.campaign_send import CampaignSendConfig  # noqa: E402
from extracted_content_pipeline.campaign_sender import (  # noqa: E402
    DEFAULT_RESEND_TIMEOUT_SECONDS,
    DEFAULT_SES_REGION,
    RESEND_API_URL,
    create_campaign_sender,
)


DEFAULT_CAMPAIGN_SENDER_PROVIDER = "resend"
DATABASE_URL_ENV = ("EXTRACTED_DATABASE_URL", "DATABASE_URL")
SENDER_PROVIDER_ENV = (
    "EXTRACTED_CAMPAIGN_SENDER_TYPE",
    "EXTRACTED_CAMPAIGN_SEQUENCE_SENDER_TYPE",
)
FROM_EMAIL_ENV = (
    "EXTRACTED_CAMPAIGN_FROM_EMAIL",
    "EXTRACTED_RESEND_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_RESEND_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_SEQ_RESEND_FROM_EMAIL",
    "EXTRACTED_SES_FROM_EMAIL",
)
RESEND_API_KEY_ENV = (
    "EXTRACTED_RESEND_API_KEY",
    "EXTRACTED_CAMPAIGN_RESEND_API_KEY",
    "EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY",
)
SENDER_TIMEOUT_ENV = (
    "EXTRACTED_CAMPAIGN_SENDER_TIMEOUT_SECONDS",
    "EXTRACTED_CAMPAIGN_SEQ_SENDER_TIMEOUT_SECONDS",
)


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    send_defaults = CampaignSendConfig()
    parser = argparse.ArgumentParser(
        description="Send queued campaign emails from the extracted product database."
    )
    parser.add_argument(
        "--database-url",
        default=_env(*DATABASE_URL_ENV),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--provider",
        choices=("resend", "ses"),
        default=_env(
            *SENDER_PROVIDER_ENV,
            default=DEFAULT_CAMPAIGN_SENDER_PROVIDER,
        ),
        help="Email provider to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_env_int(("EXTRACTED_CAMPAIGN_SEND_LIMIT",), send_defaults.limit),
        help="Maximum queued campaigns to evaluate.",
    )
    parser.add_argument(
        "--default-from-email",
        default=_env(
            *FROM_EMAIL_ENV,
            default=send_defaults.default_from_email,
        ),
        help="Fallback From email when a queued row has no from_email.",
    )
    parser.add_argument(
        "--reply-to",
        default=_env("EXTRACTED_CAMPAIGN_REPLY_TO", default=send_defaults.default_reply_to),
        help="Optional Reply-To email.",
    )
    parser.add_argument(
        "--unsubscribe-base-url",
        default=_env(
            "EXTRACTED_CAMPAIGN_UNSUBSCRIBE_BASE_URL",
            default=send_defaults.unsubscribe_base_url,
        ),
        help="Base URL used for List-Unsubscribe headers and footer links.",
    )
    parser.add_argument(
        "--company-address",
        default=_env("EXTRACTED_CAMPAIGN_COMPANY_ADDRESS", default=send_defaults.company_address),
        help="Physical address appended to outbound email footers.",
    )
    parser.add_argument(
        "--resend-api-key",
        default=_env(
            *RESEND_API_KEY_ENV,
        ),
        help="Resend API key.",
    )
    parser.add_argument(
        "--resend-api-url",
        default=_env("EXTRACTED_RESEND_API_URL", default=RESEND_API_URL),
        help="Resend email API URL.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_env_float(
            SENDER_TIMEOUT_ENV,
            DEFAULT_RESEND_TIMEOUT_SECONDS,
        ),
        help="HTTP timeout for provider calls.",
    )
    parser.add_argument(
        "--ses-from-email",
        default=_env("EXTRACTED_SES_FROM_EMAIL"),
        help="SES verified sender email.",
    )
    parser.add_argument(
        "--ses-region",
        default=_env("EXTRACTED_SES_REGION", default=DEFAULT_SES_REGION),
        help="AWS SES region.",
    )
    parser.add_argument(
        "--ses-access-key-id",
        default=_env("EXTRACTED_SES_ACCESS_KEY_ID"),
        help="Optional AWS access key id.",
    )
    parser.add_argument(
        "--ses-secret-access-key",
        default=_env("EXTRACTED_SES_SECRET_ACCESS_KEY"),
        help="Optional AWS secret access key.",
    )
    parser.add_argument(
        "--ses-configuration-set",
        default=_env("EXTRACTED_SES_CONFIGURATION_SET"),
        help="Optional SES configuration set.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of a concise text summary.",
    )
    return parser.parse_args(argv)


def _sender_config(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    provider = str(args.provider or "").strip().lower()
    if provider == "ses":
        return (
            "ses",
            {
                "from_email": args.ses_from_email or args.default_from_email,
                "region": args.ses_region,
                "access_key_id": args.ses_access_key_id,
                "secret_access_key": args.ses_secret_access_key,
                "configuration_set": args.ses_configuration_set,
            },
        )
    return (
        "resend",
        {
            "api_key": args.resend_api_key,
            "api_url": args.resend_api_url,
            "timeout_seconds": args.timeout_seconds,
        },
    )


def _configured(value: Any) -> bool:
    return bool(str(value or "").strip())


def _validate_sender_config(provider: str, config: dict[str, Any]) -> None:
    if provider == "ses":
        if not _configured(config.get("from_email")):
            raise SystemExit(
                "Missing --ses-from-email, --default-from-email, "
                "EXTRACTED_SES_FROM_EMAIL, or EXTRACTED_CAMPAIGN_FROM_EMAIL"
            )
        return
    if not _configured(config.get("api_key")):
        raise SystemExit(
            "Missing --resend-api-key, EXTRACTED_RESEND_API_KEY, "
            "EXTRACTED_CAMPAIGN_RESEND_API_KEY, or "
            "EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY"
        )


def _validate_send_args(args: argparse.Namespace) -> None:
    if int(args.limit) <= 0:
        raise SystemExit("Invalid --limit: must be greater than 0")
    default_from_email = args.default_from_email or args.ses_from_email
    if not _configured(default_from_email):
        raise SystemExit(
            "Missing --default-from-email, EXTRACTED_CAMPAIGN_FROM_EMAIL, "
            "or a provider-specific From email"
        )


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to send campaigns; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    provider, provider_config = _sender_config(args)
    _validate_sender_config(provider, provider_config)
    _validate_send_args(args)
    sender = create_campaign_sender(provider, provider_config)
    config = CampaignSendConfig(
        default_from_email=args.default_from_email or args.ses_from_email or "",
        default_reply_to=args.reply_to,
        unsubscribe_base_url=args.unsubscribe_base_url,
        company_address=args.company_address,
        limit=args.limit,
    )
    pool = await _create_pool(args.database_url)
    try:
        summary = await send_due_campaigns_from_postgres(
            pool,
            sender=sender,
            config=config,
            limit=args.limit,
        )
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

    if args.json:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    else:
        print(
            "sent={sent} failed={failed} suppressed={suppressed} skipped={skipped}".format(
                **summary.as_dict()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
