#!/usr/bin/env python3
"""Guarded live smoke for publishing one approved FAQ draft to Zendesk macros."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain._content_ops_macro_writeback import (  # noqa: E402
    ConfigZendeskMacroCredentialsProvider,
    TenantZendeskMacroCredentialsProvider,
)
from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.faq_macro_writeback import (  # noqa: E402
    build_macro_writeback_preview,
)
from extracted_content_pipeline.faq_macro_writeback_postgres import (  # noqa: E402
    PostgresFAQMacroPublishAttemptRepository,
    PostgresFAQMacroWritebackMappingRepository,
)
from extracted_content_pipeline.faq_macro_writeback_publish import (  # noqa: E402
    FAQMacroWritebackPublishService,
)
from extracted_content_pipeline.faq_macro_writeback_zendesk import (  # noqa: E402
    StaticZendeskMacroCredentialsProvider,
    ZendeskMacroCredentialsProvider,
    ZendeskMacroPublishProvider,
)
from extracted_content_pipeline.ticket_faq_postgres import (  # noqa: E402
    PostgresTicketFAQRepository,
)


SKIPPED_EXIT = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        required=True,
        help="Postgres DSN. Required explicitly; this script does not read env vars.",
    )
    parser.add_argument("--account-id", required=True, help="Tenant/account id.")
    parser.add_argument("--faq-id", required=True, help="Existing FAQ draft id to publish.")
    parser.add_argument("--user-id", default="", help="Optional operator user id.")
    parser.add_argument(
        "--expected-zendesk-base-url",
        default="",
        help="Optional exact Zendesk base URL guard, for example https://acme.zendesk.com.",
    )
    parser.add_argument(
        "--confirm-live-zendesk-write",
        action="store_true",
        help="Required to allow the script to create or update real Zendesk macros.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Accepted for operator consistency; output is always JSON.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not _clean(args.database_url):
        raise SystemExit("--database-url is required")
    if not _clean(args.account_id):
        raise SystemExit("--account-id is required")
    if not _clean(args.faq_id):
        raise SystemExit("--faq-id is required")


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for the live Zendesk smoke; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def run_live_zendesk_smoke(
    args: argparse.Namespace,
    pool: Any,
    *,
    credentials_provider: ZendeskMacroCredentialsProvider | None = None,
    faq_repository: Any | None = None,
    provider: Any | None = None,
    attempt_repository: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the guarded smoke, returning an exit code and JSON payload."""

    scope = TenantScope(
        account_id=_clean(args.account_id),
        user_id=_clean(getattr(args, "user_id", "")) or None,
    )
    faq_id = _clean(args.faq_id)
    if not getattr(args, "confirm_live_zendesk_write", False):
        return _not_run(
            "missing_confirm_live_zendesk_write",
            account_id=scope.account_id or "",
            faq_id=faq_id,
        )

    credentials_source = credentials_provider or _build_credentials_provider(pool)
    credentials = await credentials_source.credentials_for_scope(scope)
    if credentials is None or not credentials.is_complete():
        return _not_run(
            "zendesk_credentials_missing",
            account_id=scope.account_id or "",
            faq_id=faq_id,
        )

    zendesk_base_url = credentials.normalized_base_url().rstrip("/")
    expected_url = _clean(getattr(args, "expected_zendesk_base_url", "")).rstrip("/")
    if expected_url and zendesk_base_url != expected_url:
        return _not_run(
            "unexpected_zendesk_base_url",
            account_id=scope.account_id or "",
            faq_id=faq_id,
            expected_zendesk_base_url=expected_url,
            observed_zendesk_base_url=zendesk_base_url,
        )

    faq_repo = faq_repository or PostgresTicketFAQRepository(pool)
    draft = await faq_repo.get_draft(faq_id, scope=scope)
    if draft is None:
        return _not_run(
            "faq_draft_not_found",
            account_id=scope.account_id or "",
            faq_id=faq_id,
            zendesk_base_url=zendesk_base_url,
        )

    preview = build_macro_writeback_preview((draft,))
    if preview.publishable_count < 1:
        return _not_run(
            "no_publishable_macros",
            account_id=scope.account_id or "",
            faq_id=faq_id,
            zendesk_base_url=zendesk_base_url,
            draft_status=_clean(getattr(draft, "status", "")),
            skipped=[dict(item) for item in _as_dicts(preview.skipped)],
        )

    publish_provider = provider or ZendeskMacroPublishProvider(
        credentials_provider=StaticZendeskMacroCredentialsProvider(credentials),
        mapping_repository=PostgresFAQMacroWritebackMappingRepository(pool),
    )
    service = FAQMacroWritebackPublishService(
        faq_repository=faq_repo,
        provider=publish_provider,
        attempt_repository=attempt_repository
        if attempt_repository is not None
        else PostgresFAQMacroPublishAttemptRepository(pool),
    )
    summary = await service.publish_faq_draft(faq_id, scope=scope)
    errors = _summary_errors(summary)
    ok = summary.ok and not errors
    return (
        0 if ok else 1,
        {
            "ok": ok,
            "skipped": False,
            "account_id": scope.account_id or "",
            "faq_id": faq_id,
            "zendesk_base_url": zendesk_base_url,
            "publishable_count": preview.publishable_count,
            "summary": summary.as_dict(),
            "errors": errors,
        },
    )


def _build_credentials_provider(pool: Any) -> TenantZendeskMacroCredentialsProvider:
    return TenantZendeskMacroCredentialsProvider(
        pool=pool,
        fallback_provider=ConfigZendeskMacroCredentialsProvider(_host_config()),
    )


def _host_config() -> Any:
    from atlas_brain.config import settings

    return settings.b2b_campaign


def _summary_errors(summary: Any) -> list[str]:
    errors: list[str] = []
    if not summary.found:
        errors.append("faq_draft_not_found")
    if summary.failed_count:
        errors.append("macro_publish_failed")
    if summary.pending_reconcile_count:
        errors.append("macro_publish_pending_reconcile")
    if summary.skipped_count:
        errors.append("macro_publish_skipped_items")
    return errors


def _not_run(reason: str, **extra: Any) -> tuple[int, dict[str, Any]]:
    payload = {
        "ok": False,
        "skipped": True,
        "not_run_reason": reason,
    }
    payload.update(extra)
    return SKIPPED_EXIT, payload


def _as_dicts(items: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in items:
        as_dict = getattr(item, "as_dict", None)
        rows.append(dict(as_dict() if callable(as_dict) else item))
    return tuple(rows)


def _write_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not args.confirm_live_zendesk_write:
        code, payload = _not_run(
            "missing_confirm_live_zendesk_write",
            account_id=_clean(args.account_id),
            faq_id=_clean(args.faq_id),
        )
        _write_payload(payload, args)
        return code

    pool = await _create_pool(_clean(args.database_url))
    try:
        code, payload = await run_live_zendesk_smoke(args, pool)
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    _write_payload(payload, args)
    return code


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
