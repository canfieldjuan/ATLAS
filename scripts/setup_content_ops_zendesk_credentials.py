#!/usr/bin/env python3
"""Provision tenant-scoped Zendesk credentials for Content Ops macro writeback."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable
import uuid as _uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SKIPPED_EXIT = 2
TOKEN_PREFIX_LEN = 8


PoolFactory = Callable[[str], Awaitable[Any]]
CredentialProvider = Callable[[], Any | None]
UpsertFunc = Callable[..., Awaitable[Any]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default="",
        help=(
            "Postgres DSN. Required for writes; not required with --dry-run. "
            "This script does not read database credentials from env vars."
        ),
    )
    parser.add_argument("--account-id", required=True, help="Tenant/account UUID.")
    parser.add_argument(
        "--label",
        default="Content Ops Zendesk",
        help="Display label for the stored Zendesk credential row.",
    )
    parser.add_argument(
        "--expected-zendesk-base-url",
        default="",
        help="Optional exact Zendesk base URL guard, for example https://acme.zendesk.com.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and guards without opening a database pool or writing.",
    )
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required to write encrypted tenant Zendesk credentials.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Accepted for operator consistency; output is always JSON.",
    )
    return parser.parse_args(argv)


async def run_setup(
    args: argparse.Namespace,
    *,
    pool_factory: PoolFactory | None = None,
    credentials_provider: CredentialProvider | None = None,
    upsert_func: UpsertFunc | None = None,
) -> tuple[int, dict[str, Any]]:
    """Validate and optionally store one tenant Zendesk credential."""

    try:
        account_id = _uuid.UUID(str(args.account_id))
    except (TypeError, ValueError):
        return _failure("invalid_account_id", "Account id must be a UUID.")

    credentials = (credentials_provider or _credentials_from_settings)()
    if credentials is None or not credentials.is_complete():
        return _failure(
            "zendesk_credentials_incomplete",
            "Zendesk credentials are incomplete in Atlas config.",
            account_id=str(account_id),
        )

    zendesk_base_url = credentials.normalized_base_url().rstrip("/")
    expected_url = _clean(getattr(args, "expected_zendesk_base_url", "")).rstrip("/")
    if expected_url and zendesk_base_url != expected_url:
        return _failure(
            "unexpected_zendesk_base_url",
            "Configured Zendesk endpoint did not match the expected guard.",
            account_id=str(account_id),
            expected_zendesk_base_url=expected_url,
            observed_zendesk_base_url=zendesk_base_url,
        )

    label = _clean(getattr(args, "label", ""))
    preview = _display_payload(
        account_id=str(account_id),
        zendesk_base_url=zendesk_base_url,
        label=label,
        api_token_prefix=_token_prefix(credentials.api_token),
    )
    if bool(getattr(args, "dry_run", False)):
        return 0, {
            "ok": True,
            "dry_run": True,
            "would_write": True,
            **preview,
        }

    if not bool(getattr(args, "confirm_write", False)):
        return SKIPPED_EXIT, {
            "ok": False,
            "skipped": True,
            "not_run_reason": "missing_confirm_write",
            **preview,
        }

    database_url = _clean(getattr(args, "database_url", ""))
    if not database_url:
        return _failure(
            "database_url_missing",
            "Database URL is required for writes.",
            **preview,
        )

    try:
        pool = await (pool_factory or _create_pool)(database_url)
    except Exception:
        return _failure(
            "database_pool_unavailable",
            "Database pool could not be opened.",
            **preview,
        )

    close_warning = ""
    try:
        upsert = upsert_func or _upsert_zendesk_credentials
        record = await upsert(
            pool,
            account_id=account_id,
            email=credentials.email,
            api_token=credentials.api_token,
            subdomain=credentials.subdomain,
            base_url=credentials.base_url,
            label=label,
        )
    except ValueError as exc:
        close_warning = await _close_warning(pool)
        _code, payload = (
            _failure(
                "credential_encryption_unavailable",
                "Credential encryption is not configured correctly.",
                **preview,
            )
            if _is_encryption_setup_error(exc)
            else _failure(
                "zendesk_credentials_invalid",
                "Zendesk credentials are invalid.",
                **preview,
            )
        )
        return 1, _with_optional_warning(payload, close_warning)
    except Exception:
        close_warning = await _close_warning(pool)
        _code, payload = _failure(
            "zendesk_credentials_write_failed",
            "Zendesk credential write failed.",
            **preview,
        )
        return 1, _with_optional_warning(payload, close_warning)
    close_warning = await _close_warning(pool)

    payload = {
        "ok": True,
        "dry_run": False,
        "credential_id": str(record.id),
        "account_id": str(record.account_id),
        "zendesk_base_url": str(record.base_url or zendesk_base_url),
        "subdomain": str(record.subdomain or credentials.subdomain),
        "label": str(record.label or label),
        "api_token_prefix": str(record.api_token_prefix or ""),
    }
    return 0, _with_optional_warning(payload, close_warning)


def _credentials_from_settings() -> Any | None:
    from atlas_brain._content_ops_macro_writeback import (
        zendesk_macro_credentials_from_config,
    )
    from atlas_brain.config import settings

    return zendesk_macro_credentials_from_config(settings.b2b_campaign)


async def _upsert_zendesk_credentials(pool: Any, **kwargs: Any) -> Any:
    from atlas_brain._content_ops_zendesk_credentials import upsert_zendesk_credentials

    return await upsert_zendesk_credentials(pool, **kwargs)


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for Zendesk credential setup; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _close_warning(pool: Any) -> str:
    close = getattr(pool, "close", None)
    if close is None:
        return ""
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception:
        return "database_pool_close_failed"
    return ""


def _with_optional_warning(payload: dict[str, Any], warning: str) -> dict[str, Any]:
    if not warning:
        return payload
    return {**payload, "warnings": [warning]}


def _display_payload(
    *,
    account_id: str,
    zendesk_base_url: str,
    label: str,
    api_token_prefix: str,
) -> dict[str, str]:
    return {
        "account_id": account_id,
        "zendesk_base_url": zendesk_base_url,
        "label": label,
        "api_token_prefix": api_token_prefix,
    }


def _failure(reason: str, message: str, **extra: Any) -> tuple[int, dict[str, Any]]:
    payload = {
        "ok": False,
        "reason": reason,
        "message": message,
    }
    payload.update(extra)
    return 1, payload


def _token_prefix(token: str) -> str:
    cleaned = str(token or "").strip()
    if len(cleaned) <= TOKEN_PREFIX_LEN:
        return ""
    return cleaned[:TOKEN_PREFIX_LEN]


def _is_encryption_setup_error(exc: ValueError) -> bool:
    message = str(exc)
    return (
        "ATLAS_SAAS_BYOK_ENCRYPTION_KEK" in message
        or "BYOK KEK" in message
        or "encrypt_secret" in message
    )


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _write_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    code, payload = await run_setup(args)
    _write_payload(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
