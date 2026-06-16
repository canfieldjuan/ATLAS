#!/usr/bin/env python3
"""Clean up one sandbox FAQ macro writeback proof by explicit FAQ id."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain._content_ops_macro_writeback import (  # noqa: E402
    ConfigZendeskMacroCredentialsProvider,
    TenantZendeskMacroCredentialsProvider,
)
from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.faq_macro_writeback_zendesk import (  # noqa: E402
    ZENDESK_PLATFORM,
    ZendeskHTTPMacroTransport,
    ZendeskMacroCredentialsProvider,
    ZendeskMacroTransport,
    ZendeskMacroTransportError,
)


SKIPPED_EXIT = 2


@dataclass(frozen=True)
class CleanupSnapshot:
    """Preview of local rows tied to one FAQ macro writeback proof."""

    found: bool
    draft_status: str = ""
    macro_mappings: tuple[dict[str, Any], ...] = ()
    publish_attempt_count: int = 0
    search_document_count: int = 0

    @property
    def external_ids(self) -> tuple[str, ...]:
        return tuple(
            mapping["external_id"]
            for mapping in self.macro_mappings
            if _clean(mapping.get("external_id"))
        )

    @property
    def missing_external_id_count(self) -> int:
        return sum(
            1
            for mapping in self.macro_mappings
            if not _clean(mapping.get("external_id"))
        )

    def unexpected_external_urls(
        self,
        *,
        expected_base_url: str,
    ) -> tuple[dict[str, str], ...]:
        expected = _url_base(expected_base_url)
        unexpected: list[dict[str, str]] = []
        for mapping in self.macro_mappings:
            external_url = _clean(mapping.get("external_url"))
            if external_url and _url_base(external_url) != expected:
                unexpected.append({
                    "external_id": _clean(mapping.get("external_id")),
                    "external_url": external_url,
                })
        return tuple(unexpected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "draft_status": self.draft_status,
            "macro_mapping_count": len(self.macro_mappings),
            "external_ids": list(self.external_ids),
            "missing_external_id_count": self.missing_external_id_count,
            "publish_attempt_count": self.publish_attempt_count,
            "search_document_count": self.search_document_count,
            "macro_mappings": [dict(mapping) for mapping in self.macro_mappings],
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        required=True,
        help="Postgres DSN. Required explicitly; this script does not read env vars.",
    )
    parser.add_argument("--account-id", required=True, help="Tenant/account id.")
    parser.add_argument("--faq-id", required=True, help="FAQ draft id to clean up.")
    parser.add_argument(
        "--expected-zendesk-base-url",
        default="",
        help="Required exact Zendesk base URL guard in execute mode.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete external Zendesk macros and the local FAQ draft.",
    )
    parser.add_argument(
        "--confirm-live-zendesk-delete",
        action="store_true",
        help="Required with --execute to delete real Zendesk macros.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
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
            "asyncpg is required for FAQ macro sandbox cleanup; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def cleanup_sandbox_macro_writeback(
    args: argparse.Namespace,
    pool: Any,
    *,
    credentials_provider: ZendeskMacroCredentialsProvider | None = None,
    transport: ZendeskMacroTransport | None = None,
) -> tuple[int, dict[str, Any]]:
    """Preview or execute cleanup for one tenant-scoped FAQ draft."""

    scope = TenantScope(account_id=_clean(args.account_id))
    faq_id = _clean(args.faq_id)
    if getattr(args, "execute", False):
        preflight = _execute_preflight(args)
        if preflight is not None:
            return preflight

    snapshot = await _snapshot(pool, scope=scope, faq_id=faq_id)
    if not getattr(args, "execute", False):
        return 0, _payload(
            ok=True,
            execute=False,
            stage="preview",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
        )
    if not snapshot.found:
        return SKIPPED_EXIT, _payload(
            ok=False,
            execute=True,
            stage="local_lookup",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
            skipped=True,
            not_run_reason="faq_draft_not_found",
        )
    if snapshot.missing_external_id_count:
        return 1, _payload(
            ok=False,
            execute=True,
            stage="mapping_preflight",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
            errors=("macro_mapping_missing_external_id",),
        )

    credentials_source = credentials_provider or _build_credentials_provider(pool)
    credentials = await credentials_source.credentials_for_scope(scope)
    if credentials is None or not credentials.is_complete():
        return SKIPPED_EXIT, _payload(
            ok=False,
            execute=True,
            stage="credential_preflight",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
            skipped=True,
            not_run_reason="zendesk_credentials_missing",
        )

    zendesk_base_url = credentials.normalized_base_url().rstrip("/")
    expected_url = _clean(args.expected_zendesk_base_url).rstrip("/")
    if zendesk_base_url != expected_url:
        return SKIPPED_EXIT, _payload(
            ok=False,
            execute=True,
            stage="credential_preflight",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
            skipped=True,
            not_run_reason="unexpected_zendesk_base_url",
            expected_zendesk_base_url=expected_url,
            observed_zendesk_base_url=zendesk_base_url,
        )
    unexpected_urls = snapshot.unexpected_external_urls(
        expected_base_url=expected_url,
    )
    if unexpected_urls:
        return 1, _payload(
            ok=False,
            execute=True,
            stage="mapping_preflight",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
            zendesk_base_url=zendesk_base_url,
            unexpected_external_urls=unexpected_urls,
            errors=("zendesk_macro_url_base_mismatch",),
        )

    delete_results = await _delete_external_macros(
        snapshot.external_ids,
        credentials=credentials,
        transport=transport or ZendeskHTTPMacroTransport(),
    )
    failed_deletes = [item for item in delete_results if not item["ok"]]
    if failed_deletes:
        return 1, _payload(
            ok=False,
            execute=True,
            stage="external_delete",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
            zendesk_base_url=zendesk_base_url,
            external_deletes=delete_results,
            errors=("zendesk_macro_delete_failed",),
        )

    deleted_faq_count = await _delete_local_faq(pool, scope=scope, faq_id=faq_id)
    ok = deleted_faq_count == 1
    return (
        0 if ok else 1,
        _payload(
            ok=ok,
            execute=True,
            stage="complete" if ok else "local_delete",
            scope=scope,
            faq_id=faq_id,
            snapshot=snapshot,
            zendesk_base_url=zendesk_base_url,
            external_deletes=delete_results,
            deleted_faq_count=deleted_faq_count,
            errors=() if ok else ("faq_draft_delete_missed",),
        ),
    )


def _execute_preflight(args: argparse.Namespace) -> tuple[int, dict[str, Any]] | None:
    if not getattr(args, "confirm_live_zendesk_delete", False):
        return _not_run("missing_confirm_live_zendesk_delete", args)
    if not _clean(getattr(args, "expected_zendesk_base_url", "")):
        return _not_run("missing_expected_zendesk_base_url", args)
    return None


def _not_run(reason: str, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return (
        SKIPPED_EXIT,
        {
            "ok": False,
            "execute": bool(getattr(args, "execute", False)),
            "skipped": True,
            "stage": "preflight",
            "not_run_reason": reason,
            "account_id": _clean(getattr(args, "account_id", "")),
            "faq_id": _clean(getattr(args, "faq_id", "")),
            "expected_zendesk_base_url": _clean(
                getattr(args, "expected_zendesk_base_url", "")
            ),
        },
    )


async def _snapshot(pool: Any, *, scope: TenantScope, faq_id: str) -> CleanupSnapshot:
    row = await pool.fetchrow(
        """
        SELECT id, status
          FROM ticket_faq_markdown
         WHERE account_id = $1
           AND id = $2
         LIMIT 1
        """,
        scope.account_id or "",
        faq_id,
    )
    if row is None:
        return CleanupSnapshot(found=False)

    mappings = await pool.fetch(
        """
        SELECT platform, faq_item_id, external_id, external_url, publish_status
         FROM ticket_faq_macro_writebacks
         WHERE account_id = $1
           AND faq_draft_id = $2
           AND platform = $3
         ORDER BY platform ASC, faq_item_id ASC
        """,
        scope.account_id or "",
        faq_id,
        ZENDESK_PLATFORM,
    )
    attempt_count = await pool.fetchval(
        """
        SELECT COUNT(*)
          FROM ticket_faq_macro_publish_attempts
         WHERE account_id = $1
           AND faq_draft_id = $2
        """,
        scope.account_id or "",
        faq_id,
    )
    search_count = await pool.fetchval(
        """
        SELECT COUNT(*)
          FROM ticket_faq_search_documents
         WHERE account_id = $1
           AND faq_id = $2
        """,
        scope.account_id or "",
        faq_id,
    )
    return CleanupSnapshot(
        found=True,
        draft_status=_clean(_row_get(row, "status")),
        macro_mappings=tuple(_mapping_row_to_dict(mapping) for mapping in mappings),
        publish_attempt_count=int(attempt_count or 0),
        search_document_count=int(search_count or 0),
    )


async def _delete_external_macros(
    external_ids: Sequence[str],
    *,
    credentials: Any,
    transport: ZendeskMacroTransport,
) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    base_url = credentials.normalized_base_url().rstrip("/")
    for external_id in external_ids:
        cleaned_id = _clean(external_id)
        if not cleaned_id:
            continue
        try:
            await transport.request(
                "DELETE",
                f"{base_url}/api/v2/macros/{cleaned_id}",
                headers=_headers(credentials),
                json={},
            )
            results.append({"external_id": cleaned_id, "ok": True, "error": ""})
        except Exception as exc:
            if (
                isinstance(exc, ZendeskMacroTransportError)
                and exc.status_code == 404
            ):
                results.append({
                    "external_id": cleaned_id,
                    "ok": True,
                    "error": "zendesk_macro_not_found",
                    "already_deleted": True,
                })
                continue
            results.append({
                "external_id": cleaned_id,
                "ok": False,
                "error": _clean(str(exc)) or exc.__class__.__name__,
            })
    return tuple(results)


async def _delete_local_faq(pool: Any, *, scope: TenantScope, faq_id: str) -> int:
    rows = await pool.fetch(
        """
        DELETE FROM ticket_faq_markdown
         WHERE account_id = $1
           AND id = $2
        RETURNING id
        """,
        scope.account_id or "",
        faq_id,
    )
    return len(rows)


def _payload(
    *,
    ok: bool,
    execute: bool,
    stage: str,
    scope: TenantScope,
    faq_id: str,
    snapshot: CleanupSnapshot,
    skipped: bool = False,
    not_run_reason: str = "",
    zendesk_base_url: str = "",
    expected_zendesk_base_url: str = "",
    observed_zendesk_base_url: str = "",
    external_deletes: Sequence[Mapping[str, Any]] = (),
    unexpected_external_urls: Sequence[Mapping[str, str]] = (),
    deleted_faq_count: int = 0,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    payload = {
        "ok": ok,
        "execute": execute,
        "skipped": skipped,
        "stage": stage,
        "account_id": scope.account_id or "",
        "faq_id": faq_id,
        "platform": ZENDESK_PLATFORM,
        "snapshot": snapshot.as_dict(),
        "zendesk_base_url": zendesk_base_url,
        "external_deletes": [dict(item) for item in external_deletes],
        "unexpected_external_urls": [
            dict(item) for item in unexpected_external_urls
        ],
        "deleted_faq_count": deleted_faq_count,
        "errors": list(errors),
    }
    if not_run_reason:
        payload["not_run_reason"] = not_run_reason
    if expected_zendesk_base_url:
        payload["expected_zendesk_base_url"] = expected_zendesk_base_url
    if observed_zendesk_base_url:
        payload["observed_zendesk_base_url"] = observed_zendesk_base_url
    return payload


def _build_credentials_provider(pool: Any) -> TenantZendeskMacroCredentialsProvider:
    return TenantZendeskMacroCredentialsProvider(
        pool=pool,
        fallback_provider=ConfigZendeskMacroCredentialsProvider(_host_config()),
    )


def _host_config() -> Any:
    from atlas_brain.config import settings

    return settings.b2b_campaign


def _headers(credentials: Any) -> dict[str, str]:
    return {
        "Authorization": credentials.authorization_header(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _mapping_row_to_dict(row: Any) -> dict[str, Any]:
    return {
        "platform": _clean(_row_get(row, "platform")),
        "faq_item_id": _clean(_row_get(row, "faq_item_id")),
        "external_id": _clean(_row_get(row, "external_id")),
        "external_url": _clean(_row_get(row, "external_url")),
        "publish_status": _clean(_row_get(row, "publish_status")),
    }


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return row[key]


def _write_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if args.execute:
        preflight = _execute_preflight(args)
        if preflight is not None:
            code, payload = preflight
            _write_payload(payload, args)
            return code

    pool = await _create_pool(_clean(args.database_url))
    try:
        code, payload = await cleanup_sandbox_macro_writeback(args, pool)
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


def _url_base(value: str) -> str:
    parsed = urlparse(_clean(value).rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".lower()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
