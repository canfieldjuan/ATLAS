#!/usr/bin/env python3
"""Upsert Content Ops campaign reasoning contexts into Postgres."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_reasoning_postgres import (  # noqa: E402
    PostgresCampaignReasoningContextRepository,
)


_ROW_KEYS = ("contexts", "rows", "data", "reasoning_contexts")
_MATCH_KEYS = (
    "target_id",
    "id",
    "company",
    "company_name",
    "account",
    "account_name",
    "email",
    "contact_email",
    "vendor",
    "vendor_name",
)
_META_KEYS = {"account_id", "target_mode", "selectors", *_MATCH_KEYS}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upsert host-produced campaign reasoning contexts into "
            "campaign_reasoning_contexts."
        )
    )
    parser.add_argument("path", help="JSON object, array, or wrapper containing contexts.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument("--account-id", default="", help="Default tenant/account id.")
    parser.add_argument("--target-mode", default="", help="Default target_mode.")
    parser.add_argument(
        "--selector",
        action="append",
        default=[],
        help="Extra selector to apply to every row; repeatable.",
    )
    parser.add_argument(
        "--table",
        default="campaign_reasoning_contexts",
        help="Reasoning context table name.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input rows and report how many would be upserted.",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="Optional JSONL file for one metadata-only entry per saved row.",
    )
    return parser.parse_args(argv)


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to upsert campaign reasoning contexts; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


def _load_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _context_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("reasoning context JSON must be an object or array")
    for key in _ROW_KEYS:
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [row for row in value if isinstance(row, Mapping)]
        if isinstance(value, Mapping):
            return [
                {"target_id": str(row_key), "context": row_value}
                for row_key, row_value in value.items()
                if isinstance(row_value, Mapping)
            ]
    return [payload]


def _clean_values(values: Sequence[Any]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            cleaned.extend(_clean_values(value))
            continue
        text = str(value or "").strip()
        if not text:
            continue
        cleaned.append(text)
    return tuple(cleaned)


def _row_selectors(row: Mapping[str, Any], extra_selectors: Sequence[str]) -> tuple[str, ...]:
    values: list[Any] = []
    values.extend(_clean_values(extra_selectors))
    values.append(row.get("selectors"))
    values.extend(row.get(key) for key in _MATCH_KEYS)
    return _clean_values(values)


def _row_context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("context")
    if isinstance(nested, Mapping):
        return nested
    return {
        str(key): value
        for key, value in row.items()
        if key not in _META_KEYS and value not in (None, "", [], {})
    }


def _prepare_contexts(
    *,
    payload: Any,
    default_account_id: str,
    default_target_mode: str,
    extra_selectors: Sequence[str],
) -> list[dict[str, Any]]:
    rows = _context_rows(payload)
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        selectors = _row_selectors(row, extra_selectors)
        if not selectors:
            raise ValueError(f"row {index} has no selectors")
        context = _row_context(row)
        if not context:
            raise ValueError(f"row {index} has no context")
        prepared.append(
            {
                "scope": TenantScope(
                    account_id=str(row.get("account_id") or default_account_id or ""),
                ),
                "selectors": selectors,
                "target_mode": str(row.get("target_mode") or default_target_mode or ""),
                "context": context,
            }
        )
    return prepared


async def _upsert_contexts(
    repository: PostgresCampaignReasoningContextRepository,
    *,
    payload: Any,
    default_account_id: str,
    default_target_mode: str,
    extra_selectors: Sequence[str],
    audit_log: Path | None = None,
) -> dict[str, Any]:
    prepared = _prepare_contexts(
        payload=payload,
        default_account_id=default_account_id,
        default_target_mode=default_target_mode,
        extra_selectors=extra_selectors,
    )

    saved_ids: list[str] = []
    for index, item in enumerate(prepared, start=1):
        saved_id = await repository.save_context(
            scope=item["scope"],
            selectors=item["selectors"],
            target_mode=item["target_mode"],
            context=item["context"],
        )
        saved_ids.append(saved_id)
        if audit_log is not None:
            _append_audit_log(
                audit_log,
                [_audit_entry(saved_id=saved_id, row_index=index, item=item)],
            )
    return {"status": "ok", "upserted": len(saved_ids), "ids": saved_ids}


def _audit_entry(*, saved_id: str, row_index: int, item: Mapping[str, Any]) -> dict[str, Any]:
    scope = item["scope"]
    context = item["context"]
    return {
        "action": "upsert_campaign_reasoning_context",
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "row_index": row_index,
        "context_id": saved_id,
        "account_id": scope.account_id,
        "target_mode": item["target_mode"],
        "selectors": list(item["selectors"]),
        "context_keys": sorted(str(key) for key in context.keys()),
    }


def _append_audit_log(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _dry_run_contexts(
    *,
    payload: Any,
    default_account_id: str,
    default_target_mode: str,
    extra_selectors: Sequence[str],
) -> dict[str, Any]:
    prepared = _prepare_contexts(
        payload=payload,
        default_account_id=default_account_id,
        default_target_mode=default_target_mode,
        extra_selectors=extra_selectors,
    )
    return {"status": "dry_run", "would_upsert": len(prepared)}


async def _main_from_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _load_payload(args.path)
    if args.dry_run:
        result = _dry_run_contexts(
            payload=payload,
            default_account_id=str(args.account_id or ""),
            default_target_mode=str(args.target_mode or ""),
            extra_selectors=tuple(args.selector or ()),
        )
    else:
        if not args.database_url:
            raise SystemExit(
                "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL"
            )
        pool = await _create_pool(args.database_url)
        try:
            repository = PostgresCampaignReasoningContextRepository(
                pool=pool,
                table=args.table,
            )
            result = await _upsert_contexts(
                repository,
                payload=payload,
                default_account_id=str(args.account_id or ""),
                default_target_mode=str(args.target_mode or ""),
                extra_selectors=tuple(args.selector or ()),
                audit_log=args.audit_log,
            )
        finally:
            await pool.close()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.dry_run:
        print(
            "dry-run: would upsert "
            f"{result['would_upsert']} campaign reasoning context rows"
        )
    else:
        print(f"upserted {result['upserted']} campaign reasoning context rows")
    return 0


async def _main() -> int:
    return await _main_from_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
