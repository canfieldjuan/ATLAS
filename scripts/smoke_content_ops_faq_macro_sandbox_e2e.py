#!/usr/bin/env python3
"""Seed one FAQ draft and run the guarded Zendesk macro live smoke."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SKIPPED_EXIT = 2


def _load_script_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEED_SCRIPT = _load_script_module(
    "seed_faq_macro_writeback_live_smoke_draft_for_e2e",
    ROOT / "scripts" / "seed_faq_macro_writeback_live_smoke_draft.py",
)
_LIVE_SMOKE_SCRIPT = _load_script_module(
    "smoke_content_ops_faq_macro_live_zendesk_for_e2e",
    ROOT / "scripts" / "smoke_content_ops_faq_macro_live_zendesk.py",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        required=True,
        help="Postgres DSN. Required explicitly; this script does not read env vars.",
    )
    parser.add_argument("--account-id", required=True, help="Tenant/account id.")
    parser.add_argument("--user-id", default="", help="Optional operator user id.")
    parser.add_argument(
        "--expected-zendesk-base-url",
        default="",
        help="Required exact Zendesk base URL guard, for example https://acme.zendesk.com.",
    )
    parser.add_argument(
        "--target-id",
        default="macro-writeback-live-smoke-seed",
        help="Target id stored on the seeded FAQ draft.",
    )
    parser.add_argument(
        "--target-mode",
        default="support_ticket_faq",
        help="Target mode stored on the seeded FAQ draft.",
    )
    parser.add_argument(
        "--title",
        default="Zendesk macro writeback sandbox E2E smoke seed",
        help="FAQ draft title.",
    )
    parser.add_argument(
        "--question",
        default="How do I refund a duplicate charge?",
        help="Question used for the publishable FAQ item.",
    )
    parser.add_argument(
        "--resolution-text",
        default=(
            "Open Billing, select the duplicate charge, and click Refund payment. "
            "Confirm the refund and tell the customer it may take 3-5 business days."
        ),
        help="Verified answer text used for the publishable macro body.",
    )
    parser.add_argument(
        "--confirm-create-draft",
        action="store_true",
        help="Required to create one approved FAQ draft in Postgres.",
    )
    parser.add_argument(
        "--confirm-live-zendesk-write",
        action="store_true",
        help="Required to create or update real Zendesk macros.",
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
    if not _clean(args.question):
        raise SystemExit("--question is required")
    if not _clean(args.resolution_text):
        raise SystemExit("--resolution-text is required")


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for the sandbox FAQ macro writeback smoke; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def run_sandbox_e2e_smoke(
    args: argparse.Namespace,
    pool: Any,
    *,
    seed_stage: Any | None = None,
    live_stage: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run seed then live-smoke stages and return one combined artifact."""

    preflight = _preflight(args)
    if preflight is not None:
        return preflight

    seed_runner = seed_stage or _seed_stage
    live_runner = live_stage or _live_stage
    seed_code, seed_payload = await seed_runner(args, pool)
    if seed_code != 0 or not seed_payload.get("ok"):
        return _stage_failure(
            "seed",
            seed_code,
            account_id=_clean(args.account_id),
            seed_payload=seed_payload,
        )

    faq_id = _clean(seed_payload.get("faq_id"))
    if not faq_id:
        return _stage_failure(
            "seed",
            1,
            account_id=_clean(args.account_id),
            seed_payload={
                **dict(seed_payload),
                "error": "seed_faq_id_missing",
            },
        )

    live_args = _live_args(args, faq_id=faq_id)
    live_code, live_payload = await live_runner(live_args, pool)
    ok = live_code == 0 and bool(live_payload.get("ok"))
    return (
        0 if ok else live_code or 1,
        {
            "ok": ok,
            "skipped": False,
            "live_smoke_skipped": bool(live_payload.get("skipped", False)),
            "stage": "complete" if ok else "live_smoke",
            "account_id": _clean(args.account_id),
            "faq_id": faq_id,
            "zendesk_base_url": _clean(live_payload.get("zendesk_base_url")),
            "seed": seed_payload,
            "live_smoke": live_payload,
            "errors": list(live_payload.get("errors") or ()),
        },
    )


async def _seed_stage(args: argparse.Namespace, pool: Any) -> tuple[int, dict[str, Any]]:
    return await _SEED_SCRIPT.seed_live_smoke_draft(args, pool)


async def _live_stage(args: argparse.Namespace, pool: Any) -> tuple[int, dict[str, Any]]:
    return await _LIVE_SMOKE_SCRIPT.run_live_zendesk_smoke(args, pool)


def _preflight(args: argparse.Namespace) -> tuple[int, dict[str, Any]] | None:
    if not getattr(args, "confirm_create_draft", False):
        return _not_run("missing_confirm_create_draft", args)
    if not getattr(args, "confirm_live_zendesk_write", False):
        return _not_run("missing_confirm_live_zendesk_write", args)
    if not _clean(getattr(args, "expected_zendesk_base_url", "")):
        return _not_run("missing_expected_zendesk_base_url", args)
    return None


def _not_run(reason: str, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    return (
        SKIPPED_EXIT,
        {
            "ok": False,
            "skipped": True,
            "stage": "preflight",
            "not_run_reason": reason,
            "account_id": _clean(getattr(args, "account_id", "")),
            "expected_zendesk_base_url": _clean(
                getattr(args, "expected_zendesk_base_url", "")
            ),
        },
    )


def _stage_failure(
    stage: str,
    code: int,
    *,
    account_id: str,
    seed_payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    return (
        code or 1,
        {
            "ok": False,
            "skipped": bool(seed_payload.get("skipped", False)),
            "stage": stage,
            "account_id": account_id,
            "faq_id": _clean(seed_payload.get("faq_id")),
            "seed": seed_payload,
            "live_smoke": None,
            "errors": [_clean(seed_payload.get("error")) or f"{stage}_stage_failed"],
        },
    )


def _live_args(args: argparse.Namespace, *, faq_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        database_url=_clean(args.database_url),
        account_id=_clean(args.account_id),
        faq_id=faq_id,
        user_id=_clean(getattr(args, "user_id", "")),
        expected_zendesk_base_url=_clean(args.expected_zendesk_base_url),
        confirm_live_zendesk_write=True,
        output=None,
        json=True,
    )


def _write_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    preflight = _preflight(args)
    if preflight is not None:
        code, payload = preflight
        _write_payload(payload, args)
        return code

    pool = await _create_pool(_clean(args.database_url))
    try:
        code, payload = await run_sandbox_e2e_smoke(args, pool)
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
