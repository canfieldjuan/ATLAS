#!/usr/bin/env python3
"""Run the guarded FAQ macro sandbox write and cleanup lifecycle."""

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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_E2E_SCRIPT = _load_script_module(
    "smoke_content_ops_faq_macro_sandbox_e2e_for_lifecycle",
    ROOT / "scripts" / "smoke_content_ops_faq_macro_sandbox_e2e.py",
)
_CLEANUP_SCRIPT = _load_script_module(
    "cleanup_content_ops_faq_macro_sandbox_for_lifecycle",
    ROOT / "scripts" / "cleanup_content_ops_faq_macro_sandbox.py",
)
_CHECKER_SCRIPT = _load_script_module(
    "check_content_ops_faq_macro_lifecycle_artifact_for_lifecycle",
    ROOT / "scripts" / "check_content_ops_faq_macro_lifecycle_artifact.py",
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
        help="Required exact Zendesk base URL guard.",
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
        default="Zendesk macro writeback sandbox lifecycle smoke seed",
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
        "--confirm-live-zendesk-delete",
        action="store_true",
        help="Required to delete real Zendesk macros during cleanup.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional sanitized Markdown proof summary output path.",
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
            "asyncpg is required for the sandbox FAQ macro lifecycle smoke; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def run_sandbox_lifecycle_smoke(
    args: argparse.Namespace,
    pool: Any,
    *,
    write_stage: Any | None = None,
    cleanup_stage: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run sandbox write then cleanup stages and return one lifecycle artifact."""

    preflight = _preflight(args)
    if preflight is not None:
        return preflight

    write_runner = write_stage or _write_stage
    cleanup_runner = cleanup_stage or _cleanup_stage
    write_code, write_payload = await write_runner(args, pool)
    if write_code != 0 or not write_payload.get("ok"):
        return _stage_failure(
            "write",
            write_code,
            account_id=_clean(args.account_id),
            write_payload=write_payload,
        )

    faq_id = _clean(write_payload.get("faq_id"))
    if not faq_id:
        return _stage_failure(
            "write",
            1,
            account_id=_clean(args.account_id),
            write_payload={
                **dict(write_payload),
                "error": "write_faq_id_missing",
            },
        )

    cleanup_args = _cleanup_args(args, faq_id=faq_id)
    cleanup_code, cleanup_payload = await cleanup_runner(cleanup_args, pool)
    ok = cleanup_code == 0 and bool(cleanup_payload.get("ok"))
    return (
        0 if ok else cleanup_code or 1,
        {
            "ok": ok,
            "skipped": False,
            "cleanup_skipped": bool(cleanup_payload.get("skipped", False)),
            "stage": "complete" if ok else "cleanup",
            "account_id": _clean(args.account_id),
            "faq_id": faq_id,
            "zendesk_base_url": _clean(
                cleanup_payload.get("zendesk_base_url")
                or write_payload.get("zendesk_base_url")
            ),
            "write": write_payload,
            "cleanup": cleanup_payload,
            "errors": list(cleanup_payload.get("errors") or ()),
        },
    )


async def _write_stage(args: argparse.Namespace, pool: Any) -> tuple[int, dict[str, Any]]:
    return await _E2E_SCRIPT.run_sandbox_e2e_smoke(args, pool)


async def _cleanup_stage(args: argparse.Namespace, pool: Any) -> tuple[int, dict[str, Any]]:
    return await _CLEANUP_SCRIPT.cleanup_sandbox_macro_writeback(args, pool)


def _preflight(args: argparse.Namespace) -> tuple[int, dict[str, Any]] | None:
    if not getattr(args, "confirm_create_draft", False):
        return _not_run("missing_confirm_create_draft", args)
    if not getattr(args, "confirm_live_zendesk_write", False):
        return _not_run("missing_confirm_live_zendesk_write", args)
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
    write_payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    return (
        code or 1,
        {
            "ok": False,
            "skipped": bool(write_payload.get("skipped", False)),
            "cleanup_skipped": False,
            "stage": stage,
            "account_id": account_id,
            "faq_id": _clean(write_payload.get("faq_id")),
            "write": write_payload,
            "cleanup": None,
            "errors": [_clean(write_payload.get("error")) or f"{stage}_stage_failed"],
        },
    )


def _cleanup_args(args: argparse.Namespace, *, faq_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        database_url=_clean(args.database_url),
        account_id=_clean(args.account_id),
        faq_id=faq_id,
        expected_zendesk_base_url=_clean(args.expected_zendesk_base_url),
        execute=True,
        confirm_live_zendesk_delete=True,
        output=None,
        json=True,
    )


def _write_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def _write_proof_summary(
    code: int,
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    if not args.summary_output:
        return code, payload

    result = _CHECKER_SCRIPT.validate_lifecycle_artifact(payload)
    args.summary_output.write_text(
        _CHECKER_SCRIPT.render_summary(result),
        encoding="utf-8",
    )
    if code == 0 and not result.get("ok"):
        errors = list(payload.get("errors") or ())
        errors.append("proof_summary_validation_failed")
        return (
            1,
            {
                **payload,
                "ok": False,
                "stage": "proof_summary",
                "errors": errors,
                "proof_summary_errors": list(result.get("errors") or ()),
            },
        )
    return code, payload


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    preflight = _preflight(args)
    if preflight is not None:
        code, payload = preflight
        code, payload = _write_proof_summary(code, payload, args)
        _write_payload(payload, args)
        return code

    pool = await _create_pool(_clean(args.database_url))
    try:
        code, payload = await run_sandbox_lifecycle_smoke(args, pool)
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    code, payload = _write_proof_summary(code, payload, args)
    _write_payload(payload, args)
    return code


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
