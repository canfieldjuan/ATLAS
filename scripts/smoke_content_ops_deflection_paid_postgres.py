#!/usr/bin/env python3
"""Guarded Postgres smoke for the FAQ deflection paid gate."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any
from uuid import UUID, uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.api import billing  # noqa: E402
from extracted_content_pipeline.deflection_report_access import (  # noqa: E402
    PostgresDeflectionReportArtifactStore,
)
from extracted_content_pipeline.storage._jsonb_helpers import (  # noqa: E402
    parse_command_tag,
)


SKIPPED_EXIT = 2
SMOKE_REQUEST_PREFIX = "smoke-"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        required=True,
        help="Postgres DSN. Required explicitly; this script does not read env vars.",
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="UUID account id used for the ephemeral smoke row.",
    )
    parser.add_argument(
        "--request-id",
        default="",
        help="Optional request id. Must start with smoke- when supplied.",
    )
    parser.add_argument(
        "--payment-reference",
        default="",
        help="Optional fake Checkout session id/payment reference.",
    )
    parser.add_argument(
        "--amount-cents",
        type=int,
        default=150000,
        help="Fake Checkout amount_total in cents.",
    )
    parser.add_argument(
        "--currency",
        default="usd",
        help="Fake Checkout currency.",
    )
    parser.add_argument(
        "--confirm-postgres-write",
        action="store_true",
        help="Required before creating or updating the Postgres smoke row.",
    )
    parser.add_argument(
        "--cleanup-on-success",
        action="store_true",
        help="Delete the ephemeral smoke row after the paid-gate proof succeeds.",
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
    if int(args.amount_cents) < 1:
        raise SystemExit("--amount-cents must be positive")
    if not _clean(args.currency):
        raise SystemExit("--currency is required")


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for the deflection paid Postgres smoke"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def run_deflection_paid_postgres_smoke(
    args: argparse.Namespace,
    pool: Any,
    *,
    store: PostgresDeflectionReportArtifactStore | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the guarded smoke, returning an exit code and JSON payload."""

    preflight = _preflight(args)
    if preflight is not None:
        return preflight

    account_id = _account_id(args)
    request_id = _request_id(args)
    payment_reference = _payment_reference(args)
    artifact_store = store or PostgresDeflectionReportArtifactStore(pool=pool)

    existing = await artifact_store.get_artifact_record(
        account_id=account_id,
        request_id=request_id,
    )
    if existing is not None:
        return _not_run(
            "ephemeral_request_already_exists",
            account_id=account_id,
            request_id=request_id,
        )

    snapshot, artifact = _smoke_payload(request_id)
    await artifact_store.save_report(
        account_id=account_id,
        request_id=request_id,
        snapshot=snapshot,
        artifact=artifact,
    )
    locked = await artifact_store.get_artifact_record(
        account_id=account_id,
        request_id=request_id,
    )
    if locked is None:
        return _failed(
            "seeded_report_missing",
            account_id=account_id,
            request_id=request_id,
        )
    if locked.paid:
        return _failed(
            "seeded_report_unexpectedly_paid",
            account_id=account_id,
            request_id=request_id,
        )

    metadata = {
        "source": "content_ops_deflection_report",
        "account_id": account_id,
        "request_id": request_id,
    }
    session = SimpleNamespace(
        id=payment_reference,
        mode="payment",
        payment_status="paid",
        amount_total=int(args.amount_cents),
        currency=_clean(args.currency).lower(),
        metadata=metadata,
        to_dict=lambda: {
            "id": payment_reference,
            "mode": "payment",
            "payment_status": "paid",
            "amount_total": int(args.amount_cents),
            "currency": _clean(args.currency).lower(),
            "metadata": dict(metadata),
        },
    )
    await billing._handle_content_ops_deflection_report_checkout_completed(
        pool,
        session,
        metadata,
    )
    unlocked = await artifact_store.get_artifact_record(
        account_id=account_id,
        request_id=request_id,
    )
    if unlocked is None or not unlocked.paid:
        return _failed(
            "paid_gate_did_not_unlock",
            account_id=account_id,
            request_id=request_id,
        )
    cleanup_requested = bool(getattr(args, "cleanup_on_success", False))
    cleanup_deleted = False
    if cleanup_requested:
        cleanup_deleted = await _cleanup_ephemeral_report(
            pool,
            account_id=account_id,
            request_id=request_id,
        )
        if not cleanup_deleted:
            return _failed(
                "cleanup_failed",
                account_id=account_id,
                request_id=request_id,
                payment_reference=unlocked.payment_reference,
                cleanup_requested=True,
                cleanup_deleted=False,
                paid_after_checkout=True,
            )
    return (
        0,
        {
            "ok": True,
            "skipped": False,
            "account_id": account_id,
            "request_id": request_id,
            "payment_reference": unlocked.payment_reference,
            "locked_before_paid": not locked.paid,
            "paid_after_checkout": unlocked.paid,
            "cleanup_requested": cleanup_requested,
            "cleanup_deleted": cleanup_deleted,
            "snapshot_summary": dict(unlocked.snapshot.get("summary") or {}),
            "artifact_summary": dict((unlocked.artifact or {}).get("summary") or {}),
        },
    )


def _preflight(args: argparse.Namespace) -> tuple[int, dict[str, Any]] | None:
    account_id = _clean(args.account_id)
    request_id = _clean(getattr(args, "request_id", ""))
    if not getattr(args, "confirm_postgres_write", False):
        return _not_run(
            "missing_confirm_postgres_write",
            account_id=account_id,
            request_id=request_id,
        )
    try:
        UUID(account_id)
    except ValueError:
        return _not_run(
            "invalid_account_id",
            account_id=account_id,
            request_id=request_id,
        )
    if request_id and not request_id.startswith(SMOKE_REQUEST_PREFIX):
        return _not_run(
            "request_id_not_ephemeral",
            account_id=account_id,
            request_id=request_id,
        )
    return None


def _smoke_payload(request_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = {
        "summary": {
            "generated": 1,
            "drafted_answer_count": 1,
            "no_proven_answer_count": 0,
        },
        "top_questions": [
            {
                "rank": 1,
                "question": "How do customers export attribution reports?",
                "weighted_frequency": 2,
                "customer_wording": "How do I export attribution reports?",
            }
        ],
    }
    artifact = {
        "markdown": "# Smoke FAQ Deflection Report\n\nOpen Analytics and export.",
        "summary": dict(snapshot["summary"]),
        "faq_result": {
            "items": [
                {
                    "question": "How do customers export attribution reports?",
                    "answer": "Open Analytics and export the report.",
                    "source_ids": [f"{request_id}:ticket-1"],
                }
            ]
        },
    }
    return snapshot, artifact


def _account_id(args: argparse.Namespace) -> str:
    return str(UUID(_clean(args.account_id)))


def _request_id(args: argparse.Namespace) -> str:
    return _clean(getattr(args, "request_id", "")) or (
        f"{SMOKE_REQUEST_PREFIX}deflection-paid-postgres-{uuid4().hex}"
    )


def _payment_reference(args: argparse.Namespace) -> str:
    return _clean(getattr(args, "payment_reference", "")) or f"cs_{uuid4().hex}"


async def _cleanup_ephemeral_report(
    pool: Any,
    *,
    account_id: str,
    request_id: str,
) -> bool:
    if not request_id.startswith(SMOKE_REQUEST_PREFIX):
        return False
    result = await pool.execute(
        """
        DELETE FROM content_ops_deflection_reports
        WHERE account_id = $1
          AND request_id = $2
          AND request_id LIKE 'smoke-%'
        """,
        account_id,
        request_id,
    )
    return parse_command_tag(result)


def _not_run(reason: str, **extra: Any) -> tuple[int, dict[str, Any]]:
    payload = {
        "ok": False,
        "skipped": True,
        "not_run_reason": reason,
    }
    payload.update(extra)
    return SKIPPED_EXIT, payload


def _failed(reason: str, **extra: Any) -> tuple[int, dict[str, Any]]:
    payload = {
        "ok": False,
        "skipped": False,
        "error": reason,
    }
    payload.update(extra)
    return 1, payload


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
        code, payload = await run_deflection_paid_postgres_smoke(args, pool)
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
