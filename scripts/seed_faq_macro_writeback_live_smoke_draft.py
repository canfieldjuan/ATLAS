#!/usr/bin/env python3
"""Seed one approved FAQ draft for the guarded Zendesk macro live smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.faq_macro_writeback import (  # noqa: E402
    APPROVED_FAQ_STATUS,
    build_macro_writeback_preview,
)
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft  # noqa: E402
from extracted_content_pipeline.ticket_faq_postgres import (  # noqa: E402
    PostgresTicketFAQRepository,
)


SKIPPED_EXIT = 2
DATABASE_URL_ENV_EXPR = "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        required=True,
        help="Postgres DSN. Required explicitly; the script does not read env vars.",
    )
    parser.add_argument("--account-id", required=True, help="Tenant/account id.")
    parser.add_argument("--user-id", default="", help="Optional operator user id.")
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
        default="Zendesk macro writeback live smoke seed",
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
        "--expected-zendesk-base-url",
        default="",
        help=(
            "Optional expected Zendesk base URL included in the printed next "
            "live-smoke command."
        ),
    )
    parser.add_argument(
        "--confirm-create-draft",
        action="store_true",
        help="Required to write one approved FAQ draft to Postgres.",
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
    if not _clean(args.question):
        raise SystemExit("--question is required")
    if not _clean(args.resolution_text):
        raise SystemExit("--resolution-text is required")


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to seed the macro writeback live-smoke draft; "
            "install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def seed_live_smoke_draft(
    args: argparse.Namespace,
    pool: Any,
    *,
    faq_repository: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Seed one approved FAQ draft and return a machine-readable payload."""

    scope = TenantScope(
        account_id=_clean(args.account_id),
        user_id=_clean(getattr(args, "user_id", "")) or None,
    )
    if not getattr(args, "confirm_create_draft", False):
        return _not_run(
            "missing_confirm_create_draft",
            account_id=scope.account_id or "",
        )

    repo = faq_repository or PostgresTicketFAQRepository(pool)
    draft = _seed_draft(args)
    saved_ids = tuple(await repo.save_drafts((draft,), scope=scope))
    if len(saved_ids) != 1 or not _clean(saved_ids[0]):
        return 1, {
            "ok": False,
            "skipped": False,
            "account_id": scope.account_id or "",
            "error": "seed_faq_draft_save_failed",
        }
    faq_id = _clean(saved_ids[0])
    if not await repo.update_status(faq_id, APPROVED_FAQ_STATUS, scope=scope):
        return 1, {
            "ok": False,
            "skipped": False,
            "account_id": scope.account_id or "",
            "faq_id": faq_id,
            "error": "seed_faq_draft_approve_failed",
        }

    approved = _with_identity(draft, faq_id=faq_id, status=APPROVED_FAQ_STATUS)
    preview = build_macro_writeback_preview((approved,))
    if preview.publishable_count < 1:
        return 1, {
            "ok": False,
            "skipped": False,
            "account_id": scope.account_id or "",
            "faq_id": faq_id,
            "error": "seed_faq_draft_not_publishable",
            "preview": preview.as_dict(),
        }

    return 0, {
        "ok": True,
        "skipped": False,
        "account_id": scope.account_id or "",
        "faq_id": faq_id,
        "draft_status": APPROVED_FAQ_STATUS,
        "publishable_count": preview.publishable_count,
        "macro_titles": [macro.title for macro in preview.macros],
        "next_command": _next_live_smoke_command(args, faq_id=faq_id),
    }


def _seed_draft(args: argparse.Namespace) -> TicketFAQDraft:
    question = _clean(args.question)
    resolution_text = _clean(args.resolution_text)
    item = {
        "faq_item_id": "macro-writeback-live-smoke-item",
        "topic": "billing",
        "question": question,
        "answer": resolution_text,
        "resolution_text": resolution_text,
        "answer_evidence_status": "resolution_evidence",
        "resolution_evidence_scope": "scoped",
        "source_ids": ("macro-writeback-live-smoke-ticket",),
    }
    return TicketFAQDraft(
        target_id=_clean(args.target_id),
        target_mode=_clean(args.target_mode),
        title=_clean(args.title),
        markdown=_seed_markdown(question=question, resolution_text=resolution_text),
        items=(item,),
        source_count=1,
        ticket_source_count=1,
        output_checks={
            "condensed": True,
            "has_action_items": True,
            "resolution_evidence_scoped": True,
            "uses_user_vocabulary": True,
        },
        warnings=(),
        metadata={
            "macro_writeback_live_smoke_seed": True,
            "zendesk_write": False,
        },
    )


def _with_identity(draft: TicketFAQDraft, *, faq_id: str, status: str) -> TicketFAQDraft:
    return TicketFAQDraft(
        target_id=draft.target_id,
        target_mode=draft.target_mode,
        title=draft.title,
        markdown=draft.markdown,
        items=draft.items,
        source_count=draft.source_count,
        ticket_source_count=draft.ticket_source_count,
        output_checks=draft.output_checks,
        warnings=draft.warnings,
        metadata=draft.metadata,
        id=faq_id,
        status=status,
    )


def _seed_markdown(*, question: str, resolution_text: str) -> str:
    return (
        "# Zendesk macro writeback live smoke seed\n\n"
        f"## {question}\n\n"
        f"{resolution_text}\n"
    )


def _next_live_smoke_command(args: argparse.Namespace, *, faq_id: str) -> str:
    command = [
        "python",
        "scripts/smoke_content_ops_faq_macro_live_zendesk.py",
        "--database-url",
        DATABASE_URL_ENV_EXPR,
        "--account-id",
        _clean(args.account_id),
        "--faq-id",
        faq_id,
    ]
    if _clean(getattr(args, "user_id", "")):
        command.extend(["--user-id", _clean(args.user_id)])
    expected_base_url = _clean(getattr(args, "expected_zendesk_base_url", ""))
    command.extend([
        "--expected-zendesk-base-url",
        expected_base_url or "<expected-zendesk-base-url>",
        "--confirm-live-zendesk-write",
        "--json",
    ])
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(part: str) -> str:
    if part == DATABASE_URL_ENV_EXPR:
        return f'"{DATABASE_URL_ENV_EXPR}"'
    return shlex.quote(part)


def _not_run(reason: str, **extra: Any) -> tuple[int, dict[str, Any]]:
    payload = {
        "ok": False,
        "skipped": True,
        "not_run_reason": reason,
    }
    payload.update(extra)
    return SKIPPED_EXIT, payload


def _write_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not args.confirm_create_draft:
        code, payload = _not_run(
            "missing_confirm_create_draft",
            account_id=_clean(args.account_id),
        )
        _write_payload(payload, args)
        return code

    pool = await _create_pool(_clean(args.database_url))
    try:
        code, payload = await seed_live_smoke_draft(args, pool)
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            await close()
    _write_payload(payload, args)
    return code


def _clean(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
