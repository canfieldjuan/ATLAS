#!/usr/bin/env python3
"""Seed the synthetic B2B SaaS FAQ demo into ticket FAQ search."""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    source_rows_to_campaign_opportunities,
)
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
    build_ticket_faq_markdown,
)
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft  # noqa: E402
from extracted_content_pipeline.ticket_faq_postgres import (  # noqa: E402
    PostgresTicketFAQRepository,
)
from extracted_content_pipeline.ticket_faq_search import (  # noqa: E402
    PostgresTicketFAQSearchRepository,
    build_ticket_faq_search_documents,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional host dependency
    load_dotenv = None


DEMO_SOURCE_PATH = ROOT / "extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv"
DEMO_TITLE = "Synthetic B2B SaaS Support FAQ Demo"
DEMO_DATASET_LABEL = "synthetic_b2b_saas_demo"
DEFAULT_CORPUS_ID = "synthetic-b2b-saas-demo"
DEFAULT_TARGET_ID = "support-synthetic-b2b-saas-demo"
DEFAULT_TARGET_MODE = "support_account"
DEFAULT_STATUS = "approved"
DEFAULT_SUPPORT_CONTACT = "https://example.com/support"
DEFAULT_QUERY = "export attribution reports"


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    _load_dotenv_files()
    parser = argparse.ArgumentParser(
        description="Seed the synthetic B2B SaaS FAQ demo into FAQ search."
    )
    parser.add_argument(
        "--database-url",
        default=_env("EXTRACTED_DATABASE_URL", "DATABASE_URL"),
    )
    parser.add_argument(
        "--account-id",
        default=_env("ATLAS_FAQ_SEARCH_ACCOUNT_ID", "ATLAS_ACCOUNT_ID"),
    )
    parser.add_argument("--user-id", default="")
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument("--target-id", default=DEFAULT_TARGET_ID)
    parser.add_argument("--status", default=DEFAULT_STATUS)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not str(args.database_url or "").strip():
        errors.append("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    if not str(args.account_id or "").strip():
        errors.append("ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required")
    if not str(args.corpus_id or "").strip():
        errors.append("--corpus-id is required")
    if not str(args.target_id or "").strip():
        errors.append("--target-id is required")
    if not str(args.status or "").strip():
        errors.append("--status is required")
    if not str(args.query or "").strip():
        errors.append("--query is required")
    if int(args.limit) < 1:
        errors.append("--limit must be positive")
    return errors


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError("asyncpg is required to seed the SaaS FAQ demo") from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


def _demo_rows(path: Path = DEMO_SOURCE_PATH) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_saas_demo_faq_draft(
    *,
    corpus_id: str = DEFAULT_CORPUS_ID,
    target_id: str = DEFAULT_TARGET_ID,
) -> TicketFAQDraft:
    rows = _demo_rows()
    normalized = source_rows_to_campaign_opportunities(
        rows,
        target_mode=DEFAULT_TARGET_MODE,
    )
    result = build_ticket_faq_markdown(
        normalized.opportunities,
        title=DEMO_TITLE,
        max_items=12,
        max_evidence_per_item=3,
        support_contact=DEFAULT_SUPPORT_CONTACT,
    )
    failed_checks = [
        name for name, passed in sorted(result.output_checks.items())
        if passed is not True
    ]
    if normalized.warnings:
        raise ValueError("SaaS demo source rows produced normalization warnings")
    if failed_checks:
        raise ValueError(f"SaaS demo FAQ output checks failed: {', '.join(failed_checks)}")
    return TicketFAQDraft(
        target_id=target_id,
        target_mode=DEFAULT_TARGET_MODE,
        title=DEMO_TITLE,
        markdown=result.markdown,
        items=result.items,
        source_count=result.source_count,
        ticket_source_count=result.ticket_source_count,
        output_checks=result.output_checks,
        warnings=result.warnings,
        metadata={
            "corpus_id": corpus_id,
            "dataset_label": DEMO_DATASET_LABEL,
            "demo_source_path": str(DEMO_SOURCE_PATH.relative_to(ROOT)),
            "synthetic": True,
        },
    )


async def seed_saas_demo_faq(
    pool: Any,
    *,
    account_id: str,
    user_id: str = "",
    corpus_id: str = DEFAULT_CORPUS_ID,
    target_id: str = DEFAULT_TARGET_ID,
    status: str = DEFAULT_STATUS,
    query: str = DEFAULT_QUERY,
    limit: int = 5,
) -> dict[str, Any]:
    scope = TenantScope(account_id=account_id, user_id=user_id)
    draft = build_saas_demo_faq_draft(corpus_id=corpus_id, target_id=target_id)
    faq_repo = PostgresTicketFAQRepository(pool)
    search_repo = PostgresTicketFAQSearchRepository(pool)
    saved_ids = tuple(await faq_repo.save_drafts([draft], scope=scope))
    faq_id = saved_ids[0] if saved_ids else ""
    errors: list[str] = []
    if not faq_id:
        errors.append("FAQ repository did not return a saved FAQ id")
    updated = False
    if faq_id:
        updated = bool(await faq_repo.update_status(faq_id, status, scope=scope))
        if not updated:
            errors.append("FAQ repository did not approve the saved FAQ id")
    # Projection persistence happens inside update_status(...). This projection is
    # only a diagnostic parity count for the summary payload.
    projected_draft = replace(draft, id=faq_id, status=status)
    projected_documents = build_ticket_faq_search_documents(
        projected_draft,
        account_id=account_id,
        corpus_id=corpus_id,
    )
    response = await search_repo.search(
        query=query,
        account_id=account_id,
        corpus_id=corpus_id,
        status=status,
        limit=limit,
    )
    search_payload = response.as_dict()
    search_results = search_payload.get("results") or []
    matched_seeded_faq = any(
        isinstance(row, dict) and row.get("faq_id") == faq_id
        for row in search_results
    )
    if not matched_seeded_faq:
        errors.append("Seeded SaaS FAQ id was not present in verification search results")
    return {
        "ok": not errors,
        "errors": errors,
        "account_id": account_id,
        "corpus_id": corpus_id,
        "faq_id": faq_id,
        "status": status,
        "source_count": draft.source_count,
        "ticket_source_count": draft.ticket_source_count,
        "generated_items": len(draft.items),
        "projected_documents": len(projected_documents),
        "search": {
            "query": search_payload.get("query"),
            "count": search_payload.get("count"),
            "matched_seeded_faq": matched_seeded_faq,
            "first_result": (
                search_results[0]
                if search_results
                else None
            ),
        },
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    pool = await _create_pool(str(args.database_url))
    try:
        return await seed_saas_demo_faq(
            pool,
            account_id=str(args.account_id),
            user_id=str(args.user_id or ""),
            corpus_id=str(args.corpus_id),
            target_id=str(args.target_id),
            status=str(args.status),
            query=str(args.query),
            limit=int(args.limit),
        )
    finally:
        await pool.close()


def _write_result(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_result(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(
        "SaaS FAQ demo seed: "
        f"ok={payload['ok']} faq_id={payload['faq_id']} "
        f"corpus_id={payload['corpus_id']} search_count={payload['search']['count']} "
        f"errors={len(payload['errors'])}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = _validate_args(args)
    if errors:
        raise SystemExit("; ".join(errors))
    payload = asyncio.run(_run(args))
    _write_result(args.output_result, payload)
    _print_result(payload, as_json=bool(args.json))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
