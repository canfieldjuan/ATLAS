#!/usr/bin/env python3
"""Smoke-test concurrent FAQ search retrieval against Postgres."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.ticket_faq_search import (  # noqa: E402
    PostgresTicketFAQSearchRepository,
    TicketFAQSearchDocument,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - host dependency
    load_dotenv = None


@dataclass(frozen=True)
class SearchCase:
    account_id: str
    corpus_id: str
    faq_id: str
    query: str
    expected_hit: bool


def _default_database_url() -> str | None:
    raw = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if raw:
        return raw
    try:
        from atlas_brain.storage.config import db_settings
    except Exception:
        return None
    dsn = str(getattr(db_settings, "dsn", "") or "").strip()
    return dsn or None


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    _load_dotenv_files()
    parser = argparse.ArgumentParser(
        description="Run concurrent FAQ search retrieval against Postgres."
    )
    parser.add_argument("--database-url", default=_default_database_url())
    parser.add_argument("--account-count", type=int, default=3)
    parser.add_argument("--corpora-per-account", type=int, default=2)
    parser.add_argument("--documents-per-corpus", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    for name in (
        "account_count",
        "corpora_per_account",
        "documents_per_corpus",
        "iterations",
        "concurrency",
        "pool_size",
    ):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")


async def _create_pool(database_url: str, *, pool_size: int):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for the FAQ search concurrency smoke; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=pool_size)


def _build_cases(
    *,
    run_id: str,
    account_count: int,
    corpora_per_account: int,
) -> tuple[SearchCase, ...]:
    cases: list[SearchCase] = []
    for account_index in range(account_count):
        account_id = f"faq-search-{run_id}-acct-{account_index}"
        for corpus_index in range(corpora_per_account):
            corpus_id = f"faq-search-{run_id}-corpus-{corpus_index}"
            faq_id = str(uuid4())
            cases.append(
                SearchCase(
                    account_id=account_id,
                    corpus_id=corpus_id,
                    faq_id=faq_id,
                    query="password reset",
                    expected_hit=True,
                )
            )
            cases.append(
                SearchCase(
                    account_id=account_id,
                    corpus_id=corpus_id,
                    faq_id=faq_id,
                    query="escrow shortage",
                    expected_hit=False,
                )
            )
    return tuple(cases)


def _documents_for_case(case: SearchCase, *, documents_per_corpus: int) -> tuple[TicketFAQSearchDocument, ...]:
    documents: list[TicketFAQSearchDocument] = []
    for rank in range(1, documents_per_corpus + 1):
        documents.append(
            TicketFAQSearchDocument(
                account_id=case.account_id,
                corpus_id=case.corpus_id,
                faq_id=case.faq_id,
                target_id=f"support-{case.corpus_id}",
                target_mode="support_account",
                status="approved",
                rank=rank,
                topic="password reset",
                question=f"How do I reset my password for corpus {case.corpus_id}?",
                answer_summary="Use the password reset email and contact support if it expires.",
                source_ids=(f"{case.corpus_id}-ticket-{rank}",),
                ticket_count=rank,
                search_text="password reset email login support account",
            )
        )
    return tuple(documents)


async def _apply_migrations(pool: Any) -> None:
    for relative in (
        "atlas_brain/storage/migrations/325_ticket_faq_markdown.sql",
        "atlas_brain/storage/migrations/327_ticket_faq_search_documents.sql",
    ):
        await pool.execute((ROOT / relative).read_text(encoding="utf-8"))


async def _seed(pool: Any, repo: PostgresTicketFAQSearchRepository, cases: Sequence[SearchCase], *, documents_per_corpus: int) -> None:
    seen_faq_ids: set[str] = set()
    for case in cases:
        if case.faq_id in seen_faq_ids:
            continue
        seen_faq_ids.add(case.faq_id)
        await pool.execute(
            """
            INSERT INTO ticket_faq_markdown (
                id, account_id, target_id, target_mode, title, markdown,
                items, source_count, ticket_source_count, output_checks,
                warnings, metadata, status
            )
            VALUES (
                $1::uuid, $2, $3, 'support_account', 'FAQ Search Smoke',
                '# FAQ Search Smoke', '[]'::jsonb, $4, $4,
                '{}'::jsonb, '[]'::jsonb, '{}'::jsonb, 'approved'
            )
            """,
            case.faq_id,
            case.account_id,
            f"support-{case.corpus_id}",
            documents_per_corpus,
        )
        await repo.replace_documents(
            _documents_for_case(case, documents_per_corpus=documents_per_corpus)
        )


async def _cleanup(pool: Any, cases: Sequence[SearchCase]) -> None:
    account_ids = sorted({case.account_id for case in cases})
    if account_ids:
        await pool.execute(
            "DELETE FROM ticket_faq_markdown WHERE account_id = ANY($1::text[])",
            account_ids,
        )


async def _run_case(
    repo: PostgresTicketFAQSearchRepository,
    case: SearchCase,
    *,
    limit: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await repo.search(
        query=case.query,
        account_id=case.account_id,
        corpus_id=case.corpus_id,
        status="approved",
        limit=limit,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    rows = response.as_dict()["results"]
    failures: list[str] = []
    if case.expected_hit and not rows:
        failures.append("expected hit returned no rows")
    if not case.expected_hit and rows:
        failures.append("expected miss returned rows")
    for row in rows:
        if row.get("account_id") != case.account_id:
            failures.append("wrong account_id returned")
        if row.get("corpus_id") != case.corpus_id:
            failures.append("wrong corpus_id returned")
    return {
        "account_id": case.account_id,
        "corpus_id": case.corpus_id,
        "query": case.query,
        "expected_hit": case.expected_hit,
        "result_count": len(rows),
        "elapsed_ms": round(elapsed_ms, 6),
        "failures": failures,
    }


async def _run_concurrent_searches(
    repo: PostgresTicketFAQSearchRepository,
    cases: Sequence[SearchCase],
    *,
    iterations: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    selected = [cases[index % len(cases)] for index in range(iterations)]

    async def worker(case: SearchCase) -> dict[str, Any]:
        async with semaphore:
            try:
                return await _run_case(repo, case, limit=5)
            except Exception as exc:  # pragma: no cover - exercised by live smoke failures.
                return {
                    "account_id": case.account_id,
                    "corpus_id": case.corpus_id,
                    "query": case.query,
                    "expected_hit": case.expected_hit,
                    "result_count": 0,
                    "elapsed_ms": 0.0,
                    "failures": [f"{type(exc).__name__}: {exc}"],
                }

    return list(await asyncio.gather(*(worker(case) for case in selected)))


def _latency_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row.get("elapsed_ms") or 0.0) for row in results)
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    p95_index = min(len(values) - 1, max(0, int(len(values) * 0.95) - 1))
    return {
        "count": len(values),
        "p50_ms": round(float(statistics.median(values)), 6),
        "p95_ms": round(values[p95_index], 6),
        "max_ms": round(values[-1], 6),
    }


def _failure_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failures = [
        {
            "account_id": row.get("account_id"),
            "corpus_id": row.get("corpus_id"),
            "query": row.get("query"),
            "failures": row.get("failures"),
        }
        for row in results
        if row.get("failures")
    ]
    return {
        "count": len(failures),
        "items": failures[:20],
        "truncated": len(failures) > 20,
    }


async def run_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    _validate_args(args)
    run_id = uuid4().hex[:10]
    cases = _build_cases(
        run_id=run_id,
        account_count=int(args.account_count),
        corpora_per_account=int(args.corpora_per_account),
    )
    pool = await _create_pool(str(args.database_url), pool_size=int(args.pool_size))
    repo = PostgresTicketFAQSearchRepository(pool)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        await _apply_migrations(pool)
        await _seed(pool, repo, cases, documents_per_corpus=int(args.documents_per_corpus))
        results = await _run_concurrent_searches(
            repo,
            cases,
            iterations=int(args.iterations),
            concurrency=int(args.concurrency),
        )
    finally:
        if not bool(args.keep_data):
            await _cleanup(pool, cases)
        await pool.close()
    failure = _failure_summary(results)
    summary = {
        "ok": failure["count"] == 0,
        "run_id": run_id,
        "requests": {
            "total": len(results),
            "iterations": int(args.iterations),
            "concurrency": int(args.concurrency),
        },
        "seed": {
            "accounts": int(args.account_count),
            "corpora_per_account": int(args.corpora_per_account),
            "documents_per_corpus": int(args.documents_per_corpus),
            "search_cases": len(cases),
        },
        "latency": _latency_summary(results),
        "isolation": failure,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    return (0 if summary["ok"] else 1), summary


def _write_result(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    latency = summary["latency"]
    print(
        "FAQ search concurrency smoke: "
        f"ok={summary['ok']} requests={summary['requests']['total']} "
        f"p50_ms={latency['p50_ms']} p95_ms={latency['p95_ms']} "
        f"max_ms={latency['max_ms']} failures={summary['isolation']['count']}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    code, summary = asyncio.run(run_smoke(args))
    if args.output_result:
        _write_result(Path(args.output_result), summary)
    _print_summary(summary, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
