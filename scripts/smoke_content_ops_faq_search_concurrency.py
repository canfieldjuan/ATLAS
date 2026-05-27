#!/usr/bin/env python3
"""Smoke-test concurrent FAQ search retrieval against Postgres."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
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

SEEDED_FAQ_TARGET_MODE = "support_account"
SEEDED_FAQ_TITLE = "FAQ Search Smoke"
SEEDED_FAQ_MARKDOWN = "# FAQ Search Smoke"
SEEDED_FAQ_STATUS = "approved"
SEEDED_HIT_QUERY = "export attribution report"
SEEDED_MISS_QUERY = "saml domain verification"

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
    parser.add_argument("--account-id", default="")
    parser.add_argument("--corpora-per-account", type=int, default=2)
    parser.add_argument("--documents-per-corpus", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-single-request-ms", type=float)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--route-case-file-output", type=Path)
    parser.add_argument("--cleanup-manifest-output", type=Path)
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
    for name in ("max_p95_ms", "max_single_request_ms"):
        value = getattr(args, name)
        if value is not None and float(value) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if str(args.account_id or "").strip() and int(args.account_count) != 1:
        raise SystemExit("--account-id requires --account-count 1")
    if args.route_case_file_output and not bool(args.keep_data):
        raise SystemExit("--route-case-file-output requires --keep-data")
    if args.cleanup_manifest_output and not bool(args.keep_data):
        raise SystemExit("--cleanup-manifest-output requires --keep-data")


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
    account_id: str = "",
) -> tuple[SearchCase, ...]:
    cases: list[SearchCase] = []
    for account_index in range(account_count):
        resolved_account_id = account_id.strip() or f"faq-search-{run_id}-acct-{account_index}"
        for corpus_index in range(corpora_per_account):
            corpus_id = f"faq-search-{run_id}-corpus-{corpus_index}"
            faq_id = str(uuid4())
            cases.append(
                SearchCase(
                    account_id=resolved_account_id,
                    corpus_id=corpus_id,
                    faq_id=faq_id,
                    query=SEEDED_HIT_QUERY,
                    expected_hit=True,
                )
            )
            cases.append(
                SearchCase(
                    account_id=resolved_account_id,
                    corpus_id=corpus_id,
                    faq_id=faq_id,
                    query=SEEDED_MISS_QUERY,
                    expected_hit=False,
                )
            )
    return tuple(cases)


def _seeded_faq_target_id(case: SearchCase) -> str:
    return f"support-{case.corpus_id}"


def _route_case_payload(
    cases: Sequence[SearchCase],
    *,
    documents_per_corpus: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    expected_hit_count = min(documents_per_corpus, limit)
    payload: list[dict[str, Any]] = []
    for case in cases:
        row = {
            "query": case.query,
            "corpus_id": case.corpus_id,
            "status": SEEDED_FAQ_STATUS,
            "limit": limit,
            "require_results": case.expected_hit,
            "expected_count": expected_hit_count if case.expected_hit else 0,
        }
        if case.expected_hit:
            row.update(
                {
                    "expected_first_account_id": case.account_id,
                    "expected_first_corpus_id": case.corpus_id,
                    "expected_first_faq_id": case.faq_id,
                    "expected_detail_account_id": case.account_id,
                    "expected_detail_target_id": _seeded_faq_target_id(case),
                    "expected_detail_target_mode": SEEDED_FAQ_TARGET_MODE,
                    "expected_detail_title": SEEDED_FAQ_TITLE,
                    "expected_detail_status": SEEDED_FAQ_STATUS,
                }
            )
        payload.append(row)
    return payload


def _write_route_case_file(
    path: Path | None,
    cases: Sequence[SearchCase],
    *,
    documents_per_corpus: int,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _route_case_payload(cases, documents_per_corpus=documents_per_corpus),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _cleanup_manifest_payload(cases: Sequence[SearchCase]) -> dict[str, Any]:
    return {
        "account_ids": sorted({case.account_id for case in cases}),
        "corpus_ids": sorted({case.corpus_id for case in cases}),
        "faq_ids": sorted({case.faq_id for case in cases}),
        "search_cases": len(cases),
    }


def _write_cleanup_manifest(path: Path | None, cases: Sequence[SearchCase]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_cleanup_manifest_payload(cases), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
                topic="reporting export",
                question=f"How do I export attribution reports for corpus {case.corpus_id}?",
                answer_summary="Use the reporting export workflow and contact support if the CSV is missing.",
                source_ids=(f"{case.corpus_id}-ticket-{rank}",),
                ticket_count=rank,
                search_text="export attribution report dashboard csv support account",
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
                $1::uuid, $2, $3, $4, $5, $6, '[]'::jsonb, $7, $7,
                '{}'::jsonb, '[]'::jsonb, '{}'::jsonb, $8
            )
            """,
            case.faq_id,
            case.account_id,
            _seeded_faq_target_id(case),
            SEEDED_FAQ_TARGET_MODE,
            SEEDED_FAQ_TITLE,
            SEEDED_FAQ_MARKDOWN,
            documents_per_corpus,
            SEEDED_FAQ_STATUS,
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
    p95_index = min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))
    return {
        "count": len(values),
        "p50_ms": round(float(statistics.median(values)), 6),
        "p95_ms": round(values[p95_index], 6),
        "max_ms": round(values[-1], 6),
    }


def _latency_budget_summary(
    latency: dict[str, Any],
    *,
    max_p95_ms: float | None,
    max_single_request_ms: float | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for metric, limit in (
        ("p95_ms", max_p95_ms),
        ("max_ms", max_single_request_ms),
    ):
        if limit is None:
            continue
        actual = float(latency[metric])
        max_ms = round(float(limit), 6)
        ok = actual <= max_ms
        checks.append(
            {
                "metric": metric,
                "actual_ms": actual,
                "max_ms": max_ms,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(f"{metric} exceeded {max_ms} ms")
    return {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
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


def _cleanup_result(*, attempted: bool, error: BaseException | None = None) -> dict[str, Any]:
    return {
        "ok": error is None,
        "attempted": attempted,
        "error": None
        if error is None
        else {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _with_cleanup_result(
    summary: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(summary)
    updated["cleanup"] = cleanup
    updated["ok"] = bool(summary["ok"]) and bool(cleanup["ok"])
    return updated


def _summary_payload(
    *,
    ok: bool,
    run_id: str,
    args: argparse.Namespace,
    cases: Sequence[SearchCase],
    results: Sequence[dict[str, Any]],
    setup: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    failure = _failure_summary(results)
    latency = _latency_summary(results)
    latency_budget = _latency_budget_summary(
        latency,
        max_p95_ms=args.max_p95_ms,
        max_single_request_ms=args.max_single_request_ms,
    )
    return {
        "ok": (
            ok
            and failure["count"] == 0
            and latency_budget["ok"]
            and setup["ok"]
        ),
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
        "setup": setup,
        "cleanup": _cleanup_result(attempted=False),
        "latency": latency,
        "latency_budget": latency_budget,
        "isolation": failure,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _setup_failure_summary(
    *,
    run_id: str,
    args: argparse.Namespace,
    cases: Sequence[SearchCase],
    phase: str,
    error: BaseException,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return _summary_payload(
        ok=False,
        run_id=run_id,
        args=args,
        cases=cases,
        results=[],
        setup={
            "ok": False,
            "phase": phase,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        },
        elapsed_seconds=elapsed_seconds,
    )


async def run_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    _validate_args(args)
    run_id = uuid4().hex[:10]
    cases = _build_cases(
        run_id=run_id,
        account_count=int(args.account_count),
        corpora_per_account=int(args.corpora_per_account),
        account_id=str(args.account_id or ""),
    )
    started = time.perf_counter()
    try:
        pool = await _create_pool(str(args.database_url), pool_size=int(args.pool_size))
    except Exception as exc:
        summary = _summary_payload(
            ok=False,
            run_id=run_id,
            args=args,
            cases=cases,
            results=[],
            setup={
                "ok": False,
                "phase": "pool_create",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        return 1, summary
    repo = PostgresTicketFAQSearchRepository(pool)
    results: list[dict[str, Any]] = []
    cleanup_ready = False
    summary: dict[str, Any] | None = None
    cleanup = _cleanup_result(attempted=False)
    try:
        try:
            await _apply_migrations(pool)
        except Exception as exc:
            summary = _setup_failure_summary(
                run_id=run_id,
                args=args,
                cases=cases,
                phase="migrations",
                error=exc,
                elapsed_seconds=time.perf_counter() - started,
            )
        if summary is None:
            cleanup_ready = True
            try:
                _write_cleanup_manifest(args.cleanup_manifest_output, cases)
            except Exception as exc:
                summary = _setup_failure_summary(
                    run_id=run_id,
                    args=args,
                    cases=cases,
                    phase="cleanup_manifest_output",
                    error=exc,
                    elapsed_seconds=time.perf_counter() - started,
                )
        if summary is None:
            try:
                await _seed(pool, repo, cases, documents_per_corpus=int(args.documents_per_corpus))
            except Exception as exc:
                summary = _setup_failure_summary(
                    run_id=run_id,
                    args=args,
                    cases=cases,
                    phase="seed",
                    error=exc,
                    elapsed_seconds=time.perf_counter() - started,
                )
        if summary is None:
            try:
                _write_route_case_file(
                    args.route_case_file_output,
                    cases,
                    documents_per_corpus=int(args.documents_per_corpus),
                )
            except Exception as exc:
                summary = _setup_failure_summary(
                    run_id=run_id,
                    args=args,
                    cases=cases,
                    phase="route_case_file_output",
                    error=exc,
                    elapsed_seconds=time.perf_counter() - started,
                )
        if summary is None:
            results = await _run_concurrent_searches(
                repo,
                cases,
                iterations=int(args.iterations),
                concurrency=int(args.concurrency),
            )
            summary = _summary_payload(
                ok=True,
                run_id=run_id,
                args=args,
                cases=cases,
                results=results,
                setup={"ok": True, "phase": "complete", "error": None},
                elapsed_seconds=time.perf_counter() - started,
            )
    finally:
        if cleanup_ready and not bool(args.keep_data):
            try:
                await _cleanup(pool, cases)
                cleanup = _cleanup_result(attempted=True)
            except Exception as exc:
                cleanup = _cleanup_result(attempted=True, error=exc)
        await pool.close()
    if summary is None:  # pragma: no cover - defensive guard for unexpected control flow.
        raise RuntimeError("FAQ search smoke did not produce a summary")
    summary = _with_cleanup_result(summary, cleanup)
    return (0 if summary["ok"] else 1), summary


def _write_result(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    latency = summary["latency"]
    setup_error = ""
    setup = summary.get("setup")
    if isinstance(setup, dict):
        error = setup.get("error")
        if isinstance(error, dict) and error.get("message"):
            setup_error = f" setup_error={error['message']}"
    print(
        "FAQ search concurrency smoke: "
        f"ok={summary['ok']} requests={summary['requests']['total']} "
        f"p50_ms={latency['p50_ms']} p95_ms={latency['p95_ms']} "
        f"max_ms={latency['max_ms']} failures={summary['isolation']['count']} "
        f"latency_budget_failures={len(summary['latency_budget']['failures'])}"
        f"{setup_error}"
    )


def _preflight_summary(
    args: argparse.Namespace,
    *,
    message: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return _summary_payload(
        ok=False,
        run_id="preflight",
        args=args,
        cases=[],
        results=[],
        setup={
            "ok": False,
            "phase": "preflight",
            "error": {
                "type": "SystemExit",
                "message": message,
            },
        },
        elapsed_seconds=elapsed_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    try:
        code, summary = asyncio.run(run_smoke(args))
    except SystemExit as exc:
        summary = _preflight_summary(
            args,
            message=str(exc.code or ""),
            elapsed_seconds=time.perf_counter() - started,
        )
        code = 2
    if args.output_result:
        _write_result(Path(args.output_result), summary)
    _print_summary(summary, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
