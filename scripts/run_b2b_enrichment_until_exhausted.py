#!/usr/bin/env python3
"""Run deterministic B2B enrichment until only terminal states remain.

This wrapper hardens the enrichment task with:
- one batch per cycle for bounded progress
- per-cycle timeout
- cancellation and orphan recovery on timeout
- exhausted pending rows marked failed between cycles

Usage:
  ATLAS_B2B_CHURN_ENABLED=true python scripts/run_b2b_enrichment_until_exhausted.py
  ATLAS_B2B_CHURN_ENABLED=true python scripts/run_b2b_enrichment_until_exhausted.py --batch-size 4 --concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.config import settings
from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.storage.database import close_database, get_db_pool, init_database

_TERMINAL_STATUSES = (
    "enriched",
    "no_signal",
    "failed",
    "filtered",
    "not_applicable",
    "quarantined",
)


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


async def _get_counts(pool) -> dict[str, int]:
    rows = await pool.fetch(
        """
        SELECT enrichment_status, count(*)
        FROM b2b_reviews
        GROUP BY 1
        ORDER BY 1
        """
    )
    return {str(row["enrichment_status"]): int(row["count"] or 0) for row in rows}


async def _run_cycle(*, batch_size: int, concurrency: int, inter_batch_delay: float) -> dict:
    task = SimpleNamespace(
        metadata={
            "enrichment_max_per_batch": int(batch_size),
            "enrichment_max_rounds_per_run": 1,
            "enrichment_concurrency": int(concurrency),
            "enrichment_inter_batch_delay_seconds": float(inter_batch_delay),
        }
    )
    return await b2b_enrichment.run(task)


async def _recover(pool, max_attempts: int) -> dict[str, int]:
    orphaned = await b2b_enrichment._recover_orphaned_enriching(pool, max_attempts)
    exhausted = await b2b_enrichment._mark_exhausted_pending_failed(pool, max_attempts)
    return {"orphaned_recovered": int(orphaned), "exhausted_marked_failed": int(exhausted)}


async def _main(args: argparse.Namespace) -> None:
    await init_database()
    pool = get_db_pool()
    cfg = settings.b2b_churn
    if not cfg.enabled:
        raise SystemExit("B2B churn pipeline disabled; set ATLAS_B2B_CHURN_ENABLED=true for this run")
    if not pool.is_initialized:
        raise SystemExit("Database pool not initialized")

    cycle = 0
    idle_cycles = 0
    started_at = time.time()

    try:
        while True:
            before = await _get_counts(pool)
            pending_before = int(before.get("pending", 0) or 0)
            enriching_before = int(before.get("enriching", 0) or 0)
            cycle += 1
            print(_dump({"cycle": cycle, "phase": "before", "counts": before}))

            if pending_before == 0 and enriching_before == 0:
                print(_dump({"cycle": cycle, "phase": "done", "reason": "terminal_only"}))
                break

            if pending_before == 0 and enriching_before > 0:
                recovered = await _recover(pool, cfg.enrichment_max_attempts)
                after_recover = await _get_counts(pool)
                print(_dump({
                    "cycle": cycle,
                    "phase": "recover_only",
                    "recovery": recovered,
                    "counts": after_recover,
                }))
                continue

            run_task = asyncio.create_task(
                _run_cycle(
                    batch_size=args.batch_size,
                    concurrency=args.concurrency,
                    inter_batch_delay=args.inter_batch_delay,
                )
            )
            timed_out = False
            result: dict | None = None
            try:
                result = await asyncio.wait_for(run_task, timeout=args.cycle_timeout_seconds)
            except asyncio.TimeoutError:
                timed_out = True
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
                await asyncio.sleep(args.cancel_grace_seconds)
                result = {
                    "timed_out": True,
                    "cycle_timeout_seconds": args.cycle_timeout_seconds,
                }
            except Exception as exc:
                result = {
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }

            recovery = await _recover(pool, cfg.enrichment_max_attempts)
            after = await _get_counts(pool)
            progress = after != before
            idle_cycles = 0 if progress else idle_cycles + 1

            print(_dump({
                "cycle": cycle,
                "phase": "after",
                "timed_out": timed_out,
                "result": result,
                "recovery": recovery,
                "counts": after,
                "progress": progress,
                "idle_cycles": idle_cycles,
            }))

            if idle_cycles >= args.max_idle_cycles:
                print(_dump({
                    "cycle": cycle,
                    "phase": "done",
                    "reason": "max_idle_cycles",
                    "max_idle_cycles": args.max_idle_cycles,
                }))
                break
    finally:
        elapsed = round(time.time() - started_at, 2)
        final_counts = await _get_counts(pool) if pool.is_initialized else {}
        print(_dump({"phase": "summary", "elapsed_seconds": elapsed, "counts": final_counts}))
        await close_database()


if __name__ == "__main__":
    default_cycle_timeout = max(
        30.0,
        float(settings.b2b_churn.enrichment_full_extraction_timeout_seconds) * 1.5,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--inter-batch-delay", type=float, default=0.0)
    parser.add_argument("--cycle-timeout-seconds", type=float, default=default_cycle_timeout)
    parser.add_argument("--cancel-grace-seconds", type=float, default=2.0)
    parser.add_argument("--max-idle-cycles", type=int, default=3)
    asyncio.run(_main(parser.parse_args()))
