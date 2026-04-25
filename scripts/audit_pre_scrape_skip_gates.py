#!/usr/bin/env python3
"""Audit the pre-scrape skip gates after deploy.

Six checks (matching the deploy plan):
  1. saved_calls (count of skipped_* rows since UTC midnight)
  2. Skip-status breakdown: skipped_low_incremental_yield vs skipped_redundant
  3. Top skipped vendors/sources with unique_insert_ratio, total_found,
     real_runs, parser_version
  4. Sanity invariants (FAIL = potential bug):
       - skip rows for non-paid sources
       - parser_version IS NULL on skip rows
       - total_found < pre_scrape_low_yield_skip_min_found_total in a skip
         decision (the gate is supposed to refuse to fire below the floor)
  5. Paid-source reviews_inserted today vs yesterday (freshness cliff)
  6. No paid-source target was disabled in the audit window with a
     skip-only recent log history (proves the pruner JOIN filter holds)

Exit codes:
  0 = all sanity invariants pass (saved_calls rising is the goal, not a
      failure)
  1 = at least one sanity invariant failed; consider rolling back via
      ATLAS_B2B_SCRAPE_PRE_SCRAPE_SKIP_ENABLED=false  and/or
      ATLAS_B2B_SCRAPE_PRE_SCRAPE_LOW_YIELD_SKIP_ENABLED=false

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/audit_pre_scrape_skip_gates.py --window-hours 1
  python scripts/audit_pre_scrape_skip_gates.py --window-hours 24 --output /tmp/skip_gate_audit.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.config import settings
from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_pre_scrape_skip_gates")


def _derive_paid_sources() -> list[str]:
    """Same derivation as the gate uses (verified + web_unlocker/residential)."""
    from atlas_brain.services.scraping.capabilities import (
        AccessPattern,
        DataQuality,
        ProxyClass,
        get_all_capabilities,
    )
    paid: set[str] = set()
    for source, profile in get_all_capabilities().items():
        if profile.data_quality is not DataQuality.verified:
            continue
        uses_wu = AccessPattern.web_unlocker in profile.access_patterns
        uses_res = profile.proxy_class is ProxyClass.residential
        if uses_wu or uses_res:
            paid.add(str(source).strip().lower())
    return sorted(paid)


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return out


async def _check_saved_calls(pool, *, window_hours: int) -> tuple[list[str], int]:
    md = ["## 1. saved_calls"]
    rows = await pool.fetch(
        """
        SELECT
          COUNT(*) FILTER (WHERE COALESCE(status,'') LIKE 'skipped%')
            AS saved_in_window,
          COUNT(*) FILTER (
            WHERE COALESCE(status,'') LIKE 'skipped%'
              AND started_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
          ) AS saved_today_utc,
          COUNT(*) AS total_runs_in_window
        FROM b2b_scrape_log
        WHERE started_at >= NOW() - ($1::int * INTERVAL '1 hour')
        """,
        window_hours,
    )
    r = rows[0]
    saved = int(r["saved_in_window"])
    today = int(r["saved_today_utc"])
    total = int(r["total_runs_in_window"])
    md.append(f"- saved_in_window ({window_hours}h): **{saved}**")
    md.append(f"- saved_today_utc: **{today}**")
    md.append(f"- total_runs_in_window: {total}")
    md.append("")
    return md, saved


async def _check_status_breakdown(pool, *, window_hours: int) -> list[str]:
    md = ["## 2. Skip-status breakdown"]
    rows = await pool.fetch(
        """
        SELECT status, COUNT(*) AS n
        FROM b2b_scrape_log
        WHERE COALESCE(status,'') LIKE 'skipped%'
          AND started_at >= NOW() - ($1::int * INTERVAL '1 hour')
        GROUP BY status
        ORDER BY n DESC
        """,
        window_hours,
    )
    if not rows:
        md.append("- (no skip rows in window)")
        md.append("")
        return md
    md.extend(_md_table(["status", "count"], [[r["status"], r["n"]] for r in rows]))
    md.append("")
    return md


async def _check_top_skipped(pool, *, window_hours: int) -> list[str]:
    md = ["## 3. Top skipped (source, vendor) in window"]
    rows = await pool.fetch(
        """
        SELECT
          l.source,
          t.vendor_name,
          l.parser_version,
          l.status,
          l.errors,
          l.started_at
        FROM b2b_scrape_log l
        JOIN b2b_scrape_targets t ON t.id = l.target_id
        WHERE COALESCE(l.status,'') LIKE 'skipped%'
          AND l.started_at >= NOW() - ($1::int * INTERVAL '1 hour')
        ORDER BY l.started_at DESC
        LIMIT 30
        """,
        window_hours,
    )
    if not rows:
        md.append("- (no skip rows in window)")
        md.append("")
        return md
    table_rows = []
    for r in rows:
        ratio = ""
        found = ""
        runs = ""
        raw_errs = r["errors"]
        if isinstance(raw_errs, str):
            try:
                raw_errs = json.loads(raw_errs)
            except (ValueError, TypeError):
                raw_errs = []
        errs = raw_errs or []
        if isinstance(errs, list) and errs:
            payload = errs[0] if isinstance(errs[0], dict) else {}
            ratio_val = payload.get("unique_insert_ratio")
            if ratio_val is None:
                ratio_val = payload.get("duplicate_ratio")
            if ratio_val is not None:
                ratio = f"{float(ratio_val):.4f}"
            tf = payload.get("total_found")
            if tf is not None:
                found = str(tf)
            rr = payload.get("real_runs")
            if rr is not None:
                runs = str(rr)
        table_rows.append([
            r["source"],
            r["vendor_name"],
            r["parser_version"] or "",
            r["status"],
            ratio,
            found,
            runs,
            str(r["started_at"])[:19],
        ])
    md.extend(_md_table(
        ["source", "vendor", "parser_version", "status", "ratio", "found", "runs", "started_at"],
        table_rows,
    ))
    md.append("")
    return md


async def _check_sanity_invariants(
    pool, *, window_hours: int, paid_sources: list[str], min_found_floor: int,
) -> tuple[list[str], bool]:
    md = ["## 4. Sanity invariants"]
    paid_set = "{" + ",".join(repr(s) for s in paid_sources) + "}"
    bad_sources = await pool.fetch(
        """
        SELECT id, source, started_at, status
        FROM b2b_scrape_log
        WHERE COALESCE(status,'') LIKE 'skipped%'
          AND started_at >= NOW() - ($1::int * INTERVAL '1 hour')
          AND source NOT IN (
            SELECT unnest($2::text[])
          )
        """,
        window_hours,
        paid_sources,
    )
    bad_pv = await pool.fetch(
        """
        SELECT id, source, started_at, status
        FROM b2b_scrape_log
        WHERE COALESCE(status,'') LIKE 'skipped%'
          AND started_at >= NOW() - ($1::int * INTERVAL '1 hour')
          AND parser_version IS NULL
        """,
        window_hours,
    )
    bad_floor = await pool.fetch(
        """
        SELECT id, source, started_at, status, errors
        FROM b2b_scrape_log
        WHERE COALESCE(status,'') LIKE 'skipped%'
          AND started_at >= NOW() - ($1::int * INTERVAL '1 hour')
          AND EXISTS (
            SELECT 1
            FROM jsonb_array_elements(errors) e
            WHERE (e->>'total_found')::int < $2::int
          )
        """,
        window_hours,
        min_found_floor,
    )
    all_pass = True
    if bad_sources:
        all_pass = False
        md.append(f"- **FAIL** skip rows on non-paid sources: {len(bad_sources)}")
        for r in bad_sources[:5]:
            md.append(f"  - {r['source']} status={r['status']} at {r['started_at']}")
    else:
        md.append("- PASS skip rows are all on paid sources")
    if bad_pv:
        all_pass = False
        md.append(f"- **FAIL** skip rows with parser_version IS NULL: {len(bad_pv)}")
        for r in bad_pv[:5]:
            md.append(f"  - {r['source']} status={r['status']} at {r['started_at']}")
    else:
        md.append("- PASS all skip rows have a parser_version")
    if bad_floor:
        all_pass = False
        md.append(f"- **FAIL** skip rows with total_found < {min_found_floor}: {len(bad_floor)}")
        for r in bad_floor[:5]:
            md.append(f"  - {r['source']} status={r['status']} at {r['started_at']}")
    else:
        md.append(f"- PASS all skip decisions have total_found >= {min_found_floor}")
    md.append("")
    _ = paid_set  # keep linter happy
    return md, all_pass


async def _check_freshness_cliff(pool, *, paid_sources: list[str]) -> tuple[list[str], bool]:
    md = ["## 5. Freshness cliff (paid-source reviews_inserted)"]
    rows = await pool.fetch(
        """
        SELECT
          COALESCE(SUM(reviews_inserted) FILTER (
            WHERE started_at >= date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
          ), 0) AS today_utc,
          COALESCE(SUM(reviews_inserted) FILTER (
            WHERE started_at >= (date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
                              - INTERVAL '1 day'
              AND started_at <  date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
          ), 0) AS yesterday_utc
        FROM b2b_scrape_log
        WHERE source = ANY($1::text[])
          AND COALESCE(status,'') NOT LIKE 'skipped%'
        """,
        paid_sources,
    )
    today = int(rows[0]["today_utc"])
    yesterday = int(rows[0]["yesterday_utc"])
    md.append(f"- today_utc: **{today}**")
    md.append(f"- yesterday_utc: **{yesterday}**")
    cliff_threshold = 0.5
    cliff = False
    if yesterday > 0:
        delta_pct = (today - yesterday) / yesterday
        md.append(f"- delta vs yesterday: {delta_pct*100:.1f}%")
        if today < yesterday * cliff_threshold and yesterday >= 50:
            cliff = True
            md.append(f"- **WARN** today is below {int(cliff_threshold*100)}% of yesterday on a non-trivial baseline; possible freshness cliff")
        else:
            md.append("- PASS no freshness cliff (within tolerance or baseline too small)")
    else:
        md.append("- (yesterday baseline is 0; cannot compute delta)")
    md.append("")
    return md, not cliff


async def _check_pruner_safety(
    pool, *, window_hours: int, paid_sources: list[str],
) -> tuple[list[str], bool]:
    md = ["## 6. Pruner safety (no paid target disabled with skip-only history)"]
    rows = await pool.fetch(
        """
        WITH disabled_recent AS (
          SELECT id, source, vendor_name, last_scraped_at, updated_at
          FROM b2b_scrape_targets
          WHERE enabled = false
            AND source = ANY($1::text[])
            AND updated_at >= NOW() - ($2::int * INTERVAL '1 hour')
        ),
        log_summary AS (
          SELECT t.id AS target_id,
                 COUNT(*) AS n,
                 COUNT(*) FILTER (
                   WHERE COALESCE(l.status,'') LIKE 'skipped%'
                 ) AS n_skipped
          FROM disabled_recent t
          JOIN b2b_scrape_log l ON l.target_id = t.id
          WHERE l.started_at >= NOW() - INTERVAL '7 days'
          GROUP BY t.id
        )
        SELECT d.id, d.source, d.vendor_name, d.last_scraped_at, d.updated_at,
               COALESCE(s.n, 0) AS recent_runs,
               COALESCE(s.n_skipped, 0) AS recent_skipped
        FROM disabled_recent d
        LEFT JOIN log_summary s ON s.target_id = d.id
        ORDER BY d.updated_at DESC
        """,
        paid_sources,
        window_hours,
    )
    bug_hits: list[Any] = []
    for r in rows:
        n = int(r["recent_runs"])
        ns = int(r["recent_skipped"])
        if n > 0 and ns > 0 and (ns / n) >= 0.5:
            bug_hits.append(r)
    if bug_hits:
        md.append(f"- **FAIL** {len(bug_hits)} paid target(s) disabled in window with skip-dominated recent history")
        for r in bug_hits[:5]:
            md.append(
                f"  - {r['source']}/{r['vendor_name']} disabled at {r['updated_at']} "
                f"(recent_runs={r['recent_runs']} recent_skipped={r['recent_skipped']})"
            )
        md.append("")
        return md, False
    if not rows:
        md.append("- PASS no paid-source targets disabled in audit window")
    else:
        md.append(f"- PASS {len(rows)} paid target(s) disabled in window, none with skip-dominated history")
    md.append("")
    return md, True


async def _main_async(args: argparse.Namespace) -> int:
    await init_database()
    pool = get_db_pool()
    if not pool.is_initialized:
        print("ERROR: db pool not initialized", file=sys.stderr)
        return 1
    cfg = settings.b2b_scrape
    paid_sources = _derive_paid_sources()
    min_found_floor = int(cfg.pre_scrape_low_yield_skip_min_found_total or 50)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    md: list[str] = [
        f"# Pre-scrape skip gate audit",
        f"- started_at_utc: {started_at_utc}",
        f"- window_hours: {args.window_hours}",
        f"- paid_sources: {paid_sources}",
        f"- min_found_floor: {min_found_floor}",
        "",
    ]
    try:
        section_1, _saved = await _check_saved_calls(pool, window_hours=args.window_hours)
        md.extend(section_1)
        md.extend(await _check_status_breakdown(pool, window_hours=args.window_hours))
        md.extend(await _check_top_skipped(pool, window_hours=args.window_hours))
        section_4, sanity_pass = await _check_sanity_invariants(
            pool, window_hours=args.window_hours,
            paid_sources=paid_sources, min_found_floor=min_found_floor,
        )
        md.extend(section_4)
        section_5, freshness_pass = await _check_freshness_cliff(
            pool, paid_sources=paid_sources,
        )
        md.extend(section_5)
        section_6, pruner_pass = await _check_pruner_safety(
            pool, window_hours=args.window_hours, paid_sources=paid_sources,
        )
        md.extend(section_6)
        verdict_pass = sanity_pass and freshness_pass and pruner_pass
        md.append("## Verdict")
        if verdict_pass:
            md.append("**PASS** all sanity invariants hold; saved_calls rising is the intended outcome.")
        else:
            md.append("**FAIL** at least one invariant failed; review sections above and consider rollback:")
            md.append("  - ATLAS_B2B_SCRAPE_PRE_SCRAPE_SKIP_ENABLED=false")
            md.append("  - ATLAS_B2B_SCRAPE_PRE_SCRAPE_LOW_YIELD_SKIP_ENABLED=false")
        md.append("")
        report = "\n".join(md)
        print(report)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report)
            print(f"\nreport written to: {out_path}")
        return 0 if verdict_pass else 1
    finally:
        await close_database()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Audit window in hours (default 24).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional markdown output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
