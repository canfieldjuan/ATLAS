#!/usr/bin/env python3
"""Remove polluted company signals and rebuild affected accounts-in-motion rows.

Usage:
  python scripts/cleanup_accounts_in_motion_pollution.py
  python scripts/cleanup_accounts_in_motion_pollution.py --apply
  python scripts/cleanup_accounts_in_motion_pollution.py --apply --vendors "Salesforce,ClickUp"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.autonomous.tasks._b2b_shared import (
    _build_company_signal_blocked_names_by_vendor,
    _canonicalize_vendor,
    _company_signal_exclusion_reason,
    _fetch_existing_company_signals,
    _high_intent_row_has_signal_evidence,
    _high_intent_signal_evidence_enabled,
    build_account_intelligence,
)
from atlas_brain.autonomous.tasks.b2b_accounts_in_motion import (
    run as run_accounts_in_motion,
)
from atlas_brain.config import settings
from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _canonicalize_vendors(vendors: list[str]) -> list[str]:
    canonical: list[str] = []
    seen: set[str] = set()
    for vendor in vendors:
        normalized = _canonicalize_vendor(vendor)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        canonical.append(normalized)
    return canonical


def _polluted_company_signal_reason(
    row: dict[str, object],
) -> str | None:
    vendor_name = _canonicalize_vendor(str(row.get("vendor_name") or ""))
    company_name = str(row.get("company_name") or "").strip()
    exclusion_reason = _company_signal_exclusion_reason(
        company_name,
        current_vendor=vendor_name,
        source=row.get("source"),
        confidence_score=row.get("confidence_score"),
    )
    if exclusion_reason:
        return exclusion_reason
    if _high_intent_signal_evidence_enabled() and not _high_intent_row_has_signal_evidence(row):
        return "missing_signal_evidence"
    return None


async def _latest_core_report_date(pool) -> date | None:
    return await pool.fetchval(
        """
        SELECT report_date
        FROM b2b_intelligence
        WHERE report_type = 'core_run_complete'
        ORDER BY report_date DESC, created_at DESC
        LIMIT 1
        """
    )


async def _fetch_company_signal_rows(
    pool,
    *,
    vendors: list[str],
    window_days: int,
) -> list[dict[str, object]]:
    params: list[object] = [window_days]
    vendor_filter = ""
    if vendors:
        params.append([vendor.lower() for vendor in vendors])
        vendor_filter = "AND LOWER(cs.vendor_name) = ANY($2::text[])"
    # APPROVED-ENRICHMENT-READ: churn_signals.intent_to_leave, churn_signals.actively_evaluating, churn_signals.contract_renewal_mentioned, urgency_indicators.explicit_cancel_language, urgency_indicators.active_migration_language, urgency_indicators.active_evaluation_language, urgency_indicators.completed_switch_language
    # Reason: cleanup audit for accounts-in-motion signal pollution
    rows = await pool.fetch(
        f"""
        SELECT
            cs.company_name,
            cs.vendor_name,
            cs.review_id,
            cs.source,
            cs.urgency_score,
            cs.confidence_score,
            cs.first_seen_at,
            cs.last_seen_at,
            r.content_type,
            (r.enrichment->'churn_signals'->>'intent_to_leave')::boolean AS intent_to_leave,
            (r.enrichment->'churn_signals'->>'actively_evaluating')::boolean AS actively_evaluating,
            (r.enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean AS contract_renewal_mentioned,
            (r.enrichment->'urgency_indicators'->>'explicit_cancel_language')::boolean AS indicator_cancel,
            (r.enrichment->'urgency_indicators'->>'active_migration_language')::boolean AS indicator_migration,
            (r.enrichment->'urgency_indicators'->>'active_evaluation_language')::boolean AS indicator_evaluation,
            (r.enrichment->'urgency_indicators'->>'completed_switch_language')::boolean AS indicator_switch
        FROM b2b_company_signals cs
        LEFT JOIN b2b_reviews r ON r.id = cs.review_id
        WHERE cs.last_seen_at >= NOW() - make_interval(days => $1)
          {vendor_filter}
        ORDER BY cs.vendor_name, cs.last_seen_at DESC, cs.company_name
        """,
        *params,
    )
    return [
        {
            "company_name": row["company_name"],
            "vendor_name": _canonicalize_vendor(row["vendor_name"]),
            "review_id": str(row["review_id"]) if row["review_id"] else None,
            "source": row["source"],
            "urgency_score": float(row["urgency_score"]) if row["urgency_score"] is not None else None,
            "confidence_score": float(row["confidence_score"]) if row["confidence_score"] is not None else None,
            "first_seen_at": str(row["first_seen_at"]) if row["first_seen_at"] else None,
            "last_seen_at": str(row["last_seen_at"]) if row["last_seen_at"] else None,
            "content_type": row["content_type"],
            "intent_to_leave": row["intent_to_leave"],
            "actively_evaluating": row["actively_evaluating"],
            "contract_renewal_mentioned": row["contract_renewal_mentioned"],
            "indicator_cancel": row["indicator_cancel"],
            "indicator_migration": row["indicator_migration"],
            "indicator_evaluation": row["indicator_evaluation"],
            "indicator_switch": row["indicator_switch"],
        }
        for row in rows
    ]


async def _delete_company_signal_rows(
    pool,
    rows: list[dict[str, object]],
) -> int:
    if not rows:
        return 0
    deleted = 0
    seen_keys: set[tuple[str, str]] = set()
    async with pool.transaction() as conn:
        for row in rows:
            key = (
                str(row.get("company_name") or ""),
                str(row.get("vendor_name") or ""),
            )
            if not key[0] or not key[1] or key in seen_keys:
                continue
            seen_keys.add(key)
            await conn.execute(
                """
                DELETE FROM b2b_company_signals
                WHERE company_name = $1
                  AND vendor_name = $2
                """,
                key[0],
                key[1],
            )
            deleted += 1
    return deleted


async def _rebuild_account_intelligence(
    pool,
    *,
    vendors: list[str],
    as_of: date,
    analysis_window_days: int,
) -> int:
    if not vendors:
        return 0
    signal_lookup = await _fetch_existing_company_signals(
        pool,
        window_days=analysis_window_days,
    )
    blocked_names_by_vendor = _build_company_signal_blocked_names_by_vendor(vendors)
    rows: list[tuple[object, ...]] = []
    for vendor in vendors:
        account_intelligence = build_account_intelligence(
            vendor_name=vendor,
            persisted_signals=signal_lookup.get(vendor, []),
            blocked_names=blocked_names_by_vendor.get(vendor),
            analysis_window_days=analysis_window_days,
        )
        rows.append(
            (
                vendor,
                as_of,
                analysis_window_days,
                account_intelligence["schema_version"],
                json.dumps(account_intelligence, default=str),
            )
        )
    await pool.executemany(
        """
        INSERT INTO b2b_account_intelligence
            (vendor_name, as_of_date, analysis_window_days,
             schema_version, accounts)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        ON CONFLICT (vendor_name, as_of_date,
                     analysis_window_days, schema_version)
        DO UPDATE SET accounts = EXCLUDED.accounts,
                      created_at = NOW()
        """,
        rows,
    )
    return len(rows)


async def _rebuild_accounts_in_motion(
    *,
    vendors: list[str],
    as_of: date,
) -> dict[str, object]:
    task = SimpleNamespace(
        id=uuid4(),
        metadata={
            "_execution_id": str(uuid4()),
            "maintenance_run": True,
            "test_vendors": vendors,
        },
    )
    return await run_accounts_in_motion(task, as_of=as_of)


async def _main(args: argparse.Namespace) -> None:
    selected_vendors = _canonicalize_vendors(_parse_csv(args.vendors))
    window_days = int(args.window_days or settings.b2b_churn.intelligence_window_days)

    await init_database()
    pool = get_db_pool()
    try:
        as_of = await _latest_core_report_date(pool)
        if as_of is None:
            raise SystemExit("No core_run_complete marker found in b2b_intelligence")

        signal_rows = await _fetch_company_signal_rows(
            pool,
            vendors=selected_vendors,
            window_days=window_days,
        )
        polluted_rows: list[dict[str, object]] = []
        for row in signal_rows:
            reason = _polluted_company_signal_reason(row)
            if not reason:
                continue
            polluted_rows.append({**row, "pollution_reason": reason})

        affected_vendors = sorted(
            {
                str(row.get("vendor_name") or "")
                for row in polluted_rows
                if row.get("vendor_name")
            }
        )
        rebuild_vendors = affected_vendors or selected_vendors
        preview = polluted_rows[: max(1, int(args.preview_limit))]
        result: dict[str, object] = {
            "apply": bool(args.apply),
            "window_days": window_days,
            "core_as_of": str(as_of),
            "requested_vendors": selected_vendors,
            "polluted_row_count": len(polluted_rows),
            "affected_vendors": affected_vendors,
            "rebuild_vendors": rebuild_vendors,
            "preview_rows": preview,
        }

        if not args.apply:
            print(json.dumps(result, indent=2, default=str))
            return

        deleted = await _delete_company_signal_rows(pool, polluted_rows)
        rebuilt_account_intelligence = await _rebuild_account_intelligence(
            pool,
            vendors=rebuild_vendors,
            as_of=as_of,
            analysis_window_days=window_days,
        )
        if rebuild_vendors:
            rebuild_result = await _rebuild_accounts_in_motion(
                vendors=rebuild_vendors,
                as_of=as_of,
            )
        else:
            rebuild_result = {"_skip_synthesis": "No affected vendors to rebuild"}
        result.update(
            {
                "deleted_company_signals": deleted,
                "rebuilt_account_intelligence": rebuilt_account_intelligence,
                "accounts_in_motion_rebuild": rebuild_result,
            }
        )
        print(json.dumps(result, indent=2, default=str))
    finally:
        await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Remove polluted b2b_company_signals rows that poison accounts_in_motion "
            "and republish affected vendors."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete polluted rows and rebuild affected vendor artifacts",
    )
    parser.add_argument(
        "--vendors",
        help="Optional comma-separated vendor filter",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=settings.b2b_churn.intelligence_window_days,
        help="Analysis window to inspect for polluted signals",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=25,
        help="Max polluted rows to print in preview output",
    )
    asyncio.run(_main(parser.parse_args()))
