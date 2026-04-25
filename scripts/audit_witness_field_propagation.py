#!/usr/bin/env python3
"""Audit propagation of witness-quality fields into downstream JSON surfaces.

This is read-only. It compares the source witness table with persisted
reasoning/report JSON payloads and reports which witness-like objects preserve
the Phase 5/6/7 fields:

  grounding_status, phrase_polarity, phrase_subject, phrase_role,
  phrase_verbatim, pain_confidence

Usage:
  python scripts/audit_witness_field_propagation.py --days 7
  python scripts/audit_witness_field_propagation.py --days 14 --output /tmp/witness_field_propagation.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.services.reasoning_delivery_audit import summarize_witness_field_propagation
from atlas_brain.storage.database import close_database, get_db_pool, init_database


async def _main(*, days: int, row_limit: int, output: str | None) -> None:
    await init_database()
    pool = get_db_pool()
    try:
        result = await summarize_witness_field_propagation(
            pool,
            days=days,
            row_limit=row_limit,
        )
    finally:
        await close_database()

    payload = json.dumps(result, ensure_ascii=True, indent=2, default=str)
    print(payload)
    if output:
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="Lookback window for downstream surfaces.")
    parser.add_argument("--row-limit", type=int, default=250, help="Max rows per downstream table.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    asyncio.run(_main(days=args.days, row_limit=args.row_limit, output=args.output))
