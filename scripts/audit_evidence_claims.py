#!/usr/bin/env python3
"""Pretty-print or JSON-dump the b2b_evidence_claims audit summary.

Phase 9 step 6 CLI. Wraps
``atlas_brain.services.reasoning_delivery_audit.summarize_claim_validation``
for ad-hoc canary triage. The daily autonomous task
(``atlas_brain.autonomous.tasks.b2b_evidence_claim_audit``) runs the same
summary on a schedule and emits to the operator inbox; this script is for
when you want to look at any single day on demand.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate

  # today, pretty
  python scripts/audit_evidence_claims.py
  # specific day
  python scripts/audit_evidence_claims.py --as-of 2026-04-25
  # raw JSON for piping into jq
  python scripts/audit_evidence_claims.py --as-of 2026-04-25 --json
  # narrower invalid examples
  python scripts/audit_evidence_claims.py --invalid-examples 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.services.reasoning_delivery_audit import summarize_claim_validation
from atlas_brain.storage.database import close_database, get_db_pool, init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _print_pretty(summary: dict) -> None:
    scope = summary.get("scope", {})
    totals = summary.get("totals", {})
    print(f"\n=== EvidenceClaim audit — {summary['as_of_date']} ===\n")
    print(
        f"Scope: rows={scope.get('total_rows', 0)} "
        f"vendors={scope.get('distinct_vendors', 0)} "
        f"artifacts={scope.get('distinct_artifacts', 0)}"
    )
    print(
        f"Totals: valid={totals.get('valid', 0)} "
        f"invalid={totals.get('invalid', 0)} "
        f"cannot_validate={totals.get('cannot_validate', 0)}"
    )

    print("\n-- by claim_type --")
    by_type = summary.get("by_claim_type", {})
    for ct, bucket in sorted(by_type.items()):
        print(
            f"  {ct}: total={bucket['total']:>4} "
            f"valid={bucket['valid']:>4} "
            f"invalid={bucket['invalid']:>4} "
            f"cannot_validate={bucket['cannot_validate']:>4}"
        )

    print("\n-- top rejection reasons (invalid) --")
    for ct, entries in sorted(summary.get("rejection_reasons_by_claim_type", {}).items()):
        if not entries:
            continue
        print(f"  {ct}:")
        for entry in entries:
            print(f"    {entry['count']:>4}  {entry['rejection_reason']}")

    print("\n-- top cannot_validate reasons --")
    for ct, entries in sorted(summary.get("cannot_validate_reasons_by_claim_type", {}).items()):
        if not entries:
            continue
        print(f"  {ct}:")
        for entry in entries:
            print(f"    {entry['count']:>4}  {entry['rejection_reason']}")

    print("\n-- by vendor (top 20) --")
    for vendor in summary.get("by_vendor", [])[:20]:
        print(
            f"  {vendor['vendor_name']:<28} "
            f"total={vendor['total']:>4} "
            f"valid={vendor['valid']:>4} "
            f"invalid={vendor['invalid']:>4} "
            f"cannot_validate={vendor['cannot_validate']:>4}"
        )

    print("\n-- by source --")
    for src, bucket in sorted(summary.get("by_source", {}).items()):
        print(
            f"  {src:<14} total={bucket['total']:>4} "
            f"valid={bucket['valid']:>4} "
            f"invalid={bucket['invalid']:>4} "
            f"cannot_validate={bucket['cannot_validate']:>4}"
        )

    print("\n-- by pain_category --")
    for cat, bucket in sorted(summary.get("by_pain_category", {}).items()):
        print(
            f"  {cat:<26} total={bucket['total']:>4} "
            f"valid={bucket['valid']:>4} "
            f"invalid={bucket['invalid']:>4} "
            f"cannot_validate={bucket['cannot_validate']:>4}"
        )

    print("\n-- by enrichment_schema_version --")
    for sv, bucket in sorted(summary.get("by_schema_version", {}).items()):
        print(
            f"  {sv:<10} total={bucket['total']:>4} "
            f"valid={bucket['valid']:>4} "
            f"invalid={bucket['invalid']:>4} "
            f"cannot_validate={bucket['cannot_validate']:>4}"
        )

    examples = summary.get("invalid_examples", [])
    if examples:
        print("\n-- invalid examples (sample) --")
        for ex in examples:
            preview = (ex.get("excerpt_preview") or "").strip()
            if len(preview) > 90:
                preview = preview[:90] + "..."
            print(
                f"  [{ex['claim_type']}/{ex['rejection_reason']}] "
                f"{ex['vendor_name']}: {preview}"
            )

    print()


async def _run(args: argparse.Namespace) -> int:
    await init_database()
    try:
        pool = get_db_pool()
        as_of = (
            date.fromisoformat(args.as_of) if args.as_of else date.today()
        )
        summary = await summarize_claim_validation(
            pool,
            as_of_date=as_of,
            invalid_examples_per_reason=int(args.invalid_examples),
            rejection_reasons_per_claim_type=int(args.rejection_reasons),
        )
        if args.json:
            print(json.dumps(summary, default=str, indent=2))
        else:
            _print_pretty(summary)
        return 0
    finally:
        await close_database()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="ISO date (default: today)")
    parser.add_argument(
        "--json", action="store_true", help="Print raw JSON instead of pretty text"
    )
    parser.add_argument(
        "--invalid-examples",
        type=int,
        default=3,
        help="Sample N invalid rows per (claim_type, rejection_reason) (default: 3)",
    )
    parser.add_argument(
        "--rejection-reasons",
        type=int,
        default=10,
        help="Top N rejection_reasons per claim_type (default: 10)",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
