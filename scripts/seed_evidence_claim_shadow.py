#!/usr/bin/env python3
"""One-shot: run scoped synthesis to seed b2b_evidence_claims for canary triage.

Phase 9 step 7 helper. Picks a small canary vendor set, forces a fresh
synthesis run with the shadow-capture flag on, then dumps the audit
summary so the operator can inspect claim quality before the full
nightly cycle.

The shadow flag must be ON before this script imports atlas_brain.config
(pydantic BaseSettings caches at import time). The shell wrapper at the
bottom of this docstring sets it.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  ATLAS_B2B_CHURN_EVIDENCE_CLAIM_SHADOW_ENABLED=true \\
      python scripts/seed_evidence_claim_shadow.py

  # Pick vendors explicitly:
  ATLAS_B2B_CHURN_EVIDENCE_CLAIM_SHADOW_ENABLED=true \\
      python scripts/seed_evidence_claim_shadow.py --vendors Pipedrive Monday.com ClickUp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)


_DEFAULT_CANARY_VENDORS = ["Pipedrive", "Monday.com", "ClickUp"]


def _ensure_flag_on() -> None:
    """Hard-fail if the operator forgot to set the shadow flag.

    Settings are cached at first import; flipping the env var after
    import would silently no-op and the script would appear to succeed
    while writing zero claim rows."""
    if os.environ.get("ATLAS_B2B_CHURN_EVIDENCE_CLAIM_SHADOW_ENABLED", "").lower() not in (
        "true",
        "1",
        "yes",
        "on",
    ):
        sys.stderr.write(
            "ERROR: ATLAS_B2B_CHURN_EVIDENCE_CLAIM_SHADOW_ENABLED must be 'true' "
            "in the environment BEFORE running this script. The settings module "
            "caches at import time; setting it later would no-op.\n"
        )
        sys.exit(2)


async def _main(args: argparse.Namespace) -> int:
    _ensure_flag_on()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("seed_evidence_claim_shadow")

    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis
    from atlas_brain.config import settings
    from atlas_brain.services.reasoning_delivery_audit import summarize_claim_validation
    from atlas_brain.storage.database import close_database, get_db_pool, init_database

    if not settings.b2b_churn.evidence_claim_shadow_enabled:
        sys.stderr.write(
            "ERROR: settings.b2b_churn.evidence_claim_shadow_enabled is False "
            "after env load. Did pydantic miss the env var?\n"
        )
        return 2

    vendors = args.vendors or _DEFAULT_CANARY_VENDORS
    logger.info("Canary vendors: %s", vendors)

    await init_database()
    try:
        task = SimpleNamespace(
            metadata={
                "test_vendors": vendors,
                "force": True,
                "force_cross_vendor": False,
            }
        )
        logger.info("Running scoped synthesis with force=True...")
        result = await b2b_reasoning_synthesis.run(task)

        logger.info(
            "Synthesis result: reasoned=%d failed=%d",
            (result or {}).get("vendors_reasoned", 0) or 0,
            (result or {}).get("vendors_failed", 0) or 0,
        )

        pool = get_db_pool()
        as_of = date.today()
        summary = await summarize_claim_validation(pool, as_of_date=as_of)
        if args.json:
            print(json.dumps(summary, default=str, indent=2))
        else:
            scope = summary["scope"]
            totals = summary["totals"]
            print(f"\n=== Shadow audit {summary['as_of_date']} ===")
            print(
                f"rows={scope['total_rows']} vendors={scope['distinct_vendors']} "
                f"artifacts={scope['distinct_artifacts']}"
            )
            print(
                f"valid={totals['valid']} invalid={totals['invalid']} "
                f"cannot_validate={totals['cannot_validate']}"
            )
            print("\nby_claim_type:")
            for ct, b in sorted(summary["by_claim_type"].items()):
                print(
                    f"  {ct:<35} total={b['total']:>3} valid={b['valid']:>3} "
                    f"invalid={b['invalid']:>3} cannot_validate={b['cannot_validate']:>3}"
                )
            print("\nby_vendor:")
            for v in summary["by_vendor"]:
                print(
                    f"  {v['vendor_name']:<28} total={v['total']:>3} "
                    f"valid={v['valid']:>3} invalid={v['invalid']:>3} "
                    f"cannot_validate={v['cannot_validate']:>3}"
                )
            print("\nrejection_reasons (invalid) by claim_type:")
            for ct, entries in sorted(summary["rejection_reasons_by_claim_type"].items()):
                if not entries:
                    continue
                print(f"  {ct}:")
                for e in entries:
                    print(f"    {e['count']:>3}  {e['rejection_reason']}")
            print("\ncannot_validate reasons by claim_type:")
            for ct, entries in sorted(summary["cannot_validate_reasons_by_claim_type"].items()):
                if not entries:
                    continue
                print(f"  {ct}:")
                for e in entries:
                    print(f"    {e['count']:>3}  {e['rejection_reason']}")
        return 0
    finally:
        await close_database()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendors",
        nargs="+",
        default=None,
        help=f"Vendor names to synthesize (default: {_DEFAULT_CANARY_VENDORS})",
    )
    parser.add_argument(
        "--json", action="store_true", help="Dump audit summary as JSON"
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
