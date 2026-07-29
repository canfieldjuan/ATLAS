#!/usr/bin/env python3
"""Drain unfinished EOM estimate bookings before rolling this app back.

Run this while the current Atlas version is still deployed and Calendar
credentials are still available. The command replays unfinished durable booking
operations through the current booking service. It exits non-zero if any
operation remains pending/projecting/retryable, which means rollback is not yet
safe because the database contact-state fence would outlive the code that can
complete or terminalize that row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Callable, Sequence

from atlas_brain.eom_api.config import funnel_settings
from atlas_brain.services.eom_lead_booking import EOMLeadBookingService
from atlas_brain.storage.database import get_db_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finish or terminalize unfinished EOM estimate-booking operations "
            "before rolling back the application."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum unfinished operations to attempt in this run.",
    )
    return parser


async def _run(
    args: argparse.Namespace,
    *,
    pool_provider: Callable[[], Any] = get_db_pool,
    service_factory: Callable[..., EOMLeadBookingService] = EOMLeadBookingService,
    config: Any = funnel_settings,
) -> int:
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    pool = pool_provider()
    await pool.initialize()
    try:
        service = service_factory(pool=pool, config=config)
        summary = await service.drain_unfinished_for_rollback(limit=args.limit)
    finally:
        await pool.close()
    print(json.dumps(summary, sort_keys=True, indent=2))
    if not summary["ok"]:
        print(
            "rollback unsafe: unfinished EOM estimate-booking operations remain",
        )
        return 1
    print("rollback drain complete: no unfinished EOM estimate bookings remain")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
