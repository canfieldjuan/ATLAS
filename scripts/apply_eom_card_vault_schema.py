#!/usr/bin/env python3
"""Preflight or apply controlled EOM card-vault migration 398."""

from __future__ import annotations

import argparse
import asyncio
import sys

from apply_eom_first_clean_completion_schema import CARD_VAULT_MIGRATION_NAME, _main


def _card_vault_argv(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Preflight or apply controlled EOM card-vault schema."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migration 398 after the read-only DBA preflight.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the redacted preflight/result payload as JSON.",
    )
    args = parser.parse_args(argv)
    forwarded = ["--migration", CARD_VAULT_MIGRATION_NAME]
    if args.apply:
        forwarded.append("--apply")
    if args.json:
        forwarded.append("--json")
    return forwarded


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_card_vault_argv(sys.argv[1:]))))
