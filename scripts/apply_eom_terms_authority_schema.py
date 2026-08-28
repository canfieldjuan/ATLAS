#!/usr/bin/env python3
"""Preflight or apply controlled EOM Terms-authority migration 396."""

from __future__ import annotations

import argparse
import asyncio
import sys

from apply_eom_first_clean_completion_schema import (
    TERMS_AUTHORITY_MIGRATION_NAME,
    _main,
)


def _terms_argv(argv: list[str]) -> list[str]:
    """Pin the shared controlled runner to the Terms migration."""

    parser = argparse.ArgumentParser(
        description="Preflight or apply controlled EOM Terms-authority schema."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migration 396 after the read-only DBA preflight.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the redacted preflight/result payload as JSON.",
    )
    args = parser.parse_args(argv)
    forwarded = ["--migration", TERMS_AUTHORITY_MIGRATION_NAME]
    if args.apply:
        forwarded.append("--apply")
    if args.json:
        forwarded.append("--json")
    return forwarded


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_terms_argv(sys.argv[1:]))))
