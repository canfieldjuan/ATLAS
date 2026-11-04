#!/usr/bin/env python3
"""Preflight or apply controlled EOM operator-contact receipt index migration 401."""

from __future__ import annotations

import argparse
import asyncio
import sys

from apply_eom_first_clean_completion_schema import (
    OPERATOR_CONTACT_RECEIPT_INDEX_MIGRATION_NAME,
    _main,
)


def _receipt_index_argv(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or apply the controlled EOM operator-contact receipt index."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migration 401 after the read-only DBA preflight.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the redacted preflight/result payload as JSON.",
    )
    args = parser.parse_args(argv)
    forwarded = ["--migration", OPERATOR_CONTACT_RECEIPT_INDEX_MIGRATION_NAME]
    if args.apply:
        forwarded.append("--apply")
    if args.json:
        forwarded.append("--json")
    return forwarded


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_receipt_index_argv(sys.argv[1:]))))
