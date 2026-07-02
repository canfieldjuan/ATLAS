"""CLI entry point: ``python -m atlas_reddit``.

S3 ships the ``digest`` command; later slices add ``poll`` (S4),
``track`` (S5), and ``purge`` (S6). The wall clock enters exactly here
(``--date`` defaults to today, UTC) -- everything below the CLI takes
dates and timestamps as data.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import RedditListeningSettings
from .digest import write_digest
from .store import ListeningStore, StoreError

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_date(value: str) -> str:
    if not _DATE_RE.match(value):
        raise argparse.ArgumentTypeError(
            f"date must be YYYY-MM-DD, got {value!r}"
        )
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: {exc}") from exc
    return value


def _finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float {value!r}") from exc
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(
            f"--min-score must be finite, got {value!r}"
        )
    return number


def _build_parser(defaults: RedditListeningSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m atlas_reddit",
        description="Read-only Reddit listening tool (Resolution Audit).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    digest = subparsers.add_parser(
        "digest", help="Render today's Markdown digest from local state."
    )
    digest.add_argument(
        "--db",
        type=Path,
        default=defaults.db_path,
        help=f"SQLite state file (default: {defaults.db_path})",
    )
    digest.add_argument(
        "--digest-dir",
        type=Path,
        default=defaults.digest_dir,
        help=f"Output directory (default: {defaults.digest_dir})",
    )
    digest.add_argument(
        "--date",
        type=_valid_date,
        default=None,
        help="Digest date YYYY-MM-DD (default: today, UTC)",
    )
    digest.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum radar candidates (default: 20)",
    )
    digest.add_argument(
        "--min-score",
        type=_finite_float,
        default=None,
        help="Minimum final score for radar candidates (default: no floor)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = RedditListeningSettings()
    parser = _build_parser(settings)
    args = parser.parse_args(argv)

    if args.command == "digest":
        generated_on = args.date or datetime.now(tz=timezone.utc).date().isoformat()
        if args.limit < 1:
            parser.error(f"--limit must be at least 1, got {args.limit}")
        try:
            with ListeningStore(args.db) as store:
                path = write_digest(
                    store,
                    digest_dir=args.digest_dir,
                    generated_on=generated_on,
                    limit=args.limit,
                    min_final_score=args.min_score,
                )
        except (StoreError, OSError) as exc:
            # One operator-error contract: bad state OR bad output path
            # both report cleanly on stderr with exit 2, never a traceback.
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(path)
        return 0

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
