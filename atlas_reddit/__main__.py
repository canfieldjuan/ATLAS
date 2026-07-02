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

from pydantic import ValidationError

from .config import (
    MAX_DORMANT_AFTER_HOURS,
    MAX_FRESHNESS_HOURS,
    MAX_HISTORY_LIMIT,
    MAX_PACE_SECONDS,
    MAX_PER_SUBREDDIT_LIMIT,
    RedditListeningSettings,
    WatchlistError,
    load_watchlist,
)
from .digest import write_digest
from .reddit_client import (
    PrawDeletionSource,
    PrawHistorySource,
    PrawListingSource,
    RedditAuthError,
)
from .poller import poll_once
from .tracker import track_once
from .purge import purge_once
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

    poll = subparsers.add_parser(
        "poll", help="One read-only polling pass over the watchlist."
    )
    poll.add_argument(
        "--db",
        type=Path,
        default=defaults.db_path,
        help=f"SQLite state file (default: {defaults.db_path})",
    )
    poll.add_argument(
        "--watchlist",
        type=Path,
        default=defaults.watchlist_path,
        help=f"Watchlist TOML (default: {defaults.watchlist_path})",
    )
    poll.add_argument(
        "--limit-per-subreddit",
        type=int,
        default=defaults.per_subreddit_limit,
        help=f"Newest posts fetched per subreddit (default: {defaults.per_subreddit_limit})",
    )
    poll.add_argument(
        "--freshness-hours",
        type=int,
        default=defaults.freshness_hours,
        help=f"Admit posts younger than this (default: {defaults.freshness_hours}h)",
    )
    poll.add_argument(
        "--min-score",
        type=_finite_float,
        default=defaults.poll_min_score,
        help=f"Store candidates at or above this score (default: {defaults.poll_min_score})",
    )
    poll.add_argument(
        "--pace-seconds",
        type=_finite_float,
        default=defaults.pace_seconds,
        help=f"Sleep between subreddit fetches (default: {defaults.pace_seconds}s)",
    )

    track = subparsers.add_parser(
        "track", help="One read-only reply-tracking pass over own threads."
    )
    track.add_argument(
        "--db",
        type=Path,
        default=defaults.db_path,
        help=f"SQLite state file (default: {defaults.db_path})",
    )
    track.add_argument(
        "--history-limit",
        type=int,
        default=defaults.history_limit,
        help=f"Own recent items fetched (default: {defaults.history_limit})",
    )
    track.add_argument(
        "--dormant-after-hours",
        type=int,
        default=defaults.dormant_after_hours,
        help=f"Quiet window before a thread sleeps (default: {defaults.dormant_after_hours}h)",
    )
    track.add_argument(
        "--pace-seconds",
        type=_finite_float,
        default=defaults.pace_seconds,
        help=f"Sleep between thread fetches (default: {defaults.pace_seconds}s)",
    )

    purge = subparsers.add_parser(
        "purge",
        help="Deletion-compliance pass: drop stored content that is "
        "deleted/removed/missing on Reddit (run at least every 48h).",
    )
    purge.add_argument(
        "--db",
        type=Path,
        default=defaults.db_path,
        help=f"SQLite state file (default: {defaults.db_path})",
    )
    purge.add_argument(
        "--pace-seconds",
        type=_finite_float,
        default=defaults.pace_seconds,
        help=f"Sleep between check batches (default: {defaults.pace_seconds}s)",
    )
    purge.add_argument(
        "--digest-dir",
        type=Path,
        default=defaults.digest_dir,
        help=(
            "Digest directory; rendered digests are removed when content "
            f"was purged (default: {defaults.digest_dir})"
        ),
    )

    mark = subparsers.add_parser(
        "mark-read", help="Mark one reply as seen (drops it from the digest)."
    )
    mark.add_argument("reply_id", help="Reply id, e.g. t1_abc123")
    mark.add_argument(
        "--db",
        type=Path,
        default=defaults.db_path,
        help=f"SQLite state file (default: {defaults.db_path})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        settings = RedditListeningSettings()
    except ValidationError as exc:
        # An env typo (ATLAS_REDDIT_FRESHNESS_HOURS=abc) is an operator
        # error for EVERY command, not a traceback.
        print(f"error: invalid ATLAS_REDDIT_* environment: {exc}", file=sys.stderr)
        return 2
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

    if args.command == "poll":
        if not 1 <= args.limit_per_subreddit <= MAX_PER_SUBREDDIT_LIMIT:
            parser.error(
                f"--limit-per-subreddit must be 1..{MAX_PER_SUBREDDIT_LIMIT}, "
                f"got {args.limit_per_subreddit}"
            )
        if not 1 <= args.freshness_hours <= MAX_FRESHNESS_HOURS:
            parser.error(
                f"--freshness-hours must be 1..{MAX_FRESHNESS_HOURS}, "
                f"got {args.freshness_hours}"
            )
        if not 0 <= args.pace_seconds <= MAX_PACE_SECONDS:
            parser.error(
                f"--pace-seconds must be 0..{MAX_PACE_SECONDS}, got {args.pace_seconds}"
            )
        if args.min_score < 0:
            # Same contract as the poll_min_score setting (ge=0): a CLI
            # override must not broaden what gets stored vs the env path.
            parser.error(f"--min-score must be >= 0, got {args.min_score}")
        try:
            watchlist = load_watchlist(args.watchlist)
            source = PrawListingSource(settings)
            with ListeningStore(args.db) as store:
                stats = poll_once(
                    store,
                    watchlist,
                    source,
                    now=int(datetime.now(tz=timezone.utc).timestamp()),
                    freshness_hours=args.freshness_hours,
                    per_subreddit_limit=args.limit_per_subreddit,
                    min_final_score=args.min_score,
                    pace_seconds=args.pace_seconds,
                )
        except (StoreError, WatchlistError, RedditAuthError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"fetched={stats.fetched} admitted={stats.admitted} "
            f"not_text={stats.skipped_not_text} stale={stats.skipped_stale} "
            f"below_floor={stats.skipped_below_floor} errors={len(stats.errors)}"
        )
        for line in stats.errors:
            print(f"warning: {line}", file=sys.stderr)
        return 0 if not stats.errors else 1

    if args.command == "track":
        if not 1 <= args.history_limit <= MAX_HISTORY_LIMIT:
            parser.error(
                f"--history-limit must be 1..{MAX_HISTORY_LIMIT}, got {args.history_limit}"
            )
        if not 1 <= args.dormant_after_hours <= MAX_DORMANT_AFTER_HOURS:
            parser.error(
                f"--dormant-after-hours must be 1..{MAX_DORMANT_AFTER_HOURS}, "
                f"got {args.dormant_after_hours}"
            )
        if not 0 <= args.pace_seconds <= MAX_PACE_SECONDS:
            parser.error(
                f"--pace-seconds must be 0..{MAX_PACE_SECONDS}, got {args.pace_seconds}"
            )
        try:
            source = PrawHistorySource(settings)
            with ListeningStore(args.db) as store:
                stats = track_once(
                    store,
                    source,
                    now=int(datetime.now(tz=timezone.utc).timestamp()),
                    history_limit=args.history_limit,
                    dormant_after_hours=args.dormant_after_hours,
                    pace_seconds=args.pace_seconds,
                )
        except (StoreError, RedditAuthError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"discovered={stats.threads_discovered} checked={stats.threads_checked} "
            f"woken={stats.threads_woken} dormant={stats.threads_marked_dormant} "
            f"new_replies={stats.replies_new} replayed={stats.replies_replayed} "
            f"errors={len(stats.errors)}"
        )
        for line in stats.errors:
            print(f"warning: {line}", file=sys.stderr)
        return 0 if not stats.errors else 1

    if args.command == "purge":
        if not 0 <= args.pace_seconds <= MAX_PACE_SECONDS:
            parser.error(
                f"--pace-seconds must be 0..{MAX_PACE_SECONDS}, got {args.pace_seconds}"
            )
        try:
            source = PrawDeletionSource(settings)
            with ListeningStore(args.db) as store:
                stats = purge_once(
                    store,
                    source,
                    now=int(datetime.now(tz=timezone.utc).timestamp()),
                    pace_seconds=args.pace_seconds,
                    digest_dir=args.digest_dir,
                )
        except (StoreError, RedditAuthError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"checked={stats.checked} purged_candidates={stats.purged_candidates} "
            f"purged_replies={stats.purged_replies} "
            f"digests_removed={stats.digests_removed} errors={len(stats.errors)}"
        )
        for line in stats.errors:
            print(f"warning: {line}", file=sys.stderr)
        return 0 if not stats.errors else 1

    if args.command == "mark-read":
        try:
            with ListeningStore(args.db) as store:
                store.mark_reply_seen(args.reply_id)
        except (StoreError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"marked seen: {args.reply_id}")
        return 0

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
