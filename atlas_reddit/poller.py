"""Radar polling: listings -> filters -> scorer -> store (S4, #1934).

Pure orchestration over injected boundaries: the transport arrives as a
:class:`~atlas_reddit.reddit_client.ListingSource`, the clock as ``now``,
and pacing as an injectable ``sleep`` -- so the whole loop is
deterministic under test with a fake source and a recording sleep.

Filters here decide what is worth STORING (fresh, text post, scores
above the floor). What SURFACES stays the digest's job: triage state
(seen/dismissed/responded) is preserved by the store's replay-safe
upsert and filtered at read time, so re-polling can never resurrect a
dismissed thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .config import Watchlist
from .reddit_client import ListingSource
from .scoring import score_post
from .store import ListeningStore


@dataclass
class PollStats:
    fetched: int = 0
    admitted: int = 0
    skipped_not_text: int = 0
    skipped_stale: int = 0
    skipped_below_floor: int = 0
    errors: list[str] = field(default_factory=list)


def poll_once(
    store: ListeningStore,
    watchlist: Watchlist,
    source: ListingSource,
    *,
    now: int,
    freshness_hours: int,
    per_subreddit_limit: int,
    min_final_score: float,
    pace_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> PollStats:
    """One polite polling pass over every watched subreddit."""
    stats = PollStats()
    cutoff = now - freshness_hours * 3600
    subreddits = list(watchlist.subreddits)

    for index, entry in enumerate(subreddits):
        if index and pace_seconds > 0:
            sleep(pace_seconds)
        try:
            posts = source.fetch_new(entry.name, limit=per_subreddit_limit)
        except Exception as exc:  # noqa: BLE001 -- one bad subreddit must not
            # abort the whole pass; the error is surfaced, not swallowed.
            stats.errors.append(f"r/{entry.name}: {exc}")
            continue

        for post in posts:
            stats.fetched += 1
            if not post.is_self:
                stats.skipped_not_text += 1
                continue
            if post.created_utc < cutoff:
                stats.skipped_stale += 1
                continue
            breakdown = score_post(
                title=post.title,
                body=post.selftext,
                subreddit_weight=entry.weight,
                watchlist=watchlist,
            )
            if breakdown.total <= 0 or breakdown.total < min_final_score:
                stats.skipped_below_floor += 1
                continue
            store.upsert_candidate(
                post_id=post.post_id,
                subreddit=post.subreddit,
                title=post.title,
                url=post.url,
                author=post.author,
                created_utc=post.created_utc,
                reddit_score=post.score,
                num_comments=post.num_comments,
                keyword_score=breakdown.total / entry.weight,
                final_score=breakdown.total,
                matched_topics=tuple(hit.topic for hit in breakdown.topic_hits),
                observed_at=now,
            )
            stats.admitted += 1

    return stats
