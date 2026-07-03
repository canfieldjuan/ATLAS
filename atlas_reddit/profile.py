"""Own-profile sync: profile listing -> own_posts store.

The operator's profile ("home page") is the one feed that already carries
every post they made and the subreddit each landed in. This pass mirrors
that feed into the local ``own_posts`` table so all their posts live in
one inspectable place, each readable individually (together with the
replies the tracker collects on the same thread fullnames).

Same orchestration posture as the poller and tracker: the transport
arrives as a :class:`~atlas_reddit.reddit_client.ProfileSource` and the
clock as ``now`` -- fully deterministic under test.

Unlike the radar there are NO admission filters: the radar's
is_self/freshness/score gates triage strangers' posts, but every own post
belongs in the profile mirror (link posts and old posts included).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .reddit_client import ProfileSource
from .store import ListeningStore


@dataclass
class ProfileStats:
    fetched: int = 0
    new: int = 0
    refreshed: int = 0
    errors: list[str] = field(default_factory=list)


def sync_profile_once(
    store: ListeningStore,
    source: ProfileSource,
    *,
    now: int,
    limit: int,
) -> ProfileStats:
    """One sync pass over the operator's own submission listing (a single
    paginated PRAW listing request -- same request-budget family as the
    tracker's history fetches)."""
    stats = ProfileStats()
    try:
        posts = source.fetch_my_posts(limit=limit)
    except Exception as exc:  # noqa: BLE001 -- a transport failure is a
        # pass-level error surfaced to the operator, never a traceback.
        stats.errors.append(f"profile fetch: {exc}")
        return stats

    for post in posts:
        stats.fetched += 1
        inserted = store.upsert_own_post(
            post_id=post.post_id,
            subreddit=post.subreddit,
            title=post.title,
            url=post.url,
            created_utc=post.created_utc,
            reddit_score=post.score,
            num_comments=post.num_comments,
            selftext=post.selftext,
            observed_at=now,
        )
        if inserted:
            stats.new += 1
        else:
            stats.refreshed += 1

    return stats
