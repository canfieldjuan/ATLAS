"""Reply tracking: own history -> tracked threads -> replies (S5, #1934).

Same orchestration posture as the poller: the transport arrives as a
:class:`~atlas_reddit.reddit_client.HistorySource`, the clock as ``now``,
pacing as an injectable ``sleep`` -- fully deterministic under test.

Lifecycle:

1. Discovery: the operator's recent comments and submissions are grouped
   by thread and upserted (the store's set-union merge grows
   ``my_comment_ids`` and never drops known ids). A thread with fresh
   own activity is woken if it was dormant -- rediscovery is the only
   wake signal, so dormancy means "stop polling until I engage again".
2. Reply fetch: ACTIVE threads only, paced politely. Replies insert
   replay-safe (duplicate reply ids are ignored; integrity violations
   raise); unseen state drives the digest's warm-replies section.
3. Dormancy: after fetching, a thread goes dormant when its newest known
   activity (own activity seen this pass or newest stored reply) is
   older than the quiet window -- and also when no activity timestamp is
   known at all, which happens exactly when the operator's engagement
   has aged out of the recent-history window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .reddit_client import HistorySource
from .store import ListeningStore


@dataclass
class TrackStats:
    threads_discovered: int = 0
    threads_checked: int = 0
    threads_woken: int = 0
    threads_marked_dormant: int = 0
    replies_new: int = 0
    replies_replayed: int = 0
    errors: list[str] = field(default_factory=list)


def track_once(
    store: ListeningStore,
    source: HistorySource,
    *,
    now: int,
    history_limit: int,
    dormant_after_hours: int,
    pace_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> TrackStats:
    """One polite tracking pass: discover own threads, fetch replies on
    active ones, apply the dormancy lifecycle."""
    stats = TrackStats()
    cutoff = now - dormant_after_hours * 3600

    comments = source.fetch_my_recent_comments(limit=history_limit)
    submissions = source.fetch_my_recent_submissions(limit=history_limit)

    fresh_activity: dict[str, int] = {}
    comment_ids_by_thread: dict[str, list[str]] = {}
    for activity in [*comments, *submissions]:
        fresh_activity[activity.thread_id] = max(
            fresh_activity.get(activity.thread_id, 0), activity.created_utc
        )
        if activity.item_id.startswith("t1_"):
            comment_ids_by_thread.setdefault(activity.thread_id, []).append(
                activity.item_id
            )

    known_before = {
        thread.thread_id
        for thread in store.list_tracked_threads(include_dormant=True)
    }
    dormant_before = {
        thread.thread_id
        for thread in store.list_tracked_threads(include_dormant=True)
        if thread.dormant
    }

    for thread_id, newest in fresh_activity.items():
        store.upsert_tracked_thread(
            thread_id=thread_id,
            my_comment_ids=tuple(comment_ids_by_thread.get(thread_id, ())),
            checked_at=now,
        )
        if thread_id not in known_before:
            stats.threads_discovered += 1
        # Rediscovery with activity inside the quiet window is the wake
        # signal for a dormant thread.
        if thread_id in dormant_before and newest >= cutoff:
            store.set_thread_dormant(thread_id, False)
            stats.threads_woken += 1

    active = store.list_tracked_threads(include_dormant=False)
    for index, thread in enumerate(active):
        if index and pace_seconds > 0:
            sleep(pace_seconds)
        stats.threads_checked += 1
        try:
            replies = source.fetch_thread_replies(
                thread.thread_id, my_comment_ids=frozenset(thread.my_comment_ids)
            )
        except Exception as exc:  # noqa: BLE001 -- one bad thread must not
            # abort the pass; the error is surfaced, not swallowed.
            stats.errors.append(f"{thread.thread_id}: {exc}")
            continue
        for reply in replies:
            inserted = store.insert_reply(
                reply_id=reply.reply_id,
                thread_id=reply.thread_id,
                parent_id=reply.parent_id,
                author=reply.author,
                body=reply.body,
                created_utc=reply.created_utc,
                is_reply_to_me=reply.is_reply_to_me,
            )
            if inserted:
                stats.replies_new += 1
            else:
                stats.replies_replayed += 1

        newest_reply = max(
            (r.created_utc for r in store.list_replies(thread_id=thread.thread_id)),
            default=None,
        )
        newest_known = max(
            value
            for value in (newest_reply, fresh_activity.get(thread.thread_id))
            if value is not None
        ) if (newest_reply is not None or thread.thread_id in fresh_activity) else None

        if newest_known is None or newest_known < cutoff:
            store.set_thread_dormant(thread.thread_id, True)
            stats.threads_marked_dormant += 1

    return stats
