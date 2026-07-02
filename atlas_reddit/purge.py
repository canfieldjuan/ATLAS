"""Deletion-compliance purge (S6, #1934 -- final arc slice).

The epic's compliance contract: stored third-party content that is later
deleted or removed on Reddit (or 404s entirely) must be purged locally,
with a target cleanup window of 48 hours. This module is the purge JOB
on top of the fields S2 shipped: it re-checks every stored candidate
post and reply through the DeletionSource boundary (one batched read
per 100 items), deletes the gone rows, and records each purge in
``purge_log`` with the detection reason.

The operator cadence note lives in the runbook: run
``python -m atlas_reddit purge`` at least every 48 hours while any
long-lived content is stored. Tracked-thread rows themselves are only
ids and the operator's own comment ids -- no third-party content -- so
they are retained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .reddit_client import DeletionSource
from .store import ListeningStore

BATCH_SIZE = 100  # reddit info() accepts up to 100 fullnames per request


@dataclass
class PurgeStats:
    checked: int = 0
    purged_candidates: int = 0
    purged_replies: int = 0
    errors: list[str] = field(default_factory=list)


def purge_once(
    store: ListeningStore,
    source: DeletionSource,
    *,
    now: int,
    pace_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> PurgeStats:
    """One deletion-compliance pass over everything stored."""
    stats = PurgeStats()

    candidates = {c.post_id for c in store.list_candidates()}
    replies = {r.reply_id for r in store.list_replies()}
    all_items = sorted(candidates | replies)

    batches = [
        all_items[i : i + BATCH_SIZE] for i in range(0, len(all_items), BATCH_SIZE)
    ]
    for index, batch in enumerate(batches):
        if index and pace_seconds > 0:
            sleep(pace_seconds)
        stats.checked += len(batch)
        try:
            gone = source.fetch_gone_items(list(batch))
        except Exception as exc:  # noqa: BLE001 -- one failed batch must not
            # abort the compliance pass; the error is surfaced, not
            # swallowed, and the remaining batches still run.
            stats.errors.append(f"batch {index + 1}/{len(batches)}: {exc}")
            continue
        for item_id, reason in sorted(gone.items()):
            if item_id in candidates and store.purge_candidate(item_id):
                item_type = "candidate"
                stats.purged_candidates += 1
            elif item_id in replies and store.purge_reply(item_id):
                item_type = "reply"
                stats.purged_replies += 1
            else:
                continue  # raced away already; nothing was removed
            store.record_purge(
                item_id=item_id,
                item_type=item_type,
                deleted_detected_at=now,
                purged_at=now,
                reason=reason,
            )

    return stats
