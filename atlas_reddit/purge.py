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

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .reddit_client import DeletionSource
from .store import ListeningStore

BATCH_SIZE = 100  # reddit info() accepts up to 100 fullnames per request

# Every stored id must be a Reddit fullname of the RIGHT KIND for its
# table: candidates hold submissions (t3_), replies hold comments (t1_).
# Anything else -- including other valid fullname kinds like t2_ users --
# cannot be liveness-checked for that table and MUST NOT be treated as
# missing: never delete on a data-shape mismatch.
_KIND_RE = {
    "candidate": re.compile(r"^t3_[a-z0-9]+$"),
    "reply": re.compile(r"^t1_[a-z0-9]+$"),
}


@dataclass
class PurgeStats:
    checked: int = 0
    purged_candidates: int = 0
    purged_replies: int = 0
    digests_removed: int = 0
    errors: list[str] = field(default_factory=list)


def purge_once(
    store: ListeningStore,
    source: DeletionSource,
    *,
    now: int,
    pace_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    digest_dir: Path | None = None,
) -> PurgeStats:
    """One deletion-compliance pass over everything stored. When content
    was purged and ``digest_dir`` is provided, previously rendered digest
    files are removed too (they are regenerable projections that may
    contain the purged content); a pass that purges nothing leaves them."""
    stats = PurgeStats()

    candidates = {c.post_id for c in store.list_candidates()}
    replies = {r.reply_id for r in store.list_replies()}
    malformed = sorted(
        item
        for kind, items in (("candidate", candidates), ("reply", replies))
        for item in items
        if not _KIND_RE[kind].match(item)
    )
    for item in malformed:
        stats.errors.append(
            f"{item}: not a Reddit fullname of the right kind for its table; "
            "cannot verify liveness -- row retained (never delete on a "
            "data-shape mismatch)"
        )
    all_items = sorted((candidates | replies) - set(malformed))

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
            item_type = "candidate" if item_id in candidates else "reply"
            # Atomic delete+log in one store transaction: no content ever
            # disappears without its audit record.
            if store.purge_item(
                item_id,
                item_type,
                deleted_detected_at=now,
                purged_at=now,
                reason=reason,
            ):
                if item_type == "candidate":
                    stats.purged_candidates += 1
                else:
                    stats.purged_replies += 1

    if digest_dir is not None and (stats.purged_candidates or stats.purged_replies):
        # Rendered digests are local storage too: files written before the
        # deletion may still carry the purged content. They are pure
        # projections, so removal is lossless -- re-render from the clean
        # store afterwards.
        for artifact in sorted(digest_dir.glob("*.md")):
            artifact.unlink()
            stats.digests_removed += 1

    return stats
