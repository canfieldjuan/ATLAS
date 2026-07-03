"""Producer-fidelity fixture factory for atlas_reddit consumer tests.

Fable-arc codification slice 2 (plan:
plans/PR-Producer-Fidelity-Fixture-Factory.md). The S6 mass-deletion
class was consumer tests hand-seeding id shapes (bare ids) the real
producer never emits (fullnames) and staying green. This factory removes
the hand-rolled encoding:

- ids enter fixtures as BARE Reddit ids; the kind prefix is added only
  by the praw doubles below, mirroring praw's FullnameMixin contract
  (``fullname == f"{kind}_{id}"``; t3_ submissions, t1_ comments) --
  encoded exactly ONCE, here;
- fixture objects flow through the REAL producer mappings
  (:class:`PrawListingSource` / :class:`PrawHistorySource` over a
  stubbed praw module, the pattern the purge tests established);
- store rows are written by the REAL consumers (:func:`poll_once` /
  :func:`track_once`), asserted loudly if anything is silently filtered.

A consumer test built on these helpers physically cannot seed a shape
the pipeline never emits.
"""

from __future__ import annotations

import os
import re
import sys
import types
from contextlib import contextmanager
from unittest import mock

from atlas_reddit.config import SubredditEntry, Topic, Watchlist
from atlas_reddit.poller import poll_once
from atlas_reddit.store import ListeningStore
from atlas_reddit.tracker import track_once

_SUBMISSION_KIND = "t3"
_COMMENT_KIND = "t1"
_PROBE_PHRASE = "producer fixture probe"
_USERNAME = "juan_c"
_SCOPES = ["read", "identity", "history"]
_ENV = {
    "ATLAS_REDDIT_CLIENT_ID": "cid",
    "ATLAS_REDDIT_CLIENT_SECRET": "cs",
    "ATLAS_REDDIT_REFRESH_TOKEN": "rt",
    "ATLAS_REDDIT_USERNAME": _USERNAME,
}
_PREFIXED_RE = re.compile(r"^t\d_")
# Reddit ids are lowercase base36; anything else could not have come off
# the wire, so the factory refuses it at construction time.
_BARE_ID_RE = re.compile(r"^[a-z0-9]+$")


def _require_bare_id(value: str) -> str:
    if _PREFIXED_RE.match(value):
        raise ValueError(
            f"pass a bare Reddit id, not a fullname: {value!r} -- the kind "
            "prefix is added by the producer path, never by a test"
        )
    if not _BARE_ID_RE.match(value):
        raise ValueError(
            f"not a Reddit id shape: {value!r} (lowercase base36 only)"
        )
    return value


def fake_submission(
    bare_id: str,
    *,
    title: str | None = None,
    selftext: str = "body text",
    author: str | None = "someone",
    created_utc: int = 1_751_596_400,
    score: int = 5,
    num_comments: int = 1,
    is_self: bool = True,
):
    """A praw-shaped submission. The default title carries the probe
    phrase the seeding watchlist matches on; override it and admission
    fails loudly in :func:`seed_candidates` rather than silently."""
    _require_bare_id(bare_id)
    return types.SimpleNamespace(
        id=bare_id,
        fullname=f"{_SUBMISSION_KIND}_{bare_id}",
        title=title if title is not None else f"{_PROBE_PHRASE} {bare_id}",
        permalink=f"/r/x/comments/{bare_id}/probe/",
        author=types.SimpleNamespace(name=author) if author else None,
        created_utc=float(created_utc),
        score=score,
        num_comments=num_comments,
        is_self=is_self,
        selftext=selftext,
    )


def fake_own_comment(bare_id: str, *, thread_bare_id: str, created_utc: int):
    """One of the operator's own comments, as praw's history listing
    shapes it (fullname + t3_ link_id)."""
    _require_bare_id(bare_id)
    _require_bare_id(thread_bare_id)
    return types.SimpleNamespace(
        fullname=f"{_COMMENT_KIND}_{bare_id}",
        link_id=f"{_SUBMISSION_KIND}_{thread_bare_id}",
        created_utc=float(created_utc),
    )


def fake_reply(
    bare_id: str,
    *,
    author: str | None = "other_user",
    body: str = "hello there",
    created_utc: int = 1_751_596_400,
):
    """A praw-shaped third-party reply comment."""
    _require_bare_id(bare_id)
    return types.SimpleNamespace(
        fullname=f"{_COMMENT_KIND}_{bare_id}",
        author=types.SimpleNamespace(name=author) if author else None,
        body=body,
        created_utc=float(created_utc),
    )


@contextmanager
def _patched(reddit_cls):
    stub = types.ModuleType("praw")
    stub.Reddit = reddit_cls
    with mock.patch.dict(sys.modules, {"praw": stub}), mock.patch.dict(
        os.environ, _ENV
    ):
        from atlas_reddit.config import RedditListeningSettings

        yield RedditListeningSettings(_env_file=None)


@contextmanager
def real_listing_source(submissions):
    """Yield a REAL PrawListingSource whose transport returns the given
    praw-shaped submissions -- the production fullname mapping runs."""
    items = list(submissions)

    class _Subreddit:
        def new(self, *, limit):
            return iter(items[:limit])

    class _Reddit:
        def __init__(self, **kwargs):
            self.auth = types.SimpleNamespace(scopes=lambda: list(_SCOPES))

        def subreddit(self, name):
            return _Subreddit()

    with _patched(_Reddit) as settings:
        from atlas_reddit.reddit_client import PrawListingSource

        yield PrawListingSource(settings)


@contextmanager
def real_history_source(*, own_comments=(), replies_by_parent=None):
    """Yield a REAL PrawHistorySource: own-history listings and per-own-
    comment reply children flow through the production admission code
    (including the removeprefix('t1_') refresh lookup). Own submissions
    are deferred to the tracker-test conversion slice."""
    children = {k: list(v) for k, v in (replies_by_parent or {}).items()}
    me = types.SimpleNamespace(
        name=_USERNAME,
        comments=types.SimpleNamespace(
            new=lambda *, limit: iter(list(own_comments)[:limit])
        ),
        submissions=types.SimpleNamespace(new=lambda *, limit: iter(())),
    )

    class _Replies(list):
        def replace_more(self, *, limit):
            return []

    class _OwnComment:
        def __init__(self, kids):
            self.replies = _Replies(kids)

        def refresh(self):
            return self

    class _Reddit:
        def __init__(self, **kwargs):
            self.auth = types.SimpleNamespace(scopes=lambda: list(_SCOPES))
            self.user = types.SimpleNamespace(me=lambda: me)

        def comment(self, *, id):
            return _OwnComment(children.get(id, []))

    with _patched(_Reddit) as settings:
        from atlas_reddit.reddit_client import PrawHistorySource

        yield PrawHistorySource(settings)


def seed_candidates(
    store: ListeningStore,
    submissions,
    *,
    now: int,
    subreddit: str = "CustomerSuccess",
) -> list[str]:
    """Store candidate rows through the REAL pipeline (producer mapping +
    poll_once). Returns the producer-emitted post ids, in input order.
    Asserts every fixture was admitted -- a silently filtered fixture is
    a factory misuse, not a soft no-op."""
    items = list(submissions)
    watchlist = Watchlist(
        version=1,
        subreddits=(SubredditEntry(name=subreddit),),
        topics=(Topic(name="probe", phrases=(_PROBE_PHRASE,)),),
    )
    with real_listing_source(items) as source:
        produced = [
            post.post_id
            for post in source.fetch_new(subreddit, limit=max(len(items), 1))
        ]
        stats = poll_once(
            store,
            watchlist,
            source,
            now=now,
            freshness_hours=24 * 365 * 20,
            per_subreddit_limit=max(len(items), 1),
            min_final_score=0.0,
            pace_seconds=0.0,
        )
    assert stats.admitted == len(items), (
        f"factory fixtures must never be silently filtered: admitted "
        f"{stats.admitted} of {len(items)} ({stats})"
    )
    return produced


def seed_replies(
    store: ListeningStore,
    replies,
    *,
    now: int,
    thread_bare_id: str = "thread",
    my_comment_bare_id: str = "mine",
) -> list[str]:
    """Store reply rows through the REAL pipeline (own-history discovery +
    track_once reply admission). Returns the producer-emitted reply ids,
    in input order. Repeat calls against the SAME thread are replay-safe;
    seeding several DISTINCT threads needs a distinct my_comment_bare_id
    per thread (track_once polls every active thread each pass)."""
    items = list(replies)
    own = fake_own_comment(
        my_comment_bare_id, thread_bare_id=thread_bare_id, created_utc=now - 60
    )
    with real_history_source(
        own_comments=[own], replies_by_parent={my_comment_bare_id: items}
    ) as source:
        produced = [
            reply.reply_id
            for reply in source.fetch_thread_replies(
                own.link_id,
                my_comment_ids=frozenset({own.fullname}),
                include_top_level=False,
            )
        ]
        stats = track_once(
            store,
            source,
            now=now,
            history_limit=10,
            dormant_after_hours=24 * 365 * 20,
            pace_seconds=0.0,
        )
    assert stats.replies_new == len(items), (
        f"factory fixtures must never be silently filtered: inserted "
        f"{stats.replies_new} of {len(items)} ({stats})"
    )
    return produced
