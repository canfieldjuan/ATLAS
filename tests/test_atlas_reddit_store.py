"""SQLite store tests for atlas_reddit (slice S2, #1934).

Real adapter throughout: every test runs against a real SQLite file
under tmp_path through the real ListeningStore API -- no mocks, no fake
pools. Timestamps are explicit ints (the clock enters as data), so
everything is deterministic. Coverage follows the trial rules: happy
paths, idempotent/replayed writes, wrong state transitions, boundary
values, injection-shaped input, schema-version fail-closed, and both
enforcement layers (API validation and SQL CHECK constraints).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from atlas_reddit.store import (
    CANDIDATE_STATUSES,
    SCHEMA_VERSION,
    ListeningStore,
    StoreError,
)


@pytest.fixture()
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _add_candidate(store: ListeningStore, post_id: str = "t3_abc", **overrides: object) -> None:
    kwargs: dict = dict(
        post_id=post_id,
        subreddit="CustomerSuccess",
        title="Measuring ticket deflection",
        url=f"https://reddit.com/r/CustomerSuccess/comments/{post_id}",
        author="someone",
        created_utc=1_700_000_000,
        reddit_score=10,
        num_comments=3,
        keyword_score=1.0,
        final_score=1.5,
        matched_topics=("ticket-deflection",),
        observed_at=1_700_000_100,
    )
    kwargs.update(overrides)
    store.upsert_candidate(**kwargs)


# -- schema ---------------------------------------------------------------


def test_schema_created_and_versioned(tmp_path: Path) -> None:
    path = tmp_path / "listening.db"
    with ListeningStore(path) as store:
        assert store.list_candidates() == []
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()
    assert {"candidates", "tracked_threads", "replies", "purge_log"} <= tables


def test_reopen_existing_store_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "listening.db"
    with ListeningStore(path) as store:
        _add_candidate(store)
    with ListeningStore(path) as store:
        assert store.get_candidate("t3_abc") is not None


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    """A store written by a newer build must not be silently misread."""
    path = tmp_path / "listening.db"
    with ListeningStore(path):
        pass
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(StoreError, match="schema version 99"):
        ListeningStore(path)


def test_parent_directory_created(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "atlas_reddit" / "listening.db"
    with ListeningStore(nested) as store:
        _add_candidate(store)
    assert nested.exists()


# -- candidates -------------------------------------------------------------


def test_upsert_and_get_roundtrip(store: ListeningStore) -> None:
    _add_candidate(store)
    row = store.get_candidate("t3_abc")
    assert row is not None
    assert row.subreddit == "CustomerSuccess"
    assert row.matched_topics == ("ticket-deflection",)
    assert row.status == "new"
    assert row.first_seen == 1_700_000_100
    assert row.last_seen == 1_700_000_100
    assert store.get_candidate("t3_missing") is None


def test_upsert_replay_preserves_first_seen_and_status(store: ListeningStore) -> None:
    """Replay-safety: re-observing a post refreshes volatile fields but
    never resets triage state or the first-seen timestamp."""
    _add_candidate(store)
    store.set_candidate_status("t3_abc", "dismissed")
    _add_candidate(
        store,
        reddit_score=50,
        num_comments=12,
        final_score=2.5,
        observed_at=1_700_000_900,
    )
    row = store.get_candidate("t3_abc")
    assert row is not None
    assert row.status == "dismissed"  # preserved
    assert row.first_seen == 1_700_000_100  # preserved
    assert row.last_seen == 1_700_000_900  # refreshed
    assert row.reddit_score == 50
    assert row.final_score == 2.5
    assert len(store.list_candidates()) == 1  # no duplicate row


@pytest.mark.parametrize("status", CANDIDATE_STATUSES)
def test_all_documented_statuses_settable(store: ListeningStore, status: str) -> None:
    _add_candidate(store)
    store.set_candidate_status("t3_abc", status)
    row = store.get_candidate("t3_abc")
    assert row is not None and row.status == status


def test_invalid_status_rejected_at_api(store: ListeningStore) -> None:
    _add_candidate(store)
    with pytest.raises(StoreError, match="invalid candidate status"):
        store.set_candidate_status("t3_abc", "archived")
    row = store.get_candidate("t3_abc")
    assert row is not None and row.status == "new"  # unchanged


def test_invalid_status_rejected_by_check_constraint(store: ListeningStore) -> None:
    """Second enforcement layer: raw SQL writes cannot bypass the enum."""
    _add_candidate(store)
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "UPDATE candidates SET status = 'archived' WHERE post_id = 't3_abc'"
        )


def test_set_status_on_unknown_candidate_fails_closed(store: ListeningStore) -> None:
    with pytest.raises(StoreError, match="unknown candidate"):
        store.set_candidate_status("t3_missing", "seen")


def test_list_candidates_filters_and_ordering(store: ListeningStore) -> None:
    _add_candidate(store, post_id="t3_low", final_score=0.5, created_utc=1)
    _add_candidate(store, post_id="t3_high", final_score=3.0, created_utc=2)
    _add_candidate(store, post_id="t3_mid", final_score=1.5, created_utc=3)
    store.set_candidate_status("t3_mid", "dismissed")

    ordered = [c.post_id for c in store.list_candidates()]
    assert ordered == ["t3_high", "t3_mid", "t3_low"]

    new_only = [c.post_id for c in store.list_candidates(status="new")]
    assert new_only == ["t3_high", "t3_low"]

    # Boundary: min_final_score is inclusive.
    scored = [c.post_id for c in store.list_candidates(min_final_score=1.5)]
    assert scored == ["t3_high", "t3_mid"]

    limited = [c.post_id for c in store.list_candidates(limit=1)]
    assert limited == ["t3_high"]

    assert store.list_candidates(status="responded") == []


def test_list_candidates_invalid_status_filter_rejected(store: ListeningStore) -> None:
    with pytest.raises(StoreError, match="invalid candidate status"):
        store.list_candidates(status="everything")


def test_equal_scores_have_stable_tiebreak(store: ListeningStore) -> None:
    _add_candidate(store, post_id="t3_b", final_score=1.0, created_utc=100)
    _add_candidate(store, post_id="t3_a", final_score=1.0, created_utc=100)
    ordered = [c.post_id for c in store.list_candidates()]
    assert ordered == ["t3_a", "t3_b"]


def test_injection_shaped_values_stored_literally(store: ListeningStore) -> None:
    """Parameterized SQL: quotes/semicolons in input are data, not syntax."""
    hostile_id = "t3_x'; DROP TABLE candidates; --"
    hostile_title = 'Robert"); DELETE FROM candidates; --'
    _add_candidate(store, post_id=hostile_id, title=hostile_title)
    row = store.get_candidate(hostile_id)
    assert row is not None
    assert row.title == hostile_title
    assert len(store.list_candidates()) == 1  # table intact


def test_unicode_roundtrip(store: ListeningStore) -> None:
    title = "Deflexi\u00f3n de tickets \N{FIRE} \u2014 mes 1"
    _add_candidate(store, title=title)
    row = store.get_candidate("t3_abc")
    assert row is not None and row.title == title


def test_matched_topics_empty_and_multiple(store: ListeningStore) -> None:
    _add_candidate(store, post_id="t3_none", matched_topics=())
    _add_candidate(
        store, post_id="t3_two", matched_topics=("ticket-deflection", "repeat-tickets")
    )
    none_row = store.get_candidate("t3_none")
    two_row = store.get_candidate("t3_two")
    assert none_row is not None and none_row.matched_topics == ()
    assert two_row is not None and two_row.matched_topics == (
        "ticket-deflection",
        "repeat-tickets",
    )


# -- tracked threads ---------------------------------------------------------


def test_tracked_thread_upsert_merges_comment_ids(store: ListeningStore) -> None:
    store.upsert_tracked_thread(
        thread_id="t3_thread", my_comment_ids=("t1_a",), checked_at=100
    )
    store.upsert_tracked_thread(
        thread_id="t3_thread", my_comment_ids=("t1_b", "t1_a"), checked_at=200
    )
    threads = store.list_tracked_threads()
    assert len(threads) == 1
    thread = threads[0]
    assert thread.my_comment_ids == ("t1_a", "t1_b")  # union, order-stable, deduped
    assert thread.last_checked == 200
    assert thread.dormant is False


def test_dormant_flag_and_filtering(store: ListeningStore) -> None:
    store.upsert_tracked_thread(thread_id="t3_active", my_comment_ids=(), checked_at=1)
    store.upsert_tracked_thread(thread_id="t3_quiet", my_comment_ids=(), checked_at=1)
    store.set_thread_dormant("t3_quiet", True)

    active = [t.thread_id for t in store.list_tracked_threads()]
    assert active == ["t3_active"]
    everything = [t.thread_id for t in store.list_tracked_threads(include_dormant=True)]
    assert everything == ["t3_active", "t3_quiet"]

    # Waking a dormant thread flips it back.
    store.set_thread_dormant("t3_quiet", False)
    assert len(store.list_tracked_threads()) == 2


def test_dormant_preserved_across_upsert_replay(store: ListeningStore) -> None:
    store.upsert_tracked_thread(thread_id="t3_thread", my_comment_ids=(), checked_at=1)
    store.set_thread_dormant("t3_thread", True)
    store.upsert_tracked_thread(
        thread_id="t3_thread", my_comment_ids=("t1_new",), checked_at=2
    )
    thread = store.list_tracked_threads(include_dormant=True)[0]
    assert thread.dormant is True  # replay must not resurrect the thread


def test_set_dormant_unknown_thread_fails_closed(store: ListeningStore) -> None:
    with pytest.raises(StoreError, match="unknown tracked thread"):
        store.set_thread_dormant("t3_missing", True)


# -- replies -------------------------------------------------------------------


def _add_reply(store: ListeningStore, reply_id: str = "t1_r1", **overrides: object) -> bool:
    kwargs: dict = dict(
        reply_id=reply_id,
        thread_id="t3_thread",
        parent_id="t1_mine",
        author="other_user",
        body="Interesting, how did you measure it?",
        created_utc=1_700_000_000,
        is_reply_to_me=True,
    )
    kwargs.update(overrides)
    return store.insert_reply(**kwargs)


def test_insert_reply_and_replay_ignored(store: ListeningStore) -> None:
    assert _add_reply(store) is True
    assert _add_reply(store, body="changed body on replay") is False
    replies = store.list_replies()
    assert len(replies) == 1
    assert replies[0].body == "Interesting, how did you measure it?"  # original kept
    assert replies[0].seen is False


def test_mark_reply_seen_and_unseen_filter(store: ListeningStore) -> None:
    _add_reply(store, reply_id="t1_r1", created_utc=1)
    _add_reply(store, reply_id="t1_r2", created_utc=2)
    store.mark_reply_seen("t1_r1")
    unseen = [r.reply_id for r in store.list_replies(only_unseen=True)]
    assert unseen == ["t1_r2"]


def test_mark_unknown_reply_fails_closed(store: ListeningStore) -> None:
    with pytest.raises(StoreError, match="unknown reply"):
        store.mark_reply_seen("t1_missing")


def test_reply_filters_thread_and_to_me(store: ListeningStore) -> None:
    _add_reply(store, reply_id="t1_mine", thread_id="t3_a", is_reply_to_me=True, created_utc=1)
    _add_reply(store, reply_id="t1_other", thread_id="t3_a", is_reply_to_me=False, created_utc=2)
    _add_reply(store, reply_id="t1_elsewhere", thread_id="t3_b", is_reply_to_me=True, created_utc=3)

    thread_a = [r.reply_id for r in store.list_replies(thread_id="t3_a")]
    assert thread_a == ["t1_mine", "t1_other"]

    to_me = [r.reply_id for r in store.list_replies(only_to_me=True)]
    assert to_me == ["t1_mine", "t1_elsewhere"]

    combined = [
        r.reply_id
        for r in store.list_replies(thread_id="t3_a", only_to_me=True, only_unseen=True)
    ]
    assert combined == ["t1_mine"]


def test_reply_boolean_check_constraint(store: ListeningStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO replies (
                reply_id, thread_id, created_utc, is_reply_to_me, seen
            ) VALUES ('t1_bad', 't3_x', 1, 2, 0)
            """
        )


# -- purge log --------------------------------------------------------------


def test_purge_log_roundtrip(store: ListeningStore) -> None:
    store.record_purge(
        item_id="t3_gone",
        item_type="candidate",
        deleted_detected_at=1_700_000_000,
        purged_at=1_700_003_600,
        reason="post returned 404",
    )
    records = store.list_purge_log()
    assert len(records) == 1
    record = records[0]
    assert record.item_id == "t3_gone"
    assert record.item_type == "candidate"
    assert record.reason == "post returned 404"


def test_purge_log_invalid_item_type_rejected(store: ListeningStore) -> None:
    with pytest.raises(StoreError, match="invalid purge item_type"):
        store.record_purge(
            item_id="x",
            item_type="watchlist",
            deleted_detected_at=1,
            purged_at=2,
            reason="nope",
        )
    assert store.list_purge_log() == []
