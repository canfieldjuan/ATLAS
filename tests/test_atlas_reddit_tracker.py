"""Reply-tracker tests for atlas_reddit (slice S5, #1934).

Same posture as the poller suite: the Reddit API is the one true
external boundary, faked at the HistorySource protocol (or at the praw
module for constructor-path probes); everything else is real -- real
SQLite stores, real CLI main() in-process, explicit clocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_reddit.reddit_client import (
    OwnActivity,
    PrawHistorySource,
    RedditAuthError,
    ThreadReply,
    validate_scopes,
)
from atlas_reddit.store import ListeningStore
from atlas_reddit.tracker import track_once

NOW = 1_751_500_000
WEEK = 168


def _comment(item_id: str, thread_id: str, *, age_hours: int = 1) -> OwnActivity:
    return OwnActivity(
        item_id=item_id, thread_id=thread_id, created_utc=NOW - age_hours * 3600
    )


def _submission(thread_id: str, *, age_hours: int = 1) -> OwnActivity:
    return OwnActivity(
        item_id=thread_id, thread_id=thread_id, created_utc=NOW - age_hours * 3600
    )


def _reply(reply_id: str, thread_id: str, *, parent_id: str = "t1_mine",
           age_hours: int = 1, body: str = "interesting point") -> ThreadReply:
    return ThreadReply(
        reply_id=reply_id,
        thread_id=thread_id,
        parent_id=parent_id,
        author="other_user",
        body=body,
        created_utc=NOW - age_hours * 3600,
        is_reply_to_me=True,
    )


class FakeHistorySource:
    def __init__(
        self,
        comments: list[OwnActivity] | None = None,
        submissions: list[OwnActivity] | None = None,
        replies_by_thread: dict[str, list[ThreadReply]] | None = None,
        error_on: set[str] | None = None,
    ) -> None:
        self.comments = comments or []
        self.submissions = submissions or []
        self.replies_by_thread = replies_by_thread or {}
        self.error_on = error_on or set()
        self.reply_calls: list[str] = []
        self.top_level_flags: dict[str, bool] = {}

    def fetch_my_recent_comments(self, *, limit: int) -> list[OwnActivity]:
        return self.comments[:limit]

    def fetch_my_recent_submissions(self, *, limit: int) -> list[OwnActivity]:
        return self.submissions[:limit]

    def fetch_thread_replies(
        self, thread_id: str, *, my_comment_ids: frozenset[str], include_top_level: bool
    ):
        self.reply_calls.append(thread_id)
        self.top_level_flags[thread_id] = include_top_level
        if thread_id in self.error_on:
            raise ConnectionError("boom")
        return self.replies_by_thread.get(thread_id, [])


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture()
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _track(store, source, **overrides: object):
    kwargs: dict = dict(
        now=NOW,
        history_limit=50,
        dormant_after_hours=WEEK,
        pace_seconds=2.0,
        sleep=RecordingSleep(),
    )
    kwargs.update(overrides)
    return track_once(store, source, **kwargs)


# -- discovery -----------------------------------------------------------------


def test_discovers_threads_from_comments_and_submissions(store: ListeningStore) -> None:
    source = FakeHistorySource(
        comments=[_comment("t1_a", "t3_thread1"), _comment("t1_b", "t3_thread1")],
        submissions=[_submission("t3_mypost")],
    )
    stats = _track(store, source)
    assert stats.threads_discovered == 2
    threads = {t.thread_id: t for t in store.list_tracked_threads()}
    assert set(threads) == {"t3_thread1", "t3_mypost"}
    assert threads["t3_thread1"].my_comment_ids == ("t1_a", "t1_b")
    assert threads["t3_mypost"].my_comment_ids == ()  # submission, not comment


def test_rediscovery_grows_comment_ids_without_duplicates(store: ListeningStore) -> None:
    _track(store, FakeHistorySource(comments=[_comment("t1_a", "t3_x")]))
    _track(store, FakeHistorySource(comments=[_comment("t1_a", "t3_x"), _comment("t1_b", "t3_x")]))
    thread = store.list_tracked_threads()[0]
    assert thread.my_comment_ids == ("t1_a", "t1_b")
    assert len(store.list_tracked_threads(include_dormant=True)) == 1


# -- replies --------------------------------------------------------------------


def test_replies_stored_and_replay_safe(store: ListeningStore) -> None:
    source = FakeHistorySource(
        comments=[_comment("t1_mine", "t3_x")],
        replies_by_thread={"t3_x": [_reply("t1_r1", "t3_x")]},
    )
    stats = _track(store, source)
    assert stats.replies_new == 1
    stats2 = _track(store, source)
    assert stats2.replies_new == 0
    assert stats2.replies_replayed == 1
    assert len(store.list_replies()) == 1
    assert store.list_replies(only_unseen=True, only_to_me=True)[0].reply_id == "t1_r1"


def test_new_replies_feed_the_digest_warm_section(store: ListeningStore, tmp_path: Path) -> None:
    """S5 acceptance: the S3 digest's warm-replies section carries live
    tracker data with zero renderer changes."""
    from atlas_reddit.digest import write_digest

    source = FakeHistorySource(
        comments=[_comment("t1_mine", "t3_x")],
        replies_by_thread={"t3_x": [_reply("t1_r1", "t3_x", body="great question about deflection")]},
    )
    _track(store, source)
    path = write_digest(store, digest_dir=tmp_path / "d", generated_on="2026-07-02")
    content = path.read_text(encoding="utf-8")
    assert "great question about deflection" in content
    assert "No tracked-thread activity yet" not in content


def test_one_failing_thread_does_not_abort_pass(store: ListeningStore) -> None:
    source = FakeHistorySource(
        comments=[_comment("t1_a", "t3_bad"), _comment("t1_b", "t3_good")],
        replies_by_thread={"t3_good": [_reply("t1_r1", "t3_good", parent_id="t1_b")]},
        error_on={"t3_bad"},
    )
    stats = _track(store, source)
    assert stats.replies_new == 1
    assert len(stats.errors) == 1
    assert "t3_bad" in stats.errors[0]


def test_pacing_between_threads_not_before_first(store: ListeningStore) -> None:
    sleeper = RecordingSleep()
    source = FakeHistorySource(
        comments=[_comment("t1_a", "t3_one"), _comment("t1_b", "t3_two")]
    )
    _track(store, source, sleep=sleeper)
    assert sleeper.calls == [2.0]  # n-1 sleeps across 2 active threads


# -- dormancy lifecycle ----------------------------------------------------------


def test_quiet_thread_goes_dormant(store: ListeningStore) -> None:
    """Own activity older than the window and no replies -> dormant."""
    source = FakeHistorySource(comments=[_comment("t1_old", "t3_quiet", age_hours=WEEK + 24)])
    stats = _track(store, source)
    assert stats.threads_marked_dormant == 1
    assert store.list_tracked_threads() == []  # active view is empty
    assert store.list_tracked_threads(include_dormant=True)[0].dormant is True


def test_fresh_reply_keeps_thread_active(store: ListeningStore) -> None:
    source = FakeHistorySource(
        comments=[_comment("t1_old", "t3_x", age_hours=WEEK + 24)],
        replies_by_thread={"t3_x": [_reply("t1_r1", "t3_x", parent_id="t1_old", age_hours=2)]},
    )
    stats = _track(store, source)
    assert stats.threads_marked_dormant == 0
    assert store.list_tracked_threads()[0].dormant is False


def test_stale_replies_do_not_keep_thread_awake(store: ListeningStore) -> None:
    source = FakeHistorySource(
        comments=[_comment("t1_old", "t3_x", age_hours=WEEK + 48)],
        replies_by_thread={"t3_x": [_reply("t1_r1", "t3_x", parent_id="t1_old", age_hours=WEEK + 24)]},
    )
    stats = _track(store, source)
    assert stats.threads_marked_dormant == 1


def test_dormant_thread_not_polled(store: ListeningStore) -> None:
    """Dormancy means stop polling: no reply fetch for sleeping threads."""
    store.upsert_tracked_thread(thread_id="t3_sleeping", my_comment_ids=("t1_z",), checked_at=0)
    store.set_thread_dormant("t3_sleeping", True)
    source = FakeHistorySource()
    _track(store, source)
    assert source.reply_calls == []


def test_fresh_own_activity_wakes_dormant_thread(store: ListeningStore) -> None:
    """Rediscovery with in-window activity is the wake signal."""
    store.upsert_tracked_thread(thread_id="t3_sleeping", my_comment_ids=("t1_z",), checked_at=0)
    store.set_thread_dormant("t3_sleeping", True)
    source = FakeHistorySource(comments=[_comment("t1_new", "t3_sleeping", age_hours=1)])
    stats = _track(store, source)
    assert stats.threads_woken == 1
    thread = store.list_tracked_threads()[0]
    assert thread.dormant is False
    assert thread.my_comment_ids == ("t1_z", "t1_new")
    assert source.reply_calls == ["t3_sleeping"]  # woken threads are polled


def test_stale_rediscovery_does_not_wake_dormant_thread(store: ListeningStore) -> None:
    """Old own-activity aging back into the history window must not wake
    a thread the quiet rule already put to sleep."""
    store.upsert_tracked_thread(thread_id="t3_sleeping", my_comment_ids=("t1_z",), checked_at=0)
    store.set_thread_dormant("t3_sleeping", True)
    source = FakeHistorySource(
        comments=[_comment("t1_z", "t3_sleeping", age_hours=WEEK + 100)]
    )
    stats = _track(store, source)
    assert stats.threads_woken == 0
    assert store.list_tracked_threads(include_dormant=True)[0].dormant is True


# -- history source auth floor ----------------------------------------------------


def test_history_source_requires_full_scope_floor() -> None:
    with pytest.raises(RedditAuthError, match="missing required"):
        validate_scopes(["read"], required=PrawHistorySource._REQUIRED)
    with pytest.raises(RedditAuthError, match="missing required"):
        validate_scopes(["read", "identity"], required=PrawHistorySource._REQUIRED)
    assert validate_scopes(
        ["read", "identity", "history"], required=PrawHistorySource._REQUIRED
    ) == frozenset({"read", "identity", "history"})


def test_history_source_missing_credentials_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_reddit.config import RedditListeningSettings

    for var in (
        "ATLAS_REDDIT_CLIENT_ID",
        "ATLAS_REDDIT_CLIENT_SECRET",
        "ATLAS_REDDIT_REFRESH_TOKEN",
        "ATLAS_REDDIT_USERNAME",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RedditAuthError, match="missing Reddit credentials"):
        PrawHistorySource(RedditListeningSettings(_env_file=None))


def test_history_source_read_only_public_surface() -> None:
    public = {
        name
        for name in dir(PrawHistorySource)
        if not name.startswith("_") and callable(getattr(PrawHistorySource, name))
    }
    assert public == {
        "fetch_my_recent_comments",
        "fetch_my_recent_submissions",
        "fetch_thread_replies",
        "granted_scopes",
    }


# -- CLI ----------------------------------------------------------------------------


def test_cli_track_without_credentials_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from atlas_reddit.__main__ import main

    for var in (
        "ATLAS_REDDIT_CLIENT_ID",
        "ATLAS_REDDIT_CLIENT_SECRET",
        "ATLAS_REDDIT_REFRESH_TOKEN",
        "ATLAS_REDDIT_USERNAME",
    ):
        monkeypatch.delenv(var, raising=False)
    code = main(["track", "--db", str(tmp_path / "listening.db")])
    assert code == 2
    assert "missing Reddit credentials" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--history-limit", "0"),
        ("--history-limit", "101"),
        ("--dormant-after-hours", "0"),
        ("--dormant-after-hours", "8761"),
        ("--pace-seconds", "-1"),
        ("--pace-seconds", "61"),
    ],
)
def test_cli_track_rejects_out_of_contract_knobs(tmp_path: Path, flag: str, value: str) -> None:
    from atlas_reddit.__main__ import main

    with pytest.raises(SystemExit) as excinfo:
        main(["track", "--db", str(tmp_path / "x.db"), flag, value])
    assert excinfo.value.code == 2


def test_cli_mark_read_end_to_end(store: ListeningStore, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from atlas_reddit.__main__ import main

    db = tmp_path / "cli.db"
    with ListeningStore(db) as s:
        s.upsert_tracked_thread(thread_id="t3_x", my_comment_ids=("t1_m",), checked_at=0)
        s.insert_reply(
            reply_id="t1_r1",
            thread_id="t3_x",
            parent_id="t1_m",
            author="a",
            body="b",
            created_utc=1,
            is_reply_to_me=True,
        )
    code = main(["mark-read", "t1_r1", "--db", str(db)])
    assert code == 0
    with ListeningStore(db) as s:
        assert s.list_replies(only_unseen=True) == []


def test_cli_mark_read_unknown_reply_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from atlas_reddit.__main__ import main

    code = main(["mark-read", "t1_missing", "--db", str(tmp_path / "x.db")])
    assert code == 2
    assert "unknown reply" in capsys.readouterr().err


# -- wave-1 class probes (P1 gating, activity persistence, error aging) ---------


def test_top_level_flag_gated_by_own_submission(store: ListeningStore) -> None:
    """P1 class: top-level comments are replies-to-me ONLY on my own
    submissions; comment-discovered threads must not receive them."""
    source = FakeHistorySource(
        comments=[_comment("t1_a", "t3_theirs")],
        submissions=[_submission("t3_mine")],
    )
    _track(store, source)
    assert source.top_level_flags["t3_theirs"] is False
    assert source.top_level_flags["t3_mine"] is True


def test_own_submission_flag_is_sticky(store: ListeningStore) -> None:
    """Once a thread is known to be my post, a later comment-only
    rediscovery must not demote it (top-level replies would vanish)."""
    _track(store, FakeHistorySource(submissions=[_submission("t3_mine")]))
    source = FakeHistorySource(comments=[_comment("t1_late", "t3_mine")])
    _track(store, source)
    thread = store.list_tracked_threads()[0]
    assert thread.is_own_submission is True
    assert source.top_level_flags["t3_mine"] is True


def test_history_fetch_failure_contained_and_polling_continues(
    store: ListeningStore,
) -> None:
    """Transport failures at the pass level are contained like per-thread
    errors: recorded, and existing active threads still get polled."""

    class ExplodingHistorySource(FakeHistorySource):
        def fetch_my_recent_comments(self, *, limit: int):
            raise ConnectionError("rate limited")

        def fetch_my_recent_submissions(self, *, limit: int):
            raise ConnectionError("rate limited")

    store.upsert_tracked_thread(thread_id="t3_known", my_comment_ids=("t1_k",), checked_at=0)
    store.record_thread_activity("t3_known", NOW - 3600)
    source = ExplodingHistorySource()
    stats = _track(store, source)
    assert len(stats.errors) == 2
    assert source.reply_calls == ["t3_known"]  # polling continued
    assert store.list_tracked_threads()[0].dormant is False


def test_history_window_eviction_is_not_inactivity(store: ListeningStore) -> None:
    """Wave-1 class: a busy account pushes a recent thread out of the
    history window; the persisted high-water mark keeps it active."""
    _track(store, FakeHistorySource(comments=[_comment("t1_a", "t3_busy", age_hours=2)]))
    # Next pass: the thread is absent from history entirely.
    stats = _track(store, FakeHistorySource())
    assert stats.threads_marked_dormant == 0
    thread = store.list_tracked_threads()[0]
    assert thread.dormant is False
    assert thread.last_activity == NOW - 2 * 3600


def test_failed_fetch_still_ages_out_stale_threads(store: ListeningStore) -> None:
    """Wave-1 class: a dead/private thread with stale activity must not be
    retried forever -- dormancy is evaluated on the error path too.
    Setup uses store primitives so the thread is ACTIVE with stale
    persisted activity (a tracking pass would already have slept it)."""
    store.upsert_tracked_thread(thread_id="t3_dead", my_comment_ids=("t1_a",), checked_at=0)
    store.record_thread_activity("t3_dead", NOW - (WEEK + 24) * 3600)
    source = FakeHistorySource(error_on={"t3_dead"})
    stats = _track(store, source)
    assert len(stats.errors) == 1
    assert stats.threads_marked_dormant == 1
    assert store.list_tracked_threads(include_dormant=True)[0].dormant is True


def test_v1_store_migrates_to_current_preserving_data(tmp_path: Path) -> None:
    """Real migration probe: a database created with the v1 DDL opens,
    walks the full migration ladder to the current version, gains the new
    columns with correct backfill, and loses nothing."""
    import sqlite3

    db = tmp_path / "v1.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE candidates (
            post_id TEXT PRIMARY KEY NOT NULL, subreddit TEXT NOT NULL,
            title TEXT NOT NULL, url TEXT NOT NULL, author TEXT,
            created_utc INTEGER NOT NULL, reddit_score INTEGER NOT NULL DEFAULT 0,
            num_comments INTEGER NOT NULL DEFAULT 0, keyword_score REAL NOT NULL,
            final_score REAL NOT NULL, matched_topics TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'new'
                CHECK (status IN ('new', 'seen', 'dismissed', 'responded')),
            first_seen INTEGER NOT NULL, last_seen INTEGER NOT NULL
        );
        CREATE TABLE tracked_threads (
            thread_id TEXT PRIMARY KEY NOT NULL,
            my_comment_ids TEXT NOT NULL DEFAULT '[]',
            last_checked INTEGER,
            dormant INTEGER NOT NULL DEFAULT 0 CHECK (dormant IN (0, 1))
        );
        CREATE TABLE replies (
            reply_id TEXT PRIMARY KEY NOT NULL, thread_id TEXT NOT NULL,
            parent_id TEXT, author TEXT, body TEXT NOT NULL DEFAULT '',
            created_utc INTEGER NOT NULL,
            is_reply_to_me INTEGER NOT NULL CHECK (is_reply_to_me IN (0, 1)),
            seen INTEGER NOT NULL DEFAULT 0 CHECK (seen IN (0, 1)),
            FOREIGN KEY (thread_id) REFERENCES tracked_threads (thread_id)
        );
        CREATE TABLE purge_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT NOT NULL,
            item_type TEXT NOT NULL
                CHECK (item_type IN ('candidate', 'reply', 'thread')),
            deleted_detected_at INTEGER NOT NULL, purged_at INTEGER NOT NULL,
            reason TEXT NOT NULL
        );
        INSERT INTO tracked_threads (thread_id, my_comment_ids, last_checked, dormant)
        VALUES ('t3_old', '["t1_a"]', 100, 0);
        INSERT INTO replies (reply_id, thread_id, parent_id, author, body,
                             created_utc, is_reply_to_me)
        VALUES ('t1_r', 't3_old', 't1_a', 'x', 'hello', 12345, 1);
        PRAGMA user_version = 1;
        """
    )
    conn.commit()
    conn.close()

    with ListeningStore(db) as migrated:
        thread = migrated.list_tracked_threads()[0]
        assert thread.thread_id == "t3_old"
        assert thread.my_comment_ids == ("t1_a",)
        assert thread.is_own_submission is False  # additive default
        assert thread.last_activity == 12345  # backfilled from replies
        assert migrated.list_replies()[0].reply_id == "t1_r"
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    conn.close()
