"""Tests for the own-profile watcher (plan:
plans/PR-Reddit-Own-Profile-Watcher.md).

Producer-fidelity discipline (slice from #1947): sync-path tests run the
REAL producer mapping (PrawHistorySource.fetch_my_posts over a stubbed
praw module) and the REAL consumer (sync_profile_once) via the fixture
factory -- a test physically cannot seed an own-post shape the pipeline
never emits. Store, migration, and CLI branches are covered directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from atlas_reddit.__main__ import main
from atlas_reddit.profile import sync_profile_once
from atlas_reddit.store import SCHEMA_VERSION, ListeningStore, StoreError
from tests.atlas_reddit_fixtures import fake_submission, seed_own_posts

NOW = 1_751_600_000


@pytest.fixture
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _add_own_post(
    store: ListeningStore,
    *,
    post_id: str = "t3_abc",
    subreddit: str = "CustomerSuccess",
    title: str = "My post",
    url: str = "https://www.reddit.com/r/x/comments/abc/",
    created_utc: int = 1_700_000_000,
    reddit_score: int = 5,
    num_comments: int = 2,
    selftext: str = "body",
    observed_at: int = 1_700_000_100,
) -> bool:
    return store.upsert_own_post(
        post_id=post_id,
        subreddit=subreddit,
        title=title,
        url=url,
        created_utc=created_utc,
        reddit_score=reddit_score,
        num_comments=num_comments,
        selftext=selftext,
        observed_at=observed_at,
    )


# -- producer mapping ----------------------------------------------------------


def test_fetch_my_posts_maps_rich_fields_including_subreddit() -> None:
    """The profile producer emits fullnames and reads the subreddit off
    each submission (a profile listing spans subreddits)."""
    from tests.atlas_reddit_fixtures import real_history_source

    subs = [
        fake_submission("aaa", subreddit="CustomerSuccess", title="one"),
        fake_submission("bbb", subreddit="SaaS", title="two", is_self=False),
    ]
    with real_history_source(own_submissions=subs) as source:
        posts = source.fetch_my_posts(limit=10)
    assert [p.post_id for p in posts] == ["t3_aaa", "t3_bbb"]
    assert [p.subreddit for p in posts] == ["CustomerSuccess", "SaaS"]
    assert posts[0].url.startswith("https://www.reddit.com/")
    # Link posts are mapped, not filtered (unlike the radar).
    assert posts[1].is_self is False


# -- sync pass (producer fidelity) ---------------------------------------------


def test_sync_stores_all_own_posts_through_real_pipeline(store: ListeningStore) -> None:
    ids = seed_own_posts(
        store,
        [
            fake_submission("aaa", subreddit="CustomerSuccess"),
            fake_submission("bbb", subreddit="SaaS", is_self=False),
        ],
        now=NOW,
    )
    assert ids == ["t3_aaa", "t3_bbb"]
    stored = {p.post_id: p for p in store.list_own_posts()}
    assert set(stored) == {"t3_aaa", "t3_bbb"}
    assert stored["t3_bbb"].subreddit == "SaaS"


def test_sync_stats_new_then_refreshed(store: ListeningStore) -> None:
    from tests.atlas_reddit_fixtures import real_history_source

    subs = [fake_submission("aaa"), fake_submission("bbb")]
    with real_history_source(own_submissions=subs) as source:
        first = sync_profile_once(store, source, now=NOW, limit=10)
        second = sync_profile_once(store, source, now=NOW + 60, limit=10)
    assert (first.fetched, first.new, first.refreshed) == (2, 2, 0)
    assert (second.fetched, second.new, second.refreshed) == (2, 0, 2)
    assert len(store.list_own_posts()) == 2  # no duplicates


def test_sync_fetch_error_surfaced_not_raised(store: ListeningStore) -> None:
    class _Boom:
        def fetch_my_posts(self, *, limit):
            raise RuntimeError("429 rate limited")

    stats = sync_profile_once(store, _Boom(), now=NOW, limit=10)
    assert stats.fetched == 0 and stats.new == 0
    assert len(stats.errors) == 1 and "429" in stats.errors[0]
    assert store.list_own_posts() == []


# -- store: upsert / get / list ------------------------------------------------


def test_upsert_and_get_roundtrip(store: ListeningStore) -> None:
    assert _add_own_post(store) is True  # newly inserted
    row = store.get_own_post("t3_abc")
    assert row is not None
    assert row.subreddit == "CustomerSuccess"
    assert row.title == "My post"
    assert row.first_seen == 1_700_000_100
    assert row.last_seen == 1_700_000_100
    assert store.get_own_post("t3_missing") is None


def test_upsert_replay_preserves_first_seen(store: ListeningStore) -> None:
    _add_own_post(store)
    assert _add_own_post(
        store, reddit_score=99, num_comments=40, observed_at=1_700_000_900
    ) is False  # refresh, not insert
    row = store.get_own_post("t3_abc")
    assert row is not None
    assert row.first_seen == 1_700_000_100  # preserved
    assert row.last_seen == 1_700_000_900  # refreshed
    assert row.reddit_score == 99
    assert len(store.list_own_posts()) == 1


def test_upsert_stale_observation_regresses_nothing(store: ListeningStore) -> None:
    _add_own_post(store, reddit_score=50, observed_at=1_700_000_500)
    # An out-of-order (older) observation must not overwrite fresher state.
    _add_own_post(store, reddit_score=1, observed_at=1_700_000_100)
    row = store.get_own_post("t3_abc")
    assert row is not None
    assert row.reddit_score == 50  # fresher value held
    assert row.last_seen == 1_700_000_500


def test_list_own_posts_ordering_and_subreddit_filter(store: ListeningStore) -> None:
    _add_own_post(store, post_id="t3_old", created_utc=1, subreddit="SaaS")
    _add_own_post(store, post_id="t3_new", created_utc=3, subreddit="CustomerSuccess")
    _add_own_post(store, post_id="t3_mid", created_utc=2, subreddit="saas")
    ordered = [p.post_id for p in store.list_own_posts()]
    assert ordered == ["t3_new", "t3_mid", "t3_old"]  # newest first
    # Subreddit filter is case-insensitive (SaaS vs saas).
    saas = [p.post_id for p in store.list_own_posts(subreddit="SaaS")]
    assert saas == ["t3_mid", "t3_old"]
    assert [p.post_id for p in store.list_own_posts(limit=1)] == ["t3_new"]


def test_upsert_rejects_malformed_ids_and_ints(store: ListeningStore) -> None:
    with pytest.raises(StoreError, match="post_id"):
        _add_own_post(store, post_id="")
    with pytest.raises(StoreError, match="created_utc"):
        store.upsert_own_post(
            post_id="t3_x",
            subreddit="X",
            title="t",
            url="u",
            created_utc="soon",  # type: ignore[arg-type]
            reddit_score=0,
            num_comments=0,
            selftext="",
            observed_at=NOW,
        )


# -- migration -----------------------------------------------------------------


def test_v3_store_migrates_to_current_additively(tmp_path: Path) -> None:
    """A v3 database opens, walks the ladder (v4 tombstone, v5 own_posts),
    gains own_posts, keeps existing candidate rows, and lands on the
    current schema version."""
    db = tmp_path / "v3.db"
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
            my_comment_ids TEXT NOT NULL DEFAULT '[]', last_checked INTEGER,
            dormant INTEGER NOT NULL DEFAULT 0 CHECK (dormant IN (0, 1)),
            is_own_submission INTEGER NOT NULL DEFAULT 0
                CHECK (is_own_submission IN (0, 1)),
            last_activity INTEGER
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
        INSERT INTO candidates (post_id, subreddit, title, url, created_utc,
            keyword_score, final_score, first_seen, last_seen)
        VALUES ('t3_keep', 'X', 'kept', 'u', 1, 1.0, 1.0, 1, 1);
        PRAGMA user_version = 3;
        """
    )
    conn.commit()
    conn.close()

    with ListeningStore(db) as migrated:
        assert migrated.get_candidate("t3_keep") is not None  # preserved
        assert migrated.list_own_posts() == []  # new table usable
        _add_own_post(migrated, post_id="t3_new")
        assert migrated.get_own_post("t3_new") is not None
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


# -- CLI -----------------------------------------------------------------------


def test_cli_posts_lists_synced_posts(store, tmp_path, capsys) -> None:
    _add_own_post(store, post_id="t3_abc", subreddit="SaaS", title="Hello world")
    store.close()
    rc = main(["posts", "--db", str(tmp_path / "listening.db")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "r/SaaS" in out and "t3_abc" in out and "Hello world" in out


def test_cli_posts_defaults_to_all_no_silent_cap(store, tmp_path, capsys) -> None:
    """The default listing lists every synced post (no silent newest-N cap),
    so "all your posts in one place" is literal even past the old 50."""
    for i in range(60):
        _add_own_post(
            store, post_id=f"t3_p{i:03d}", created_utc=1_700_000_000 + i
        )
    store.close()
    rc = main(["posts", "--db", str(tmp_path / "listening.db")])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("\n") >= 60  # all 60, not a silent 50
    assert "t3_p000" in out and "t3_p059" in out
    # --limit still narrows on request.
    rc = main(["posts", "--db", str(tmp_path / "listening.db"), "--limit", "5"])
    assert rc == 0
    assert capsys.readouterr().out.count("\n") == 5


def test_cli_posts_empty_hint(tmp_path, capsys) -> None:
    with ListeningStore(tmp_path / "listening.db"):
        pass
    rc = main(["posts", "--db", str(tmp_path / "listening.db")])
    assert rc == 0
    assert "no synced posts" in capsys.readouterr().out


def test_cli_post_shows_post_and_accepts_bare_id(store, tmp_path, capsys) -> None:
    _add_own_post(store, post_id="t3_abc", title="Deep dive", selftext="the body")
    store.close()
    # Bare id is accepted and prefixed to the stored fullname.
    rc = main(["post", "abc", "--db", str(tmp_path / "listening.db")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Deep dive" in out and "the body" in out and "t3_abc" in out


def test_cli_post_unknown_exits_2(tmp_path, capsys) -> None:
    with ListeningStore(tmp_path / "listening.db"):
        pass
    rc = main(["post", "t3_nope", "--db", str(tmp_path / "listening.db")])
    assert rc == 2
    assert "unknown own post: t3_nope" in capsys.readouterr().err


def test_cli_profile_limit_out_of_range_errors(tmp_path) -> None:
    from atlas_reddit.config import MAX_PROFILE_LIMIT

    db = str(tmp_path / "listening.db")
    # Below the floor and above the profile ceiling (1000, not the tracker's
    # 100) both reject; the raised cap is what lets the mirror hold "all my
    # posts" for established accounts.
    with pytest.raises(SystemExit):
        main(["profile", "--db", db, "--limit", "0"])
    with pytest.raises(SystemExit):
        main(["profile", "--db", db, "--limit", str(MAX_PROFILE_LIMIT + 1)])


# -- read-only contract still holds --------------------------------------------


def test_profile_producer_adds_no_write_surface() -> None:
    """The profile method is another read; the package's static no-write
    probe (in test_atlas_reddit_poller) still governs, and the history
    source's public surface stays read-only."""
    from atlas_reddit.reddit_client import PrawHistorySource

    public = {
        name
        for name in dir(PrawHistorySource)
        if not name.startswith("_") and callable(getattr(PrawHistorySource, name))
    }
    assert public == {
        "fetch_my_recent_comments",
        "fetch_my_recent_submissions",
        "fetch_my_posts",
        "fetch_thread_replies",
        "granted_scopes",
    }
