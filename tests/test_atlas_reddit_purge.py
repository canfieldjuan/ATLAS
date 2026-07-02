"""Deletion-compliance purge tests for atlas_reddit (slice S6, #1934).

Same posture as the rest of the arc: the Reddit API is faked at the
DeletionSource boundary; everything else is real -- real SQLite stores
seeded through the real APIs, the real digest writer proving purged
content disappears from the surface, real CLI main() in-process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_reddit.purge import BATCH_SIZE, purge_once
from atlas_reddit.reddit_client import PrawDeletionSource, RedditAuthError, validate_scopes
from atlas_reddit.store import ListeningStore

NOW = 1_751_600_000


class FakeDeletionSource:
    def __init__(self, gone: dict[str, str] | None = None,
                 error_on_batch: set[int] | None = None) -> None:
        self.gone = gone or {}
        self.error_on_batch = error_on_batch or set()
        self.batches: list[list[str]] = []

    def fetch_gone_items(self, fullnames: list[str]) -> dict[str, str]:
        self.batches.append(list(fullnames))
        if len(self.batches) in self.error_on_batch:
            raise ConnectionError("boom")
        return {name: reason for name, reason in self.gone.items() if name in fullnames}


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture()
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _seed_candidate(store: ListeningStore, post_id: str) -> None:
    store.upsert_candidate(
        post_id=post_id,
        subreddit="CustomerSuccess",
        title=f"Post {post_id}",
        url=f"https://www.reddit.com/r/CustomerSuccess/comments/{post_id}/",
        author="someone",
        created_utc=NOW - 3600,
        reddit_score=5,
        num_comments=1,
        keyword_score=1.0,
        final_score=1.5,
        matched_topics=("ticket-deflection",),
        observed_at=NOW - 3600,
    )


def _seed_reply(store: ListeningStore, reply_id: str, thread_id: str = "t3_thread") -> None:
    store.upsert_tracked_thread(thread_id=thread_id, my_comment_ids=("t1_mine",), checked_at=0)
    store.insert_reply(
        reply_id=reply_id,
        thread_id=thread_id,
        parent_id="t1_mine",
        author="other",
        body="hello there",
        created_utc=NOW - 3600,
        is_reply_to_me=True,
    )


def _purge(store, source, **overrides: object):
    kwargs: dict = dict(now=NOW, pace_seconds=2.0, sleep=RecordingSleep())
    kwargs.update(overrides)
    return purge_once(store, source, **kwargs)


# -- boundary both ways ---------------------------------------------------------


def test_live_items_survive_and_gone_items_are_actually_deleted(
    store: ListeningStore,
) -> None:
    """The guard probed on BOTH sides: live content stays, deleted content
    is really gone from the database, and the purge is logged."""
    _seed_candidate(store, "t3_live")
    _seed_candidate(store, "t3_gone")
    _seed_reply(store, "t1_live")
    _seed_reply(store, "t1_gone")
    source = FakeDeletionSource(
        gone={
            "t3_gone": "content shows [deleted]",
            "t1_gone": "missing (not returned by the API)",
        }
    )
    stats = _purge(store, source)
    assert stats.purged_candidates == 1
    assert stats.purged_replies == 1
    assert store.get_candidate("t3_live") is not None
    assert store.get_candidate("t3_gone") is None  # actually gone
    remaining = {r.reply_id for r in store.list_replies()}
    assert remaining == {"t1_live"}
    log = {rec.item_id: rec for rec in store.list_purge_log()}
    assert log["t3_gone"].item_type == "candidate"
    assert log["t3_gone"].reason == "content shows [deleted]"
    assert log["t1_gone"].item_type == "reply"
    assert log["t3_gone"].purged_at == NOW


def test_purged_content_disappears_from_the_digest(
    store: ListeningStore, tmp_path: Path
) -> None:
    """Compliance end-to-end: after a purge the digest no longer surfaces
    the deleted content."""
    from atlas_reddit.digest import write_digest

    _seed_candidate(store, "t3_gone")
    _seed_reply(store, "t1_gone")
    _purge(store, FakeDeletionSource(gone={
        "t3_gone": "content shows [removed]",
        "t1_gone": "content shows [deleted]",
    }))
    content = write_digest(
        store, digest_dir=tmp_path / "d", generated_on="2026-07-02"
    ).read_text(encoding="utf-8")
    assert "t3_gone" not in content
    assert "hello there" not in content
    assert "No new candidates." in content


def test_empty_store_purge_is_a_noop(store: ListeningStore) -> None:
    source = FakeDeletionSource()
    stats = _purge(store, source)
    assert stats.checked == 0
    assert source.batches == []


def test_replay_second_purge_pass_is_idempotent(store: ListeningStore) -> None:
    _seed_candidate(store, "t3_gone")
    gone = {"t3_gone": "content shows [deleted]"}
    _purge(store, FakeDeletionSource(gone=gone))
    stats2 = _purge(store, FakeDeletionSource(gone=gone))
    assert stats2.checked == 0  # nothing left to check
    assert len(store.list_purge_log()) == 1  # no duplicate log entries


def test_batching_respects_reddit_info_limit(store: ListeningStore) -> None:
    for i in range(BATCH_SIZE + 30):
        _seed_candidate(store, f"t3_{i:04d}")
    sleeper = RecordingSleep()
    source = FakeDeletionSource()
    stats = _purge(store, source, sleep=sleeper)
    assert stats.checked == BATCH_SIZE + 30
    assert [len(batch) for batch in source.batches] == [BATCH_SIZE, 30]
    assert sleeper.calls == [2.0]  # n-1 sleeps between batches


def test_failed_batch_is_contained_and_pass_continues(store: ListeningStore) -> None:
    for i in range(BATCH_SIZE + 10):
        _seed_candidate(store, f"t3_{i:04d}")
    # The last item sorts into batch 2; batch 1 errors.
    gone_id = f"t3_{BATCH_SIZE + 9:04d}"
    source = FakeDeletionSource(
        gone={gone_id: "content shows [deleted]"}, error_on_batch={1}
    )
    stats = _purge(store, source)
    assert len(stats.errors) == 1
    assert stats.purged_candidates == 1
    assert store.get_candidate(gone_id) is None


def test_tracked_thread_rows_are_retained(store: ListeningStore) -> None:
    """Thread rows hold only ids (ours), no third-party content: the purge
    never touches them even when every reply on the thread is purged."""
    _seed_reply(store, "t1_gone")
    _purge(store, FakeDeletionSource(gone={"t1_gone": "missing (not returned by the API)"}))
    assert store.list_replies() == []
    assert len(store.list_tracked_threads(include_dormant=True)) == 1


# -- deletion source auth ---------------------------------------------------------


def test_deletion_source_requires_only_read() -> None:
    assert validate_scopes(["read"], required=PrawDeletionSource._REQUIRED)
    with pytest.raises(RedditAuthError):
        validate_scopes([], required=PrawDeletionSource._REQUIRED)


def test_deletion_source_missing_credentials_fail_closed(
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
        PrawDeletionSource(RedditListeningSettings(_env_file=None))


def test_deletion_source_read_only_public_surface() -> None:
    public = {
        name
        for name in dir(PrawDeletionSource)
        if not name.startswith("_") and callable(getattr(PrawDeletionSource, name))
    }
    assert public == {"fetch_gone_items", "granted_scopes"}


# -- CLI ---------------------------------------------------------------------------


def test_cli_purge_without_credentials_exits_cleanly(
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
    code = main(["purge", "--db", str(tmp_path / "listening.db")])
    assert code == 2
    assert "missing Reddit credentials" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["-1", "61"])
def test_cli_purge_rejects_out_of_contract_pace(tmp_path: Path, value: str) -> None:
    from atlas_reddit.__main__ import main

    with pytest.raises(SystemExit) as excinfo:
        main(["purge", "--db", str(tmp_path / "x.db"), "--pace-seconds", value])
    assert excinfo.value.code == 2
