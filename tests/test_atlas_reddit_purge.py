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


# -- wave-1 class probes: id-format consistency, classification, shape guard ---


def test_malformed_stored_id_is_error_never_missing(store: ListeningStore) -> None:
    """P1 class, defensive layer: an id that is not a Reddit fullname
    cannot be liveness-checked -- it must be surfaced as an error and the
    row RETAINED, never classified as missing and deleted."""
    _seed_candidate(store, "abc123")  # bare id, not a fullname
    _seed_candidate(store, "t3_good")
    source = FakeDeletionSource()
    stats = _purge(store, source)
    assert len(stats.errors) == 1
    assert "abc123" in stats.errors[0]
    assert store.get_candidate("abc123") is not None  # retained
    assert all("abc123" not in batch for batch in source.batches)
    assert any("t3_good" in batch for batch in source.batches)


def _stub_praw_with(monkeypatch: pytest.MonkeyPatch, *, submissions=None, info_items=None):
    """Fake the praw MODULE (the transport boundary) to exercise the REAL
    Praw* source mappings without praw installed."""
    import sys
    import types

    class _Auth:
        def scopes(self):
            return ["read", "identity", "history"]

    class _Subreddit:
        def __init__(self, items):
            self._items = items

        def new(self, *, limit):
            return iter(self._items[:limit])

    class _Reddit:
        def __init__(self, **kwargs):
            self.auth = _Auth()

        def subreddit(self, name):
            return _Subreddit(submissions or [])

        def info(self, *, fullnames):
            return iter(info_items or [])

    stub = types.ModuleType("praw")
    stub.Reddit = _Reddit
    monkeypatch.setitem(sys.modules, "praw", stub)


def _creds(monkeypatch: pytest.MonkeyPatch):
    from atlas_reddit.config import RedditListeningSettings

    monkeypatch.setenv("ATLAS_REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("ATLAS_REDDIT_CLIENT_SECRET", "cs")
    monkeypatch.setenv("ATLAS_REDDIT_REFRESH_TOKEN", "rt")
    monkeypatch.setenv("ATLAS_REDDIT_USERNAME", "juan_c")
    return RedditListeningSettings(_env_file=None)


def test_real_poller_mapping_stores_fullnames(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1 root probe: the REAL producer mapping must store fullnames so
    stored ids feed straight into reddit.info(). This is the
    fixture-vs-producer drift the fakes alone could not catch."""
    import types

    from atlas_reddit.reddit_client import PrawListingSource

    submission = types.SimpleNamespace(
        id="abc123",
        fullname="t3_abc123",
        title="t",
        permalink="/r/x/comments/abc123/t/",
        author=types.SimpleNamespace(name="a"),
        created_utc=1.0,
        score=1,
        num_comments=0,
        is_self=True,
        selftext="body",
    )
    _stub_praw_with(monkeypatch, submissions=[submission])
    source = PrawListingSource(_creds(monkeypatch))
    posts = source.fetch_new("x", limit=10)
    assert posts[0].post_id == "t3_abc123"  # fullname, not the bare id


def _info_item(fullname: str, *, body=None, selftext=None, author="someone",
               removed_by_category=None):
    import types

    item = types.SimpleNamespace(
        fullname=fullname,
        author=types.SimpleNamespace(name=author) if author else None,
        removed_by_category=removed_by_category,
    )
    if body is not None:
        item.body = body
    else:
        item.selftext = selftext or ""
    return item


def test_removed_body_with_author_still_present_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wave-1 class: mod-removed comments can keep their author while the
    body shows [removed] -- classification keys on CONTENT state."""
    from atlas_reddit.reddit_client import PrawDeletionSource

    _stub_praw_with(
        monkeypatch,
        info_items=[_info_item("t1_x", body="[removed]", author="still_here")],
    )
    source = PrawDeletionSource(_creds(monkeypatch))
    gone = source.fetch_gone_items(["t1_x"])
    assert gone == {"t1_x": "content shows [removed]"}


def test_author_deleted_but_content_intact_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second side: an account-deleted author with intact content is NOT
    deleted content -- the reply stays."""
    from atlas_reddit.reddit_client import PrawDeletionSource

    _stub_praw_with(
        monkeypatch,
        info_items=[_info_item("t1_x", body="real words remain", author=None)],
    )
    source = PrawDeletionSource(_creds(monkeypatch))
    assert source.fetch_gone_items(["t1_x"]) == {}


def test_absent_from_info_response_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas_reddit.reddit_client import PrawDeletionSource

    _stub_praw_with(monkeypatch, info_items=[])
    source = PrawDeletionSource(_creds(monkeypatch))
    gone = source.fetch_gone_items(["t3_vanished"])
    assert "t3_vanished" in gone
    assert "missing" in gone["t3_vanished"]


# -- wave-2 class probes: kinds, atomicity, artifacts, tombstones ----------------


def test_wrong_kind_fullname_per_table_retained(store: ListeningStore) -> None:
    """Wave-2 class: the shape guard must validate the KIND per table --
    a t2_ user id in candidates or a t3_ post id in replies cannot be
    liveness-checked for that table and must be retained + surfaced."""
    _seed_candidate(store, "t2_useridx")  # wrong kind for candidates
    _seed_reply(store, "t3_postidx")      # wrong kind for replies
    source = FakeDeletionSource()
    stats = _purge(store, source)
    assert len(stats.errors) == 2
    assert store.get_candidate("t2_useridx") is not None
    assert {r.reply_id for r in store.list_replies()} == {"t3_postidx"}
    assert source.batches == []  # nothing checkable was sent


def test_purge_item_is_atomic_no_log_without_delete(store: ListeningStore) -> None:
    """Wave-2 class: the audit record exists IFF the content row was
    deleted -- one store transaction, probed from both directions."""
    _seed_candidate(store, "t3_gone")
    assert store.purge_item(
        "t3_gone", "candidate", deleted_detected_at=NOW, purged_at=NOW, reason="r"
    ) is True
    assert store.get_candidate("t3_gone") is None
    assert len(store.list_purge_log()) == 1
    # Second call: no row to delete -> False AND no second log entry.
    assert store.purge_item(
        "t3_gone", "candidate", deleted_detected_at=NOW, purged_at=NOW, reason="r"
    ) is False
    assert len(store.list_purge_log()) == 1


def test_purged_ids_are_tombstones_for_reingestion(store: ListeningStore) -> None:
    """Wave-2 class: Reddit can keep returning removed items in listings;
    the write boundary refuses tombstoned ids so purged content never
    resurrects and never generates repeat purge cycles."""
    _seed_candidate(store, "t3_gone")
    _seed_reply(store, "t1_gone")
    _purge(store, FakeDeletionSource(gone={
        "t3_gone": "content shows [removed]",
        "t1_gone": "content shows [deleted]",
    }))
    # Re-ingestion attempts (a later poll/track pass).
    _seed_candidate(store, "t3_gone")
    assert store.get_candidate("t3_gone") is None  # refused
    assert store.insert_reply(
        reply_id="t1_gone", thread_id="t3_thread", parent_id="t1_mine",
        author="other", body="back again", created_utc=NOW,
        is_reply_to_me=True,
    ) is False
    assert store.list_replies() == []
    assert len(store.list_purge_log()) == 2  # no repeat purge entries


def _write_digest_file(digest_dir: Path, name: str, *, mtime: int, body: str = "digest") -> Path:
    import os

    digest_dir.mkdir(exist_ok=True)
    artifact = digest_dir / name
    artifact.write_text(body, encoding="utf-8")
    os.utime(artifact, (mtime, mtime))
    return artifact


def test_digest_artifacts_older_than_latest_purge_removed(
    store: ListeningStore, tmp_path: Path
) -> None:
    """Wave-2/3 class: any digest file predating the newest purge may
    carry purged content and is removed from PERSISTED state."""
    digest_dir = tmp_path / "digests"
    _write_digest_file(digest_dir, "2026-07-01.md", mtime=NOW - 86400,
                       body="old digest with Post t3_gone")
    _seed_candidate(store, "t3_gone")
    stats = _purge(
        store,
        FakeDeletionSource(gone={"t3_gone": "content shows [deleted]"}),
        digest_dir=digest_dir,
    )
    assert stats.digests_removed == 1
    assert list(digest_dir.glob("*.md")) == []


def test_digest_artifacts_kept_when_no_purge_log(
    store: ListeningStore, tmp_path: Path
) -> None:
    """Second side: with no purge history the artifacts are provably
    consistent with the store -- they stay."""
    digest_dir = tmp_path / "digests"
    _write_digest_file(digest_dir, "2026-07-01.md", mtime=NOW - 86400)
    _seed_candidate(store, "t3_live")
    stats = _purge(store, FakeDeletionSource(), digest_dir=digest_dir)
    assert stats.digests_removed == 0
    assert len(list(digest_dir.glob("*.md"))) == 1


def test_fresh_digest_rendered_after_purge_is_kept(
    store: ListeningStore, tmp_path: Path
) -> None:
    """A digest re-rendered AFTER the latest purge reflects the clean
    store and must survive later zero-purge passes."""
    digest_dir = tmp_path / "digests"
    _seed_candidate(store, "t3_gone")
    _purge(store, FakeDeletionSource(gone={"t3_gone": "content shows [deleted]"}))
    _write_digest_file(digest_dir, "2026-07-02.md", mtime=NOW + 3600,
                       body="fresh post-purge digest")
    stats = _purge(store, FakeDeletionSource(), digest_dir=digest_dir)
    assert stats.digests_removed == 0
    assert len(list(digest_dir.glob("*.md"))) == 1


def test_failed_digest_cleanup_retries_on_next_pass(
    store: ListeningStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wave-3 class: a failed unlink is surfaced and -- because the
    invariant reads persisted purge state -- retried on the next pass."""
    digest_dir = tmp_path / "digests"
    _write_digest_file(digest_dir, "2026-07-01.md", mtime=NOW - 86400)
    _seed_candidate(store, "t3_gone")

    real_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        raise OSError("transient filesystem error")

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    stats = _purge(
        store,
        FakeDeletionSource(gone={"t3_gone": "content shows [deleted]"}),
        digest_dir=digest_dir,
    )
    assert stats.digests_removed == 0
    assert any("digest cleanup" in error for error in stats.errors)
    assert len(list(digest_dir.glob("*.md"))) == 1  # file survived the failure

    monkeypatch.setattr(Path, "unlink", real_unlink)
    stats2 = _purge(store, FakeDeletionSource(), digest_dir=digest_dir)
    assert stats2.digests_removed == 1  # retried from persisted state
    assert list(digest_dir.glob("*.md")) == []


def test_same_second_digest_is_still_removed(
    store: ListeningStore, tmp_path: Path
) -> None:
    """Wave-4 class: purged_at is an integer second, mtimes are
    fractional -- a digest rendered in the same wall-clock second as the
    purge must still be treated as predating it."""
    digest_dir = tmp_path / "digests"
    _write_digest_file(digest_dir, "2026-07-01.md", mtime=NOW)  # same second
    import os
    os.utime(digest_dir / "2026-07-01.md", (NOW + 0.8, NOW + 0.8))
    _seed_candidate(store, "t3_gone")
    stats = _purge(
        store,
        FakeDeletionSource(gone={"t3_gone": "content shows [deleted]"}),
        digest_dir=digest_dir,
    )
    assert stats.digests_removed == 1
    assert list(digest_dir.glob("*.md")) == []


def test_unrelated_markdown_is_never_deleted(
    store: ListeningStore, tmp_path: Path
) -> None:
    """Wave-4 class: a misconfigured digest dir containing unrelated
    Markdown must not lose data -- only the tool's own YYYY-MM-DD.md
    artifacts are eligible for cleanup."""
    digest_dir = tmp_path / "digests"
    _write_digest_file(digest_dir, "2026-07-01.md", mtime=NOW - 86400)
    _write_digest_file(digest_dir, "notes.md", mtime=NOW - 86400, body="my notes")
    _write_digest_file(digest_dir, "2026-13-99.txt.md", mtime=NOW - 86400, body="odd")
    _seed_candidate(store, "t3_gone")
    stats = _purge(
        store,
        FakeDeletionSource(gone={"t3_gone": "content shows [deleted]"}),
        digest_dir=digest_dir,
    )
    assert stats.digests_removed == 1  # only the generated artifact
    remaining = {a.name for a in digest_dir.glob("*.md")}
    assert remaining == {"notes.md", "2026-13-99.txt.md"}


def test_cross_table_id_twin_is_not_shielded(store: ListeningStore) -> None:
    """Wave-3 class: a corrupt reply holding a t3_ id must not shield the
    legitimate candidate with the same id from purging."""
    _seed_candidate(store, "t3_x")
    _seed_reply(store, "t3_x")  # wrong kind for replies (corrupt row)
    source = FakeDeletionSource(gone={"t3_x": "content shows [deleted]"})
    stats = _purge(store, source)
    assert len(stats.errors) == 1  # the corrupt reply is surfaced
    assert stats.purged_candidates == 1  # the twin candidate still purges
    assert store.get_candidate("t3_x") is None
    assert {r.reply_id for r in store.list_replies()} == {"t3_x"}  # retained


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
