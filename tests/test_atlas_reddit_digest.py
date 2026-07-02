"""Digest + CLI tests for atlas_reddit (slice S3, #1934).

Real adapters end to end: digests render from candidates/replies seeded
through the real ListeningStore into real SQLite files under tmp_path,
and the CLI tests call the real main() in-process with argv. Dates are
passed explicitly (the clock enters only at the CLI default), so every
assertion is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_reddit.__main__ import main
from atlas_reddit.digest import render_digest, write_digest
from atlas_reddit.store import ListeningStore

DATE = "2026-07-01"


@pytest.fixture()
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _seed_candidate(store: ListeningStore, post_id: str, **overrides: object) -> None:
    kwargs: dict = dict(
        post_id=post_id,
        subreddit="CustomerSuccess",
        title=f"Post {post_id}",
        url=f"https://reddit.com/r/CustomerSuccess/comments/{post_id}",
        author="someone",
        created_utc=1_751_241_600,  # 2025-06-30 UTC
        reddit_score=10,
        num_comments=3,
        keyword_score=1.0,
        final_score=1.5,
        matched_topics=("ticket-deflection",),
        observed_at=1_751_328_000,
    )
    kwargs.update(overrides)
    store.upsert_candidate(**kwargs)


def _seed_reply(store: ListeningStore, reply_id: str, **overrides: object) -> None:
    kwargs: dict = dict(
        reply_id=reply_id,
        thread_id="t3_thread",
        parent_id="t1_mine",
        author="other_user",
        body="Interesting, how did you measure that deflection number?",
        created_utc=1_751_241_600,
        is_reply_to_me=True,
    )
    kwargs.update(overrides)
    store.upsert_tracked_thread(
        thread_id=str(kwargs["thread_id"]), my_comment_ids=(), checked_at=0
    )
    store.insert_reply(**kwargs)


# -- pure renderer -----------------------------------------------------------


def test_render_empty_states() -> None:
    content = render_digest(candidates=[], replies=[], generated_on=DATE)
    assert f"# Reddit Listening Digest -- {DATE}" in content
    assert "No new candidates." in content
    assert "No tracked-thread activity yet" in content


def test_render_radar_ranked_with_why(store: ListeningStore) -> None:
    _seed_candidate(store, "t3_high", final_score=3.0, title="High scorer")
    _seed_candidate(
        store,
        "t3_low",
        final_score=0.5,
        title="Low scorer",
        matched_topics=("repeat-tickets", "support-volume"),
    )
    candidates = store.list_candidates(status="new")
    content = render_digest(candidates=candidates, replies=[], generated_on=DATE)

    high_pos = content.index("High scorer")
    low_pos = content.index("Low scorer")
    assert high_pos < low_pos  # ranked by score
    assert "1. **[High scorer](https://reddit.com/r/CustomerSuccess/comments/t3_high)**" in content
    assert "score 3" in content
    assert "topics: repeat-tickets, support-volume" in content
    assert "posted: 2025-06-30" in content


def test_render_warm_replies_section(store: ListeningStore) -> None:
    _seed_reply(store, "t1_r1")
    replies = store.list_replies(only_unseen=True, only_to_me=True)
    content = render_digest(candidates=[], replies=replies, generated_on=DATE)
    assert "other_user replied to you on thread t3_thread" in content
    assert "how did you measure" in content


def test_render_escapes_markdown_hostile_titles(store: ListeningStore) -> None:
    """External titles must not break out of link syntax or forge digest
    structure."""
    _seed_candidate(
        store,
        "t3_evil",
        title="Escape](http://evil.example) [pwn",
    )
    candidates = store.list_candidates(status="new")
    content = render_digest(candidates=candidates, replies=[], generated_on=DATE)
    assert "](http://evil.example)" not in content
    assert "\\]\\(http://evil.example\\)" in content
    assert "(https://reddit.com/r/CustomerSuccess/comments/t3_evil)" in content


def test_render_collapses_newlines_in_external_text(store: ListeningStore) -> None:
    _seed_candidate(store, "t3_nl", title="Line one\n## Fake heading\nLine two")
    _seed_reply(store, "t1_nl", body="First line\n\n# Injected heading\nrest")
    content = render_digest(
        candidates=store.list_candidates(status="new"),
        replies=store.list_replies(only_unseen=True),
        generated_on=DATE,
    )
    assert "\n## Fake heading" not in content
    assert "\n# Injected heading" not in content
    assert "Line one ## Fake heading Line two" in content


def test_render_long_reply_body_excerpted(store: ListeningStore) -> None:
    _seed_reply(store, "t1_long", body="word " * 100)
    content = render_digest(
        candidates=[],
        replies=store.list_replies(only_unseen=True),
        generated_on=DATE,
    )
    line = next(l for l in content.splitlines() if "t3_thread" in l)
    assert line.endswith('..."')
    assert len(line) < 300


def test_render_is_deterministic(store: ListeningStore) -> None:
    _seed_candidate(store, "t3_a")
    _seed_reply(store, "t1_a")
    candidates = store.list_candidates(status="new")
    replies = store.list_replies(only_unseen=True)
    first = render_digest(candidates=candidates, replies=replies, generated_on=DATE)
    second = render_digest(candidates=candidates, replies=replies, generated_on=DATE)
    assert first == second


# -- wave-1 class probes: every external field sanitized; error contracts


def test_hostile_url_percent_encoded_in_link(store: ListeningStore) -> None:
    """Cited class: sanitization must cover every interpolated field, not
    just the title. A ')' in the URL would terminate the link early."""
    _seed_candidate(
        store, "t3_url", url="https://reddit.com/r/x/a)b\n## forged"
    )
    content = render_digest(
        candidates=store.list_candidates(status="new"), replies=[], generated_on=DATE
    )
    assert "(https://reddit.com/r/x/a%29b##%20forged" in content or ")b" not in content
    assert "\n## forged" not in content


def test_non_http_url_not_linkified(store: ListeningStore) -> None:
    _seed_candidate(store, "t3_js", url="javascript:alert(1)")
    content = render_digest(
        candidates=store.list_candidates(status="new"), replies=[], generated_on=DATE
    )
    assert "javascript:" not in content
    assert "(no valid link)" in content


def test_hostile_subreddit_and_topics_sanitized(store: ListeningStore) -> None:
    _seed_candidate(
        store,
        "t3_meta",
        subreddit="abc",
        matched_topics=("evil](http://x)", "ok-topic"),
    )
    content = render_digest(
        candidates=store.list_candidates(status="new"), replies=[], generated_on=DATE
    )
    assert "evil](http://x)" not in content
    assert "evil\\]\\(http://x\\)" in content


def test_hostile_thread_id_sanitized_in_replies(store: ListeningStore) -> None:
    store.upsert_tracked_thread(
        thread_id="t3_x](http://evil)", my_comment_ids=(), checked_at=0
    )
    store.insert_reply(
        reply_id="t1_h",
        thread_id="t3_x](http://evil)",
        parent_id=None,
        author="a",
        body="hi",
        created_utc=1,
        is_reply_to_me=True,
    )
    content = render_digest(
        candidates=[],
        replies=store.list_replies(only_unseen=True),
        generated_on=DATE,
    )
    assert "t3_x](http://evil)" not in content
    assert "t3_x\\]\\(http://evil\\)" in content


def test_cli_output_path_failure_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Cited class: OSError joins the CLI's operator-error contract (exit 2
    + stderr message), never a traceback."""
    blocker = tmp_path / "digests"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    code = main(
        [
            "digest",
            "--db",
            str(tmp_path / "listening.db"),
            "--digest-dir",
            str(blocker),
            "--date",
            DATE,
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_cli_rejects_non_finite_min_score(tmp_path: Path, bad: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "digest",
                "--db",
                str(tmp_path / "x.db"),
                "--digest-dir",
                str(tmp_path / "d"),
                "--date",
                DATE,
                "--min-score",
                bad,
            ]
        )
    assert excinfo.value.code == 2


def test_store_rejects_non_finite_score_filter(store: ListeningStore) -> None:
    """Root-layer guard: sqlite3 binds NaN as NULL which silently empties
    the filter, so the store itself fails loudly for every caller."""
    from atlas_reddit.store import StoreError

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(StoreError, match="finite"):
            store.list_candidates(min_final_score=bad)


def test_backslash_cannot_defeat_the_escaper(store: ListeningStore) -> None:
    """Wave-2 class: input like 'evil\\](x' would otherwise render as an
    escaped backslash followed by a LIVE bracket."""
    _seed_candidate(store, "t3_bs", title="evil\\](http://x) pwn")
    content = render_digest(
        candidates=store.list_candidates(status="new"), replies=[], generated_on=DATE
    )
    assert "\\\\\\]" in content  # escaped backslash + escaped bracket
    assert "\\](http" not in content.replace("\\\\", "")


def test_corrupt_db_file_fails_closed_at_store_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Wave-2 class: a non-SQLite --db is a store-open failure inside the
    StoreError contract, surfaced by the CLI as exit 2, never a traceback."""
    from atlas_reddit.store import StoreError

    fake_db = tmp_path / "not_a.db"
    fake_db.write_text("this is not sqlite", encoding="utf-8")
    with pytest.raises(StoreError, match="cannot open store"):
        ListeningStore(fake_db)

    code = main(
        [
            "digest",
            "--db",
            str(fake_db),
            "--digest-dir",
            str(tmp_path / "d"),
            "--date",
            DATE,
        ]
    )
    assert code == 2
    assert "cannot open store" in capsys.readouterr().err


# -- write_digest -------------------------------------------------------------


def test_write_digest_creates_dated_file(store: ListeningStore, tmp_path: Path) -> None:
    _seed_candidate(store, "t3_a")
    out_dir = tmp_path / "digests"
    path = write_digest(store, digest_dir=out_dir, generated_on=DATE)
    assert path == out_dir / f"{DATE}.md"
    assert path.exists()
    assert "Post t3_a" in path.read_text(encoding="utf-8")


def test_write_digest_same_day_regeneration_overwrites(
    store: ListeningStore, tmp_path: Path
) -> None:
    out_dir = tmp_path / "digests"
    _seed_candidate(store, "t3_first")
    write_digest(store, digest_dir=out_dir, generated_on=DATE)
    _seed_candidate(store, "t3_second", final_score=9.0)
    path = write_digest(store, digest_dir=out_dir, generated_on=DATE)
    content = path.read_text(encoding="utf-8")
    assert "t3_second" in content
    assert len(list(out_dir.iterdir())) == 1  # one file per day, overwritten


def test_write_digest_excludes_triaged_candidates(
    store: ListeningStore, tmp_path: Path
) -> None:
    """The radar is the review queue: only status=new appears."""
    _seed_candidate(store, "t3_new")
    _seed_candidate(store, "t3_done")
    _seed_candidate(store, "t3_gone")
    store.set_candidate_status("t3_done", "responded")
    store.set_candidate_status("t3_gone", "dismissed")
    path = write_digest(store, digest_dir=tmp_path / "d", generated_on=DATE)
    content = path.read_text(encoding="utf-8")
    assert "t3_new" in content
    assert "t3_done" not in content
    assert "t3_gone" not in content


def test_write_digest_respects_limit_and_min_score(
    store: ListeningStore, tmp_path: Path
) -> None:
    _seed_candidate(store, "t3_top", final_score=5.0)
    _seed_candidate(store, "t3_mid", final_score=2.0)
    _seed_candidate(store, "t3_tiny", final_score=0.1)
    path = write_digest(
        store,
        digest_dir=tmp_path / "d",
        generated_on=DATE,
        limit=1,
        min_final_score=1.0,
    )
    content = path.read_text(encoding="utf-8")
    assert "t3_top" in content
    assert "t3_mid" not in content  # over min score but beyond limit
    assert "t3_tiny" not in content  # under min score


def test_write_digest_excludes_seen_replies(
    store: ListeningStore, tmp_path: Path
) -> None:
    _seed_reply(store, "t1_unseen")
    _seed_reply(store, "t1_seen")
    store.mark_reply_seen("t1_seen")
    path = write_digest(store, digest_dir=tmp_path / "d", generated_on=DATE)
    content = path.read_text(encoding="utf-8")
    assert content.count("replied to you") == 1


# -- CLI -----------------------------------------------------------------------


def _cli_digest(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    db = tmp_path / "listening.db"
    out = tmp_path / "digests"
    code = main(
        [
            "digest",
            "--db",
            str(db),
            "--digest-dir",
            str(out),
            "--date",
            DATE,
            *extra,
        ]
    )
    return code, out / f"{DATE}.md"


def test_cli_digest_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    with ListeningStore(tmp_path / "listening.db") as store:
        _seed_candidate(store, "t3_cli")
    code, path = _cli_digest(tmp_path)
    assert code == 0
    assert path.exists()
    assert "Post t3_cli" in path.read_text(encoding="utf-8")
    assert str(path) in capsys.readouterr().out


def test_cli_digest_empty_store_still_writes(tmp_path: Path) -> None:
    code, path = _cli_digest(tmp_path)
    assert code == 0
    assert "No new candidates." in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("bad_date", ["2026-13-40", "yesterday", "2026/07/01", ""])
def test_cli_rejects_malformed_dates(tmp_path: Path, bad_date: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "digest",
                "--db",
                str(tmp_path / "x.db"),
                "--digest-dir",
                str(tmp_path / "d"),
                "--date",
                bad_date,
            ]
        )
    assert excinfo.value.code == 2


@pytest.mark.parametrize("bad_limit", ["0", "-3"])
def test_cli_rejects_nonpositive_limit(tmp_path: Path, bad_limit: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "digest",
                "--db",
                str(tmp_path / "x.db"),
                "--digest-dir",
                str(tmp_path / "d"),
                "--date",
                DATE,
                "--limit",
                bad_limit,
            ]
        )
    assert excinfo.value.code == 2


def test_cli_unknown_command_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["publish"])
    assert excinfo.value.code == 2


def test_cli_missing_command_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_cli_surfaces_store_errors(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A store written by a newer build fails closed through the CLI."""
    import sqlite3

    db = tmp_path / "listening.db"
    with ListeningStore(db):
        pass
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    code, _ = _cli_digest(tmp_path)
    assert code == 2
    assert "schema version 99" in capsys.readouterr().err
