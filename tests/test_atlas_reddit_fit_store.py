"""Store schema v5 + manual fit-import tests (v2 S4, #1931).

Nothing is faked at the store boundary: real SQLite, real migrations, real
parser + guard, real CLI main() in-process, and candidates seeded through
the REAL poll path via the producer-fidelity fixture factory.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from atlas_reddit.__main__ import main
from atlas_reddit.store import (
    MAX_BODY_EXCERPT_CHARS,
    ListeningStore,
    StoreError,
    fit_input_hash,
)
from tests.atlas_reddit_fixtures import fake_submission, seed_candidates

NOW = 1_751_600_000


@pytest.fixture()
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _seed(store: ListeningStore, post_id: str = "t3_a", body: str = "docs body") -> None:
    store.upsert_candidate(
        post_id=post_id,
        subreddit="CustomerSuccess",
        title="Docs vs product",
        url="https://www.reddit.com/r/CustomerSuccess/x/",
        author="u",
        created_utc=NOW - 3600,
        reddit_score=5,
        num_comments=3,
        keyword_score=1.0,
        final_score=2.0,
        matched_topics=("repeat-tickets",),
        observed_at=NOW - 3600,
        body_excerpt=body,
    )


def _review_kwargs(post_id: str = "t3_a", **overrides) -> dict:
    base = dict(
        post_id=post_id,
        verdict="yes",
        reason="Repeat questions despite docs.",
        angle="Ask what the ticket history shows.",
        risk_flags=(),
        guard_ok=True,
        guard_codes=(),
        source="manual",
        model_id="",
        prompt_version="fit.v1",
        input_hash="abc123",
        reviewed_at=NOW,
    )
    base.update(overrides)
    return base


# -- schema v5 migration ----------------------------------------------------


def test_fresh_v5_has_body_excerpt_and_cascade(store: ListeningStore) -> None:
    cols = [r[1] for r in store._conn.execute("PRAGMA table_info(candidates)")]
    assert "body_excerpt" in cols
    tables = [
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    ]
    assert "candidate_fit_reviews" in tables
    fk = store._conn.execute(
        "PRAGMA foreign_key_list(candidate_fit_reviews)"
    ).fetchall()
    assert fk and fk[0]["on_delete"] == "CASCADE"


def test_v4_store_migrates_to_v5_preserving_data(tmp_path: Path) -> None:
    """A real v4 store (old DDL, no fit additions) opens at v5 with data
    intact and body_excerpt defaulting to empty."""
    db = tmp_path / "v4.db"
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
                CHECK (status IN ('new','seen','dismissed','responded')),
            first_seen INTEGER NOT NULL, last_seen INTEGER NOT NULL
        );
        INSERT INTO candidates (post_id, subreddit, title, url, created_utc,
            keyword_score, final_score, first_seen, last_seen)
        VALUES ('t3_old', 'CS', 'Old post', 'http://x', 100, 1.0, 2.0, 100, 100);
        PRAGMA user_version = 4;
        """
    )
    conn.commit()
    conn.close()
    with ListeningStore(db) as migrated:
        candidate = migrated.get_candidate("t3_old")
        assert candidate.title == "Old post"
        assert candidate.body_excerpt == ""
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
    conn.close()


def test_body_excerpt_persisted_through_the_real_poll_path(
    store: ListeningStore,
) -> None:
    """The poller writes a bounded, whitespace-collapsed body_excerpt; a
    silently-empty column would leave the fit runner with nothing to judge."""
    submission = fake_submission(
        "bodyid", selftext="We  have\n\nKB   articles  but users still ask."
    )
    post_id = seed_candidates(store, [submission], now=NOW)[0]
    candidate = store.get_candidate(post_id)
    assert candidate.body_excerpt == "We have KB articles but users still ask."


def test_body_excerpt_is_bounded(store: ListeningStore) -> None:
    submission = fake_submission("longid", selftext="x " * 4000)
    post_id = seed_candidates(store, [submission], now=NOW)[0]
    assert len(store.get_candidate(post_id).body_excerpt) <= MAX_BODY_EXCERPT_CHARS


# -- upsert_fit_review ------------------------------------------------------


def test_upsert_get_and_idempotent_by_post_id(store: ListeningStore) -> None:
    _seed(store)
    store.upsert_fit_review(**_review_kwargs())
    store.upsert_fit_review(**_review_kwargs(verdict="maybe", reason="Softer read."))
    review = store.get_fit_review("t3_a")
    assert review.verdict == "maybe"
    assert review.reason == "Softer read."
    assert len(store.list_fit_reviews()) == 1  # replaced, not duplicated


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"verdict": "definitely"}, "invalid fit verdict"),
        ({"source": "robot"}, "invalid fit source"),
        ({"risk_flags": "promo_risk"}, "list/tuple of strings"),
        ({"guard_codes": [{"code": "x"}]}, "only strings"),
        ({"guard_ok": "yes"}, "must be a bool"),
    ],
)
def test_invalid_review_fields_fail_closed(
    store: ListeningStore, overrides: dict, match: str
) -> None:
    _seed(store)
    with pytest.raises(StoreError, match=match):
        store.upsert_fit_review(**_review_kwargs(**overrides))


def test_guard_rejected_review_persists_flagged_and_redacted(
    store: ListeningStore,
) -> None:
    """A blocked review keeps verdict + codes + provenance but its text is
    REDACTED -- a PII/claim leak the guard caught must not sit in SQLite."""
    _seed(store)
    store.upsert_fit_review(
        **_review_kwargs(
            reason="An audit cuts tickets 40%.",
            angle="Lead with the ROI story.",
            guard_ok=False,
            guard_codes=("GUARANTEED_DEFLECTION", "ROI_SAVINGS"),
            source="model",
            model_id="m1",
        )
    )
    review = store.get_fit_review("t3_a")
    assert review.guard_ok is False
    assert review.reason == ""
    assert review.angle is None
    assert review.guard_codes == ("GUARANTEED_DEFLECTION", "ROI_SAVINGS")
    # the raw text is not anywhere in the row
    row = store._conn.execute(
        "SELECT reason, angle FROM candidate_fit_reviews WHERE post_id = 't3_a'"
    ).fetchone()
    assert row["reason"] == "" and row["angle"] == ""


def test_no_verdict_angle_round_trips_as_none(store: ListeningStore) -> None:
    _seed(store)
    store.upsert_fit_review(**_review_kwargs(verdict="no", angle=None))
    assert store.get_fit_review("t3_a").angle is None


def test_review_for_unknown_candidate_fails_closed(store: ListeningStore) -> None:
    with pytest.raises(StoreError, match="unknown candidate"):
        store.upsert_fit_review(**_review_kwargs(post_id="t3_ghost"))


# -- purge CASCADE ----------------------------------------------------------


def test_purge_cascades_the_fit_review(store: ListeningStore) -> None:
    """Deletion compliance extends to fit output: purging a candidate
    removes its review in the same transaction (FK ON DELETE CASCADE)."""
    _seed(store)
    store.upsert_fit_review(**_review_kwargs())
    assert store.get_fit_review("t3_a") is not None
    store.purge_item(
        "t3_a", "candidate", deleted_detected_at=NOW, purged_at=NOW,
        reason="content shows [deleted]",
    )
    assert store.get_candidate("t3_a") is None
    assert store.get_fit_review("t3_a") is None


# -- list_fit_reviews filtering ---------------------------------------------


def test_list_filters_by_post_ids_and_guard_ok(store: ListeningStore) -> None:
    _seed(store, "t3_a")
    _seed(store, "t3_b")
    store.upsert_fit_review(**_review_kwargs(post_id="t3_a"))
    store.upsert_fit_review(
        **_review_kwargs(
            post_id="t3_b", guard_ok=False, guard_codes=("PII_EMAIL",),
            reason="x", angle="y",
        )
    )
    assert set(store.list_fit_reviews()) == {"t3_a", "t3_b"}
    assert set(store.list_fit_reviews(("t3_a",))) == {"t3_a"}
    assert set(store.list_fit_reviews(only_guard_ok=True)) == {"t3_a"}
    assert store.list_fit_reviews(()) == {}  # empty subset short-circuits


# -- fit_input_hash ---------------------------------------------------------


def test_input_hash_is_stable_and_input_sensitive() -> None:
    base = dict(
        post_id="t3_a", subreddit="CS", title="T",
        body_excerpt="body", matched_topics=("a", "b"),
    )
    assert fit_input_hash(**base) == fit_input_hash(**base)
    assert fit_input_hash(**{**base, "body_excerpt": "changed"}) != fit_input_hash(
        **base
    )


# -- import-fit CLI ---------------------------------------------------------


def _prediction(**overrides) -> dict:
    base = {
        "verdict": "yes",
        "reason": "Repeat questions despite docs.",
        "angle": "Ask what the ticket history shows.",
        "risk_flags": [],
    }
    base.update(overrides)
    return base


def test_import_fit_persists_through_parser_and_guard(
    tmp_path: Path, capsys
) -> None:
    db = tmp_path / "s.db"
    with ListeningStore(db) as store:
        _seed(store, "t3_a")
        _seed(store, "t3_bad")
    preds = tmp_path / "p.jsonl"
    preds.write_text(
        json.dumps({"post_id": "t3_a", "prediction": _prediction()})
        + "\n"
        + json.dumps(
            {
                "post_id": "t3_bad",
                "prediction": _prediction(
                    angle="An audit guarantees a 40% ticket reduction."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code = main(["import-fit", str(preds), "--db", str(db)])
    assert code == 0
    assert "imported=2 errors=0" in capsys.readouterr().out
    with ListeningStore(db) as store:
        good = store.get_fit_review("t3_a")
        assert good.guard_ok is True and good.source == "manual"
        assert good.prompt_version == "fit.v1"
        # the guard-blocked prediction persisted flagged + redacted
        bad = store.get_fit_review("t3_bad")
        assert bad.guard_ok is False
        assert "GUARANTEED_DEFLECTION" in bad.guard_codes
        assert bad.reason == "" and bad.angle is None


def test_import_fit_partial_errors_exit_one(tmp_path: Path, capsys) -> None:
    db = tmp_path / "s.db"
    with ListeningStore(db) as store:
        _seed(store, "t3_a")
    preds = tmp_path / "p.jsonl"
    preds.write_text(
        json.dumps({"post_id": "t3_a", "prediction": _prediction()})
        + "\n"
        + json.dumps({"post_id": "t3_missing", "prediction": _prediction()})
        + "\n"
        + '{"post_id": "t3_a", "prediction":\n'  # malformed JSON line
        + json.dumps({"post_id": "t3_a", "prediction": {"verdict": "definitely"}})
        + "\n",
        encoding="utf-8",
    )
    code = main(["import-fit", str(preds), "--db", str(db)])
    assert code == 1
    err = capsys.readouterr().err
    assert "unknown candidate t3_missing" in err
    assert "malformed JSON" in err
    # the one valid line still persisted
    with ListeningStore(db) as store:
        assert store.get_fit_review("t3_a") is not None


def test_import_fit_missing_file_exits_two(tmp_path: Path, capsys) -> None:
    code = main(["import-fit", str(tmp_path / "nope.jsonl"), "--db", str(tmp_path / "s.db")])
    assert code == 2
    assert "error:" in capsys.readouterr().err
