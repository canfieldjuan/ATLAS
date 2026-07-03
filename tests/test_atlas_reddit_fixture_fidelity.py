"""Enforcing tests for the producer-fidelity fixture factory (slice 2,
plan: plans/PR-Producer-Fidelity-Fixture-Factory.md).

The lockstep tests import BOTH real sides -- the producer path (via the
factory) and the purge kind contract (``_KIND_RE``) -- so the S6 class
(consumer fixtures drifting from producer output) fails HERE, at
test-authoring time, instead of in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_reddit.purge import _KIND_RE, purge_once
from atlas_reddit.store import ListeningStore
from tests.atlas_reddit_fixtures import (
    fake_own_comment,
    fake_reply,
    fake_submission,
    seed_candidates,
    seed_replies,
)

NOW = 1_751_600_000


@pytest.fixture
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


class _GoneSource:
    def __init__(self, gone: set[str]) -> None:
        self._gone = gone

    def fetch_gone_items(self, fullnames: list[str]) -> dict[str, str]:
        return {
            item: "missing (not returned by the API)"
            for item in fullnames
            if item in self._gone
        }


def test_candidate_ids_lockstep_with_purge_kind_contract(store) -> None:
    ids = seed_candidates(
        store, [fake_submission("livex"), fake_submission("gonex")], now=NOW
    )
    assert ids == ["t3_livex", "t3_gonex"]
    for item in ids:
        assert _KIND_RE["candidate"].match(item)
    assert {c.post_id for c in store.list_candidates()} == set(ids)


def test_reply_ids_lockstep_with_purge_kind_contract(store) -> None:
    ids = seed_replies(store, [fake_reply("livey"), fake_reply("goney")], now=NOW)
    assert ids == ["t1_livey", "t1_goney"]
    for item in ids:
        assert _KIND_RE["reply"].match(item)
    assert {r.reply_id for r in store.list_replies()} == set(ids)


def test_pipeline_seeded_rows_purge_cleanly_end_to_end(store) -> None:
    """Producer mapping -> poller/tracker -> store -> purge, all real
    components, fakes only at the praw/deletion transports: nothing the
    factory seeds is ever flagged as malformed."""
    live_c, gone_c = seed_candidates(
        store, [fake_submission("livex"), fake_submission("gonex")], now=NOW
    )
    live_r, gone_r = seed_replies(
        store, [fake_reply("livey"), fake_reply("goney")], now=NOW
    )
    stats = purge_once(
        store, _GoneSource({gone_c, gone_r}), now=NOW, pace_seconds=0.0
    )
    assert stats.errors == []  # zero data-shape mismatches
    assert stats.purged_candidates == 1 and stats.purged_replies == 1
    assert store.get_candidate(live_c) is not None
    assert store.get_candidate(gone_c) is None
    assert {r.reply_id for r in store.list_replies()} == {live_r}


def test_factory_rejects_prefixed_and_non_reddit_ids() -> None:
    for bad in ("t3_abc", "t1_abc", "t2_abc"):
        with pytest.raises(ValueError, match="bare Reddit id"):
            fake_submission(bad)
        with pytest.raises(ValueError, match="bare Reddit id"):
            fake_reply(bad)
    with pytest.raises(ValueError, match="not a Reddit id shape"):
        fake_submission("ABC!")
    with pytest.raises(ValueError, match="bare Reddit id"):
        fake_own_comment("t1_x", thread_bare_id="y", created_utc=NOW)


def test_fake_praw_fullname_contract() -> None:
    """The praw fullname contract (fullname == f"{kind}_{id}") is encoded
    exactly once, in the factory doubles; this pins it."""
    assert fake_submission("abc").fullname == "t3_abc"
    assert fake_reply("def2").fullname == "t1_def2"
    own = fake_own_comment("ghi", thread_bare_id="jkl", created_utc=NOW)
    assert own.fullname == "t1_ghi" and own.link_id == "t3_jkl"


def test_silently_filtered_fixture_fails_loudly(store) -> None:
    """A fixture the pipeline would drop (link post) is a factory-misuse
    error, not a silent no-op."""
    with pytest.raises(AssertionError, match="silently filtered"):
        seed_candidates(store, [fake_submission("linkpost", is_self=False)], now=NOW)
