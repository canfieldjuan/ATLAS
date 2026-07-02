"""Poller + read-only client tests for atlas_reddit (slice S4, #1934).

The Reddit API is the one true external boundary in this package, so the
transport is faked at the ListingSource protocol -- everything else is
real: real parser-built watchlists, real SQLite stores, real scorer, and
the real CLI main() in-process. praw itself is never imported here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from atlas_reddit.config import parse_watchlist
from atlas_reddit.poller import poll_once
from atlas_reddit.reddit_client import (
    ALLOWED_SCOPES,
    ListingPost,
    PrawListingSource,
    RedditAuthError,
    build_user_agent,
    validate_scopes,
)
from atlas_reddit.store import ListeningStore

NOW = 1_751_500_000


def _watchlist(**overrides: object):
    raw: dict = {
        "version": 1,
        "help_signals": ["how do you"],
        "subreddits": [
            {"name": "CustomerSuccess", "weight": 1.0},
            {"name": "SaaS", "weight": 0.8},
        ],
        "topics": [
            {
                "name": "ticket-deflection",
                "weight": 1.0,
                "phrases": ["ticket deflection", "deflection rate"],
            }
        ],
    }
    raw.update(overrides)
    return parse_watchlist(raw)


def _post(post_id: str, **overrides: object) -> ListingPost:
    kwargs: dict = dict(
        post_id=post_id,
        subreddit="CustomerSuccess",
        title="Our ticket deflection rate dropped",
        url=f"https://www.reddit.com/r/CustomerSuccess/comments/{post_id}/",
        author="someone",
        created_utc=NOW - 3600,
        score=5,
        num_comments=2,
        is_self=True,
        selftext="How do you measure it?",
    )
    kwargs.update(overrides)
    return ListingPost(**kwargs)


class FakeListingSource:
    """Canned transport. Records calls so pacing/order are assertable."""

    def __init__(self, by_subreddit: dict[str, list[ListingPost]] | None = None,
                 error_on: set[str] | None = None) -> None:
        self.by_subreddit = by_subreddit or {}
        self.error_on = error_on or set()
        self.calls: list[str] = []

    def fetch_new(self, subreddit: str, *, limit: int) -> list[ListingPost]:
        self.calls.append(subreddit)
        if subreddit in self.error_on:
            raise ConnectionError("boom")
        return self.by_subreddit.get(subreddit, [])[:limit]


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture()
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _poll(store, source, **overrides: object):
    kwargs: dict = dict(
        now=NOW,
        freshness_hours=48,
        per_subreddit_limit=50,
        min_final_score=0.5,
        pace_seconds=2.0,
        sleep=RecordingSleep(),
    )
    kwargs.update(overrides)
    return poll_once(store, _watchlist(), source, **kwargs)


# -- scope guard (fail closed) ------------------------------------------------


def test_exact_allowed_scopes_pass() -> None:
    assert validate_scopes(["identity", "history", "read"]) == ALLOWED_SCOPES


def test_scope_subset_passes() -> None:
    assert validate_scopes(["read"]) == frozenset({"read"})


@pytest.mark.parametrize(
    "granted",
    [
        ["identity", "history", "read", "submit"],  # write scope smuggled in
        ["*"],  # password-grant wildcard
        ["read", "vote"],
        ["edit"],
        [],
    ],
)
def test_excess_or_empty_scopes_fail_closed(granted: list[str]) -> None:
    with pytest.raises(RedditAuthError):
        validate_scopes(granted)


def test_missing_credentials_fail_before_praw_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """No creds -> RedditAuthError naming the env vars, raised before any
    praw import (proven by the fact praw is not installed for the suite)."""
    from atlas_reddit.config import RedditListeningSettings

    for var in (
        "ATLAS_REDDIT_CLIENT_ID",
        "ATLAS_REDDIT_CLIENT_SECRET",
        "ATLAS_REDDIT_REFRESH_TOKEN",
        "ATLAS_REDDIT_USERNAME",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RedditAuthError, match="missing Reddit credentials"):
        PrawListingSource(RedditListeningSettings(_env_file=None))


@pytest.mark.parametrize(
    "granted",
    [["identity"], ["history"], ["identity", "history"]],
)
def test_missing_read_scope_fails_closed(granted: list[str]) -> None:
    """Wave-1 class: the guard must validate the floor, not only the
    ceiling -- a token without read would fail on every fetch downstream
    instead of failing here."""
    with pytest.raises(RedditAuthError, match="missing required"):
        validate_scopes(granted)


def test_required_override_for_future_sources() -> None:
    validate_scopes(["identity", "history", "read"],
                    required=frozenset({"read", "history"}))
    with pytest.raises(RedditAuthError, match="missing required"):
        validate_scopes(["read"], required=frozenset({"read", "history"}))


def _full_creds_settings(monkeypatch: pytest.MonkeyPatch):
    from atlas_reddit.config import RedditListeningSettings

    monkeypatch.setenv("ATLAS_REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("ATLAS_REDDIT_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("ATLAS_REDDIT_REFRESH_TOKEN", "rtoken")
    monkeypatch.setenv("ATLAS_REDDIT_USERNAME", "juan_c")
    return RedditListeningSettings(_env_file=None)


def test_praw_absent_maps_to_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """praw is deliberately not installed for this suite: with full creds
    the lazy import itself must surface through the error contract."""
    import sys

    monkeypatch.setitem(sys.modules, "praw", None)  # ensure import fails
    settings = _full_creds_settings(monkeypatch)
    with pytest.raises(RedditAuthError, match="praw is not installed|Reddit authentication failed"):
        PrawListingSource(settings)


class _StubAuth:
    def __init__(self, scopes):
        self._scopes = scopes

    def scopes(self):
        return self._scopes


class _StubReddit:
    def __init__(self, *, fail=None, scopes=("read",), **kwargs):
        if fail:
            raise fail
        self.auth = _StubAuth(list(scopes))


def _install_stub_praw(monkeypatch: pytest.MonkeyPatch, **reddit_kwargs):
    """Fake the praw MODULE -- the external transport boundary -- so the
    real PrawListingSource constructor path is exercised without praw."""
    import sys
    import types

    stub = types.ModuleType("praw")
    stub.Reddit = lambda **kwargs: _StubReddit(**reddit_kwargs)
    monkeypatch.setitem(sys.modules, "praw", stub)


def test_invalid_grant_maps_to_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wave-1 cited class: prawcore-style failures during construction or
    the scope probe surface as RedditAuthError, never raw tracebacks."""
    _install_stub_praw(monkeypatch, fail=RuntimeError("invalid_grant error processing request"))
    settings = _full_creds_settings(monkeypatch)
    with pytest.raises(RedditAuthError, match="Reddit authentication failed.*invalid_grant"):
        PrawListingSource(settings)


def test_wildcard_token_refused_through_real_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard engages on the real constructor path, not just as a pure
    function: a password-grant wildcard token is refused at startup."""
    _install_stub_praw(monkeypatch, scopes=("*",))
    settings = _full_creds_settings(monkeypatch)
    with pytest.raises(RedditAuthError, match="exceed the read-only contract"):
        PrawListingSource(settings)


def test_scoped_token_accepted_through_real_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_praw(monkeypatch, scopes=("identity", "history", "read"))
    settings = _full_creds_settings(monkeypatch)
    source = PrawListingSource(settings)
    assert source.granted_scopes() == frozenset({"identity", "history", "read"})


# -- user agent ---------------------------------------------------------------


def test_user_agent_matches_reddit_required_format() -> None:
    ua = build_user_agent("juan_c-1")
    assert re.fullmatch(
        r"linux:atlas-reddit-listening:v[0-9.]+ \(by /u/juan_c-1\)", ua
    )


@pytest.mark.parametrize("bad", ["", "ab", "a" * 21, "has space", "semi;colon", None])
def test_invalid_username_rejected(bad: object) -> None:
    with pytest.raises(RedditAuthError, match="USERNAME"):
        build_user_agent(bad)  # type: ignore[arg-type]


# -- read-only surface (static compliance probe) -------------------------------


def test_praw_listing_source_public_surface_is_read_only() -> None:
    public = {
        name
        for name in dir(PrawListingSource)
        if not name.startswith("_") and callable(getattr(PrawListingSource, name))
    }
    assert public == {"fetch_new", "granted_scopes"}


def test_no_reddit_write_calls_anywhere() -> None:
    """Static probe over the whole package: no Reddit write-API attribute
    usage may appear in any atlas_reddit source file."""
    forbidden = re.compile(
        r"\.(submit|reply|upvote|downvote|edit|delete|report|message|"
        r"send_message|mark_read|hide|save|follow|subscribe)\("
    )
    package = Path(__file__).resolve().parent.parent / "atlas_reddit"
    offenders = []
    for source_file in sorted(package.glob("*.py")):
        for line_number, line in enumerate(
            source_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if forbidden.search(line):
                offenders.append(f"{source_file.name}:{line_number}: {line.strip()}")
    assert offenders == []


# -- poll_once ------------------------------------------------------------------


def test_poll_admits_topical_fresh_text_posts(store: ListeningStore) -> None:
    source = FakeListingSource({"CustomerSuccess": [_post("t3_hit")]})
    stats = _poll(store, source)
    assert stats.admitted == 1
    assert stats.errors == []
    row = store.get_candidate("t3_hit")
    assert row is not None
    assert row.subreddit == "CustomerSuccess"
    assert row.matched_topics == ("ticket-deflection",)
    assert row.final_score > 0
    assert row.first_seen == NOW
    assert row.last_seen == NOW


def test_poll_skips_link_posts_stale_posts_and_zero_scores(store: ListeningStore) -> None:
    source = FakeListingSource(
        {
            "CustomerSuccess": [
                _post("t3_link", is_self=False),
                _post("t3_old", created_utc=NOW - 72 * 3600),
                _post("t3_offtopic", title="Standing desk question?", selftext="chairs"),
            ]
        }
    )
    stats = _poll(store, source)
    assert stats.admitted == 0
    assert stats.skipped_not_text == 1
    assert stats.skipped_stale == 1
    assert stats.skipped_below_floor == 1
    assert store.list_candidates() == []


def test_poll_freshness_boundary_inclusive(store: ListeningStore) -> None:
    exactly_at_cutoff = _post("t3_edge", created_utc=NOW - 48 * 3600)
    source = FakeListingSource({"CustomerSuccess": [exactly_at_cutoff]})
    stats = _poll(store, source)
    assert stats.admitted == 1  # >= cutoff admits


def test_poll_min_score_floor_inclusive(store: ListeningStore) -> None:
    # SaaS weight 0.8; topic 1.0 + help signal 0.25 + question bonus 0.5
    # -> total = 0.8 * 1.75 = 1.4
    post = _post("t3_floor", subreddit="SaaS")
    source = FakeListingSource({"SaaS": [post]})
    stats = _poll(store, source, min_final_score=1.4)
    assert stats.admitted == 1  # >= floor admits (inclusive)
    stats2 = _poll(store, FakeListingSource({"SaaS": [post]}), min_final_score=1.41)
    assert stats2.admitted == 0


def test_poll_paces_between_subreddits_not_before_first(store: ListeningStore) -> None:
    sleeper = RecordingSleep()
    source = FakeListingSource()
    _poll(store, source, sleep=sleeper)
    assert source.calls == ["CustomerSuccess", "SaaS"]
    assert sleeper.calls == [2.0]  # n-1 sleeps


def test_poll_zero_pace_never_sleeps(store: ListeningStore) -> None:
    sleeper = RecordingSleep()
    _poll(store, FakeListingSource(), pace_seconds=0.0, sleep=sleeper)
    assert sleeper.calls == []


def test_poll_one_failing_subreddit_does_not_abort_pass(store: ListeningStore) -> None:
    source = FakeListingSource(
        {"SaaS": [_post("t3_ok", subreddit="SaaS")]},
        error_on={"CustomerSuccess"},
    )
    stats = _poll(store, source)
    assert stats.admitted == 1
    assert len(stats.errors) == 1
    assert "CustomerSuccess" in stats.errors[0]


def test_poll_replay_preserves_triage_state(store: ListeningStore) -> None:
    """Re-polling can never resurrect a dismissed thread: the store's
    replay-safe upsert preserves status, and the digest filters on it."""
    source = FakeListingSource({"CustomerSuccess": [_post("t3_seen")]})
    _poll(store, source)
    store.set_candidate_status("t3_seen", "dismissed")
    _poll(store, FakeListingSource({"CustomerSuccess": [_post("t3_seen", score=99)]}),
          now=NOW + 60)
    row = store.get_candidate("t3_seen")
    assert row is not None
    assert row.status == "dismissed"  # preserved
    assert row.reddit_score == 99  # volatile fields refreshed
    assert [c.post_id for c in store.list_candidates(status="new")] == []


def test_poll_respects_per_subreddit_limit(store: ListeningStore) -> None:
    posts = [_post(f"t3_{i}") for i in range(10)]
    source = FakeListingSource({"CustomerSuccess": posts})
    stats = _poll(store, source, per_subreddit_limit=3)
    assert stats.fetched == 3


# -- CLI poll -------------------------------------------------------------------


def test_cli_poll_without_credentials_exits_cleanly(
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
    watchlist = tmp_path / "watchlist.toml"
    watchlist.write_text(
        "\n".join(
            [
                "version = 1",
                "[[subreddits]]",
                'name = "SaaS"',
                "[[topics]]",
                'name = "t"',
                'phrases = ["ticket deflection"]',
            ]
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "poll",
            "--db",
            str(tmp_path / "listening.db"),
            "--watchlist",
            str(watchlist),
        ]
    )
    assert code == 2
    assert "missing Reddit credentials" in capsys.readouterr().err


def test_cli_poll_missing_watchlist_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from atlas_reddit.__main__ import main

    code = main(
        [
            "poll",
            "--db",
            str(tmp_path / "listening.db"),
            "--watchlist",
            str(tmp_path / "nope.toml"),
        ]
    )
    assert code == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--limit-per-subreddit", "0"),
        ("--freshness-hours", "0"),
        ("--pace-seconds", "-1"),
        # Wave-1 class: CLI overrides must honor the same ceilings as the
        # typed settings fields (a 1000-post listing multiplies PRAW's
        # paginated requests past the verified budget posture).
        ("--limit-per-subreddit", "101"),
        ("--limit-per-subreddit", "1000"),
        ("--freshness-hours", "721"),
        ("--pace-seconds", "61"),
        # Wave-2: the floor knob joins the same shared-contract class.
        ("--min-score", "-1"),
        ("--min-score", "-0.5"),
    ],
)
def test_cli_poll_rejects_nonsense_knobs(tmp_path: Path, flag: str, value: str) -> None:
    from atlas_reddit.__main__ import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "poll",
                "--db",
                str(tmp_path / "x.db"),
                "--watchlist",
                str(tmp_path / "w.toml"),
                flag,
                value,
            ]
        )
    assert excinfo.value.code == 2


@pytest.mark.parametrize("bad", ["abc", "0", "-5"])
def test_env_typo_is_operator_error_for_every_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, bad: str
) -> None:
    """Wave-1 class: settings construction joins the exit-2 contract --
    a poller env typo must not traceback even for the digest command."""
    from atlas_reddit.__main__ import main

    monkeypatch.setenv("ATLAS_REDDIT_FRESHNESS_HOURS", bad)
    code = main(
        [
            "digest",
            "--db",
            str(tmp_path / "listening.db"),
            "--digest-dir",
            str(tmp_path / "d"),
            "--date",
            "2026-07-01",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "invalid ATLAS_REDDIT_" in err
    assert "Traceback" not in err
