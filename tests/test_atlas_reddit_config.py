"""Watchlist config validation tests for atlas_reddit (slice S1, #1934).

The watchlist parser is guard-shaped, so every detection branch gets a
negative fixture (AGENTS.md 3i): unknown keys, bad versions, invalid
names, out-of-range weights, bool-typed numbers, duplicates, and
malformed/missing files. Boundary values are probed on both sides.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from atlas_reddit.config import (
    RedditListeningSettings,
    Watchlist,
    WatchlistError,
    load_watchlist,
    parse_watchlist,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _valid_raw() -> dict:
    return {
        "version": 1,
        "help_signal_bonus": 0.25,
        "question_bonus": 0.5,
        "help_signals": ["how do you", "any recommendations"],
        "subreddits": [
            {"name": "CustomerSuccess", "weight": 1.0},
            {"name": "SaaS", "weight": 0.8},
        ],
        "topics": [
            {
                "name": "ticket-deflection",
                "weight": 1.0,
                "phrases": ["ticket deflection", "deflection rate"],
            },
            {
                "name": "repeat-tickets",
                "weight": 0.9,
                "phrases": ["same question", "repeat tickets"],
            },
        ],
    }


def test_parse_valid_full() -> None:
    watchlist = parse_watchlist(_valid_raw())
    assert isinstance(watchlist, Watchlist)
    assert watchlist.version == 1
    assert [s.name for s in watchlist.subreddits] == ["CustomerSuccess", "SaaS"]
    assert watchlist.subreddits[1].weight == 0.8
    assert [t.name for t in watchlist.topics] == ["ticket-deflection", "repeat-tickets"]
    assert watchlist.topics[0].phrases == ("ticket deflection", "deflection rate")
    assert watchlist.help_signals == ("how do you", "any recommendations")
    assert watchlist.question_bonus == 0.5
    assert watchlist.help_signal_bonus == 0.25


def test_parse_defaults_applied() -> None:
    raw = {
        "version": 1,
        "subreddits": [{"name": "SaaS"}],
        "topics": [{"name": "t", "phrases": ["ticket deflection"]}],
    }
    watchlist = parse_watchlist(raw)
    assert watchlist.subreddits[0].weight == 1.0
    assert watchlist.topics[0].weight == 1.0
    assert watchlist.help_signals == ()
    assert watchlist.question_bonus == 0.5
    assert watchlist.help_signal_bonus == 0.25


def test_subreddit_weight_lookup_is_case_insensitive() -> None:
    watchlist = parse_watchlist(_valid_raw())
    assert watchlist.subreddit_weight("customersuccess") == 1.0
    assert watchlist.subreddit_weight("CUSTOMERSUCCESS") == 1.0
    assert watchlist.subreddit_weight("NotWatched") is None


@pytest.mark.parametrize(
    "version",
    [
        None,
        0,
        2,
        "1",
        True,
        1.0,  # Codex-cited: float compares equal to int 1 under !=
        0.0,
        2.0,
        1.5,
    ],
)
def test_version_must_be_exactly_integer_one(version: object) -> None:
    raw = _valid_raw()
    if version is None:
        del raw["version"]
    else:
        raw["version"] = version
    with pytest.raises(WatchlistError, match="version"):
        parse_watchlist(raw)


@pytest.mark.parametrize("raw", ["not a table", [{"version": 1}], 42, None])
def test_non_table_document_rejected_with_watchlist_error(raw: object) -> None:
    """Public-API contract: invalid input raises WatchlistError, never a
    bare AttributeError/TypeError from the internals."""
    with pytest.raises(WatchlistError, match="table"):
        parse_watchlist(raw)  # type: ignore[arg-type]


def test_unknown_top_level_key_rejected() -> None:
    raw = _valid_raw()
    raw["surprise"] = True
    with pytest.raises(WatchlistError, match="surprise"):
        parse_watchlist(raw)


def test_unknown_subreddit_key_rejected() -> None:
    raw = _valid_raw()
    raw["subreddits"][0]["list_type"] = "core"
    with pytest.raises(WatchlistError, match="list_type"):
        parse_watchlist(raw)


def test_unknown_topic_key_rejected() -> None:
    raw = _valid_raw()
    raw["topics"][0]["llm"] = "never"
    with pytest.raises(WatchlistError, match="llm"):
        parse_watchlist(raw)


@pytest.mark.parametrize("value", [None, [], "SaaS", [1, 2]])
def test_subreddits_must_be_nonempty_array_of_tables(value: object) -> None:
    raw = _valid_raw()
    if value is None:
        del raw["subreddits"]
    else:
        raw["subreddits"] = value
    with pytest.raises(WatchlistError, match="subreddits"):
        parse_watchlist(raw)


@pytest.mark.parametrize(
    "name", ["", "ab", "a" * 22, "bad name", "bad-name", "_lead", 123, None]
)
def test_subreddit_name_invalid(name: object) -> None:
    raw = _valid_raw()
    raw["subreddits"][0] = {"name": name} if name is not None else {}
    with pytest.raises(WatchlistError, match="name"):
        parse_watchlist(raw)


def test_subreddit_name_length_boundaries_pass() -> None:
    raw = _valid_raw()
    raw["subreddits"] = [{"name": "abc"}, {"name": "a" * 21}]
    watchlist = parse_watchlist(raw)
    assert [s.name for s in watchlist.subreddits] == ["abc", "a" * 21]


@pytest.mark.parametrize("weight", [0, 0.0, -1, 10.5, "1.0", True, False])
def test_subreddit_weight_invalid(weight: object) -> None:
    raw = _valid_raw()
    raw["subreddits"][0]["weight"] = weight
    with pytest.raises(WatchlistError, match="weight"):
        parse_watchlist(raw)


def test_weight_boundaries_pass() -> None:
    raw = _valid_raw()
    raw["subreddits"][0]["weight"] = 10.0
    raw["topics"][0]["weight"] = 0.0001
    watchlist = parse_watchlist(raw)
    assert watchlist.subreddits[0].weight == 10.0
    assert watchlist.topics[0].weight == 0.0001


def test_duplicate_subreddit_names_rejected_case_insensitive() -> None:
    raw = _valid_raw()
    raw["subreddits"].append({"name": "customersuccess"})
    with pytest.raises(WatchlistError, match="duplicate subreddit"):
        parse_watchlist(raw)


def test_mixed_valid_and_invalid_subreddits_rejected() -> None:
    raw = _valid_raw()
    raw["subreddits"] = [{"name": "GoodName"}, {"name": "bad name"}]
    with pytest.raises(WatchlistError, match=r"subreddits\[1\]"):
        parse_watchlist(raw)


@pytest.mark.parametrize("value", [None, [], "topic"])
def test_topics_must_be_nonempty_array_of_tables(value: object) -> None:
    raw = _valid_raw()
    if value is None:
        del raw["topics"]
    else:
        raw["topics"] = value
    with pytest.raises(WatchlistError, match="topics"):
        parse_watchlist(raw)


@pytest.mark.parametrize("phrases", [None, [], [""], ["  "], [123], ["ok", "ok"]])
def test_topic_phrases_invalid(phrases: object) -> None:
    raw = _valid_raw()
    if phrases is None:
        del raw["topics"][0]["phrases"]
    else:
        raw["topics"][0]["phrases"] = phrases
    with pytest.raises(WatchlistError, match="phrases"):
        parse_watchlist(raw)


def test_duplicate_topic_names_rejected() -> None:
    raw = _valid_raw()
    raw["topics"][1]["name"] = "Ticket-Deflection"
    with pytest.raises(WatchlistError, match="duplicate topic"):
        parse_watchlist(raw)


@pytest.mark.parametrize(
    "variant",
    [
        " ticket-deflection ",  # Codex-cited case
        "\tticket-deflection",
        "TICKET-DEFLECTION  ",
        "  Ticket-Deflection\t",
        "ticket-deflection\n",
    ],
)
def test_topic_duplicate_check_normalizes_whitespace_and_case(variant: str) -> None:
    """Class fix for the check/store mismatch: the duplicate check must
    compare the same normalized value that gets stored, or a curation
    typo silently double-counts the topic's weight in score_post."""
    raw = _valid_raw()
    raw["topics"][1]["name"] = variant
    with pytest.raises(WatchlistError, match="duplicate topic"):
        parse_watchlist(raw)


def test_topic_name_stored_stripped() -> None:
    raw = _valid_raw()
    raw["topics"][0]["name"] = "  spaced-name\t"
    watchlist = parse_watchlist(raw)
    assert watchlist.topics[0].name == "spaced-name"


@pytest.mark.parametrize("name", [None, "", "   ", 42, ["x"]])
def test_topic_name_invalid(name: object) -> None:
    raw = _valid_raw()
    if name is None:
        del raw["topics"][0]["name"]
    else:
        raw["topics"][0]["name"] = name
    with pytest.raises(WatchlistError, match="name"):
        parse_watchlist(raw)


@pytest.mark.parametrize(
    "name",
    [
        "SaaS\n",  # Codex-cited case: "$" would match before the final newline
        "abc\n",
        "a1b\n",
        "\nSaaS",
        "Sa\naS",
        "SaaS\n\n",
        "helpdesk\r",
    ],
)
def test_subreddit_name_with_newline_or_cr_rejected(name: str) -> None:
    """Class fix for the anchor trap: fullmatch leaves no position where
    a stray newline survives validation and then breaks
    subreddit_weight() lookups or a future poller's subreddit fetch."""
    raw = _valid_raw()
    raw["subreddits"][0]["name"] = name
    with pytest.raises(WatchlistError, match="name"):
        parse_watchlist(raw)


@pytest.mark.parametrize("signals", [[""], [42], ["dup", "DUP"], "how do you"])
def test_help_signals_invalid(signals: object) -> None:
    raw = _valid_raw()
    raw["help_signals"] = signals
    with pytest.raises(WatchlistError, match="help_signals"):
        parse_watchlist(raw)


def test_help_signals_empty_list_allowed() -> None:
    """Explicitly empty help_signals is valid: the bonus feature unused."""
    raw = _valid_raw()
    raw["help_signals"] = []
    assert parse_watchlist(raw).help_signals == ()


@pytest.mark.parametrize("key", ["question_bonus", "help_signal_bonus"])
@pytest.mark.parametrize("value", [-0.1, 5.1, True, "0.5"])
def test_bonus_out_of_range_or_wrong_type(key: str, value: object) -> None:
    raw = _valid_raw()
    raw[key] = value
    with pytest.raises(WatchlistError, match=key):
        parse_watchlist(raw)


@pytest.mark.parametrize("key", ["question_bonus", "help_signal_bonus"])
def test_bonus_zero_is_allowed(key: str) -> None:
    raw = _valid_raw()
    raw[key] = 0.0
    watchlist = parse_watchlist(raw)
    assert getattr(watchlist, key) == 0.0


def test_parse_does_not_mutate_input() -> None:
    raw = _valid_raw()
    snapshot = copy.deepcopy(raw)
    parse_watchlist(raw)
    assert raw == snapshot


def test_load_watchlist_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "watchlist.toml"
    path.write_text(
        "\n".join(
            [
                "version = 1",
                "[[subreddits]]",
                'name = "SaaS"',
                "weight = 0.8",
                "[[topics]]",
                'name = "deflection"',
                'phrases = ["ticket deflection"]',
            ]
        ),
        encoding="utf-8",
    )
    watchlist = load_watchlist(path)
    assert watchlist.subreddit_weight("saas") == 0.8
    assert watchlist.topics[0].phrases == ("ticket deflection",)


def test_load_watchlist_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(WatchlistError, match="not found"):
        load_watchlist(missing)


def test_load_watchlist_malformed_toml(tmp_path: Path) -> None:
    path = tmp_path / "watchlist.toml"
    path.write_text("version = = 1", encoding="utf-8")
    with pytest.raises(WatchlistError, match="malformed"):
        load_watchlist(path)


def test_committed_sample_watchlist_is_valid() -> None:
    """The shipped sample must always parse through the real loader."""
    sample = REPO_ROOT / "atlas_reddit" / "watchlist.sample.toml"
    watchlist = load_watchlist(sample)
    assert watchlist.subreddit_weight("CustomerSuccess") == 1.0
    assert any(t.name == "ticket-deflection" for t in watchlist.topics)
    assert watchlist.help_signals


def test_settings_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_REDDIT_WATCHLIST_PATH", raising=False)
    settings = RedditListeningSettings(_env_file=None)
    assert settings.watchlist_path == Path("data/atlas_reddit/watchlist.toml")


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_REDDIT_WATCHLIST_PATH", "/tmp/custom.toml")
    settings = RedditListeningSettings(_env_file=None)
    assert settings.watchlist_path == Path("/tmp/custom.toml")


def test_settings_db_path_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_REDDIT_DB_PATH", raising=False)
    assert RedditListeningSettings(_env_file=None).db_path == Path(
        "data/atlas_reddit/listening.db"
    )
    monkeypatch.setenv("ATLAS_REDDIT_DB_PATH", "/tmp/custom.db")
    assert RedditListeningSettings(_env_file=None).db_path == Path("/tmp/custom.db")
