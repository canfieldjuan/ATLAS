"""Scoring tests for atlas_reddit (slice S1, #1934).

Watchlists are built through the real parser (parse_watchlist), not
hand-built dataclasses, so the tests exercise the same objects
production uses. Scoring is pure, so every assertion is exact.
"""

from __future__ import annotations

import pytest

from atlas_reddit.config import parse_watchlist
from atlas_reddit.scoring import score_post


def _watchlist(**overrides: object):
    raw: dict = {
        "version": 1,
        "help_signal_bonus": 0.25,
        "question_bonus": 0.5,
        "help_signals": ["how do you", "any recommendations"],
        "subreddits": [{"name": "CustomerSuccess", "weight": 1.0}],
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
    raw.update(overrides)
    return parse_watchlist(raw)


def test_single_topic_single_phrase() -> None:
    result = score_post(
        title="Measuring ticket deflection",
        body="We rolled out a portal last month.",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.total == 1.0
    assert [h.topic for h in result.topic_hits] == ["ticket-deflection"]
    assert result.topic_hits[0].matched_phrases == ("ticket deflection",)
    assert result.question_bonus_applied == 0.0
    assert result.help_signal_bonus_applied == 0.0


def test_no_topic_match_scores_zero_even_with_question_and_help_signal() -> None:
    """Bonuses only amplify topical posts; they never surface off-topic ones."""
    result = score_post(
        title="How do you like your standing desk?",
        body="Any recommendations for a home office chair?",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.total == 0.0
    assert result.topic_hits == ()
    assert result.help_signal_hits == ()
    assert result.question_bonus_applied == 0.0


def test_multiple_topics_sum() -> None:
    result = score_post(
        title="Ticket deflection is up but customers ask the same question daily",
        body="",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert {h.topic for h in result.topic_hits} == {"ticket-deflection", "repeat-tickets"}
    assert result.total == 1.9  # 1.0 + 0.9, no bonuses


def test_multiple_phrases_in_one_topic_count_once() -> None:
    result = score_post(
        title="Our ticket deflection rate: deflection rate math inside",
        body="ticket deflection numbers for the quarter.",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert len(result.topic_hits) == 1
    hit = result.topic_hits[0]
    assert set(hit.matched_phrases) == {"ticket deflection", "deflection rate"}
    assert result.total == 1.0  # weight once, not per phrase


def test_question_bonus_applied_once() -> None:
    result = score_post(
        title="Is our ticket deflection rate normal?",
        body="Numbers inside? Really?",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.question_bonus_applied == 0.5
    assert result.total == 1.5


def test_help_signal_bonus_applied_once_for_multiple_signals() -> None:
    result = score_post(
        title="How do you improve ticket deflection",
        body="Any recommendations appreciated.",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert set(result.help_signal_hits) == {"how do you", "any recommendations"}
    assert result.help_signal_bonus_applied == 0.25
    assert result.total == 1.25


def test_matching_is_case_insensitive() -> None:
    result = score_post(
        title="TICKET DEFLECTION strategies",
        body="",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.total == 1.0


def test_phrase_does_not_match_inside_longer_word() -> None:
    watchlist = _watchlist(
        topics=[{"name": "sla", "weight": 1.0, "phrases": ["sla"]}],
        help_signals=[],
    )
    result = score_post(
        title="Visited an island to translate documents",
        body="",
        subreddit_weight=1.0,
        watchlist=watchlist,
    )
    assert result.total == 0.0

    hit = score_post(
        title="Our SLA is slipping",
        body="",
        subreddit_weight=1.0,
        watchlist=watchlist,
    )
    assert hit.total == 1.0


def test_no_stemming_plural_needs_own_phrase() -> None:
    """Documented behavior: 'repeat tickets' does not match 'repeat ticket'."""
    result = score_post(
        title="One repeat ticket ruined my week",
        body="",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.total == 0.0


def test_hyphenated_phrase_matches() -> None:
    watchlist = _watchlist(
        topics=[{"name": "self-service", "weight": 1.0, "phrases": ["self-service"]}],
        help_signals=[],
    )
    result = score_post(
        title="Improving our self-service portal.",
        body="",
        subreddit_weight=1.0,
        watchlist=watchlist,
    )
    assert result.total == 1.0


def test_subreddit_weight_scales_total() -> None:
    weighted = score_post(
        title="ticket deflection?",
        body="",
        subreddit_weight=0.8,
        watchlist=_watchlist(),
    )
    assert weighted.total == pytest.approx(0.8 * 1.5)
    assert weighted.subreddit_weight == 0.8


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_nonpositive_subreddit_weight_raises(weight: float) -> None:
    with pytest.raises(ValueError, match="subreddit_weight"):
        score_post(
            title="ticket deflection",
            body="",
            subreddit_weight=weight,
            watchlist=_watchlist(),
        )


def test_near_miss_neutral_text_scores_zero() -> None:
    """Topic vocabulary nearby is not a match: 'ticket' alone is not
    'ticket deflection', and measurement language stays neutral."""
    result = score_post(
        title="We use page views as one signal for the ticket queue",
        body="Tracking deflection-adjacent metrics like search exits.",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.total == 0.0


def test_regex_metacharacters_in_phrases_match_literally() -> None:
    """Phrases are operator-typed text, not regexes: metachars must not
    crash compilation or change matching semantics."""
    watchlist = _watchlist(
        topics=[
            {"name": "cpp", "weight": 1.0, "phrases": ["c++"]},
            {"name": "paren", "weight": 1.0, "phrases": ["(deflection)"]},
        ],
        help_signals=[],
    )
    result = score_post(
        title="Moving our c++ tooling",
        body="Our (deflection) numbers are flat.",
        subreddit_weight=1.0,
        watchlist=watchlist,
    )
    assert {h.topic for h in result.topic_hits} == {"cpp", "paren"}
    assert result.total == 2.0

    miss = score_post(
        title="Moving our cpp tooling",
        body="",
        subreddit_weight=1.0,
        watchlist=watchlist,
    )
    assert miss.total == 0.0


def test_emoji_adjacent_phrase_still_matches() -> None:
    """Reddit titles wrap words in emoji; non-word neighbors must not
    defeat the word-boundary lookarounds."""
    result = score_post(
        title="\N{FIRE}ticket deflection\N{FIRE} wins this quarter",
        body="",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.total == 1.0


def test_body_only_match_counts() -> None:
    result = score_post(
        title="Quarterly support review",
        body="The deflection rate dropped after the docs migration.",
        subreddit_weight=1.0,
        watchlist=_watchlist(),
    )
    assert result.total == 1.0


def test_deterministic_repeat_calls() -> None:
    kwargs = dict(
        title="Is our ticket deflection rate normal?",
        body="how do you measure it",
        subreddit_weight=0.9,
        watchlist=_watchlist(),
    )
    assert score_post(**kwargs) == score_post(**kwargs)
