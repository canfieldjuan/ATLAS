"""Pure keyword/topic scoring for Reddit listening candidates.

No I/O, no network, no clock, no randomness: :func:`score_post` is a
deterministic function of its inputs so ranking is reproducible and
directly testable.

Semantics:

- A post scores 0.0 unless at least one topic phrase matches. The
  help-signal and question bonuses only amplify already-topical posts;
  they never make an off-topic post surface.
- Each matched topic contributes its weight once, regardless of how
  many of its phrases hit (matched phrases are reported in the
  breakdown for digest display, not multiplied into the score).
- Matching is case-insensitive on word boundaries, so a phrase does
  not match inside a longer word ("sla" does not match "island").
  Plural or variant forms need their own phrase entries in the
  watchlist; there is no stemming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import Watchlist


@dataclass(frozen=True)
class TopicHit:
    topic: str
    weight: float
    matched_phrases: tuple[str, ...]


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    subreddit_weight: float
    topic_hits: tuple[TopicHit, ...]
    help_signal_hits: tuple[str, ...]
    question_bonus_applied: float
    help_signal_bonus_applied: float


@lru_cache(maxsize=4096)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # Lookarounds instead of \b so phrases that start or end with a
    # non-word character still anchor correctly after re.escape.
    return re.compile(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)")


def score_post(
    *,
    title: str,
    body: str,
    subreddit_weight: float,
    watchlist: Watchlist,
) -> ScoreBreakdown:
    """Score one post against the watchlist. Deterministic and pure."""
    if subreddit_weight <= 0:
        raise ValueError(
            f"subreddit_weight must be positive, got {subreddit_weight!r} "
            "(weights come from a validated watchlist)"
        )
    text = f"{title}\n{body}".casefold()

    topic_hits: list[TopicHit] = []
    for topic in watchlist.topics:
        matched = tuple(
            phrase for phrase in topic.phrases if _phrase_pattern(phrase).search(text)
        )
        if matched:
            topic_hits.append(
                TopicHit(topic=topic.name, weight=topic.weight, matched_phrases=matched)
            )

    if not topic_hits:
        return ScoreBreakdown(
            total=0.0,
            subreddit_weight=subreddit_weight,
            topic_hits=(),
            help_signal_hits=(),
            question_bonus_applied=0.0,
            help_signal_bonus_applied=0.0,
        )

    help_hits = tuple(
        signal for signal in watchlist.help_signals if _phrase_pattern(signal).search(text)
    )
    help_bonus = watchlist.help_signal_bonus if help_hits else 0.0
    question_bonus = watchlist.question_bonus if "?" in title or "?" in body else 0.0

    base = sum(hit.weight for hit in topic_hits)
    total = subreddit_weight * (base + help_bonus + question_bonus)
    return ScoreBreakdown(
        total=round(total, 4),
        subreddit_weight=subreddit_weight,
        topic_hits=tuple(topic_hits),
        help_signal_hits=help_hits,
        question_bonus_applied=question_bonus,
        help_signal_bonus_applied=help_bonus,
    )
