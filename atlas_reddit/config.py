"""Configuration for the read-only Reddit listening tool.

Two config surfaces:

- Environment settings: typed ``ATLAS_REDDIT_*`` fields on
  :class:`RedditListeningSettings` (pydantic-settings; never raw
  ``os.environ``). Paths and, in later slices, credentials live here.
- Watchlist: a human-edited TOML file (stdlib ``tomllib``, so no new
  dependency) holding the subreddit list, topic phrase clusters, and
  scoring knobs. Curation data lives in the file, not in env vars and
  not in the database.

The watchlist parser fails closed: unknown keys, missing required
sections, out-of-range weights, and malformed entries raise
:class:`WatchlistError` instead of being defaulted or skipped.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")

WATCHLIST_VERSION = 1

# Reddit subreddit names: 3-21 chars, letters/digits/underscore, and the
# first character may not be an underscore. Matched with fullmatch(): a
# "$" anchor would tolerate a trailing newline ("SaaS\n") and admit a
# name that later breaks subreddit_weight() lookups.
_SUBREDDIT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{2,20}")

# Weights are multipliers, not probabilities; 10 is an arbitrary sanity
# ceiling so a typo like 100 is rejected rather than dominating ranking.
_MAX_WEIGHT = 10.0
_MAX_BONUS = 5.0

# Poller knob ceilings: single source of truth for the settings fields'
# le= bounds AND the CLI override checks, so the two entry paths cannot
# drift (a CLI value above the cap would multiply PRAW's paginated
# requests and violate the verified request-budget posture).
MAX_FRESHNESS_HOURS = 720
MAX_PER_SUBREDDIT_LIMIT = 100
MAX_PACE_SECONDS = 60.0

_ALLOWED_TOP_KEYS = {
    "version",
    "subreddits",
    "topics",
    "help_signals",
    "question_bonus",
    "help_signal_bonus",
}
_ALLOWED_SUBREDDIT_KEYS = {"name", "weight"}
_ALLOWED_TOPIC_KEYS = {"name", "phrases", "weight"}


class RedditListeningSettings(BaseSettings):
    """Environment-backed settings (env prefix ``ATLAS_REDDIT_``)."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_REDDIT_",
        env_file=ENV_FILES,
        extra="ignore",
    )

    watchlist_path: Path = Field(
        default=Path("data/atlas_reddit/watchlist.toml"),
        description=(
            "Path to the local watchlist TOML. The default sits under the "
            "gitignored data/ tree; copy atlas_reddit/watchlist.sample.toml "
            "there to start."
        ),
    )
    db_path: Path = Field(
        default=Path("data/atlas_reddit/listening.db"),
        description=(
            "Path to the local SQLite state file (candidates, tracked "
            "threads, replies, purge log). Defaults under the gitignored "
            "data/ tree."
        ),
    )
    digest_dir: Path = Field(
        default=Path("data/atlas_reddit/digests"),
        description=(
            "Directory for daily Markdown digests (YYYY-MM-DD.md). "
            "Defaults under the gitignored data/ tree."
        ),
    )
    client_id: str = Field(
        default="",
        description="Reddit script-app client id (env only, never committed).",
    )
    client_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Reddit script-app client secret (env only).",
    )
    refresh_token: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Scoped refresh token minted once via the documented "
            "authorization-code flow with scopes identity/history/read "
            "only (docs/REDDIT_LISTENING_SETUP_RUNBOOK.md)."
        ),
    )
    username: str = Field(
        default="",
        description="Reddit username for the descriptive User-Agent.",
    )
    freshness_hours: int = Field(
        default=48,
        ge=1,
        le=MAX_FRESHNESS_HOURS,
        description="Radar admits posts younger than this many hours.",
    )
    per_subreddit_limit: int = Field(
        default=50,
        ge=1,
        le=MAX_PER_SUBREDDIT_LIMIT,
        description="Newest submissions fetched per subreddit per pass.",
    )
    pace_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=MAX_PACE_SECONDS,
        description=(
            "Polite sleep between subreddit fetches (Reddit allows 60 "
            "requests/min for OAuth clients; this keeps a wide margin)."
        ),
    )
    poll_min_score: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum final score for a post to be stored as a candidate.",
    )


class WatchlistError(ValueError):
    """Raised when the watchlist file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class SubredditEntry:
    name: str
    weight: float = 1.0


@dataclass(frozen=True)
class Topic:
    name: str
    phrases: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class Watchlist:
    version: int
    subreddits: tuple[SubredditEntry, ...]
    topics: tuple[Topic, ...]
    help_signals: tuple[str, ...] = ()
    question_bonus: float = 0.5
    help_signal_bonus: float = 0.25
    _weights_by_name: dict[str, float] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_weights_by_name",
            {entry.name.casefold(): entry.weight for entry in self.subreddits},
        )

    def subreddit_weight(self, name: str) -> float | None:
        """Weight for a watched subreddit, or None if it is not watched."""
        return self._weights_by_name.get(name.casefold())


def _require_number(value: object, *, context: str, maximum: float, allow_zero: bool) -> float:
    # bool is a subclass of int, and TOML `weight = true` would otherwise
    # silently become 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WatchlistError(f"{context} must be a number, got {value!r}")
    number = float(value)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not lower_ok or number > maximum:
        bound = "0" if allow_zero else "greater than 0"
        raise WatchlistError(
            f"{context} must be {bound} and at most {maximum}, got {value!r}"
        )
    return number


def _require_table(value: object, *, context: str) -> dict:
    if not isinstance(value, dict):
        raise WatchlistError(f"{context} must be a table, got {value!r}")
    return value


def _reject_unknown_keys(raw: dict, allowed: set[str], *, context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise WatchlistError(f"{context} has unknown keys: {', '.join(unknown)}")


def _parse_subreddits(raw: object) -> tuple[SubredditEntry, ...]:
    if not isinstance(raw, list) or not raw:
        raise WatchlistError("subreddits must be a non-empty array of tables")
    entries: list[SubredditEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        context = f"subreddits[{index}]"
        table = _require_table(item, context=context)
        _reject_unknown_keys(table, _ALLOWED_SUBREDDIT_KEYS, context=context)
        name = table.get("name")
        if not isinstance(name, str) or not _SUBREDDIT_NAME_RE.fullmatch(name):
            raise WatchlistError(
                f"{context}.name must fully match {_SUBREDDIT_NAME_RE.pattern}, got {name!r}"
            )
        if name.casefold() in seen:
            raise WatchlistError(f"duplicate subreddit name: {name!r}")
        seen.add(name.casefold())
        weight = _require_number(
            table.get("weight", 1.0),
            context=f"{context}.weight",
            maximum=_MAX_WEIGHT,
            allow_zero=False,
        )
        entries.append(SubredditEntry(name=name, weight=weight))
    return tuple(entries)


def _parse_phrases(raw: object, *, context: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(raw, list) or (not raw and not allow_empty):
        shape = "an array" if allow_empty else "a non-empty array"
        raise WatchlistError(f"{context} must be {shape} of strings")
    phrases: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise WatchlistError(f"{context}[{index}] must be a non-empty string, got {item!r}")
        phrase = item.strip()
        if phrase.casefold() in seen:
            raise WatchlistError(f"{context} has duplicate phrase: {phrase!r}")
        seen.add(phrase.casefold())
        phrases.append(phrase)
    return tuple(phrases)


def _parse_topics(raw: object) -> tuple[Topic, ...]:
    if not isinstance(raw, list) or not raw:
        raise WatchlistError("topics must be a non-empty array of tables")
    topics: list[Topic] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        context = f"topics[{index}]"
        table = _require_table(item, context=context)
        _reject_unknown_keys(table, _ALLOWED_TOPIC_KEYS, context=context)
        raw_name = table.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise WatchlistError(
                f"{context}.name must be a non-empty string, got {raw_name!r}"
            )
        # Normalize BEFORE the duplicate check so it compares the same
        # value that gets stored; checking the untrimmed name would let
        # " ticket-deflection " slip past "ticket-deflection" and
        # double-count that topic's weight in score_post.
        name = raw_name.strip()
        if name.casefold() in seen:
            raise WatchlistError(f"duplicate topic name: {name!r}")
        seen.add(name.casefold())
        phrases = _parse_phrases(table.get("phrases"), context=f"{context}.phrases")
        weight = _require_number(
            table.get("weight", 1.0),
            context=f"{context}.weight",
            maximum=_MAX_WEIGHT,
            allow_zero=False,
        )
        topics.append(Topic(name=name, phrases=phrases, weight=weight))
    return tuple(topics)


def parse_watchlist(raw: dict) -> Watchlist:
    """Validate a parsed TOML document into a :class:`Watchlist`.

    Pure function of the input dict; all validation errors raise
    :class:`WatchlistError` naming the offending key.
    """
    if not isinstance(raw, dict):
        raise WatchlistError(f"watchlist must be a table, got {type(raw).__name__}")
    _reject_unknown_keys(raw, _ALLOWED_TOP_KEYS, context="watchlist")
    version = raw.get("version")
    # Exact-int check: bool is a subclass of int (version = true), and a
    # TOML float compares equal to the int (1.0 == 1), so both would
    # otherwise slip past a plain != comparison.
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != WATCHLIST_VERSION
    ):
        raise WatchlistError(
            f"watchlist must declare version = {WATCHLIST_VERSION} (integer), got {version!r}"
        )
    subreddits = _parse_subreddits(raw.get("subreddits"))
    topics = _parse_topics(raw.get("topics"))
    # An explicitly empty help_signals array is valid config (the bonus
    # feature unused), unlike empty subreddits/topics which are nonsensical.
    help_signals = (
        _parse_phrases(raw["help_signals"], context="help_signals", allow_empty=True)
        if "help_signals" in raw
        else ()
    )
    question_bonus = _require_number(
        raw.get("question_bonus", 0.5),
        context="question_bonus",
        maximum=_MAX_BONUS,
        allow_zero=True,
    )
    help_signal_bonus = _require_number(
        raw.get("help_signal_bonus", 0.25),
        context="help_signal_bonus",
        maximum=_MAX_BONUS,
        allow_zero=True,
    )
    return Watchlist(
        version=WATCHLIST_VERSION,
        subreddits=subreddits,
        topics=topics,
        help_signals=help_signals,
        question_bonus=question_bonus,
        help_signal_bonus=help_signal_bonus,
    )


def load_watchlist(path: Path) -> Watchlist:
    """Load and validate the watchlist TOML at ``path``."""
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise WatchlistError(
            f"watchlist file not found: {path} "
            "(copy atlas_reddit/watchlist.sample.toml there to start)"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise WatchlistError(f"watchlist TOML is malformed: {path}: {exc}") from exc
    return parse_watchlist(raw)
