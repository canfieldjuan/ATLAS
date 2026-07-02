"""Read-only Reddit access for the listening tool (S4, #1934).

Compliance is enforced in code, not prose (doc-verified 2026-07-02
against the PRAW authentication docs, the PRAW refresh-token tutorial,
and the reddit-archive OAuth2/API wikis; see
docs/REDDIT_LISTENING_SETUP_RUNBOOK.md):

- Auth is a scoped refresh token minted once via the documented
  authorization-code flow (``duration="permanent"``). Password-grant
  scope restriction is not documented anywhere, so the scoped refresh
  token is the only documented way to hold exactly
  ``identity``/``history``/``read``.
- :func:`validate_scopes` fails closed on ANY grant outside the allowed
  read-only set -- including the all-scopes wildcard ``*`` -- before a
  single listing request is made.
- The wrapper's public surface is read-only by construction: it exposes
  listing fetches and scope introspection, nothing else. A static test
  (``test_no_reddit_write_calls_anywhere``) additionally greps this
  package for Reddit write-API attribute usage.
- ``praw`` is imported lazily inside :class:`PrawListingSource` so the
  test suite (which fakes the transport boundary per the trial rules)
  never needs it installed, and ``check_for_updates=False`` is passed so
  PRAW cannot phone home to PyPI.
- The User-Agent follows the verified required format
  ``<platform>:<app ID>:<version> (by /u/<username>)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from . import __version__
from .config import RedditListeningSettings

# The full read-only grant set for this tool. Anything beyond these --
# including the '*' wildcard a password-grant token carries -- is refused.
ALLOWED_SCOPES = frozenset({"identity", "history", "read"})

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


class RedditAuthError(RuntimeError):
    """Raised when credentials are missing or the token's grants exceed
    the read-only contract."""


@dataclass(frozen=True)
class ListingPost:
    """Transport-neutral view of one listing submission."""

    post_id: str
    subreddit: str
    title: str
    url: str
    author: str | None
    created_utc: int
    score: int
    num_comments: int
    is_self: bool
    selftext: str


class ListingSource(Protocol):
    """The transport boundary the poller consumes. Tests provide a fake;
    production provides :class:`PrawListingSource`."""

    def fetch_new(self, subreddit: str, *, limit: int) -> list[ListingPost]:
        """Return newest submissions for one subreddit."""
        ...


@dataclass(frozen=True)
class OwnActivity:
    """One of the operator's own recent comments or submissions."""

    item_id: str
    thread_id: str
    created_utc: int


@dataclass(frozen=True)
class ThreadReply:
    """A reply observed on a tracked thread."""

    reply_id: str
    thread_id: str
    parent_id: str | None
    author: str | None
    body: str
    created_utc: int
    is_reply_to_me: bool


class HistorySource(Protocol):
    """The transport boundary the reply tracker consumes (S5). Tests
    provide a fake; production provides :class:`PrawHistorySource`."""

    def fetch_my_recent_comments(self, *, limit: int) -> list[OwnActivity]:
        ...

    def fetch_my_recent_submissions(self, *, limit: int) -> list[OwnActivity]:
        ...

    def fetch_thread_replies(
        self,
        thread_id: str,
        *,
        my_comment_ids: frozenset[str],
        include_top_level: bool,
    ) -> list[ThreadReply]:
        """Replies relevant to the operator on one thread: direct replies
        to their comments, plus top-level comments only when the thread is
        the operator's own submission (include_top_level)."""
        ...


def validate_scopes(
    granted: Iterable[str],
    *,
    required: frozenset[str] = frozenset({"read"}),
) -> frozenset[str]:
    """Fail closed on BOTH sides of the boundary: the grants must not
    exceed the read-only ceiling (the wildcard '*' is treated as the
    superset it is), and they must include the floor the caller needs --
    a token missing ``read`` would pass a subset-only check and then
    fail on every listing fetch downstream instead of failing here."""
    scopes = frozenset(granted)
    if not scopes:
        raise RedditAuthError("token carries no scopes; expected a subset of "
                              f"{sorted(ALLOWED_SCOPES)}")
    excess = scopes - ALLOWED_SCOPES
    if excess:
        raise RedditAuthError(
            "token grants exceed the read-only contract: "
            f"unexpected scopes {sorted(excess)}; allowed: {sorted(ALLOWED_SCOPES)}. "
            "Mint a scoped refresh token per docs/REDDIT_LISTENING_SETUP_RUNBOOK.md"
        )
    missing = required - scopes
    if missing:
        raise RedditAuthError(
            f"token is missing required scopes {sorted(missing)} "
            f"(granted: {sorted(scopes)}). Mint a scoped refresh token per "
            "docs/REDDIT_LISTENING_SETUP_RUNBOOK.md"
        )
    return scopes


def build_user_agent(username: str) -> str:
    """Descriptive UA in the format Reddit's API rules require:
    ``<platform>:<app ID>:<version> (by /u/<username>)``."""
    if not _USERNAME_RE.match(username or ""):
        raise RedditAuthError(
            f"ATLAS_REDDIT_USERNAME must match {_USERNAME_RE.pattern}, got {username!r}"
        )
    return f"linux:atlas-reddit-listening:v{__version__} (by /u/{username})"


class PrawListingSource:
    """Production ListingSource over PRAW. Read-only by construction:
    the public surface is fetch_new + granted_scopes, nothing else."""

    def __init__(self, settings: RedditListeningSettings) -> None:
        client_id = settings.client_id
        client_secret = settings.client_secret.get_secret_value()
        refresh_token = settings.refresh_token.get_secret_value()
        if not (client_id and client_secret and refresh_token and settings.username):
            raise RedditAuthError(
                "missing Reddit credentials: set ATLAS_REDDIT_CLIENT_ID, "
                "ATLAS_REDDIT_CLIENT_SECRET, ATLAS_REDDIT_REFRESH_TOKEN, and "
                "ATLAS_REDDIT_USERNAME (see docs/REDDIT_LISTENING_SETUP_RUNBOOK.md)"
            )
        user_agent = build_user_agent(settings.username)

        try:
            import praw  # lazy: tests fake the transport and never need it
        except ImportError as exc:
            raise RedditAuthError(
                "praw is not installed; run pip install -r requirements.txt"
            ) from exc

        # The whole praw-touching block maps into this class's error
        # contract: invalid/expired credentials surface from the
        # constructor or the scope probe as prawcore exceptions
        # (e.g. invalid_grant), never as raw tracebacks.
        try:
            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
                user_agent=user_agent,
                check_for_updates=False,
            )
            granted = self._reddit.auth.scopes()
        except Exception as exc:
            raise RedditAuthError(f"Reddit authentication failed: {exc}") from exc
        self._scopes = validate_scopes(granted)

    def granted_scopes(self) -> frozenset[str]:
        return self._scopes

    def fetch_new(self, subreddit: str, *, limit: int) -> list[ListingPost]:
        posts: list[ListingPost] = []
        for submission in self._reddit.subreddit(subreddit).new(limit=limit):
            author = getattr(submission.author, "name", None)
            posts.append(
                ListingPost(
                    post_id=submission.id,
                    subreddit=subreddit,
                    title=submission.title or "",
                    url=f"https://www.reddit.com{submission.permalink}",
                    author=author,
                    created_utc=int(submission.created_utc),
                    score=int(submission.score),
                    num_comments=int(submission.num_comments),
                    is_self=bool(submission.is_self),
                    selftext=submission.selftext or "",
                )
            )
        return posts


class PrawHistorySource:
    """Production HistorySource over PRAW. Read-only by construction;
    requires the full read/identity/history floor because own-activity
    discovery needs identity+history and reply reads need read."""

    _REQUIRED = frozenset({"read", "identity", "history"})

    def __init__(self, settings: RedditListeningSettings) -> None:
        client_id = settings.client_id
        client_secret = settings.client_secret.get_secret_value()
        refresh_token = settings.refresh_token.get_secret_value()
        if not (client_id and client_secret and refresh_token and settings.username):
            raise RedditAuthError(
                "missing Reddit credentials: set ATLAS_REDDIT_CLIENT_ID, "
                "ATLAS_REDDIT_CLIENT_SECRET, ATLAS_REDDIT_REFRESH_TOKEN, and "
                "ATLAS_REDDIT_USERNAME (see docs/REDDIT_LISTENING_SETUP_RUNBOOK.md)"
            )
        user_agent = build_user_agent(settings.username)

        try:
            import praw  # lazy: tests fake the transport and never need it
        except ImportError as exc:
            raise RedditAuthError(
                "praw is not installed; run pip install -r requirements.txt"
            ) from exc

        try:
            self._reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
                user_agent=user_agent,
                check_for_updates=False,
            )
            granted = self._reddit.auth.scopes()
        except Exception as exc:
            raise RedditAuthError(f"Reddit authentication failed: {exc}") from exc
        self._scopes = validate_scopes(granted, required=self._REQUIRED)
        try:
            self._me = self._reddit.user.me()
        except Exception as exc:
            raise RedditAuthError(f"Reddit identity lookup failed: {exc}") from exc

    def granted_scopes(self) -> frozenset[str]:
        return self._scopes

    def fetch_my_recent_comments(self, *, limit: int) -> list[OwnActivity]:
        items: list[OwnActivity] = []
        for comment in self._me.comments.new(limit=limit):
            items.append(
                OwnActivity(
                    item_id=comment.fullname,
                    thread_id=comment.link_id,
                    created_utc=int(comment.created_utc),
                )
            )
        return items

    def fetch_my_recent_submissions(self, *, limit: int) -> list[OwnActivity]:
        items: list[OwnActivity] = []
        for submission in self._me.submissions.new(limit=limit):
            items.append(
                OwnActivity(
                    item_id=submission.fullname,
                    thread_id=submission.fullname,
                    created_utc=int(submission.created_utc),
                )
            )
        return items

    def fetch_thread_replies(
        self,
        thread_id: str,
        *,
        my_comment_ids: frozenset[str],
        include_top_level: bool,
    ) -> list[ThreadReply]:
        """Direct replies come from refreshing each of the operator's own
        comments (precise and bounded by the operator's comment count --
        replace_more(limit=0) would silently DISCARD deep replies hidden
        behind MoreComments placeholders). Top-level responses are scanned
        only when the thread is the operator's own submission, with a
        bounded MoreComments budget (a single-user post rarely exceeds it;
        the budget is a named trade-off, not an accident)."""
        my_name = getattr(self._me, "name", None)
        replies: list[ThreadReply] = []
        seen_ids: set[str] = set()

        def _admit(comment, parent: str) -> None:
            author = getattr(comment.author, "name", None)
            if my_name and author == my_name:
                return  # own comments are not replies to surface
            if comment.fullname in seen_ids:
                return
            seen_ids.add(comment.fullname)
            replies.append(
                ThreadReply(
                    reply_id=comment.fullname,
                    thread_id=thread_id,
                    parent_id=parent,
                    author=author,
                    body=comment.body or "",
                    created_utc=int(comment.created_utc),
                    is_reply_to_me=True,
                )
            )

        for comment_fullname in sorted(my_comment_ids):
            own = self._reddit.comment(id=comment_fullname.removeprefix("t1_"))
            own.refresh()
            own.replies.replace_more(limit=None)  # one comment's children only
            for child in own.replies:
                _admit(child, comment_fullname)

        if include_top_level:
            submission = self._reddit.submission(id=thread_id.removeprefix("t3_"))
            submission.comments.replace_more(limit=16)
            for top in submission.comments:
                _admit(top, thread_id)

        return replies
