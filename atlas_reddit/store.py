"""SQLite persistence for the read-only Reddit listening tool (S2, #1934).

Local, inspectable state behind stdlib ``sqlite3`` -- no new
dependencies, no network, no ORM. One store file (default
``data/atlas_reddit/listening.db``, gitignored) holds:

- ``candidates``: scored radar posts with a review status
  (new/seen/dismissed/responded).
- ``tracked_threads``: threads Juan participated in (reply tracker).
- ``replies``: replies observed on tracked threads, with seen state.
- ``purge_log``: the deletion-compliance audit trail. The purge *fields*
  exist from this slice, before any live ingestion; the purge *job* is
  slice S6.

Design rules:

- Timestamps are caller-supplied unix ints. The clock is an external
  boundary, so it enters as data instead of being read inside the store;
  tests stay deterministic without mocking.
- Writes are idempotent where ingestion can replay: candidate upserts
  preserve ``first_seen`` and ``status`` and refuse to regress fresher
  state from stale out-of-order windows; reply inserts ignore only
  duplicate ``reply_id`` replays -- any other integrity violation raises
  instead of being masked as a replay.
- State mutations fail closed: an unknown id or status raises
  :class:`StoreError` instead of silently doing nothing, and the schema
  double-enforces enums via CHECK constraints.
- The schema is versioned with ``PRAGMA user_version``; opening a store
  written by a *newer* schema raises instead of guessing.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 2

CANDIDATE_STATUSES = ("new", "seen", "dismissed", "responded")
PURGE_ITEM_TYPES = ("candidate", "reply", "thread")

_SCHEMA_DDL = """
CREATE TABLE candidates (
    post_id TEXT PRIMARY KEY NOT NULL,
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    author TEXT,
    created_utc INTEGER NOT NULL,
    reddit_score INTEGER NOT NULL DEFAULT 0,
    num_comments INTEGER NOT NULL DEFAULT 0,
    keyword_score REAL NOT NULL,
    final_score REAL NOT NULL,
    matched_topics TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'new'
        CHECK (status IN ('new', 'seen', 'dismissed', 'responded')),
    first_seen INTEGER NOT NULL,
    last_seen INTEGER NOT NULL
);
CREATE INDEX idx_candidates_status_score
    ON candidates (status, final_score DESC);
CREATE TABLE tracked_threads (
    thread_id TEXT PRIMARY KEY NOT NULL,
    my_comment_ids TEXT NOT NULL DEFAULT '[]',
    last_checked INTEGER,
    dormant INTEGER NOT NULL DEFAULT 0 CHECK (dormant IN (0, 1)),
    is_own_submission INTEGER NOT NULL DEFAULT 0
        CHECK (is_own_submission IN (0, 1)),
    last_activity INTEGER
);
CREATE TABLE replies (
    reply_id TEXT PRIMARY KEY NOT NULL,
    thread_id TEXT NOT NULL,
    parent_id TEXT,
    author TEXT,
    body TEXT NOT NULL DEFAULT '',
    created_utc INTEGER NOT NULL,
    is_reply_to_me INTEGER NOT NULL CHECK (is_reply_to_me IN (0, 1)),
    seen INTEGER NOT NULL DEFAULT 0 CHECK (seen IN (0, 1)),
    FOREIGN KEY (thread_id) REFERENCES tracked_threads (thread_id)
);
CREATE INDEX idx_replies_thread ON replies (thread_id, seen);
CREATE TABLE purge_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    item_type TEXT NOT NULL
        CHECK (item_type IN ('candidate', 'reply', 'thread')),
    deleted_detected_at INTEGER NOT NULL,
    purged_at INTEGER NOT NULL,
    reason TEXT NOT NULL
);
"""


class StoreError(RuntimeError):
    """Raised on invalid store operations (unknown ids, bad enums,
    incompatible schema versions)."""


def _require_id(value: object, *, field: str) -> str:
    """Fail closed on malformed external ids. SQLite TEXT PRIMARY KEY does
    not imply NOT NULL (the DDL adds it explicitly), and an empty string
    would satisfy NOT NULL while still being a corrupt id -- both are
    rejected here before any SQL runs."""
    if not isinstance(value, str) or not value:
        raise StoreError(f"{field} must be a non-empty string, got {value!r}")
    return value


def _require_int(value: object, *, field: str) -> int:
    """Fail closed on non-int numerics. SQLite type affinity stores text
    in INTEGER columns at runtime, which would flow into rendered output
    or crash formatters downstream."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreError(f"{field} must be an int, got {value!r}")
    return value


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StoreError(f"{field} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise StoreError(f"{field} must be finite, got {value!r}")
    return number


def _require_bool(value: object, *, field: str) -> bool:
    """Fail closed on non-bool flags. Truthiness coercion would turn a
    malformed parser value like "false" or None into valid persisted
    state, bypassing the CHECK constraint's intent."""
    if not isinstance(value, bool):
        raise StoreError(f"{field} must be a bool, got {value!r}")
    return value


def _require_comment_ids(value: object) -> list[str]:
    """Validate a collection of comment ids. A bare string is the classic
    degenerate iterable (it would persist one character per id), so
    str/bytes collections are rejected, and every element must be a
    non-empty string."""
    if isinstance(value, (str, bytes)):
        raise StoreError(
            f"my_comment_ids must be a collection of ids, not a bare {type(value).__name__}"
        )
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise StoreError(f"my_comment_ids must be iterable, got {value!r}") from exc
    return [_require_id(item, field="my_comment_ids element") for item in items]


@dataclass(frozen=True)
class Candidate:
    post_id: str
    subreddit: str
    title: str
    url: str
    author: str | None
    created_utc: int
    reddit_score: int
    num_comments: int
    keyword_score: float
    final_score: float
    matched_topics: tuple[str, ...]
    status: str
    first_seen: int
    last_seen: int


@dataclass(frozen=True)
class TrackedThread:
    thread_id: str
    my_comment_ids: tuple[str, ...]
    last_checked: int | None
    dormant: bool
    is_own_submission: bool
    last_activity: int | None


@dataclass(frozen=True)
class Reply:
    reply_id: str
    thread_id: str
    parent_id: str | None
    author: str | None
    body: str
    created_utc: int
    is_reply_to_me: bool
    seen: bool


@dataclass(frozen=True)
class PurgeRecord:
    item_id: str
    item_type: str
    deleted_detected_at: int
    purged_at: int
    reason: str


class ListeningStore:
    """Repository over one local SQLite file. Context-manager friendly."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # The whole open path is wrapped, not just the schema probe:
            # connect() itself raises (e.g. path is a directory) before
            # _ensure_schema could translate anything.
            self._conn = sqlite3.connect(path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open store at {path}: {exc}") from exc
        self._ensure_schema()

    def __enter__(self) -> "ListeningStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        try:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.Error as exc:
            # A corrupt or non-SQLite file is a store-open failure and
            # belongs to this class's StoreError contract, not a raw
            # sqlite3 traceback for every caller to re-handle.
            raise StoreError(f"cannot open store at {self._path}: {exc}") from exc
        if version == SCHEMA_VERSION:
            return
        if version == 0:
            with self._conn:
                self._conn.executescript(_SCHEMA_DDL)
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if version == 1:
            # v1 -> v2 (S5): tracked_threads gains the own-submission flag
            # (top-level replies are only "to me" on my own posts) and a
            # persisted last_activity timestamp (history-window eviction
            # must not read as inactivity). Additive, backfilled from
            # stored replies; no data is dropped.
            with self._conn:
                self._conn.execute(
                    "ALTER TABLE tracked_threads ADD COLUMN is_own_submission "
                    "INTEGER NOT NULL DEFAULT 0 CHECK (is_own_submission IN (0, 1))"
                )
                self._conn.execute(
                    "ALTER TABLE tracked_threads ADD COLUMN last_activity INTEGER"
                )
                self._conn.execute(
                    """
                    UPDATE tracked_threads SET last_activity = (
                        SELECT MAX(created_utc) FROM replies
                        WHERE replies.thread_id = tracked_threads.thread_id
                    )
                    """
                )
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        # Fail closed on schemas this code does not know how to read --
        # both newer versions and unknown values.
        raise StoreError(
            f"unsupported schema version {version} at {self._path} "
            f"(this build supports version {SCHEMA_VERSION})"
        )

    # -- candidates ------------------------------------------------------

    def upsert_candidate(
        self,
        *,
        post_id: str,
        subreddit: str,
        title: str,
        url: str,
        author: str | None,
        created_utc: int,
        reddit_score: int,
        num_comments: int,
        keyword_score: float,
        final_score: float,
        matched_topics: tuple[str, ...],
        observed_at: int,
    ) -> None:
        """Insert or refresh a candidate. Replay-safe two ways: on conflict
        the volatile fields update while ``first_seen``/``status`` are
        preserved, and a stale (out-of-order) observation -- one whose
        ``observed_at`` is older than the stored ``last_seen`` -- updates
        nothing, so replayed old polling windows can never regress fresher
        state."""
        _require_id(post_id, field="post_id")
        _require_int(created_utc, field="created_utc")
        _require_int(reddit_score, field="reddit_score")
        _require_int(num_comments, field="num_comments")
        _require_int(observed_at, field="observed_at")
        _require_finite_number(keyword_score, field="keyword_score")
        _require_finite_number(final_score, field="final_score")
        if self.is_purged(post_id):
            # Tombstone: purged content must never resurrect, even when
            # Reddit keeps returning the removed item in listings.
            return
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO candidates (
                    post_id, subreddit, title, url, author, created_utc,
                    reddit_score, num_comments, keyword_score, final_score,
                    matched_topics, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    subreddit = excluded.subreddit,
                    title = excluded.title,
                    url = excluded.url,
                    author = excluded.author,
                    reddit_score = excluded.reddit_score,
                    num_comments = excluded.num_comments,
                    keyword_score = excluded.keyword_score,
                    final_score = excluded.final_score,
                    matched_topics = excluded.matched_topics,
                    last_seen = excluded.last_seen
                WHERE excluded.last_seen >= candidates.last_seen
                """,
                (
                    post_id,
                    subreddit,
                    title,
                    url,
                    author,
                    created_utc,
                    reddit_score,
                    num_comments,
                    keyword_score,
                    final_score,
                    json.dumps(list(matched_topics)),
                    observed_at,
                    observed_at,
                ),
            )

    def get_candidate(self, post_id: str) -> Candidate | None:
        row = self._conn.execute(
            "SELECT * FROM candidates WHERE post_id = ?", (post_id,)
        ).fetchone()
        return _candidate_from_row(row) if row else None

    def set_candidate_status(self, post_id: str, status: str) -> None:
        if status not in CANDIDATE_STATUSES:
            raise StoreError(
                f"invalid candidate status {status!r}; allowed: {CANDIDATE_STATUSES}"
            )
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE candidates SET status = ? WHERE post_id = ?",
                (status, post_id),
            )
        if cursor.rowcount == 0:
            raise StoreError(f"unknown candidate: {post_id!r}")

    def list_candidates(
        self,
        *,
        status: str | None = None,
        min_final_score: float | None = None,
        limit: int | None = None,
    ) -> list[Candidate]:
        if status is not None and status not in CANDIDATE_STATUSES:
            raise StoreError(
                f"invalid candidate status {status!r}; allowed: {CANDIDATE_STATUSES}"
            )
        # sqlite3 binds NaN as NULL, which silently empties the filter;
        # reject non-finite floors here so every caller fails loudly.
        if min_final_score is not None and not math.isfinite(min_final_score):
            raise StoreError(
                f"min_final_score must be finite, got {min_final_score!r}"
            )
        query = "SELECT * FROM candidates"
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if min_final_score is not None:
            clauses.append("final_score >= ?")
            params.append(min_final_score)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY final_score DESC, created_utc DESC, post_id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [_candidate_from_row(row) for row in rows]

    # -- tracked threads --------------------------------------------------

    def upsert_tracked_thread(
        self,
        *,
        thread_id: str,
        my_comment_ids: tuple[str, ...],
        checked_at: int,
        is_own_submission: bool = False,
    ) -> None:
        """Insert or refresh a tracked thread. Comment ids are merged as a
        set union (replays and re-discoveries never drop known ids);
        ``dormant`` is preserved on conflict; ``is_own_submission`` is
        sticky once true (a thread that is my post stays my post)."""
        _require_id(thread_id, field="thread_id")
        _require_int(checked_at, field="checked_at")
        _require_bool(is_own_submission, field="is_own_submission")
        incoming = _require_comment_ids(my_comment_ids)
        existing = self._conn.execute(
            "SELECT my_comment_ids FROM tracked_threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if existing is None:
            merged = list(dict.fromkeys(incoming))
        else:
            known: list[str] = json.loads(existing["my_comment_ids"])
            merged = list(dict.fromkeys([*known, *incoming]))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO tracked_threads (
                    thread_id, my_comment_ids, last_checked, is_own_submission
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    my_comment_ids = excluded.my_comment_ids,
                    last_checked = excluded.last_checked,
                    is_own_submission = MAX(
                        tracked_threads.is_own_submission,
                        excluded.is_own_submission
                    )
                """,
                (thread_id, json.dumps(merged), checked_at,
                 1 if is_own_submission else 0),
            )

    def record_thread_activity(self, thread_id: str, activity_at: int) -> None:
        """Advance a thread's last-known-activity timestamp (monotonic:
        stale values cannot move it backwards). Unknown ids fail closed."""
        _require_id(thread_id, field="thread_id")
        _require_int(activity_at, field="activity_at")
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE tracked_threads
                SET last_activity = MAX(COALESCE(last_activity, 0), ?)
                WHERE thread_id = ?
                """,
                (activity_at, thread_id),
            )
        if cursor.rowcount == 0:
            raise StoreError(f"unknown tracked thread: {thread_id!r}")

    def set_thread_dormant(self, thread_id: str, dormant: bool) -> None:
        _require_bool(dormant, field="dormant")
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE tracked_threads SET dormant = ? WHERE thread_id = ?",
                (1 if dormant else 0, thread_id),
            )
        if cursor.rowcount == 0:
            raise StoreError(f"unknown tracked thread: {thread_id!r}")

    def list_tracked_threads(self, *, include_dormant: bool = False) -> list[TrackedThread]:
        query = "SELECT * FROM tracked_threads"
        if not include_dormant:
            query += " WHERE dormant = 0"
        query += " ORDER BY thread_id ASC"
        rows = self._conn.execute(query).fetchall()
        return [_thread_from_row(row) for row in rows]

    # -- replies -----------------------------------------------------------

    def insert_reply(
        self,
        *,
        reply_id: str,
        thread_id: str,
        parent_id: str | None,
        author: str | None,
        body: str,
        created_utc: int,
        is_reply_to_me: bool,
    ) -> bool:
        """Insert a reply. Returns True when the row was inserted, False
        on a replayed duplicate ``reply_id``. The conflict target is
        deliberately narrow: NOT NULL/CHECK/foreign-key violations raise
        (sqlite3.IntegrityError) instead of being masked as replays, and a
        reply must reference a registered tracked thread."""
        _require_id(reply_id, field="reply_id")
        _require_id(thread_id, field="thread_id")
        _require_int(created_utc, field="created_utc")
        _require_bool(is_reply_to_me, field="is_reply_to_me")
        if self.is_purged(reply_id):
            return False  # tombstoned: purged content must never resurrect
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO replies (
                    reply_id, thread_id, parent_id, author, body,
                    created_utc, is_reply_to_me
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reply_id) DO NOTHING
                """,
                (
                    reply_id,
                    thread_id,
                    parent_id,
                    author,
                    body,
                    created_utc,
                    1 if is_reply_to_me else 0,
                ),
            )
        return cursor.rowcount == 1

    def mark_reply_seen(self, reply_id: str) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE replies SET seen = 1 WHERE reply_id = ?", (reply_id,)
            )
        if cursor.rowcount == 0:
            raise StoreError(f"unknown reply: {reply_id!r}")

    def list_replies(
        self,
        *,
        thread_id: str | None = None,
        only_unseen: bool = False,
        only_to_me: bool = False,
    ) -> list[Reply]:
        query = "SELECT * FROM replies"
        clauses: list[str] = []
        params: list[object] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if only_unseen:
            clauses.append("seen = 0")
        if only_to_me:
            clauses.append("is_reply_to_me = 1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_utc ASC, reply_id ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [_reply_from_row(row) for row in rows]

    # -- purge (deletion compliance, S6) -------------------------------------

    def purge_item(
        self,
        item_id: str,
        item_type: str,
        *,
        deleted_detected_at: int,
        purged_at: int,
        reason: str,
    ) -> bool:
        """Atomically delete one stored content row AND record the audit
        entry in a single transaction: the compliance contract must never
        end up with content deleted but no record (or vice versa), even
        under an I/O error or interrupt between the two writes. Returns
        True when a row was actually removed; no row means no log entry."""
        _require_id(item_id, field="item_id")
        _require_int(deleted_detected_at, field="deleted_detected_at")
        _require_int(purged_at, field="purged_at")
        if item_type == "candidate":
            table, key = "candidates", "post_id"
        elif item_type == "reply":
            table, key = "replies", "reply_id"
        else:
            raise StoreError(
                f"invalid purge item_type {item_type!r}; allowed: ('candidate', 'reply')"
            )
        with self._conn:
            cursor = self._conn.execute(
                f"DELETE FROM {table} WHERE {key} = ?", (item_id,)
            )
            if cursor.rowcount != 1:
                return False
            self._conn.execute(
                """
                INSERT INTO purge_log (
                    item_id, item_type, deleted_detected_at, purged_at, reason
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (item_id, item_type, deleted_detected_at, purged_at, reason),
            )
        return True

    def is_purged(self, item_id: str) -> bool:
        """Tombstone check: has this id ever been purged? Ingestion paths
        consult this so re-listed removed content cannot resurrect."""
        _require_id(item_id, field="item_id")
        row = self._conn.execute(
            "SELECT 1 FROM purge_log WHERE item_id = ? LIMIT 1", (item_id,)
        ).fetchone()
        return row is not None

    # -- purge log ----------------------------------------------------------

    def record_purge(
        self,
        *,
        item_id: str,
        item_type: str,
        deleted_detected_at: int,
        purged_at: int,
        reason: str,
    ) -> None:
        _require_id(item_id, field="item_id")
        _require_int(deleted_detected_at, field="deleted_detected_at")
        _require_int(purged_at, field="purged_at")
        if item_type not in PURGE_ITEM_TYPES:
            raise StoreError(
                f"invalid purge item_type {item_type!r}; allowed: {PURGE_ITEM_TYPES}"
            )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO purge_log (
                    item_id, item_type, deleted_detected_at, purged_at, reason
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (item_id, item_type, deleted_detected_at, purged_at, reason),
            )

    def list_purge_log(self) -> list[PurgeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM purge_log ORDER BY id ASC"
        ).fetchall()
        return [
            PurgeRecord(
                item_id=row["item_id"],
                item_type=row["item_type"],
                deleted_detected_at=row["deleted_detected_at"],
                purged_at=row["purged_at"],
                reason=row["reason"],
            )
            for row in rows
        ]


def _candidate_from_row(row: sqlite3.Row) -> Candidate:
    return Candidate(
        post_id=row["post_id"],
        subreddit=row["subreddit"],
        title=row["title"],
        url=row["url"],
        author=row["author"],
        created_utc=row["created_utc"],
        reddit_score=row["reddit_score"],
        num_comments=row["num_comments"],
        keyword_score=row["keyword_score"],
        final_score=row["final_score"],
        matched_topics=tuple(json.loads(row["matched_topics"])),
        status=row["status"],
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
    )


def _thread_from_row(row: sqlite3.Row) -> TrackedThread:
    return TrackedThread(
        thread_id=row["thread_id"],
        my_comment_ids=tuple(json.loads(row["my_comment_ids"])),
        last_checked=row["last_checked"],
        dormant=bool(row["dormant"]),
        is_own_submission=bool(row["is_own_submission"]),
        last_activity=row["last_activity"],
    )


def _reply_from_row(row: sqlite3.Row) -> Reply:
    return Reply(
        reply_id=row["reply_id"],
        thread_id=row["thread_id"],
        parent_id=row["parent_id"],
        author=row["author"],
        body=row["body"],
        created_utc=row["created_utc"],
        is_reply_to_me=bool(row["is_reply_to_me"]),
        seen=bool(row["seen"]),
    )
