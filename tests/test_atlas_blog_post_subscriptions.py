"""Pin the B2B blog-post subscription reader.

`atlas_brain/_blog_post_subscriptions.py` is the typed
contract PR-Subscriptions-2's autonomous task fanout will
import. This file pins:
- `list_active_blog_post_subscriptions(pool)` filters
  enabled-only and returns typed dataclasses.
- `BlogPostSubscription.matches(topic_type, target_mode)`
  semantics: empty filters accept-all, populated filters do
  subset matching.

Test inventory (8 tests):

1. Empty table returns an empty list.
2. Active rows are converted to `BlogPostSubscription`s.
3. The reader's WHERE clause keeps disabled rows out (the
   fake pool delivers whatever rows the test sets, so the
   contract here is the WHERE clause we ship in the SQL --
   pin it via the recorded query string).
4. Empty array columns deserialize to empty tuples.
5. `matches()` with empty filters accepts everything.
6. `matches()` with `topic_types` filter rejects mismatches.
7. `matches()` with `target_modes` filter rejects mismatches.
8. `matches()` with both filters requires both to pass.
"""

from __future__ import annotations

from typing import Any

import pytest

from atlas_brain._blog_post_subscriptions import (
    BlogPostSubscription,
    list_active_blog_post_subscriptions,
)


class _Pool:
    def __init__(self) -> None:
        self.fetch_rows: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows


@pytest.mark.asyncio
async def test_empty_table_returns_empty_list() -> None:
    pool = _Pool()
    pool.fetch_rows = []

    subs = await list_active_blog_post_subscriptions(pool)

    assert subs == []


@pytest.mark.asyncio
async def test_active_rows_become_typed_subscriptions() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "account_id": "acct-1",
            "topic_types": ["vendor_alternative", "churn_report"],
            "target_modes": ["vendor_retention"],
        },
        {
            "account_id": "acct-2",
            "topic_types": [],
            "target_modes": [],
        },
    ]

    subs = await list_active_blog_post_subscriptions(pool)

    assert len(subs) == 2
    assert subs[0].account_id == "acct-1"
    assert subs[0].topic_types == ("vendor_alternative", "churn_report")
    assert subs[0].target_modes == ("vendor_retention",)
    assert subs[1].account_id == "acct-2"
    assert subs[1].topic_types == ()
    assert subs[1].target_modes == ()


@pytest.mark.asyncio
async def test_reader_query_filters_enabled_rows() -> None:
    """The reader's contract for skipping disabled rows is
    the SQL WHERE clause itself. Pin the query shape so a
    refactor that drops the predicate is caught."""

    pool = _Pool()
    pool.fetch_rows = []

    await list_active_blog_post_subscriptions(pool)

    query = pool.fetch_calls[0]["query"]
    assert "WHERE enabled = TRUE" in query
    assert "FROM b2b_blog_post_subscriptions" in query
    assert "ORDER BY account_id" in query


@pytest.mark.asyncio
async def test_null_array_columns_deserialize_to_empty_tuples() -> None:
    """Defensive canary: asyncpg returns NULL columns as
    `None`, not as empty arrays. The `or ()` coercion in the
    reader guards against that. Without it, the dataclass
    would be constructed with `topic_types=None`, breaking
    the `matches()` predicate."""

    pool = _Pool()
    pool.fetch_rows = [
        {"account_id": "acct-1", "topic_types": None, "target_modes": None},
    ]

    subs = await list_active_blog_post_subscriptions(pool)

    assert subs[0].topic_types == ()
    assert subs[0].target_modes == ()


def test_matches_empty_filters_accepts_everything() -> None:
    sub = BlogPostSubscription(account_id="acct-1")

    assert sub.matches(topic_type="vendor_alternative", target_mode="vendor_retention")
    assert sub.matches(topic_type="churn_report", target_mode="account")
    assert sub.matches(topic_type="anything", target_mode="anything")


def test_matches_topic_types_filter_rejects_mismatches() -> None:
    sub = BlogPostSubscription(
        account_id="acct-1",
        topic_types=("vendor_alternative",),
    )

    assert sub.matches(topic_type="vendor_alternative", target_mode="anything")
    assert not sub.matches(topic_type="churn_report", target_mode="anything")


def test_matches_target_modes_filter_rejects_mismatches() -> None:
    sub = BlogPostSubscription(
        account_id="acct-1",
        target_modes=("vendor_retention",),
    )

    assert sub.matches(topic_type="anything", target_mode="vendor_retention")
    assert not sub.matches(topic_type="anything", target_mode="account")


def test_matches_both_filters_requires_both_to_pass() -> None:
    sub = BlogPostSubscription(
        account_id="acct-1",
        topic_types=("vendor_alternative",),
        target_modes=("vendor_retention",),
    )

    assert sub.matches(
        topic_type="vendor_alternative",
        target_mode="vendor_retention",
    )
    # Topic matches, target doesn't.
    assert not sub.matches(
        topic_type="vendor_alternative",
        target_mode="account",
    )
    # Target matches, topic doesn't.
    assert not sub.matches(
        topic_type="churn_report",
        target_mode="vendor_retention",
    )
    # Neither matches.
    assert not sub.matches(
        topic_type="churn_report",
        target_mode="account",
    )
