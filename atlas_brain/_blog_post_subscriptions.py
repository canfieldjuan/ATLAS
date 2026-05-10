"""B2B blog post subscription reader.

Per-tenant opt-in list for the autonomous blog-post task's
blueprint fanout. PR-Subscriptions-2 wires the autonomous task
to read this list once per run and fan blueprints out per
matching subscription via
``PostgresBlogBlueprintRepository.save_blueprints``.

Subscriptions land in the host's ``b2b_blog_post_subscriptions``
table (migration 324). This module is the typed reader; the
autonomous task imports `list_active_blog_post_subscriptions`
and `BlogPostSubscription.matches`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BlogPostSubscription:
    """One tenant's opt-in row from ``b2b_blog_post_subscriptions``.

    Empty ``topic_types`` / ``target_modes`` mean "accept all
    values" -- the natural representation of "no restriction".
    The ``matches`` predicate enforces the semantics so callers
    don't conditionalize.
    """

    account_id: str
    topic_types: tuple[str, ...] = ()
    target_modes: tuple[str, ...] = ()

    def matches(self, *, topic_type: str, target_mode: str) -> bool:
        """Return whether a generated blueprint should fan out
        to this subscription."""

        if self.topic_types and topic_type not in self.topic_types:
            return False
        if self.target_modes and target_mode not in self.target_modes:
            return False
        return True


async def list_active_blog_post_subscriptions(
    pool: Any,
) -> list[BlogPostSubscription]:
    """Return enabled subscription rows ordered by account_id.

    The expected subscription count is small (per-tenant
    business-tier opt-in), so a list rather than an async
    iterator keeps the call site simple. If counts grow,
    switch to async iteration without a contract change.
    """

    rows = await pool.fetch(
        """
        SELECT account_id, topic_types, target_modes
        FROM b2b_blog_post_subscriptions
        WHERE enabled = TRUE
        ORDER BY account_id
        """
    )
    return [
        BlogPostSubscription(
            account_id=str(row["account_id"]),
            topic_types=tuple(row["topic_types"] or ()),
            target_modes=tuple(row["target_modes"] or ()),
        )
        for row in rows
    ]


__all__ = [
    "BlogPostSubscription",
    "list_active_blog_post_subscriptions",
]
