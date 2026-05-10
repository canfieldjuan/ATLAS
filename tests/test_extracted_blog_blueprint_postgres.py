"""Pin the Postgres adapter for blog blueprints.

`extracted_content_pipeline/blog_blueprint_postgres.py` adds
the missing concrete implementation of the package's
`BlogBlueprintRepository` Protocol (PR #456 left this as the
last unwired Content Ops slot). The bundle factory's blog_post
wiring slice depends on this storage layer landing first.

Test inventory (8 tests):

1. `save_blueprints` round-trips payload through JSONB.
2. `save_blueprints` upsert resets `consumed_at` so the row
   is eligible for re-generation.
3. `read_blog_blueprints` filters by account_id + target_mode
   + `consumed_at IS NULL` and applies the LIMIT.
4. `read_blog_blueprints` honors the `topic_type` filter.
5. `read_blog_blueprints` returns merged payload + row metadata
   so the generator sees a self-contained dict.
6. `read_blog_blueprints` with no rows returns `()`
   (empty-table canary).
7. `mark_consumed` issues the UPDATE with the expected
   tenant-scoped predicate AND returns the integer count
   parsed from the "UPDATE N" command tag.
8. `mark_consumed` with empty ids short-circuits (no DB
   roundtrip, returns 0).

Test harness uses an asyncpg-shaped fake pool (matches
`tests/test_extracted_blog_post_postgres.py`).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from extracted_content_pipeline.blog_blueprint_postgres import (
    BlogBlueprint,
    PostgresBlogBlueprintRepository,
)
from extracted_content_pipeline.campaign_ports import TenantScope


class _Pool:
    def __init__(self) -> None:
        self.fetchval_results: list[Any] = []
        self.fetch_rows: list[dict[str, Any]] = []
        self.fetchval_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []
        self.execute_calls: list[dict[str, Any]] = []
        self.execute_result: Any = "UPDATE 1"

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def execute(self, query: str, *args: Any) -> Any:
        self.execute_calls.append({"query": query, "args": args})
        return self.execute_result


def _blueprint() -> BlogBlueprint:
    return BlogBlueprint(
        target_mode="vendor_alternative",
        topic_type="pricing_pressure",
        slug="acme-pricing-pressure",
        suggested_title="Acme Pricing Pressure",
        payload={
            "sections": [
                {"id": "intro", "heading": "Why Acme is losing price",
                 "goal": "Frame the pressure"},
            ],
            "charts": [{"id": "pricing_mentions", "type": "bar"}],
            "tags": ["pricing", "retention"],
        },
    )


@pytest.mark.asyncio
async def test_save_blueprints_round_trips_payload_jsonb() -> None:
    pool = _Pool()
    pool.fetchval_results = ["bp-uuid-1"]
    repo = PostgresBlogBlueprintRepository(pool=pool)

    saved = await repo.save_blueprints(
        [_blueprint()],
        scope=TenantScope(account_id="acct-1"),
    )

    assert saved == ("bp-uuid-1",)
    args = pool.fetchval_calls[0]["args"]
    assert args[0] == "acct-1"
    assert args[1] == "vendor_alternative"
    assert args[2] == "pricing_pressure"
    assert args[3] == "acme-pricing-pressure"
    assert args[4] == "Acme Pricing Pressure"
    payload = json.loads(args[5])
    assert payload["tags"] == ["pricing", "retention"]
    assert payload["sections"][0]["id"] == "intro"


@pytest.mark.asyncio
async def test_save_blueprints_upsert_resets_consumed_at() -> None:
    """Pinning the ON CONFLICT branch -- without
    `consumed_at = NULL`, a re-saved blueprint would stay
    invisible to `read_blog_blueprints` and the generator
    would never re-pick it up after the first run."""

    pool = _Pool()
    pool.fetchval_results = ["bp-uuid-1"]
    repo = PostgresBlogBlueprintRepository(pool=pool)

    await repo.save_blueprints([_blueprint()], scope=TenantScope(account_id="acct-1"))

    query = pool.fetchval_calls[0]["query"]
    assert "ON CONFLICT (account_id, target_mode, slug) DO UPDATE" in query
    assert "consumed_at = NULL" in query


@pytest.mark.asyncio
async def test_read_blog_blueprints_filters_by_scope_target_mode_and_unconsumed() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "bp-uuid-1",
            "target_mode": "vendor_alternative",
            "topic_type": "pricing_pressure",
            "slug": "acme-pricing-pressure",
            "suggested_title": "Acme Pricing Pressure",
            "payload": {"sections": [], "tags": ["pricing"]},
        },
    ]
    repo = PostgresBlogBlueprintRepository(pool=pool)

    rows = await repo.read_blog_blueprints(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_alternative",
        limit=5,
    )

    assert len(rows) == 1
    query = pool.fetch_calls[0]["query"]
    assert "account_id = $1" in query
    assert "target_mode = $2" in query
    assert "consumed_at IS NULL" in query
    args = pool.fetch_calls[0]["args"]
    assert args == ("acct-1", "vendor_alternative", 5)


@pytest.mark.asyncio
async def test_read_blog_blueprints_honors_topic_type_filter() -> None:
    pool = _Pool()
    pool.fetch_rows = []
    repo = PostgresBlogBlueprintRepository(pool=pool)

    await repo.read_blog_blueprints(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_alternative",
        limit=10,
        filters={"topic_type": "pricing_pressure"},
    )

    args = pool.fetch_calls[0]["args"]
    # account_id, target_mode, topic_type, limit
    assert args == ("acct-1", "vendor_alternative", "pricing_pressure", 10)
    assert "topic_type = $3" in pool.fetch_calls[0]["query"]


@pytest.mark.asyncio
async def test_read_blog_blueprints_merges_payload_with_row_metadata() -> None:
    """The generator consumes a single dict; row-level metadata
    (id / topic_type / slug / suggested_title / target_mode)
    must land alongside the payload's sections / charts / tags."""

    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "bp-uuid-1",
            "target_mode": "vendor_alternative",
            "topic_type": "pricing_pressure",
            "slug": "acme-pricing-pressure",
            "suggested_title": "Acme Pricing Pressure",
            "payload": json.dumps(
                {"sections": [{"id": "intro"}], "tags": ["pricing"]}
            ),
        },
    ]
    repo = PostgresBlogBlueprintRepository(pool=pool)

    rows = await repo.read_blog_blueprints(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_alternative",
        limit=5,
    )

    blueprint = rows[0]
    assert blueprint["id"] == "bp-uuid-1"
    assert blueprint["target_mode"] == "vendor_alternative"
    assert blueprint["topic_type"] == "pricing_pressure"
    assert blueprint["slug"] == "acme-pricing-pressure"
    assert blueprint["suggested_title"] == "Acme Pricing Pressure"
    assert blueprint["sections"] == [{"id": "intro"}]
    assert blueprint["tags"] == ["pricing"]


@pytest.mark.asyncio
async def test_read_blog_blueprints_empty_table_returns_empty_tuple() -> None:
    """Defensive canary -- the generator's loop must see an
    empty Sequence (not a one-element tuple of None / partial
    row) when the tenant has nothing to consume."""

    pool = _Pool()
    pool.fetch_rows = []
    repo = PostgresBlogBlueprintRepository(pool=pool)

    rows = await repo.read_blog_blueprints(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_alternative",
        limit=5,
    )
    assert rows == ()


@pytest.mark.asyncio
async def test_mark_consumed_issues_tenant_scoped_update() -> None:
    """Pin the actual integer count parse from asyncpg's
    "UPDATE N" command tag -- callers compare the return to
    `len(blueprint_ids)` to detect partial-batch failures, so
    coercing to a 0/1 bool would silently mis-report."""

    pool = _Pool()
    pool.execute_result = "UPDATE 2"
    repo = PostgresBlogBlueprintRepository(pool=pool)

    hits = await repo.mark_consumed(
        ["bp-uuid-1", "bp-uuid-2"],
        scope=TenantScope(account_id="acct-1"),
    )

    assert hits == 2
    query = pool.execute_calls[0]["query"]
    assert "account_id = $1" in query
    assert "id = ANY($2::uuid[])" in query
    assert "consumed_at IS NULL" in query
    args = pool.execute_calls[0]["args"]
    assert args[0] == "acct-1"
    assert args[1] == ["bp-uuid-1", "bp-uuid-2"]


@pytest.mark.asyncio
async def test_mark_consumed_with_empty_ids_short_circuits() -> None:
    """Empty input must skip the DB roundtrip -- otherwise an
    `id = ANY('{}'::uuid[])` predicate would still execute
    once per call. Returns 0."""

    pool = _Pool()
    repo = PostgresBlogBlueprintRepository(pool=pool)

    hits = await repo.mark_consumed([], scope=TenantScope(account_id="acct-1"))

    assert hits == 0
    assert pool.execute_calls == []
