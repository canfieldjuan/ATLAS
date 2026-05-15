"""Pin the Postgres adapter for Content Ops campaign reasoning contexts.

`extracted_content_pipeline/campaign_reasoning_postgres.py` is
the DB-backed counterpart to PR #462's file-backed
`FileCampaignReasoningContextProvider`. Both implement
`CampaignReasoningProviderPort.read_campaign_reasoning_context`
so the host route mount (PR #402 / PR #462) can swap providers
without touching the bundle's `with_reasoning_context()`
derivation.

Test inventory (19 tests):

1. `read_campaign_reasoning_context` builds the candidate
   selector array from `target_id` + opportunity keys
   (mirrors the file-backed provider's `_candidate_keys`)
   and issues the `account_id = $1 AND selectors && $2::text[]`
   read with the expected args.
2. `read_campaign_reasoning_context` returns the normalized
   `CampaignReasoningContext` decoded from the JSONB payload.
3. `read_campaign_reasoning_context` returns `None` when the
   row is absent (no match).
4. `read_campaign_reasoning_context` returns `None` when the
   stored payload is empty / not a Mapping (defensive).
5. `read_campaign_reasoning_context` short-circuits without a
   DB roundtrip when no selectors can be built (empty target_id
   + empty opportunity).
6. `save_context` round-trips a normalized context through
   JSONB and persists the deduped selectors.
7. `save_context` raises `ValueError` when given an
   all-empty selectors list (a row with no selectors is
   unreachable -- almost certainly an upstream ETL bug).
8. `save_context` accepts a raw mapping (not just a
   `CampaignReasoningContext`) and round-trips through
   `normalize_campaign_reasoning_context` so the persisted
   payload matches the file-backed loader's expected layout.
9. `read_campaign_reasoning_context` filters by target_mode,
   with blank target-mode rows kept as fallback matches.
10. Exact target-mode rows rank before blank fallback rows at
    the same selector priority.
11. Target-mode caller input is normalized before filtering.
12. `save_context` normalizes target-mode values before writing.
13. `save_context` upserts by account, target_mode, and selector_key.
14. Selector keys are order-independent for the same logical selector set.
15. `save_context` accepts raw mappings after the selector-key SQL shape change.
16. Migration 278 defines the selector_key column and unique replay index.
17. `delete_stale_contexts` counts stale rows in dry-run mode.
18. `delete_stale_contexts` deletes stale rows only when requested.
19. `delete_stale_contexts` rejects non-positive age thresholds.

Test harness uses an asyncpg-shaped fake pool (matches
`tests/test_extracted_blog_blueprint_postgres.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    TenantScope,
)
from extracted_content_pipeline.campaign_reasoning_postgres import (
    PostgresCampaignReasoningContextRepository,
)


class _Pool:
    def __init__(self) -> None:
        self.fetchval_results: list[Any] = []
        self.fetchrow_result: Any = None
        self.fetchval_calls: list[dict[str, Any]] = []
        self.fetchrow_calls: list[dict[str, Any]] = []

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append({"query": query, "args": args})
        return self.fetchrow_result


@pytest.mark.asyncio
async def test_read_uses_candidate_selectors_array() -> None:
    """The TEXT[] arg must include `target_id` plus the
    opportunity selectors, in both case-as-given and lowercase
    forms -- the file-backed provider's `_candidate_keys`
    contract is what existing tenants' file payloads index
    against."""

    pool = _Pool()
    pool.fetchrow_result = None
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="ACME-123",
        target_mode="vendor",
        opportunity={
            "company_name": "Acme Corp",
            "contact_email": "buyer@acme.com",
        },
    )

    args = pool.fetchrow_calls[0]["args"]
    assert args[0] == "acct-1"
    selectors = args[1]
    assert args[2] == "vendor"
    # case-as-given + lowercase variants for each non-empty key
    assert "ACME-123" in selectors
    assert "acme-123" in selectors
    assert "Acme Corp" in selectors
    assert "acme corp" in selectors
    assert "buyer@acme.com" in selectors
    query = pool.fetchrow_calls[0]["query"]
    assert "account_id = $1" in query
    assert "selectors && $2::text[]" in query
    assert "target_mode = $3 OR target_mode = ''" in query
    # Priority ordering (Codex P2): exact target_id should beat a
    # broader newer company-name match. Selector-position MIN
    # subquery ranks rows by which of the candidates they matched.
    assert "unnest($2::text[]) WITH ORDINALITY" in query
    assert "ORDER BY" in query and "updated_at DESC" in query
    assert "CASE WHEN $3 <> '' AND target_mode = $3" in query
    assert "LIMIT 1" in query


@pytest.mark.asyncio
async def test_read_priority_subquery_ranks_before_updated_at() -> None:
    """Pin the ORDER-BY shape: candidate-selector-position MIN
    must rank before ``updated_at DESC``. Without this the
    file-backed provider's first-key-wins semantics diverge
    from the DB-backed read -- a broader newer company-name
    row would beat an exact ``target_id`` row whose selectors
    also overlap. (Codex P2 follow-up)"""

    pool = _Pool()
    pool.fetchrow_result = None
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="ACME-123",
        target_mode="vendor",
        opportunity={"company_name": "Acme Corp"},
    )

    query = pool.fetchrow_calls[0]["query"]
    # Priority subquery must appear in the ORDER BY clause,
    # before the target-mode and updated_at tie-breakers.
    priority_idx = query.find("MIN(c.idx)")
    target_mode_idx = query.find("CASE WHEN $3 <> '' AND target_mode = $3")
    updated_idx = query.find("updated_at DESC")
    assert priority_idx >= 0
    assert target_mode_idx >= 0
    assert updated_idx >= 0
    assert priority_idx < target_mode_idx < updated_idx


@pytest.mark.asyncio
async def test_read_filters_by_target_mode_with_blank_fallback() -> None:
    """A nonblank request mode must not read a row saved for a
    different nonblank mode. Blank target-mode rows remain shared
    fallback contexts for legacy/global seed data."""

    pool = _Pool()
    pool.fetchrow_result = None
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="ACME-123",
        target_mode="vendor_retention",
        opportunity={},
    )

    query = pool.fetchrow_calls[0]["query"]
    args = pool.fetchrow_calls[0]["args"]
    assert args[2] == "vendor_retention"
    assert "AND ($3 = '' OR target_mode = $3 OR target_mode = '')" in query


@pytest.mark.asyncio
async def test_read_normalizes_target_mode_before_filtering() -> None:
    """Target-mode values are persisted lowercase by convention;
    normalize caller input so mixed-case request values still match."""

    pool = _Pool()
    pool.fetchrow_result = None
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="ACME-123",
        target_mode="Vendor_Retention",
        opportunity={},
    )

    args = pool.fetchrow_calls[0]["args"]
    assert args[2] == "vendor_retention"


@pytest.mark.asyncio
async def test_read_empty_target_mode_preserves_legacy_unfiltered_lookup() -> None:
    """Empty target_mode keeps the old broad lookup behavior for
    callers that do not know their mode yet."""

    pool = _Pool()
    pool.fetchrow_result = None
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="ACME-123",
        target_mode="",
        opportunity={},
    )

    args = pool.fetchrow_calls[0]["args"]
    assert args[2] == ""


@pytest.mark.asyncio
async def test_read_returns_normalized_context() -> None:
    """Decoded JSONB payload flows through
    `normalize_campaign_reasoning_context` so callers get the
    same shape regardless of whether the row was written as
    Atlas-style `reasoning_*` keys or already-normalized
    fields."""

    pool = _Pool()
    pool.fetchrow_result = {
        "payload": {
            "reasoning_top_theses": [
                {"wedge": "price", "summary": "Acme is losing on price"},
            ],
            "reasoning_account_signals": [
                {"company": "Acme", "primary_pain": "renewal pressure"},
            ],
        }
    }
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    context = await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="acme",
        target_mode="vendor",
        opportunity={},
    )

    assert isinstance(context, CampaignReasoningContext)
    assert context.has_content()
    assert context.top_theses[0]["summary"] == "Acme is losing on price"


@pytest.mark.asyncio
async def test_read_no_match_returns_none() -> None:
    """A miss must resolve to `None`, not an empty context --
    the bundle's `with_reasoning_context()` short-circuits on
    `None` and falls back to zero-context defaults."""

    pool = _Pool()
    pool.fetchrow_result = None
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    result = await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="acme",
        target_mode="vendor",
        opportunity={},
    )
    assert result is None


@pytest.mark.asyncio
async def test_read_empty_payload_returns_none() -> None:
    """Defensive: a row whose JSONB payload decodes to
    empty / non-Mapping must resolve to `None` so callers
    don't pass an empty bundle through to the prompt."""

    pool = _Pool()
    pool.fetchrow_result = {"payload": "not-a-mapping"}
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    result = await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="acme",
        target_mode="vendor",
        opportunity={},
    )
    assert result is None


@pytest.mark.asyncio
async def test_read_no_selectors_short_circuits_db() -> None:
    """An empty `target_id` + empty opportunity yields no
    candidate selectors -- the read must skip the DB roundtrip
    rather than issue `selectors && '{}'::text[]` (which
    would scan with no useful predicate)."""

    pool = _Pool()
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    result = await repo.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="",
        target_mode="vendor",
        opportunity={},
    )

    assert result is None
    assert pool.fetchrow_calls == []


@pytest.mark.asyncio
async def test_save_context_round_trips_payload_jsonb() -> None:
    """Persist a normalized context and verify the deduped
    selectors land in the TEXT[] column with both case-as-given
    and lowercase variants."""

    pool = _Pool()
    pool.fetchval_results = ["ctx-uuid-1"]
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    context = CampaignReasoningContext(
        top_theses=({"wedge": "price", "summary": "losing on price"},),
    )

    saved = await repo.save_context(
        scope=TenantScope(account_id="acct-1"),
        selectors=("Acme Corp", "acme corp", "ACME-123"),
        context=context,
        target_mode="vendor",
    )

    assert saved == "ctx-uuid-1"
    args = pool.fetchval_calls[0]["args"]
    assert args[0] == "acct-1"
    assert args[1] == "vendor"
    selectors = args[2]
    # case-as-given + lowercase, deduped
    assert "Acme Corp" in selectors
    assert "acme corp" in selectors
    assert "ACME-123" in selectors
    assert "acme-123" in selectors
    payload = json.loads(args[4])
    assert "reasoning_context" in payload
    assert payload["reasoning_context"]["top_theses"][0]["summary"] == "losing on price"


@pytest.mark.asyncio
async def test_save_context_upserts_by_selector_key() -> None:
    """Replay writes update the existing row for the same account,
    target_mode, and logical selector set."""

    pool = _Pool()
    pool.fetchval_results = ["ctx-uuid-upsert"]
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    saved = await repo.save_context(
        scope=TenantScope(account_id="acct-1"),
        selectors=("Acme Corp", "ACME-123"),
        context=CampaignReasoningContext(
            top_theses=({"summary": "Acme is losing on price"},),
        ),
        target_mode="vendor_retention",
    )

    assert saved == "ctx-uuid-upsert"
    call = pool.fetchval_calls[0]
    query = call["query"]
    args = call["args"]
    assert "selector_key" in query
    assert "ON CONFLICT (account_id, target_mode, selector_key)" in query
    assert "DO UPDATE SET" in query
    assert "payload = EXCLUDED.payload" in query
    assert args[0] == "acct-1"
    assert args[1] == "vendor_retention"
    assert "Acme Corp" in args[2]
    assert len(args[3]) == 32
    assert json.loads(args[4])["reasoning_context"]["top_theses"]


@pytest.mark.asyncio
async def test_save_context_selector_key_is_order_independent() -> None:
    """The replay key depends on the set of cleaned selectors, not
    caller ordering."""

    pool = _Pool()
    pool.fetchval_results = ["ctx-1", "ctx-1"]
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    context = CampaignReasoningContext(top_theses=({"summary": "x"},))
    await repo.save_context(
        scope=TenantScope(account_id="acct-1"),
        selectors=("Acme Corp", "ACME-123"),
        context=context,
        target_mode="vendor",
    )
    await repo.save_context(
        scope=TenantScope(account_id="acct-1"),
        selectors=("ACME-123", "Acme Corp"),
        context=context,
        target_mode="vendor",
    )

    first_args = pool.fetchval_calls[0]["args"]
    second_args = pool.fetchval_calls[1]["args"]
    assert first_args[3] == second_args[3]
    assert first_args[2] != []
    assert second_args[2] != []


@pytest.mark.asyncio
async def test_save_context_normalizes_target_mode_before_writing() -> None:
    """Persisted target_mode must match the read-path lowercase
    convention so mixed-case writer input stays readable."""

    pool = _Pool()
    pool.fetchval_results = ["ctx-uuid-mode"]
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    await repo.save_context(
        scope=TenantScope(account_id="acct-1"),
        selectors=("Acme Corp",),
        context=CampaignReasoningContext(
            top_theses=({"summary": "Acme losing on price"},),
        ),
        target_mode=" Vendor_Retention ",
    )

    args = pool.fetchval_calls[0]["args"]
    assert args[1] == "vendor_retention"


@pytest.mark.asyncio
async def test_save_context_rejects_empty_selectors() -> None:
    """A row with no selectors is unreachable -- almost
    certainly an upstream ETL bug. Surface it loudly rather
    than persisting a row that no read will ever match."""

    pool = _Pool()
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    with pytest.raises(ValueError):
        await repo.save_context(
            scope=TenantScope(account_id="acct-1"),
            selectors=("", "  ", None),  # type: ignore[arg-type]
            context=CampaignReasoningContext(
                top_theses=({"summary": "x"},),
            ),
        )
    assert pool.fetchval_calls == []


@pytest.mark.asyncio
async def test_save_context_accepts_raw_mapping() -> None:
    """Hosts that already produce reasoning context as a raw
    mapping (the same shape `FileCampaignReasoningContextProvider`
    accepts) should be able to save it directly without
    first constructing a `CampaignReasoningContext` --
    `normalize_campaign_reasoning_context` round-trip
    canonicalizes the persisted JSONB."""

    pool = _Pool()
    pool.fetchval_results = ["ctx-uuid-2"]
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    raw = {
        "reasoning_top_theses": [
            {"wedge": "price", "summary": "Acme losing on price"},
        ],
    }

    saved = await repo.save_context(
        scope=TenantScope(account_id="acct-1"),
        selectors=("acme",),
        context=raw,
    )

    assert saved == "ctx-uuid-2"
    payload = json.loads(pool.fetchval_calls[0]["args"][4])
    assert payload["reasoning_context"]["top_theses"][0]["summary"] == (
        "Acme losing on price"
    )


def test_campaign_reasoning_context_upsert_migration_shape() -> None:
    """Migration 278 owns the selector_key column and replay index."""

    migration = (
        Path(__file__).resolve().parents[1]
        / "extracted_content_pipeline/storage/migrations"
        / "278_campaign_reasoning_context_upsert.sql"
    ).read_text()

    assert "ADD COLUMN IF NOT EXISTS selector_key TEXT" in migration
    assert "md5(" in migration
    assert "DELETE FROM campaign_reasoning_contexts AS stale" in migration
    assert "ALTER COLUMN selector_key SET NOT NULL" in migration
    assert "idx_campaign_reasoning_contexts_scope_mode_selector_key" in migration
    assert "(account_id, target_mode, selector_key)" in migration


@pytest.mark.asyncio
async def test_delete_stale_contexts_dry_run_counts_matching_rows() -> None:
    """Dry-run mode must count stale rows without deleting them."""

    pool = _Pool()
    pool.fetchval_results = [3]
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    affected = await repo.delete_stale_contexts(
        older_than_days=30,
        scope=TenantScope(account_id="acct-1"),
        target_mode="Vendor_Retention",
        dry_run=True,
    )

    assert affected == 3
    call = pool.fetchval_calls[0]
    query = call["query"]
    args = call["args"]
    assert "SELECT COUNT(*) FROM stale" in query
    assert "DELETE FROM campaign_reasoning_contexts" not in query
    assert "updated_at < NOW() - ($1::int * INTERVAL '1 day')" in query
    assert args == (30, "acct-1", "vendor_retention")


@pytest.mark.asyncio
async def test_delete_stale_contexts_deletes_matching_rows_when_not_dry_run() -> None:
    """Apply mode deletes through the same stale-row predicate."""

    pool = _Pool()
    pool.fetchval_results = [2]
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    affected = await repo.delete_stale_contexts(
        older_than_days=45,
        scope=None,
        target_mode=None,
        dry_run=False,
    )

    assert affected == 2
    call = pool.fetchval_calls[0]
    query = call["query"]
    args = call["args"]
    assert "DELETE FROM campaign_reasoning_contexts" in query
    assert "SELECT COUNT(*) FROM deleted" in query
    assert args == (45, None, None)


@pytest.mark.asyncio
async def test_delete_stale_contexts_rejects_non_positive_days() -> None:
    """A zero-day cleanup would be too broad for an operator action."""

    pool = _Pool()
    repo = PostgresCampaignReasoningContextRepository(pool=pool)

    with pytest.raises(ValueError):
        await repo.delete_stale_contexts(older_than_days=0)
    assert pool.fetchval_calls == []
