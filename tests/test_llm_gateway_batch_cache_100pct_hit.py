"""Regression tests for PR-D6g-2: 100%-hit short-circuit for /batch
cache prefilter.

Pins six contracts:

1. The cache-hit usage INSERT SQL exists with the right shape (zero
   tokens, ``llm_gateway.batch`` span/endpoint).
2. The prefilter helper imports cache primitives lazily and uses
   the same namespace as /chat (cross-endpoint sharing).
3. The prefilter returns ``None`` on any miss and on lookup error
   (fail-open contract).
4. The short-circuit hook fires AFTER the idempotency replay /
   resume checks (only when ``row is None``).
5. The 100%-hit insert sets ``status='ended'``, all three count
   fields equal to ``len(items)``, and the row has no
   ``provider_batch_id``.
6. Cache savings metadata key is shared between write and read
   sites (no drift).

File-text inspection only -- bypasses the gateway module's full
settings stack.

See plans/PR-D6g-2-batch-cache-100pct-hit-shortcircuit.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = ROOT / "atlas_brain" / "services" / "llm_gateway_batch.py"


@pytest.fixture(scope="module")
def batch_source() -> str:
    return BATCH_PATH.read_text(encoding="utf-8")


def test_cache_hit_usage_insert_sql_exists_with_zero_token_shape(batch_source):
    """The INSERT writes all-zero token counts and tags the row with
    'llm_gateway.batch' so /usage rollups can attribute savings."""
    assert "_CACHE_HIT_BATCH_USAGE_INSERT_SQL" in batch_source
    sql_block = batch_source.split("_CACHE_HIT_BATCH_USAGE_INSERT_SQL = ", 1)[1].split(
        '"""', 2
    )[1]
    assert "INSERT INTO llm_usage" in sql_block
    assert "'llm_gateway.batch'" in sql_block
    assert "0, 0, 0" in sql_block, "must write zero token counts"
    assert "'completed'" in sql_block
    assert "$3::jsonb" in sql_block, "metadata must be jsonb"


def test_cache_savings_metadata_key_constant_exists(batch_source):
    """A module-level constant pins the savings metadata key so write
    and read sites can't drift (matches PR-D6c on /chat)."""
    assert '_CACHE_SAVINGS_METADATA_KEY = "cache_savings_usd"' in batch_source


def test_prefilter_helper_lazy_imports_cache_primitives(batch_source):
    """The cache helpers are lazy-imported inside the prefilter
    function -- avoids widening this module's import surface."""
    assert "async def _try_prefilter_batch_through_cache" in batch_source
    helper_block = batch_source.split(
        "async def _try_prefilter_batch_through_cache", 1
    )[1].split("\nasync def ", 1)[0]
    # Lazy import is INSIDE the function body, not at module top.
    assert "from .b2b.llm_exact_cache import" in helper_block
    assert "build_request_envelope" in helper_block
    assert "is_llm_gateway_exact_cache_enabled" in helper_block
    assert "lookup_cached_text" in helper_block


def test_prefilter_returns_none_on_any_miss(batch_source):
    """Any miss -> caller falls through to normal submission (no
    short-circuit). Pin the contract via source assertion."""
    helper_block = batch_source.split(
        "async def _try_prefilter_batch_through_cache", 1
    )[1].split("\nasync def ", 1)[0]
    assert "if cache_hit is None:" in helper_block
    assert "return None  # Any miss" in helper_block


def test_prefilter_fails_open_on_lookup_error(batch_source):
    """Cache lookup exception -> return None (fall through to
    normal submission). Customer pays Anthropic instead of seeing a
    500 from a cache implementation detail."""
    helper_block = batch_source.split(
        "async def _try_prefilter_batch_through_cache", 1
    )[1].split("\nasync def ", 1)[0]
    assert "except Exception:" in helper_block
    assert "return None  # Fail-open" in helper_block


def test_prefilter_uses_chat_cache_namespace(batch_source):
    """Cross-endpoint cache sharing: /batch and /chat share the
    'llm_gateway.chat' namespace so they hit each other's stored
    responses for identical prompts."""
    helper_block = batch_source.split(
        "async def _try_prefilter_batch_through_cache", 1
    )[1].split("\nasync def ", 1)[0]
    assert 'namespace = "llm_gateway.chat"' in helper_block


def test_short_circuit_only_fires_when_row_is_none(batch_source):
    """The prefilter hook lives inside ``if row is None:`` so a
    replay or resume always wins over the cache short-circuit."""
    submit_block = batch_source.split("async def submit_customer_batch", 1)[1].split(
        "\nasync def ", 1
    )[0]
    # The hook is a NEW ``if row is None`` block (the second one;
    # the first is the existing INSERT path that follows it).
    assert "if row is None:" in submit_block
    assert "_try_prefilter_batch_through_cache" in submit_block
    # Hook calls _insert_all_cache_hit_batch on 100% hits.
    assert "_insert_all_cache_hit_batch" in submit_block


def test_insert_all_cache_hit_batch_marks_status_ended(batch_source):
    """The 100%-hit INSERT writes status='ended', all three counts
    equal to total_items, no provider_batch_id, both timestamps NOW,
    usage_tracked=TRUE."""
    insert_block = batch_source.split("async def _insert_all_cache_hit_batch", 1)[
        1
    ].split("\nasync def ", 1)[0]
    assert "'ended'" in insert_block
    assert "cache_prefiltered_items" in insert_block
    assert "completed_items" in insert_block
    # All three counts (total_items, completed_items, cache_prefiltered_items)
    # use the same $3 placeholder (item_count). The SQL spans multiple
    # lines so we count occurrences in the VALUES clause.
    insert_sql = insert_block.split("INSERT INTO", 1)[1].split("RETURNING", 1)[0]
    values_clause = insert_sql.split("VALUES", 1)[1]
    assert values_clause.count("$3") == 3, (
        "total_items, completed_items, cache_prefiltered_items must all "
        f"use $3 placeholder; found {values_clause.count('$3')}"
    )
    # Timestamps marked NOW; usage_tracked TRUE; no provider_batch_id
    # column in the INSERT (defaults to NULL).
    assert "NOW(), NOW(), TRUE" in insert_block
    assert "provider_batch_id" not in insert_block.split("INSERT", 1)[1].split(
        "RETURNING", 1
    )[0], "provider_batch_id should not be in the INSERT column list (defaults to NULL)"


def test_insert_writes_per_item_zero_token_usage_row(batch_source):
    """Each cache hit gets a zero-token llm_usage row in the same
    transaction so /usage rollups attribute the savings."""
    insert_block = batch_source.split("async def _insert_all_cache_hit_batch", 1)[
        1
    ].split("\nasync def ", 1)[0]
    assert "for hit in all_hits:" in insert_block
    assert "_CACHE_HIT_BATCH_USAGE_INSERT_SQL" in insert_block
    # Metadata includes cache_hit, savings, batch_id, custom_id.
    assert '"cache_hit": True' in insert_block
    assert "_CACHE_SAVINGS_METADATA_KEY" in insert_block
    assert '"batch_id"' in insert_block
    assert '"custom_id"' in insert_block
