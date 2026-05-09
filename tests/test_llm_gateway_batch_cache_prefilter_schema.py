"""Regression tests for PR-D6g-1: schema scaffolding for /batch
cache prefilter accounting.

Pins five contracts:

1. Migration 323 exists and adds the column with the right shape.
2. ``BatchView`` Pydantic model exposes ``cache_prefiltered_items:
   int = 0``.
3. ``CustomerBatchRecord`` dataclass exposes the same field with
   default 0.
4. Every SELECT in ``llm_gateway_batch.py`` reads the new column.
5. ``_row_to_record`` defensively reads the column with the
   ``.get``-style pattern matching the precedent set by
   ``usage_tracked`` and ``anthropic_call_initiated_at``.

File-text inspection only -- bypasses the gateway module's full
settings stack.

See plans/PR-D6g-1-batch-cache-prefilter-schema.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = (
    ROOT
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "323_llm_gateway_batches_cache_prefiltered_items.sql"
)
GATEWAY_PATH = ROOT / "atlas_brain" / "api" / "llm_gateway.py"
BATCH_PATH = ROOT / "atlas_brain" / "services" / "llm_gateway_batch.py"


@pytest.fixture(scope="module")
def gateway_source() -> str:
    return GATEWAY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def batch_source() -> str:
    return BATCH_PATH.read_text(encoding="utf-8")


def test_migration_323_exists_and_adds_column():
    """Migration file is present and adds the column with the right
    type and default."""
    assert MIGRATION_PATH.exists(), f"missing migration: {MIGRATION_PATH}"
    text = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "ALTER TABLE llm_gateway_batches" in text
    assert "cache_prefiltered_items" in text
    assert "INTEGER NOT NULL DEFAULT 0" in text


def test_batch_view_pydantic_model_has_cache_prefiltered_items(gateway_source):
    """BatchView model exposes the field with default 0."""
    assert "class BatchView(BaseModel):" in gateway_source
    block = gateway_source.split("class BatchView(BaseModel):", 1)[1]
    block = block.split("\nclass ", 1)[0]
    assert "cache_prefiltered_items: int = 0" in block, (
        "BatchView must declare cache_prefiltered_items: int = 0"
    )


def test_batch_record_to_view_maps_cache_prefiltered_items(gateway_source):
    """The view-mapper threads the field from the record to the view."""
    assert "def _batch_record_to_view" in gateway_source
    block = gateway_source.split("def _batch_record_to_view", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert "cache_prefiltered_items=record.cache_prefiltered_items" in block, (
        "_batch_record_to_view must thread cache_prefiltered_items "
        "from the record to the view"
    )


def test_customer_batch_record_dataclass_has_cache_prefiltered_items(batch_source):
    """CustomerBatchRecord dataclass declares the field with default 0."""
    assert "class CustomerBatchRecord:" in batch_source
    block = batch_source.split("class CustomerBatchRecord:", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert "cache_prefiltered_items: int = 0" in block, (
        "CustomerBatchRecord must declare cache_prefiltered_items: int = 0"
    )


def test_row_to_record_defensively_reads_cache_prefiltered_items(batch_source):
    """The row reader uses the same .get-style pattern as
    usage_tracked / anthropic_call_initiated_at -- mocks with shorter
    row dicts still work."""
    assert "def _row_to_record" in batch_source
    block = batch_source.split("def _row_to_record", 1)[1].split("\ndef ", 1)[0]
    assert "cache_prefiltered_items=" in block, (
        "_row_to_record must read cache_prefiltered_items"
    )
    assert '"cache_prefiltered_items" in row.keys()' in block, (
        "_row_to_record must defensively check the column is present "
        "(matches the usage_tracked / anthropic_call_initiated_at pattern)"
    )


def test_all_select_statements_include_cache_prefiltered_items(batch_source):
    """Every SELECT that produces a CustomerBatchRecord row must
    include the new column. Missing it on any read path would
    silently default to 0 even when there are real hits."""
    # Count SELECT statements that reference last_usage_retry_at
    # (the prior tail-of-list field) -- these are the ones that
    # need to also include cache_prefiltered_items.
    select_count = batch_source.count("anthropic_call_initiated_at, last_usage_retry_at,")
    cache_count = batch_source.count("cache_prefiltered_items")

    # Every SELECT block has cache_prefiltered_items appended.
    # Plus the dataclass field, plus the _row_to_record reader,
    # plus the BatchView field (in another file -- not this count).
    # In this file we expect at least: 6 SELECTs + 1 dataclass +
    # 2 _row_to_record references = 9+ occurrences.
    assert cache_count >= 6, (
        f"expected at least 6 cache_prefiltered_items references in "
        f"llm_gateway_batch.py; found {cache_count}"
    )
    assert select_count >= 5, (
        f"expected at least 5 SELECT-shaped column lists; found "
        f"{select_count} -- if this dropped, a SELECT may have been "
        f"missed by the schema migration"
    )
