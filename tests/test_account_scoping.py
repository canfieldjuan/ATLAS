"""Tests for per-account scoping on llm_usage + caches (PR-D3).

Pins the contract that:

  - Atlas's existing pipeline (which doesn't know about accounts)
    keeps working unmodified -- writes get the sentinel account_id
    via the column DEFAULT and reads via the constructor default.
  - Customer-facing callers (PR-D4 LLM Gateway router) pass an
    explicit account_id and get an isolated cache namespace.
  - The migration drops the cache_key/pattern_sig single-column
    PK/UNIQUE and replaces with composite (..., account_id).
  - The ON CONFLICT clauses target the new composite constraint.
  - llm_usage INSERT carries account_id so per-tenant cost queries
    work.

DB-backed integration tests live alongside other auth integration
tests and are gated on a running Postgres; this file does pure unit
+ source-text inspection so the suite stays runnable without DB.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from pathlib import Path

import pytest


# ---- Migration files exist + carry the sentinel + composite keys --------

_MIG_DIR = Path(__file__).resolve().parent.parent / "atlas_brain" / "storage" / "migrations"


def _read_migration(filename: str) -> str:
    return (_MIG_DIR / filename).read_text(encoding="utf-8")


def test_migration_313_adds_account_id_with_sentinel():
    sql = _read_migration("313_llm_usage_account_scoping.sql")
    assert "ALTER TABLE llm_usage" in sql
    assert "ADD COLUMN IF NOT EXISTS account_id UUID NOT NULL" in sql
    assert "'00000000-0000-0000-0000-000000000000'" in sql


def test_migration_313_indexes_per_account_analytics():
    sql = _read_migration("313_llm_usage_account_scoping.sql")
    assert "idx_llm_usage_account_created" in sql
    assert "(account_id, created_at DESC)" in sql


def test_migration_314_replaces_cache_key_pk_with_composite():
    sql = _read_migration("314_b2b_llm_exact_cache_account_scoping.sql")
    assert "DROP CONSTRAINT IF EXISTS b2b_llm_exact_cache_pkey" in sql
    assert "PRIMARY KEY (cache_key, account_id)" in sql
    # Migration must still install the sentinel default so atlas's
    # existing rows survive the constraint swap.
    assert "'00000000-0000-0000-0000-000000000000'" in sql


def test_migration_315_replaces_pattern_sig_unique_with_composite():
    sql = _read_migration("315_reasoning_semantic_cache_account_scoping.sql")
    assert "DROP CONSTRAINT IF EXISTS reasoning_semantic_cache_pattern_sig_key" in sql
    assert "UNIQUE (pattern_sig, account_id)" in sql
    assert "'00000000-0000-0000-0000-000000000000'" in sql


# ---- SemanticCache: constructor + query SQL -------------------------------


def test_semantic_cache_constructor_accepts_account_id_with_default():
    """The atlas pipeline instantiates ``SemanticCache(pool)`` without
    knowing about accounts. The default sentinel must keep that
    working."""
    from atlas_brain.reasoning.semantic_cache import SemanticCache

    sig = inspect.signature(SemanticCache.__init__)
    assert "account_id" in sig.parameters
    assert (
        sig.parameters["account_id"].default
        == "00000000-0000-0000-0000-000000000000"
    )
    assert sig.parameters["account_id"].kind == inspect.Parameter.KEYWORD_ONLY


def test_semantic_cache_lookup_filters_by_account_id():
    """The lookup query must scope on account_id so cross-tenant hits
    are impossible at the storage layer."""
    from atlas_brain.reasoning import semantic_cache as sc_mod

    src = inspect.getsource(sc_mod.SemanticCache.lookup)
    assert "WHERE pattern_sig = $1 AND account_id = $2" in src
    assert "self._account_id" in src


def test_semantic_cache_store_uses_composite_on_conflict():
    """The store path must target ``(pattern_sig, account_id)``. If
    it still says ``ON CONFLICT (pattern_sig)``, the migration's
    constraint replacement breaks it at runtime."""
    from atlas_brain.reasoning import semantic_cache as sc_mod

    src = inspect.getsource(sc_mod.SemanticCache.store)
    assert "ON CONFLICT (pattern_sig, account_id) DO UPDATE" in src
    assert "self._account_id" in src


def test_semantic_cache_validate_filters_by_account():
    from atlas_brain.reasoning import semantic_cache as sc_mod

    src = inspect.getsource(sc_mod.SemanticCache.validate)
    assert "AND account_id = $2" in src


def test_semantic_cache_invalidate_filters_by_account():
    from atlas_brain.reasoning import semantic_cache as sc_mod

    src = inspect.getsource(sc_mod.SemanticCache.invalidate)
    assert "AND account_id = $2" in src


def test_semantic_cache_lookup_by_class_filters_by_account():
    from atlas_brain.reasoning import semantic_cache as sc_mod

    src = inspect.getsource(sc_mod.SemanticCache.lookup_by_class)
    assert "self._account_id" in src
    # All three branches (vendor only / class+vendor / class only)
    # must scope on account_id; verify each appears at least once.
    assert src.count("AND account_id =") >= 3


# ---- llm_exact_cache: account_id kwarg + composite ON CONFLICT ----------


def test_lookup_cached_text_accepts_account_id_default_sentinel():
    from atlas_brain.services.b2b.llm_exact_cache import lookup_cached_text

    sig = inspect.signature(lookup_cached_text)
    assert "account_id" in sig.parameters
    assert (
        sig.parameters["account_id"].default
        == "00000000-0000-0000-0000-000000000000"
    )


def test_store_cached_text_accepts_account_id_default_sentinel():
    from atlas_brain.services.b2b.llm_exact_cache import store_cached_text

    sig = inspect.signature(store_cached_text)
    assert "account_id" in sig.parameters
    assert (
        sig.parameters["account_id"].default
        == "00000000-0000-0000-0000-000000000000"
    )


def test_lookup_cached_text_query_scopes_by_account_id():
    from atlas_brain.services.b2b import llm_exact_cache as ec_mod

    src = inspect.getsource(ec_mod.lookup_cached_text)
    assert "WHERE cache_key = $1 AND account_id = $2" in src


def test_store_cached_text_uses_composite_on_conflict():
    from atlas_brain.services.b2b import llm_exact_cache as ec_mod

    src = inspect.getsource(ec_mod.store_cached_text)
    assert "ON CONFLICT (cache_key, account_id) DO UPDATE" in src
    assert "INSERT INTO b2b_llm_exact_cache" in src
    assert "account_id" in src


# ---- llm_usage tracing INSERT -------------------------------------------


def test_tracing_store_local_insert_includes_account_id_column():
    """The INSERT INTO llm_usage statement must include the
    account_id column so per-tenant cost queries (PR-D4 ``GET
    /api/v1/llm/usage``) return the right numbers."""
    from atlas_brain.services import tracing

    src = inspect.getsource(tracing.FTLTracingClient._store_local)
    assert "INSERT INTO llm_usage" in src
    # Column list must include account_id
    assert "account_id)" in src
    # 27 placeholders (was 26 + new account_id)
    assert "$27" in src


def test_tracing_store_local_reads_account_id_from_metadata():
    """Callers thread account_id via ``metadata={"account_id": ...}``;
    the existing pattern for vendor_name / run_id / etc."""
    from atlas_brain.services import tracing

    src = inspect.getsource(tracing.FTLTracingClient._store_local)
    assert '_metadata_text_value(meta, "account_id")' in src


def test_tracing_store_local_falls_back_to_sentinel():
    """When metadata lacks account_id (atlas's existing pipeline),
    the INSERT writes the sentinel UUID."""
    from atlas_brain.services import tracing

    src = inspect.getsource(tracing.FTLTracingClient._store_local)
    assert "00000000-0000-0000-0000-000000000000" in src


# ---- Atlas-pipeline regression: ecosystem_analysis ON CONFLICT ----------


def test_ecosystem_analysis_uses_composite_on_conflict():
    """``ecosystem_analysis.py`` writes directly to
    reasoning_semantic_cache via raw SQL (bypasses SemanticCache).
    Migration 315 drops the pattern_sig-only UNIQUE so its ON
    CONFLICT target must update too."""
    from atlas_brain.autonomous.tasks import ecosystem_analysis

    src = inspect.getsource(ecosystem_analysis)
    # Must NOT contain the old single-column target...
    assert "ON CONFLICT (pattern_sig) DO UPDATE" not in src
    # ...but must contain the composite target (T3 + T4 INSERTs).
    assert src.count("ON CONFLICT (pattern_sig, account_id) DO UPDATE") >= 2


# ---- Sentinel UUID is the documented constant ---------------------------


def test_sentinel_account_id_is_zero_uuid():
    """The sentinel value is documented in two places (SemanticCache
    + llm_exact_cache); both must agree on the all-zero UUID."""
    from atlas_brain.reasoning.semantic_cache import SemanticCache
    from atlas_brain.services.b2b.llm_exact_cache import SENTINEL_ACCOUNT_ID

    assert SemanticCache.SENTINEL_ACCOUNT_ID == "00000000-0000-0000-0000-000000000000"
    assert SENTINEL_ACCOUNT_ID == "00000000-0000-0000-0000-000000000000"


# ---- Sync invariant: extracted copies updated ---------------------------


def test_extracted_llm_exact_cache_synced_with_account_id():
    """``services/b2b/llm_exact_cache.py`` is synced from atlas to
    extracted. The extracted copy must carry the same account_id
    threading."""
    extracted_path = (
        Path(__file__).resolve().parent.parent
        / "extracted_llm_infrastructure"
        / "services"
        / "b2b"
        / "llm_exact_cache.py"
    )
    src = extracted_path.read_text(encoding="utf-8")
    assert "SENTINEL_ACCOUNT_ID = " in src
    assert "WHERE cache_key = $1 AND account_id = $2" in src
    assert "ON CONFLICT (cache_key, account_id) DO UPDATE" in src


def test_extracted_semantic_cache_synced_with_account_id():
    extracted_path = (
        Path(__file__).resolve().parent.parent
        / "extracted_llm_infrastructure"
        / "reasoning"
        / "semantic_cache.py"
    )
    src = extracted_path.read_text(encoding="utf-8")
    assert "SENTINEL_ACCOUNT_ID = " in src
    assert "ON CONFLICT (pattern_sig, account_id) DO UPDATE" in src


def test_extracted_tracing_synced_with_account_id():
    extracted_path = (
        Path(__file__).resolve().parent.parent
        / "extracted_llm_infrastructure"
        / "services"
        / "tracing.py"
    )
    src = extracted_path.read_text(encoding="utf-8")
    assert "$27" in src  # the new account_id placeholder
    assert '_metadata_text_value(meta, "account_id")' in src
