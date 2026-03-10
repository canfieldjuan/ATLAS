"""
Canonical vendor name registry.

Resolves free-text vendor names to their canonical form using a DB-backed
registry (b2b_vendors table) with an in-process cache.  Every B2B ingest
point should call ``resolve_vendor_name()`` (async) or
``resolve_vendor_name_cached()`` (sync, cache-only) before storing
vendor_name values.

Cache is a flat dict[str, str] mapping lowered keys (canonical names +
aliases) to canonical form, rebuilt every 5 minutes from the DB.
"""

import asyncio
import logging
import time
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger("atlas.services.vendor_registry")

# ---------------------------------------------------------------------------
# Bootstrap aliases -- identical to the former _B2B_COMPETITOR_ALIASES dict
# in b2b_churn_intelligence.py.  Used as a cold-start fallback when the DB
# cache has not been populated yet.
# ---------------------------------------------------------------------------

_BOOTSTRAP_ALIASES: dict[str, str] = {
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "ms teams": "Microsoft Teams",
    "ms 365": "Microsoft 365",
    "office 365": "Microsoft 365",
    "o365": "Microsoft 365",
    "sf": "Salesforce",
    "sfdc": "Salesforce",
    "hubspot crm": "HubSpot",
    "g suite": "Google Workspace",
    "google workspace": "Google Workspace",
    "gsuite": "Google Workspace",
}

# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_cache: dict[str, str] = {}  # lowered key -> canonical_name
_cache_ts: float = 0.0
_cache_lock: asyncio.Lock | None = None
_CACHE_TTL_SECONDS = 300  # 5 minutes
_CACHE_FALLBACK_RETRY_SECONDS = 30  # retry sooner when using bootstrap fallback
_FUZZY_THRESHOLD = 0.85  # minimum similarity ratio for fuzzy match


def _get_lock() -> asyncio.Lock:
    """Lazy-init the asyncio lock (must be created inside a running loop)."""
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


async def _ensure_cache() -> None:
    """Rebuild cache from b2b_vendors if stale (older than TTL)."""
    global _cache, _cache_ts

    if time.monotonic() - _cache_ts < _CACHE_TTL_SECONDS and _cache:
        return

    lock = _get_lock()
    async with lock:
        # Double-check after acquiring lock
        if time.monotonic() - _cache_ts < _CACHE_TTL_SECONDS and _cache:
            return

        try:
            from ..storage.database import get_db_pool

            pool = get_db_pool()
            if not pool.is_initialized:
                logger.debug("DB not ready, using bootstrap aliases")
                if not _cache:
                    _cache = dict(_BOOTSTRAP_ALIASES)
                # Retry soon rather than waiting full TTL
                _cache_ts = time.monotonic() - _CACHE_TTL_SECONDS + _CACHE_FALLBACK_RETRY_SECONDS
                return

            rows = await pool.fetch(
                "SELECT canonical_name, aliases FROM b2b_vendors"
            )
            new_cache: dict[str, str] = {}
            for row in rows:
                canonical = row["canonical_name"]
                new_cache[canonical.lower()] = canonical
                aliases = row["aliases"]
                if isinstance(aliases, list):
                    for alias in aliases:
                        if isinstance(alias, str) and alias:
                            new_cache[alias.lower()] = canonical
                elif isinstance(aliases, str):
                    # Defensive: asyncpg may return JSONB as string in edge cases
                    import json as _json
                    try:
                        alias_list = _json.loads(aliases)
                        for alias in alias_list:
                            if isinstance(alias, str) and alias:
                                new_cache[alias.lower()] = canonical
                    except (ValueError, TypeError):
                        pass

            # Merge bootstrap entries that aren't already covered by DB
            for key, canonical in _BOOTSTRAP_ALIASES.items():
                if key not in new_cache:
                    new_cache[key] = canonical

            _cache = new_cache
            _cache_ts = time.monotonic()
            logger.debug("Vendor cache rebuilt: %d entries", len(_cache))

        except Exception:
            logger.exception("Failed to rebuild vendor cache")
            # Keep stale cache or fall back to bootstrap; retry soon
            if not _cache:
                _cache = dict(_BOOTSTRAP_ALIASES)
            _cache_ts = time.monotonic() - _CACHE_TTL_SECONDS + _CACHE_FALLBACK_RETRY_SECONDS


def invalidate_cache() -> None:
    """Force a refresh on the next call to _ensure_cache()."""
    global _cache_ts, _cache_lock
    _cache_ts = 0.0
    # Reset the lock so it works correctly if the event loop changed (e.g. tests)
    _cache_lock = None


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------

def _resolve_from_cache(raw: str) -> str:
    """Resolve a vendor name using the current cache contents.

    1. Exact match against cache keys (canonical names + aliases).
    2. Fuzzy fallback: find best match above ``_FUZZY_THRESHOLD`` (0.85).
    3. Return title-cased original if no match found.
    """
    stripped = raw.strip()
    if not stripped:
        return stripped
    lowered = stripped.lower()

    # 1. Direct cache hit (alias or canonical)
    if lowered in _cache:
        return _cache[lowered]

    # 2. Fuzzy fallback -- only for inputs >= 4 chars (short strings match too broadly)
    if len(lowered) >= 4 and _cache:
        best_score = 0.0
        best_canonical = ""
        for key, canonical in _cache.items():
            ratio = SequenceMatcher(None, lowered, key).ratio()
            if ratio > best_score:
                best_score = ratio
                best_canonical = canonical
        if best_score >= _FUZZY_THRESHOLD:
            return best_canonical

    # 3. Not found -- title-case if all-lowercase, else preserve original casing
    return stripped.title() if stripped.islower() else stripped


async def resolve_vendor_name(raw: str) -> str:
    """Resolve a raw vendor name to its canonical form (async, DB-backed).

    Suitable for ingest points (scrape intake, API, MCP).
    """
    if not raw:
        return raw or ""
    await _ensure_cache()
    return _resolve_from_cache(raw)


def resolve_vendor_name_cached(raw: str) -> str:
    """Resolve a raw vendor name using cache only (sync).

    Falls back to bootstrap aliases if the cache has not been populated.
    Suitable for hot-path sync code like intelligence synthesis.
    """
    if not raw:
        return raw or ""
    if not _cache:
        # Cold start: use bootstrap aliases directly
        stripped = raw.strip()
        if not stripped:
            return stripped
        lowered = stripped.lower()
        if lowered in _BOOTSTRAP_ALIASES:
            return _BOOTSTRAP_ALIASES[lowered]
        return stripped.title() if stripped.islower() else stripped
    return _resolve_from_cache(raw)


# ---------------------------------------------------------------------------
# Registry management (async)
# ---------------------------------------------------------------------------

async def add_vendor(canonical_name: str, aliases: list[str] | None = None) -> dict[str, Any]:
    """Insert or update a vendor in the registry.

    On conflict, merges the new aliases into the existing set (does not replace).
    Returns the row.
    """
    from ..storage.database import get_db_pool
    import json as _json

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    alias_list = [a.lower().strip() for a in (aliases or []) if a.strip()]

    row = await pool.fetchrow(
        """
        INSERT INTO b2b_vendors (canonical_name, aliases)
        VALUES ($1, $2::jsonb)
        ON CONFLICT (canonical_name) DO UPDATE SET
            aliases = (
                SELECT jsonb_agg(DISTINCT val)
                FROM jsonb_array_elements_text(
                    b2b_vendors.aliases || EXCLUDED.aliases
                ) AS t(val)
            ),
            updated_at = NOW()
        RETURNING id, canonical_name, aliases, created_at, updated_at
        """,
        canonical_name.strip(),
        _json.dumps(alias_list),
    )
    invalidate_cache()
    return dict(row)


async def add_alias(canonical_name: str, alias: str) -> dict[str, Any] | None:
    """Append an alias to an existing vendor. Returns updated row or None."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    alias_lower = alias.strip().lower()
    if not alias_lower:
        return None

    row = await pool.fetchrow(
        """
        UPDATE b2b_vendors
        SET aliases = CASE
                WHEN NOT aliases @> to_jsonb($2::text)
                THEN aliases || to_jsonb($2::text)
                ELSE aliases
            END,
            updated_at = NOW()
        WHERE canonical_name = $1
        RETURNING id, canonical_name, aliases, created_at, updated_at
        """,
        canonical_name.strip(),
        alias_lower,
    )
    if row:
        invalidate_cache()
        return dict(row)
    return None


async def list_vendors() -> list[dict[str, Any]]:
    """Return all vendors from the registry."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    rows = await pool.fetch(
        "SELECT id, canonical_name, aliases, metadata, created_at, updated_at "
        "FROM b2b_vendors ORDER BY canonical_name"
    )
    return [dict(r) for r in rows]


async def fuzzy_search_vendors(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> list[dict[str, Any]]:
    """Search vendors using pg_trgm trigram similarity.

    Returns vendors sorted by similarity score (descending).
    Requires the pg_trgm extension (migration 114).
    """
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    query = query.strip()
    if not query:
        return []

    limit = max(1, min(limit, 100))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    rows = await pool.fetch(
        """
        SELECT id, canonical_name, aliases,
               similarity(canonical_name, $1) AS sim_score
        FROM b2b_vendors
        WHERE similarity(canonical_name, $1) >= $2
        ORDER BY sim_score DESC
        LIMIT $3
        """,
        query,
        min_similarity,
        limit,
    )
    return [
        {
            "id": str(r["id"]),
            "canonical_name": r["canonical_name"],
            "aliases": list(r["aliases"]) if isinstance(r["aliases"], list) else [],
            "similarity": round(float(r["sim_score"]), 4),
        }
        for r in rows
    ]


async def fuzzy_search_companies(
    query: str,
    vendor_name: str | None = None,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> list[dict[str, Any]]:
    """Search company names using pg_trgm trigram similarity.

    Optionally scoped to a specific vendor.  Returns companies sorted by
    similarity score (descending).
    """
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    query = query.strip()
    if not query:
        return []

    limit = max(1, min(limit, 100))
    min_similarity = max(0.0, min(min_similarity, 1.0))

    if vendor_name:
        rows = await pool.fetch(
            """
            SELECT DISTINCT ON (company_name)
                   company_name, vendor_name, urgency_score, buying_stage,
                   similarity(company_name, $1) AS sim_score
            FROM b2b_company_signals
            WHERE similarity(company_name, $1) >= $2
              AND vendor_name ILIKE $4
            ORDER BY company_name, sim_score DESC
            LIMIT $3
            """,
            query, min_similarity, limit, vendor_name,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT DISTINCT ON (company_name)
                   company_name, vendor_name, urgency_score, buying_stage,
                   similarity(company_name, $1) AS sim_score
            FROM b2b_company_signals
            WHERE similarity(company_name, $1) >= $2
            ORDER BY company_name, sim_score DESC
            LIMIT $3
            """,
            query, min_similarity, limit,
        )
    return [
        {
            "company_name": r["company_name"],
            "vendor_name": r["vendor_name"],
            "urgency_score": float(r["urgency_score"]) if r["urgency_score"] else None,
            "buying_stage": r["buying_stage"],
            "similarity": round(float(r["sim_score"]), 4),
        }
        for r in rows
    ]
