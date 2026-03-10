"""
Canonical consumer brand name registry.

Resolves free-text brand names to their canonical form using a DB-backed
registry (consumer_brand_registry table) with an in-process cache.  The
consumer intelligence pipeline should call ``resolve_brand_name()`` (async)
or ``resolve_brand_name_cached()`` (sync, cache-only) before persisting
brand name values into brand_intelligence, snapshots, or change events.

Mirrors the B2B vendor registry pattern (vendor_registry.py) adapted for
consumer product brands.
"""

import asyncio
import logging
import time
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger("atlas.services.brand_registry")

# ---------------------------------------------------------------------------
# Bootstrap aliases -- common consumer brand name variations.
# Used as a cold-start fallback before the DB cache is populated.
# ---------------------------------------------------------------------------

_BOOTSTRAP_ALIASES: dict[str, str] = {
    "kitchen aid": "KitchenAid",
    "kitchenaid": "KitchenAid",
    "de'longhi": "De'Longhi",
    "delonghi": "De'Longhi",
    "black+decker": "BLACK+DECKER",
    "black and decker": "BLACK+DECKER",
    "black & decker": "BLACK+DECKER",
    "hamilton beach": "Hamilton Beach",
    "mr. coffee": "Mr. Coffee",
    "mr coffee": "Mr. Coffee",
    "t-fal": "T-fal",
    "tfal": "T-fal",
    "instant pot": "Instant Pot",
    "instantpot": "Instant Pot",
    "le creuset": "Le Creuset",
    "oxo": "OXO",
    "all-clad": "All-Clad",
    "all clad": "All-Clad",
    "lodge cast iron": "Lodge",
    "dyson": "Dyson",
    "shark": "Shark",
    "ninja": "Ninja",
    "vitamix": "Vitamix",
    "breville": "Breville",
}

# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_cache: dict[str, str] = {}  # lowered key -> canonical_name
_cache_ts: float = 0.0
_cache_lock: asyncio.Lock | None = None
_CACHE_TTL_SECONDS = 300  # 5 minutes
_CACHE_FALLBACK_RETRY_SECONDS = 30
_FUZZY_THRESHOLD = 0.85


def _get_lock() -> asyncio.Lock:
    """Lazy-init the asyncio lock (must be created inside a running loop)."""
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


async def _ensure_cache() -> None:
    """Rebuild cache from consumer_brand_registry if stale."""
    global _cache, _cache_ts

    if time.monotonic() - _cache_ts < _CACHE_TTL_SECONDS and _cache:
        return

    lock = _get_lock()
    async with lock:
        if time.monotonic() - _cache_ts < _CACHE_TTL_SECONDS and _cache:
            return

        try:
            from ..storage.database import get_db_pool

            pool = get_db_pool()
            if not pool.is_initialized:
                logger.debug("DB not ready, using bootstrap aliases")
                if not _cache:
                    _cache = dict(_BOOTSTRAP_ALIASES)
                _cache_ts = time.monotonic() - _CACHE_TTL_SECONDS + _CACHE_FALLBACK_RETRY_SECONDS
                return

            rows = await pool.fetch(
                "SELECT canonical_name, aliases FROM consumer_brand_registry"
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
                    import json as _json
                    try:
                        alias_list = _json.loads(aliases)
                        for alias in alias_list:
                            if isinstance(alias, str) and alias:
                                new_cache[alias.lower()] = canonical
                    except (ValueError, TypeError):
                        pass

            for key, canonical in _BOOTSTRAP_ALIASES.items():
                if key not in new_cache:
                    new_cache[key] = canonical

            _cache = new_cache
            _cache_ts = time.monotonic()
            logger.debug("Brand cache rebuilt: %d entries", len(_cache))

        except Exception:
            logger.exception("Failed to rebuild brand cache")
            if not _cache:
                _cache = dict(_BOOTSTRAP_ALIASES)
            _cache_ts = time.monotonic() - _CACHE_TTL_SECONDS + _CACHE_FALLBACK_RETRY_SECONDS


def invalidate_cache() -> None:
    """Force a refresh on the next call to _ensure_cache()."""
    global _cache_ts, _cache_lock
    _cache_ts = 0.0
    _cache_lock = None


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------


def _resolve_from_cache(raw: str) -> str:
    """Resolve a brand name using the current cache contents.

    1. Exact match against cache keys (canonical names + aliases).
    2. Fuzzy fallback: best match above _FUZZY_THRESHOLD (0.85).
    3. Return original casing if no match found.
    """
    stripped = raw.strip()
    if not stripped:
        return stripped
    lowered = stripped.lower()

    if lowered in _cache:
        return _cache[lowered]

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

    return stripped.title() if stripped.islower() else stripped


async def resolve_brand_name(raw: str) -> str:
    """Resolve a raw brand name to its canonical form (async, DB-backed)."""
    if not raw:
        return raw or ""
    await _ensure_cache()
    return _resolve_from_cache(raw)


def resolve_brand_name_cached(raw: str) -> str:
    """Resolve a raw brand name using cache only (sync).

    Falls back to bootstrap aliases if the cache has not been populated.
    """
    if not raw:
        return raw or ""
    if not _cache:
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


async def add_brand(canonical_name: str, aliases: list[str] | None = None) -> dict[str, Any]:
    """Insert or update a brand in the registry.

    On conflict, merges the new aliases into the existing set.
    """
    from ..storage.database import get_db_pool
    import json as _json

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    alias_list = [a.lower().strip() for a in (aliases or []) if a.strip()]

    row = await pool.fetchrow(
        """
        INSERT INTO consumer_brand_registry (canonical_name, aliases)
        VALUES ($1, $2::jsonb)
        ON CONFLICT (canonical_name) DO UPDATE SET
            aliases = COALESCE(
                (SELECT jsonb_agg(DISTINCT val)
                 FROM jsonb_array_elements_text(
                     consumer_brand_registry.aliases || EXCLUDED.aliases
                 ) AS t(val)),
                '[]'::jsonb
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
    """Append an alias to an existing brand. Returns updated row or None."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    alias_lower = alias.strip().lower()
    if not alias_lower:
        return None

    row = await pool.fetchrow(
        """
        UPDATE consumer_brand_registry
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


async def list_brands() -> list[dict[str, Any]]:
    """Return all brands from the registry."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise RuntimeError("Database pool not initialized")

    rows = await pool.fetch(
        "SELECT id, canonical_name, aliases, metadata, created_at, updated_at "
        "FROM consumer_brand_registry ORDER BY canonical_name"
    )
    return [dict(r) for r in rows]


async def fuzzy_search_brands(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> list[dict[str, Any]]:
    """Search brands using pg_trgm trigram similarity."""
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
        FROM consumer_brand_registry
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
