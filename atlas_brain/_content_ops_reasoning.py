"""Host factories for the Content Ops reasoning context provider.

PR #402 shipped the route-level ``reasoning_context_provider``
seam in ``extracted_content_pipeline/api/control_surfaces.py``
plus the bundle's per-request ``with_reasoning_context()``
derivation. This module is the host adapter the route mount
calls to obtain a configured provider (or ``None``).

Two providers are offered side-by-side:

* ``build_content_ops_reasoning_context_provider`` (PR #462) --
  file-backed reference adapter. Operators opt in by setting
  ``ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH`` to a JSON file
  readable by ``FileCampaignReasoningContextProvider``.
* ``build_postgres_content_ops_reasoning_context_provider`` --
  DB-backed adapter (this slice). Operators opt in by setting
  ``ATLAS_CONTENT_OPS_REASONING_DB_ENABLED=true``; the factory
  binds to the host's existing asyncpg pool.

``select_content_ops_reasoning_context_provider`` is the chooser
the route mount passes as the ``reasoning_context_provider``
kwarg: it tries DB first, falls back to file, returns ``None``
when neither is configured. Both factories preserve the
WARN-and-fall-back contract -- a misconfigured provider must
not crash the route mount or block the entire Content Ops
surface for all tenants.

Lives at ``atlas_brain/`` root with underscore prefix to dodge
the heavy ``atlas_brain.services`` import chain (same pattern as
``_content_ops_infrastructure.py`` / ``_content_ops_scope.py``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


_ENV_VAR = "ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH"
_DB_ENV_VAR = "ATLAS_CONTENT_OPS_REASONING_DB_ENABLED"


def build_content_ops_reasoning_context_provider(
    *,
    path_factory: Callable[[], str | None] | None = None,
    loader_factory: Callable[[str], Any] | None = None,
) -> Any | None:
    """Return the configured reasoning context provider, or
    ``None`` when the host hasn't opted in.

    Hosts opt in by setting
    ``ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH`` to a JSON file
    readable by ``FileCampaignReasoningContextProvider``. Failures
    (missing file, parse errors) resolve to ``None`` with a
    warning logged -- the reasoning provider is enrichment, a
    bad file must not block the entire Content Ops surface for
    all tenants.

    DI kwargs let tests stub the env-var read and the loader
    without touching the filesystem or the heavy
    ``extracted_content_pipeline.campaign_reasoning_data`` module.
    """

    path = (path_factory or _read_env_path)()
    if not path:
        return None
    if not Path(path).is_file():
        logger.warning(
            "Content Ops reasoning context path %s does not exist; "
            "provider stays unwired.",
            path,
        )
        return None

    loader = loader_factory or _default_loader
    try:
        return loader(path)
    except Exception as exc:
        logger.warning(
            "Failed to load Content Ops reasoning context from %s: %s",
            path,
            exc,
        )
        return None


def _read_env_path() -> str | None:
    """Read the env var; return ``None`` for unset / empty."""

    value = os.environ.get(_ENV_VAR)
    return value or None


def _default_loader(path: str) -> Any:
    """Lazy import the package's file-backed loader.

    Keeps this module light enough to import in dependency-light
    dev envs; the heavier
    ``extracted_content_pipeline.campaign_reasoning_data`` only
    loads when the env var is actually set.
    """

    from extracted_content_pipeline.campaign_reasoning_data import (
        load_campaign_reasoning_context_provider,
    )

    return load_campaign_reasoning_context_provider(path)


def build_postgres_content_ops_reasoning_context_provider(
    *,
    enabled_factory: Callable[[], bool] | None = None,
    pool_factory: Callable[[], Any] | None = None,
    repository_factory: Callable[[Any], Any] | None = None,
) -> Any | None:
    """Return a Postgres-backed reasoning context provider, or
    ``None`` when the host hasn't opted in or the pool is not
    available yet.

    Hosts opt in by setting ``ATLAS_CONTENT_OPS_REASONING_DB_ENABLED``
    to a truthy value (``true`` / ``1`` / ``yes``). The factory
    pulls the host's existing asyncpg pool via
    ``atlas_brain.storage.database.get_db_pool`` and binds a
    ``PostgresCampaignReasoningContextRepository`` against it.

    A missing pool (DB not initialized yet, or the host runs
    without Postgres) resolves to ``None`` with a WARN log -- the
    route mount stays unwired rather than crashing.

    DI kwargs let tests stub the env-var read, the pool acquire,
    and the repository construction without touching the
    filesystem or the heavy ``extracted_content_pipeline``
    storage modules.
    """

    if not (enabled_factory or _read_db_enabled)():
        return None

    try:
        pool = (pool_factory or _default_pool_factory)()
    except Exception as exc:
        logger.warning(
            "Content Ops reasoning DB pool acquire failed: %s; "
            "DB-backed provider stays unwired.",
            exc,
        )
        return None
    if pool is None:
        logger.warning(
            "Content Ops reasoning DB enabled but pool is not "
            "available; DB-backed provider stays unwired.",
        )
        return None

    factory = repository_factory or _default_repository_factory
    try:
        return factory(pool)
    except Exception as exc:
        logger.warning(
            "Failed to construct Content Ops reasoning DB "
            "repository: %s; DB-backed provider stays unwired.",
            exc,
        )
        return None


def select_content_ops_reasoning_context_provider(
    *,
    db_factory: Callable[[], Any | None] | None = None,
    file_factory: Callable[[], Any | None] | None = None,
) -> Any | None:
    """Pick the configured reasoning provider (DB > file > None).

    The route mount passes this chooser as the
    ``reasoning_context_provider`` kwarg. DB takes precedence
    because it scales per-tenant; the file-backed adapter stays
    available as a single-tenant / staging fallback. ``None``
    means no provider is configured -- the bundle's existing
    ``with_reasoning_context()`` derivation already handles that.

    Both factories own their own WARN-and-fall-back behavior so
    the chooser stays trivial -- failing factories return
    ``None`` and the chooser advances to the next.
    """

    db_pick = (db_factory or build_postgres_content_ops_reasoning_context_provider)()
    if db_pick is not None:
        return db_pick
    return (file_factory or build_content_ops_reasoning_context_provider)()


def _read_db_enabled() -> bool:
    """Parse the DB opt-in env var; default ``False`` when unset
    or unrecognized so a typo'd value resolves to "off"."""

    value = os.environ.get(_DB_ENV_VAR, "").strip().lower()
    return value in {"true", "1", "yes", "on"}


def _default_pool_factory() -> Any:
    """Lazy import the host's asyncpg pool getter.

    Keeps this module light enough to import in dependency-light
    dev envs; ``atlas_brain.storage.database`` only loads when
    the DB env var is actually set.
    """

    from atlas_brain.storage.database import get_db_pool

    return get_db_pool()


def _default_repository_factory(pool: Any) -> Any:
    """Lazy import the package's Postgres repository.

    Mirrors ``_default_loader``: the heavier
    ``extracted_content_pipeline.campaign_reasoning_postgres``
    module loads only when the DB env var is set + a pool is
    available.
    """

    from extracted_content_pipeline.campaign_reasoning_postgres import (
        PostgresCampaignReasoningContextRepository,
    )

    return PostgresCampaignReasoningContextRepository(pool=pool)


__all__ = [
    "build_content_ops_reasoning_context_provider",
    "build_postgres_content_ops_reasoning_context_provider",
    "select_content_ops_reasoning_context_provider",
]
