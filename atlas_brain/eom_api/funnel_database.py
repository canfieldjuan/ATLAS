"""Dedicated canonical-CRM database pool for the slim EOM funnel routes."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlsplit

import asyncpg

from .config import EOMFunnelConfig, funnel_settings

logger = logging.getLogger("atlas.eom.funnel_database")


class EOMFunnelDatabasePool:
    """Owns the slim funnel pool pointed at the authoritative Atlas CRM DB."""

    def __init__(self, *, dsn: str) -> None:
        self._dsn = dsn.strip()
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self._pool is not None

    @property
    def target_label(self) -> str:
        try:
            parsed = urlsplit(self._dsn)
            database = parsed.path.lstrip("/") or "<database>"
            host = parsed.hostname or "connection-string"
            port = f":{parsed.port}" if parsed.port else ""
            return f"dsn={host}{port}/{database}"
        except ValueError:
            return "dsn=<connection-string>"

    async def initialize(self) -> None:
        if self._initialized:
            logger.debug("EOM funnel CRM database pool already initialized")
            return
        if not self._dsn:
            raise RuntimeError(
                "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING is required when "
                "ATLAS_EOM_FUNNEL_API_ENABLED=true"
            )
        logger.info(
            "Initializing EOM funnel CRM database pool (%s)",
            self.target_label,
        )
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=1,
            max_size=5,
            command_timeout=30,
        )
        self._initialized = True
        logger.info("EOM funnel CRM database pool initialized")

    async def close(self) -> None:
        if self._pool is not None:
            logger.info("Closing EOM funnel CRM database pool")
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("EOM funnel CRM database pool closed")

    async def acquire(self) -> asyncpg.Connection:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel CRM database pool not initialized")
        return await self._pool.acquire()

    async def release(self, connection: asyncpg.Connection) -> None:
        if self._pool is not None:
            await self._pool.release(connection)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        if not self.is_initialized:
            raise RuntimeError("EOM funnel CRM database pool not initialized")
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def fetch(self, query: str, *args):
        if not self.is_initialized:
            raise RuntimeError("EOM funnel CRM database pool not initialized")
        return await self._pool.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        if not self.is_initialized:
            raise RuntimeError("EOM funnel CRM database pool not initialized")
        return await self._pool.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        if not self.is_initialized:
            raise RuntimeError("EOM funnel CRM database pool not initialized")
        return await self._pool.fetchval(query, *args)

    async def execute(self, query: str, *args):
        if not self.is_initialized:
            raise RuntimeError("EOM funnel CRM database pool not initialized")
        return await self._pool.execute(query, *args)


_eom_funnel_db_pool: EOMFunnelDatabasePool | None = None


def validate_eom_funnel_canonical_crm_config(
    config: EOMFunnelConfig | None = None,
    *,
    canonical_crm_database_confirmed: bool,
) -> None:
    """Require explicit canonical CRM admission before this pool is opened.

    Admission is owed by whoever OPENS the pool, not by the funnel flag that
    first needed it. The receivables billing-recipient routes read canonical
    contacts through this same pool and are enabled independently, so gating
    on ``api_enabled`` alone would open a configured DSN unadmitted: if it
    pointed at a reachable non-canonical Atlas database holding
    ``effingham_maids`` contacts, the receivables bearer could read those
    names and addresses through routes that exist to disclose exactly that.
    """
    from .config import invoicing_settings

    resolved = config or funnel_settings
    # Mirrors the condition in init_eom_funnel_database. Both must agree, or
    # a pool opens without the admission this function exists to require.
    receivables_opens_pool = bool(
        invoicing_settings.receivables_api_enabled
        and resolved.db_connection_string.strip()
    )
    if not resolved.api_enabled and not receivables_opens_pool:
        return
    trigger = (
        "ATLAS_EOM_FUNNEL_API_ENABLED=true"
        if resolved.api_enabled
        else "ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true with "
        "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING set"
    )
    if not canonical_crm_database_confirmed:
        raise RuntimeError(
            "ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true is required when "
            + trigger
        )
    if not resolved.db_connection_string.strip():
        # Only reachable via api_enabled; the receivables branch requires a
        # DSN to be considered at all.
        raise RuntimeError(
            "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING is required when "
            "ATLAS_EOM_FUNNEL_API_ENABLED=true"
        )


def get_eom_funnel_db_pool(
    config: EOMFunnelConfig | None = None,
) -> EOMFunnelDatabasePool:
    """Return the dedicated slim funnel pool."""
    global _eom_funnel_db_pool
    if _eom_funnel_db_pool is None:
        resolved = config or funnel_settings
        _eom_funnel_db_pool = EOMFunnelDatabasePool(
            dsn=resolved.db_connection_string,
        )
    return _eom_funnel_db_pool


async def init_eom_funnel_database(config: EOMFunnelConfig | None = None) -> None:
    """Initialize the dedicated slim funnel CRM pool when anything needs it.

    Gated on the funnel API alone until the receivables billing-recipient
    routes started reading canonical contacts through this pool. Those routes
    are enabled independently, so a deployment with receivables on and the
    funnel API off would leave the pool uninitialized and fail them at runtime.
    Ownership of the contacts data, not the feature flag that first needed it,
    decides when the pool comes up.
    """
    from .config import invoicing_settings

    resolved = config or funnel_settings
    if resolved.api_enabled:
        pool = get_eom_funnel_db_pool(resolved)
        await pool.initialize()
        return
    # Receivables needs this pool for billing recipients, but only when the
    # deployment actually configured one. Raising here would stop a profile
    # that runs receivables without billing recipients from booting at all --
    # a worse failure than the gap being closed. The routes that need the pool
    # fail closed instead.
    if invoicing_settings.receivables_api_enabled and resolved.db_connection_string.strip():
        pool = get_eom_funnel_db_pool(resolved)
        await pool.initialize()
    return


async def close_eom_funnel_database() -> None:
    """Close the dedicated slim funnel CRM pool."""
    if _eom_funnel_db_pool is None:
        return
    pool = _eom_funnel_db_pool
    await pool.close()


def get_eom_funnel_crm_provider():
    """Build a CRM provider pinned to the dedicated slim funnel CRM pool."""
    from ..services.crm_provider import DatabaseCRMProvider

    return DatabaseCRMProvider(pool=get_eom_funnel_db_pool())
