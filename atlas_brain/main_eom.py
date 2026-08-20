"""Slim Atlas API entrypoint for Effingham Office Maids business systems.

This profile is intentionally separate from ``atlas_brain.main``. It avoids the
full Atlas router aggregate, local model startup, voice/ASR boot, B2B scraping,
Content Ops, and autonomous scheduler surfaces. The first supported vertical is
the service-to-service receivables API consumed by the EOM time tracker.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


def _dotenv_values_with_context(
    dotenv_path: Path,
    context: Mapping[str, str],
) -> dict[str, str | None]:
    """Parse a dotenv file while allowing earlier files to satisfy interpolation."""
    added_keys: list[str] = []
    for key, value in context.items():
        if key not in os.environ:
            os.environ[key] = value
            added_keys.append(key)
    try:
        return dotenv_values(dotenv_path)
    finally:
        for key in added_keys:
            os.environ.pop(key, None)


def _load_local_env_files(env_root: Path) -> None:
    """Load local env defaults while preserving real process env values.

    ``.env.local`` is the machine-local override for ``.env``. Values that were
    already present in the launching process still win over both files, which
    keeps Render/operator-injected environment variables authoritative.
    """

    process_keys = set(os.environ)
    merged: dict[str, str] = {}
    for filename in (".env", ".env.local"):
        values = _dotenv_values_with_context(env_root / filename, merged)
        for key, value in values.items():
            if value is not None:
                merged[key] = value
    for key, value in merged.items():
        if key not in process_keys:
            os.environ[key] = value


_env_root = Path(__file__).parent.parent
_load_local_env_files(_env_root)

import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .eom_api.auth import validate_receivables_api_config
from .eom_api.config import (
    eom_profile_settings,
    eom_settings,
    funnel_settings,
    invoicing_settings,
)
from .eom_api.funnel import router as funnel_router
from .eom_api.funnel_auth import validate_eom_funnel_api_config
from .eom_api.funnel_database import (
    close_eom_funnel_database,
    get_eom_funnel_crm_provider,
    get_eom_funnel_db_pool,
    init_eom_funnel_database,
    validate_eom_funnel_canonical_crm_config,
)
from .eom_api.funnel_store import (
    require_eom_funnel_data_store as _require_eom_funnel_data_store_with_pool,
)
from .eom_api.receivables import router as receivables_router
from .logging_config import configure_logging
from .storage.config import db_settings
from .storage.database import close_database, get_db_pool, init_database

configure_logging(level=eom_settings.log_level, log_format=eom_settings.log_format)
logger = logging.getLogger("atlas.eom")

MigrationRunner = Callable[..., Awaitable[None]]

# Closure for the EOM startup migration set: CLOSED and ENUMERATED from the
# current receivables readiness contract plus its SQL prerequisite chain.
# Migrations outside this set are intentionally not run by the EOM profile; a
# new receivables readiness table/index or SQL prerequisite must update this
# tuple and the live readiness migration test in the same slice.
EOM_RECEIVABLES_READINESS_MIGRATIONS: tuple[str, ...] = (
    "012_appointments",
    "035_contacts",
    "045_invoices",
    "344_receivables_payments",
    "345_receivables_event_key_lookup",
    "368_receivables_payment_check_metadata",
    "369_receivables_payment_receipt_outbox",
    "378_receivables_payment_receipt_delivery",
    "379_receivables_payment_receipt_delivery_recovery",
    "385_invoices_billing_period_dedup",
)


async def _apply_eom_receivables_migrations(
    pool: Any,
    run_migrations_fn: MigrationRunner | None = None,
) -> None:
    if run_migrations_fn is None:
        from .storage.migrations import run_migrations

        runner = run_migrations
    else:
        runner = run_migrations_fn

    await runner(pool, only=EOM_RECEIVABLES_READINESS_MIGRATIONS)


async def _run_startup_migrations() -> None:
    pool = get_db_pool()
    if pool.is_initialized:
        await _apply_eom_receivables_migrations(pool)
        logger.info(
            "EOM receivables readiness migrations checked: %s",
            ", ".join(EOM_RECEIVABLES_READINESS_MIGRATIONS),
        )


class ReceivablesSchemaUnavailableError(RuntimeError):
    """Raised when the enabled receivables API's schema prerequisites are missing.

    ``eom_profile_settings.run_migrations`` defaults to False, so this
    profile does not always run migrations on startup. An enabled
    receivables API must still be fenced against serving requests on a
    schema that is missing a required column/index -- mirrors the
    ``receivables_ready_fn`` fence already used by the invoicing MCP
    servers' ``_database_lifespan``.
    """


async def _require_receivables_schema_ready() -> None:
    from .services.receivables import ReceivablesService

    pool = get_db_pool()
    if not pool.is_initialized:
        return
    if not await ReceivablesService(pool).is_ready():
        raise ReceivablesSchemaUnavailableError(
            "EOM API has receivables_api_enabled=true but the receivables "
            "schema is not ready (a required column or index is missing). "
            "Run pending migrations before starting this profile with the "
            "receivables API enabled."
        )
    logger.info("EOM receivables schema readiness verified")


async def _require_eom_funnel_data_store(
    config: object,
    *,
    database_enabled: bool,
) -> None:
    """Fail closed if an enabled handoff cannot use the canonical CRM store."""
    await _require_eom_funnel_data_store_with_pool(
        config,
        database_enabled=database_enabled,
        get_db_pool_fn=get_eom_funnel_db_pool,
    )


async def _validate_eom_funnel_startup() -> None:
    """Run enabled funnel datastore preflight after DB init and before migrations."""
    await _require_eom_funnel_data_store(
        funnel_settings,
        database_enabled=db_settings.enabled,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize only the dependencies required by the EOM API profile."""
    logger.info("Atlas EOM API starting up")
    validate_receivables_api_config(invoicing_settings)
    validate_eom_funnel_api_config(funnel_settings)
    validate_eom_funnel_canonical_crm_config(
        funnel_settings,
        canonical_crm_database_confirmed=(
            eom_profile_settings.canonical_crm_database_confirmed
        ),
    )

    try:
        if db_settings.enabled:
            await init_database()
            logger.info("Database connection pool initialized")
        else:
            logger.warning("Database persistence is disabled")
        await init_eom_funnel_database(funnel_settings)
        await _validate_eom_funnel_startup()
        if db_settings.enabled and eom_profile_settings.run_migrations:
            await _run_startup_migrations()
        if db_settings.enabled and invoicing_settings.receivables_api_enabled:
            await _require_receivables_schema_ready()
        yield
    finally:
        try:
            # Mirrors init_eom_funnel_database's condition. Closing only when
            # the funnel API is on would leak the pool in the deployment that
            # opened it for receivables alone.
            if funnel_settings.api_enabled or (
                invoicing_settings.receivables_api_enabled
                and funnel_settings.db_connection_string.strip()
            ):
                await close_eom_funnel_database()
        finally:
            if db_settings.enabled:
                await close_database()
                logger.info("Database connection pool closed")
        logger.info("Atlas EOM API shutdown complete")


app = FastAPI(
    title="Atlas EOM API",
    description=(
        "Slim service-to-service API for Effingham Office Maids business "
        "systems. Heavy Atlas, B2B, voice, and local-model surfaces are not "
        "mounted in this profile."
    ),
    version="0.1.0",
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/api/v1/ping", tags=["Health"])
async def ping() -> dict[str, str]:
    """Simple Render/liveness endpoint that does not require database access."""
    return {"status": "ok", "profile": "eom"}


app.include_router(receivables_router, prefix="/api/v1")
app.state.eom_funnel_crm_provider = get_eom_funnel_crm_provider
app.include_router(funnel_router, prefix="/api/v1")
