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
from .eom_api.funnel_readiness import require_eom_funnel_data_store
from .eom_api.receivables import router as receivables_router
from .logging_config import configure_logging
from .storage.config import db_settings
from .storage.database import close_database, get_db_pool, init_database

configure_logging(level=eom_settings.log_level, log_format=eom_settings.log_format)
logger = logging.getLogger("atlas.eom")

MigrationRunner = Callable[..., Awaitable[None]]

# Closure for the EOM receivables migration set: CLOSED and ENUMERATED from the
# current receivables readiness contract plus its SQL prerequisite chain.
EOM_RECEIVABLES_READINESS_MIGRATIONS: tuple[str, ...] = (
    "012_appointments",
    "035_contacts",
    "045_invoices",
    "344_receivables_payments",
    "345_receivables_event_key_lookup",
)

# Closure for the EOM funnel migration set: CLOSED and ENUMERATED from the
# current lead-review/customer-handoff contracts plus their SQL prerequisite
# chain.
EOM_FUNNEL_READINESS_MIGRATIONS: tuple[str, ...] = (
    "346_contact_lead_pipeline",
    "351_eom_lead_lifecycle_events",
    "353_eom_customer_handoffs",
    "354_eom_customer_handoff_privileges",
    "355_eom_lead_review_queue_index",
)

# Migrations outside this set are intentionally not run by the EOM profile; a
# new EOM API table/index or SQL prerequisite must update this tuple and the
# live readiness migration test in the same slice.
EOM_API_READINESS_MIGRATIONS: tuple[str, ...] = (
    *EOM_RECEIVABLES_READINESS_MIGRATIONS,
    *EOM_FUNNEL_READINESS_MIGRATIONS,
)


async def _apply_eom_api_migrations(
    pool: Any,
    *,
    migrations: tuple[str, ...] = EOM_API_READINESS_MIGRATIONS,
    run_migrations_fn: MigrationRunner | None = None,
) -> None:
    if run_migrations_fn is None:
        from .storage.migrations import run_migrations

        runner = run_migrations
    else:
        runner = run_migrations_fn

    await runner(pool, only=migrations)


def _startup_migrations_for_enabled_apis() -> tuple[str, ...]:
    if funnel_settings.api_enabled:
        _require_canonical_crm_database_for_funnel()
        return EOM_API_READINESS_MIGRATIONS
    return EOM_RECEIVABLES_READINESS_MIGRATIONS


def _require_canonical_crm_database_for_funnel() -> None:
    if (
        funnel_settings.api_enabled
        and not eom_profile_settings.canonical_crm_database_confirmed
    ):
        raise RuntimeError(
            "EOM funnel requires the canonical Atlas CRM database before startup; "
            "set ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true only after "
            "ATLAS_DB_CONNECTION_STRING points at the canonical CRM store."
        )


async def _run_startup_migrations() -> None:
    pool = get_db_pool()
    if pool.is_initialized:
        migrations = _startup_migrations_for_enabled_apis()
        await _apply_eom_api_migrations(pool, migrations=migrations)
        logger.info(
            "EOM API readiness migrations checked: %s",
            ", ".join(migrations),
        )


async def _validate_eom_funnel_startup() -> None:
    _require_canonical_crm_database_for_funnel()
    await require_eom_funnel_data_store(
        funnel_settings,
        database_enabled=db_settings.enabled,
        pool_getter=get_db_pool,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize only the dependencies required by the EOM API profile."""
    logger.info("Atlas EOM API starting up")
    validate_receivables_api_config(invoicing_settings)
    validate_eom_funnel_api_config(funnel_settings)
    _require_canonical_crm_database_for_funnel()

    try:
        if db_settings.enabled:
            await init_database()
            logger.info("Database connection pool initialized")
            if eom_profile_settings.run_migrations:
                await _run_startup_migrations()
        else:
            logger.warning("Database persistence is disabled")
        await _validate_eom_funnel_startup()
        yield
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
app.include_router(funnel_router, prefix="/api/v1")
