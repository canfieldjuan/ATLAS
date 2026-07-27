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
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .eom_api.auth import validate_receivables_api_config
from .eom_api.config import (
    eom_profile_settings,
    eom_settings,
    invoicing_settings,
)
from .eom_api.receivables import router as receivables_router
from .logging_config import configure_logging
from .storage.config import db_settings
from .storage.database import close_database, get_db_pool, init_database

configure_logging(level=eom_settings.log_level, log_format=eom_settings.log_format)
logger = logging.getLogger("atlas.eom")


async def _run_startup_migrations() -> None:
    from .storage.migrations import run_migrations

    pool = get_db_pool()
    if pool.is_initialized:
        await run_migrations(pool)
        logger.info("Database migrations checked")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize only the dependencies required by the EOM API profile."""
    logger.info("Atlas EOM API starting up")
    validate_receivables_api_config(invoicing_settings)

    try:
        if db_settings.enabled:
            await init_database()
            logger.info("Database connection pool initialized")
            if eom_profile_settings.run_migrations:
                await _run_startup_migrations()
        else:
            logger.warning("Database persistence is disabled")
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
