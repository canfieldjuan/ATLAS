"""Production-mounted live-app smoke for calibration-library admin routes (#1497)."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
import uuid

import httpx
import pytest
from fastapi import FastAPI


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
JWT_SECRET = "test-jwt-secret-not-for-prod"


def _database_url() -> str | None:
    return os.environ.get(DATABASE_URL_ENV)


def _migration(name: str) -> str:
    return (MIGRATIONS_DIR / name).read_text(encoding="utf-8")


def _point_atlas_db_settings_at(database_url: str, *, storage_config, database_module) -> None:
    parsed = urlparse(database_url)
    storage_config.db_settings.enabled = True
    storage_config.db_settings.host = parsed.hostname or "localhost"
    storage_config.db_settings.port = parsed.port or 5432
    storage_config.db_settings.database = parsed.path.lstrip("/")
    storage_config.db_settings.user = parsed.username or ""
    storage_config.db_settings.password = parsed.password or ""
    storage_config.db_settings.socket_path = None
    database_module.db_settings = storage_config.db_settings


async def _apply_required_migrations(conn) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    for name in (
        "076_saas_accounts.sql",
        "079_b2b_saas.sql",
        "338_saas_users_is_admin.sql",
        "334_content_ops_claim_registry.sql",
        "335_content_ops_calibration_library.sql",
    ):
        await conn.execute(_migration(name))


async def _seed_user(conn, *, role: str, account_id: uuid.UUID) -> uuid.UUID:
    user_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO saas_users
            (id, account_id, email, password_hash, role, is_active)
        VALUES ($1, $2, $3, 'not-used-in-jwt-smoke', $4, TRUE)
        """,
        user_id,
        account_id,
        f"{user_id}@example.test",
        role,
    )
    return user_id


async def _seed_account(
    conn,
    *,
    name: str,
    plan: str = "b2b_growth",
    product: str = "b2b_challenger",
) -> uuid.UUID:
    account_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO saas_accounts
            (id, name, plan, plan_status, product)
        VALUES ($1, $2, $3, 'active', $4)
        """,
        account_id,
        name,
        plan,
        product,
    )
    return account_id


def _auth_header(user_id: uuid.UUID, account_id: uuid.UUID) -> dict[str, str]:
    from atlas_brain.auth.jwt import create_access_token

    token = create_access_token(
        str(user_id),
        str(account_id),
        "b2b_growth",
    )
    return {"Authorization": f"Bearer {token}"}


def _calibration_body(example_id: str = "voice-001") -> dict[str, object]:
    return {
        "example_id": example_id,
        "label": "voice_drift",
        "excerpt": "Crush your quota instantly.",
        "reasoning": "The phrasing is too aggressive for this brand voice.",
        "source": "live-app-smoke",
        "metadata": {"suite": "content-ops-calibration-live-app"},
    }


@pytest.mark.asyncio
async def test_calibration_library_admin_router_live_app_auth_tenant_and_db_smoke() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = _database_url()
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")

    from atlas_brain.config import settings
    from atlas_brain.storage import config as storage_config
    from atlas_brain.storage import database as database_module

    original_saas_auth = {
        "enabled": settings.saas_auth.enabled,
        "jwt_secret": settings.saas_auth.jwt_secret,
        "api_key_pepper": settings.saas_auth.api_key_pepper,
        "byok_encryption_kek": settings.saas_auth.byok_encryption_kek,
    }
    original_db_settings = {
        "enabled": storage_config.db_settings.enabled,
        "host": storage_config.db_settings.host,
        "port": storage_config.db_settings.port,
        "database": storage_config.db_settings.database,
        "user": storage_config.db_settings.user,
        "password": storage_config.db_settings.password,
        "socket_path": storage_config.db_settings.socket_path,
    }

    _point_atlas_db_settings_at(
        database_url,
        storage_config=storage_config,
        database_module=database_module,
    )
    settings.saas_auth.enabled = True
    settings.saas_auth.jwt_secret = JWT_SECRET
    settings.saas_auth.api_key_pepper = "test-api-key-pepper-not-for-prod"
    settings.saas_auth.byok_encryption_kek = "test:DEj0-fNH6mOs5JYXn3Uv6ejEfP4PQ6XIqWla36eIR_U="
    database_module._db_pool = None

    conn = await asyncpg.connect(database_url)
    seeded_account_ids: list[uuid.UUID] = []
    try:
        await _apply_required_migrations(conn)
        owner_account_id = await _seed_account(conn, name="Calibration smoke owner")
        other_account_id = await _seed_account(conn, name="Calibration smoke other")
        trial_account_id = await _seed_account(
            conn, name="Calibration smoke trial", plan="b2b_trial"
        )
        seeded_account_ids.extend([owner_account_id, other_account_id, trial_account_id])
        owner_user_id = await _seed_user(conn, role="owner", account_id=owner_account_id)
        member_user_id = await _seed_user(conn, role="member", account_id=owner_account_id)
        other_owner_user_id = await _seed_user(
            conn, role="owner", account_id=other_account_id
        )
        trial_owner_user_id = await _seed_user(
            conn, role="owner", account_id=trial_account_id
        )

        await database_module.init_database()

        from atlas_brain.api import router as api_router

        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            route = "/api/v1/content-ops/calibration-library"

            unauthenticated = await client.get(route)
            assert unauthenticated.status_code == 401

            below_plan = await client.get(
                route,
                headers=_auth_header(trial_owner_user_id, trial_account_id),
            )
            assert below_plan.status_code == 403

            member_write = await client.post(
                route,
                headers=_auth_header(member_user_id, owner_account_id),
                json=_calibration_body(),
            )
            assert member_write.status_code == 403

            created = await client.post(
                route,
                headers=_auth_header(owner_user_id, owner_account_id),
                json=_calibration_body(),
            )
            assert created.status_code == 201
            created_body = created.json()
            assert created_body["account_id"] == str(owner_account_id)
            assert created_body["example_id"] == "voice-001"

            duplicate = await client.post(
                route,
                headers=_auth_header(owner_user_id, owner_account_id),
                json=_calibration_body(" voice-001 "),
            )
            assert duplicate.status_code == 409

            owner_list = await client.get(
                route,
                headers=_auth_header(owner_user_id, owner_account_id),
            )
            assert owner_list.status_code == 200
            assert [row["example_id"] for row in owner_list.json()] == ["voice-001"]

            other_tenant_list = await client.get(
                route,
                headers=_auth_header(other_owner_user_id, other_account_id),
            )
            assert other_tenant_list.status_code == 200
            assert other_tenant_list.json() == []

            deleted = await client.delete(
                f"{route}/{created_body['id']}",
                headers=_auth_header(owner_user_id, owner_account_id),
            )
            assert deleted.status_code == 204

        archived_at = await conn.fetchval(
            """
            SELECT archived_at
            FROM content_ops_calibration_library
            WHERE id = $1 AND account_id = $2
            """,
            uuid.UUID(created_body["id"]),
            owner_account_id,
        )
        assert archived_at is not None
    finally:
        try:
            await database_module.close_database()
            database_module._db_pool = None
            for account_id in seeded_account_ids:
                await conn.execute("DELETE FROM saas_accounts WHERE id = $1", account_id)
            await conn.close()
        finally:
            for key, value in original_saas_auth.items():
                setattr(settings.saas_auth, key, value)
            for key, value in original_db_settings.items():
                setattr(storage_config.db_settings, key, value)
            database_module.db_settings = storage_config.db_settings
