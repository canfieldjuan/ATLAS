from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import uuid

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from atlas_brain.auth.dependencies import AuthUser


# Load the router module by file path to bypass the heavy atlas_brain.api
# package __init__ (which pulls in asyncpg/numpy via the storage chain).
API_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "api"
    / "content_ops_calibration_library.py"
)
api = None


def setup_module() -> None:
    global api
    spec = importlib.util.spec_from_file_location(
        "atlas_brain.api.content_ops_calibration_library_for_test",
        API_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    api = module


ACCOUNT_ID = uuid.uuid4()


class _Pool:
    def __init__(self, *, fetchrow_result=None, fetch_rows=None) -> None:
        self.fetchrow_result = fetchrow_result
        self.fetch_rows = list(fetch_rows or [])
        self.fetchrow_calls: list[tuple] = []
        self.fetch_calls: list[tuple] = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append(args)
        return self.fetchrow_result

    async def fetch(self, query, *args):
        self.fetch_calls.append(args)
        return self.fetch_rows


def _row(**overrides):
    row = {
        "id": uuid.uuid4(),
        "account_id": ACCOUNT_ID,
        "example_id": "overclaim-001",
        "label": "overclaim",
        "excerpt": "guaranteed 99.99% uptime",
        "reasoning": "No SLA backs this number.",
        "source": "curated",
        "metadata": "{}",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "archived_at": None,
    }
    row.update(overrides)
    return row


def _user(account_id=ACCOUNT_ID, *, role: str = "owner") -> AuthUser:
    return AuthUser(
        user_id="33333333-3333-3333-3333-333333333333",
        account_id=str(account_id),
        plan="b2b_growth",
        plan_status="active",
        role=role,
        product="b2b_challenger",
    )


def _client(*, pool=None, user: AuthUser | None = None) -> TestClient:
    app = fastapi.FastAPI()

    def pool_provider():
        return pool if pool is not None else _Pool()

    def auth_dependency():
        return user or _user()

    app.include_router(
        api.create_content_ops_calibration_library_router(
            pool_provider=pool_provider,
            auth_dependency=auth_dependency,
        )
    )
    return TestClient(app)


_VALID_BODY = {
    "example_id": "overclaim-001",
    "label": "overclaim",
    "excerpt": "guaranteed 99.99% uptime",
    "reasoning": "No SLA backs this number.",
}


def test_list_returns_tenant_examples() -> None:
    pool = _Pool(fetch_rows=[_row()])
    resp = _client(pool=pool).get("/content-ops/calibration-library")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["example_id"] == "overclaim-001"
    assert body[0]["label"] == "overclaim"
    assert pool.fetch_calls[0] == (ACCOUNT_ID,)


def test_create_requires_admin_role() -> None:
    resp = _client(user=_user(role="member")).post(
        "/content-ops/calibration-library", json=_VALID_BODY
    )
    assert resp.status_code == 403


def test_create_inserts_and_returns_view() -> None:
    pool = _Pool(fetchrow_result=_row())
    resp = _client(pool=pool).post("/content-ops/calibration-library", json=_VALID_BODY)
    assert resp.status_code == 201
    assert resp.json()["example_id"] == "overclaim-001"
    # account scoping: first insert arg is the tenant account id.
    assert pool.fetchrow_calls[0][0] == ACCOUNT_ID


def test_create_rejects_invalid_label() -> None:
    pool = _Pool(fetchrow_result=_row())
    body = dict(_VALID_BODY, label="not_a_label")
    resp = _client(pool=pool).post("/content-ops/calibration-library", json=body)
    assert resp.status_code == 400
    assert pool.fetchrow_calls == []  # never reached the DB


def test_create_translates_unique_violation_to_409() -> None:
    class UniqueViolationError(Exception):  # matches asyncpg's class name
        pass

    class _ConflictPool(_Pool):
        async def fetchrow(self, query, *args):
            raise UniqueViolationError()

    resp = _client(pool=_ConflictPool()).post("/content-ops/calibration-library", json=_VALID_BODY)
    assert resp.status_code == 409


def test_update_missing_row_is_404() -> None:
    pool = _Pool(fetchrow_result=None)
    resp = _client(pool=pool).put(
        f"/content-ops/calibration-library/{uuid.uuid4()}", json=_VALID_BODY
    )
    assert resp.status_code == 404


def test_update_returns_view_for_existing_row() -> None:
    pool = _Pool(fetchrow_result=_row())
    resp = _client(pool=pool).put(
        f"/content-ops/calibration-library/{uuid.uuid4()}", json=_VALID_BODY
    )
    assert resp.status_code == 200
    assert resp.json()["label"] == "overclaim"


def test_delete_archives_and_returns_204() -> None:
    pool = _Pool(fetchrow_result={"id": uuid.uuid4()})
    resp = _client(pool=pool).delete(f"/content-ops/calibration-library/{uuid.uuid4()}")
    assert resp.status_code == 204


def test_delete_missing_row_is_404() -> None:
    pool = _Pool(fetchrow_result=None)
    resp = _client(pool=pool).delete(f"/content-ops/calibration-library/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_invalid_tenant_scope_is_401() -> None:
    resp = _client(user=_user(account_id="not-a-uuid")).get("/content-ops/calibration-library")
    assert resp.status_code == 401
