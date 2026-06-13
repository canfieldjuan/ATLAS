from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any
import uuid

import httpx
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from atlas_brain.auth.dependencies import AuthUser
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZendeskMacroCredentials,
)
from extracted_content_pipeline.support_ticket_zendesk_export import (
    ZendeskTicketExportError,
)


API_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "api"
    / "content_ops_zendesk_export.py"
)
ACCOUNT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
api = None


def setup_module() -> None:
    global api
    api = _load_api_module()


def _load_api_module():
    spec = importlib.util.spec_from_file_location(
        "atlas_brain.api.content_ops_zendesk_export_for_test",
        API_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Pool:
    is_initialized = True


def _user(account_id: Any = ACCOUNT_ID) -> AuthUser:
    return AuthUser(
        user_id="33333333-3333-3333-3333-333333333333",
        account_id=str(account_id),
        plan="b2b_growth",
        plan_status="active",
        role="owner",
        product="b2b_challenger",
    )


def _credentials() -> ZendeskMacroCredentials:
    return ZendeskMacroCredentials(
        email="agent@example.com",
        api_token="secret-token",
        subdomain="acme",
    )


def _client(
    *,
    pool: Any | None = None,
    user: AuthUser | None = None,
    credential_lookup: Any,
    exporter: Any,
) -> TestClient:
    app = fastapi.FastAPI()

    def pool_provider():
        return pool if pool is not None else _Pool()

    def auth_dependency():
        return user or _user()

    app.include_router(api.create_content_ops_zendesk_export_router(
        pool_provider=pool_provider,
        auth_dependency=auth_dependency,
        credential_lookup=credential_lookup,
        exporter=exporter,
    ))
    return TestClient(app)


def test_zendesk_export_route_uses_tenant_credentials_and_returns_artifact() -> None:
    calls: list[dict[str, Any]] = []

    async def lookup(pool, *, account_id):
        calls.append({"kind": "lookup", "pool": pool, "account_id": account_id})
        return _credentials()

    async def exporter(credentials, *, limit, start_time):
        calls.append({
            "kind": "export",
            "credentials": credentials,
            "limit": limit,
            "start_time": start_time,
        })
        return {"tickets": [{"ticket": {"id": 101}, "comments": []}]}

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post(
        "/content-ops/zendesk-export/full-thread",
        json={
            "limit": 2,
            "start_time": 1_700_000_000,
            "account_id": "99999999-9999-9999-9999-999999999999",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["importer_mode"] == "full_thread"
    assert body["support_platform"] == "zendesk"
    assert body["ticket_count"] == 1
    assert body["limit"] == 2
    assert body["start_time"] == 1_700_000_000
    assert body["artifact"] == {"tickets": [{"ticket": {"id": 101}, "comments": []}]}
    assert calls[0]["kind"] == "lookup"
    assert calls[0]["account_id"] == ACCOUNT_ID
    assert calls[1]["kind"] == "export"
    assert calls[1]["credentials"].normalized_base_url() == "https://acme.zendesk.com"
    assert "secret-token" not in str(body)


def test_zendesk_export_route_fails_closed_without_tenant_credentials() -> None:
    calls: list[str] = []

    async def lookup(pool, *, account_id):
        calls.append(f"lookup:{account_id}")
        return None

    async def exporter(*args, **kwargs):
        raise AssertionError("missing tenant credentials must not export")

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 404
    assert response.json()["detail"]["reason"] == "zendesk_credentials_missing"
    assert calls == [f"lookup:{ACCOUNT_ID}"]


def test_zendesk_export_route_sanitizes_export_errors() -> None:
    async def lookup(pool, *, account_id):
        return _credentials()

    async def exporter(credentials, *, limit, start_time):
        raise ZendeskTicketExportError(
            "zendesk_export_request_failed",
            status_code=429,
        )

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 502
    assert response.json()["detail"] == {
        "reason": "zendesk_export_request_failed",
        "zendesk_status_code": 429,
    }
    assert "secret-token" not in response.text
    assert "Authorization" not in response.text


def test_zendesk_export_route_maps_exporter_missing_credentials_to_404() -> None:
    async def lookup(pool, *, account_id):
        return _credentials()

    async def exporter(credentials, *, limit, start_time):
        raise ZendeskTicketExportError("zendesk_credentials_missing")

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 404
    assert response.json()["detail"] == {"reason": "zendesk_credentials_missing"}


def test_zendesk_export_route_sanitizes_transport_errors() -> None:
    async def lookup(pool, *, account_id):
        return _credentials()

    async def exporter(credentials, *, limit, start_time):
        request = httpx.Request("GET", "https://secret-token.zendesk.com/api/v2")
        raise httpx.ConnectTimeout("timeout secret-token", request=request)

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 502
    assert response.json()["detail"] == {"reason": "zendesk_export_unavailable"}
    assert "secret-token" not in response.text
    assert "Authorization" not in response.text


def test_zendesk_export_route_maps_credential_lookup_failure_to_503() -> None:
    async def lookup(pool, *, account_id):
        raise RuntimeError("database down secret-token")

    async def exporter(*args, **kwargs):
        raise AssertionError("credential lookup failure must stop before export")

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == {"reason": "zendesk_credentials_unavailable"}
    assert "secret-token" not in response.text


def test_zendesk_export_route_rejects_invalid_authenticated_account_id() -> None:
    async def lookup(pool, *, account_id):
        raise AssertionError("invalid auth scope must stop before lookup")

    async def exporter(*args, **kwargs):
        raise AssertionError("invalid auth scope must stop before export")

    response = _client(
        user=_user(account_id="not-a-uuid"),
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid tenant scope"


def test_zendesk_export_route_validates_bounds_before_lookup() -> None:
    calls: list[str] = []

    async def lookup(pool, *, account_id):
        calls.append("lookup")
        return _credentials()

    async def exporter(*args, **kwargs):
        calls.append("export")
        return {"tickets": []}

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 0})

    assert response.status_code == 422
    assert calls == []


def test_zendesk_export_route_fails_closed_on_bad_export_artifact() -> None:
    async def lookup(pool, *, account_id):
        return _credentials()

    async def exporter(credentials, *, limit, start_time):
        return {"ticket": []}

    response = _client(
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 502
    assert response.json()["detail"] == {"reason": "zendesk_export_artifact_invalid"}


def test_zendesk_export_route_fails_when_database_unavailable() -> None:
    class _UninitializedPool:
        is_initialized = False

    async def lookup(pool, *, account_id):
        raise AssertionError("database unavailable should stop before lookup")

    async def exporter(*args, **kwargs):
        raise AssertionError("database unavailable should stop before export")

    response = _client(
        pool=_UninitializedPool(),
        credential_lookup=lookup,
        exporter=exporter,
    ).post("/content-ops/zendesk-export/full-thread", json={"limit": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Database not ready"


def test_zendesk_export_api_import_does_not_require_asyncpg() -> None:
    script = textwrap.dedent("""
import sys

class BlockAsyncpg:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'asyncpg' or fullname.startswith('asyncpg.'):
            raise ModuleNotFoundError("No module named 'asyncpg'")
        return None

sys.meta_path.insert(0, BlockAsyncpg())
import importlib.util
from pathlib import Path

path = Path('atlas_brain/api/content_ops_zendesk_export.py').resolve()
spec = importlib.util.spec_from_file_location(
    'atlas_brain.api.content_ops_zendesk_export_block_asyncpg_test',
    path,
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module.ZendeskFullThreadExportResponse.__name__)
""")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ZendeskFullThreadExportResponse" in result.stdout
