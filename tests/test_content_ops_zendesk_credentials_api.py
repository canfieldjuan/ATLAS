from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap
import uuid

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from atlas_brain._content_ops_zendesk_credentials import ContentOpsZendeskCredentialRecord
from atlas_brain.auth.dependencies import AuthUser


API_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "api"
    / "content_ops_zendesk_credentials.py"
)
ACCOUNT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CREDENTIAL_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
api = None


def setup_module() -> None:
    global api
    api = _load_api_module()


def _load_api_module():
    spec = importlib.util.spec_from_file_location(
        "atlas_brain.api.content_ops_zendesk_credentials_for_test",
        API_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Pool:
    is_initialized = True


def _user(account_id: uuid.UUID = ACCOUNT_ID, *, role: str = "owner") -> AuthUser:
    return AuthUser(
        user_id="33333333-3333-3333-3333-333333333333",
        account_id=str(account_id),
        plan="b2b_growth",
        plan_status="active",
        role=role,
        product="b2b_challenger",
    )


def _record(**overrides) -> ContentOpsZendeskCredentialRecord:
    row = {
        "id": CREDENTIAL_ID,
        "account_id": ACCOUNT_ID,
        "email": "agent@example.com",
        "api_token_prefix": "secret-t",
        "subdomain": "acme",
        "base_url": "",
        "label": "Primary",
        "added_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "last_used_at": None,
        "revoked_at": None,
    }
    row.update(overrides)
    return ContentOpsZendeskCredentialRecord(**row)


def _client(*, pool=None, user: AuthUser | None = None) -> TestClient:
    app = fastapi.FastAPI()

    def pool_provider():
        return pool if pool is not None else _Pool()

    def auth_dependency():
        return user or _user()

    app.include_router(api.create_content_ops_zendesk_credentials_router(
        pool_provider=pool_provider,
        auth_dependency=auth_dependency,
    ))
    return TestClient(app)


def test_list_zendesk_credentials_returns_display_safe_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def list_credentials(pool, *, account_id):
        calls.append({"pool": pool, "account_id": account_id})
        return [_record()]

    monkeypatch.setattr(api, "list_zendesk_credentials", list_credentials)
    client = _client()

    response = client.get("/content-ops/zendesk-credentials")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == str(CREDENTIAL_ID)
    assert body[0]["account_id"] == str(ACCOUNT_ID)
    assert body[0]["api_token_prefix"] == "secret-t"
    assert "api_token" not in body[0]
    assert "encrypted_api_token" not in body[0]
    assert calls[0]["account_id"] == ACCOUNT_ID


def test_add_zendesk_credential_uses_authenticated_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def upsert(pool, **kwargs):
        calls.append({"pool": pool, **kwargs})
        return _record(label=kwargs["label"], subdomain=kwargs["subdomain"])

    monkeypatch.setattr(api, "upsert_zendesk_credentials", upsert)
    client = _client()

    response = client.post(
        "/content-ops/zendesk-credentials",
        json={
            "email": "agent@example.com",
            "api_token": "secret-token",
            "subdomain": "acme",
            "label": "Primary",
        },
    )

    assert response.status_code == 201
    assert response.json()["label"] == "Primary"
    assert calls == [{
        "pool": calls[0]["pool"],
        "account_id": ACCOUNT_ID,
        "email": "agent@example.com",
        "api_token": "secret-token",
        "subdomain": "acme",
        "base_url": "",
        "label": "Primary",
    }]


def test_add_zendesk_credential_maps_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def upsert(pool, **kwargs):
        raise ValueError("Complete Zendesk email, API token, and endpoint are required")

    monkeypatch.setattr(api, "upsert_zendesk_credentials", upsert)
    client = _client()

    response = client.post(
        "/content-ops/zendesk-credentials",
        json={"email": "agent@example.com", "api_token": "secret-token"},
    )

    assert response.status_code == 400
    assert "Complete Zendesk" in response.json()["detail"]


def test_add_zendesk_credential_requires_admin_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def upsert(pool, **kwargs):
        raise AssertionError("member should not be able to write credentials")

    monkeypatch.setattr(api, "upsert_zendesk_credentials", upsert)
    client = _client(user=_user(role="member"))

    response = client.post(
        "/content-ops/zendesk-credentials",
        json={
            "email": "agent@example.com",
            "api_token": "secret-token",
            "subdomain": "acme",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_revoke_zendesk_credential_is_account_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def revoke(pool, *, account_id, credential_id):
        calls.append({"account_id": account_id, "credential_id": credential_id})
        return True

    monkeypatch.setattr(api, "revoke_zendesk_credentials", revoke)
    client = _client()

    response = client.delete(f"/content-ops/zendesk-credentials/{CREDENTIAL_ID}")

    assert response.status_code == 204
    assert calls == [{"account_id": ACCOUNT_ID, "credential_id": CREDENTIAL_ID}]


def test_revoke_zendesk_credential_requires_admin_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def revoke(pool, *, account_id, credential_id):
        raise AssertionError("member should not be able to revoke credentials")

    monkeypatch.setattr(api, "revoke_zendesk_credentials", revoke)
    client = _client(user=_user(role="member"))

    response = client.delete(f"/content-ops/zendesk-credentials/{CREDENTIAL_ID}")

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_revoke_zendesk_credential_returns_404_for_invalid_or_missing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def revoke(pool, *, account_id, credential_id):
        calls.append({"account_id": account_id, "credential_id": credential_id})
        return False

    monkeypatch.setattr(api, "revoke_zendesk_credentials", revoke)
    client = _client()

    invalid = client.delete("/content-ops/zendesk-credentials/not-a-uuid")
    missing = client.delete(f"/content-ops/zendesk-credentials/{CREDENTIAL_ID}")

    assert invalid.status_code == 404
    assert missing.status_code == 404
    assert calls == [{"account_id": ACCOUNT_ID, "credential_id": CREDENTIAL_ID}]


def test_zendesk_credential_routes_fail_when_database_unavailable() -> None:
    class _UninitializedPool:
        is_initialized = False

    client = _client(pool=_UninitializedPool())

    response = client.get("/content-ops/zendesk-credentials")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database not ready"


def test_zendesk_credential_api_import_does_not_require_asyncpg() -> None:
    script = textwrap.dedent("""
import sys

class BlockAsyncpg:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'asyncpg' or fullname.startswith('asyncpg.'):
            raise ModuleNotFoundError(\"No module named 'asyncpg'\")
        return None

sys.meta_path.insert(0, BlockAsyncpg())
import importlib.util
from pathlib import Path

path = Path('atlas_brain/api/content_ops_zendesk_credentials.py').resolve()
spec = importlib.util.spec_from_file_location(
    'atlas_brain.api.content_ops_zendesk_credentials_block_asyncpg_test',
    path,
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module.ZendeskCredentialView.__name__)
""")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ZendeskCredentialView" in result.stdout
