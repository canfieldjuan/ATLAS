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

from atlas_brain._content_ops_brand_voice_profiles import (
    ContentOpsBrandVoiceProfileRecord,
)
from atlas_brain.auth.dependencies import AuthUser


API_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "api"
    / "content_ops_brand_voice_profiles.py"
)
ACCOUNT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
PROFILE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
api = None


def setup_module() -> None:
    global api
    api = _load_api_module()


def _load_api_module():
    spec = importlib.util.spec_from_file_location(
        "atlas_brain.api.content_ops_brand_voice_profiles_for_test",
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


def _record(**overrides) -> ContentOpsBrandVoiceProfileRecord:
    row = {
        "id": PROFILE_ID,
        "account_id": ACCOUNT_ID,
        "name": "Acme editorial",
        "descriptors": ("plainspoken",),
        "exemplars": ("Write like this.",),
        "banned_terms": ("synergy",),
        "preferred_pov": "second_person",
        "reading_level": "plain",
        "metadata": {"source": "operator"},
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "archived_at": None,
    }
    row.update(overrides)
    return ContentOpsBrandVoiceProfileRecord(**row)


def _client(*, pool=None, user: AuthUser | None = None) -> TestClient:
    app = fastapi.FastAPI()

    def pool_provider():
        return pool if pool is not None else _Pool()

    def auth_dependency():
        return user or _user()

    app.include_router(api.create_content_ops_brand_voice_profiles_router(
        pool_provider=pool_provider,
        auth_dependency=auth_dependency,
    ))
    return TestClient(app)


def test_list_brand_voice_profiles_returns_tenant_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def list_profiles(pool, *, account_id):
        calls.append({"pool": pool, "account_id": account_id})
        return [_record()]

    monkeypatch.setattr(api, "list_brand_voice_profiles", list_profiles)
    client = _client()

    response = client.get("/content-ops/brand-voice-profiles")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == str(PROFILE_ID)
    assert body[0]["account_id"] == str(ACCOUNT_ID)
    assert body[0]["descriptors"] == ["plainspoken"]
    assert body[0]["banned_terms"] == ["synergy"]
    assert calls[0]["account_id"] == ACCOUNT_ID


def test_add_brand_voice_profile_uses_authenticated_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def create_profile(pool, *, account_id, payload):
        calls.append({"pool": pool, "account_id": account_id, "payload": payload})
        return _record(name=payload["name"])

    monkeypatch.setattr(api, "create_brand_voice_profile", create_profile)
    client = _client()

    response = client.post(
        "/content-ops/brand-voice-profiles",
        json={"name": "Acme editorial", "descriptors": ["plainspoken"]},
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Acme editorial"
    assert calls[0]["account_id"] == ACCOUNT_ID
    assert calls[0]["payload"]["descriptors"] == ("plainspoken",)


def test_add_brand_voice_profile_requires_admin_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def create_profile(pool, *, account_id, payload):
        raise AssertionError("member should not create profiles")

    monkeypatch.setattr(api, "create_brand_voice_profile", create_profile)
    client = _client(user=_user(role="member"))

    response = client.post(
        "/content-ops/brand-voice-profiles",
        json={"name": "Acme editorial", "descriptors": ["plainspoken"]},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_update_brand_voice_profile_is_account_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def update_profile(pool, *, account_id, profile_id, payload):
        calls.append({
            "account_id": account_id,
            "profile_id": profile_id,
            "payload": payload,
        })
        return _record(id=profile_id, account_id=account_id, name=payload["name"])

    monkeypatch.setattr(api, "update_brand_voice_profile", update_profile)
    client = _client()

    response = client.put(
        f"/content-ops/brand-voice-profiles/{PROFILE_ID}",
        json={"name": "Acme v2", "descriptors": ["direct"]},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Acme v2"
    assert calls == [{
        "account_id": ACCOUNT_ID,
        "profile_id": PROFILE_ID,
        "payload": calls[0]["payload"],
    }]
    assert calls[0]["payload"]["descriptors"] == ("direct",)


def test_update_brand_voice_profile_returns_404_for_missing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def update_profile(pool, *, account_id, profile_id, payload):
        return None

    monkeypatch.setattr(api, "update_brand_voice_profile", update_profile)
    client = _client()

    invalid = client.put(
        "/content-ops/brand-voice-profiles/not-a-uuid",
        json={"name": "Acme v2", "descriptors": ["direct"]},
    )
    missing = client.put(
        f"/content-ops/brand-voice-profiles/{PROFILE_ID}",
        json={"name": "Acme v2", "descriptors": ["direct"]},
    )

    assert invalid.status_code == 404
    assert missing.status_code == 404


def test_delete_brand_voice_profile_is_account_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def archive(pool, *, account_id, profile_id):
        calls.append({"account_id": account_id, "profile_id": profile_id})
        return True

    monkeypatch.setattr(api, "archive_brand_voice_profile", archive)
    client = _client()

    response = client.delete(f"/content-ops/brand-voice-profiles/{PROFILE_ID}")

    assert response.status_code == 204
    assert calls == [{"account_id": ACCOUNT_ID, "profile_id": PROFILE_ID}]


def test_brand_voice_profile_routes_fail_when_database_unavailable() -> None:
    class _UninitializedPool:
        is_initialized = False

    client = _client(pool=_UninitializedPool())

    response = client.get("/content-ops/brand-voice-profiles")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database not ready"


def test_brand_voice_profile_api_import_does_not_require_asyncpg() -> None:
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

path = Path('atlas_brain/api/content_ops_brand_voice_profiles.py').resolve()
spec = importlib.util.spec_from_file_location(
    'atlas_brain.api.content_ops_brand_voice_profiles_block_asyncpg_test',
    path,
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(module.BrandVoiceProfileView.__name__)
""")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "BrandVoiceProfileView" in result.stdout
