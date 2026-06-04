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
    sys.modules[spec.name] = module
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


class _FakeSampleUrlResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self._body = body
        self.status = status
        self.content_type = content_type
        self.closed = False

    def read(self, size: int) -> bytes:
        return self._body[:size]

    def getheader(self, name: str, default: str = "") -> str:
        if name.lower() == "content-type":
            return self.content_type
        return default

    def close(self) -> None:
        self.closed = True


def _allow_public_sample_url_dns(
    monkeypatch: pytest.MonkeyPatch,
    *,
    address: str = "93.184.216.34",
) -> None:
    monkeypatch.setattr(
        api.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (api.socket.AF_INET, api.socket.SOCK_STREAM, 6, "", (address, 443)),
        ],
    )


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


def test_sample_url_fetch_returns_readable_text_and_requires_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_public_sample_url_dns(monkeypatch)
    responses: list[_FakeSampleUrlResponse] = []
    calls: list[dict] = []

    def open_request(target, *, timeout):
        calls.append({"target": target, "timeout": timeout})
        response = _FakeSampleUrlResponse(
            b"""
            <html>
              <head><title>Acme About</title><style>.x{}</style></head>
              <body>
                <h1>Launch secure workflows faster.</h1>
                <script>doNotInclude()</script>
                <p>Your team gets clean automation without waiting.</p>
              </body>
            </html>
            """,
        )
        responses.append(response)
        return response

    monkeypatch.setattr(api, "_open_https_sample_url_request", open_request)
    client = _client()

    response = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://example.test/about?ref=atlas"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == "https://example.test/about?ref=atlas"
    assert body["title"] == "Acme About"
    assert body["text"] == (
        "Launch secure workflows faster. "
        "Your team gets clean automation without waiting."
    )
    assert body["source_character_count"] == len(body["text"])
    assert calls[0]["target"].host == "example.test"
    assert calls[0]["target"].connect_host == "93.184.216.34"
    assert calls[0]["target"].path == "/about?ref=atlas"
    assert responses[0].closed is True

    member_response = _client(user=_user(role="member")).post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://example.test/about"},
    )
    assert member_response.status_code == 403
    assert member_response.json()["detail"] == "Admin access required"


def test_sample_url_rejects_private_dns_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (api.socket.AF_INET, api.socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )

    def open_request(target, *, timeout):
        raise AssertionError("private DNS target must be rejected before fetch")

    monkeypatch.setattr(api, "_open_https_sample_url_request", open_request)
    client = _client()

    response = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://example.test/about"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Sample URL host is not allowed"


def test_sample_url_rejects_shared_address_space_dns_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (api.socket.AF_INET, api.socket.SOCK_STREAM, 6, "", ("100.64.0.1", 443)),
        ],
    )

    def open_request(target, *, timeout):
        raise AssertionError("shared-address-space target must be rejected first")

    monkeypatch.setattr(api, "_open_https_sample_url_request", open_request)
    client = _client()

    response = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://example.test/about"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Sample URL host is not allowed"


def test_sample_url_rejects_redirect_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_public_sample_url_dns(monkeypatch)

    def open_request(target, *, timeout):
        return _FakeSampleUrlResponse(b"", status=302)

    monkeypatch.setattr(api, "_open_https_sample_url_request", open_request)
    client = _client()

    response = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://example.test/about"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Sample URL redirects are not allowed."


def test_sample_url_rejects_non_success_response_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_public_sample_url_dns(monkeypatch)

    def open_request(target, *, timeout):
        return _FakeSampleUrlResponse(b"", status=500)

    monkeypatch.setattr(api, "_open_https_sample_url_request", open_request)
    client = _client()

    response = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://example.test/about"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Sample URL could not be fetched."


def test_sample_url_rejects_non_https_credentials_and_literal_private_hosts() -> None:
    client = _client()

    non_https = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "http://example.test/about"},
    )
    file_scheme = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "file:///etc/passwd"},
    )
    credentials = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://user:secret@example.test/about"},
    )
    metadata_ip = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://169.254.169.254/latest/meta-data"},
    )
    shared_space = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://100.64.0.1/about"},
    )

    assert non_https.status_code == 400
    assert non_https.json()["detail"] == "Sample URL must be an https URL"
    assert file_scheme.status_code == 400
    assert file_scheme.json()["detail"] == "Sample URL must be an https URL"
    assert credentials.status_code == 400
    assert credentials.json()["detail"] == "Sample URL must not include credentials"
    assert metadata_ip.status_code == 400
    assert metadata_ip.json()["detail"] == "Sample URL host is not allowed"
    assert shared_space.status_code == 400
    assert shared_space.json()["detail"] == "Sample URL host is not allowed"


def test_sample_url_rejects_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_public_sample_url_dns(monkeypatch)
    body = b"a" * (api._SAMPLE_URL_MAX_BYTES + 1)

    def open_request(target, *, timeout):
        return _FakeSampleUrlResponse(body, content_type="text/plain")

    monkeypatch.setattr(api, "_open_https_sample_url_request", open_request)
    client = _client()

    response = client.post(
        "/content-ops/brand-voice-profiles/sample-url",
        json={"url": "https://example.test/about"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == {
        "reason": "brand_voice_sample_url_too_large",
        "max_bytes": api._SAMPLE_URL_MAX_BYTES,
    }


def test_sample_url_plain_text_extraction_falls_back_without_html() -> None:
    title, text = api._extract_readable_text(
        b"  First line.\n\nSecond\tline.  ",
        content_type="text/plain",
    )

    assert title is None
    assert text == "First line. Second line."


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
sys.modules[spec.name] = module
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
