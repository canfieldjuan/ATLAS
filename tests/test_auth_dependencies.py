import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from atlas_brain.auth.dependencies import (
    AuthUser,
    B2B_PLAN_ORDER,
    CONTENT_OPS_API_KEY_SCOPES,
    PLAN_ORDER,
    _effective_is_admin,
    require_b2b_plan,
    require_b2b_plan_or_api_key,
    require_plan,
)


class _FakeRequest:
    def __init__(self, token: str):
        self.headers = {"authorization": f"Bearer {token}"}
        self.client = SimpleNamespace(host="127.0.0.1")


class _FakePool:
    def __init__(self, account_row: dict[str, object]):
        self.account_row = account_row
        self.touched: tuple[uuid.UUID, str | None] | None = None

    async def fetchrow(self, *_args, **_kwargs):
        return self.account_row


def test_effective_is_admin_true_from_flag():
    assert _effective_is_admin("member", True) is True


def test_effective_is_admin_true_from_owner_role():
    assert _effective_is_admin("owner", False) is True


def test_effective_is_admin_true_from_admin_role():
    assert _effective_is_admin("admin", False) is True


def test_effective_is_admin_false_for_member_without_flag():
    assert _effective_is_admin("member", False) is False


def test_auth_user_keeps_platform_admin_separate_from_account_admin():
    user = AuthUser(
        user_id="user-1",
        account_id="account-1",
        plan="b2b_growth",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
        is_platform_admin=False,
    )

    assert user.is_admin is True
    assert user.is_platform_admin is False


def test_require_plan_rejects_unknown_consumer_tier():
    with pytest.raises(ValueError, match="Invalid consumer plan tier 'bogus'"):
        require_plan("bogus")


def test_require_b2b_plan_rejects_unknown_b2b_tier():
    with pytest.raises(ValueError, match="Invalid B2B plan tier 'starter'"):
        require_b2b_plan("starter")


def test_require_b2b_plan_accepts_known_b2b_tiers():
    for plan in B2B_PLAN_ORDER:
        dependency = require_b2b_plan(plan)
        assert callable(dependency)


def test_require_b2b_plan_or_api_key_rejects_unknown_b2b_tier():
    with pytest.raises(ValueError, match="Invalid B2B plan tier 'starter'"):
        require_b2b_plan_or_api_key("starter")


def test_require_b2b_plan_or_api_key_accepts_known_b2b_tiers():
    for plan in B2B_PLAN_ORDER:
        dependency = require_b2b_plan_or_api_key(plan)
        assert callable(dependency)


def test_require_plan_accepts_known_consumer_tiers():
    for plan in PLAN_ORDER:
        dependency = require_plan(plan)
        assert callable(dependency)


def _b2b_user(**overrides) -> AuthUser:
    defaults = {
        "user_id": "user-1",
        "account_id": "account-1",
        "plan": "b2b_growth",
        "plan_status": "active",
        "role": "member",
        "product": "b2b_retention",
    }
    defaults.update(overrides)
    return AuthUser(**defaults)


async def _api_key_user_from_dispatch(monkeypatch, *, scopes: tuple[str, ...]) -> tuple[AuthUser, _FakePool]:
    import atlas_brain.auth.api_keys as api_keys_mod
    import atlas_brain.auth.dependencies as dependencies_mod
    import atlas_brain.storage.database as database_mod

    raw_key = "atls_live_abcdefghijklmnopqrstuvwxyz234567"
    key_id = uuid.uuid4()
    account_id = uuid.uuid4()
    user_id = uuid.uuid4()
    pool = _FakePool(
        {
            "plan": "b2b_growth",
            "plan_status": "active",
            "product": "b2b_retention",
            "trial_ends_at": None,
        }
    )

    async def fake_lookup_api_key(_pool, presented_key):
        assert _pool is pool
        assert presented_key == raw_key
        return {
            "id": key_id,
            "account_id": account_id,
            "user_id": user_id,
            "scopes": list(scopes),
        }

    async def fake_touch_api_key(_pool, *, key_id: uuid.UUID, client_ip: str | None):
        assert _pool is pool
        pool.touched = (key_id, client_ip)

    monkeypatch.setattr(
        dependencies_mod,
        "settings",
        SimpleNamespace(saas_auth=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(database_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(api_keys_mod, "lookup_api_key", fake_lookup_api_key)
    monkeypatch.setattr(api_keys_mod, "touch_api_key", fake_touch_api_key)

    user = await dependencies_mod.require_auth_or_api_key(_FakeRequest(raw_key))
    return user, pool


@pytest.mark.asyncio
async def test_require_b2b_plan_or_api_key_accepts_jwt_dashboard_user_without_scopes():
    dependency = require_b2b_plan_or_api_key("b2b_growth")
    user = _b2b_user(auth_method="jwt", api_key_scopes=())

    assert await dependency(user) is user


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", sorted(CONTENT_OPS_API_KEY_SCOPES))
async def test_require_b2b_plan_or_api_key_accepts_content_ops_api_key_scope(scope):
    dependency = require_b2b_plan_or_api_key("b2b_growth")
    user = _b2b_user(auth_method="api_key", api_key_scopes=(scope,))

    assert await dependency(user) is user


@pytest.mark.asyncio
@pytest.mark.parametrize("scopes", [(), ("*",), ("llm:*",), ("content_ops:macro:*",)])
async def test_require_b2b_plan_or_api_key_rejects_api_key_without_content_ops_scope(scopes):
    dependency = require_b2b_plan_or_api_key("b2b_growth")
    user = _b2b_user(auth_method="api_key", api_key_scopes=scopes)

    with pytest.raises(HTTPException) as exc:
        await dependency(user)

    assert exc.value.status_code == 403
    assert exc.value.detail == "Content Ops API key scope required"


@pytest.mark.asyncio
async def test_require_b2b_plan_or_api_key_rejects_api_key_scopes_from_dispatch(monkeypatch):
    user, pool = await _api_key_user_from_dispatch(monkeypatch, scopes=("llm:*",))
    dependency = require_b2b_plan_or_api_key("b2b_growth")

    assert user.auth_method == "api_key"
    assert user.api_key_scopes == ("llm:*",)
    assert pool.touched is not None
    with pytest.raises(HTTPException) as exc:
        await dependency(user)

    assert exc.value.status_code == 403
    assert exc.value.detail == "Content Ops API key scope required"


@pytest.mark.asyncio
async def test_require_b2b_plan_or_api_key_accepts_content_ops_scope_from_dispatch(monkeypatch):
    user, pool = await _api_key_user_from_dispatch(monkeypatch, scopes=("content_ops:*",))
    dependency = require_b2b_plan_or_api_key("b2b_growth")

    assert user.auth_method == "api_key"
    assert user.api_key_scopes == ("content_ops:*",)
    assert pool.touched is not None
    assert await dependency(user) is user


@pytest.mark.asyncio
async def test_require_b2b_plan_or_api_key_rejects_non_b2b_product():
    dependency = require_b2b_plan_or_api_key("b2b_growth")
    user = _b2b_user(product="consumer")

    with pytest.raises(HTTPException) as exc:
        await dependency(user)

    assert exc.value.status_code == 403
    assert exc.value.detail == "B2B product required"


@pytest.mark.asyncio
async def test_require_b2b_plan_or_api_key_rejects_lower_b2b_plan():
    dependency = require_b2b_plan_or_api_key("b2b_growth")
    user = _b2b_user(plan="b2b_starter")

    with pytest.raises(HTTPException) as exc:
        await dependency(user)

    assert exc.value.status_code == 403
    assert exc.value.detail == (
        "Plan 'b2b_growth' or higher required (current: 'b2b_starter')"
    )


@pytest.mark.asyncio
async def test_require_b2b_plan_or_api_key_rejects_past_due_account():
    dependency = require_b2b_plan_or_api_key("b2b_growth")
    user = _b2b_user(plan_status="past_due")

    with pytest.raises(HTTPException) as exc:
        await dependency(user)

    assert exc.value.status_code == 402
    assert exc.value.detail == "Payment past due"


def test_content_ops_route_mount_uses_b2b_api_key_auth_dependency():
    from pathlib import Path

    source = Path("atlas_brain/api/__init__.py").read_text(encoding="utf-8")

    assert 'Depends(require_b2b_plan_or_api_key("b2b_growth"))' in source
    assert 'Depends(require_b2b_plan("b2b_growth"))' not in source
