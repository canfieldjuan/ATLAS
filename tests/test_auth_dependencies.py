import pytest

from atlas_brain.auth.dependencies import (
    AuthUser,
    B2B_PLAN_ORDER,
    PLAN_ORDER,
    _effective_is_admin,
    require_b2b_plan,
    require_plan,
)


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


def test_require_plan_accepts_known_consumer_tiers():
    for plan in PLAN_ORDER:
        dependency = require_plan(plan)
        assert callable(dependency)
