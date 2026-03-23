from atlas_brain.auth.dependencies import _effective_is_admin


def test_effective_is_admin_true_from_flag():
    assert _effective_is_admin("member", True) is True


def test_effective_is_admin_true_from_owner_role():
    assert _effective_is_admin("owner", False) is True


def test_effective_is_admin_true_from_admin_role():
    assert _effective_is_admin("admin", False) is True


def test_effective_is_admin_false_for_member_without_flag():
    assert _effective_is_admin("member", False) is False
