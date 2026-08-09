"""Tests for the guard class-closure advisory lint.

Exercises the pure detection core (scan_diff / file_is_guard_shaped /
diff_has_property_test) with synthetic diffs -- both directions per AGENTS 3i:
a guard-shaped change without a property test is flagged; the same change with
a property test, a non-guard change, and the near-miss single-signal cases are
clean.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_guard_class_closure",
    Path(__file__).resolve().parent.parent / "scripts" / "check_guard_class_closure.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod  # dataclass resolution needs the module registered
_SPEC.loader.exec_module(mod)


# --- guard-shape detection ---------------------------------------------------


def test_path_name_stem_marks_guard() -> None:
    assert mod.file_is_guard_shaped("extracted_content_pipeline/support_ticket_privacy.py", "+x = 1")
    assert mod.file_is_guard_shaped("atlas_brain/services/message_sanitizer.py", "+pass")
    assert mod.file_is_guard_shaped("scripts/redact_pii.py", "+pass")


def test_content_signals_both_required() -> None:
    verdict_and_open = (
        "+def _marker_is_private(value):\n"
        "+    if isinstance(value, (str, dict)):\n"
        "+        return value.strip().lower() in _DENY\n"
    )
    assert mod.file_is_guard_shaped("atlas_brain/services/foo.py", verdict_and_open)

    # verdict only, no open-input inspection -> not guard-shaped
    verdict_only = "+def is_ready(count):\n+    return count > 0\n"
    assert not mod.file_is_guard_shaped("atlas_brain/services/foo.py", verdict_only)
    assert mod.file_is_guard_shaped("atlas_brain/services/foo.py", verdict_only, strict=True)

    # open-input only, no verdict def -> not guard-shaped
    open_only = "+    if isinstance(value, dict):\n+        data = frozenset({'a', 'b'})\n"
    assert not mod.file_is_guard_shaped("atlas_brain/services/foo.py", open_only)


def test_non_python_and_test_files_are_never_guard_shaped() -> None:
    guardish = "+def is_private(v):\n+    return isinstance(v, str)\n"
    assert not mod.file_is_guard_shaped("docs/privacy.md", guardish)
    assert not mod.file_is_guard_shaped("tests/test_privacy_guard.py", guardish)
    assert not mod.file_is_guard_shaped("atlas_brain/privacy_config.json", guardish)


# --- property-test detection -------------------------------------------------


@pytest.mark.parametrize(
    "added",
    [
        "+@pytest.mark.parametrize('v', VALUES)\n+def test_x(v): ...\n",
        "+    for key, val in itertools.product(KEYS, VALUES):\n",
        "+from hypothesis import given\n+@given(st.text())\n",
        "+    for wrapped in product(containers, values):\n",
    ],
)
def test_property_test_signals_recognized(added: str) -> None:
    assert mod.diff_has_property_test({"tests/test_x.py": added})


def test_strict_property_test_rejects_bare_parametrize_fixture() -> None:
    weak = "+@pytest.mark.parametrize('v', ['kept private'])\n+def test_x(v): ...\n"
    strong = (
        "+    # grammar axes: tokens x containers x key families\n"
        "+    contract_oracle = {'private': False, 'public': True}\n"
        "+    for key, val in itertools.product(KEY_FAMILIES, TOKEN_VALUES):\n"
        "+        assert guard(key, [val]) == contract_oracle[val]\n"
    )

    assert mod.diff_has_property_test({"tests/test_x.py": weak}) is True
    assert mod.diff_has_property_test({"tests/test_x.py": weak}, strict=True) is False
    assert mod.diff_has_property_test({"tests/test_x.py": strong}, strict=True) is True


def test_strict_property_test_rejects_product_without_oracle_axes() -> None:
    fixture_matrix = (
        "+def test_support_ticket_privacy_matrix():\n"
        "+    for value, flag in product(['kept private'], [0]):\n"
        "+        assert support_ticket_privacy_guard(value) == flag\n"
    )
    unused_hypothesis = "+from hypothesis import given\n+def test_support_ticket_privacy_literal(): ...\n"

    assert mod.diff_has_property_test({"tests/test_x.py": fixture_matrix}, strict=True) is False
    assert mod.diff_has_property_test({"tests/test_x.py": unused_hypothesis}, strict=True) is False


def test_strict_property_test_rejects_string_scoped_product_costume() -> None:
    product_costume = (
        "+tokens = ['kept private']\n"
        "+containers = [False]\n"
        "+expected = False\n"
        "+def test_support_ticket_privacy_matrix():\n"
        "+    for token, container in product(tokens, containers):\n"
        "+        assert support_ticket_privacy_guard(token) == expected\n"
    )

    assert mod.diff_has_property_test({"tests/test_x.py": product_costume}, strict=True) is False


def test_plain_fixture_list_is_not_a_property_test() -> None:
    fixture_only = (
        "+def test_kept_private_rejects():\n"
        "+    assert guard('kept private') is True\n"
        "+def test_no_longer_public_rejects():\n"
        "+    assert guard('no longer public') is True\n"
    )
    assert not mod.diff_has_property_test({"tests/test_x.py": fixture_only})


# --- scan_diff end to end ----------------------------------------------------


GUARD_CHANGE = (
    "+def _marker_is_private(value):\n"
    "+    if isinstance(value, (str, dict)):\n"
    "+        return value.strip().lower() in _DENY\n"
)
PROPERTY_TEST = "+@pytest.mark.parametrize('v', VALUES)\n+def test_matrix(v): ...\n"
FIXTURE_TEST = "+def test_one():\n+    assert guard('kept private') is True\n"


def test_guard_change_without_property_test_is_flagged() -> None:
    findings = mod.scan_diff(
        {"extracted_content_pipeline/support_ticket_privacy.py": GUARD_CHANGE}
    )
    assert len(findings) == 1
    assert findings[0].path == "extracted_content_pipeline/support_ticket_privacy.py"


def test_guard_change_with_property_test_is_clean() -> None:
    findings = mod.scan_diff(
        {
            "extracted_content_pipeline/support_ticket_privacy.py": GUARD_CHANGE,
            "tests/test_support_ticket_privacy_sweep.py": PROPERTY_TEST,
        }
    )
    assert findings == []


def test_guard_change_with_weak_parametrize_stays_flagged_in_strict_mode() -> None:
    findings = mod.scan_diff(
        {
            "extracted_content_pipeline/support_ticket_privacy.py": GUARD_CHANGE,
            "tests/test_support_ticket_privacy_sweep.py": PROPERTY_TEST,
        },
        strict=True,
    )
    assert len(findings) == 1


def test_guard_change_with_generative_product_is_clean_in_strict_mode() -> None:
    product_test = (
        "+import itertools\n"
        "+# grammar axes: tokens x containers x key families\n"
        "+SPEC_ORACLE = {'kept private': False, 'published': True}\n"
        "+def test_support_ticket_privacy_matrix():\n"
        "+    for key, value in itertools.product(KEY_FAMILIES, TOKEN_VALUES):\n"
        "+        assert support_ticket_privacy_guard(key, [value]) == SPEC_ORACLE[value]\n"
    )
    findings = mod.scan_diff(
        {
            "extracted_content_pipeline/support_ticket_privacy.py": GUARD_CHANGE,
            "tests/test_support_ticket_privacy_sweep.py": product_test,
        },
        strict=True,
    )
    assert findings == []


def test_guard_change_with_only_a_fixture_list_is_flagged() -> None:
    findings = mod.scan_diff(
        {
            "extracted_content_pipeline/support_ticket_privacy.py": GUARD_CHANGE,
            "tests/test_support_ticket_privacy_sweep.py": FIXTURE_TEST,
        }
    )
    assert len(findings) == 1


def test_non_guard_change_is_clean() -> None:
    findings = mod.scan_diff(
        {"atlas_brain/api/health.py": "+def ping():\n+    return {'ok': True}\n"}
    )
    assert findings == []


def test_ignore_globs_opt_out() -> None:
    findings = mod.scan_diff(
        {"extracted_content_pipeline/support_ticket_privacy.py": GUARD_CHANGE},
        ignore_globs=["extracted_content_pipeline/*_privacy.py"],
    )
    assert findings == []


# --- failure branches (raise paths) ------------------------------------------


def test_git_failure_raises_system_exit() -> None:
    with pytest.raises(SystemExit, match="git .* failed"):
        mod._git(["rev-parse", "--verify", "definitely-not-a-ref-xyz"])


def test_bad_ignore_globs_config_raises_system_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "guard_class_closure_ignore.json"
    bad.write_text('{"ignore_globs": "not-a-list"}', encoding="utf-8")
    monkeypatch.setattr(mod, "CONFIG_PATH", bad)
    with pytest.raises(SystemExit, match="ignore_globs must be a list"):
        mod.load_ignore_globs()


def test_bare_fixture_loop_is_not_a_property_test() -> None:
    # Trailing and mid-diff plain loops must both be rejected (the old regex
    # was order-dependent: a trailing bare loop falsely counted).
    trailing = "+def test_cases():\n+    for case in FIXTURE_CASES:"
    mid = "+for case in CASES:\n+    assert guard(case)\n+\n+def other(): ...\n"
    assert mod.diff_has_property_test({"tests/t.py": trailing}) is False
    assert mod.diff_has_property_test({"tests/t.py": mid}) is False


def test_generative_product_loop_is_a_property_test() -> None:
    gen = "+import itertools\n+for k, v in itertools.product(KEYS, VALUES):\n+    assert guard({k: v})\n"
    assert mod.diff_has_property_test({"tests/t.py": gen}) is True


def test_unrelated_property_test_does_not_suppress_guard_finding() -> None:
    findings = mod.scan_diff({
        "pkg/privacy_guard.py": "+def is_private(v):\n+    return _verdict(v)\n",
        "tests/test_other_helper.py": "+from hypothesis import given\n+@given(st.text())\n+def test_helper(s): ...\n",
    })
    assert [f.path for f in findings] == ["pkg/privacy_guard.py"]


def test_stem_tied_property_test_suppresses_guard_finding() -> None:
    findings = mod.scan_diff({
        "pkg/privacy_guard.py": "+def is_private(v):\n+    return _verdict(v)\n",
        "tests/test_privacy_guard.py": (
            "+from hypothesis import given\n"
            "+@given(token=st.text(), key=st.sampled_from(KEY_FAMILIES))\n"
            "+def test_privacy_guard_closed(token, key):\n"
            "+    expected = contract_oracle(token)\n"
            "+    assert privacy_guard(key, [token]) == expected\n"
        ),
    })
    assert findings == []


def test_strict_mode_requires_oracle_axes_for_guard_scan() -> None:
    findings = mod.scan_diff(
        {
            "pkg/privacy_guard.py": "+def is_private(v):\n+    return _verdict(v)\n",
            "tests/test_privacy_guard.py": (
                "+def test_privacy_guard_cases():\n"
                "+    for token, flag in product(['kept private'], [False]):\n"
                "+        assert privacy_guard(token) == flag\n"
            ),
        },
        strict=True,
    )
    assert [f.path for f in findings] == ["pkg/privacy_guard.py"]


def test_strict_mode_flags_existing_guard_body_edit_without_open_input_token() -> None:
    findings = mod.scan_diff(
        {
            "pkg/rules.py": (
                "@@ -12,0 +13,1 @@ def validate_payload(value):\n"
                "+    return value == 'kept private'\n"
            )
        },
        strict=True,
    )

    assert [f.path for f in findings] == ["pkg/rules.py"]


def test_ignore_config_stays_on_trusted_root_when_inspecting_pr_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = tmp_path / "trusted"
    inspected = tmp_path / "pr"
    (trusted / "scripts").mkdir(parents=True)
    (inspected / "scripts").mkdir(parents=True)
    (trusted / "scripts" / "guard_class_closure_config.json").write_text(
        '{"ignore_globs": ["trusted_only.py"]}',
        encoding="utf-8",
    )
    (inspected / "scripts" / "guard_class_closure_config.json").write_text(
        '{"ignore_globs": ["**"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "TRUSTED_REPO_ROOT", trusted)
    monkeypatch.setattr(mod, "INSPECTED_REPO_ROOT", inspected)
    monkeypatch.setattr(
        mod,
        "CONFIG_PATH",
        mod.TRUSTED_REPO_ROOT / "scripts" / "guard_class_closure_config.json",
    )

    assert mod.load_ignore_globs() == ["trusted_only.py"]


def test_async_guard_def_is_detected() -> None:
    added = "+async def classify_payload(value):\n+    if isinstance(value, str):\n+        return False\n"
    assert mod.file_is_guard_shaped("pkg/handlers.py", added) is True


def test_body_edit_inside_existing_guard_is_detected_via_hunk_header() -> None:
    # -U0 headers name the enclosing function even when no new def is added.
    added = "@@ -40,0 +41,2 @@ def is_private(value):\n+    if isinstance(value, str):\n+        return value in _DENY\n"
    assert mod.file_is_guard_shaped("pkg/rules.py", added) is True


def test_hunk_header_of_non_guard_function_is_not_a_verdict() -> None:
    added = "@@ -10,0 +11,1 @@ def render_report(rows):\n+    total = sum(rows)\n"
    assert mod.file_is_guard_shaped("pkg/report.py", added) is False
