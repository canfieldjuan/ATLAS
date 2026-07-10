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
