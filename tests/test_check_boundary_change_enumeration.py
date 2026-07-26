from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_boundary_change_enumeration",
    Path(__file__).resolve().parent.parent / "scripts" / "check_boundary_change_enumeration.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

BOUNDARY_CHANGE = (
    "+def resolve_contact_identity(payload):\n"
    "+    return payload.get('email') or payload.get('phone')\n"
)
TS_BOUNDARY_CHANGE = (
    "+export function validateClaimGate(input) {\n"
    "+  return Boolean(input.accountId)\n"
    "+}\n"
)
PATH_ONLY_BOUNDARY_CHANGE = "+VALUE = 1\n"
CONST_BOUNDARY_CHANGE = "+const resolveClaimGate = (input) => Boolean(input.accountId)\n"
CLASS_BOUNDARY_CHANGE = "+class TenantResolver {\n+  resolve(input) { return input.tenantId }\n+}\n"
PLAN_WITH_ENUMERATION = """
### Boundary-change enumeration

- Replaced-path behaviors: preserved existing email-first lookup.
- Guard-relevant fields: email, phone, source_ref.
- Caller x input shape: intake x email-only -> preserved.
"""
SCAFFOLD_PLACEHOLDER = """
### Boundary-change enumeration

- Replaced-path behaviors: TODO/N/A.
- Guard-relevant fields: TODO/N/A.
- Caller x input shape: TODO/N/A.
"""


def test_boundary_change_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/services/crm_provider.py": BOUNDARY_CHANGE}, [])
    assert len(findings) == 1
    assert findings[0].path == "atlas_brain/services/crm_provider.py"
    assert mod.RULE in findings[0].reason


def test_typescript_boundary_change_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas-churn-ui/src/components/ProductClaimGate.tsx": TS_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1
    assert findings[0].path == "atlas-churn-ui/src/components/ProductClaimGate.tsx"


def test_boundary_path_signal_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/services/admission_gate.py": PATH_ONLY_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_const_boundary_signal_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas-churn-ui/src/components/ClaimPanel.tsx": CONST_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_class_boundary_signal_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"portfolio-ui/src/auth.ts": CLASS_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_boundary_change_with_plan_enumeration_is_clean() -> None:
    findings = mod.scan_diff(
        {"atlas_brain/services/crm_provider.py": BOUNDARY_CHANGE},
        [PLAN_WITH_ENUMERATION],
    )
    assert findings == []


def test_non_boundary_change_is_clean() -> None:
    findings = mod.scan_diff({"atlas_brain/api/health.py": "+def ping():\n+    return {'ok': True}\n"}, [])
    assert findings == []


def test_process_detector_change_is_clean_with_na_plan() -> None:
    findings = mod.scan_diff({"scripts/check_boundary_change_enumeration.py": BOUNDARY_CHANGE}, [])
    assert findings == []


def test_other_checker_boundary_change_is_not_exempt() -> None:
    findings = mod.scan_diff({"scripts/check_guard_class_closure.py": BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_test_file_boundary_words_are_ignored() -> None:
    findings = mod.scan_diff({"tests/test_contact_resolver.py": BOUNDARY_CHANGE}, [])
    assert findings == []


def test_plan_requires_all_enumeration_markers() -> None:
    incomplete = "### Boundary-change enumeration\n- Replaced-path behaviors: preserved."
    narrative = (
        "## Why this slice exists\n"
        "replaced-path behaviors documented; guard-relevant fields documented; "
        "caller x input shape documented.\n"
        + SCAFFOLD_PLACEHOLDER
    )
    assert not mod.plan_has_boundary_enumeration(incomplete)
    assert not mod.plan_has_boundary_enumeration(SCAFFOLD_PLACEHOLDER)
    assert not mod.plan_has_boundary_enumeration(narrative)
    assert mod.plan_has_boundary_enumeration(PLAN_WITH_ENUMERATION)


def test_cli_entrypoint_warns_advisory_and_fails_strict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        mod,
        "changed_added_lines",
        lambda base: {"atlas_brain/services/crm_provider.py": BOUNDARY_CHANGE},
    )
    monkeypatch.setattr(mod, "changed_plan_texts", lambda base: [])

    assert mod.main(["--base", "ignored"]) == 0
    out = capsys.readouterr().out
    assert "::warning file=atlas_brain/services/crm_provider.py::" in out
    assert mod.RULE in out

    assert mod.main(["--base", "ignored", "--strict"]) == 1


def test_git_failure_raises_system_exit() -> None:
    with pytest.raises(SystemExit, match="git .* failed"):
        mod._git(["rev-parse", "--verify", "definitely-not-a-ref-xyz"])
