from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_deployed_config_probing",
    Path(__file__).resolve().parent.parent / "scripts" / "check_deployed_config_probing.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

ENV_FALLBACK = "+timeout = os.getenv('ATLAS_TIMEOUT_SECONDS', '30')\n"
PYTHON_OR_FALLBACK = '+workload = os.getenv("EXTRACTED_CAMPAIGN_LLM_WORKLOAD") or "draft"\n'
REMOVED_ENV_FALLBACK = "-timeout = os.getenv('ATLAS_TIMEOUT_SECONDS', '30')\n+timeout = settings.timeout\n"
TS_ENV_FALLBACK = "+const timeout = process.env.ATLAS_TIMEOUT_SECONDS || '30'\n"
SHELL_ENV_FALLBACK = "+: ${ATLAS_TIMEOUT_SECONDS:=30}\n+echo \"$ATLAS_TIMEOUT_SECONDS\"\n"
GUARD_CHANGE = "+def validate_context(payload):\n+    return payload.get('business_context_id') is not None\n"
CLASS_CONTEXT_BOUNDARY_CHANGE = "@@ -4,3 +4,3 @@ class AdmissionGate:\n+        return bool(value)\n"
PLAN_WITH_PROBES = """
### Deployed-config probing

- Deployed/default config values: ATLAS_TIMEOUT_SECONDS=30 from render.yaml.
- Explicit value probe: ATLAS_TIMEOUT_SECONDS=10 passes.
- Absent value probe: unset env uses documented default.
- Default-session/default-context probe: missing session context rejects.
- Side-effect ordering: no write occurs before validate_context passes.
"""
SCAFFOLD_PLACEHOLDER = """
### Deployed-config probing

- Deployed/default config values: TODO/N/A.
- Explicit value probe: TODO/N/A.
- Absent value probe: TODO/N/A.
- Default-session/default-context probe: TODO/N/A.
- Side-effect ordering: TODO/N/A.
"""


def test_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/config.py": ENV_FALLBACK}, [])
    assert len(findings) == 1
    assert mod.RULE in findings[0].reason


def test_python_or_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/runtime_config.py": PYTHON_OR_FALLBACK}, [])
    assert len(findings) == 1


def test_removed_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/config.py": REMOVED_ENV_FALLBACK}, [])
    assert len(findings) == 1


def test_typescript_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"atlas-churn-ui/src/runtime.ts": TS_ENV_FALLBACK}, [])
    assert len(findings) == 1


def test_shell_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"scripts/render_env.sh": SHELL_ENV_FALLBACK}, [])
    assert len(findings) == 1


def test_guard_change_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/mcp/crm_server.py": GUARD_CHANGE}, [])
    assert len(findings) == 1
    assert findings[0].path == "atlas_brain/mcp/crm_server.py"


def test_boundary_class_context_without_function_name_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/services/admission.py": CLASS_CONTEXT_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_guard_change_with_plan_probe_is_clean() -> None:
    findings = mod.scan_diff(
        {"atlas_brain/mcp/crm_server.py": GUARD_CHANGE},
        [PLAN_WITH_PROBES],
    )
    assert findings == []


def test_non_guard_non_config_change_is_clean() -> None:
    findings = mod.scan_diff({"atlas_brain/api/health.py": "+def ping():\n+    return {'ok': True}\n"}, [])
    assert findings == []


def test_substring_collision_path_is_clean() -> None:
    findings = mod.scan_diff({"atlas_brain/services/b2b/enrichment_buyer_authority.py": "+VALUE = 1\n"}, [])
    assert findings == []


def test_process_detector_change_is_clean() -> None:
    findings = mod.scan_diff({"scripts/check_deployed_config_probing.py": GUARD_CHANGE}, [])
    assert findings == []


def test_plan_requires_all_probe_markers() -> None:
    incomplete = "### Deployed-config probing\n- Explicit value probe: done."
    assert not mod.plan_has_deployed_config_probing(incomplete)
    assert not mod.plan_has_deployed_config_probing(SCAFFOLD_PLACEHOLDER)
    assert mod.plan_has_deployed_config_probing(PLAN_WITH_PROBES)


def test_cli_entrypoint_warns_advisory_and_fails_strict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        mod,
        "changed_lines",
        lambda base: {"atlas_brain/config.py": ENV_FALLBACK},
    )
    monkeypatch.setattr(mod, "changed_plan_texts", lambda base: [])

    assert mod.main(["--base", "ignored"]) == 0
    out = capsys.readouterr().out
    assert "::warning file=atlas_brain/config.py::" in out
    assert mod.RULE in out

    assert mod.main(["--base", "ignored", "--strict"]) == 1


def test_git_failure_raises_system_exit() -> None:
    with pytest.raises(SystemExit, match="git .* failed"):
        mod._git(["rev-parse", "--verify", "definitely-not-a-ref-xyz"])
