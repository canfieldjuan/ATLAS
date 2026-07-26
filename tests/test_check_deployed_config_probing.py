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
YAML_ENV_FALLBACK = "+      POSTGRES_USER: ${ATLAS_DB_USER:-atlas}\n"
DOCKERFILE_ENV_FALLBACK = "+CMD exec uvicorn app:app --port ${PORT:-8000}\n"
GUARD_CHANGE = "+def validate_context(payload):\n+    return payload.get('business_context_id') is not None\n"
CLASS_CONTEXT_BOUNDARY_CHANGE = "@@ -4,3 +4,3 @@ class AdmissionGate:\n+        return bool(value)\n"
BILLING_VALIDATOR_CHANGE = "+def _deflection_checkout_amount_is_valid(amount):\n+    return amount > 0\n"
TWO_ENV_FALLBACKS = (
    "+only_a = os.getenv('ONLY_A', 'one')\n"
    "+unmentioned_b = os.getenv('UNMENTIONED_B', 'two')\n"
)
PLAN_WITH_PROBES = """
### Deployed-config probing

- Deployed/default config values: ATLAS_TIMEOUT_SECONDS=30 from render.yaml.
- Explicit value probe: ATLAS_TIMEOUT_SECONDS=10 passes.
- Absent value probe: ATLAS_TIMEOUT_SECONDS unset uses documented default.
- Default-session/default-context probe: ATLAS_TIMEOUT_SECONDS default session rejects.
- Side-effect ordering: no write occurs before ATLAS_TIMEOUT_SECONDS admission passes.
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


def test_yaml_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"docker-compose.yml": YAML_ENV_FALLBACK}, [])
    assert len(findings) == 1


def test_workflow_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({".github/workflows/content_ops_deflection_report_ttl_purge.yml": YAML_ENV_FALLBACK}, [])
    assert len(findings) == 1


def test_dockerfile_env_fallback_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"Dockerfile.graphiti": DOCKERFILE_ENV_FALLBACK}, [])
    assert len(findings) == 1


def test_guard_change_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/mcp/crm_server.py": GUARD_CHANGE}, [])
    assert len(findings) == 1
    assert findings[0].path == "atlas_brain/mcp/crm_server.py"


def test_boundary_class_context_without_function_name_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/services/admission.py": CLASS_CONTEXT_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_is_valid_boundary_change_without_plan_probe_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/api/billing.py": BILLING_VALIDATOR_CHANGE}, [])
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


def test_plan_rejects_unresolved_probe_dispositions() -> None:
    unresolved = """
### Deployed-config probing

- Deployed/default config values: unknown.
- Explicit value probe: pending before push.
- Absent value probe: skipped.
- Default-session/default-context probe: not verified.
- Side-effect ordering: TBD.
"""
    assert not mod.plan_has_deployed_config_probing(unresolved)


def test_plan_rejects_negative_probe_outcomes() -> None:
    negative = """
### Deployed-config probing

- Deployed/default config values: ATLAS_TIMEOUT_SECONDS=30 from render.yaml.
- Explicit value probe: ATLAS_TIMEOUT_SECONDS never passes.
- Absent value probe: ATLAS_TIMEOUT_SECONDS failed.
- Default-session/default-context probe: ATLAS_TIMEOUT_SECONDS does not pass.
- Side-effect ordering: write before admission.
"""
    assert not mod.plan_has_deployed_config_probing(negative)


def test_could_not_determine_requires_settling_source() -> None:
    missing_source = PLAN_WITH_PROBES.replace(
        "ATLAS_TIMEOUT_SECONDS=30 from render.yaml",
        "could-not-determine",
    )
    with_source = PLAN_WITH_PROBES.replace(
        "ATLAS_TIMEOUT_SECONDS=30 from render.yaml",
        "could-not-determine; deployment provider source would settle it",
    )
    assert not mod.plan_has_deployed_config_probing(missing_source)
    assert mod.plan_has_deployed_config_probing(with_source)


def test_could_not_determine_is_only_allowed_for_deployed_values() -> None:
    all_indeterminate = """
### Deployed-config probing

- Deployed/default config values: could-not-determine; deployment provider source would settle it.
- Explicit value probe: could-not-determine; deployment provider source would settle it.
- Absent value probe: could-not-determine; deployment provider source would settle it.
- Default-session/default-context probe: could-not-determine; deployment provider source would settle it.
- Side-effect ordering: could-not-determine; deployment provider source would settle it.
"""
    assert not mod.plan_has_deployed_config_probing(all_indeterminate)


def test_plan_must_cover_every_changed_config_key() -> None:
    findings = mod.scan_diff({"atlas_brain/config.py": TWO_ENV_FALLBACKS}, [PLAN_WITH_PROBES])
    assert len(findings) == 1


def test_config_key_coverage_uses_exact_deployed_config_section() -> None:
    plan = PLAN_WITH_PROBES.replace("ATLAS_TIMEOUT_SECONDS", "ONLY_A")
    plan += "\n## Deferred\n- FOOBAR coverage is unrelated.\n"
    findings = mod.scan_diff({"atlas_brain/config.py": "+foo = os.getenv('FOO', 'x')\n"}, [plan])
    assert len(findings) == 1


def test_each_config_key_must_appear_in_each_probe_row() -> None:
    plan = """
### Deployed-config probing

- Deployed/default config values: ONLY_A=1 from render.yaml; UNMENTIONED_B=2 from render.yaml.
- Explicit value probe: ONLY_A=override passes.
- Absent value probe: ONLY_A unset uses default.
- Default-session/default-context probe: ONLY_A default session rejects.
- Side-effect ordering: no write occurs before ONLY_A admission passes; UNMENTIONED_B stray mention.
"""
    findings = mod.scan_diff({"atlas_brain/config.py": TWO_ENV_FALLBACKS}, [plan])
    assert len(findings) == 1


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
