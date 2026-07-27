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


# ---------------------------------------------------------------------------
# Pydantic-Settings idiom (the repository's own config boundary)
#
# Atlas law forbids reading os.environ directly (CLAUDE.md: "Never read
# `os.environ` directly -- add a typed field on the relevant BaseSettings
# subclass"). A detector that recognized only os.getenv could not fire on a
# single compliant Atlas guard, including the one behind the incident that
# motivated this rule, so these cases pin the repository idiom.
# ---------------------------------------------------------------------------

SETTINGS_FALLBACK_INCIDENT = '''
def _default_context() -> "str | None":
    """Deployment-default tenant for read scoping."""
    from ..config import settings

    return settings.mcp.crm_default_business_context or None
'''

ALIASED_SETTINGS_FALLBACK = '''
def resolve_token() -> str:
    return funnel_settings.service_token or ""
'''


def test_settings_fallback_from_the_motivating_incident_is_detected() -> None:
    """#2216: the deployed-default read whose explicit-vs-default divergence
    claimed a legacy row on a rejected operation. Verbatim shape from
    atlas_brain/mcp/crm_server.py."""
    assert mod.CONFIG_FALLBACK_RE.search(SETTINGS_FALLBACK_INCIDENT)
    assert mod.config_keys(SETTINGS_FALLBACK_INCIDENT) == {
        "crm_default_business_context"
    }


def test_aliased_settings_module_fallback_is_detected() -> None:
    """Per-profile settings objects (funnel_settings, invoicing_settings, ...)
    are the same boundary under a different name."""
    assert mod.CONFIG_FALLBACK_RE.search(ALIASED_SETTINGS_FALLBACK)
    assert mod.config_keys(ALIASED_SETTINGS_FALLBACK) == {"service_token"}


def test_getattr_settings_default_is_detected() -> None:
    source = 'value = getattr(settings.mcp, "crm_default_business_context", None)'
    assert mod.CONFIG_FALLBACK_RE.search(source)


def test_non_config_or_default_stays_silent() -> None:
    """The other direction: `or` on an ordinary local is not a config fallback,
    so the detector must not fire on every default-valued expression."""
    source = "def resolve(candidate):\n    return candidate.strip() or fallback\n"
    assert not mod.CONFIG_FALLBACK_RE.search(source)
    assert mod.config_keys(source) == set()


def test_settings_read_without_a_fallback_stays_silent() -> None:
    """A plain settings read decides nothing about unlisted values; only a
    fallback expresses the deployed-vs-default divergence this rule probes."""
    source = "def resolve():\n    return settings.mcp.crm_default_business_context\n"
    assert not mod.CONFIG_FALLBACK_RE.search(source)


def test_guard_touching_settings_fallback_requires_the_plan_section() -> None:
    """End-to-end both directions on the repository idiom: a guard-shaped file
    with a settings fallback and no deployed-config section warns; the same
    diff with a complete section does not."""
    diff = {"atlas_brain/mcp/crm_server.py": SETTINGS_FALLBACK_INCIDENT}
    assert mod.scan_diff(diff, []) != []

    plan = """### Deployed-config probing
- Deployed/default config values: ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT observed as effingham_maids from the deployed profile, source render.eom.yaml.
- Explicit value probe: an explicit business_context_id passes and crm_default_business_context is unused.
- Absent value probe: with crm_default_business_context unset the legacy unscoped read passes.
- Default-session/default-context probe: a session carrying only crm_default_business_context rejects the EOM stage change.
- Side-effect ordering: no write occurs before the crm_default_business_context admission passes.
"""
    assert mod.scan_diff(diff, [plan]) == []


def test_wrapped_probe_disposition_is_read_whole() -> None:
    """A disposition that wraps onto an indented continuation line is one
    value. Reading only the first physical line reported a complete probe as
    missing evidence -- a false warning on correct input, which is the
    expensive direction for an advisory check."""
    section = (
        "- Explicit value probe: an explicit business_context_id\n"
        "  passes and crm_default_business_context is unused.\n"
        "- Absent value probe: unset crm_default_business_context passes.\n"
    )
    value = mod._marker_value(section, "explicit value probe")
    assert value is not None
    assert "crm_default_business_context is unused" in value
    assert mod._is_dispositioned_value(value, marker="explicit value probe")


def test_continuation_folding_does_not_swallow_the_next_marker() -> None:
    """The other side: a following list item ends the value, so one probe
    cannot absorb the next one and mask a missing disposition."""
    section = (
        "- Explicit value probe: explicit id passes.\n"
        "- Absent value probe: TODO.\n"
    )
    assert mod._marker_value(section, "explicit value probe") == "explicit id passes"
    assert not mod._is_dispositioned_value(
        mod._marker_value(section, "absent value probe"), marker="absent value probe"
    )
