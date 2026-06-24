from __future__ import annotations

import ast
import importlib.util
import re
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SECURITY_MD = REPO_ROOT / "SECURITY.md"
INCIDENT_RESPONSE_MD = REPO_ROOT / "docs" / "INCIDENT_RESPONSE.md"
SECURITY_GUARDRAILS_MD = REPO_ROOT / "docs" / "SECURITY_GUARDRAILS.md"
DEPENDABOT_YML = REPO_ROOT / ".github" / "dependabot.yml"
SECURITY_POLICY_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "atlas_security_policy_docs_checks.yml"
)
REPO_LABELS_JSON = REPO_ROOT / ".github" / "labels.json"
REPO_LABELS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "repo_labels.yml"
LABEL_SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_github_labels.py"
GITLEAKS_ROTATION_SCRIPT = REPO_ROOT / "scripts" / "check_gitleaks_baseline_rotation.py"
PAID_FUNNEL_EMITTERS = [
    REPO_ROOT / "atlas_brain" / "api" / "billing.py",
    REPO_ROOT / "atlas_brain" / "content_ops_deflection_delivery.py",
]


class SecurityPolicyDocsTest(unittest.TestCase):
    def test_security_policy_links_incident_response_runbook(self) -> None:
        security_text = SECURITY_MD.read_text(encoding="utf-8")

        self.assertIn("## Incident Response", security_text)
        self.assertIn("[docs/INCIDENT_RESPONSE.md](docs/INCIDENT_RESPONSE.md)", security_text)

    def test_security_policy_documents_cve_remediation_sla(self) -> None:
        security_text = SECURITY_MD.read_text(encoding="utf-8")

        required_markers = [
            "Critical: fix, mitigate, or document acceptance within 7 calendar days.",
            "High: fix, mitigate, or document acceptance within 30 calendar days.",
            "Moderate: fix, mitigate, or document acceptance within 90 calendar days.",
            "`dependencies`, `security`, and",
            "`cve-remediation-sla` labels",
            "Dependabot PRs",
            "GitHub Security Advisories",
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, security_text)

    def test_dependabot_updates_carry_security_sla_labels(self) -> None:
        config_text = DEPENDABOT_YML.read_text(encoding="utf-8")
        update_labels = _dependabot_update_label_sets(config_text)

        _assert_dependabot_security_sla_labels(
            update_labels,
            expected_update_count=_dependabot_package_ecosystem_count(config_text),
        )

    def test_dependabot_label_contract_fails_without_update_blocks(self) -> None:
        with self.assertRaisesRegex(AssertionError, "no Dependabot update blocks found"):
            _dependabot_update_label_sets("version: 2\nupdates:\n")

    def test_dependabot_label_contract_rejects_missing_sla_label(self) -> None:
        update_labels = _dependabot_update_label_sets(
            '\n'.join(
                [
                    "version: 2",
                    "updates:",
                    '  - package-ecosystem: "pip"',
                    '    directory: "/"',
                    "    labels:",
                    "      - dependencies",
                    "      - security",
                ]
            )
        )

        with self.assertRaisesRegex(AssertionError, "pip labels missing"):
            _assert_dependabot_security_sla_labels(
                update_labels,
                expected_update_count=1,
            )

    def test_dependabot_label_contract_rejects_partial_parse(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "parsed 1 Dependabot update block.*2 package-ecosystem markers",
        ):
            _assert_dependabot_security_sla_labels(
                [("pip", {"dependencies", "security", "cve-remediation-sla"})],
                expected_update_count=2,
            )

    def test_security_policy_docs_workflow_runs_on_dependabot_config_changes(self) -> None:
        workflow = SECURITY_POLICY_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn('".github/dependabot.yml"', workflow)

    def test_security_policy_docs_workflow_runs_on_label_contract_changes(self) -> None:
        workflow = SECURITY_POLICY_WORKFLOW.read_text(encoding="utf-8")

        required_paths = [
            '".github/labels.json"',
            '".github/workflows/repo_labels.yml"',
            '"docs/SECURITY_GUARDRAILS.md"',
            '"scripts/check_gitleaks_baseline_rotation.py"',
            '"scripts/sync_github_labels.py"',
        ]
        for path in required_paths:
            with self.subTest(path=path):
                self.assertIn(path, workflow)

    def test_repo_label_manifest_covers_security_policy_references(self) -> None:
        manifest_labels = _repo_label_manifest_names()
        dependabot_labels = set().union(
            *[labels for _, labels in _dependabot_update_label_sets(DEPENDABOT_YML.read_text(encoding="utf-8"))]
        )
        required_labels = (
            dependabot_labels
            | _security_policy_label_references(SECURITY_MD.read_text(encoding="utf-8"))
            | {_gitleaks_rotation_label()}
        )

        _assert_repo_label_manifest_covers(required_labels, manifest_labels)

    def test_repo_label_manifest_rejects_missing_referenced_label(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing repo label definitions.*security"):
            _assert_repo_label_manifest_covers({"security"}, {"dependencies"})

    def test_repo_label_manifest_rejects_duplicate_names(self) -> None:
        syncer = _load_label_syncer()

        with self.assertRaisesRegex(ValueError, "duplicate repo label definitions: security"):
            syncer.load_manifest_text(
                """
                [
                  {"name": "security", "color": "b60205", "description": "one"},
                  {"name": "security", "color": "d73a4a", "description": "two"}
                ]
                """
            )

    def test_repo_label_manifest_rejects_bad_color(self) -> None:
        syncer = _load_label_syncer()

        with self.assertRaisesRegex(ValueError, "security: color must be a 6-digit hex value"):
            syncer.load_manifest_text(
                '[{"name": "security", "color": "red", "description": "bad"}]'
            )

    def test_repo_label_manifest_rejects_overlong_description(self) -> None:
        syncer = _load_label_syncer()
        overlong = "x" * 101

        with self.assertRaisesRegex(ValueError, "description must be 100 characters or fewer"):
            syncer.load_manifest_text(
                f'[{{"name": "security", "color": "b60205", "description": "{overlong}"}}]'
            )

    def test_repo_label_sync_plans_missing_and_stale_live_labels(self) -> None:
        syncer = _load_label_syncer()
        manifest = (
            syncer.LabelSpec("security", "b60205", "Security-relevant dependency or vulnerability work"),
            syncer.LabelSpec("security-rotation", "d93f0b", "Controlled credential or security-baseline rotation"),
        )
        live = {
            "security": syncer.LabelSpec("security", "ededed", "Old description"),
        }

        plan = syncer.plan_label_sync(manifest, live)

        self.assertEqual([label.name for label in plan.create], ["security-rotation"])
        self.assertEqual([update.wanted.name for update in plan.update], ["security"])
        self.assertEqual([update.current_name for update in plan.update], ["security"])
        self.assertEqual(plan.unchanged, ())

    def test_repo_label_sync_plans_case_only_label_drift_as_update(self) -> None:
        syncer = _load_label_syncer()
        manifest = (
            syncer.LabelSpec("security", "b60205", "Security-relevant dependency or vulnerability work"),
        )
        live = syncer.load_repo_labels_text(
            """
            [
              {
                "name": "Security",
                "color": "b60205",
                "description": "Security-relevant dependency or vulnerability work"
              }
            ]
            """
        )

        plan = syncer.plan_label_sync(manifest, live)

        self.assertEqual(plan.create, ())
        self.assertEqual(plan.update[0].current_name, "Security")
        self.assertEqual(plan.update[0].wanted.name, "security")

    def test_repo_label_sync_plans_noop_when_live_labels_match(self) -> None:
        syncer = _load_label_syncer()
        manifest = (
            syncer.LabelSpec("cve-remediation-sla", "5319e7", "Dependabot/CVE remediation governed by SECURITY.md SLA"),
        )
        live = {
            "cve-remediation-sla": syncer.LabelSpec(
                "cve-remediation-sla",
                "5319e7",
                "Dependabot/CVE remediation governed by SECURITY.md SLA",
            )
        }

        plan = syncer.plan_label_sync(manifest, live)

        self.assertFalse(plan.has_changes)
        self.assertEqual(plan.unchanged, ("cve-remediation-sla",))

    def test_repo_labels_workflow_syncs_manifest_from_trusted_events_only(self) -> None:
        workflow = REPO_LABELS_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("push:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("schedule:", workflow)
        self.assertIn('cron: "37 10 * * 1"', workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertIn("if: github.event_name != 'workflow_dispatch' || github.ref == 'refs/heads/main'", workflow)
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", workflow)
        self.assertIn("issues: write", workflow)
        self.assertIn("python scripts/sync_github_labels.py --manifest .github/labels.json --apply", workflow)

    def test_security_guardrails_documents_repo_label_manifest(self) -> None:
        guardrails = SECURITY_GUARDRAILS_MD.read_text(encoding="utf-8")

        required_markers = [
            ".github/labels.json",
            "`Repo Labels` workflow",
            "`scripts/sync_github_labels.py`",
            "trusted `main` updates and manual dispatch",
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, guardrails)

    def test_incident_response_runbook_covers_required_sections(self) -> None:
        runbook = INCIDENT_RESPONSE_MD.read_text(encoding="utf-8")

        required_markers = [
            "# Incident Response",
            "## Severity Levels",
            "SEV0",
            "SEV1",
            "SEV2",
            "SEV3",
            "## Roles and Ownership",
            "## Intake and Triage",
            "## Communications",
            "## Paid Funnel Incident Types",
            "## Credential Rotation",
            "## Containment and Recovery",
            "## Postmortem Template",
            "Incident:",
            "Root cause:",
            "Timeline:",
            "Follow-up actions:",
        ]
        for marker in required_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, runbook)

    def test_incident_response_names_current_paid_funnel_incidents(self) -> None:
        runbook = INCIDENT_RESPONSE_MD.read_text(encoding="utf-8")

        incident_types = sorted(_paid_funnel_incident_types_from_emitters())
        for incident_type in incident_types:
            with self.subTest(incident_type=incident_type):
                self.assertIn(incident_type, runbook)


def _load_script_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_label_syncer():
    return _load_script_module("sync_github_labels", LABEL_SYNC_SCRIPT)


def _load_gitleaks_checker():
    return _load_script_module("check_gitleaks_baseline_rotation", GITLEAKS_ROTATION_SCRIPT)


def _repo_label_manifest_names() -> set[str]:
    syncer = _load_label_syncer()
    labels = syncer.load_manifest(REPO_LABELS_JSON)
    return {label.name for label in labels}


def _gitleaks_rotation_label() -> str:
    return _load_gitleaks_checker().DEFAULT_ROTATION_LABEL


def _security_policy_label_references(security_text: str) -> set[str]:
    match = re.search(r"Dependabot PRs carry the (?P<labels>.*?) labels\.", security_text, re.S)
    if not match:
        raise AssertionError("SECURITY.md does not document Dependabot PR labels")
    labels = set(re.findall(r"`([^`]+)`", match.group("labels")))
    if not labels:
        raise AssertionError("SECURITY.md Dependabot label sentence has no backtick labels")
    return labels


def _assert_repo_label_manifest_covers(
    required_labels: set[str],
    manifest_labels: set[str],
) -> None:
    missing = sorted(required_labels - manifest_labels)
    if missing:
        raise AssertionError(f"missing repo label definitions: {missing}")


def _paid_funnel_incident_types_from_emitters() -> set[str]:
    incident_types: set[str] = set()
    for emitter_path in PAID_FUNNEL_EMITTERS:
        tree = ast.parse(emitter_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node.func) not in {
                "_emit_delivery_incident",
                "emit_deflection_paid_funnel_incident_alert",
            }:
                continue
            incident_type = _literal_incident_type(node)
            if incident_type:
                incident_types.add(incident_type)
    if not incident_types:
        raise AssertionError("no paid-funnel incident emitters found")
    return incident_types


def _dependabot_update_label_sets(config_text: str) -> list[tuple[str, set[str]]]:
    update_labels: list[tuple[str, set[str]]] = []
    current_ecosystem: str | None = None
    current_labels: set[str] = set()
    in_labels = False

    def flush_current() -> None:
        nonlocal current_ecosystem, current_labels, in_labels
        if current_ecosystem is not None:
            update_labels.append((current_ecosystem, set(current_labels)))
        current_ecosystem = None
        current_labels = set()
        in_labels = False

    for raw_line in config_text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- package-ecosystem:"):
            flush_current()
            current_ecosystem = stripped.split(":", 1)[1].strip().strip('"')
            continue
        if current_ecosystem is None:
            continue
        if stripped == "labels:":
            in_labels = True
            continue
        if in_labels and stripped.startswith("- "):
            current_labels.add(stripped[2:].strip().strip('"'))
            continue
        if in_labels and stripped and not stripped.startswith("#"):
            in_labels = False

    flush_current()
    if not update_labels:
        raise AssertionError("no Dependabot update blocks found")
    return update_labels


def _assert_dependabot_security_sla_labels(
    update_labels: list[tuple[str, set[str]]],
    *,
    expected_update_count: int,
) -> None:
    if len(update_labels) != expected_update_count:
        raise AssertionError(
            "parsed "
            f"{len(update_labels)} Dependabot update block(s), but found "
            f"{expected_update_count} package-ecosystem markers"
        )
    required_labels = {"dependencies", "security", "cve-remediation-sla"}
    for ecosystem, labels in update_labels:
        missing = required_labels - labels
        if missing:
            raise AssertionError(f"{ecosystem} labels missing {sorted(missing)}")


def _dependabot_package_ecosystem_count(config_text: str) -> int:
    return sum(
        1
        for line in config_text.splitlines()
        if line.strip().startswith("- package-ecosystem:")
    )


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literal_incident_type(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "incident_type" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else None
    if node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        return value if isinstance(value, str) else None
    return None


if __name__ == "__main__":
    unittest.main()
