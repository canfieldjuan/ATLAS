from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SECURITY_MD = REPO_ROOT / "SECURITY.md"
INCIDENT_RESPONSE_MD = REPO_ROOT / "docs" / "INCIDENT_RESPONSE.md"
PAID_FUNNEL_EMITTERS = [
    REPO_ROOT / "atlas_brain" / "api" / "billing.py",
    REPO_ROOT / "atlas_brain" / "content_ops_deflection_delivery.py",
]


class SecurityPolicyDocsTest(unittest.TestCase):
    def test_security_policy_links_incident_response_runbook(self) -> None:
        security_text = SECURITY_MD.read_text(encoding="utf-8")

        self.assertIn("## Incident Response", security_text)
        self.assertIn("[docs/INCIDENT_RESPONSE.md](docs/INCIDENT_RESPONSE.md)", security_text)

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
