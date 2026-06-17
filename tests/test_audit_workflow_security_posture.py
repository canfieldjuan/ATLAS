from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_workflow_security_posture.py"


def load_auditor():
    name = "audit_workflow_security_posture"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
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


def _write_workflow(tmp_path: Path, name: str, text: str) -> Path:
    workflow = tmp_path / name
    workflow.write_text(text, encoding="utf-8")
    return workflow


def test_unapproved_pull_request_target_is_error(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "unsafe.yml",
        """
name: Unsafe
on:
  pull_request_target:
jobs:
  test:
    runs-on: ubuntu-latest
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "pull_request_target" in f.detail for f in findings)


def test_approved_security_guardrails_pull_request_target_is_allowed(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "security_guardrails.yml",
        """
name: Security
on:
  pull_request_target:
jobs:
  guard:
    runs-on: ubuntu-latest
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert not [f for f in findings if f.level == "ERROR"]


def test_unapproved_oidc_write_is_error(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "oidc.yml",
        """
name: OIDC
on: pull_request
jobs:
  cloud:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "id-token" in f.detail for f in findings)


def test_claude_oidc_write_is_warn_only(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "claude.yml",
        """
name: Claude
on: issue_comment
jobs:
  claude:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "WARN" and "allowed id-token" in f.detail for f in findings)
    assert not [f for f in findings if f.level == "ERROR"]


def test_mutable_action_ref_is_warn_only(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "mutable.yml",
        """
name: Mutable
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: local/action
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "WARN" and "actions/checkout@v4" in f.detail for f in findings)
    assert any(f.level == "WARN" and "local/action@<missing ref>" in f.detail for f in findings)


def test_sha_pinned_action_ref_is_clean(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "pinned.yml",
        """
name: Pinned
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@1234567890abcdef1234567890abcdef12345678
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert findings == []
