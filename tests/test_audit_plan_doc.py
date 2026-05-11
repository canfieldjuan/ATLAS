from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_plan_doc.py"


def load_auditor():
    name = "audit_plan_doc"
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


def _statuses(text: str) -> dict[str, str]:
    audit_plan_doc = load_auditor()
    return {row.canonical: row.status for row in audit_plan_doc.audit_plan_text(text)}


def test_happy_path_accepts_required_sections_in_order():
    text = textwrap.dedent(
        """\
        # Plan

        ## Why this slice exists
        ## Scope (this PR)
        ## Mechanism
        ## Intentional
        ## Deferred
        ## Verification
        ## Estimated diff size
        """
    )

    assert set(_statuses(text).values()) == {"OK"}


def test_out_of_scope_does_not_satisfy_scope():
    text = textwrap.dedent(
        """\
        # Plan

        ## Why this slice exists
        ## Out of scope
        ## Mechanism
        ## Intentional
        ## Deferred
        ## Verification
        ## Estimated diff size
        """
    )

    statuses = _statuses(text)

    assert statuses["Scope"] == "MISSING"


def test_out_of_order_section_is_reported():
    text = textwrap.dedent(
        """\
        # Plan

        ## Why this slice exists
        ## Mechanism
        ## Scope
        ## Intentional
        ## Deferred
        ## Verification
        ## Estimated diff size
        """
    )

    statuses = _statuses(text)

    assert statuses["Mechanism"] == "OUT OF ORDER"


def test_duplicate_required_section_is_reported():
    text = textwrap.dedent(
        """\
        # Plan

        ## Why this slice exists
        ## Scope
        ## Scope (this PR)
        ## Mechanism
        ## Intentional
        ## Deferred
        ## Verification
        ## Estimated diff size
        """
    )

    statuses = _statuses(text)

    assert statuses["Scope"] == "DUPLICATE"


def test_cli_returns_nonzero_for_missing_file():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "/tmp/atlas-plan-doc-does-not-exist.md"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "plan doc not found" in result.stderr


def test_cli_accepts_valid_plan(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(
        textwrap.dedent(
            """\
            # Plan

            ## Why this slice exists
            ## Scope
            ## Mechanism
            ## Intentional
            ## Deferred
            ## Verification
            ## Estimated diff size
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_cli_returns_nonzero_for_plan_drift(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(
        textwrap.dedent(
            """\
            # Plan

            ## Why this slice exists
            ## Out of scope
            ## Mechanism
            ## Intentional
            ## Deferred
            ## Verification
            ## Estimated diff size
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "MISSING" in result.stdout
    assert "## Scope" in result.stdout
