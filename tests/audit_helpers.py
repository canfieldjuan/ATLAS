"""Shared loader for the pre-push-gate audit scripts under scripts/.

The audit scripts live in scripts/ and are not on Python's import path.
They also are not under a package (no scripts/__init__.py). This helper
loads each script as a module via importlib.util so tests can import
the parsing functions directly without subprocess overhead.

Tests use the helper like this:

    from tests.audit_helpers import load_auditor

    @pytest.fixture(scope="module")
    def auditor():
        return load_auditor("audit_mcp_port_assignments")

    def test_b2b_churn_port_matches(auditor):
        claims = auditor.doc_claims("ATLAS_MCP_B2B_CHURN_PORT=8062\\n")
        assert any(name == "b2b_churn" for _, name, _, _ in claims)

If the auditor script does not exist on the current branch (because
the auditor's parent PR has not merged yet), load_auditor() raises
pytest.skip with a clear message naming the dependency. That lets
this slice ship off main while PRs #483 / #484 / #485 / #486 are
still in review; tests skip cleanly today and activate as each
underlying PR lands.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_auditor(name: str) -> ModuleType:
    """Return the auditor module named `<name>` from scripts/.

    Skips the calling test (via pytest.skip) if the script file does
    not exist on the current branch. Caches successful loads in
    sys.modules so the importlib hop is one-shot per session.
    """
    if name in sys.modules:
        return sys.modules[name]
    path = SCRIPTS_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(
            f"requires {path.relative_to(REPO_ROOT)} to exist; "
            f"depends on a prior audit PR merging first"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
