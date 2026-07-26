"""The unit gate must survive a PR head that predates the selector.

#2207 merged `scripts/select_impacted_tests.py`. For a `pull_request` event
GitHub takes the workflow definition from the merged ref but checks out the PR
HEAD, so every branch cut before that merge invokes a selector its tree does not
contain -- 20+ open PRs died on ENOENT. The workflow guards that with a
`[ ! -f ... ]` fallback to FULL.

The guard is shell inside YAML, so nothing else in the suite can execute it.
These tests extract the Select step's script and run it for real in a temp tree,
with and without the selector present -- per AGENTS.md 3i, proving the failure
branch fires rather than only the happy path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github/workflows/unit_gate.yml"


def _select_step_script() -> str:
    """The `run:` body of the Select step, straight from the workflow."""
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = doc["jobs"]["unit-gate"]["steps"]
    for step in steps:
        if step.get("name") == "Select impacted tests":
            return step["run"]
    raise AssertionError("Select impacted tests step not found in unit_gate.yml")


def _run_select(tmp_path: Path, *, selector_present: bool) -> tuple[int, str]:
    """Execute the real Select script in a throwaway tree; return (rc, selection)."""
    script = _select_step_script()
    # The step fetches the base ref and diffs against it; neither is available
    # here, so stub git. The branch under test is the file-existence guard.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "git").chmod(0o755)
    # The workflow calls `python`; CI's setup-python provides it, a bare
    # PATH here does not. Shim it to this interpreter so the test exercises
    # the guard rather than a missing binary.
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)

    work = tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    if selector_present:
        # A sentinel selector, not the real one: with git stubbed the real
        # selector escalates to FULL by its own empty-diff rule, so its output
        # is indistinguishable from the guard escalating. The sentinel proves
        # the else-branch actually INVOKED the script.
        (work / "scripts/select_impacted_tests.py").write_text(
            "print('SENTINEL_SELECTOR_RAN')\n", encoding="utf-8")

    selection = tmp_path / "selected.txt"
    script = script.replace("/tmp/selected.txt", str(selection))
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=work,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH','')}",
             "BASE_REF": "main"},
    )
    return proc.returncode, (selection.read_text() if selection.exists() else "")


def test_absent_selector_falls_back_to_full(tmp_path):
    """The branch that was broken: no selector in the tree must yield FULL, not ENOENT."""
    rc, selection = _run_select(tmp_path, selector_present=False)
    assert rc == 0, f"Select step failed on a pre-selector head: {rc}"
    assert selection.strip() == "FULL"


def test_absent_selector_never_yields_an_empty_selection(tmp_path):
    """Empty would run the growth-guard-only path and skip the suite entirely."""
    _, selection = _run_select(tmp_path, selector_present=False)
    assert selection.strip() != ""


def test_present_selector_is_still_invoked(tmp_path):
    """Over-correction guard: the fallback must not swallow the normal path.

    Asserting only "non-empty" cannot distinguish the selector running from the
    guard escalating -- both produce output. The sentinel selector emits a token
    the guard never could, so this fails if the fallback swallows the normal
    path and silently sends every PR to the expensive FULL suite.
    """
    rc, selection = _run_select(tmp_path, selector_present=True)
    assert rc == 0
    assert selection.strip() == "SENTINEL_SELECTOR_RAN"
    assert selection.strip() != "FULL"


def test_workflow_guards_the_selector_invocation():
    """Pins the guard itself so a later 'simplification' cannot drop it silently."""
    script = _select_step_script()
    assert "! -f scripts/select_impacted_tests.py" in script
    assert "echo FULL" in script


# --- the merge-base growth-guard resolution, against a REAL git repo --------


def _baseline_step_script() -> str:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for step in doc["jobs"]["unit-gate"]["steps"]:
        if step.get("name", "").startswith("Resolve base-branch baseline"):
            return step["run"]
    raise AssertionError("Resolve base-branch baseline step not found")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def test_growth_guard_resolves_the_merge_base_baseline(tmp_path):
    """A branch cut before main shrank the baseline must not read as growth.

    Everything else here stubs git; this one builds a real repository, because
    the defect being prevented lives entirely in git history resolution and a
    stub cannot express "main moved after the branch forked".
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@t"); _git(origin, "config", "user.name", "t")
    baseline = origin / "tests"
    baseline.mkdir()
    (baseline / "unit_gate_baseline.txt").write_text("a::t1\nb::t2\nc::t3\n")
    _git(origin, "add", "-A"); _git(origin, "commit", "-q", "-m", "seed")

    # branch forks here, with the 3-entry baseline
    _git(origin, "checkout", "-q", "-b", "feature")
    (origin / "note.txt").write_text("branch work\n")
    _git(origin, "add", "-A"); _git(origin, "commit", "-q", "-m", "branch work")

    # main then SHRINKS the baseline
    _git(origin, "checkout", "-q", "main")
    (baseline / "unit_gate_baseline.txt").write_text("a::t1\n")
    _git(origin, "add", "-A"); _git(origin, "commit", "-q", "-m", "shrink baseline")

    # a checkout of the branch, with origin/main present, as CI has it
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)],
                   check=True, capture_output=True, text=True)
    _git(work, "checkout", "-q", "feature")

    out = tmp_path / "base_baseline.txt"
    script = _baseline_step_script().replace("/tmp/base_baseline.txt", str(out))
    proc = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True,
                          text=True, env={**os.environ, "BASE_REF": "main"})
    assert proc.returncode == 0, proc.stderr

    resolved = out.read_text().split()
    # The merge base still has all three. Resolving against current main would
    # yield one, and the branch's own 3-entry baseline would then look like
    # +2 growth -- failing a PR that added nothing.
    assert resolved == ["a::t1", "b::t2", "c::t3"], (
        f"growth guard resolved the wrong baseline: {resolved}"
    )
