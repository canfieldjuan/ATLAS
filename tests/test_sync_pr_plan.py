from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync_pr_plan.py"


def load_sync_pr_plan():
    name = "sync_pr_plan"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_plan_text_rewrites_machine_checked_sections() -> None:
    module = load_sync_pr_plan()
    text = textwrap.dedent(
        """\
        # Plan

        ## Why this slice exists

        Test.

        ## Scope (this PR)

        Ownership lane: workflow
        Slice phase: Workflow/process

        ### Files touched

        - `stale.py`

        ## Mechanism

        Keep this prose.

        ## Estimated diff size

        | File | LOC |
        |---|---:|
        | `stale.py` | 99 |
        | **Total** | **99** |
        """
    )

    updated = module.sync_plan_text(
        text,
        [
            module.DiffEntry("plans/PR-Test.md", 10, 1),
            module.DiffEntry("scripts/tool.py", 3, 2),
        ],
    )

    assert "- `stale.py`" not in updated
    assert "- `plans/PR-Test.md`" in updated
    assert "- `scripts/tool.py`" in updated
    assert "## Mechanism\n\nKeep this prose." in updated
    assert "| `plans/PR-Test.md` | 11 |" in updated
    assert "| `scripts/tool.py` | 5 |" in updated
    assert "| **Total** | **16** |" in updated


def test_sync_plan_text_inserts_files_touched_when_missing() -> None:
    module = load_sync_pr_plan()
    text = textwrap.dedent(
        """\
        # Plan

        ## Scope (this PR)

        Ownership lane: workflow
        Slice phase: Workflow/process

        ## Mechanism

        Test.

        ## Estimated diff size

        | File | LOC |
        |---|---:|
        | **Total** | **0** |
        """
    )

    updated = module.sync_plan_text(text, [module.DiffEntry("scripts/tool.py", 1, 0)])

    assert "### Files touched\n\n- `scripts/tool.py`\n\n## Mechanism" in updated


def test_sync_plan_text_requires_scope_heading() -> None:
    module = load_sync_pr_plan()
    text = "# Plan\n\n## Estimated diff size\n"

    try:
        module.sync_plan_text(text, [])
    except ValueError as exc:
        assert "## Scope (this PR)" in str(exc)
    else:
        raise AssertionError("expected missing scope heading to fail")


def test_tracked_entries_count_renamed_destination_path(tmp_path: Path) -> None:
    module = load_sync_pr_plan()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    old_path = tmp_path / "old.txt"
    old_path.write_text("one\ntwo\n", encoding="utf-8")
    subprocess.run(["git", "add", "old.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    subprocess.run(["git", "mv", "old.txt", "new.txt"], cwd=tmp_path, check=True)
    (tmp_path / "new.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    entries = module._tracked_entries("HEAD", cwd=tmp_path)

    assert entries == [module.DiffEntry(path="new.txt", added=1, deleted=0)]


def test_parse_numstat_z_rejects_malformed_rename_destination() -> None:
    module = load_sync_pr_plan()

    try:
        module._parse_numstat_z("1\t0\t\0old.txt\0")
    except ValueError as exc:
        assert "missing rename destination" in str(exc)
    else:
        raise AssertionError("expected malformed rename output to fail")
