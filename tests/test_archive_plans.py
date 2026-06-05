from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "archive_plans.py"


def load_tool():
    name = "archive_plans"
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


def _plan(text_title: str, *, lane: str | None = None, phase: str | None = None) -> str:
    lines = [f"# {text_title}", ""]
    if lane:
        lines.append(f"Ownership lane: {lane}")
    if phase:
        lines.append(f"Slice phase: {phase}")
    lines.append("")
    lines.append("## Why this slice exists")
    return "\n".join(lines) + "\n"


def _make_plans(tmp_path: Path, names: list[str]) -> Path:
    plans = tmp_path / "plans"
    plans.mkdir()
    for n in names:
        (plans / f"{n}.md").write_text(_plan(n, lane="x/y", phase="Workflow/process"), encoding="utf-8")
    return plans


def test_archive_moves_root_plans_and_writes_index(tmp_path):
    tool = load_tool()
    plans = _make_plans(tmp_path, ["PR-Alpha", "PR-Beta"])

    moved = tool.archive_plans(plans)
    tool.write_index(plans)

    assert {p.name for p in moved} == {"PR-Alpha.md", "PR-Beta.md"}
    # root no longer holds the plan docs; archive does
    assert tool.root_plan_files(plans) == []
    assert {p.name for p in tool.archived_plan_files(plans)} == {
        "PR-Alpha.md",
        "PR-Beta.md",
    }
    index = (plans / "INDEX.md").read_text(encoding="utf-8")
    assert "PR-Alpha" in index and "PR-Beta" in index
    assert "archive/PR-Alpha.md" in index
    assert "lane: x/y" in index and "phase: Workflow/process" in index


def test_archive_is_idempotent(tmp_path):
    tool = load_tool()
    plans = _make_plans(tmp_path, ["PR-Alpha"])

    tool.archive_plans(plans)
    # second run finds nothing left in root, leaves the archive intact
    moved_again = tool.archive_plans(plans)
    assert moved_again == []
    assert len(tool.archived_plan_files(plans)) == 1


def test_index_ignores_non_plan_and_index_files(tmp_path):
    tool = load_tool()
    plans = _make_plans(tmp_path, ["PR-Alpha"])
    (plans / "README.md").write_text("# not a plan\n", encoding="utf-8")
    (plans / "notes.txt").write_text("scratch\n", encoding="utf-8")

    tool.archive_plans(plans)
    # README/notes stay in root (only PR-*.md is archived)
    assert (plans / "README.md").is_file()
    assert (plans / "notes.txt").is_file()
    index = tool.build_index(plans)
    assert "README" not in index
    assert "PR-Alpha" in index


def test_over_threshold_flags_only_when_root_exceeds(tmp_path):
    tool = load_tool()
    plans = _make_plans(tmp_path, ["PR-A", "PR-B", "PR-C"])

    count, over = tool.over_threshold(plans, threshold=5)
    assert count == 3 and over is False

    count, over = tool.over_threshold(plans, threshold=2)
    assert count == 3 and over is True


def test_check_command_is_non_blocking_even_over_threshold(tmp_path, capsys):
    tool = load_tool()
    plans = _make_plans(tmp_path, ["PR-A", "PR-B"])

    rc = tool.main(["check", "--plans-dir", str(plans), "--threshold", "1"])
    out = capsys.readouterr().out
    assert rc == 0  # never fails a PR
    assert "WARNING" in out and "archive_plans.py archive" in out


def test_plan_metadata_extracts_title_lane_phase():
    tool = load_tool()
    meta = tool.plan_metadata(_plan("PR-Foo", lane="intel/ci", phase="Robust testing"))
    assert meta["title"] == "PR-Foo"
    assert meta["lane"] == "intel/ci"
    assert meta["phase"] == "Robust testing"
