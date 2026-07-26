"""Fixture tests for the impacted-test selector and the scoped unit gate.

Per AGENTS.md 3i a checker must prove its FAILURE detection, not only its happy
path. The selector's dangerous failure is selecting too FEW tests (a green gate
that ran nothing relevant), so most of these assert that a test IS selected or
that the run escalates to FULL.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sel = _load("select_impacted_tests")
gate = _load("check_unit_gate")


def _mkrepo(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return tmp_path


# --- selection: the direction that matters (too few) ------------------------


def test_direct_importer_is_selected(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/svc.py": "VALUE = 1\n",
        "tests/test_svc.py": "from atlas_brain.svc import VALUE\n",
    })
    assert sel.select(["atlas_brain/svc.py"], repo) == ["tests/test_svc.py"]


def test_transitive_importer_is_selected(tmp_path):
    """test -> helper -> changed module. A one-hop grep would miss this."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/deep.py": "VALUE = 1\n",
        "atlas_brain/mid.py": "from atlas_brain.deep import VALUE\n",
        "tests/test_top.py": "from atlas_brain.mid import VALUE\n",
    })
    assert sel.select(["atlas_brain/deep.py"], repo) == ["tests/test_top.py"]


def test_four_hop_chain_is_selected(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/a.py": "X = 1\n",
        "atlas_brain/b.py": "from atlas_brain.a import X\n",
        "atlas_brain/c.py": "from atlas_brain.b import X\n",
        "tests/test_d.py": "from atlas_brain.c import X\n",
    })
    assert sel.select(["atlas_brain/a.py"], repo) == ["tests/test_d.py"]


def test_symbol_import_records_the_module_edge(tmp_path):
    """`from pkg.mod import symbol` must map to pkg/mod.py, not be dropped."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/svc.py": "def helper():\n    return 1\n",
        "tests/test_sym.py": "from atlas_brain.svc import helper\n",
    })
    assert sel.select(["atlas_brain/svc.py"], repo) == ["tests/test_sym.py"]


def test_relative_import_is_followed(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/pkg/__init__.py": "",
        "atlas_brain/pkg/leaf.py": "X = 1\n",
        "atlas_brain/pkg/user.py": "from .leaf import X\n",
        "tests/test_rel.py": "from atlas_brain.pkg.user import X\n",
    })
    assert sel.select(["atlas_brain/pkg/leaf.py"], repo) == ["tests/test_rel.py"]


def test_changed_test_file_selects_itself(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "tests/test_standalone.py": "def test_x():\n    assert True\n",
    })
    assert sel.select(["tests/test_standalone.py"], repo) == ["tests/test_standalone.py"]


def test_unrelated_module_is_not_selected(tmp_path):
    """Over-selection direction: an unrelated test must not be pulled in."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/svc.py": "X = 1\n",
        "atlas_brain/other.py": "Y = 2\n",
        "tests/test_other.py": "from atlas_brain.other import Y\n",
    })
    assert sel.select(["atlas_brain/svc.py"], repo) == []


# --- escalation: unresolvable input must never yield an empty selection -----


@pytest.mark.parametrize("path", [
    "tests/conftest.py",
    "atlas_brain/services/conftest.py",
    "requirements.txt",
    "requirements.content_ops_ci.txt",
    "pytest.ini",
    "pyproject.toml",
    "tests/unit_gate_baseline.txt",
    "scripts/check_unit_gate.py",
    "scripts/select_impacted_tests.py",
    ".github/workflows/unit_gate.yml",
])
def test_global_files_escalate_to_full(tmp_path, path):
    repo = _mkrepo(tmp_path, {"atlas_brain/__init__.py": ""})
    assert sel.select([path], repo) == sel.FULL


def test_unparseable_module_escalates_to_full(tmp_path):
    """Unknown import edges must not be read as 'imports nothing'."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/svc.py": "X = 1\n",
        "atlas_brain/broken.py": "def (((\n",
        "tests/test_svc.py": "from atlas_brain.svc import X\n",
    })
    assert sel.select(["atlas_brain/svc.py"], repo) == sel.FULL


def test_empty_diff_escalates_to_full(tmp_path):
    """No diff is 'something is wrong with the base ref', not 'nothing to run'."""
    repo = _mkrepo(tmp_path, {"atlas_brain/__init__.py": ""})
    assert sel.select([], repo) == sel.FULL


def test_docs_only_change_selects_nothing(tmp_path):
    """The one case where empty is correct: mapped, and reachable from no test."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/svc.py": "X = 1\n",
        "tests/test_svc.py": "from atlas_brain.svc import X\n",
    })
    assert sel.select(["docs/GUIDE.md", "plans/PR-Thing.md"], repo) == []


# --- the scoped baseline: the trap that makes naive scoping fail ------------


def test_restrict_baseline_keeps_only_selected_files():
    baseline = {
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two",
        "tests/test_b.py::test_three",
    }
    assert gate.restrict_baseline(baseline, {"tests/test_a.py"}) == {
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_two",
    }


def test_unrestricted_baseline_would_fail_a_scoped_run():
    """Without restriction every unselected baseline entry reads as newly-passing.

    This is the defect the --selected-files flag exists to prevent; asserting it
    directly keeps the reason from being refactored away.
    """
    baseline = {"tests/test_a.py::t", "tests/test_b.py::t", "tests/test_c.py::t"}
    failing_from_scoped_run = {"tests/test_a.py::t"}

    _, fixed_unrestricted = gate.compare(failing_from_scoped_run, baseline)
    assert len(fixed_unrestricted) == 2  # b and c look "fixed" -- gate fails

    restricted = gate.restrict_baseline(baseline, {"tests/test_a.py"})
    regressions, fixed = gate.compare(failing_from_scoped_run, restricted)
    assert regressions == [] and fixed == []


def test_regression_inside_scope_is_still_caught():
    baseline = {"tests/test_a.py::known"}
    failing = {"tests/test_a.py::known", "tests/test_a.py::NEW"}
    restricted = gate.restrict_baseline(baseline, {"tests/test_a.py"})
    regressions, _ = gate.compare(failing, restricted)
    assert regressions == ["tests/test_a.py::NEW"]


def test_stale_entry_inside_scope_is_still_caught():
    baseline = {"tests/test_a.py::known", "tests/test_a.py::now_passing"}
    failing = {"tests/test_a.py::known"}
    restricted = gate.restrict_baseline(baseline, {"tests/test_a.py"})
    _, fixed = gate.compare(failing, restricted)
    assert fixed == ["tests/test_a.py::now_passing"]


def test_node_file_handles_parametrized_ids_with_colons():
    assert gate.node_file("tests/test_a.py::test_k[a::b]") == "tests/test_a.py"
