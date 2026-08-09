"""Fixture tests for the impacted-test selector and the scoped unit gate.

Per AGENTS.md 3i a checker must prove its FAILURE detection, not only its happy
path. The selector's dangerous failure is selecting too FEW tests (a green gate
that ran nothing relevant), so most of these assert that a test IS selected or
that the run escalates to FULL.
"""
from __future__ import annotations

import ast
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


def test_package_initializer_edge_is_recorded(tmp_path):
    """Importing pkg.leaf executes pkg/__init__.py too; changing it must select."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/pkg/__init__.py": "INIT = 1\n",
        "atlas_brain/pkg/leaf.py": "X = 1\n",
        "tests/test_pkg.py": "import atlas_brain.pkg.leaf\n",
    })
    assert sel.select(["atlas_brain/pkg/__init__.py"], repo) == ["tests/test_pkg.py"]


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
])
def test_global_files_escalate_to_full(tmp_path, path):
    repo = _mkrepo(tmp_path, {"atlas_brain/__init__.py": ""})
    assert sel.select([path], repo) == sel.FULL


@pytest.mark.parametrize(("path", "owners"), [
    (
        ".github/workflows/branch_protection_required_checks.yml",
        ["tests/test_security_guardrails_workflow.py"],
    ),
    (
        ".github/workflows/ai_reconciliation_live.yml",
        [
            "tests/test_audit_workflow_security_posture.py",
            "tests/test_check_ai_reconciliation_live.py",
        ],
    ),
    (
        ".github/workflows/ai_reconciliation_review_retrigger.yml",
        ["tests/test_check_ai_reconciliation_live.py"],
    ),
    (
        ".github/workflows/pr_body_contract.yml",
        ["tests/test_pr_body_contract_workflow.py"],
    ),
    (
        ".github/workflows/unit_gate.yml",
        [
            "tests/test_check_unit_gate.py",
            "tests/test_select_impacted_tests.py",
            "tests/test_unit_gate_selector_fallback.py",
        ],
    ),
    ("ci/gates.yml", ["tests/test_security_guardrails_workflow.py"]),
    (
        "docs/SECURITY_GUARDRAILS.md",
        [
            "tests/test_security_guardrails_workflow.py",
            "tests/test_security_policy_docs.py",
        ],
    ),
    ("docs/ci_cd_autonomous_coding_map.md", ["tests/test_audit_pr_watcher_safety.py"]),
    ("docs/ci_cd_runtime_duplication_audit.md", ["tests/test_security_guardrails_workflow.py"]),
    (
        "docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md",
        ["tests/test_security_guardrails_workflow.py"],
    ),
    (
        "docs/audits/required-workflow-enrollment-audit-2026-08-04.md",
        ["tests/test_security_guardrails_workflow.py"],
    ),
    (
        "scripts/audit_ai_reconciliation.py",
        [
            "tests/test_audit_ai_reconciliation.py",
            "tests/test_audit_fix_loop_disposition.py",
            "tests/test_audit_pr_body.py",
            "tests/test_check_ai_reconciliation_live.py",
        ],
    ),
    (
        "scripts/audit_fix_loop_disposition.py",
        [
            "tests/test_audit_fix_loop_disposition.py",
            "tests/test_local_pr_review.py",
        ],
    ),
    (
        "scripts/audit_pr_body.py",
        [
            "tests/test_audit_pr_body.py",
            "tests/test_check_ai_reconciliation_live.py",
            "tests/test_local_pr_review.py",
            "tests/test_open_pr_wrapper.py",
            "tests/test_push_pr_wrapper.py",
        ],
    ),
    (
        "scripts/audit_workflow_security_posture.py",
        ["tests/test_audit_workflow_security_posture.py"],
    ),
    (
        "scripts/check_ai_reconciliation_live.py",
        ["tests/test_check_ai_reconciliation_live.py"],
    ),
    (
        "scripts/check_required_status_checks.py",
        ["tests/test_security_guardrails_workflow.py"],
    ),
    (
        "scripts/codex_wake_bridge.py",
        ["tests/test_codex_wake_bridge.py"],
    ),
    (
        "scripts/check_unit_gate.py",
        ["tests/test_check_unit_gate.py", "tests/test_select_impacted_tests.py"],
    ),
    (
        "extracted/_shared/scripts/check_ascii_python.sh",
        ["tests/test_pre_push_audit.py"],
    ),
    ("scripts/local_pr_review.sh", ["tests/test_local_pr_review.py"]),
    ("scripts/open_pr.sh", ["tests/test_open_pr_wrapper.py"]),
    ("scripts/pre_push_audit.sh", ["tests/test_pre_push_audit.py"]),
    ("scripts/pr_watcher.py", ["tests/test_pr_watcher.py"]),
    ("scripts/push_pr.sh", ["tests/test_push_pr_wrapper.py"]),
    ("scripts/select_impacted_tests.py", ["tests/test_select_impacted_tests.py"]),
    ("scripts/watch_owned_pr.sh", ["tests/test_watch_owned_pr.py"]),
])
def test_explicit_ci_surface_owners_are_selected(tmp_path, path, owners):
    files = {"atlas_brain/__init__.py": "", path: "VALUE = 1\n"}
    files.update({owner: "def test_owner():\n    assert True\n" for owner in owners})
    repo = _mkrepo(tmp_path, files)

    assert sel.select([path], repo) == sorted(owners)


def test_explicit_owner_map_has_no_duplicate_keys():
    text = (REPO / "scripts" / "select_impacted_tests.py").read_text(encoding="utf-8")
    tree = compile(text, "select_impacted_tests.py", "exec", ast.PyCF_ONLY_AST)
    owner_map = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "EXPLICIT_TEST_OWNERS"
    )
    assert isinstance(owner_map.value, ast.Dict)
    keys = [
        key.value
        for key in owner_map.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]

    assert len(keys) == len(set(keys))


def test_explicit_ci_surface_with_missing_owner_escalates_to_full(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "scripts/pr_watcher.py": "VALUE = 1\n",
    })

    assert sel.select(["scripts/pr_watcher.py"], repo) == sel.FULL


def test_explicit_ci_surface_deletion_escalates_to_full(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "tests/test_pr_watcher.py": "def test_owner():\n    assert True\n",
    })

    assert sel.select(["scripts/pr_watcher.py"], repo) == sel.FULL


def test_explicit_ci_surface_owners_union_with_import_graph(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/svc.py": "VALUE = 1\n",
        "tests/test_svc.py": "from atlas_brain.svc import VALUE\n",
        "scripts/pr_watcher.py": "VALUE = 1\n",
        "tests/test_pr_watcher.py": "def test_owner():\n    assert True\n",
    })

    assert sel.select(["atlas_brain/svc.py", "scripts/pr_watcher.py"], repo) == [
        "tests/test_pr_watcher.py",
        "tests/test_svc.py",
    ]


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


def test_deleted_changed_path_escalates_to_full(tmp_path):
    """A deleted module's old import edges are absent from PR-head parsing."""
    repo = _mkrepo(tmp_path, {"atlas_brain/__init__.py": ""})
    assert sel.select(["atlas_brain/removed.py"], repo) == sel.FULL


def test_unknown_python_root_escalates_to_full(tmp_path):
    """New/omitted first-party roots must not be interpreted as test-free."""
    repo = _mkrepo(tmp_path, {
        "atlas_reddit/__init__.py": "",
        "atlas_reddit/config.py": "TOKEN = 'x'\n",
        "tests/test_atlas_reddit_config.py": "from atlas_reddit.config import TOKEN\n",
    })
    assert sel.select(["atlas_reddit/config.py"], repo) == sel.FULL


def test_runtime_asset_change_escalates_to_full(tmp_path):
    """YAML/JSON/etc. may be loaded at runtime without a Python import edge."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/skills/brand/brand_voice.yml": "tone: warm\n",
        "tests/test_brand_voice_validator.py": (
            "from pathlib import Path\n"
            "def test_asset():\n"
            "    assert Path('atlas_brain/skills/brand/brand_voice.yml')\n"
        ),
    })
    assert sel.select(["atlas_brain/skills/brand/brand_voice.yml"], repo) == sel.FULL


def test_workflow_change_escalates_to_full(tmp_path):
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        ".github/workflows/other_gate.yml": "name: other\n",
    })
    assert sel.select([".github/workflows/other_gate.yml"], repo) == sel.FULL


def test_script_change_escalates_for_path_based_loading(tmp_path):
    """Tests often load scripts via importlib/runpy/subprocess path strings."""
    repo = _mkrepo(tmp_path, {
        "scripts/audit_claims.py": "VALUE = 1\n",
        "tests/test_audit_claims.py": (
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location("
            "'audit_claims', 'scripts/audit_claims.py')\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
        ),
    })
    assert sel.select(["scripts/audit_claims.py"], repo) == sel.FULL


def test_conftest_dependency_escalates_to_full(tmp_path):
    """Once conftest is reached, fixture consumers are not in the import graph."""
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/fixture_source.py": "VALUE = 1\n",
        "tests/conftest.py": (
            "import pytest\n"
            "from atlas_brain.fixture_source import VALUE\n"
            "@pytest.fixture\n"
            "def shared_value():\n"
            "    return VALUE\n"
        ),
        "tests/test_fixture_user.py": (
            "def test_uses_fixture(shared_value):\n"
            "    assert shared_value == 1\n"
        ),
    })
    assert sel.select(["atlas_brain/fixture_source.py"], repo) == sel.FULL


def test_docs_only_change_selects_nothing(tmp_path):
    """The one case where empty is correct: mapped, and reachable from no test.

    The docs must EXIST in the head. A changed path absent from the head is a
    deletion and escalates to FULL, so a fixture that omits the files would
    pass for the wrong reason and hide a real docs-only regression.
    """
    repo = _mkrepo(tmp_path, {
        "atlas_brain/__init__.py": "",
        "atlas_brain/svc.py": "X = 1\n",
        "tests/test_svc.py": "from atlas_brain.svc import X\n",
        "docs/GUIDE.md": "# guide\n",
        "plans/PR-Thing.md": "# plan\n",
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


def test_growth_only_runs_growth_guard_without_pytest_report(tmp_path, capsys):
    baseline = tmp_path / "unit_gate_baseline.txt"
    base_baseline = tmp_path / "base_unit_gate_baseline.txt"
    baseline.write_text("tests/test_a.py::known\n", encoding="utf-8")
    base_baseline.write_text("tests/test_a.py::known\n", encoding="utf-8")

    assert gate.main([
        "--baseline", str(baseline),
        "--base-baseline", str(base_baseline),
        "--growth-only",
    ]) == 0
    assert "growth guard passed" in capsys.readouterr().out


def test_growth_only_still_rejects_baseline_growth(tmp_path):
    baseline = tmp_path / "unit_gate_baseline.txt"
    base_baseline = tmp_path / "base_unit_gate_baseline.txt"
    baseline.write_text(
        "tests/test_a.py::known\n"
        "tests/test_a.py::newly_added\n",
        encoding="utf-8",
    )
    base_baseline.write_text("tests/test_a.py::known\n", encoding="utf-8")

    assert gate.main([
        "--baseline", str(baseline),
        "--base-baseline", str(base_baseline),
        "--growth-only",
    ]) == 3


def test_scoped_run_allows_marker_filtered_no_tests_collected():
    gate.ensure_pytest_ran(5, allow_no_tests=True)


def test_unscoped_no_tests_collected_still_fails_infrastructure():
    with pytest.raises(RuntimeError):
        gate.ensure_pytest_ran(5)
