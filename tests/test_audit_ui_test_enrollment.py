from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_ui_test_enrollment.py"


def load_auditor():
    name = "audit_ui_test_enrollment"
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


def _make_ui(root: Path, name: str, test_scripts: list[str]) -> None:
    ui = root / name
    ui.mkdir(parents=True, exist_ok=True)
    scripts = {t: f"node --test scripts/{t}.test.mjs" for t in test_scripts}
    (ui / "package.json").write_text(
        json.dumps({"scripts": {"lint": "eslint .", **scripts}}),
        encoding="utf-8",
    )


def _make_workflow(root: Path, filename: str, run_tests: list[str]) -> None:
    wf_dir = root / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    steps = "\n".join(
        f"      - name: {t}\n        run: npm run {t}" for t in run_tests
    )
    (wf_dir / filename).write_text(
        f"name: checks\non:\n  pull_request:\njobs:\n  c:\n    steps:\n{steps}\n",
        encoding="utf-8",
    )


def _rows_by_ui(root: Path) -> dict:
    audit = load_auditor()
    return {row.ui: row for row in audit.audit_root(root)}


def test_happy_all_enrolled_is_not_drift(tmp_path):
    audit = load_auditor()
    _make_ui(tmp_path, "foo-ui", ["test:a", "test:b"])
    _make_workflow(tmp_path, "foo_ui_checks.yml", ["test:a", "test:b"])

    row = _rows_by_ui(tmp_path)["foo-ui"]
    assert row.status == "OK"
    assert row.missing == ()
    assert not audit.row_is_drift(row)


def test_declared_but_unrun_test_is_flagged(tmp_path):
    audit = load_auditor()
    _make_ui(tmp_path, "foo-ui", ["test:a", "test:b"])
    _make_workflow(tmp_path, "foo_ui_checks.yml", ["test:a"])  # test:b dropped

    row = _rows_by_ui(tmp_path)["foo-ui"]
    assert row.status == "UNENROLLED"
    assert row.missing == ("test:b",)
    assert audit.row_is_drift(row)


def test_tests_without_workflow_is_flagged(tmp_path):
    audit = load_auditor()
    _make_ui(tmp_path, "foo-ui", ["test:a"])
    # no foo_ui_checks.yml created

    row = _rows_by_ui(tmp_path)["foo-ui"]
    assert row.status == "MISSING_WORKFLOW"
    assert row.missing == ("test:a",)
    assert audit.row_is_drift(row)


def test_ui_without_test_scripts_is_not_drift(tmp_path):
    audit = load_auditor()
    _make_ui(tmp_path, "bare-ui", [])  # only a lint script

    row = _rows_by_ui(tmp_path)["bare-ui"]
    assert row.status == "NO_TESTS"
    assert not audit.row_is_drift(row)


def test_prefix_name_does_not_count_as_enrolled(tmp_path):
    # A longer-named run step must not satisfy a shorter declared test name.
    audit = load_auditor()
    _make_ui(tmp_path, "foo-ui", ["test:abc"])
    _make_workflow(tmp_path, "foo_ui_checks.yml", ["test:abcd"])

    row = _rows_by_ui(tmp_path)["foo-ui"]
    assert row.status == "UNENROLLED"
    assert row.missing == ("test:abc",)


def test_parse_test_scripts_raises_on_malformed_json():
    audit = load_auditor()
    with pytest.raises(ValueError):
        audit.parse_test_scripts("{not valid json")
    # Valid JSON whose scripts is not a dict has no test scripts (not an error).
    assert audit.parse_test_scripts(json.dumps({"scripts": "oops"})) == set()
    assert audit.parse_test_scripts(
        json.dumps({"scripts": {"test:x": "x", "build": "b"}})
    ) == {"test:x"}


def test_malformed_package_json_is_drift(tmp_path):
    audit = load_auditor()
    ui = tmp_path / "foo-ui"
    ui.mkdir(parents=True)
    (ui / "package.json").write_text("{not valid json", encoding="utf-8")

    row = _rows_by_ui(tmp_path)["foo-ui"]
    assert row.status == "MALFORMED_PACKAGE"
    assert audit.row_is_drift(row)


def test_npm_run_outside_run_step_is_not_counted(tmp_path):
    # test:b appears only in a step name and a YAML comment, never in a run body.
    audit = load_auditor()
    _make_ui(tmp_path, "foo-ui", ["test:a", "test:b"])
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "foo_ui_checks.yml").write_text(
        "name: c\n"
        "on:\n  pull_request:\n"
        "jobs:\n  c:\n    steps:\n"
        "      - name: mentions npm run test:b in the step name only\n"
        "        run: npm run test:a\n"
        "      # run: npm run test:b\n",
        encoding="utf-8",
    )

    row = _rows_by_ui(tmp_path)["foo-ui"]
    assert row.status == "UNENROLLED"
    assert row.missing == ("test:b",)


def test_workflow_name_convention():
    audit = load_auditor()
    assert audit.workflow_name_for("atlas-intel-ui") == "atlas_intel_ui_checks.yml"
    assert audit.workflow_name_for("portfolio-ui") == "portfolio_ui_checks.yml"
