from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_extracted_pipeline_ci_enrollment.py"


def _load_ci_enrollment_auditor():
    name = "audit_extracted_pipeline_ci_enrollment"
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


def _repo(
    tmp_path: Path,
    *,
    runner_paths: tuple[str, ...] = ("tests/test_extracted_content_alpha.py",),
    pull_request_filters: tuple[str, ...] = ("tests/test_extracted_content_*.py",),
    push_filters: tuple[str, ...] = ("tests/test_extracted_content_*.py",),
    include_push: bool = True,
    include_candidate: bool = True,
) -> Path:
    root = tmp_path / "repo"
    (root / ".github/workflows").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    if include_candidate:
        (root / "tests/test_extracted_content_alpha.py").write_text(
            "def test_alpha():\n    assert True\n",
            encoding="utf-8",
        )

    runner = "pytest \\\n" + "".join(f"  {path} \\\n" for path in runner_paths)
    (root / "scripts/run_extracted_pipeline_checks.sh").write_text(
        runner,
        encoding="utf-8",
    )

    workflow = _workflow_text(
        pull_request_filters=pull_request_filters,
        push_filters=push_filters,
        include_push=include_push,
    )
    (root / ".github/workflows/extracted_pipeline_checks.yml").write_text(
        workflow,
        encoding="utf-8",
    )
    return root


def _workflow_text(
    *,
    pull_request_filters: tuple[str, ...],
    push_filters: tuple[str, ...],
    include_push: bool,
) -> str:
    lines = [
        "name: Extracted Pipeline Checks",
        "",
        "on:",
        "  pull_request:",
        "    paths:",
    ]
    lines.extend(f'      - "{path}"' for path in pull_request_filters)
    if include_push:
        lines.extend(["  push:", "    paths:"])
        lines.extend(f'      - "{path}"' for path in push_filters)
    lines.extend(
        [
            "",
            "jobs:",
            "  extracted-checks:",
            "    runs-on: ubuntu-latest",
            "",
        ]
    )
    return "\n".join(lines)


def test_audit_accepts_enrolled_candidate(tmp_path: Path) -> None:
    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(_repo(tmp_path))

    assert audit.ok
    assert audit.candidates == ("tests/test_extracted_content_alpha.py",)


def test_audit_fixture_test_is_a_candidate(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        runner_paths=("tests/test_audit_extracted_pipeline_ci_enrollment.py",),
        pull_request_filters=("tests/test_audit_extracted_pipeline_ci_enrollment.py",),
        push_filters=("tests/test_audit_extracted_pipeline_ci_enrollment.py",),
        include_candidate=False,
    )
    (root / "tests/test_audit_extracted_pipeline_ci_enrollment.py").write_text(
        "def test_audit_fixture():\n    assert True\n",
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(root)

    assert audit.ok
    assert audit.candidates == ("tests/test_audit_extracted_pipeline_ci_enrollment.py",)


def test_audit_checker_and_evaluator_tests_are_candidates(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        runner_paths=(
            "tests/test_check_content_ops_faq_search_route_contract.py",
            "tests/test_evaluate_support_ticket_generated_content.py",
        ),
        pull_request_filters=(
            "tests/test_check_content_ops_*.py",
            "tests/test_evaluate_support_ticket_*.py",
        ),
        push_filters=(
            "tests/test_check_content_ops_*.py",
            "tests/test_evaluate_support_ticket_*.py",
        ),
        include_candidate=False,
    )
    (root / "tests/test_check_content_ops_faq_search_route_contract.py").write_text(
        "def test_checker():\n    assert True\n",
        encoding="utf-8",
    )
    (root / "tests/test_evaluate_support_ticket_generated_content.py").write_text(
        "def test_evaluator():\n    assert True\n",
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(root)

    assert audit.ok
    assert audit.candidates == (
        "tests/test_check_content_ops_faq_search_route_contract.py",
        "tests/test_evaluate_support_ticket_generated_content.py",
    )


def test_audit_card_generation_tests_are_candidates(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        runner_paths=(
            "tests/test_extracted_quote_card_generation.py",
            "tests/test_extracted_stat_card_generation.py",
        ),
        pull_request_filters=("tests/test_extracted_*_card_generation.py",),
        push_filters=("tests/test_extracted_*_card_generation.py",),
        include_candidate=False,
    )
    (root / "tests/test_extracted_quote_card_generation.py").write_text(
        "def test_quote_card():\n    assert True\n",
        encoding="utf-8",
    )
    (root / "tests/test_extracted_stat_card_generation.py").write_text(
        "def test_stat_card():\n    assert True\n",
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(root)

    assert audit.ok
    assert audit.candidates == (
        "tests/test_extracted_quote_card_generation.py",
        "tests/test_extracted_stat_card_generation.py",
    )


def test_audit_reports_missing_runner_entry(tmp_path: Path) -> None:
    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        _repo(tmp_path, runner_paths=())
    )

    assert audit.missing_from_runner == ("tests/test_extracted_content_alpha.py",)
    assert not audit.ok


def test_audit_reports_missing_pull_request_filter(tmp_path: Path) -> None:
    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        _repo(tmp_path, pull_request_filters=("tests/test_extracted_blog_*.py",))
    )

    assert audit.missing_from_pull_request_filters == (
        "tests/test_extracted_content_alpha.py",
    )
    assert not audit.ok


def test_audit_reports_missing_push_filter(tmp_path: Path) -> None:
    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        _repo(tmp_path, push_filters=("tests/test_extracted_blog_*.py",))
    )

    assert audit.missing_from_push_filters == (
        "tests/test_extracted_content_alpha.py",
    )
    assert not audit.ok


def test_audit_reports_zero_candidate_scan(tmp_path: Path) -> None:
    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        _repo(tmp_path, include_candidate=False)
    )

    assert "CI enrollment scanner matched zero test files" in audit.failures
    assert not audit.ok


def test_audit_reports_missing_workflow_event(tmp_path: Path) -> None:
    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        _repo(tmp_path, include_push=False)
    )

    assert "extracted workflow missing push trigger" in audit.workflow_errors
    assert not audit.ok


def test_cli_returns_nonzero_for_incomplete_enrollment(tmp_path: Path) -> None:
    repo = _repo(tmp_path, runner_paths=())

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing from runner" in result.stdout


def _write_atlas_workflow(
    root: Path,
    *,
    workflow_name: str = "atlas_widget_checks.yml",
    pull_request_filters: tuple[str, ...] = ("tests/test_atlas_widget.py",),
    push_filters: tuple[str, ...] = ("tests/test_atlas_widget.py",),
    runner_paths: tuple[str, ...] = ("tests/test_atlas_widget.py",),
    inline_run: bool = False,
) -> None:
    lines = [
        "name: Atlas Widget Checks",
        "",
        "on:",
        "  pull_request:",
        "    paths:",
    ]
    lines.extend(f'      - "{path}"' for path in pull_request_filters)
    lines.extend(["  push:", "    paths:"])
    lines.extend(f'      - "{path}"' for path in push_filters)
    lines.extend(
        [
            "",
            "jobs:",
            "  atlas-widget-checks:",
            "    runs-on: ubuntu-latest",
            "    steps:",
        ]
    )
    if inline_run:
        lines.append("      - run: python -m pytest " + " ".join(runner_paths) + " -q")
    else:
        lines.extend(
            [
                "      - name: Run atlas widget tests",
                "        run: python -m pytest " + " ".join(runner_paths) + " -q",
            ]
        )
    lines.append("")
    (root / f".github/workflows/{workflow_name}").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _write_atlas_importing_test(root: Path) -> str:
    path = "tests/test_atlas_widget.py"
    (root / path).write_text(
        "from atlas_brain.widgets import build_widget\n\n"
        "def test_widget():\n"
        "    assert build_widget\n",
        encoding="utf-8",
    )
    return path


def test_atlas_brain_changed_test_accepts_dedicated_workflow(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    _write_atlas_workflow(root)

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.ok
    assert audit.atlas_brain_test_errors == ()


def test_atlas_brain_changed_test_accepts_inline_run_step(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    _write_atlas_workflow(root, inline_run=True)

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.ok
    assert audit.atlas_brain_test_errors == ()


def _write_repo_wide_backstop(root: Path) -> None:
    (root / ".github/workflows/repo_wide_unit_backstop.yml").write_text(
        "name: Repo-Wide Unit Backstop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  repo-wide-unit-backstop:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        '      - run: python -m pytest tests/ -m "not integration and not e2e" -q\n',
        encoding="utf-8",
    )


def test_atlas_brain_changed_test_accepts_repo_wide_backstop(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    # No dedicated atlas_*_checks.yml; the repo-wide backstop is the catch-all.
    _write_repo_wide_backstop(root)

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.ok
    assert audit.atlas_brain_test_errors == ()


def test_atlas_brain_integration_test_exempt_from_unit_enrollment(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    # An integration-marked atlas_brain test is service-lane: not subject to the
    # unit-workflow enrollment requirement, and the unit backstop intentionally
    # skips it. No atlas workflow, no backstop -- still exempt.
    path = "tests/test_atlas_widget.py"
    (root / path).write_text(
        "import pytest\n"
        "from atlas_brain.widgets import build_widget\n\n"
        "pytestmark = pytest.mark.integration\n\n"
        "def test_widget():\n"
        "    assert build_widget\n",
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(path,),
    )

    assert audit.ok
    assert audit.atlas_brain_test_errors == ()


def test_atlas_brain_changed_test_rejects_comment_only_backstop(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    # A backstop file whose marker text appears only in a comment (no real
    # pytest command) must NOT credit coverage.
    (root / ".github/workflows/repo_wide_unit_backstop.yml").write_text(
        "name: Repo-Wide Unit Backstop\n"
        "# pytest run (not integration and not e2e) was removed from this file\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  repo-wide-unit-backstop:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo disabled\n",
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert not audit.ok
    assert audit.atlas_brain_test_errors != ()


def test_atlas_brain_changed_test_rejects_run_block_comment_backstop(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    # The pytest command appears only as a shell comment inside a `run: |`
    # block -- not a real invocation, so it must not credit the backstop.
    (root / ".github/workflows/repo_wide_unit_backstop.yml").write_text(
        "name: Repo-Wide Unit Backstop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  repo-wide-unit-backstop:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: |\n"
        '          # python -m pytest -m "not integration and not e2e" -q (off)\n'
        "          echo disabled\n",
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert not audit.ok
    assert audit.atlas_brain_test_errors != ()


def test_atlas_brain_changed_test_rejects_directory_scoped_backstop(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    # A directory-scoped run has the marker but is not the repo-wide catch-all.
    (root / ".github/workflows/repo_wide_unit_backstop.yml").write_text(
        "name: Repo-Wide Unit Backstop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  repo-wide-unit-backstop:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        '      - run: python -m pytest tests/unit/ -m "not integration and not e2e" -q\n',
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert not audit.ok
    assert audit.atlas_brain_test_errors != ()


def test_atlas_brain_docstring_marker_mention_not_exempt(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    path = "tests/test_atlas_widget.py"
    # A docstring/comment mention of the marker is not a real marker; the test
    # must NOT be exempted from unit enrollment.
    (root / path).write_text(
        '"""Example that mentions pytest.mark.integration in prose."""\n'
        "from atlas_brain.widgets import build_widget\n\n"
        "def test_widget():\n"
        "    assert build_widget\n",
        encoding="utf-8",
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(path,),
    )

    assert not audit.ok
    assert audit.atlas_brain_test_errors != ()


def test_atlas_brain_changed_test_requires_atlas_workflow(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.atlas_brain_test_errors == (
        "tests/test_atlas_widget.py imports atlas_brain.* but no "
        "atlas_*_checks.yml workflow exists",
    )
    assert not audit.ok


def test_atlas_brain_changed_test_reports_missing_pr_filter(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    _write_atlas_workflow(root, pull_request_filters=("tests/test_other.py",))

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.atlas_brain_test_errors == (
        "tests/test_atlas_widget.py imports atlas_brain.* but is missing "
        "dedicated atlas workflow enrollment: pull_request path filter",
    )
    assert not audit.ok


def test_atlas_brain_changed_test_reports_missing_push_filter(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    _write_atlas_workflow(root, push_filters=("tests/test_other.py",))

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.atlas_brain_test_errors == (
        "tests/test_atlas_widget.py imports atlas_brain.* but is missing "
        "dedicated atlas workflow enrollment: push path filter",
    )
    assert not audit.ok


def test_atlas_brain_changed_test_reports_missing_pytest_runner(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    _write_atlas_workflow(root, runner_paths=("tests/test_other.py",))

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.atlas_brain_test_errors == (
        "tests/test_atlas_widget.py imports atlas_brain.* but is missing "
        "dedicated atlas workflow enrollment: pytest run step",
    )
    assert not audit.ok


def test_atlas_brain_changed_test_rejects_split_workflow_enrollment(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    test_path = _write_atlas_importing_test(root)
    _write_atlas_workflow(
        root,
        workflow_name="atlas_widget_pr_checks.yml",
        push_filters=("tests/test_other.py",),
        runner_paths=("tests/test_other.py",),
    )
    _write_atlas_workflow(
        root,
        workflow_name="atlas_widget_push_checks.yml",
        pull_request_filters=("tests/test_other.py",),
        runner_paths=("tests/test_other.py",),
    )
    _write_atlas_workflow(
        root,
        workflow_name="atlas_widget_run_checks.yml",
        pull_request_filters=("tests/test_other.py",),
        push_filters=("tests/test_other.py",),
    )

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(test_path,),
    )

    assert audit.atlas_brain_test_errors == (
        "tests/test_atlas_widget.py imports atlas_brain.* but is missing "
        "dedicated atlas workflow enrollment: single atlas workflow with "
        "pull_request path filter, push path filter, and pytest run step",
    )
    assert not audit.ok


def test_non_atlas_brain_changed_test_does_not_need_atlas_workflow(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    path = "tests/test_plain_widget.py"
    (root / path).write_text("def test_plain():\n    assert True\n", encoding="utf-8")

    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        root,
        atlas_brain_test_paths=(path,),
    )

    assert audit.ok
    assert audit.atlas_brain_test_errors == ()
