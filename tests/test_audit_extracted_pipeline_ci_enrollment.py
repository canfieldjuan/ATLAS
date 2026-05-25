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
