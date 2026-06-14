from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/extracted_pipeline_checks.yml"
RUNNER = ROOT / "scripts/run_extracted_pipeline_checks.sh"
AUDITOR = ROOT / "scripts/audit_extracted_pipeline_ci_enrollment.py"


def _load_ci_enrollment_auditor():
    name = "audit_extracted_pipeline_ci_enrollment"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, AUDITOR)
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


def _read_extracted_ci_contract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _workflow_install_command() -> str:
    text = _read_extracted_ci_contract_text(WORKFLOW)
    for match in re.finditer(r"python -m pip install (?P<packages>[^\n]+)", text):
        packages = match.group("packages")
        if "pytest" in packages.split():
            return packages
    raise AssertionError("extracted workflow must install test dependencies")


def test_extracted_pipeline_installs_fastapi_route_dependencies() -> None:
    packages = set(shlex.split(_workflow_install_command()))

    assert "fastapi<0.137" in packages
    assert "fastapi" not in packages
    assert "python-multipart" in packages


def test_extracted_pipeline_runner_includes_route_tests() -> None:
    runner = _read_extracted_ci_contract_text(RUNNER)

    assert "tests/test_extracted_content_control_surface_api.py" in runner
    assert "tests/test_smoke_content_ops_ingestion_file_route.py" in runner


def test_extracted_pipeline_ci_enrolls_matching_tests() -> None:
    audit = _load_ci_enrollment_auditor().audit_ci_enrollment(
        ROOT,
        workflow_path=WORKFLOW,
        runner_path=RUNNER,
    )

    assert audit.failures == ()
