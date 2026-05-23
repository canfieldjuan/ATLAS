from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/extracted_pipeline_checks.yml"
RUNNER = ROOT / "scripts/run_extracted_pipeline_checks.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _workflow_install_command() -> str:
    text = _read(WORKFLOW)
    for match in re.finditer(r"python -m pip install (?P<packages>[^\n]+)", text):
        packages = match.group("packages")
        if "pytest" in packages.split():
            return packages
    raise AssertionError("extracted workflow must install test dependencies")


def test_extracted_pipeline_installs_fastapi_route_dependencies() -> None:
    packages = set(_workflow_install_command().split())

    assert "fastapi" in packages
    assert "python-multipart" in packages


def test_extracted_pipeline_runner_includes_route_tests() -> None:
    runner = _read(RUNNER)

    assert "tests/test_extracted_content_control_surface_api.py" in runner
    assert "tests/test_smoke_content_ops_ingestion_file_route.py" in runner
