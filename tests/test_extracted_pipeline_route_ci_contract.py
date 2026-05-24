from __future__ import annotations

import fnmatch
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/extracted_pipeline_checks.yml"
RUNNER = ROOT / "scripts/run_extracted_pipeline_checks.sh"
ENROLLED_TEST_PATTERNS = (
    "tests/test_extracted_content*.py",
    "tests/test_extracted_campaign*.py",
    "tests/test_extracted_blog*.py",
    "tests/test_extracted_landing_page*.py",
    "tests/test_extracted_report*.py",
    "tests/test_extracted_sales_brief*.py",
    "tests/test_extracted_ticket_faq*.py",
    "tests/test_extracted_support_ticket*.py",
    "tests/test_smoke_content_ops*.py",
    "tests/test_atlas_content_ops*.py",
    "tests/test_content_ops*.py",
    "tests/test_support_ticket*.py",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _event_path_filters(event_name: str) -> tuple[str, ...]:
    text = _read(WORKFLOW)
    start = re.search(rf"^  {re.escape(event_name)}:\n", text, re.MULTILINE)
    if start is None:
        raise AssertionError(f"extracted workflow missing {event_name} trigger")
    end = re.search(r"^  [a-z_]+:\n|^jobs:\n", text[start.end():], re.MULTILINE)
    block = (
        text[start.end():]
        if end is None
        else text[start.end():start.end() + end.start()]
    )
    return tuple(
        match.group("path")
        for match in re.finditer(r'^\s+- "(?P<path>[^"]+)"\s*$', block, re.MULTILINE)
    )


def _covered_by_path_filter(path: str, filters: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, item) for item in filters)


def _runner_test_paths() -> set[str]:
    return set(
        re.findall(r"tests/[A-Za-z0-9_./-]+\.py", _read(RUNNER))
    )


def _enrollment_candidate_tests() -> tuple[str, ...]:
    paths = {
        path.relative_to(ROOT).as_posix()
        for pattern in ENROLLED_TEST_PATTERNS
        for path in ROOT.glob(pattern)
    }
    return tuple(sorted(paths))


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


def test_extracted_pipeline_ci_enrolls_matching_tests() -> None:
    candidates = _enrollment_candidate_tests()
    assert candidates, "CI enrollment scanner matched zero test files"

    runner_paths = _runner_test_paths()
    missing_from_runner = [
        path for path in candidates if path not in runner_paths
    ]
    assert missing_from_runner == []

    pull_request_filters = _event_path_filters("pull_request")
    push_filters = _event_path_filters("push")
    missing_from_pull_request_filters = [
        path for path in candidates
        if not _covered_by_path_filter(path, pull_request_filters)
    ]
    missing_from_push_filters = [
        path for path in candidates
        if not _covered_by_path_filter(path, push_filters)
    ]

    assert missing_from_pull_request_filters == []
    assert missing_from_push_filters == []
