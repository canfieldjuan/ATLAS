from __future__ import annotations

import argparse
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ENROLLED_TEST_PATTERNS = (
    "tests/test_audit_extracted_pipeline_ci_enrollment.py",
    "tests/test_extracted_content*.py",
    "tests/test_extracted_campaign*.py",
    "tests/test_extracted_blog*.py",
    "tests/test_extracted_landing_page*.py",
    "tests/test_extracted_report*.py",
    "tests/test_extracted_sales_brief*.py",
    "tests/test_extracted_ticket_faq*.py",
    "tests/test_extracted_support_ticket*.py",
    "tests/test_smoke_content_ops*.py",
    "tests/test_check_content_ops*.py",
    "tests/test_atlas_content_ops*.py",
    "tests/test_content_ops*.py",
    "tests/test_support_ticket*.py",
    "tests/test_evaluate_*.py",
)


@dataclass(frozen=True)
class EnrollmentAudit:
    candidates: tuple[str, ...]
    missing_from_runner: tuple[str, ...]
    missing_from_pull_request_filters: tuple[str, ...]
    missing_from_push_filters: tuple[str, ...]
    workflow_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.candidates:
            failures.append("CI enrollment scanner matched zero test files")
        failures.extend(self.workflow_errors)
        if self.missing_from_runner:
            failures.append(
                "missing from runner: " + ", ".join(self.missing_from_runner)
            )
        if self.missing_from_pull_request_filters:
            failures.append(
                "missing from pull_request filters: "
                + ", ".join(self.missing_from_pull_request_filters)
            )
        if self.missing_from_push_filters:
            failures.append(
                "missing from push filters: "
                + ", ".join(self.missing_from_push_filters)
            )
        return tuple(failures)


def event_path_filters(workflow_text: str, event_name: str) -> tuple[str, ...]:
    start = re.search(rf"^  {re.escape(event_name)}:\n", workflow_text, re.MULTILINE)
    if start is None:
        raise ValueError(f"extracted workflow missing {event_name} trigger")
    end = re.search(r"^  [a-z_]+:\n|^jobs:\n", workflow_text[start.end():], re.MULTILINE)
    block = (
        workflow_text[start.end():]
        if end is None
        else workflow_text[start.end():start.end() + end.start()]
    )
    return tuple(
        match.group("path")
        for match in re.finditer(r'^\s+- "(?P<path>[^"]+)"\s*$', block, re.MULTILINE)
    )


def runner_test_paths(runner_text: str) -> set[str]:
    return set(re.findall(r"tests/[A-Za-z0-9_./-]+\.py", runner_text))


def enrollment_candidate_tests(
    root: Path,
    patterns: Iterable[str] = DEFAULT_ENROLLED_TEST_PATTERNS,
) -> tuple[str, ...]:
    paths = {
        path.relative_to(root).as_posix()
        for pattern in patterns
        for path in root.glob(pattern)
    }
    return tuple(sorted(paths))


def covered_by_path_filter(path: str, filters: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, item) for item in filters)


def audit_ci_enrollment(
    root: Path,
    *,
    workflow_path: Path | None = None,
    runner_path: Path | None = None,
    patterns: Iterable[str] = DEFAULT_ENROLLED_TEST_PATTERNS,
) -> EnrollmentAudit:
    workflow = workflow_path or root / ".github/workflows/extracted_pipeline_checks.yml"
    runner = runner_path or root / "scripts/run_extracted_pipeline_checks.sh"
    candidates = enrollment_candidate_tests(root, patterns)
    runner_paths = runner_test_paths(runner.read_text(encoding="utf-8"))
    workflow_text = workflow.read_text(encoding="utf-8")
    workflow_errors: list[str] = []

    try:
        pull_request_filters = event_path_filters(workflow_text, "pull_request")
    except ValueError as exc:
        pull_request_filters = ()
        workflow_errors.append(str(exc))

    try:
        push_filters = event_path_filters(workflow_text, "push")
    except ValueError as exc:
        push_filters = ()
        workflow_errors.append(str(exc))

    return EnrollmentAudit(
        candidates=candidates,
        missing_from_runner=tuple(
            path for path in candidates if path not in runner_paths
        ),
        missing_from_pull_request_filters=tuple(
            path for path in candidates
            if not covered_by_path_filter(path, pull_request_filters)
        ),
        missing_from_push_filters=tuple(
            path for path in candidates
            if not covered_by_path_filter(path, push_filters)
        ),
        workflow_errors=tuple(workflow_errors),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit extracted pipeline test enrollment."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root to audit.",
    )
    args = parser.parse_args(argv)
    audit = audit_ci_enrollment(args.root)
    if audit.ok:
        print(f"OK: {len(audit.candidates)} matching tests are enrolled.")
        return 0

    print("DRIFT: extracted pipeline CI enrollment is incomplete.")
    for failure in audit.failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
