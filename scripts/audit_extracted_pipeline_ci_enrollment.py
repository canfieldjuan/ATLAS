from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
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
    atlas_brain_test_errors: tuple[str, ...] = ()
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
        failures.extend(self.atlas_brain_test_errors)
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


def workflow_run_test_paths(workflow_text: str) -> set[str]:
    paths: set[str] = set()
    lines = workflow_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"^(?P<indent>\s*)(?:-\s*)?run:\s*(?P<command>.*)$", line)
        if match is None:
            index += 1
            continue

        indent = len(match.group("indent"))
        command_lines = [match.group("command")]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= indent:
                break
            command_lines.append(next_line)
            index += 1
        paths.update(runner_test_paths("\n".join(command_lines)))
    return paths


def changed_test_paths(root: Path, base_ref: str) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=AM",
            f"{base_ref}...HEAD",
            "--",
            "tests/test_*.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("tests/test_") and line.strip().endswith(".py")
    )


def imports_atlas_brain(test_text: str) -> bool:
    return bool(
        re.search(
            r"^\s*(?:from\s+atlas_brain(?:\.|\s+import)|import\s+atlas_brain(?:\.|\s|$))",
            test_text,
            re.MULTILINE,
        )
    )


def atlas_brain_importing_tests(
    root: Path,
    test_paths: Iterable[str],
) -> tuple[str, ...]:
    matches: list[str] = []
    for path in test_paths:
        test_path = root / path
        if test_path.is_file() and imports_atlas_brain(
            test_path.read_text(encoding="utf-8")
        ):
            matches.append(path)
    return tuple(sorted(matches))


@dataclass(frozen=True)
class AtlasWorkflowEnrollment:
    pull_request_paths: tuple[str, ...]
    push_paths: tuple[str, ...]
    runner_paths: frozenset[str]


def workflow_covers_atlas_test(workflow: AtlasWorkflowEnrollment, path: str) -> bool:
    return (
        covered_by_path_filter(path, workflow.pull_request_paths)
        and covered_by_path_filter(path, workflow.push_paths)
        and path in workflow.runner_paths
    )


def atlas_workflow_enrollments(root: Path) -> tuple[AtlasWorkflowEnrollment, ...]:
    workflows: list[AtlasWorkflowEnrollment] = []
    for workflow_path in sorted((root / ".github/workflows").glob("atlas_*_checks.yml")):
        workflow_text = workflow_path.read_text(encoding="utf-8")
        try:
            pull_request_paths = event_path_filters(workflow_text, "pull_request")
            push_paths = event_path_filters(workflow_text, "push")
        except ValueError:
            continue
        workflows.append(
            AtlasWorkflowEnrollment(
                pull_request_paths=pull_request_paths,
                push_paths=push_paths,
                runner_paths=frozenset(workflow_run_test_paths(workflow_text)),
            )
        )
    return tuple(workflows)


def atlas_brain_test_workflow_errors(
    root: Path,
    test_paths: Iterable[str],
) -> tuple[str, ...]:
    workflows = atlas_workflow_enrollments(root)
    errors: list[str] = []
    for path in atlas_brain_importing_tests(root, test_paths):
        if not workflows:
            errors.append(
                f"{path} imports atlas_brain.* but no atlas_*_checks.yml workflow exists"
            )
            continue
        if any(workflow_covers_atlas_test(workflow, path) for workflow in workflows):
            continue
        missing: list[str] = []
        if not any(
            covered_by_path_filter(path, workflow.pull_request_paths)
            for workflow in workflows
        ):
            missing.append("pull_request path filter")
        if not any(
            covered_by_path_filter(path, workflow.push_paths) for workflow in workflows
        ):
            missing.append("push path filter")
        if not any(path in workflow.runner_paths for workflow in workflows):
            missing.append("pytest run step")
        if not missing:
            missing.append(
                "single atlas workflow with pull_request path filter, "
                "push path filter, and pytest run step"
            )
        errors.append(
            f"{path} imports atlas_brain.* but is missing dedicated atlas workflow enrollment: "
            + ", ".join(missing)
        )
    return tuple(errors)


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
    atlas_brain_test_paths: Iterable[str] = (),
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
        atlas_brain_test_errors=atlas_brain_test_workflow_errors(
            root,
            atlas_brain_test_paths,
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
    parser.add_argument(
        "--atlas-brain-tests-from",
        metavar="BASE_REF",
        help=(
            "Also audit changed tests importing atlas_brain.* against dedicated "
            "atlas_*_checks.yml workflow enrollment."
        ),
    )
    args = parser.parse_args(argv)
    atlas_brain_test_paths = (
        changed_test_paths(args.root, args.atlas_brain_tests_from)
        if args.atlas_brain_tests_from
        else ()
    )
    audit = audit_ci_enrollment(
        args.root,
        atlas_brain_test_paths=atlas_brain_test_paths,
    )
    if audit.ok:
        print(f"OK: {len(audit.candidates)} matching tests are enrolled.")
        return 0

    print("DRIFT: extracted pipeline CI enrollment is incomplete.")
    for failure in audit.failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
