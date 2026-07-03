#!/usr/bin/env python3
"""Advisory detector for retired autonomous-coding failure modes.

This tool notices likely recurrences; it does not block merges. Findings are
signals for a later ledger/reviewer disposition step.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = 1
DETECTOR_VERSION = "retired-failure-mode-detector.v1"
MODE_PLAN_WEAKENING = "plan_weakening"
MODE_TEST_WEAKENING = "test_weakening"
MODE_SCOPE_DRIFT = "scope_drift"
MODE_SYMPTOM_PATCHING = "symptom_patching"

OBLIGATION_RE = re.compile(
    r"\b("
    r"acceptance|must|must not|required|requirement|fail(?:s|ed|ing)?|"
    r"block(?:s|ed|ing)?|negative|edge|adversarial|malformed|boundary|"
    r"regression|reviewer rules|risk areas|source of truth|real adapter"
    r")\b",
    re.IGNORECASE,
)
ASSERTION_RE = re.compile(
    r"\b(assert|pytest\.raises|parametrize|fixture|expect\(|toEqual|toStrictEqual|"
    r"toThrow|not\.to|rejects|raises)\b"
)
TEST_SKIP_RE = re.compile(r"\b(skip|skipif|xfail|todo|only)\b", re.IGNORECASE)
ROOT_CAUSE_RE = re.compile(r"\broot cause\b", re.IGNORECASE)
FIX_WORD_RE = re.compile(
    r"(?<![-\w])(?:fix(?:es|ed|ing)?|bug|defect|regression|review finding|blocker)\b",
    re.IGNORECASE,
)
DOWNSTREAM_PATH_RE = re.compile(r"(^|/)(api|routes|views|ui|templates|render|presentation|adapter)s?(/|$)")
UPSTREAM_PATH_RE = re.compile(r"(^|/)(domain|core|services|models|schemas|store|storage|pipeline)s?(/|$)")
PATH_TOKEN_RE = re.compile(r"`([^`\n]+)`")


@dataclass(frozen=True)
class Signal:
    mode: str
    signature: str
    confidence: str
    paths: tuple[str, ...]
    evidence: tuple[str, ...]
    explanation: str
    detector_version: str = DETECTOR_VERSION


def git(args: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def merge_base(base_ref: str, *, cwd: Path) -> str:
    return git(["merge-base", "HEAD", base_ref], cwd=cwd).strip()


def head_sha(*, cwd: Path) -> str:
    return git(["rev-parse", "HEAD"], cwd=cwd).strip()


def changed_paths(base: str, *, cwd: Path) -> tuple[str, ...]:
    output = git(["diff", "--name-only", f"{base}...HEAD"], cwd=cwd)
    return tuple(line for line in output.splitlines() if line.strip())


def diff_lines(base: str, path: str, *, cwd: Path) -> tuple[str, ...]:
    output = git(["diff", "--unified=0", f"{base}...HEAD", "--", path], cwd=cwd)
    return tuple(output.splitlines())


def changed_line_groups(base: str, path: str, *, cwd: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    removed: list[str] = []
    added: list[str] = []
    for line in diff_lines(base, path, cwd=cwd):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
    return (("removed", tuple(removed)), ("added", tuple(added)))


def file_at(ref: str, path: str, *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def read_worktree(path: str, *, cwd: Path) -> str:
    try:
        return (cwd / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def is_plan_path(path: str) -> bool:
    return path.startswith("plans/PR-") and path.endswith(".md")


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name.lower()
    return (
        "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )


def is_code_path(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix not in {".py", ".js", ".jsx", ".ts", ".tsx", ".sh"}:
        return False
    return not is_test_path(path) and not is_plan_path(path)


def evidence(lines: Iterable[str], pattern: re.Pattern[str], *, limit: int = 4) -> tuple[str, ...]:
    hits = []
    for line in lines:
        stripped = line.strip()
        if stripped and pattern.search(stripped):
            hits.append(stripped[:180])
        if len(hits) >= limit:
            break
    return tuple(hits)


def claimed_files_from_plan(text: str) -> set[str]:
    in_files = False
    claimed: set[str] = set()
    for line in text.splitlines():
        if line.startswith("## "):
            in_files = False
        elif line.strip().lower() == "### files touched":
            in_files = True
            continue
        if in_files:
            claimed.update(token.strip() for token in PATH_TOKEN_RE.findall(line) if token.strip())
    return claimed


def detect_plan_weakening(base: str, paths: Sequence[str], *, cwd: Path) -> list[Signal]:
    plan_paths = [path for path in paths if is_plan_path(path)]
    code_paths = [path for path in paths if is_code_path(path)]
    if not plan_paths or not code_paths:
        return []

    signals: list[Signal] = []
    for plan_path in plan_paths:
        groups = dict(changed_line_groups(base, plan_path, cwd=cwd))
        removed_obligations = evidence(groups["removed"], OBLIGATION_RE)
        if not removed_obligations:
            continue
        signals.append(
            Signal(
                mode=MODE_PLAN_WEAKENING,
                signature="plan_obligation_removed_with_code_change",
                confidence="medium",
                paths=tuple([plan_path, *code_paths[:6]]),
                evidence=removed_obligations,
                explanation=(
                    "A plan obligation disappeared in the same diff as code changes. "
                    "This can be legitimate, but it is the known shape of plan weakening."
                ),
            )
        )
    return signals


def detect_test_weakening(base: str, paths: Sequence[str], *, cwd: Path) -> list[Signal]:
    test_paths = [path for path in paths if is_test_path(path)]
    code_paths = [path for path in paths if is_code_path(path)]
    if not test_paths or not code_paths:
        return []

    signals: list[Signal] = []
    for test_path in test_paths:
        groups = dict(changed_line_groups(base, test_path, cwd=cwd))
        removed_assertions = evidence(groups["removed"], ASSERTION_RE)
        added_skips = evidence(groups["added"], TEST_SKIP_RE)
        if not removed_assertions and not added_skips:
            continue
        signal_evidence = tuple([*removed_assertions, *added_skips])
        signature = "test_assertion_removed_with_code_change"
        if added_skips:
            signature = "test_skip_or_xfail_added_with_code_change"
        signals.append(
            Signal(
                mode=MODE_TEST_WEAKENING,
                signature=signature,
                confidence="medium",
                paths=tuple([test_path, *code_paths[:6]]),
                evidence=signal_evidence,
                explanation=(
                    "A test lost assertions/cases or gained skip-like behavior while "
                    "production code changed. This is the known shape of test weakening."
                ),
            )
        )
    return signals


def detect_scope_drift(base: str, paths: Sequence[str], *, cwd: Path) -> list[Signal]:
    plan_paths = [path for path in paths if is_plan_path(path)]
    non_plan_paths = {path for path in paths if not is_plan_path(path)}
    signals: list[Signal] = []

    if not plan_paths and any(is_code_path(path) for path in paths):
        signals.append(
            Signal(
                mode=MODE_SCOPE_DRIFT,
                signature="code_change_without_plan_doc",
                confidence="low",
                paths=tuple(sorted(path for path in paths if is_code_path(path))[:8]),
                evidence=("No changed plans/PR-*.md file found for code changes.",),
                explanation=(
                    "Code changed without a visible slice plan. Dependabot and trivial "
                    "changes may be valid exceptions, but unplanned code is a scope-drift shape."
                ),
            )
        )
        return signals

    for plan_path in plan_paths:
        plan_text = read_worktree(plan_path, cwd=cwd)
        claimed = claimed_files_from_plan(plan_text)
        if not claimed:
            continue
        missing = sorted(non_plan_paths - claimed)
        if missing:
            signals.append(
                Signal(
                    mode=MODE_SCOPE_DRIFT,
                    signature="changed_files_outside_plan_files_touched",
                    confidence="high",
                    paths=tuple([plan_path, *missing[:8]]),
                    evidence=tuple(f"outside Files touched: {path}" for path in missing[:6]),
                    explanation=(
                        "The diff touches files not declared in the plan's Files touched section. "
                        "The hard plan/code gate may also catch this; this detector records recurrence."
                    ),
                )
            )
    return signals


def detect_symptom_patching(base: str, paths: Sequence[str], *, cwd: Path) -> list[Signal]:
    plan_paths = [path for path in paths if is_plan_path(path)]
    code_paths = [path for path in paths if is_code_path(path)]
    if not plan_paths or not code_paths:
        return []

    downstream = [path for path in code_paths if DOWNSTREAM_PATH_RE.search(path)]
    upstream = [path for path in code_paths if UPSTREAM_PATH_RE.search(path)]
    signals: list[Signal] = []
    for plan_path in plan_paths:
        current = read_worktree(plan_path, cwd=cwd)
        prior = file_at(base, plan_path, cwd=cwd)
        plan_text = current or prior
        if not FIX_WORD_RE.search(plan_text):
            continue
        if ROOT_CAUSE_RE.search(plan_text) and upstream:
            continue
        if not ROOT_CAUSE_RE.search(plan_text):
            signals.append(
                Signal(
                    mode=MODE_SYMPTOM_PATCHING,
                    signature="fix_plan_missing_root_cause_language",
                    confidence="low",
                    paths=tuple([plan_path, *code_paths[:8]]),
                    evidence=("Fix-type plan text found without explicit 'root cause' language.",),
                    explanation=(
                        "Fix-type work should name the upstream root cause before code. "
                        "This is a weak signal because some feature plans use fix-like words casually."
                    ),
                )
            )
        elif downstream and not upstream:
            signals.append(
                Signal(
                    mode=MODE_SYMPTOM_PATCHING,
                    signature="root_cause_fix_changes_downstream_only",
                    confidence="medium",
                    paths=tuple([plan_path, *downstream[:8]]),
                    evidence=tuple(f"downstream-only changed path: {path}" for path in downstream[:6]),
                    explanation=(
                        "A fix-type PR names root cause but appears localized to downstream "
                        "presentation/adapter paths. That can be valid, but it is the known symptom-patch shape."
                    ),
                )
            )
    return signals


def build_report(base_ref: str, *, cwd: Path) -> dict[str, object]:
    base = merge_base(base_ref, cwd=cwd)
    paths = changed_paths(base, cwd=cwd)
    signals: list[Signal] = []
    signals.extend(detect_plan_weakening(base, paths, cwd=cwd))
    signals.extend(detect_test_weakening(base, paths, cwd=cwd))
    signals.extend(detect_scope_drift(base, paths, cwd=cwd))
    signals.extend(detect_symptom_patching(base, paths, cwd=cwd))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{head_sha(cwd=cwd)[:12]}:{base_ref}",
        "base_ref": base_ref,
        "head_sha": head_sha(cwd=cwd),
        "signal_type": "retired_failure_recurrence",
        "detector_version": DETECTOR_VERSION,
        "signals": [asdict(signal) for signal in signals],
    }


def print_summary(report: dict[str, object]) -> None:
    signals = report["signals"]
    if not isinstance(signals, list) or not signals:
        print("retired-failure-mode detector: no recurrence signatures detected.")
        return
    print(
        f"retired-failure-mode detector: {len(signals)} advisory signal(s) detected. "
        "These are labels, not blockers."
    )
    for raw in signals:
        signal = raw if isinstance(raw, dict) else {}
        print()
        print(f"- {signal.get('mode')} [{signal.get('confidence')}] {signal.get('signature')}")
        for path in signal.get("paths", [])[:8]:
            print(f"  path: {path}")
        for item in signal.get("evidence", [])[:4]:
            print(f"  evidence: {item}")
        print(f"  note: {signal.get('explanation')}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="base ref to diff against")
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument("--json-out", help="write JSON report to this path")
    args = parser.parse_args(argv)

    try:
        report = build_report(args.base, cwd=Path.cwd())
    except RuntimeError as exc:
        print(f"retired-failure-mode detector error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
