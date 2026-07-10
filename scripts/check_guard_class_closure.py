#!/usr/bin/env python3
"""Advisory lint: flag guard-shaped diffs that lack a property/generative test.

Enforces the acceptance bar in ``docs/GUARD_CLASS_CLOSURE.md`` / ``AGENTS.md``
section 3k.1: a change to a guard over an OPEN input space (privacy/safety
classifier, sanitizer, parser-admission rule) must ship a grammar-derived
property test, not a fixture list of the reported strings. Codifying the rule
as prose is not enough; this makes it visible on every PR.

Heuristic and advisory by design. "Guard-shaped" over open input cannot be
detected precisely, so this reports ``::warning::`` annotations and exits 0 by
default (advisory-first, the same rollout the CI-enforcement arc uses before a
gate is promoted to required). ``--strict`` exits non-zero for a future
required enrollment. False positives are opted out per path in
``scripts/guard_class_closure_config.json`` or waived inline with a
``guard-class-closure: waived`` marker in the PR body / commit message.

The detection core is pure (``scan_diff`` takes file->content maps, no git), so
tests exercise the real logic with synthetic diffs and only the git transport
is mocked.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "scripts" / "guard_class_closure_config.json"
WAIVER_MARKER = "guard-class-closure: waived"

# --- Guard-shape signals -----------------------------------------------------

# Path-name stems that on their own mark a file as a guard over open input.
_GUARD_PATH_STEMS = (
    "privacy",
    "sanitiz",
    "redact",
    "scrub",
    "denylist",
    "admission",
    "moderation",
    "profanity",
    "classifier",
)

# Content signals that the code decides an admit/reject verdict.
_VERDICT_DEF_RE = re.compile(
    r"^\+\s*def\s+("
    r"[a-z0-9_]*is_(?:private|public|blocked|allowed|safe|valid)"
    r"|[a-z0-9_]*_(?:marker|verdict|admit|reject|classif[a-z]*|sanitiz[a-z]*)"
    r"|(?:is|has|should|can)_[a-z0-9_]+"
    r"|[a-z0-9_]*(?:validate|classify|admit|reject|scrub|redact)[a-z0-9_]*"
    r")\s*\(",
    re.MULTILINE,
)
# Content signals that the code inspects free-form / nested producer values.
_OPEN_INPUT_RE = re.compile(
    r"isinstance\([^)]*\b(?:str|dict|list|tuple|set|Mapping|Sequence)\b"
    r"|\bfrozenset\(\{"
    r"|\.strip\(\)\.lower\(\)|\.lower\(\)\.strip\(\)"
    r"|_TOKEN_RE|_tokens\(|re\.compile\("
)

# --- Property-test signals ---------------------------------------------------

# A bare for-loop is NOT a property signal: a plain loop over a fixture list
# is exactly the shape the lint exists to reject, and the old trailing-loop
# alternative (no MULTILINE) was order-dependent noise in both directions.
_PROPERTY_TEST_RE = re.compile(
    r"@pytest\.mark\.parametrize"
    r"|itertools\.product|\bproduct\("
    r"|\bhypothesis\b|@given\b"
)


def _is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    return p.name.startswith("test_") or "tests/" in path or p.name.endswith("_test.py")


def _added_lines(diff_hunk: str) -> str:
    """Return only the added (``+``) lines of a unified-diff hunk body."""
    return "\n".join(
        line for line in diff_hunk.splitlines() if line.startswith("+") and not line.startswith("+++")
    )


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def file_is_guard_shaped(path: str, added: str) -> bool:
    """True if a non-test .py file's change looks like a guard over open input.

    Guard-shaped iff a guard path-name stem is present, OR the added lines show
    BOTH a verdict definition and an open-input inspection signal. Requiring
    both content signals (not either) keeps the false-positive rate low enough
    for an advisory gate.
    """
    if not path.endswith(".py") or _is_test_path(path):
        return False
    compact = path.lower()
    if any(stem in compact for stem in _GUARD_PATH_STEMS):
        return True
    has_verdict = _VERDICT_DEF_RE.search(added) is not None
    has_open_input = _OPEN_INPUT_RE.search(added) is not None
    return has_verdict and has_open_input


def diff_has_property_test(test_added: Mapping[str, str]) -> bool:
    """True if any co-changed test file adds a property/generative test."""
    return any(_PROPERTY_TEST_RE.search(added) for added in test_added.values())


def guard_has_property_test(guard_path: str, test_added: Mapping[str, str]) -> bool:
    """True if a property/generative test is tied to THIS guard.

    Evidence must both carry a property signal AND reference the guard's
    module stem (in the test path or its added lines), so an unrelated
    property test elsewhere in the same PR cannot suppress a guard finding.
    """
    stem = PurePosixPath(guard_path).stem.lower()
    for test_path, added in test_added.items():
        if not _PROPERTY_TEST_RE.search(added):
            continue
        if stem in test_path.lower() or stem in added.lower():
            return True
    return False


def scan_diff(
    added_by_file: Mapping[str, str],
    *,
    ignore_globs: Sequence[str] = (),
) -> list[Finding]:
    """Pure core: given {path: added-lines}, return advisory findings.

    A finding is raised for each guard-shaped source file when the SAME diff
    adds no property/generative test.
    """
    test_added = {p: a for p, a in added_by_file.items() if _is_test_path(p)}

    findings: list[Finding] = []
    for path, added in sorted(added_by_file.items()):
        if _is_test_path(path):
            continue
        if any(PurePosixPath(path).match(glob) for glob in ignore_globs):
            continue
        if not file_is_guard_shaped(path, added):
            continue
        # Evidence is per-guard: an unrelated property test in the same PR
        # must not suppress this guard's finding.
        if guard_has_property_test(path, test_added):
            continue
        findings.append(
            Finding(
                path=path,
                reason=(
                    "guard-shaped change over open input with no co-changed "
                    "property/generative test (docs/GUARD_CLASS_CLOSURE.md req 3): "
                    "add a grammar-derived test (tokens x containers x families) "
                    "with a spec-derived oracle, not a fixture list"
                ),
            )
        )
    return findings


# --- Git transport (thin; mocked in tests) -----------------------------------


def _git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def changed_added_lines(base: str) -> dict[str, str]:
    """Return {path: added-lines} for every file changed in base...HEAD."""
    names = [n for n in _git(["diff", "--name-only", f"{base}...HEAD"]).splitlines() if n.strip()]
    out: dict[str, str] = {}
    for name in names:
        if not name.endswith(".py"):
            continue
        hunk = _git(["diff", "--unified=0", f"{base}...HEAD", "--", name])
        out[name] = _added_lines(hunk)
    return out


def load_ignore_globs() -> list[str]:
    if not CONFIG_PATH.exists():
        return []
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    globs = data.get("ignore_globs", [])
    if not isinstance(globs, list) or not all(isinstance(g, str) for g in globs):
        raise SystemExit(f"{CONFIG_PATH.name}: ignore_globs must be a list of strings")
    return globs


def _pr_text_waives() -> bool:
    body = ""
    for env in ("ATLAS_CURRENT_PR_BODY_FILE",):
        import os

        path = os.environ.get(env)
        if path and Path(path).exists():
            body += Path(path).read_text(encoding="utf-8")
    return WAIVER_MARKER in body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main", help="diff base ref")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on findings (for a future required enrollment)",
    )
    args = parser.parse_args(argv)

    added_by_file = changed_added_lines(args.base)
    findings = scan_diff(added_by_file, ignore_globs=load_ignore_globs())

    print("guard class-closure lint (advisory)")
    print("-" * 60)
    if not findings:
        print("OK: no guard-shaped change without a property test.")
        return 0
    if _pr_text_waives():
        print(f"WAIVED: '{WAIVER_MARKER}' present; {len(findings)} finding(s) not enforced.")
        for f in findings:
            print(f"  (waived) {f.path}")
        return 0
    for f in findings:
        # GitHub annotation so the advisory surfaces without failing the check.
        print(f"::warning file={f.path}::{f.reason}")
        print(f"  {f.path}: {f.reason}")
    print(f"{len(findings)} guard-shaped file(s) lack a co-changed property test.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
