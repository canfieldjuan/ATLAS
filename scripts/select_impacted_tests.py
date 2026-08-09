#!/usr/bin/env python3
"""Select the unit tests impacted by a changed file set.

The unit gate runs the whole suite on every push (~7.5 min of pytest), which
re-pays for a guarantee `repo_wide_unit_backstop` already holds daily. This
picks the tests a change can actually reach, so a push pays for its own blast
radius instead of the repo's.

Soundness rests on three properties, in order of how much they matter:

1. **Transitive, not one-hop.** Reachability is computed over the whole
   first-party import graph, so `tests/test_a.py -> helpers -> changed_module`
   selects `test_a`. A one-hop "which test names this module" grep silently
   drops exactly the indirect dependencies a refactor breaks.
2. **Unresolvable input means FULL, never empty.** A file this script cannot
   prove scoped (a global config, an unparseable module, a deleted path, an
   unknown Python root, an unowned runtime asset, or an unowned path-loaded
   script) escalates to the full suite. The failure direction is "run too
   much", never "run too little" -- a selector that silently returns nothing
   is worse than no selector, because it reports green.
3. **Empty is only for provably test-free changes.** An empty selection means
   every changed file was mapped and none of them is reachable from any test
   (documentation, plans). It is not the fallback for "I could not tell".

Output: newline-separated test paths on stdout, or the single token ``FULL``.

    python scripts/select_impacted_tests.py --base origin/main
    python scripts/select_impacted_tests.py --changed-file /tmp/changed.txt
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

# First-party import roots. A module outside these is third-party: it cannot be
# changed by a PR, so it is not part of the graph.
FIRST_PARTY_ROOTS = ("atlas_brain", "tests", "scripts", "extracted_content_pipeline")

# Changing any of these invalidates selection itself -- they alter collection,
# the interpreter environment, or the gate's own logic, so no per-file mapping
# is meaningful. Matched against the changed path's parts, so a nested
# conftest.py escalates too.
GLOBAL_FILES = {
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "unit_gate_baseline.txt",
    "check_unit_gate.py",
    "select_impacted_tests.py",
    "unit_gate.yml",
}
GLOBAL_PREFIXES = ("requirements",)

FULL = "FULL"
TEST_FREE_SUFFIXES = {".md"}

# CI governance files are loaded by workflow/path instead of first-party Python
# imports, so the import graph cannot discover their tests. Keep this list
# deliberately small and auditable: unknown workflows, registries, and scripts
# still escalate to FULL.
EXPLICIT_TEST_OWNERS: dict[str, tuple[str, ...]] = {
    ".github/workflows/branch_protection_required_checks.yml": (
        "tests/test_security_guardrails_workflow.py",
    ),
    ".github/workflows/ai_reconciliation_live.yml": (
        "tests/test_audit_workflow_security_posture.py",
        "tests/test_check_ai_reconciliation_live.py",
    ),
    ".github/workflows/ai_reconciliation_review_retrigger.yml": (
        "tests/test_check_ai_reconciliation_live.py",
    ),
    ".github/workflows/pr_body_contract.yml": (
        "tests/test_pr_body_contract_workflow.py",
    ),
    ".github/workflows/unit_gate.yml": (
        "tests/test_check_unit_gate.py",
        "tests/test_select_impacted_tests.py",
        "tests/test_unit_gate_selector_fallback.py",
    ),
    "ci/gates.yml": (
        "tests/test_security_guardrails_workflow.py",
    ),
    "docs/SECURITY_GUARDRAILS.md": (
        "tests/test_security_guardrails_workflow.py",
        "tests/test_security_policy_docs.py",
    ),
    "docs/ci_cd_autonomous_coding_map.md": (
        "tests/test_audit_pr_watcher_safety.py",
    ),
    "docs/ci_cd_runtime_duplication_audit.md": (
        "tests/test_security_guardrails_workflow.py",
    ),
    "docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md": (
        "tests/test_security_guardrails_workflow.py",
    ),
    "docs/audits/required-workflow-enrollment-audit-2026-08-04.md": (
        "tests/test_security_guardrails_workflow.py",
    ),
    "scripts/audit_ai_reconciliation.py": (
        "tests/test_audit_ai_reconciliation.py",
        "tests/test_audit_fix_loop_disposition.py",
        "tests/test_audit_pr_body.py",
        "tests/test_check_ai_reconciliation_live.py",
    ),
    "scripts/audit_fix_loop_disposition.py": (
        "tests/test_audit_fix_loop_disposition.py",
        "tests/test_local_pr_review.py",
    ),
    "scripts/audit_pr_body.py": (
        "tests/test_audit_pr_body.py",
        "tests/test_check_ai_reconciliation_live.py",
        "tests/test_local_pr_review.py",
        "tests/test_open_pr_wrapper.py",
        "tests/test_push_pr_wrapper.py",
    ),
    "scripts/audit_workflow_security_posture.py": (
        "tests/test_audit_workflow_security_posture.py",
    ),
    "scripts/check_ai_reconciliation_live.py": (
        "tests/test_check_ai_reconciliation_live.py",
    ),
    "scripts/check_required_status_checks.py": (
        "tests/test_security_guardrails_workflow.py",
    ),
    "scripts/codex_wake_bridge.py": (
        "tests/test_codex_wake_bridge.py",
    ),
    "scripts/check_unit_gate.py": (
        "tests/test_check_unit_gate.py",
        "tests/test_select_impacted_tests.py",
    ),
    "extracted/_shared/scripts/check_ascii_python.sh": (
        "tests/test_pre_push_audit.py",
    ),
    "scripts/local_pr_review.sh": (
        "tests/test_local_pr_review.py",
    ),
    "scripts/open_pr.sh": (
        "tests/test_open_pr_wrapper.py",
    ),
    "scripts/update_pr_body.sh": (
        "tests/test_update_pr_body_wrapper.py",
    ),
    "scripts/pre_push_audit.sh": (
        "tests/test_pre_push_audit.py",
    ),
    "scripts/pr_watcher.py": (
        "tests/test_pr_watcher.py",
    ),
    "scripts/push_pr.sh": (
        "tests/test_push_pr_wrapper.py",
    ),
    "scripts/select_impacted_tests.py": (
        "tests/test_select_impacted_tests.py",
    ),
    "scripts/watch_owned_pr.sh": (
        "tests/test_watch_owned_pr.py",
    ),
}


def changed_files_from_git(base: str) -> list[str]:
    """Changed paths vs the merge base with ``base``."""
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{merge_base}..HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def is_global_change(path: str) -> bool:
    """True when ``path`` invalidates per-file selection (-> FULL)."""
    name = Path(path).name
    if name in GLOBAL_FILES:
        return True
    return any(name.startswith(prefix) for prefix in GLOBAL_PREFIXES)


def module_name_for(path: Path) -> str | None:
    """Dotted module name for a first-party .py path, else None."""
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if not parts or parts[0] not in FIRST_PARTY_ROOTS:
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def is_provably_test_free_path(path: Path) -> bool:
    """True for paths that can safely produce an empty unit-test selection."""
    return path.suffix in TEST_FREE_SUFFIXES


def is_conftest_module(name: str) -> bool:
    """True for any ``tests/.../conftest.py`` module name."""
    return name == "tests.conftest" or name.endswith(".conftest")


def explicit_test_owners(path: str, repo: Path) -> set[str] | None | str:
    """Owning tests for non-import-graph CI surfaces, or FULL when stale."""
    owners = EXPLICIT_TEST_OWNERS.get(path)
    if owners is None:
        return None

    missing = [owner for owner in owners if not (repo / owner).is_file()]
    if missing:
        print(
            f"select_impacted_tests: explicit test owner(s) missing for {path}: "
            f"{', '.join(missing)}; escalating to FULL",
            file=sys.stderr,
        )
        return FULL
    return set(owners)


def _imports_of(path: Path, rel: Path) -> set[str]:
    """First-party dotted names imported by ``path``.

    ``rel`` is the repo-relative path and is what relative imports resolve
    against -- using the absolute path here would prefix every ``from .x``
    with the checkout directory, so the edge would never match a real module
    and the importer would silently look independent.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        # Unparseable: caller escalates. Never treat as "imports nothing".
        raise

    names: set[str] = set()
    pkg_parts = list(rel.with_suffix("").parts)[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                prefix = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if prefix:
                names.add(prefix)
                for alias in node.names:
                    names.add(f"{prefix}.{alias.name}")
    return {n for n in names if n.split(".")[0] in FIRST_PARTY_ROOTS}


def build_reverse_graph(repo: Path) -> tuple[dict[str, set[str]], set[str]]:
    """(module -> modules importing it, unparseable paths).

    Unparseable files are returned rather than skipped: a file whose imports we
    could not read has unknown edges, and unknown edges mean FULL.
    """
    reverse: dict[str, set[str]] = defaultdict(set)
    unparseable: set[str] = set()
    modules: dict[str, Path] = {}

    for root in FIRST_PARTY_ROOTS:
        root_dir = repo / root
        if not root_dir.is_dir():
            continue
        for path in root_dir.rglob("*.py"):
            rel = path.relative_to(repo)
            name = module_name_for(rel)
            if name:
                modules[name] = rel

    for name, rel in modules.items():
        try:
            imported = _imports_of(repo / rel, rel)
        except (SyntaxError, UnicodeDecodeError, OSError):
            unparseable.add(str(rel))
            continue
        for target in imported:
            # Attribute imports (pkg.mod.symbol) resolve to the longest prefix
            # that is a real module, so importing a symbol still records the
            # edge to the module that defines it. Package initializers are also
            # executable dependencies, so keep ancestor package edges too:
            # importing atlas_brain.pkg.leaf runs atlas_brain/pkg/__init__.py.
            parts = target.split(".")
            for i in range(len(parts), 0, -1):
                candidate = ".".join(parts[:i])
                if candidate in modules:
                    reverse[candidate].add(name)
    return reverse, unparseable


def impacted_tests(
    changed: Iterable[str], reverse: dict[str, set[str]], repo: Path
) -> set[str] | str:
    """Test files transitively reachable from the changed modules."""
    seen: set[str] = set()
    queue: deque[str] = deque()

    for path in changed:
        name = module_name_for(Path(path))
        if name:
            queue.append(name)
            seen.add(name)

    tests: set[str] = set()
    while queue:
        current = queue.popleft()
        if is_conftest_module(current):
            print(
                f"select_impacted_tests: {current} is reachable; "
                "fixture consumers are collection-scoped, escalating to FULL",
                file=sys.stderr,
            )
            return FULL
        if current.startswith("tests."):
            rel = Path(*current.split(".")).with_suffix(".py")
            if (repo / rel).exists():
                tests.add(str(rel))
        for importer in reverse.get(current, ()):
            if importer not in seen:
                seen.add(importer)
                queue.append(importer)
    return tests


def select(changed: list[str], repo: Path) -> list[str] | str:
    """FULL, or the sorted impacted test paths."""
    if not changed:
        # No diff at all is not a provably test-free change; something is off
        # with the base ref. Escalate.
        return FULL

    owned_tests: set[str] = set()
    graph_changed: list[str] = []
    for path in changed:
        p = Path(path)
        abs_path = repo / p
        if not abs_path.exists():
            print(
                f"select_impacted_tests: {path} is absent in the PR head; "
                "deleted/renamed dependencies require FULL",
                file=sys.stderr,
            )
            return FULL
        owners = explicit_test_owners(path, repo)
        if owners == FULL:
            return FULL
        if owners is not None:
            owned_tests.update(owners)
            continue
        if is_global_change(path):
            return FULL

        if p.suffix != ".py" and is_provably_test_free_path(p):
            continue
        if p.suffix != ".py":
            print(
                f"select_impacted_tests: {path} is a non-Python runtime/config "
                "surface; escalating to FULL",
                file=sys.stderr,
            )
            return FULL
        if p.parts and p.parts[0] == "scripts":
            print(
                f"select_impacted_tests: {path} may be loaded by filesystem "
                "path; escalating to FULL",
                file=sys.stderr,
            )
            return FULL
        if module_name_for(p) is None:
            print(
                f"select_impacted_tests: cannot map {path}; escalating to FULL",
                file=sys.stderr,
            )
            return FULL
        graph_changed.append(path)

    if not graph_changed:
        return sorted(owned_tests)

    reverse, unparseable = build_reverse_graph(repo)
    if unparseable:
        print(
            f"select_impacted_tests: {len(unparseable)} unparseable module(s); "
            "escalating to FULL",
            file=sys.stderr,
        )
        return FULL

    result = impacted_tests(graph_changed, reverse, repo)
    if result == FULL:
        return FULL
    return sorted(result | owned_tests)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="origin/main",
                    help="Base ref to diff against (default: origin/main).")
    ap.add_argument("--changed-file", type=Path,
                    help="Read the changed paths from this file instead of git.")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)

    if args.changed_file is not None:
        changed = [
            line.strip()
            for line in args.changed_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        try:
            changed = changed_files_from_git(args.base)
        except subprocess.CalledProcessError as exc:
            print(f"select_impacted_tests: git failed ({exc}); escalating to FULL",
                  file=sys.stderr)
            print(FULL)
            return 0

    result = select(changed, args.repo)
    if result == FULL:
        print(FULL)
    else:
        for path in result:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
