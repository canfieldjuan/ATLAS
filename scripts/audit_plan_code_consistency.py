#!/usr/bin/env python3
"""Verify a plan doc's backticked code/path claims match shipped code."""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

BACKTICK_TOKEN = re.compile(r"`([^`]+)`")
BACKTICK_FUNC = re.compile(r"^([a-z_][a-z0-9_]{3,})\(\)$")
PATH_EXTENSIONS = (".md", ".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".txt")
PATH_SEARCH_ROOTS = ("scripts", "plans", "docs", "tests", "atlas_brain")


def _slice_sections(plan_text: str, section_titles: tuple[str, ...]) -> str:
    """Return bodies for exact matching section headings."""
    out: list[str] = []
    in_section = False
    allowed = {title.lower() for title in section_titles}

    for line in plan_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            base = re.sub(r"\s*\([^)]*\)\s*$", "", heading).strip()
            in_section = base in allowed
            continue
        if in_section:
            out.append(line)
    return "\n".join(out)


def _is_path_token(token: str) -> bool:
    return token.endswith(PATH_EXTENSIONS) and not token.startswith("-")


def parse_claims(plan_text: str) -> tuple[set[str], set[str]]:
    """Return (path_claims, function_claims) from enforceable sections."""
    path_body = _slice_sections(plan_text, ("Scope", "Mechanism", "Verification"))
    func_body = _slice_sections(plan_text, ("Mechanism", "Verification"))

    paths: set[str] = set()
    for token in BACKTICK_TOKEN.findall(path_body):
        if _is_path_token(token):
            paths.add(token)

    funcs: set[str] = set()
    for token in BACKTICK_TOKEN.findall(func_body):
        match = BACKTICK_FUNC.match(token)
        if match:
            funcs.add(match.group(1))

    return paths, funcs


def _candidate_roots() -> list[Path]:
    roots = [REPO_ROOT / root for root in PATH_SEARCH_ROOTS]
    roots.extend(path for path in REPO_ROOT.glob("extracted_*") if path.is_dir())
    return roots


def _path_resolves(claim: str) -> bool:
    direct = REPO_ROOT / claim
    if direct.exists():
        return True
    if "/" in claim:
        return False
    for root in _candidate_roots():
        if not root.is_dir():
            continue
        for match in root.rglob(claim):
            if match.is_file():
                return True
    return False


def _path_is_gitignored(claim: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", claim],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError):
        return False
    return result.returncode == 0


def collect_def_names() -> set[str]:
    names: set[str] = set()
    for root in ("scripts", "atlas_brain"):
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for py_file in base.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
    return names


def audit_claims(plan_text: str) -> tuple[list[str], list[str]]:
    path_claims, function_claims = parse_claims(plan_text)
    missing_paths = sorted(
        claim
        for claim in path_claims
        if not _path_is_gitignored(claim) and not _path_resolves(claim)
    )
    defs = collect_def_names() if function_claims else set()
    missing_functions = sorted(function_claims - defs)
    return missing_paths, missing_functions


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_plan_code_consistency.py PLAN_PATH", file=sys.stderr)
        return 2

    plan_path = Path(sys.argv[1])
    if not plan_path.exists():
        print(f"plan doc not found: {plan_path}", file=sys.stderr)
        return 2

    plan_text = plan_path.read_text(encoding="utf-8")
    claimed_paths, claimed_functions = parse_claims(plan_text)
    missing_paths, missing_functions = audit_claims(plan_text)

    print(f"plan doc: {plan_path}")
    print(f"path claims:     {len(claimed_paths)}")
    print(f"function claims: {len(claimed_functions)}")
    print("-" * 60)

    drift = False
    if missing_paths:
        drift = True
        print(f"MISSING PATHS ({len(missing_paths)}):")
        for claim in missing_paths:
            print(f"  - {claim}")
    else:
        print(f"OK: all {len(claimed_paths)} path claims exist on disk.")

    if missing_functions:
        drift = True
        print(
            f"MISSING FUNCTION DEFS ({len(missing_functions)}); "
            f"checked scripts/ and atlas_brain/:"
        )
        for function in missing_functions:
            print(f"  - {function}()")
    else:
        print(f"OK: all {len(claimed_functions)} function claims resolve to a def.")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
