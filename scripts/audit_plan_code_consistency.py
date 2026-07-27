#!/usr/bin/env python3
"""Verify a plan doc's backticked code/path claims match shipped code."""
from __future__ import annotations

import ast
import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_repo_root import audit_repo_root

REPO_ROOT = audit_repo_root(__file__)

BACKTICK_TOKEN = re.compile(r"`([^`]+)`")
BACKTICK_FUNC = re.compile(r"^([a-z_][a-z0-9_]{3,})\(\)$")
PATH_EXTENSIONS = (".md", ".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".txt")
PATH_SEARCH_ROOTS = (
    "scripts",
    "plans",
    "docs",
    "tests",
    "atlas_brain",
    "node_distributions",
)
PATH_CLAIM_ROOTS = set(PATH_SEARCH_ROOTS)
EXECUTABLE_EXTENSIONS = (".py", ".sh")
SHELL_CONTROL_TOKENS = {"&&", "||", "|", ";", ">", "<"}
PATH_HEAD_EXECUTABLE_NAMES = {
    "bash",
    "node",
    "npm",
    "npx",
    "python",
    "python3",
    "pytest",
    "sh",
    "uv",
}


class DeletedPathInventoryError(RuntimeError):
    """Raised when the branch-deletion inventory cannot be read from Git."""


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
    if token.startswith("-") or not token.endswith(PATH_EXTENSIONS):
        return False
    return not _looks_like_command_token(token)


def _looks_like_command_token(token: str) -> bool:
    if not any(char.isspace() for char in token):
        return False

    try:
        parts = shlex.split(token)
    except ValueError:
        parts = token.split()
    if len(parts) < 2:
        return False

    executable_index = 0
    while executable_index < len(parts) and _looks_like_env_assignment(
        parts[executable_index]
    ):
        executable_index += 1
    if executable_index >= len(parts) - 1:
        return False

    executable = parts[executable_index]
    if executable in SHELL_CONTROL_TOKENS or executable.startswith("-"):
        return False
    if Path(executable).name in PATH_HEAD_EXECUTABLE_NAMES:
        return True
    if executable.endswith(EXECUTABLE_EXTENSIONS):
        return True
    if "/" in executable:
        if _looks_like_repository_path_claim(token):
            return False
        return _looks_like_path_headed_executable(executable) or any(
            _looks_like_command_argument(part)
            for part in parts[executable_index + 1 :]
        )
    if _looks_like_repository_path_claim(token):
        return False
    if executable.endswith(PATH_EXTENSIONS):
        return False
    return any(
        _looks_like_command_argument(part)
        for part in parts[executable_index + 1 :]
    )


def _looks_like_env_assignment(part: str) -> bool:
    if "=" not in part:
        return False
    name, _value = part.split("=", 1)
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _looks_like_command_argument(part: str) -> bool:
    return part in SHELL_CONTROL_TOKENS or (part.startswith("-") and part != "-")


def _looks_like_path_headed_executable(executable: str) -> bool:
    path = Path(executable)
    if path.name in PATH_HEAD_EXECUTABLE_NAMES:
        return True
    if executable.startswith(("./", "../", "/")) and "." not in path.name:
        return True
    return path.parent.name == "bin" and "." not in path.name


def _looks_like_repository_path_claim(token: str) -> bool:
    if not token.endswith(PATH_EXTENSIONS):
        return False
    normalized = token.removeprefix("./")
    if "/" not in normalized:
        return True
    root = normalized.split("/", 1)[0]
    return root in PATH_CLAIM_ROOTS or root.startswith("extracted_")


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


def _path_resolves(claim: str, base_ref: str = "origin/main") -> bool:
    direct = REPO_ROOT / claim
    if direct.exists():
        return True
    if "/" in claim:
        if _path_deleted_in_branch_diff(claim, base_ref):
            return True
        return False
    for root in _candidate_roots():
        if not root.is_dir():
            continue
        for match in root.rglob(claim):
            if match.is_file():
                return True
    if _path_deleted_in_branch_diff(claim, base_ref):
        return True
    return False


def _path_deleted_in_branch_diff(claim: str, base_ref: str = "origin/main") -> bool:
    normalized_claim = _normalize_repo_relative_path(claim)
    if "/" not in normalized_claim:
        return any(
            _deleted_path_can_satisfy_basename_claim(path, normalized_claim)
            for path in _deleted_paths_in_branch_diff(base_ref)
        )
    return normalized_claim in {
        _normalize_repo_relative_path(path)
        for path in _deleted_paths_in_branch_diff(base_ref)
    }


def _normalize_repo_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _deleted_path_can_satisfy_basename_claim(path: str, basename: str) -> bool:
    normalized = _normalize_repo_relative_path(path)
    if Path(normalized).name != basename:
        return False
    if "/" not in normalized:
        return True
    root = normalized.split("/", 1)[0]
    return root in PATH_CLAIM_ROOTS or root.startswith("extracted_")


def _deleted_paths_in_branch_diff(base_ref: str = "origin/main") -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=DR",
            f"{base_ref}...HEAD",
        ],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        diagnostic = result.stderr.decode("utf-8", "replace").strip()
        if not diagnostic:
            diagnostic = f"git diff exited with status {result.returncode}"
        raise DeletedPathInventoryError(
            f"could not inspect deleted paths against {base_ref}: {diagnostic}"
        )
    deleted: list[str] = []
    parts = [
        item.decode("utf-8", "surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]
    i = 0
    while i < len(parts):
        status = parts[i]
        i += 1
        if status == "D":
            if i < len(parts):
                deleted.append(parts[i])
                i += 1
            continue
        if status.startswith("R"):
            if i < len(parts):
                deleted.append(parts[i])
                i += 1
            if i < len(parts):
                i += 1
            continue
        if i < len(parts):
            i += 1
    return deleted


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


def audit_claims(
    plan_text: str, base_ref: str = "origin/main"
) -> tuple[list[str], list[str]]:
    path_claims, function_claims = parse_claims(plan_text)
    missing_paths = sorted(
        claim
        for claim in path_claims
        if not _path_is_gitignored(claim) and not _path_resolves(claim, base_ref)
    )
    defs = collect_def_names() if function_claims else set()
    missing_functions = sorted(function_claims - defs)
    return missing_paths, missing_functions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a plan doc's backticked code/path claims match shipped code."
    )
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("plan_path")
    args = parser.parse_args()

    plan_path = Path(args.plan_path)
    if not plan_path.exists():
        print(f"plan doc not found: {plan_path}", file=sys.stderr)
        return 2

    plan_text = plan_path.read_text(encoding="utf-8")
    claimed_paths, claimed_functions = parse_claims(plan_text)
    try:
        missing_paths, missing_functions = audit_claims(plan_text, args.base_ref)
    except DeletedPathInventoryError as exc:
        print(f"deleted-path inventory failed: {exc}", file=sys.stderr)
        return 2

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
        print(f"OK: all {len(claimed_paths)} path claims resolve.")

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
