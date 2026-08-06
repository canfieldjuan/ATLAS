#!/usr/bin/env python3
"""Audit PR-side CI/security docs-test consistency without executing PR code."""
from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audit_repo_root import audit_repo_root

REPO_ROOT = audit_repo_root(__file__)
SCRIPT_ROOT = Path(os.environ.get("ATLAS_AUDIT_SCRIPT_ROOT", Path(__file__).resolve().parents[1]))

GATE_REGISTRY = Path("ci/gates.yml")
SECURITY_GUARDRAILS_DOC = Path("docs/SECURITY_GUARDRAILS.md")
BRANCH_PROTECTION_WORKFLOW = Path(
    ".github/workflows/branch_protection_required_checks.yml"
)
SECURITY_GUARDRAILS_TEST = Path("tests/test_security_guardrails_workflow.py")

EXTRA_REQUIRED_STATUS_TEST_PATHS = ("ci/gates.yml",)
EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS = (
    ".github/workflows/branch_protection_required_checks.yml",
    "docs/SECURITY_GUARDRAILS.md",
    "scripts/check_required_status_checks.py",
    "tests/test_security_guardrails_workflow.py",
)


class AuditFailure(Exception):
    """Raised when the PR-side consistency contract is violated."""


def _load_required_status_checker():
    path = SCRIPT_ROOT / "scripts" / "check_required_status_checks.py"
    spec = importlib.util.spec_from_file_location("check_required_status_checks", path)
    if spec is None or spec.loader is None:
        raise AuditFailure(f"could not load trusted checker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(repo_root: Path, relative: Path) -> str:
    path = repo_root / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditFailure(f"{relative}: could not read file: {exc}") from exc


def _registry_inventory(
    repo_root: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    checker = _load_required_status_checker()
    gates = checker.parse_gate_registry(_read(repo_root, GATE_REGISTRY))
    contexts = tuple(
        str(gate["context"])
        for gate in gates
        if gate["enforcement"] == checker.BRANCH_REQUIRED
    )
    workflows = tuple(
        str(gate["workflow"])
        for gate in gates
        if gate["enforcement"] == checker.BRANCH_REQUIRED
    )
    all_workflows = tuple(str(gate["workflow"]) for gate in gates)
    return contexts, workflows, all_workflows


def _target_contains_name(target: ast.AST | None, name: str) -> bool:
    if target is None:
        return False
    if isinstance(target, ast.Name):
        return target.id == name
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(target))


def _alias_binds_name(alias: ast.alias, name: str) -> bool:
    bound_name = alias.asname or alias.name.split(".", 1)[0]
    return bound_name == name


def _pattern_binds_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return node.name == name
    if isinstance(node, ast.MatchMapping):
        return node.rest == name
    return False


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_namespace_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"globals", "locals", "vars"}
    if isinstance(node, ast.Attribute) and node.attr == "__dict__":
        return True
    return False


def _subscript_binds_name(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    return _is_namespace_expr(node.value) and _literal_string(node.slice) == name


def _target_indirectly_binds_name(target: ast.AST | None, name: str) -> bool:
    if target is None:
        return False
    return any(_subscript_binds_name(node, name) for node in ast.walk(target))


def _dict_literal_has_key(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    return any(_literal_string(key) == name for key in node.keys if key is not None)


def _call_indirectly_binds_name(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        if node.func.id in {"globals", "locals", "vars"}:
            return True
        if node.func.id in {"setattr", "delattr"} and len(node.args) >= 2:
            return _literal_string(node.args[1]) == name
        if node.func.id in {"exec", "eval"}:
            return any(name in value for arg in node.args if (value := _literal_string(arg)))
        return False
    if not isinstance(node.func, ast.Attribute) or not _is_namespace_expr(node.func.value):
        return False
    if node.func.attr in {"__setitem__", "setdefault", "pop"} and node.args:
        return _literal_string(node.args[0]) == name
    if node.func.attr == "update":
        return any(keyword.arg == name for keyword in node.keywords) or any(
            _dict_literal_has_key(arg, name) for arg in node.args
        )
    return False


def _literal_assignment_tuple(source: str, name: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: invalid Python syntax") from exc

    matches: list[ast.expr | None] = []
    module_assignments: set[ast.AST] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            _target_contains_name(target, name) for target in node.targets
        ):
            if len(node.targets) != 1 or not (
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
            ):
                raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
            matches.append(node.value)
            module_assignments.add(node)
        elif isinstance(node, ast.AnnAssign) and _target_contains_name(node.target, name):
            if not (isinstance(node.target, ast.Name) and node.target.id == name):
                raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
            matches.append(node.value)
            module_assignments.add(node)

    for node in ast.walk(tree):
        if node in module_assignments:
            continue
        if isinstance(node, ast.Assign) and any(
            _target_contains_name(target, name) for target in node.targets
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.Assign) and any(
            _target_indirectly_binds_name(target, name) for target in node.targets
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.AnnAssign) and _target_contains_name(node.target, name):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.AnnAssign) and _target_indirectly_binds_name(
            node.target, name
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.AugAssign) and _target_contains_name(node.target, name):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: mutating assignment for {name}")
        elif isinstance(node, ast.AugAssign) and _target_indirectly_binds_name(
            node.target, name
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, (ast.For, ast.AsyncFor)) and _target_contains_name(
            node.target, name
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.comprehension) and _target_contains_name(node.target, name):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, (ast.With, ast.AsyncWith)) and any(
            _target_contains_name(item.optional_vars, name) for item in node.items
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.NamedExpr) and _target_contains_name(node.target, name):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.Delete) and any(
            _target_contains_name(target, name) for target in node.targets
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            node.name == name
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            _alias_binds_name(alias, name) for alias in node.names
        ):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif _pattern_binds_name(node, name):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")
        elif _call_indirectly_binds_name(node, name):
            raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: runtime binding for {name}")

    if not matches:
        raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: missing {name}")
    if len(matches) > 1:
        raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: multiple assignments for {name}")
    value_node = matches[0]
    if value_node is None:
        raise AuditFailure(f"{SECURITY_GUARDRAILS_TEST}: {name} must be assigned a value")
    try:
        value = ast.literal_eval(value_node)
    except (ValueError, TypeError) as exc:
        raise AuditFailure(
            f"{SECURITY_GUARDRAILS_TEST}: {name} must be a literal tuple/list"
        ) from exc
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise AuditFailure(
            f"{SECURITY_GUARDRAILS_TEST}: {name} must be a literal tuple of strings"
        )
    return tuple(value)


def _strip_yaml_comment(raw_line: str, *, relative: Path, lineno: int) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(raw_line):
        if quote is not None:
            if quote == '"' and escaped:
                escaped = False
                continue
            if quote == '"' and char == "\\":
                escaped = True
                continue
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (index == 0 or raw_line[index - 1].isspace()):
            return raw_line[:index].rstrip()
    if quote is not None:
        raise AuditFailure(f"{relative}:{lineno}: malformed quoted scalar")
    return raw_line.rstrip()


def _yaml_indent(raw_line: str, *, relative: Path, lineno: int) -> int:
    if "\t" in raw_line:
        raise AuditFailure(f"{relative}:{lineno}: tabs are not supported")
    return len(raw_line) - len(raw_line.lstrip(" "))


def _decode_workflow_path(raw_value: str, *, relative: Path, lineno: int) -> str:
    value = raw_value.strip()
    if not value:
        raise AuditFailure(f"{relative}:{lineno}: empty push path")
    if value[0] in {"'", '"'}:
        if len(value) < 2 or value[-1] != value[0]:
            raise AuditFailure(f"{relative}:{lineno}: malformed quoted push path")
        return value[1:-1]
    if value.startswith(("[", "{")):
        raise AuditFailure(f"{relative}:{lineno}: push paths must be a scalar list")
    return value


def _github_pattern_matches(pattern: str, ref_name: str) -> bool:
    return fnmatch.fnmatchcase(ref_name, pattern)


def _branch_patterns_admit_ref(
    patterns: tuple[str, ...],
    ref_name: str,
    *,
    relative: Path,
) -> bool:
    admitted = False
    saw_positive = False
    for pattern in patterns:
        negative = pattern.startswith("!")
        effective_pattern = pattern[1:] if negative else pattern
        if not effective_pattern:
            raise AuditFailure(f"{relative}: empty on.push.branches pattern")
        if not negative:
            saw_positive = True
        if _github_pattern_matches(effective_pattern, ref_name):
            admitted = not negative
    if not saw_positive:
        raise AuditFailure(f"{relative}: on.push.branches must include a positive pattern")
    return admitted


def _branch_ignore_patterns_exclude_ref(
    patterns: tuple[str, ...],
    ref_name: str,
    *,
    relative: Path,
) -> bool:
    for pattern in patterns:
        if pattern.startswith("!"):
            raise AuditFailure(
                f"{relative}: negative on.push.branches-ignore patterns are unsupported"
            )
        if _github_pattern_matches(pattern, ref_name):
            return True
    return False


def _mapping_entry(text: str) -> tuple[str, str] | None:
    if text.startswith("- ") or ":" not in text:
        return None
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _split_flow_sequence_items(raw_value: str, *, relative: Path, lineno: int) -> tuple[str, ...]:
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in raw_value:
        if quote is not None:
            current.append(char)
            if quote == '"' and escaped:
                escaped = False
                continue
            if quote == '"' and char == "\\":
                escaped = True
                continue
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char == ",":
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)
    if quote is not None:
        raise AuditFailure(f"{relative}:{lineno}: malformed quoted scalar")
    item = "".join(current).strip()
    if item:
        items.append(item)
    return tuple(items)


def _decode_flow_sequence(
    raw_value: str,
    *,
    relative: Path,
    lineno: int,
    label: str,
) -> tuple[str, ...]:
    value = raw_value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise AuditFailure(f"{relative}:{lineno}: unsupported {label} entry")
    body = value[1:-1].strip()
    if not body:
        return ()
    return tuple(
        _decode_workflow_path(item.strip(), relative=relative, lineno=lineno)
        for item in _split_flow_sequence_items(body, relative=relative, lineno=lineno)
    )


def _collect_direct_sequence(
    parsed: list[tuple[int, int, str]],
    parent_index: int,
    *,
    label: str,
    relative: Path,
    inline_value: str = "",
) -> tuple[str, ...]:
    if inline_value:
        return _decode_flow_sequence(
            inline_value,
            relative=relative,
            lineno=parsed[parent_index][1],
            label=label,
        )
    parent_indent = parsed[parent_index][0]
    values: list[str] = []
    item_indent: int | None = None
    for index in range(parent_index + 1, len(parsed)):
        indent, lineno, text = parsed[index]
        if indent <= parent_indent:
            break
        if item_indent is None:
            item_indent = indent
        elif indent != item_indent:
            raise AuditFailure(f"{relative}:{lineno}: nested {label} entries are unsupported")
        if not text.startswith("- "):
            raise AuditFailure(f"{relative}:{lineno}: unsupported {label} entry")
        values.append(_decode_workflow_path(text[2:], relative=relative, lineno=lineno))
    return tuple(values)


def _workflow_push_paths(
    source: str,
    relative: Path = BRANCH_PROTECTION_WORKFLOW,
) -> tuple[str, ...]:
    parsed: list[tuple[int, int, str]] = []
    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        line = _strip_yaml_comment(raw_line, relative=relative, lineno=lineno)
        if not line.strip():
            continue
        parsed.append(
            (_yaml_indent(line, relative=relative, lineno=lineno), lineno, line.strip())
        )

    on_indexes = [
        index
        for index, (indent, _lineno, text) in enumerate(parsed)
        if indent == 0 and text == "on:"
    ]
    if len(on_indexes) != 1:
        raise AuditFailure(f"{relative}: expected exactly one top-level on: mapping")
    on_index = on_indexes[0]
    on_direct_child_indent = min(
        (
            indent
            for indent, _lineno, _text in parsed[on_index + 1 :]
            if indent > 0
        ),
        default=None,
    )
    if on_direct_child_indent is None:
        raise AuditFailure(f"{relative}: top-level on: mapping is empty")

    push_index: int | None = None
    for index in range(on_index + 1, len(parsed)):
        indent, _lineno, text = parsed[index]
        if indent <= 0:
            break
        if text == "push:":
            if indent != on_direct_child_indent:
                raise AuditFailure(f"{relative}: on.push must be a direct child")
            if push_index is not None:
                raise AuditFailure(f"{relative}: multiple on.push mappings")
            push_index = index
    if push_index is None:
        raise AuditFailure(f"{relative}: missing on.push mapping")

    push_indent = parsed[push_index][0]
    direct_child_indent = min(
        (
            indent
            for indent, _lineno, _text in parsed[push_index + 1 :]
            if indent > push_indent
        ),
        default=None,
    )
    if direct_child_indent is None:
        raise AuditFailure(f"{relative}: on.push mapping is empty")

    paths_index: int | None = None
    branches_index: int | None = None
    branches_ignore_index: int | None = None
    for index in range(push_index + 1, len(parsed)):
        indent, _lineno, text = parsed[index]
        if indent <= push_indent:
            break
        entry = _mapping_entry(text)
        key = entry[0] if entry is not None else None
        value = entry[1] if entry is not None else ""
        if key in {"paths", "branches", "branches-ignore"} and indent != direct_child_indent:
            raise AuditFailure(f"{relative}: on.push.{key} must be a direct child")
        if key == "paths":
            if indent != direct_child_indent:
                raise AuditFailure(f"{relative}: on.push.paths must be a direct child")
            if value:
                raise AuditFailure(f"{relative}: on.push.paths must be a block scalar list")
            if paths_index is not None:
                raise AuditFailure(f"{relative}: multiple on.push.paths mappings")
            paths_index = index
        elif key == "branches":
            if branches_index is not None:
                raise AuditFailure(f"{relative}: multiple on.push.branches mappings")
            branches_index = index
        elif key == "branches-ignore":
            if branches_ignore_index is not None:
                raise AuditFailure(f"{relative}: multiple on.push.branches-ignore mappings")
            branches_ignore_index = index
    if paths_index is None:
        raise AuditFailure(f"{relative}: missing on.push.paths mapping")

    paths = _collect_direct_sequence(
        parsed,
        paths_index,
        label="on.push.paths",
        relative=relative,
    )
    if not paths:
        raise AuditFailure(f"{relative}: on.push.paths is empty")
    branches = (
        _collect_direct_sequence(
            parsed,
            branches_index,
            label="on.push.branches",
            relative=relative,
            inline_value=_mapping_entry(parsed[branches_index][2])[1],
        )
        if branches_index is not None
        else ()
    )
    branches_ignore = (
        _collect_direct_sequence(
            parsed,
            branches_ignore_index,
            label="on.push.branches-ignore",
            relative=relative,
            inline_value=_mapping_entry(parsed[branches_ignore_index][2])[1],
        )
        if branches_ignore_index is not None
        else ()
    )
    if branches and branches_ignore:
        raise AuditFailure(f"{relative}: on.push cannot define both branches and branches-ignore")
    if branches and not _branch_patterns_admit_ref(branches, "main", relative=relative):
        raise AuditFailure(f"{relative}: on.push.branches must admit main")
    if _branch_ignore_patterns_exclude_ref(branches_ignore, "main", relative=relative):
        raise AuditFailure(f"{relative}: on.push.branches-ignore must not exclude main")
    return tuple(paths)


def _ordered_unique(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _format_missing(items: list[str]) -> str:
    return ", ".join(items)


def _invalid_repo_relative_paths(repo_root: Path, paths: tuple[str, ...]) -> list[str]:
    invalid: list[str] = []
    repo_root_resolved = repo_root.resolve()
    for path in paths:
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            invalid.append(path)
            continue
        resolved = (repo_root / candidate).resolve(strict=False)
        if not resolved.is_relative_to(repo_root_resolved):
            invalid.append(path)
    return invalid


def _missing_regular_files(repo_root: Path, paths: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for path in paths:
        if not (repo_root / path).is_file():
            missing.append(path)
    return missing


def audit_repo(repo_root: Path = REPO_ROOT) -> list[str]:
    contexts, workflows, all_workflows = _registry_inventory(repo_root)
    failures: list[str] = []

    invalid_registry_workflows = _invalid_repo_relative_paths(repo_root, all_workflows)
    if invalid_registry_workflows:
        failures.append(
            f"{GATE_REGISTRY}: registry workflow path(s) must stay inside the PR tree: "
            f"{_format_missing(invalid_registry_workflows)}"
        )

    missing_registry_workflows = _missing_regular_files(repo_root, all_workflows)
    if missing_registry_workflows:
        failures.append(
            f"{GATE_REGISTRY}: registry workflow path(s) missing from PR tree: "
            f"{_format_missing(missing_registry_workflows)}"
        )

    doc_text = _read(repo_root, SECURITY_GUARDRAILS_DOC)
    missing_doc_contexts = [context for context in contexts if f"`{context}`" not in doc_text]
    if missing_doc_contexts:
        failures.append(
            f"{SECURITY_GUARDRAILS_DOC}: missing branch-required context(s): "
            f"{_format_missing(missing_doc_contexts)}"
        )

    workflow_text = _read(repo_root, BRANCH_PROTECTION_WORKFLOW)
    workflow_push_paths = _workflow_push_paths(workflow_text)
    expected_test_workflow_paths = _ordered_unique(
        (*workflows, *EXTRA_REQUIRED_STATUS_TEST_PATHS)
    )
    expected_workflow_paths = _ordered_unique(
        (*expected_test_workflow_paths, *EXTRA_BRANCH_PROTECTION_TRIGGER_PATHS)
    )
    missing_workflow_paths = [
        path for path in expected_workflow_paths if path not in workflow_push_paths
    ]
    if missing_workflow_paths:
        failures.append(
            f"{BRANCH_PROTECTION_WORKFLOW}: missing push path trigger(s): "
            f"{_format_missing(missing_workflow_paths)}"
        )
    excluded_workflow_paths = [
        path for path in workflow_push_paths if path.startswith("!")
    ]
    if excluded_workflow_paths:
        failures.append(
            f"{BRANCH_PROTECTION_WORKFLOW}: unsupported negative push path trigger(s): "
            f"{_format_missing(excluded_workflow_paths)}"
        )

    test_text = _read(repo_root, SECURITY_GUARDRAILS_TEST)
    declared_contexts = _literal_assignment_tuple(test_text, "REQUIRED_STATUS_CONTEXTS")
    declared_workflows = _literal_assignment_tuple(test_text, "REQUIRED_STATUS_WORKFLOW_PATHS")

    if declared_contexts != contexts:
        failures.append(
            f"{SECURITY_GUARDRAILS_TEST}: REQUIRED_STATUS_CONTEXTS differs from "
            f"{GATE_REGISTRY} branch_required contexts"
        )
    if set(declared_workflows) != set(expected_test_workflow_paths):
        failures.append(
            f"{SECURITY_GUARDRAILS_TEST}: REQUIRED_STATUS_WORKFLOW_PATHS differs from "
            f"{GATE_REGISTRY} branch_required workflows"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit CI/security docs-test consistency from PR files as data. "
            "Safe for trusted-base pull_request_target workflows."
        )
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()

    failures = audit_repo(Path(args.repo_root))
    print("PR-side docs/test consistency audit")
    print(f"repo root: {Path(args.repo_root)}")
    print("-" * 60)
    if not failures:
        print("OK: branch-required registry, docs, workflow triggers, and tests agree.")
        return 0
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
