#!/usr/bin/env python3
"""PreToolUse hook: confine edits to the fix-mode allowed-files set.

Reads the PreToolUse payload on stdin. When an active fix-mode baton
(.claude/fix-mode-state.json) is present, it denies an Edit/Write/MultiEdit
whose target path is outside the baton's `allowed` globs, surfacing the reason
to the model via `permissionDecision: "deny"`.

Fail-open by construction: no baton, an inactive/empty/malformed baton, or any
unexpected error exits 0 with no output, so normal (non-fix-mode) sessions and
any committed-but-unarmed checkout are never blocked. The companion push-time
gate (scripts/audit_plan_doc_files_touched.py --max-files / `Max files:`)
enforces the file-count budget; this hook only enforces the allowed *set*.
"""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from fix_loop_trace_contract import (
    is_placeholder_text,
    normalize_repo_path,
    source_trace_is_valid,
)

_REQUIRED_ROOT_TRACE_FIELDS = (
    "activation_head",
    "symptom",
    "root_cause",
    "source_trace",
    "fix_strategy",
    "upstream_files",
)
_FIX_STRATEGIES = {"upstream-root", "symptom-only-deferred"}
_SUPPORT_PATHS = {"AGENTS.md", "CLAUDE.md", "docs/SESSION_STATE_TEMPLATE.md"}
_SUPPORT_PREFIXES = ("tests/", "plans/", ".claude/skills/")


def _project_dir() -> str:
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def _targets(tool_input: dict) -> list[str]:
    targets: list[str] = []
    fp = tool_input.get("file_path")
    if isinstance(fp, str) and fp:
        targets.append(fp)
    for edit in tool_input.get("edits") or []:
        if isinstance(edit, dict):
            efp = edit.get("file_path")
            if isinstance(efp, str) and efp:
                targets.append(efp)
    return targets


# Control files the armed session must always be able to edit, regardless of the
# allowed set -- so `/fix-mode off`, widening the baton, and updating the human
# state file are never locked out.
_ALWAYS_ALLOWED = {".claude/fix-mode-state.json", "SESSION_STATE.local.md"}


def _relativize(path: str, project_dir: str) -> str:
    """Repo-relative, normalized POSIX path (collapses '..'/'.', unifies seps).

    Normalizing before glob matching closes the `scripts/../tests/foo.py` bypass
    (it resolves to `tests/foo.py`, which no longer matches `scripts/*`) and
    makes Windows `\\` separators match `/`-based globs.
    """
    try:
        return normalize_repo_path(path, project_dir)
    except ValueError:
        return path


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def _has_text(value: object) -> bool:
    return isinstance(value, str) and not is_placeholder_text(value)


def _has_string_list(value: object) -> bool:
    return isinstance(value, list) and any(_has_text(item) for item in value)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if _has_text(item)]


def _path_set(value: object, project_dir: str) -> set[str]:
    return {
        _relativize(path, project_dir)
        for path in _string_list(value)
    }


def _root_trace_errors(baton: dict) -> list[str]:
    missing: list[str] = []
    for field in _REQUIRED_ROOT_TRACE_FIELDS:
        value = baton.get(field)
        if field == "upstream_files":
            if not _has_string_list(value):
                missing.append(field)
        elif not _has_text(value):
            missing.append(field)
    if missing:
        return ["missing " + ", ".join(missing)]
    if not isinstance(baton.get("activation_dirty_paths"), list):
        return ["activation_dirty_paths must snapshot staged/working/untracked paths when fix mode is armed"]
    if not source_trace_is_valid(baton.get("source_trace")):
        return ["source_trace must name the chain from symptom -> upstream source with non-placeholder endpoints"]

    strategy = str(baton.get("fix_strategy", "")).strip().lower()
    if strategy not in _FIX_STRATEGIES:
        return [
            "fix_strategy must be one of "
            + ", ".join(sorted(_FIX_STRATEGIES))
            + f", got {strategy!r}"
        ]
    if strategy == "symptom-only-deferred":
        symptom_missing = [
            field
            for field in ("symptom_only_reason", "follow_up")
            if not _has_text(baton.get(field))
        ]
        if symptom_missing:
            return ["symptom-only-deferred requires " + ", ".join(symptom_missing)]
    return []


def _is_support_path(path: str) -> bool:
    return path in _SUPPORT_PATHS or path.startswith(_SUPPORT_PREFIXES)


def _changed_paths(project_dir: str, base_ref: str | None) -> set[str]:
    changed: set[str] = set()
    commands = [
        ["git", "diff", "--name-only", "--cached"],
        ["git", "diff", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    if base_ref:
        commands.insert(0, ["git", "diff", "--name-only", f"{base_ref}...HEAD"])
    for command in commands:
        proc = subprocess.run(command, cwd=project_dir, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            continue
        changed.update(_relativize(line.strip(), project_dir) for line in proc.stdout.splitlines() if line.strip())
    return changed


def _upstream_source_is_changed(project_dir: str, baton: dict, upstream_files: set[str]) -> bool:
    activation_head = str(baton.get("activation_head") or "").strip() or None
    activation_dirty_paths = _path_set(baton.get("activation_dirty_paths"), project_dir)
    current_pass_paths = _changed_paths(project_dir, activation_head).difference(activation_dirty_paths)
    return bool(current_pass_paths.intersection(upstream_files))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # cannot parse input -> do not block

    try:
        project_dir = _project_dir()
        baton_path = os.path.join(project_dir, ".claude", "fix-mode-state.json")
        if not os.path.isfile(baton_path):
            return 0

        with open(baton_path, encoding="utf-8") as fh:
            baton = json.load(fh)
        if not isinstance(baton, dict) or not baton.get("active"):
            return 0

        allowed = baton.get("allowed")
        if not isinstance(allowed, list) or not allowed:
            return 0  # active but no constraint declared -> do not block

        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return 0

        targets = _targets(tool_input)
        normal_targets = []
        for target in targets:
            rel = _relativize(target, project_dir)
            if rel in _ALWAYS_ALLOWED:
                continue  # control files stay editable so /fix-mode off + widen work
            normal_targets.append(rel)
        if not normal_targets:
            return 0

        trace_errors = _root_trace_errors(baton)
        if trace_errors:
            _deny(
                "fix-mode root-cause trace is incomplete ("
                + "; ".join(trace_errors)
                + "). Fill symptom, root_cause, source_trace, fix_strategy, "
                "and upstream_files before editing; symptom-only-deferred also "
                "requires symptom_only_reason and follow_up (AGENTS.md 3k)."
            )
            return 0

        for rel in normal_targets:
            if not any(fnmatch.fnmatch(rel, str(pat)) for pat in allowed):
                _deny(
                    f"{rel} is outside the fix-mode allowed set "
                    f"({', '.join(str(p) for p in allowed)}). Widen the baton's "
                    "allowed list with the upstream reason (AGENTS.md 3k/3l) "
                    "before editing it."
                )
                return 0
        strategy = str(baton.get("fix_strategy", "")).strip().lower()
        upstream_files = _path_set(baton.get("upstream_files"), project_dir)
        if strategy == "upstream-root":
            downstream_targets = [
                rel
                for rel in normal_targets
                if rel not in upstream_files and not _is_support_path(rel)
            ]
            if downstream_targets and not _upstream_source_is_changed(project_dir, baton, upstream_files):
                _deny(
                    "fix-mode upstream-root requires editing the declared upstream "
                    "source before downstream symptom targets. Edit one of "
                    f"{', '.join(sorted(upstream_files))} first, or change the "
                    "baton to symptom-only-deferred with reason and follow_up."
                )
                return 0
        return 0
    except Exception:
        return 0  # never block on an unexpected hook error


if __name__ == "__main__":
    sys.exit(main())
