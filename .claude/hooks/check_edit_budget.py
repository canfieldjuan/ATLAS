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
import sys


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
        candidate = path.replace("\\", "/")
        if os.path.isabs(path) or os.path.isabs(candidate):
            candidate = os.path.relpath(path, project_dir)
        return os.path.normpath(candidate).replace("\\", "/")
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

        for target in _targets(tool_input):
            rel = _relativize(target, project_dir)
            if rel in _ALWAYS_ALLOWED:
                continue  # control files stay editable so /fix-mode off + widen work
            if not any(fnmatch.fnmatch(rel, str(pat)) for pat in allowed):
                _deny(
                    f"{rel} is outside the fix-mode allowed set "
                    f"({', '.join(str(p) for p in allowed)}). Widen the baton's "
                    "allowed list with the upstream reason (AGENTS.md 3k/3l) "
                    "before editing it."
                )
                return 0
        return 0
    except Exception:
        return 0  # never block on an unexpected hook error


if __name__ == "__main__":
    sys.exit(main())
