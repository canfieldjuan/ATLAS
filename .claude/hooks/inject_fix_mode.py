#!/usr/bin/env python3
"""SessionStart hook: re-inject the fix-mode baton after start/resume/compact.

When an active fix-mode baton (.claude/fix-mode-state.json) exists, emits its
key fields as `additionalContext` so a session -- especially one resuming after
auto-compaction -- continues the fix loop instead of re-exploring. No baton, an
inactive/malformed baton, or any error produces no output and exits 0.
"""

from __future__ import annotations

import json
import os
import sys


def _project_dir() -> str:
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def _summary(baton: dict) -> str:
    lines = ["PR Fix Mode is ACTIVE -- stay inside the allowed set:"]
    fields = [
        ("PR", baton.get("pr")),
        ("Branch", baton.get("branch")),
        ("Latest commit", baton.get("latest_commit")),
        ("Allowed files", baton.get("allowed")),
        ("Max files", baton.get("max_files")),
        ("Current failing check", baton.get("failing_check")),
        ("Last useful finding", baton.get("last_finding")),
        ("Next exact action", baton.get("next_action")),
        ("Do-NOT-redo", baton.get("do_not_redo")),
    ]
    for label, value in fields:
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def main() -> int:
    try:
        json.load(sys.stdin)
    except Exception:
        pass  # SessionStart payload is unused; never fail on it

    try:
        baton_path = os.path.join(_project_dir(), ".claude", "fix-mode-state.json")
        if not os.path.isfile(baton_path):
            return 0
        with open(baton_path, encoding="utf-8") as fh:
            baton = json.load(fh)
        if not isinstance(baton, dict) or not baton.get("active"):
            return 0
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": _summary(baton),
                    }
                }
            )
        )
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(main())
