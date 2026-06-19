"""Tests for the Claude Code fix-mode hooks.

The hooks are deny/inject scripts driven by stdin JSON. These tests subprocess
them with a tmp CLAUDE_PROJECT_DIR and crafted payloads, mirroring the
subprocess style of test_install_local_pr_hook.py. The load-bearing property is
fail-open: with no/inactive/malformed baton the PreToolUse hook never blocks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_HOOK = REPO_ROOT / ".claude" / "hooks" / "check_edit_budget.py"
INJECT_HOOK = REPO_ROOT / ".claude" / "hooks" / "inject_fix_mode.py"


def _run(hook: Path, payload: dict, project_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        cwd=str(project_dir),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)},
    )


def _write_baton(project_dir: Path, baton: dict) -> None:
    state_dir = project_dir / ".claude"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "fix-mode-state.json").write_text(json.dumps(baton), encoding="utf-8")


def _edit(file_path: str) -> dict:
    return {"tool_name": "Edit", "tool_input": {"file_path": file_path}}


def _decision(stdout: str) -> str | None:
    if not stdout.strip():
        return None
    return json.loads(stdout)["hookSpecificOutput"]["permissionDecision"]


def test_no_baton_allows(tmp_path):
    result = _run(CHECK_HOOK, _edit("anything.py"), tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_inactive_baton_allows(tmp_path):
    _write_baton(tmp_path, {"active": False, "allowed": ["scripts/*"]})
    result = _run(CHECK_HOOK, _edit("tests/foo.py"), tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_malformed_baton_allows(tmp_path):
    (tmp_path / ".claude").mkdir(parents=True)
    (tmp_path / ".claude" / "fix-mode-state.json").write_text("{not json", encoding="utf-8")
    result = _run(CHECK_HOOK, _edit("tests/foo.py"), tmp_path)
    assert result.returncode == 0
    assert _decision(result.stdout) is None


def test_active_empty_allowed_does_not_block(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": []})
    result = _run(CHECK_HOOK, _edit("tests/foo.py"), tmp_path)
    assert result.returncode == 0
    assert _decision(result.stdout) is None


def test_edit_inside_allowed_is_allowed(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"]})
    result = _run(CHECK_HOOK, _edit("scripts/audit_x.py"), tmp_path)
    assert result.returncode == 0
    assert _decision(result.stdout) is None


def test_edit_outside_allowed_is_denied(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"]})
    result = _run(CHECK_HOOK, _edit("tests/foo.py"), tmp_path)
    assert result.returncode == 0
    assert _decision(result.stdout) == "deny"
    assert "allowed set" in result.stdout


def test_multiedit_any_outside_target_is_denied(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"]})
    payload = {
        "tool_name": "MultiEdit",
        "tool_input": {
            "edits": [
                {"file_path": "scripts/ok.py"},
                {"file_path": "tests/bad.py"},
            ]
        },
    }
    result = _run(CHECK_HOOK, payload, tmp_path)
    assert _decision(result.stdout) == "deny"


def test_absolute_path_is_relativized_before_match(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"]})
    abs_target = str(tmp_path / "scripts" / "audit_x.py")
    result = _run(CHECK_HOOK, _edit(abs_target), tmp_path)
    assert _decision(result.stdout) is None


def test_inject_emits_context_when_active(tmp_path):
    _write_baton(
        tmp_path,
        {"active": True, "pr": "#42", "allowed": ["scripts/*"], "next_action": "fix foo"},
    )
    result = _run(INJECT_HOOK, {"hook_event_name": "SessionStart", "source": "compact"}, tmp_path)
    assert result.returncode == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "PR Fix Mode is ACTIVE" in ctx
    assert "#42" in ctx


def test_inject_silent_when_no_baton(tmp_path):
    result = _run(INJECT_HOOK, {"hook_event_name": "SessionStart", "source": "startup"}, tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == ""
