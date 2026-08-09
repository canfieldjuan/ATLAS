"""Tests for the Claude Code fix-mode hooks.

The hooks are deny/inject scripts driven by stdin JSON. These tests subprocess
them with a tmp CLAUDE_PROJECT_DIR and crafted payloads, mirroring the
subprocess style of test_install_local_pr_hook.py. The load-bearing property is
fail-open: with no/inactive/malformed baton the PreToolUse hook never blocks.
"""

from __future__ import annotations

import itertools
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


def _root_trace() -> dict:
    return {
        "symptom": "Codex says parser.py accepts malformed x",
        "root_cause": "parser admission grammar does not reject x before callers branch",
        "source_trace": "review claim -> parser accepts x -> admission grammar lacks x rejection",
        "fix_strategy": "upstream-root",
        "upstream_files": ["scripts/parser.py"],
    }


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
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"], **_root_trace()})
    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)
    assert result.returncode == 0
    assert _decision(result.stdout) is None


def test_edit_outside_allowed_is_denied(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"], **_root_trace()})
    result = _run(CHECK_HOOK, _edit("tests/foo.py"), tmp_path)
    assert result.returncode == 0
    assert _decision(result.stdout) == "deny"
    assert "allowed set" in result.stdout


def test_multiedit_any_outside_target_is_denied(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"], **_root_trace()})
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
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"], **_root_trace()})
    abs_target = str(tmp_path / "scripts" / "parser.py")
    result = _run(CHECK_HOOK, _edit(abs_target), tmp_path)
    assert _decision(result.stdout) is None


def test_traversal_path_cannot_bypass_allowed_set(tmp_path):
    # `scripts/../tests/foo.py` matches `scripts/*` by raw fnmatch but resolves
    # to `tests/foo.py`; normalization must catch the escape and deny it.
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"], **_root_trace()})
    result = _run(CHECK_HOOK, _edit("scripts/../tests/foo.py"), tmp_path)
    assert _decision(result.stdout) == "deny"


def test_baton_control_file_is_always_editable(tmp_path):
    # Even with an allowed set that omits it, the baton must stay editable so
    # `/fix-mode off` / widen are never locked out.
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"]})
    result = _run(CHECK_HOOK, _edit(".claude/fix-mode-state.json"), tmp_path)
    assert result.returncode == 0
    assert _decision(result.stdout) is None


def test_session_state_file_is_always_editable(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"]})
    result = _run(CHECK_HOOK, _edit("SESSION_STATE.local.md"), tmp_path)
    assert _decision(result.stdout) is None


def test_active_fix_mode_denies_edit_without_root_trace(tmp_path):
    _write_baton(tmp_path, {"active": True, "allowed": ["scripts/*"]})
    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert result.returncode == 0
    assert _decision(result.stdout) == "deny"
    assert "root-cause trace is incomplete" in result.stdout
    assert "symptom, root_cause, source_trace" in result.stdout


def test_active_fix_mode_denies_placeholder_source_trace(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "source_trace": "TBD -> TBD",
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) == "deny"
    assert "source_trace must name the chain" in result.stdout


def test_active_fix_mode_denies_decorated_template_source_trace(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "fix_strategy": "symptom-only-deferred",
        "symptom_only_reason": "review finding is non-blocking",
        "follow_up": "HARDENING.md ROOT-TRACE-2",
        "source_trace": "<symptom -> intermediate cause -> upstream source>",
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) == "deny"
    assert "source_trace must name the chain" in result.stdout


def test_check_edit_budget_source_trace_endpoint_grammar(tmp_path):
    trace_tokens_by_expected = {
        "review claim": True,
        "parser branch": True,
        "症状": True,
        "admission source": True,
        "TBD": False,
        "unknown": False,
        "<symptom": False,
        "intermediate cause": False,
        "upstream source>": False,
        "...": False,
    }
    trace_containers = {
        "bare": lambda value: value,
        "padded": lambda value: f"  {value}  ",
    }
    trace_families = {
        "symptom": 0,
        "middle": 1,
        "source": 2,
    }

    for token, container, family in itertools.product(
        trace_tokens_by_expected,
        trace_containers,
        trace_families,
    ):
        endpoints = ["review claim", "parser branch", "admission source"]
        endpoints[trace_families[family]] = trace_containers[container](token)
        source_trace = " -> ".join(endpoints)
        spec_derived_oracle = trace_tokens_by_expected[token]
        baton = {
            "active": True,
            "allowed": ["scripts/*"],
            **_root_trace(),
            "source_trace": source_trace,
        }
        _write_baton(tmp_path, baton)

        result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

        assert (_decision(result.stdout) is None) is spec_derived_oracle


def test_symptom_only_strategy_requires_reason_and_followup(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "fix_strategy": "symptom-only-deferred",
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) == "deny"
    assert "symptom-only-deferred requires symptom_only_reason, follow_up" in result.stdout


def test_symptom_only_strategy_allows_with_reason_and_followup(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "fix_strategy": "symptom-only-deferred",
        "symptom_only_reason": "upstream shared parser is owned by another active lane",
        "follow_up": "HARDENING.md ROOT-TRACE-1",
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) is None


def test_upstream_root_denies_downstream_before_source_changed(tmp_path):
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
        },
    )

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) == "deny"
    assert "requires editing the declared upstream source" in result.stdout


def test_upstream_root_allows_downstream_after_source_changed(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'fixed'\n", encoding="utf-8")
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
        },
    )

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) is None


def test_upstream_root_normalizes_declared_upstream_paths(tmp_path):
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*"],
            **_root_trace(),
            "upstream_files": ["./scripts\\parser.py"],
        },
    )

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) is None


def test_upstream_root_allows_downstream_after_untracked_source_created(tmp_path):
    _git_fixture(tmp_path)
    baton = {
        "active": True,
        "allowed": ["scripts/*", "templates/*"],
        **_root_trace(),
        "upstream_files": ["scripts/new_parser.py"],
    }
    (tmp_path / "scripts" / "new_parser.py").write_text("def parse():\n    return 'new'\n", encoding="utf-8")
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) is None


def _git_fixture(path: Path) -> None:
    (path / "scripts").mkdir()
    (path / "templates").mkdir()
    (path / "scripts" / "parser.py").write_text("def parse():\n    return 'old'\n", encoding="utf-8")
    (path / "templates" / "downstream.html").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "base"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=path, check=True)


def test_inject_emits_context_when_active(tmp_path):
    _write_baton(
        tmp_path,
        {"active": True, "pr": "#42", "allowed": ["scripts/*"], "next_action": "fix foo", **_root_trace()},
    )
    result = _run(INJECT_HOOK, {"hook_event_name": "SessionStart", "source": "compact"}, tmp_path)
    assert result.returncode == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "PR Fix Mode is ACTIVE" in ctx
    assert "#42" in ctx
    assert "Source trace" in ctx


def test_inject_silent_when_no_baton(tmp_path):
    result = _run(INJECT_HOOK, {"hook_event_name": "SessionStart", "source": "startup"}, tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == ""
