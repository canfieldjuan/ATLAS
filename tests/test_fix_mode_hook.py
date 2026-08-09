"""Tests for the Claude Code fix-mode hooks.

The hooks are deny/inject scripts driven by stdin JSON. These tests subprocess
them with a tmp CLAUDE_PROJECT_DIR and crafted payloads, mirroring the
subprocess style of test_install_local_pr_hook.py. The load-bearing property is
fail-open: with no/inactive/malformed baton the PreToolUse hook never blocks.
"""

from __future__ import annotations

import itertools
import hashlib
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
        "activation_head": "HEAD",
        "activation_dirty_paths": [],
        "symptom": "Codex says parser.py accepts malformed x",
        "root_cause": "parser admission grammar does not reject x before callers branch",
        "source_trace": "review claim -> parser accepts x -> admission grammar lacks x rejection",
        "fix_strategy": "upstream-root",
        "upstream_files": ["scripts/parser.py"],
    }


def _edit(file_path: str) -> dict:
    return {"tool_name": "Edit", "tool_input": {"file_path": file_path}}


def _post_edit(file_path: str) -> dict:
    return {"hook_event_name": "PostToolUse", "tool_name": "Edit", "tool_input": {"file_path": file_path}}


def _decision(stdout: str) -> str | None:
    if not stdout.strip():
        return None
    return json.loads(stdout)["hookSpecificOutput"]["permissionDecision"]


def _file_fingerprint(project_dir: Path, rel_path: str) -> str:
    proc = subprocess.run(
        ["git", "ls-files", "-s", "--", rel_path],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        parts = proc.stdout.splitlines()[0].split()
        index_blob = parts[1] if len(parts) > 1 else "unknown"
    else:
        index_blob = "none"
    path = project_dir / rel_path
    try:
        worktree_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        worktree_hash = "missing"
    return f"index:{index_blob}|worktree:{worktree_hash}"


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


def test_active_fix_mode_denies_embedded_placeholder_source_trace(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "source_trace": "TBD symptom -> TBD upstream source",
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


def test_active_fix_mode_denies_producer_template_root_fields(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "fix_strategy": "symptom-only-deferred",
        "symptom_only_reason": "review finding is non-blocking",
        "follow_up": "HARDENING.md ROOT-TRACE-2",
        "symptom": "<failing check or review claim being addressed>",
        "root_cause": "<upstream defect, not the visible leaf symptom>",
        "upstream_files": ["<repo-relative file(s) where the source is fixed>"],
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) == "deny"
    assert "root-cause trace is incomplete" in result.stdout
    assert "symptom, root_cause, upstream_files" in result.stdout


def test_active_fix_mode_allows_substantive_root_fields_with_sentinel_terms(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "symptom": "unknown input reaches parser",
        "root_cause": "parser maps None to the accepted default",
        "source_trace": "unknown input reaches parser -> parser maps None to the accepted default",
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) is None


def test_check_edit_budget_source_trace_endpoint_grammar(tmp_path):
    trace_tokens_by_expected = {
        "review claim": True,
        "parser branch": True,
        "症状": True,
        "admission source": True,
        "TBD": False,
        "TBD symptom": False,
        "TBD upstream source": False,
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


def test_symptom_only_strategy_denies_decorated_template_reason_and_followup(tmp_path):
    baton = {
        "active": True,
        "allowed": ["scripts/*"],
        **_root_trace(),
        "fix_strategy": "symptom-only-deferred",
        "symptom_only_reason": "<required only for symptom-only-deferred>",
        "follow_up": "<required only for symptom-only-deferred>",
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)

    assert _decision(result.stdout) == "deny"
    assert "symptom-only-deferred requires symptom_only_reason, follow_up" in result.stdout


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


def test_upstream_root_denies_downstream_when_worktree_source_predates_activation(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'pre baton dirty'\n", encoding="utf-8")
    activation_fingerprint = _file_fingerprint(tmp_path, "scripts/parser.py")
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_dirty_paths": ["scripts/parser.py"],
            "activation_dirty_fingerprints": {"scripts/parser.py": activation_fingerprint},
        },
    )

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) == "deny"


def test_upstream_root_denies_downstream_with_uncaptured_activation_fingerprint(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'pre baton dirty'\n", encoding="utf-8")
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_dirty_paths": ["scripts/parser.py"],
            "activation_dirty_fingerprints": {
                "scripts/parser.py": "<index/worktree fingerprint for that path when armed>"
            },
        },
    )

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) == "deny"
    assert "activation_dirty_fingerprints must snapshot file state" in result.stdout


def test_upstream_root_allows_downstream_when_initially_dirty_source_changes_again(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'pre baton dirty'\n", encoding="utf-8")
    activation_fingerprint = _file_fingerprint(tmp_path, "scripts/parser.py")
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_dirty_paths": ["scripts/parser.py"],
            "activation_dirty_fingerprints": {"scripts/parser.py": activation_fingerprint},
        },
    )
    upstream_result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)
    assert _decision(upstream_result.stdout) is None
    state = json.loads((tmp_path / ".claude" / "fix-mode-state.json").read_text(encoding="utf-8"))
    assert state["pending_upstream_edits"]["scripts/parser.py"] == activation_fingerprint
    assert "upstream_edit_receipts" not in state

    parser.write_text("def parse():\n    return 'post baton dirty'\n", encoding="utf-8")
    post_result = _run(CHECK_HOOK, _post_edit("scripts/parser.py"), tmp_path)
    assert _decision(post_result.stdout) is None
    state = json.loads((tmp_path / ".claude" / "fix-mode-state.json").read_text(encoding="utf-8"))
    assert set(state["upstream_edit_receipts"]) == {"scripts/parser.py"}
    assert state["upstream_edit_receipts"]["scripts/parser.py"] == _file_fingerprint(tmp_path, "scripts/parser.py")
    assert "pending_upstream_edits" not in state

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) is None


def test_upstream_root_invalidates_receipt_when_source_reverts_to_activation_state(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    activation_text = "def parse():\n    return 'pre baton dirty'\n"
    parser.write_text(activation_text, encoding="utf-8")
    activation_fingerprint = _file_fingerprint(tmp_path, "scripts/parser.py")
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_dirty_paths": ["scripts/parser.py"],
            "activation_dirty_fingerprints": {"scripts/parser.py": activation_fingerprint},
        },
    )
    assert _decision(_run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path).stdout) is None
    parser.write_text("def parse():\n    return 'post baton dirty'\n", encoding="utf-8")
    assert _decision(_run(CHECK_HOOK, _post_edit("scripts/parser.py"), tmp_path).stdout) is None
    assert _decision(_run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path).stdout) is None

    assert _decision(_run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path).stdout) is None
    parser.write_text(activation_text, encoding="utf-8")
    assert _decision(_run(CHECK_HOOK, _post_edit("scripts/parser.py"), tmp_path).stdout) is None
    state = json.loads((tmp_path / ".claude" / "fix-mode-state.json").read_text(encoding="utf-8"))
    assert "upstream_edit_receipts" not in state

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) == "deny"


def test_upstream_root_does_not_record_receipt_when_admitted_source_edit_does_not_change_file(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'pre baton dirty'\n", encoding="utf-8")
    activation_fingerprint = _file_fingerprint(tmp_path, "scripts/parser.py")
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_dirty_paths": ["scripts/parser.py"],
            "activation_dirty_fingerprints": {"scripts/parser.py": activation_fingerprint},
        },
    )

    upstream_result = _run(CHECK_HOOK, _edit("scripts/parser.py"), tmp_path)
    assert _decision(upstream_result.stdout) is None
    post_result = _run(CHECK_HOOK, _post_edit("scripts/parser.py"), tmp_path)
    assert _decision(post_result.stdout) is None
    state = json.loads((tmp_path / ".claude" / "fix-mode-state.json").read_text(encoding="utf-8"))
    assert "upstream_edit_receipts" not in state
    assert state["pending_upstream_edits"]["scripts/parser.py"] == activation_fingerprint

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) == "deny"


def test_upstream_root_denies_same_batch_downstream_when_only_source_target_is_present(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'pre baton dirty'\n", encoding="utf-8")
    activation_fingerprint = _file_fingerprint(tmp_path, "scripts/parser.py")
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_dirty_paths": ["scripts/parser.py"],
            "activation_dirty_fingerprints": {"scripts/parser.py": activation_fingerprint},
        },
    )
    payload = {
        "tool_name": "MultiEdit",
        "tool_input": {
            "edits": [
                {"file_path": "scripts/parser.py"},
                {"file_path": "templates/downstream.html"},
            ]
        },
    }

    result = _run(CHECK_HOOK, payload, tmp_path)

    assert _decision(result.stdout) == "deny"


def test_upstream_root_allows_downstream_after_worktree_source_changed_since_activation(tmp_path):
    _git_fixture(tmp_path)
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_dirty_paths": [],
        },
    )
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'post baton dirty'\n", encoding="utf-8")

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) is None


def test_upstream_root_denies_downstream_when_source_changed_before_activation(tmp_path):
    _git_fixture(tmp_path)
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'pre baton'\n", encoding="utf-8")
    subprocess.run(["git", "add", "scripts/parser.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "pre baton"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    activation_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_head": activation_head,
            "activation_dirty_paths": [],
        },
    )

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) == "deny"


def test_upstream_root_allows_downstream_after_source_commit_since_activation(tmp_path):
    _git_fixture(tmp_path)
    activation_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    parser = tmp_path / "scripts" / "parser.py"
    parser.write_text("def parse():\n    return 'post baton'\n", encoding="utf-8")
    subprocess.run(["git", "add", "scripts/parser.py"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "post baton"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    _write_baton(
        tmp_path,
        {
            "active": True,
            "allowed": ["scripts/*", "templates/*"],
            **_root_trace(),
            "activation_head": activation_head,
            "activation_dirty_paths": [],
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


def test_upstream_root_denies_downstream_when_untracked_source_predates_activation(tmp_path):
    _git_fixture(tmp_path)
    (tmp_path / "scripts" / "new_parser.py").write_text("def parse():\n    return 'old dirty'\n", encoding="utf-8")
    activation_fingerprint = _file_fingerprint(tmp_path, "scripts/new_parser.py")
    baton = {
        "active": True,
        "allowed": ["scripts/*", "templates/*"],
        **_root_trace(),
        "upstream_files": ["scripts/new_parser.py"],
        "activation_dirty_paths": ["scripts/new_parser.py"],
        "activation_dirty_fingerprints": {"scripts/new_parser.py": activation_fingerprint},
    }
    _write_baton(tmp_path, baton)

    result = _run(CHECK_HOOK, _edit("templates/downstream.html"), tmp_path)

    assert _decision(result.stdout) == "deny"


def test_upstream_root_allows_downstream_after_untracked_source_created_since_activation(tmp_path):
    _git_fixture(tmp_path)
    baton = {
        "active": True,
        "allowed": ["scripts/*", "templates/*"],
        **_root_trace(),
        "upstream_files": ["scripts/new_parser.py"],
        "activation_dirty_paths": [],
    }
    _write_baton(tmp_path, baton)
    (tmp_path / "scripts" / "new_parser.py").write_text("def parse():\n    return 'new'\n", encoding="utf-8")

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


def test_inject_emits_upstream_edit_receipts_when_active(tmp_path):
    _write_baton(
        tmp_path,
        {
            "active": True,
            "pr": "#42",
            "allowed": ["scripts/*"],
            "upstream_edit_receipts": {
                "scripts/parser.py": "index:none|worktree:missing",
            },
            **_root_trace(),
        },
    )
    result = _run(INJECT_HOOK, {"hook_event_name": "SessionStart", "source": "compact"}, tmp_path)
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]

    assert "Upstream edit receipts" in ctx
    assert "scripts/parser.py" in ctx


def test_inject_emits_pending_upstream_edits_when_active(tmp_path):
    _write_baton(
        tmp_path,
        {
            "active": True,
            "pr": "#42",
            "allowed": ["scripts/*"],
            "pending_upstream_edits": {
                "scripts/parser.py": "index:none|worktree:missing",
            },
            **_root_trace(),
        },
    )
    result = _run(INJECT_HOOK, {"hook_event_name": "SessionStart", "source": "compact"}, tmp_path)
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]

    assert "Pending upstream edits" in ctx
    assert "scripts/parser.py" in ctx


def test_inject_silent_when_no_baton(tmp_path):
    result = _run(INJECT_HOOK, {"hook_event_name": "SessionStart", "source": "startup"}, tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == ""
