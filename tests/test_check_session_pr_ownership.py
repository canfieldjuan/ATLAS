from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_session_pr_ownership.py"
SPEC = importlib.util.spec_from_file_location("check_session_pr_ownership", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _state_text(
    *,
    owned_pr: str = "#1189",
    branch: str = "claude/pr-agents-session-pr-map",
    head_sha: str = "abc123",
    may_touch: str = "- #1189 PR-Agents-Session-PR-Map -- owned\n",
    must_not_touch: str = "- #1190 PR-FAQ -- not ours\n",
) -> str:
    return f"""# Atlas Builder Session State

## Owned Active PR

PR: {owned_pr}
Branch: {branch}
Expected head SHA: {head_sha}

## PRs This Session May Touch

{may_touch}
## PRs This Session Must Not Touch

{must_not_touch}
"""


def test_parse_session_state_extracts_owned_and_blocked_prs() -> None:
    ownership = guard.parse_session_state(_state_text())

    assert ownership.owned_pr == 1189
    assert ownership.owned_branch == "claude/pr-agents-session-pr-map"
    assert ownership.owned_head == "abc123"
    assert ownership.may_touch == frozenset({1189})
    assert ownership.must_not_touch == frozenset({1190})


def test_ownership_errors_accepts_owned_pr_branch_and_head() -> None:
    ownership = guard.parse_session_state(_state_text())

    assert guard.ownership_errors(
        ownership,
        pr=1189,
        branch="claude/pr-agents-session-pr-map",
        head_sha="abc123",
    ) == []


def test_ownership_errors_blocks_must_not_touch_before_other_checks() -> None:
    ownership = guard.parse_session_state(_state_text())

    errors = guard.ownership_errors(
        ownership,
        pr=1190,
        branch="claude/pr-faq-deflection",
        head_sha="other",
    )

    assert errors == ["PR #1190 is listed under PRs This Session Must Not Touch"]


def test_ownership_errors_rejects_unlisted_pr() -> None:
    ownership = guard.parse_session_state(_state_text(must_not_touch=""))

    errors = guard.ownership_errors(
        ownership,
        pr=1191,
        branch="claude/pr-other",
        head_sha="abc123",
    )

    assert errors == [
        "PR #1191 is not listed under Owned Active PR or PRs This Session May Touch"
    ]


def test_ownership_errors_rejects_owned_branch_mismatch() -> None:
    ownership = guard.parse_session_state(_state_text())

    errors = guard.ownership_errors(
        ownership,
        pr=1189,
        branch="claude/pr-other",
        head_sha="abc123",
    )

    assert errors == [
        "branch mismatch for PR #1189: state has claude/pr-agents-session-pr-map, "
        "target has claude/pr-other"
    ]


def test_ownership_errors_rejects_owned_head_mismatch() -> None:
    ownership = guard.parse_session_state(_state_text())

    errors = guard.ownership_errors(
        ownership,
        pr=1189,
        branch="claude/pr-agents-session-pr-map",
        head_sha="def456",
    )

    assert errors == [
        "head SHA mismatch for PR #1189: state has abc123, target has def456"
    ]


def test_ownership_errors_requires_target_head_when_state_records_one() -> None:
    ownership = guard.parse_session_state(_state_text())

    errors = guard.ownership_errors(
        ownership,
        pr=1189,
        branch="claude/pr-agents-session-pr-map",
    )

    assert errors == [
        "head SHA required for PR #1189 because session state records expected head abc123"
    ]


def test_main_fails_when_state_file_is_missing(tmp_path, capsys) -> None:
    code = guard.main([
        "--state-file",
        str(tmp_path / "missing.md"),
        "--pr",
        "1189",
        "--branch",
        "claude/pr-agents-session-pr-map",
    ])

    assert code == 2
    assert "session state file not found" in capsys.readouterr().err


def test_main_accepts_may_touch_pr_when_owned_slot_is_empty(tmp_path, capsys) -> None:
    state_file = tmp_path / "SESSION_STATE.local.md"
    state_file.write_text(
        _state_text(
            owned_pr="none",
            branch="none",
            head_sha="none",
            may_touch="- #1189 PR-Agents-Session-PR-Map -- reassigned\n",
            must_not_touch="",
        ),
        encoding="utf-8",
    )

    code = guard.main([
        "--state-file",
        str(state_file),
        "--pr",
        "1189",
        "--branch",
        "claude/pr-agents-session-pr-map",
    ])

    assert code == 0
    assert "session PR ownership check passed for PR #1189" in capsys.readouterr().out
