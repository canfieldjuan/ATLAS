from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_decision_recording",
    Path(__file__).resolve().parent.parent / "scripts" / "check_decision_recording.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

UNRECORDED = """
## Why this slice exists

Routine fix.

### Decision recording

- Recorded decision URL: N/A.
- Umbrella issue: https://github.com/canfieldjuan/ATLAS/issues/123
- Scope effect: narrows issue #123 to backend-only work.
"""
RECORDED = """
## Why this slice exists

Routine fix.

### Decision recording

- Recorded decision URL: https://github.com/canfieldjuan/ATLAS/issues/2188#issuecomment-5085162036
- Umbrella issue: https://github.com/canfieldjuan/ATLAS/issues/2188
- Scope effect: narrows issue #2188 to this process rule.
"""


def test_rescope_decision_without_url_is_flagged() -> None:
    findings = mod.scan_plans({"plans/PR-Thing.md": UNRECORDED})
    assert len(findings) == 1
    assert findings[0].path == "plans/PR-Thing.md"
    assert mod.RULE in findings[0].reason


def test_rescope_decision_with_url_in_decision_section_is_clean() -> None:
    assert mod.scan_plans({"plans/PR-Thing.md": RECORDED}) == []


def test_root_issue_url_does_not_satisfy_recorded_comment_rule() -> None:
    text = RECORDED.replace("#issuecomment-5085162036", "")
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_comment_url_must_match_cited_umbrella() -> None:
    text = RECORDED.replace(
        "Umbrella issue: https://github.com/canfieldjuan/ATLAS/issues/2188",
        "Umbrella issue: https://github.com/canfieldjuan/ATLAS/issues/9999",
    )
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_scaffold_prompt_in_decision_section_does_not_trigger() -> None:
    text = """
## Why this slice exists

Routine fix.

### Decision recording

Required when citing an operator decision that re-scopes an umbrella issue;
otherwise write N/A.

- Recorded decision URL: N/A.
"""
    assert mod.scan_plans({"plans/PR-Thing.md": text}) == []


def test_non_rescope_plan_with_na_section_is_clean() -> None:
    text = "## Why this slice exists\nRoutine fix.\n### Decision recording\n- Recorded decision URL: N/A."
    assert mod.scan_plans({"plans/PR-Thing.md": text}) == []


@pytest.mark.parametrize(
    "scope_effect",
    [
        "widens umbrella #123 to include CLI behavior",
        "narrows umbrella #123 to backend-only scope",
        "reinterprets umbrella #123 as process-only",
        "defers frontend scope from umbrella #123",
    ],
)
def test_structural_decision_scope_without_comment_url_is_flagged(scope_effect: str) -> None:
    text = f"""
## Why this slice exists

Routine fix.

### Decision recording

- Recorded decision URL: N/A.
- Umbrella issue: https://github.com/canfieldjuan/ATLAS/issues/123
- Scope effect: {scope_effect}.
"""
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_prose_phrase_alone_does_not_trigger_without_structural_decision_section() -> None:
    text = """
## Why this slice exists

Per Juan, this plan narrows umbrella issue #123.

### Decision recording

Required when citing an operator decision that re-scopes an umbrella issue;
otherwise write N/A.

- Recorded decision URL: N/A.
- Umbrella issue: N/A.
- Scope effect: N/A.
"""
    assert mod.scan_plans({"plans/PR-Thing.md": text}) == []


def test_url_outside_decision_section_does_not_satisfy_rule() -> None:
    text = UNRECORDED + "\n## Deferred\n- See https://github.com/canfieldjuan/ATLAS/issues/1"
    assert len(mod.scan_plans({"plans/PR-Thing.md": text})) == 1


def test_cli_entrypoint_warns_advisory_and_fails_strict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(mod, "changed_plan_texts", lambda base: {"plans/PR-Thing.md": UNRECORDED})

    assert mod.main(["--base", "ignored"]) == 0
    out = capsys.readouterr().out
    assert "::warning file=plans/PR-Thing.md::" in out
    assert mod.RULE in out

    assert mod.main(["--base", "ignored", "--strict"]) == 1


def test_git_failure_raises_system_exit() -> None:
    with pytest.raises(SystemExit, match="git .* failed"):
        mod._git(["rev-parse", "--verify", "definitely-not-a-ref-xyz"])
