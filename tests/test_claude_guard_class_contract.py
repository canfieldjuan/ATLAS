from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_claude_names_guard_class_closure_stop_command() -> None:
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "Guard-class closure is a mechanical stop" in text
    assert "python scripts/check_guard_class_closure.py --base origin/main --strict" in text
    assert "docs/GUARD_CLASS_CLOSURE.md" in text
    assert "guard-class-closure: waived" in text


def test_claude_rejects_instance_patching_after_guard_finding() -> None:
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "do not add another token, regex, example fixture, or downstream filter" in normalized
    assert "close the class" in normalized
