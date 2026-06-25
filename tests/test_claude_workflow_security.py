from __future__ import annotations

import re
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claude.yml"

# Actions that must stay present in claude.yml and pinned to an immutable
# commit SHA. Asserting a 40-char hex SHA (rather than a hard-coded value)
# keeps the "no mutable tag/branch ref" security tripwire while tolerating
# routine Dependabot SHA bumps that would otherwise require a fixture edit.
PINNED_ACTIONS = ("actions/checkout", "anthropics/claude-code-action")


def test_claude_oidc_job_is_owner_gated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.actor == github.repository_owner" in text
    assert "id-token: write" in text


def test_claude_actions_are_sha_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for action in PINNED_ACTIONS:
        match = re.search(re.escape(action) + r"@([^\s#]+)", text)
        assert match is not None, f"{action} must be used in claude.yml"
        ref = match.group(1)
        assert re.fullmatch(r"[0-9a-f]{40}", ref), (
            f"{action} must be pinned to a 40-char commit SHA, got {ref!r}"
        )
