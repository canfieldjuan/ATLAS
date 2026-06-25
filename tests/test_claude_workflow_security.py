from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claude.yml"


def test_claude_oidc_job_is_owner_gated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.actor == github.repository_owner" in text
    assert "id-token: write" in text


def test_claude_actions_are_sha_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6.0.3" in text
    assert "anthropics/claude-code-action@80b31826338489861333dc17217865dfe8085cdc # v1" in text
