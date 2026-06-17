from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claude.yml"


def test_claude_oidc_job_is_owner_gated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.actor == github.repository_owner" in text
    assert "id-token: write" in text


def test_claude_actions_are_sha_pinned() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4" in text
    assert "anthropics/claude-code-action@9dd8b95a392eb34b6f5fb56cf5a64cb735912d4b # v1" in text
