from __future__ import annotations

import re
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claude.yml"

# Actions that must stay present in claude.yml and pinned to an immutable
# commit SHA. Asserting a 40-char hex SHA (rather than a hard-coded value)
# keeps the "no mutable tag/branch ref" security tripwire while tolerating
# routine Dependabot SHA bumps that would otherwise require a fixture edit.
PINNED_ACTIONS = ("actions/checkout", "anthropics/claude-code-action")

# Capture the action ref from real `uses:` steps only (optionally a "- uses:"
# list item, optional surrounding quote). Anchoring to `uses:` means a pinned
# SHA left in a comment or example cannot satisfy the check, and the exact
# owner/repo comparison below means a look-alike such as
# `some-fork/actions/checkout` cannot stand in for `actions/checkout`.
_USES_RE = re.compile(r"^\s*-?\s*uses:\s*['\"]?([^\s'\"#]+)", re.MULTILINE)
_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _uses_refs(text: str) -> list[str]:
    return _USES_RE.findall(text)


def test_claude_oidc_job_is_owner_gated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.actor == github.repository_owner" in text
    assert "id-token: write" in text


def test_claude_actions_are_sha_pinned() -> None:
    uses_refs = _uses_refs(WORKFLOW.read_text(encoding="utf-8"))

    for action in PINNED_ACTIONS:
        pinned = [ref for ref in uses_refs if ref.split("@", 1)[0] == action]
        assert pinned, f"{action} must be used in a `uses:` step in claude.yml"
        for ref in pinned:
            _, _, sha = ref.partition("@")
            assert _SHA_RE.fullmatch(sha), (
                f"{action} must be pinned to a 40-char commit SHA in every "
                f"`uses:` step, got {ref!r}"
            )
