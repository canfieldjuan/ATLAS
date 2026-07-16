from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "session_lane.yml"


def test_session_lane_workflow_runs_as_trusted_base_pr_target() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request_target:" in text
    assert "session-lane:" in text
    assert "if: github.event_name == 'pull_request_target'" in text
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0" in text
    assert "ref: ${{ github.event.pull_request.base.sha }}" in text
    assert "pull-requests: read" in text


def test_session_lane_workflow_materializes_pr_head_as_data() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '"+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}"' in text
    assert '"pull/${PR_NUMBER}/head:refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert 'git worktree add "$RUNNER_TEMP/pr-tree" "refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert 'cd "$RUNNER_TEMP/pr-tree"' in text


def test_session_lane_workflow_passes_current_body_to_base_owned_auditor() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Write current PR body" in text
    assert 'Path(os.environ["RUNNER_TEMP"], "current-pr-body.md")' in text
    assert 'python "$GITHUB_WORKSPACE/scripts/audit_pr_session_drift.py" \\' in text
    assert '--current-pr-body-file "$RUNNER_TEMP/current-pr-body.md"' in text
    assert "--require-current-pr-body" in text
    assert '"origin/${BASE_REF}"' in text
    assert "GITHUB_HEAD_REF: ${{ github.event.pull_request.head.ref }}" in text
