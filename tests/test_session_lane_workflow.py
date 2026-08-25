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


def test_session_lane_workflow_snapshots_live_base_before_state_gate() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    refresh_live_base = '"+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}"'
    materialize_pr_tree = 'git worktree add "$RUNNER_TEMP/pr-tree" "refs/remotes/origin/pr-${PR_NUMBER}"'
    query_current_state = (
        'current_pr_state="$(gh pr view "${PR_NUMBER}" --json state --jq \'.state\')"'
    )
    invoke_auditor = 'python "$GITHUB_WORKSPACE/scripts/audit_pr_session_drift.py" \\'

    assert refresh_live_base in text
    assert '"pull/${PR_NUMBER}/head:refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert "BASE_SHA: ${{ github.event.pull_request.base.sha }}" not in text
    assert "git update-ref" not in text
    assert materialize_pr_tree in text
    assert query_current_state in text
    assert text.count("PR_NUMBER: ${{ github.event.pull_request.number }}") == 2
    assert text.index(refresh_live_base) < text.index(query_current_state)
    assert text.index(query_current_state) < text.index(invoke_auditor)
    assert 'cd "$RUNNER_TEMP/pr-tree"' in text


def test_session_lane_workflow_closes_pr_state_class() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'case "${current_pr_state}" in' in text
    assert "            OPEN)" in text
    assert "            CLOSED|MERGED)" in text
    assert 'echo "Session Lane skipped: PR #${PR_NUMBER} is ${current_pr_state}."' in text
    assert "              exit 0" in text
    assert "            *)" in text
    assert 'echo "::error::unexpected PR state: ${current_pr_state:-<empty>}"' in text
    assert "              exit 1" in text


def test_session_lane_workflow_passes_current_body_to_base_owned_auditor() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Write current PR body" in text
    assert 'Path(os.environ["RUNNER_TEMP"], "current-pr-body.md")' in text
    assert 'python "$GITHUB_WORKSPACE/scripts/audit_pr_session_drift.py" \\' in text
    assert '--current-pr-body-file "$RUNNER_TEMP/current-pr-body.md"' in text
    assert "--require-current-pr-body" in text
    assert '"origin/${BASE_REF}"' in text
    assert "GITHUB_HEAD_REF: ${{ github.event.pull_request.head.ref }}" in text
