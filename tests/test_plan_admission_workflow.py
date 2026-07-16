from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "plan_admission.yml"


def test_plan_admission_workflow_runs_as_trusted_base_pr_target() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request_target:" in text
    assert "plan-admission:" in text
    assert "if: github.event_name == 'pull_request_target'" in text
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0" in text
    assert "ref: ${{ github.event.pull_request.base.sha }}" in text
    assert "contents: read" in text
    assert "pull-requests: read" in text


def test_plan_admission_workflow_materializes_pr_head_as_data() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '"+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}"' in text
    assert '"pull/${PR_NUMBER}/head:refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert 'git worktree add "$RUNNER_TEMP/pr-tree" "refs/remotes/origin/pr-${PR_NUMBER}"' in text
    assert 'cd "$RUNNER_TEMP/pr-tree"' in text


def test_plan_admission_workflow_runs_base_owned_auditor_against_pr_data() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Audit plan admission" in text
    assert 'ATLAS_AUDIT_REPO_ROOT="$RUNNER_TEMP/pr-tree"' in text
    assert 'python "$GITHUB_WORKSPACE/scripts/audit_pr_plan_presence.py" \\' in text
    assert '"origin/${BASE_REF}"' in text
    assert '--pr-author "$PR_AUTHOR"' in text
    assert "PR_AUTHOR: ${{ github.event.pull_request.user.login }}" in text
