from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pr_body_contract.yml"


def test_pr_body_contract_workflow_validates_docs_only_against_fetched_pr_head() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "BASE_REF: ${{ github.event.pull_request.base.ref }}" in text
    assert '"+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}"' in text
    assert '--base-ref "origin/${BASE_REF}"' in text
    assert '--head-ref "refs/remotes/origin/pr-${PR_NUMBER}"' in text
