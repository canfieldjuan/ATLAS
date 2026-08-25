from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "session_lane.yml"


def _audit_step_script() -> str:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = document["jobs"]["session-lane"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == "Audit session lane")


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_lifecycle_gate(
    tmp_path: Path,
    *,
    state: str,
) -> tuple[subprocess.CompletedProcess[str], tuple[str, ...]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "gh", '#!/bin/sh\nprintf "%s\\n" "${PR_STATE}"\n')

    audit_log = tmp_path / "audit.log"
    _write_executable(
        bin_dir / "python",
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "${AUDIT_LOG}"\n',
    )

    runner_temp = tmp_path / "runner"
    (runner_temp / "pr-tree").mkdir(parents=True)
    result = subprocess.run(
        ["bash", "-c", _audit_step_script()],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "AUDIT_LOG": str(audit_log),
            "BASE_REF": "main",
            "GH_TOKEN": "test-token",
            "GITHUB_HEAD_REF": "claude/test-branch",
            "GITHUB_WORKSPACE": "/trusted-base",
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "PR_NUMBER": "2496",
            "PR_STATE": state,
            "RUNNER_TEMP": str(runner_temp),
        },
    )
    audit_calls = tuple(audit_log.read_text(encoding="utf-8").splitlines()) if audit_log.exists() else ()
    return result, audit_calls


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


@pytest.mark.parametrize(
    ("state", "expected_returncode", "expected_auditor_calls"),
    [
        pytest.param("OPEN", 0, 1, id="open-audits"),
        pytest.param("CLOSED", 0, 0, id="closed-skips"),
        pytest.param("MERGED", 0, 0, id="merged-skips"),
        pytest.param("UNRECOGNIZED", 1, 0, id="unknown-fails"),
    ],
)
def test_session_lane_workflow_executes_lifecycle_gate(
    tmp_path: Path,
    state: str,
    expected_returncode: int,
    expected_auditor_calls: int,
) -> None:
    result, audit_calls = _run_lifecycle_gate(tmp_path, state=state)

    assert result.returncode == expected_returncode
    assert len(audit_calls) == expected_auditor_calls
    if state == "OPEN":
        assert audit_calls == (
            "/trusted-base/scripts/audit_pr_session_drift.py origin/main "
            f"--current-pr-body-file {tmp_path}/runner/current-pr-body.md "
            "--require-current-pr-body",
        )


def test_session_lane_workflow_passes_current_body_to_base_owned_auditor() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Write current PR body" in text
    assert 'Path(os.environ["RUNNER_TEMP"], "current-pr-body.md")' in text
    assert 'python "$GITHUB_WORKSPACE/scripts/audit_pr_session_drift.py" \\' in text
    assert '--current-pr-body-file "$RUNNER_TEMP/current-pr-body.md"' in text
    assert "--require-current-pr-body" in text
    assert '"origin/${BASE_REF}"' in text
    assert "GITHUB_HEAD_REF: ${{ github.event.pull_request.head.ref }}" in text
