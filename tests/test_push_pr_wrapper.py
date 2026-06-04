from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "push_pr.sh"


def test_push_pr_dry_run_wires_body_file_env(tmp_path: Path) -> None:
    body = tmp_path / "body.md"
    body.write_text("Plan: plans/PR-Test.md\nSlice phase: Workflow/process\n", encoding="utf-8")
    env = {**os.environ, "ATLAS_PUSH_PR_DRY_RUN": "1"}

    result = subprocess.run(
        ["bash", str(SCRIPT), str(body), "-u", "origin", "HEAD"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"ATLAS_CURRENT_PR_BODY_FILE={body}" in result.stdout
    assert f"--current-pr-body-file {body}" in result.stdout
    assert "git push -u origin HEAD" in result.stdout


def test_push_pr_missing_body_file_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"

    result = subprocess.run(
        ["bash", str(SCRIPT), str(missing), "-u", "origin", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "PR body file not found" in result.stderr
    assert str(missing) in result.stderr
