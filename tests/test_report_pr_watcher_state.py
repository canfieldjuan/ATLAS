from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "report_pr_watcher_state.py"


def _write_state(
    path: Path,
    *,
    state: str,
    pr_state: str = "OPEN",
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "state": state,
        "observed_at": "2026-07-03T14:00:00-05:00",
        "check_failures": ["unit"] if state == "attention" else [],
        "check_pending": ["ci"] if state == "pending" else [],
        "reconciliation_exit_code": 0,
        "pr": {
            "number": 123,
            "title": f"{state} PR",
            "state": pr_state,
            "headRefOid": "abc123",
        },
    }
    if extra:
        payload.update(extra)
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _run(state_dir: Path, *, skip_github: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), "--state-dir", str(state_dir)]
    if skip_github:
        cmd.append("--skip-github")
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_reports_ready_as_manual_merge_decision(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state(state_dir / "ready.json", state="ready_for_human_merge")

    result = _run(state_dir)

    assert result.returncode == 0
    assert "Ready for active-agent merge decision" in result.stdout
    assert "ready_for_human_merge PR" in result.stdout


def test_ready_snapshot_with_failure_details_reports_attention(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state(
        state_dir / "contradictory-ready.json",
        state="ready_for_human_merge",
        extra={"head_mismatch": True, "reconciliation_exit_code": 1},
    )

    result = _run(state_dir)

    assert "Needs active-agent attention" in result.stdout
    assert "Ready for active-agent merge decision" not in result.stdout
    assert "head_mismatch=true" in result.stdout
    assert "reconciliation_exit_code=1" in result.stdout


def test_ready_snapshot_with_pending_checks_reports_pending(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state(
        state_dir / "pending-ready.json",
        state="ready_for_human_merge",
        extra={"check_pending": ["maturity-sweep"]},
    )

    result = _run(state_dir)

    assert "Still pending" in result.stdout
    assert "Ready for active-agent merge decision" not in result.stdout
    assert "pending=maturity-sweep" in result.stdout


def test_ignores_wake_bridge_handoff_json(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "session.wake.json").write_text(
        json.dumps({"wake_kind": "scheduled-ready"}),
        encoding="utf-8",
    )
    _write_state(state_dir / "session.json", state="ready_for_human_merge")

    result = _run(state_dir)

    assert "Ready for active-agent merge decision" in result.stdout
    assert "Other watcher states" not in result.stdout
    assert "watcher=unknown" not in result.stdout


def test_reports_attention_with_failures(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state(state_dir / "attention.json", state="attention")

    result = _run(state_dir)

    assert "Needs active-agent attention" in result.stdout
    assert "failures=unit" in result.stdout


def test_reports_attention_status_details(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state(
        state_dir / "attention.json",
        state="attention",
        extra={
            "check_failures": [],
            "head_mismatch": True,
            "reconciliation_exit_code": 1,
        },
    )

    result = _run(state_dir)

    assert "Needs active-agent attention" in result.stdout
    assert "head_mismatch=true" in result.stdout
    assert "reconciliation_exit_code=1" in result.stdout


def test_reports_merged_state_as_stale_cleanup(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state(state_dir / "stale.json", state="ready_for_human_merge", pr_state="MERGED")

    result = _run(state_dir)

    assert "Stale/closed watcher state to clean up" in result.stdout
    assert "Ready for active-agent merge decision" not in result.stdout


def test_ignores_archived_state_directories(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    (state_dir / "archive-1").mkdir(parents=True)
    _write_state(state_dir / "archive-1" / "old.json", state="ready_for_human_merge")

    result = _run(state_dir)

    assert "No watcher state files found" in result.stdout


def test_reports_unreadable_state_file_path_and_summary(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    bad = state_dir / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    result = _run(state_dir)

    assert "Needs active-agent attention" in result.stdout
    assert str(bad) in result.stdout
    assert "unreadable watcher JSON" in result.stdout


def test_uses_stored_closed_state_when_github_refresh_is_unavailable(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state(state_dir / "stale.json", state="ready_for_human_merge", pr_state="MERGED")

    result = _run(state_dir, skip_github=False, env={"PATH": ""})

    assert result.returncode == 0
    assert "Stale/closed watcher state to clean up" in result.stdout
    assert "Ready for active-agent merge decision" not in result.stdout
