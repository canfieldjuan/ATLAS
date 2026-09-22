"""End-to-end smoke of the real wake entrypoint.

Every other test in this area exercises the runner, the installer, and the
bridge separately. That leaves the seam the operator actually depends on
untested: the bridge reading a watcher config, splitting `CODEX_WAKE_COMMAND`,
and invoking the INSTALLED runner, which then records a thread and a wake log.
An integration error anywhere along that path -- a command string the bridge
cannot split, an installed artifact that does not accept the documented flags,
a state-directory mismatch -- would leave the real workflow dead while the unit
tests stayed green.

Only the external Codex binary is faked.
"""
from __future__ import annotations

import json
import re
import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_codex_wake_bridge.py"
THREAD_ID = "01a0bfda-9ffb-7873-a20f-b085a7ae0a92"


def _install(bin_dir: Path, systemd_dir: Path) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--bin-dir",
            str(bin_dir),
            "--systemd-dir",
            str(systemd_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    runner = bin_dir / "atlas-codex-wake-run"
    assert runner.exists(), "the installer must place the wake runner"
    return runner


def _fake_codex(tmp_path: Path) -> tuple[Path, Path]:
    record = tmp_path / "codex-invocation.json"
    fake = tmp_path / "fake codex bin"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"record = {str(record)!r}\n"
        "with open(record, 'w', encoding='utf-8') as handle:\n"
        "    json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd(),"
        " 'stdin': sys.stdin.read()}, handle)\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {THREAD_ID!r}}}))\n"
        "print(json.dumps({'type': 'item.completed', 'item':"
        " {'id': 'i', 'type': 'agent_message', 'text': 'addressed the review'}}))\n"
        "print(json.dumps({'type': 'turn.completed',"
        " 'usage': {'input_tokens': 29048, 'output_tokens': 5}}))\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake, record


def _watcher_status(head_sha: str) -> dict[str, object]:
    """An attention-shaped snapshot: unresolved review activity on the PR."""
    return {
        "watcher_id": "e2e-wake",
        "label": "End-to-end wake smoke",
        "observed_at": "2026-09-20T12:00:00-05:00",
        "next_poll_at": "2026-09-20T12:30:00-05:00",
        "state": "review_changed",
        "review_changed": True,
        "pr": {
            "number": 4242,
            "title": "A pull request with a new review thread",
            "url": "https://github.com/example/repo/pull/4242",
            "headRefName": "claude/pr-example",
            "headRefOid": head_sha,
            "state": "OPEN",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "reviewDecision": "",
        },
        "check_failures": [],
        "check_pending": False,
        "reconciliation_exit_code": 0,
    }


def test_bridge_wakes_the_installed_runner_and_records_the_thread(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    systemd_dir = tmp_path / "systemd"
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    repo_dir = tmp_path / "my repo dir"
    for path in (config_dir, state_dir, repo_dir):
        path.mkdir(parents=True, exist_ok=True)

    runner = _install(bin_dir, systemd_dir)
    fake_codex, codex_record = _fake_codex(tmp_path)
    head_sha = "a" * 40

    # The wake command exactly as the handoff doc tells an operator to write
    # it: absolute paths, each quoted on its own. The bridge shlex-splits this
    # value with no shell, so the outer quotes do not survive to protect the
    # individual arguments and an unquoted path containing a space would break
    # into stray argv entries. The paths here contain spaces to enforce that.
    wake_command = (
        f"'{sys.executable}' '{runner}' --watcher-id 'e2e-wake' "
        f"--repo-dir '{repo_dir}' --state-dir '{state_dir}' "
        f"--sandbox read-only --codex-bin '{fake_codex}'"
    )
    (config_dir / "e2e-wake.env").write_text(
        "\n".join(
            [
                'LABEL="End-to-end wake smoke"',
                f'REPO_DIR="{repo_dir}"',
                'PR="4242"',
                'REPO="example/repo"',
                f'SESSION_STATE="{repo_dir}/SESSION_STATE.local.md"',
                f'HEAD_SHA="{head_sha}"',
                'POLL_MINUTES="30"',
                'AUTO_MERGE="0"',
                f'CODEX_WAKE_COMMAND="{wake_command}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (state_dir / "e2e-wake.json").write_text(
        json.dumps(_watcher_status(head_sha), indent=2) + "\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "codex_wake_bridge.py"),
            "e2e-wake",
            "--source",
            "event",
            "--config-dir",
            str(config_dir),
            "--state-dir",
            str(state_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "wake_kind=event-attention" in result.stdout

    # The bridge actually reached the installed runner.
    assert codex_record.exists(), "the installed runner never invoked Codex"
    invocation = json.loads(codex_record.read_text(encoding="utf-8"))
    assert invocation["argv"][0] == "exec"
    assert 'sandbox_mode="read-only"' in invocation["argv"]
    assert Path(invocation["cwd"]).resolve() == repo_dir.resolve()
    # The prompt Codex received is the bridge's handoff, naming the real PR.
    assert "#4242" in invocation["stdin"]

    # The runner recorded resumable state and an auditable wake record.
    # One thread file per effective Codex home; this watcher ran under one.
    thread_files = [
        p for p in state_dir.iterdir()
        if re.fullmatch(r"e2e-wake\.codex-thread\.[0-9a-f]{16}", p.name)
    ]
    assert len(thread_files) == 1, [p.name for p in state_dir.iterdir()]
    assert thread_files[0].read_text(encoding="utf-8").strip() == THREAD_ID
    assert (state_dir / "e2e-wake.codex-wake.last.md").read_text(
        encoding="utf-8"
    ).strip() == "addressed the review"
    log_text = (state_dir / "e2e-wake.codex-wake.log").read_text(encoding="utf-8")
    assert "wake fresh watcher=e2e-wake" in log_text
    assert "input_tokens" in log_text


def test_second_bridge_wake_resumes_the_recorded_thread(tmp_path: Path) -> None:
    """The second event on the same PR must continue the arc, not restart it."""
    bin_dir = tmp_path / "bin"
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    repo_dir = tmp_path / "my repo dir"
    for path in (config_dir, state_dir, repo_dir):
        path.mkdir(parents=True, exist_ok=True)

    runner = _install(bin_dir, tmp_path / "systemd")
    fake_codex, codex_record = _fake_codex(tmp_path)
    head_sha = "b" * 40
    wake_command = (
        f"'{sys.executable}' '{runner}' --watcher-id 'e2e-wake' "
        f"--repo-dir '{repo_dir}' --state-dir '{state_dir}' "
        f"--sandbox read-only --codex-bin '{fake_codex}'"
    )
    (config_dir / "e2e-wake.env").write_text(
        f'REPO_DIR="{repo_dir}"\nPR="4242"\nREPO="example/repo"\n'
        f'HEAD_SHA="{head_sha}"\nPOLL_MINUTES="30"\nAUTO_MERGE="0"\n'
        f'CODEX_WAKE_COMMAND="{wake_command}"\n',
        encoding="utf-8",
    )
    (state_dir / "e2e-wake.json").write_text(
        json.dumps(_watcher_status(head_sha), indent=2) + "\n", encoding="utf-8"
    )

    def run_bridge() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "codex_wake_bridge.py"),
                "e2e-wake",
                "--source",
                "event",
                "--config-dir",
                str(config_dir),
                "--state-dir",
                str(state_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )

    assert run_bridge().returncode == 0
    first = json.loads(codex_record.read_text(encoding="utf-8"))["argv"]
    assert "resume" not in first

    assert run_bridge().returncode == 0
    second = json.loads(codex_record.read_text(encoding="utf-8"))["argv"]
    assert second[:3] == ["exec", "resume", THREAD_ID], (
        "the second wake must continue the thread the first one recorded"
    )
