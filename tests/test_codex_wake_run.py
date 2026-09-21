from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codex_wake_run.py"
SPEC = importlib.util.spec_from_file_location("codex_wake_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

THREAD_A = "01a0bfda-9ffb-7873-a20f-b085a7ae0a92"
THREAD_B = "01a0bfd5-3b96-7701-b18f-0278ff55548d"


def _fake_codex(tmp_path: Path, *, thread_id: str, exit_code: int = 0) -> tuple[Path, Path]:
    """Install a stand-in for the Codex CLI.

    Codex is a third-party binary, so this is the one real external boundary in
    the wake path. The stand-in records the argv and cwd it was actually
    invoked with, which is what the argv-shape assertions read, and emits the
    same JSONL envelope the real CLI emits under --json.
    """
    record = tmp_path / "invocation.json"
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"record = {str(record)!r}\n"
        "payload = {\n"
        "    'argv': sys.argv[1:],\n"
        "    'cwd': os.getcwd(),\n"
        "    'stdin': sys.stdin.read(),\n"
        "}\n"
        "with open(record, 'w', encoding='utf-8') as handle:\n"
        "    json.dump(payload, handle)\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {thread_id!r}}}))\n"
        "print(json.dumps({'type': 'item.completed',"
        " 'item': {'id': 'item_0', 'type': 'agent_message', 'text': 'ok'}}))\n"
        "print(json.dumps({'type': 'turn.completed',"
        " 'usage': {'input_tokens': 29048, 'cached_input_tokens': 0, 'output_tokens': 5}}))\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake, record


def _run(
    tmp_path: Path,
    *,
    fake: Path,
    watcher_id: str = "slice-123",
    prompt: str = "wake prompt",
    sandbox: str | None = None,
    repo_dir: Path | None = None,
    state_dir: Path | None = None,
) -> int:
    argv = [
        "--watcher-id",
        watcher_id,
        "--repo-dir",
        str(repo_dir if repo_dir is not None else (tmp_path / "repo")),
        "--state-dir",
        str(state_dir if state_dir is not None else (tmp_path / "state")),
        "--codex-bin",
        str(fake),
    ]
    if sandbox is not None:
        argv.extend(["--sandbox", sandbox])

    class _Stdin:
        @staticmethod
        def read() -> str:
            return prompt

    original = sys.stdin
    sys.stdin = _Stdin()  # type: ignore[assignment]
    try:
        return runner.main(argv)
    finally:
        sys.stdin = original


@pytest.fixture()
def repo_dir(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    return path


def test_source_has_no_merge_or_push_command() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "shell=True" not in source
    for forbidden in ("gh pr merge", "gh pr close", "--delete-branch", "git push"):
        assert forbidden not in source


def test_fresh_wake_argv_shape(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake) == 0

    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert argv == ["exec", "--json", "-c", 'sandbox_mode="workspace-write"', "-"]


def test_resume_wake_argv_shape(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "slice-123.codex-thread").write_text(THREAD_A + "\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake) == 0

    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert argv == [
        "exec",
        "resume",
        THREAD_A,
        "--json",
        "-c",
        'sandbox_mode="workspace-write"',
        "-",
    ]


def test_resume_never_passes_cd_or_sandbox_flags() -> None:
    """`codex exec resume` on 0.155.1 accepts neither -C/--cd nor -s/--sandbox."""
    argv = runner.build_argv(codex_bin="codex", thread_id=THREAD_A, sandbox="read-only")

    for rejected in ("-C", "--cd", "-s", "--sandbox", "--ask-for-approval"):
        assert rejected not in argv


def test_fresh_argv_also_avoids_the_flag_that_broke_the_old_runner() -> None:
    argv = runner.build_argv(codex_bin="codex", thread_id=None, sandbox="read-only")

    assert "--ask-for-approval" not in argv


def test_repo_dir_is_passed_as_cwd_on_both_paths(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"

    assert _run(tmp_path, fake=fake) == 0
    fresh_cwd = json.loads(record.read_text(encoding="utf-8"))["cwd"]

    assert (state_dir / "slice-123.codex-thread").read_text(encoding="utf-8").strip() == THREAD_A
    assert _run(tmp_path, fake=fake) == 0
    resume_payload = json.loads(record.read_text(encoding="utf-8"))

    assert Path(fresh_cwd).resolve() == repo_dir.resolve()
    assert Path(resume_payload["cwd"]).resolve() == repo_dir.resolve()
    assert "resume" in resume_payload["argv"]


def test_sandbox_default_is_workspace_write(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake) == 0

    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert 'sandbox_mode="workspace-write"' in argv
    assert 'sandbox_mode="danger-full-access"' not in argv


def test_explicit_sandbox_reaches_codex(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake, sandbox="read-only") == 0

    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert 'sandbox_mode="read-only"' in argv


def test_thread_id_round_trip(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    thread_path = tmp_path / "state" / "slice-123.codex-thread"

    assert _run(tmp_path, fake=fake) == 0
    assert thread_path.read_text(encoding="utf-8").strip() == THREAD_A

    assert _run(tmp_path, fake=fake) == 0
    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert argv[:3] == ["exec", "resume", THREAD_A]


def test_prompt_reaches_codex_on_stdin(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake, prompt="address the new review thread") == 0

    assert json.loads(record.read_text(encoding="utf-8"))["stdin"] == (
        "address the new review thread"
    )


@pytest.mark.parametrize(
    "stored",
    [
        "",
        "   \n",
        "not-a-uuid",
        "--dangerously-bypass-approvals-and-sandbox",
        f"{THREAD_A} extra",
        f"{THREAD_A}/../../etc/passwd",
        "x" * 300,
    ],
)
def test_rejects_malformed_thread_id(
    tmp_path: Path, repo_dir: Path, stored: str
) -> None:
    """A bad stored id degrades to a fresh thread, never into argv."""
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_B)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "slice-123.codex-thread").write_text(stored, encoding="utf-8")

    assert _run(tmp_path, fake=fake) == 0

    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert "resume" not in argv
    assert stored.strip() not in argv
    # The fresh run repairs the file with the real id it just started.
    assert (state_dir / "slice-123.codex-thread").read_text(
        encoding="utf-8"
    ).strip() == THREAD_B


def test_trailing_newline_thread_id_is_accepted(tmp_path: Path, repo_dir: Path) -> None:
    """The stored id is stripped before matching, so a newline is not a reject."""
    assert runner.valid_thread_id(THREAD_A)
    assert not runner.valid_thread_id(THREAD_A + "\n")

    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "slice-123.codex-thread").write_text(f"{THREAD_A}\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake) == 0

    assert json.loads(record.read_text(encoding="utf-8"))["argv"][:3] == [
        "exec",
        "resume",
        THREAD_A,
    ]


def test_thread_id_is_read_inside_the_lock(tmp_path: Path, repo_dir: Path) -> None:
    """Regression for the stale-read interleaving.

    Process B reads the thread file before A has written it, A records a new
    id and releases, then B acquires the lock. If B used its pre-lock read it
    would start a second thread and overwrite A's id, breaking the
    one-thread-per-watcher contract even though both respected the lock.
    """
    # The fake reports the same id the wake resumes: this test is about lock
    # ordering, and a mismatched id is refused by a separate rule.
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    thread_path = state_dir / "slice-123.codex-thread"
    lock_path = state_dir / "slice-123.codex-wake.lock"

    # Stand in for "A finished while B was blocked": the id appears after B
    # would have taken its pre-lock read, but before B can take the lock.
    real_flock = fcntl.flock

    def flock_then_publish(fileno: int, operation: int) -> None:
        real_flock(fileno, operation)
        if operation & fcntl.LOCK_EX and not thread_path.exists():
            thread_path.write_text(THREAD_A + "\n", encoding="utf-8")

    original = runner.fcntl.flock
    runner.fcntl.flock = flock_then_publish
    try:
        assert _run(tmp_path, fake=fake, state_dir=state_dir) == 0
    finally:
        runner.fcntl.flock = original
        lock_path.unlink(missing_ok=True)

    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert argv[:3] == ["exec", "resume", THREAD_A], (
        "the wake must use the id published before it took the lock"
    )


# The exact stderr codex-cli 0.155.1 prints for a resume against an id it has
# no rollout for. Captured from a real run, not invented.
def missing_session_stderr(thread_id: str) -> str:
    """The exact stderr codex-cli 0.155.1 prints for an unknown session.

    Parameterized by thread id because quarantine now requires the diagnostic
    to name the thread being resumed; a message about some other thread is not
    evidence about this one.
    """
    return (
        "Error: thread/resume: thread/resume failed: no rollout found for "
        f"thread id {thread_id} (code -32600)"
    )


def _failing_codex(tmp_path: Path, *, stderr_text: str, name: str = "fake-codex") -> Path:
    fake = tmp_path / name
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stderr.write({stderr_text!r} + '\\n')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def test_confirmed_missing_session_quarantines_the_id(
    tmp_path: Path, repo_dir: Path
) -> None:
    """A cleared or foreign session must not wedge every future wake."""
    fake = _failing_codex(tmp_path, stderr_text=missing_session_stderr(THREAD_A))
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    thread_path = state_dir / "slice-123.codex-thread"
    thread_path.write_text(THREAD_A + "\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 1

    assert not thread_path.exists()
    stale = state_dir / "slice-123.codex-thread.stale"
    assert stale.read_text(encoding="utf-8").strip() == THREAD_A
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "quarantined" in log_text
    assert "no rollout found for thread" in log_text


@pytest.mark.parametrize(
    "stderr_text",
    [
        "Error: request failed: connection reset by peer",
        "Error: unauthorized: refresh your credentials",
        "Error: invalid config at ~/.codex/config.toml",
        "",
    ],
)
def test_transient_resume_failure_keeps_the_id(
    tmp_path: Path, repo_dir: Path, stderr_text: str
) -> None:
    """A network, auth, or config failure must not discard a valid session.

    Quarantining on any nonzero exit would permanently lose the PR arc this
    runner exists to preserve, so only a confirmed missing session qualifies.
    """
    fake = _failing_codex(tmp_path, stderr_text=stderr_text)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    thread_path = state_dir / "slice-123.codex-thread"
    thread_path.write_text(THREAD_A + "\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 1

    assert thread_path.read_text(encoding="utf-8").strip() == THREAD_A
    assert not (state_dir / "slice-123.codex-thread.stale").exists()
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "keeping the id" in log_text


def test_next_wake_after_quarantine_starts_fresh(tmp_path: Path, repo_dir: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "slice-123.codex-thread").write_text(THREAD_A + "\n", encoding="utf-8")

    dead = _failing_codex(
        tmp_path, stderr_text=missing_session_stderr(THREAD_A), name="dead-codex"
    )
    assert _run(tmp_path, fake=dead, state_dir=state_dir) == 1

    fake, record = _fake_codex(tmp_path, thread_id=THREAD_B)
    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 0

    argv = json.loads(record.read_text(encoding="utf-8"))["argv"]
    assert "resume" not in argv, "the wake after a quarantine must start fresh"
    assert (state_dir / "slice-123.codex-thread").read_text(
        encoding="utf-8"
    ).strip() == THREAD_B


def test_failed_resume_that_did_attach_keeps_the_id(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Only a resume that never attached is quarantined.

    A turn that started and then failed still owns a usable thread; discarding
    it would throw away the arc on any ordinary mid-turn error.
    """
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A, exit_code=4)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    thread_path = state_dir / "slice-123.codex-thread"
    thread_path.write_text(THREAD_A + "\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 4

    assert thread_path.read_text(encoding="utf-8").strip() == THREAD_A
    assert not (state_dir / "slice-123.codex-thread.stale").exists()


def test_thread_id_persists_when_the_turn_is_killed_mid_flight(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Codex creates the session at thread.started, before the turn ends.

    If the runner is killed, the unit stopped, or the host restarts between
    those points, an unrecorded id means the next wake starts a second thread
    and the arc is lost. So the id is written as soon as the event is consumed.
    """
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, signal, sys\n"
        "sys.stdin.read()\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {THREAD_A!r}}}))\n"
        "sys.stdout.flush()\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    state_dir = tmp_path / "state"

    _run(tmp_path, fake=fake, state_dir=state_dir)

    assert (state_dir / "slice-123.codex-thread").read_text(
        encoding="utf-8"
    ).strip() == THREAD_A, "the id must survive a turn that never completed"


def test_codex_stderr_is_recorded_in_the_log(tmp_path: Path, repo_dir: Path) -> None:
    """stderr is captured for inspection, but must still reach the operator."""
    fake = _failing_codex(tmp_path, stderr_text="Error: something went wrong")

    assert _run(tmp_path, fake=fake) == 1

    log_text = (tmp_path / "state" / "slice-123.codex-wake.log").read_text(
        encoding="utf-8"
    )
    assert "codex stderr: Error: something went wrong" in log_text


def test_a_blocked_wake_waits_and_then_runs_its_own_turn(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrency is handled by waiting, not by handing the prompt away.

    There is no queue to strand: a wake that acquires the lock runs the prompt
    it was given.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lock_path = state_dir / "slice-123.codex-wake.lock"

    holder = lock_path.open("w", encoding="utf-8")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    real_sleep = time.sleep
    released: list[bool] = []

    def releasing_sleep(seconds: float) -> None:
        if not released:
            released.append(True)
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()
        real_sleep(0.01)

    monkeypatch.setattr(runner.time, "sleep", releasing_sleep)

    assert _run(tmp_path, fake=fake, state_dir=state_dir, prompt="my own prompt") == 0

    assert released, "the test must have exercised the wait path"
    assert json.loads(record.read_text(encoding="utf-8"))["stdin"] == "my own prompt"


def test_lock_timeout_reports_failure_rather_than_dropping_the_wake(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "LOCK_WAIT_SECONDS", 0)
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lock_path = state_dir / "slice-123.codex-wake.lock"

    with lock_path.open("w", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        assert _run(tmp_path, fake=fake, state_dir=state_dir) == runner.EXIT_LOCK_TIMEOUT

    assert not record.exists()
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "gave up" in log_text


def test_every_wake_that_gets_the_lock_runs_its_prompt(
    tmp_path: Path, repo_dir: Path
) -> None:
    """There is no skip path, so no state can suppress a wake.

    Burst coalescing was removed: it was an optimization this slice never
    required, and every version of it turned out to be able to drop a wake
    under some interleaving or clock change. Serializing on the lock makes the
    invariant checkable in one sentence.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # Anything a previous design might have used to suppress this wake.
    for name in (
        "slice-123.codex-wake.completed",
        "slice-123.codex-wake.generation",
        "slice-123.codex-wake.pending",
    ):
        (state_dir / name).write_text(f"{time.time_ns() + 10**12}\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 0

    assert record.exists(), "no leftover state may suppress a wake"


def test_a_backward_clock_cannot_suppress_a_wake(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Regression for the wall-clock ordering defect.

    An earlier design compared `time.time_ns()` stamps, so a completion
    recorded before a backward clock correction would silently skip the wake
    carrying the newer review prompt. Nothing orders wakes by time any more.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    future = time.time_ns() + 60 * 10**9
    (state_dir / "slice-123.codex-wake.completed").write_text(
        f"{future}\n", encoding="utf-8"
    )

    assert _run(tmp_path, fake=fake, state_dir=state_dir, prompt="the newest prompt") == 0

    assert json.loads(record.read_text(encoding="utf-8"))["stdin"] == "the newest prompt"


def test_an_unwritable_audit_path_does_not_fail_a_completed_turn(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Regression for the post-turn audit write.

    Codex may already have edited files, pushed, or commented. Failing the wake
    because the diagnostic copy of its message could not be written would make
    the caller retry and repeat those side effects.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # An unwritable audit path: replaced by a directory.
    (state_dir / "slice-123.codex-wake.last.md").mkdir()

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 0

    assert record.exists(), "the turn must still have run"
    assert (state_dir / "slice-123.codex-thread").read_text(
        encoding="utf-8"
    ).strip() == THREAD_A
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "not being failed for this" in log_text
    # The message still reaches the operator through the log.
    assert "agent message: ok" in log_text


def test_zero_exit_without_a_thread_event_is_a_protocol_failure(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Success with no thread id would silently forfeit continuity."""
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}))\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    state_dir = tmp_path / "state"

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == runner.EXIT_NO_THREAD_EVENT

    assert not (state_dir / "slice-123.codex-thread").exists()
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "emitted no valid thread.started event" in log_text


def test_zero_exit_with_a_malformed_thread_id_is_a_protocol_failure(
    tmp_path: Path, repo_dir: Path
) -> None:
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'not-a-uuid'}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}))\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    assert _run(tmp_path, fake=fake) == runner.EXIT_NO_THREAD_EVENT


def test_atomic_write_flushes_the_file_and_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The thread id must survive a host restart, not only process death.

    `os.replace` makes the swap atomic but says nothing about when the bytes or
    the directory entry reach disk. Without both flushes a power loss can leave
    the id missing and the next wake starts a second thread.
    """
    synced: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        try:
            synced.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        except OSError:  # pragma: no cover - defensive
            synced.append("unknown")
        real_fsync(fd)

    monkeypatch.setattr(runner.os, "fsync", recording_fsync)
    runner._atomic_write(tmp_path / "sub" / "value", "payload\n")

    assert (tmp_path / "sub" / "value").read_text(encoding="utf-8") == "payload\n"
    assert "file" in synced, "the staged file must be flushed"
    assert "dir" in synced, "the rename is directory metadata and needs its own flush"


def test_atomic_write_leaves_no_staged_file(tmp_path: Path) -> None:
    target = tmp_path / "value"
    runner._atomic_write(target, "payload\n")

    assert [q.name for q in tmp_path.iterdir()] == ["value"]


def test_codex_child_is_asked_to_die_with_the_runner(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed runner releases the lock, so its child must not outlive it.

    Otherwise an orphan keeps editing the checkout while a later wake acquires
    the lock and starts another turn against the same thread. A signal handler
    cannot cover a SIGKILLed parent; the kernel's parent-death signal can.
    """
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A)
    captured: dict[str, object] = {}
    real_popen = runner.subprocess.Popen

    def recording_popen(argv: list[str], **kwargs: object) -> object:
        captured["preexec_fn"] = kwargs.get("preexec_fn")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", recording_popen)

    assert _run(tmp_path, fake=fake) == 0

    assert callable(captured["preexec_fn"]), "a pre-exec hook must be installed"


def test_parent_death_support_is_probed_in_the_parent(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable kernel feature must be logged, not silently skipped.

    The probe cannot report from the pre-exec hook, which runs after fork.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    monkeypatch.setattr(
        runner, "parent_death_signal_support", lambda: (None, "no libc here")
    )
    captured: dict[str, object] = {}
    real_popen = runner.subprocess.Popen

    def recording_popen(argv: list[str], **kwargs: object) -> object:
        captured["preexec_fn"] = kwargs.get("preexec_fn")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", recording_popen)

    assert _run(tmp_path, fake=fake, state_dir=tmp_path / "state") == 0

    assert record.exists(), "the wake must still run without the kernel feature"
    # The hook is installed regardless: the orphan check does not depend on the
    # kernel facility, only the death signal does.
    assert callable(captured["preexec_fn"])
    log_text = (tmp_path / "state" / "slice-123.codex-wake.log").read_text(
        encoding="utf-8"
    )
    assert "no libc here" in log_text
    assert "cannot signal this Codex process" in log_text


def test_die_with_parent_sets_the_parent_death_signal() -> None:
    """Exercised in a real child, because it only takes effect after fork."""
    probe = (
        "import ctypes, os, signal, sys;"
        "libc = ctypes.CDLL('libc.so.6', use_errno=True);"
        # PR_GET_PDEATHSIG == 2; read it back into a buffer.
        "out = ctypes.c_int(0);"
        "rc = libc.prctl(2, ctypes.byref(out), 0, 0, 0);"
        "sys.exit(0 if rc == 0 and out.value == int(signal.SIGTERM) else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        preexec_fn=runner.make_die_with_parent(os.getpid()),
        check=False,
    )

    assert result.returncode == 0, "the child should carry PDEATHSIG=SIGTERM"


def test_hook_refuses_to_exec_when_already_reparented(tmp_path: Path) -> None:
    """Regression for the window between fork and setting the death signal.

    If the runner dies in that window the kernel has already reparented the
    child, and setting the flag afterwards delivers nothing because it is not
    retroactive. The hook compares against the pid captured before the fork
    and leaves rather than exec'ing Codex into an orphan.
    """
    evidence = tmp_path / "exec_happened.txt"
    # A pid that is not this child's parent stands in for "already reparented".
    hook = runner.make_die_with_parent(expected_ppid=os.getpid() + 1_000_000)

    result = subprocess.run(
        [sys.executable, "-c", f"open({str(evidence)!r}, 'w').write('ran')"],
        preexec_fn=hook,
        check=False,
    )

    assert not evidence.exists(), "the child must not reach exec once orphaned"
    assert result.returncode != 0


def test_hook_execs_normally_when_the_parent_is_still_alive(
    tmp_path: Path,
) -> None:
    """The other side of that boundary: a live parent must not block exec."""
    evidence = tmp_path / "exec_happened.txt"
    hook = runner.make_die_with_parent(expected_ppid=os.getpid())

    result = subprocess.run(
        [sys.executable, "-c", f"open({str(evidence)!r}, 'w').write('ran')"],
        preexec_fn=hook,
        check=False,
    )

    assert result.returncode == 0
    assert evidence.read_text(encoding="utf-8") == "ran"


def test_unusable_thread_path_refuses_before_launching_codex(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Regression: a turn whose id cannot be kept must not happen at all.

    Discovering it afterwards means Codex has already edited files while
    nothing can resume the arc, so every retry repeats that work.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "slice-123.codex-thread").mkdir()  # not a regular file

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == runner.EXIT_STATE_UNUSABLE

    assert not record.exists(), "Codex must not be launched at all"
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "refusing to start a turn" in log_text
    assert "safe to retry" in log_text


def test_thread_path_problem_accepts_a_usable_path(tmp_path: Path) -> None:
    """The other side of that boundary, and it must leave no probe behind."""
    assert runner.thread_path_problem(tmp_path / "sub" / "w.codex-thread") is None
    assert list((tmp_path / "sub").iterdir()) == []


def test_thread_path_problem_names_a_directory(tmp_path: Path) -> None:
    target = tmp_path / "w.codex-thread"
    target.mkdir()

    problem = runner.thread_path_problem(target)

    assert problem is not None and "not a regular file" in problem


def test_a_persist_failure_mid_turn_is_a_controlled_outcome(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the path breaks after launch, the turn still finishes cleanly.

    Raising out of a running turn would abandon work that is already having
    effects, and leave a traceback instead of an actionable exit code.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    def exploding_write(path: Path, thread_id: str) -> None:
        raise OSError("the state directory went away")

    monkeypatch.setattr(runner, "write_thread_id", exploding_write)

    assert _run(tmp_path, fake=fake) == runner.EXIT_STATE_UNUSABLE

    assert record.exists(), "the turn itself still ran to completion"
    log_text = (tmp_path / "state" / "slice-123.codex-wake.log").read_text(
        encoding="utf-8"
    )
    assert "do not retry blindly" in log_text


def test_supervisor_is_interposed_between_the_runner_and_codex(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent-death signal reaches one pid; a process group covers a tree."""
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A)
    captured: dict[str, object] = {}
    real_popen = runner.subprocess.Popen

    def recording_popen(argv: list[str], **kwargs: object) -> object:
        captured.setdefault("argv", argv)
        captured.setdefault("kwargs", kwargs)
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", recording_popen)

    assert _run(tmp_path, fake=fake) == 0

    argv = captured["argv"]
    assert "--supervise" in argv, "codex must be started under the supervisor"
    assert argv[argv.index("--supervise") + 1] == str(os.getpid())
    assert "--" in argv and argv[argv.index("--") + 1] == str(fake)
    # The lock descriptor is handed down so the group holds it.
    assert captured["kwargs"]["pass_fds"], "the wake lock fd must be inherited"


@pytest.mark.parametrize(
    "argv",
    [
        ["--supervise"],
        ["--supervise", "1"],
        ["--supervise", "1", "--"],
        ["--supervise", "not-a-pid", "--", "true"],
        ["--supervise", "1", "no-dashes", "true"],
    ],
)
def test_supervisor_entrypoint_rejects_a_malformed_invocation(
    argv: list[str],
) -> None:
    assert runner.main(argv) == 2


def test_supervise_rejects_an_empty_command() -> None:
    assert runner.supervise([], expected_ppid=os.getpid()) == 2


def test_supervisor_reports_a_missing_command_without_a_traceback(
    tmp_path: Path,
) -> None:
    """Run it as a real subprocess: setsid only works in a fresh process."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--supervise",
            str(os.getpid()),
            "--",
            str(tmp_path / "definitely-not-here"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "could not start" in result.stderr
    assert "Traceback" not in result.stderr


def test_supervisor_runs_its_command_and_returns_its_exit_code(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "ran.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--supervise",
            str(os.getpid()),
            "--",
            sys.executable,
            "-c",
            f"open({str(evidence)!r}, 'w').write('ran'); raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert evidence.read_text(encoding="utf-8") == "ran"
    assert result.returncode == 7, "the command's exit code must propagate"


def test_a_background_process_does_not_outlive_the_lock(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Regression: the turn is not over when its own process exits.

    Codex can start a background command that survives it. Returning then
    would release the wake lock while that command is still editing the
    checkout, so the next wake could overlap it.
    """
    background_pid = tmp_path / "background.pid"
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(40)'],"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open({str(background_pid)!r}, 'w').write(str(p.pid))\n"
        "sys.stdin.read()\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {THREAD_A!r}}}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}))\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    state_dir = tmp_path / "state"

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 0

    pid = int(background_pid.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{pid}").exists(), (
        "a process the turn left running must be stopped before the lock frees"
    )
    # The turn itself still succeeded and is still resumable.
    assert (state_dir / "slice-123.codex-thread").read_text(
        encoding="utf-8"
    ).strip() == THREAD_A


def test_a_clean_turn_is_not_slowed_by_the_drain(
    tmp_path: Path, repo_dir: Path
) -> None:
    """The other side: nothing left behind means nothing to wait for."""
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A)

    started = time.monotonic()
    assert _run(tmp_path, fake=fake) == 0
    elapsed = time.monotonic() - started

    assert elapsed < runner.GROUP_DRAIN_SECONDS, (
        "a turn that left nothing running must not wait out the drain deadline"
    )


def test_scan_reports_what_it_could_not_read(tmp_path: Path) -> None:
    """A scan that read almost nothing must not look like an empty group."""
    scan = runner.scan_process_group(os.getpgid(0), exclude_pid=os.getpid())

    assert isinstance(scan.members, list)
    assert scan.unreadable >= 0 and scan.vanished >= 0


def test_reap_children_returns_zero_with_nothing_to_reap() -> None:
    """ChildProcessError is how the loop ends, not a failure to hide."""
    assert runner._reap_children() == 0


@pytest.mark.parametrize(
    ("stat_line", "expected"),
    [
        # comm is parenthesized and may contain spaces, parentheses and ") ".
        # Splitting on the first ") " reads the ppid as the pgid and silently
        # drops that process from a scan.
        ("4242 (worker) hidden) S 1000 7777 7777 0 -1", 7777),
        ("9 ((paren)) S 5 2121 2121", 2121),
        ("9 (a b c) S 5 3131 3131", 3131),
        ("1 (systemd) S 0 1 1 0 -1", 1),
        ("1 (comm) S 1 4242 0 0", 4242),
        ("", None),
        ("1 (x) S", None),
        ("1 (comm with spaces) S 1 notanumber", None),
    ],
)
def test_pgid_of_parses_every_comm_shape(
    stat_line: str, expected: int | None
) -> None:
    assert runner._pgid_of(stat_line) == expected


def test_scan_finds_a_child_whose_name_contains_the_delimiter(
    tmp_path: Path,
) -> None:
    """The parser bug, driven through a real process rather than a fixture.

    A process named "worker) hidden" stays in the group. If its stat line is
    mis-parsed it vanishes from the scan, and the drain then reports an empty
    group and releases the lock while it is still running.
    """
    # comm comes from the executable name, capped at 15 characters, so this
    # copies a real binary rather than using a shebang script, whose comm
    # would become the interpreter's name instead.
    source = shutil.which("sleep")
    if source is None:
        pytest.skip("no sleep binary to copy")
    script = tmp_path / "worker) hidden"
    shutil.copy(source, script)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    child = subprocess.Popen([str(script), "30"])
    try:
        time.sleep(0.5)
        comm = Path(f"/proc/{child.pid}/comm").read_text(encoding="utf-8").strip()
        if ") " not in comm:
            pytest.skip(f"kernel did not keep the delimiter in comm: {comm!r}")

        members = runner.process_group_members(
            os.getpgid(child.pid), exclude_pid=os.getpid()
        )

        assert child.pid in members, (
            "a process whose name contains the stat delimiter must still be found"
        )
    finally:
        child.kill()
        child.wait()


def test_process_group_members_excludes_the_caller(tmp_path: Path) -> None:
    pgid = os.getpgid(0)

    members = runner.process_group_members(pgid, exclude_pid=os.getpid())

    assert os.getpid() not in members


def test_process_group_members_of_an_unused_group_is_empty() -> None:
    # A pgid that cannot be in use: larger than the configured pid ceiling.
    assert runner.process_group_members(2**31 - 1, exclude_pid=os.getpid()) == []


def test_supervisor_handlers_are_installed_before_the_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the spawn-to-handler window.

    If the runner dies after Codex starts but before the handlers exist, the
    parent-death signal takes its default action and kills only the
    supervisor. Codex survives without the lock descriptor, so the next wake
    can overlap it in the same checkout.
    """
    order: list[str] = []
    real_signal = runner.signal.signal
    real_popen = runner.subprocess.Popen

    def recording_signal(sig: int, handler: object) -> object:
        if sig in (runner.signal.SIGTERM, runner.signal.SIGINT, runner.signal.SIGHUP):
            order.append(f"handler:{sig}")
        return real_signal(sig, handler)

    def recording_popen(argv: list[str], **kwargs: object) -> object:
        order.append("spawn")
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(runner.signal, "signal", recording_signal)
    monkeypatch.setattr(runner.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(runner.os, "setsid", lambda: None)
    monkeypatch.setattr(runner, "drain_process_group", lambda **kwargs: None)

    exit_code = runner.supervise(
        [sys.executable, "-c", "pass"], expected_ppid=os.getppid()
    )

    assert exit_code == 0
    assert "spawn" in order
    assert order.index("spawn") > 0, "no handler was installed before the spawn"
    first_spawn = order.index("spawn")
    installed = [item for item in order[:first_spawn] if item.startswith("handler:")]
    assert len(installed) == 3, "all three termination signals must be covered first"


@pytest.mark.parametrize("shape", ["under-a-file", "unwritable-parent"])
def test_an_unusable_state_directory_exits_cleanly(
    tmp_path: Path, repo_dir: Path, shape: str
) -> None:
    """The first filesystem touch must not be the one that tracebacks.

    It happens before the log and lock exist, so it cannot rely on their
    guards; the runner promises a controlled diagnostic exit either way.
    """
    if shape == "under-a-file":
        blocker = tmp_path / "a-regular-file"
        blocker.write_text("not a directory", encoding="utf-8")
        state_dir = blocker / "state"
    else:
        parent = tmp_path / "locked"
        parent.mkdir()
        parent.chmod(0o500)
        state_dir = parent / "state"

    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    try:
        assert (
            _run(tmp_path, fake=fake, state_dir=state_dir)
            == runner.EXIT_STATE_UNUSABLE
        )
    finally:
        if shape == "unwritable-parent":
            (tmp_path / "locked").chmod(0o700)

    assert not record.exists(), "Codex must not be launched"


def test_drain_does_not_claim_to_cover_escaped_descendants() -> None:
    """The claim and the mechanism have to match.

    A descendant that calls setsid leaves the process group, and nothing built
    from process groups can stop it. Saying otherwise in the one place a
    maintainer looks is how a false guarantee survives.
    """
    doc = runner.drain_process_group.__doc__ or ""

    assert "setsid" in doc
    assert "does NOT cover" in doc
    assert "cgroup" in doc or "systemd scope" in doc


@pytest.mark.parametrize("shape", ["fifo", "directory"])
def test_a_special_thread_file_is_rejected_without_being_opened(
    tmp_path: Path, repo_dir: Path, shape: str
) -> None:
    """Regression: the type check must precede the open, not follow it.

    A FIFO reports a zero size and blocks on open until a writer appears, so
    a later is_file() check never runs. A real wake would hang holding the
    wake lock and every later wake would wait or time out.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    target = state_dir / "slice-123.codex-thread"
    if shape == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()

    # The read must return rather than block; pytest-timeout is not assumed,
    # so a hang here shows up as the suite itself never finishing, which the
    # wake log assertion below would never reach.
    stored, reason = runner.read_thread_id(target)

    assert stored is None
    assert reason is not None and "not a regular file" in reason

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == runner.EXIT_STATE_UNUSABLE
    assert not record.exists(), "Codex must not be launched"


def test_a_symlinked_thread_file_is_judged_on_its_own(tmp_path: Path) -> None:
    """lstat, so a link to a FIFO cannot smuggle the blocking open back in."""
    fifo = tmp_path / "target-fifo"
    os.mkfifo(fifo)
    link = tmp_path / "slice-123.codex-thread"
    link.symlink_to(fifo)

    stored, reason = runner.read_thread_id(link)

    assert stored is None
    assert reason is not None and "not a regular file" in reason


@pytest.mark.parametrize(
    "stderr_text",
    [
        "Error: MCP server session not found",
        "Error: upstream session not found",
        "Error: conversation not found in cache",
        "Error: no such thread in the pool",
        "Error: thread not found: retrying",
    ],
)
def test_an_unrelated_not_found_does_not_quarantine(
    tmp_path: Path, repo_dir: Path, stderr_text: str
) -> None:
    """Quarantine needs the canonical diagnostic, not any 'not found' text.

    An earlier matcher accepted bare phrases, so an unrelated subsystem saying
    "session not found" threw away a perfectly good thread. Losing the arc is
    the exact harm quarantining exists to prevent.
    """
    fake = _failing_codex(tmp_path, stderr_text=stderr_text)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    thread_path = state_dir / "slice-123.codex-thread"
    thread_path.write_text(THREAD_A + "\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 1

    assert thread_path.read_text(encoding="utf-8").strip() == THREAD_A
    assert not (state_dir / "slice-123.codex-thread.stale").exists()


def test_reports_missing_session_requires_the_phrase_and_the_id() -> None:
    """Both halves, because either alone appears in unrelated output."""
    canonical = (
        "Error: thread/resume: thread/resume failed: no rollout found for "
        f"thread id {THREAD_A} (code -32600)"
    )

    assert runner.reports_missing_session(canonical, THREAD_A)
    # The canonical phrase, but about a different thread.
    assert not runner.reports_missing_session(canonical, THREAD_B)
    # The id echoed by some other failure, without the diagnostic.
    assert not runner.reports_missing_session(f"Error: timeout for {THREAD_A}", THREAD_A)
    assert not runner.reports_missing_session("", THREAD_A)


def test_a_resume_reporting_a_different_thread_is_refused(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Regression: a resume must come back as the thread it asked for.

    Accepting a different id would silently redirect this arc and every later
    wake with it, which breaks the one promise this runner makes.
    """
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_B)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    thread_path = state_dir / "slice-123.codex-thread"
    thread_path.write_text(THREAD_A + "\n", encoding="utf-8")

    exit_code = _run(tmp_path, fake=fake, state_dir=state_dir)

    assert exit_code == runner.EXIT_NO_THREAD_EVENT
    assert thread_path.read_text(encoding="utf-8").strip() == THREAD_A, (
        "the stored arc must survive a mismatched resume"
    )
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "rather than switching arcs" in log_text


def test_a_fresh_turn_still_records_whatever_thread_it_started(
    tmp_path: Path, repo_dir: Path
) -> None:
    """The other side: with nothing stored, any valid id is the arc."""
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_B)
    state_dir = tmp_path / "state"

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 0

    assert (state_dir / "slice-123.codex-thread").read_text(
        encoding="utf-8"
    ).strip() == THREAD_B


def test_a_split_diagnostic_does_not_quarantine(tmp_path: Path) -> None:
    """The id must come out of the diagnostic, not be searched for separately.

    stderr that names the resumed thread on one line and reports a missing
    rollout for a different thread on another is evidence about that other
    conversation, not this one.
    """
    mixed = (
        f"Error: resuming thread {THREAD_A}\n"
        f"Error: no rollout found for thread id {THREAD_B} (code -32600)"
    )

    assert not runner.reports_missing_session(mixed, THREAD_A)
    assert runner.reports_missing_session(mixed, THREAD_B)


def test_a_wrong_thread_turn_is_stopped_before_it_can_act(
    tmp_path: Path, repo_dir: Path
) -> None:
    """Regression: refuse at the event that names it, not after the turn.

    Checking only after the subprocess exits gives the wrong conversation the
    whole turn to edit the checkout, push, or comment first.
    """
    marker = tmp_path / "wrong_conversation_acted.txt"
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        "sys.stdin.read()\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {THREAD_B!r}}}))\n"
        "sys.stdout.flush()\n"
        "time.sleep(3)\n"
        f"open({str(marker)!r}, 'w').write('acted')\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    thread_path = state_dir / "slice-123.codex-thread"
    thread_path.write_text(THREAD_A + "\n", encoding="utf-8")

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == runner.EXIT_NO_THREAD_EVENT

    assert not marker.exists(), "the wrong conversation must be stopped, not awaited"
    assert thread_path.read_text(encoding="utf-8").strip() == THREAD_A
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "stopping the turn now" in log_text


def test_drain_reports_how_many_it_could_not_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leak has to be countable, because the lock frees regardless."""
    messages: list[str] = []
    monkeypatch.setattr(runner, "process_group_members", lambda *a, **k: [999999])
    monkeypatch.setattr(
        runner,
        "scan_process_group",
        lambda *a, **k: runner.GroupScan(members=[999999], unreadable=0, vanished=0),
    )
    monkeypatch.setattr(runner.os, "killpg", lambda *a, **k: None)
    monkeypatch.setattr(runner.os, "kill", lambda *a, **k: None)

    leaked = runner.drain_process_group(
        deadline_seconds=0.0, report=messages.append
    )

    assert leaked == 1
    assert any("released when this process exits" in m for m in messages)


def test_usage_is_recorded_for_cost_visibility(tmp_path: Path, repo_dir: Path) -> None:
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake) == 0

    log_text = (tmp_path / "state" / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "input_tokens" in log_text
    assert "29048" in log_text


def test_agent_message_is_recorded_for_the_absent_operator(
    tmp_path: Path, repo_dir: Path
) -> None:
    """A wake runs while nobody is watching; its output must survive it."""
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake) == 0

    state_dir = tmp_path / "state"
    last = state_dir / "slice-123.codex-wake.last.md"
    assert last.read_text(encoding="utf-8").strip() == "ok"
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "agent message: ok" in log_text


def test_long_agent_message_is_truncated_in_the_log_but_kept_in_full(
    tmp_path: Path, repo_dir: Path
) -> None:
    long_text = "x" * 900
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {THREAD_A!r}}}))\n"
        "print(json.dumps({'type': 'item.completed', 'item': "
        f"{{'id': 'i', 'type': 'agent_message', 'text': {long_text!r}}}}}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}))\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    assert _run(tmp_path, fake=fake) == 0

    state_dir = tmp_path / "state"
    assert len(
        (state_dir / "slice-123.codex-wake.last.md").read_text(encoding="utf-8").strip()
    ) == 900
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "[truncated]" in log_text


def test_codex_exit_code_is_propagated(tmp_path: Path, repo_dir: Path) -> None:
    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A, exit_code=3)

    assert _run(tmp_path, fake=fake) == 3


def test_empty_prompt_is_refused(tmp_path: Path, repo_dir: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake, prompt="   \n") == 2
    assert not record.exists()


@pytest.mark.parametrize("watcher_id", ["../escape", ".hidden", "bad id", ""])
def test_rejects_unsafe_watcher_id(tmp_path: Path, repo_dir: Path, watcher_id: str) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake, watcher_id=watcher_id) == 2
    assert not record.exists()


def test_rejects_missing_repo_dir(tmp_path: Path) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    assert _run(tmp_path, fake=fake, repo_dir=tmp_path / "nope") == 2
    assert not record.exists()


def test_dry_run_prints_argv_without_invoking_codex(
    tmp_path: Path, repo_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    exit_code = runner.main(
        [
            "--watcher-id",
            "slice-123",
            "--repo-dir",
            str(repo_dir),
            "--state-dir",
            str(tmp_path / "state"),
            "--codex-bin",
            str(fake),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not record.exists()
    out = capsys.readouterr().out
    assert "mode=fresh" in out
    assert "--json" in out


def test_non_json_stdout_lines_are_counted_not_silently_dropped(
    tmp_path: Path, repo_dir: Path
) -> None:
    """A garbled stream must not look identical to a quiet one."""
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        "print('warning: this line is not json')\n"
        "print('[1, 2, 3]')\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {THREAD_A!r}}}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}))\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    assert _run(tmp_path, fake=fake) == 0

    log_text = (tmp_path / "state" / "slice-123.codex-wake.log").read_text(
        encoding="utf-8"
    )
    assert "2 stdout line(s) were not JSON events" in log_text
    assert "warning: this line is not json" in log_text
    # The valid events around them are still processed.
    assert (tmp_path / "state" / "slice-123.codex-thread").read_text(
        encoding="utf-8"
    ).strip() == THREAD_A


def test_build_argv_rejects_an_unknown_sandbox_mode() -> None:
    """build_argv is importable and interpolates this value straight into argv."""
    with pytest.raises(ValueError, match="unknown sandbox mode"):
        runner.build_argv(codex_bin="codex", thread_id=None, sandbox="wide-open")

    with pytest.raises(ValueError, match="unknown sandbox mode"):
        runner.build_argv(codex_bin="codex", thread_id=THREAD_A, sandbox="")


def test_unwritable_state_dir_exits_cleanly(tmp_path: Path, repo_dir: Path) -> None:
    """A background wake must not surface a traceback when its log cannot open."""
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "slice-123.codex-wake.log").mkdir()  # an open("a") on a dir fails

    assert _run(tmp_path, fake=fake, state_dir=state_dir) == 2
    assert not record.exists()


def test_temp_prompt_file_is_cleaned_up(tmp_path: Path, repo_dir: Path) -> None:
    import tempfile as _tempfile

    fake, _record = _fake_codex(tmp_path, thread_id=THREAD_A)
    before = set(Path(_tempfile.gettempdir()).glob("codex-wake-*.txt"))

    assert _run(tmp_path, fake=fake) == 0

    after = set(Path(_tempfile.gettempdir()).glob("codex-wake-*.txt"))
    assert after <= before


def test_missing_codex_binary_is_reported(tmp_path: Path, repo_dir: Path) -> None:
    assert _run(tmp_path, fake=tmp_path / "definitely-not-here") == 2
