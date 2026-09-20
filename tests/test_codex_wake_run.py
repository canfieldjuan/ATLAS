from __future__ import annotations

import fcntl
import importlib.util
import json
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
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_B)
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
MISSING_SESSION_STDERR = (
    "Error: thread/resume: thread/resume failed: no rollout found for thread id "
    "01a0bfff-dead-7000-a000-000000000000 (code -32600)"
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
    fake = _failing_codex(tmp_path, stderr_text=MISSING_SESSION_STDERR)
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
        tmp_path, stderr_text=MISSING_SESSION_STDERR, name="dead-codex"
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
