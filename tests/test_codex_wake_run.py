from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys

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


def test_concurrent_wake_queues_instead_of_dropping(
    tmp_path: Path, repo_dir: Path
) -> None:
    """A wake blocked on the lock must not discard its event.

    The running turn may already have taken its PR snapshot, so it cannot see a
    review posted after that point. The newer prompt is queued for it.
    """
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lock_path = state_dir / "slice-123.codex-wake.lock"

    with lock_path.open("w", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        assert _run(tmp_path, fake=fake, prompt="newer review thread") == 0

        assert not record.exists(), "a second wake must not invoke Codex directly"

    pending = state_dir / "slice-123.codex-wake.pending"
    assert pending.read_text(encoding="utf-8") == "newer review thread"
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "already running" in log_text
    assert "queued this prompt" in log_text


def _queueing_codex(tmp_path: Path, *, thread_id: str, queue_times: int) -> Path:
    """A fake Codex that simulates events arriving while the turn is running."""
    counter = tmp_path / "invocations"
    pending = tmp_path / "state" / "slice-123.codex-wake.pending"
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"counter = Path({str(counter)!r})\n"
        f"pending = Path({str(pending)!r})\n"
        "sys.stdin.read()\n"
        "n = int(counter.read_text()) if counter.exists() else 0\n"
        "n += 1\n"
        "counter.write_text(str(n))\n"
        f"if n <= {queue_times}:\n"
        "    pending.parent.mkdir(parents=True, exist_ok=True)\n"
        "    pending.write_text(f'queued prompt {n}')\n"
        f"print(json.dumps({{'type': 'thread.started', 'thread_id': {thread_id!r}}}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {}}))\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def test_lock_holder_drains_a_prompt_queued_mid_turn(
    tmp_path: Path, repo_dir: Path
) -> None:
    fake = _queueing_codex(tmp_path, thread_id=THREAD_A, queue_times=1)

    assert _run(tmp_path, fake=fake) == 0

    assert (tmp_path / "invocations").read_text(encoding="utf-8") == "2"
    state_dir = tmp_path / "state"
    assert not (state_dir / "slice-123.codex-wake.pending").exists()
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "draining a queued prompt; starting turn 2" in log_text
    assert "wake complete turns=2" in log_text


def test_coalescing_is_capped_and_hands_off(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review burst must not chain Codex turns in one process without bound.

    The remaining prompt is not stranded: a follow-up consumer is started for
    it before this process returns.
    """
    fake = _queueing_codex(tmp_path, thread_id=THREAD_A, queue_times=50)
    spawned: list[int] = []
    monkeypatch.setattr(
        runner,
        "spawn_handoff",
        lambda **kwargs: (spawned.append(kwargs["chain_depth"]), True)[1],
    )

    assert _run(tmp_path, fake=fake) == 0

    invocations = int((tmp_path / "invocations").read_text(encoding="utf-8"))
    assert invocations == runner.MAX_COALESCED_TURNS
    state_dir = tmp_path / "state"
    # The event that did not fit is preserved AND has a consumer.
    assert (state_dir / "slice-123.codex-wake.pending").exists()
    assert spawned == [1]
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "handed the queued prompt to a follow-up consumer" in log_text


def test_exhausted_handoff_chain_fails_loudly(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the end of the chain a strand must surface, not rot silently."""
    fake = _queueing_codex(tmp_path, thread_id=THREAD_A, queue_times=50)
    monkeypatch.setattr(runner, "spawn_handoff", lambda **kwargs: False)

    exit_code = runner.run_wake(
        watcher_id="slice-123",
        repo_dir=repo_dir,
        state_dir=tmp_path / "state",
        prompt="wake prompt",
        sandbox="workspace-write",
        codex_bin=str(fake),
        dry_run=False,
        chain_depth=runner.MAX_HANDOFF_CHAIN,
    )

    assert exit_code == runner.EXIT_QUEUE_NOT_DRAINED
    state_dir = tmp_path / "state"
    assert (state_dir / "slice-123.codex-wake.pending").exists()
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "no follow-up consumer" in log_text


def test_prompt_queued_during_a_claim_is_not_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the read-then-unlink race.

    A contender can atomically replace the pending file between a read and a
    delete. Deleting afterwards would destroy a prompt nobody read, while that
    contender already returned success. The claim is a rename, so the bytes
    taken are exactly the bytes removed.
    """
    pending = tmp_path / "slice-123.codex-wake.pending"
    runner.queue_pending_prompt(pending, "older snapshot")

    real_read_text = Path.read_text

    def racing_read(self: Path, *args: object, **kwargs: object) -> str:
        if ".claim." in self.name:
            # A second wake queues a newer snapshot mid-claim.
            runner.queue_pending_prompt(pending, "newer snapshot")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", racing_read)
    first, reason = runner.take_pending_prompt(pending)
    monkeypatch.undo()

    assert first == "older snapshot"
    assert reason is None
    second, _ = runner.take_pending_prompt(pending)
    assert second == "newer snapshot", "a prompt queued mid-claim must survive"


def test_claim_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    pending = tmp_path / "slice-123.codex-wake.pending"
    runner.queue_pending_prompt(pending, "a prompt")

    runner.take_pending_prompt(pending)

    assert list(tmp_path.glob("*claim*")) == []
    assert not pending.exists()


def test_spawn_handoff_builds_a_drain_invocation(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    log = tmp_path / "log.txt"
    with log.open("w", encoding="utf-8") as handle:
        assert runner.spawn_handoff(
            watcher_id="slice-123",
            repo_dir=repo_dir,
            state_dir=tmp_path / "state",
            sandbox="read-only",
            codex_bin="codex",
            chain_depth=2,
            log_handle=handle,
        )

    argv = captured["argv"]
    assert "--drain-pending" in argv
    assert argv[argv.index("--chain-depth") + 1] == "2"
    assert argv[argv.index("--watcher-id") + 1] == "slice-123"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert captured["kwargs"]["start_new_session"] is True


def test_drain_pending_consumes_the_queue_without_stdin(
    tmp_path: Path, repo_dir: Path
) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    runner.queue_pending_prompt(
        state_dir / "slice-123.codex-wake.pending", "queued work"
    )

    exit_code = runner.run_wake(
        watcher_id="slice-123",
        repo_dir=repo_dir,
        state_dir=state_dir,
        prompt="",
        sandbox="workspace-write",
        codex_bin=str(fake),
        dry_run=False,
        drain_pending=True,
        chain_depth=1,
    )

    assert exit_code == 0
    assert json.loads(record.read_text(encoding="utf-8"))["stdin"] == "queued work"
    assert not (state_dir / "slice-123.codex-wake.pending").exists()


def test_drain_pending_with_an_empty_queue_is_a_noop(
    tmp_path: Path, repo_dir: Path
) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    exit_code = runner.run_wake(
        watcher_id="slice-123",
        repo_dir=repo_dir,
        state_dir=tmp_path / "state",
        prompt="",
        sandbox="workspace-write",
        codex_bin=str(fake),
        dry_run=False,
        drain_pending=True,
        chain_depth=1,
    )

    assert exit_code == 0
    assert not record.exists()


def test_drain_pending_gives_up_when_the_lock_is_held(
    tmp_path: Path, repo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Giving up is safe: the holder re-checks the queue after every turn."""
    monkeypatch.setattr(runner, "LOCK_WAIT_SECONDS", 1)
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    runner.queue_pending_prompt(
        state_dir / "slice-123.codex-wake.pending", "queued work"
    )
    lock_path = state_dir / "slice-123.codex-wake.lock"

    with lock_path.open("w", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        exit_code = runner.run_wake(
            watcher_id="slice-123",
            repo_dir=repo_dir,
            state_dir=state_dir,
            prompt="",
            sandbox="workspace-write",
            codex_bin=str(fake),
            dry_run=False,
            drain_pending=True,
            chain_depth=1,
        )

    assert exit_code == 0
    assert not record.exists()
    # The prompt is still queued for whoever holds the lock.
    assert (state_dir / "slice-123.codex-wake.pending").exists()
    log_text = (state_dir / "slice-123.codex-wake.log").read_text(encoding="utf-8")
    assert "gave up waiting" in log_text


@pytest.mark.parametrize("depth", [-1, 99])
def test_rejects_out_of_range_chain_depth(
    tmp_path: Path, repo_dir: Path, depth: int
) -> None:
    fake, record = _fake_codex(tmp_path, thread_id=THREAD_A)

    exit_code = runner.main(
        [
            "--watcher-id", "slice-123",
            "--repo-dir", str(repo_dir),
            "--state-dir", str(tmp_path / "state"),
            "--codex-bin", str(fake),
            "--drain-pending",
            "--chain-depth", str(depth),
        ]
    )

    assert exit_code == 2
    assert not record.exists()


def test_queued_prompt_is_the_newest_one(tmp_path: Path, repo_dir: Path) -> None:
    """Each bridge prompt is a full PR snapshot, so newest supersedes older."""
    state_dir = tmp_path / "state"
    pending = state_dir / "slice-123.codex-wake.pending"

    runner.queue_pending_prompt(pending, "older snapshot")
    runner.queue_pending_prompt(pending, "newer snapshot")

    taken, reason = runner.take_pending_prompt(pending)
    assert taken == "newer snapshot"
    assert reason is None
    assert not pending.exists()


def test_empty_queued_prompt_is_discarded(tmp_path: Path) -> None:
    pending = tmp_path / "state" / "slice-123.codex-wake.pending"
    runner.queue_pending_prompt(pending, "   \n")

    taken, reason = runner.take_pending_prompt(pending)

    assert taken is None
    assert reason == "queued prompt was empty"


def test_oversized_queued_prompt_is_discarded(tmp_path: Path) -> None:
    pending = tmp_path / "state" / "slice-123.codex-wake.pending"
    runner.queue_pending_prompt(pending, "x" * (runner.MAX_PENDING_FILE_BYTES + 1))

    taken, reason = runner.take_pending_prompt(pending)

    assert taken is None
    assert reason == "queued prompt was too large; discarded"
    assert not pending.exists()


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
