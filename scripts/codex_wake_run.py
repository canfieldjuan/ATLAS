#!/usr/bin/env python3
"""Run Codex wake turns against a persistent per-watcher thread.

The PR watcher and `codex_wake_bridge.py` decide whether a wake should happen
and what the prompt says. This runner is the last stage: it takes that prompt
on stdin and spends a Codex turn on it, resuming the watcher's existing Codex
thread when there is one.

Resuming matters for cost and for continuity. A fresh `codex exec` rebuilds the
whole conversation from nothing, so the woken agent does not remember the arc it
is continuing and the operator pays full cold-start context on every wake.

This runner never merges, closes, or pushes anything. It starts Codex turns and
records what they cost.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, NamedTuple, Sequence


DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "atlas-pr-watchers"
DEFAULT_SANDBOX = "workspace-write"
DEFAULT_CODEX_BIN = "codex"

SAFE_WATCHER_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
# Codex session ids are UUID-shaped. Anything else must not reach argv.
THREAD_ID_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
# A stored thread id is one short line; refuse to read more than that.
MAX_THREAD_FILE_BYTES = 256
# A wake waits this long for the lock before reporting failure. Waiting is how
# a concurrent wake is preserved: it runs its own turn rather than handing its
# prompt to someone else, so there is no queue to strand.
LOCK_WAIT_SECONDS = 600
# Exit code when a wake never got the lock. Distinct so a caller can tell a
# contended wake from a Codex failure.
EXIT_LOCK_TIMEOUT = 75
# Exit code when Codex reports success but never named a thread. Without that
# id there is nothing to resume, so reporting success would silently forfeit
# the continuity this runner exists to provide.
EXIT_NO_THREAD_EVENT = 76
# Exit code used by the pre-exec hook when the runner died before Codex was
# started. Never seen by the runner itself, which is gone by then.
EXIT_PARENT_GONE = 77

SANDBOX_CHOICES = ("read-only", "workspace-write", "danger-full-access")

# Codex reports a resume against an unknown session on stderr, not as a JSON
# event. Verified against codex-cli 0.155.1, which prints:
#   Error: thread/resume: thread/resume failed: no rollout found for thread
#   id <uuid> (code -32600)
# Quarantining is keyed to this signature specifically. A generic pre-attach
# failure -- network, expired auth, bad local config -- must NOT discard a
# valid id, because that permanently loses the arc this runner exists to keep.
MISSING_SESSION_RE = re.compile(
    r"no rollout found for thread|(?:thread|session|conversation) not found"
    r"|no such (?:thread|session|conversation)",
    re.IGNORECASE,
)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# PR_SET_PDEATHSIG on Linux.
_PR_SET_PDEATHSIG = 1


def parent_death_signal_support() -> tuple[Any, str | None]:
    """Resolve libc in the PARENT, so an unavailable kernel feature is visible.

    The probe cannot live in the pre-exec hook: that code runs after fork in a
    child with no safe way to report anything, so a failure there would be
    silent exactly where it matters. Resolving here lets the caller log that
    orphan protection is not in force on this host.
    """
    if not sys.platform.startswith("linux"):
        return None, f"parent-death signal is Linux-only; not available on {sys.platform}"
    try:
        import ctypes

        return ctypes.CDLL("libc.so.6", use_errno=True), None
    except (OSError, ImportError) as exc:
        return None, f"could not load libc for the parent-death signal: {exc}"


def make_die_with_parent(expected_ppid: int) -> Any:
    """Build the pre-exec hook that binds this child to the runner's life.

    Without it the Codex child outlives a killed runner. The runner's death
    releases the wake lock, so a later wake can acquire it and start a second
    turn against the same thread and the same checkout while the orphan is
    still editing files. A signal handler cannot cover that, because the
    runner may be SIGKILLed; PR_SET_PDEATHSIG is enforced by the kernel.

    The flag alone leaves a window. It is set after fork, and if the runner
    dies in between, the kernel has already reparented this child and setting
    the flag afterwards delivers nothing, because it is not retroactive. So
    the hook also compares its parent against the pid captured before the
    fork and leaves if it has already been orphaned. That check matters even
    where the kernel facility is unavailable, so it is not conditional on it.
    """

    def _hook() -> None:
        libc, _reason = parent_death_signal_support()
        if libc is not None:
            libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if os.getppid() != expected_ppid:
            # Already reparented: no death signal is coming, and the lock this
            # child was covered by is gone. Leave before exec'ing Codex.
            os._exit(EXIT_PARENT_GONE)

    return _hook


def _log(handle: Any, message: str) -> None:
    handle.write(f"[{_now()}] {message}\n")
    handle.flush()


def valid_watcher_id(watcher_id: str) -> bool:
    """Reject ids that could escape the state directory."""
    if not SAFE_WATCHER_ID_RE.fullmatch(watcher_id):
        return False
    return ".." not in watcher_id and not watcher_id.startswith(".")


def valid_thread_id(value: str) -> bool:
    """Accept only the canonical Codex session-id shape.

    `fullmatch` on the already-stripped value, so a trailing newline or a
    leading dash cannot smuggle argv through.
    """
    return bool(THREAD_ID_RE.fullmatch(value))


def read_thread_id(path: Path) -> tuple[str | None, str | None]:
    """Return (thread_id, reason_ignored).

    Any unusable stored value degrades to a fresh thread rather than failing the
    wake: a wake that runs on a new thread still does the operator's work, but a
    wake that refuses to run loses the review event entirely.
    """
    if not path.exists():
        return None, None
    try:
        if path.stat().st_size > MAX_THREAD_FILE_BYTES:
            return None, "stored thread id file is too large"
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not read stored thread id: {exc}"
    except UnicodeDecodeError:
        return None, "stored thread id is not valid UTF-8"
    candidate = raw.strip()
    if not candidate:
        return None, "stored thread id is empty"
    if not valid_thread_id(candidate):
        return None, "stored thread id is not a Codex session id"
    return candidate, None


def _atomic_write(path: Path, text: str) -> None:
    """Write durably enough to survive a host restart, not just process death.

    `os.replace` alone makes the swap atomic but says nothing about when the
    bytes or the directory entry reach the disk. Without both fsyncs, a power
    loss right after a wake can leave the thread id missing or empty, and the
    next wake starts a second thread against a session Codex already created.
    The thread id is the one piece of state this runner exists to keep, so it
    is worth two fsyncs on a file written once per wake.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with staged.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
        # The rename itself is directory metadata and needs its own flush.
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        staged.unlink(missing_ok=True)


def write_thread_id(path: Path, thread_id: str) -> None:
    """Persist atomically so a killed wake cannot leave a partial id."""
    _atomic_write(path, thread_id + "\n")


def build_argv(*, codex_bin: str, thread_id: str | None, sandbox: str) -> list[str]:
    """Build the Codex argv for a fresh or resumed wake.

    `codex exec` and `codex exec resume` do not share a flag surface. Verified
    against codex-cli 0.155.1: `exec` accepts `-C/--cd` and `-s/--sandbox`;
    `exec resume` accepts neither. So this uses only what both accept -- the
    working directory is passed as the subprocess cwd, and the sandbox as a
    generic `-c` config override -- and the two paths stay one shape.
    """
    if sandbox not in SANDBOX_CHOICES:
        # argparse already constrains the CLI, but build_argv is importable and
        # this value is interpolated straight into argv.
        raise ValueError(f"unknown sandbox mode: {sandbox!r}")
    argv = [codex_bin, "exec"]
    if thread_id is not None:
        argv.append("resume")
        argv.append(thread_id)
    argv.extend(["--json", "-c", f'sandbox_mode="{sandbox}"'])
    # Trailing "-" makes Codex read the prompt from stdin.
    argv.append("-")
    return argv


def _consume_events(
    stream: Any,
    log_handle: Any,
    on_thread_started: Any = None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Scan the JSONL event stream for the thread id, usage, and final message.

    The final message is the only record of what the woken agent actually did.
    A wake happens while the operator is away, so dropping it would leave the
    wake unauditable.
    """
    thread_id: str | None = None
    usage: dict[str, Any] | None = None
    last_message: str | None = None
    # Codex can interleave non-JSON lines into stdout. Skipping them is right,
    # but skipping them silently makes a garbled stream look like a quiet one,
    # so they are counted and reported once at the end of the turn.
    unparsed = 0
    first_unparsed = ""
    for line in stream:
        text = line.strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            unparsed += 1
            if not first_unparsed:
                first_unparsed = text[:200]
            continue
        if not isinstance(event, dict):
            unparsed += 1
            if not first_unparsed:
                first_unparsed = text[:200]
            continue
        kind = event.get("type")
        if kind == "thread.started":
            candidate = event.get("thread_id")
            if isinstance(candidate, str) and valid_thread_id(candidate.strip()):
                thread_id = candidate.strip()
                # Persist immediately, not at end of turn. Codex has already
                # created the session by this point; if the runner is killed,
                # the unit stopped, or the host restarts before the turn ends,
                # an unrecorded id means the next wake starts a second thread
                # and the arc is lost.
                if on_thread_started is not None:
                    on_thread_started(thread_id)
        elif kind == "turn.completed":
            reported = event.get("usage")
            if isinstance(reported, dict):
                usage = reported
        elif kind == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                message = item.get("text")
                if isinstance(message, str) and message.strip():
                    last_message = message
        elif kind == "error":
            _log(log_handle, f"codex error event: {text[:500]}")
    if unparsed:
        _log(
            log_handle,
            f"{unparsed} stdout line(s) were not JSON events; first: {first_unparsed}",
        )
    return thread_id, usage, last_message


class TurnResult(NamedTuple):
    """Outcome of one Codex turn."""

    exit_code: int
    binary_missing: bool = False


def run_one_turn(
    *,
    watcher_id: str,
    repo_dir: Path,
    thread_path: Path,
    state_dir: Path,
    prompt: str,
    sandbox: str,
    codex_bin: str,
    log_handle: Any,
) -> TurnResult:
    """Spend exactly one Codex turn. The caller must already hold the wake lock."""
    # Read the thread id under the lock. Reading it before acquiring the lock
    # leaves a stale-read interleaving: this process can read "no thread" while
    # another holds the lock, that one records a new id and releases, and this
    # one then acquires the lock still holding None, starts a second thread,
    # and overwrites the id. One thread per watcher is the contract, so the
    # read has to happen inside the critical section.
    thread_id, ignored_reason = read_thread_id(thread_path)
    if ignored_reason:
        _log(log_handle, f"starting fresh thread: {ignored_reason}")

    argv = build_argv(codex_bin=codex_bin, thread_id=thread_id, sandbox=sandbox)
    mode = "resume" if thread_id else "fresh"
    _log(log_handle, f"wake {mode} watcher={watcher_id} repo={repo_dir}")
    _log(log_handle, "argv=" + " ".join(argv))

    _libc, pdeath_reason = parent_death_signal_support()
    if pdeath_reason:
        _log(
            log_handle,
            f"{pdeath_reason}; the orphan check still applies, but a runner "
            "killed mid-turn cannot signal this Codex process",
        )
    # Captured before the fork: the hook compares against it to detect having
    # been reparented while it was setting the death signal.
    die_with_parent = make_die_with_parent(os.getpid())

    def _persist(new_id: str) -> None:
        if new_id != thread_id:
            write_thread_id(thread_path, new_id)
            _log(log_handle, f"recorded thread id {new_id}")

    prompt_fd, prompt_name = tempfile.mkstemp(prefix="codex-wake-", suffix=".txt")
    prompt_file = Path(prompt_name)
    # Codex reports a missing session on stderr, so it is captured rather than
    # streamed straight to the log: the text has to be inspected before the
    # stored id can be judged dead. It is appended to the log either way.
    stderr_fd, stderr_name = tempfile.mkstemp(prefix="codex-wake-err-", suffix=".txt")
    stderr_file = Path(stderr_name)
    stderr_text = ""
    try:
        with os.fdopen(prompt_fd, "w", encoding="utf-8") as prompt_handle:
            prompt_handle.write(prompt)
        with prompt_file.open("r", encoding="utf-8") as stdin_handle:
            with os.fdopen(stderr_fd, "w", encoding="utf-8") as stderr_handle:
                try:
                    process = subprocess.Popen(
                        argv,
                        cwd=str(repo_dir),
                        stdin=stdin_handle,
                        stdout=subprocess.PIPE,
                        stderr=stderr_handle,
                        text=True,
                        preexec_fn=die_with_parent,
                    )
                except FileNotFoundError:
                    _log(log_handle, f"codex binary not found: {codex_bin}")
                    print(f"codex binary not found: {codex_bin}", file=sys.stderr)
                    return TurnResult(2, binary_missing=True)
                with process:
                    observed_thread, usage, last_message = _consume_events(
                        process.stdout, log_handle, on_thread_started=_persist
                    )
                exit_code = process.returncode
        try:
            stderr_text = stderr_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _log(log_handle, f"could not read codex stderr: {exc}")
        if stderr_text.strip():
            for line in stderr_text.splitlines():
                if line.strip():
                    _log(log_handle, f"codex stderr: {line.rstrip()}")
    finally:
        for temp in (prompt_file, stderr_file):
            try:
                temp.unlink()
            except OSError as exc:
                _log(log_handle, f"could not remove temp file {temp}: {exc}")

    if thread_id is not None and observed_thread is None and exit_code != 0:
        if MISSING_SESSION_RE.search(stderr_text):
            # Confirmed: Codex says this session does not exist. Leaving the id
            # in place would make every future wake retry the same dead resume,
            # so it is quarantined and the next wake starts fresh. Quarantine
            # rather than delete, so the id stays inspectable.
            stale_path = thread_path.with_name(thread_path.name + ".stale")
            try:
                os.replace(thread_path, stale_path)
            except OSError as exc:
                _log(log_handle, f"could not quarantine missing-session id: {exc}")
            else:
                _log(
                    log_handle,
                    f"codex reports no such session; quarantined {thread_id} to "
                    f"{stale_path}. The next wake starts fresh.",
                )
        else:
            # Unrecognized pre-attach failure: network, expired auth, bad local
            # config, or a reworded missing-session error. Keep the id. A wrong
            # quarantine permanently loses the arc, while keeping it costs only
            # a retry on the next wake. Log loudly so a genuinely dead session
            # that stops matching the signature is still visible.
            _log(
                log_handle,
                f"resume of {thread_id} failed before attaching (exit {exit_code}) "
                "and codex did not report a missing session; keeping the id. If "
                "wakes keep failing this way, remove "
                f"{thread_path} to force a fresh thread.",
            )

    if usage is not None:
        _log(log_handle, "usage=" + json.dumps(usage, sort_keys=True))

    if last_message is not None:
        summary = " ".join(last_message.split())
        if len(summary) > 500:
            summary = summary[:500] + " [truncated]"
        _log(log_handle, f"agent message: {summary}")
        # Everything past this point is diagnostics about a turn that already
        # happened. Codex may have edited files, pushed, or commented, so a
        # failure to record that must not turn a completed turn into a reported
        # failure: the caller would retry and repeat real side effects.
        last_path = state_dir / f"{watcher_id}.codex-wake.last.md"
        try:
            last_path.write_text(last_message + "\n", encoding="utf-8")
        except OSError as exc:
            _log(
                log_handle,
                f"could not write the agent message to {last_path}: {exc}; "
                "the turn itself completed and is not being failed for this",
            )
        else:
            _log(log_handle, f"full agent message written to {last_path}")
    else:
        _log(log_handle, "no agent message in this turn")

    if exit_code == 0 and observed_thread is None:
        # Codex claimed success but never named a thread, so nothing was
        # recorded to resume. A schema change or a truncated event stream both
        # land here. Reporting success would mean the next wake silently starts
        # yet another fresh thread and the continuity guarantee quietly lapses.
        _log(
            log_handle,
            "codex exited 0 but emitted no valid thread.started event; "
            "no thread id was recorded, so this turn cannot be resumed",
        )
        _log(log_handle, f"turn complete exit={EXIT_NO_THREAD_EVENT}")
        return TurnResult(EXIT_NO_THREAD_EVENT)

    _log(log_handle, f"turn complete exit={exit_code}")
    return TurnResult(exit_code)


def run_wake(
    *,
    watcher_id: str,
    repo_dir: Path,
    state_dir: Path,
    prompt: str,
    sandbox: str,
    codex_bin: str,
    dry_run: bool,
) -> int:
    state_dir.mkdir(parents=True, exist_ok=True)
    thread_path = state_dir / f"{watcher_id}.codex-thread"
    lock_path = state_dir / f"{watcher_id}.codex-wake.lock"
    log_path = state_dir / f"{watcher_id}.codex-wake.log"

    if dry_run:
        thread_id, ignored_reason = read_thread_id(thread_path)
        argv = build_argv(codex_bin=codex_bin, thread_id=thread_id, sandbox=sandbox)
        print(f"mode={'resume' if thread_id else 'fresh'}")
        print(f"cwd={repo_dir}")
        if ignored_reason:
            print(f"ignored_stored_thread_id={ignored_reason}")
        print("argv=" + " ".join(argv))
        return 0

    try:
        log_handle = log_path.open("a", encoding="utf-8")
    except OSError as exc:
        print(f"cannot open wake log {log_path}: {exc}", file=sys.stderr)
        return 2

    with log_handle:
        try:
            lock_handle = lock_path.open("w", encoding="utf-8")
        except OSError as exc:
            _log(log_handle, f"cannot open wake lock {lock_path}: {exc}")
            print(f"cannot open wake lock {lock_path}: {exc}", file=sys.stderr)
            return 2

        with lock_handle:
            # Wait for the lock instead of handing this prompt to whoever holds
            # it. A file-based handoff cannot be made strand-free: every version
            # of it left a window between the holder's last queue check and its
            # release in which a contender could enqueue and return success with
            # nobody left to consume. Waiting removes the queue, so the only
            # shared state is the lock itself and every wake that acquires it
            # runs its own turn with its own prompt.
            deadline = time.monotonic() + LOCK_WAIT_SECONDS
            acquired = False
            while True:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(1.0)
                    continue
                acquired = True
                break

            if not acquired:
                _log(
                    log_handle,
                    f"gave up after {LOCK_WAIT_SECONDS}s waiting for the "
                    f"{watcher_id} wake lock; this wake did not run",
                )
                print(
                    f"timed out waiting for the {watcher_id} wake lock",
                    file=sys.stderr,
                )
                return EXIT_LOCK_TIMEOUT

            result = run_one_turn(
                watcher_id=watcher_id,
                repo_dir=repo_dir,
                thread_path=thread_path,
                state_dir=state_dir,
                prompt=prompt,
                sandbox=sandbox,
                codex_bin=codex_bin,
                log_handle=log_handle,
            )
            _log(log_handle, f"wake complete exit={result.exit_code}")
            return result.exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watcher-id", required=True)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--sandbox",
        choices=SANDBOX_CHOICES,
        default=DEFAULT_SANDBOX,
        help=(
            "Codex sandbox policy for this wake. The prompt is built from PR "
            "and review text, which the watcher treats as untrusted, so the "
            "default stays workspace-write."
        ),
    )
    parser.add_argument("--codex-bin", default=DEFAULT_CODEX_BIN)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the argv this wake would run and exit without calling Codex.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not valid_watcher_id(args.watcher_id):
        print(f"invalid watcher id: {args.watcher_id!r}", file=sys.stderr)
        return 2

    repo_dir = args.repo_dir.expanduser()
    if not repo_dir.is_dir():
        print(f"repo dir is not a directory: {repo_dir}", file=sys.stderr)
        return 2

    prompt = "" if args.dry_run else sys.stdin.read()
    if not args.dry_run and not prompt.strip():
        print("refusing to start a Codex turn with an empty prompt", file=sys.stderr)
        return 2

    return run_wake(
        watcher_id=args.watcher_id,
        repo_dir=repo_dir,
        state_dir=args.state_dir.expanduser(),
        prompt=prompt,
        sandbox=args.sandbox,
        codex_bin=args.codex_bin,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
