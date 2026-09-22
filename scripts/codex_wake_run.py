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
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
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
# Exit code when the runner cannot keep the state a turn would produce.
# Raised before launch where possible, so retrying is safe.
EXIT_STATE_UNUSABLE = 78
# Exit code when a turn ended with processes of its own still running.
# The lock cannot be held past process exit, so the only honest option is
# to make the leak loud instead of returning success over it.
EXIT_TURN_NOT_CONTAINED = 79
# Exit code when a turn finished without ever saying what it did. The wake
# record is the only account of work done while nobody was watching.
EXIT_NO_AGENT_MESSAGE = 80
# Exit code when a turn finished without ever saying what it cost. This runner
# exists because wakes were burning a weekly quota unattended, so a turn whose
# price is unknown is a failed wake even when its work succeeded.
EXIT_NO_USAGE = 81
# How long the supervisor gives a turn's leftovers to exit on SIGTERM
# before killing them, once the turn itself has finished.
GROUP_DRAIN_SECONDS = 5.0

SANDBOX_CHOICES = ("read-only", "workspace-write", "danger-full-access")

# Codex reports a resume against an unknown session on stderr, not as a JSON
# event. Verified against codex-cli 0.155.1, which prints:
#   Error: thread/resume: thread/resume failed: no rollout found for thread
#   id <uuid> (code -32600)
# Quarantining is keyed to this signature specifically. A generic pre-attach
# failure -- network, expired auth, bad local config -- must NOT discard a
# valid id, because that permanently loses the arc this runner exists to keep.
# Only the diagnostic Codex actually emits for an unknown session, verified
# against codex-cli 0.155.1:
#   Error: thread/resume: thread/resume failed: no rollout found for thread
#   id <uuid> (code -32600)
# An earlier version also accepted bare phrases like "session not found", which
# an unrelated message such as "MCP server session not found" satisfies. That
# threw away a valid thread over a failure that had nothing to do with it,
# which is the exact loss quarantining exists to prevent.
MISSING_SESSION_RE = re.compile(
    r"no rollout found for thread id\s+(?P<thread>[0-9a-fA-F-]{36})",
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


class GroupScan(NamedTuple):
    """Live pids in a process group, plus what the scan could not read.

    Scanning /proc races with processes exiting, so entries disappearing is
    normal. The count is still carried out rather than discarded: a scan that
    could read almost nothing would otherwise look identical to an empty group
    and quietly license releasing the lock.
    """

    members: list[int]
    unreadable: int
    vanished: int


# A process in Z state has already exited and is only waiting to be reaped.
# It holds no files and runs no code, so counting it as a live group member
# turns a finished turn into a reported failure.
ZOMBIE_STATE = "Z"


def _pgid_of(stat_line: str) -> int | None:
    """Parse the process group of a LIVE process from one /proc/<pid>/stat line.

    Returns None for a zombie, because the question this answers is whether
    anything from the turn is still running, not whether an entry still exists.

    The comm field is parenthesized and may itself contain spaces, parentheses
    and even ") ". Splitting on the FIRST ") " therefore mis-parses a process
    named something like "worker) hidden", reading its parent pid as its group
    and dropping it from a scan. comm is the only parenthesized field, so the
    LAST ")" is where it really ends.
    """
    _, _, rest = stat_line.rpartition(")")
    fields = rest.split()
    # After the state field come ppid and pgrp.
    state, _ppid, pgrp, *_remainder = (*fields, None, None, None)
    if state == ZOMBIE_STATE:
        return None
    if not isinstance(pgrp, str) or not pgrp.isdigit():
        return None
    return int(pgrp)


def scan_process_group(pgid: int, *, exclude_pid: int) -> GroupScan:
    """Find the live pids in `pgid`, ignoring one pid (normally ourselves)."""
    members: list[int] = []
    unreadable = 0
    vanished = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return GroupScan(members, unreadable=1, vanished=0)
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == exclude_pid:
            continue
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
                stat_line = handle.read()
        except FileNotFoundError:
            # Exited between listing and reading. Expected while a group is
            # winding down, and counted so a caller can tell a quiet scan from
            # one that raced with a stampede of exits.
            vanished += 1
            continue
        except (OSError, UnicodeDecodeError):
            unreadable += 1
            continue
        if _pgid_of(stat_line) == pgid:
            members.append(pid)
    return GroupScan(members, unreadable, vanished)


def process_group_members(pgid: int, *, exclude_pid: int) -> list[int]:
    """Convenience wrapper for callers that only need the pids."""
    return scan_process_group(pgid, exclude_pid=exclude_pid).members


def drain_process_group(*, deadline_seconds: float, report: Any) -> int:
    """Stop what the turn left running in this process group.

    The turn's own process exiting does not mean the turn is over. Codex can
    start a background command that outlives it, and returning here would
    release the wake lock while that command is still editing the checkout, so
    the next wake could overlap it.

    This covers the supervisor's process group, which is what every ordinary
    child and grandchild inherits. It does NOT cover a descendant that calls
    `setsid` and deliberately leaves the group: nothing built from process
    groups can, because leaving is the descendant's choice. Containment a
    descendant cannot opt out of needs a cgroup or a systemd scope, which is
    tracked separately and is not built here.
    """
    me = os.getpid()
    try:
        pgid = os.getpgid(0)
    except OSError as exc:
        report(f"supervisor could not read its process group: {exc}")
        return 0

    scan = scan_process_group(pgid, exclude_pid=me)
    if scan.vanished:
        report(f"{scan.vanished} process(es) exited while the group was scanned")
    if scan.unreadable:
        # Not knowing whether anything is left is not the same as nothing
        # being left. Reporting a clean turn here would release the lock on an
        # assumption, so it is counted as uncontained and the caller fails.
        report(
            f"could not read {scan.unreadable} /proc entries, so this turn's "
            "containment could not be established"
        )
        return scan.unreadable
    if not scan.members:
        return 0
    report(f"turn left {len(scan.members)} process(es) running; stopping them")

    # Ignore the signal we are about to send to the whole group, so the
    # supervisor survives long enough to reap and report.
    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError as exc:
            report(f"supervisor could not signal the turn's group: {exc}")

        end = time.monotonic() + deadline_seconds
        while time.monotonic() < end:
            _reap_children()
            if not process_group_members(pgid, exclude_pid=me):
                return 0
            time.sleep(0.1)

        already_gone = 0
        for pid in process_group_members(pgid, exclude_pid=me):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                # Exited on its own between the scan and the kill.
                already_gone += 1
            except OSError as exc:
                report(f"supervisor could not stop pid {pid}: {exc}")
        reaped = _reap_children()
        if already_gone or reaped:
            report(
                f"drain reaped {reaped} and found {already_gone} already gone"
            )
        remaining = process_group_members(pgid, exclude_pid=me)
        if remaining:
            # The lock cannot be held past this process exiting, so it WILL be
            # released with these still alive. Saying so is the only honest
            # option; the caller turns it into a non-zero exit so a leak is
            # visible rather than reported as a clean turn.
            report(
                f"supervisor could not stop {remaining}; the wake lock is "
                "released when this process exits, so a later wake can overlap "
                "them"
            )
        return len(remaining)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _reap_children() -> int:
    """Collect finished children so they do not linger as zombies.

    Returns how many were reaped. ChildProcessError means there is nothing
    left to reap, which is how this loop is meant to end rather than a failure,
    so it is turned into the count instead of being re-raised.
    """
    reaped = 0
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        except OSError as exc:
            print(f"supervisor could not reap children: {exc}", file=sys.stderr)
            return reaped
        if pid == 0:
            return reaped
        reaped += 1


def supervise(codex_argv: Sequence[str], *, expected_ppid: int) -> int:
    """Run Codex as the leader of its own process group and take the group down.

    The runner cannot reap Codex's descendants itself. The parent-death signal
    reaches exactly one pid, and a shell or test runner Codex starts underneath
    survives it; the wake lock is then released while that descendant is still
    editing the checkout, so the next wake can overlap it. Passing the lock
    descriptor does not close the gap either, because an intermediate process
    that closes inherited descriptors breaks the chain.

    So a supervisor sits between them. It calls setsid, which makes it the
    leader of a new process group that every descendant inherits, and it holds
    the inherited wake-lock descriptor. When the runner dies it is signalled
    and kills the entire group before exiting, which releases the lock only
    once nothing from this turn is left.
    """
    command = list(codex_argv)
    if not command:
        print("supervisor was given no command to run", file=sys.stderr)
        return 2

    own_group = True
    try:
        os.setsid()
    except OSError as exc:
        # Already a group leader, or not permitted. Without our own group the
        # take-down below could reach processes that are not part of this turn,
        # so it is disabled rather than aimed at the wrong target. The death
        # signal below still covers the immediate Codex process.
        own_group = False
        print(
            f"supervisor could not create its own process group: {exc}. Only "
            "the immediate Codex process can be stopped, so this turn cannot "
            "report a contained result.",
            file=sys.stderr,
        )
    libc, _reason = parent_death_signal_support()
    if libc is not None:
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    if os.getppid() != expected_ppid:
        # Orphaned in the window before the signal was registered.
        os._exit(EXIT_PARENT_GONE)

    # Installed BEFORE anything is spawned. Between the spawn and the handler
    # there would otherwise be a window where the parent-death signal takes its
    # default action: it would kill only this supervisor, leaving Codex running
    # without the lock descriptor, so the next wake could overlap it. The
    # handler works whether or not a child exists yet, because everything this
    # supervisor spawns joins its process group at fork.
    spawned: list[subprocess.Popen[bytes]] = []

    def _take_down_the_group(_signum: int, _frame: Any) -> None:
        # SIGKILL because anything here may be mid-write; the point is that
        # nothing from this turn outlives the lock. stderr is the runner's
        # wake log, so a failure to tear down is still recorded.
        try:
            if own_group:
                os.killpg(0, signal.SIGKILL)
            else:
                for started in spawned:
                    started.kill()
        except OSError as exc:
            print(f"supervisor could not stop the turn: {exc}", file=sys.stderr)
        os._exit(EXIT_PARENT_GONE)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _take_down_the_group)

    try:
        child = subprocess.Popen(command)
    except OSError as exc:
        executable = next(iter(command), "<none>")
        print(f"supervisor could not start {executable}: {exc}", file=sys.stderr)
        return 2
    spawned.append(child)

    exit_code = child.wait()

    if own_group:
        # The turn is not over just because its own process exited.
        leaked = drain_process_group(
            deadline_seconds=GROUP_DRAIN_SECONDS,
            report=lambda message: print(message, file=sys.stderr),
        )
        if leaked and exit_code == 0:
            # Do not report a clean turn when part of it is still running, or
            # when we could not tell.
            return EXIT_TURN_NOT_CONTAINED
    elif exit_code == 0:
        # Without our own process group the drain cannot cover descendants, so
        # a clean result here would be an assumption rather than a check.
        return EXIT_TURN_NOT_CONTAINED
    return exit_code


def make_die_with_parent(expected_ppid: int) -> Any:
    """Build the pre-exec hook that binds this child to the runner's life.

    Without it the Codex child outlives a killed runner. The runner's death
    releases the wake lock, so a later wake can acquire it and start a second
    turn against the same thread and the same checkout while the orphan is
    still editing files. A signal handler cannot cover that, because the
    runner may be SIGKILLed; PR_SET_PDEATHSIG is enforced by the kernel.

    Covers the immediate child and, through the supervisor's process group,
    the ordinary descendants it starts. A descendant that calls `setsid` to
    leave that group is outside what this can reach; see `drain_process_group`.

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


def _write_if_possible(stream: Any, text: str) -> str | None:
    """Write `text`; return None on success, or why the stream refused it."""
    try:
        stream.write(text)
        stream.flush()
    except OSError as exc:
        return str(exc)
    return None


def _log(handle: Any, message: str) -> None:
    """Write one wake-log line, falling back to stderr if the log cannot take it.

    This is a diagnostic channel, and a diagnostic channel must not be able to
    fail the operation it is describing. A full disk after Codex has already
    edited files, pushed, or commented would otherwise raise out of a finished
    turn and make the caller retry work that has already happened. If stderr is
    gone too there is nothing left to tell, and still nothing worth failing a
    finished turn over.
    """
    line = f"[{_now()}] {message}"
    problem = _write_if_possible(handle, line + "\n")
    if problem is None:
        return
    _write_if_possible(sys.stderr, f"{line}  (wake log unavailable: {problem})\n")


def valid_watcher_id(watcher_id: str) -> bool:
    """Reject ids that could escape the state directory."""
    if not SAFE_WATCHER_ID_RE.fullmatch(watcher_id):
        return False
    return ".." not in watcher_id and not watcher_id.startswith(".")


def canonical_thread_id(value: str) -> str:
    """The one spelling this runner compares and stores.

    Codex session ids are UUIDs, and a UUID's identity is not its casing. The
    shape check accepts uppercase hex, so without normalising here a stored
    `01A0...` and a reported `01a0...` are the same conversation spelled two
    ways, and a case-sensitive comparison would kill the correct turn.
    """
    return value.strip().lower()


def valid_thread_id(value: str) -> bool:
    """Accept only the canonical Codex session-id shape.

    `fullmatch` on the already-stripped value, so a trailing newline or a
    leading dash cannot smuggle argv through.
    """
    return bool(THREAD_ID_RE.fullmatch(value))


def _read_bounded(fd: int, limit: int) -> bytes:
    """Read at most `limit` bytes from an already-opened descriptor."""
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_thread_id(path: Path) -> tuple[str | None, str | None]:
    """Return (thread_id, reason_ignored).

    Any unusable stored value degrades to a fresh thread rather than failing the
    wake: a wake that runs on a new thread still does the operator's work, but a
    wake that refuses to run loses the review event entirely.
    """
    # One path lookup, and everything after it is judged on the descriptor.
    # Validating the name and then opening the name leaves a window in which
    # the checked regular file is replaced by a FIFO, and that open blocks
    # forever while this wake holds the lock. O_NONBLOCK means even a FIFO
    # that wins the race opens immediately instead of waiting for a writer,
    # O_NOFOLLOW refuses a symlink at the final component, and the fstat below
    # judges what was actually opened rather than what the name once pointed
    # at. The read is bounded because st_size cannot be trusted for anything
    # that is not a regular file.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None, f"stored thread id at {path} is not a regular file"
        return None, f"could not open the stored thread id: {exc}"
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None, f"stored thread id at {path} is not a regular file"
        if info.st_size > MAX_THREAD_FILE_BYTES:
            return None, "stored thread id file is too large"
        raw = _read_bounded(fd, MAX_THREAD_FILE_BYTES).decode("utf-8")
    except OSError as exc:
        return None, f"could not read stored thread id: {exc}"
    except UnicodeDecodeError:
        return None, "stored thread id is not valid UTF-8"
    finally:
        os.close(fd)
    candidate = canonical_thread_id(raw)
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
        # Same rule as reading the id: never write through a name something
        # else may have prepared. O_EXCL refuses to reuse anything already at
        # the staging name, which also makes a stale file from a recycled pid
        # an error rather than a silent overwrite, and O_NOFOLLOW refuses a
        # symlink planted there to redirect the write.
        fd = os.open(
            staged,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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


PROFILE_UNSET = "<unset>"


class WakeProfile(NamedTuple):
    """Which Codex profile this wake runs under, and where that came from.

    Codex reads its injected context from two roots resolved from two separate
    environment variables: `CODEX_HOME` supplies config, memories and built-in
    skills, and `$HOME/.agents/skills` supplies the shared skills catalogue.
    Neither is a per-invocation flag, so the only way an unattended wake can
    avoid inheriting the operator's interactive surface is to run under a
    different profile.
    """

    codex_home: str | None
    agent_home: str | None
    effective_codex_home: str
    effective_home: str
    codex_home_origin: str
    home_origin: str

    @property
    def isolates_anything(self) -> bool:
        return self.codex_home is not None or self.agent_home is not None

    @property
    def partial_reason(self) -> str | None:
        """Name the root this wake did NOT isolate, or None when moot.

        Partial isolation measurably under-delivers -- `CODEX_HOME` alone still
        carried all 26 shared skills -- so it is allowed but must not read like
        full isolation in the log.
        """
        if not self.isolates_anything:
            return None
        if self.codex_home is None:
            return (
                "CODEX_HOME is not isolated, so this wake still uses that "
                "profile's config, memories and built-in skills"
            )
        if self.agent_home is None:
            return (
                "HOME is not isolated, so this wake still loads the shared "
                "skills catalogue from that home's .agents/skills"
            )
        return None


def resolve_profile(
    *,
    codex_home: str | None,
    agent_home: str | None,
    environ: Any = None,
) -> WakeProfile:
    """Decide the profile and record what the child will actually see.

    The effective values matter more than the arguments. An operator can set
    either variable in a systemd unit, in which case the child uses it and no
    argument names it; a receipt built from arguments alone would be blind in
    exactly that case.
    """
    source = os.environ if environ is None else environ

    def effective(argument: str | None, key: str) -> tuple[str, str]:
        if argument is not None:
            return argument, "argument"
        inherited = source.get(key)
        if inherited:
            return inherited, "inherited"
        return PROFILE_UNSET, "unset"

    codex_value, codex_origin = effective(codex_home, "CODEX_HOME")
    home_value, home_origin = effective(agent_home, "HOME")
    return WakeProfile(
        codex_home=codex_home,
        agent_home=agent_home,
        effective_codex_home=codex_value,
        effective_home=home_value,
        codex_home_origin=codex_origin,
        home_origin=home_origin,
    )


def child_environment(profile: WakeProfile) -> dict[str, str] | None:
    """Return the child environment, or None to inherit unchanged.

    None rather than a copy of `os.environ` on purpose: with no profile
    configured the child must be launched exactly as it is today, and not
    passing `env=` at all is the only way to guarantee that.
    """
    if not profile.isolates_anything:
        return None
    env = dict(os.environ)
    if profile.codex_home is not None:
        env["CODEX_HOME"] = profile.codex_home
    if profile.agent_home is not None:
        env["HOME"] = profile.agent_home
    return env


def profile_marker_path(thread_path: Path) -> Path:
    """Where the profile that owns a stored thread id is recorded."""
    return thread_path.with_name(thread_path.name + ".profile")


def stored_thread_profile(thread_path: Path) -> tuple[str | None, str | None]:
    """Return (owner, problem) for the profile that owns the stored thread id.

    A missing marker and an unreadable one are different states and must not
    collapse into one. Missing is the expected state for every watcher that
    predates the marker, and resuming is right. Unreadable means the owner
    cannot be determined at all, which is an anomaly the caller has to hear
    about rather than a silent "no owner".
    """
    marker = profile_marker_path(thread_path)
    try:
        text = marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"could not read the thread profile marker: {exc}"
    except UnicodeDecodeError:
        return None, "the thread profile marker is not valid UTF-8"
    value = text.strip()
    if not value:
        return None, "the thread profile marker is empty"
    return value, None


def record_thread_profile(thread_path: Path, profile: WakeProfile) -> None:
    """Remember which profile owns the thread id now stored at `thread_path`."""
    _atomic_write(profile_marker_path(thread_path), profile.effective_codex_home + "\n")


def foreign_thread_reason(thread_path: Path, profile: WakeProfile) -> str | None:
    """Say why a stored thread id belongs to a different profile.

    A rollout lives only inside the `CODEX_HOME` that created it, so a stored
    id is meaningless under another one. Resuming it anyway costs a wake and
    ends in a quarantine that discards the arc, so the mismatch is detected
    here instead and the wake simply starts fresh. An unrecorded owner is
    treated as a match, which keeps every watcher that predates this marker
    resuming exactly as before.
    """
    owner, problem = stored_thread_profile(thread_path)
    if problem:
        # The owner cannot be determined. Starting fresh loses the arc; resuming
        # anyway risks a cross-profile resume, which loses the arc AND spends a
        # wake discovering it. The cheaper failure wins, and it is logged.
        return (
            f"{problem}; this wake cannot tell which profile owns the stored "
            "thread id, so it starts a fresh thread rather than risk a resume "
            "into a profile that does not hold the rollout"
        )
    if owner is None or owner == profile.effective_codex_home:
        return None
    return (
        f"the stored thread id was created under CODEX_HOME {owner} and this "
        f"wake runs under {profile.effective_codex_home}; a rollout does not "
        "exist outside the profile that created it, so this wake starts a "
        "fresh thread rather than failing a resume"
    )


def write_thread_id(path: Path, thread_id: str) -> None:
    """Persist atomically so a killed wake cannot leave a partial id."""
    _atomic_write(path, thread_id + "\n")


def thread_path_problem(path: Path) -> str | None:
    """Say why the thread id could not be stored at `path`, before launching.

    Checked ahead of the turn on purpose. Discovering it afterwards means
    Codex has already edited files, pushed, or commented, and there is no
    resumable id to show for it, so every retry repeats that work.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        info = None
    except OSError as exc:
        return f"cannot inspect {path}: {exc}"
    if info is not None and not stat.S_ISREG(info.st_mode):
        return f"{path} exists but is not a regular file"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"cannot create the state directory {path.parent}: {exc}"
    probe = path.with_name(f".{path.name}.probe.{os.getpid()}")
    try:
        probe.write_text("", encoding="utf-8")
    except OSError as exc:
        return f"cannot write beside {path}: {exc}"
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            return f"cannot clean up a probe file beside {path}"
    return None


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


def reports_missing_session(stderr_text: str, thread_id: str) -> bool:
    """True only when Codex says THIS thread does not exist.

    The id is captured out of the diagnostic itself rather than searched for
    separately. Searching independently lets unrelated output satisfy both
    halves: stderr mentioning the resumed thread on one line while reporting a
    missing rollout for a different thread on another would quarantine valid
    state on evidence about some other conversation.
    """
    wanted = canonical_thread_id(thread_id)
    for match in MISSING_SESSION_RE.finditer(stderr_text):
        if canonical_thread_id(match.group("thread")) == wanted:
            return True
    return False


def _usage_receipt(usage: Any) -> bool:
    """Say whether a reported usage object actually prices the turn.

    An empty object, or one carrying no token count, is the same as no usage
    at all for the operator who has to answer what this wake cost. A count of
    zero is a real answer and is accepted; the receipt is the number being
    present, not the number being large. `bool` is excluded because it is an
    `int` subclass and `True` is not a token count.
    """
    if not isinstance(usage, dict):
        return False
    return any(
        key.endswith("_tokens") and isinstance(value, int) and not isinstance(value, bool)
        for key, value in usage.items()
    )


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
            if isinstance(candidate, str) and valid_thread_id(
                canonical_thread_id(candidate)
            ):
                thread_id = canonical_thread_id(candidate)
                # Persist immediately, not at end of turn. Codex has already
                # created the session by this point; if the runner is killed,
                # the unit stopped, or the host restarts before the turn ends,
                # an unrecorded id means the next wake starts a second thread
                # and the arc is lost.
                if on_thread_started is not None:
                    on_thread_started(thread_id)
        elif kind == "turn.completed":
            reported = event.get("usage")
            if _usage_receipt(reported):
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
    profile: WakeProfile | None = None,
    lock_fd: int | None = None,
) -> TurnResult:
    """Spend exactly one Codex turn. The caller must already hold the wake lock."""
    if profile is None:
        profile = resolve_profile(codex_home=None, agent_home=None)
    # Read the thread id under the lock. Reading it before acquiring the lock
    # leaves a stale-read interleaving: this process can read "no thread" while
    # another holds the lock, that one records a new id and releases, and this
    # one then acquires the lock still holding None, starts a second thread,
    # and overwrites the id. One thread per watcher is the contract, so the
    # read has to happen inside the critical section.
    thread_id, ignored_reason = read_thread_id(thread_path)
    if ignored_reason:
        _log(log_handle, f"starting fresh thread: {ignored_reason}")
    if thread_id is not None:
        foreign = foreign_thread_reason(thread_path, profile)
        if foreign:
            # Deliberately not a quarantine. The session is not dead, it simply
            # lives in another profile, and quarantining it would discard an
            # arc that is still resumable by switching back.
            _log(log_handle, f"starting fresh thread: {foreign}")
            thread_id = None

    if shutil.which(codex_bin) is None and not os.access(codex_bin, os.X_OK):
        # Resolved here rather than at the spawn, because the supervisor now
        # sits in between and a failure inside it would surface as its exit
        # code instead of a clear message about the binary.
        _log(log_handle, f"codex binary not found or not executable: {codex_bin}")
        print(f"codex binary not found: {codex_bin}", file=sys.stderr)
        return TurnResult(2, binary_missing=True)

    state_problem = thread_path_problem(thread_path)
    if state_problem:
        # Refuse before Codex can do anything. A turn whose id cannot be kept
        # is worse than no turn: the work happens and no one can resume it.
        _log(
            log_handle,
            f"refusing to start a turn: {state_problem}. Codex was not "
            "launched, so nothing was changed and this is safe to retry.",
        )
        print(f"cannot store the thread id: {state_problem}", file=sys.stderr)
        return TurnResult(EXIT_STATE_UNUSABLE)

    argv = build_argv(codex_bin=codex_bin, thread_id=thread_id, sandbox=sandbox)
    mode = "resume" if thread_id else "fresh"
    _log(log_handle, f"wake {mode} watcher={watcher_id} repo={repo_dir}")
    _log(log_handle, "argv=" + " ".join(argv))
    # The receipt for which profile this turn ran under. Logged on every
    # launching turn, including when nothing is configured, because "no line"
    # and "ran on the inherited profile" have to read differently to whoever
    # comes back to the log.
    _log(
        log_handle,
        "profile "
        f"codex_home={profile.effective_codex_home} ({profile.codex_home_origin}) "
        f"home={profile.effective_home} ({profile.home_origin})",
    )
    partial = profile.partial_reason
    if partial:
        _log(log_handle, f"partial profile isolation: {partial}")

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

    persist_failure: list[str] = []

    # The wake record is what the operator reads when they come back, so it
    # must describe THIS wake or nothing. Every post-launch exit goes through
    # `finish`, which is the only place the record is written and the only
    # place a turn result is returned. Leaving the write to each exit path is
    # what let the protocol-failure returns keep presenting the previous
    # wake's message as the current result.
    last_path = state_dir / f"{watcher_id}.codex-wake.last.md"

    def finish(code: int, record: str) -> TurnResult:
        try:
            last_path.write_text(record, encoding="utf-8")
        except OSError as exc:
            # The turn already ran. Codex may have edited files, pushed, or
            # commented, so a failure to write the record must not turn a
            # completed turn into a reported failure the caller would retry.
            _log(
                log_handle,
                f"could not refresh the wake record at {last_path}: {exc}; "
                "the turn itself completed and is not being failed for this",
            )
        _log(log_handle, f"turn complete exit={code}")
        return TurnResult(code)

    # Holds the running turn so the event callback can stop it the moment it
    # identifies itself as the wrong conversation.
    live_process: list[Any] = []
    wrong_thread: list[str] = []

    # The turn's identity is whatever was decided first: the stored id when
    # resuming, or the first thread.started on a fresh turn. Both paths are
    # pinned by the same rule, because a later event naming a different
    # conversation is the same defect either way.
    pinned: list[str] = [thread_id] if thread_id is not None else []

    def _persist(new_id: str) -> None:
        expected = next(iter(pinned), None)
        if expected is None:
            # Fresh turn: the first valid id this stream names IS the arc.
            pinned.append(new_id)
        elif new_id != expected:
            # A turn reporting a conversation other than the one it is pinned
            # to is the wrong conversation. Stop it at the event that names it
            # rather than after the turn, because by then it has had the whole
            # turn to edit the checkout, push, or comment in that context.
            wrong_thread.append(new_id)
            _log(
                log_handle,
                f"turn pinned to thread {expected} reported thread {new_id}; "
                "stopping it now rather than letting the wrong conversation "
                "continue",
            )
            running = next(iter(live_process), None)
            if running is not None:
                try:
                    running.terminate()
                except OSError as exc:
                    _log(log_handle, f"could not stop the wrong-thread turn: {exc}")
            return
        else:
            return
        if new_id == thread_id:
            return
        try:
            write_thread_id(thread_path, new_id)
            record_thread_profile(thread_path, profile)
        except OSError as exc:
            # Raising here would abandon a turn that is already running and
            # already having effects. Record it and report a controlled
            # outcome once the turn is over.
            persist_failure.append(f"could not record thread id {new_id}: {exc}")
        else:
            _log(log_handle, f"recorded thread id {new_id}")

    prompt_fd, prompt_name = tempfile.mkstemp(prefix="codex-wake-", suffix=".txt")
    prompt_file = Path(prompt_name)
    # Codex reports a missing session on stderr, so it is captured to a file
    # rather than streamed straight to the log: the text has to be inspected
    # before the stored id can be judged dead. It reaches the log either way.
    stderr_fd, stderr_name = tempfile.mkstemp(prefix="codex-wake-err-", suffix=".txt")
    stderr_file = Path(stderr_name)
    stderr_text = ""
    try:
        with os.fdopen(prompt_fd, "w", encoding="utf-8") as prompt_handle:
            prompt_handle.write(prompt)
        with prompt_file.open("r", encoding="utf-8") as stdin_handle:
            with os.fdopen(stderr_fd, "w", encoding="utf-8") as stderr_handle:
                try:
                    supervisor_argv = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--supervise",
                        str(os.getpid()),
                        "--",
                        *argv,
                    ]
                    process = subprocess.Popen(
                        supervisor_argv,
                        cwd=str(repo_dir),
                        stdin=stdin_handle,
                        stdout=subprocess.PIPE,
                        stderr=stderr_handle,
                        text=True,
                        # Replace undecodable bytes instead of raising. Strict
                        # decoding lets one bad byte abort the whole stream,
                        # and the decoder buffers, so even a valid
                        # thread.started already emitted is lost with it. The
                        # turn has run by then, so that would discard the only
                        # way to resume work that already happened.
                        encoding="utf-8",
                        errors="replace",
                        env=child_environment(profile),
                        preexec_fn=die_with_parent,
                        # The wake lock is an open file description, which fork
                        # and exec preserve. The supervisor holds it for the
                        # whole group, so the lock is released only once every
                        # descendant of this turn is gone.
                        pass_fds=() if lock_fd is None else (lock_fd,),
                    )
                except FileNotFoundError:
                    # Not a post-launch exit: exec never happened, so no turn
                    # ran and the record still correctly describes the last
                    # wake that did run. The two returns above are the same
                    # case. Every exit from here on is a turn that happened,
                    # and those all go through `finish`.
                    _log(log_handle, f"codex binary not found: {codex_bin}")
                    print(f"codex binary not found: {codex_bin}", file=sys.stderr)
                    return TurnResult(2, binary_missing=True)
                live_process.append(process)
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
        for scratch_file in (prompt_file, stderr_file):
            try:
                scratch_file.unlink()
            except OSError as exc:
                _log(
                    log_handle,
                    f"could not remove the scratch file {scratch_file}: {exc}",
                )

    if thread_id is not None and observed_thread is None and exit_code != 0:
        if reports_missing_session(stderr_text, thread_id):
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
    else:
        # Logged even on a failed turn. "No usage line" and "this wake was
        # never priced" have to read differently to someone scanning the log
        # for what the night cost.
        _log(log_handle, "usage=unavailable; codex reported no token counts")

    for failure in persist_failure:
        _log(
            log_handle,
            f"{failure}. The turn ran, so its work is done, but nothing can "
            "resume it; do not retry blindly.",
        )
    if persist_failure:
        return finish(
            EXIT_STATE_UNUSABLE,
            f"[{_now()}] this wake ran a turn but could not record its thread "
            f"id: {persist_failure[0]}. The work is done and cannot be "
            "resumed; do not retry blindly.\n",
        )

    if wrong_thread or (
        thread_id is not None
        and observed_thread is not None
        and observed_thread != thread_id
    ):
        reported = next(iter(wrong_thread), observed_thread)
        expected = next(iter(pinned), thread_id)
        _log(
            log_handle,
            f"turn pinned to thread {expected} reported thread {reported}; kept "
            f"{expected} and failed this turn rather than switching arcs",
        )
        return finish(
            EXIT_NO_THREAD_EVENT,
            f"[{_now()}] this wake was stopped: the turn was pinned to thread "
            f"{expected} and reported thread {reported}, so it was killed "
            "rather than allowed to continue the wrong conversation\n",
        )

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
        return finish(
            EXIT_NO_THREAD_EVENT,
            f"[{_now()}] this wake exited 0 but never named a thread, so "
            "nothing was recorded to resume\n",
        )

    if last_message is None:
        # A turn that reports success while saying nothing about what it did
        # is a protocol failure. A turn that already failed keeps its own exit
        # code, which is the more useful diagnosis.
        return finish(
            EXIT_NO_AGENT_MESSAGE if exit_code == 0 else exit_code,
            f"[{_now()}] this wake produced no agent message; "
            f"the turn exited {exit_code} and left no account of what it did\n",
        )

    summary = " ".join(last_message.split())
    if len(summary) > 500:
        summary = summary[:500] + " [truncated]"
    _log(log_handle, f"agent message: {summary}")

    if exit_code == 0 and usage is None:
        # The turn worked and said what it did, but nothing priced it. This
        # runner was built because unattended wakes burned a weekly quota, so
        # an unpriced wake is a failed wake: the agent message is kept, and
        # the distinct exit code is what makes the missing receipt visible to
        # an operator who was asleep for it.
        _log(
            log_handle,
            "codex exited 0 and reported no token usage for this turn; the "
            "work is done but its cost is unknown",
        )
        return finish(
            EXIT_NO_USAGE,
            last_message
            + f"\n\n[{_now()}] codex reported no token usage for this turn, "
            "so this wake is unpriced\n",
        )

    return finish(exit_code, last_message + "\n")


def run_wake(
    *,
    watcher_id: str,
    repo_dir: Path,
    state_dir: Path,
    prompt: str,
    sandbox: str,
    codex_bin: str,
    dry_run: bool,
    profile: WakeProfile | None = None,
) -> int:
    if profile is None:
        profile = resolve_profile(codex_home=None, agent_home=None)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Reached before the log exists, so stderr is the only channel. The
        # runner promises a controlled diagnostic exit, and a traceback from
        # the very first filesystem touch is not one.
        print(f"cannot use the state directory {state_dir}: {exc}", file=sys.stderr)
        return EXIT_STATE_UNUSABLE
    thread_path = state_dir / f"{watcher_id}.codex-thread"
    lock_path = state_dir / f"{watcher_id}.codex-wake.lock"
    log_path = state_dir / f"{watcher_id}.codex-wake.log"

    if dry_run:
        thread_id, ignored_reason = read_thread_id(thread_path)
        foreign = (
            foreign_thread_reason(thread_path, profile)
            if thread_id is not None
            else None
        )
        if foreign:
            thread_id = None
        argv = build_argv(codex_bin=codex_bin, thread_id=thread_id, sandbox=sandbox)
        print(f"mode={'resume' if thread_id else 'fresh'}")
        print(f"cwd={repo_dir}")
        if ignored_reason:
            print(f"ignored_stored_thread_id={ignored_reason}")
        if foreign:
            print(f"ignored_stored_thread_id={foreign}")
        print(
            "profile "
            f"codex_home={profile.effective_codex_home} ({profile.codex_home_origin}) "
            f"home={profile.effective_home} ({profile.home_origin})"
        )
        partial = profile.partial_reason
        if partial:
            print(f"partial_profile_isolation={partial}")
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
                except OSError as exc:
                    if exc.errno == errno.EINTR:
                        # A signal arrived mid-call; that is not contention.
                        continue
                    if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                        # EIO, EBADF, ENOLCK and friends are not another
                        # holder. Retrying them for the full wait window would
                        # stall every wake for ten minutes and then report a
                        # lock timeout that hides the real fault.
                        _log(
                            log_handle,
                            f"cannot lock {lock_path}: {exc}. This is not "
                            "contention, so the wake is failing immediately "
                            "rather than waiting out the timeout.",
                        )
                        print(f"cannot lock {lock_path}: {exc}", file=sys.stderr)
                        return EXIT_STATE_UNUSABLE
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
                profile=profile,
                lock_fd=lock_handle.fileno(),
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
    parser.add_argument(
        "--codex-home",
        default=None,
        help=(
            "Run Codex under this CODEX_HOME instead of the one this process "
            "inherited. Isolates config, memories and built-in skills."
        ),
    )
    parser.add_argument(
        "--agent-home",
        default=None,
        help=(
            "Run Codex under this HOME instead of the inherited one. The "
            "shared skills catalogue lives at $HOME/.agents/skills, so this is "
            "a separate root from --codex-home and each takes effect alone."
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
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["--supervise"]:
        # Internal entrypoint: `--supervise <runner pid> -- <codex argv...>`.
        # Not part of the operator surface, so it is handled before argparse.
        head, separator, command = raw[1:2], raw[2:3], raw[3:]
        if separator != ["--"] or not command:
            print("usage: --supervise <ppid> -- <command...>", file=sys.stderr)
            return 2
        parent_pid = next(iter(head), "")
        try:
            expected = int(parent_pid)
        except ValueError:
            print(f"invalid supervisor parent pid: {parent_pid!r}", file=sys.stderr)
            return 2
        return supervise(command, expected_ppid=expected)

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

    profile = resolve_profile(
        codex_home=args.codex_home,
        agent_home=args.agent_home,
    )

    return run_wake(
        watcher_id=args.watcher_id,
        repo_dir=repo_dir,
        state_dir=args.state_dir.expanduser(),
        prompt=prompt,
        sandbox=args.sandbox,
        codex_bin=args.codex_bin,
        dry_run=args.dry_run,
        profile=profile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
