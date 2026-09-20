#!/usr/bin/env python3
"""Run one Codex wake turn against a persistent per-watcher thread.

The PR watcher and `codex_wake_bridge.py` decide whether a wake should happen
and what the prompt says. This runner is the last stage: it takes that prompt
on stdin and spends exactly one Codex turn on it, resuming the watcher's
existing Codex thread when there is one.

Resuming matters for cost and for continuity. A fresh `codex exec` rebuilds the
whole conversation from nothing, so the woken agent does not remember the arc it
is continuing and the operator pays full cold-start context on every wake.

This runner never merges, closes, or pushes anything. It starts a Codex turn and
records what that turn cost.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence


DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "atlas-pr-watchers"
DEFAULT_SANDBOX = "workspace-write"
DEFAULT_CODEX_BIN = "codex"

SAFE_WATCHER_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
# Codex session ids are UUID-shaped. Anything else must not reach argv.
THREAD_ID_RE = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
# A stored thread id is one short line; refuse to read more than that.
MAX_THREAD_FILE_BYTES = 256

SANDBOX_CHOICES = ("read-only", "workspace-write", "danger-full-access")


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


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


def write_thread_id(path: Path, thread_id: str) -> None:
    """Persist atomically so a killed wake cannot leave a partial id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_text(thread_id + "\n", encoding="utf-8")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def build_argv(
    *,
    codex_bin: str,
    thread_id: str | None,
    sandbox: str,
) -> list[str]:
    """Build the Codex argv for a fresh or resumed wake.

    `codex exec` and `codex exec resume` do not share a flag surface. Verified
    against codex-cli 0.155.1: `exec` accepts `-C/--cd` and `-s/--sandbox`;
    `exec resume` accepts neither. So this uses only what both accept -- the
    working directory is passed as the subprocess cwd, and the sandbox as a
    generic `-c` config override -- and the two paths stay one shape.
    """
    argv = [codex_bin, "exec"]
    if thread_id is not None:
        argv.append("resume")
        argv.append(thread_id)
    argv.extend(["--json", "-c", f'sandbox_mode="{sandbox}"'])
    # Trailing "-" makes Codex read the prompt from stdin.
    argv.append("-")
    return argv


def _log(handle: Any, message: str) -> None:
    handle.write(f"[{_now()}] {message}\n")
    handle.flush()


def _consume_events(
    stream: Any, log_handle: Any
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Scan the JSONL event stream for the thread id, usage, and final message.

    The final message is the only record of what the woken agent actually did.
    A wake happens while the operator is away, so dropping it would leave the
    wake unauditable.
    """
    thread_id: str | None = None
    usage: dict[str, Any] | None = None
    last_message: str | None = None
    for line in stream:
        text = line.strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "thread.started":
            candidate = event.get("thread_id")
            if isinstance(candidate, str) and valid_thread_id(candidate.strip()):
                thread_id = candidate.strip()
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
    return thread_id, usage, last_message


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

    thread_id, ignored_reason = read_thread_id(thread_path)

    if dry_run:
        argv = build_argv(codex_bin=codex_bin, thread_id=thread_id, sandbox=sandbox)
        mode = "resume" if thread_id else "fresh"
        print(f"mode={mode}")
        print(f"cwd={repo_dir}")
        if ignored_reason:
            print(f"ignored_stored_thread_id={ignored_reason}")
        print("argv=" + " ".join(argv))
        return 0

    with log_path.open("a", encoding="utf-8") as log_handle:
        with lock_path.open("w", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                # A wake is already in flight. It will observe the same PR state
                # this one would have, so dropping this wake loses nothing.
                _log(log_handle, f"skipped: a Codex wake is already running for {watcher_id}")
                return 0

            if ignored_reason:
                _log(log_handle, f"starting fresh thread: {ignored_reason}")

            argv = build_argv(codex_bin=codex_bin, thread_id=thread_id, sandbox=sandbox)
            mode = "resume" if thread_id else "fresh"
            _log(log_handle, f"wake {mode} watcher={watcher_id} repo={repo_dir}")
            _log(log_handle, "argv=" + " ".join(argv))

            prompt_file = Path(
                tempfile.mkstemp(prefix="codex-wake-", suffix=".txt")[1]
            )
            try:
                prompt_file.write_text(prompt, encoding="utf-8")
                with prompt_file.open("r", encoding="utf-8") as stdin_handle:
                    try:
                        process = subprocess.Popen(
                            argv,
                            cwd=str(repo_dir),
                            stdin=stdin_handle,
                            stdout=subprocess.PIPE,
                            stderr=log_handle,
                            text=True,
                        )
                    except FileNotFoundError:
                        _log(log_handle, f"codex binary not found: {codex_bin}")
                        print(f"codex binary not found: {codex_bin}", file=sys.stderr)
                        return 2
                    with process:
                        observed_thread, usage, last_message = _consume_events(
                            process.stdout, log_handle
                        )
                    exit_code = process.returncode
            finally:
                with contextlib.suppress(OSError):
                    prompt_file.unlink()

            if observed_thread and observed_thread != thread_id:
                write_thread_id(thread_path, observed_thread)
                _log(log_handle, f"recorded thread id {observed_thread}")

            if usage is not None:
                _log(log_handle, "usage=" + json.dumps(usage, sort_keys=True))

            if last_message is not None:
                # Full text goes to its own file so the log stays scannable.
                last_path = state_dir / f"{watcher_id}.codex-wake.last.md"
                last_path.write_text(last_message + "\n", encoding="utf-8")
                summary = " ".join(last_message.split())
                if len(summary) > 500:
                    summary = summary[:500] + " [truncated]"
                _log(log_handle, f"agent message: {summary}")
                _log(log_handle, f"full agent message written to {last_path}")
            else:
                _log(log_handle, "no agent message in this turn")

            _log(log_handle, f"wake complete exit={exit_code}")
            return exit_code


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
