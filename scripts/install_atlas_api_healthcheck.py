#!/usr/bin/env python3
"""Install or verify the standalone Atlas API liveness monitor.

Run this installer from a checked-out Atlas source tree after the liveness
change is merged.  It deliberately installs the monitor outside the Atlas
runtime worktree, reloads user systemd, enables its timer, and invokes the
installed service once so deployment proves the actual systemd path.
"""
from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence


SERVICE_NAME = "atlas-api-healthcheck.service"
TIMER_NAME = "atlas-api-healthcheck.timer"
INSTALLED_MONITOR_NAME = "atlas-api-healthcheck.py"
LEGACY_MONITOR_NAME = "atlas-api-healthcheck.sh"
TOPIC_ENV = "ATLAS_API_HEALTHCHECK_NTFY_TOPIC"
TOPIC_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
LEGACY_TOPIC_RE = re.compile(
    r'^NTFY_TOPIC="\$\{ATLAS_HC_NTFY_TOPIC:-(?P<topic>[A-Za-z0-9][A-Za-z0-9._-]{0,199})\}"$'
)


@dataclass(frozen=True)
class InstallPaths:
    source_root: Path
    bin_dir: Path
    systemd_dir: Path
    config_dir: Path
    legacy_monitor: Path

    @property
    def installed_monitor(self) -> Path:
        return self.bin_dir / INSTALLED_MONITOR_NAME

    @property
    def notification_env(self) -> Path:
        return self.config_dir / "atlas-api-healthcheck.env"


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def default_paths() -> InstallPaths:
    bin_dir = Path.home() / ".local" / "bin"
    return InstallPaths(
        source_root=Path(__file__).resolve().parents[1],
        bin_dir=bin_dir,
        systemd_dir=Path.home() / ".config" / "systemd" / "user",
        config_dir=Path.home() / ".config" / "atlas",
        legacy_monitor=bin_dir / LEGACY_MONITOR_NAME,
    )


def _source_files(paths: InstallPaths) -> tuple[tuple[Path, Path, bool], ...]:
    monitor = paths.source_root / "scripts" / "atlas_api_healthcheck.py"
    service = paths.source_root / "config" / SERVICE_NAME
    timer = paths.source_root / "config" / TIMER_NAME
    sources = (
        (monitor, paths.installed_monitor, True),
        (service, paths.systemd_dir / SERVICE_NAME, False),
        (timer, paths.systemd_dir / TIMER_NAME, False),
    )
    missing = [str(source) for source, _destination, _executable in sources if not source.is_file()]
    if missing:
        raise RuntimeError("installer must run from an Atlas source checkout; missing " + ", ".join(missing))
    return sources


def _validated_topic(value: str, *, source: str) -> str:
    topic = value.strip()
    if not TOPIC_RE.fullmatch(topic):
        raise RuntimeError(f"invalid notification topic from {source}")
    return topic


def _topic_from_env_file(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        values = [
            line.partition("=")[2]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.partition("=")[0] == TOPIC_ENV
        ]
    except OSError as exc:
        raise RuntimeError(f"cannot read notification environment file: {exc}") from exc
    if not values:
        return None
    if len(values) != 1:
        raise RuntimeError("notification environment file has multiple topic entries")
    return _validated_topic(values[0], source="notification environment file")


def _topic_from_legacy_monitor(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read legacy monitor for topic migration: {exc}") from exc
    for line in lines:
        match = LEGACY_TOPIC_RE.fullmatch(line.strip())
        if match:
            return match.group("topic")
    return None


def _append_topic(path: Path, topic: str) -> None:
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise RuntimeError(f"cannot read notification environment file: {exc}") from exc
    separator = "" if not existing or existing.endswith("\n") else "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{existing}{separator}{TOPIC_ENV}={topic}\n", encoding="utf-8")
    path.chmod(0o600)


def ensure_notification_topic(paths: InstallPaths, environment: Mapping[str, str]) -> str:
    existing = _topic_from_env_file(paths.notification_env)
    if existing is not None:
        paths.notification_env.chmod(0o600)
        return "preserved private notification topic"

    candidate = environment.get(TOPIC_ENV, "")
    source = "environment"
    if not candidate.strip():
        candidate = _topic_from_legacy_monitor(paths.legacy_monitor) or ""
        source = "legacy monitor"
    if not candidate.strip():
        raise RuntimeError(
            f"{TOPIC_ENV} is not configured; set it or retain a parseable legacy monitor before installation"
        )
    _append_topic(paths.notification_env, _validated_topic(candidate, source=source))
    return "migrated private notification topic" if source == "legacy monitor" else "wrote private notification topic"


def _write_copy(source: Path, destination: Path, *, executable: bool) -> None:
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read install source {source}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    destination.chmod(0o755 if executable else 0o644)


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    text = " ".join(part for part in (result.stderr, result.stdout) if part)
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if len(text) > 240:
        text = text[:237] + "..."
    return f"exit {result.returncode}: {text or 'no diagnostic output'}"


def _run_required(runner: Runner, command: Sequence[str], *, action: str) -> None:
    result = runner(command)
    if result.returncode != 0:
        raise RuntimeError(f"{action} failed ({_command_detail(result)})")


def install(paths: InstallPaths, *, runner: Runner = _run, environment: Mapping[str, str] | None = None) -> list[str]:
    """Install the monitor, enable the timer, then invoke the installed service once."""
    environment = os.environ if environment is None else environment
    sources = _source_files(paths)
    topic_message = ensure_notification_topic(paths, environment)
    messages = [topic_message]
    for source, destination, executable in sources:
        _write_copy(source, destination, executable=executable)
        messages.append(f"wrote: {destination}")
    _run_required(runner, ("systemctl", "--user", "daemon-reload"), action="systemd reload")
    _run_required(
        runner,
        ("systemctl", "--user", "enable", "--now", TIMER_NAME),
        action="timer enable",
    )
    _run_required(
        runner,
        ("systemctl", "--user", "start", "--wait", SERVICE_NAME),
        action="initial installed-monitor invocation",
    )
    messages.append("enabled timer and invoked installed health service")
    return messages


def _matches(source: Path, destination: Path, *, executable: bool) -> tuple[bool, str]:
    if not destination.is_file():
        return False, f"missing: {destination}"
    try:
        matches = source.read_bytes() == destination.read_bytes()
    except OSError as exc:
        return False, f"unreadable {destination}: {exc}"
    if not matches:
        return False, f"content drift: {destination}"
    if executable and not os.access(destination, os.X_OK):
        return False, f"not executable: {destination}"
    return True, f"ok: {destination}"


def check_install(paths: InstallPaths, *, runner: Runner = _run) -> tuple[bool, list[str]]:
    """Read-only verification of the installed copies and enabled timer."""
    checks = [_matches(source, destination, executable=executable) for source, destination, executable in _source_files(paths)]
    try:
        topic = _topic_from_env_file(paths.notification_env)
        mode = stat.S_IMODE(paths.notification_env.stat().st_mode) if paths.notification_env.exists() else 0
        if topic is None:
            checks.append((False, f"missing notification topic: {paths.notification_env}"))
        elif mode & 0o077:
            checks.append((False, f"notification environment file is not private: {paths.notification_env}"))
        else:
            checks.append((True, f"ok: private notification environment {paths.notification_env}"))
    except (OSError, RuntimeError) as exc:
        checks.append((False, str(exc)))
    for command, label in (
        (("systemctl", "--user", "is-enabled", "--quiet", TIMER_NAME), "timer is not enabled"),
        (("systemctl", "--user", "is-active", "--quiet", TIMER_NAME), "timer is not active"),
    ):
        result = runner(command)
        checks.append((result.returncode == 0, f"ok: {TIMER_NAME}" if result.returncode == 0 else label))
    return all(passed for passed, _message in checks), [message for _passed, message in checks]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--install", action="store_true", help="install, enable, and invoke the monitor")
    action.add_argument("--check", action="store_true", help="verify installed monitor/timer without writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.install:
            messages = install(default_paths())
            for message in messages:
                print(message)
            return 0
        ok, messages = check_install(default_paths())
        for message in messages:
            print(message)
        return 0 if ok else 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"atlas-api-healthcheck installer: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
