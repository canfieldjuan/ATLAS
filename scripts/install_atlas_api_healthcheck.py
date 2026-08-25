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
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import atlas_api_healthcheck as healthcheck_runtime


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


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    payload: bytes | None
    mode: int | None
    symlink_target: str | None = None


@dataclass(frozen=True)
class TimerState:
    enablement: str
    active: bool

    @property
    def enabled(self) -> bool:
        return self.enablement in {"enabled", "enabled-runtime"}


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
    payload = f"{existing}{separator}{TOPIC_ENV}={topic}\n".encode("utf-8")
    _atomic_write(path, payload, mode=0o600)


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


def _atomic_write(destination: Path, payload: bytes, *, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.fchmod(handle.fileno(), mode)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as exc:
        raise RuntimeError(f"cannot atomically write install destination {destination}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError as exc:
                print(
                    f"WARNING cannot remove installer temporary file: {type(exc).__name__}",
                    file=sys.stderr,
                )


def _write_copy(source: Path, destination: Path, *, executable: bool) -> None:
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read install source {source}: {exc}") from exc
    _atomic_write(destination, payload, mode=0o755 if executable else 0o644)


def _snapshot_file(path: Path) -> FileSnapshot:
    if path.is_symlink():
        try:
            return FileSnapshot(
                path=path,
                payload=None,
                mode=None,
                symlink_target=os.readlink(path),
            )
        except OSError as exc:
            raise RuntimeError(f"cannot snapshot install destination {path}") from exc
    if not path.exists():
        return FileSnapshot(path=path, payload=None, mode=None)
    if not path.is_file():
        raise RuntimeError(f"install destination is not a regular file: {path}")
    try:
        return FileSnapshot(
            path=path,
            payload=path.read_bytes(),
            mode=stat.S_IMODE(path.stat().st_mode),
        )
    except OSError as exc:
        raise RuntimeError(f"cannot snapshot install destination {path}") from exc


def _restore_file(snapshot: FileSnapshot) -> None:
    if snapshot.symlink_target is not None:
        temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            snapshot.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_dir = tempfile.TemporaryDirectory(
                dir=snapshot.path.parent,
                prefix=f".{snapshot.path.name}.link.",
            )
            temporary_link = Path(temporary_dir.name) / snapshot.path.name
            os.symlink(snapshot.symlink_target, temporary_link)
            os.replace(temporary_link, snapshot.path)
        except OSError as exc:
            raise RuntimeError(f"cannot restore install symlink {snapshot.path}") from exc
        finally:
            if temporary_dir is not None:
                try:
                    temporary_dir.cleanup()
                except OSError as exc:
                    print(
                        f"WARNING cannot remove installer symlink staging directory: {type(exc).__name__}",
                        file=sys.stderr,
                    )
        return
    if snapshot.payload is None:
        try:
            snapshot.path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"cannot remove new install destination {snapshot.path}") from exc
        return
    mode = snapshot.mode if snapshot.mode is not None else 0o600
    _atomic_write(snapshot.path, snapshot.payload, mode=mode)


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


def _timer_state(runner: Runner) -> TimerState:
    unit_state, query_error = healthcheck_runtime.query_unit_state(TIMER_NAME, runner)
    if unit_state is None:
        raise RuntimeError(query_error or "cannot query existing health timer")
    if unit_state.load_state == "not-found":
        return TimerState(enablement="disabled", active=False)
    if unit_state.load_state != "loaded":
        raise RuntimeError(f"health timer has unsupported LoadState={unit_state.load_state}")
    if unit_state.active_state not in {"active", "inactive", "failed"}:
        raise RuntimeError(f"health timer has unsupported ActiveState={unit_state.active_state}")

    command = (
        "systemctl",
        "--user",
        "show",
        "--property=UnitFileState",
        "--value",
        TIMER_NAME,
    )
    result = runner(command)
    if result.returncode != 0:
        raise RuntimeError(f"cannot query health timer enablement ({_command_detail(result)})")
    unit_file_state = result.stdout.strip().lower()
    if unit_file_state not in {"disabled", "enabled", "enabled-runtime", "static"}:
        raise RuntimeError(
            f"health timer has unsupported UnitFileState={unit_file_state or 'empty'}"
        )
    return TimerState(
        enablement=unit_file_state,
        active=unit_state.active_state == "active",
    )


def _existing_health_service_is_loaded(runner: Runner) -> bool:
    unit_state, query_error = healthcheck_runtime.query_unit_state(SERVICE_NAME, runner)
    if unit_state is None:
        raise RuntimeError(query_error or "cannot query existing health service")
    if unit_state.load_state == "not-found":
        return False
    if unit_state.load_state != "loaded":
        raise RuntimeError(
            f"existing health service has unsupported LoadState={unit_state.load_state}"
        )
    return True


def _loaded_unit_is_current(unit: str, runner: Runner) -> tuple[bool, str]:
    command = (
        "systemctl",
        "--user",
        "show",
        "--property=NeedDaemonReload",
        "--value",
        unit,
    )
    result = runner(command)
    if result.returncode != 0:
        return False, f"cannot query loaded definition for {unit} ({_command_detail(result)})"
    needs_reload = result.stdout.strip().lower()
    if needs_reload == "no":
        return True, f"ok: loaded definition is current for {unit}"
    if needs_reload == "yes":
        return False, f"systemd requires daemon-reload for {unit}"
    return False, f"invalid NeedDaemonReload state for {unit}: {needs_reload or 'empty'}"


def _rollback_install(
    snapshots: Sequence[FileSnapshot],
    timer_state: TimerState,
    runner: Runner,
    *,
    enrollment_attempted: bool,
) -> list[str]:
    errors: list[str] = []

    def run_rollback(command: Sequence[str], action: str) -> None:
        try:
            result = runner(command)
        except OSError as exc:
            errors.append(f"{action} ({type(exc).__name__})")
            return
        if result.returncode != 0:
            errors.append(f"{action} ({_command_detail(result)})")

    if enrollment_attempted:
        run_rollback(
            ("systemctl", "--user", "disable", "--now", TIMER_NAME),
            "cannot remove failed timer enrollment",
        )
    for snapshot in snapshots:
        try:
            _restore_file(snapshot)
        except RuntimeError as exc:
            errors.append(str(exc))
    run_rollback(("systemctl", "--user", "daemon-reload"), "cannot reload restored systemd files")
    if timer_state.enablement == "enabled-runtime":
        run_rollback(
            ("systemctl", "--user", "enable", "--runtime", TIMER_NAME),
            "cannot restore runtime-enabled timer",
        )
    elif timer_state.enablement == "enabled":
        run_rollback(("systemctl", "--user", "enable", TIMER_NAME), "cannot restore enabled timer")
    if timer_state.active:
        run_rollback(("systemctl", "--user", "start", TIMER_NAME), "cannot restore active timer")
    return errors


def install(paths: InstallPaths, *, runner: Runner = _run, environment: Mapping[str, str] | None = None) -> list[str]:
    """Install and prove the monitor before enrolling its timer."""
    environment = os.environ if environment is None else environment
    sources = _source_files(paths)
    destinations = [destination for _source, destination, _executable in sources]
    snapshots = [_snapshot_file(path) for path in (*destinations, paths.notification_env)]
    previous_timer = _timer_state(runner)
    enrollment_attempted = False
    try:
        if previous_timer.active:
            _run_required(
                runner,
                ("systemctl", "--user", "stop", TIMER_NAME),
                action="existing timer stop",
            )
        topic_message = ensure_notification_topic(paths, environment)
        messages = [topic_message]
        if _existing_health_service_is_loaded(runner):
            _run_required(
                runner,
                ("systemctl", "--user", "stop", SERVICE_NAME),
                action="existing health service stop",
            )
        for source, destination, executable in sources:
            _write_copy(source, destination, executable=executable)
            messages.append(f"wrote: {destination}")
        _run_required(runner, ("systemctl", "--user", "daemon-reload"), action="systemd reload")
        _run_required(
            runner,
            ("systemctl", "--user", "start", "--wait", SERVICE_NAME),
            action="initial installed-monitor invocation",
        )
        enrollment_attempted = True
        _run_required(
            runner,
            ("systemctl", "--user", "enable", "--now", TIMER_NAME),
            action="timer enable",
        )
        messages.append("proved installed health service and enabled timer")
        return messages
    except KeyboardInterrupt as exc:
        rollback_errors = _rollback_install(
            snapshots,
            previous_timer,
            runner,
            enrollment_attempted=enrollment_attempted,
        )
        if rollback_errors:
            exc.add_note(f"rollback failed: {'; '.join(rollback_errors)}")
        raise
    except (OSError, RuntimeError) as exc:
        rollback_errors = _rollback_install(
            snapshots,
            previous_timer,
            runner,
            enrollment_attempted=enrollment_attempted,
        )
        if rollback_errors:
            raise RuntimeError(
                f"{exc}; rollback failed: {'; '.join(rollback_errors)}"
            ) from exc
        raise


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
    try:
        timer_state = _timer_state(runner)
        checks.append(
            (timer_state.enabled, f"ok: {TIMER_NAME} is enabled" if timer_state.enabled else "timer is not enabled")
        )
        checks.append(
            (timer_state.active, f"ok: {TIMER_NAME} is active" if timer_state.active else "timer is not active")
        )
    except RuntimeError as exc:
        checks.append((False, str(exc)))
    for unit in (SERVICE_NAME, TIMER_NAME):
        checks.append(_loaded_unit_is_current(unit, runner))
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
