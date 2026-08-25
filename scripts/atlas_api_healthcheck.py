#!/usr/bin/env python3
"""Monitor and recover an unexpectedly inactive local Atlas API service.

This program is intentionally stdlib-only and is installed outside the Atlas
runtime worktree. It can therefore observe the provider and attempt recovery
even when a runtime worktree is stale, missing, or unable to start.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Sequence, TextIO


EXIT_OK = 0
EXIT_RECOVERED = 2
EXIT_DOWN = 3
EXIT_ALERT_UNDELIVERED = 4

DEFAULT_SERVICE = "atlas-api.service"
DEFAULT_PROBE_URL = "http://127.0.0.1:8012/api/v1/leads/intake"
DEFAULT_NTFY_URL = "https://ntfy.sh"
DEFAULT_REALERT_EVERY = 6
DEFAULT_RECOVERY_ATTEMPTS = 8
DEFAULT_RECOVERY_INTERVAL_SECONDS = 1.0
DEFAULT_STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "atlas-api-health"
DEFAULT_MAINTENANCE_LOCK = Path.home() / ".config" / "atlas" / "atlas-api.maintenance"
PENDING_ALERTS = frozenset({"down", "recovered", "auto-recovered"})
STATE_STATUSES = frozenset({"healthy", "down", "maintenance"})
MAX_COMMAND_DETAIL = 240


@dataclass(frozen=True)
class Settings:
    service: str
    probe_url: str
    ntfy_url: str
    ntfy_topic: str
    state_dir: Path
    maintenance_lock: Path
    realert_every: int
    recovery_attempts: int
    recovery_interval_seconds: float
    no_alert: bool


@dataclass(frozen=True)
class Observation:
    status: str
    detail: str


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Opener = Callable[..., object]
Notifier = Callable[[str, str, str, str, str, str], bool]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def command_failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Return bounded diagnostic context without carrying terminal controls."""
    text = " ".join(part for part in (result.stderr, result.stdout) if part)
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    if len(text) > MAX_COMMAND_DETAIL:
        text = text[: MAX_COMMAND_DETAIL - 3] + "..."
    return f"exit {result.returncode}: {text or 'no diagnostic output'}"


@contextmanager
def serialized_monitor_state(settings: Settings) -> Iterator[TextIO]:
    """Serialize health cycles and supported maintenance transitions."""
    lock_path = settings.state_dir / "state.lock"
    try:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("w", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"unable to open Atlas API health lock at {lock_path}") from exc
    with lock_handle as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError as exc:
            raise RuntimeError(f"unable to acquire Atlas API health lock at {lock_path}") from exc
        yield lock


def service_is_active(service: str, runner: Runner) -> bool:
    return runner(("systemctl", "--user", "is-active", "--quiet", service)).returncode == 0


def enter_maintenance(settings: Settings, *, runner: Runner = _run) -> int:
    """Create the maintenance marker and stop the service under the monitor lock."""
    with serialized_monitor_state(settings):
        try:
            settings.maintenance_lock.parent.mkdir(parents=True, exist_ok=True)
            settings.maintenance_lock.touch(mode=0o600, exist_ok=True)
            settings.maintenance_lock.chmod(0o600)
        except OSError as exc:
            raise RuntimeError(
                f"unable to create Atlas API maintenance lock at {settings.maintenance_lock}"
            ) from exc
        stopped = runner(("systemctl", "--user", "stop", settings.service))
        if stopped.returncode != 0:
            raise RuntimeError(
                f"failed to stop {settings.service} for maintenance ({command_failure_detail(stopped)})"
            )
    print(f"MAINTENANCE: stopped {settings.service}; lock present at {settings.maintenance_lock}")
    return EXIT_OK


def exit_maintenance(settings: Settings) -> int:
    """Remove the maintenance marker under the same lock used by recovery."""
    with serialized_monitor_state(settings):
        try:
            settings.maintenance_lock.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"unable to remove Atlas API maintenance lock at {settings.maintenance_lock}"
            ) from exc
    print(f"MAINTENANCE CLEARED: {settings.maintenance_lock}")
    return EXIT_OK


def probe_lead_intake(probe_url: str, opener: Opener) -> tuple[bool, str]:
    request = urllib.request.Request(
        probe_url,
        headers={
            "Origin": "https://effinghamofficemaids.com",
            "Access-Control-Request-Method": "POST",
        },
        method="OPTIONS",
    )
    try:
        with opener(request, timeout=8) as response:
            status = int(response.status)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, f"lead-intake probe failed: {type(exc).__name__}"
    if status in (200, 204):
        return True, f"lead-intake probe returned HTTP {status}"
    return False, f"lead-intake probe returned HTTP {status}"


def observe(
    settings: Settings,
    runner: Runner,
    opener: Opener,
    sleeper: Callable[[float], None],
    *,
    before_start: Callable[[], None] | None = None,
) -> Observation:
    """Observe the provider and recover only an inactive, non-maintenance unit."""
    if settings.maintenance_lock.exists():
        return Observation("maintenance", f"maintenance lock present: {settings.maintenance_lock}")

    if service_is_active(settings.service, runner):
        healthy, detail = probe_lead_intake(settings.probe_url, opener)
        return Observation("healthy" if healthy else "down", detail)

    if before_start is not None:
        before_start()
    started = runner(("systemctl", "--user", "start", settings.service))
    if started.returncode != 0:
        return Observation(
            "down",
            f"failed to start inactive unit {settings.service} ({command_failure_detail(started)})",
        )

    last_detail = f"started inactive unit {settings.service}; waiting for lead-intake probe"
    for attempt in range(settings.recovery_attempts):
        if service_is_active(settings.service, runner):
            healthy, detail = probe_lead_intake(settings.probe_url, opener)
            if healthy:
                return Observation("recovered", f"auto-recovered {settings.service}: {detail}")
            last_detail = detail
        else:
            last_detail = f"unit {settings.service} is still inactive after start"
        if attempt + 1 < settings.recovery_attempts:
            sleeper(settings.recovery_interval_seconds)
    return Observation("down", f"auto-recovery did not restore {settings.service}: {last_detail}")


def _normalized_base_state(value: object) -> dict[str, object] | None:
    if value == {}:
        return {}
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    consecutive = value.get("consecutive")
    if (
        not isinstance(status, str)
        or status not in STATE_STATUSES
        or type(consecutive) is not int
        or consecutive < 0
    ):
        return None
    if status == "down" and consecutive < 1:
        return None
    if status != "down" and consecutive != 0:
        return None
    return {"status": status, "consecutive": consecutive}


def _normalized_notification(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    alert = value.get("alert")
    detail = value.get("detail")
    if not isinstance(alert, str) or alert not in PENDING_ALERTS or not isinstance(detail, str):
        return None
    return {"alert": alert, "detail": detail}


def _normalized_recovery_intent(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    service = value.get("service")
    if not isinstance(service, str) or not service.strip():
        return None
    return {"service": service}


def normalize_state(value: object) -> dict[str, object] | None:
    """Validate the persisted state and migrate the previous singular queue."""
    if not isinstance(value, dict):
        return None

    queued_value = value.get("pending_notifications")
    legacy_value = value.get("pending_notification")
    recovery_value = value.get("recovery_intent")
    if queued_value is not None and legacy_value is not None:
        return None
    if legacy_value is not None:
        if recovery_value is not None:
            return None
        notification = _normalized_notification(legacy_value)
        next_state = _normalized_base_state(
            legacy_value.get("next_state") if isinstance(legacy_value, dict) else None
        )
        exit_code = legacy_value.get("exit_code") if isinstance(legacy_value, dict) else None
        if (
            notification is None
            or next_state is None
            or type(exit_code) is not int
            or exit_code not in {EXIT_OK, EXIT_RECOVERED, EXIT_DOWN}
        ):
            return None
        return {**next_state, "pending_notifications": [notification]}

    state = _normalized_base_state(value)
    if state is None:
        return None
    if queued_value is not None:
        if not isinstance(queued_value, list):
            return None
        notifications: list[dict[str, str]] = []
        for candidate in queued_value:
            notification = _normalized_notification(candidate)
            if notification is None:
                return None
            notifications.append(notification)
        if notifications:
            state["pending_notifications"] = notifications
    if recovery_value is not None:
        recovery_intent = _normalized_recovery_intent(recovery_value)
        if recovery_intent is None:
            return None
        state["recovery_intent"] = recovery_intent
    return state


def read_state(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        value = {}
    except OSError as exc:
        raise RuntimeError(f"unable to read Atlas API health state at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"unable to decode Atlas API health state at {path}") from exc
    normalized = normalize_state(value)
    if normalized is None:
        print("WARNING state schema is invalid; resetting monitor state", file=sys.stderr)
        return {}
    return normalized


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, sort_keys=True).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary_path = None
    except OSError as exc:
        raise RuntimeError(f"unable to atomically persist Atlas API health state at {path}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError as exc:
                print(
                    f"WARNING state temporary cleanup failed: {type(exc).__name__}",
                    file=sys.stderr,
                )


def append_log(state_dir: Path, message: str) -> bool:
    log_path = state_dir / "health.log"
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
    except OSError as exc:
        print(
            f"WARNING health log append failed at {log_path}: {type(exc).__name__}",
            file=sys.stderr,
        )
        return False
    return True


def decide_alert(previous: dict[str, object], observation: Observation, realert_every: int) -> tuple[dict[str, object], str | None, int]:
    previous_status = previous.get("status", "healthy")
    if not isinstance(previous_status, str) or previous_status not in STATE_STATUSES:
        previous_status = "healthy"
    if observation.status == "maintenance":
        return {"status": "maintenance", "consecutive": 0}, None, EXIT_OK
    if observation.status == "healthy":
        alert = "auto-recovered" if recovery_intent_service(previous) is not None else None
        if alert is None and previous_status == "down":
            alert = "recovered"
        return {"status": "healthy", "consecutive": 0}, alert, EXIT_OK
    if observation.status == "recovered":
        return {"status": "healthy", "consecutive": 0}, "auto-recovered", EXIT_RECOVERED

    previous_consecutive = previous.get("consecutive", 0)
    if type(previous_consecutive) is not int or previous_consecutive < 0:
        previous_consecutive = 0
    consecutive = previous_consecutive + 1 if previous_status == "down" else 1
    alert = "down" if previous_status != "down" else None
    if alert is None and realert_every > 0 and consecutive % realert_every == 0:
        alert = "down"
    return {"status": "down", "consecutive": consecutive}, alert, EXIT_DOWN


def pending_notifications(state: dict[str, object]) -> list[dict[str, str]]:
    value = state.get("pending_notifications", [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def state_with_pending_notifications(
    next_state: dict[str, object], notifications: list[dict[str, str]]
) -> dict[str, object]:
    state = dict(next_state)
    if notifications:
        state["pending_notifications"] = notifications
    return state


def state_with_recovery_intent(state: dict[str, object], service: str) -> dict[str, object]:
    pending_state = dict(state)
    pending_state.setdefault("status", "healthy")
    pending_state.setdefault("consecutive", 0)
    pending_state["recovery_intent"] = {"service": service}
    return pending_state


def recovery_intent_service(state: dict[str, object]) -> str | None:
    intent = state.get("recovery_intent")
    if not isinstance(intent, dict):
        return None
    service = intent.get("service")
    return service if isinstance(service, str) and service.strip() else None


def publish(ntfy_url: str, topic: str, title: str, body: str, priority: str, tags: str) -> bool:
    """Deliver one alert without putting the private topic in a process argv."""
    delivered = False
    if not topic.strip():
        print("WARNING alert topic is not configured")
    else:
        request = urllib.request.Request(
            f"{ntfy_url.rstrip('/')}/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                delivered = 200 <= int(response.status) < 300
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"WARNING alert delivery failed: {type(exc).__name__}")
    try:
        subprocess.run(
            ("notify-send", "-u", "critical", title, body),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        print(f"WARNING desktop notification failed: {type(exc).__name__}", file=sys.stderr)
    return delivered


def alert_message(alert: str, detail: str) -> tuple[str, str, str, str]:
    if alert == "auto-recovered":
        return (
            "atlas-api auto-recovered",
            detail,
            "urgent",
            "white_check_mark,warning",
        )
    if alert == "recovered":
        return ("atlas-api recovered", detail, "default", "white_check_mark")
    return ("atlas-api DOWN", detail, "urgent", "rotating_light,warning")


def run_healthcheck(
    settings: Settings,
    *,
    runner: Runner = _run,
    opener: Opener = urllib.request.urlopen,
    notifier: Notifier = publish,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Run one serialized observation/recovery/notification cycle."""
    state_path = settings.state_dir / "state.json"
    with serialized_monitor_state(settings):
        previous = read_state(state_path)
        observation = observe(
            settings,
            runner,
            opener,
            sleeper,
            before_start=lambda: write_state(
                state_path, state_with_recovery_intent(previous, settings.service)
            ),
        )
        next_state, alert, exit_code = decide_alert(previous, observation, settings.realert_every)
        notifications = pending_notifications(previous)
        if alert is not None:
            detail = observation.detail
            recovered_service = recovery_intent_service(previous)
            if alert == "auto-recovered" and recovered_service is not None:
                detail = f"auto-recovered {recovered_service}: {detail}"
            notifications.append({"alert": alert, "detail": detail})
        append_log(settings.state_dir, f"{observation.status.upper()} {observation.detail}")
        print(f"{observation.status.upper()}: {observation.detail}")

        if not notifications:
            write_state(state_path, next_state)
            return exit_code
        write_state(state_path, state_with_pending_notifications(next_state, notifications))
        if settings.no_alert:
            print("WARNING --no-alert: notifications queued for retry")
            return EXIT_ALERT_UNDELIVERED

        for index, notification in enumerate(notifications):
            title, body, priority, tags = alert_message(
                notification["alert"], notification["detail"]
            )
            if not notifier(settings.ntfy_url, settings.ntfy_topic, title, body, priority, tags):
                print("WARNING alert undelivered; transition remains queued for retry")
                return EXIT_ALERT_UNDELIVERED
            write_state(
                state_path,
                state_with_pending_notifications(next_state, notifications[index + 1 :]),
            )

        return exit_code


def _settings_from_args(argv: Sequence[str] | None) -> tuple[Settings, str]:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--enter-maintenance", action="store_true")
    action.add_argument("--exit-maintenance", action="store_true")
    parser.add_argument("--service", default=os.environ.get("ATLAS_API_HEALTHCHECK_SERVICE", DEFAULT_SERVICE))
    parser.add_argument("--probe-url", default=os.environ.get("ATLAS_API_HEALTHCHECK_PROBE_URL", DEFAULT_PROBE_URL))
    parser.add_argument("--ntfy-url", default=os.environ.get("ATLAS_API_HEALTHCHECK_NTFY_URL", DEFAULT_NTFY_URL))
    parser.add_argument("--ntfy-topic", default=os.environ.get("ATLAS_API_HEALTHCHECK_NTFY_TOPIC", ""))
    parser.add_argument("--state-dir", default=os.environ.get("ATLAS_API_HEALTHCHECK_STATE_DIR", str(DEFAULT_STATE_DIR)))
    parser.add_argument("--maintenance-lock", default=os.environ.get("ATLAS_API_HEALTHCHECK_MAINTENANCE_LOCK", str(DEFAULT_MAINTENANCE_LOCK)))
    parser.add_argument("--realert-every", type=int, default=int(os.environ.get("ATLAS_API_HEALTHCHECK_REALERT_EVERY", DEFAULT_REALERT_EVERY)))
    parser.add_argument("--recovery-attempts", type=int, default=int(os.environ.get("ATLAS_API_HEALTHCHECK_RECOVERY_ATTEMPTS", DEFAULT_RECOVERY_ATTEMPTS)))
    parser.add_argument("--recovery-interval-seconds", type=float, default=float(os.environ.get("ATLAS_API_HEALTHCHECK_RECOVERY_INTERVAL_SECONDS", DEFAULT_RECOVERY_INTERVAL_SECONDS)))
    parser.add_argument("--no-alert", action="store_true")
    args = parser.parse_args(argv)
    if not args.service.strip():
        raise ValueError("service must not be blank")
    if args.realert_every < 0:
        raise ValueError("realert-every must not be negative")
    if args.recovery_attempts < 1:
        raise ValueError("recovery-attempts must be at least one")
    if args.recovery_interval_seconds < 0:
        raise ValueError("recovery-interval-seconds must not be negative")
    settings = Settings(
        service=args.service,
        probe_url=args.probe_url,
        ntfy_url=args.ntfy_url,
        ntfy_topic=args.ntfy_topic,
        state_dir=Path(args.state_dir),
        maintenance_lock=Path(args.maintenance_lock),
        realert_every=args.realert_every,
        recovery_attempts=args.recovery_attempts,
        recovery_interval_seconds=args.recovery_interval_seconds,
        no_alert=args.no_alert,
    )
    selected_action = (
        "enter-maintenance"
        if args.enter_maintenance
        else "exit-maintenance"
        if args.exit_maintenance
        else "healthcheck"
    )
    return settings, selected_action


def main(argv: Sequence[str] | None = None) -> int:
    settings, action = _settings_from_args(argv)
    if action == "enter-maintenance":
        return enter_maintenance(settings)
    if action == "exit-maintenance":
        return exit_maintenance(settings)
    return run_healthcheck(settings)


if __name__ == "__main__":
    sys.exit(main())
