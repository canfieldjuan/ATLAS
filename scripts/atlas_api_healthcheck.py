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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


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


def service_is_active(service: str, runner: Runner) -> bool:
    return runner(("systemctl", "--user", "is-active", "--quiet", service)).returncode == 0


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


def observe(settings: Settings, runner: Runner, opener: Opener, sleeper: Callable[[float], None]) -> Observation:
    """Observe the provider and recover only an inactive, non-maintenance unit."""
    if settings.maintenance_lock.exists():
        return Observation("maintenance", f"maintenance lock present: {settings.maintenance_lock}")

    if service_is_active(settings.service, runner):
        healthy, detail = probe_lead_intake(settings.probe_url, opener)
        return Observation("healthy" if healthy else "down", detail)

    started = runner(("systemctl", "--user", "start", settings.service))
    if started.returncode != 0:
        return Observation("down", f"failed to start inactive unit {settings.service}")

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


def read_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")


def append_log(state_dir: Path, message: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
    with (state_dir / "health.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def decide_alert(previous: dict[str, object], observation: Observation, realert_every: int) -> tuple[dict[str, object], str | None, int]:
    previous_status = str(previous.get("status", "healthy"))
    if observation.status == "maintenance":
        return {"status": "maintenance", "consecutive": 0}, None, EXIT_OK
    if observation.status == "healthy":
        alert = "recovered" if previous_status == "down" else None
        return {"status": "healthy", "consecutive": 0}, alert, EXIT_OK
    if observation.status == "recovered":
        return {"status": "healthy", "consecutive": 0}, "auto-recovered", EXIT_RECOVERED

    consecutive = int(previous.get("consecutive", 0)) + 1 if previous_status == "down" else 1
    alert = "down" if previous_status != "down" else None
    if alert is None and realert_every > 0 and consecutive % realert_every == 0:
        alert = "down"
    return {"status": "down", "consecutive": consecutive}, alert, EXIT_DOWN


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
    except OSError:
        pass
    return delivered


def alert_message(alert: str, observation: Observation) -> tuple[str, str, str, str]:
    if alert == "auto-recovered":
        return (
            "atlas-api auto-recovered",
            observation.detail,
            "urgent",
            "white_check_mark,warning",
        )
    if alert == "recovered":
        return ("atlas-api recovered", observation.detail, "default", "white_check_mark")
    return ("atlas-api DOWN", observation.detail, "urgent", "rotating_light,warning")


def run_healthcheck(
    settings: Settings,
    *,
    runner: Runner = _run,
    opener: Opener = urllib.request.urlopen,
    notifier: Notifier = publish,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Run one serialized observation/recovery/notification cycle."""
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = settings.state_dir / "state.json"
    lock_path = settings.state_dir / "state.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        observation = observe(settings, runner, opener, sleeper)
        previous = read_state(state_path)
        next_state, alert, exit_code = decide_alert(previous, observation, settings.realert_every)
        append_log(settings.state_dir, f"{observation.status.upper()} {observation.detail}")
        print(f"{observation.status.upper()}: {observation.detail}")

        delivered = True
        if alert and not settings.no_alert:
            title, body, priority, tags = alert_message(alert, observation)
            delivered = notifier(settings.ntfy_url, settings.ntfy_topic, title, body, priority, tags)
        elif alert:
            print(f"WARNING --no-alert: {alert} notification not sent")

        write_state(state_path, next_state)
        if alert and not settings.no_alert and not delivered:
            return EXIT_ALERT_UNDELIVERED
        return exit_code


def _settings_from_args(argv: Sequence[str] | None) -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return Settings(
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


def main(argv: Sequence[str] | None = None) -> int:
    return run_healthcheck(_settings_from_args(argv))


if __name__ == "__main__":
    sys.exit(main())
