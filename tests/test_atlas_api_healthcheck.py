"""Regression coverage for the standalone Atlas API liveness monitor."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "atlas_api_healthcheck", REPO_ROOT / "scripts" / "atlas_api_healthcheck.py"
)
healthcheck = importlib.util.module_from_spec(_SPEC)
sys.modules["atlas_api_healthcheck"] = healthcheck
assert _SPEC.loader is not None
_SPEC.loader.exec_module(healthcheck)


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _Runner:
    def __init__(self, *, active: bool, start_returncode: int = 0) -> None:
        self.active = active
        self.start_returncode = start_returncode
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command):
        command = tuple(command)
        self.commands.append(command)
        if "is-active" in command:
            return subprocess.CompletedProcess(command, 0 if self.active else 3, "", "")
        assert "start" in command
        if self.start_returncode == 0:
            self.active = True
        return subprocess.CompletedProcess(command, self.start_returncode, "", "")


def _settings(tmp_path: Path, **overrides):
    values = {
        "service": "atlas-api.service",
        "probe_url": "http://example.test/api/v1/leads/intake",
        "ntfy_url": "https://ntfy.test",
        "ntfy_topic": "private-topic",
        "state_dir": tmp_path / "state",
        "maintenance_lock": tmp_path / "atlas-api.maintenance",
        "realert_every": 6,
        "recovery_attempts": 1,
        "recovery_interval_seconds": 0.0,
        "no_alert": False,
    }
    values.update(overrides)
    return healthcheck.Settings(**values)


def _opener(status: int, seen: list | None = None):
    def _open(request, *, timeout):
        if seen is not None:
            seen.append((request.method, request.headers, timeout))
        return _Response(status)

    return _open


def _start_commands(runner: _Runner) -> list[tuple[str, ...]]:
    return [command for command in runner.commands if "start" in command]


def _state(settings) -> dict:
    return json.loads((settings.state_dir / "state.json").read_text(encoding="utf-8"))


def test_inactive_service_is_started_and_reprobed(tmp_path):
    settings = _settings(tmp_path)
    runner = _Runner(active=False)
    sent: list[tuple] = []
    seen: list = []

    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(204, seen),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_RECOVERED
    assert _start_commands(runner) == [
        ("systemctl", "--user", "start", "atlas-api.service")
    ]
    assert seen == [("OPTIONS", {"Origin": "https://effinghamofficemaids.com", "Access-control-request-method": "POST"}, 8)]
    assert sent[0][2] == "atlas-api auto-recovered"
    assert _state(settings) == {"consecutive": 0, "status": "healthy"}


def test_maintenance_lock_never_starts_service(tmp_path):
    settings = _settings(tmp_path)
    settings.maintenance_lock.touch()
    runner = _Runner(active=False)

    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(204),
        notifier=lambda *args: (_ for _ in ()).throw(AssertionError("must not notify")),
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_OK
    assert _start_commands(runner) == []
    assert _state(settings) == {"consecutive": 0, "status": "maintenance"}


def test_failed_start_remains_a_visible_down_result(tmp_path):
    settings = _settings(tmp_path)
    runner = _Runner(active=False, start_returncode=1)
    sent: list[tuple] = []

    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(204),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert len(_start_commands(runner)) == 1
    assert sent[0][2] == "atlas-api DOWN"
    assert _state(settings) == {"consecutive": 1, "status": "down"}


def test_failed_reprobe_after_start_remains_down(tmp_path):
    settings = _settings(tmp_path)
    runner = _Runner(active=False)

    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(503),
        notifier=lambda *args: True,
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert len(_start_commands(runner)) == 1
    assert _state(settings) == {"consecutive": 1, "status": "down"}


def test_active_unhealthy_service_alerts_without_restart(tmp_path):
    settings = _settings(tmp_path)
    runner = _Runner(active=True)

    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(503),
        notifier=lambda *args: True,
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert _start_commands(runner) == []
    assert _state(settings) == {"consecutive": 1, "status": "down"}


def test_missing_notification_configuration_does_not_block_recovery(tmp_path):
    settings = _settings(tmp_path, ntfy_topic="")
    runner = _Runner(active=False)

    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(204),
        notifier=lambda *args: False,
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_ALERT_UNDELIVERED
    assert len(_start_commands(runner)) == 1
    assert _state(settings) == {"consecutive": 0, "status": "healthy"}


def test_publish_preserves_the_desktop_notification_channel(monkeypatch):
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", lambda *args, **kwargs: _Response(200))
    monkeypatch.setattr(
        healthcheck.subprocess,
        "run",
        lambda command, **kwargs: (commands.append(tuple(command)), subprocess.CompletedProcess(command, 0))[1],
    )

    assert healthcheck.publish("https://ntfy.test", "private-topic", "Title", "Body", "urgent", "warning")
    assert commands == [("notify-send", "-u", "critical", "Title", "Body")]


def test_missing_ntfy_topic_still_preserves_desktop_notification(monkeypatch):
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        healthcheck.subprocess,
        "run",
        lambda command, **kwargs: (commands.append(tuple(command)), subprocess.CompletedProcess(command, 0))[1],
    )

    assert not healthcheck.publish("https://ntfy.test", "", "Title", "Body", "urgent", "warning")
    assert commands == [("notify-send", "-u", "critical", "Title", "Body")]


def test_service_template_uses_installed_script_and_private_environment():
    service = (REPO_ROOT / "config" / "atlas-api-healthcheck.service").read_text(encoding="utf-8")
    timer = (REPO_ROOT / "config" / "atlas-api-healthcheck.timer").read_text(encoding="utf-8")

    assert "ExecStart=/usr/bin/python3 %h/.local/bin/atlas-api-healthcheck.py" in service
    assert "EnvironmentFile=-%h/.config/atlas/atlas-api-healthcheck.env" in service
    assert "Environment=DISPLAY=:0" in service
    assert "Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus" in service
    assert "ATLAS_API_HEALTHCHECK_NTFY_TOPIC=<private-topic>" in service
    assert "eom-atlas-api-health-" not in service
    assert "OnUnitActiveSec=5min" in timer
