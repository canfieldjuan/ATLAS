"""Regression coverage for the standalone Atlas API liveness monitor."""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import string
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "atlas_api_healthcheck", REPO_ROOT / "scripts" / "atlas_api_healthcheck.py"
)
healthcheck = importlib.util.module_from_spec(_SPEC)
sys.modules["atlas_api_healthcheck"] = healthcheck
assert _SPEC.loader is not None
_SPEC.loader.exec_module(healthcheck)

_INSTALLER_SPEC = importlib.util.spec_from_file_location(
    "install_atlas_api_healthcheck", REPO_ROOT / "scripts" / "install_atlas_api_healthcheck.py"
)
installer = importlib.util.module_from_spec(_INSTALLER_SPEC)
sys.modules["install_atlas_api_healthcheck"] = installer
assert _INSTALLER_SPEC.loader is not None
_INSTALLER_SPEC.loader.exec_module(installer)


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _Runner:
    def __init__(
        self,
        *,
        active: bool,
        start_returncode: int = 0,
        start_stdout: str = "",
        start_stderr: str = "",
    ) -> None:
        self.active = active
        self.start_returncode = start_returncode
        self.start_stdout = start_stdout
        self.start_stderr = start_stderr
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command):
        command = tuple(command)
        self.commands.append(command)
        if "is-active" in command:
            return subprocess.CompletedProcess(command, 0 if self.active else 3, "", "")
        assert "start" in command
        if self.start_returncode == 0:
            self.active = True
        return subprocess.CompletedProcess(
            command,
            self.start_returncode,
            self.start_stdout,
            self.start_stderr,
        )


class _InstallerRunner:
    def __init__(self, *, failing_command: tuple[str, ...] | None = None) -> None:
        self.failing_command = failing_command
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command):
        command = tuple(command)
        self.commands.append(command)
        if command == self.failing_command:
            return subprocess.CompletedProcess(command, 1, "", "unit\n\x1b[31mfailed")
        return subprocess.CompletedProcess(command, 0, "", "")


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


def _install_paths(tmp_path: Path) -> installer.InstallPaths:
    bin_dir = tmp_path / "bin"
    return installer.InstallPaths(
        source_root=REPO_ROOT,
        bin_dir=bin_dir,
        systemd_dir=tmp_path / "systemd",
        config_dir=tmp_path / "config",
        legacy_monitor=bin_dir / installer.LEGACY_MONITOR_NAME,
    )


def _topic_grammar_oracle(value: str) -> bool:
    normalized = value.strip()
    allowed_leading = set(string.ascii_letters + string.digits)
    allowed_body = allowed_leading | {".", "_", "-"}
    return bool(
        1 <= len(normalized) <= 200
        and normalized[0] in allowed_leading
        and all(character in allowed_body for character in normalized)
    )


def _topic_leading_tokens() -> str:
    return string.ascii_letters + string.digits + "._-/ \u00e9"


def _topic_body_families() -> str:
    return string.ascii_letters + string.digits + "._-/ \u00e9"


def _topic_wrappers() -> tuple[str, ...]:
    return "", " ", "\n"


def test_notification_topic_grammar_class_closure():
    """Check the bounded topic grammar across token, family, and wrapper axes."""
    for leading_token, body_family, length, wrapper in itertools.product(
        _topic_leading_tokens(),
        _topic_body_families(),
        (0, 1, 2, 200, 201),
        _topic_wrappers(),
    ):
        base = "" if length == 0 else leading_token + body_family * (length - 1)
        candidate = f"{wrapper}{base}{wrapper}"
        expected = _topic_grammar_oracle(candidate)
        if expected:
            assert installer._validated_topic(candidate, source="grammar property") == candidate.strip()
        else:
            with pytest.raises(RuntimeError):
                installer._validated_topic(candidate, source="grammar property")


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


def test_failed_start_keeps_bounded_sanitized_diagnostics(tmp_path):
    runner = _Runner(
        active=False,
        start_returncode=1,
        start_stderr="unit failed\n\x1b[31m" + "x" * 400,
    )

    observation = healthcheck.observe(
        _settings(tmp_path), runner, _opener(204), sleeper=lambda _: None
    )

    assert observation.status == "down"
    assert "exit 1" in observation.detail
    assert "\n" not in observation.detail
    assert "\x1b" not in observation.detail
    assert len(observation.detail) <= 320


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


def test_undelivered_down_transition_is_retried(tmp_path):
    settings = _settings(tmp_path)
    runner = _Runner(active=True)
    attempts: list[tuple] = []

    def notifier(*args):
        attempts.append(args)
        return len(attempts) == 2

    first = healthcheck.run_healthcheck(
        settings, runner=runner, opener=_opener(503), notifier=notifier, sleeper=lambda _: None
    )
    assert first == healthcheck.EXIT_ALERT_UNDELIVERED
    assert _state(settings)["pending_notification"]["alert"] == "down"

    second = healthcheck.run_healthcheck(
        settings, runner=runner, opener=_opener(503), notifier=notifier, sleeper=lambda _: None
    )
    assert second == healthcheck.EXIT_DOWN
    assert len(attempts) == 2
    assert _state(settings) == {"consecutive": 1, "status": "down"}


def test_undelivered_recovery_transition_is_retried(tmp_path):
    settings = _settings(tmp_path)
    settings.state_dir.mkdir(parents=True)
    (settings.state_dir / "state.json").write_text(
        json.dumps({"consecutive": 2, "status": "down"}), encoding="utf-8"
    )
    runner = _Runner(active=True)
    attempts: list[tuple] = []

    def notifier(*args):
        attempts.append(args)
        return len(attempts) == 2

    first = healthcheck.run_healthcheck(
        settings, runner=runner, opener=_opener(204), notifier=notifier, sleeper=lambda _: None
    )
    assert first == healthcheck.EXIT_ALERT_UNDELIVERED
    assert _state(settings)["pending_notification"]["alert"] == "recovered"

    second = healthcheck.run_healthcheck(
        settings, runner=runner, opener=_opener(204), notifier=notifier, sleeper=lambda _: None
    )
    assert second == healthcheck.EXIT_OK
    assert attempts[1][2] == "atlas-api recovered"
    assert _state(settings) == {"consecutive": 0, "status": "healthy"}


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
    assert _state(settings)["pending_notification"]["alert"] == "auto-recovered"


def test_no_alert_does_not_consume_a_transition(tmp_path):
    settings = _settings(tmp_path, no_alert=True)

    result = healthcheck.run_healthcheck(
        settings,
        runner=_Runner(active=True),
        opener=_opener(503),
        notifier=lambda *args: (_ for _ in ()).throw(AssertionError("must not notify")),
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_ALERT_UNDELIVERED
    assert not (settings.state_dir / "state.json").exists()


def test_invalid_state_is_visible_before_monitor_resets_it(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    state_path.write_text("not-json", encoding="utf-8")

    assert healthcheck.read_state(state_path) == {}
    assert "WARNING state read failed: JSONDecodeError" in capsys.readouterr().err


def test_health_log_failure_has_context(tmp_path):
    invalid_state_dir = tmp_path / "not-a-directory"
    invalid_state_dir.write_text("occupied", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unable to append Atlas API health log"):
        healthcheck.append_log(invalid_state_dir, "DOWN test")


def test_health_lock_failure_has_context(tmp_path):
    settings = _settings(tmp_path)
    settings.state_dir.mkdir(parents=True)
    (settings.state_dir / "state.lock").mkdir()

    with pytest.raises(RuntimeError, match="unable to open Atlas API health lock"):
        healthcheck.run_healthcheck(
            settings,
            runner=_Runner(active=True),
            opener=_opener(204),
            notifier=lambda *args: True,
            sleeper=lambda _: None,
        )


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


def test_publish_reports_an_unavailable_desktop_notification(monkeypatch, capsys):
    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", lambda *args, **kwargs: _Response(200))
    monkeypatch.setattr(
        healthcheck.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("notify-send")),
    )

    assert healthcheck.publish("https://ntfy.test", "private-topic", "Title", "Body", "urgent", "warning")
    assert "WARNING desktop notification failed: FileNotFoundError" in capsys.readouterr().err


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


def test_installer_deploys_source_and_invokes_enabled_timer_path(tmp_path):
    paths = _install_paths(tmp_path)
    paths.legacy_monitor.parent.mkdir(parents=True)
    paths.legacy_monitor.write_text(
        'NTFY_TOPIC="${ATLAS_HC_NTFY_TOPIC:-test-private-topic}"\n', encoding="utf-8"
    )
    runner = _InstallerRunner()

    messages = installer.install(paths, runner=runner, environment={})

    assert paths.installed_monitor.read_bytes() == (
        REPO_ROOT / "scripts" / "atlas_api_healthcheck.py"
    ).read_bytes()
    assert os.access(paths.installed_monitor, os.X_OK)
    assert (paths.systemd_dir / installer.SERVICE_NAME).read_bytes() == (
        REPO_ROOT / "config" / installer.SERVICE_NAME
    ).read_bytes()
    assert (paths.systemd_dir / installer.TIMER_NAME).read_bytes() == (
        REPO_ROOT / "config" / installer.TIMER_NAME
    ).read_bytes()
    assert paths.notification_env.read_text(encoding="utf-8") == (
        f"{installer.TOPIC_ENV}=test-private-topic\n"
    )
    assert paths.notification_env.stat().st_mode & 0o077 == 0
    assert "test-private-topic" not in "\n".join(messages)
    assert runner.commands == [
        ("systemctl", "--user", "daemon-reload"),
        ("systemctl", "--user", "enable", "--now", installer.TIMER_NAME),
        ("systemctl", "--user", "start", "--wait", installer.SERVICE_NAME),
    ]

    check_runner = _InstallerRunner()
    ok, check_messages = installer.check_install(paths, runner=check_runner)
    assert ok
    assert all(message.startswith("ok:") for message in check_messages)
    assert check_runner.commands == [
        ("systemctl", "--user", "is-enabled", "--quiet", installer.TIMER_NAME),
        ("systemctl", "--user", "is-active", "--quiet", installer.TIMER_NAME),
    ]


def test_installer_requires_private_topic_before_writing_or_enabling(tmp_path):
    paths = _install_paths(tmp_path)
    runner = _InstallerRunner()

    with pytest.raises(RuntimeError, match=installer.TOPIC_ENV):
        installer.install(paths, runner=runner, environment={})

    assert not paths.installed_monitor.exists()
    assert runner.commands == []


def test_installer_surfaces_sanitized_systemd_failure(tmp_path):
    paths = _install_paths(tmp_path)
    runner = _InstallerRunner(failing_command=("systemctl", "--user", "daemon-reload"))

    with pytest.raises(RuntimeError, match="systemd reload failed") as error:
        installer.install(
            paths,
            runner=runner,
            environment={installer.TOPIC_ENV: "test-private-topic"},
        )

    assert "\n" not in str(error.value)
    assert "\x1b" not in str(error.value)
