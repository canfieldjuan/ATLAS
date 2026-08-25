"""Regression coverage for the standalone Atlas API liveness monitor."""
from __future__ import annotations

import importlib.util
import http.client
import itertools
import json
import os
import shutil
import stat
import string
import subprocess
import sys
import threading
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
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = (
            {
                "Access-Control-Allow-Origin": healthcheck.PROBE_ORIGIN,
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
            if headers is None
            else headers
        )

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
        load_state: str = "loaded",
        active_state: str | None = None,
        query_returncode: int = 0,
        query_stdout: str | None = None,
        query_stderr: str = "",
    ) -> None:
        self.active = active
        self.load_state = load_state
        self.active_state = active_state or ("active" if active else "inactive")
        self.start_returncode = start_returncode
        self.start_stdout = start_stdout
        self.start_stderr = start_stderr
        self.query_returncode = query_returncode
        self.query_stdout = query_stdout
        self.query_stderr = query_stderr
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command):
        command = tuple(command)
        self.commands.append(command)
        if "show" in command:
            stdout = self.query_stdout
            if stdout is None:
                stdout = f"LoadState={self.load_state}\nActiveState={self.active_state}\n"
            return subprocess.CompletedProcess(
                command, self.query_returncode, stdout, self.query_stderr
            )
        assert "start" in command
        if self.start_returncode == 0:
            self.active = True
            self.active_state = "active"
        return subprocess.CompletedProcess(
            command,
            self.start_returncode,
            self.start_stdout,
            self.start_stderr,
        )


class _InstallerRunner:
    def __init__(
        self,
        *,
        failing_command: tuple[str, ...] | None = None,
        timer_enabled: bool = False,
        timer_runtime: bool = False,
        timer_active: bool = False,
        timer_load_state: str = "loaded",
        health_service_load_state: str = "loaded",
        health_service_active_state: str = "inactive",
        need_daemon_reload: dict[str, str] | None = None,
    ) -> None:
        self.failing_command = failing_command
        self.timer_enabled = timer_enabled or timer_runtime
        self.timer_runtime = timer_runtime
        self.timer_active = timer_active
        self.timer_load_state = timer_load_state
        self.health_service_load_state = health_service_load_state
        self.health_service_active_state = health_service_active_state
        self.need_daemon_reload = need_daemon_reload or {
            installer.SERVICE_NAME: "no",
            installer.TIMER_NAME: "no",
        }
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command):
        command = tuple(command)
        self.commands.append(command)
        if command == self.failing_command:
            return subprocess.CompletedProcess(command, 1, "", "unit\n\x1b[31mfailed")
        if "show" in command and "--property=LoadState" in command:
            if command[-1] == installer.TIMER_NAME:
                load_state = self.timer_load_state
                active_state = "active" if self.timer_active else "inactive"
            else:
                load_state = self.health_service_load_state
                active_state = self.health_service_active_state
            return subprocess.CompletedProcess(
                command,
                0,
                f"LoadState={load_state}\nActiveState={active_state}\n",
                "",
            )
        if "show" in command and "--property=NeedDaemonReload" in command:
            return subprocess.CompletedProcess(
                command, 0, self.need_daemon_reload.get(command[-1], "") + "\n", ""
            )
        if "show" in command and "--property=UnitFileState" in command:
            state = (
                "enabled-runtime"
                if self.timer_runtime
                else "enabled"
                if self.timer_enabled
                else "disabled"
            )
            return subprocess.CompletedProcess(command, 0, state + "\n", "")
        if command[2:4] == ("enable", "--now"):
            self.timer_enabled = True
            self.timer_runtime = False
            self.timer_active = True
        elif command[2:4] == ("disable", "--now"):
            self.timer_enabled = False
            self.timer_runtime = False
            self.timer_active = False
        elif command[2:4] == ("enable", "--runtime"):
            self.timer_enabled = True
            self.timer_runtime = True
        elif command[2:] == ("enable", installer.TIMER_NAME):
            self.timer_enabled = True
            self.timer_runtime = False
        elif command[2:] == ("start", installer.TIMER_NAME):
            self.timer_active = True
        elif command[2:] == ("stop", installer.TIMER_NAME):
            self.timer_active = False
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


def _opener(
    status: int,
    seen: list | None = None,
    headers: dict[str, str] | None = None,
):
    def _open(request, *, timeout):
        if seen is not None:
            seen.append((request.method, request.headers, timeout))
        return _Response(status, headers)

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
    assert seen == [
        (
            "OPTIONS",
            {
                "Origin": healthcheck.PROBE_ORIGIN,
                "Access-control-request-method": "POST",
                "Access-control-request-headers": "Content-Type",
            },
            8,
        )
    ]
    assert sent[0][2] == "atlas-api auto-recovered"
    assert _state(settings) == {"consecutive": 0, "status": "healthy"}


@pytest.mark.parametrize(
    ("headers", "expected_detail"),
    [
        ({}, "CORS origin"),
        (
            {
                "Access-Control-Allow-Origin": "https://example.invalid",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "CORS origin",
        ),
        (
            {
                "Access-Control-Allow-Origin": healthcheck.PROBE_ORIGIN,
                "Access-Control-Allow-Methods": "OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "POST",
        ),
        (
            {
                "Access-Control-Allow-Origin": healthcheck.PROBE_ORIGIN,
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "X-Request-ID",
            },
            "Content-Type",
        ),
    ],
)
def test_probe_rejects_non_browser_ready_cors_responses(headers, expected_detail):
    healthy, detail = healthcheck.probe_lead_intake(
        "http://example.test/api/v1/leads/intake",
        _opener(204, headers=headers),
    )

    assert not healthy
    assert expected_detail in detail


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


def test_unreadable_maintenance_lock_fails_before_query_or_start(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    runner = _Runner(active=False)
    real_stat = Path.stat

    def fail_only_for_maintenance(path):
        if path == settings.maintenance_lock:
            raise PermissionError("simulated unreadable maintenance marker")
        return real_stat(path)

    monkeypatch.setattr(Path, "stat", fail_only_for_maintenance)

    with pytest.raises(RuntimeError, match="unable to inspect Atlas API maintenance lock"):
        healthcheck.run_healthcheck(
            settings,
            runner=runner,
            opener=_opener(204),
            notifier=lambda *args: True,
            sleeper=lambda _: None,
        )

    assert runner.commands == []


def test_enter_maintenance_serializes_with_an_inflight_recovery(tmp_path):
    settings = _settings(tmp_path)
    first_status_read = threading.Event()
    release_status_read = threading.Event()
    maintenance_started = threading.Event()

    class CoordinatedRunner:
        def __init__(self):
            self.active = False
            self.commands: list[tuple[str, ...]] = []

        def __call__(self, command):
            command = tuple(command)
            self.commands.append(command)
            if "show" in command:
                if not first_status_read.is_set():
                    first_status_read.set()
                    assert release_status_read.wait(timeout=2)
                active_state = "active" if self.active else "inactive"
                return subprocess.CompletedProcess(
                    command, 0, f"LoadState=loaded\nActiveState={active_state}\n", ""
                )
            if "start" in command:
                self.active = True
            elif "stop" in command:
                self.active = False
            return subprocess.CompletedProcess(command, 0, "", "")

    runner = CoordinatedRunner()
    health_results: list[int] = []
    maintenance_results: list[int] = []
    health_thread = threading.Thread(
        target=lambda: health_results.append(
            healthcheck.run_healthcheck(
                settings,
                runner=runner,
                opener=_opener(204),
                notifier=lambda *args: True,
                sleeper=lambda _: None,
            )
        )
    )

    def enter() -> None:
        maintenance_started.set()
        maintenance_results.append(healthcheck.enter_maintenance(settings, runner=runner))

    maintenance_thread = threading.Thread(target=enter)
    health_thread.start()
    assert first_status_read.wait(timeout=1)
    maintenance_thread.start()
    assert maintenance_started.wait(timeout=1)
    assert not settings.maintenance_lock.exists()
    release_status_read.set()
    health_thread.join(timeout=2)
    maintenance_thread.join(timeout=2)

    assert not health_thread.is_alive()
    assert not maintenance_thread.is_alive()
    assert health_results == [healthcheck.EXIT_RECOVERED]
    assert maintenance_results == [healthcheck.EXIT_OK]
    assert settings.maintenance_lock.exists()
    assert not runner.active
    assert runner.commands.index(("systemctl", "--user", "start", settings.service)) < runner.commands.index(
        ("systemctl", "--user", "stop", settings.service)
    )


def test_exit_maintenance_removes_marker_under_monitor_lock(tmp_path):
    settings = _settings(tmp_path)
    settings.maintenance_lock.parent.mkdir(parents=True, exist_ok=True)
    settings.maintenance_lock.touch()

    assert healthcheck.exit_maintenance(settings) == healthcheck.EXIT_OK
    assert not settings.maintenance_lock.exists()


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


def test_http_protocol_failure_becomes_a_down_observation(tmp_path):
    settings = _settings(tmp_path)
    sent: list[tuple] = []

    result = healthcheck.run_healthcheck(
        settings,
        runner=_Runner(active=True),
        opener=lambda *args, **kwargs: (_ for _ in ()).throw(
            http.client.BadStatusLine("malformed status")
        ),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert "BadStatusLine" in sent[0][3]
    assert _state(settings) == {"consecutive": 1, "status": "down"}


@pytest.mark.parametrize(
    ("runner", "expected_detail"),
    [
        (
            _Runner(active=False, load_state="not-found"),
            "LoadState=not-found",
        ),
        (
            _Runner(active=False, active_state="activating"),
            "ActiveState=activating",
        ),
        (
            _Runner(active=False, query_returncode=1, query_stderr="manager unavailable"),
            "unit-state query failed",
        ),
        (
            _Runner(active=False, query_stdout="LoadState=loaded\n"),
            "incomplete LoadState/ActiveState",
        ),
    ],
)
def test_only_explicit_loaded_inactive_state_is_recoverable(
    tmp_path, runner, expected_detail
):
    settings = _settings(tmp_path)
    sent: list[tuple] = []

    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(204),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert _start_commands(runner) == []
    assert expected_detail in sent[0][3]


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
    assert first == healthcheck.EXIT_DOWN
    assert _state(settings)["pending_notifications"][0]["alert"] == "down"

    second = healthcheck.run_healthcheck(
        settings, runner=runner, opener=_opener(503), notifier=notifier, sleeper=lambda _: None
    )
    assert second == healthcheck.EXIT_DOWN
    assert len(attempts) == 2
    assert _state(settings) == {"consecutive": 2, "status": "down"}


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
    assert _state(settings)["pending_notifications"][0]["alert"] == "recovered"

    second = healthcheck.run_healthcheck(
        settings, runner=runner, opener=_opener(204), notifier=notifier, sleeper=lambda _: None
    )
    assert second == healthcheck.EXIT_OK
    assert attempts[1][2] == "atlas-api recovered"
    assert _state(settings) == {"consecutive": 0, "status": "healthy"}


def test_pending_recovery_does_not_hide_a_current_outage(tmp_path):
    settings = _settings(tmp_path)
    settings.state_dir.mkdir(parents=True)
    (settings.state_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "healthy",
                "consecutive": 0,
                "pending_notifications": [
                    {"alert": "auto-recovered", "detail": "earlier recovery"}
                ],
            }
        ),
        encoding="utf-8",
    )
    sent: list[tuple] = []

    result = healthcheck.run_healthcheck(
        settings,
        runner=_Runner(active=False, start_returncode=1),
        opener=_opener(204),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert [message[2] for message in sent] == [
        "atlas-api auto-recovered",
        "atlas-api DOWN",
    ]
    assert _state(settings) == {"consecutive": 1, "status": "down"}


def test_recovery_outbox_is_persisted_before_a_delivery_crash(tmp_path):
    settings = _settings(tmp_path)
    runner = _Runner(active=False)

    with pytest.raises(SystemExit, match="simulated crash"):
        healthcheck.run_healthcheck(
            settings,
            runner=runner,
            opener=_opener(204),
            notifier=lambda *args: (_ for _ in ()).throw(SystemExit("simulated crash")),
            sleeper=lambda _: None,
        )

    assert _state(settings) == {
        "status": "healthy",
        "consecutive": 0,
        "pending_notifications": [
            {
                "alert": "auto-recovered",
                "detail": "auto-recovered atlas-api.service: lead-intake probe returned browser-ready HTTP 204",
            }
        ],
    }

    sent: list[tuple] = []
    result = healthcheck.run_healthcheck(
        settings,
        runner=runner,
        opener=_opener(204),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_OK
    assert sent[0][2] == "atlas-api auto-recovered"
    assert _state(settings) == {"consecutive": 0, "status": "healthy"}


def test_recovery_outbox_is_persisted_before_console_output_crash(tmp_path, monkeypatch):
    settings = _settings(tmp_path)

    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: (_ for _ in ()).throw(BrokenPipeError("simulated console loss")),
    )

    with pytest.raises(BrokenPipeError, match="simulated console loss"):
        healthcheck.run_healthcheck(
            settings,
            runner=_Runner(active=False),
            opener=_opener(204),
            notifier=lambda *args: True,
            sleeper=lambda _: None,
        )

    assert _state(settings) == {
        "status": "healthy",
        "consecutive": 0,
        "pending_notifications": [
            {
                "alert": "auto-recovered",
                "detail": "auto-recovered atlas-api.service: lead-intake probe returned browser-ready HTTP 204",
            }
        ],
    }


def test_recovery_intent_is_persisted_before_start_and_replayed_after_crash(tmp_path):
    settings = _settings(tmp_path)
    crashed_commands: list[tuple[str, ...]] = []

    def crash_after_start(command):
        command = tuple(command)
        crashed_commands.append(command)
        if "show" in command:
            return subprocess.CompletedProcess(
                command, 0, "LoadState=loaded\nActiveState=inactive\n", ""
            )
        raise SystemExit("simulated crash after systemd accepted start")

    with pytest.raises(SystemExit, match="simulated crash"):
        healthcheck.run_healthcheck(
            settings,
            runner=crash_after_start,
            opener=_opener(204),
            notifier=lambda *args: True,
            sleeper=lambda _: None,
        )

    assert crashed_commands[-1] == ("systemctl", "--user", "start", settings.service)
    assert _state(settings) == {
        "status": "healthy",
        "consecutive": 0,
        "recovery_intent": {"service": settings.service},
    }

    sent: list[tuple] = []
    result = healthcheck.run_healthcheck(
        settings,
        runner=_Runner(active=True),
        opener=_opener(204),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_OK
    assert sent[0][2] == "atlas-api auto-recovered"
    assert _state(settings) == {"status": "healthy", "consecutive": 0}


def test_state_read_error_prevents_recovery_and_state_replacement(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.state_dir.mkdir(parents=True)
    state_path = settings.state_dir / "state.json"
    original_payload = '{"status":"down","consecutive":1}'
    state_path.write_text(original_payload, encoding="utf-8")
    original_read_text = Path.read_text

    def fail_state_read(path, *args, **kwargs):
        if path == state_path:
            raise OSError("transient read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_state_read)
    runner = _Runner(active=False)

    with pytest.raises(RuntimeError, match="unable to read Atlas API health state"):
        healthcheck.run_healthcheck(
            settings,
            runner=runner,
            opener=_opener(204),
            notifier=lambda *args: True,
            sleeper=lambda _: None,
        )

    assert runner.commands == []
    with state_path.open(encoding="utf-8") as handle:
        assert handle.read() == original_payload


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
    assert _state(settings)["pending_notifications"][0]["alert"] == "auto-recovered"


def test_failed_recovery_with_undelivered_alert_preserves_down_exit(tmp_path):
    settings = _settings(tmp_path)

    result = healthcheck.run_healthcheck(
        settings,
        runner=_Runner(active=False, start_returncode=1),
        opener=_opener(204),
        notifier=lambda *args: False,
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert _state(settings)["pending_notifications"][0]["alert"] == "down"


def test_no_alert_does_not_consume_a_transition(tmp_path):
    settings = _settings(tmp_path, no_alert=True)

    result = healthcheck.run_healthcheck(
        settings,
        runner=_Runner(active=True),
        opener=_opener(503),
        notifier=lambda *args: (_ for _ in ()).throw(AssertionError("must not notify")),
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_DOWN
    assert _state(settings)["pending_notifications"][0]["alert"] == "down"


def test_invalid_json_state_is_not_reset_or_replaced(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unable to decode Atlas API health state"):
        healthcheck.read_state(state_path)

    assert state_path.read_text(encoding="utf-8") == "not-json"


@pytest.mark.parametrize(
    "value",
    [
        {"status": [], "consecutive": 0},
        {"status": "unknown", "consecutive": 0},
        {"status": "down", "consecutive": "1"},
        {
            "status": "healthy",
            "consecutive": 0,
            "pending_notification": {
                "alert": "recovered",
                "detail": "old",
                "next_state": {"status": "healthy", "consecutive": 0},
                "exit_code": [],
            },
        },
        {
            "status": "healthy",
            "consecutive": 0,
            "pending_notifications": [{"alert": "unknown", "detail": "old"}],
        },
        {
            "status": "healthy",
            "consecutive": 0,
            "recovery_intent": {"service": ""},
        },
    ],
)
def test_malformed_state_shapes_reset_safely(tmp_path, capsys, value):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(value), encoding="utf-8")

    assert healthcheck.read_state(state_path) == {}
    assert "WARNING state schema is invalid" in capsys.readouterr().err


def test_previous_singular_pending_state_migrates_without_an_outer_status(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "pending_notification": {
                    "alert": "down",
                    "detail": "earlier outage",
                    "next_state": {"status": "down", "consecutive": 1},
                    "exit_code": healthcheck.EXIT_DOWN,
                }
            }
        ),
        encoding="utf-8",
    )

    assert healthcheck.read_state(state_path) == {
        "status": "down",
        "consecutive": 1,
        "pending_notifications": [
            {"alert": "down", "detail": "earlier outage"}
        ],
    }


def test_health_log_failure_is_visible_but_best_effort(tmp_path, capsys):
    invalid_state_dir = tmp_path / "not-a-directory"
    invalid_state_dir.write_text("occupied", encoding="utf-8")

    assert not healthcheck.append_log(invalid_state_dir, "DOWN test")
    assert "WARNING health log append failed" in capsys.readouterr().err


def test_health_log_failure_does_not_suppress_recovery_alert(tmp_path, capsys):
    settings = _settings(tmp_path)
    settings.state_dir.mkdir(parents=True)
    (settings.state_dir / "health.log").mkdir()
    sent: list[tuple] = []

    result = healthcheck.run_healthcheck(
        settings,
        runner=_Runner(active=False),
        opener=_opener(204),
        notifier=lambda *args: (sent.append(args), True)[1],
        sleeper=lambda _: None,
    )

    assert result == healthcheck.EXIT_RECOVERED
    assert sent[0][2] == "atlas-api auto-recovered"
    assert "WARNING health log append failed" in capsys.readouterr().err


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


def test_publish_treats_an_http_protocol_failure_as_undelivered(monkeypatch, capsys):
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        healthcheck.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            http.client.BadStatusLine("malformed status")
        ),
    )
    monkeypatch.setattr(
        healthcheck.subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append(tuple(command)),
            subprocess.CompletedProcess(command, 0),
        )[1],
    )

    assert not healthcheck.publish(
        "https://ntfy.test", "private-topic", "Title", "Body", "urgent", "warning"
    )
    assert "WARNING alert delivery failed: BadStatusLine" in capsys.readouterr().out
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
    assert "DBUS_SESSION_BUS_ADDRESS=" not in service
    assert "/run/user/1000" not in service
    assert "ATLAS_API_HEALTHCHECK_NTFY_TOPIC=<private-topic>" in service
    assert "eom-atlas-api-health-" not in service
    assert "SuccessExitStatus=0 2 4" in service
    assert "OnUnitActiveSec=5min" in timer


def test_systemd_templates_verify_when_systemd_analyze_is_available():
    systemd_analyze = shutil.which("systemd-analyze")
    if systemd_analyze is None:
        pytest.skip("systemd-analyze is unavailable")

    result = subprocess.run(
        [
            systemd_analyze,
            "verify",
            str(REPO_ROOT / "config" / installer.SERVICE_NAME),
            str(REPO_ROOT / "config" / installer.TIMER_NAME),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("argument", "expected_action"),
    [
        ("--enter-maintenance", "enter-maintenance"),
        ("--exit-maintenance", "exit-maintenance"),
    ],
)
def test_maintenance_actions_ignore_invalid_healthcheck_numeric_environment(
    monkeypatch, argument, expected_action
):
    monkeypatch.setenv("ATLAS_API_HEALTHCHECK_REALERT_EVERY", "bad")
    monkeypatch.setenv("ATLAS_API_HEALTHCHECK_RECOVERY_ATTEMPTS", "bad")
    monkeypatch.setenv("ATLAS_API_HEALTHCHECK_RECOVERY_INTERVAL_SECONDS", "bad")

    settings, action = healthcheck._settings_from_args([argument])

    assert action == expected_action
    assert settings.realert_every == healthcheck.DEFAULT_REALERT_EVERY
    assert settings.recovery_attempts == healthcheck.DEFAULT_RECOVERY_ATTEMPTS
    assert (
        settings.recovery_interval_seconds
        == healthcheck.DEFAULT_RECOVERY_INTERVAL_SECONDS
    )


def test_healthcheck_action_rejects_invalid_numeric_environment(monkeypatch):
    monkeypatch.setenv("ATLAS_API_HEALTHCHECK_REALERT_EVERY", "bad")

    with pytest.raises(ValueError, match="realert-every must be an integer"):
        healthcheck._settings_from_args([])


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
        (
            "systemctl",
            "--user",
            "show",
            "--property=LoadState",
            "--property=ActiveState",
            installer.TIMER_NAME,
        ),
        (
            "systemctl",
            "--user",
            "show",
            "--property=UnitFileState",
            "--value",
            installer.TIMER_NAME,
        ),
        (
            "systemctl",
            "--user",
            "show",
            "--property=LoadState",
            "--property=ActiveState",
            installer.SERVICE_NAME,
        ),
        ("systemctl", "--user", "stop", installer.SERVICE_NAME),
        ("systemctl", "--user", "daemon-reload"),
        ("systemctl", "--user", "start", "--wait", installer.SERVICE_NAME),
        ("systemctl", "--user", "enable", "--now", installer.TIMER_NAME),
    ]

    check_runner = _InstallerRunner(timer_enabled=True, timer_active=True)
    ok, check_messages = installer.check_install(paths, runner=check_runner)
    assert ok
    assert all(message.startswith("ok:") for message in check_messages)
    assert check_runner.commands == [
        (
            "systemctl",
            "--user",
            "show",
            "--property=LoadState",
            "--property=ActiveState",
            installer.TIMER_NAME,
        ),
        (
            "systemctl",
            "--user",
            "show",
            "--property=UnitFileState",
            "--value",
            installer.TIMER_NAME,
        ),
        (
            "systemctl",
            "--user",
            "show",
            "--property=NeedDaemonReload",
            "--value",
            installer.SERVICE_NAME,
        ),
        (
            "systemctl",
            "--user",
            "show",
            "--property=NeedDaemonReload",
            "--value",
            installer.TIMER_NAME,
        ),
    ]


def test_installer_requires_private_topic_before_writing_or_enabling(tmp_path):
    paths = _install_paths(tmp_path)
    runner = _InstallerRunner()

    with pytest.raises(RuntimeError, match=installer.TOPIC_ENV):
        installer.install(paths, runner=runner, environment={})

    assert not paths.installed_monitor.exists()
    assert runner.commands == []


@pytest.mark.parametrize("configuration_source", ["notification", "legacy"])
def test_installer_rejects_malformed_utf8_before_systemd_or_mutation(
    tmp_path, configuration_source
):
    paths = _install_paths(tmp_path)
    target = (
        paths.notification_env
        if configuration_source == "notification"
        else paths.legacy_monitor
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\xff\xfe")
    runner = _InstallerRunner(timer_active=True)

    with pytest.raises(RuntimeError, match="UnicodeDecodeError"):
        installer.install(paths, runner=runner, environment={})

    assert runner.commands == []
    assert runner.timer_active
    assert target.read_bytes() == b"\xff\xfe"


def test_clean_install_allows_an_absent_old_health_service(tmp_path):
    paths = _install_paths(tmp_path)
    runner = _InstallerRunner(
        timer_load_state="not-found", health_service_load_state="not-found"
    )

    installer.install(
        paths,
        runner=runner,
        environment={installer.TOPIC_ENV: "test-private-topic"},
    )

    assert ("systemctl", "--user", "stop", installer.SERVICE_NAME) not in runner.commands
    assert (
        "systemctl",
        "--user",
        "start",
        "--wait",
        installer.SERVICE_NAME,
    ) in runner.commands
    assert paths.installed_monitor.exists()


def test_installer_rejects_an_unsupported_old_unit_load_state(tmp_path):
    paths = _install_paths(tmp_path)
    runner = _InstallerRunner(health_service_load_state="error")

    with pytest.raises(RuntimeError, match="unsupported LoadState=error"):
        installer.install(
            paths,
            runner=runner,
            environment={installer.TOPIC_ENV: "test-private-topic"},
        )

    assert not paths.installed_monitor.exists()


def test_installer_fails_closed_when_existing_timer_state_cannot_be_queried(tmp_path):
    paths = _install_paths(tmp_path)
    timer_query = (
        "systemctl",
        "--user",
        "show",
        "--property=LoadState",
        "--property=ActiveState",
        installer.TIMER_NAME,
    )
    runner = _InstallerRunner(failing_command=timer_query)

    with pytest.raises(RuntimeError, match="unit-state query failed"):
        installer.install(
            paths,
            runner=runner,
            environment={installer.TOPIC_ENV: "test-private-topic"},
        )

    assert runner.commands == [timer_query]
    assert not paths.installed_monitor.exists()


@pytest.mark.parametrize(
    ("active_state", "unit_file_state", "expected"),
    [
        ("activating", "disabled", "unsupported ActiveState=activating"),
        ("inactive", "masked", "unsupported UnitFileState=masked"),
    ],
)
def test_installer_rejects_timer_states_outside_the_restorable_sets(
    active_state, unit_file_state, expected
):
    def runner(command):
        if "--property=LoadState" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                f"LoadState=loaded\nActiveState={active_state}\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, unit_file_state + "\n", "")

    with pytest.raises(RuntimeError, match=expected):
        installer._timer_state(runner)


def test_install_check_rejects_stale_loaded_unit_definitions(tmp_path):
    paths = _install_paths(tmp_path)
    installer.install(
        paths,
        runner=_InstallerRunner(
            timer_load_state="not-found", health_service_load_state="not-found"
        ),
        environment={installer.TOPIC_ENV: "test-private-topic"},
    )
    check_runner = _InstallerRunner(
        timer_enabled=True,
        timer_active=True,
        need_daemon_reload={
            installer.SERVICE_NAME: "yes",
            installer.TIMER_NAME: "no",
        },
    )

    ok, messages = installer.check_install(paths, runner=check_runner)

    assert not ok
    assert f"systemd requires daemon-reload for {installer.SERVICE_NAME}" in messages


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_install_check_rejects_monitor_alias_to_checkout_source(tmp_path, link_kind):
    paths = _install_paths(tmp_path)
    installer.install(
        paths,
        runner=_InstallerRunner(
            timer_load_state="not-found", health_service_load_state="not-found"
        ),
        environment={installer.TOPIC_ENV: "test-private-topic"},
    )
    source = REPO_ROOT / "scripts" / "atlas_api_healthcheck.py"
    paths.installed_monitor.unlink()
    if link_kind == "symlink":
        paths.installed_monitor.symlink_to(source)
    else:
        os.link(source, paths.installed_monitor)

    ok, messages = installer.check_install(
        paths, runner=_InstallerRunner(timer_enabled=True, timer_active=True)
    )

    assert not ok
    assert f"not an independent regular-file copy: {paths.installed_monitor}" in messages


def test_installer_quiesces_old_service_before_replacing_files(tmp_path, monkeypatch):
    paths = _install_paths(tmp_path)
    events: list[tuple[str, ...]] = []
    runner = _InstallerRunner()
    original_write_copy = installer._write_copy

    def record_runner(command):
        events.append(("command", *tuple(command)))
        return runner(command)

    def record_copy(source, destination, *, executable):
        events.append(("copy", str(destination)))
        return original_write_copy(source, destination, executable=executable)

    monkeypatch.setattr(installer, "_write_copy", record_copy)

    installer.install(
        paths,
        runner=record_runner,
        environment={installer.TOPIC_ENV: "test-private-topic"},
    )

    stop_index = events.index(
        ("command", "systemctl", "--user", "stop", installer.SERVICE_NAME)
    )
    first_copy_index = next(index for index, event in enumerate(events) if event[0] == "copy")
    proof_index = events.index(
        (
            "command",
            "systemctl",
            "--user",
            "start",
            "--wait",
            installer.SERVICE_NAME,
        )
    )

    assert stop_index < first_copy_index < proof_index


def test_topic_file_is_private_before_atomic_publish(tmp_path, monkeypatch):
    paths = _install_paths(tmp_path)
    destination = paths.notification_env
    observed_modes: list[int] = []
    real_replace = installer.os.replace

    def inspect_replace(source, target):
        if Path(target) == destination:
            observed_modes.append(Path(source).stat().st_mode & 0o777)
        return real_replace(source, target)

    monkeypatch.setattr(installer.os, "replace", inspect_replace)

    plan = installer._notification_topic_plan(
        paths, {installer.TOPIC_ENV: "test-private-topic"}
    )
    installer.ensure_notification_topic(paths, plan)

    assert observed_modes == [0o600]
    assert destination.stat().st_mode & 0o777 == 0o600


def test_file_restore_preserves_a_falsy_zero_mode(tmp_path):
    destination = tmp_path / "restored"

    installer._restore_file(
        installer.FileSnapshot(path=destination, payload=b"previous", mode=0)
    )

    assert destination.stat().st_mode & 0o777 == 0
    destination.chmod(0o600)
    assert destination.read_bytes() == b"previous"


@pytest.mark.parametrize(
    ("timer_runtime", "restore_enable_command"),
    [
        (False, ("systemctl", "--user", "enable", installer.TIMER_NAME)),
        (
            True,
            ("systemctl", "--user", "enable", "--runtime", installer.TIMER_NAME),
        ),
    ],
)
def test_installer_restores_files_and_timer_when_service_proof_fails(
    tmp_path, timer_runtime, restore_enable_command
):
    paths = _install_paths(tmp_path)
    installed_files = [
        paths.installed_monitor,
        paths.systemd_dir / installer.SERVICE_NAME,
        paths.systemd_dir / installer.TIMER_NAME,
        paths.notification_env,
    ]
    for index, path in enumerate(installed_files):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"previous-{index}".encode("utf-8"))
        path.chmod(0o600 if path == paths.notification_env else 0o644)
    paths.notification_env.write_text(
        f"{installer.TOPIC_ENV}=previous-private-topic\n", encoding="utf-8"
    )
    previous_payloads = {path: path.read_bytes() for path in installed_files}
    runner = _InstallerRunner(
        failing_command=("systemctl", "--user", "start", "--wait", installer.SERVICE_NAME),
        timer_enabled=not timer_runtime,
        timer_runtime=timer_runtime,
        timer_active=True,
    )

    with pytest.raises(RuntimeError, match="initial installed-monitor invocation failed"):
        installer.install(paths, runner=runner, environment={})

    assert {path: path.read_bytes() for path in installed_files} == previous_payloads
    assert runner.timer_enabled
    assert runner.timer_active
    assert ("systemctl", "--user", "enable", "--now", installer.TIMER_NAME) not in runner.commands
    assert runner.commands[-3:] == [
        ("systemctl", "--user", "daemon-reload"),
        restore_enable_command,
        ("systemctl", "--user", "start", installer.TIMER_NAME),
    ]


@pytest.mark.parametrize("target_exists", [True, False])
def test_installer_failure_restores_live_and_broken_relative_symlinks(
    tmp_path, target_exists
):
    paths = _install_paths(tmp_path)
    paths.legacy_monitor.parent.mkdir(parents=True, exist_ok=True)
    paths.legacy_monitor.write_text(
        'NTFY_TOPIC="${ATLAS_HC_NTFY_TOPIC:-test-private-topic}"\n', encoding="utf-8"
    )
    paths.systemd_dir.mkdir(parents=True, exist_ok=True)
    target = paths.systemd_dir / "managed-healthcheck.service"
    original_target_payload = b"managed-service"
    if target_exists:
        target.write_bytes(original_target_payload)
    destination = paths.systemd_dir / installer.SERVICE_NAME
    destination.symlink_to(target.name)
    original_link_target = os.readlink(destination)
    runner = _InstallerRunner(
        failing_command=("systemctl", "--user", "start", "--wait", installer.SERVICE_NAME)
    )

    with pytest.raises(RuntimeError, match="initial installed-monitor invocation failed"):
        installer.install(paths, runner=runner, environment={})

    assert destination.is_symlink()
    assert os.readlink(destination) == original_link_target
    if target_exists:
        assert target.read_bytes() == original_target_payload
    else:
        assert not destination.exists()


def test_installer_rejects_nonprivate_notification_symlink_before_systemd_or_mutation(
    tmp_path,
):
    paths = _install_paths(tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    target = paths.config_dir / "managed-healthcheck.env"
    original_payload = f"{installer.TOPIC_ENV}=test-private-topic\n"
    target.write_text(original_payload, encoding="utf-8")
    target.chmod(0o640)
    paths.notification_env.symlink_to(target.name)
    runner = _InstallerRunner(timer_active=True)

    with pytest.raises(RuntimeError, match="symlink target is not private"):
        installer.install(paths, runner=runner, environment={})

    assert runner.commands == []
    assert paths.notification_env.is_symlink()
    assert os.readlink(paths.notification_env) == target.name
    assert target.read_text(encoding="utf-8") == original_payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.parametrize("proof_fails", [False, True])
def test_installer_preserves_private_notification_symlink_without_chmod(
    tmp_path, proof_fails
):
    paths = _install_paths(tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    target = paths.config_dir / "managed-healthcheck.env"
    original_payload = f"{installer.TOPIC_ENV}=test-private-topic\n"
    target.write_text(original_payload, encoding="utf-8")
    target.chmod(0o600)
    paths.notification_env.symlink_to(target.name)
    failing_command = (
        ("systemctl", "--user", "start", "--wait", installer.SERVICE_NAME)
        if proof_fails
        else None
    )
    runner = _InstallerRunner(failing_command=failing_command)

    if proof_fails:
        with pytest.raises(RuntimeError, match="initial installed-monitor invocation failed"):
            installer.install(paths, runner=runner, environment={})
    else:
        installer.install(paths, runner=runner, environment={})

    assert paths.notification_env.is_symlink()
    assert os.readlink(paths.notification_env) == target.name
    assert target.read_text(encoding="utf-8") == original_payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_installer_rejects_private_missing_topic_symlink_before_systemd_or_mutation(
    tmp_path,
):
    paths = _install_paths(tmp_path)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    target = paths.config_dir / "managed-healthcheck.env"
    original_payload = "UNRELATED=value\n"
    target.write_text(original_payload, encoding="utf-8")
    target.chmod(0o600)
    paths.notification_env.symlink_to(target.name)
    runner = _InstallerRunner(timer_active=True)

    with pytest.raises(RuntimeError, match="symlink must already contain"):
        installer.install(
            paths,
            runner=runner,
            environment={installer.TOPIC_ENV: "test-private-topic"},
        )

    assert paths.notification_env.is_symlink()
    assert os.readlink(paths.notification_env) == target.name
    assert target.read_text(encoding="utf-8") == original_payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert runner.commands == []


def test_installer_rolls_back_before_reraising_operator_cancellation(tmp_path):
    paths = _install_paths(tmp_path)
    paths.installed_monitor.parent.mkdir(parents=True, exist_ok=True)
    paths.installed_monitor.write_bytes(b"previous-monitor")
    paths.installed_monitor.chmod(0o700)
    paths.notification_env.parent.mkdir(parents=True, exist_ok=True)
    paths.notification_env.write_text(
        f"{installer.TOPIC_ENV}=previous-private-topic\n", encoding="utf-8"
    )
    paths.notification_env.chmod(0o600)
    runner = _InstallerRunner(timer_enabled=True, timer_active=True)
    proof_command = (
        "systemctl",
        "--user",
        "start",
        "--wait",
        installer.SERVICE_NAME,
    )

    def interrupting_runner(command):
        if tuple(command) == proof_command:
            runner.commands.append(proof_command)
            raise KeyboardInterrupt
        return runner(command)

    with pytest.raises(KeyboardInterrupt):
        installer.install(paths, runner=interrupting_runner, environment={})

    assert paths.installed_monitor.read_bytes() == b"previous-monitor"
    assert not (paths.systemd_dir / installer.SERVICE_NAME).exists()
    assert not (paths.systemd_dir / installer.TIMER_NAME).exists()
    assert paths.notification_env.read_text(encoding="utf-8") == (
        f"{installer.TOPIC_ENV}=previous-private-topic\n"
    )
    assert runner.timer_enabled
    assert runner.timer_active
    assert runner.commands[-3:] == [
        ("systemctl", "--user", "daemon-reload"),
        ("systemctl", "--user", "enable", installer.TIMER_NAME),
        ("systemctl", "--user", "start", installer.TIMER_NAME),
    ]


def test_installer_removes_partial_enrollment_when_timer_enable_fails(tmp_path):
    paths = _install_paths(tmp_path)
    runner = _InstallerRunner(
        failing_command=("systemctl", "--user", "enable", "--now", installer.TIMER_NAME)
    )

    with pytest.raises(RuntimeError, match="timer enable failed"):
        installer.install(
            paths,
            runner=runner,
            environment={installer.TOPIC_ENV: "test-private-topic"},
        )

    assert not runner.timer_enabled
    assert not runner.timer_active
    assert not paths.installed_monitor.exists()
    assert not paths.notification_env.exists()
    assert ("systemctl", "--user", "disable", "--now", installer.TIMER_NAME) in runner.commands


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
