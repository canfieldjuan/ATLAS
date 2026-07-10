from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_codex_wake_bridge.py"
SPEC = importlib.util.spec_from_file_location("install_codex_wake_bridge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_install_writes_wrapper_and_systemd_dropin(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    systemd_dir = tmp_path / "systemd"

    result = _run("--bin-dir", str(bin_dir), "--systemd-dir", str(systemd_dir))

    assert result.returncode == 0, result.stdout + result.stderr
    wrapper = bin_dir / "atlas-pr-watch-and-wake"
    bridge = bin_dir / "atlas-codex-wake-bridge"
    watcher = bin_dir / "atlas-pr-watch"
    dropin = systemd_dir / "atlas-pr-watch@.service.d" / "wake-bridge.conf"
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert wrapper_text == installer._wrapper_text(watcher, bridge)
    assert bridge.read_text(encoding="utf-8") == (ROOT / "scripts" / "codex_wake_bridge.py").read_text(encoding="utf-8")
    assert watcher.read_text(encoding="utf-8") == (ROOT / "scripts" / "pr_watcher.py").read_text(encoding="utf-8")
    assert dropin.read_text(encoding="utf-8") == installer._dropin_text(wrapper)
    assert os.access(wrapper, os.X_OK)
    assert os.access(bridge, os.X_OK)
    assert os.access(watcher, os.X_OK)
    assert "invalid watcher id" in wrapper_text
    assert str(watcher) in wrapper_text
    assert str(bridge) in wrapper_text
    assert "scripts/codex_wake_bridge.py" not in wrapper_text
    assert 'cd "$REPO_DIR"' not in wrapper_text
    assert "gh pr merge" not in wrapper_text
    assert "--delete-branch" not in wrapper_text
    assert f'ExecStart="{wrapper}" %i' in dropin.read_text(encoding="utf-8")


def test_check_mode_fails_when_dropin_points_at_default_wrapper_with_custom_bin_dir(tmp_path: Path) -> None:
    bin_dir = tmp_path / "custom-bin"
    systemd_dir = tmp_path / "systemd"
    wrapper = bin_dir / "atlas-pr-watch-and-wake"
    bridge = bin_dir / "atlas-codex-wake-bridge"
    watcher = bin_dir / "atlas-pr-watch"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(installer._wrapper_text(watcher, bridge), encoding="utf-8")
    bridge.write_text(installer._bridge_text(), encoding="utf-8")
    watcher.write_text(installer._watcher_text(), encoding="utf-8")
    bridge.chmod(bridge.stat().st_mode | stat.S_IXUSR)
    watcher.chmod(watcher.stat().st_mode | stat.S_IXUSR)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    dropin = systemd_dir / "atlas-pr-watch@.service.d" / "wake-bridge.conf"
    dropin.parent.mkdir(parents=True)
    dropin.write_text(
        "[Service]\nExecStart=\nExecStart=%h/.local/bin/atlas-pr-watch-and-wake %i\n",
        encoding="utf-8",
    )

    result = _run(
        "--check",
        "--bin-dir",
        str(bin_dir),
        "--systemd-dir",
        str(systemd_dir),
    )

    assert result.returncode == 1
    assert "content drift" in result.stdout


def test_dropin_uses_absolute_quoted_wrapper_for_relative_bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    bin_dir = Path("relative bin")
    systemd_dir = tmp_path / "systemd"

    result = _run("--bin-dir", str(bin_dir), "--systemd-dir", str(systemd_dir))

    assert result.returncode == 0, result.stdout + result.stderr
    wrapper = (tmp_path / "relative bin" / "atlas-pr-watch-and-wake").resolve(strict=False)
    bridge = (tmp_path / "relative bin" / "atlas-codex-wake-bridge").resolve(strict=False)
    watcher = (tmp_path / "relative bin" / "atlas-pr-watch").resolve(strict=False)
    dropin = systemd_dir / "atlas-pr-watch@.service.d" / "wake-bridge.conf"
    assert wrapper.read_text(encoding="utf-8") == installer._wrapper_text(watcher, bridge)
    assert dropin.read_text(encoding="utf-8") == installer._dropin_text(wrapper)
    assert f'ExecStart="{wrapper}" %i' in dropin.read_text(encoding="utf-8")


def test_dropin_with_space_path_is_valid_systemd_execstart(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin with spaces"
    systemd_dir = tmp_path / "systemd"
    result = _run("--bin-dir", str(bin_dir), "--systemd-dir", str(systemd_dir))
    assert result.returncode == 0, result.stdout + result.stderr
    dropin = systemd_dir / "atlas-pr-watch@.service.d" / "wake-bridge.conf"
    text = dropin.read_text(encoding="utf-8")
    assert f'ExecStart="{bin_dir / "atlas-pr-watch-and-wake"}" %i' in text

    systemd_analyze = shutil.which("systemd-analyze")
    if systemd_analyze is None:
        return
    unit = tmp_path / "atlas-pr-watch@example.service"
    unit.write_text(text, encoding="utf-8")
    verify = subprocess.run(
        [systemd_analyze, "verify", str(unit)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_check_mode_passes_after_install(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    systemd_dir = tmp_path / "systemd"
    assert _run("--bin-dir", str(bin_dir), "--systemd-dir", str(systemd_dir)).returncode == 0

    result = _run(
        "--check",
        "--bin-dir",
        str(bin_dir),
        "--systemd-dir",
        str(systemd_dir),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok:" in result.stdout


def test_check_mode_fails_when_installed_watcher_drifts(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    systemd_dir = tmp_path / "systemd"
    assert _run("--bin-dir", str(bin_dir), "--systemd-dir", str(systemd_dir)).returncode == 0
    (bin_dir / "atlas-pr-watch").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    result = _run(
        "--check",
        "--bin-dir",
        str(bin_dir),
        "--systemd-dir",
        str(systemd_dir),
    )

    assert result.returncode == 1
    assert f"content drift: {bin_dir / 'atlas-pr-watch'}" in result.stdout


def test_check_mode_fails_when_systemd_still_calls_bare_watcher(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    systemd_dir = tmp_path / "systemd"
    wrapper = bin_dir / "atlas-pr-watch-and-wake"
    bridge = bin_dir / "atlas-codex-wake-bridge"
    watcher = bin_dir / "atlas-pr-watch"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(installer._wrapper_text(watcher, bridge), encoding="utf-8")
    bridge.write_text(installer._bridge_text(), encoding="utf-8")
    watcher.write_text(installer._watcher_text(), encoding="utf-8")
    bridge.chmod(bridge.stat().st_mode | stat.S_IXUSR)
    watcher.chmod(watcher.stat().st_mode | stat.S_IXUSR)
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    dropin = systemd_dir / "atlas-pr-watch@.service.d" / "wake-bridge.conf"
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Service]\nExecStart=%h/.local/bin/atlas-pr-watch %i\n", encoding="utf-8")

    result = _run(
        "--check",
        "--bin-dir",
        str(bin_dir),
        "--systemd-dir",
        str(systemd_dir),
    )

    assert result.returncode == 1
    assert "content drift" in result.stdout


def test_check_mode_fails_when_wrapper_is_not_executable(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    systemd_dir = tmp_path / "systemd"
    wrapper = bin_dir / "atlas-pr-watch-and-wake"
    bridge = bin_dir / "atlas-codex-wake-bridge"
    watcher = bin_dir / "atlas-pr-watch"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(installer._wrapper_text(watcher, bridge), encoding="utf-8")
    bridge.write_text(installer._bridge_text(), encoding="utf-8")
    watcher.write_text(installer._watcher_text(), encoding="utf-8")
    bridge.chmod(bridge.stat().st_mode | stat.S_IXUSR)
    watcher.chmod(watcher.stat().st_mode | stat.S_IXUSR)
    dropin = systemd_dir / "atlas-pr-watch@.service.d" / "wake-bridge.conf"
    dropin.parent.mkdir(parents=True)
    dropin.write_text(installer._dropin_text(wrapper), encoding="utf-8")

    result = _run(
        "--check",
        "--bin-dir",
        str(bin_dir),
        "--systemd-dir",
        str(systemd_dir),
    )

    assert result.returncode == 1
    assert "not executable" in result.stdout


def test_main_rejects_unknown_argument() -> None:
    with pytest.raises(SystemExit):
        installer.main(["--definitely-not-a-real-installer-flag"])


def test_reload_systemd_failure_returns_nonzero(tmp_path: Path) -> None:
    failure = subprocess.CompletedProcess(
        ["systemctl", "--user", "daemon-reload"],
        1,
        stdout="",
        stderr="no user bus",
    )

    with mock.patch.object(installer.subprocess, "run", return_value=failure):
        code, messages = installer.install(
            tmp_path / "bin",
            tmp_path / "systemd",
            reload_systemd=True,
        )

    assert code == 1
    assert "systemctl --user daemon-reload exited 1" in messages
    assert "no user bus" in messages
