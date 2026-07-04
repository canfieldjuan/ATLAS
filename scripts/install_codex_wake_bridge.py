#!/usr/bin/env python3
"""Install or verify the local Codex watcher wake wrapper.

This writes local user files only. The wrapper keeps the watcher read-only:
it runs the one-shot watcher, then asks the repo bridge to build/run the Codex
handoff for scheduled wakes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import subprocess
from typing import Sequence


WRAPPER_NAME = "atlas-pr-watch-and-wake"
BRIDGE_NAME = "atlas-codex-wake-bridge"
DROPIN_REL = Path("atlas-pr-watch@.service.d") / "wake-bridge.conf"
BRIDGE_SOURCE = Path(__file__).with_name("codex_wake_bridge.py")


def _shell_token(path: Path) -> str:
    escaped = str(path).replace("'", "'\"'\"'")
    return f"'{escaped}'"


def _wrapper_text(bridge: Path) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
session_id="${{1:?watcher session id required}}"
bridge={_shell_token(bridge)}

if [[ ! "$session_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || [[ "$session_id" == *..* ]]; then
  echo "invalid watcher id: $session_id" >&2
  exit 2
fi

config="${{HOME}}/.config/atlas-pr-watchers/${{session_id}}.env"

if [ ! -f "$config" ]; then
  echo "watcher config not found: $config" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$config"

if [ -z "${{REPO_DIR:-}}" ] || [ ! -d "$REPO_DIR" ]; then
  echo "invalid REPO_DIR for ${{session_id}}: ${{REPO_DIR:-}}" >&2
  exit 2
fi

~/.local/bin/atlas-pr-watch "${{session_id}}"
python "$bridge" "${{session_id}}" --source scheduled
"""


def _bridge_text() -> str:
    return BRIDGE_SOURCE.read_text(encoding="utf-8")

def _systemd_exec_token(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    escaped = (
        str(resolved)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
    )
    return f'"{escaped}"'


def _dropin_text(wrapper: Path) -> str:
    return f"""[Service]
ExecStart=
ExecStart={_systemd_exec_token(wrapper)} %i
"""


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _matches(path: Path, expected: str, *, executable: bool = False) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path}"
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"unreadable {path}: {exc}"
    if actual != expected:
        return False, f"content drift: {path}"
    if executable and not os.access(path, os.X_OK):
        return False, f"not executable: {path}"
    return True, f"ok: {path}"


def check_install(bin_dir: Path, systemd_dir: Path) -> tuple[bool, list[str]]:
    wrapper = bin_dir / WRAPPER_NAME
    bridge = bin_dir / BRIDGE_NAME
    checks = [
        _matches(wrapper, _wrapper_text(bridge), executable=True),
        _matches(bridge, _bridge_text(), executable=True),
        _matches(systemd_dir / DROPIN_REL, _dropin_text(wrapper)),
    ]
    ok = all(item[0] for item in checks)
    return ok, [item[1] for item in checks]


def install(bin_dir: Path, systemd_dir: Path, *, reload_systemd: bool) -> tuple[int, list[str]]:
    wrapper = bin_dir / WRAPPER_NAME
    bridge = bin_dir / BRIDGE_NAME
    dropin = systemd_dir / DROPIN_REL
    _write(wrapper, _wrapper_text(bridge), executable=True)
    _write(bridge, _bridge_text(), executable=True)
    _write(dropin, _dropin_text(wrapper))
    messages = [f"wrote: {wrapper}", f"wrote: {bridge}", f"wrote: {dropin}"]
    exit_code = 0
    if reload_systemd:
        proc = subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        messages.append(f"systemctl --user daemon-reload exited {proc.returncode}")
        exit_code = proc.returncode
        if proc.stdout.strip():
            messages.append(proc.stdout.strip())
        if proc.stderr.strip():
            messages.append(proc.stderr.strip())
    return exit_code, messages


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local" / "bin")
    parser.add_argument(
        "--systemd-dir",
        type=Path,
        default=Path.home() / ".config" / "systemd" / "user",
    )
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument(
        "--reload-systemd",
        action="store_true",
        help="run systemctl --user daemon-reload after installing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bin_dir = args.bin_dir.expanduser().resolve(strict=False)
    systemd_dir = args.systemd_dir.expanduser().resolve(strict=False)
    if args.check:
        ok, messages = check_install(bin_dir, systemd_dir)
        for message in messages:
            print(message)
        return 0 if ok else 1
    exit_code, messages = install(bin_dir, systemd_dir, reload_systemd=args.reload_systemd)
    for message in messages:
        print(message)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
