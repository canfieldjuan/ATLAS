import os
import subprocess
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "test_adapter_live.py"


def _script_env():
    env = os.environ.copy()
    for key in ("ATLAS_TOKEN", "ATLAS_VENDOR"):
        env.pop(key, None)
    return env


def test_adapter_live_cli_requires_token():
    result = subprocess.run(
        ["python3", str(_SCRIPT_PATH), "--base-url", "http://127.0.0.1:1"],
        capture_output=True,
        text=True,
        env=_script_env(),
        check=False,
    )

    assert result.returncode == 2
    assert "ATLAS_TOKEN or --token is required for tenant route validation" in result.stdout


def test_adapter_live_cli_requires_vendor_when_discovery_fails():
    env = _script_env()
    env["ATLAS_TOKEN"] = "token-123"

    result = subprocess.run(
        ["python3", str(_SCRIPT_PATH), "--base-url", "http://127.0.0.1:1"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "Unable to discover a tracked vendor. Provide --vendor or set ATLAS_VENDOR." in result.stdout


def test_adapter_live_cli_reports_unreachable_server_for_explicit_vendor():
    env = _script_env()
    env["ATLAS_TOKEN"] = "token-123"

    result = subprocess.run(
        [
            "python3",
            str(_SCRIPT_PATH),
            "--base-url",
            "http://127.0.0.1:1",
            "--vendor",
            "HubSpot",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR: Server not reachable at http://127.0.0.1:1" in result.stdout
