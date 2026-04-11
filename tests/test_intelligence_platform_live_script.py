import os
import subprocess
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "test_intelligence_platform_live.sh"


def _script_env():
    env = os.environ.copy()
    for key in (
        "ATLAS_TOKEN",
        "ATLAS_VENDOR",
        "ATLAS_COMPARISON_VENDOR",
        "ATLAS_REVIEW_ID",
        "ATLAS_SEQUENCE_ID",
        "ATLAS_REPORT_ID",
        "ATLAS_CORRECTION_ID",
    ):
        env.pop(key, None)
    return env


def test_live_platform_script_requires_token():
    result = subprocess.run(
        ["bash", str(_SCRIPT_PATH), "1"],
        capture_output=True,
        text=True,
        env=_script_env(),
        check=False,
    )

    assert result.returncode == 2
    assert "ATLAS_TOKEN or a second positional bearer token argument is required" in result.stdout


def test_live_platform_script_uses_env_token_before_vendor_discovery():
    env = _script_env()
    env["ATLAS_TOKEN"] = "token-123"

    result = subprocess.run(
        ["bash", str(_SCRIPT_PATH), "1"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "Unable to discover a tracked vendor. Set ATLAS_VENDOR and rerun." in result.stdout
    assert "ATLAS_TOKEN or a second positional bearer token argument is required" not in result.stdout


def test_live_platform_script_accepts_positional_token():
    result = subprocess.run(
        ["bash", str(_SCRIPT_PATH), "1", "token-123"],
        capture_output=True,
        text=True,
        env=_script_env(),
        check=False,
    )

    assert result.returncode == 2
    assert "Unable to discover a tracked vendor. Set ATLAS_VENDOR and rerun." in result.stdout
    assert "ATLAS_TOKEN or a second positional bearer token argument is required" not in result.stdout
