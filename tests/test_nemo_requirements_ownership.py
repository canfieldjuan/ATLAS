from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NEMO_REQUIREMENT = (
    "nemo_toolkit[asr] @ "
    "git+https://github.com/NVIDIA/NeMo.git@"
    "0f378e9d8dd72630c911025b555f18658d44cc8f"
)


def _nemo_requirements(text: str) -> tuple[str, ...]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        name = re.split(r"[\s<>=!~;\[]", line, maxsplit=1)[0]
        if name.lower().replace("-", "_") == "nemo_toolkit":
            lines.append(line)
    return tuple(lines)


def _assert_nemo_ownership(root_requirements: str, asr_requirements: str) -> None:
    assert _nemo_requirements(root_requirements) == ()
    assert _nemo_requirements(asr_requirements) == (CANONICAL_NEMO_REQUIREMENT,)


def test_standalone_asr_requirements_solely_own_nemo() -> None:
    _assert_nemo_ownership(
        (ROOT / "requirements.txt").read_text(encoding="utf-8"),
        (ROOT / "requirements.asr.txt").read_text(encoding="utf-8"),
    )


@pytest.mark.parametrize(
    ("root_requirements", "asr_requirements"),
    [
        ("nemo_toolkit[asr]\n", CANONICAL_NEMO_REQUIREMENT),
        ("", ""),
        ("", "nemo_toolkit[asr]==2.4.0"),
        (
            "",
            "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main",
        ),
    ],
)
def test_nemo_ownership_contract_rejects_duplicate_or_moved_authority(
    root_requirements: str,
    asr_requirements: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_nemo_ownership(root_requirements, asr_requirements)


def test_full_security_sweep_jointly_audits_root_and_asr_requirements() -> None:
    workflow = (ROOT / ".github/workflows/security_full_sweep.yml").read_text(
        encoding="utf-8"
    )

    assert "pip-audit -r requirements.txt -r requirements.asr.txt" in workflow


def test_asr_server_directs_optional_dependency_setup_to_asr_requirements() -> None:
    entrypoint = (ROOT / "asr_server.py").read_text(encoding="utf-8")

    assert "pip install -r requirements.asr.txt" in entrypoint
