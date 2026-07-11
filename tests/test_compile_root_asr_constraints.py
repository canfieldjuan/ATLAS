from __future__ import annotations

import hashlib
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compile_root_asr_constraints",
    ROOT / "scripts/compile_root_asr_constraints.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONSTRAINTS_NAME = MODULE.CONSTRAINTS_NAME
DIGEST_PREFIX = MODULE.DIGEST_PREFIX
merge_resolutions = MODULE.merge_resolutions
parse_resolution = MODULE.parse_resolution


NEMO_SHA = "0f378e9d8dd72630c911025b555f18658d44cc8f"


def test_merge_resolutions_marks_only_interpreter_divergence() -> None:
    py310 = parse_resolution(
        """
        common==1.0
        older==2.0
        only-310==3.0
        direct @ git+https://example.test/repo.git@abc123
        """
    )
    py311 = parse_resolution(
        """
        common==1.0
        older==2.1
        only-311==4.0
        direct @ git+https://example.test/repo.git@abc123
        """
    )

    assert merge_resolutions(py310, py311) == [
        "common==1.0",
        "direct @ git+https://example.test/repo.git@abc123",
        "older==2.0 ; python_version < '3.11'",
        "older==2.1 ; python_version >= '3.11'",
        "only-310==3.0 ; python_version < '3.11'",
        "only-311==4.0 ; python_version >= '3.11'",
    ]


@pytest.mark.parametrize(
    "bad_line",
    [
        "floating>=1.0",
        "floating",
        "name==1.0 ; python_version >= '3.11'",
        "--extra-index-url https://example.test/simple",
    ],
)
def test_parse_resolution_rejects_non_concrete_or_pre_marked_cells(
    bad_line: str,
) -> None:
    with pytest.raises(ValueError, match="concrete resolver output"):
        parse_resolution(bad_line)


def test_real_lock_is_consumed_and_digest_bound() -> None:
    constraints = ROOT / CONSTRAINTS_NAME
    lock_bytes = constraints.read_bytes()
    digest = hashlib.sha256(lock_bytes).hexdigest()
    root_lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    asr_lines = (
        ROOT / "requirements.asr.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert f"-c {CONSTRAINTS_NAME}" in root_lines
    assert f"-c {CONSTRAINTS_NAME}" in asr_lines
    assert f"{DIGEST_PREFIX}{digest}" in root_lines
    assert sum(line.startswith(DIGEST_PREFIX) for line in root_lines) == 1

    lock_text = lock_bytes.decode("utf-8")
    assert NEMO_SHA in lock_text
    assert not re.search(r"(?m)^[A-Za-z0-9_.-]+(?:\[[^]]+\])?$", lock_text)


def test_real_lock_marker_cells_are_canonical_and_non_overlapping() -> None:
    cells: dict[str, list[str]] = defaultdict(list)
    lock_lines = (ROOT / CONSTRAINTS_NAME).read_text(encoding="utf-8").splitlines()
    for raw_line in lock_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement, separator, marker = line.partition(" ; ")
        cells[MODULE._normalized_name(requirement)].append(marker if separator else "")

    for name, markers in cells.items():
        assert markers in (
            [""],
            ["python_version < '3.11'"],
            ["python_version >= '3.11'"],
            ["python_version < '3.11'", "python_version >= '3.11'"],
        ), f"non-canonical or overlapping marker cells for {name}: {markers}"


def test_constraints_regeneration_is_enrolled_in_ci() -> None:
    workflow = (
        ROOT / ".github/workflows/python_constraints_checks.yml"
    ).read_text(encoding="utf-8")

    assert 'uv==0.10.10 pytest==9.1.1' in workflow
    assert "python scripts/compile_root_asr_constraints.py --check" in workflow
    assert (
        "python -m pytest tests/test_compile_root_asr_constraints.py -q" in workflow
    )
    assert workflow.count('"constraints.root-asr.txt"') == 2


def test_root_docker_image_copies_constraints_before_install() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    copy_index = dockerfile.index(
        "COPY requirements.txt constraints.root-asr.txt ./"
    )
    install_index = dockerfile.index("pip install --no-cache-dir -r requirements.txt")
    assert copy_index < install_index
