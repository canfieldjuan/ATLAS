from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fix_loop_trace_contract.py"

SPEC = importlib.util.spec_from_file_location("fix_loop_trace_contract", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


@pytest.mark.parametrize(
    "trace",
    [
        "review claim -> parser accepts x -> admission grammar",
        "症状 -> 根因",
    ],
)
def test_source_trace_accepts_filled_endpoint_chain(trace: str) -> None:
    assert contract.source_trace_is_valid(trace)


@pytest.mark.parametrize(
    "trace",
    [
        "TBD -> TBD",
        "TBD symptom -> TBD upstream source",
        "<symptom -> intermediate cause -> upstream source>",
        "review claim",
    ],
)
def test_source_trace_rejects_invalid_placeholder_or_incomplete_chain(trace: str) -> None:
    assert not contract.source_trace_is_valid(trace)


def test_parse_repo_path_tokens_normalizes_valid_repo_paths() -> None:
    assert contract.parse_repo_path_tokens(r"./scripts/parser.py, `tests\test_parser.py`") == {
        "scripts/parser.py",
        "tests/test_parser.py",
    }


def test_parse_repo_path_tokens_rejects_placeholders_and_traversal() -> None:
    assert contract.parse_repo_path_tokens("none, ../escape.py, `...`") == set()


def test_normalize_repo_path_relativizes_absolute_path(tmp_path: Path) -> None:
    source = tmp_path / "scripts" / "parser.py"

    assert contract.normalize_repo_path(str(source), tmp_path) == "scripts/parser.py"


def test_normalize_repo_path_raises_for_invalid_path_type(tmp_path: Path) -> None:
    with pytest.raises(AttributeError):
        contract.normalize_repo_path(None, tmp_path)
