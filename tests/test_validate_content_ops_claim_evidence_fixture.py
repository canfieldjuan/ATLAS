from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_content_ops_claim_evidence_fixture.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "validate_content_ops_claim_evidence_fixture",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    triple_id: str,
    expected: bool = True,
    difficulty: str = "easy",
) -> dict[str, object]:
    return {
        "triple_id": triple_id,
        "claim_id": f"claim-{triple_id}",
        "evidence_quote": f"evidence {triple_id}",
        "source_id": f"source-{triple_id}",
        "expected_supports": expected,
        "difficulty": difficulty,
    }


def test_cli_validates_json_fixture_file(tmp_path: Path) -> None:
    cli = _load_cli_module()
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps([_row("json-1")]), encoding="utf-8")

    exit_code, payload = cli.validate_fixture_file(path)

    assert exit_code == 0
    assert payload == {
        "ok": True,
        "source_format": "json",
        "errors": [],
        "triple_count": 1,
        "easy_supports_count": 1,
        "easy_not_supports_count": 0,
        "hard_count": 0,
    }


def test_cli_validates_explicit_jsonl_fixture_file(tmp_path: Path) -> None:
    cli = _load_cli_module()
    path = tmp_path / "fixture.txt"
    path.write_text(
        "\n".join(
            [
                json.dumps(_row("jsonl-1", True, "easy")),
                json.dumps(_row("jsonl-2", False, "hard")),
            ]
        ),
        encoding="utf-8",
    )

    exit_code, payload = cli.validate_fixture_file(path, requested_format="jsonl")

    assert exit_code == 0
    assert payload["source_format"] == "jsonl"
    assert payload["triple_count"] == 2
    assert payload["easy_supports_count"] == 1
    assert payload["hard_count"] == 1


def test_cli_rejects_missing_fixture_file(tmp_path: Path) -> None:
    cli = _load_cli_module()
    path = tmp_path / "missing.json"

    exit_code, payload = cli.validate_fixture_file(path)

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["source_format"] == "json"
    assert payload["errors"] == [f"fixture file not found: {path}"]


def test_cli_rejects_unknown_suffix_in_auto_mode(tmp_path: Path) -> None:
    cli = _load_cli_module()
    path = tmp_path / "fixture.txt"
    path.write_text("[]", encoding="utf-8")

    exit_code, payload = cli.validate_fixture_file(path)

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["source_format"] is None
    assert payload["errors"] == [
        "fixture format auto-detection requires .json or .jsonl suffix"
    ]


def test_cli_returns_nonzero_for_loader_errors(tmp_path: Path) -> None:
    cli = _load_cli_module()
    path = tmp_path / "fixture.jsonl"
    path.write_text("[{}]", encoding="utf-8")

    exit_code, payload = cli.validate_fixture_file(path)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["source_format"] == "jsonl"
    assert payload["errors"] == ["line 1: jsonl fixture line must be an object"]


def test_cli_delegates_final_shape_validation(tmp_path: Path) -> None:
    cli = _load_cli_module()
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps([_row("json-1")]), encoding="utf-8")

    exit_code, payload = cli.validate_fixture_file(path, require_final_shape=True)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["triple_count"] == 1
    assert payload["errors"] == [
        "final fixture requires 15 easy support rows; got 1",
        "final fixture requires 15 easy non-support rows; got 0",
        "final fixture requires 10 hard rows; got 0",
    ]


def test_direct_script_invocation_prints_json_envelope(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps([_row("json-1")]), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert result.returncode == 0
    assert result.stderr == ""
    assert output["ok"] is True
    assert output["triple_count"] == 1
