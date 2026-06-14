from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_content_ops_claim_evidence_prompt_packets.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "export_content_ops_claim_evidence_prompt_packets",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(triple_id: str, expected: bool, difficulty: str) -> dict[str, object]:
    return {
        "triple_id": triple_id,
        "claim_id": f"claim-{triple_id}",
        "claim_text": f"Claim statement {triple_id}",
        "evidence_quote": f"evidence {triple_id}",
        "source_id": f"source-{triple_id}",
        "expected_supports": expected,
        "difficulty": difficulty,
    }


def _write_fixture(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                _row("easy", True, "easy"),
                _row("hard", False, "hard"),
            ]
        ),
        encoding="utf-8",
    )


def test_prompt_packet_cli_writes_json_packets_without_labels(tmp_path: Path) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "packets" / "claim_evidence_prompts.json"
    _write_fixture(fixture_path)

    exit_code, payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=("claude-sonnet", "gpt"),
    )

    packets = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload == {
        "ok": True,
        "output_path": str(output_path),
        "output_format": "json",
        "packet_count": 4,
        "triple_count": 2,
        "model_count": 2,
        "model_ids": ["claude-sonnet", "gpt"],
        "errors": [],
    }
    assert [packet["model_id"] for packet in packets] == [
        "claude-sonnet",
        "claude-sonnet",
        "gpt",
        "gpt",
    ]
    assert packets[0]["triple_id"] == "easy"
    assert packets[0]["contract_version"] == "verify_claim_evidence.v1"
    assert "Claim: Claim statement easy" in packets[0]["prompt"]
    assert packets[0]["response_schema"]["additionalProperties"] is False
    assert "expected_supports" not in packets[0]
    assert "expected_supports" not in packets[0]["prompt"]


def test_prompt_packet_cli_writes_jsonl_packets(tmp_path: Path) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.jsonl"
    output_path = tmp_path / "packets.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(_row("easy", True, "easy")),
                json.dumps(_row("hard", False, "hard")),
            ]
        ),
        encoding="utf-8",
    )

    exit_code, payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=("claude-sonnet",),
    )

    packets = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 0
    assert payload["output_format"] == "jsonl"
    assert payload["packet_count"] == 2
    assert [packet["triple_id"] for packet in packets] == ["easy", "hard"]


def test_prompt_packet_cli_rejects_missing_and_duplicate_model_ids(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "packets.json"
    _write_fixture(fixture_path)

    exit_code, payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=(" ", "gpt", "gpt"),
    )

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["errors"] == [
        "model_id 1 missing",
        "model_id duplicated from position 2: gpt",
    ]
    assert output_path.exists() is False


def test_prompt_packet_cli_rejects_invalid_direct_call_options(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "packets.json"
    _write_fixture(fixture_path)

    fixture_exit, fixture_payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=("gpt",),
        fixture_format="csv",
    )
    output_exit, output_payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=("gpt",),
        output_format="xml",
    )
    model_exit, model_payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids="gpt",
    )

    assert fixture_exit == 2
    assert fixture_payload["errors"] == ["fixture format must be auto, json, or jsonl"]
    assert output_exit == 2
    assert output_payload["errors"] == ["output format must be auto, json, or jsonl"]
    assert model_exit == 2
    assert model_payload["errors"] == ["model_ids must be a sequence"]
    assert output_path.exists() is False


def test_prompt_packet_cli_rejects_invalid_fixture_without_writing(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "packets.json"
    fixture_path.write_text("{", encoding="utf-8")

    exit_code, payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=("gpt",),
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["errors"] == ["json fixture is malformed: Expecting property name enclosed in double quotes"]
    assert output_path.exists() is False


def test_prompt_packet_cli_rejects_non_utf8_fixture_as_io_error(
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "packets.json"
    fixture_path.write_bytes(b"\xff\xfe")

    exit_code, payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=("gpt",),
    )

    assert exit_code == 2
    assert payload["ok"] is False
    assert len(payload["errors"]) == 1
    assert payload["errors"][0].startswith(f"fixture file could not be read: {fixture_path}: ")
    assert "utf-8" in payload["errors"][0]
    assert output_path.exists() is False


def test_prompt_packet_cli_rejects_symlink_output(tmp_path: Path) -> None:
    cli = _load_cli_module()
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "packets.json"
    external_path = tmp_path / "external.json"
    _write_fixture(fixture_path)
    external_path.write_text("external stays intact", encoding="utf-8")
    output_path.symlink_to(external_path)

    exit_code, payload = cli.export_prompt_packets_from_fixture_file(
        fixture_path,
        output_path,
        model_ids=("gpt",),
    )

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["errors"] == [f"output path is a symlink: {output_path}"]
    assert external_path.read_text(encoding="utf-8") == "external stays intact"


def test_direct_script_invocation_prints_json_envelope(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "packets.json"
    _write_fixture(fixture_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(fixture_path),
            str(output_path),
            "--model-id",
            "gpt",
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert result.returncode == 0
    assert result.stderr == ""
    assert output["ok"] is True
    assert output["packet_count"] == 2
    assert output_path.exists()
