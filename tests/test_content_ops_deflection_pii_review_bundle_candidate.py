from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote_deflection_pii_review_bundle.py"
SCORE_SCRIPT = ROOT / "scripts" / "score_deflection_pii_recall.py"
TINY_FIXTURE = (
    ROOT
    / "docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json"
)


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CLI = _load_script(SCRIPT, "promote_deflection_pii_review_bundle")
SCORE_CLI = _load_script(SCORE_SCRIPT, "score_deflection_pii_recall_for_candidate_tests")


def _tiny_corpus() -> dict:
    return json.loads(TINY_FIXTURE.read_text(encoding="utf-8"))


def _refresh_counts(corpus: dict) -> dict:
    tickets = [
        ticket
        for ticket in corpus.get("tickets", [])
        if isinstance(ticket, dict)
    ]
    label_count = sum(
        len(ticket.get("labels", []))
        for ticket in tickets
        if isinstance(ticket.get("labels"), list)
    )
    must_survive_count = sum(
        len(ticket.get("must_survive", []))
        for ticket in tickets
        if isinstance(ticket.get("must_survive"), list)
    )
    corpus["summary"]["ticket_count"] = len(tickets)
    corpus["summary"]["label_count"] = label_count
    corpus["summary"]["must_survive_count"] = must_survive_count
    return corpus


def _one_ticket_corpus() -> dict:
    corpus = _tiny_corpus()
    corpus["tickets"] = corpus["tickets"][:1]
    return _refresh_counts(corpus)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ready_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME).write_text(
        TINY_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    corpus = _tiny_corpus()
    _write_json(
        bundle_dir / CLI.REVIEW_BUNDLE_MANIFEST_NAME,
        {
            "schema_version": CLI.REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "status": "ok",
            "files": {
                "source_intake_summary": {
                    "path": "source-intake-summary.json",
                    "present": True,
                    "ok": True,
                    "schema_version": "deflection_pii_source_intake_summary.v1",
                },
                "source_intake_markdown": {
                    "path": "source-intake-summary.md",
                    "present": True,
                },
                "surrogate_eval_corpus": {
                    "path": CLI.REVIEW_BUNDLE_ARTIFACT_NAME,
                    "present": True,
                    "schema_version": corpus["schema_version"],
                    "ticket_count": corpus["summary"]["ticket_count"],
                    "label_count": corpus["summary"]["label_count"],
                },
            },
        },
    )
    assert SCORE_CLI.main(["--review-bundle-dir", str(bundle_dir)]) == 0
    return bundle_dir


def _manifest(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / CLI.REVIEW_BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8"))


def _rewrite_manifest(bundle_dir: Path, manifest: dict) -> None:
    _write_json(bundle_dir / CLI.REVIEW_BUNDLE_MANIFEST_NAME, manifest)


def _score(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / CLI.REVIEW_BUNDLE_SCORE_NAME).read_text(encoding="utf-8"))


def _rewrite_score(bundle_dir: Path, score: dict) -> None:
    _write_json(bundle_dir / CLI.REVIEW_BUNDLE_SCORE_NAME, score)


def _rewrite_corpus(bundle_dir: Path, corpus: dict) -> None:
    _write_json(bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME, corpus)


def _error_codes(capsys: pytest.CaptureFixture[str]) -> list[str]:
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    return [error["code"] for error in payload["errors"]]


def test_promotes_ready_review_bundle_candidate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    output = tmp_path / "versioned" / "deflection-pii-surrogate-eval-corpus.json"

    assert CLI.main([str(bundle_dir), "--output", str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert output.read_text(encoding="utf-8") == (
        bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME
    ).read_text(encoding="utf-8")
    assert payload["schema_version"] == CLI.EXPORT_SCHEMA_VERSION
    assert payload["ticket_count"] == 3
    assert payload["label_count"] == 11
    assert payload["headline"] == {
        "deferred_open_set_name_leak_count": 1,
        "free_high_severity_gate_eligible_leak_count": 0,
        "free_high_severity_leak_count": 1,
    }


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("schema_version", "wrong", "manifest_schema_version_mismatch"),
        ("status", "blocked", "manifest_status_not_ok"),
        ("score_status", "failed", "manifest_score_status_not_ok"),
    ),
)
def test_rejects_manifest_not_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: str,
    code: str,
) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    manifest = _manifest(bundle_dir)
    manifest[field] = value
    _rewrite_manifest(bundle_dir, manifest)
    output = tmp_path / "candidate.json"

    assert CLI.main([str(bundle_dir), "--output", str(output)]) == 1

    assert code in _error_codes(capsys)
    assert not output.exists()


@pytest.mark.parametrize(
    ("file_key", "mutation", "code"),
    (
        ("surrogate_eval_corpus", {"present": False}, "manifest_file_not_present"),
        ("surrogate_eval_corpus", {"path": "../raw.json"}, "manifest_file_path_mismatch"),
        ("recall_score", {"present": False}, "manifest_file_not_present"),
        ("recall_score", {"path": "/tmp/score.json"}, "manifest_file_path_mismatch"),
    ),
)
def test_rejects_manifest_file_contract_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    file_key: str,
    mutation: dict,
    code: str,
) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    manifest = _manifest(bundle_dir)
    manifest["files"][file_key].update(mutation)
    _rewrite_manifest(bundle_dir, manifest)
    output = tmp_path / "candidate.json"

    assert CLI.main([str(bundle_dir), "--output", str(output)]) == 1

    assert code in _error_codes(capsys)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutate", "code"),
    (
        (lambda bundle_dir: (bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME).unlink(), "corpus_load_failed"),
        (
            lambda bundle_dir: _rewrite_corpus(
                bundle_dir,
                {**_tiny_corpus(), "schema_version": "wrong"},
            ),
            "corpus_schema_version_mismatch",
        ),
        (
            lambda bundle_dir: _rewrite_corpus(
                bundle_dir,
                {
                    **_tiny_corpus(),
                    "source": {**_tiny_corpus()["source"], "raw_source_persisted": True},
                },
            ),
            "corpus_not_surrogate_only",
        ),
        (
            lambda bundle_dir: _rewrite_corpus(
                bundle_dir,
                {**_tiny_corpus(), "tickets": []},
            ),
            "corpus_empty_tickets",
        ),
        (
            lambda bundle_dir: _rewrite_corpus(
                bundle_dir,
                {
                    **_tiny_corpus(),
                    "tickets": [
                        {
                            **_tiny_corpus()["tickets"][0],
                            "labels": [
                                {
                                    **_tiny_corpus()["tickets"][0]["labels"][0],
                                    "span": "",
                                },
                            ],
                        },
                    ],
                },
            ),
            "label_missing_span",
        ),
    ),
)
def test_rejects_invalid_corpus_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutate,
    code: str,
) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    mutate(bundle_dir)
    output = tmp_path / "candidate.json"

    assert CLI.main([str(bundle_dir), "--output", str(output)]) == 1

    assert code in _error_codes(capsys)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutate", "code"),
    (
        (lambda bundle_dir: (bundle_dir / CLI.REVIEW_BUNDLE_SCORE_NAME).unlink(), "score_load_failed"),
        (
            lambda bundle_dir: _rewrite_score(
                bundle_dir,
                {**_score(bundle_dir), "schema_version": "wrong"},
            ),
            "score_schema_version_mismatch",
        ),
        (
            lambda bundle_dir: _rewrite_score(
                bundle_dir,
                {**_score(bundle_dir), "status": "failed"},
            ),
            "score_status_not_ok",
        ),
        (
            lambda bundle_dir: _rewrite_score(
                bundle_dir,
                {
                    key: value
                    for key, value in _score(bundle_dir).items()
                    if key != "headline"
                },
            ),
            "score_headline_missing",
        ),
        (
            lambda bundle_dir: _rewrite_score(
                bundle_dir,
                {
                    **_score(bundle_dir),
                    "headline": {
                        **_score(bundle_dir)["headline"],
                        "free_high_severity_gate_eligible_leak_count": "0",
                    },
                },
            ),
            "score_headline_metric_not_integer",
        ),
    ),
)
def test_rejects_invalid_score_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutate,
    code: str,
) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    mutate(bundle_dir)
    output = tmp_path / "candidate.json"

    assert CLI.main([str(bundle_dir), "--output", str(output)]) == 1

    assert code in _error_codes(capsys)
    assert not output.exists()


def test_rejects_score_input_and_manifest_counts_that_do_not_match_corpus(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    _rewrite_corpus(bundle_dir, _one_ticket_corpus())
    output = tmp_path / "candidate.json"

    assert CLI.main([str(bundle_dir), "--output", str(output)]) == 1

    codes = _error_codes(capsys)
    assert "manifest_corpus_count_mismatch" in codes
    assert "score_input_count_mismatch" in codes
    assert not output.exists()


def test_existing_output_requires_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    output = tmp_path / "candidate.json"
    output.write_text("keep me\n", encoding="utf-8")

    assert CLI.main([str(bundle_dir), "--output", str(output)]) == 1

    assert _error_codes(capsys) == ["output_exists"]
    assert output.read_text(encoding="utf-8") == "keep me\n"

    assert CLI.main([str(bundle_dir), "--output", str(output), "--force"]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "deflection_pii_eval_corpus.v1"


def test_force_cannot_overwrite_reserved_bundle_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    output = bundle_dir / CLI.REVIEW_BUNDLE_SCORE_NAME
    original_score = output.read_text(encoding="utf-8")

    assert CLI.main([str(bundle_dir), "--output", str(output), "--force"]) == 1

    assert _error_codes(capsys) == ["output_reserved_bundle_artifact"]
    assert output.read_text(encoding="utf-8") == original_score


def test_rejects_output_same_as_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_dir = _ready_bundle(tmp_path)
    output = bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME

    assert CLI.main([str(bundle_dir), "--output", str(output), "--force"]) == 1

    assert _error_codes(capsys) == ["output_same_as_source"]
