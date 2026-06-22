from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from extracted_content_pipeline.deflection_pii_eval_corpus import (
    SCHEMA_VERSION,
    SOURCE_INTAKE_SUMMARY_SCHEMA_VERSION,
    build_surrogate_eval_corpus,
    format_labeled_source_intake_summary_markdown,
    summarize_labeled_source,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_deflection_pii_surrogate_eval_corpus.py"
TINY_FIXTURE = (
    ROOT
    / "docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json"
)
SPEC = importlib.util.spec_from_file_location(
    "build_deflection_pii_surrogate_eval_corpus",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)


def _valid_source() -> dict:
    return {
        "schema_version": "deflection_pii_labeled_source.v1",
        "records": [
            {
                "fields": {
                    "subject": "Refund for Alice Baker",
                    "customer_message": (
                        "Email alice.baker@example.com or call 202-555-0188. "
                        "Order ORD-98765 ships to 99 Real Street."
                    ),
                    "agent_reply": (
                        "Customer Alice Baker can keep CVE-2021-44228 and ISO 27001 "
                        "references in the report. DOB 1977-06-05 was removed."
                    ),
                    "private_note": "SSN 111-22-3333 and card 4242 4242 4242 4242.",
                    "source_id": "ticket-eval-safe-001",
                },
                "labels": [
                    {
                        "span": "Alice Baker",
                        "class": "person_name",
                        "origin_field": "subject",
                        "name_subtype": "cue_less",
                    },
                    {
                        "span": "alice.baker@example.com",
                        "class": "email",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "202-555-0188",
                        "class": "phone",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "ORD-98765",
                        "class": "order_id",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "99 Real Street",
                        "class": "street_address",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "Alice Baker",
                        "class": "person_name",
                        "origin_field": "agent_reply",
                        "name_subtype": "cue_prefixed",
                    },
                    {
                        "span": "1977-06-05",
                        "class": "dob",
                        "origin_field": "agent_reply",
                    },
                    {
                        "span": "111-22-3333",
                        "class": "ssn",
                        "origin_field": "private_note",
                    },
                    {
                        "span": "4242 4242 4242 4242",
                        "class": "payment_card",
                        "origin_field": "private_note",
                    },
                ],
                "must_survive": [
                    {
                        "span": "CVE-2021-44228",
                        "origin_field": "agent_reply",
                        "reason": "security_reference",
                    },
                    {
                        "span": "ISO 27001",
                        "origin_field": "agent_reply",
                        "reason": "compliance_reference",
                    },
                    {
                        "span": "ticket-eval-safe-001",
                        "origin_field": "source_id",
                        "reason": "tenant_source_id",
                    },
                ],
            }
        ],
    }


def test_surrogate_artifact_rewrites_labels_and_drops_raw_pii() -> None:
    result = build_surrogate_eval_corpus(_valid_source())

    assert result.ok
    assert result.artifact is not None
    artifact = result.artifact
    rendered = json.dumps(artifact, sort_keys=True)
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["source"] == {
        "kind": "surrogated_eval",
        "raw_source_persisted": False,
        "raw_label_spans_persisted": False,
        "surrogate_positions_are_recall_labels": True,
    }
    for raw in (
        "Alice Baker",
        "alice.baker@example.com",
        "202-555-0188",
        "ORD-98765",
        "99 Real Street",
        "1977-06-05",
        "111-22-3333",
        "4242 4242 4242 4242",
    ):
        assert raw not in rendered

    ticket = artifact["tickets"][0]
    fields = ticket["fields"]
    labels = ticket["labels"]
    for label in labels:
        text = fields[label["origin_field"]]
        assert text[label["start"]:label["end"]] == label["span"]
    assert artifact["summary"]["labels_by_severity"] == {
        "high": 7,
        "medium": 2,
    }
    assert artifact["summary"]["cue_less_person_name_count"] == 1
    assert artifact["summary"]["cue_prefixed_person_name_count"] == 1


def test_labeled_source_intake_summary_reports_mix_without_raw_echo() -> None:
    summary = summarize_labeled_source(_valid_source())
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["ok"] is True
    assert summary["schema_version"] == SOURCE_INTAKE_SUMMARY_SCHEMA_VERSION
    assert summary["source_schema_version"] == "deflection_pii_labeled_source.v1"
    assert summary["artifact_schema_version"] == SCHEMA_VERSION
    assert summary["raw_source_persisted"] is False
    assert summary["raw_label_spans_persisted"] is False
    assert summary["ticket_count"] == 1
    assert summary["label_count"] == 9
    assert summary["labels_by_class"] == {
        "dob": 1,
        "email": 1,
        "order_id": 1,
        "payment_card": 1,
        "person_name": 2,
        "phone": 1,
        "ssn": 1,
        "street_address": 1,
    }
    assert summary["labels_by_origin_field"] == {
        "agent_reply": 2,
        "customer_message": 4,
        "private_note": 2,
        "subject": 1,
    }
    assert summary["person_name"] == {"cue_less": 1, "cue_prefixed": 1}
    assert summary["must_survive"] == {
        "count": 3,
        "by_reason": {
            "compliance_reference": 1,
            "security_reference": 1,
            "tenant_source_id": 1,
        },
    }
    for raw in (
        "Alice Baker",
        "alice.baker@example.com",
        "202-555-0188",
        "ORD-98765",
        "99 Real Street",
        "1977-06-05",
        "111-22-3333",
        "4242 4242 4242 4242",
    ):
        assert raw not in rendered


def test_labeled_source_intake_summary_requires_schema_without_raw_echo() -> None:
    source = _valid_source()
    source["schema_version"] = "alice@example.com"

    summary = summarize_labeled_source(source)
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["ok"] is False
    assert summary["source_schema_version"] == "invalid"
    assert summary["errors"] == [
        {
            "code": "source_schema_version_mismatch",
            "expected": "deflection_pii_labeled_source.v1",
            "actual": "invalid",
        }
    ]
    assert "Alice Baker" not in rendered
    assert "alice.baker@example.com" not in rendered
    assert "alice@example.com" not in rendered


def test_labeled_source_intake_summary_buckets_unsafe_must_survive_reasons() -> None:
    source = _valid_source()
    source["records"][0]["must_survive"][0]["reason"] = "alice@example.com"

    summary = summarize_labeled_source(source)
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["ok"] is True
    assert summary["must_survive"] == {
        "count": 3,
        "by_reason": {
            "compliance_reference": 1,
            "other": 1,
            "tenant_source_id": 1,
        },
    }
    assert "alice@example.com" not in rendered


def test_labeled_source_intake_markdown_reports_mix_without_raw_echo() -> None:
    summary = summarize_labeled_source(_valid_source())

    markdown = format_labeled_source_intake_summary_markdown(summary)

    assert "# Deflection PII Source Intake Summary" in markdown
    assert "| Status | ok |" in markdown
    assert "| Tickets | 1 |" in markdown
    assert "| Labels | 9 |" in markdown
    assert "| person_name | 2 |" in markdown
    assert "| high | 7 |" in markdown
    assert "| customer_message | 4 |" in markdown
    assert "| security_reference | 1 |" in markdown
    for raw in (
        "Alice Baker",
        "alice.baker@example.com",
        "202-555-0188",
        "ORD-98765",
        "99 Real Street",
        "1977-06-05",
        "111-22-3333",
        "4242 4242 4242 4242",
    ):
        assert raw not in markdown


def test_labeled_source_intake_markdown_sanitizes_invalid_source() -> None:
    source = _valid_source()
    source["schema_version"] = "alice@example.com"

    summary = summarize_labeled_source(source)
    markdown = format_labeled_source_intake_summary_markdown(summary)

    assert "| Status | blocked |" in markdown
    assert "source_schema_version_mismatch" in markdown
    assert "actual=invalid" in markdown
    assert "alice@example.com" not in markdown
    assert "Alice Baker" not in markdown


def test_must_survive_tokens_are_preserved_for_precision_scoring() -> None:
    artifact = build_surrogate_eval_corpus(_valid_source()).artifact
    assert artifact is not None
    ticket = artifact["tickets"][0]
    records = {item["span"]: item for item in ticket["must_survive"]}

    assert set(records) == {
        "CVE-2021-44228",
        "ISO 27001",
        "ticket-eval-safe-001",
    }
    for token, record in records.items():
        text = ticket["fields"][record["origin_field"]]
        assert text[record["start"]:record["end"]] == token


def test_repeated_raw_spans_are_surrogated_by_occurrence() -> None:
    result = build_surrogate_eval_corpus({
        "records": [
            {
                "fields": {
                    "customer_message": (
                        "Email repeated@example.com first, then repeated@example.com again."
                    )
                },
                "labels": [
                    {
                        "span": "repeated@example.com",
                        "class": "email",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "repeated@example.com",
                        "class": "email",
                        "origin_field": "customer_message",
                    },
                ],
            }
        ]
    })

    assert result.ok
    assert result.artifact is not None
    rendered = json.dumps(result.artifact, sort_keys=True)
    assert "repeated@example.com" not in rendered
    labels = result.artifact["tickets"][0]["labels"]
    assert [label["span"] for label in labels] == [
        "alex.rivera@example.test",
        "maya.chen@example.test",
    ]


def test_unlabeled_pii_in_rendered_output_fails_closed_without_raw_echo() -> None:
    result = build_surrogate_eval_corpus({
        "records": [
            {
                "fields": {
                    "customer_message": (
                        "Customer is John Doe. Email leak.real@gmail.com, "
                        "call 212-555-7788, SSN 987-65-4321, card "
                        "4000 0000 0000 0002, DOB 1977-06-05, ship to "
                        "44 Cedar Street."
                    )
                },
                "labels": [
                    {
                        "span": "John Doe",
                        "class": "person_name",
                        "origin_field": "customer_message",
                        "name_subtype": "cue_prefixed",
                    }
                ],
            }
        ]
    })

    assert not result.ok
    rendered = json.dumps({"errors": result.errors}, sort_keys=True)
    for raw in (
        "leak.real@gmail.com",
        "212-555-7788",
        "987-65-4321",
        "4000 0000 0000 0002",
        "1977-06-05",
        "44 Cedar Street",
    ):
        assert raw not in rendered
    assert {error["code"] for error in result.errors} == {"unlabeled_pii_detected"}
    assert {"email", "phone", "ssn", "payment_card", "dob", "street_address"} <= {
        error["detector"] for error in result.errors
    }


def test_unlabeled_dob_common_cues_fail_closed_without_raw_echo() -> None:
    for text, raw_dob in (
        ("Customer is John Doe. Birthday is 1977-06-05.", "1977-06-05"),
        ("Customer is John Doe. Born 06/05/1977.", "06/05/1977"),
    ):
        result = build_surrogate_eval_corpus({
            "records": [
                {
                    "fields": {"customer_message": text},
                    "labels": [
                        {
                            "span": "John Doe",
                            "class": "person_name",
                            "origin_field": "customer_message",
                            "name_subtype": "cue_prefixed",
                        }
                    ],
                }
            ]
        })

        assert not result.ok
        rendered = json.dumps({"errors": result.errors}, sort_keys=True)
        assert raw_dob not in rendered
        assert {error["detector"] for error in result.errors} == {"dob"}


def test_surrogate_never_reuses_matching_raw_span() -> None:
    result = build_surrogate_eval_corpus({
        "records": [
            {
                "fields": {"subject": "Maya Chen asked for an export."},
                "labels": [
                    {
                        "span": "Maya Chen",
                        "class": "person_name",
                        "origin_field": "subject",
                        "name_subtype": "cue_less",
                    }
                ],
            }
        ]
    })

    assert result.ok
    assert result.artifact is not None
    rendered = json.dumps(result.artifact, sort_keys=True)
    assert "Maya Chen" not in rendered
    label = result.artifact["tickets"][0]["labels"][0]
    assert label["span"] == "Jordan Lee"


def test_raw_label_span_remaining_after_partial_occurrence_replacement_fails() -> None:
    result = build_surrogate_eval_corpus({
        "records": [
            {
                "fields": {
                    "customer_message": (
                        "Email repeat.raw@example.com first, then "
                        "repeat.raw@example.com again."
                    )
                },
                "labels": [
                    {
                        "span": "repeat.raw@example.com",
                        "class": "email",
                        "origin_field": "customer_message",
                        "occurrence": 2,
                    }
                ],
            }
        ]
    })

    assert not result.ok
    rendered = json.dumps({"errors": result.errors}, sort_keys=True)
    assert "repeat.raw@example.com" not in rendered
    assert "raw_label_span_remaining" in {error["code"] for error in result.errors}


def test_empty_record_sets_are_rejected() -> None:
    result = build_surrogate_eval_corpus({"records": []})

    assert not result.ok
    assert result.artifact is None
    assert result.errors == ({"code": "source_empty_records"},)


def test_must_survive_offsets_track_occurrences_after_surrogation() -> None:
    result = build_surrogate_eval_corpus({
        "records": [
            {
                "fields": {
                    "customer_message": (
                        "Email raw@example.com. Keep ISO 27001 before ISO 27001."
                    )
                },
                "labels": [
                    {
                        "span": "raw@example.com",
                        "class": "email",
                        "origin_field": "customer_message",
                    }
                ],
                "must_survive": [
                    {
                        "span": "ISO 27001",
                        "origin_field": "customer_message",
                        "reason": "compliance_reference",
                    },
                    {
                        "span": "ISO 27001",
                        "origin_field": "customer_message",
                        "reason": "compliance_reference",
                    },
                ],
            }
        ]
    })

    assert result.ok
    assert result.artifact is not None
    ticket = result.artifact["tickets"][0]
    records = ticket["must_survive"]
    assert records[0]["start"] != records[1]["start"]
    for record in records:
        text = ticket["fields"][record["origin_field"]]
        assert text[record["start"]:record["end"]] == "ISO 27001"


def test_malformed_decoded_input_returns_sanitized_errors() -> None:
    result = build_surrogate_eval_corpus({
        "records": [
            None,
            {
                "fields": {"customer_message": "Email secret@example.com"},
                "labels": [
                    {
                        "span": "secret@example.com",
                        "class": "person_name",
                        "origin_field": "customer_message",
                    },
                    {
                        "span": "missing@example.com",
                        "class": "email",
                        "origin_field": "customer_message",
                    },
                ],
            },
        ]
    })

    assert not result.ok
    rendered = json.dumps({"errors": result.errors}, sort_keys=True)
    assert "secret@example.com" not in rendered
    assert "missing@example.com" not in rendered
    assert {error["code"] for error in result.errors} == {
        "record_not_object",
        "person_name_missing_subtype",
        "label_span_not_found",
    }


def test_cli_writes_artifact_and_rejects_invalid_input_without_raw_echo(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "artifact.json"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")

    assert CLI.main([str(source), "--output", str(output), "--pretty"]) == 0
    assert output.is_file()
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["summary"]["ticket_count"] == 1

    bad_source = tmp_path / "bad.json"
    bad_source.write_text(
        json.dumps({
            "records": [{
                "fields": {"customer_message": "Email raw.bad@example.com"},
                "labels": [{
                    "span": "missing.bad@example.com",
                    "class": "email",
                    "origin_field": "customer_message",
                }],
            }]
        }),
        encoding="utf-8",
    )
    assert CLI.main([str(bad_source), "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert "raw.bad@example.com" not in captured.err
    assert "missing.bad@example.com" not in captured.err
    assert "label_span_not_found" in captured.err


def test_cli_writes_summary_without_artifact_and_sanitizes_invalid_summary(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.json"
    summary_output = tmp_path / "summary.json"
    artifact_output = tmp_path / "artifact.json"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")

    assert CLI.main([str(source), "--summary-output", str(summary_output), "--pretty"]) == 0
    assert summary_output.is_file()
    assert not artifact_output.exists()
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["label_count"] == 9

    bad_source = tmp_path / "bad-source.json"
    bad_source.write_text(
        json.dumps({
            "schema_version": "deflection_pii_labeled_source.v1",
            "records": [{
                "fields": {"customer_message": "Email raw.bad@example.com"},
                "labels": [{
                    "span": "missing.bad@example.com",
                    "class": "email",
                    "origin_field": "customer_message",
                }],
            }],
        }),
        encoding="utf-8",
    )
    assert CLI.main([str(bad_source), "--summary-output", str(summary_output)]) == 1
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    rendered = json.dumps(summary, sort_keys=True) + captured.err
    assert summary["ok"] is False
    assert "label_span_not_found" in rendered
    assert "raw.bad@example.com" not in rendered
    assert "missing.bad@example.com" not in rendered


def test_cli_writes_markdown_summary_without_artifact_and_sanitizes_invalid_summary(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.json"
    markdown_output = tmp_path / "summary.md"
    artifact_output = tmp_path / "artifact.json"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")

    assert CLI.main([str(source), "--summary-markdown-output", str(markdown_output)]) == 0
    assert markdown_output.is_file()
    assert not artifact_output.exists()
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "| Labels | 9 |" in markdown
    assert "| payment_card | 1 |" in markdown
    assert "Alice Baker" not in markdown

    bad_source = tmp_path / "bad-source.json"
    bad_source.write_text(
        json.dumps({
            "schema_version": "deflection_pii_labeled_source.v1",
            "records": [{
                "fields": {"customer_message": "Email raw.bad@example.com"},
                "labels": [{
                    "span": "missing.bad@example.com",
                    "class": "email",
                    "origin_field": "customer_message",
                }],
            }],
        }),
        encoding="utf-8",
    )
    assert CLI.main([str(bad_source), "--summary-markdown-output", str(markdown_output)]) == 1
    markdown = markdown_output.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    rendered = markdown + captured.err
    assert "| Status | blocked |" in rendered
    assert "label_span_not_found" in rendered
    assert "record_index=1, label_index=1, origin_field=customer_message" in rendered
    assert "raw.bad@example.com" not in rendered
    assert "missing.bad@example.com" not in rendered


def test_cli_writes_review_bundle_with_artifact_and_summaries(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.json"
    bundle_dir = tmp_path / "review-bundle"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")

    assert CLI.main([str(source), "--review-bundle-dir", str(bundle_dir), "--pretty"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact_output = bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME
    summary_output = bundle_dir / CLI.REVIEW_BUNDLE_SUMMARY_NAME
    markdown_output = bundle_dir / CLI.REVIEW_BUNDLE_MARKDOWN_NAME

    assert payload["review_bundle_dir"] == str(bundle_dir)
    assert payload["output"] == str(artifact_output)
    assert payload["summary_output"] == str(summary_output)
    assert payload["summary_markdown_output"] == str(markdown_output)
    assert artifact_output.is_file()
    assert summary_output.is_file()
    assert markdown_output.is_file()
    artifact = json.loads(artifact_output.read_text(encoding="utf-8"))
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    rendered = (
        json.dumps(artifact, sort_keys=True)
        + json.dumps(summary, sort_keys=True)
        + markdown
        + captured.out
        + captured.err
    )
    assert artifact["summary"]["label_count"] == 9
    assert summary["ok"] is True
    assert "| Labels | 9 |" in markdown
    for raw in (
        "Alice Baker",
        "alice.baker@example.com",
        "202-555-0188",
        "ORD-98765",
        "99 Real Street",
        "1977-06-05",
        "111-22-3333",
        "4242 4242 4242 4242",
    ):
        assert raw not in rendered


def test_cli_review_bundle_sanitizes_invalid_source_without_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "bad-source.json"
    bundle_dir = tmp_path / "review-bundle"
    source.write_text(
        json.dumps({
            "schema_version": "deflection_pii_labeled_source.v1",
            "records": [{
                "fields": {"customer_message": "Email raw.bad@example.com"},
                "labels": [{
                    "span": "missing.bad@example.com",
                    "class": "email",
                    "origin_field": "customer_message",
                }],
            }],
        }),
        encoding="utf-8",
    )

    assert CLI.main([str(source), "--review-bundle-dir", str(bundle_dir), "--pretty"]) == 1
    captured = capsys.readouterr()
    artifact_output = bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME
    summary_output = bundle_dir / CLI.REVIEW_BUNDLE_SUMMARY_NAME
    markdown_output = bundle_dir / CLI.REVIEW_BUNDLE_MARKDOWN_NAME

    assert not artifact_output.exists()
    assert summary_output.is_file()
    assert markdown_output.is_file()
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    rendered = json.dumps(summary, sort_keys=True) + markdown + captured.out + captured.err
    assert summary["ok"] is False
    assert "| Status | blocked |" in markdown
    assert "label_span_not_found" in rendered
    assert "record_index=1, label_index=1, origin_field=customer_message" in rendered
    assert "raw.bad@example.com" not in rendered
    assert "missing.bad@example.com" not in rendered


def test_cli_review_bundle_removes_stale_artifact_on_invalid_rebuild(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.json"
    bad_source = tmp_path / "bad-source.json"
    bundle_dir = tmp_path / "review-bundle"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")
    bad_source.write_text(
        json.dumps({
            "schema_version": "alice@example.com",
            "records": [{
                "fields": {"customer_message": "Email raw.bad@example.com"},
                "labels": [{
                    "span": "missing.bad@example.com",
                    "class": "email",
                    "origin_field": "customer_message",
                }],
            }],
        }),
        encoding="utf-8",
    )

    assert CLI.main([str(source), "--review-bundle-dir", str(bundle_dir)]) == 0
    artifact_output = bundle_dir / CLI.REVIEW_BUNDLE_ARTIFACT_NAME
    assert artifact_output.is_file()
    valid_artifact = json.loads(artifact_output.read_text(encoding="utf-8"))
    assert valid_artifact["summary"]["label_count"] == 9
    capsys.readouterr()

    assert CLI.main([str(bad_source), "--review-bundle-dir", str(bundle_dir)]) == 1
    captured = capsys.readouterr()
    summary_output = bundle_dir / CLI.REVIEW_BUNDLE_SUMMARY_NAME
    markdown_output = bundle_dir / CLI.REVIEW_BUNDLE_MARKDOWN_NAME

    assert not artifact_output.exists()
    assert summary_output.is_file()
    assert markdown_output.is_file()
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    rendered = json.dumps(summary, sort_keys=True) + markdown + captured.out + captured.err
    assert summary["ok"] is False
    assert "source_schema_version_mismatch" in rendered
    assert "| Status | blocked |" in markdown
    assert "alice@example.com" not in rendered
    assert "raw.bad@example.com" not in rendered
    assert "missing.bad@example.com" not in rendered


@pytest.mark.parametrize(
    "output_flag",
    ("--output", "--summary-output", "--summary-markdown-output"),
)
def test_cli_rejects_review_bundle_with_individual_outputs(
    tmp_path: Path,
    output_flag: str,
) -> None:
    source = tmp_path / "source.json"
    bundle_dir = tmp_path / "review-bundle"
    explicit_output = tmp_path / "explicit-output"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        CLI.main([
            str(source),
            "--review-bundle-dir",
            str(bundle_dir),
            output_flag,
            str(explicit_output),
        ])

    assert exc.value.code == 2
    assert not bundle_dir.exists()
    assert not explicit_output.exists()


def test_cli_rejects_review_bundle_file_path_without_raw_echo(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.json"
    bundle_path = tmp_path / "review-bundle"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")
    bundle_path.write_text("not a directory", encoding="utf-8")

    assert CLI.main([str(source), "--review-bundle-dir", str(bundle_path)]) == 1
    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert "review_bundle_dir_not_directory" in rendered
    assert "Alice Baker" not in rendered
    assert "alice.baker@example.com" not in rendered


@pytest.mark.parametrize(
    ("first_flag", "second_flag"),
    (
        ("--summary-output", "--output"),
        ("--summary-markdown-output", "--output"),
        ("--summary-output", "--summary-markdown-output"),
    ),
)
def test_cli_rejects_colliding_output_paths(
    tmp_path: Path,
    first_flag: str,
    second_flag: str,
) -> None:
    source = tmp_path / "source.json"
    shared_output = tmp_path / "pii-output.json"
    source.write_text(json.dumps(_valid_source()), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        CLI.main([
            str(source),
            first_flag,
            str(shared_output),
            second_flag,
            str(shared_output),
        ])

    assert exc.value.code == 2
    assert not shared_output.exists()


def test_committed_tiny_fixture_is_surrogate_only() -> None:
    artifact = json.loads(TINY_FIXTURE.read_text(encoding="utf-8"))
    rendered = json.dumps(artifact, sort_keys=True)

    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["summary"]["ticket_count"] == 3
    assert artifact["summary"]["cue_less_person_name_count"] == 1
    assert artifact["summary"]["cue_prefixed_person_name_count"] == 3
    for raw in (
        "Alice Baker",
        "alice.baker@example.com",
        "202-555-0188",
        "ORD-98765",
        "99 Real Street",
        "1977-06-05",
        "111-22-3333",
        "4242 4242 4242 4242",
    ):
        assert raw not in rendered
    assert artifact["summary"]["labels_by_class"]["dob"] == 1
    assert artifact["summary"]["labels_by_severity"]["high"] == 9
    residual_ticket = artifact["tickets"][1]
    assert residual_ticket["fields"]["agent_reply"] == (
        "Customer Mary Jane Watson Report plan was upgraded."
    )
    assert residual_ticket["labels"] == [
        {
            "class": "person_name",
            "end": 25,
            "name_subtype": "cue_prefixed",
            "origin_field": "agent_reply",
            "severity": "high",
            "span": "Mary Jane Watson",
            "start": 9,
            "surrogate_id": "person_name-003",
        }
    ]
    short_surname_ticket = artifact["tickets"][2]
    assert short_surname_ticket["fields"]["agent_reply"] == (
        "Customer Amy Rae Li Report plan was upgraded."
    )
    assert short_surname_ticket["labels"] == [
        {
            "class": "person_name",
            "end": 19,
            "name_subtype": "cue_prefixed",
            "origin_field": "agent_reply",
            "severity": "high",
            "span": "Amy Rae Li",
            "start": 9,
            "surrogate_id": "person_name-004",
        }
    ]
