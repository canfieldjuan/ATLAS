from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_zendesk_product_proof_corpus",
    ROOT / "scripts" / "evaluate_zendesk_product_proof_corpus.py",
)
assert SPEC is not None
assert SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _labels():
    return {
        "zd-proof-001": {
            "should_publish_answer": True,
            "unresolved": False,
            "reopened": False,
            "has_private_note": False,
        },
        "zd-proof-002": {
            "should_publish_answer": False,
            "unresolved": True,
            "reopened": False,
            "has_private_note": False,
        },
        "zd-proof-003": {
            "should_publish_answer": False,
            "unresolved": False,
            "reopened": True,
            "has_private_note": False,
        },
        "zd-proof-004": {
            "should_publish_answer": False,
            "unresolved": False,
            "reopened": False,
            "has_private_note": True,
        },
    }


def _item(source_ids, question="How do I reset MFA?"):
    return {
        "question": question,
        "answer_evidence_status": "resolution_evidence",
        "ticket_count": len(source_ids),
        "source_ids": tuple(source_ids),
    }


def test_report_excerpt_skips_complete_evidence_bodies_but_keeps_next_question() -> None:
    markdown = """# Support Ticket Deflection Report

## Question Details and Evidence

### 1. How do I get a duplicate charge refunded?

**Publishable answer draft:**

Refund the duplicate charge.

**Complete evidence:**

**Source IDs (full list):** zd-proof-020

- `zd-proof-020` - [Atlas seed 01] raw seeded ticket text

### 2. How do I reset MFA?

**Publishable answer draft:**

Reset MFA from Admin Settings.
"""

    excerpt = MOD._report_excerpt(markdown, max_lines=40)

    assert "[Atlas seed" not in excerpt
    assert "### 2. How do I reset MFA?" in excerpt
    assert "Reset MFA from Admin Settings." in excerpt


def test_evaluate_items_accepts_clean_publishable_sources() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-001",)),),
        markdown="Use the admin panel to reset MFA.",
        private_texts={"zd-proof-004": ("internal refund quietly",)},
    )

    assert result["status"] == "ok"
    assert result["blocking_violation_codes"] == []
    assert result["covered_publishable_source_count"] == 1


def test_evaluate_items_records_faq_warnings_without_making_them_silent() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-001",)),),
        markdown="Use the admin panel to reset MFA.",
        private_texts={},
        faq_warnings=({
            "code": "duplicate_source_policy_questions",
            "question": "What should I do about reset?",
        },),
    )

    assert result["status"] == "ok"
    assert result["faq_warning_count"] == 1
    assert result["faq_warning_codes"] == ["duplicate_source_policy_questions"]


def test_evaluate_items_blocks_non_publishable_source_in_publishable_answer() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-001", "zd-proof-002")),),
        markdown="Use the admin panel to reset MFA.",
        private_texts={},
    )

    assert result["status"] == "failed"
    assert "publishable_false_positive" in result["blocking_violation_codes"]
    assert result["publishable_false_positive_source_ids"] == ["zd-proof-002"]


def test_evaluate_items_blocks_unresolved_source_in_publishable_answer() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-002",)),),
        markdown="Use the admin panel to reset MFA.",
        private_texts={},
    )

    assert result["status"] == "failed"
    assert "unresolved_source_published" in result["blocking_violation_codes"]
    assert result["unresolved_publishable_source_ids"] == ["zd-proof-002"]


def test_evaluate_items_blocks_reopened_source_in_publishable_answer() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-003",)),),
        markdown="Use the admin panel to reset MFA.",
        private_texts={},
    )

    assert result["status"] == "failed"
    assert "reopened_source_published" in result["blocking_violation_codes"]
    assert result["reopened_publishable_source_ids"] == ["zd-proof-003"]


def test_evaluate_items_blocks_private_note_text_leak() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-001",)),),
        markdown="The internal note says refund quietly after the billing bug.",
        private_texts={"zd-proof-004": ("refund quietly after the billing bug",)},
    )

    assert result["status"] == "failed"
    assert "private_note_leak" in result["blocking_violation_codes"]
    assert result["private_note_leaks"] == [
        {
            "source_id": "zd-proof-004",
            "text": "refund quietly after the billing bug",
        }
    ]


def test_evaluate_items_blocks_truncated_private_note_windows() -> None:
    private_notes = {
        "zd-proof-004": (
            "refund quietly after the billing bug because proration details are internal",
            "known outage root cause do not share until incident review closes",
            "manual approval bypass was used for this account during escalation",
            "customer threatened chargeback but waive the fee without mentioning policy",
            "agent saw duplicate records from importer bug keep that explanation internal",
            "security review found stale token reuse escalate privately before replying",
        )
    }
    markdown = "\n".join(
        (
            "Evidence: refund quietly after the billing bug...",
            "Evidence: known outage root cause do not share...",
            "Evidence: manual approval bypass was used for this...",
            "Evidence: customer threatened chargeback but waive the fee...",
            "Evidence: duplicate records from importer bug keep that...",
            "Evidence: stale token reuse escalate privately before replying...",
        )
    )

    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-001",)),),
        markdown=markdown,
        private_texts=private_notes,
    )

    assert result["status"] == "failed"
    assert "private_note_leak" in result["blocking_violation_codes"]
    assert [leak["source_id"] for leak in result["private_note_leaks"]] == [
        "zd-proof-004",
        "zd-proof-004",
        "zd-proof-004",
        "zd-proof-004",
        "zd-proof-004",
        "zd-proof-004",
    ]


def test_evaluate_items_allows_private_note_near_miss_without_token_window() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-001",)),),
        markdown="The public answer says refund after confirming the billing event.",
        private_texts={
            "zd-proof-004": (
                "refund quietly after the billing bug because proration details are internal",
            )
        },
    )

    assert result["status"] == "ok"
    assert result["private_note_leaks"] == []


def test_evaluate_items_blocks_degraded_question_labels() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(
            _item(("zd-proof-001",), question="[Atlas seed 10] Login and MFA"),
            _item(("zd-proof-001",), question="What should I do about atla?"),
            _item(
                ("zd-proof-001",),
                question="Localized support question Como fa\u00e7o para recuperar uma conta bloqueada por MFA?",
            ),
        ),
        markdown="Use the admin panel to reset MFA.",
        private_texts={},
    )

    assert result["status"] == "failed"
    assert "degraded_question_label" in result["blocking_violation_codes"]
    assert result["degraded_question_labels"] == [
        "[Atlas seed 10] Login and MFA",
        "What should I do about atla?",
        "Localized support question Como fa\u00e7o para recuperar uma conta bloqueada por MFA?",
    ]


def test_evaluate_items_records_draft_degraded_question_labels_without_blocking() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(
            {
                **_item(("zd-proof-001",), question="What should I do about reset?"),
                "answer_evidence_status": "draft_needs_review",
            },
            {
                **_item(("zd-proof-001",), question="What should I do about duplicate?"),
                "answer_evidence_status": "draft_needs_review",
            },
        ),
        markdown="Use the admin panel to reset MFA.",
        private_texts={},
    )

    assert result["status"] == "ok"
    assert result["degraded_question_labels"] == []
    assert result["degraded_draft_question_labels"] == [
        "What should I do about reset?",
        "What should I do about duplicate?",
    ]


def test_evaluate_items_blocks_failed_artifact_output_checks() -> None:
    result = MOD.evaluate_items(
        labels=_labels(),
        items=(_item(("zd-proof-001",)),),
        markdown="Use the admin panel to reset MFA.",
        private_texts={},
        output_checks={
            "uses_user_vocabulary": True,
            "resolution_evidence_scoped": False,
        },
    )

    assert result["status"] == "failed"
    assert "failed_output_checks" in result["blocking_violation_codes"]
    assert result["failed_output_checks"] == ["resolution_evidence_scoped"]


def test_committed_zendesk_product_proof_corpus_passes_eval(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifact"
    doc = tmp_path / "proof.md"

    code = MOD.main(["--out-dir", str(out_dir), "--doc", str(doc)])

    assert code == 0
    summary = (out_dir / "summary.json").read_text(encoding="utf-8")
    assert '"status": "ok"' in summary
    report_excerpt = (out_dir / "report_excerpt.md").read_text(encoding="utf-8")
    assert "[Atlas seed" not in report_excerpt
    assert "What should I do about atla?" not in report_excerpt
    assert doc.exists()
