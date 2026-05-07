from __future__ import annotations

import json
from pathlib import Path

import pytest

from extracted_evidence_to_story.claims import (
    CLAIMS_FILENAME,
    ClaimLedger,
    ClaimRecord,
    ClaimSchemaError,
    InvalidClaimType,
    SourceLocator,
    apply_soft_rewrite,
    load_claim_ledger,
    validate_claim,
    validate_ledger,
    write_claim_ledger,
)


def _factual(**overrides: object) -> ClaimRecord:
    base = dict(
        claim_id="clm_001",
        story_id="case_001",
        text="The car was found near the river.",
        claim_type="factual",
        source_id="src_news_01",
        quote="The car was found near the river at dawn.",
        source_locator=SourceLocator(paragraph=3),
        confidence="verified",
    )
    base.update(overrides)
    return ClaimRecord(**base)  # type: ignore[arg-type]


def _ledger(*claims: ClaimRecord, story_id: str = "case_001") -> ClaimLedger:
    return ClaimLedger(story_id=story_id, claims=tuple(claims))


# ----- per-type fact-check invariants (§5 table) -------------------------


def test_factual_requires_source_id_quote_and_locator() -> None:
    validate_claim(_factual())  # baseline ok

    with pytest.raises(ClaimSchemaError, match="non-empty source_id"):
        validate_claim(_factual(source_id=""))

    with pytest.raises(ClaimSchemaError, match="non-empty quote"):
        validate_claim(_factual(quote=""))

    with pytest.raises(ClaimSchemaError, match="paragraph or timestamp"):
        validate_claim(_factual(source_locator=SourceLocator(quote_offset=12)))


def test_timeline_requires_locator_with_paragraph_or_timestamp() -> None:
    claim = _factual(
        claim_id="clm_t1",
        claim_type="timeline",
        text="911 call placed at 8:43 PM.",
        source_locator=SourceLocator(timestamp="00:14:23"),
    )
    validate_claim(claim)


def test_entity_requires_full_locator_set() -> None:
    claim = _factual(
        claim_id="clm_e1",
        claim_type="entity",
        text="John Doe lived in Chicago.",
        source_locator=SourceLocator(paragraph=2),
    )
    validate_claim(claim)


def test_emotional_inference_requires_rewrite_applied_and_original_text() -> None:
    raw = _factual(
        claim_id="clm_emo",
        claim_type="emotional_inference",
        text="She felt isolated.",
        confidence="inferred",
    )
    with pytest.raises(ClaimSchemaError, match="rewrite_applied is False"):
        validate_claim(raw)

    flagged_no_original = _factual(
        claim_id="clm_emo2",
        claim_type="emotional_inference",
        text="She felt isolated.",
        confidence="inferred",
        rewrite_applied=True,
        original_text="",
    )
    with pytest.raises(ClaimSchemaError, match="original_text is empty"):
        validate_claim(flagged_no_original)


def test_disputed_requires_dispute_group_id() -> None:
    claim = _factual(
        claim_id="clm_d1",
        claim_type="disputed",
        text="The defense says the meeting never happened.",
        confidence="disputed",
    )
    with pytest.raises(ClaimSchemaError, match="dispute_group_id"):
        validate_claim(claim)


def test_disputed_group_requires_two_distinct_source_ids_across_members() -> None:
    a = _factual(
        claim_id="clm_d_a",
        claim_type="disputed",
        text="Sources disagree on what time the call happened.",
        confidence="disputed",
        dispute_group_id="dispute_call_time",
    )
    b = _factual(
        claim_id="clm_d_b",
        claim_type="disputed",
        text="Sources disagree on what time the call happened.",
        confidence="disputed",
        dispute_group_id="dispute_call_time",
        source_id="src_yt_01",
        quote="It happened at 8 PM, the host says.",
    )

    # One distinct source_id -> fail.
    with pytest.raises(ClaimSchemaError, match="distinct source_id"):
        validate_ledger(_ledger(a, _factual(claim_id="clm_d_c", claim_type="disputed",
                                            text="same", dispute_group_id="dispute_call_time")))
    # Two distinct -> ok.
    validate_ledger(_ledger(a, b))


def test_reveal_requires_source_id_quote_and_locator() -> None:
    claim = _factual(
        claim_id="clm_r1",
        claim_type="reveal",
        text="Police arrested the neighbor.",
    )
    validate_claim(claim)

    with pytest.raises(ClaimSchemaError, match="paragraph or timestamp"):
        validate_claim(_factual(claim_id="clm_r2", claim_type="reveal",
                                source_locator=SourceLocator()))


def test_transition_forbids_source_fields() -> None:
    transition = ClaimRecord(
        claim_id="clm_tr",
        story_id="case_001",
        text="Hours passed before anyone noticed.",
        claim_type="transition",
    )
    validate_claim(transition)

    with pytest.raises(ClaimSchemaError, match="transition"):
        validate_claim(ClaimRecord(
            claim_id="clm_tr2",
            story_id="case_001",
            text="bad",
            claim_type="transition",
            source_id="src_news_01",
        ))


# ----- type and structural errors ----------------------------------------


def test_invalid_claim_type_raises_specific_subclass() -> None:
    bad = ClaimRecord(
        claim_id="clm_x",
        story_id="case_001",
        text="x",
        claim_type="reconstructed",  # type: ignore[arg-type]
    )
    with pytest.raises(InvalidClaimType, match="reconstructed"):
        validate_claim(bad)


# ----- soft-rewrite ------------------------------------------------------


def test_apply_soft_rewrite_stamps_original_text_and_flag() -> None:
    raw = _factual(
        claim_id="clm_emo",
        claim_type="emotional_inference",
        text="She felt isolated.",
        confidence="inferred",
    )
    rewritten = apply_soft_rewrite(
        raw,
        rewritten_text="Based on the reporting, she appears to have been left isolated.",
    )
    assert rewritten.rewrite_applied is True
    assert rewritten.original_text == "She felt isolated."
    assert rewritten.text.startswith("Based on the reporting")
    # All other fields preserved.
    assert rewritten.claim_id == raw.claim_id
    assert rewritten.source_id == raw.source_id
    assert rewritten.confidence == raw.confidence
    # Validates after rewrite.
    validate_claim(rewritten)


def test_apply_soft_rewrite_rejects_non_emotional_inference_claims() -> None:
    with pytest.raises(ClaimSchemaError, match="emotional_inference"):
        apply_soft_rewrite(_factual(), rewritten_text="hedged version")


def test_apply_soft_rewrite_requires_non_empty_rewritten_text() -> None:
    raw = _factual(
        claim_id="clm_emo3",
        claim_type="emotional_inference",
        text="She felt isolated.",
    )
    with pytest.raises(ClaimSchemaError, match="non-empty rewritten_text"):
        apply_soft_rewrite(raw, rewritten_text="   ")


# ----- ledger-level invariants -------------------------------------------


def test_ledger_rejects_duplicate_claim_ids() -> None:
    a = _factual(claim_id="clm_dup")
    b = _factual(claim_id="clm_dup", text="another claim with same id",
                 quote="other quote")
    with pytest.raises(ClaimSchemaError, match="duplicate claim_id"):
        validate_ledger(_ledger(a, b))


def test_ledger_rejects_claim_story_id_mismatch() -> None:
    a = _factual(story_id="case_OTHER")
    with pytest.raises(ClaimSchemaError, match="does not match ledger story_id"):
        validate_ledger(_ledger(a))


def test_ledger_round_trips_through_json(tmp_path: Path) -> None:
    rewritten = apply_soft_rewrite(
        _factual(
            claim_id="clm_emo_rt",
            claim_type="emotional_inference",
            text="She felt isolated.",
            confidence="inferred",
        ),
        rewritten_text="The reporting suggests she was isolated.",
    )
    transition = ClaimRecord(
        claim_id="clm_tr_rt",
        story_id="case_001",
        text="Hours later...",
        claim_type="transition",
    )
    ledger = _ledger(_factual(), rewritten, transition)

    output_path = write_claim_ledger(ledger, tmp_path)

    assert output_path == tmp_path / CLAIMS_FILENAME
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["story_id"] == "case_001"
    assert len(payload["claims"]) == 3
    # JSON output sorts keys; the reload parses back to an identical ledger.
    reloaded = load_claim_ledger(output_path)
    assert reloaded.as_dict() == ledger.as_dict()


def test_load_claim_ledger_validates_on_read(tmp_path: Path) -> None:
    bad_ledger = ClaimLedger(
        story_id="case_001",
        claims=(_factual(source_id=""),),  # missing source_id is invalid
    )
    output_path = tmp_path / CLAIMS_FILENAME
    # Bypass write_claim_ledger (which validates) so we can write a bad file.
    output_path.write_text(
        json.dumps(bad_ledger.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ClaimSchemaError, match="non-empty source_id"):
        load_claim_ledger(output_path)


def test_load_claim_ledger_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ClaimSchemaError, match="not found"):
        load_claim_ledger(tmp_path / "nonexistent.json")


def test_load_claim_ledger_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("this is not json", encoding="utf-8")
    with pytest.raises(ClaimSchemaError, match="not valid JSON"):
        load_claim_ledger(bad)


def test_write_claim_ledger_validates_before_writing(tmp_path: Path) -> None:
    bad_ledger = ClaimLedger(
        story_id="case_001",
        claims=(_factual(quote=""),),
    )
    with pytest.raises(ClaimSchemaError):
        write_claim_ledger(bad_ledger, tmp_path)
    assert not (tmp_path / CLAIMS_FILENAME).exists()
