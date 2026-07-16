"""Contract tests for the Content Factory artifact schemas.

Fixtures mirror the shapes produced by the Phase 1.4 end-to-end run
(jobs/20260715-155227-resolution-audit-linkedin), including the real
empty-evidence packet, so the contracts are tested against artifacts a worker
actually emits rather than idealized ones.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas_brain.schemas.content_factory import (
    ArtifactManifest,
    ContentBrief,
    DraftArtifact,
    EditorialAudit,
    EvidencePacket,
    EvidenceRow,
    model_for,
)

BRIEF = {
    "schema": "content_brief.v1",
    "project_id": "resolution-audit",
    "request_raw": "Write a short LinkedIn post ...",
    "channel": "LinkedIn",
    "audience": "support and CX leaders",
    "goal": "Invite target audience to a free Resolution Audit snapshot.",
    "angle": "The hidden cost of repetitive support tickets.",
    "must_include": ["hidden cost of repetitive tickets", "invitation to free snapshot"],
    "must_avoid": ["guaranteed savings"],
    "target_length": "150 words",
    "open_questions": [],
}

# The real Researcher output: no citable quotes -> gaps, never fabrication.
EVIDENCE_EMPTY = {
    "schema": "evidence_packet.v1",
    "project_id": "resolution-audit",
    "evidence": [],
    "gaps": ["no first-party savings number available", "no on-topic quote found"],
    "retrieved_from": [],
}

DRAFT = {
    "schema": "draft.v1",
    "project_id": "resolution-audit",
    "revision": 1,
    "body_markdown": "Support and CX leaders: ...",
    "claims": [],
    "word_count": 98,
}


@pytest.mark.parametrize(
    "data,model",
    [
        (BRIEF, ContentBrief),
        (EVIDENCE_EMPTY, EvidencePacket),
        (DRAFT, DraftArtifact),
    ],
)
def test_round_trip_preserves_schema_tag(data, model):
    obj = model.model_validate(data)
    dumped = obj.model_dump(by_alias=True)
    assert dumped["schema"] == data["schema"]
    assert dumped["project_id"] == "resolution-audit"
    # re-parsing the dumped form yields an equal object
    assert model.model_validate(dumped) == obj


def test_model_for_dispatches_by_tag():
    assert model_for(BRIEF) is ContentBrief
    assert model_for(EVIDENCE_EMPTY) is EvidencePacket
    with pytest.raises(ValueError):
        model_for({"schema": "not_a_real.v1"})


def test_empty_evidence_packet_with_gaps_is_valid():
    # A packet with zero rows but logged gaps is the honest "no evidence" result.
    pkt = EvidencePacket.model_validate(EVIDENCE_EMPTY)
    assert pkt.evidence == []
    assert len(pkt.gaps) == 2


def test_evidence_packet_without_evidence_or_gaps_rejected():
    # A packet with neither evidence nor gaps cannot masquerade as an honest
    # empty result -- it is indistinguishable from truncated worker output.
    with pytest.raises(ValidationError):
        EvidencePacket.model_validate(
            {"schema": "evidence_packet.v1", "project_id": "p"}
        )


@pytest.mark.parametrize("blank_gaps", [[""], ["   "], ["real gap", ""]])
def test_evidence_packet_blank_gap_rejected(blank_gaps):
    # A blank/whitespace gap is not a logged gap; it must not satisfy the
    # honest-empty-packet guard (gaps=[''] would otherwise slip past it).
    with pytest.raises(ValidationError):
        EvidencePacket.model_validate(
            {"schema": "evidence_packet.v1", "project_id": "p", "gaps": blank_gaps}
        )


def test_evidence_row_without_source_id_is_rejected():
    # Load-bearing guard: uncited evidence must not validate.
    with pytest.raises(ValidationError):
        EvidenceRow.model_validate({"id": "e1", "quote": "some text"})


def test_evidence_row_without_quote_is_rejected():
    with pytest.raises(ValidationError):
        EvidenceRow.model_validate({"id": "e1", "source_id": "f#1"})


def test_evidence_row_valid_minimal():
    row = EvidenceRow.model_validate(
        {"id": "e1", "quote": "agents answer this weekly", "source_id": "evidence.jsonl#42"}
    )
    assert row.confidence == "medium"  # default applied


def test_draft_requires_body():
    with pytest.raises(ValidationError):
        DraftArtifact.model_validate(
            {"schema": "draft.v1", "project_id": "p", "claims": []}
        )


def test_brief_requires_project_and_request():
    with pytest.raises(ValidationError):
        ContentBrief.model_validate({"schema": "content_brief.v1"})


def test_extra_keys_rejected():
    # extra='forbid': an unmodeled key fails closed rather than riding through.
    with pytest.raises(ValidationError):
        ContentBrief.model_validate({**BRIEF, "experimental_hint": "keep me"})


def test_editorial_audit_defaults_to_revise():
    audit = EditorialAudit.model_validate(
        {"schema": "editorial_audit.v1", "project_id": "resolution-audit"}
    )
    assert audit.recommendation == "revise"
    assert audit.voice_pass is False


def test_manifest_default_approval_pending():
    m = ArtifactManifest.model_validate(
        {"schema": "manifest.v1", "job_id": "j1", "project_id": "resolution-audit"}
    )
    assert m.approval.status == "pending"


# --- load-bearing invariants (from Codex review of #2116) ---


def test_evidence_row_blank_quote_rejected():
    with pytest.raises(ValidationError):
        EvidenceRow.model_validate({"id": "e1", "quote": "", "source_id": "f#1"})


def test_evidence_row_blank_source_id_rejected():
    with pytest.raises(ValidationError):
        EvidenceRow.model_validate({"id": "e1", "quote": "real quote", "source_id": ""})


def test_evidence_row_blank_id_rejected():
    # A blank id cannot be referenced by Claim.source_id -> no-orphan-claim guard.
    with pytest.raises(ValidationError):
        EvidenceRow.model_validate({"id": "  ", "quote": "real quote", "source_id": "f#1"})


def test_evidence_row_whitespace_only_rejected():
    with pytest.raises(ValidationError):
        EvidenceRow.model_validate({"id": "e1", "quote": "   ", "source_id": "f#1"})


def test_evidence_row_strips_whitespace():
    row = EvidenceRow.model_validate(
        {"id": "e1", "quote": "  q  ", "source_id": "  f#1 "}
    )
    assert row.quote == "q"
    assert row.source_id == "f#1"


def test_claim_blank_source_id_rejected():
    from atlas_brain.schemas.content_factory import Claim

    with pytest.raises(ValidationError):
        Claim.model_validate({"text": "a claim", "source_id": ""})


def test_missing_schema_tag_rejected():
    # The version/type tag is required on top-level artifacts (version boundary).
    with pytest.raises(ValidationError):
        ContentBrief.model_validate({"project_id": "resolution-audit", "request_raw": "x"})


def test_attribute_name_only_tag_rejected():
    # The canonical "schema" key is the sole admission rule: supplying the tag
    # only under the attribute name "artifact_schema" must fail, since that raw
    # dict could not be dispatched by model_for() (which reads "schema").
    with pytest.raises(ValidationError):
        ContentBrief.model_validate(
            {"artifact_schema": "content_brief.v1", "project_id": "p", "request_raw": "x"}
        )


def test_audit_promote_without_verification_rejected():
    with pytest.raises(ValidationError):
        EditorialAudit.model_validate(
            {
                "schema": "editorial_audit.v1",
                "project_id": "resolution-audit",
                "recommendation": "promote",
            }
        )


def test_audit_promote_with_failed_verdict_rejected():
    with pytest.raises(ValidationError):
        EditorialAudit.model_validate(
            {
                "schema": "editorial_audit.v1",
                "project_id": "resolution-audit",
                "recommendation": "promote",
                "copy_verification": {"verdict": "fail", "hits": ["guaranteed savings"]},
            }
        )


def test_audit_promote_with_passing_verdict_accepted():
    audit = EditorialAudit.model_validate(
        {
            "schema": "editorial_audit.v1",
            "project_id": "resolution-audit",
            "recommendation": "promote",
            "copy_verification": {"verdict": "pass", "hits": []},
        }
    )
    assert audit.recommendation == "promote"


def test_default_dump_uses_canonical_schema_key():
    # Default model_dump() (no by_alias) emits the "schema" key, so model_for
    # round-trips without callers remembering by_alias=True (version boundary).
    brief = ContentBrief.model_validate(BRIEF)
    dumped = brief.model_dump()
    assert dumped["schema"] == "content_brief.v1"
    assert "artifact_schema" not in dumped
    assert model_for(dumped) is ContentBrief


def test_model_dump_json_uses_canonical_schema_key():
    import json

    draft = DraftArtifact.model_validate(DRAFT)
    reparsed = json.loads(draft.model_dump_json())
    assert reparsed["schema"] == "draft.v1"
    assert model_for(reparsed) is DraftArtifact
