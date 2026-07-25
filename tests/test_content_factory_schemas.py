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
    from atlas_brain.schemas.content_factory import EditorialAuditV2

    audit = EditorialAuditV2.model_validate(
        {"schema": "editorial_audit.v2", "project_id": "resolution-audit"}
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


def _v2():
    from atlas_brain.schemas.content_factory import EditorialAuditV2

    return EditorialAuditV2


def test_audit_promote_without_verification_rejected():
    with pytest.raises(ValidationError):
        _v2().model_validate(
            {
                "schema": "editorial_audit.v2",
                "project_id": "resolution-audit",
                "recommendation": "promote",
            }
        )


def test_audit_promote_with_failed_verdict_rejected():
    with pytest.raises(ValidationError):
        _v2().model_validate(
            {
                "schema": "editorial_audit.v2",
                "project_id": "resolution-audit",
                "recommendation": "promote",
                "copy_verification": {"verdict": "fail", "hits": ["guaranteed savings"]},
            }
        )


def test_audit_promote_with_passing_verdict_accepted():
    audit = _v2().model_validate(
        {
            "schema": "editorial_audit.v2",
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


# --- editorial_audit versioning (#2181 round 2): v1 is FROZEN ---


def test_editorial_audit_v1_still_validates_old_artifacts():
    audit = EditorialAudit.model_validate(
        {"schema": "editorial_audit.v1", "project_id": "resolution-audit"}
    )
    assert audit.recommendation == "revise"


def test_editorial_audit_v1_rejects_advisory_warnings_field():
    """Rollback safety: the v1 shape is frozen, so a v1-tagged artifact can
    never carry the v2-only field (and an old reader never sees one)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EditorialAudit.model_validate(
            {
                "schema": "editorial_audit.v1",
                "project_id": "p",
                "advisory_warnings": ["x"],
            }
        )


def test_editorial_audit_v1_promote_gate_still_enforced():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EditorialAudit.model_validate(
            {
                "schema": "editorial_audit.v1",
                "project_id": "p",
                "recommendation": "promote",
            }
        )


def test_model_for_dispatches_both_audit_versions():
    from atlas_brain.schemas.content_factory import (
        EditorialAudit,
        EditorialAuditV2,
        model_for,
    )

    assert model_for({"schema": "editorial_audit.v1"}) is EditorialAudit
    assert model_for({"schema": "editorial_audit.v2"}) is EditorialAuditV2


# --- Phase 6 contracts: repurposing variants + image prompts (#2109) ---


def _variant(channel="linkedin", body="Clean copy about repeat tickets.", verdict="pass"):
    return {
        "channel": channel,
        "body_markdown": body,
        "derived_from_claims": ["e1"],
        "copy_verification": {"verdict": verdict, "hits": []},
    }


def _package(variants, ready=False):
    return {
        "schema": "repurposing.v1",
        "project_id": "resolution-audit",
        "variants": variants,
        "ready_to_publish": ready,
    }


def test_repurposing_requires_at_least_one_variant():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(_package([]))


def test_repurposing_rejects_duplicate_channels():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(
            _package([_variant(channel="linkedin"), _variant(channel="LinkedIn")])
        )


def test_repurposing_rejects_blank_variant_body():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(_package([_variant(body="   ")]))


def test_ready_to_publish_requires_every_variant_passing():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(
            _package(
                [_variant(channel="linkedin"), _variant(channel="x", verdict="fail")],
                ready=True,
            )
        )


def test_ready_to_publish_accepted_when_all_pass():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    pkg = RepurposingPackage.model_validate(
        _package([_variant(channel="linkedin"), _variant(channel="x")], ready=True)
    )
    assert pkg.ready_to_publish is True


def test_not_ready_package_may_carry_failing_variant():
    """A failing variant is a legitimate intermediate state -- it just cannot
    be declared shippable."""
    from atlas_brain.schemas.content_factory import RepurposingPackage

    pkg = RepurposingPackage.model_validate(
        _package([_variant(verdict="fail")], ready=False)
    )
    assert pkg.variants[0].copy_verification.verdict == "fail"


def test_variant_advisory_warnings_share_the_bounded_grammar():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    variant = _variant()
    variant["advisory_warnings"] = ["Contact bob@example.com"]
    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(_package([variant]))


def test_image_prompt_set_requires_a_prompt():
    from atlas_brain.schemas.content_factory import ImagePromptSet

    with pytest.raises(ValidationError):
        ImagePromptSet.model_validate(
            {"schema": "image_prompt.v1", "project_id": "p", "prompts": []}
        )


def test_image_prompt_set_accepts_valid_prompt():
    from atlas_brain.schemas.content_factory import ImagePromptSet

    ps = ImagePromptSet.model_validate(
        {
            "schema": "image_prompt.v1",
            "project_id": "p",
            "prompts": [{"purpose": "hero", "prompt_text": "a clean desk, soft light"}],
        }
    )
    assert ps.prompts[0].aspect_ratio == "1:1"


def test_phase6_schemas_dispatch():
    from atlas_brain.schemas.content_factory import (
        ImagePromptSet,
        RepurposingPackage,
        model_for,
    )

    assert model_for({"schema": "repurposing.v1"}) is RepurposingPackage
    assert model_for({"schema": "image_prompt.v1"}) is ImagePromptSet


# --- review round 1 on #2192 ---


def test_variant_without_lineage_is_rejected():
    """Orphan variants must be unrepresentable, not merely discouraged."""
    from atlas_brain.schemas.content_factory import RepurposingPackage

    v = _variant()
    del v["derived_from_claims"]
    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(_package([v]))


def test_variant_with_empty_lineage_is_rejected():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(
            _package([{**_variant(), "derived_from_claims": []}])
        )


def test_mixed_lineage_package_is_rejected():
    """One traceable variant does not license an untraceable sibling."""
    from atlas_brain.schemas.content_factory import RepurposingPackage

    good = _variant(channel="linkedin")
    bad = {**_variant(channel="x"), "derived_from_claims": []}
    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(_package([good, bad], ready=True))


def test_blank_lineage_id_is_rejected():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(
            _package([{**_variant(), "derived_from_claims": ["  "]}])
        )


def test_ready_to_generate_requires_passing_verdict():
    from atlas_brain.schemas.content_factory import ImagePromptSet

    with pytest.raises(ValidationError):
        ImagePromptSet.model_validate({
            "schema": "image_prompt.v1", "project_id": "p",
            "prompts": [{"purpose": "hero", "prompt_text": "a desk"}],
            "copy_verification": {"verdict": "fail", "hits": ["guaranteed-savings: x"]},
            "ready_to_generate": True,
        })


def test_ready_to_generate_rejected_without_any_verdict():
    from atlas_brain.schemas.content_factory import ImagePromptSet

    with pytest.raises(ValidationError):
        ImagePromptSet.model_validate({
            "schema": "image_prompt.v1", "project_id": "p",
            "prompts": [{"purpose": "hero", "prompt_text": "a desk"}],
            "ready_to_generate": True,
        })


def test_ready_to_generate_accepted_when_passing():
    from atlas_brain.schemas.content_factory import ImagePromptSet

    ps = ImagePromptSet.model_validate({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": "a desk"}],
        "copy_verification": {"verdict": "pass", "hits": []},
        "ready_to_generate": True,
    })
    assert ps.ready_to_generate is True


def test_failing_set_may_persist_when_not_ready():
    """A failing verdict is a legitimate intermediate state."""
    from atlas_brain.schemas.content_factory import ImagePromptSet

    ps = ImagePromptSet.model_validate({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": "guaranteed savings poster"}],
        "copy_verification": {"verdict": "fail", "hits": ["guaranteed-savings: x"]},
    })
    assert ps.ready_to_generate is False


@pytest.mark.parametrize("invisible", ["​", "­‌", "⁠", "   "])
def test_invisible_only_variant_body_rejected(invisible):
    """Zero-width and format-only text renders as nothing; it is not copy."""
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(
            _package([{**_variant(), "body_markdown": invisible}])
        )


@pytest.mark.parametrize("invisible", ["​", "­", "⁠"])
def test_invisible_only_prompt_text_rejected(invisible):
    from atlas_brain.schemas.content_factory import ImagePromptSet

    with pytest.raises(ValidationError):
        ImagePromptSet.model_validate({
            "schema": "image_prompt.v1", "project_id": "p",
            "prompts": [{"purpose": "hero", "prompt_text": invisible}],
        })


def test_visible_text_with_incidental_zero_width_accepted():
    """The other side: real copy is not rejected for containing one."""
    from atlas_brain.schemas.content_factory import RepurposingPackage

    pkg = RepurposingPackage.model_validate(
        _package([{**_variant(), "body_markdown": "real​copy here"}])
    )
    assert "copy" in pkg.variants[0].body_markdown


@pytest.mark.parametrize("mark_only", ["️", "́", "︎"])
def test_combining_mark_only_text_rejected(mark_only):
    """A lone variation selector/combining mark renders nothing."""
    from atlas_brain.schemas.content_factory import RepurposingPackage

    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(
            _package([{**_variant(), "body_markdown": mark_only}])
        )


def test_emoji_with_variation_selector_accepted():
    """The other side: real content carrying a mark is still content."""
    from atlas_brain.schemas.content_factory import RepurposingPackage

    pkg = RepurposingPackage.model_validate(
        _package([{**_variant(), "body_markdown": "spotless results ❤️"}])
    )
    assert "spotless" in pkg.variants[0].body_markdown


def test_canonically_equivalent_channels_are_duplicates():
    from atlas_brain.schemas.content_factory import RepurposingPackage

    nfc, nfd = "café", "café"
    with pytest.raises(ValidationError):
        RepurposingPackage.model_validate(
            _package([_variant(channel=nfc), _variant(channel=nfd)])
        )
