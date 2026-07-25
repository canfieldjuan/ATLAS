"""Pydantic v2 contracts for the local Content Factory artifact pipeline.

The Content Factory (Open WebUI + LM Studio, GitHub epic #2109) runs a rough
request through stage-scoped workers and drops one artifact per stage into a
git-init'd job folder. These models are the *shapes* of those artifacts, so a
stage output can be validated deterministically -- independent of which local
model produced it.

Five artifact surfaces, one per stage:

  - ContentBrief     -> brief.json     (Brief Architect)
  - EvidencePacket   -> evidence.json  (Researcher / Evidence Builder)
  - DraftArtifact    -> draft.json     (Long-Form Writer)
  - EditorialAudit   -> audit.json     (Editor + Verifier)
  - ArtifactManifest -> manifest.json  (the job index)

House style (matches atlas_brain/schemas/campaigns.py): every model opens with
``schema_version=1``. The iteration-phase ``extra='allow'`` has now been flipped
to ``extra='forbid'`` (the content_factory_store/runner consumers round-trip
artifacts, so the flip is validated against known-good shapes): any unmodeled key
is rejected rather than silently carried, terminally closing the schema-key leak
class (a reserved ``artifact_schema`` key can no longer ride through as an extra).

The artifact's type tag is stored under the JSON key ``"schema"`` via an alias
(the attribute is ``artifact_schema`` to avoid shadowing ``BaseModel.schema``).
The canonical ``"schema"`` key is the sole admission rule: raw input is accepted
only under ``"schema"`` (not the attribute name), and ``serialize_by_alias``
emits ``"schema"`` on dump, so a raw artifact always round-trips through
``model_for()``. The tag is **required** on the five top-level artifacts -- this
contract is the version boundary, so an artifact that omits its version/type tag,
or supplies it only under the attribute name, must fail rather than be silently
accepted or tagged on dump.

Three load-bearing invariants are enforced here, not merely documented:
  - an evidence row is inadmissible without a non-blank ``quote`` and
    ``source_id`` (stops "writer invents research"), and a claim's ``source_id``
    must be non-blank (stops orphan claims);
  - an editorial audit may not recommend ``promote`` unless the deterministic
    copy-verification verdict is ``pass`` (the model cannot self-promote).

These are contracts only -- nothing here is wired into a runtime path yet. The
Action function that writes artifacts (Phase 2.2) and the Filter that enforces
these contracts (Phase 4.2) are separate slices. JSON schema export lives in
docs/schemas/ (generated, not hand-written).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


# The canonical "schema" key is the SOLE admission rule for the version tag,
# in both directions:
#   - no populate_by_name, so a raw artifact is accepted only under "schema"
#     (the "artifact_schema" attribute name is not a valid raw input key); and
#   - serialize_by_alias, so the default model_dump()/model_dump_json() emit
#     "schema" too (not "artifact_schema").
# One key both ways means a raw artifact always round-trips through model_for(),
# and the Phase 2.2 writer and Phase 4.2 filter cannot disagree on the key.
# extra='forbid' rejects any non-canonical key (including a reserved
# "artifact_schema" duplicate of the tag), so unmodeled worker output fails closed.
_BASE_CONFIG = ConfigDict(extra="forbid", serialize_by_alias=True)

# A string that must carry real content: whitespace is stripped, then a
# non-empty result is required. Used for citation fields whose blankness would
# silently defeat the "every claim is traceable" invariant.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Canonical advisory-warning strings + bounded grammar (#2181 round 5).
# The schema is the choke point: EVERY persisted v2 warning must be either a
# known static line or match the deterministic locator grammar (code +
# sentence number + short alphabetic keyword). Free text -- and therefore
# anything PII-shaped -- is unrepresentable, no matter which writer produced
# the artifact.
ADVISORY_CTA_REMINDER = (
    "reminder: confirm the CTA matches the channel and offer posture"
)
ADVISORY_OWNER_ROUTING_WARNING = (
    "owner-routing-coverage: draft explains the report shape but omits "
    "owner routing or who should review the fix"
)
_ADVISORY_STATIC_WARNINGS = frozenset(
    {ADVISORY_CTA_REMINDER, ADVISORY_OWNER_ROUTING_WARNING}
)
# Locator bound: up to 10 digits covers any physically possible draft (a
# sentence needs >= 2 characters, so 10^9 sentences implies a multi-GB
# body no worker response can carry) -- the producer can never emit a
# locator this grammar rejects.
_ADVISORY_GRAMMAR_RE = re.compile(
    r"^(?:unqualified-answer-claim|unqualified-ownership-claim): "
    r"sentence [1-9]\d{0,9}$"
)


def _routing_key(channel: str) -> str:
    """Comparison key for a routing label.

    NFKC first (canonically equivalent spellings are one channel to a reader),
    then DROP format and control characters -- zero-width joiners, bidi marks
    and the like occupy no visible width, so a label carrying them is
    indistinguishable from one that does not and must not read as a second,
    separate channel (#2192 round 9).
    """
    normalized = unicodedata.normalize("NFKC", channel)
    stripped = "".join(
        ch for ch in normalized if not unicodedata.category(ch) in ("Cf", "Cc")
    )
    return stripped.strip().casefold()


def _has_visible_content(text: str) -> bool:
    """True when ``text`` contains at least one VISIBLE character.

    ``str.strip()`` only removes ASCII/Unicode whitespace, so a
    producer-supplied zero-width space, soft hyphen, or bidi control passes
    a non-blank check while rendering as nothing. Phase 6 artifacts are
    shippable copy and renderer instructions, so "looks empty" must be
    treated as empty (#2192 round 4). Categories C* (control/format/
    surrogate/private-use/unassigned), Z* (separators) and M* (combining
    marks, e.g. a lone U+FE0F) cannot stand alone as content.
    """
    return any(
        not unicodedata.category(char).startswith(("C", "Z", "M")) for char in text
    )


VisibleStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(
        lambda value: value
        if _has_visible_content(value)
        else _raise_invisible()
    ),
]


def _raise_invisible():
    raise ValueError(
        "value must contain visible characters (invisible/format-only text "
        "is not content)"
    )


def _validate_advisory_warnings(warnings: "list[str]") -> None:
    """Persistence choke point for ALL writers (runner or direct) on EVERY
    artifact that carries advisory warnings.

    A warning must be a known static line or match the deterministic locator
    grammar -- free-text (and therefore PII-shaped) entries are rejected at
    validation, before the store persists anything. Kept as one function so
    the audit, the repurposing variants, and the image prompts cannot drift
    into three different grammars.
    """
    for warning in warnings:
        if warning in _ADVISORY_STATIC_WARNINGS:
            continue
        if _ADVISORY_GRAMMAR_RE.fullmatch(warning):
            continue
        raise ValueError(
            "advisory_warnings entries must be deterministic checklist "
            "lines (bounded locator grammar); free-text evidence is not "
            "representable"
        )


Confidence = Literal["high", "medium", "low"]
Recommendation = Literal["promote", "revise"]
Verdict = Literal["pass", "fail"]


class ContentBrief(BaseModel):
    """brief.json -- a rough request turned into a structured brief."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["content_brief.v1"] = Field(alias="schema")
    schema_version: int = 1
    project_id: str
    request_raw: str
    channel: str = ""
    audience: str = ""
    goal: str = ""
    angle: str = ""
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    target_length: str = ""
    open_questions: list[str] = Field(default_factory=list)
    created_by_model: Optional[str] = None
    prompt_version: Optional[str] = None


class EvidenceRow(BaseModel):
    """One cited evidence row. A row without a usable id, quote, and source_id is
    not admissible -- blank strings are rejected, not just missing keys."""

    model_config = _BASE_CONFIG

    # Non-blank: a blank id cannot be referenced by Claim.source_id, so it would
    # silently defeat the no-orphan-claim invariant.
    id: NonEmptyStr
    claim_candidate: str = ""
    quote: NonEmptyStr
    source_id: NonEmptyStr
    source_doc: Optional[str] = None
    confidence: Confidence = "medium"


class EvidencePacket(BaseModel):
    """evidence.json -- cited quotes from local knowledge; gaps, never fabrication."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["evidence_packet.v1"] = Field(alias="schema")
    schema_version: int = 1
    project_id: str
    evidence: list[EvidenceRow] = Field(default_factory=list)
    # Each gap must carry real content: a blank/whitespace gap is not a logged gap
    # and would otherwise let a packet slip past the honest-empty-packet guard below
    # with gaps=[''] -- the same blankness this contract rejects on citation fields.
    gaps: list[NonEmptyStr] = Field(default_factory=list)
    retrieved_from: list[str] = Field(default_factory=list)
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def _evidence_or_gaps_required(self) -> "EvidencePacket":
        # A packet with neither evidence nor gaps is not the honest "no evidence,
        # logged gaps" result -- it is indistinguishable from a truncated/empty
        # worker output. At least one must be present, and a gap must be a real
        # non-blank gap (enforced by the list[NonEmptyStr] type above), so gaps=['']
        # cannot defeat this guard.
        if not self.evidence and not self.gaps:
            raise ValueError(
                "an evidence packet must carry at least one evidence row or one "
                "non-blank gap; an empty packet cannot masquerade as an honest "
                "'no evidence' result"
            )
        return self


class Claim(BaseModel):
    """A factual claim in a draft, traced to an evidence row id."""

    model_config = _BASE_CONFIG

    text: str
    source_id: NonEmptyStr


class DraftArtifact(BaseModel):
    """draft.json -- the primary asset; every claim references an evidence id."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["draft.v1"] = Field(alias="schema")
    schema_version: int = 1
    project_id: str
    revision: int = 1
    body_markdown: str
    claims: list[Claim] = Field(default_factory=list)
    word_count: int = 0
    created_by_model: Optional[str] = None
    prompt_version: Optional[str] = None
    parent_evidence: Optional[str] = None


class CopyVerification(BaseModel):
    """Result of the deterministic copy_verification gate."""

    model_config = _BASE_CONFIG

    verdict: Verdict
    hits: list[str] = Field(default_factory=list)


class EditorialAudit(BaseModel):
    """audit.json (v1, FROZEN) -- voice edit + the verify verdict.

    Keeps the pre-#2181 class name and shape so existing consumers of
    ``EditorialAudit.model_validate(v1_payload)`` are unaffected. The v1
    shape is frozen so artifacts already on disk (and any rolled-back
    reader) keep validating byte-for-byte; ``advisory_warnings`` lives only
    on ``EditorialAuditV2`` (#2136 item 2). Do not add fields here.
    """

    model_config = _BASE_CONFIG

    artifact_schema: Literal["editorial_audit.v1"] = Field(alias="schema")
    schema_version: int = 1
    project_id: str
    draft_revision: int = 1
    edited_body_markdown: str = ""
    voice_pass: bool = False
    orphan_claims: list[str] = Field(default_factory=list)
    copy_verification: Optional[CopyVerification] = None
    recommendation: Recommendation = "revise"
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def _promote_requires_passing_verdict(self) -> "EditorialAudit":
        # The model cannot self-promote: a 'promote' recommendation is only valid
        # when the deterministic copy-verification verdict is 'pass'.
        if self.recommendation == "promote":
            cv = self.copy_verification
            if cv is None or cv.verdict != "pass":
                raise ValueError(
                    "recommendation 'promote' requires copy_verification.verdict == 'pass'"
                )
        return self


class EditorialAuditV2(BaseModel):
    """audit.json (v2, FROZEN) -- v1 plus the non-blocking advisory checklist.

    Frozen for the same reason as v1: v2 shipped in #2181, so artifacts
    already on disk and any rolled-back reader must keep validating
    byte-for-byte. ``extra="forbid"`` makes that symmetric -- a field added
    here would make every newly written audit UNREADABLE to a v2 consumer,
    not merely unknown to it. ``source_draft_fingerprint`` therefore lives
    on ``EditorialAuditV3`` (#2192 round 8). Do not add fields here.
    """

    model_config = _BASE_CONFIG

    artifact_schema: Literal["editorial_audit.v2"] = Field(alias="schema")
    schema_version: Literal[2] = 2
    project_id: str
    draft_revision: int = 1
    edited_body_markdown: str = ""
    voice_pass: bool = False
    orphan_claims: list[str] = Field(default_factory=list)
    copy_verification: Optional[CopyVerification] = None
    # Non-blocking reviewer checklist (#2136 item 2): deterministic advisory
    # warnings from the copy-verification module. Deliberately NOT referenced
    # by any validator -- warnings never gate the recommendation.
    advisory_warnings: list[str] = Field(default_factory=list)
    recommendation: Recommendation = "revise"
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def _advisory_warnings_bounded(self) -> "EditorialAuditV2":
        _validate_advisory_warnings(self.advisory_warnings)
        return self

    @model_validator(mode="after")
    def _promote_requires_passing_verdict(self) -> "EditorialAuditV2":
        # The model cannot self-promote: a 'promote' recommendation is only valid
        # when the deterministic copy-verification verdict is 'pass'.
        if self.recommendation == "promote":
            cv = self.copy_verification
            if cv is None or cv.verdict != "pass":
                raise ValueError(
                    "recommendation 'promote' requires copy_verification.verdict == 'pass'"
                )
        return self


class EditorialAuditV3(EditorialAuditV2):
    """audit.json (v3) -- v2 plus the runner-stamped draft content identity.

    v2 cannot express WHICH draft an approval covers. ``draft_revision`` is a
    producer-supplied label, and a same-revision rerun replaces the body
    while the number stays 1, so a stale approval carried over to copy
    nobody reviewed (#2192 round 7). v3 binds approval to content instead.

    A NEW VERSION rather than a v2 field: v2 shipped in #2181 and forbids
    extras, so adding the key there would make every newly written audit
    unreadable to a v2 consumer or a rolled-back reader -- the exact
    breakage v1's freeze note exists to prevent (#2192 round 8).

    Inherits v2's validators by subclassing so the promote-gate and the
    advisory-warning grammar cannot drift between the two versions.
    """

    artifact_schema: Literal["editorial_audit.v3"] = Field(alias="schema")
    schema_version: Literal[3] = 3
    # Stamped by the runner, never by the worker -- the same self-report
    # discipline as copy_verification. Optional so a direct writer can still
    # persist a v3 audit, but the runner's readiness gate REJECTS an
    # unstamped audit rather than treating absence as approval.
    source_draft_fingerprint: Optional[str] = None


class StageEntry(BaseModel):
    """One stage's row in the job manifest."""

    model_config = _BASE_CONFIG

    stage: str
    artifact: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    status: Optional[str] = None
    validation: Optional[str] = None
    seconds: Optional[int] = None


class Approval(BaseModel):
    """Human approval record. Nothing publishes while status != approved."""

    model_config = _BASE_CONFIG

    status: Literal["pending", "approved", "rejected"] = "pending"
    note: Optional[str] = None
    approved_at: Optional[str] = None


class ArtifactManifest(BaseModel):
    """manifest.json -- the job index: what each stage produced and its approval."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["manifest.v1"] = Field(alias="schema")
    schema_version: int = 1
    job_id: str
    project_id: str
    request: Optional[str] = None
    stages: list[StageEntry] = Field(default_factory=list)
    approval: Approval = Field(default_factory=Approval)
    sources_used: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class ChannelVariant(BaseModel):
    """One channel-specific rewrite of an approved draft.

    A variant is the copy that actually SHIPS, so it carries its own
    deterministic verdict rather than inheriting the draft's: a rewrite can
    introduce an overclaim the source never made.
    """

    model_config = _BASE_CONFIG

    # A channel must be VISIBLE, not merely non-blank: a label made only of
    # U+200B/U+200C/U+FE0F satisfies "non-empty" while being unroutable, and
    # several distinct invisible labels also slip past the duplicate check as
    # different strings. Same class as the body/prompt fix, applied to the
    # routing identifier (#2192 round 9).
    channel: VisibleStr
    body_markdown: VisibleStr
    # Claim lineage back to the source draft's claims/evidence ids. REQUIRED
    # and non-empty: a variant with no lineage is an orphan -- it asserts
    # something the approved draft never established, which is exactly the
    # "repurposer invents claims" failure this field exists to prevent. A
    # default would make that orphan representable (review round 1).
    derived_from_claims: list[NonEmptyStr] = Field(min_length=1)
    copy_verification: Optional[CopyVerification] = None
    advisory_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _advisory_warnings_bounded(self) -> "ChannelVariant":
        _validate_advisory_warnings(self.advisory_warnings)
        return self


class RepurposingPackage(BaseModel):
    """repurposing.json -- channel variants derived from an approved draft.

    ``ready_to_publish`` is the variant-level analogue of the editorial
    audit's promote gate: the model cannot declare a package shippable while
    any variant carries a failing deterministic verdict.
    """

    model_config = _BASE_CONFIG

    artifact_schema: Literal["repurposing.v1"] = Field(alias="schema")
    schema_version: Literal[1] = 1
    project_id: str
    source_draft_revision: int = 1
    variants: list[ChannelVariant] = Field(default_factory=list)
    ready_to_publish: bool = False
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def _package_invariants(self) -> "RepurposingPackage":
        # An empty package is not a repurposing result; it is a no-op that
        # would otherwise persist as if work had been done.
        if not self.variants:
            raise ValueError("repurposing package requires at least one variant")
        # One channel per variant: two variants on the same channel means an
        # ambiguous "which one ships?" downstream.
        # NFKC first: canonically equivalent spellings (NFC vs NFD) are the
        # same channel to a reader, so casefold alone leaves the very
        # ambiguity this check exists to prevent (round 5).
        # Zero-width and control characters are dropped before comparing: they
        # carry no visible width, so "email" and "email​" are the SAME
        # channel to a reader and to anything routing on the label, and
        # keeping them made a visually identical duplicate pass (round 9).
        channels = [_routing_key(variant.channel) for variant in self.variants]
        if len(channels) != len(set(channels)):
            raise ValueError("duplicate channel in repurposing variants")
        if self.ready_to_publish:
            for variant in self.variants:
                verdict = variant.copy_verification
                if verdict is None or verdict.verdict != "pass":
                    raise ValueError(
                        "ready_to_publish requires every variant to carry "
                        "copy_verification.verdict == 'pass'"
                    )
        return self


class ImagePrompt(BaseModel):
    """One text-to-image prompt. TEXT ONLY -- generation is a separate,
    human-triggered, VRAM-guarded step (epic #2109 Phase 6 keeps the prompt
    designer and the generator split)."""

    model_config = _BASE_CONFIG

    purpose: NonEmptyStr
    prompt_text: VisibleStr
    negative_prompt: str = ""
    aspect_ratio: str = "1:1"


class ImagePromptSet(BaseModel):
    """image_prompt.json -- image prompts derived from an approved draft.

    Prompt text is gated like body copy: a diffusion model will happily
    render a banned claim or a contact string INTO the artwork, where no
    downstream text check would ever see it.
    """

    model_config = _BASE_CONFIG

    artifact_schema: Literal["image_prompt.v1"] = Field(alias="schema")
    schema_version: Literal[1] = 1
    project_id: str
    source_draft_revision: int = 1
    prompts: list[ImagePrompt] = Field(default_factory=list)
    copy_verification: Optional[CopyVerification] = None
    advisory_warnings: list[str] = Field(default_factory=list)
    # The generation gate, mirroring RepurposingPackage.ready_to_publish: a
    # set may not be marked renderable while its deterministic verdict
    # fails. Recording a verdict nobody gates on is verification, not a gate
    # (review round 1).
    ready_to_generate: bool = False
    prompt_version: Optional[str] = None

    @model_validator(mode="after")
    def _prompt_set_invariants(self) -> "ImagePromptSet":
        if not self.prompts:
            raise ValueError("image prompt set requires at least one prompt")
        if self.ready_to_generate:
            verdict = self.copy_verification
            if verdict is None or verdict.verdict != "pass":
                raise ValueError(
                    "ready_to_generate requires copy_verification.verdict == 'pass'"
                )
        _validate_advisory_warnings(self.advisory_warnings)
        return self


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "content_brief.v1": ContentBrief,
    "evidence_packet.v1": EvidencePacket,
    "draft.v1": DraftArtifact,
    "editorial_audit.v1": EditorialAudit,
    "editorial_audit.v2": EditorialAuditV2,
    "editorial_audit.v3": EditorialAuditV3,
    "manifest.v1": ArtifactManifest,
    "repurposing.v1": RepurposingPackage,
    "image_prompt.v1": ImagePromptSet,
}


def model_for(data: dict[str, Any]) -> type[BaseModel]:
    """Return the contract class for an artifact dict, keyed by its "schema" tag."""

    tag = data.get("schema")
    if tag not in ARTIFACT_MODELS:
        raise ValueError(f"unknown artifact schema tag: {tag!r}")
    return ARTIFACT_MODELS[tag]
