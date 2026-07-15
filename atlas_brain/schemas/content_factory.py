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
``extra='allow'`` and ``schema_version=1``. ``extra='allow'`` lets artifacts
written before a field was added round-trip without validation errors while the
pipeline is still iterating; flip to ``extra='forbid'`` in a later slice once the
shapes are stable. The artifact's type tag is stored under the JSON key
``"schema"`` via an alias (the attribute is ``artifact_schema`` to avoid shadowing
``BaseModel.schema``); ``populate_by_name=True`` accepts either name on input.

These are contracts only -- nothing here is wired into a runtime path yet. The
Action function that writes artifacts (Phase 2.2) and the Filter that enforces
these contracts (Phase 4.2) are separate slices. JSON schema export lives in
docs/schemas/ (generated, not hand-written).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


_BASE_CONFIG = ConfigDict(extra="allow", populate_by_name=True)

Confidence = Literal["high", "medium", "low"]
Recommendation = Literal["promote", "revise"]
Verdict = Literal["pass", "fail"]


class ContentBrief(BaseModel):
    """brief.json -- a rough request turned into a structured brief."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["content_brief.v1"] = Field(
        "content_brief.v1", alias="schema"
    )
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
    """One cited evidence row. A row without a source_id is not admissible."""

    model_config = _BASE_CONFIG

    id: str
    claim_candidate: str = ""
    quote: str
    source_id: str
    source_doc: Optional[str] = None
    confidence: Confidence = "medium"


class EvidencePacket(BaseModel):
    """evidence.json -- cited quotes from local knowledge; gaps, never fabrication."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["evidence_packet.v1"] = Field(
        "evidence_packet.v1", alias="schema"
    )
    schema_version: int = 1
    project_id: str
    evidence: list[EvidenceRow] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    retrieved_from: list[str] = Field(default_factory=list)
    prompt_version: Optional[str] = None


class Claim(BaseModel):
    """A factual claim in a draft, traced to an evidence row id."""

    model_config = _BASE_CONFIG

    text: str
    source_id: str


class DraftArtifact(BaseModel):
    """draft.json -- the primary asset; every claim references an evidence id."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["draft.v1"] = Field("draft.v1", alias="schema")
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
    """audit.json -- voice edit + the verify verdict; the model cannot self-promote."""

    model_config = _BASE_CONFIG

    artifact_schema: Literal["editorial_audit.v1"] = Field(
        "editorial_audit.v1", alias="schema"
    )
    schema_version: int = 1
    project_id: str
    draft_revision: int = 1
    edited_body_markdown: str = ""
    voice_pass: bool = False
    orphan_claims: list[str] = Field(default_factory=list)
    copy_verification: Optional[CopyVerification] = None
    recommendation: Recommendation = "revise"
    prompt_version: Optional[str] = None


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

    artifact_schema: Literal["manifest.v1"] = Field("manifest.v1", alias="schema")
    schema_version: int = 1
    job_id: str
    project_id: str
    request: Optional[str] = None
    stages: list[StageEntry] = Field(default_factory=list)
    approval: Approval = Field(default_factory=Approval)
    sources_used: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "content_brief.v1": ContentBrief,
    "evidence_packet.v1": EvidencePacket,
    "draft.v1": DraftArtifact,
    "editorial_audit.v1": EditorialAudit,
    "manifest.v1": ArtifactManifest,
}


def model_for(data: dict[str, Any]) -> type[BaseModel]:
    """Return the contract class for an artifact dict, keyed by its "schema" tag."""

    tag = data.get("schema")
    if tag not in ARTIFACT_MODELS:
        raise ValueError(f"unknown artifact schema tag: {tag!r}")
    return ARTIFACT_MODELS[tag]
