"""Host wiring for the Content Ops review-contract engine.

The extracted package owns the deterministic review logic. This module is the
first host callable around it: tenant scope comes from Atlas, approved claims
come from an injected tenant registry reader, and the verdict still delegates to
the pure Content-PR engine.

No DB, FastAPI, MCP, or LLM code lives here. Those layers should wrap this
service instead of reimplementing review logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Protocol, Sequence

from extracted_content_pipeline.adversarial_pass import (
    AdversarialPass,
    comment_from_finding,
    corroborated_categories_across,
)
from extracted_content_pipeline.calibration_anchors import anchors_for_finding_categories
from extracted_content_pipeline.calibration_library import (
    CalibrationExample,
    CalibrationLibrary,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.claims_map import (
    ExtractedClaim,
    MappedClaim,
    RegistryClaim,
    build_claims_map,
)
from extracted_content_pipeline.content_pr import (
    ContentPR,
    CoverageRow,
    ReviewComment,
    RulePacketVersions,
    review_verdict,
    verdict_reasons,
)
from extracted_content_pipeline.coverage_rows import (
    brand_voice_coverage_rows,
    quality_gate_coverage_rows,
)
from extracted_content_pipeline.review_contract import ReviewDecision


logger = logging.getLogger("atlas.content_ops_review_workflow")


class TenantClaimRegistryReadError(RuntimeError):
    """Raised when tenant registry data cannot be read safely."""


class TenantClaimRegistryReader(Protocol):
    """Read approved marketing claims for one tenant scope."""

    async def list_registry_claims(
        self,
        *,
        scope: TenantScope,
    ) -> Mapping[str, RegistryClaim]:
        """Return registry claims keyed by registry id."""


class TenantCalibrationLibraryReader(Protocol):
    """Read curated calibration examples for one tenant scope.

    Symmetric to :class:`TenantClaimRegistryReader`, with one deliberate
    difference: calibration anchors are *evidence*, not a gate, so a read
    failure must degrade (fall back to request-supplied anchors), never block
    the verdict. The review treats any exception from this reader as "no
    server-side anchors available" rather than a hard error.
    """

    async def list_calibration_examples(
        self,
        *,
        scope: TenantScope,
    ) -> tuple[CalibrationExample, ...]:
        """Return the tenant's curated calibration examples."""


class ContentOpsAccountResolver(Protocol):
    """Resolve the connector-bound tenant account for one service call."""

    async def resolve_account_id(self) -> Any:
        """Return the bound account ID, or a missing value when unavailable."""


@dataclass(frozen=True)
class ContentOpsReviewRequest:
    """Structured request a future marketer MCP tool can pass through."""

    asset_id: str = ""
    rule_packet: RulePacketVersions = RulePacketVersions()
    coverage: tuple[CoverageRow, ...] = ()
    quality_reports: tuple[Any, ...] = ()
    brand_voice_payload: Mapping[str, Any] | None = None
    extracted_claims: tuple[ExtractedClaim, ...] = ()
    comments: tuple[ReviewComment, ...] = ()
    adversarial_passes: tuple[AdversarialPass, ...] = ()
    calibration_examples: tuple[CalibrationExample, ...] = ()
    as_of: date | None = None


async def run_content_ops_review_for_bound_tenant(
    request: ContentOpsReviewRequest,
    *,
    account_resolver: ContentOpsAccountResolver,
    registry_reader: TenantClaimRegistryReader,
    calibration_reader: TenantCalibrationLibraryReader | None = None,
) -> ContentOpsReviewResult:
    """Resolve connector tenant binding before running the review service."""

    try:
        account_id = await account_resolver.resolve_account_id()
    except Exception:
        return _blocked_result(request, "tenant binding resolution failed")

    return await run_content_ops_review(
        request,
        scope=_tenant_scope_from_account_binding(account_id),
        registry_reader=registry_reader,
        calibration_reader=calibration_reader,
    )


@dataclass(frozen=True)
class ContentOpsReviewResult:
    """Tool-shaped review result plus the Content-PR envelope."""

    decision: ReviewDecision
    reasons: tuple[str, ...]
    mapped_claims: tuple[MappedClaim, ...]
    content_pr: ContentPR
    calibration_anchors: tuple[CalibrationExample, ...] = ()
    corroborated_objection_categories: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible shape for future transport wrappers."""

        return {
            "ok": self.decision == ReviewDecision.APPROVED,
            "decision": _value(self.decision),
            "reasons": list(self.reasons),
            "mapped_claims": [_mapped_claim_as_dict(claim) for claim in self.mapped_claims],
            "content_pr": {
                "asset_id": self.content_pr.asset_id,
                "rule_packet": {
                    "brief": self.content_pr.rule_packet.brief,
                    "brand_voice": self.content_pr.rule_packet.brand_voice,
                    "claim_registry": self.content_pr.rule_packet.claim_registry,
                    "compliance": self.content_pr.rule_packet.compliance,
                    "channel_schema": self.content_pr.rule_packet.channel_schema,
                },
                "coverage": [_coverage_row_as_dict(row) for row in self.content_pr.coverage],
                "comments": [_comment_as_dict(comment) for comment in self.content_pr.comments],
            },
            "calibration_anchors": [
                _calibration_anchor_as_dict(anchor) for anchor in self.calibration_anchors
            ],
            "corroborated_objection_categories": list(self.corroborated_objection_categories),
        }


async def run_content_ops_review(
    request: ContentOpsReviewRequest,
    *,
    scope: TenantScope | None,
    registry_reader: TenantClaimRegistryReader,
    calibration_reader: TenantCalibrationLibraryReader | None = None,
) -> ContentOpsReviewResult:
    """Build a tenant claims map and compute the deterministic review verdict."""

    if not _scope_account_id(scope):
        return _blocked_result(request, "tenant scope required")

    try:
        registry = await registry_reader.list_registry_claims(scope=scope)
    except TenantClaimRegistryReadError as exc:
        return _blocked_result(request, str(exc) or "claim registry read failed")
    mapped_claims = build_claims_map(
        request.extracted_claims,
        registry,
        as_of=request.as_of or date.today(),
    )
    coverage = _coverage_rows_for_request(request)
    content_pr = ContentPR(
        asset_id=request.asset_id,
        rule_packet=request.rule_packet,
        coverage=coverage,
        claims=mapped_claims,
        comments=_comments_for_request(request),
    )
    decision = review_verdict(content_pr)
    merged_examples = await _merged_calibration_examples(request, scope, calibration_reader)
    return ContentOpsReviewResult(
        decision=decision,
        reasons=verdict_reasons(content_pr),
        mapped_claims=mapped_claims,
        content_pr=content_pr,
        calibration_anchors=_calibration_anchors_for_examples(merged_examples, request),
        corroborated_objection_categories=_corroborated_categories_for_request(request),
    )


def _blocked_result(
    request: ContentOpsReviewRequest,
    reason: str,
) -> ContentOpsReviewResult:
    content_pr = ContentPR(
        asset_id=request.asset_id,
        rule_packet=request.rule_packet,
        coverage=_coverage_rows_for_request(request),
        claims=(),
        comments=_comments_for_request(request),
    )
    return ContentOpsReviewResult(
        decision=ReviewDecision.BLOCKED,
        reasons=(reason,),
        mapped_claims=(),
        content_pr=content_pr,
        calibration_anchors=_calibration_anchors_for_request(request),
        corroborated_objection_categories=_corroborated_categories_for_request(request),
    )


def _corroborated_categories_for_request(
    request: ContentOpsReviewRequest,
) -> tuple[str, ...]:
    """Objection categories raised by two or more independent adversarial passes.

    The strongest signal in the verify result: when two passes independently
    flag the same failure mode, the editor should weight that objection highest.
    Returned as sorted category values for a stable, JSON-friendly surface.
    """

    passes = tuple(
        pass_ for pass_ in _items(request.adversarial_passes)
        if isinstance(pass_, AdversarialPass)
    )
    corroborated = corroborated_categories_across(passes)
    return tuple(sorted(_value(category) for category in corroborated))


async def _merged_calibration_examples(
    request: ContentOpsReviewRequest,
    scope: TenantScope,
    calibration_reader: TenantCalibrationLibraryReader | None,
) -> tuple[CalibrationExample, ...]:
    """Tenant server-side anchors unioned with request-supplied ones.

    The tenant's curated set is canonical (server-first), so a server anchor wins
    over a request anchor with the same ``example_id``; the connector can still
    supplement with new ids. Calibration anchors are evidence, not a gate, so any
    reader failure degrades to request-supplied anchors -- it never blocks or
    raises.
    """

    if calibration_reader is None:
        return request.calibration_examples
    try:
        server_examples = await calibration_reader.list_calibration_examples(scope=scope)
    except Exception:
        logger.warning(
            "calibration library read failed; using request-supplied anchors",
            exc_info=True,
        )
        return request.calibration_examples

    merged: list[CalibrationExample] = []
    seen_ids: set[str] = set()
    for example in tuple(server_examples) + tuple(request.calibration_examples):
        if not isinstance(example, CalibrationExample) or example.example_id in seen_ids:
            continue
        seen_ids.add(example.example_id)
        merged.append(example)
    return tuple(merged)


def _calibration_anchors_for_request(
    request: ContentOpsReviewRequest,
) -> tuple[CalibrationExample, ...]:
    """Request-supplied anchors only (blocked-result path; no reader consulted)."""

    return _calibration_anchors_for_examples(request.calibration_examples, request)


def _calibration_anchors_for_examples(
    examples: tuple[CalibrationExample, ...],
    request: ContentOpsReviewRequest,
) -> tuple[CalibrationExample, ...]:
    """Anchors illustrating the failure modes the adversarial passes raised.

    Builds a library from ``examples`` and selects the teachable anchors whose
    label maps to a fired finding category, so the editor sees a worked example
    of each failure mode the draft tripped. Returns nothing when no anchors are
    available or no fired category maps.
    """

    if not examples:
        return ()
    fired_categories: list[object] = []
    for pass_ in _items(request.adversarial_passes):
        if not isinstance(pass_, AdversarialPass):
            continue
        for finding in pass_.substantiated():
            fired_categories.append(finding.category)
    library = CalibrationLibrary(examples=tuple(examples))
    return anchors_for_finding_categories(library, fired_categories)


def _scope_account_id(scope: TenantScope | None) -> str:
    value = getattr(scope, "account_id", None)
    if not isinstance(value, str):
        return ""
    return value.strip()


def _tenant_scope_from_account_binding(account_id: Any) -> TenantScope | None:
    account = account_id.strip() if isinstance(account_id, str) else ""
    if not account:
        return None
    return TenantScope(account_id=account)


def _comments_for_request(request: ContentOpsReviewRequest) -> tuple[ReviewComment, ...]:
    """Explicit reviewer comments plus the adversarial-pass findings as comments.

    Explicit comments keep their order; adversarial-derived comments follow.
    Every adversarial comment is non-blocking (see ``comment_from_finding``), so
    folding them in never changes the verdict on its own -- they are evidence the
    editor reads, not a gate (the doc's "still not a judge").
    """

    return tuple(request.comments) + _adversarial_comments(request.adversarial_passes)


def _adversarial_comments(
    passes: Sequence[AdversarialPass],
) -> tuple[ReviewComment, ...]:
    """Convert each pass's substantiated findings into never-blocking comments.

    Only substantiated findings (carrying both an objection and evidence) are
    folded; an empty/decoration finding would add a bare ``[adversarial:x]``
    comment with no objection, so it is dropped to keep the result signal-dense.
    """

    comments: list[ReviewComment] = []
    for pass_ in _items(passes):
        if not isinstance(pass_, AdversarialPass):
            continue
        for finding in pass_.substantiated():
            comments.append(comment_from_finding(finding))
    return tuple(comments)


def _coverage_rows_for_request(request: ContentOpsReviewRequest) -> tuple[CoverageRow, ...]:
    rows = list(request.coverage or ())
    for report in _items(request.quality_reports):
        rows.extend(quality_gate_coverage_rows(report))
    if request.brand_voice_payload is not None:
        rows.extend(brand_voice_coverage_rows(request.brand_voice_payload))
    return tuple(rows)


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _mapped_claim_as_dict(claim: MappedClaim) -> dict[str, Any]:
    return {
        "text": claim.text,
        "location": claim.location,
        "registry_id": claim.registry_id,
        "approved_wording": claim.approved_wording,
        "status": _value(claim.status),
        "risk_tier": _value(claim.risk_tier),
    }


def _coverage_row_as_dict(row: CoverageRow) -> dict[str, Any]:
    return {
        "rule_id": row.rule_id,
        "requirement": row.requirement,
        "required": row.required,
        "status": _value(row.status),
        "evidence": row.evidence,
    }


def _comment_as_dict(comment: ReviewComment) -> dict[str, Any]:
    return {
        "category": _value(comment.category),
        "message": comment.message,
        "evidence": comment.evidence,
        "blocking": comment.blocking,
    }


def _calibration_anchor_as_dict(anchor: CalibrationExample) -> dict[str, Any]:
    return {
        "example_id": anchor.example_id,
        "label": _value(anchor.label),
        "excerpt": anchor.excerpt,
        "reasoning": anchor.reasoning,
        "source": anchor.source,
    }


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


__all__ = [
    "ContentOpsAccountResolver",
    "ContentOpsReviewRequest",
    "ContentOpsReviewResult",
    "TenantCalibrationLibraryReader",
    "TenantClaimRegistryReadError",
    "TenantClaimRegistryReader",
    "run_content_ops_review",
    "run_content_ops_review_for_bound_tenant",
]
