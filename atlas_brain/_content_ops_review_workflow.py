"""Host wiring for the Content Ops review-contract engine.

The extracted package owns the deterministic review logic. This module is the
first host callable around it: tenant scope comes from Atlas, approved claims
come from an injected tenant registry reader, and the verdict still delegates to
the pure Content-PR engine.

No DB, FastAPI, MCP, or LLM code lives here. Those layers should wrap this
service instead of reimplementing review logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Protocol

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
from extracted_content_pipeline.review_contract import ReviewDecision


class TenantClaimRegistryReader(Protocol):
    """Read approved marketing claims for one tenant scope."""

    async def list_registry_claims(
        self,
        *,
        scope: TenantScope,
    ) -> Mapping[str, RegistryClaim]:
        """Return registry claims keyed by registry id."""


@dataclass(frozen=True)
class ContentOpsReviewRequest:
    """Structured request a future marketer MCP tool can pass through."""

    asset_id: str = ""
    rule_packet: RulePacketVersions = RulePacketVersions()
    coverage: tuple[CoverageRow, ...] = ()
    extracted_claims: tuple[ExtractedClaim, ...] = ()
    comments: tuple[ReviewComment, ...] = ()
    as_of: date | None = None


@dataclass(frozen=True)
class ContentOpsReviewResult:
    """Tool-shaped review result plus the Content-PR envelope."""

    decision: ReviewDecision
    reasons: tuple[str, ...]
    mapped_claims: tuple[MappedClaim, ...]
    content_pr: ContentPR

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
        }


async def run_content_ops_review(
    request: ContentOpsReviewRequest,
    *,
    scope: TenantScope | None,
    registry_reader: TenantClaimRegistryReader,
) -> ContentOpsReviewResult:
    """Build a tenant claims map and compute the deterministic review verdict."""

    if not _scope_account_id(scope):
        return _blocked_result(request, "tenant scope required")

    registry = await registry_reader.list_registry_claims(scope=scope)
    mapped_claims = build_claims_map(
        request.extracted_claims,
        registry,
        as_of=request.as_of or date.today(),
    )
    content_pr = ContentPR(
        asset_id=request.asset_id,
        rule_packet=request.rule_packet,
        coverage=request.coverage,
        claims=mapped_claims,
        comments=request.comments,
    )
    decision = review_verdict(content_pr)
    return ContentOpsReviewResult(
        decision=decision,
        reasons=verdict_reasons(content_pr),
        mapped_claims=mapped_claims,
        content_pr=content_pr,
    )


def _blocked_result(
    request: ContentOpsReviewRequest,
    reason: str,
) -> ContentOpsReviewResult:
    content_pr = ContentPR(
        asset_id=request.asset_id,
        rule_packet=request.rule_packet,
        coverage=request.coverage,
        claims=(),
        comments=request.comments,
    )
    return ContentOpsReviewResult(
        decision=ReviewDecision.BLOCKED,
        reasons=(reason,),
        mapped_claims=(),
        content_pr=content_pr,
    )


def _scope_account_id(scope: TenantScope | None) -> str:
    value = getattr(scope, "account_id", None)
    if not isinstance(value, str):
        return ""
    return value.strip()


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


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


__all__ = [
    "ContentOpsReviewRequest",
    "ContentOpsReviewResult",
    "TenantClaimRegistryReader",
    "run_content_ops_review",
]
