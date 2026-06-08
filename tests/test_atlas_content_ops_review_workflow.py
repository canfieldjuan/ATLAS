from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from atlas_brain._content_ops_review_workflow import (
    ContentOpsReviewRequest,
    TenantClaimRegistryReadError,
    run_content_ops_review,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.claims_map import ClaimStatus, ExtractedClaim, RegistryClaim
from extracted_content_pipeline.content_pr import (
    CommentCategory,
    CoverageRow,
    CoverageStatus,
    ReviewComment,
    RulePacketVersions,
)
from extracted_content_pipeline.review_contract import ReviewDecision, RiskTier


_AS_OF = date(2026, 6, 7)
_PINNED = RulePacketVersions(
    brief="brief-v1",
    brand_voice="voice-v1",
    claim_registry="claims-v1",
    compliance="compliance-v1",
    channel_schema="channel-v1",
)


@dataclass
class _RegistryReader:
    registry: dict[str, RegistryClaim]
    scopes: list[TenantScope]

    async def list_registry_claims(self, *, scope: TenantScope):
        self.scopes.append(scope)
        return self.registry


@dataclass
class _FailingRegistryReader:
    scopes: list[TenantScope]

    async def list_registry_claims(self, *, scope: TenantScope):
        self.scopes.append(scope)
        raise TenantClaimRegistryReadError("claim registry read failed")


def _reader() -> _RegistryReader:
    return _RegistryReader(
        registry={
            "pricing.discount": RegistryClaim(
                id="pricing.discount",
                approved_wording="Save up to 30% on eligible annual plans",
                risk_tier=RiskTier.HIGH,
                expiration=date(2026, 1, 31),
            ),
            "feature.sso": RegistryClaim(
                id="feature.sso",
                approved_wording="SSO is included on every plan",
                risk_tier=RiskTier.MEDIUM,
            ),
        },
        scopes=[],
    )


def _failing_reader() -> _FailingRegistryReader:
    return _FailingRegistryReader(scopes=[])


def _pass_row(rule_id: str = "VOICE-01") -> CoverageRow:
    return CoverageRow(
        rule_id=rule_id,
        requirement="Rule must be evidenced",
        status=CoverageStatus.PASS,
        evidence="quoted draft span",
    )


def _request(**overrides) -> ContentOpsReviewRequest:
    values = {
        "asset_id": "asset-1",
        "rule_packet": _PINNED,
        "coverage": (_pass_row(),),
        "extracted_claims": (
            ExtractedClaim(
                text="SSO is included on every plan",
                location="hero",
                registry_id="feature.sso",
            ),
        ),
        "comments": (),
        "as_of": _AS_OF,
    }
    values.update(overrides)
    return ContentOpsReviewRequest(**values)


@pytest.mark.asyncio
async def test_missing_tenant_scope_blocks_without_reading_registry() -> None:
    reader = _reader()

    result = await run_content_ops_review(
        _request(),
        scope=TenantScope(account_id=" "),
        registry_reader=reader,
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert result.reasons == ("tenant scope required",)
    assert result.mapped_claims == ()
    assert reader.scopes == []


@pytest.mark.asyncio
async def test_registry_reader_receives_tenant_scope_not_request_account_id() -> None:
    reader = _reader()
    scope = TenantScope(account_id="acct-1", user_id="user-1")

    result = await run_content_ops_review(
        _request(),
        scope=scope,
        registry_reader=reader,
    )

    assert result.decision == ReviewDecision.APPROVED
    assert reader.scopes == [scope]
    assert result.content_pr.asset_id == "asset-1"
    assert result.mapped_claims[0].status == ClaimStatus.MATCH


@pytest.mark.asyncio
async def test_registry_reader_failure_blocks_review() -> None:
    reader = _failing_reader()
    scope = TenantScope(account_id="acct-1")

    result = await run_content_ops_review(
        _request(),
        scope=scope,
        registry_reader=reader,
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert result.reasons == ("claim registry read failed",)
    assert result.mapped_claims == ()
    assert reader.scopes == [scope]


@pytest.mark.asyncio
async def test_result_as_dict_is_tool_shaped() -> None:
    result = await run_content_ops_review(
        _request(),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    payload = result.as_dict()

    assert payload["ok"] is True
    assert payload["decision"] == "approved"
    assert payload["reasons"] == []
    assert payload["mapped_claims"][0]["status"] == "match"
    assert payload["mapped_claims"][0]["risk_tier"] == "medium"
    assert payload["content_pr"]["asset_id"] == "asset-1"
    assert payload["content_pr"]["coverage"][0]["status"] == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("review_request", "expected_reason"),
    [
        (
            _request(coverage=()),
            "missing coverage matrix",
        ),
        (
            _request(coverage=(CoverageRow(rule_id="VOICE-02"),)),
            "unresolved required coverage",
        ),
    ],
)
async def test_incomplete_review_blocks(
    review_request: ContentOpsReviewRequest,
    expected_reason: str,
) -> None:
    result = await run_content_ops_review(
        review_request,
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert any(expected_reason in reason for reason in result.reasons)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "review_request",
    [
        _request(
            coverage=(
                CoverageRow(
                    rule_id="CLAIM-01",
                    status=CoverageStatus.FAIL,
                    evidence="discount overclaim",
                ),
            ),
        ),
        _request(
            coverage=(
                CoverageRow(
                    rule_id="CLAIM-01",
                    status="fail",
                    evidence="decoded string status",
                ),
            ),
        ),
    ],
)
async def test_failed_coverage_requires_revision(
    review_request: ContentOpsReviewRequest,
) -> None:
    result = await run_content_ops_review(
        review_request,
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.REVISION_REQUIRED
    assert any("failed required coverage" in reason for reason in result.reasons)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim",
    [
        ExtractedClaim(
            text="Save 30% on all plans",
            location="hero",
            registry_id="pricing.discount",
        ),
        ExtractedClaim(
            text="Save up to 30% on eligible annual plans",
            location="hero",
            registry_id="pricing.discount",
        ),
    ],
)
async def test_blocking_claims_require_revision(claim: ExtractedClaim) -> None:
    result = await run_content_ops_review(
        _request(extracted_claims=(claim,)),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.REVISION_REQUIRED
    assert any("blocking claim" in reason for reason in result.reasons)


@pytest.mark.asyncio
async def test_blocking_comment_requires_revision() -> None:
    result = await run_content_ops_review(
        _request(
            comments=(
                ReviewComment(
                    category=CommentCategory.BRIEF,
                    message="Promise does not match brief",
                    evidence="brief primary promise",
                    blocking=True,
                ),
            ),
        ),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.REVISION_REQUIRED
    assert any("blocking comment" in reason for reason in result.reasons)
