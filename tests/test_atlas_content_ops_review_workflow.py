from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from atlas_brain._content_ops_review_workflow import (
    ContentOpsReviewRequest,
    TenantClaimRegistryReadError,
    run_content_ops_review,
    run_content_ops_review_for_bound_tenant,
)
from extracted_content_pipeline.adversarial_pass import (
    AdversarialFinding,
    AdversarialFindingCategory,
    AdversarialPass,
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
from extracted_quality_gate.types import (
    GateDecision,
    GateFinding,
    GateSeverity,
    QualityReport,
)


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


@dataclass
class _AccountResolver:
    account_id: object
    calls: int = 0

    async def resolve_account_id(self):
        self.calls += 1
        return self.account_id


@dataclass
class _FailingAccountResolver:
    calls: int = 0

    async def resolve_account_id(self):
        self.calls += 1
        raise RuntimeError("oauth token lookup failed")


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
async def test_bound_tenant_review_uses_resolved_account_scope() -> None:
    reader = _reader()
    resolver = _AccountResolver(" acct-1 ")

    result = await run_content_ops_review_for_bound_tenant(
        _request(),
        account_resolver=resolver,
        registry_reader=reader,
    )

    assert result.decision == ReviewDecision.APPROVED
    assert resolver.calls == 1
    assert reader.scopes == [TenantScope(account_id="acct-1")]
    assert result.mapped_claims[0].status == ClaimStatus.MATCH


@pytest.mark.asyncio
@pytest.mark.parametrize("account_id", [None, "", " ", 123])
async def test_bound_tenant_review_blocks_missing_or_malformed_binding(
    account_id: object,
) -> None:
    reader = _reader()
    resolver = _AccountResolver(account_id)

    result = await run_content_ops_review_for_bound_tenant(
        _request(),
        account_resolver=resolver,
        registry_reader=reader,
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert result.reasons == ("tenant scope required",)
    assert result.mapped_claims == ()
    assert resolver.calls == 1
    assert reader.scopes == []


@pytest.mark.asyncio
async def test_bound_tenant_review_blocks_resolver_failure_before_registry() -> None:
    reader = _reader()
    resolver = _FailingAccountResolver()

    result = await run_content_ops_review_for_bound_tenant(
        _request(),
        account_resolver=resolver,
        registry_reader=reader,
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert result.reasons == ("tenant binding resolution failed",)
    assert result.mapped_claims == ()
    assert resolver.calls == 1
    assert reader.scopes == []


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
async def test_quality_report_can_supply_required_coverage() -> None:
    result = await run_content_ops_review(
        _request(
            coverage=(),
            quality_reports=(QualityReport(passed=True, decision=GateDecision.PASS),),
        ),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.APPROVED
    assert [row.rule_id for row in result.content_pr.coverage] == [
        "QUALITY-GATE:report",
    ]
    assert result.content_pr.coverage[0].status == CoverageStatus.PASS


@pytest.mark.asyncio
async def test_caller_supplied_coverage_is_preserved_before_quality_rows() -> None:
    result = await run_content_ops_review(
        _request(
            coverage=(_pass_row("MANUAL-VOICE"),),
            quality_reports=(QualityReport(passed=True, decision=GateDecision.PASS),),
        ),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.APPROVED
    assert [row.rule_id for row in result.content_pr.coverage] == [
        "MANUAL-VOICE",
        "QUALITY-GATE:report",
    ]


@pytest.mark.asyncio
async def test_quality_report_blocker_requires_revision() -> None:
    result = await run_content_ops_review(
        _request(
            coverage=(),
            quality_reports=(
                QualityReport(
                    passed=False,
                    decision=GateDecision.BLOCK,
                    findings=(
                        GateFinding(
                            code="no_cta",
                            message="CTA is missing",
                            severity=GateSeverity.BLOCKER,
                            field_name="cta",
                        ),
                    ),
                ),
            ),
        ),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.REVISION_REQUIRED
    assert result.content_pr.coverage[0].rule_id == "QUALITY-GATE:no-cta"
    assert any("failed required coverage" in reason for reason in result.reasons)


@pytest.mark.asyncio
async def test_malformed_quality_evidence_blocks_as_unresolved_coverage() -> None:
    result = await run_content_ops_review(
        _request(coverage=(), quality_reports=(None,)),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert result.content_pr.coverage[0].rule_id == "QUALITY-GATE:report"
    assert result.content_pr.coverage[0].status == CoverageStatus.UNRESOLVED
    assert any("unresolved required coverage" in reason for reason in result.reasons)


@pytest.mark.asyncio
async def test_contradictory_quality_evidence_blocks_as_unresolved_coverage() -> None:
    result = await run_content_ops_review(
        _request(coverage=(), quality_reports=({"passed": True, "decision": "block"},)),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert result.content_pr.coverage[0].rule_id == "QUALITY-GATE:contradictory-decision"
    assert result.content_pr.coverage[0].status == CoverageStatus.UNRESOLVED
    assert any("unresolved required coverage" in reason for reason in result.reasons)


@pytest.mark.asyncio
async def test_brand_voice_public_metadata_can_supply_required_coverage() -> None:
    result = await run_content_ops_review(
        _request(
            coverage=(),
            brand_voice_payload={"brand_voice_audit": {"passed": True}},
        ),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.APPROVED
    assert [row.rule_id for row in result.content_pr.coverage] == [
        "BRAND-VOICE:audit",
    ]
    assert result.content_pr.coverage[0].status == CoverageStatus.PASS


@pytest.mark.asyncio
async def test_brand_voice_warning_requires_revision() -> None:
    result = await run_content_ops_review(
        _request(
            coverage=(),
            brand_voice_payload={
                "brand_voice_audit": {
                    "passed": False,
                    "warnings": ["preferred_pov_second_person_not_detected"],
                }
            },
        ),
        scope=TenantScope(account_id="acct-1"),
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.REVISION_REQUIRED
    assert result.content_pr.coverage[0].rule_id == (
        "BRAND-VOICE:warning-preferred-pov-second-person-not-detected"
    )
    assert any("failed required coverage" in reason for reason in result.reasons)


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


# -- adversarial pass folding (slice 6) --------------------------------------


_SCOPE = TenantScope(account_id="acct-1", user_id="user-1")


def _finding(
    category: AdversarialFindingCategory,
    *,
    message: str = "strongest reason not to ship",
    evidence: str = "quoted draft span",
) -> AdversarialFinding:
    return AdversarialFinding(category=category, message=message, evidence=evidence)


def _adversarial_comments(result):
    return [
        c for c in result.content_pr.comments if c.message.startswith("[adversarial:")
    ]


@pytest.mark.asyncio
async def test_adversarial_findings_fold_in_as_nonblocking_comments() -> None:
    passes = (
        AdversarialPass(
            pass_id="p1",
            findings=(
                _finding(AdversarialFindingCategory.OVERCLAIM),
                _finding(AdversarialFindingCategory.AMBIGUITY, message="", evidence=""),
            ),
        ),
    )

    result = await run_content_ops_review(
        _request(adversarial_passes=passes),
        scope=_SCOPE,
        registry_reader=_reader(),
    )

    folded = _adversarial_comments(result)
    # The unsubstantiated finding (no message/evidence) is dropped.
    assert [c.message for c in folded] == ["[adversarial:overclaim] strongest reason not to ship"]
    assert all(c.blocking is False for c in folded)
    assert folded[0].category == CommentCategory.EDITORIAL_JUDGMENT


@pytest.mark.asyncio
async def test_voice_slip_finding_routes_to_brand_rule_lane() -> None:
    passes = (AdversarialPass(pass_id="p1", findings=(_finding(AdversarialFindingCategory.VOICE_SLIP),)),)

    result = await run_content_ops_review(
        _request(adversarial_passes=passes),
        scope=_SCOPE,
        registry_reader=_reader(),
    )

    folded = _adversarial_comments(result)
    assert folded[0].category == CommentCategory.BRAND_RULE
    assert folded[0].blocking is False


@pytest.mark.asyncio
async def test_adversarial_findings_do_not_change_an_approved_verdict() -> None:
    # The default request approves; folding never-blocking evidence keeps it APPROVED.
    passes = (
        AdversarialPass(
            pass_id="p1",
            findings=(
                _finding(AdversarialFindingCategory.OVERCLAIM),
                _finding(AdversarialFindingCategory.MISSING_PROOF),
            ),
        ),
    )

    result = await run_content_ops_review(
        _request(adversarial_passes=passes),
        scope=_SCOPE,
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.APPROVED
    assert len(_adversarial_comments(result)) == 2


@pytest.mark.asyncio
async def test_explicit_comments_precede_adversarial_comments() -> None:
    explicit = ReviewComment(
        category=CommentCategory.EDITORIAL_JUDGMENT,
        message="human note",
        evidence="span",
    )
    passes = (AdversarialPass(pass_id="p1", findings=(_finding(AdversarialFindingCategory.GENERIC_STRETCH),)),)

    result = await run_content_ops_review(
        _request(comments=(explicit,), adversarial_passes=passes),
        scope=_SCOPE,
        registry_reader=_reader(),
    )

    messages = [c.message for c in result.content_pr.comments]
    assert messages == ["human note", "[adversarial:generic_stretch] strongest reason not to ship"]


@pytest.mark.asyncio
async def test_blocked_path_still_folds_adversarial_evidence() -> None:
    # An unpinned rule packet blocks; the adversarial evidence still surfaces.
    passes = (AdversarialPass(pass_id="p1", findings=(_finding(AdversarialFindingCategory.OVERCLAIM),)),)

    result = await run_content_ops_review(
        _request(rule_packet=RulePacketVersions(), adversarial_passes=passes),
        scope=_SCOPE,
        registry_reader=_reader(),
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert len(_adversarial_comments(result)) == 1
