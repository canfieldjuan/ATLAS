"""Tests for extracted_quality_gate.evidence_pack.

Pure unit tests against the lifted helpers + the
``evaluate_evidence_coverage`` pack-contract entry point. The async
DB query is exercised against a fake pool that emulates the
``b2b_evidence_claims`` filter so tests stay self-contained.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from extracted_quality_gate.evidence_pack import (
    _coerce_review_ids,
    _rank_floor,
    audit_witness_evidence_coverage,
    evaluate_evidence_coverage,
)
from extracted_quality_gate.types import (
    GateDecision,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)


class _NullPool:
    """Pool stub that always returns no rows -- nothing covered."""

    async def fetch(self, *_args: Any) -> list[dict[str, Any]]:
        return []


class _FakePool:
    """Pool stub that mirrors the SQL: filter by status/vendor/IDs/rank.

    Tests seed ``self.claims`` with dicts containing
    ``status``, ``vendor_name``, ``source_review_id`` (UUID), and
    ``pain_confidence_rank`` (int 0/1/2). ``fetch`` returns the
    distinct review IDs that pass the filter, matching the production
    query shape so the production SQL can change later without
    silently desyncing tests.
    """

    def __init__(self) -> None:
        self.claims: list[dict[str, Any]] = []
        self.received_args: list[tuple[Any, ...]] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        # Sanity-check that the production SQL keeps the shape this
        # fake emulates. Catches accidental regressions to the schema
        # column names or the placeholder ordering.
        assert "FROM b2b_evidence_claims" in sql
        assert "status = $1" in sql
        assert "vendor_name = $2" in sql
        assert "source_review_id = ANY($3::uuid[])" in sql
        assert "pain_confidence_rank <= $4" in sql
        self.received_args.append(args)
        valid_status, vendor, ids, floor = args
        ids_set = {str(i) for i in ids}
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for claim in self.claims:
            if claim["status"] != valid_status:
                continue
            if claim["vendor_name"] != vendor:
                continue
            if str(claim["source_review_id"]) not in ids_set:
                continue
            if claim["pain_confidence_rank"] > floor:
                continue
            review_id = str(claim["source_review_id"])
            if review_id in seen:
                continue
            seen.add(review_id)
            out.append({"review_id": review_id})
        return out


# ---- _rank_floor ----


def test_rank_floor_strong_maps_to_zero():
    assert _rank_floor("strong") == 0


def test_rank_floor_weak_maps_to_one():
    assert _rank_floor("weak") == 1


def test_rank_floor_none_maps_to_two():
    assert _rank_floor("none") == 2


def test_rank_floor_unknown_falls_back_to_strong():
    assert _rank_floor("unknown") == 0
    assert _rank_floor("") == 0
    assert _rank_floor(None) == 0  # type: ignore[arg-type]


def test_rank_floor_case_insensitive():
    assert _rank_floor("STRONG") == 0
    assert _rank_floor("Weak") == 1


def test_rank_floor_whitespace_stripped():
    assert _rank_floor("  strong  ") == 0


# ---- _coerce_review_ids ----


def test_coerce_drops_none_and_blank():
    assert _coerce_review_ids([None, "", "   "]) == []


def test_coerce_drops_invalid_uuids():
    out = _coerce_review_ids(["not-a-uuid", "abc"])
    assert out == []


def test_coerce_dedupes_uuid_and_str_forms():
    rid = uuid4()
    out = _coerce_review_ids([rid, str(rid), rid])
    assert len(out) == 1
    assert out[0] == rid


def test_coerce_preserves_input_order():
    a = uuid4()
    b = uuid4()
    out = _coerce_review_ids([a, b])
    assert out == [a, b]


# ---- audit_witness_evidence_coverage (legacy entry point) ----


@pytest.mark.asyncio
async def test_audit_returns_full_coverage_for_empty_review_ids():
    result = await audit_witness_evidence_coverage(
        _NullPool(),
        vendor_name="acme",
        source_review_ids=[],
    )
    assert result["total_review_ids"] == 0
    assert result["covered_count"] == 0
    assert result["uncovered_count"] == 0
    assert result["coverage_ratio"] == 1.0


@pytest.mark.asyncio
async def test_audit_returns_zero_coverage_for_empty_vendor():
    rid = uuid4()
    result = await audit_witness_evidence_coverage(
        _NullPool(),
        vendor_name="",
        source_review_ids=[rid],
    )
    assert result["total_review_ids"] == 1
    assert result["covered_count"] == 0
    assert result["uncovered_count"] == 1
    assert result["coverage_ratio"] == 0.0


@pytest.mark.asyncio
async def test_audit_dedups_duplicate_review_ids():
    rid = uuid4()
    result = await audit_witness_evidence_coverage(
        _NullPool(),
        vendor_name="acme",
        source_review_ids=[rid, rid, str(rid)],
    )
    assert result["total_review_ids"] == 1


@pytest.mark.asyncio
async def test_audit_skips_invalid_uuid_strings():
    result = await audit_witness_evidence_coverage(
        _NullPool(),
        vendor_name="acme",
        source_review_ids=["not-a-uuid", None, ""],
    )
    assert result["total_review_ids"] == 0


@pytest.mark.asyncio
async def test_audit_counts_strong_backed_review_ids():
    pool = _FakePool()
    vendor = "vendor-x"
    strong_rid = uuid4()
    weak_rid = uuid4()
    unbacked_rid = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": strong_rid, "pain_confidence_rank": 0},
        {"status": "valid", "vendor_name": vendor, "source_review_id": weak_rid, "pain_confidence_rank": 1},
    ]
    result = await audit_witness_evidence_coverage(
        pool,
        vendor_name=vendor,
        source_review_ids=[strong_rid, weak_rid, unbacked_rid],
        min_pain_confidence="strong",
    )
    assert result["total_review_ids"] == 3
    assert str(strong_rid) in result["covered_review_ids"]
    assert str(weak_rid) not in result["covered_review_ids"]
    assert str(unbacked_rid) not in result["covered_review_ids"]
    assert result["covered_count"] == 1
    assert result["uncovered_count"] == 2
    assert result["coverage_ratio"] == round(1 / 3, 3)


@pytest.mark.asyncio
async def test_audit_loosened_to_weak_includes_weak_claims():
    pool = _FakePool()
    vendor = "vendor-y"
    strong_rid = uuid4()
    weak_rid = uuid4()
    unbacked_rid = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": strong_rid, "pain_confidence_rank": 0},
        {"status": "valid", "vendor_name": vendor, "source_review_id": weak_rid, "pain_confidence_rank": 1},
    ]
    result = await audit_witness_evidence_coverage(
        pool,
        vendor_name=vendor,
        source_review_ids=[strong_rid, weak_rid, unbacked_rid],
        min_pain_confidence="weak",
    )
    assert result["covered_count"] == 2
    assert str(strong_rid) in result["covered_review_ids"]
    assert str(weak_rid) in result["covered_review_ids"]


@pytest.mark.asyncio
async def test_audit_ignores_invalid_status_rows():
    pool = _FakePool()
    vendor = "vendor-z"
    rid = uuid4()
    pool.claims = [
        {"status": "rejected", "vendor_name": vendor, "source_review_id": rid, "pain_confidence_rank": 0},
    ]
    result = await audit_witness_evidence_coverage(
        pool,
        vendor_name=vendor,
        source_review_ids=[rid],
        min_pain_confidence="strong",
    )
    assert result["covered_count"] == 0
    assert result["uncovered_count"] == 1


@pytest.mark.asyncio
async def test_audit_respects_custom_valid_status():
    pool = _FakePool()
    vendor = "vendor-q"
    rid = uuid4()
    pool.claims = [
        {"status": "approved", "vendor_name": vendor, "source_review_id": rid, "pain_confidence_rank": 0},
    ]
    # Override the status filter via the new keyword arg.
    result = await audit_witness_evidence_coverage(
        pool,
        vendor_name=vendor,
        source_review_ids=[rid],
        valid_status="approved",
    )
    assert result["covered_count"] == 1
    assert pool.received_args[-1][0] == "approved"


@pytest.mark.asyncio
async def test_audit_respects_custom_precision():
    pool = _FakePool()
    vendor = "vendor-p"
    rid = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": rid, "pain_confidence_rank": 0},
    ]
    result = await audit_witness_evidence_coverage(
        pool,
        vendor_name=vendor,
        source_review_ids=[rid, uuid4(), uuid4()],
        coverage_precision=5,
    )
    # coverage_ratio is 1/3 -> rounded to 5 dp
    assert result["coverage_ratio"] == round(1 / 3, 5)


# ---- evaluate_evidence_coverage (pack contract) ----


def _build_input(
    *,
    vendor_name: str = "acme",
    source_review_ids: tuple = (),
    min_pain_confidence: str | None = None,
) -> QualityInput:
    context: dict[str, Any] = {
        "vendor_name": vendor_name,
        "source_review_ids": source_review_ids,
    }
    if min_pain_confidence is not None:
        context["min_pain_confidence"] = min_pain_confidence
    return QualityInput(
        artifact_type="campaign_email",
        artifact_id="test",
        content=None,
        context=context,
    )


@pytest.mark.asyncio
async def test_pack_contract_returns_quality_report():
    report = await evaluate_evidence_coverage(_NullPool(), _build_input())
    assert isinstance(report, QualityReport)


@pytest.mark.asyncio
async def test_pack_contract_passes_when_no_review_ids():
    report = await evaluate_evidence_coverage(_NullPool(), _build_input())
    assert report.decision == GateDecision.PASS
    assert report.findings == ()
    assert report.metadata["coverage_ratio"] == 1.0


@pytest.mark.asyncio
async def test_pack_contract_warns_on_partial_coverage_default_thresholds():
    # Default block_threshold=0.0 -> never blocks. warn_threshold=1.0
    # -> any uncovered review yields warnings.
    pool = _FakePool()
    vendor = "vendor-w"
    rid_covered = uuid4()
    rid_uncovered = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": rid_covered, "pain_confidence_rank": 0},
    ]
    report = await evaluate_evidence_coverage(
        pool,
        _build_input(vendor_name=vendor, source_review_ids=(rid_covered, rid_uncovered)),
    )
    assert report.decision == GateDecision.WARN
    # One per-review finding for the uncovered one.
    warnings = [f for f in report.findings if f.severity == GateSeverity.WARNING]
    assert len(warnings) == 1
    assert warnings[0].code == "unbacked_review"
    assert str(rid_uncovered) in warnings[0].message


@pytest.mark.asyncio
async def test_pack_contract_blocks_when_ratio_below_block_threshold():
    pool = _FakePool()
    vendor = "vendor-b"
    rid_covered = uuid4()
    rid_uncovered_a = uuid4()
    rid_uncovered_b = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": rid_covered, "pain_confidence_rank": 0},
    ]
    policy = QualityPolicy(
        name="evidence",
        thresholds={"coverage_block_threshold": 0.5},
    )
    # 1/3 coverage -> below 0.5 block threshold -> BLOCK
    report = await evaluate_evidence_coverage(
        pool,
        _build_input(
            vendor_name=vendor,
            source_review_ids=(rid_covered, rid_uncovered_a, rid_uncovered_b),
        ),
        policy=policy,
    )
    assert report.decision == GateDecision.BLOCK
    blockers = [f for f in report.findings if f.severity == GateSeverity.BLOCKER]
    # One summary + per-uncovered findings.
    assert any(f.code == "low_evidence_coverage" for f in blockers)
    unbacked = [f for f in blockers if f.code == "unbacked_review"]
    assert len(unbacked) == 2


@pytest.mark.asyncio
async def test_pack_contract_passes_when_coverage_meets_warn_threshold():
    pool = _FakePool()
    vendor = "vendor-p2"
    rid = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": rid, "pain_confidence_rank": 0},
    ]
    policy = QualityPolicy(
        name="evidence",
        thresholds={"coverage_warn_threshold": 0.5},
    )
    # 1/1 coverage = 1.0 > warn 0.5 -> PASS
    report = await evaluate_evidence_coverage(
        pool,
        _build_input(vendor_name=vendor, source_review_ids=(rid,)),
        policy=policy,
    )
    assert report.decision == GateDecision.PASS
    assert report.findings == ()


@pytest.mark.asyncio
async def test_pack_contract_metadata_carries_audit_dict_fields():
    pool = _FakePool()
    vendor = "vendor-m"
    rid = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": rid, "pain_confidence_rank": 0},
    ]
    report = await evaluate_evidence_coverage(
        pool,
        _build_input(vendor_name=vendor, source_review_ids=(rid,)),
    )
    md = report.metadata
    assert md["vendor_name"] == vendor
    assert md["min_pain_confidence"] == "strong"
    assert md["total_review_ids"] == 1
    assert md["covered_count"] == 1
    assert md["coverage_ratio"] == 1.0


@pytest.mark.asyncio
async def test_pack_contract_input_overrides_min_pain_confidence():
    pool = _FakePool()
    vendor = "vendor-o"
    weak_rid = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": weak_rid, "pain_confidence_rank": 1},
    ]
    # default policy -> floor=strong (rank 0); weak (rank 1) excluded
    report_default = await evaluate_evidence_coverage(
        pool,
        _build_input(vendor_name=vendor, source_review_ids=(weak_rid,)),
    )
    assert report_default.metadata["covered_count"] == 0

    # input override -> floor=weak (rank 1); now included
    report_override = await evaluate_evidence_coverage(
        pool,
        _build_input(
            vendor_name=vendor,
            source_review_ids=(weak_rid,),
            min_pain_confidence="weak",
        ),
    )
    assert report_override.metadata["covered_count"] == 1


@pytest.mark.asyncio
async def test_pack_contract_policy_overrides_min_pain_confidence():
    pool = _FakePool()
    vendor = "vendor-p3"
    weak_rid = uuid4()
    pool.claims = [
        {"status": "valid", "vendor_name": vendor, "source_review_id": weak_rid, "pain_confidence_rank": 1},
    ]
    policy = QualityPolicy(
        name="evidence",
        thresholds={"min_pain_confidence": "weak"},
    )
    report = await evaluate_evidence_coverage(
        pool,
        _build_input(vendor_name=vendor, source_review_ids=(weak_rid,)),
        policy=policy,
    )
    assert report.metadata["covered_count"] == 1


def test_quality_report_is_frozen():
    # Build a minimal report by reaching the helper directly.
    from extracted_quality_gate.evidence_pack import _build_quality_report

    audit = {
        "vendor_name": "acme",
        "min_pain_confidence": "strong",
        "total_review_ids": 0,
        "covered_review_ids": [],
        "uncovered_review_ids": [],
        "covered_count": 0,
        "uncovered_count": 0,
        "coverage_ratio": 1.0,
    }
    report = _build_quality_report(audit, block_threshold=0.0, warn_threshold=1.0)
    with pytest.raises(Exception):
        report.passed = False  # type: ignore[misc]


# ---- Atlas re-export sanity ----


def test_atlas_re_export_paths_match():
    from atlas_brain.services.b2b.evidence_gate import (
        audit_witness_evidence_coverage as atlas_audit,
        evaluate_evidence_coverage as atlas_evaluate,
    )
    assert atlas_audit is audit_witness_evidence_coverage
    assert atlas_evaluate is evaluate_evidence_coverage
