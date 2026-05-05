from __future__ import annotations

import sys
from datetime import date
from importlib import import_module

from extracted_quality_gate import product_claim as quality_gate_product_claim

_COMPETITIVE_PRODUCT_CLAIM = (
    "extracted_competitive_intelligence.services.b2b.product_claim"
)
_ATLAS_PRODUCT_CLAIM = "atlas_brain.services.b2b.product_claim"


def _load_product_claim_module():
    sys.modules.pop(_COMPETITIVE_PRODUCT_CLAIM, None)
    sys.modules.pop(_ATLAS_PRODUCT_CLAIM, None)
    return import_module(_COMPETITIVE_PRODUCT_CLAIM)


def test_competitive_product_claim_reexports_quality_gate_contract() -> None:
    product_claim = _load_product_claim_module()

    assert product_claim.ProductClaim is quality_gate_product_claim.ProductClaim
    assert product_claim.SuppressionReason is quality_gate_product_claim.SuppressionReason
    assert product_claim.ClaimScope is quality_gate_product_claim.ClaimScope
    assert product_claim.build_product_claim is quality_gate_product_claim.build_product_claim
    assert product_claim.__all__ == quality_gate_product_claim.__all__


def test_competitive_product_claim_builds_quality_gate_claim() -> None:
    product_claim = _load_product_claim_module()
    product_claim.reset_policy_registry()

    claim = product_claim.build_product_claim(
        claim_scope=product_claim.ClaimScope.COMPETITOR_PAIR,
        claim_type="direct_displacement",
        claim_key="support",
        claim_text="Acme displaces Globex on support",
        target_entity="Acme",
        secondary_target="Globex",
        supporting_count=5,
        direct_evidence_count=2,
        contradiction_count=0,
        witness_count=3,
        sample_size=20,
        denominator=20,
        evidence_links=("review-1",),
        contradicting_links=(),
        as_of_date=date(2026, 5, 5),
        analysis_window_days=90,
    )

    assert isinstance(claim, quality_gate_product_claim.ProductClaim)
    assert claim.report_allowed is True
    assert claim.suppression_reason is None
    assert claim.claim_scope == quality_gate_product_claim.ClaimScope.COMPETITOR_PAIR
    assert claim.claim_type == "direct_displacement"
    assert claim.target_entity == "Acme"
    assert claim.secondary_target == "Globex"


def test_competitive_product_claim_does_not_import_atlas_bridge() -> None:
    _load_product_claim_module()

    assert _ATLAS_PRODUCT_CLAIM not in sys.modules
