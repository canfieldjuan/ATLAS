import pytest
from unittest.mock import AsyncMock, MagicMock

from atlas_brain.reasoning.market_pulse import MarketPulseReasoner
from atlas_brain.reasoning.temporal import TemporalEvidence, VendorVelocity

@pytest.mark.asyncio
async def test_category_payload_includes_market_regime():
    """Verify _build_category_payload accepts and includes market_regime."""
    from atlas_brain.reasoning.cross_vendor import _build_category_payload
    
    payload = _build_category_payload(
        "CRM", 
        vendor_evidence={}, 
        ecosystem={}, 
        displacement_flows=[], 
        market_regime={"regime_type": "high_churn"}
    )
    assert "market_pulse" in payload
    assert payload["market_pulse"]["regime_type"] == "high_churn"

@pytest.mark.asyncio
async def test_analyze_category_signature():
    """Verify CrossVendorReasoner.analyze_category accepts market_regime."""
    from atlas_brain.reasoning.cross_vendor import CrossVendorReasoner
    
    cv = CrossVendorReasoner(MagicMock())
    cv._call_llm = AsyncMock()
    
    await cv.analyze_category(
        "CRM", 
        vendor_evidence={}, 
        ecosystem={}, 
        displacement_flows=[], 
        market_regime={"regime_type": "test"}
    )
    
    # Check if _call_llm was called
    assert cv._call_llm.called
    # Check payload passed to _call_llm
    call_args = cv._call_llm.call_args
    payload = call_args[0][1] # 2nd arg
    assert "market_pulse" in payload
    assert payload["market_pulse"]["regime_type"] == "test"

@pytest.mark.asyncio
async def test_temporal_evidence_reconstruction_logic():
    """Temporal reconstruction preserves the vendor name for a single entry."""
    from atlas_brain.reasoning.temporal import TemporalEvidence, VendorVelocity

    vname = "TestVendor"
    td = {
        "snapshot_days": 10,
        "velocity_churn_density": 0.5
    }

    velocities = []
    for k, v in td.items():
        if k.startswith("velocity_"):
            metric = k.replace("velocity_", "")
            velocities.append(VendorVelocity(
                vendor_name=vname,
                metric=metric,
                current_value=0, previous_value=0,
                velocity=float(v),
                days_between=1
            ))

    te = TemporalEvidence(
        vendor_name=vname,
        snapshot_days=td.get("snapshot_days", 0),
        velocities=velocities
    )

    assert te.vendor_name == "TestVendor"
    assert len(te.velocities) == 1
    assert te.velocities[0].vendor_name == "TestVendor"


def test_temporal_evidence_reconstruction_preserves_each_vendor_name():
    """Category grouping must not leak the last vendor name across entries."""
    from atlas_brain.reasoning.temporal import TemporalEvidence, VendorVelocity

    grouped = {
        "CRM": [
            ("VendorA", {"snapshot_days": 7, "velocity_churn_density": 0.4}),
            ("VendorB", {"snapshot_days": 9, "velocity_avg_urgency": -0.2}),
        ]
    }

    te_list = []
    for vname, td in grouped["CRM"]:
        velocities = []
        for key, value in td.items():
            if key.startswith("velocity_"):
                velocities.append(VendorVelocity(
                    vendor_name=vname,
                    metric=key.replace("velocity_", ""),
                    current_value=0,
                    previous_value=0,
                    velocity=float(value),
                    days_between=1,
                ))
        te_list.append(TemporalEvidence(
            vendor_name=vname,
            snapshot_days=td.get("snapshot_days", 0),
            velocities=velocities,
        ))

    assert [te.vendor_name for te in te_list] == ["VendorA", "VendorB"]
    assert te_list[0].velocities[0].vendor_name == "VendorA"
    assert te_list[1].velocities[0].vendor_name == "VendorB"


def test_market_pulse_uses_support_and_feature_dynamics():
    """Support decay plus shipping velocity should register as disruption."""
    reasoner = MarketPulseReasoner()
    vendor_evidence = [
        TemporalEvidence(
            vendor_name="VendorA",
            snapshot_days=30,
            velocities=[
                VendorVelocity("VendorA", "churn_density", 0, 0, 0.02, 7),
                VendorVelocity("VendorA", "support_sentiment", 0, 0, -0.08, 7),
                VendorVelocity("VendorA", "new_feature_velocity", 0, 0, 0.12, 7),
            ],
        ),
        TemporalEvidence(
            vendor_name="VendorB",
            snapshot_days=30,
            velocities=[
                VendorVelocity("VendorB", "churn_density", 0, 0, 0.01, 7),
                VendorVelocity("VendorB", "support_sentiment", 0, 0, -0.06, 7),
                VendorVelocity("VendorB", "new_feature_velocity", 0, 0, 0.09, 7),
            ],
        ),
    ]

    result = reasoner.analyze_category("CRM", vendor_evidence)

    assert result.regime_type == "disruption"
    assert "support sentiment" in result.narrative.lower()
