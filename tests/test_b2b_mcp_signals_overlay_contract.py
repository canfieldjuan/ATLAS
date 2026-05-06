from atlas_brain.mcp.b2b import signals


class _DummyView:
    pass


def test_overlay_reasoning_summary_from_view_uses_adapter(monkeypatch):
    def _fake_summary_fields(_view):
        return {
            "archetype": "feature_parity",
            "archetype_confidence": 0.55,
            "reasoning_mode": "synthesis",
            "reasoning_risk_level": "medium",
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_reasoning_consumer_adapter.reasoning_summary_fields_from_view",
        _fake_summary_fields,
    )

    payload = {
        "vendor_name": "Acme",
        "archetype": None,
        "archetype_confidence": None,
        "reasoning_mode": None,
        "reasoning_risk_level": None,
        "keyword_spike_count": 3,
    }
    signals._overlay_reasoning_summary_from_view(payload, _DummyView())

    assert payload["archetype"] == "feature_parity"
    assert payload["archetype_confidence"] == 0.55
    assert payload["reasoning_mode"] == "synthesis"
    assert payload["reasoning_risk_level"] == "medium"
    assert payload["keyword_spike_count"] == 3


def test_overlay_reasoning_detail_from_view_uses_adapter(monkeypatch):
    def _fake_detail_fields(_view):
        return {
            "archetype": "support_erosion",
            "archetype_confidence": 0.81,
            "reasoning_mode": "synthesis",
            "reasoning_risk_level": "high",
            "reasoning_executive_summary": "summary",
            "reasoning_key_signals": ["k1"],
            "reasoning_uncertainty_sources": ["u1"],
            "falsification_conditions": ["f1"],
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_reasoning_consumer_adapter.reasoning_detail_fields_from_view",
        _fake_detail_fields,
    )

    payload = {
        "vendor_name": "Acme",
        "reasoning_executive_summary": None,
        "reasoning_key_signals": [],
        "reasoning_uncertainty_sources": [],
        "falsification_conditions": [],
        "source_distribution": {"reddit": 3},
    }
    signals._overlay_reasoning_detail_from_view(payload, _DummyView())

    assert payload["archetype"] == "support_erosion"
    assert payload["archetype_confidence"] == 0.81
    assert payload["reasoning_mode"] == "synthesis"
    assert payload["reasoning_risk_level"] == "high"
    assert payload["reasoning_executive_summary"] == "summary"
    assert payload["reasoning_key_signals"] == ["k1"]
    assert payload["reasoning_uncertainty_sources"] == ["u1"]
    assert payload["falsification_conditions"] == ["f1"]
    assert payload["source_distribution"] == {"reddit": 3}
