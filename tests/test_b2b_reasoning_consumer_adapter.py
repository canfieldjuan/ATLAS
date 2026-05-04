from atlas_brain.autonomous.tasks import _b2b_reasoning_consumer_adapter as adapter


class _DummyView:
    pass


def test_reasoning_summary_fields_from_view(monkeypatch):
    def _fake_entry(_view):
        return {
            "archetype": "price_squeeze",
            "confidence": 0.82,
            "mode": "synthesis",
            "risk_level": "high",
            "executive_summary": "ignore",
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.synthesis_view_to_reasoning_entry",
        _fake_entry,
    )

    out = adapter.reasoning_summary_fields_from_view(_DummyView())
    assert out == {
        "archetype": "price_squeeze",
        "archetype_confidence": 0.82,
        "reasoning_mode": "synthesis",
        "reasoning_risk_level": "high",
    }


def test_reasoning_detail_fields_from_view_preserves_contract_defaults(monkeypatch):
    def _fake_entry(_view):
        return {
            "archetype": "support_erosion",
            "confidence": 0.44,
            "mode": "synthesis",
            "risk_level": "medium",
            "executive_summary": "summary",
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.synthesis_view_to_reasoning_entry",
        _fake_entry,
    )

    out = adapter.reasoning_detail_fields_from_view(_DummyView())
    assert out["archetype"] == "support_erosion"
    assert out["archetype_confidence"] == 0.44
    assert out["reasoning_mode"] == "synthesis"
    assert out["reasoning_risk_level"] == "medium"
    assert out["reasoning_executive_summary"] == "summary"
    assert out["reasoning_key_signals"] == []
    assert out["reasoning_uncertainty_sources"] == []
    assert out["falsification_conditions"] == []


def test_reasoning_detail_fields_from_view_sparse_entry_has_stable_keys(monkeypatch):
    def _fake_entry(_view):
        return {}

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.synthesis_view_to_reasoning_entry",
        _fake_entry,
    )

    out = adapter.reasoning_detail_fields_from_view(_DummyView())
    assert set(out.keys()) == {
        "archetype",
        "archetype_confidence",
        "reasoning_mode",
        "reasoning_risk_level",
        "reasoning_executive_summary",
        "reasoning_key_signals",
        "reasoning_uncertainty_sources",
        "falsification_conditions",
    }
    assert out["archetype"] is None
    assert out["archetype_confidence"] is None
    assert out["reasoning_mode"] is None
    assert out["reasoning_risk_level"] is None
    assert out["reasoning_executive_summary"] is None
    assert out["reasoning_key_signals"] == []
    assert out["reasoning_uncertainty_sources"] == []
    assert out["falsification_conditions"] == []


def test_reasoning_detail_fields_from_view_explicit_null_lists_stay_lists(monkeypatch):
    """Explicit-null list fields must coerce to [] -- dict.get(k, default)
    only uses the default when k is missing, so present-but-null upstream
    values would otherwise leak through as None and break consumer contracts.
    """
    def _fake_entry(_view):
        return {
            "archetype": "support_erosion",
            "confidence": 0.5,
            "mode": "synthesis",
            "risk_level": "medium",
            "executive_summary": "summary",
            "key_signals": None,
            "uncertainty_sources": None,
            "falsification_conditions": None,
        }

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.synthesis_view_to_reasoning_entry",
        _fake_entry,
    )

    out = adapter.reasoning_detail_fields_from_view(_DummyView())
    assert out["reasoning_key_signals"] == []
    assert out["reasoning_uncertainty_sources"] == []
    assert out["falsification_conditions"] == []
    # Scalars passed through (None is acceptable for those, list contract is the concern)
    assert out["archetype"] == "support_erosion"
    assert out["reasoning_executive_summary"] == "summary"
