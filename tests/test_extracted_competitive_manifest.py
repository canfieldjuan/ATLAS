from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return json.loads(
        (ROOT / "extracted_competitive_intelligence/manifest.json").read_text()
    )


def _owned_targets() -> set[str]:
    return {entry["target"] for entry in _manifest().get("owned", [])}


def _mapped_targets() -> set[str]:
    return {entry["target"] for entry in _manifest().get("mappings", [])}


def test_manifest_tracks_product_owned_competitive_modules() -> None:
    owned = _owned_targets()

    assert "extracted_competitive_intelligence/services/vendor_registry.py" in owned
    assert "extracted_competitive_intelligence/mcp/b2b/vendor_registry.py" in owned
    assert "extracted_competitive_intelligence/mcp/b2b/displacement.py" in owned
    assert "extracted_competitive_intelligence/mcp/b2b/cross_vendor.py" in owned
    assert "extracted_competitive_intelligence/mcp/b2b/write_intelligence.py" in owned
    assert "extracted_competitive_intelligence/mcp/b2b/write_ports.py" in owned
    assert "extracted_competitive_intelligence/services/scraping/capabilities.py" in owned
    assert "extracted_competitive_intelligence/services/b2b/source_impact.py" in owned
    assert "extracted_competitive_intelligence/services/b2b/challenger_dashboard_claims.py" in owned
    assert "extracted_competitive_intelligence/services/b2b/competitive_set_ports.py" in owned
    assert "extracted_competitive_intelligence/services/b2b/battle_card_ports.py" in owned
    assert "extracted_competitive_intelligence/services/b2b/vendor_briefing_ports.py" in owned
    assert "extracted_competitive_intelligence/services/b2b_competitive_sets.py" in owned
    assert "extracted_competitive_intelligence/autonomous/tasks/_b2b_batch_utils.py" in owned
    assert "extracted_competitive_intelligence/autonomous/tasks/_b2b_cross_vendor_synthesis.py" in owned
    assert "extracted_competitive_intelligence/templates/email/vendor_briefing.py" in owned
    assert "extracted_competitive_intelligence/reasoning/ecosystem.py" in owned
    assert "extracted_competitive_intelligence/reasoning/cross_vendor_selection.py" in owned
    assert "extracted_competitive_intelligence/reasoning/single_pass_prompts/cross_vendor_battle.py" in owned
    assert "extracted_competitive_intelligence/reasoning/single_pass_prompts/battle_card_reasoning.py" in owned


def test_product_owned_competitive_modules_are_not_manifest_synced() -> None:
    mapped = _mapped_targets()

    assert "extracted_competitive_intelligence/services/vendor_registry.py" not in mapped
    assert "extracted_competitive_intelligence/mcp/b2b/write_intelligence.py" not in mapped
    assert "extracted_competitive_intelligence/mcp/b2b/write_ports.py" not in mapped
    assert "extracted_competitive_intelligence/services/scraping/capabilities.py" not in mapped
    assert "extracted_competitive_intelligence/services/b2b/source_impact.py" not in mapped
    assert "extracted_competitive_intelligence/services/b2b/challenger_dashboard_claims.py" not in mapped
    assert "extracted_competitive_intelligence/services/b2b/competitive_set_ports.py" not in mapped
    assert "extracted_competitive_intelligence/services/b2b/battle_card_ports.py" not in mapped
    assert "extracted_competitive_intelligence/services/b2b/vendor_briefing_ports.py" not in mapped
    assert "extracted_competitive_intelligence/services/b2b_competitive_sets.py" not in mapped
    assert "extracted_competitive_intelligence/autonomous/tasks/_b2b_batch_utils.py" not in mapped
    assert "extracted_competitive_intelligence/autonomous/tasks/_b2b_cross_vendor_synthesis.py" not in mapped
    assert "extracted_competitive_intelligence/templates/email/vendor_briefing.py" not in mapped
    assert "extracted_competitive_intelligence/reasoning/ecosystem.py" not in mapped
    assert "extracted_competitive_intelligence/reasoning/cross_vendor_selection.py" not in mapped
    assert "extracted_competitive_intelligence/reasoning/single_pass_prompts/cross_vendor_battle.py" not in mapped
    assert "extracted_competitive_intelligence/reasoning/single_pass_prompts/battle_card_reasoning.py" not in mapped


def test_product_owned_competitive_modules_do_not_import_atlas() -> None:
    for target in _owned_targets():
        path = ROOT / target
        if path.suffix != ".py":
            continue
        assert "atlas_brain." not in path.read_text()
