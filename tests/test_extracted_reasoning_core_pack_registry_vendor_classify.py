"""Integration test: vendor_classify registers with the pack registry.

PR-C3d -- third concrete pack slice on top of PR-C3a (registry skeleton),
PR-C3b (battle_card_reasoning), and PR-C3c (cross_vendor_battle x2).

The vendor_classify pack produces archetype assignments + self-check
output for the b2b churn intelligence task. Atlas's autonomous tasks
consume it directly; competitive_intelligence's battle card builders
read the assigned archetype downstream.

**Not wired into the standalone extracted-pipeline CI** for the same
reason as the prior PR-C3* pack tests: importing the atlas-side
prompt module transitively pulls in pydantic via
``atlas_brain.reasoning.config``. This suite runs in the full
atlas-side test runs.
"""

from __future__ import annotations

import importlib

from extracted_reasoning_core.pack_registry import get_pack, list_packs


def _ensure_vendor_classify_registered():
    from atlas_brain.reasoning.single_pass_prompts import vendor_classify

    importlib.reload(vendor_classify)
    return vendor_classify


def test_vendor_classify_pack_registers_on_import() -> None:
    module = _ensure_vendor_classify_registered()

    pack = get_pack("vendor_classify")
    assert pack is not None
    assert pack.name == "vendor_classify"
    assert len(pack.version) == 8
    assert all(c in "0123456789abcdef" for c in pack.version)
    assert pack.version == module.VENDOR_CLASSIFY_PROMPT_VERSION
    assert pack.prompts["classify_single_pass"] == module.VENDOR_CLASSIFY_SINGLE_PASS


def test_vendor_classify_pack_carries_owner_metadata() -> None:
    _ensure_vendor_classify_registered()

    pack = get_pack("vendor_classify")
    assert pack is not None
    assert pack.metadata["output_artifact"] == "vendor_archetype_classification"
    assert pack.metadata["owner_product"] == "atlas"
    assert pack.metadata["synthesis_mode"] == "single_pass_with_self_check"


def test_vendor_classify_pack_appears_in_list_packs() -> None:
    _ensure_vendor_classify_registered()

    pack_names = [p.name for p in list_packs()]
    assert "vendor_classify" in pack_names
