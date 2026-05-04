"""Integration test: cross_vendor_battle prompts register with the pack registry.

PR-C3c -- second concrete pack slice on top of the PR-C3a registry skeleton
and PR-C3b's battle_card_reasoning pack. Two related prompts are registered:

  - ``cross_vendor_battle_single_pass`` -- one-shot battle analysis with
    self-check + grounding rules baked into a single prompt
  - ``cross_vendor_battle_synthesis`` -- structured-synthesis variant that
    consumes deterministic displacement evidence and produces the JSON
    battle conclusion downstream

Both packs share ``owner_product: competitive_intelligence`` and ship
distinct ``output_artifact`` / ``synthesis_mode`` metadata so callers
can disambiguate.

**Not wired into the standalone extracted-pipeline CI** for the same
reason as PR-C3b's pack-registration test: importing the atlas-side
prompt modules transitively pulls in pydantic via
``atlas_brain.reasoning.config``. This suite runs in the full atlas-side
test runs.

The single-pass prompt is *owned* by extracted_competitive_intelligence
per the package manifest; the parallel atlas-side copy carries the same
registration code. Both register_pack calls are idempotent against the
registry (identical content -> no ValueError) so importing either or
both modules in any order yields the same registered pack.
"""

from __future__ import annotations

import importlib

from extracted_reasoning_core.pack_registry import get_pack, list_packs


def _ensure_single_pass_registered():
    from atlas_brain.reasoning.single_pass_prompts import cross_vendor_battle

    importlib.reload(cross_vendor_battle)
    return cross_vendor_battle


def _ensure_synthesis_registered():
    from atlas_brain.reasoning.single_pass_prompts import cross_vendor_battle_synthesis

    importlib.reload(cross_vendor_battle_synthesis)
    return cross_vendor_battle_synthesis


# ----------------------------------------------------------------------
# cross_vendor_battle_single_pass
# ----------------------------------------------------------------------


def test_single_pass_pack_registers_on_import() -> None:
    module = _ensure_single_pass_registered()

    pack = get_pack("cross_vendor_battle_single_pass")
    assert pack is not None
    assert pack.name == "cross_vendor_battle_single_pass"
    assert len(pack.version) == 8
    assert all(c in "0123456789abcdef" for c in pack.version)
    assert pack.version == module.CROSS_VENDOR_BATTLE_SINGLE_PASS_VERSION
    assert pack.prompts["battle_single_pass"] == module.CROSS_VENDOR_BATTLE_SINGLE_PASS


def test_single_pass_pack_carries_owner_metadata() -> None:
    _ensure_single_pass_registered()

    pack = get_pack("cross_vendor_battle_single_pass")
    assert pack is not None
    assert pack.metadata["output_artifact"] == "cross_vendor_battle_conclusion"
    assert pack.metadata["owner_product"] == "competitive_intelligence"
    assert pack.metadata["synthesis_mode"] == "single_pass_with_self_check"


# ----------------------------------------------------------------------
# cross_vendor_battle_synthesis
# ----------------------------------------------------------------------


def test_synthesis_pack_registers_on_import() -> None:
    module = _ensure_synthesis_registered()

    pack = get_pack("cross_vendor_battle_synthesis")
    assert pack is not None
    assert pack.name == "cross_vendor_battle_synthesis"
    assert len(pack.version) == 8
    assert all(c in "0123456789abcdef" for c in pack.version)
    assert pack.version == module.CROSS_VENDOR_BATTLE_SYNTHESIS_PROMPT_VERSION
    assert pack.prompts["battle_synthesis"] == module.CROSS_VENDOR_BATTLE_SYNTHESIS_PROMPT


def test_synthesis_pack_carries_owner_metadata() -> None:
    _ensure_synthesis_registered()

    pack = get_pack("cross_vendor_battle_synthesis")
    assert pack is not None
    assert pack.metadata["output_artifact"] == "cross_vendor_battle_synthesis"
    assert pack.metadata["owner_product"] == "competitive_intelligence"
    assert pack.metadata["synthesis_mode"] == "structured_synthesis_v1"


# ----------------------------------------------------------------------
# Combined registry surface
# ----------------------------------------------------------------------


def test_both_cross_vendor_packs_appear_in_list() -> None:
    _ensure_single_pass_registered()
    _ensure_synthesis_registered()

    pack_names = {p.name for p in list_packs()}
    assert "cross_vendor_battle_single_pass" in pack_names
    assert "cross_vendor_battle_synthesis" in pack_names


def test_extracted_side_registration_idempotent() -> None:
    # The extracted_competitive_intelligence copy registers the same
    # cross_vendor_battle_single_pass pack with identical content.
    # Importing it after the atlas-side import should be idempotent
    # (no ValueError).
    _ensure_single_pass_registered()

    from extracted_competitive_intelligence.reasoning.single_pass_prompts import (
        cross_vendor_battle as ext_cross_vendor_battle,
    )

    importlib.reload(ext_cross_vendor_battle)
    pack = get_pack("cross_vendor_battle_single_pass")
    assert pack is not None
    assert pack.version == ext_cross_vendor_battle.CROSS_VENDOR_BATTLE_SINGLE_PASS_VERSION
