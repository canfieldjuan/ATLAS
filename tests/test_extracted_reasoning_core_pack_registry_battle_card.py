"""Integration test: battle_card_reasoning registers with the pack registry.

PR-C3b -- first concrete pack added on top of the PR-C3a registry skeleton.
The pack lives at ``atlas_brain/reasoning/single_pass_prompts/battle_card_reasoning.py``
for now (PR 7 will migrate it into ``extracted_competitive_intelligence``).
This test confirms the registration call fires when the module is imported
and that the registry exposes the same prompt content + version that
direct callers see.

**Not wired into the standalone extracted-pipeline CI**: importing the
atlas-side battle_card_reasoning module transitively pulls in
``atlas_brain.reasoning.config`` -> ``pydantic``, which isn't part of
the standalone CI's minimal pip install. This test runs as part of
the full atlas-side test suite (which has ``requirements.txt`` with
pydantic). The pack registry's own contract is covered by
``test_extracted_reasoning_core_pack_registry.py`` which IS wired
into the standalone CI.

No ``clear_packs`` fixture: the pack-registry-skeleton test file
tests the registry's own contract under isolation; this file tests
that the *module side effect* registers the right pack, so it relies
on the registration having actually happened. Each test reloads the
module under test to re-trigger registration after any prior
``clear_packs`` call leaked in from a sibling suite.
"""

from __future__ import annotations

import importlib

from extracted_reasoning_core.pack_registry import get_pack, list_packs


def _ensure_battle_card_pack_registered():
    """Re-trigger the registration side effect by reloading the module.

    Returns the module so callers can read its constants.
    """
    from atlas_brain.reasoning.single_pass_prompts import battle_card_reasoning

    importlib.reload(battle_card_reasoning)
    return battle_card_reasoning


def test_battle_card_pack_registers_on_import() -> None:
    module = _ensure_battle_card_pack_registered()

    pack = get_pack("battle_card_reasoning")
    assert pack is not None
    assert pack.name == "battle_card_reasoning"
    # The version is a sha256[:8] hex digest of the prompt text.
    assert len(pack.version) == 8
    assert all(c in "0123456789abcdef" for c in pack.version)
    # Same value as the module-level constant.
    assert pack.version == module.BATTLE_CARD_REASONING_PROMPT_VERSION
    # Prompt content matches the module-level constant.
    assert pack.prompts["reasoning"] == module.BATTLE_CARD_REASONING_PROMPT


def test_battle_card_pack_carries_owner_metadata() -> None:
    module = _ensure_battle_card_pack_registered()

    pack = get_pack("battle_card_reasoning")
    assert pack is not None
    assert pack.metadata["output_artifact"] == "battle_card"
    assert pack.metadata["owner_product"] == "competitive_intelligence"
    # valid_wedges came from the shared wedge_registry; carry it through
    # so callers selecting wedge enum values from the pack don't have to
    # reach into wedge_registry separately.
    assert isinstance(pack.metadata["valid_wedges"], tuple)
    assert pack.metadata["valid_wedges"] == module.VALID_WEDGE_TYPES


def test_battle_card_pack_appears_in_list_packs() -> None:
    _ensure_battle_card_pack_registered()

    pack_names = [p.name for p in list_packs()]
    assert "battle_card_reasoning" in pack_names


def test_battle_card_pack_re_register_is_idempotent() -> None:
    # The reload-helper triggers registration once; reloading twice
    # exercises the idempotent re-registration path (same content ->
    # no ValueError raised). This matters because pytest can re-execute
    # module bodies under module reload scenarios.
    module = _ensure_battle_card_pack_registered()
    importlib.reload(module)

    pack = get_pack("battle_card_reasoning")
    assert pack is not None
    assert pack.version == module.BATTLE_CARD_REASONING_PROMPT_VERSION
