"""Integration tests: remaining synthesis prompts register with the pack registry.

PR-C3f -- final slice of the PR-C3 sequence. Two synthesis prompts
that weren't named in the original audit's PR 5 list but live in
``atlas_brain/reasoning/single_pass_prompts/`` and follow the same
single-pass synthesis pattern:

  - ``category_council_synthesis`` -- market-regime assessment for a
    product category from category dynamics + ecosystem context
  - ``resource_asymmetry_synthesis`` -- resource divergence assessment
    between two vendors with similar churn pressure but different
    market positions

Audit deviation note (recorded in the PR description): the audit's
"content/campaign placeholder packs" entry turned out to have no
corresponding atlas-side prompt files -- atlas's content/campaign
work is in autonomous tasks (``campaign_audit``, ``blog_post_generation``,
etc.), not in single-pass prompt modules. PR-C3f treats those as PR 7
territory (Product Migration) where the campaign content packs
naturally land alongside the migrated tasks.

**Not wired into the standalone extracted-pipeline CI** -- same
atlas->pydantic transitive-import constraint as prior PR-C3* tests.
Runs in atlas-side full test suites.
"""

from __future__ import annotations

import importlib

from extracted_reasoning_core.pack_registry import get_pack, list_packs


def _ensure_category_council_registered():
    from atlas_brain.reasoning.single_pass_prompts import category_council_synthesis

    importlib.reload(category_council_synthesis)
    return category_council_synthesis


def _ensure_resource_asymmetry_registered():
    from atlas_brain.reasoning.single_pass_prompts import resource_asymmetry_synthesis

    importlib.reload(resource_asymmetry_synthesis)
    return resource_asymmetry_synthesis


# ----------------------------------------------------------------------
# category_council_synthesis
# ----------------------------------------------------------------------


def test_category_council_pack_registers_on_import() -> None:
    module = _ensure_category_council_registered()

    pack = get_pack("category_council_synthesis")
    assert pack is not None
    assert pack.name == "category_council_synthesis"
    assert len(pack.version) == 8
    assert pack.version == module.CATEGORY_COUNCIL_SYNTHESIS_PROMPT_VERSION
    assert pack.prompts["synthesis"] == module.CATEGORY_COUNCIL_SYNTHESIS_PROMPT


def test_category_council_pack_carries_owner_metadata() -> None:
    _ensure_category_council_registered()

    pack = get_pack("category_council_synthesis")
    assert pack is not None
    assert pack.metadata["output_artifact"] == "category_market_regime"
    assert pack.metadata["owner_product"] == "atlas"
    assert pack.metadata["synthesis_mode"] == "structured_synthesis_v1"


# ----------------------------------------------------------------------
# resource_asymmetry_synthesis
# ----------------------------------------------------------------------


def test_resource_asymmetry_pack_registers_on_import() -> None:
    module = _ensure_resource_asymmetry_registered()

    pack = get_pack("resource_asymmetry_synthesis")
    assert pack is not None
    assert pack.name == "resource_asymmetry_synthesis"
    assert len(pack.version) == 8
    assert pack.version == module.RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT_VERSION
    assert pack.prompts["synthesis"] == module.RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT


def test_resource_asymmetry_pack_carries_owner_metadata() -> None:
    _ensure_resource_asymmetry_registered()

    pack = get_pack("resource_asymmetry_synthesis")
    assert pack is not None
    assert pack.metadata["output_artifact"] == "resource_asymmetry_assessment"
    assert pack.metadata["owner_product"] == "competitive_intelligence"
    assert pack.metadata["synthesis_mode"] == "structured_synthesis_v1"


# ----------------------------------------------------------------------
# Combined: closes the PR-C3 sequence
# ----------------------------------------------------------------------


def test_both_remaining_packs_appear_in_list() -> None:
    _ensure_category_council_registered()
    _ensure_resource_asymmetry_registered()

    pack_names = {p.name for p in list_packs()}
    assert "category_council_synthesis" in pack_names
    assert "resource_asymmetry_synthesis" in pack_names
