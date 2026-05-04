"""Integration test: reasoning_synthesis registers with the pack registry.

PR-C3e -- fourth concrete pack slice. The reasoning synthesis pack
produces consumer-neutral analytical contracts that downstream
consumers (battle cards, reports, blogs, campaigns) translate into
their own language.

**Not wired into the standalone extracted-pipeline CI** -- same
atlas-pydantic transitive-import constraint as the prior PR-C3*
integration tests. Runs in atlas-side full test suites.
"""

from __future__ import annotations

import importlib

from extracted_reasoning_core.pack_registry import get_pack, list_packs


def _ensure_registered():
    from atlas_brain.reasoning.single_pass_prompts import reasoning_synthesis

    importlib.reload(reasoning_synthesis)
    return reasoning_synthesis


def test_reasoning_synthesis_pack_registers_on_import() -> None:
    module = _ensure_registered()

    pack = get_pack("reasoning_synthesis")
    assert pack is not None
    assert pack.name == "reasoning_synthesis"
    assert len(pack.version) == 8
    assert all(c in "0123456789abcdef" for c in pack.version)
    assert pack.version == module.REASONING_SYNTHESIS_PROMPT_VERSION
    assert pack.prompts["synthesis"] == module.REASONING_SYNTHESIS_PROMPT


def test_reasoning_synthesis_pack_carries_owner_metadata() -> None:
    _ensure_registered()

    pack = get_pack("reasoning_synthesis")
    assert pack is not None
    assert pack.metadata["output_artifact"] == "reasoning_contracts"
    assert pack.metadata["owner_product"] == "atlas"
    assert pack.metadata["synthesis_mode"] == "consumer_neutral_contracts"
    # valid_wedges: shared wedge_registry enum values surfaced via
    # metadata so callers picking wedges from the synthesis output
    # don't have to reach into wedge_registry separately.
    assert isinstance(pack.metadata["valid_wedges"], tuple)
    assert len(pack.metadata["valid_wedges"]) > 0


def test_reasoning_synthesis_pack_appears_in_list() -> None:
    _ensure_registered()

    pack_names = [p.name for p in list_packs()]
    assert "reasoning_synthesis" in pack_names
