import re

from extracted_competitive_intelligence.reasoning.single_pass_prompts.battle_card_reasoning import (
    BATTLE_CARD_REASONING_PROMPT,
    BATTLE_CARD_REASONING_PROMPT_VERSION,
    VALID_WEDGE_TYPES,
)
from extracted_competitive_intelligence.reasoning.single_pass_prompts.cross_vendor_battle import (
    CROSS_VENDOR_BATTLE_SINGLE_PASS,
)
from extracted_competitive_intelligence.reasoning.wedge_registry import WEDGE_ENUM_VALUES


def test_cross_vendor_battle_prompt_preserves_locked_direction_contract():
    prompt = CROSS_VENDOR_BATTLE_SINGLE_PASS

    assert "locked_direction" in prompt
    assert "Copy winner/loser from locked_direction exactly" in prompt
    assert "Do NOT flip winner/loser" in prompt
    assert "Output ONLY valid JSON" in prompt


def test_battle_card_reasoning_prompt_exports_stable_schema_contract():
    prompt = BATTLE_CARD_REASONING_PROMPT

    assert '"schema_version": "2.2"' in prompt
    assert '"reasoning_shape": "contracts_first_v1"' in prompt
    assert '"reasoning_contracts"' in prompt
    assert "Output ONLY valid JSON" in prompt


def test_battle_card_reasoning_prompt_uses_shared_wedge_registry():
    assert VALID_WEDGE_TYPES == tuple(sorted(WEDGE_ENUM_VALUES))
    assert VALID_WEDGE_TYPES

    for wedge in VALID_WEDGE_TYPES:
        assert wedge in BATTLE_CARD_REASONING_PROMPT


def test_battle_card_reasoning_prompt_version_is_hash_prefix():
    assert re.fullmatch(r"[0-9a-f]{8}", BATTLE_CARD_REASONING_PROMPT_VERSION)
