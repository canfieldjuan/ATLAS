from __future__ import annotations

import pytest

from extracted_reasoning_core.api import load_reasoning_pack
from extracted_reasoning_core.pack_registry import Pack, clear_packs, register_pack
from extracted_reasoning_core.types import ReasoningPack


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_packs()
    yield
    clear_packs()


def test_load_reasoning_pack_returns_none_for_unknown_name() -> None:
    assert load_reasoning_pack("never_registered") is None


def test_load_reasoning_pack_adapts_registered_pack_to_reasoning_pack() -> None:
    register_pack(Pack(
        name="test_pack",
        version="1.0.0",
        prompts={"reasoning_synthesis": "do the thing"},
        metadata={"max_attempts": 3, "temperature": 0.2},
    ))

    pack = load_reasoning_pack("test_pack")

    assert isinstance(pack, ReasoningPack)
    assert pack.name == "test_pack"
    assert pack.version == "1.0.0"
    assert pack.prompts == {"reasoning_synthesis": "do the thing"}
    # Registry metadata maps to ReasoningPack.policies so consumers like
    # synthesis_config_from_pack pick up policy flags directly.
    assert pack.policies == {"max_attempts": 3, "temperature": 0.2}


def test_load_reasoning_pack_returns_highest_version_when_multiple_registered() -> None:
    register_pack(Pack(name="versioned", version="1.0.0", prompts={"k": "old"}))
    register_pack(Pack(name="versioned", version="2.0.0", prompts={"k": "new"}))
    register_pack(Pack(name="versioned", version="1.5.0", prompts={"k": "mid"}))

    pack = load_reasoning_pack("versioned")

    assert pack is not None
    assert pack.version == "2.0.0"
    assert pack.prompts == {"k": "new"}


def test_load_reasoning_pack_handles_pack_with_no_metadata() -> None:
    register_pack(Pack(name="bare", version="1.0.0", prompts={"k": "v"}))

    pack = load_reasoning_pack("bare")

    assert pack is not None
    assert pack.policies == {}
