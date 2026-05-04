"""Unit tests for extracted_reasoning_core.pack_registry (PR-C3a / PR 5).

Exercises the pack registry skeleton landed in PR-C3a:

  - ``Pack`` dataclass is frozen + slots
  - ``register_pack`` adds packs and is idempotent for identical re-registration
  - ``register_pack`` raises on conflicting re-registration (same key, different content)
  - ``get_pack`` returns latest version when no version requested
  - ``get_pack`` returns specific version when requested
  - ``get_pack`` returns ``None`` for unknown names (no-pack-registered tolerated)
  - ``list_packs`` returns all registered packs ordered deterministically
  - ``clear_packs`` resets registry (test isolation)

Concrete packs (battle card, cross-vendor battle, vendor classify, reasoning
synthesis, content/campaign) land in subsequent PR-C3 slices. This file
covers the registry contract only.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from extracted_reasoning_core.pack_registry import (
    Pack,
    clear_packs,
    get_pack,
    list_packs,
    register_pack,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test starts with an empty registry and resets at teardown."""
    clear_packs()
    yield
    clear_packs()


# ----------------------------------------------------------------------
# Pack dataclass shape
# ----------------------------------------------------------------------


def test_pack_is_frozen() -> None:
    pack = Pack(name="example", version="1.0.0", prompts={"intro": "hello"})
    with pytest.raises(FrozenInstanceError):
        pack.name = "renamed"  # type: ignore[misc]


def test_pack_metadata_defaults_to_empty_dict() -> None:
    pack = Pack(name="example", version="1.0.0", prompts={})
    assert pack.metadata == {}


def test_pack_with_metadata() -> None:
    pack = Pack(
        name="example",
        version="1.0.0",
        prompts={"intro": "hi"},
        metadata={"owner": "competitive_intelligence", "schema": "battle_card_v1"},
    )
    assert pack.metadata == {
        "owner": "competitive_intelligence",
        "schema": "battle_card_v1",
    }


# ----------------------------------------------------------------------
# Empty registry behaviour (acceptance criterion: core can run without
# any pack registered)
# ----------------------------------------------------------------------


def test_get_pack_returns_none_for_unknown_name() -> None:
    assert get_pack("never_registered") is None


def test_get_pack_returns_none_for_unknown_version() -> None:
    register_pack(Pack(name="example", version="1.0.0", prompts={}))
    assert get_pack("example", version="2.0.0") is None


def test_list_packs_returns_empty_when_unregistered() -> None:
    assert list_packs() == []


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def test_register_pack_makes_pack_retrievable() -> None:
    pack = Pack(name="example", version="1.0.0", prompts={"intro": "hi"})
    register_pack(pack)
    assert get_pack("example") is pack


def test_register_pack_is_idempotent_for_identical_content() -> None:
    pack = Pack(name="example", version="1.0.0", prompts={"intro": "hi"})
    register_pack(pack)
    register_pack(pack)  # must not raise
    assert get_pack("example") is pack


def test_register_pack_raises_on_conflicting_content() -> None:
    register_pack(Pack(name="example", version="1.0.0", prompts={"a": "1"}))
    with pytest.raises(ValueError, match="already registered"):
        register_pack(Pack(name="example", version="1.0.0", prompts={"a": "2"}))


def test_register_pack_allows_multiple_versions_of_same_name() -> None:
    v1 = Pack(name="example", version="1.0.0", prompts={"intro": "v1 text"})
    v2 = Pack(name="example", version="2.0.0", prompts={"intro": "v2 text"})
    register_pack(v1)
    register_pack(v2)

    assert get_pack("example", version="1.0.0") is v1
    assert get_pack("example", version="2.0.0") is v2


# ----------------------------------------------------------------------
# Version selection
# ----------------------------------------------------------------------


def test_get_pack_without_version_returns_latest() -> None:
    v1 = Pack(name="example", version="1.0.0", prompts={})
    v2 = Pack(name="example", version="2.0.0", prompts={})
    v110 = Pack(name="example", version="1.10.0", prompts={})
    register_pack(v1)
    register_pack(v2)
    register_pack(v110)

    # Lexicographic on version string -- "2.0.0" > "1.10.0" > "1.0.0"
    assert get_pack("example") is v2


def test_get_pack_explicit_version_overrides_latest() -> None:
    v1 = Pack(name="example", version="1.0.0", prompts={"k": "old"})
    v2 = Pack(name="example", version="2.0.0", prompts={"k": "new"})
    register_pack(v1)
    register_pack(v2)

    assert get_pack("example", version="1.0.0") is v1
    assert get_pack("example", version="2.0.0") is v2


# ----------------------------------------------------------------------
# Listing + ordering
# ----------------------------------------------------------------------


def test_list_packs_orders_by_name_then_version() -> None:
    a1 = Pack(name="alpha", version="1.0.0", prompts={})
    a2 = Pack(name="alpha", version="2.0.0", prompts={})
    b1 = Pack(name="beta", version="1.0.0", prompts={})
    register_pack(b1)  # registered out of order to verify sort
    register_pack(a2)
    register_pack(a1)

    assert list_packs() == [a1, a2, b1]


# ----------------------------------------------------------------------
# Test isolation
# ----------------------------------------------------------------------


def test_clear_packs_empties_registry() -> None:
    register_pack(Pack(name="example", version="1.0.0", prompts={}))
    assert list_packs() != []
    clear_packs()
    assert list_packs() == []
