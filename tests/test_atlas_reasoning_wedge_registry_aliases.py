"""Pin atlas's re-exports from ``atlas_brain.reasoning.wedge_registry``.

PR-D7b2 promoted the wedge registry module into
``extracted_reasoning_core.wedge_registry`` and replaced atlas's
159-line fork with a thin re-export wrapper. Internal callers
(B2B synthesis validation/contracts/reader, blog post generation,
``tests/test_reasoning_synthesis_v2.py``) keep their existing import
sites working through the wrapper.

Mirrors ``test_atlas_reasoning_tiers_aliases.py`` (PR-D7b1) and
``test_atlas_reasoning_graph_aliases.py`` (PR-C4e1): each alias must
point to the exact same object as the core export, plus an AST-level
guard that the wrapper body contains no def/class redefinitions.
"""

from __future__ import annotations

from atlas_brain.reasoning import wedge_registry as atlas_wedge_registry
from extracted_reasoning_core import wedge_registry as core_wedge_registry


def test_wedge_alias_identity() -> None:
    assert atlas_wedge_registry.Wedge is core_wedge_registry.Wedge


def test_wedge_meta_alias_identity() -> None:
    assert atlas_wedge_registry.WedgeMeta is core_wedge_registry.WedgeMeta


def test_wedge_enum_values_alias_identity() -> None:
    # The frozenset itself must be the same object so a caller that
    # holds a reference compares ``is`` against any other reader.
    assert (
        atlas_wedge_registry.WEDGE_ENUM_VALUES
        is core_wedge_registry.WEDGE_ENUM_VALUES
    )


def test_wedge_from_archetype_alias_identity() -> None:
    assert (
        atlas_wedge_registry.wedge_from_archetype
        is core_wedge_registry.wedge_from_archetype
    )


def test_validate_wedge_alias_identity() -> None:
    assert atlas_wedge_registry.validate_wedge is core_wedge_registry.validate_wedge


def test_get_wedge_meta_alias_identity() -> None:
    assert atlas_wedge_registry.get_wedge_meta is core_wedge_registry.get_wedge_meta


def test_get_sales_motion_alias_identity() -> None:
    assert (
        atlas_wedge_registry.get_sales_motion
        is core_wedge_registry.get_sales_motion
    )


def test_get_required_pools_alias_identity() -> None:
    assert (
        atlas_wedge_registry.get_required_pools
        is core_wedge_registry.get_required_pools
    )


def test_atlas_wedge_registry_all_matches_re_export_set() -> None:
    # __all__ pins the public-surface contract.
    assert set(atlas_wedge_registry.__all__) == {
        "WEDGE_ENUM_VALUES",
        "Wedge",
        "WedgeMeta",
        "get_required_pools",
        "get_sales_motion",
        "get_wedge_meta",
        "validate_wedge",
        "wedge_from_archetype",
    }


def test_atlas_wedge_registry_does_not_redefine_symbols() -> None:
    # Defensive: the wrapper module body should not contain any
    # ``def``/``class`` re-implementations -- only the re-export
    # imports + ``__all__`` literal. Catches a refactor that
    # accidentally forks the implementation back into atlas-side.
    import ast
    import inspect

    source = inspect.getsource(atlas_wedge_registry)
    tree = ast.parse(source)

    redefinitions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            redefinitions.append(node.name)

    assert redefinitions == [], (
        f"atlas_brain.reasoning.wedge_registry should be a pure re-export "
        f"wrapper, but it defines: {redefinitions}"
    )
