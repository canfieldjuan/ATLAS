from __future__ import annotations

from pathlib import Path

import extracted_competitive_intelligence.reasoning.wedge_registry as comp_wedges
import extracted_content_pipeline.reasoning.wedge_registry as content_wedges
from extracted_reasoning_core.api import (
    WEDGE_ENUM_VALUES,
    Wedge,
    get_required_pools,
    get_sales_motion,
    get_wedge_meta,
    validate_wedge,
    wedge_from_archetype,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_shared_wedge_registry_exposes_full_atlas_contract() -> None:
    assert len(WEDGE_ENUM_VALUES) == 10
    assert WEDGE_ENUM_VALUES == {wedge.value for wedge in Wedge}

    for value in WEDGE_ENUM_VALUES:
        assert validate_wedge(value) is Wedge(value)


def test_wedge_from_archetype_maps_known_and_unknown_shapes() -> None:
    assert wedge_from_archetype("pricing_shock") is Wedge.PRICE_SQUEEZE
    assert wedge_from_archetype("support_collapse") is Wedge.SUPPORT_EROSION
    assert wedge_from_archetype("not_a_known_archetype") is Wedge.SEGMENT_MISMATCH


def test_invalid_or_legacy_compound_wedges_do_not_validate() -> None:
    assert validate_wedge("pricing_support") is None
    assert validate_wedge("feature_reliability") is None
    assert validate_wedge("support") is None
    assert validate_wedge("") is None


def test_wedge_metadata_helpers_return_catalog_values() -> None:
    meta = get_wedge_meta(Wedge.PRICE_SQUEEZE)

    assert meta.wedge is Wedge.PRICE_SQUEEZE
    assert meta.label == "Price Squeeze"
    assert "pricing_shock" in meta.archetype_map
    assert get_sales_motion(Wedge.PRICE_SQUEEZE) == meta.sales_motion
    assert tuple(get_required_pools(Wedge.PRICE_SQUEEZE)) == meta.required_pools


def test_product_wedge_wrappers_reexport_shared_core_objects() -> None:
    assert content_wedges.Wedge is Wedge
    assert comp_wedges.Wedge is Wedge
    assert content_wedges.WEDGE_ENUM_VALUES is WEDGE_ENUM_VALUES
    assert comp_wedges.WEDGE_ENUM_VALUES is WEDGE_ENUM_VALUES
    assert content_wedges.validate_wedge("price_squeeze") is Wedge.PRICE_SQUEEZE
    assert comp_wedges.validate_wedge("price_squeeze") is Wedge.PRICE_SQUEEZE


def test_product_wedge_wrappers_do_not_bridge_to_atlas() -> None:
    for path in (
        "extracted_content_pipeline/reasoning/wedge_registry.py",
        "extracted_competitive_intelligence/reasoning/wedge_registry.py",
    ):
        text = _read(path)
        assert "extracted_reasoning_core.api" in text
        assert "atlas_brain.reasoning.wedge_registry" not in text
        assert "importlib.import_module" not in text
