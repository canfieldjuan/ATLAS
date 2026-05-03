"""Compatibility wrapper for the shared extracted reasoning wedge registry."""

from __future__ import annotations

from extracted_reasoning_core.api import (
    WEDGE_ENUM_VALUES,
    Wedge,
    WedgeMeta,
    get_required_pools,
    get_sales_motion,
    get_wedge_meta,
    validate_wedge,
    wedge_from_archetype,
)


__all__ = [
    "WEDGE_ENUM_VALUES",
    "Wedge",
    "WedgeMeta",
    "get_required_pools",
    "get_sales_motion",
    "get_wedge_meta",
    "validate_wedge",
    "wedge_from_archetype",
]
