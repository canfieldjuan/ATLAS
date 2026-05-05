"""Consumer adapter helpers for reasoning payload overlays.

Thin compatibility shim for MCP/API consumers that overlay reasoning
fields onto response payloads. Existing callers (MCP signals overlay,
adapter regression tests) keep their two-function surface; the actual
projection logic now lives behind the M5-alpha typed envelope in
``atlas_brain.reasoning.vendor_pressure``.

Wire shape preserved bit-for-bit: ``reasoning_summary_fields_from_view``
and ``reasoning_detail_fields_from_view`` return the same flat dicts
they always have, including the sparse-entry contract from PR #184
(stable 8-key set, ``None`` scalars, empty lists).
"""

from __future__ import annotations

from typing import Any


def reasoning_summary_fields_from_view(view: object) -> dict[str, Any]:
    """Return stable reasoning summary fields derived from a synthesis view."""
    from atlas_brain.reasoning.vendor_pressure import (
        VendorPressureConsumer,
        vendor_pressure_result_from_synthesis_view,
    )

    result = vendor_pressure_result_from_synthesis_view(view)
    return dict(VendorPressureConsumer().to_summary_fields(result))


def reasoning_detail_fields_from_view(view: object) -> dict[str, Any]:
    """Return stable reasoning detail fields derived from a synthesis view.

    List-valued fields are guaranteed to be lists (never None) even when
    the upstream entry has explicit null values for those keys -- the
    typed envelope pathway preserves the same sparse-entry guard the
    legacy implementation provided (PR #184).
    """
    from atlas_brain.reasoning.vendor_pressure import (
        VendorPressureConsumer,
        vendor_pressure_result_from_synthesis_view,
    )

    result = vendor_pressure_result_from_synthesis_view(view)
    return dict(VendorPressureConsumer().to_detail_fields(result))
