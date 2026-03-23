"""Helpers for backfilling legacy B2B reasoning rows into contract-first shape."""

from __future__ import annotations

import json
from typing import Any

from ..autonomous.tasks._b2b_synthesis_reader import load_synthesis_view

_FLAT_SECTION_KEYS = (
    "causal_narrative",
    "segment_playbook",
    "timing_intelligence",
    "competitive_reframes",
    "migration_proof",
)

_CONTRACT_MIRROR_KEYS = (
    "vendor_core_reasoning",
    "displacement_reasoning",
    "category_reasoning",
)


def _json_clone(value: Any) -> Any:
    """Return a detached clone of JSON-like payloads."""
    return json.loads(json.dumps(value))


def _vendor_name_for_payload(payload: dict[str, Any], fallback: str = "") -> str:
    """Pick the best vendor hint available for synthesis-view normalization."""
    vendor = str(payload.get("vendor") or fallback or "").strip()
    return vendor


def normalize_reasoning_payload(
    payload: Any,
    *,
    vendor_name: str = "",
    synthesis_mode: bool = False,
) -> Any:
    """Normalize one reasoning-bearing payload to contract-first form.

    This is safe for current canonical rows; unchanged inputs are returned
    semantically identical. Legacy rows with flat sections or top-level
    contract mirrors are rewritten to store only ``reasoning_contracts`` plus
    the existing wedge/meta fields already used by consumers.
    """
    if isinstance(payload, list):
        return [
            normalize_reasoning_payload(
                item,
                vendor_name=vendor_name,
                synthesis_mode=synthesis_mode,
            )
            for item in payload
        ]
    if not isinstance(payload, dict):
        return payload

    has_reasoning_shape = any(
        key in payload for key in ("reasoning_contracts", *_FLAT_SECTION_KEYS, *_CONTRACT_MIRROR_KEYS)
    )
    if not has_reasoning_shape:
        return payload

    view = load_synthesis_view(
        payload,
        _vendor_name_for_payload(payload, fallback=vendor_name),
        schema_version=str(
            payload.get("schema_version")
            or payload.get("synthesis_schema_version")
            or ""
        ),
    )
    contracts = view.materialized_contracts()
    if not contracts:
        return payload

    normalized = dict(payload)
    normalized["reasoning_contracts"] = contracts
    for key in (*_FLAT_SECTION_KEYS, *_CONTRACT_MIRROR_KEYS):
        normalized.pop(key, None)

    if synthesis_mode:
        normalized["reasoning_shape"] = "contracts_first_v1"
        if view.meta:
            normalized["meta"] = _json_clone(view.meta)
    if view.primary_wedge:
        normalized["synthesis_wedge"] = view.primary_wedge.value
        normalized["synthesis_wedge_label"] = view.wedge_label

    return normalized

