"""Standalone substrate for the _b2b_witnesses helper module.

Real implementation lives in atlas_brain; this module exists only so
that mirrored modules importing build_vendor_witness_artifacts (e.g.,
``autonomous/tasks/_b2b_pool_compression.py``) can be imported under
``EXTRACTED_PIPELINE_STANDALONE=1`` without touching atlas_brain.

Runtime behavior is no-op: build_vendor_witness_artifacts returns an
empty witness pack and empty section packets regardless of inputs.
Tasks that need real witness extraction should run in atlas_brain
mode.

See plans/PR-Decouple-PoolCompression-1.md.
"""
from __future__ import annotations

from typing import Any


def build_vendor_witness_artifacts(
    vendor_name: str,
    reviews: list[dict[str, Any]] | None,
    **_kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {}
