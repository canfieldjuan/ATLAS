"""Standalone substrate for the brand registry service.

Real implementation lives in atlas_brain; this module exists only so
that mirrored modules importing brand_registry (e.g.,
``autonomous/tasks/competitive_intelligence.py``) can be imported
under ``EXTRACTED_PIPELINE_STANDALONE=1`` without touching atlas_brain.

Runtime behavior is no-op: ``resolve_brand_name_cached`` returns the
input unchanged; ``_ensure_cache`` is a no-op coroutine. Tasks that
need real brand canonicalization should run in atlas_brain mode.

See plans/PR-Decouple-CompIntel-1.md.
"""
from __future__ import annotations

from typing import Any


def resolve_brand_name_cached(name: Any) -> Any:
    return name


async def _ensure_cache() -> None:
    return None
