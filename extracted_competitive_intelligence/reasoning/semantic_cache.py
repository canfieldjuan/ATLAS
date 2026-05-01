"""Phase 1 bridge: re-exports atlas_brain.reasoning.semantic_cache.

Programmatically copies every non-dunder name (including underscore-
prefixed helpers that from X import * would drop). Required because
many scaffolded modules import private helpers from atlas_brain peers
via from .X import _foo lazily inside function bodies. Phase 2
replaces this with a standalone implementation gated on
EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from __future__ import annotations

import importlib as _importlib

_src = _importlib.import_module("atlas_brain.reasoning.semantic_cache")
_g = globals()
for _name in dir(_src):
    if not _name.startswith("__"):
        _g[_name] = getattr(_src, _name)
del _importlib, _src, _g, _name  # type: ignore[name-defined]
