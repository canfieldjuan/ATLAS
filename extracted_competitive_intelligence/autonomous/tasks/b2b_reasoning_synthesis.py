"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.

Programmatically copies every non-dunder name (including underscore-
prefixed helpers that ``from X import *`` would drop). Phase 2 replaces
this with a standalone implementation gated on
EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from __future__ import annotations

import importlib as _importlib

_src = _importlib.import_module("atlas_brain.autonomous.tasks.b2b_reasoning_synthesis")
_g = globals()
for _name in dir(_src):
    if not _name.startswith("__"):
        _g[_name] = getattr(_src, _name)
del _importlib, _src, _g, _name  # type: ignore[name-defined]
