"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks.b2b_reasoning_synthesis.

Programmatically copies every non-dunder name (including underscore-
prefixed helpers that ``from X import *`` would drop). Phase 2 replaces
this with a standalone implementation gated on
EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from __future__ import annotations

import importlib as _importlib

def _bridge() -> None:
    src = _importlib.import_module("atlas_brain.autonomous.tasks.b2b_reasoning_synthesis")
    g = globals()
    for name in dir(src):
        if not name.startswith("__"):
            g[name] = getattr(src, name)


_bridge()
del _bridge, _importlib
