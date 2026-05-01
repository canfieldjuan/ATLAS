"""Phase 1 package bridge: re-exports every non-dunder name from
atlas_brain.services.b2b so scaffolded modules with from PACKAGE import name
imports (where name is a submodule or attribute defined in atlas's
package __init__.py) resolve cleanly.

Phase 2 replaces with a standalone implementation gated on
EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from __future__ import annotations

import importlib as _importlib

def _bridge() -> None:
    src = _importlib.import_module("atlas_brain.services.b2b")
    g = globals()
    for name in dir(src):
        if not name.startswith("__"):
            g[name] = getattr(src, name)


_bridge()
del _bridge, _importlib
