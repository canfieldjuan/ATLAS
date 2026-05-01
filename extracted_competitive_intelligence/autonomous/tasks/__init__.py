"""Phase 1 package bridge: re-exports every non-dunder name from
atlas_brain.autonomous.tasks so scaffolded modules with from PACKAGE import name
imports (where name is a submodule or attribute defined in atlas's
package __init__.py) resolve cleanly.

Phase 2 replaces with a standalone implementation gated on
EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from __future__ import annotations

import importlib as _importlib

_src = _importlib.import_module("atlas_brain.autonomous.tasks")
_g = globals()
for _name in dir(_src):
    if not _name.startswith("__"):
        _g[_name] = getattr(_src, _name)
del _importlib, _src, _g, _name  # type: ignore[name-defined]
