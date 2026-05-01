"""Auth bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.auth.dependencies. Standalone mode
uses fail-closed local auth hooks; host apps should inject their own
dependencies before serving routes.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    from .._standalone.auth import AuthUser, require_auth  # noqa: F401
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.auth.dependencies")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
