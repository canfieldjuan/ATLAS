"""Protocol bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.services.protocols. Standalone mode
reuses the extracted LLM infrastructure protocol definitions.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    from .._standalone.protocols import (  # noqa: F401
        InferenceMetrics,
        LLMService,
        Message,
        ModelInfo,
    )
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.services.protocols")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
