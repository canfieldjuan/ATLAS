"""LLM bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.pipelines.llm. Standalone mode
delegates to extracted_llm_infrastructure so competitive intelligence
depends on the extracted LLM product instead of the monolith.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    _os.environ.setdefault("EXTRACTED_LLM_INFRA_STANDALONE", "1")
    from extracted_llm_infrastructure.pipelines.llm import *  # noqa: F401,F403
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.pipelines.llm")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
