"""LLM exact-cache bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.services.b2b.llm_exact_cache.
Standalone mode delegates to extracted_llm_infrastructure so battle-card
prompt envelopes use the extracted LLM substrate instead of Atlas.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    _os.environ.setdefault("EXTRACTED_LLM_INFRA_STANDALONE", "1")
    from extracted_llm_infrastructure.services.b2b.llm_exact_cache import *  # noqa: F401,F403
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.services.b2b.llm_exact_cache")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
