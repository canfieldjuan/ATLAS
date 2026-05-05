"""LLM router bridge for extracted competitive intelligence.

Default mode preserves Atlas routing. Standalone mode delegates to
extracted_llm_infrastructure so competitive intelligence does not import
the monolith for campaign/vendor briefing LLM selection.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os


def _bridge(module_name: str) -> None:
    src = _importlib.import_module(module_name)
    globals_dict = globals()
    for name in dir(src):
        if not name.startswith("__"):
            globals_dict[name] = getattr(src, name)


if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    _os.environ.setdefault("EXTRACTED_LLM_INFRA_STANDALONE", "1")
    _bridge("extracted_llm_infrastructure.services.llm_router")
else:
    _bridge("atlas_brain.services.llm_router")

del _bridge, _importlib, _os
