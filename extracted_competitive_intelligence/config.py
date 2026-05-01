"""Config bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.config. Standalone mode uses the
product-local settings in _standalone.config.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    from ._standalone.config import (  # noqa: F401
        B2BChurnSubConfig,
        B2BScrapeSubConfig,
        CampaignSequenceSubConfig,
        CompIntelSettings,
        LLMSubConfig,
        MCPSubConfig,
        SaasAuthSubConfig,
        settings,
    )
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.config")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
