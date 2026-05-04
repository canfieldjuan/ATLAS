"""CRM provider bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.services.crm_provider. Standalone
mode exposes a configurable CRM provider port and fails closed until a
host adapter is registered.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    from .._standalone.crm_provider import (  # noqa: F401
        CRMProvider,
        CRMProviderNotConfigured,
        configure_crm_provider,
        get_crm_provider,
    )
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.services.crm_provider")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
