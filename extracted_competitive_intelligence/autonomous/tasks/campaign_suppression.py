"""Suppression bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain suppression helpers. Standalone mode
exposes a configurable suppression policy and fails closed until a host
adapter is registered.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    from ..._standalone.campaign_suppression import (  # noqa: F401
        SuppressionPolicy,
        SuppressionPolicyNotConfigured,
        configure_suppression_policy,
        is_suppressed,
    )
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.autonomous.tasks.campaign_suppression")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
