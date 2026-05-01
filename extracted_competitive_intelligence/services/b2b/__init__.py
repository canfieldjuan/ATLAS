"""Phase 1 package bridge: lazily exposes names from atlas_brain.services.b2b.

PEP 562 ``__getattr__`` resolves ``from PACKAGE import some_name`` at
runtime by delegating to the atlas_brain peer package's __init__
namespace. This avoids triggering the atlas_brain peer's heavy import
chain at scaffold-load time -- e.g., importing
``extracted_competitive_intelligence.services.vendor_registry`` no
longer eagerly loads ``atlas_brain.services`` (which pulls in the
torch/llm chain) just to satisfy a hypothetical
``from ...services import llm_registry`` runtime fallback.

Submodule imports of the form ``from PACKAGE import submodule_name``
are handled by Python's native import machinery from the scaffold
filesystem; this hook only fires for non-submodule attributes.

Phase 2 replaces this with a standalone implementation gated on
EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from __future__ import annotations

import importlib
import os
from typing import Any


def __getattr__(name: str) -> Any:
    if os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
        raise AttributeError(
            f"module {__name__!r} has no standalone attribute {name!r}; "
            "register an explicit product service instead"
        )
    src = importlib.import_module("atlas_brain.services.b2b")
    try:
        return getattr(src, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
