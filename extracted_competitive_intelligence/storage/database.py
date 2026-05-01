"""Database bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.storage.database. Standalone mode
uses the slim asyncpg wrapper from extracted_llm_infrastructure so the
two extracted products share one database substrate.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    from extracted_llm_infrastructure.storage.database import (  # noqa: F401
        DatabasePool,
        get_db_pool,
    )
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.storage.database")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
