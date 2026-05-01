"""Phase 2 bridge: registry entry point for the LLM-infrastructure scaffold.

Default mode: re-export from ``atlas_brain.services.registry``. The
``llm_registry`` global is shared with atlas_brain.

Standalone mode (``EXTRACTED_LLM_INFRA_STANDALONE=1``): use the local
copy under ``_standalone/registry.py``. The ``llm_registry`` global is
process-local to the standalone package -- atlas_brain is not on
``sys.path`` in this mode.
"""

from __future__ import annotations

import os as _os

if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
    from .._standalone.registry import (  # noqa: F401
        ServiceRegistry,
        llm_registry,
        register_llm,
    )
else:
    from atlas_brain.services.registry import *  # noqa: F401,F403
    from atlas_brain.services.registry import llm_registry, register_llm  # noqa: F401
