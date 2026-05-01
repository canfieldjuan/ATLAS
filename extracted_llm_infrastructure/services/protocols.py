"""Phase 2 bridge: protocol/dataclass entry point for the LLM-infrastructure
scaffold.

Default mode: re-export from ``atlas_brain.services.protocols``.
Standalone mode (``EXTRACTED_LLM_INFRA_STANDALONE=1``): use the
torch-free local copy under ``_standalone/protocols.py``.
"""

from __future__ import annotations

import os as _os

if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
    from .._standalone.protocols import (  # noqa: F401
        InferenceMetrics,
        LLMService,
        Message,
        ModelInfo,
    )
else:
    from atlas_brain.services.protocols import *  # noqa: F401,F403
