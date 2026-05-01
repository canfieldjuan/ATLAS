"""Phase 2 bridge: ``BaseModelService`` entry point for the LLM-infrastructure
scaffold.

Default mode: re-export from ``atlas_brain.services.base`` (which imports
torch).

Standalone mode (``EXTRACTED_LLM_INFRA_STANDALONE=1``): use the
torch-free copy under ``_standalone/base.py`` so the scaffold can be
imported in environments without GPU tooling installed.
"""

from __future__ import annotations

import os as _os

if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
    from .._standalone.base import (  # noqa: F401
        BaseModelService,
        InferenceTimer,
    )
else:
    from atlas_brain.services.base import *  # noqa: F401,F403
