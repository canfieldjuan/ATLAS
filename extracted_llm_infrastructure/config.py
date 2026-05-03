"""Phase 2 bridge: settings entry point for the LLM-infrastructure scaffold.

Default mode (``EXTRACTED_LLM_INFRA_STANDALONE`` unset/false): re-export
from ``atlas_brain.config`` so the scaffold runs as a sibling of Atlas
with shared global settings.

Standalone mode (``EXTRACTED_LLM_INFRA_STANDALONE=1``): use the slim
``LLMInfraSettings`` carved out of atlas_brain. This makes the package
runnable without atlas_brain on ``sys.path``.
"""

from __future__ import annotations

import os as _os

if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
    from ._standalone.config import (  # noqa: F401
        FTLTracingSubConfig,
        LLMInfraSettings,
        LLMSubConfig,
        ModelPricingConfig,
        ReasoningSubConfig,
        B2BChurnSubConfig,
        ProviderCostSubConfig,
        settings,
    )
else:
    from atlas_brain.config import *  # noqa: F401,F403
    from atlas_brain.config import settings  # noqa: F401
