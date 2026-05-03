"""Phase 2 bridge: skills entry point for the LLM-infrastructure scaffold.

The exact LLM cache (`services/b2b/llm_exact_cache.py:160`) lazily imports
`get_skill_registry` from `...skills` to build skill-message envelopes. In
default mode that resolves to `atlas_brain.skills`. A standalone substrate
for skills is not yet carved out -- skill prompts are owned by Atlas's
content/competitive-intelligence pipelines and have not been classified for
extraction yet. Phase 3 work that decouples cache helpers from skills (or
extracts skills behind a Protocol) will replace this bridge with a proper
substrate.

Default mode (``EXTRACTED_LLM_INFRA_STANDALONE`` unset/false): re-export
from ``atlas_brain.skills`` so the cache helpers run as a sibling of Atlas.

Standalone mode (``EXTRACTED_LLM_INFRA_STANDALONE=1``): raise on access.
The cache helpers that need skills are not callable in standalone mode
until Phase 3 extracts (or stubs) the skills layer; lookup/store paths
that do not call `build_skill_messages` / `build_skill_request_envelope`
remain usable.
"""

from __future__ import annotations

import os as _os

if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
    def get_skill_registry(*_args, **_kwargs):  # type: ignore[no-redef]
        raise NotImplementedError(
            "extracted_llm_infrastructure.skills is not implemented in "
            "standalone mode. The skills layer is owned by content/"
            "competitive-intelligence pipelines and has not been extracted. "
            "Phase 3 will decouple cache helpers from skills."
        )
else:
    from atlas_brain.skills import *  # noqa: F401,F403
    from atlas_brain.skills import get_skill_registry  # noqa: F401
