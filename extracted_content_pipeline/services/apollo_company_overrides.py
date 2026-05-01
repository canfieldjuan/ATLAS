from __future__ import annotations

import os

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    async def fetch_company_override_map(*args, **kwargs) -> dict[str, str]:
        return {}
else:
    from atlas_brain.services.apollo_company_overrides import *
