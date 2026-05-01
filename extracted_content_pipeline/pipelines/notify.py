from __future__ import annotations

import os
from typing import Any

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    async def send_pipeline_notification(*args: Any, **kwargs: Any) -> None:
        return None
else:
    from atlas_brain.pipelines.notify import *
