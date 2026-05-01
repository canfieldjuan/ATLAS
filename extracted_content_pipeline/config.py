from __future__ import annotations

import os

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    from .settings import build_settings

    settings = build_settings()
else:
    from atlas_brain.config import settings

__all__ = ["settings"]
