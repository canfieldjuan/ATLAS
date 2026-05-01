from __future__ import annotations

import os
from dataclasses import dataclass

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    @dataclass
    class _StandalonePool:
        is_initialized: bool = False

        async def initialize(self) -> None:
            self.is_initialized = True

    _POOL = _StandalonePool()

    def get_db_pool() -> _StandalonePool:
        return _POOL
else:
    from atlas_brain.storage.database import *
