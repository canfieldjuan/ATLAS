"""Phase 2 bridge: database entry point for the LLM-infrastructure scaffold.

Default mode: re-export from ``atlas_brain.storage.database``. The
``DatabasePool`` global singleton is shared with atlas_brain.

Standalone mode (``EXTRACTED_LLM_INFRA_STANDALONE=1``): use the slim
asyncpg wrapper under ``_standalone/database.py``. Configuration via
``ATLAS_DB_*`` environment variables (matches atlas_brain for
compatibility).
"""

from __future__ import annotations

import os as _os

if _os.environ.get("EXTRACTED_LLM_INFRA_STANDALONE") == "1":
    from .._standalone.database import (  # noqa: F401
        DatabasePool,
        get_db_pool,
    )
else:
    from atlas_brain.storage.database import *  # noqa: F401,F403
    from atlas_brain.storage.database import get_db_pool  # noqa: F401
