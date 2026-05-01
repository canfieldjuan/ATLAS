"""Phase 1 bridge: re-exports atlas_brain.storage.database so scaffolded
modules with `from ...storage.database import get_db_pool` resolve
cleanly. The ``DatabasePool`` global is shared with atlas_brain by virtue
of this re-export.

Phase 3 work introduces a ``LLMInfraStorage`` Protocol so the scaffold
can run against any asyncpg-compatible pool, not just atlas_brain's
singleton.
"""
from atlas_brain.storage.database import *  # noqa: F401,F403
from atlas_brain.storage.database import get_db_pool  # noqa: F401
