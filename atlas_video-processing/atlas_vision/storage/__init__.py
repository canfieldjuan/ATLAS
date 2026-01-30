"""
Database storage for atlas_vision.

Provides PostgreSQL access for person recognition data.
"""

from .config import db_settings
from .database import (
    DatabasePool,
    get_db_pool,
    init_database,
    close_database,
)

__all__ = [
    "db_settings",
    "DatabasePool",
    "get_db_pool",
    "init_database",
    "close_database",
]
