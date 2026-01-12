"""
Storage module for Atlas Brain.

Provides persistent storage for:
- Conversation history
- User sessions
- Terminal registrations
"""

from .config import DatabaseConfig, db_settings
from .database import DatabasePool, get_db_pool

__all__ = [
    "DatabaseConfig",
    "db_settings",
    "DatabasePool",
    "get_db_pool",
]
