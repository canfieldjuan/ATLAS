from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _StandalonePool:
    is_initialized: bool = False

    async def initialize(self) -> None:
        self.is_initialized = True


_POOL = _StandalonePool()


def get_db_pool() -> _StandalonePool:
    return _POOL
