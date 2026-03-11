"""Reasoning Agent -- cross-domain event-driven intelligence layer."""

from .config import ReasoningConfig
from .events import AtlasEvent, EventType, emit_event
from .event_bus import EventBus
from .entity_locks import EntityLockManager

__all__ = [
    "ReasoningConfig",
    "AtlasEvent",
    "EventType",
    "emit_event",
    "EventBus",
    "EntityLockManager",
]

# ---------------------------------------------------------------------------
# Stratified reasoning engine (lazy singleton)
# ---------------------------------------------------------------------------

_stratified_reasoner = None


def get_stratified_reasoner():
    """Return the app-scoped StratifiedReasoner, or None if not initialised."""
    return _stratified_reasoner


async def init_stratified_reasoner(db_pool) -> None:
    """Initialise the stratified reasoning engine (call from app startup).

    *db_pool*: atlas_brain.storage.database.DatabasePool (must be initialised).
    """
    global _stratified_reasoner

    from .semantic_cache import SemanticCache
    from .episodic_store import EpisodicStore
    from .metacognition import MetacognitiveMonitor
    from .stratified_reasoner import StratifiedReasoner

    cache = SemanticCache(db_pool)
    episodic = EpisodicStore()
    meta = MetacognitiveMonitor(db_pool)

    try:
        await episodic.ensure_indexes()
    except Exception:
        import logging
        logging.getLogger("atlas.reasoning").warning(
            "Failed to ensure episodic indexes (Neo4j may be unavailable)",
            exc_info=True,
        )

    _stratified_reasoner = StratifiedReasoner(cache, episodic, metacognition=meta)

    # Attach temporal engine for ad-hoc vendor temporal queries
    try:
        from .temporal import TemporalEngine
        _stratified_reasoner._temporal = TemporalEngine(db_pool)
    except Exception:
        import logging
        logging.getLogger("atlas.reasoning").warning(
            "Failed to init TemporalEngine (non-fatal)", exc_info=True,
        )


async def close_stratified_reasoner() -> None:
    """Flush metacognition and shutdown the episodic store connection."""
    global _stratified_reasoner
    if _stratified_reasoner is not None:
        if _stratified_reasoner._meta:
            try:
                await _stratified_reasoner._meta.flush()
            except Exception:
                pass
        await _stratified_reasoner._episodic.close()
        _stratified_reasoner = None
