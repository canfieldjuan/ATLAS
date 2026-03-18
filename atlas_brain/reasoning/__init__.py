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

    import asyncio as _aio
    try:
        await _aio.wait_for(episodic.ensure_indexes(), timeout=10.0)
    except _aio.TimeoutError:
        import logging
        logging.getLogger("atlas.reasoning").warning(
            "Episodic index setup timed out after 10s -- Neo4j may be slow or unreachable. "
            "Episodic memory will operate in degraded mode (no similarity search).",
        )
        episodic._degraded = True
    except Exception:
        import logging
        logging.getLogger("atlas.reasoning").warning(
            "Failed to ensure episodic indexes (Neo4j may be unavailable). "
            "Episodic memory will operate in degraded mode.",
            exc_info=True,
        )
        episodic._degraded = True

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

    # Attach ecosystem analyzer + narrative engine
    try:
        from .ecosystem import EcosystemAnalyzer
        from .narrative import NarrativeEngine

        _stratified_reasoner._ecosystem = EcosystemAnalyzer(db_pool)
        _stratified_reasoner._narrative = NarrativeEngine(db_pool)
    except Exception:
        import logging
        logging.getLogger("atlas.reasoning").warning(
            "Ecosystem/Narrative engine failed to init (non-fatal)",
            exc_info=True,
        )

    # Attach knowledge graph query + trigger correlator
    try:
        from neo4j import AsyncGraphDatabase
        from .knowledge_graph import KnowledgeGraphQuery
        from .trigger_events import TriggerCorrelator

        from .config import ReasoningConfig
        _rcfg = ReasoningConfig()
        _neo4j_driver = AsyncGraphDatabase.driver(
            _rcfg.neo4j_bolt_url,
            auth=(_rcfg.neo4j_user, _rcfg.neo4j_password),
            connection_timeout=10.0,
        )
        _stratified_reasoner._graph = KnowledgeGraphQuery(_neo4j_driver)
        _stratified_reasoner._triggers = TriggerCorrelator(db_pool)
        _stratified_reasoner._neo4j_driver = _neo4j_driver
    except Exception:
        import logging
        logging.getLogger("atlas.reasoning").warning(
            "Knowledge graph / trigger engine failed to start (non-fatal)",
            exc_info=True,
        )


async def close_stratified_reasoner() -> None:
    """Flush metacognition, release LLM slots, and shutdown connections."""
    global _stratified_reasoner
    if _stratified_reasoner is not None:
        if _stratified_reasoner._meta:
            try:
                await _stratified_reasoner._meta.flush()
            except Exception:
                pass
        # Close Neo4j driver if attached
        if hasattr(_stratified_reasoner, "_neo4j_driver"):
            try:
                await _stratified_reasoner._neo4j_driver.close()
            except Exception:
                pass
        await _stratified_reasoner._episodic.close()
        _stratified_reasoner = None

    # Release reasoning LLM slots (heavy + light) without touching the
    # primary registry slot used by voice/chat/agents.
    try:
        from ..services import llm_registry
        from .llm_utils import _SLOT_HEAVY, _SLOT_LIGHT
        llm_registry.release_slot(_SLOT_HEAVY)
        llm_registry.release_slot(_SLOT_LIGHT)
    except Exception:
        pass
