"""
Atlas Memory MCP Server.

Exposes the Atlas memory system (PostgreSQL short-term + Neo4j/Graphiti
long-term knowledge graph) to any MCP-compatible client.

Graph tools (via Graphiti wrapper at localhost:8001):
    search_memory            -- vector search facts in knowledge graph
    search_memory_enhanced   -- full pipeline: decompose, expand, rerank, dedupe
    search_memory_temporal   -- temporal-aware search with date filtering
    get_entity               -- get entity and all its relationships
    traverse_graph           -- multi-hop relationship traversal
    find_shortest_path       -- shortest path between two entities
    add_fact                 -- store a single fact in the graph
    add_episode              -- add a structured episode with metadata
    delete_episode           -- remove an episode from the graph
    enhance_prompt           -- enrich a prompt with graph context
    analyze_sentiment        -- semantic similarity sentiment analysis

Conversation tools (via PostgreSQL):
    search_conversations -- ILIKE search on conversation_turns.content
    get_session_history  -- full transcript for a session
    list_sessions        -- list recent sessions

Combined:
    get_context          -- parallel graph + conversation search, unified result

Run:
    python -m atlas_brain.mcp.memory_server          # stdio (Claude Desktop / Cursor)
    python -m atlas_brain.mcp.memory_server --sse    # SSE HTTP transport
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.memory")

# Hard limits to prevent abuse
MAX_RESULTS = 50
MAX_EDGES = 100
MAX_HOPS = 5
MAX_SHORTEST_PATH_HOPS = 10
MAX_ENHANCE_SOURCES = 20
MAX_CONVERSATION_LIMIT = 200
MAX_SESSION_LIMIT = 100
MAX_DAYS = 365


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool (fatal) and check Graphiti health (warn-only)."""
    from ..storage.database import init_database, close_database

    await init_database()
    logger.info("Memory MCP: DB pool initialized")

    # Non-blocking Graphiti health check
    try:
        rag = _rag()
        healthy = await rag.health_check()
        if healthy:
            logger.info("Memory MCP: Graphiti wrapper is healthy")
        else:
            logger.warning("Memory MCP: Graphiti wrapper unhealthy -- graph tools will return empty results")
    except Exception as e:
        logger.warning("Memory MCP: Graphiti health check failed: %s -- graph tools will return empty results", e)

    yield
    await close_database()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "atlas-memory",
    instructions=(
        "Memory server for Atlas. Provides access to the knowledge graph "
        "(long-term facts about entities and relationships) and conversation "
        "history (short-term PostgreSQL-backed session transcripts). "
        "Use search_memory for semantic fact retrieval, search_memory_enhanced "
        "for complex queries (decomposition + reranking), search_memory_temporal "
        "for date-filtered searches, get_entity for entity-centric lookups, "
        "find_shortest_path for entity connections, add_episode for structured "
        "knowledge ingestion, enhance_prompt for RAG context injection, "
        "analyze_sentiment for text sentiment, and search_conversations for "
        "keyword matches in past dialogue. Use get_context for a combined view."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Lazy accessors
# ---------------------------------------------------------------------------

def _rag():
    from ..memory.rag_client import get_rag_client
    return get_rag_client()


def _db():
    from ..storage.database import get_db_pool
    return get_db_pool()


def _conversation_repo():
    from ..storage.repositories.conversation import get_conversation_repo
    return get_conversation_repo()


# ---------------------------------------------------------------------------
# Graph tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_memory(query: str, max_results: int = 10) -> str:
    """Search the knowledge graph for facts matching a query.

    Uses vector similarity search against the Graphiti knowledge graph.
    Returns facts with their names, content, confidence, and timestamps.

    Args:
        query: Natural language search query
        max_results: Maximum facts to return (default 10, max 50)
    """
    max_results = min(max_results, MAX_RESULTS)
    try:
        rag = _rag()
        result = await rag.search(query, max_facts=max_results)
        facts = [
            {
                "uuid": f.uuid,
                "name": f.name,
                "fact": f.fact,
                "confidence": f.confidence,
                "created_at": f.created_at,
            }
            for f in result.facts
        ]
        return json.dumps({"facts": facts, "count": len(facts)}, default=str)
    except Exception as e:
        logger.exception("search_memory error")
        return json.dumps({"error": str(e), "facts": [], "count": 0})


@mcp.tool()
async def get_entity(entity_name: str, max_edges: int = 20) -> str:
    """Get an entity and all its relationships from the knowledge graph.

    Retrieves all edges (relationships/facts) connected to a named entity.
    Complements search_memory by returning ALL known facts about an entity
    rather than just the top-k similar ones.

    Args:
        entity_name: Entity name to look up (e.g. "Juan", "Atlas")
        max_edges: Maximum edges to return (default 20, max 100)
    """
    max_edges = min(max_edges, MAX_EDGES)
    try:
        rag = _rag()
        result = await rag.get_entity_edges(entity_name, max_edges=max_edges)
        edges = [
            {
                "uuid": f.uuid,
                "name": f.name,
                "fact": f.fact,
                "confidence": f.confidence,
                "created_at": f.created_at,
            }
            for f in result.facts
        ]
        return json.dumps({
            "entity": entity_name,
            "edges": edges,
            "count": len(edges),
        }, default=str)
    except Exception as e:
        logger.exception("get_entity error")
        return json.dumps({"error": str(e), "entity": entity_name, "edges": [], "count": 0})


@mcp.tool()
async def traverse_graph(
    entity_name: str,
    max_hops: int = 2,
    direction: str = "both",
) -> str:
    """Multi-hop graph traversal starting from a named entity.

    Walks the knowledge graph outward from an entity, following
    relationships up to max_hops deep. Useful for discovering indirect
    connections (e.g. "Juan -> Atlas -> Memory System -> Neo4j").

    Args:
        entity_name: Starting entity name
        max_hops: Maximum traversal depth (default 2, max 5)
        direction: 'outgoing', 'incoming', or 'both' (default 'both')
    """
    max_hops = min(max_hops, MAX_HOPS)
    if direction not in ("outgoing", "incoming", "both"):
        direction = "both"
    try:
        rag = _rag()
        paths = await rag.traverse_graph(
            entity_name, max_hops=max_hops, direction=direction,
        )
        return json.dumps({
            "entity": entity_name,
            "paths": paths,
            "count": len(paths),
            "max_hops": max_hops,
            "direction": direction,
        }, default=str)
    except Exception as e:
        logger.exception("traverse_graph error")
        return json.dumps({"error": str(e), "entity": entity_name, "paths": [], "count": 0})


@mcp.tool()
async def add_fact(fact: str, source: str = "mcp-client") -> str:
    """Store a single fact in the knowledge graph.

    Writes directly to the Graphiti /messages endpoint, bypassing the
    store_conversations guard (which blocks bulk real-time writes).
    Use this for deliberate, explicit fact storage.

    Args:
        fact: The fact to store (e.g. "Juan prefers morning meetings")
        source: Source description (default "mcp-client")
    """
    if not fact or not fact.strip():
        return json.dumps({"success": False, "error": "Fact cannot be empty"})
    try:
        rag = _rag()
        # Bypass add_fact() which checks store_conversations=False.
        # Post directly to the Graphiti wrapper /messages endpoint.
        from ..config import settings
        from datetime import datetime, timezone

        client = await rag._get_client()
        payload = {
            "group_id": settings.memory.group_id,
            "messages": [
                {
                    "content": fact.strip(),
                    "role_type": "system",
                    "role": None,
                    "source_description": source,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
            ],
        }
        resp = await client.post("/messages", json=payload)
        if resp.status_code in (200, 202):
            data = resp.json()
            return json.dumps({
                "success": data.get("success", True),
                "fact": fact.strip(),
                "source": source,
            })
        else:
            return json.dumps({
                "success": False,
                "error": "Graphiti returned %d: %s" % (resp.status_code, resp.text[:200]),
            })
    except Exception as e:
        logger.exception("add_fact error")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def search_memory_enhanced(
    query: str,
    max_results: int = 10,
    threshold: float = 0.3,
    enable_expansion: bool = True,
    enable_reranking: bool = True,
    enable_decomposition: bool = True,
) -> str:
    """Enhanced knowledge graph search with full query processing pipeline.

    Runs: classify -> decompose -> expand -> search -> rerank -> deduplicate.
    Much smarter than basic search_memory -- handles complex multi-part
    questions, expands abbreviations, and cross-encoder reranks results.

    Args:
        query: Natural language search query
        max_results: Maximum facts to return (default 10, max 50)
        threshold: Minimum relevance score 0.0-1.0 (default 0.3)
        enable_expansion: Expand query with synonyms/variants (default True)
        enable_reranking: Cross-encoder reranking (default True)
        enable_decomposition: Decompose complex queries (default True)
    """
    max_results = min(max_results, MAX_RESULTS)
    threshold = max(0.0, min(1.0, threshold))
    try:
        from ..config import settings
        rag = _rag()
        client = await rag._get_client()

        gid = settings.memory.group_id or ""
        payload = {
            "query": query,
            "group_ids": gid,
            "num_results": max_results,
            "threshold": threshold,
            "enable_expansion": enable_expansion,
            "enable_reranking": enable_reranking,
            "reranker_type": "cross-encoder",
            "enable_deduplication": True,
            "enable_decomposition": enable_decomposition,
            "enable_fallback": True,
        }
        resp = await client.post("/search/enhanced", json=payload)

        if resp.status_code == 404:
            logger.info("Enhanced search not available, falling back to basic search")
            return await search_memory(query, max_results=max_results)

        if resp.status_code != 200:
            return json.dumps({
                "error": "Graphiti returned %d" % resp.status_code,
                "edges": [], "count": 0,
            })

        data = resp.json()
        edges = data.get("edges", [])
        metadata = data.get("metadata", {})

        return json.dumps({
            "edges": edges,
            "count": len(edges),
            "metadata": {
                "query_time_ms": metadata.get("query_time_ms"),
                "search_queries": metadata.get("search_queries"),
                "decomposition": metadata.get("decomposition"),
                "expansion": metadata.get("expansion"),
                "reranker_type": metadata.get("reranker_type"),
                "raw_results_count": metadata.get("raw_results_count"),
            },
        }, default=str)
    except Exception as e:
        logger.exception("search_memory_enhanced error")
        return json.dumps({"error": str(e), "edges": [], "count": 0})


@mcp.tool()
async def search_memory_temporal(
    query: str,
    max_results: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    """Temporal-aware knowledge graph search with date filtering.

    Searches with automatic temporal intent detection (e.g. "last week",
    "in January"). Returns source/target entity nodes for each edge.
    Use this when the query involves time or you need to filter by date range.

    Args:
        query: Natural language search query
        max_results: Maximum facts to return (default 10, max 50)
        date_from: Optional ISO 8601 start date filter (e.g. "2026-01-01")
        date_to: Optional ISO 8601 end date filter (e.g. "2026-03-18")
    """
    max_results = min(max_results, MAX_RESULTS)
    try:
        from ..config import settings
        rag = _rag()
        client = await rag._get_client()

        gid = settings.memory.group_id or ""
        params = {
            "query": query,
            "group_ids": gid,
            "num_results": max_results,
        }
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        resp = await client.get("/search", params=params)

        if resp.status_code == 404:
            logger.info("Temporal search not available, falling back to basic search")
            return await search_memory(query, max_results=max_results)

        if resp.status_code != 200:
            return json.dumps({
                "error": "Graphiti returned %d" % resp.status_code,
                "facts": [], "count": 0,
            })

        data = resp.json()
        edges = data.get("edges", [])
        return json.dumps({
            "edges": edges,
            "count": len(edges),
            "query": query,
            "date_from": date_from,
            "date_to": date_to,
        }, default=str)
    except Exception as e:
        logger.exception("search_memory_temporal error")
        return json.dumps({"error": str(e), "edges": [], "count": 0})


@mcp.tool()
async def find_shortest_path(
    start_entity: str,
    end_entity: str,
    max_hops: int = 5,
) -> str:
    """Find the shortest path between two entities in the knowledge graph.

    Uses Neo4j's shortestPath algorithm to discover how two entities
    are connected. Returns the full path with intermediate entities
    and relationships.

    Args:
        start_entity: Starting entity name (e.g. "Juan")
        end_entity: Target entity name (e.g. "Atlas")
        max_hops: Maximum path length (default 5, max 10)
    """
    max_hops = min(max_hops, MAX_SHORTEST_PATH_HOPS)
    try:
        from ..config import settings
        rag = _rag()
        client = await rag._get_client()

        gid = settings.memory.group_id or ""
        payload = {
            "start_entity": start_entity,
            "end_entity": end_entity,
            "group_id": gid,
            "max_hops": max_hops,
        }
        resp = await client.post("/shortest-path", json=payload)

        if resp.status_code == 404:
            return json.dumps({
                "error": "Shortest path endpoint not available on this Graphiti instance",
                "found": False,
            })

        if resp.status_code != 200:
            return json.dumps({
                "error": "Graphiti returned %d" % resp.status_code,
                "found": False,
            })

        data = resp.json()
        return json.dumps({
            "found": data.get("found", False),
            "path": data.get("path"),
            "start_entity": start_entity,
            "end_entity": end_entity,
        }, default=str)
    except Exception as e:
        logger.exception("find_shortest_path error")
        return json.dumps({"error": str(e), "found": False})


@mcp.tool()
async def add_episode(
    name: str,
    episode_body: str,
    source_description: str = "mcp-client",
    reference_time: Optional[str] = None,
    is_historical: bool = False,
    data_source_type: str = "current",
) -> str:
    """Add a structured episode to the knowledge graph.

    Episodes are richer than facts -- they include metadata like timestamps,
    source descriptions, and historical flags. The graph extracts entities
    and relationships from the episode body automatically.

    Args:
        name: Short name for the episode (e.g. "Meeting with client")
        episode_body: Full text content of the episode
        source_description: Where this came from (default "mcp-client")
        reference_time: ISO 8601 timestamp (default now)
        is_historical: True if this describes a past event (default False)
        data_source_type: One of "current", "historical", "training", "archived"
    """
    if not episode_body or not episode_body.strip():
        return json.dumps({"success": False, "error": "Episode body cannot be empty"})
    if data_source_type not in ("current", "historical", "training", "archived"):
        data_source_type = "current"
    try:
        from ..config import settings
        from datetime import datetime, timezone

        rag = _rag()
        client = await rag._get_client()

        gid = settings.memory.group_id or ""
        ref_time = reference_time or (datetime.now(timezone.utc).isoformat() + "Z")
        payload = {
            "name": name,
            "episode_body": episode_body.strip(),
            "source_description": source_description,
            "reference_time": ref_time,
            "group_id": gid,
            "is_historical": is_historical,
            "data_source_type": data_source_type,
        }
        resp = await client.post("/episodes", json=payload, timeout=300.0)

        if resp.status_code not in (200, 201):
            return json.dumps({
                "success": False,
                "error": "Graphiti returned %d: %s" % (resp.status_code, resp.text[:200]),
            })

        data = resp.json()
        return json.dumps({
            "success": True,
            "episode_id": data.get("episode_id", ""),
            "entities_created": data.get("entities_created", 0),
            "relations_created": data.get("relations_created", 0),
        }, default=str)
    except Exception as e:
        logger.exception("add_episode error")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def delete_episode(episode_id: str) -> str:
    """Delete an episode from the knowledge graph.

    Removes the episodic node and its associations. Use with caution --
    this permanently removes knowledge from the graph.

    Args:
        episode_id: UUID of the episode to delete
    """
    if not episode_id or not episode_id.strip():
        return json.dumps({"success": False, "error": "Episode ID cannot be empty"})
    try:
        rag = _rag()
        client = await rag._get_client()

        resp = await client.delete("/episodes/%s" % episode_id.strip())

        if resp.status_code == 404:
            return json.dumps({"success": False, "error": "Episode not found"})
        if resp.status_code != 200:
            return json.dumps({
                "success": False,
                "error": "Graphiti returned %d: %s" % (resp.status_code, resp.text[:200]),
            })

        return json.dumps({"success": True, "episode_id": episode_id.strip()})
    except Exception as e:
        logger.exception("delete_episode error")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool()
async def enhance_prompt(
    query: str,
    max_sources: int = 5,
    min_confidence: float = 0.3,
    max_context_length: int = 4000,
) -> str:
    """Enhance a prompt with knowledge graph context.

    Searches the graph for relevant facts and returns a context-enriched
    prompt ready for LLM consumption. Useful for building RAG pipelines
    where you want the graph to inform the response.

    Args:
        query: The user query or prompt to enhance
        max_sources: Maximum context sources to include (default 5, max 20)
        min_confidence: Minimum confidence threshold 0.0-1.0 (default 0.3)
        max_context_length: Maximum context chars to inject (default 4000)
    """
    max_sources = min(max_sources, MAX_ENHANCE_SOURCES)
    min_confidence = max(0.0, min(1.0, min_confidence))
    try:
        from ..config import settings
        rag = _rag()
        client = await rag._get_client()

        gid = settings.memory.group_id or ""
        payload = {
            "query": query,
            "group_ids": gid,
            "min_confidence": min_confidence,
            "max_sources": max_sources,
            "max_context_length": max_context_length,
            "compress_context": False,
            "include_metadata": True,
        }
        resp = await client.post("/enhance", json=payload)

        if resp.status_code == 404:
            return json.dumps({
                "error": "Enhance endpoint not available on this Graphiti instance",
                "context_used": False,
            })
        if resp.status_code != 200:
            return json.dumps({
                "error": "Graphiti returned %d" % resp.status_code,
                "context_used": False,
            })

        data = resp.json()
        return json.dumps({
            "prompt": data.get("prompt", query),
            "context_used": data.get("context_used", False),
            "sources": data.get("sources", []),
            "metadata": data.get("metadata", {}),
        }, default=str)
    except Exception as e:
        logger.exception("enhance_prompt error")
        return json.dumps({"error": str(e), "context_used": False})


@mcp.tool()
async def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text using semantic similarity.

    Uses sentence embeddings with sentiment anchor phrases to determine
    positive/neutral/negative sentiment. Returns a score (-1 to +1)
    and confidence level.

    Args:
        text: Text to analyze for sentiment
    """
    if not text or not text.strip():
        return json.dumps({"error": "Text cannot be empty"})
    try:
        rag = _rag()
        client = await rag._get_client()

        resp = await client.post(
            "/analyze/sentiment",
            json={"text": text.strip()},
        )

        if resp.status_code == 404:
            return json.dumps({"error": "Sentiment endpoint not available on this Graphiti instance"})
        if resp.status_code != 200:
            return json.dumps({"error": "Graphiti returned %d" % resp.status_code})

        data = resp.json()
        return json.dumps({
            "text": data.get("text", text.strip()),
            "sentiment": data.get("sentiment", "unknown"),
            "score": data.get("score", 0.0),
            "confidence": data.get("confidence", 0.0),
            "method": data.get("method", "semantic_similarity"),
        }, default=str)
    except Exception as e:
        logger.exception("analyze_sentiment error")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Conversation tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_conversations(
    query: str,
    limit: int = 20,
    days: int = 30,
    role: Optional[str] = None,
) -> str:
    """Search conversation history by keyword (ILIKE).

    Searches the conversation_turns table for content matching the query.
    Time-bounded by the 'days' parameter to keep scans efficient.

    Args:
        query: Search keyword or phrase
        limit: Maximum results (default 20, max 200)
        days: Look back N days (default 30, max 365)
        role: Filter by role ('user' or 'assistant'), or None for all
    """
    limit = min(limit, MAX_CONVERSATION_LIMIT)
    days = min(days, MAX_DAYS)
    if role and role not in ("user", "assistant"):
        role = None
    try:
        pool = _db()
        search_pattern = "%" + query.replace("%", "\\%").replace("_", "\\_") + "%"

        if role:
            rows = await pool.fetch(
                """
                SELECT id, session_id, role, content, speaker_id, intent,
                       turn_type, created_at
                FROM conversation_turns
                WHERE content ILIKE $1
                  AND created_at > NOW() - make_interval(days => $2)
                  AND role = $3
                ORDER BY created_at DESC
                LIMIT $4
                """,
                search_pattern,
                days,
                role,
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT id, session_id, role, content, speaker_id, intent,
                       turn_type, created_at
                FROM conversation_turns
                WHERE content ILIKE $1
                  AND created_at > NOW() - make_interval(days => $2)
                ORDER BY created_at DESC
                LIMIT $3
                """,
                search_pattern,
                days,
                limit,
            )

        turns = [
            {
                "id": str(row["id"]),
                "session_id": str(row["session_id"]),
                "role": row["role"],
                "content": row["content"],
                "speaker_id": row["speaker_id"],
                "intent": row["intent"],
                "turn_type": row["turn_type"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return json.dumps({"turns": turns, "count": len(turns), "query": query}, default=str)
    except Exception as e:
        logger.exception("search_conversations error")
        return json.dumps({"error": str(e), "turns": [], "count": 0})


@mcp.tool()
async def get_session_history(session_id: str, limit: int = 50) -> str:
    """Get the full conversation transcript for a session.

    Args:
        session_id: Session UUID
        limit: Maximum turns to return (default 50, max 200)
    """
    limit = min(limit, MAX_CONVERSATION_LIMIT)
    try:
        sid = UUID(session_id)
    except (ValueError, AttributeError):
        return json.dumps({"error": "Invalid session_id UUID", "turns": [], "count": 0})

    try:
        repo = _conversation_repo()
        turns = await repo.get_history(sid, limit=limit)
        result = [
            {
                "id": str(t.id),
                "role": t.role,
                "content": t.content,
                "speaker_id": t.speaker_id,
                "intent": t.intent,
                "turn_type": t.turn_type,
                "created_at": t.created_at,
            }
            for t in turns
        ]
        return json.dumps({
            "session_id": session_id,
            "turns": result,
            "count": len(result),
        }, default=str)
    except Exception as e:
        logger.exception("get_session_history error")
        return json.dumps({"error": str(e), "turns": [], "count": 0})


@mcp.tool()
async def list_sessions(limit: int = 20, active_only: bool = False) -> str:
    """List recent sessions.

    Args:
        limit: Maximum sessions to return (default 20, max 100)
        active_only: If True, only return currently active sessions
    """
    limit = min(limit, MAX_SESSION_LIMIT)
    try:
        pool = _db()
        if active_only:
            rows = await pool.fetch(
                """
                SELECT id, user_id, terminal_id, started_at, last_activity_at,
                       is_active, session_date
                FROM sessions
                WHERE is_active = true
                ORDER BY last_activity_at DESC
                LIMIT $1
                """,
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT id, user_id, terminal_id, started_at, last_activity_at,
                       is_active, session_date
                FROM sessions
                ORDER BY last_activity_at DESC
                LIMIT $1
                """,
                limit,
            )

        sessions = [
            {
                "id": str(row["id"]),
                "user_id": str(row["user_id"]) if row["user_id"] else None,
                "terminal_id": row["terminal_id"],
                "started_at": row["started_at"],
                "last_activity_at": row["last_activity_at"],
                "is_active": row["is_active"],
                "session_date": row["session_date"],
            }
            for row in rows
        ]
        return json.dumps({
            "sessions": sessions,
            "count": len(sessions),
            "active_only": active_only,
        }, default=str)
    except Exception as e:
        logger.exception("list_sessions error")
        return json.dumps({"error": str(e), "sessions": [], "count": 0})


# ---------------------------------------------------------------------------
# Combined tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_context(
    query: str,
    entity_name: Optional[str] = None,
    max_facts: int = 5,
) -> str:
    """Get unified context from both graph and conversation history.

    Runs graph search (and optionally entity traversal) in parallel with
    a conversation keyword search, then merges the results.

    Args:
        query: Search query for both graph and conversation search
        entity_name: Optional entity name for graph traversal
        max_facts: Maximum graph facts (default 5, max 50)
    """
    max_facts = min(max_facts, MAX_RESULTS)

    async def _graph_search():
        try:
            rag = _rag()
            result = await rag.search_with_traversal(
                query, entity_name=entity_name, max_facts=max_facts,
            )
            return [
                {
                    "uuid": f.uuid,
                    "name": f.name,
                    "fact": f.fact,
                    "confidence": f.confidence,
                    "source_type": f.source_type,
                }
                for f in result.facts
            ]
        except Exception as e:
            logger.warning("get_context graph search failed: %s", e)
            return []

    async def _conv_search():
        try:
            pool = _db()
            pattern = "%" + query.replace("%", "\\%").replace("_", "\\_") + "%"
            rows = await pool.fetch(
                """
                SELECT id, session_id, role, content, created_at
                FROM conversation_turns
                WHERE content ILIKE $1
                  AND created_at > NOW() - INTERVAL '30 days'
                ORDER BY created_at DESC
                LIMIT 10
                """,
                pattern,
            )
            return [
                {
                    "id": str(row["id"]),
                    "session_id": str(row["session_id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": str(row["created_at"]),
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning("get_context conversation search failed: %s", e)
            return []

    graph_facts, conv_turns = await asyncio.gather(
        _graph_search(), _conv_search(),
    )

    return json.dumps({
        "query": query,
        "entity_name": entity_name,
        "graph_facts": graph_facts,
        "graph_count": len(graph_facts),
        "conversation_turns": conv_turns,
        "conversation_count": len(conv_turns),
    }, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings
        from .auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.memory_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.memory_port)
    else:
        mcp.run(transport="stdio")
