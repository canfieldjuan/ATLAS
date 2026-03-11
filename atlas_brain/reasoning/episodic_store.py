"""Episodic memory store for reasoning traces (Neo4j).

Stores full reasoning traces with evidence nodes and conclusions in Neo4j,
using labels distinct from Graphiti (ReasoningTrace, EvidenceNode,
ConclusionNode).  Supports native vector index search on trace embeddings
(mxbai-embed-large-v1, 1024-dim).

All nodes use group_id="b2b-reasoning" for namespace isolation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncGraphDatabase

logger = logging.getLogger("atlas.reasoning.episodic_store")

GROUP_ID = "b2b-reasoning"
EMBEDDING_DIM = 1024
SIMILARITY_THRESHOLD = 0.75


@dataclass
class EvidenceNode:
    """A piece of evidence supporting a reasoning trace."""

    id: str = ""
    type: str = ""          # "review", "snapshot_delta", "displacement", "event"
    source: str = ""        # e.g. "g2", "capterra", "b2b_vendor_snapshots"
    value: str = ""         # human-readable summary
    review_id: str | None = None
    event_id: str | None = None
    timestamp: datetime | None = None


@dataclass
class ConclusionNode:
    """A conclusion drawn from a reasoning trace."""

    id: str = ""
    claim: str = ""
    confidence: float = 0.0
    evidence_chain: list[str] = field(default_factory=list)  # evidence node IDs


@dataclass
class ReasoningTrace:
    """A full reasoning trace with evidence and conclusions."""

    id: str = ""
    vendor_name: str = ""
    category: str = ""
    created_at: datetime | None = None
    conclusion_type: str = ""       # archetype name
    confidence: float = 0.0
    pattern_sig: str = ""
    trace_embedding: list[float] = field(default_factory=list)
    evidence: list[EvidenceNode] = field(default_factory=list)
    conclusions: list[ConclusionNode] = field(default_factory=list)
    similarity_score: float | None = None  # set on similarity search results


class EpisodicStore:
    """Neo4j-backed episodic memory for reasoning traces."""

    def __init__(
        self,
        bolt_url: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "password123"),
    ):
        self._bolt_url = bolt_url
        self._auth = auth
        self._driver = None
        self._embedder = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_driver(self):
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._bolt_url, auth=self._auth,
            )
        return self._driver

    async def close(self):
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    def _get_embedder(self):
        """Lazy-load mxbai-embed-large-v1 sentence-transformer."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
            logger.info("Loaded mxbai-embed-large-v1 embedding model for episodic store")
        return self._embedder

    def embed_text(self, text: str) -> list[float]:
        """Generate 1024-dim embedding for a text string."""
        model = self._get_embedder()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create vector index + range indexes if they don't exist."""
        driver = await self._get_driver()
        async with driver.session() as session:
            # Vector index for trace similarity
            await session.run(
                """
                CREATE VECTOR INDEX reasoning_trace_embedding IF NOT EXISTS
                FOR (n:ReasoningTrace) ON (n.trace_embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                dim=EMBEDDING_DIM,
            )
            # Range indexes for common lookups
            await session.run(
                "CREATE INDEX reasoning_trace_vendor IF NOT EXISTS "
                "FOR (n:ReasoningTrace) ON (n.vendor_name)"
            )
            await session.run(
                "CREATE INDEX reasoning_trace_pattern IF NOT EXISTS "
                "FOR (n:ReasoningTrace) ON (n.pattern_sig)"
            )
            await session.run(
                "CREATE INDEX reasoning_trace_created IF NOT EXISTS "
                "FOR (n:ReasoningTrace) ON (n.created_at)"
            )
            await session.run(
                "CREATE INDEX evidence_node_id IF NOT EXISTS "
                "FOR (n:EvidenceNode) ON (n.id)"
            )
            await session.run(
                "CREATE INDEX conclusion_node_id IF NOT EXISTS "
                "FOR (n:ConclusionNode) ON (n.id)"
            )
        logger.info("Episodic store indexes ensured")

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store_trace(self, trace: ReasoningTrace) -> str:
        """Store a full reasoning trace. Returns the trace ID."""
        if not trace.id:
            trace.id = str(uuid.uuid4())
        if trace.created_at is None:
            trace.created_at = datetime.now(timezone.utc)

        driver = await self._get_driver()
        async with driver.session() as session:
            # Create the ReasoningTrace node
            await session.run(
                """
                CREATE (t:ReasoningTrace {
                    id: $id,
                    vendor_name: $vendor_name,
                    category: $category,
                    created_at: datetime($created_at),
                    conclusion_type: $conclusion_type,
                    confidence: $confidence,
                    pattern_sig: $pattern_sig,
                    trace_embedding: $trace_embedding,
                    group_id: $group_id
                })
                """,
                id=trace.id,
                vendor_name=trace.vendor_name,
                category=trace.category,
                created_at=trace.created_at.isoformat(),
                conclusion_type=trace.conclusion_type,
                confidence=trace.confidence,
                pattern_sig=trace.pattern_sig,
                trace_embedding=trace.trace_embedding,
                group_id=GROUP_ID,
            )

            # Create evidence nodes + SUPPORTED_BY relationships
            for ev in trace.evidence:
                if not ev.id:
                    ev.id = str(uuid.uuid4())
                await session.run(
                    """
                    MATCH (t:ReasoningTrace {id: $trace_id, group_id: $group_id})
                    CREATE (e:EvidenceNode {
                        id: $id,
                        type: $type,
                        source: $source,
                        value: $value,
                        review_id: $review_id,
                        event_id: $event_id,
                        timestamp: $timestamp,
                        group_id: $group_id
                    })
                    CREATE (t)-[:SUPPORTED_BY {group_id: $group_id}]->(e)
                    """,
                    trace_id=trace.id,
                    id=ev.id,
                    type=ev.type,
                    source=ev.source,
                    value=ev.value,
                    review_id=ev.review_id,
                    event_id=ev.event_id,
                    timestamp=ev.timestamp.isoformat() if ev.timestamp else None,
                    group_id=GROUP_ID,
                )

            # Create conclusion nodes + CONCLUDED relationships
            for conc in trace.conclusions:
                if not conc.id:
                    conc.id = str(uuid.uuid4())
                await session.run(
                    """
                    MATCH (t:ReasoningTrace {id: $trace_id, group_id: $group_id})
                    CREATE (c:ConclusionNode {
                        id: $id,
                        claim: $claim,
                        confidence: $confidence,
                        evidence_chain: $evidence_chain,
                        group_id: $group_id
                    })
                    CREATE (t)-[:CONCLUDED {group_id: $group_id}]->(c)
                    """,
                    trace_id=trace.id,
                    id=conc.id,
                    claim=conc.claim,
                    confidence=conc.confidence,
                    evidence_chain=conc.evidence_chain,
                    group_id=GROUP_ID,
                )

        logger.info("Stored reasoning trace %s for %s", trace.id, trace.vendor_name)
        return trace.id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def find_similar(
        self,
        embedding: list[float],
        threshold: float = SIMILARITY_THRESHOLD,
        limit: int = 5,
    ) -> list[ReasoningTrace]:
        """Vector similarity search on trace embeddings."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                CALL db.index.vector.queryNodes(
                    'reasoning_trace_embedding', $limit, $embedding
                ) YIELD node, score
                WHERE score >= $threshold AND node.group_id = $group_id
                RETURN node, score
                ORDER BY score DESC
                """,
                embedding=embedding,
                limit=limit,
                threshold=threshold,
                group_id=GROUP_ID,
            )
            traces = []
            async for record in result:
                node = record["node"]
                trace = ReasoningTrace(
                    id=node["id"],
                    vendor_name=node["vendor_name"],
                    category=node.get("category", ""),
                    conclusion_type=node.get("conclusion_type", ""),
                    confidence=node.get("confidence", 0.0),
                    pattern_sig=node.get("pattern_sig", ""),
                    similarity_score=record["score"],
                )
                traces.append(trace)
            return traces

    async def get_trace(self, trace_id: str) -> ReasoningTrace | None:
        """Load a full trace with evidence and conclusions."""
        driver = await self._get_driver()
        async with driver.session() as session:
            # Fetch trace node
            result = await session.run(
                """
                MATCH (t:ReasoningTrace {id: $id, group_id: $group_id})
                RETURN t
                """,
                id=trace_id,
                group_id=GROUP_ID,
            )
            record = await result.single()
            if record is None:
                return None

            node = record["t"]
            trace = ReasoningTrace(
                id=node["id"],
                vendor_name=node["vendor_name"],
                category=node.get("category", ""),
                conclusion_type=node.get("conclusion_type", ""),
                confidence=node.get("confidence", 0.0),
                pattern_sig=node.get("pattern_sig", ""),
            )

            # Fetch evidence
            ev_result = await session.run(
                """
                MATCH (t:ReasoningTrace {id: $id, group_id: $group_id})
                      -[:SUPPORTED_BY]->(e:EvidenceNode)
                RETURN e
                """,
                id=trace_id,
                group_id=GROUP_ID,
            )
            async for rec in ev_result:
                e = rec["e"]
                trace.evidence.append(EvidenceNode(
                    id=e["id"],
                    type=e.get("type", ""),
                    source=e.get("source", ""),
                    value=e.get("value", ""),
                    review_id=e.get("review_id"),
                    event_id=e.get("event_id"),
                ))

            # Fetch conclusions
            conc_result = await session.run(
                """
                MATCH (t:ReasoningTrace {id: $id, group_id: $group_id})
                      -[:CONCLUDED]->(c:ConclusionNode)
                RETURN c
                """,
                id=trace_id,
                group_id=GROUP_ID,
            )
            async for rec in conc_result:
                c = rec["c"]
                trace.conclusions.append(ConclusionNode(
                    id=c["id"],
                    claim=c.get("claim", ""),
                    confidence=c.get("confidence", 0.0),
                    evidence_chain=list(c.get("evidence_chain", [])),
                ))

            return trace

    async def get_traces_for_vendor(self, vendor_name: str, limit: int = 10) -> list[ReasoningTrace]:
        """All traces for a vendor, newest first."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (t:ReasoningTrace {vendor_name: $vendor, group_id: $group_id})
                RETURN t
                ORDER BY t.created_at DESC
                LIMIT $limit
                """,
                vendor=vendor_name,
                group_id=GROUP_ID,
                limit=limit,
            )
            traces = []
            async for record in result:
                node = record["t"]
                traces.append(ReasoningTrace(
                    id=node["id"],
                    vendor_name=node["vendor_name"],
                    category=node.get("category", ""),
                    conclusion_type=node.get("conclusion_type", ""),
                    confidence=node.get("confidence", 0.0),
                    pattern_sig=node.get("pattern_sig", ""),
                ))
            return traces

    # ------------------------------------------------------------------
    # Cleanup (for tests)
    # ------------------------------------------------------------------

    async def delete_by_group(self, group_id: str = GROUP_ID) -> int:
        """Delete all nodes and relationships for a group. Returns count."""
        driver = await self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (n {group_id: $group_id})
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                group_id=group_id,
            )
            record = await result.single()
            return record["deleted"] if record else 0
