"""B2B Knowledge Graph (WS3).

Entity-relationship model in Neo4j enabling multi-hop reasoning across
vendors, competitors, integrations, pain points, and companies.

Node types (all prefixed B2b to avoid Graphiti collisions):
    B2bVendor     -- canonical vendor entity
    B2bProduct    -- product profile (one per vendor)
    B2bCompany    -- company showing buying signals
    B2bPainPoint  -- pain category per vendor
    B2bIntegration -- integration/tool dependency

Relationship types:
    COMPETES_WITH    -- bidirectional competitive relationship
    SWITCHED_TO      -- directed displacement (from_vendor -> to_vendor)
    INTEGRATES_WITH  -- vendor uses/connects to another tool
    HAS_PAIN         -- vendor exhibits a pain point category
    USES             -- company uses a vendor
    CONSIDERING      -- company evaluating alternatives

SQL -> Graph sync reads from existing b2b_* Postgres tables and upserts
into Neo4j. Designed for nightly batch sync (idempotent via MERGE).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atlas.reasoning.knowledge_graph")

GROUP_ID = "b2b-knowledge-graph"


# ------------------------------------------------------------------
# Entity definitions (WS3A)
# ------------------------------------------------------------------


@dataclass
class VendorNode:
    """Canonical vendor entity."""

    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    product_category: str = ""
    total_reviews: int = 0
    churn_density: float = 0.0
    avg_urgency: float = 0.0
    positive_review_pct: float = 0.0
    recommend_ratio: float = 0.0
    pain_count: int = 0
    competitor_count: int = 0
    confidence_score: float = 0.0


@dataclass
class ProductNode:
    """Product profile for a vendor."""

    vendor_name: str
    product_category: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    avg_rating: float = 0.0
    total_reviews: int = 0
    typical_company_size: str = ""
    profile_summary: str = ""


@dataclass
class CompanyNode:
    """Company showing buying signals."""

    company_name: str
    vendor_name: str = ""
    urgency_score: float = 0.0
    pain_category: str = ""
    buying_stage: str = ""
    decision_maker: bool = False
    seat_count: int = 0


@dataclass
class PainPointNode:
    """Pain category for a vendor."""

    vendor_name: str
    pain_category: str
    mention_count: int = 0
    avg_urgency: float = 0.0
    confidence_score: float = 0.0


@dataclass
class DisplacementEdge:
    """Directed switching: from_vendor -> to_vendor."""

    from_vendor: str
    to_vendor: str
    mention_count: int = 0
    primary_driver: str = ""  # pricing, features, reliability, support, other
    signal_strength: str = ""  # strong, moderate, weak
    confidence_score: float = 0.0


@dataclass
class IntegrationEdge:
    """Vendor integrates with a tool/platform."""

    vendor_name: str
    integration_name: str
    mention_count: int = 0
    confidence_score: float = 0.0


# ------------------------------------------------------------------
# Graph sync engine (WS3C)
# ------------------------------------------------------------------


class KnowledgeGraphSync:
    """Syncs b2b_* Postgres tables into Neo4j knowledge graph."""

    def __init__(self, pg_pool: Any, neo4j_driver: Any):
        self._pg = pg_pool
        self._driver = neo4j_driver

    async def full_sync(self) -> dict[str, int]:
        """Run a full idempotent sync. Returns counts per entity type."""
        counts = {}
        counts["vendors"] = await self._sync_vendors()
        counts["products"] = await self._sync_products()
        counts["pain_points"] = await self._sync_pain_points()
        counts["displacement_edges"] = await self._sync_displacement_edges()
        counts["integrations"] = await self._sync_integrations()
        counts["companies"] = await self._sync_companies()
        counts["competition_edges"] = await self._derive_competition_edges()

        total = sum(counts.values())
        logger.info("Knowledge graph sync complete: %d entities/edges synced", total)
        for kind, n in counts.items():
            logger.info("  %s: %d", kind, n)
        return counts

    async def ensure_indexes(self) -> None:
        """Create Neo4j indexes for the knowledge graph labels."""
        index_queries = [
            # Uniqueness constraints
            "CREATE CONSTRAINT b2b_vendor_name IF NOT EXISTS "
            "FOR (n:B2bVendor) REQUIRE n.canonical_name IS UNIQUE",
            "CREATE CONSTRAINT b2b_pain_key IF NOT EXISTS "
            "FOR (n:B2bPainPoint) REQUIRE (n.vendor_name, n.pain_category) IS UNIQUE",
            # Range indexes
            "CREATE INDEX b2b_vendor_category IF NOT EXISTS "
            "FOR (n:B2bVendor) ON (n.product_category)",
            "CREATE INDEX b2b_product_vendor IF NOT EXISTS "
            "FOR (n:B2bProduct) ON (n.vendor_name)",
            "CREATE INDEX b2b_company_name IF NOT EXISTS "
            "FOR (n:B2bCompany) ON (n.company_name)",
            "CREATE INDEX b2b_integration_name IF NOT EXISTS "
            "FOR (n:B2bIntegration) ON (n.integration_name)",
            "CREATE INDEX b2b_pain_vendor IF NOT EXISTS "
            "FOR (n:B2bPainPoint) ON (n.vendor_name)",
        ]
        async with self._driver.session() as session:
            for q in index_queries:
                try:
                    await session.run(q)
                except Exception:
                    logger.debug("Index creation skipped (may already exist): %s", q[:60])

    # ------------------------------------------------------------------
    # Sync methods
    # ------------------------------------------------------------------

    async def _sync_vendors(self) -> int:
        """Sync canonical vendor scorecard rows into B2bVendor nodes."""
        from ..autonomous.tasks._b2b_shared import read_vendor_graph_sync_rows

        rows = await read_vendor_graph_sync_rows(self._pg)

        if not rows:
            return 0

        async with self._driver.session() as session:
            for row in rows:
                aliases = row["aliases"] or []
                if isinstance(aliases, str):
                    import json
                    try:
                        aliases = json.loads(aliases)
                    except (ValueError, TypeError):
                        aliases = []

                await session.run(
                    """
                    MERGE (v:B2bVendor {canonical_name: $name})
                    SET v.group_id = $gid,
                        v.aliases = $aliases,
                        v.product_category = $category,
                        v.total_reviews = $reviews,
                        v.churn_density = $churn,
                        v.avg_urgency = $urgency,
                        v.positive_review_pct = $pos_pct,
                        v.recommend_ratio = $rec_ratio,
                        v.pain_count = $pain_cnt,
                        v.competitor_count = $comp_cnt,
                        v.confidence_score = $conf,
                        v.updated_at = datetime()
                    """,
                    name=row["canonical_name"],
                    gid=GROUP_ID,
                    aliases=aliases,
                    category=row["product_category"] or "",
                    reviews=row["total_reviews"] or 0,
                    churn=float(row["churn_density"] or 0),
                    urgency=float(row["avg_urgency"] or 0),
                    pos_pct=float(row["positive_review_pct"] or 0),
                    rec_ratio=float(row["recommend_ratio"] or 0),
                    pain_cnt=row["pain_count"] or 0,
                    comp_cnt=row["competitor_count"] or 0,
                    conf=float(row["confidence_score"] or 0),
                )

        return len(rows)

    async def _sync_products(self) -> int:
        """Sync b2b_product_profiles -> B2bProduct nodes + HAS_PRODUCT edges."""
        rows = await self._pg.fetch("""
            SELECT vendor_name, product_category, strengths, weaknesses,
                   avg_rating, total_reviews_analyzed, typical_company_size,
                   profile_summary
            FROM b2b_product_profiles
        """)

        if not rows:
            return 0

        async with self._driver.session() as session:
            for row in rows:
                strengths = row["strengths"] or []
                weaknesses = row["weaknesses"] or []
                if isinstance(strengths, str):
                    import json
                    strengths = json.loads(strengths) if strengths else []
                if isinstance(weaknesses, str):
                    import json
                    weaknesses = json.loads(weaknesses) if weaknesses else []

                # Flatten to string lists for Neo4j
                str_strengths = [str(s)[:200] if isinstance(s, dict) else str(s)[:200] for s in strengths[:10]]
                str_weaknesses = [str(w)[:200] if isinstance(w, dict) else str(w)[:200] for w in weaknesses[:10]]

                await session.run(
                    """
                    MERGE (p:B2bProduct {vendor_name: $vendor})
                    SET p.group_id = $gid,
                        p.product_category = $category,
                        p.strengths = $strengths,
                        p.weaknesses = $weaknesses,
                        p.avg_rating = $rating,
                        p.total_reviews = $reviews,
                        p.typical_company_size = $size,
                        p.profile_summary = $summary,
                        p.updated_at = datetime()
                    WITH p
                    MATCH (v:B2bVendor)
                    WHERE toLower(v.canonical_name) = toLower($vendor)
                    MERGE (v)-[:HAS_PRODUCT]->(p)
                    """,
                    vendor=row["vendor_name"],
                    gid=GROUP_ID,
                    category=row["product_category"] or "",
                    strengths=str_strengths,
                    weaknesses=str_weaknesses,
                    rating=float(row["avg_rating"] or 0),
                    reviews=row["total_reviews_analyzed"] or 0,
                    size=row["typical_company_size"] or "",
                    summary=(row["profile_summary"] or "")[:500],
                )

        return len(rows)

    async def _sync_pain_points(self) -> int:
        """Sync b2b_vendor_pain_points -> B2bPainPoint nodes + HAS_PAIN edges."""
        rows = await self._pg.fetch("""
            SELECT vendor_name, pain_category, mention_count,
                   avg_urgency, confidence_score
            FROM b2b_vendor_pain_points
            WHERE mention_count >= 2
        """)

        if not rows:
            return 0

        async with self._driver.session() as session:
            for row in rows:
                await session.run(
                    """
                    MERGE (pp:B2bPainPoint {vendor_name: $vendor, pain_category: $cat})
                    SET pp.group_id = $gid,
                        pp.mention_count = $mentions,
                        pp.avg_urgency = $urgency,
                        pp.confidence_score = $conf,
                        pp.updated_at = datetime()
                    WITH pp
                    MATCH (v:B2bVendor)
                    WHERE toLower(v.canonical_name) = toLower($vendor)
                    MERGE (v)-[:HAS_PAIN]->(pp)
                    """,
                    vendor=row["vendor_name"],
                    cat=row["pain_category"],
                    gid=GROUP_ID,
                    mentions=row["mention_count"],
                    urgency=float(row["avg_urgency"] or 0),
                    conf=float(row["confidence_score"] or 0),
                )

        return len(rows)

    async def _sync_displacement_edges(self) -> int:
        """Sync b2b_displacement_edges -> SWITCHED_TO relationships."""
        rows = await self._pg.fetch("""
            SELECT from_vendor, to_vendor, mention_count,
                   primary_driver, signal_strength, confidence_score
            FROM b2b_displacement_edges
            WHERE mention_count >= 1
        """)

        if not rows:
            return 0

        async with self._driver.session() as session:
            for row in rows:
                # Ensure both vendor nodes exist
                await session.run(
                    """
                    MERGE (f:B2bVendor {canonical_name: $from_v})
                    SET f.group_id = $gid
                    MERGE (t:B2bVendor {canonical_name: $to_v})
                    SET t.group_id = $gid
                    MERGE (f)-[r:SWITCHED_TO]->(t)
                    SET r.mention_count = $mentions,
                        r.primary_driver = $driver,
                        r.signal_strength = $strength,
                        r.confidence_score = $conf,
                        r.updated_at = datetime()
                    """,
                    from_v=row["from_vendor"],
                    to_v=row["to_vendor"],
                    gid=GROUP_ID,
                    mentions=row["mention_count"],
                    driver=row["primary_driver"] or "",
                    strength=row["signal_strength"] or "",
                    conf=float(row["confidence_score"] or 0),
                )

        return len(rows)

    async def _sync_integrations(self) -> int:
        """Sync b2b_vendor_integrations -> INTEGRATES_WITH relationships."""
        rows = await self._pg.fetch("""
            SELECT vendor_name, integration_name, mention_count, confidence_score
            FROM b2b_vendor_integrations
            WHERE mention_count >= 2
        """)

        if not rows:
            return 0

        async with self._driver.session() as session:
            for row in rows:
                await session.run(
                    """
                    MATCH (v:B2bVendor)
                    WHERE toLower(v.canonical_name) = toLower($vendor)
                    MERGE (i:B2bIntegration {integration_name: $int_name})
                    SET i.group_id = $gid, i.updated_at = datetime()
                    MERGE (v)-[r:INTEGRATES_WITH]->(i)
                    SET r.mention_count = $mentions,
                        r.confidence_score = $conf,
                        r.updated_at = datetime()
                    """,
                    vendor=row["vendor_name"],
                    int_name=row["integration_name"],
                    gid=GROUP_ID,
                    mentions=row["mention_count"],
                    conf=float(row["confidence_score"] or 0),
                )

        return len(rows)

    async def _sync_companies(self) -> int:
        """Sync b2b_company_signals -> B2bCompany nodes + USES/CONSIDERING edges."""
        rows = await self._pg.fetch("""
            SELECT company_name, vendor_name, urgency_score,
                   pain_category, buying_stage, decision_maker, seat_count
            FROM b2b_company_signals
        """)

        if not rows:
            return 0

        async with self._driver.session() as session:
            for row in rows:
                stage = row["buying_stage"] or "unknown"
                rel_type = "CONSIDERING" if stage in ("evaluation", "negotiation") else "USES"

                await session.run(
                    f"""
                    MERGE (c:B2bCompany {{company_name: $company}})
                    SET c.group_id = $gid,
                        c.urgency_score = $urgency,
                        c.pain_category = $pain,
                        c.buying_stage = $stage,
                        c.decision_maker = $dm,
                        c.seat_count = $seats,
                        c.updated_at = datetime()
                    WITH c
                    MATCH (v:B2bVendor)
                    WHERE toLower(v.canonical_name) = toLower($vendor)
                    MERGE (c)-[r:{rel_type}]->(v)
                    SET r.urgency_score = $urgency,
                        r.updated_at = datetime()
                    """,
                    company=row["company_name"],
                    vendor=row["vendor_name"],
                    gid=GROUP_ID,
                    urgency=float(row["urgency_score"] or 0),
                    pain=row["pain_category"] or "",
                    stage=stage,
                    dm=bool(row["decision_maker"]),
                    seats=row["seat_count"] or 0,
                )

        return len(rows)

    async def _derive_competition_edges(self) -> int:
        """Derive COMPETES_WITH from SWITCHED_TO (if A->B and B->A, they compete)."""
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (a:B2bVendor)-[r1:SWITCHED_TO]->(b:B2bVendor)
                WHERE a.group_id = $gid AND b.group_id = $gid
                WITH a, b,
                     r1.mention_count AS ab_mentions,
                     EXISTS { MATCH (b)-[:SWITCHED_TO]->(a) } AS bidirectional
                MERGE (a)-[c:COMPETES_WITH]->(b)
                SET c.mention_count = ab_mentions,
                    c.bidirectional = bidirectional,
                    c.updated_at = datetime()
                RETURN count(c) AS cnt
                """,
                gid=GROUP_ID,
            )
            record = await result.single()
            return record["cnt"] if record else 0


# ------------------------------------------------------------------
# Multi-hop query engine (WS3D)
# ------------------------------------------------------------------


class KnowledgeGraphQuery:
    """Multi-hop Cypher queries over the B2B knowledge graph."""

    def __init__(self, neo4j_driver: Any):
        self._driver = neo4j_driver

    async def vendor_competitive_landscape(self, vendor_name: str) -> dict[str, Any]:
        """Full competitive picture: who's winning, who's losing, why.

        Returns: {
            vendor, losing_to: [{name, mentions, driver}],
            winning_from: [{name, mentions, driver}],
            shared_integrations: [{vendor, integration, mentions}],
            shared_pains: [{vendor, pain, mentions}],
        }
        """
        async with self._driver.session() as session:
            # Vendors people are switching TO from this vendor
            losing_result = await session.run(
                """
                MATCH (v:B2bVendor {group_id: $gid})-[r:SWITCHED_TO]->(winner:B2bVendor)
                WHERE toLower(v.canonical_name) = toLower($name)
                RETURN winner.canonical_name AS name,
                       r.mention_count AS mentions,
                       r.primary_driver AS driver,
                       r.signal_strength AS strength
                ORDER BY r.mention_count DESC
                LIMIT 10
                """,
                name=vendor_name, gid=GROUP_ID,
            )
            losing_to = [dict(r) async for r in losing_result]

            # Vendors whose users are switching TO this vendor
            winning_result = await session.run(
                """
                MATCH (loser:B2bVendor)-[r:SWITCHED_TO]->(v:B2bVendor {group_id: $gid})
                WHERE toLower(v.canonical_name) = toLower($name)
                RETURN loser.canonical_name AS name,
                       r.mention_count AS mentions,
                       r.primary_driver AS driver,
                       r.signal_strength AS strength
                ORDER BY r.mention_count DESC
                LIMIT 10
                """,
                name=vendor_name, gid=GROUP_ID,
            )
            winning_from = [dict(r) async for r in winning_result]

            # Shared integrations with competitors
            shared_int_result = await session.run(
                """
                MATCH (v:B2bVendor {group_id: $gid})-[:INTEGRATES_WITH]->(i:B2bIntegration)
                      <-[:INTEGRATES_WITH]-(comp:B2bVendor)
                WHERE toLower(v.canonical_name) = toLower($name)
                  AND comp.canonical_name <> v.canonical_name
                  AND EXISTS { (v)-[:COMPETES_WITH]-(comp) }
                RETURN comp.canonical_name AS vendor,
                       i.integration_name AS integration
                ORDER BY comp.canonical_name
                LIMIT 20
                """,
                name=vendor_name, gid=GROUP_ID,
            )
            shared_ints = [dict(r) async for r in shared_int_result]

            # Shared pain points with competitors
            shared_pain_result = await session.run(
                """
                MATCH (v:B2bVendor {group_id: $gid})-[:HAS_PAIN]->(pp:B2bPainPoint)
                WHERE toLower(v.canonical_name) = toLower($name)
                WITH v, pp.pain_category AS pain
                MATCH (comp:B2bVendor)-[:HAS_PAIN]->(pp2:B2bPainPoint)
                WHERE pp2.pain_category = pain
                  AND comp.canonical_name <> v.canonical_name
                  AND EXISTS { (v)-[:COMPETES_WITH]-(comp) }
                RETURN comp.canonical_name AS vendor,
                       pain,
                       pp2.mention_count AS mentions
                ORDER BY pp2.mention_count DESC
                LIMIT 20
                """,
                name=vendor_name, gid=GROUP_ID,
            )
            shared_pains = [dict(r) async for r in shared_pain_result]

        return {
            "vendor": vendor_name,
            "losing_to": losing_to,
            "winning_from": winning_from,
            "shared_integrations": shared_ints,
            "shared_pains": shared_pains,
        }

    async def displacement_chain(
        self, vendor_name: str, max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """Multi-hop displacement chain: who's displacing whom, transitively.

        Example: Mailchimp -[pricing]-> Klaviyo -[features]-> HubSpot
        Returns list of paths as [{from, to, driver, hops}].
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH path = (start:B2bVendor {group_id: $gid})
                              -[:SWITCHED_TO*1..""" + str(max_hops) + """]->(end:B2bVendor)
                WHERE toLower(start.canonical_name) = toLower($name)
                UNWIND relationships(path) AS rel
                WITH path, collect({
                    from: startNode(rel).canonical_name,
                    to: endNode(rel).canonical_name,
                    driver: rel.primary_driver,
                    mentions: rel.mention_count
                }) AS steps
                RETURN steps, length(path) AS hops
                ORDER BY hops ASC
                LIMIT 20
                """,
                name=vendor_name, gid=GROUP_ID,
            )
            chains = []
            async for record in result:
                chains.append({
                    "steps": record["steps"],
                    "hops": record["hops"],
                })
            return chains

    async def integration_risk_assessment(self, vendor_name: str) -> dict[str, Any]:
        """Find integration dependencies at risk if vendor churns.

        Returns vendors that share integrations with the target AND show
        displacement activity -- signals integration ecosystem disruption.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (v:B2bVendor {group_id: $gid})-[vi:INTEGRATES_WITH]->(i:B2bIntegration)
                WHERE toLower(v.canonical_name) = toLower($name)
                OPTIONAL MATCH (i)<-[ci:INTEGRATES_WITH]-(comp:B2bVendor)
                WHERE comp.canonical_name <> v.canonical_name
                OPTIONAL MATCH (v)-[sw:SWITCHED_TO]->(comp)
                RETURN i.integration_name AS integration,
                       vi.mention_count AS vendor_mentions,
                       collect(DISTINCT {
                           competitor: comp.canonical_name,
                           also_integrates: ci IS NOT NULL,
                           displacement_active: sw IS NOT NULL,
                           displacement_mentions: sw.mention_count
                       }) AS competitors
                ORDER BY vi.mention_count DESC
                """,
                name=vendor_name, gid=GROUP_ID,
            )
            integrations = []
            async for record in result:
                comps = [c for c in record["competitors"] if c.get("competitor")]
                integrations.append({
                    "integration": record["integration"],
                    "vendor_mentions": record["vendor_mentions"],
                    "competitors": comps,
                    "at_risk": any(c.get("displacement_active") for c in comps),
                })

        return {
            "vendor": vendor_name,
            "integrations": integrations,
            "at_risk_count": sum(1 for i in integrations if i["at_risk"]),
        }

    async def company_churn_risk_path(self, company_name: str) -> dict[str, Any]:
        """Trace a company's vendor dependency and churn exposure.

        Path: Company -[USES]-> Vendor -[SWITCHED_TO]-> Competitor
        Shows what alternatives the company's vendor's users are moving to.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (c:B2bCompany {group_id: $gid})-[:USES|CONSIDERING]->(v:B2bVendor)
                WHERE toLower(c.company_name) = toLower($name)
                OPTIONAL MATCH (v)-[sw:SWITCHED_TO]->(alt:B2bVendor)
                OPTIONAL MATCH (v)-[:HAS_PAIN]->(pp:B2bPainPoint)
                RETURN v.canonical_name AS vendor,
                       v.churn_density AS churn_density,
                       v.avg_urgency AS avg_urgency,
                       collect(DISTINCT {
                           alternative: alt.canonical_name,
                           mentions: sw.mention_count,
                           driver: sw.primary_driver
                       }) AS alternatives,
                       collect(DISTINCT pp.pain_category) AS pain_points
                """,
                name=company_name, gid=GROUP_ID,
            )
            vendors = []
            async for record in result:
                alts = [a for a in record["alternatives"] if a.get("alternative")]
                vendors.append({
                    "vendor": record["vendor"],
                    "churn_density": record["churn_density"],
                    "avg_urgency": record["avg_urgency"],
                    "alternatives": alts,
                    "pain_points": record["pain_points"],
                })

        return {
            "company": company_name,
            "vendor_dependencies": vendors,
        }

    async def category_competitive_map(self, category: str) -> dict[str, Any]:
        """Map all vendors in a category with their competitive relationships.

        Returns the full competitive graph for a product category.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (v:B2bVendor {group_id: $gid})
                WHERE toLower(v.product_category) = toLower($cat)
                OPTIONAL MATCH (v)-[sw:SWITCHED_TO]->(comp:B2bVendor)
                OPTIONAL MATCH (v)-[:HAS_PAIN]->(pp:B2bPainPoint)
                RETURN v.canonical_name AS vendor,
                       v.churn_density AS churn_density,
                       v.total_reviews AS total_reviews,
                       v.avg_urgency AS avg_urgency,
                       collect(DISTINCT {
                           to: comp.canonical_name,
                           mentions: sw.mention_count,
                           driver: sw.primary_driver
                       }) AS outgoing_displacement,
                       collect(DISTINCT pp.pain_category) AS pains
                ORDER BY v.churn_density DESC
                """,
                cat=category, gid=GROUP_ID,
            )
            vendors = []
            async for record in result:
                disps = [d for d in record["outgoing_displacement"] if d.get("to")]
                vendors.append({
                    "vendor": record["vendor"],
                    "churn_density": record["churn_density"],
                    "total_reviews": record["total_reviews"],
                    "avg_urgency": record["avg_urgency"],
                    "losing_to": disps,
                    "pains": record["pains"],
                })

        return {
            "category": category,
            "vendor_count": len(vendors),
            "vendors": vendors,
        }

    async def graph_stats(self) -> dict[str, int]:
        """Return node and edge counts for the knowledge graph."""
        async with self._driver.session() as session:
            counts = {}
            for label in ["B2bVendor", "B2bProduct", "B2bPainPoint",
                          "B2bCompany", "B2bIntegration"]:
                result = await session.run(
                    f"MATCH (n:{label} {{group_id: $gid}}) RETURN count(n) AS cnt",
                    gid=GROUP_ID,
                )
                record = await result.single()
                counts[label] = record["cnt"] if record else 0

            for rel_type in ["SWITCHED_TO", "COMPETES_WITH", "INTEGRATES_WITH",
                             "HAS_PAIN", "HAS_PRODUCT", "USES", "CONSIDERING"]:
                result = await session.run(
                    f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt",
                )
                record = await result.single()
                counts[rel_type] = record["cnt"] if record else 0

        return counts
