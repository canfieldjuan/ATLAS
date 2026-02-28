"""
Intelligence report orchestration.

Gathers entity data from pressure baselines, enriched news articles,
reasoning journal, and knowledge graph, then calls intelligence skill
prompts to produce structured reports.

Usage:
    from atlas_brain.services.intelligence_report import generate_report

    report = await generate_report("Boeing", time_window_days=14)
    # report["report_text"] -- rendered report
    # report["structured_data"] -- parsed sections
    # report["pressure_snapshot"] -- entity pressure state at generation time
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("atlas.services.intelligence_report")


async def generate_report(
    entity_name: str,
    *,
    entity_type: str = "company",
    time_window_days: int = 7,
    report_type: str = "full",
    audience: str = "executive",
    requested_by: Optional[str] = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Generate an intelligence report for a specific entity.

    Args:
        entity_name: Entity to analyze (company, person, sector).
        entity_type: Classification of entity (company, person, sector).
        time_window_days: How far back to look for data.
        report_type: "full" (600-word report) or "executive" (200-word summary).
        audience: Target reader persona (executive, ops lead, investor).
        requested_by: Who requested the report (user, api, autonomous).
        persist: Store the report in the database.

    Returns:
        Dict with report_id, report_text, structured_data, pressure_snapshot,
        source_article_ids, and metadata.
    """
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"error": "Database not initialized"}

    # Gather data for this entity
    pressure, articles, journal, graph_facts, graph_network = await _gather_entity_data(
        pool, entity_name, entity_type, time_window_days,
    )

    if not pressure and not articles and not journal:
        return {"error": f"No data found for entity '{entity_name}'"}

    # Run behavioral risk sensors on article text
    sensor_summary = _run_sensors_on_articles(articles)

    # Build signals and evidence from articles
    signals = _extract_signals(articles)
    evidence = _extract_evidence(articles)
    risks, opportunities = _extract_risks_opportunities(articles, pressure)

    # Pick the skill based on report_type
    if report_type == "executive":
        skill_name = "intelligence/prompt_to_report"
        payload = {
            "subject": entity_name,
            "time_window": f"last {time_window_days} days",
            "behavioral_triggers": _format_behavioral_triggers(articles, pressure),
            "signals": signals,
            "evidence": evidence,
            "audience": audience,
        }
    else:
        skill_name = "intelligence/report"
        payload = {
            "subject": entity_name,
            "time_window": f"last {time_window_days} days",
            "relationships": _format_relationships(articles, graph_facts, graph_network),
            "signals": signals,
            "evidence": evidence,
            "risks": risks,
            "opportunities": opportunities,
            "audience": audience,
        }

    # Inject sensor analysis into payload
    if sensor_summary:
        payload["behavioral_sensor_analysis"] = sensor_summary

    # Inject pressure context into the payload
    if pressure:
        payload["pressure_baseline"] = {
            "score": pressure.get("pressure_score", 0.0),
            "sentiment_drift": pressure.get("sentiment_drift", 0.0),
            "soram_breakdown": pressure.get("soram_breakdown", {}),
            "linguistic_signals": pressure.get("linguistic_signals", {}),
        }
    if journal:
        payload["prior_analysis"] = [
            {
                "date": str(j.get("session_date", "")),
                "key_insights": j.get("key_insights", []),
            }
            for j in journal[:3]
        ]

    # Call LLM with skill
    from ..pipelines.llm import call_llm_with_skill

    report_text = call_llm_with_skill(
        skill_name, payload,
        max_tokens=1500 if report_type == "full" else 500,
        temperature=0.3,
    )

    if not report_text:
        return {"error": "LLM failed to generate report"}

    # Build result
    source_ids = [a["id"] for a in articles if a.get("id")]
    pressure_snapshot = pressure or {}

    result: dict[str, Any] = {
        "report_id": str(uuid4()),
        "entity_name": entity_name,
        "entity_type": entity_type,
        "report_type": report_type,
        "time_window_days": time_window_days,
        "report_text": report_text,
        "structured_data": {
            "signals_count": len(signals),
            "evidence_count": len(evidence),
            "articles_analyzed": len(articles),
            "journal_entries_referenced": len(journal),
            "graph_facts_used": len(graph_facts),
            "graph_network_paths": len(graph_network),
        },
        "pressure_snapshot": pressure_snapshot,
        "source_article_ids": source_ids,
        "requested_by": requested_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Persist
    if persist:
        try:
            await pool.execute(
                """
                INSERT INTO intelligence_reports (
                    id, entity_name, entity_type, report_type,
                    time_window_days, report_text, structured_data,
                    pressure_snapshot, source_article_ids, requested_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10)
                """,
                result["report_id"],
                entity_name,
                entity_type,
                report_type,
                time_window_days,
                report_text,
                json.dumps(result["structured_data"]),
                json.dumps(pressure_snapshot, default=str),
                source_ids,
                requested_by,
            )
            logger.info(
                "Stored intelligence report %s for %s (%s, %dd window)",
                result["report_id"], entity_name, report_type, time_window_days,
            )
        except Exception:
            logger.exception("Failed to persist intelligence report")

    return result


async def generate_report_package(
    entity_name: str,
    *,
    client_profile: str = "executive decision-maker",
    price_band: str = "four-figure",
    objectives: str = "",
    time_window_days: int = 30,
) -> dict[str, Any]:
    """Generate a high-ticket report package specification.

    This is a planning/scoping tool that defines what a premium
    report engagement would look like, not the report itself.
    """
    from ..pipelines.llm import call_llm_with_skill

    payload = {
        "subject": entity_name,
        "client_profile": client_profile,
        "price_band": price_band,
        "objectives": objectives or f"Comprehensive behavioral intelligence on {entity_name}",
        "time_window": f"last {time_window_days} days",
        "data_sources": "pressure baselines, SORAM-enriched news, knowledge graph, CRM interactions",
        "delivery_format": "PDF" if price_band == "four-figure" else "deck",
        "visual_preference": "plain-text" if price_band == "four-figure" else "visual-heavy",
        "urgency": "standard (5-7 business days)",
        "writer_model": "Claude Opus for drafting, human review for polish",
    }

    text = call_llm_with_skill(
        "intelligence/report_builder", payload,
        max_tokens=2000, temperature=0.3,
    )

    return {
        "entity_name": entity_name,
        "price_band": price_band,
        "package_spec": text or "Failed to generate package specification",
    }


async def list_reports(
    entity_name: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List recent intelligence reports, optionally filtered by entity."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    if entity_name:
        rows = await pool.fetch(
            """
            SELECT id, entity_name, entity_type, report_type,
                   time_window_days, requested_by, created_at
            FROM intelligence_reports
            WHERE entity_name ILIKE $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            f"%{entity_name}%",
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, entity_name, entity_type, report_type,
                   time_window_days, requested_by, created_at
            FROM intelligence_reports
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )

    return [dict(r) for r in rows]


async def get_report(report_id: str) -> Optional[dict[str, Any]]:
    """Retrieve a full intelligence report by ID."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return None

    row = await pool.fetchrow(
        "SELECT * FROM intelligence_reports WHERE id = $1",
        report_id,
    )
    return dict(row) if row else None


# ------------------------------------------------------------------
# Data gathering
# ------------------------------------------------------------------


async def _gather_entity_data(
    pool,
    entity_name: str,
    entity_type: str,
    window_days: int,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    """Gather all available data for an entity."""
    import asyncio

    pressure, articles, journal, graph_facts, graph_network = await asyncio.gather(
        _fetch_pressure_baseline(pool, entity_name, entity_type),
        _fetch_entity_articles(pool, entity_name, window_days),
        _fetch_entity_journal(pool, entity_name, window_days),
        _fetch_graph_facts(entity_name),
        _fetch_entity_network(entity_name),
        return_exceptions=True,
    )

    if isinstance(pressure, Exception):
        logger.warning("Pressure baseline fetch failed: %s", pressure)
        pressure = {}
    if isinstance(articles, Exception):
        logger.warning("Articles fetch failed: %s", articles)
        articles = []
    if isinstance(journal, Exception):
        logger.warning("Journal fetch failed: %s", journal)
        journal = []
    if isinstance(graph_facts, Exception):
        logger.warning("Graph facts fetch failed: %s", graph_facts)
        graph_facts = []
    if isinstance(graph_network, Exception):
        logger.warning("Graph network fetch failed: %s", graph_network)
        graph_network = []

    return pressure, articles, journal, graph_facts, graph_network


async def _fetch_pressure_baseline(
    pool, entity_name: str, entity_type: str,
) -> dict[str, Any]:
    """Get the current pressure baseline for this entity."""
    row = await pool.fetchrow(
        """
        SELECT entity_name, entity_type, pressure_score, sentiment_drift,
               narrative_frequency, soram_breakdown, linguistic_signals,
               last_computed_at
        FROM entity_pressure_baselines
        WHERE entity_name ILIKE $1 AND entity_type = $2
        """,
        entity_name,
        entity_type,
    )
    if not row:
        return {}

    result = dict(row)
    # Normalize JSONB fields
    for key in ("soram_breakdown", "linguistic_signals"):
        val = result.get(key)
        if isinstance(val, str):
            try:
                result[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                result[key] = {}
    # Convert Decimal to float
    for key in ("pressure_score", "sentiment_drift"):
        if result.get(key) is not None:
            result[key] = float(result[key])
    return result


async def _fetch_entity_articles(
    pool, entity_name: str, window_days: int,
) -> list[dict[str, Any]]:
    """Fetch enriched news articles that mention this entity."""
    rows = await pool.fetch(
        """
        SELECT id, title, source_name, url, summary, content,
               soram_channels, linguistic_indicators, entities_detected,
               pressure_direction, published_at, enriched_at
        FROM news_articles
        WHERE (
            $1 = ANY(entities_detected)
            OR title ILIKE $2
            OR content ILIKE $2
        )
        AND created_at > NOW() - make_interval(days => $3)
        AND enrichment_status = 'classified'
        ORDER BY enriched_at DESC NULLS LAST
        LIMIT 30
        """,
        entity_name,
        f"%{entity_name}%",
        window_days,
    )

    result = []
    for r in rows:
        article: dict[str, Any] = {
            "id": str(r["id"]),
            "title": r["title"],
            "source": r["source_name"],
            "summary": r["summary"],
            "published_at": r["published_at"],
        }
        if r["content"]:
            article["content_preview"] = r["content"][:500]

        soram = r["soram_channels"]
        if isinstance(soram, str):
            try:
                soram = json.loads(soram)
            except (json.JSONDecodeError, TypeError):
                soram = None
        if soram:
            article["soram_channels"] = soram

        ling = r["linguistic_indicators"]
        if isinstance(ling, str):
            try:
                ling = json.loads(ling)
            except (json.JSONDecodeError, TypeError):
                ling = None
        if ling:
            article["linguistic_indicators"] = ling

        if r.get("pressure_direction"):
            article["pressure_direction"] = r["pressure_direction"]
        if r["entities_detected"]:
            article["entities"] = list(r["entities_detected"])

        result.append(article)

    return result


async def _fetch_entity_journal(
    pool, entity_name: str, window_days: int,
) -> list[dict[str, Any]]:
    """Fetch reasoning journal entries that reference this entity."""
    rows = await pool.fetch(
        """
        SELECT session_date, reasoning_output, key_insights,
               connections_found, recommendations, pressure_readings
        FROM reasoning_journal
        WHERE created_at > NOW() - make_interval(days => $1)
        ORDER BY session_date DESC
        LIMIT 5
        """,
        window_days,
    )

    result = []
    for r in rows:
        # Filter to entries that mention the entity
        output = r["reasoning_output"] or ""
        pressure_readings = r["pressure_readings"]
        if isinstance(pressure_readings, str):
            try:
                pressure_readings = json.loads(pressure_readings)
            except (json.JSONDecodeError, TypeError):
                pressure_readings = []

        entity_mentioned = entity_name.lower() in output.lower()
        entity_in_pressure = any(
            (pr.get("entity_name", "").lower() == entity_name.lower())
            for pr in (pressure_readings or [])
            if isinstance(pr, dict)
        )

        if entity_mentioned or entity_in_pressure:
            entry: dict[str, Any] = {
                "session_date": str(r["session_date"]),
                "key_insights": r["key_insights"] if isinstance(r["key_insights"], list) else [],
                "connections_found": r["connections_found"] if isinstance(r["connections_found"], list) else [],
                "recommendations": r["recommendations"] if isinstance(r["recommendations"], list) else [],
            }
            # Include only this entity's pressure reading
            entity_pressure = [
                pr for pr in (pressure_readings or [])
                if isinstance(pr, dict) and pr.get("entity_name", "").lower() == entity_name.lower()
            ]
            if entity_pressure:
                entry["pressure_reading"] = entity_pressure[0]
            result.append(entry)

    return result


async def _fetch_graph_facts(entity_name: str) -> list[dict[str, Any]]:
    """Fetch relationship data from knowledge graph using hybrid search + traversal."""
    try:
        from ..memory.rag_client import get_rag_client

        rag = get_rag_client()
        # Use search_with_traversal for richer results: vector search + direct edges
        result = await rag.search_with_traversal(
            query=f"relationships and facts about {entity_name}",
            entity_name=entity_name,
            max_facts=15,
        )
        if result and result.facts:
            return [
                {
                    "fact": f.fact,
                    "source": f.name,
                    "confidence": f.confidence,
                    "source_type": getattr(f, "source_type", "search"),
                }
                for f in result.facts
                if f.fact
            ]
    except Exception:
        logger.debug("Graph fact fetch failed for %s", entity_name, exc_info=True)
    return []


async def _fetch_entity_network(entity_name: str) -> list[dict[str, Any]]:
    """Fetch multi-hop entity network for relationship mapping."""
    try:
        from ..memory.rag_client import get_rag_client

        rag = get_rag_client()
        paths = await rag.traverse_graph(
            entity_name=entity_name,
            max_hops=2,
            direction="both",
        )
        return paths if isinstance(paths, list) else []
    except Exception:
        logger.debug("Entity network fetch failed for %s", entity_name, exc_info=True)
    return []


# ------------------------------------------------------------------
# Payload construction helpers
# ------------------------------------------------------------------


def _extract_signals(articles: list[dict]) -> list[dict[str, str]]:
    """Extract notable signals from enriched articles."""
    signals = []
    for a in articles:
        signal: dict[str, str] = {"event": a.get("title", "")}
        if a.get("pressure_direction"):
            signal["pressure"] = a["pressure_direction"]
        soram = a.get("soram_channels", {})
        if soram:
            dominant = max(soram, key=soram.get, default="")
            if dominant and soram[dominant] >= 0.5:
                signal["dominant_channel"] = f"{dominant} ({soram[dominant]:.1f})"
        ling = a.get("linguistic_indicators", {})
        active = [k for k, v in ling.items() if v]
        if active:
            signal["linguistic_flags"] = ", ".join(active)
        signals.append(signal)
    return signals


def _extract_evidence(articles: list[dict]) -> list[dict[str, str]]:
    """Extract evidence excerpts from articles."""
    evidence = []
    for a in articles:
        preview = a.get("content_preview") or a.get("summary") or ""
        if preview:
            evidence.append({
                "excerpt": preview[:300],
                "source": a.get("source", "unknown"),
                "title": a.get("title", ""),
            })
    return evidence[:10]


def _extract_risks_opportunities(
    articles: list[dict], pressure: dict,
) -> tuple[list[dict], list[dict]]:
    """Derive risks and opportunities from article patterns."""
    risks = []
    opportunities = []

    # Pressure-based risks
    score = pressure.get("pressure_score", 0)
    if isinstance(score, (int, float)) and score >= 7:
        risks.append({
            "type": "high_pressure",
            "detail": f"Entity pressure score at {score}/10 -- approaching critical threshold",
        })

    drift = pressure.get("sentiment_drift", 0)
    if isinstance(drift, (int, float)) and drift < -0.1:
        risks.append({
            "type": "negative_sentiment_drift",
            "detail": f"Sentiment drifting negative ({drift:+.3f})",
        })

    # Article-based signals
    building_count = sum(1 for a in articles if a.get("pressure_direction") == "building")
    releasing_count = sum(1 for a in articles if a.get("pressure_direction") == "releasing")

    if building_count > len(articles) * 0.6 and len(articles) >= 3:
        risks.append({
            "type": "pressure_building",
            "detail": f"{building_count}/{len(articles)} articles show building pressure",
        })

    if releasing_count > len(articles) * 0.5 and len(articles) >= 3:
        opportunities.append({
            "type": "pressure_releasing",
            "detail": f"{releasing_count}/{len(articles)} articles show releasing pressure",
        })

    # Linguistic escalation
    urgency_count = sum(
        1 for a in articles
        if a.get("linguistic_indicators", {}).get("urgency_escalation")
    )
    if urgency_count >= 2:
        risks.append({
            "type": "urgency_escalation",
            "detail": f"Urgency escalation detected in {urgency_count} articles",
        })

    return risks, opportunities


def _format_behavioral_triggers(
    articles: list[dict], pressure: dict,
) -> list[str]:
    """Format behavioral triggers for the executive summary skill."""
    triggers = []

    score = pressure.get("pressure_score", 0)
    if isinstance(score, (int, float)) and score > 0:
        triggers.append(f"Pressure baseline: {score}/10")

    soram = pressure.get("soram_breakdown", {})
    if soram:
        dominant = max(soram, key=lambda k: soram.get(k, 0), default="")
        if dominant:
            triggers.append(f"Dominant SORAM channel: {dominant} ({soram[dominant]:.2f})")

    ling = pressure.get("linguistic_signals", {})
    active = [k for k, v in ling.items() if v]
    if active:
        triggers.append(f"Active linguistic indicators: {', '.join(active)}")

    directions = [a.get("pressure_direction") for a in articles if a.get("pressure_direction")]
    if directions:
        from collections import Counter
        counts = Counter(directions)
        dominant_dir = counts.most_common(1)[0]
        triggers.append(f"Article pressure direction: {dominant_dir[0]} ({dominant_dir[1]}/{len(directions)} articles)")

    return triggers


def _format_relationships(
    articles: list[dict],
    graph_facts: list[dict],
    graph_network: Optional[list[dict]] = None,
) -> list[dict[str, str]]:
    """Format relationship data for the full report skill."""
    rels = []

    # From graph search + traversal (includes confidence)
    for fact in graph_facts:
        entry: dict[str, str] = {
            "type": "graph",
            "description": fact.get("fact", ""),
            "source": fact.get("source", "knowledge_graph"),
        }
        conf = fact.get("confidence")
        if conf is not None:
            entry["confidence"] = f"{conf:.2f}"
        source_type = fact.get("source_type", "search")
        if source_type != "search":
            entry["retrieval"] = source_type
        rels.append(entry)

    # From multi-hop network traversal
    for path_item in (graph_network or []):
        if isinstance(path_item, dict):
            rel_desc = path_item.get("relation") or path_item.get("fact") or ""
            entity = path_item.get("entity", "")
            hop = path_item.get("hop", 1)
            if rel_desc:
                rels.append({
                    "type": f"network_hop_{hop}",
                    "description": f"{entity}: {rel_desc}" if entity else rel_desc,
                    "source": "graph_traversal",
                })
            # Handle nested path structures from Graphiti /traverse
            for step in path_item.get("path", []):
                if isinstance(step, dict):
                    step_rel = step.get("relation", {})
                    step_entity = step.get("entity", {})
                    fact_text = step_rel.get("fact", "") if isinstance(step_rel, dict) else ""
                    ent_name = step_entity.get("name", "") if isinstance(step_entity, dict) else ""
                    if fact_text:
                        rels.append({
                            "type": "network_traversal",
                            "description": f"{ent_name}: {fact_text}" if ent_name else fact_text,
                            "source": "graph_traversal",
                        })

    # From articles -- extract entity co-occurrences
    entity_pairs: dict[str, int] = {}
    for a in articles:
        entities = a.get("entities", [])
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                pair = f"{e1} <-> {e2}"
                entity_pairs[pair] = entity_pairs.get(pair, 0) + 1

    for pair, count in sorted(entity_pairs.items(), key=lambda x: -x[1])[:10]:
        rels.append({
            "type": "co_occurrence",
            "description": pair,
            "mentions": str(count),
        })

    return rels


def _run_sensors_on_articles(articles: list[dict]) -> dict[str, Any]:
    """Run behavioral risk sensors on article text and return aggregate summary."""
    try:
        from ..tools.risk_sensors import (
            alignment_sensor_tool,
            operational_urgency_tool,
            negotiation_rigidity_tool,
            correlate,
        )
    except Exception:
        return {}

    triggered_counts = {"alignment": 0, "urgency": 0, "rigidity": 0}
    high_confidence_triggers = {"alignment": 0, "urgency": 0, "rigidity": 0}
    risk_levels: list[str] = []
    patterns_seen: list[str] = []
    analyzed = 0
    confidence_dist = {"high": 0, "medium": 0, "low": 0}

    for a in articles:
        text = a.get("content_preview") or a.get("summary") or ""
        if not text or len(text) < 50:
            continue

        try:
            al = alignment_sensor_tool.analyze(text)
            ur = operational_urgency_tool.analyze(text)
            ri = negotiation_rigidity_tool.analyze(text)
            cross = correlate(al, ur, ri)

            # Cross-validate sensors with SORAM (same logic as daily_intelligence)
            soram = a.get("soram_channels", {})
            conf = _sensor_confidence_from_soram(soram, cross)

            analyzed += 1
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

            if al["triggered"]:
                triggered_counts["alignment"] += 1
                if conf == "high":
                    high_confidence_triggers["alignment"] += 1
            if ur["triggered"]:
                triggered_counts["urgency"] += 1
                if conf == "high":
                    high_confidence_triggers["urgency"] += 1
            if ri["triggered"]:
                triggered_counts["rigidity"] += 1
                if conf == "high":
                    high_confidence_triggers["rigidity"] += 1
            risk_levels.append(cross["composite_risk_level"])
            for rel in cross["relationships"]:
                if rel["label"] not in patterns_seen:
                    patterns_seen.append(rel["label"])
        except Exception:
            continue

    if not analyzed:
        return {}

    # Find dominant risk level
    from collections import Counter
    level_counts = Counter(risk_levels)
    dominant_level = level_counts.most_common(1)[0][0] if level_counts else "LOW"

    return {
        "articles_analyzed": analyzed,
        "triggered_counts": triggered_counts,
        "high_confidence_triggers": high_confidence_triggers,
        "dominant_risk_level": dominant_level,
        "risk_level_distribution": dict(level_counts),
        "cross_sensor_patterns": patterns_seen,
        "confidence_distribution": confidence_dist,
    }


def _sensor_confidence_from_soram(soram: dict, cross: dict) -> str:
    """Rate sensor confidence using SORAM as ground truth.

    Sensors are context-blind term counters. SORAM is LLM-driven and
    understands whether adversarial language is direct or quoted.
    """
    if not soram or cross.get("sensor_count", 0) == 0:
        return "low"

    supporting = max(
        soram.get("operational", 0.0),
        soram.get("alignment", 0.0),
        soram.get("societal", 0.0),
    )

    if supporting >= 0.5 and cross.get("sensor_count", 0) >= 2:
        return "high"
    if supporting >= 0.3:
        return "medium"
    if soram.get("media", 0.0) >= 0.6 and supporting < 0.3:
        return "low"
    return "medium"
