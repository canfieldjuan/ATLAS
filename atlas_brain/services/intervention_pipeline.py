"""
Intervention pipeline -- chains intelligence skills into a three-stage sequence.

Stage 1: Adaptive Intervention  -- report findings -> F.A.T.E. tactical playbook
Stage 2: Simulated Evolution    -- playbook + signals -> scenario matrix + outcome trajectories
Stage 3: Narrative Architect    -- simulation outcomes -> micro-intervention plan (gated by safety check)

Usage:
    from atlas_brain.services.intervention_pipeline import run_intervention_pipeline

    result = await run_intervention_pipeline("Boeing", objectives=["de-escalate"])
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("atlas.services.intervention_pipeline")

# Safety gate: narrative architect requires orchestration layer controls.
# Until Gap #6 (safety & compliance) is implemented, stage 3 is blocked
# unless the caller explicitly opts in with allow_narrative_architect=True.
_SAFETY_WARNING = (
    "Stage 3 (Narrative Architect) skipped: orchestration-layer safety "
    "controls (approval workflows, audit logs, content filtering, human "
    "review gates) are not yet implemented. Pass allow_narrative_architect=True "
    "to override for testing/research purposes only."
)


async def run_intervention_pipeline(
    entity_name: str,
    *,
    entity_type: str = "company",
    time_window_days: int = 7,
    objectives: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    audience: str = "executive",
    risk_tolerance: str = "moderate",
    simulation_horizon: str = "7 days",
    hours_before_event: int = 48,
    channels: Optional[list[str]] = None,
    allow_narrative_architect: bool = False,
    requested_by: Optional[str] = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Run the full intervention pipeline for an entity.

    Args:
        entity_name: Entity to analyze.
        entity_type: company, person, or sector.
        time_window_days: Lookback window for data gathering.
        objectives: Goals -- de-escalate, stabilize, capitalize.
        constraints: Legal, ethical, comms, or operational limits.
        audience: Target reader (executive, ops lead, negotiator).
        risk_tolerance: low, moderate, or high.
        simulation_horizon: Time window for projections (e.g. "7 days").
        hours_before_event: T-minus hours for calibration checkpoints.
        channels: Communication channels for interventions.
        allow_narrative_architect: Enable stage 3 (requires safety acknowledgment).
        requested_by: Caller identifier for audit trail.
        persist: Store pipeline results in the database.

    Returns:
        Dict with pipeline_id, stages (playbook, simulation, narrative_plan),
        metadata, and safety_warnings.
    """
    from ..storage.database import get_db_pool
    from .intelligence_report import _gather_entity_data, _run_sensors_on_articles

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"error": "Database not initialized"}

    objectives = objectives or ["de-escalate", "stabilize"]
    constraints = constraints or []
    channels = channels or ["internal comms"]

    # ---- Gather data (reuse from intelligence_report) ----
    pressure, articles, journal, graph_facts = await _gather_entity_data(
        pool, entity_name, entity_type, time_window_days,
    )

    if not pressure and not articles and not journal:
        return {"error": f"No data found for entity '{entity_name}'"}

    # Run behavioral sensors for signal input
    sensor_summary = _run_sensors_on_articles(articles)

    # Build shared context
    time_window_str = f"last {time_window_days} days"
    behavioral_triggers = _build_trigger_list(articles, pressure, sensor_summary)
    pressure_points = _build_pressure_points(articles, pressure, sensor_summary)
    evidence = _build_evidence(articles)

    pipeline_id = str(uuid4())
    stages: dict[str, Any] = {}
    safety_warnings: list[str] = []

    # ---- Stage 1: Adaptive Intervention (F.A.T.E. Playbook) ----
    logger.info("Pipeline %s stage 1: adaptive intervention for %s", pipeline_id, entity_name)

    # Build report findings summary for stage 1 input
    report_findings = _build_report_findings(pressure, articles, sensor_summary, journal)

    playbook_payload = {
        "subject": entity_name,
        "time_window": time_window_str,
        "report_findings": report_findings,
        "behavioral_triggers": behavioral_triggers,
        "pressure_points": pressure_points,
        "objectives": ", ".join(objectives),
        "constraints": "; ".join(constraints) if constraints else "Standard legal and ethical guidelines apply",
        "audience": audience,
        "evidence": evidence,
    }

    from ..pipelines.llm import call_llm_with_skill

    playbook_text = call_llm_with_skill(
        "intelligence/adaptive_intervention", playbook_payload,
        max_tokens=1200, temperature=0.3,
    )

    if not playbook_text:
        return {
            "pipeline_id": pipeline_id,
            "error": "Stage 1 (adaptive intervention) failed -- LLM returned no output",
            "stages": stages,
        }

    stages["playbook"] = {
        "skill": "intelligence/adaptive_intervention",
        "text": playbook_text,
        "status": "completed",
    }

    # ---- Stage 2: Simulated Evolution ----
    logger.info("Pipeline %s stage 2: simulated evolution for %s", pipeline_id, entity_name)

    simulation_payload = {
        "subject": entity_name,
        "time_window": time_window_str,
        "high_pressure_signals": pressure_points,
        "intervention_playbook": playbook_text,
        "behavioral_triggers": behavioral_triggers,
        "objectives": ", ".join(objectives),
        "simulation_horizon": simulation_horizon,
        "hours_before_event": hours_before_event,
        "risk_tolerance": risk_tolerance,
        "constraints": "; ".join(constraints) if constraints else "Standard legal and ethical guidelines apply",
        "audience": audience,
        "evidence": evidence,
    }

    if sensor_summary:
        simulation_payload["sensor_analysis"] = sensor_summary

    simulation_text = call_llm_with_skill(
        "intelligence/simulated_evolution", simulation_payload,
        max_tokens=1500, temperature=0.3,
    )

    if not simulation_text:
        stages["simulation"] = {"skill": "intelligence/simulated_evolution", "status": "failed"}
        return {
            "pipeline_id": pipeline_id,
            "error": "Stage 2 (simulated evolution) failed -- LLM returned no output",
            "stages": stages,
        }

    stages["simulation"] = {
        "skill": "intelligence/simulated_evolution",
        "text": simulation_text,
        "status": "completed",
    }

    # ---- Stage 3: Narrative Architect (safety-gated) ----
    from .safety_gate import get_safety_gate
    gate = get_safety_gate()

    if not allow_narrative_architect:
        safety_warnings.append(_SAFETY_WARNING)
        stages["narrative_plan"] = {
            "skill": "intelligence/autonomous_narrative_architect",
            "status": "blocked_by_caller",
            "reason": _SAFETY_WARNING,
        }
        logger.info("Pipeline %s stage 3 skipped: caller did not opt in", pipeline_id)
    else:
        logger.info("Pipeline %s stage 3: narrative architect for %s", pipeline_id, entity_name)

        # Run safety gate on stage 2 output before proceeding
        gate_result = await gate.gate_check(
            pipeline_id=pipeline_id,
            stage="narrative_architect",
            entity_name=entity_name,
            stage_output=simulation_text,
            sensor_summary=sensor_summary,
            pressure=pressure,
            requested_by=requested_by or "unknown",
        )

        if not gate_result["allowed"]:
            safety_warnings.append(gate_result["reason"])
            stages["narrative_plan"] = {
                "skill": "intelligence/autonomous_narrative_architect",
                "status": "blocked_by_safety_gate",
                "reason": gate_result["reason"],
                "approval_id": gate_result.get("approval_id"),
                "risk_assessment": gate_result.get("risk_assessment"),
                "content_check": gate_result.get("content_check"),
            }
            logger.info(
                "Pipeline %s stage 3 blocked by safety gate: %s",
                pipeline_id, gate_result["reason"],
            )
        else:
            # Safety gate passed -- proceed with stage 3
            current_score = pressure.get("pressure_score", 0)
            if isinstance(current_score, (int, float)):
                thresholds = {
                    "green": f"0-{max(3, current_score - 2):.0f}",
                    "yellow": f"{max(3, current_score - 2):.0f}-{max(6, current_score):.0f}",
                    "red": f"{max(6, current_score):.0f}-10",
                }
            else:
                thresholds = {"green": "0-3", "yellow": "4-6", "red": "7-10"}

            narrative_payload = {
                "subject": entity_name,
                "time_window": time_window_str,
                "high_pressure_signals": pressure_points,
                "simulation_outcomes": simulation_text,
                "core_story": _detect_core_story(articles),
                "target_clusters": [entity_name],
                "channels": channels,
                "intervention_library": "No pre-approved intervention library available -- generate from analysis",
                "pressure_thresholds": thresholds,
                "hours_before_event": hours_before_event,
                "constraints": "; ".join(constraints) if constraints else "Standard legal and ethical guidelines apply",
                "risk_tolerance": risk_tolerance,
                "audience": audience,
                "evidence": evidence,
            }

            narrative_text = call_llm_with_skill(
                "intelligence/autonomous_narrative_architect", narrative_payload,
                max_tokens=1500, temperature=0.3,
            )

            if narrative_text:
                # Run content filter on the output
                content_check = gate.check_content(narrative_text)
                if content_check["blocked"]:
                    safety_warnings.append(
                        f"Stage 3 output blocked by content filter: "
                        f"{', '.join(f['pattern'] for f in content_check['flags'])}"
                    )
                    await gate.log_event(
                        event_type="intervention.output_blocked",
                        source="safety_gate",
                        entity_name=entity_name,
                        payload={
                            "pipeline_id": pipeline_id,
                            "stage": "narrative_architect",
                            "flags": content_check["flags"],
                        },
                    )
                    stages["narrative_plan"] = {
                        "skill": "intelligence/autonomous_narrative_architect",
                        "status": "blocked_by_content_filter",
                        "content_check": content_check,
                    }
                else:
                    await gate.log_event(
                        event_type="intervention.stage_completed",
                        source=requested_by or "unknown",
                        entity_name=entity_name,
                        payload={
                            "pipeline_id": pipeline_id,
                            "stage": "narrative_architect",
                            "risk_level": gate_result.get("risk_assessment", {}).get("risk_level", "UNKNOWN"),
                        },
                    )
                    stages["narrative_plan"] = {
                        "skill": "intelligence/autonomous_narrative_architect",
                        "text": narrative_text,
                        "status": "completed",
                    }
            else:
                stages["narrative_plan"] = {
                    "skill": "intelligence/autonomous_narrative_architect",
                    "status": "failed",
                }

    # ---- Build result ----
    result: dict[str, Any] = {
        "pipeline_id": pipeline_id,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "time_window_days": time_window_days,
        "objectives": objectives,
        "stages": stages,
        "stages_completed": sum(1 for s in stages.values() if s.get("status") == "completed"),
        "stages_total": 3,
        "sensor_summary": sensor_summary,
        "pressure_snapshot": pressure or {},
        "articles_analyzed": len(articles),
        "safety_warnings": safety_warnings,
        "requested_by": requested_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Persist
    if persist:
        await _persist_pipeline(pool, result)

    return result


async def _persist_pipeline(pool, result: dict[str, Any]) -> None:
    """Store pipeline run in the intelligence_reports table with type 'intervention'."""
    try:
        # Combine all stage text into one report
        combined_text = []
        for stage_name, stage_data in result.get("stages", {}).items():
            if stage_data.get("text"):
                combined_text.append(f"=== {stage_name.upper()} ===\n\n{stage_data['text']}")

        report_text = "\n\n".join(combined_text) if combined_text else "Pipeline produced no output"

        structured = {
            "pipeline_id": result["pipeline_id"],
            "objectives": result.get("objectives", []),
            "stages_completed": result.get("stages_completed", 0),
            "stages_total": result.get("stages_total", 3),
            "sensor_summary": result.get("sensor_summary", {}),
            "safety_warnings": result.get("safety_warnings", []),
            "stage_statuses": {
                name: data.get("status", "unknown")
                for name, data in result.get("stages", {}).items()
            },
        }

        await pool.execute(
            """
            INSERT INTO intelligence_reports (
                id, entity_name, entity_type, report_type,
                time_window_days, report_text, structured_data,
                pressure_snapshot, source_article_ids, requested_by
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10)
            """,
            result["pipeline_id"],
            result["entity_name"],
            result.get("entity_type", "company"),
            "intervention",
            result.get("time_window_days", 7),
            report_text,
            json.dumps(structured),
            json.dumps(result.get("pressure_snapshot", {}), default=str),
            [],
            result.get("requested_by"),
        )
        logger.info("Stored intervention pipeline %s for %s", result["pipeline_id"], result["entity_name"])
    except Exception:
        logger.exception("Failed to persist intervention pipeline")


# ------------------------------------------------------------------
# Helper: build data payloads from gathered entity data
# ------------------------------------------------------------------


def _build_trigger_list(
    articles: list[dict], pressure: dict, sensor_summary: dict,
) -> list[str]:
    """Build a trigger list combining pressure baselines, article signals, and sensor data."""
    triggers: list[str] = []

    score = pressure.get("pressure_score", 0)
    if isinstance(score, (int, float)) and score > 0:
        triggers.append(f"Pressure score: {score}/10")

    soram = pressure.get("soram_breakdown", {})
    if soram:
        dominant = max(soram, key=lambda k: soram.get(k, 0), default="")
        if dominant:
            triggers.append(f"Dominant SORAM channel: {dominant} ({soram[dominant]:.2f})")

    ling = pressure.get("linguistic_signals", {})
    active_signals = [k for k, v in ling.items() if v]
    if active_signals:
        triggers.append(f"Active linguistic indicators: {', '.join(active_signals)}")

    # Article pressure direction
    directions = [a.get("pressure_direction") for a in articles if a.get("pressure_direction")]
    if directions:
        from collections import Counter
        counts = Counter(directions)
        dominant_dir = counts.most_common(1)[0]
        triggers.append(f"Article pressure: {dominant_dir[0]} ({dominant_dir[1]}/{len(directions)} articles)")

    # Sensor-derived triggers
    if sensor_summary:
        level = sensor_summary.get("dominant_risk_level", "LOW")
        triggers.append(f"Behavioral sensor composite: {level}")
        tc = sensor_summary.get("triggered_counts", {})
        fired = [f"{k}={v}" for k, v in tc.items() if v > 0]
        if fired:
            triggers.append(f"Sensors triggered: {', '.join(fired)}")
        patterns = sensor_summary.get("cross_sensor_patterns", [])
        if patterns:
            triggers.append(f"Cross-sensor patterns: {', '.join(patterns)}")

    return triggers


def _build_pressure_points(
    articles: list[dict], pressure: dict, sensor_summary: dict,
) -> list[dict[str, str]]:
    """Build ranked pressure points with confidence and impact."""
    points: list[dict[str, str]] = []

    score = pressure.get("pressure_score", 0)
    if isinstance(score, (int, float)) and score >= 5:
        points.append({
            "trigger": f"Elevated pressure baseline ({score}/10)",
            "confidence": "high" if score >= 7 else "medium",
            "impact": "critical" if score >= 8 else "significant",
        })

    drift = pressure.get("sentiment_drift", 0)
    if isinstance(drift, (int, float)) and drift < -0.1:
        points.append({
            "trigger": f"Negative sentiment drift ({drift:+.3f})",
            "confidence": "medium",
            "impact": "significant",
        })

    # Article-derived pressure points
    building_count = sum(1 for a in articles if a.get("pressure_direction") == "building")
    if building_count >= 2:
        points.append({
            "trigger": f"Pressure building in {building_count}/{len(articles)} articles",
            "confidence": "high" if building_count >= 3 else "medium",
            "impact": "significant",
        })

    # Sensor-derived
    if sensor_summary:
        risk = sensor_summary.get("dominant_risk_level", "LOW")
        if risk in ("HIGH", "CRITICAL"):
            points.append({
                "trigger": f"Behavioral sensor risk: {risk}",
                "confidence": "high",
                "impact": "critical" if risk == "CRITICAL" else "significant",
            })

    return points


def _build_evidence(articles: list[dict]) -> list[dict[str, str]]:
    """Build evidence list from article data."""
    evidence: list[dict[str, str]] = []
    for a in articles[:10]:
        preview = a.get("content_preview") or a.get("summary") or ""
        if preview:
            evidence.append({
                "excerpt": preview[:300],
                "source": a.get("source", "unknown"),
                "title": a.get("title", ""),
            })
    return evidence


def _build_report_findings(
    pressure: dict, articles: list[dict],
    sensor_summary: dict, journal: list[dict],
) -> str:
    """Build a concise report findings summary for stage 1 input."""
    parts: list[str] = []

    score = pressure.get("pressure_score", 0)
    if isinstance(score, (int, float)) and score > 0:
        parts.append(f"Current pressure score: {score}/10.")

    soram = pressure.get("soram_breakdown", {})
    if soram:
        channels = ", ".join(f"{k}={v:.2f}" for k, v in sorted(soram.items(), key=lambda x: -x[1]))
        parts.append(f"SORAM channels: {channels}.")

    if articles:
        parts.append(f"{len(articles)} articles analyzed in the time window.")
        directions = [a.get("pressure_direction") for a in articles if a.get("pressure_direction")]
        if directions:
            from collections import Counter
            dir_counts = Counter(directions)
            parts.append(f"Pressure directions: {dict(dir_counts)}.")

    if sensor_summary:
        level = sensor_summary.get("dominant_risk_level", "LOW")
        parts.append(f"Behavioral sensor composite risk: {level}.")
        patterns = sensor_summary.get("cross_sensor_patterns", [])
        if patterns:
            parts.append(f"Cross-sensor patterns detected: {', '.join(patterns)}.")

    if journal:
        insights = []
        for j in journal[:2]:
            for ins in j.get("key_insights", [])[:2]:
                insights.append(str(ins))
        if insights:
            parts.append(f"Prior analysis insights: {'; '.join(insights)}.")

    return " ".join(parts) if parts else "No prior report findings available."


def _detect_core_story(articles: list[dict]) -> str:
    """Detect the dominant narrative from article titles and summaries."""
    if not articles:
        return "No dominant narrative detected -- insufficient article data."

    # Use the most recent article titles as a proxy
    titles = [a.get("title", "") for a in articles[:5] if a.get("title")]
    if titles:
        return f"Dominant narrative themes from recent coverage: {'; '.join(titles)}"
    return "Narrative unclear -- article titles unavailable."
