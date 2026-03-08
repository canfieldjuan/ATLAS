"""
Shared notification helper for pipeline tasks.

Replaces identical _send_notification() functions across
complaint_analysis, daily_intelligence, and complaint_content_generation.

Formats structured LLM output into readable markdown for ntfy push
notifications.
"""

from __future__ import annotations

import logging
from typing import Any

from ..storage.models import ScheduledTask

logger = logging.getLogger("atlas.pipelines.notify")


async def send_pipeline_notification(
    message: str,
    task: ScheduledTask,
    *,
    title: str | None = None,
    default_tags: str = "brain",
    max_chars: int = 4000,
    parsed: dict[str, Any] | None = None,
) -> None:
    """Send an ntfy push notification for a pipeline task result.

    When *parsed* is provided, builds a structured markdown notification
    from known fields (key_insights, recommendations, top_pain_points,
    opportunities, pressure_readings, product_highlights).  Falls back to
    the raw *message* when *parsed* is not given.

    Checks autonomous config (notify_results), alerts config (ntfy_enabled),
    and per-task opt-out (metadata.notify). Truncates message to ``max_chars``
    (ntfy has a ~4KB limit).
    """
    from ..config import settings

    if not settings.autonomous.notify_results:
        return
    if not settings.alerts.ntfy_enabled:
        return
    if (task.metadata or {}).get("notify") is False:
        return

    if title is None:
        title = f"Atlas: {task.name.replace('_', ' ').title()}"

    priority = (task.metadata or {}).get("notify_priority", "default")
    tags = (task.metadata or {}).get("notify_tags", default_tags)

    # Build formatted message from parsed data when available
    if parsed:
        formatted = _format_parsed(parsed, message)
    else:
        formatted = message

    try:
        from ..tools.notify import notify_tool

        await notify_tool._send_notification(
            message=formatted[:max_chars],
            title=title,
            priority=priority,
            tags=tags,
            markdown=True,
        )
        logger.info("Sent notification for task '%s'", task.name)
    except Exception:
        logger.warning("Failed to send notification for task '%s'", task.name, exc_info=True)


def _format_parsed(parsed: dict[str, Any], fallback: str) -> str:
    """Build a readable markdown notification from structured LLM output.

    Looks for common fields across pipeline tasks and formats whichever
    are present.  Keeps it concise -- ntfy has a ~4KB body limit.
    """
    parts: list[str] = []

    # Lead with the narrative summary
    analysis = parsed.get("analysis_text", "")
    if analysis:
        parts.append(analysis.strip())

    # -- Complaint analysis fields --

    pain_points = parsed.get("top_pain_points", [])
    if pain_points and isinstance(pain_points, list):
        items = []
        for pp in pain_points[:5]:
            if isinstance(pp, dict):
                asin = pp.get("asin", "")
                issue = pp.get("primary_issue", "")
                score = pp.get("avg_pain_score", "")
                line = f"- **{asin}**: {issue}"
                if score:
                    line += f" (pain: {score})"
                items.append(line)
        if items:
            parts.append("\n**Top Pain Points**\n" + "\n".join(items))

    opportunities = parsed.get("opportunities", [])
    if opportunities and isinstance(opportunities, list):
        items = []
        for opp in opportunities[:3]:
            if isinstance(opp, dict):
                desc = opp.get("description", "")
                otype = opp.get("type", "")
                impact = opp.get("estimated_impact", "")
                line = f"- {desc}"
                if otype or impact:
                    line += f" [{otype}{'/' + impact if impact else ''}]"
                items.append(line)
        if items:
            parts.append("\n**Opportunities**\n" + "\n".join(items))

    product_highlights = parsed.get("product_highlights", [])
    if product_highlights and isinstance(product_highlights, list):
        items = []
        for ph in product_highlights[:5]:
            if isinstance(ph, dict):
                name = ph.get("product_name", ph.get("asin", ""))
                complaint = ph.get("top_complaint", "")
                alt = ph.get("alternative_mentioned", "")
                line = f"- **{name}**: {complaint}"
                if alt:
                    line += f" -> {alt}"
                items.append(line)
        if items:
            parts.append("\n**Product Highlights**\n" + "\n".join(items))

    # -- Daily intelligence fields --

    insights = parsed.get("key_insights", [])
    if insights and isinstance(insights, list):
        items = []
        for ins in insights[:5]:
            if isinstance(ins, dict):
                text = ins.get("insight", "")
                conf = ins.get("confidence", "")
                domain = ins.get("domain", "")
                line = f"- {text}"
                if conf or domain:
                    tags_str = "/".join(filter(None, [domain, conf]))
                    line += f" [{tags_str}]"
                items.append(line)
            elif isinstance(ins, str):
                items.append(f"- {ins}")
        if items:
            parts.append("\n**Key Insights**\n" + "\n".join(items))

    pressure = parsed.get("pressure_readings", [])
    if pressure and isinstance(pressure, list):
        items = []
        for pr in pressure[:5]:
            if isinstance(pr, dict):
                name = pr.get("entity_name", "")
                score = pr.get("pressure_score", "")
                traj = pr.get("trajectory", "")
                note = pr.get("note", "")
                line = f"- **{name}** {score}/10"
                if traj:
                    line += f" ({traj})"
                if note:
                    line += f": {note}"
                items.append(line)
        if items:
            parts.append("\n**Pressure Readings**\n" + "\n".join(items))

    connections = parsed.get("connections_found", [])
    if connections and isinstance(connections, list):
        items = []
        for conn in connections[:3]:
            if isinstance(conn, dict):
                desc = conn.get("description", "")
                sig = conn.get("significance", "")
                line = f"- {desc}"
                if sig:
                    line += f" [{sig}]"
                items.append(line)
            elif isinstance(conn, str):
                items.append(f"- {conn}")
        if items:
            parts.append("\n**Connections**\n" + "\n".join(items))

    # -- Competitive intelligence fields --

    brand_vuln = parsed.get("brand_vulnerability", []) or parsed.get("brand_scorecards", [])
    if brand_vuln and isinstance(brand_vuln, list):
        items = []
        for sc in brand_vuln[:5]:
            if isinstance(sc, dict):
                brand = sc.get("brand", "")
                if not brand:
                    continue
                score = sc.get("vulnerability_score", sc.get("health_score", ""))
                status = sc.get("status", "")
                liner = sc.get("one_liner", "")
                line = f"- **{brand}** {score}/100 vulnerability"
                if status:
                    line += f" ({status})"
                if liner:
                    line += f": {liner}"
                items.append(line)
        if items:
            parts.append("\n**Brand Vulnerability**\n" + "\n".join(items))

    comp_flows = parsed.get("competitive_flows", [])
    if comp_flows and isinstance(comp_flows, list):
        items = []
        for flow in comp_flows[:5]:
            if isinstance(flow, dict):
                frm = flow.get("from_brand", "")
                to = flow.get("to_brand", "")
                if not frm and not to:
                    continue
                reason = flow.get("primary_reason", "")
                vol = flow.get("count", "")
                line = f"- {frm} -> {to}"
                if vol:
                    line += f" ({vol} mentions)"
                if reason:
                    line += f": {reason}"
                items.append(line)
        if items:
            parts.append("\n**Competitive Flows**\n" + "\n".join(items))

    # insights (competitive intelligence uses "insights", not "key_insights")
    ci_insights = parsed.get("insights", [])
    if ci_insights and isinstance(ci_insights, list) and not parsed.get("key_insights"):
        items = []
        for ins in ci_insights[:5]:
            if isinstance(ins, dict):
                text = ins.get("insight", "")
                impact = ins.get("impact", "")
                line = f"- {text}"
                if impact:
                    line += f" [{impact}]"
                items.append(line)
            elif isinstance(ins, str):
                items.append(f"- {ins}")
        if items:
            parts.append("\n**Insights**\n" + "\n".join(items))

    # -- Common: recommendations --

    recs = parsed.get("recommendations", [])
    if recs and isinstance(recs, list):
        items = []
        for rec in recs[:3]:
            if isinstance(rec, dict):
                action = rec.get("action", "")
                urgency = rec.get("urgency", "")
                line = f"- {action}"
                if urgency:
                    line += f" [{urgency}]"
                items.append(line)
            elif isinstance(rec, str):
                items.append(f"- {rec}")
        if items:
            parts.append("\n**Recommendations**\n" + "\n".join(items))

    if not parts:
        return fallback

    return "\n".join(parts)
