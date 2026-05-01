from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..campaign_ports import VisibilitySink


logger = logging.getLogger("extracted_content_pipeline.pipelines.notify")

_visibility_sink: VisibilitySink | None = None


@dataclass(frozen=True)
class PipelineNotification:
    """Host-visible notification event emitted by extracted pipeline tasks."""

    message: str
    title: str
    task_name: str | None = None
    task_id: str | None = None
    priority: str = "default"
    tags: str = "brain"
    markdown: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "title": self.title,
            "task_name": self.task_name,
            "task_id": self.task_id,
            "priority": self.priority,
            "tags": self.tags,
            "markdown": self.markdown,
            "metadata": dict(self.metadata),
        }


def configure_pipeline_notification_sink(
    sink: VisibilitySink | None,
) -> VisibilitySink | None:
    """Set the process-wide notification sink and return the previous sink."""

    global _visibility_sink
    previous = _visibility_sink
    _visibility_sink = sink
    return previous


def get_pipeline_notification_sink() -> VisibilitySink | None:
    """Return the currently configured notification sink, if any."""

    return _visibility_sink


async def send_pipeline_notification(
    message: str,
    task: Any,
    *,
    title: str | None = None,
    default_tags: str = "brain",
    max_chars: int = 4000,
    parsed: Mapping[str, Any] | None = None,
    visibility: VisibilitySink | None = None,
) -> None:
    """Emit a host-visible pipeline notification.

    The extracted package does not own ntfy, Slack, email, or dashboard
    delivery. A host app can configure a ``VisibilitySink`` and route this event
    wherever it wants. With no sink configured, notifications remain a safe
    no-op so copied Atlas task modules can run standalone.
    """

    metadata = _task_metadata(task)
    if metadata.get("notify") is False:
        return

    sink = visibility or _visibility_sink
    if sink is None:
        return

    task_name = _task_name(task)
    notification = PipelineNotification(
        message=_truncate(
            _format_parsed(dict(parsed), message) if parsed else str(message or ""),
            max_chars=max_chars,
        ),
        title=title or _default_title(task_name),
        task_name=task_name,
        task_id=_task_id(task),
        priority=str(metadata.get("notify_priority") or "default"),
        tags=str(metadata.get("notify_tags") or default_tags),
        metadata={
            "source": "extracted_content_pipeline",
            "parsed": bool(parsed),
        },
    )

    try:
        await sink.emit("pipeline_notification", notification.as_payload())
    except Exception:
        logger.warning(
            "Failed to emit pipeline notification for task %r",
            task_name,
            exc_info=True,
        )


def _task_metadata(task: Any) -> dict[str, Any]:
    if isinstance(task, Mapping):
        raw = task.get("metadata")
    else:
        raw = getattr(task, "metadata", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _task_name(task: Any) -> str | None:
    if isinstance(task, Mapping):
        value = task.get("name")
    else:
        value = getattr(task, "name", None)
    text = str(value or "").strip()
    return text or None


def _task_id(task: Any) -> str | None:
    if isinstance(task, Mapping):
        value = task.get("id") or task.get("task_id")
    else:
        value = getattr(task, "id", None) or getattr(task, "task_id", None)
    text = str(value or "").strip()
    return text or None


def _default_title(task_name: str | None) -> str:
    if not task_name:
        return "Atlas: Pipeline Notification"
    return f"Atlas: {task_name.replace('_', ' ').title()}"


def _truncate(message: str, *, max_chars: int) -> str:
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        limit = 4000
    if limit <= 0:
        return ""
    return message[:limit]


def _format_parsed(parsed: dict[str, Any], fallback: str) -> str:
    """Build concise markdown from common structured pipeline payload fields."""

    parts: list[str] = []

    analysis = str(parsed.get("analysis_text") or "").strip()
    if analysis:
        parts.append(analysis)

    _append_top_pain_points(parts, parsed.get("top_pain_points"))
    _append_opportunities(parts, parsed.get("opportunities"))
    _append_product_highlights(parts, parsed.get("product_highlights"))
    _append_key_insights(parts, parsed.get("key_insights"))
    _append_pressure_readings(parts, parsed.get("pressure_readings"))
    _append_connections(parts, parsed.get("connections_found"))
    _append_brand_vulnerability(
        parts,
        parsed.get("brand_vulnerability") or parsed.get("brand_scorecards"),
    )
    _append_competitive_flows(parts, parsed.get("competitive_flows"))
    if not parsed.get("key_insights"):
        _append_generic_insights(parts, parsed.get("insights"))
    _append_recommendations(parts, parsed.get("recommendations"))

    return "\n".join(parts) if parts else str(fallback or "")


def _append_section(parts: list[str], title: str, items: list[str]) -> None:
    if items:
        parts.append(f"\n**{title}**\n" + "\n".join(items))


def _append_top_pain_points(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:5]:
        if not isinstance(item, Mapping):
            continue
        asin = str(item.get("asin") or "").strip()
        issue = str(item.get("primary_issue") or "").strip()
        score = str(item.get("avg_pain_score") or "").strip()
        line = f"- **{asin}**: {issue}" if asin else f"- {issue}"
        if score:
            line += f" (pain: {score})"
        if issue:
            items.append(line)
    _append_section(parts, "Top Pain Points", items)


def _append_opportunities(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:3]:
        if not isinstance(item, Mapping):
            continue
        desc = str(item.get("description") or "").strip()
        if not desc:
            continue
        otype = str(item.get("type") or "").strip()
        impact = str(item.get("estimated_impact") or "").strip()
        line = f"- {desc}"
        if otype or impact:
            line += f" [{otype}{'/' + impact if impact else ''}]"
        items.append(line)
    _append_section(parts, "Opportunities", items)


def _append_product_highlights(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:5]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("product_name") or item.get("asin") or "").strip()
        complaint = str(item.get("top_complaint") or "").strip()
        alt = str(item.get("alternative_mentioned") or "").strip()
        if not name and not complaint:
            continue
        line = f"- **{name}**: {complaint}" if name else f"- {complaint}"
        if alt:
            line += f" -> {alt}"
        items.append(line)
    _append_section(parts, "Product Highlights", items)


def _append_key_insights(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items = [_insight_line(item) for item in values[:5]]
    _append_section(parts, "Key Insights", [item for item in items if item])


def _append_pressure_readings(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:5]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("entity_name") or "").strip()
        score = str(item.get("pressure_score") or "").strip()
        traj = str(item.get("trajectory") or "").strip()
        note = str(item.get("note") or "").strip()
        line = f"- **{name}** {score}/10" if name else f"- {score}/10"
        if traj:
            line += f" ({traj})"
        if note:
            line += f": {note}"
        items.append(line)
    _append_section(parts, "Pressure Readings", items)


def _append_connections(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:3]:
        if isinstance(item, str) and item.strip():
            items.append(f"- {item.strip()}")
        elif isinstance(item, Mapping):
            desc = str(item.get("description") or "").strip()
            sig = str(item.get("significance") or "").strip()
            if desc:
                items.append(f"- {desc}{f' [{sig}]' if sig else ''}")
    _append_section(parts, "Connections", items)


def _append_brand_vulnerability(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:5]:
        if not isinstance(item, Mapping):
            continue
        brand = str(item.get("brand") or "").strip()
        if not brand:
            continue
        score = str(item.get("vulnerability_score") or item.get("health_score") or "").strip()
        status = str(item.get("status") or "").strip()
        liner = str(item.get("one_liner") or "").strip()
        line = f"- **{brand}** {score}/100 vulnerability"
        if status:
            line += f" ({status})"
        if liner:
            line += f": {liner}"
        items.append(line)
    _append_section(parts, "Brand Vulnerability", items)


def _append_competitive_flows(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:5]:
        if not isinstance(item, Mapping):
            continue
        frm = str(item.get("from_brand") or "").strip()
        to = str(item.get("to_brand") or "").strip()
        if not frm and not to:
            continue
        reason = str(item.get("primary_reason") or "").strip()
        count = str(item.get("count") or "").strip()
        line = f"- {frm} -> {to}"
        if count:
            line += f" ({count} mentions)"
        if reason:
            line += f": {reason}"
        items.append(line)
    _append_section(parts, "Competitive Flows", items)


def _append_generic_insights(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items = [_insight_line(item) for item in values[:5]]
    _append_section(parts, "Insights", [item for item in items if item])


def _append_recommendations(parts: list[str], values: Any) -> None:
    if not isinstance(values, list):
        return
    items: list[str] = []
    for item in values[:3]:
        if isinstance(item, str) and item.strip():
            items.append(f"- {item.strip()}")
        elif isinstance(item, Mapping):
            action = str(item.get("action") or "").strip()
            urgency = str(item.get("urgency") or "").strip()
            if action:
                items.append(f"- {action}{f' [{urgency}]' if urgency else ''}")
    _append_section(parts, "Recommendations", items)


def _insight_line(item: Any) -> str | None:
    if isinstance(item, str):
        text = item.strip()
        return f"- {text}" if text else None
    if not isinstance(item, Mapping):
        return None
    text = str(item.get("insight") or "").strip()
    if not text:
        return None
    confidence = str(item.get("confidence") or "").strip()
    domain = str(item.get("domain") or "").strip()
    impact = str(item.get("impact") or "").strip()
    tags = "/".join(value for value in (domain, confidence or impact) if value)
    return f"- {text}{f' [{tags}]' if tags else ''}"
