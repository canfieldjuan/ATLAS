"""Periodic cross-domain pattern detection and proactive reasoning.

Runs on a cron schedule (default 4x daily) to identify patterns that
the reactive pipeline might miss.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("atlas.reasoning.reflection")


async def run_reflection() -> dict[str, Any]:
    """Execute a full reflection cycle.

    1. Run rule-based pattern detectors
    2. Feed patterns + recent events to Claude for analysis
    3. Auto-execute high-confidence recommendations
    4. Notify owner for lower-confidence findings
    """
    from .patterns import run_all_pattern_detectors

    # 1. Rule-based pattern detection
    findings = await run_all_pattern_detectors()
    logger.info("Reflection: %d rule-based findings", len(findings))

    if not findings:
        return {"findings": 0, "actions": 0, "notifications": 0}

    # 2. LLM analysis of findings
    from .prompts import REFLECTION_SYSTEM
    from ..services.llm_router import get_llm
    from ..config import settings

    llm = get_llm("reasoning")
    if not llm:
        # No LLM available -- just notify on all findings
        await _notify_findings(findings)
        return {"findings": len(findings), "actions": 0, "notifications": len(findings)}

    prompt = (
        "Here are patterns detected across Atlas's domains:\n\n"
        + json.dumps(findings, default=str, indent=2)
    )

    try:
        from .graph import _llm_generate
        text = await _llm_generate(
            llm, prompt, REFLECTION_SYSTEM,
            max_tokens=settings.reasoning.max_tokens,
            temperature=settings.reasoning.temperature,
        )
        analyzed = json.loads(text)
        llm_findings = analyzed.get("findings", [])
    except Exception:
        logger.warning("Reflection LLM analysis failed", exc_info=True)
        llm_findings = []

    # 3. Execute high-confidence actions
    actions_taken = 0
    notifications = 0

    for finding in llm_findings:
        confidence = finding.get("confidence", 0)
        action = finding.get("recommended_action")

        if confidence >= 0.8 and action:
            try:
                await _execute_reflection_action(finding)
                actions_taken += 1
            except Exception:
                logger.warning(
                    "Reflection action failed: %s", action, exc_info=True
                )

    # 4. Notify on remaining findings
    notify_findings = [
        f for f in (llm_findings or findings)
        if f.get("confidence", 0) < 0.8 or not f.get("recommended_action")
    ]
    if notify_findings:
        await _notify_findings(notify_findings)
        notifications = len(notify_findings)

    return {
        "findings": len(findings),
        "llm_findings": len(llm_findings),
        "actions": actions_taken,
        "notifications": notifications,
    }


async def _execute_reflection_action(finding: dict[str, Any]) -> None:
    """Execute a single reflection action."""
    action = finding.get("recommended_action", "")

    if action == "generate_draft":
        # Generate a follow-up draft for stale threads
        from ..api.email_drafts import generate_draft
        draft_id = finding.get("params", {}).get("imap_uid")
        if draft_id:
            await generate_draft(draft_id)
            logger.info("Reflection: generated follow-up draft for %s", draft_id)

    elif action == "send_notification":
        message = finding.get("description", "Reflection finding")
        await _send_ntfy(f"Proactive: {message}")

    else:
        logger.debug("Reflection: unknown action %s", action)


async def _notify_findings(findings: list[dict[str, Any]]) -> None:
    """Send a summary notification of reflection findings."""
    if not findings:
        return

    lines = []
    for f in findings[:5]:  # cap at 5 for readability
        desc = f.get("description", f.get("pattern", "Unknown"))
        lines.append(f"- {desc}")

    message = f"Atlas found {len(findings)} pattern(s):\n" + "\n".join(lines)
    await _send_ntfy(message)


async def _send_ntfy(message: str) -> None:
    """Send notification via ntfy."""
    from ..config import settings

    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    url = f"{settings.alerts.ntfy_url}/{settings.alerts.ntfy_topic}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                url,
                content=message.encode("utf-8"),
                headers={
                    "Title": "Atlas Reflection",
                    "Priority": "low",
                    "Tags": "crystal_ball",
                },
            )
    except Exception:
        logger.debug("Reflection ntfy failed", exc_info=True)
