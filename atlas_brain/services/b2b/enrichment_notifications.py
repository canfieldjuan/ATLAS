from __future__ import annotations

from typing import Any


async def notify_high_urgency(
    *,
    alerts: Any,
    logger: Any,
    vendor_name: str,
    reviewer_company: str,
    urgency: float,
    pain_category: str,
    intent_to_leave: bool,
) -> None:
    if not alerts.ntfy_enabled:
        return

    import httpx

    url = f"{alerts.ntfy_url.rstrip('/')}/{alerts.ntfy_topic}"
    company_part = f" at {reviewer_company}" if reviewer_company else ""
    intent_part = " | Intent to leave" if intent_to_leave else ""
    pain_part = f" | Pain: {pain_category}" if pain_category else ""

    message = (
        f"Urgency {urgency:.0f}/10{company_part}\n"
        f"Vendor: {vendor_name}{pain_part}{intent_part}"
    )

    headers: dict[str, str] = {
        "Title": f"High-Urgency Signal: {vendor_name}",
        "Priority": "high",
        "Tags": "rotating_light,b2b,churn",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("ntfy high-urgency alert sent for %s (urgency=%s)", vendor_name, urgency)
    except Exception as exc:
        logger.warning("ntfy high-urgency alert failed for %s: %s", vendor_name, exc)
