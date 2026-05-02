from __future__ import annotations

import re


_VENDOR_ALIASES = {
    "aws": "Amazon Web Services",
    "amazonwebservices": "Amazon Web Services",
    "gcp": "Google Cloud",
    "googlecloud": "Google Cloud",
    "googlecloudplatform": "Google Cloud",
    "azure": "Microsoft Azure",
    "msazure": "Microsoft Azure",
    "microsoftazure": "Microsoft Azure",
    "sfdc": "Salesforce",
    "salesforcecom": "Salesforce",
    "salesforce": "Salesforce",
    "hubspot": "HubSpot",
    "clickup": "ClickUp",
    "notion": "Notion",
    "slack": "Slack",
    "zendesk": "Zendesk",
}


def _vendor_key(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def _title_vendor(value: str) -> str:
    words = re.split(r"(\s+)", value.strip())
    return "".join(
        item if item.isspace() else item[:1].upper() + item[1:].lower()
        for item in words
    )


def resolve_vendor_name_cached(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return _VENDOR_ALIASES.get(_vendor_key(raw), _title_vendor(raw))


async def resolve_vendor_name(value: str | None) -> str:
    """Async-compatible resolver used by copied Atlas campaign tasks."""
    return resolve_vendor_name_cached(value)
