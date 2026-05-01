from __future__ import annotations

import re
from typing import Any


def build_pain_patterns(
    keywords: dict[str, tuple[str, ...]],
) -> dict[str, re.Pattern[str]]:
    compiled: dict[str, re.Pattern[str]] = {}
    for category, needles in keywords.items():
        parts = [r"\b" + re.escape(n) + r"\b" for n in needles]
        compiled[category] = re.compile("|".join(parts), re.IGNORECASE)
    return compiled


PAIN_KEYWORDS_RAW = {
    "pricing": (
        "price", "pricing", "cost", "costly", "expensive", "overpriced", "renewal",
        "invoice", "invoiced", "billing", "billed", "charged", "charge", "overcharge",
        "fee", "fees", "refund", "cost increase", "price increase",
    ),
    "support": ("support", "ticket", "response", "customer service", "escalation", "escalated", "escalate"),
    "features": ("feature", "functionality", "capability", "missing"),
    "ux": ("ui", "ux", "interface", "clunky", "usability", "navigation"),
    "reliability": ("outage", "downtime", "crash", "bug", "unstable", "reliable"),
    "performance": ("slow", "latency", "lag", "performance", "speed"),
    "integration": ("integration", "integrate", "sync", "connector", "api"),
    "security": ("security", "permission", "access control", "compliance", "sso", "mfa"),
    "onboarding": ("onboarding", "setup", "implementation", "training", "adoption"),
    "technical_debt": ("technical debt", "legacy", "outdated", "deprecated", "workaround"),
    "contract_lock_in": (
        "lock-in", "locked in", "vendor lock", "multi-year", "exit fee", "cancel",
        "cancellation", "terminate", "termination", "auto renew", "automatic renewal",
        "renewed without notice", "notice period", "contract term", "contract trap",
        "billing dispute", "runaround",
    ),
    "data_migration": ("migration", "migrate", "import", "export", "data transfer"),
    "api_limitations": ("api", "webhook", "sdk", "rate limit", "endpoint"),
    "privacy": ("spam", "unsubscribe", "unsolicited", "data breach", "privacy violation"),
}

PAIN_KEYWORDS = PAIN_KEYWORDS_RAW
PAIN_PATTERNS = build_pain_patterns(PAIN_KEYWORDS_RAW)

PAIN_DERIVATION_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
)

COMPETITOR_RECOVERY_PATTERNS = (
    r"\b(?:switched to|moved to|replaced with|migrating to|migration to)\s+([A-Z][A-Za-z0-9.&+/\-]*(?:\s+[A-Z][A-Za-z0-9.&+/\-]*){0,3})",
    r"\b(?:evaluating|looking at|considering|shortlisting|shortlisted|poc with|proof of concept with)\s+([A-Z][A-Za-z0-9.&+/\-]*(?:\s+[A-Z][A-Za-z0-9.&+/\-]*){0,3})",
)

COMPETITOR_RECOVERY_BLOCKLIST = {
    "a", "an", "the", "another tool", "another vendor", "other tool", "other vendor",
    "new tool", "new vendor", "options", "alternative", "alternatives",
    "alternative platform", "alternative platforms", "crm",
    "provider", "providers", "competing provider", "competing providers",
}

GENERIC_COMPETITOR_TOKENS = {
    "alternative", "alternatives", "platform", "platforms", "tool", "tools",
    "vendor", "vendors", "software", "solutions", "solution", "service",
    "services", "system", "systems", "crm", "suite", "app", "apps",
    "provider", "providers", "competing",
}

COMPETITOR_CONTEXT_PATTERNS = (
    "switched to", "moved to", "replaced with", "migrating to", "migration to",
    "evaluating", "looking at", "considering", "shortlisting", "shortlisted",
    "poc with", "proof of concept with", "instead of", "compared to", "versus", " vs ",
)

KNOWN_PAIN_CATEGORIES = {
    "pricing", "features", "reliability", "support", "integration",
    "performance", "security", "ux", "onboarding", "overall_dissatisfaction",
    "technical_debt", "contract_lock_in", "data_migration", "api_limitations",
    "outcome_gap", "admin_burden", "ai_hallucination", "integration_debt",
    "privacy",
}

LEGACY_GENERIC_PAIN_CATEGORIES = {"other", "general_dissatisfaction"}


def normalize_pain_category(category: Any) -> str:
    raw = str(category or "").strip().lower()
    if not raw:
        return "overall_dissatisfaction"
    if raw in LEGACY_GENERIC_PAIN_CATEGORIES:
        return "overall_dissatisfaction"
    if raw in KNOWN_PAIN_CATEGORIES:
        return raw
    return "overall_dissatisfaction"
