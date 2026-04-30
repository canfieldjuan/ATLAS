from __future__ import annotations

import re


REPAIR_NEGATIVE_PATTERNS = (
    "cancel", "cancellation", "billing dispute", "refund denied", "runaround",
    "automatic renewal", "auto renew", "renewed without notice", "charged",
    "invoiced", "price increase", "overcharged", "switching cost",
)
REPAIR_COMPETITOR_PATTERNS = (
    "switched to", "moved to", "replaced with", "evaluating", "looking at",
    "considering", "shortlisting", "shortlisted", "poc with", "proof of concept with",
)
REPAIR_PRICING_PATTERNS = (
    "billing", "invoice", "invoiced", "charged", "refund", "renewal",
    "price increase", "cost increase", "automatic renewal", "auto renew",
    "overcharged",
)
REPAIR_RECOMMEND_PATTERNS = (
    "would not recommend", "wouldn't recommend", "do not recommend",
    "don't recommend", "stay away", "avoid", "not worth", "cannot recommend",
)
REPAIR_FEATURE_GAP_PATTERNS = (
    "missing", "lacks", "lacking", "wish it had", "wish they had",
    "need better", "needs better", "needs more", "could use", "limited",
)
REPAIR_TIMELINE_PATTERNS = (
    "renewal", "contract end", "contract expires", "deadline", "next quarter",
    "q1", "q2", "q3", "q4", "30 days", "60 days", "90 days",
)
REPAIR_CATEGORY_SHIFT_PATTERNS = (
    "async", "docs", "documentation", "notion", "confluence", "bundle",
    "workspace", "microsoft 365", "google workspace", "internal tool",
    "homegrown", "home-grown", "custom tool",
)
REPAIR_CURRENCY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?", re.I)
