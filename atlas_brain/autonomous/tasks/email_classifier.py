"""
Rule-based email classifier for Gmail digest pre-processing.

Classifies emails into categories and priorities using a deterministic
waterfall of signals (Gmail labels, unsubscribe headers, sender domain,
subject keywords) — no LLM calls needed.

The classifier runs between email fetch and LLM synthesis so the LLM
only needs to summarize pre-classified emails.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ...config import settings

logger = logging.getLogger("atlas.autonomous.tasks.email_classifier")


@dataclass
class EmailClassification:
    """Result of rule-based email classification."""

    category: str
    priority: str
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Built-in domain → category map (overridable via data/email_domains.json)
# ---------------------------------------------------------------------------

_DEFAULT_DOMAIN_MAP: dict[str, str] = {
    # Financial
    "cashapp.com": "financial",
    "square.com": "financial",
    "chase.com": "financial",
    "bankofamerica.com": "financial",
    "paypal.com": "financial",
    "venmo.com": "financial",
    "stripe.com": "financial",
    "capitalone.com": "financial",
    "wellsfargo.com": "financial",
    "americanexpress.com": "financial",
    "discover.com": "financial",
    # Shopping / Delivery
    "amazon.com": "shopping",
    "ebay.com": "shopping",
    "ups.com": "shopping",
    "fedex.com": "shopping",
    "usps.com": "shopping",
    "walmart.com": "shopping",
    "target.com": "shopping",
    # Travel
    "amtrak.com": "travel",
    "united.com": "travel",
    "delta.com": "travel",
    "southwest.com": "travel",
    "uber.com": "travel",
    "lyft.com": "travel",
    "airbnb.com": "travel",
    # Automated / Dev
    "github.com": "automated",
    "gitlab.com": "automated",
    "jenkins.io": "automated",
    "google.com": "automated",
    "noreply.google.com": "automated",
    # Social
    "facebookmail.com": "social",
    "linkedin.com": "social",
    "twitter.com": "social",
    "instagram.com": "social",
    "x.com": "social",
}

# ---------------------------------------------------------------------------
# Subject keyword patterns → (category, optional priority override)
# ---------------------------------------------------------------------------

_SUBJECT_PATTERNS: list[tuple[str, str, Optional[str]]] = [
    # (regex, category, priority_override_or_None)
    (r"password\s*(reset|changed)|verification\s*code|login\s*alert|2fa|two.factor", "security", "action_required"),
    (r"payment\s*due|invoice|statement|bill\s*is\s*ready|amount\s*due", "financial", "action_required"),
    (r"shipped|delivered|tracking|order.*confirm|your\s*order", "shopping", None),
    (r"boarding\s*pass|itinerary|flight|reservation.*confirm|check.in", "travel", None),
    (r"event\s*invite|rsvp|calendar|meeting\s*(invite|request)", "calendar", None),
]


# ---------------------------------------------------------------------------
# Gmail label → category map
# ---------------------------------------------------------------------------

_GMAIL_LABEL_MAP: dict[str, tuple[str, float]] = {
    "CATEGORY_PROMOTIONS": ("promotion", 0.95),
    "CATEGORY_SOCIAL": ("social", 0.95),
    "CATEGORY_FORUMS": ("newsletter", 0.85),
    "CATEGORY_UPDATES": ("automated", 0.60),
    "CATEGORY_PERSONAL": ("personal", 0.60),
}


def _extract_domain(from_header: str) -> str:
    """Extract the sender domain from a From header value."""
    # "Name <user@domain.com>" or "user@domain.com"
    match = re.search(r"@([\w.-]+)", from_header)
    return match.group(1).lower() if match else ""


def _lookup_domain(domain: str, domain_map: dict[str, str]) -> str | None:
    """
    Look up a domain in the map, falling back to the parent domain.

    e.g. "mail.cashapp.com" → try "mail.cashapp.com" first,
    then "cashapp.com".  Handles up to one level of subdomain stripping.
    """
    if not domain:
        return None
    result = domain_map.get(domain)
    if result:
        return result
    # Try parent domain (strip first subdomain segment)
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[1:])
        return domain_map.get(parent)
    return None


class EmailRuleClassifier:
    """
    Waterfall rule-based email classifier.

    Layers (checked in order, first match wins):
      1. Gmail category labels (CATEGORY_PROMOTIONS, etc.)
      2. List-Unsubscribe header → newsletter (default)
      3. Sender domain map
      4. Subject keyword regex patterns
      5. Default by remaining Gmail labels
      6. Fallback → "other"
    """

    def __init__(self, domain_map_file: Optional[str] = None) -> None:
        self._domain_map = dict(_DEFAULT_DOMAIN_MAP)

        # Load optional custom domain overrides
        map_path = domain_map_file or settings.tools.gmail_domain_map_file
        self._load_domain_overrides(map_path)

        # Pre-compile subject patterns
        self._subject_patterns: list[tuple[re.Pattern, str, Optional[str]]] = [
            (re.compile(pat, re.IGNORECASE), cat, pri)
            for pat, cat, pri in _SUBJECT_PATTERNS
        ]

    def _load_domain_overrides(self, path: str) -> None:
        """Merge user-provided domain→category overrides from JSON file."""
        try:
            p = Path(path)
            if p.exists():
                with p.open() as f:
                    overrides = json.load(f)
                if isinstance(overrides, dict):
                    self._domain_map.update(overrides)
                    logger.info("Loaded %d domain overrides from %s", len(overrides), path)
        except Exception as e:
            logger.warning("Could not load domain map from %s: %s", path, e)

    # ------------------------------------------------------------------
    # Priority assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_priority(
        category: str,
        subject: str,
        has_unsubscribe: bool,
        label_ids: list[str],
    ) -> str:
        """Assign priority based on category and signals."""
        subj_lower = subject.lower()

        # Action required: financial due dates, explicit action words,
        # or personal messages from real people
        action_keywords = [
            "due", "action required", "action needed", "reply needed",
            "fill out", "confirm your", "verify your", "respond by",
        ]
        if any(kw in subj_lower for kw in action_keywords):
            return "action_required"

        # Personal email from a real person (no unsubscribe, personal label)
        if (
            "CATEGORY_PERSONAL" in label_ids
            and not has_unsubscribe
            and category == "personal"
        ):
            return "action_required"

        # Low priority: bulk/automated mail
        if has_unsubscribe or category in (
            "promotion", "newsletter", "social", "automated",
        ):
            return "low_priority"

        # Everything else
        return "fyi"

    # ------------------------------------------------------------------
    # Single email classification
    # ------------------------------------------------------------------

    def classify(self, email: dict[str, Any]) -> EmailClassification:
        """
        Classify a single email dict through the waterfall.

        Expected keys: label_ids, has_unsubscribe, from, subject.
        """
        label_ids: list[str] = email.get("label_ids", [])
        has_unsubscribe: bool = email.get("has_unsubscribe", False)
        sender: str = email.get("from", "")
        subject: str = email.get("subject", "")

        # --- Layer 1: Gmail category labels ---
        for label in label_ids:
            if label in _GMAIL_LABEL_MAP:
                cat, conf = _GMAIL_LABEL_MAP[label]
                # Promotions/Social are high-confidence terminal matches
                if conf >= 0.90:
                    pri = self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                    return EmailClassification(
                        category=cat,
                        priority=pri,
                        confidence=conf,
                        reason=f"Gmail label {label}",
                    )

        # --- Layer 2: Unsubscribe header ---
        if has_unsubscribe:
            # Default to newsletter, but domain rules can override below
            domain = _extract_domain(sender)
            domain_cat = _lookup_domain(domain, self._domain_map)
            if domain_cat:
                # e.g. cashapp.com with unsubscribe → still "financial"
                pri = self._assign_priority(domain_cat, subject, has_unsubscribe, label_ids)
                return EmailClassification(
                    category=domain_cat,
                    priority=pri,
                    confidence=0.90,
                    reason=f"Unsubscribe + domain {domain} → {domain_cat}",
                )
            pri = self._assign_priority("newsletter", subject, has_unsubscribe, label_ids)
            return EmailClassification(
                category="newsletter",
                priority=pri,
                confidence=0.90,
                reason="List-Unsubscribe header present",
            )

        # --- Layer 3: Sender domain ---
        domain = _extract_domain(sender)
        domain_cat = _lookup_domain(domain, self._domain_map)
        if domain_cat:
            pri = self._assign_priority(domain_cat, subject, has_unsubscribe, label_ids)
            return EmailClassification(
                category=domain_cat,
                priority=pri,
                confidence=0.85,
                reason=f"Sender domain {domain}",
            )

        # --- Layer 4: Subject keyword patterns ---
        for pattern, cat, pri_override in self._subject_patterns:
            if pattern.search(subject):
                pri = pri_override or self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                return EmailClassification(
                    category=cat,
                    priority=pri,
                    confidence=0.75,
                    reason=f"Subject keyword match: {pattern.pattern[:40]}",
                )

        # --- Layer 5: Default by remaining Gmail labels ---
        for label in label_ids:
            if label in _GMAIL_LABEL_MAP:
                cat, conf = _GMAIL_LABEL_MAP[label]
                pri = self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                return EmailClassification(
                    category=cat,
                    priority=pri,
                    confidence=conf,
                    reason=f"Fallback Gmail label {label}",
                )

        # --- Layer 6: Fallback ---
        return EmailClassification(
            category="other",
            priority="fyi",
            confidence=0.50,
            reason="No classification signals matched",
        )

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_batch(self, emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Classify all emails in-place, injecting 'category' and 'priority' keys.

        Returns the same list (mutated) for chaining convenience.
        """
        for email in emails:
            result = self.classify(email)
            email["category"] = result.category
            email["priority"] = result.priority
            email["_classify_confidence"] = result.confidence
            email["_classify_reason"] = result.reason

        # Log summary
        if emails:
            cats = {}
            for e in emails:
                c = e["category"]
                cats[c] = cats.get(c, 0) + 1
            logger.info(
                "Classified %d emails: %s",
                len(emails),
                ", ".join(f"{k}={v}" for k, v in sorted(cats.items())),
            )

        return emails


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_classifier: Optional[EmailRuleClassifier] = None


def get_email_classifier() -> EmailRuleClassifier:
    """Get or create the module-level EmailRuleClassifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = EmailRuleClassifier()
    return _classifier
