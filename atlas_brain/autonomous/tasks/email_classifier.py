"""
Rule-based email classifier for Gmail digest pre-processing.

Classifies emails into categories and priorities using a deterministic
waterfall of signals (Gmail labels, unsubscribe headers, sender domain,
subject keywords) -- no LLM calls needed.

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
    replyable: bool | None  # None = ambiguous, needs LLM triage


# ---------------------------------------------------------------------------
# Built-in domain -> category map (overridable via data/email_domains.json)
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
# Subject keyword patterns -> (category, optional priority override)
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
# Gmail label -> category map
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

    e.g. "mail.cashapp.com" -> try "mail.cashapp.com" first,
    then "cashapp.com". Handles up to one level of subdomain stripping.
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
      2. List-Unsubscribe header -> newsletter (default)
      3. Sender domain map
      4. Subject keyword regex patterns
      5. Default by remaining Gmail labels
      6. Fallback -> "other"
    """

    # Sender patterns that indicate automated/no-reply addresses
    _NOREPLY_LOCAL_PATTERNS = (
        "noreply", "no-reply", "donotreply", "do-not-reply",
        "mailer-daemon", "postmaster",
    )
    _NOREPLY_PREFIX_PATTERNS = (
        "notifications@", "notification@", "alerts@", "alert@",
        "news@", "info@", "updates@", "marketing@",
    )

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
        """Merge user-provided domain->category overrides from JSON file."""
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
    # Replyable assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _is_noreply_sender(sender: str) -> bool:
        """Check if the sender address matches known no-reply patterns."""
        email_part = sender.lower()
        if "<" in email_part:
            email_part = email_part.split("<")[1].split(">")[0]
        local = email_part.split("@")[0] if "@" in email_part else email_part
        if local in EmailRuleClassifier._NOREPLY_LOCAL_PATTERNS:
            return True
        return any(email_part.startswith(p) for p in EmailRuleClassifier._NOREPLY_PREFIX_PATTERNS)

    @staticmethod
    def _assign_replyable(
        category: str,
        priority: str,
        has_unsubscribe: bool,
        sender: str,
    ) -> bool | None:
        """Determine if an email expects a human reply.

        Returns:
            False  - definitively not replyable (bulk, automated, noreply)
            True   - definitively replyable (personal + action_required)
            None   - ambiguous, needs LLM triage
        """
        if has_unsubscribe:
            return False
        if EmailRuleClassifier._is_noreply_sender(sender):
            return False
        if category in ("automated", "promotion", "newsletter", "social", "shopping"):
            return False
        if category == "personal" and priority == "action_required":
            return True
        return None

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

        cat: str | None = None
        pri: str | None = None
        conf: float = 0.50
        reason: str = ""

        # --- Layer 1: Gmail category labels ---
        for label in label_ids:
            if label in _GMAIL_LABEL_MAP:
                _cat, _conf = _GMAIL_LABEL_MAP[label]
                if _conf >= 0.90:
                    cat, conf = _cat, _conf
                    pri = self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                    reason = f"Gmail label {label}"
                    break

        # --- Layer 2: Unsubscribe header ---
        if cat is None and has_unsubscribe:
            domain = _extract_domain(sender)
            domain_cat = _lookup_domain(domain, self._domain_map)
            if domain_cat:
                cat, conf = domain_cat, 0.90
                pri = self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                reason = f"Unsubscribe + domain {domain} -> {cat}"
            else:
                cat, conf = "newsletter", 0.90
                pri = self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                reason = "List-Unsubscribe header present"

        # --- Layer 3: Sender domain ---
        if cat is None:
            domain = _extract_domain(sender)
            domain_cat = _lookup_domain(domain, self._domain_map)
            if domain_cat:
                cat, conf = domain_cat, 0.85
                pri = self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                reason = f"Sender domain {domain}"

        # --- Layer 4: Subject keyword patterns ---
        if cat is None:
            for pattern, _cat, pri_override in self._subject_patterns:
                if pattern.search(subject):
                    cat, conf = _cat, 0.75
                    pri = pri_override or self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                    reason = f"Subject keyword match: {pattern.pattern[:40]}"
                    break

        # --- Layer 5: Default by remaining Gmail labels ---
        if cat is None:
            for label in label_ids:
                if label in _GMAIL_LABEL_MAP:
                    _cat, _conf = _GMAIL_LABEL_MAP[label]
                    cat, conf = _cat, _conf
                    pri = self._assign_priority(cat, subject, has_unsubscribe, label_ids)
                    reason = f"Fallback Gmail label {label}"
                    break

        # --- Layer 6: Fallback ---
        if cat is None:
            cat, pri, conf = "other", "fyi", 0.50
            reason = "No classification signals matched"

        # --- Replyable determination ---
        replyable = self._assign_replyable(cat, pri, has_unsubscribe, sender)

        return EmailClassification(
            category=cat,
            priority=pri,
            confidence=conf,
            reason=reason,
            replyable=replyable,
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
            email["replyable"] = result.replyable
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
