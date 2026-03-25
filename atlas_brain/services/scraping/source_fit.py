"""Source-fit policy for B2B scrape scheduling.

Maps product categories into broad verticals and classifies each current
scrape source as core, conditional, or avoid for that vertical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SourceFit(str, Enum):
    core = "core"
    conditional = "conditional"
    avoid = "avoid"


class ScrapeVertical(str, Enum):
    crm_support_marketing = "crm_support_marketing"
    communication = "communication"
    cloud_devops_security = "cloud_devops_security"
    project_collaboration = "project_collaboration"
    data_analytics = "data_analytics"
    ecommerce_retail = "ecommerce_retail"
    hr_hcm = "hr_hcm"
    finance_erp_billing = "finance_erp_billing"
    general_b2b = "general_b2b"


@dataclass(frozen=True)
class SourceFitDecision:
    source: str
    product_category: str | None
    vertical: str | None
    fit: str
    reason: str


_CATEGORY_RULES: tuple[tuple[ScrapeVertical, tuple[re.Pattern[str], ...]], ...] = (
    (
        ScrapeVertical.communication,
        (
            re.compile(r"communication|team\s+chat|messaging|video\s+conferencing", re.I),
            re.compile(r"ucaas|voip|contact\s+center|call\s+center", re.I),
            re.compile(r"business\s+phone|meeting\s+platform", re.I),
        ),
    ),
    (
        ScrapeVertical.crm_support_marketing,
        (
            re.compile(r"\bcrm\b", re.I),
            re.compile(r"help\s*desk|helpdesk|customer\s+support|customer\s+service", re.I),
            re.compile(r"marketing|email\s+marketing|sales\s+engagement|live\s+chat", re.I),
            re.compile(r"customer\s+success|contact\s+center", re.I),
        ),
    ),
    (
        ScrapeVertical.cloud_devops_security,
        (
            re.compile(r"cloud|infrastructure|devops|observability|monitoring|logging", re.I),
            re.compile(r"security|identity|endpoint|backup|kubernetes|container", re.I),
            re.compile(r"database|developer|api\s+management|incident|sre", re.I),
        ),
    ),
    (
        ScrapeVertical.project_collaboration,
        (
            re.compile(r"project\s+management|collaboration|workflow", re.I),
            re.compile(r"knowledge\s+management|wiki|documentation|notetaking|notes", re.I),
            re.compile(r"itsm|it\s+service\s+management|service\s+desk", re.I),
        ),
    ),
    (
        ScrapeVertical.data_analytics,
        (
            re.compile(r"analytics|business\s+intelligence|\bbi\b|dashboard|reporting", re.I),
            re.compile(r"data\s+warehouse|etl|elt|data\s+pipeline|data\s+integration", re.I),
        ),
    ),
    (
        ScrapeVertical.ecommerce_retail,
        (
            re.compile(r"e-?commerce|storefront|shopping\s+cart|retail", re.I),
            re.compile(r"order\s+management|merchant", re.I),
        ),
    ),
    (
        ScrapeVertical.hr_hcm,
        (
            re.compile(r"\bhr\b|human\s+resources|hcm|payroll|benefits", re.I),
            re.compile(r"recruiting|applicant\s+tracking|\bats\b|talent", re.I),
        ),
    ),
    (
        ScrapeVertical.finance_erp_billing,
        (
            re.compile(r"finance|accounting|erp|billing|invoic", re.I),
            re.compile(r"procurement|expense|subscription|revenue", re.I),
        ),
    ),
)


_CORE_SOURCES: dict[ScrapeVertical, frozenset[str]] = {
    ScrapeVertical.communication: frozenset({
        "g2", "capterra", "trustradius", "getapp", "software_advice",
        "trustpilot", "reddit", "hackernews", "github", "stackoverflow",
    }),
    ScrapeVertical.crm_support_marketing: frozenset({
        "g2", "capterra", "trustradius", "getapp", "software_advice",
        "trustpilot", "reddit",
    }),
    ScrapeVertical.cloud_devops_security: frozenset({
        "g2", "trustradius", "gartner", "peerspot", "reddit",
        "hackernews", "github", "stackoverflow", "rss", "getapp",
    }),
    ScrapeVertical.project_collaboration: frozenset({
        "g2", "capterra", "trustradius", "getapp", "software_advice",
        "trustpilot", "reddit",
    }),
    ScrapeVertical.data_analytics: frozenset({
        "g2", "capterra", "trustradius", "gartner", "peerspot",
        "getapp", "software_advice", "reddit", "hackernews", "stackoverflow",
    }),
    ScrapeVertical.ecommerce_retail: frozenset({
        "g2", "capterra", "getapp", "software_advice",
        "trustpilot", "reddit", "producthunt",
    }),
    ScrapeVertical.hr_hcm: frozenset({
        "g2", "capterra", "trustradius", "getapp",
        "software_advice", "trustpilot", "reddit",
    }),
    ScrapeVertical.finance_erp_billing: frozenset({
        "g2", "capterra", "trustradius", "gartner",
        "getapp", "software_advice", "trustpilot", "reddit",
    }),
    ScrapeVertical.general_b2b: frozenset({
        "g2", "capterra", "trustradius", "gartner", "peerspot",
        "getapp", "software_advice", "trustpilot", "reddit",
    }),
}


_AVOID_SOURCES: dict[ScrapeVertical, frozenset[str]] = {
    ScrapeVertical.communication: frozenset({"sourceforge", "twitter"}),
    ScrapeVertical.crm_support_marketing: frozenset({"sourceforge"}),
    ScrapeVertical.cloud_devops_security: frozenset({"sourceforge"}),
    ScrapeVertical.project_collaboration: frozenset({"sourceforge"}),
    ScrapeVertical.data_analytics: frozenset({"sourceforge"}),
    ScrapeVertical.ecommerce_retail: frozenset({"github", "stackoverflow", "sourceforge", "peerspot"}),
    ScrapeVertical.hr_hcm: frozenset({"github", "stackoverflow", "hackernews", "sourceforge", "twitter"}),
    ScrapeVertical.finance_erp_billing: frozenset({"github", "stackoverflow", "hackernews", "sourceforge", "twitter"}),
    ScrapeVertical.general_b2b: frozenset(),
}


def normalize_scrape_vertical(product_category: str | None) -> str | None:
    """Map a product category string into a scraper vertical bucket."""
    text = str(product_category or "").strip()
    if not text:
        return None
    for vertical, patterns in _CATEGORY_RULES:
        if any(pattern.search(text) for pattern in patterns):
            return vertical.value
    return ScrapeVertical.general_b2b.value


def classify_source_fit(source: str, product_category: str | None) -> SourceFitDecision:
    """Classify a source as core, conditional, or avoid for the category."""
    source_name = str(source or "").strip().lower()
    vertical_name = normalize_scrape_vertical(product_category)
    if vertical_name is None:
        return SourceFitDecision(source_name, product_category, None, SourceFit.conditional.value, "missing_product_category")

    vertical = ScrapeVertical(vertical_name)
    if source_name in _CORE_SOURCES[vertical]:
        return SourceFitDecision(source_name, product_category, vertical.value, SourceFit.core.value, "vertical_core_source")
    if source_name in _AVOID_SOURCES[vertical]:
        return SourceFitDecision(source_name, product_category, vertical.value, SourceFit.avoid.value, "vertical_noise_source")
    return SourceFitDecision(source_name, product_category, vertical.value, SourceFit.conditional.value, "vertical_conditional_source")


def is_source_fit_allowed(
    source: str,
    product_category: str | None,
    *,
    allow_conditional: bool = True,
) -> bool:
    """Return True if the source/category pair should be scraped."""
    fit = classify_source_fit(source, product_category).fit
    if fit == SourceFit.core.value:
        return True
    if fit == SourceFit.conditional.value:
        return allow_conditional
    return False
