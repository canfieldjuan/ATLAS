"""Ticket FAQ input contracts for hosted Content Ops control surfaces."""

from __future__ import annotations

from typing import Any


TICKET_FAQ_INPUT_ASSET = "faq_markdown"
TICKET_FAQ_VOCABULARY_GAP_INPUT_GROUP = "vocabulary_gap"

FAQ_DOCUMENTATION_TERMS_INPUT = "faq_documentation_terms"
FAQ_VOCABULARY_GAP_RULES_INPUT = "faq_vocabulary_gap_rules"

_TICKET_FAQ_VOCABULARY_GAP_INPUT_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "key": FAQ_DOCUMENTATION_TERMS_INPUT,
        "label": "Documentation terms",
        "type": "string_list",
        "placeholder": "Single sign-on setup\nData export guide",
    },
    {
        "key": FAQ_VOCABULARY_GAP_RULES_INPUT,
        "label": "Vocabulary-gap rules",
        "type": "nested_string_list",
        "placeholder": "SSO, single sign-on\nexport, data export",
    },
)


def ticket_faq_vocabulary_gap_input_contracts() -> dict[str, dict[str, Any]]:
    """Return wire contracts for FAQ vocabulary-gap inputs."""

    return {
        item["key"]: {
            **item,
            "asset": TICKET_FAQ_INPUT_ASSET,
            "group": TICKET_FAQ_VOCABULARY_GAP_INPUT_GROUP,
        }
        for item in _TICKET_FAQ_VOCABULARY_GAP_INPUT_CONTRACTS
    }


__all__ = [
    "FAQ_DOCUMENTATION_TERMS_INPUT",
    "FAQ_VOCABULARY_GAP_RULES_INPUT",
    "TICKET_FAQ_INPUT_ASSET",
    "TICKET_FAQ_VOCABULARY_GAP_INPUT_GROUP",
    "ticket_faq_vocabulary_gap_input_contracts",
]
