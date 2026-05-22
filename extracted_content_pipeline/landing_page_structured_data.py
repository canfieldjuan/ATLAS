"""Schema.org structured-data builder for generated landing pages."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from extracted_quality_gate.landing_page_section_contract import (
    LANDING_PAGE_QUESTION_SECTION_KINDS,
    normalize_landing_page_section_kind,
)

from .landing_page_ports import LandingPageDraft, LandingPageSection


JsonDict = dict[str, Any]


def build_landing_page_structured_data(draft: LandingPageDraft) -> JsonDict:
    """Return renderer-ready JSON-LD for a generated landing-page draft.

    The builder only emits facts already present in the draft. It avoids
    inventing a public URL, organization, pricing, or proof claim that the
    generator did not provide.
    """

    graph: list[JsonDict] = [_web_page_node(draft)]
    faq_page = _faq_page_node(draft)
    if faq_page:
        graph.append(faq_page)
    return {
        "@context": "https://schema.org",
        "@graph": graph,
    }


def _web_page_node(draft: LandingPageDraft) -> JsonDict:
    meta = _mapping(draft.meta)
    canonical_url = _text(
        meta.get("canonical_url") or meta.get("public_url") or meta.get("url")
    )
    description = _text(meta.get("description"))
    node: JsonDict = {
        "@type": "WebPage",
        "name": _text(meta.get("title_tag")) or _text(draft.title),
    }
    if canonical_url:
        node["@id"] = f"{canonical_url}#webpage"
        node["url"] = canonical_url
    if description:
        node["description"] = description
    audience = _text(draft.persona)
    if audience:
        node["audience"] = {
            "@type": "Audience",
            "audienceType": audience,
        }
    value_prop = _text(draft.value_prop)
    if value_prop:
        node["about"] = {
            "@type": "Thing",
            "name": value_prop,
        }
    cta = _action_node(draft.cta)
    if cta:
        node["potentialAction"] = cta
    parts = _section_part_nodes(draft.sections)
    if parts:
        node["hasPart"] = parts
    return node


def _faq_page_node(draft: LandingPageDraft) -> JsonDict | None:
    questions = [
        question
        for section in draft.sections
        if (question := _question_node(section))
    ]
    if not questions:
        return None
    meta = _mapping(draft.meta)
    canonical_url = _text(
        meta.get("canonical_url") or meta.get("public_url") or meta.get("url")
    )
    node: JsonDict = {
        "@type": "FAQPage",
        "name": f"{_text(draft.title) or _text(draft.campaign_name)} FAQs",
        "mainEntity": questions,
    }
    if canonical_url:
        node["@id"] = f"{canonical_url}#faq"
        node["mainEntityOfPage"] = {"@id": f"{canonical_url}#webpage"}
    return node


def _section_part_nodes(sections: Any) -> list[JsonDict]:
    nodes: list[JsonDict] = []
    for index, section in enumerate(sections or (), start=1):
        title = _text(getattr(section, "title", ""))
        body = _text(getattr(section, "body_markdown", ""))
        if not title and not body:
            continue
        metadata = _mapping(getattr(section, "metadata", {}))
        node: JsonDict = {
            "@type": "WebPageElement",
            "position": _position(metadata.get("order"), index),
        }
        if title:
            node["name"] = title
        if body:
            node["text"] = body
        kind = normalize_landing_page_section_kind(metadata.get("kind"))
        if kind:
            node["additionalType"] = kind
        nodes.append(node)
    return nodes


def _question_node(section: LandingPageSection) -> JsonDict | None:
    metadata = _mapping(section.metadata)
    question = _text(metadata.get("primary_question"))
    answer = _text(metadata.get("answer_summary"))
    kind = normalize_landing_page_section_kind(metadata.get("kind"))
    if not question or not answer:
        return None
    if kind and kind not in LANDING_PAGE_QUESTION_SECTION_KINDS:
        return None
    if not _answer_visible(answer, section.body_markdown):
        return None
    return {
        "@type": "Question",
        "name": question,
        "acceptedAnswer": {
            "@type": "Answer",
            "text": answer,
        },
    }


def _action_node(value: Any) -> JsonDict | None:
    cta = _mapping(value)
    label = _text(cta.get("label"))
    url = _text(cta.get("url"))
    if not label and not url:
        return None
    node: JsonDict = {"@type": "Action"}
    if label:
        node["name"] = label
    if url:
        node["target"] = url
    return node


def _mapping(value: Any) -> JsonDict:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _position(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def _answer_visible(answer: str, body: str) -> bool:
    return bool(answer and _normalize(body).startswith(_normalize(answer)))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).lower()).strip()


__all__ = ["build_landing_page_structured_data"]
