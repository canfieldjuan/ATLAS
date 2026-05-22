from __future__ import annotations

from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
)
from extracted_content_pipeline.landing_page_structured_data import (
    build_landing_page_structured_data,
)


def _draft(**overrides) -> LandingPageDraft:
    sections = (
        LandingPageSection(
            id="problem",
            title="Support questions keep repeating",
            body_markdown=(
                "Teams reduce repeat support pressure by turning old tickets "
                "into answers customers can find before they email. Customers "
                "stop waiting on the same basic answer."
            ),
            metadata={
                "order": 1,
                "kind": "problem",
                "primary_question": (
                    "Why do repeat support questions create retention risk?"
                ),
                "answer_summary": (
                    "Teams reduce repeat support pressure by turning old "
                    "tickets into answers customers can find before they email."
                ),
            },
        ),
        LandingPageSection(
            id="proof",
            title="Internal proof",
            body_markdown=(
                "This section is not a customer question, so it should remain "
                "a WebPageElement instead of becoming FAQPage schema."
            ),
            metadata={
                "order": 2,
                "kind": "proof",
                "primary_question": "What proof exists?",
                "answer_summary": (
                    "This hidden summary does not match the visible body start."
                ),
            },
        ),
    )
    draft = LandingPageDraft(
        campaign_name="faq-report",
        persona="Small SaaS operator",
        value_prop="Turn repeat support tickets into clearer FAQ answers",
        title="FAQ Report",
        slug="faq-report",
        hero={"headline": "Stop answering the same support questions"},
        sections=sections,
        cta={"label": "Upload ticket CSV", "url": "/intake"},
        meta={
            "title_tag": "FAQ Report for Small SaaS Teams",
            "description": (
                "Turn repeat support tickets into FAQ answers customers can "
                "find before they email again."
            ),
            "canonical_url": "https://example.com/faq-report",
        },
        reference_ids=("ticket-cluster-1",),
    )
    return LandingPageDraft(
        campaign_name=overrides.get("campaign_name", draft.campaign_name),
        persona=overrides.get("persona", draft.persona),
        value_prop=overrides.get("value_prop", draft.value_prop),
        title=overrides.get("title", draft.title),
        slug=overrides.get("slug", draft.slug),
        hero=overrides.get("hero", draft.hero),
        sections=overrides.get("sections", draft.sections),
        cta=overrides.get("cta", draft.cta),
        meta=overrides.get("meta", draft.meta),
        reference_ids=overrides.get("reference_ids", draft.reference_ids),
        metadata=overrides.get("metadata", draft.metadata),
        id=overrides.get("id", draft.id),
        status=overrides.get("status", draft.status),
    )


def test_build_landing_page_structured_data_emits_webpage_and_faqpage() -> None:
    structured_data = build_landing_page_structured_data(_draft())

    assert structured_data["@context"] == "https://schema.org"
    graph = structured_data["@graph"]
    webpage = graph[0]
    faq_page = graph[1]

    assert webpage["@type"] == "WebPage"
    assert webpage["@id"] == "https://example.com/faq-report#webpage"
    assert webpage["url"] == "https://example.com/faq-report"
    assert webpage["name"] == "FAQ Report for Small SaaS Teams"
    assert webpage["description"].startswith("Turn repeat support tickets")
    assert webpage["audience"] == {
        "@type": "Audience",
        "audienceType": "Small SaaS operator",
    }
    assert webpage["about"] == {
        "@type": "Thing",
        "name": "Turn repeat support tickets into clearer FAQ answers",
    }
    assert webpage["potentialAction"] == {
        "@type": "Action",
        "name": "Upload ticket CSV",
        "target": "/intake",
    }
    assert webpage["hasPart"][0]["additionalType"] == "problem"

    assert faq_page["@type"] == "FAQPage"
    assert faq_page["@id"] == "https://example.com/faq-report#faq"
    assert faq_page["mainEntityOfPage"] == {
        "@id": "https://example.com/faq-report#webpage",
    }
    assert faq_page["mainEntity"] == [
        {
            "@type": "Question",
            "name": "Why do repeat support questions create retention risk?",
            "acceptedAnswer": {
                "@type": "Answer",
                "text": (
                    "Teams reduce repeat support pressure by turning old "
                    "tickets into answers customers can find before they email."
                ),
            },
        }
    ]


def test_build_landing_page_structured_data_omits_faqpage_without_questions() -> None:
    structured_data = build_landing_page_structured_data(
        _draft(sections=(
            LandingPageSection(
                id="overview",
                title="Overview",
                body_markdown="A plain section without question metadata.",
                metadata={"order": 1, "kind": "proof"},
            ),
        ))
    )

    assert [node["@type"] for node in structured_data["@graph"]] == ["WebPage"]


def test_build_landing_page_structured_data_does_not_invent_canonical_url() -> None:
    structured_data = build_landing_page_structured_data(
        _draft(meta={"title_tag": "FAQ Report"})
    )

    webpage = structured_data["@graph"][0]
    assert webpage["@type"] == "WebPage"
    assert "@id" not in webpage
    assert "url" not in webpage
