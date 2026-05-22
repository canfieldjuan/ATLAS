from extracted_content_pipeline.landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
)
from extracted_content_pipeline.landing_page_readiness import (
    landing_page_geo_readiness,
    landing_page_readiness_repair_issues,
    landing_page_seo_aeo_readiness,
)


def _ready_draft(**overrides) -> LandingPageDraft:
    values = {
        "campaign_name": "support-faq-report",
        "persona": "10-50 person SaaS team",
        "value_prop": "Turn repeat support tickets into customer-ready FAQs",
        "title": "Support FAQ Report for Small SaaS Teams",
        "slug": "support-faq-report",
        "hero": {
            "headline": "Turn repeat tickets into answers",
            "subheadline": (
                "10-50 person SaaS teams turn repeat support tickets into "
                "customer-ready FAQs before customers wait again."
            ),
            "cta_label": "Upload Ticket CSV -- Free Analysis",
            "cta_url": "/systems/ai-content-ops/intake",
        },
        "sections": (
            LandingPageSection(
                id="repeat_support_problem",
                title="Repeat support questions become customer frustration",
                body_markdown=(
                    "10-50 person SaaS teams lose time when customers ask "
                    "the same support questions again and again. Repeat "
                    "questions create friction because the answer is not "
                    "where customers are looking."
                ),
                metadata={
                    "kind": "problem",
                    "primary_question": "Why do repeat support questions matter?",
                    "answer_summary": (
                        "10-50 person SaaS teams lose time when customers "
                        "ask the same support questions again and again."
                    ),
                },
            ),
            LandingPageSection(
                id="faq_report_solution",
                title="A FAQ Report turns tickets into findable answers",
                body_markdown=(
                    "10-50 person SaaS teams use the FAQ Report workflow to "
                    "turn old support tickets into clear answers customers "
                    "can find before they email support."
                ),
                metadata={
                    "kind": "solution",
                    "primary_question": "How does the FAQ Report help?",
                    "answer_summary": (
                        "10-50 person SaaS teams use the FAQ Report workflow "
                        "to turn old support tickets into clear answers."
                    ),
                },
            ),
            LandingPageSection(
                id="before_upload_questions",
                title="Questions before uploading tickets",
                body_markdown=(
                    "10-50 person SaaS teams can review privacy, publishing, "
                    "and setup questions before uploading tickets. That keeps "
                    "the process clear without giving up help-center control."
                ),
                metadata={
                    "kind": "objection",
                    "primary_question": "What should teams know before upload?",
                    "answer_summary": (
                        "10-50 person SaaS teams can review privacy, "
                        "publishing, and setup questions before uploading "
                        "tickets."
                    ),
                },
            ),
        ),
        "cta": {
            "label": "Upload Ticket CSV -- Free Analysis",
            "url": "/systems/ai-content-ops/intake",
            "variant": "primary",
        },
        "meta": {
            "title_tag": "Support FAQ Report for Small SaaS Teams",
            "description": (
                "Turn repeat support tickets into customer-ready FAQ answers "
                "small SaaS teams can publish before customers wait again."
            ),
        },
        "reference_ids": ("support-ticket-sample",),
    }
    values.update(overrides)
    return LandingPageDraft(**values)


def test_landing_page_readiness_reports_ready_payloads() -> None:
    draft = _ready_draft()

    assert landing_page_seo_aeo_readiness(draft)["status"] == "ready"
    assert landing_page_geo_readiness(draft)["status"] == "ready"
    assert landing_page_readiness_repair_issues(draft) == ()


def test_landing_page_readiness_repair_issues_prefix_missing_checks() -> None:
    draft = _ready_draft(
        slug="landing-page",
        meta={"title_tag": "Support"},
        cta={"label": "Upload Ticket CSV -- Free Analysis", "url": "#"},
    )

    issues = landing_page_readiness_repair_issues(draft)

    assert "seo_aeo_readiness:meta_description" in issues
    assert "seo_aeo_readiness:slug_quality" in issues
    assert "geo_readiness:conversion_path_clarity" in issues
