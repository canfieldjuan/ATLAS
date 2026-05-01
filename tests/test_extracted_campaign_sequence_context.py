from __future__ import annotations

import json

from extracted_content_pipeline.campaign_sequence_context import (
    SequenceContextLimits,
    plain_text_preview,
    prepare_sequence_prompt_contexts,
    prepare_sequence_storage_contexts,
    prompt_email_body_preview_chars,
    prompt_max_tokens,
)


def test_prepare_prompt_contexts_removes_duplicate_and_heavy_fields():
    seq = {
        "company_context": json.dumps({
            "target_persona": "executive",
            "key_quotes": ["q1", "q2", "q3", "q4"],
            "pain_categories": [
                {"category": "pricing", "severity": "high", "extra": "drop"},
                {"category": "support", "severity": "medium"},
            ],
            "feature_gaps": ["automation", "reporting", "search", "alerts", "exports", "audit"],
            "comparison_asset": {"alternative_vendor": "ClickUp"},
            "reasoning_witness_highlights": [{"excerpt_text": "drop from prompt"}],
            "selling": {
                "sender_name": "Atlas Intel",
                "booking_url": "https://example.test/book",
                "blog_posts": [{
                    "title": "Pricing pressure",
                    "url": "https://example.test/blog/pricing",
                    "topic_type": "pricing_reality_check",
                    "slug": "drop",
                }],
            },
        }),
        "selling_context": "",
    }

    company_context, selling_context = prepare_sequence_prompt_contexts(seq)

    assert "selling" not in company_context
    assert "comparison_asset" not in company_context
    assert "reasoning_witness_highlights" not in company_context
    assert company_context["key_quotes"] == ["q1", "q2", "q3"]
    assert company_context["pain_categories"] == [
        {"category": "pricing", "severity": "high"},
        {"category": "support", "severity": "medium"},
    ]
    assert company_context["feature_gaps"] == [
        "automation",
        "reporting",
        "search",
        "alerts",
        "exports",
    ]
    assert selling_context == {
        "sender_name": "Atlas Intel",
        "booking_url": "https://example.test/book",
        "blog_posts": [{
            "title": "Pricing pressure",
            "url": "https://example.test/blog/pricing",
            "topic_type": "pricing_reality_check",
        }],
    }


def test_prepare_storage_contexts_preserves_specificity_fields():
    company_context, selling_context = prepare_sequence_storage_contexts(
        {
            "target_persona": "executive",
            "selling": {"sender_name": "Atlas Intel"},
            "comparison_asset": {"alternative_vendor": "ClickUp"},
            "reasoning_anchor_examples": {"outlier": [{"witness_id": "w1"}]},
            "reasoning_witness_highlights": [{"witness_id": "w1"}],
            "reasoning_reference_ids": {"witness_ids": ["w1"]},
            "reasoning_contracts": {"raw": "drop"},
        },
        {},
    )

    assert "selling" not in company_context
    assert "comparison_asset" not in company_context
    assert "reasoning_contracts" not in company_context
    assert company_context["reasoning_anchor_examples"]["outlier"][0]["witness_id"] == "w1"
    assert company_context["reasoning_witness_highlights"][0]["witness_id"] == "w1"
    assert company_context["reasoning_reference_ids"]["witness_ids"] == ["w1"]
    assert selling_context["sender_name"] == "Atlas Intel"


def test_custom_limits_drive_compaction_without_settings_import():
    limits = SequenceContextLimits(
        prompt_max_tokens=333,
        prompt_list_limit=2,
        prompt_quote_limit=1,
        prompt_blog_post_limit=1,
        prompt_email_body_preview_chars=12,
    )
    seq = {
        "company_context": {
            "key_quotes": ["q1", "q2"],
            "feature_gaps": ["gap1", "gap2", "gap3"],
        },
        "selling_context": {
            "blog_posts": [
                {"title": "one", "url": "https://example.test/one", "topic_type": "a"},
                {"title": "two", "url": "https://example.test/two", "topic_type": "b"},
            ],
        },
    }

    company_context, selling_context = prepare_sequence_prompt_contexts(seq, limits=limits)

    assert prompt_max_tokens(limits) == 333
    assert prompt_email_body_preview_chars(limits) == 12
    assert company_context["key_quotes"] == ["q1"]
    assert company_context["feature_gaps"] == ["gap1", "gap2"]
    assert selling_context["blog_posts"] == [{
        "title": "one",
        "url": "https://example.test/one",
        "topic_type": "a",
    }]


def test_invalid_json_contexts_compact_to_empty_dicts():
    company_context, selling_context = prepare_sequence_prompt_contexts({
        "company_context": "{not-json",
        "selling_context": "{also-not-json",
    })

    assert company_context == {}
    assert selling_context == {}


def test_plain_text_preview_strips_html_and_truncates():
    rendered = plain_text_preview(
        "<p>Hello <strong>team</strong>.</p><p>We tracked a sharp pricing shift.</p>",
        limit=24,
    )

    assert rendered == "Hello team. We tracked a..."
    assert "<p>" not in rendered
    assert "<strong>" not in rendered
