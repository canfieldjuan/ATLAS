from __future__ import annotations

import json

import pytest

from atlas_brain.services.blog_quality_backfill import (
    _BLOG_BACKFILL_REPAIR_POLICY_VERSION,
    apply_blog_quality_backfill,
    derive_blog_quality_patch,
    derive_blog_quality_patch_with_recovery,
)


def test_derive_blog_quality_patch_backfills_latest_audit():
    filler = (
        "This analysis reflects self-selected feedback from public software reviews. "
        "Shopify migration signals show broad urgency, pricing pressure, and evaluation activity "
        "across public review sources. "
    ) * 140
    patch = derive_blog_quality_patch(
        {
            "slug": "switch-to-shopify-2026-03",
            "title": "Migration Guide: Why Teams Are Switching to Shopify",
            "description": "desc",
            "topic_type": "migration_guide",
            "tags": ["shopify", "migration"],
            "content": f"<p>{filler}</p>",
            "charts": [],
            "cta": None,
            "data_context": {
                "reasoning_anchor_examples": {
                    "outlier_or_named_account": [
                        {
                            "witness_id": "w1",
                            "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                            "time_anchor": "Q2 renewal",
                            "numeric_literals": {"currency_mentions": ["$200k/year"]},
                            "competitor": "BigCommerce",
                            "pain_category": "pricing",
                        }
                    ]
                },
                "reasoning_witness_highlights": [
                    {
                        "witness_id": "w1",
                        "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "BigCommerce",
                        "pain_category": "pricing",
                    }
                ],
                "reasoning_reference_ids": {"witness_ids": ["w1"]},
            },
        }
    )

    audit = patch["data_context"]["latest_quality_audit"]
    assert audit["boundary"] == "backfill"
    assert audit["status"] == "pass"
    assert audit["min_words_required"] == 1500
    assert audit["target_words"] == 2100
    assert "Evidence anchor:" in patch["resolved_content"]["content"]
    assert "added_witness_anchor_note" in audit["fixes_applied"]
    assert patch["failure_step"] is None


def test_derive_blog_quality_patch_preserves_complete_existing_audit():
    patch = derive_blog_quality_patch(
        {
            "slug": "switch-to-shopify-2026-03",
            "title": "Migration Guide: Why Teams Are Switching to Shopify",
            "description": "desc",
            "topic_type": "migration_guide",
            "tags": ["shopify", "migration"],
            "content": "<p>Body</p>",
            "charts": [],
            "cta": None,
            "data_context": {
                "latest_quality_audit": {
                    "status": "pass",
                    "boundary": "publish",
                    "word_count": 2200,
                    "min_words_required": 1500,
                    "target_words": 2100,
                    "failure_explanation": {
                        "boundary": "publish",
                        "primary_blocker": None,
                        "cause_type": None,
                        "missing_inputs": [],
                        "context_sources": ["data_context"],
                    },
                }
            },
        }
    )

    assert patch == {}


def test_derive_blog_quality_patch_promotes_borderline_short_rejected_row_to_draft():
    patch = derive_blog_quality_patch(
        {
            "slug": "switch-to-shopify-2026-03",
            "title": "Migration Guide: Why Teams Are Switching to Shopify",
            "description": "desc",
            "topic_type": "migration_guide",
            "tags": ["shopify", "migration"],
            "content": "<p>" + ("alpha " * 1470) + "</p>",
            "charts": [],
            "cta": None,
            "status": "rejected",
            "data_context": {
                "review_period": "2025-06 to 2026-03",
                "enriched_count": 148,
                "churn_intent_count": 23,
                "source_distribution": {"g2": 60, "capterra": 41, "reddit": 12},
                "data_quality": {"confidence": "high"},
                "reasoning_scope_summary": {"witnesses_in_scope": 8},
            },
        }
    )

    assert patch["status"] == "draft"
    assert "Coverage snapshot:" in patch["resolved_content"]["content"]
    assert patch["data_context"]["latest_quality_audit"]["status"] == "pass"


def test_derive_blog_quality_patch_refreshes_stale_backfill_draft_policy():
    patch = derive_blog_quality_patch(
        {
            "slug": "zoho-desk-deep-dive-2026-04",
            "title": "Zoho Desk Deep Dive",
            "description": "desc",
            "topic_type": "vendor_deep_dive",
            "tags": ["helpdesk"],
            "content": (
                "<p>"
                + ("Zoho Desk remains central to the narrative. " * 120)
                + "</p>\n"
                + '<p><a href="https://churnsignals.co/blog/freshdesk-deep-dive">Freshdesk</a></p>'
            ),
            "charts": [],
            "cta": None,
            "status": "draft",
            "data_context": {
                "vendor": "Zoho Desk",
                "review_period": "2025-06 to 2026-03",
                "_valid_internal_slugs": ["zoho-desk-deep-dive-2026-04"],
                "latest_quality_audit": {
                    "status": "pass",
                    "boundary": "backfill",
                    "word_count": 2200,
                    "min_words_required": 1800,
                    "target_words": 2400,
                    "repair_policy_version": "v6",
                    "failure_explanation": {
                        "boundary": "backfill",
                        "primary_blocker": None,
                        "cause_type": None,
                        "missing_inputs": [],
                        "context_sources": ["data_context"],
                    },
                },
            },
        }
    )

    assert patch
    assert patch["data_context"]["latest_quality_audit"]["repair_policy_version"] == _BLOG_BACKFILL_REPAIR_POLICY_VERSION
    assert "https://churnsignals.co/blog/freshdesk-deep-dive" not in patch["resolved_content"]["content"]


class _FakePool:
    def __init__(self, rows, recovery_rows=None):
        self._rows = rows
        self._recovery_rows = recovery_rows or {}
        self.executed = []

    async def fetch(self, _query, *_args):
        return self._rows

    async def fetchrow(self, _query, *args):
        slug = str(args[0]) if args else ""
        return self._recovery_rows.get(slug)

    async def execute(self, query, *args):
        self.executed.append((query, args))


def _recovery_anchor_context() -> dict:
    return {
        "vendor": "Shopify",
        "review_period": "2025-06 to 2026-03",
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "A customer said Shopify pushed a $200k/year price jump at Q2 renewal.",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "BigCommerce",
                    "pain_category": "pricing",
                }
            ]
        },
        "reasoning_witness_highlights": [
            {
                "witness_id": "witness:r1:0",
                "excerpt_text": "A customer said Shopify pushed a $200k/year price jump at Q2 renewal.",
                "time_anchor": "Q2 renewal",
                "numeric_literals": {"currency_mentions": ["$200k/year"]},
                "competitor": "BigCommerce",
                "pain_category": "pricing",
            }
        ],
        "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
    }


def _recovery_blog_body() -> str:
    sentence = (
        "Shopify hits a $200k/year flashpoint around the Q2 renewal window while "
        "BigCommerce keeps showing up as the alternative in the witness-backed record."
    )
    return " ".join([sentence] * 120)


def _repairable_rejected_body() -> str:
    base = " ".join(
        [
            (
                "Shopify hits a $200k/year flashpoint around the Q2 renewal window while "
                "BigCommerce keeps showing up as the alternative in the witness-backed record."
            )
        ]
        * 120
    )
    return (
        f"{base}\n\n"
        "- [Close vs Zoho CRM: 102 Reviews Analyzed](/blog/close-vs-zoho-crm-2026-04)\n"
        "See [Zendesk alternatives](/blog/zendesk-alternatives) for general context.\n"
    )


@pytest.mark.asyncio
async def test_derive_blog_quality_patch_with_recovery_recovers_rejected_content():
    row = {
        "id": "33333333-3333-3333-3333-333333333333",
        "slug": "switch-to-shopify-2026-03",
        "title": "Migration Guide: Why Teams Are Switching to Shopify",
        "description": "desc",
        "topic_type": "migration_guide",
        "tags": ["shopify", "migration"],
        "content": "",
        "charts": [],
        "cta": None,
        "seo_title": None,
        "seo_description": None,
        "target_keyword": None,
        "secondary_keywords": [],
        "faq": [],
        "status": "rejected",
        "created_at": "2026-03-31T00:00:00Z",
        "data_context": _recovery_anchor_context(),
    }
    response_text = json.dumps(
        {
            "title": "Why Teams Are Switching to Shopify",
            "description": "Migration analysis",
            "content": _recovery_blog_body(),
            "seo_title": "Switch to Shopify",
            "seo_description": "Migration guide",
            "target_keyword": "switch to Shopify",
            "secondary_keywords": ["Shopify migration"],
            "faq": [],
        }
    )
    pool = _FakePool([row], recovery_rows={"switch-to-shopify-2026-03": {"response_text": response_text}})

    patch = await derive_blog_quality_patch_with_recovery(pool, row)

    assert patch["status"] == "draft"
    assert "self-selected feedback" in patch["resolved_content"]["content"]
    assert "Shopify hits a $200k/year flashpoint" in patch["resolved_content"]["content"]
    assert patch["data_context"]["latest_quality_audit"]["min_words_required"] == 1500
    assert patch["data_context"]["latest_quality_audit"]["status"] == "pass"


@pytest.mark.asyncio
async def test_apply_blog_quality_backfill_updates_recent_rows():
    pool = _FakePool(
        [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "slug": "switch-to-shopify-2026-03",
                "title": "Migration Guide: Why Teams Are Switching to Shopify",
                "description": "desc",
                "topic_type": "migration_guide",
                "tags": ["shopify", "migration"],
                "content": "<p>" + ("Generic market pressure is rising. " * 120) + "</p>",
                "charts": [],
                "data_context": {
                    "reasoning_anchor_examples": {
                        "outlier_or_named_account": [
                            {
                                "witness_id": "w1",
                                "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                                "time_anchor": "Q2 renewal",
                                "numeric_literals": {"currency_mentions": ["$200k/year"]},
                            }
                        ]
                    },
                    "reasoning_reference_ids": {"witness_ids": ["w1"]},
                },
                "cta": None,
                "seo_title": None,
                "seo_description": None,
                "target_keyword": None,
                "secondary_keywords": [],
                "faq": [],
                "status": "draft",
                "created_at": "2026-03-31T00:00:00Z",
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "slug": "already-complete",
                "title": "Already Complete",
                "description": "desc",
                "topic_type": "migration_guide",
                "tags": [],
                "content": "<p>Body</p>",
                "charts": [],
                "data_context": {
                    "latest_quality_audit": {
                        "status": "pass",
                        "boundary": "publish",
                        "word_count": 2200,
                        "min_words_required": 1500,
                        "target_words": 2100,
                        "failure_explanation": {
                            "boundary": "publish",
                            "primary_blocker": None,
                            "cause_type": None,
                            "missing_inputs": [],
                            "context_sources": ["data_context"],
                        },
                    }
                },
                "cta": None,
                "seo_title": None,
                "seo_description": None,
                "target_keyword": None,
                "secondary_keywords": [],
                "faq": [],
                "status": "draft",
                "created_at": "2026-03-31T00:00:00Z",
            },
        ]
    )

    result = await apply_blog_quality_backfill(pool, days=30)

    assert result["scanned"] == 2
    assert result["changed"] == 1
    assert result["applied"] == 1
    assert len(pool.executed) == 1
    updated_context = json.loads(pool.executed[0][1][10])
    assert updated_context["latest_quality_audit"]["boundary"] == "backfill"


@pytest.mark.asyncio
async def test_apply_blog_quality_backfill_promotes_recovered_rejected_row_to_draft():
    pool = _FakePool(
        [
            {
                "id": "44444444-4444-4444-4444-444444444444",
                "slug": "switch-to-shopify-2026-03",
                "title": "Migration Guide: Why Teams Are Switching to Shopify",
                "description": "desc",
                "topic_type": "migration_guide",
                "tags": ["shopify", "migration"],
                "content": "",
                "charts": [],
                "data_context": {"review_period": "2025-06 to 2026-03"},
                "cta": None,
                "seo_title": None,
                "seo_description": None,
                "target_keyword": None,
                "secondary_keywords": [],
                "faq": [],
                "status": "rejected",
                "created_at": "2026-03-31T00:00:00Z",
                "data_context": _recovery_anchor_context(),
            }
        ],
        recovery_rows={
            "switch-to-shopify-2026-03": {
                "response_text": json.dumps(
                    {
                        "title": "Why Teams Are Switching to Shopify",
                        "description": "Migration analysis",
                        "content": _recovery_blog_body(),
                        "seo_title": "Switch to Shopify",
                        "seo_description": "Migration guide",
                        "target_keyword": "switch to Shopify",
                        "secondary_keywords": ["Shopify migration"],
                        "faq": [],
                    }
                )
            }
        },
    )

    result = await apply_blog_quality_backfill(pool, days=30)

    assert result["changed"] == 1
    assert result["applied"] == 1
    assert len(pool.executed) == 1
    query, args = pool.executed[0]
    assert "status = $10" in query
    assert args[9] == "draft"
    assert "self-selected feedback" in args[3]
    assert "Shopify hits a $200k/year flashpoint" in args[3]


def test_derive_blog_quality_patch_repairs_existing_rejected_content():
    row = {
        "id": "55555555-5555-5555-5555-555555555555",
        "slug": "switch-to-shopify-2026-03",
        "title": "Migration Guide: Why Teams Are Switching to Shopify",
        "description": "desc",
        "topic_type": "migration_guide",
        "tags": ["shopify", "migration"],
        "content": _repairable_rejected_body(),
        "charts": [],
        "cta": None,
        "seo_title": "Switch to Shopify",
        "seo_description": "Migration guide",
        "target_keyword": "switch to Shopify",
        "secondary_keywords": ["Shopify migration"],
        "faq": [],
        "status": "rejected",
        "created_at": "2026-03-31T00:00:00Z",
        "data_context": {
            **_recovery_anchor_context(),
            "_known_vendors": ["Shopify", "Close", "Zoho CRM"],
            "_valid_internal_slugs": ["switch-to-shopify-2026-03"],
        },
    }

    patch = derive_blog_quality_patch(row)

    assert patch["status"] == "draft"
    assert patch["data_context"]["latest_quality_audit"]["status"] == "pass"
    assert "/blog/close-vs-zoho-crm-2026-04" not in patch["resolved_content"]["content"]
    assert "/blog/zendesk-alternatives" not in patch["resolved_content"]["content"]
