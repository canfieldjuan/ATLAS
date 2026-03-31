from __future__ import annotations

import json

import pytest

from atlas_brain.services.blog_quality_backfill import (
    apply_blog_quality_backfill,
    derive_blog_quality_patch,
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
    assert audit["status"] == "fail"
    assert audit["failure_explanation"]["cause_type"] == "content_ignored_available_evidence"
    assert patch["failure_step"] == "backfill"


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


class _FakePool:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def fetch(self, _query, *_args):
        return self._rows

    async def execute(self, query, *args):
        self.executed.append((query, args))


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
    updated_context = json.loads(pool.executed[0][1][1])
    assert updated_context["latest_quality_audit"]["boundary"] == "backfill"
