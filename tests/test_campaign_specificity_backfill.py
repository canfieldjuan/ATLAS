from __future__ import annotations

from atlas_brain.services.campaign_specificity_backfill import (
    derive_campaign_specificity_patch,
)


def test_derive_campaign_specificity_patch_backfills_latest_audit():
    patch = derive_campaign_specificity_patch(
        {
            "channel": "email_followup",
            "body": "<p>General market pressure is rising across the category.</p>",
            "metadata": {
                "reasoning_anchor_examples": {
                    "outlier_or_named_account": [
                        {
                            "witness_id": "w1",
                            "excerpt_text": "A customer hit a $200k/year renewal decision in Q2.",
                            "time_anchor": "Q2 renewal",
                            "numeric_literals": {"currency_mentions": ["$200k/year"]},
                            "competitor": "Freshdesk",
                            "pain_category": "pricing",
                        }
                    ]
                },
                "reasoning_witness_highlights": [
                    {
                        "witness_id": "w1",
                        "excerpt_text": "A customer hit a $200k/year renewal decision in Q2.",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    }
                ],
                "reasoning_reference_ids": {"witness_ids": ["w1"]},
            },
            "company_context": {},
        },
        min_anchor_hits=1,
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
    )

    audit = patch["metadata"]["latest_specificity_audit"]
    assert audit["boundary"] == "backfill"
    assert audit["status"] == "fail"
    assert audit["anchor_count"] == 1
    assert "content does not reference any witness-backed anchor" in audit["blocking_issues"][0]


def test_derive_campaign_specificity_patch_preserves_existing_audit():
    patch = derive_campaign_specificity_patch(
        {
            "channel": "email_followup",
            "body": "<p>Body</p>",
            "metadata": {
                "latest_specificity_audit": {
                    "status": "pass",
                    "blocking_issues": [],
                    "warnings": [],
                }
            },
            "company_context": {},
        },
        min_anchor_hits=1,
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
    )

    assert patch == {}
