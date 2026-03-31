from __future__ import annotations

import json

import pytest

from atlas_brain.services.campaign_specificity_backfill import (
    apply_campaign_specificity_backfill,
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
    assert audit["failure_explanation"]["boundary"] == "backfill"
    assert audit["failure_explanation"]["cause_type"] == "content_ignored_available_evidence"
    assert audit["failure_explanation"]["missing_inputs"] == []


def test_derive_campaign_specificity_patch_upgrades_incomplete_existing_audit():
    patch = derive_campaign_specificity_patch(
        {
            "channel": "email_followup",
            "body": "<p>Body</p>",
            "metadata": {
                "latest_specificity_audit": {
                    "status": "fail",
                    "boundary": "manual_approval",
                    "blocking_issues": [
                        "content omits a concrete timing or numeric anchor even though one is available"
                    ],
                    "warnings": [],
                    "anchor_count": 1,
                    "highlight_count": 1,
                    "reference_ids": {"witness_ids": ["w1"]},
                    "available_groups": ["numeric_terms"],
                    "matched_groups": [],
                    "reasoning_reference_ids": {"witness_ids": ["w1"]},
                },
                "reasoning_anchor_examples": {
                    "outlier_or_named_account": [
                        {
                            "witness_id": "w1",
                            "excerpt_text": "A customer hit a $200k/year renewal decision in Q2.",
                            "time_anchor": "Q2 renewal",
                            "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        }
                    ]
                }
            },
            "company_context": {},
        },
        min_anchor_hits=1,
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
    )

    audit = patch["metadata"]["latest_specificity_audit"]
    assert audit["boundary"] == "manual_approval"
    assert audit["failure_explanation"]["boundary"] == "manual_approval"
    assert audit["failure_explanation"]["cause_type"] == "content_ignored_available_evidence"


def test_derive_campaign_specificity_patch_restores_boundary_from_status():
    patch = derive_campaign_specificity_patch(
        {
            "status": "queued",
            "channel": "email_followup",
            "body": "<p>Body</p>",
            "metadata": {
                "latest_specificity_audit": {
                    "status": "fail",
                    "boundary": "backfill",
                    "blocking_issues": [
                        "content omits a concrete timing or numeric anchor even though one is available"
                    ],
                    "warnings": [],
                    "failure_explanation": {
                        "boundary": "backfill",
                        "primary_blocker": "content omits a concrete timing or numeric anchor even though one is available",
                        "cause_type": "content_ignored_available_evidence",
                        "missing_inputs": [],
                        "context_sources": ["metadata"],
                    },
                },
                "reasoning_anchor_examples": {
                    "outlier_or_named_account": [
                        {
                            "witness_id": "w1",
                            "excerpt_text": "A customer hit a $200k/year renewal decision in Q2.",
                            "time_anchor": "Q2 renewal",
                            "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        }
                    ]
                },
                "reasoning_reference_ids": {"witness_ids": ["w1"]},
            },
            "company_context": {},
        },
        min_anchor_hits=1,
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
    )

    audit = patch["metadata"]["latest_specificity_audit"]
    assert audit["boundary"] == "queue_send"
    assert audit["failure_explanation"]["boundary"] == "queue_send"


def test_derive_campaign_specificity_patch_preserves_complete_existing_audit():
    patch = derive_campaign_specificity_patch(
        {
            "channel": "email_followup",
            "body": "<p>Body</p>",
            "metadata": {
                "latest_specificity_audit": {
                    "status": "pass",
                    "blocking_issues": [],
                    "warnings": [],
                    "failure_explanation": {
                        "boundary": "manual_approval",
                        "primary_blocker": None,
                        "cause_type": None,
                        "missing_inputs": [],
                        "context_sources": ["metadata"],
                    },
                }
            },
            "company_context": {},
        },
        min_anchor_hits=1,
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
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
async def test_apply_campaign_specificity_backfill_updates_incomplete_legacy_rows():
    pool = _FakePool(
        [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "company_name": "Slack",
                "vendor_name": "Slack",
                "status": "approved",
                "channel": "email_followup",
                "subject": "Legacy subject",
                "body": "<p>General market pressure is rising across the category.</p>",
                "cta": "Book time",
                "target_mode": "vendor_retention",
                "metadata": {
                    "latest_specificity_audit": {
                        "status": "fail",
                        "boundary": "manual_approval",
                        "blocking_issues": [
                            "content omits a concrete timing or numeric anchor even though one is available"
                        ],
                        "warnings": [],
                    },
                    "reasoning_anchor_examples": {
                        "outlier_or_named_account": [
                            {
                                "witness_id": "w1",
                                "excerpt_text": "A customer hit a $200k/year renewal decision in Q2.",
                                "time_anchor": "Q2 renewal",
                                "numeric_literals": {"currency_mentions": ["$200k/year"]},
                            }
                        ]
                    },
                    "reasoning_reference_ids": {"witness_ids": ["w1"]},
                },
                "company_context": {},
            }
        ]
    )

    result = await apply_campaign_specificity_backfill(
        pool,
        min_anchor_hits=1,
        require_anchor_support=True,
        require_timing_or_numeric_when_available=True,
    )

    assert result["scanned"] == 1
    assert result["changed"] == 1
    assert result["applied"] == 1
    assert len(pool.executed) == 1
    updated_metadata = json.loads(pool.executed[0][1][1])
    assert updated_metadata["latest_specificity_audit"]["boundary"] == "manual_approval"
    assert (
        updated_metadata["latest_specificity_audit"]["failure_explanation"]["cause_type"]
        == "content_ignored_available_evidence"
    )
