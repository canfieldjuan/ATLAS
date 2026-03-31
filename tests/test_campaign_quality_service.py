from unittest.mock import AsyncMock

import pytest

from atlas_brain.services import campaign_quality as mod


def test_campaign_quality_revalidation_merges_metadata_and_boundary():
    result = mod.campaign_quality_revalidation(
        campaign={
            "subject": "Renewal pressure",
            "body": "<p>The Q2 renewal now carries a $200k/year issue.</p>",
            "cta": "Book time",
            "channel": "email_followup",
            "target_mode": "vendor_retention",
            "tier": "report",
            "metadata": {},
        },
        boundary="manual_approval",
        specificity_context={
            "anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "w1",
                        "excerpt_text": "The Q2 renewal now carries a $200k/year issue.",
                    },
                ],
            },
            "witness_highlights": [
                {
                    "witness_id": "w1",
                    "excerpt_text": "The Q2 renewal now carries a $200k/year issue.",
                },
            ],
            "reference_ids": {"witness_ids": ["w1"]},
        },
    )

    assert result["audit"]["status"] == "pass"
    assert result["metadata"]["tier"] == "report"
    assert result["metadata"]["target_mode"] == "vendor_retention"
    assert result["metadata"]["reasoning_anchor_examples"]["outlier_or_named_account"][0]["witness_id"] == "w1"
    assert result["metadata"]["latest_specificity_audit"]["boundary"] == "manual_approval"
    explanation = result["metadata"]["latest_specificity_audit"]["failure_explanation"]
    assert explanation["boundary"] == "manual_approval"
    assert explanation["cause_type"] is None
    assert explanation["anchor_count"] == 1


@pytest.mark.asyncio
async def test_campaign_quality_revalidation_with_fallback_uses_reasoning_view(monkeypatch):
    async def _fake_load_best_reasoning_view(pool, vendor_name):
        assert vendor_name == "Slack"
        return type(
            "_View",
            (),
            {
                "consumer_context": lambda self, consumer: {
                    "anchor_examples": {
                        "outlier_or_named_account": [
                            {
                                "witness_id": "w1",
                                "excerpt_text": "The Q2 renewal now carries a $200k/year issue.",
                            },
                        ],
                    },
                    "witness_highlights": [
                        {
                            "witness_id": "w1",
                            "excerpt_text": "The Q2 renewal now carries a $200k/year issue.",
                        },
                    ],
                    "reference_ids": {"witness_ids": ["w1"]},
                },
            },
        )()

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.load_best_reasoning_view",
        _fake_load_best_reasoning_view,
    )

    result = await mod.campaign_quality_revalidation_with_fallback(
        AsyncMock(),
        campaign={
            "vendor_name": "Slack",
            "subject": "Renewal pressure",
            "body": "<p>The Q2 renewal now carries a $200k/year issue.</p>",
            "cta": "Book time",
            "channel": "email_followup",
            "metadata": {},
        },
        boundary="send",
        company_context=None,
    )

    assert result["audit"]["status"] == "pass"
    assert result["metadata"]["reasoning_reference_ids"]["witness_ids"] == ["w1"]
    assert result["metadata"]["latest_specificity_audit"]["boundary"] == "send"
    explanation = result["metadata"]["latest_specificity_audit"]["failure_explanation"]
    assert explanation["fallback_used"] is True
    assert explanation["reasoning_view_found"] is True
    assert explanation["context_sources"] == ["reasoning_fallback"]
