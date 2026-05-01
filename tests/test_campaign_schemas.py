"""Round-trip tests for atlas_brain.schemas.campaigns.

Validates the four JSONB-blob models (CampaignMetadata, CompanyContext,
SellingContext, BriefingData) plus CanonicalEvent for the multi-ESP webhook
layer. The models open with extra='allow' during the soak window so
unknown keys must flow through unchanged; that posture is enforced here so
a future tightening (extra='forbid') is caught by the test suite, not by a
production row failing to parse.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from atlas_brain.schemas.campaigns import (
    BriefingData,
    CampaignMetadata,
    CanonicalEvent,
    CompanyContext,
    SellingContext,
)


def test_campaign_metadata_round_trip():
    payload = {
        "tier": "gold",
        "target_mode": "churning_company",
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [{"witness_id": "w1", "excerpt_text": "..."}],
        },
        "reasoning_witness_highlights": [{"witness_id": "w1"}],
        "reasoning_reference_ids": {"witness_ids": ["w1"]},
        "campaign_proof_terms": ["renewal", "Q1"],
        "opportunity_claim": {"score": 87},
        "generation_audit": {"model": "claude-opus", "tokens": 1820},
        "latest_specificity_audit": {"matched_groups": ["timing"]},
    }
    model = CampaignMetadata.model_validate(payload)
    assert model.tier == "gold"
    assert model.schema_version == 1
    serialized = model.model_dump(exclude_none=True)
    assert serialized["tier"] == "gold"
    assert serialized["reasoning_anchor_examples"]["outlier_or_named_account"][0]["witness_id"] == "w1"


def test_campaign_metadata_accepts_unknown_keys():
    """During the soak window, extra='allow' must permit unknown keys."""
    payload = {"tier": "silver", "future_key_we_have_not_seen_yet": {"hello": "world"}}
    model = CampaignMetadata.model_validate(payload)
    dumped = model.model_dump()
    assert dumped["future_key_we_have_not_seen_yet"] == {"hello": "world"}


def test_campaign_metadata_json_round_trip():
    payload = {"tier": "bronze", "campaign_proof_terms": ["renewal"]}
    blob = json.dumps(payload)
    model = CampaignMetadata.model_validate_json(blob)
    assert model.tier == "bronze"
    assert model.campaign_proof_terms == ["renewal"]


def test_company_context_pain_categories_typed():
    payload = {
        "company": "Acme Corp",
        "churning_from": "Vendor A",
        "pain_categories": [
            {"category": "pricing", "severity": "high"},
            {"category": "support"},
        ],
        "competitors_considering": [{"name": "Vendor B"}, {"vendor_name": "Vendor C"}],
        "feature_gaps": ["sso", "api"],
    }
    model = CompanyContext.model_validate(payload)
    assert model.pain_categories[0].category == "pricing"
    assert model.pain_categories[1].severity is None
    assert model.competitors_considering[0].name == "Vendor B"


def test_selling_context_blog_posts():
    payload = {
        "sender_name": "Sam Sender",
        "sender_company": "ChurnSignals",
        "primary_blog_post": {
            "id": "abc",
            "title": "5 reasons teams leave Vendor A",
            "url": "https://example.com/x",
            "pain_tags": ["pricing"],
        },
        "blog_posts": [
            {"id": "abc", "title": "5 reasons teams leave Vendor A"},
            {"id": "def", "title": "Migration guide"},
        ],
    }
    model = SellingContext.model_validate(payload)
    assert model.primary_blog_post.title.startswith("5 reasons")
    assert len(model.blog_posts) == 2


def test_briefing_data_roundtrips_with_extras():
    payload = {
        "vendor_name": "Vendor A",
        "pain_categories": ["pricing", "support"],
        "key_quotes": ["pricing increased 40%"],
        "battle_cards": [{"vs": "Vendor B", "wedge": "transparent pricing"}],
        "an_unknown_extension": [1, 2, 3],
    }
    model = BriefingData.model_validate(payload)
    assert model.vendor_name == "Vendor A"
    assert model.pain_categories == ["pricing", "support"]
    dumped = model.model_dump()
    assert dumped["an_unknown_extension"] == [1, 2, 3]


def test_canonical_event_minimal_payload():
    payload = {
        "provider": "resend",
        "event_type": "opened",
        "message_id": "msg_123",
        "recipient_email": "alice@example.com",
        "timestamp": "2026-04-30T12:00:00Z",
    }
    event = CanonicalEvent.model_validate(payload)
    assert event.event_type == "opened"
    assert event.bounce_type is None


def test_canonical_event_rejects_unknown_keys():
    """CanonicalEvent uses extra='forbid' (unlike the storage models): the
    webhook layer must produce only canonical fields, and a field we don't
    handle in the route handler should fail loudly rather than silently
    pass through."""
    payload = {
        "provider": "resend",
        "event_type": "clicked",
        "message_id": "msg_456",
        "recipient_email": "bob@example.com",
        "timestamp": "2026-04-30T12:00:00Z",
        "unexpected_field": "boom",
    }
    with pytest.raises(ValidationError):
        CanonicalEvent.model_validate(payload)


def test_canonical_event_rejects_invalid_event_type():
    payload = {
        "provider": "ses",
        "event_type": "spammed",  # not in CanonicalEventType
        "message_id": "msg_789",
        "recipient_email": "carol@example.com",
        "timestamp": "2026-04-30T12:00:00Z",
    }
    with pytest.raises(ValidationError):
        CanonicalEvent.model_validate(payload)


def test_models_export_json_schema():
    """Customer integrations consume model_json_schema(); make sure each model
    can produce one without crashing."""
    for model in (CampaignMetadata, CompanyContext, SellingContext, BriefingData, CanonicalEvent):
        schema = model.model_json_schema()
        assert "properties" in schema
        assert "schema_version" in schema["properties"]
