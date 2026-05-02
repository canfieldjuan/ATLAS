from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignDraft,
    SendResult,
    TenantScope,
    WebhookEvent,
)
from extracted_content_pipeline.campaign_postgres import (
    PostgresCampaignAuditSink,
    PostgresCampaignRepository,
    PostgresCampaignSequenceRepository,
    PostgresIntelligenceRepository,
    PostgresSuppressionRepository,
)


class _Pool:
    def __init__(self):
        self.fetchval_results = ["campaign-1"]
        self.fetch_rows = []
        self.fetchrow_result = None
        self.fetchval_calls = []
        self.fetch_calls = []
        self.fetchrow_calls = []
        self.execute_calls = []

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append({"query": query, "args": args})
        return self.fetchrow_result

    async def execute(self, query, *args):
        self.execute_calls.append({"query": query, "args": args})
        return "OK"


@pytest.mark.asyncio
async def test_intelligence_repository_reads_customer_opportunities_with_scope_and_filters():
    pool = _Pool()
    pool.fetch_rows = [
        {
            "id": "row-1",
            "account_id": "acct-1",
            "target_id": "opp-1",
            "company_name": "Acme",
            "vendor_name": "HubSpot",
            "contact_email": "buyer@example.com",
            "contact_title": "VP Revenue",
            "opportunity_score": 84,
            "urgency_score": 8,
            "pain_points": ["pricing"],
            "competitors": ["Salesforce", "Zoho"],
            "evidence": [{"quote": "Too expensive"}],
            "raw_payload": {
                "custom_segment": "enterprise",
                "source_system": "warehouse",
            },
        }
    ]
    repo = PostgresIntelligenceRepository(pool)

    rows = await repo.read_campaign_opportunities(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        limit=10,
        filters={"vendor_name": "HubSpot", "custom_segment": "enterprise"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["target_id"] == "opp-1"
    assert row["target_mode"] == "vendor_retention"
    assert row["company_name"] == "Acme"
    assert row["vendor_name"] == "HubSpot"
    assert row["contact_email"] == "buyer@example.com"
    assert row["contact_title"] == "VP Revenue"
    assert row["pain_points"] == ["pricing"]
    assert row["competitors"] == ["Salesforce", "Zoho"]
    assert row["evidence"] == [{"quote": "Too expensive"}]
    assert row["custom_segment"] == "enterprise"
    call = pool.fetch_calls[0]
    assert 'FROM "campaign_opportunities"' in call["query"]
    assert "account_id = $2" in call["query"]
    assert "LOWER(vendor_name) = LOWER($3)" in call["query"]
    assert "LOWER(raw_payload ->> $4) = LOWER($5)" in call["query"]
    assert call["args"] == ("vendor_retention", "acct-1", "HubSpot", "custom_segment", "enterprise", 10)


@pytest.mark.asyncio
async def test_intelligence_repository_accepts_json_string_payloads():
    pool = _Pool()
    pool.fetch_rows = [
        {
            "target_id": "opp-1",
            "company_name": "",
            "vendor_name": "",
            "pain_points": '["pricing", "support"]',
            "competitors": '["Salesforce"]',
            "evidence": '[{"quote":"Too expensive"}]',
            "raw_payload": json.dumps({
                "company": "Acme",
                "vendor": "HubSpot",
                "email": "buyer@example.com",
                "custom_segment": "midmarket",
            }),
        }
    ]
    repo = PostgresIntelligenceRepository(pool)

    rows = await repo.read_campaign_opportunities(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=5,
    )

    assert rows[0]["company_name"] == "Acme"
    assert rows[0]["vendor_name"] == "HubSpot"
    assert rows[0]["contact_email"] == "buyer@example.com"
    assert rows[0]["pain_points"] == ["pricing", "support"]
    assert rows[0]["competitors"] == ["Salesforce"]
    assert rows[0]["evidence"] == [{"quote": "Too expensive"}]
    assert rows[0]["custom_segment"] == "midmarket"


@pytest.mark.asyncio
async def test_intelligence_repository_rejects_unsafe_table_and_filter_identifiers():
    pool = _Pool()
    repo = PostgresIntelligenceRepository(pool, opportunity_table="campaign_opportunities;drop")

    with pytest.raises(ValueError, match="invalid SQL identifier"):
        await repo.read_campaign_opportunities(
            scope=TenantScope(),
            target_mode="vendor_retention",
            limit=5,
        )

    repo = PostgresIntelligenceRepository(pool)
    with pytest.raises(ValueError, match="unsupported filter key"):
        await repo.read_campaign_opportunities(
            scope=TenantScope(),
            target_mode="vendor_retention",
            limit=5,
            filters={"bad-key": "value"},
        )


@pytest.mark.asyncio
async def test_intelligence_repository_reads_vendor_targets():
    pool = _Pool()
    pool.fetch_rows = [{"id": "target-1", "company_name": "Acme"}]
    repo = PostgresIntelligenceRepository(pool)

    rows = await repo.read_vendor_targets(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        vendor_name="Acme",
    )

    assert rows == ({"id": "target-1", "company_name": "Acme"},)
    call = pool.fetch_calls[0]
    assert 'FROM "vendor_targets"' in call["query"]
    assert "target_mode = $1" in call["query"]
    assert "LOWER(company_name) = LOWER($2)" in call["query"]
    assert "account_id" not in call["query"]
    assert call["args"] == ("vendor_retention", "Acme")


@pytest.mark.asyncio
async def test_campaign_repository_saves_drafts_with_scope_and_source_metadata():
    pool = _Pool()
    repo = PostgresCampaignRepository(pool)
    draft = CampaignDraft(
        target_id="target-1",
        target_mode="vendor_retention",
        channel="email",
        subject="Pricing signal",
        body="<p>Hello</p>",
        metadata={
            "cta": "Book time",
            "generation_model": "model-1",
            "source_opportunity": {
                "company_name": "Acme",
                "vendor_name": "HubSpot",
                "product_category": "CRM",
                "recipient_email": "buyer@example.com",
            },
        },
    )

    saved = await repo.save_drafts(
        [draft],
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
    )

    assert saved == ("campaign-1",)
    call = pool.fetchval_calls[0]
    assert "INSERT INTO b2b_campaigns" in call["query"]
    assert call["args"][:9] == (
        "Acme",
        "HubSpot",
        "CRM",
        "vendor_retention",
        "email",
        "Pricing signal",
        "<p>Hello</p>",
        "Book time",
        "buyer@example.com",
    )
    metadata = json.loads(call["args"][9])
    assert metadata["target_id"] == "target-1"
    assert metadata["scope"] == {"account_id": "acct-1", "user_id": "user-1"}
    assert call["args"][10] == "model-1"


@pytest.mark.asyncio
async def test_campaign_repository_lists_due_queued_sends():
    pool = _Pool()
    pool.fetch_rows = [{"id": "campaign-1", "recipient_email": "buyer@example.com"}]
    repo = PostgresCampaignRepository(pool)
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)

    rows = await repo.list_due_sends(limit=5, now=now)

    assert rows == ({"id": "campaign-1", "recipient_email": "buyer@example.com"},)
    assert "status = 'queued'" in pool.fetch_calls[0]["query"]
    assert pool.fetch_calls[0]["args"] == (5,)


@pytest.mark.asyncio
async def test_campaign_repository_marks_sent_with_provider_metadata():
    pool = _Pool()
    repo = PostgresCampaignRepository(pool)
    sent_at = datetime(2026, 5, 1, 15, tzinfo=timezone.utc)

    await repo.mark_sent(
        campaign_id="campaign-1",
        result=SendResult(provider="resend", message_id="msg-1", raw={"id": "msg-1"}),
        sent_at=sent_at,
    )

    call = pool.execute_calls[0]
    assert "status = 'sent'" in call["query"]
    assert call["args"][:3] == ("campaign-1", sent_at, "msg-1")
    assert json.loads(call["args"][3]) == {
        "send_provider": "resend",
        "send_raw": {"id": "msg-1"},
    }


@pytest.mark.asyncio
async def test_campaign_repository_records_open_webhook_event():
    pool = _Pool()
    repo = PostgresCampaignRepository(pool)
    occurred_at = datetime(2026, 5, 1, 15, tzinfo=timezone.utc)

    await repo.record_webhook_event(
        WebhookEvent(
            provider="resend",
            event_type="opened",
            message_id="msg-1",
            email="buyer@example.com",
            occurred_at=occurred_at,
            payload={"type": "email.opened"},
        )
    )

    update_call, audit_call = pool.execute_calls
    assert "opened_at = COALESCE(opened_at, $2)" in update_call["query"]
    assert update_call["args"][:2] == ("msg-1", occurred_at)
    assert "INSERT INTO campaign_audit_log" in audit_call["query"]
    assert audit_call["args"][0] == "webhook_opened"


@pytest.mark.asyncio
async def test_sequence_repository_queues_followup_step():
    pool = _Pool()
    repo = PostgresCampaignSequenceRepository(pool)
    queued_at = datetime(2026, 5, 1, 15, tzinfo=timezone.utc)

    campaign_id = await repo.queue_sequence_step(
        sequence={
            "id": "sequence-1",
            "company_name": "Acme",
            "batch_id": "batch-1",
            "recipient_email": "buyer@example.com",
            "current_step": 1,
            "max_steps": 4,
        },
        content={
            "subject": "Following up",
            "body": "<p>Hi</p>",
            "cta": "Book time",
            "step_number": 2,
            "target_mode": "vendor_retention",
            "product_category": "CRM",
            "angle_reasoning": "Opened previous email.",
        },
        from_email="seller@example.com",
        queued_at=queued_at,
    )

    assert campaign_id == "campaign-1"
    call = pool.fetchval_calls[0]
    assert "INSERT INTO b2b_campaigns" in call["query"]
    assert call["args"][:12] == (
        "sequence-1",
        "Acme",
        "batch-1",
        "Following up",
        "<p>Hi</p>",
        "Book time",
        2,
        "buyer@example.com",
        "seller@example.com",
        "vendor_retention",
        "CRM",
        json.dumps(
            {
                "angle_reasoning": "Opened previous email.",
                "sequence_context": {"current_step": 1, "max_steps": 4},
            },
            separators=(",", ":"),
        ),
    )
    assert call["args"][12] == queued_at


@pytest.mark.asyncio
async def test_suppression_repository_checks_and_upserts_email_suppression():
    pool = _Pool()
    pool.fetchrow_result = {"id": "sup-1"}
    repo = PostgresSuppressionRepository(pool)

    assert await repo.is_suppressed(email="buyer@example.com") is True
    await repo.add_suppression(
        email="buyer@example.com",
        reason="unsubscribe",
        source="webhook",
        campaign_id="campaign-1",
    )

    assert "LOWER(email) = LOWER($1)" in pool.fetchrow_calls[0]["query"]
    assert "ON CONFLICT (LOWER(email))" in pool.execute_calls[0]["query"]
    assert pool.execute_calls[0]["args"][:4] == (
        "buyer@example.com",
        "unsubscribe",
        "webhook",
        "campaign-1",
    )


@pytest.mark.asyncio
async def test_audit_sink_writes_campaign_audit_log():
    pool = _Pool()
    sink = PostgresCampaignAuditSink(pool)

    await sink.record(
        "sent",
        campaign_id="campaign-1",
        sequence_id="sequence-1",
        metadata={"provider": "resend"},
    )

    call = pool.execute_calls[0]
    assert "INSERT INTO campaign_audit_log" in call["query"]
    assert call["args"] == (
        "campaign-1",
        "sequence-1",
        "sent",
        json.dumps({"provider": "resend"}, separators=(",", ":")),
    )
