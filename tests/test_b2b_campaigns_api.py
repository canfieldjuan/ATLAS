from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.api import b2b_campaigns as mod


def test_campaign_activity_timestamp_sql_uses_real_campaign_columns():
    assert mod._campaign_activity_timestamp_sql("bc") == (
        "COALESCE(bc.clicked_at, bc.opened_at, bc.sent_at, bc.approved_at, bc.created_at)"
    )


@pytest.mark.asyncio
async def test_review_candidates_endpoint_uses_helper(monkeypatch):
    captured = {}

    async def _fake_list_candidates(
        pool,
        *,
        min_score,
        max_score,
        limit,
        vendor_filter=None,
        company_filter=None,
        qualified_only=True,
        ignore_recent_dedup=False,
    ):
        captured.update({
            "pool": pool,
            "min_score": min_score,
            "max_score": max_score,
            "limit": limit,
            "vendor_filter": vendor_filter,
            "company_filter": company_filter,
            "qualified_only": qualified_only,
            "ignore_recent_dedup": ignore_recent_dedup,
        })
        return {"count": 1, "candidates": [{"company_name": "Acme Co"}], "summary": {"qualified": 1}}

    sentinel_pool = object()
    monkeypatch.setattr(mod, "_pool_or_503", lambda: sentinel_pool)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_campaign_generation.list_churning_company_review_candidates",
        _fake_list_candidates,
    )

    result = await mod.review_candidates(
        limit=25,
        min_score=56,
        max_score=68,
        vendor="CrowdStrike",
        company="Pax8",
        qualified_only=False,
        ignore_recent_dedup=True,
    )

    assert result["count"] == 1
    assert captured == {
        "pool": sentinel_pool,
        "min_score": 56,
        "max_score": 68,
        "limit": 25,
        "vendor_filter": "CrowdStrike",
        "company_filter": "Pax8",
        "qualified_only": False,
        "ignore_recent_dedup": True,
    }


@pytest.mark.asyncio
async def test_review_candidates_summary_returns_summary_only(monkeypatch):
    async def _fake_list_candidates(
        pool,
        *,
        min_score,
        max_score,
        limit,
        vendor_filter=None,
        company_filter=None,
        qualified_only=True,
        ignore_recent_dedup=False,
    ):
        return {
            "count": 3,
            "candidates": [{"company_name": "Acme Co"}],
            "summary": {"qualified": 2, "unqualified": 1},
        }

    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_campaign_generation.list_churning_company_review_candidates",
        _fake_list_candidates,
    )

    result = await mod.review_candidates_summary(
        min_score=55,
        max_score=69,
        vendor=None,
        company=None,
        ignore_recent_dedup=False,
    )

    assert result == {
        "count": 3,
        "summary": {"qualified": 2, "unqualified": 1},
    }


class _CampaignPool:
    def __init__(self, campaign_row):
        self.campaign_row = campaign_row
        self.execute_calls = []

    async def fetchrow(self, query, *args):
        if "SELECT bc.*, cs.company_context" in query:
            return dict(self.campaign_row)
        if "RETURNING id" in query:
            return {"id": self.campaign_row["id"]}
        return None

    async def fetchval(self, query, *args):
        if "SELECT recipient_email FROM campaign_sequences" in query:
            return self.campaign_row.get("seq_recipient")
        if "SELECT status FROM b2b_campaigns" in query:
            return self.campaign_row.get("status")
        return None

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"


class _FallbackReasoningView:
    def consumer_context(self, consumer):
        assert consumer == "campaign"
        return {
            "anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "a customer hit a $200k/year renewal decision in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    },
                ],
            },
            "witness_highlights": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "a customer hit a $200k/year renewal decision in Q2",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Freshdesk",
                    "pain_category": "pricing",
                },
            ],
            "reference_ids": {"witness_ids": ["witness:r1:0"]},
        }

    def filtered_consumer_context(self, consumer):
        return self.consumer_context(consumer)


@pytest.mark.asyncio
async def test_approve_campaign_blocks_generic_copy_when_revalidation_fails(monkeypatch):
    campaign_id = uuid4()
    pool = _CampaignPool({
        "id": campaign_id,
        "status": "draft",
        "sequence_id": None,
        "step_number": 1,
        "recipient_email": "owner@example.com",
        "subject": "Renewal pressure",
        "body": "<p>There is broad market pressure and multiple teams are feeling it.</p>",
        "channel": "email_followup",
        "metadata": {
            "reasoning_anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "a customer hit a $200k/year renewal decision in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    },
                ],
            },
            "reasoning_witness_highlights": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "a customer hit a $200k/year renewal decision in Q2",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Freshdesk",
                    "pain_category": "pricing",
                },
            ],
            "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
        },
        "company_context": {},
    })

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod.settings.b2b_campaign, "revalidate_before_manual_approval", True)
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())
    monkeypatch.setattr(mod, "record_attempt", AsyncMock())
    monkeypatch.setattr(mod, "emit_event", AsyncMock())

    with pytest.raises(mod.HTTPException) as exc:
        await mod.approve_campaign(str(campaign_id))

    assert exc.value.status_code == 409
    assert "witness-backed anchor" in str(exc.value.detail)
    assert any("SET metadata = $1::jsonb" in call[0] for call in pool.execute_calls)


@pytest.mark.asyncio
async def test_queue_campaign_for_send_blocks_generic_copy_when_revalidation_fails(monkeypatch):
    campaign_id = uuid4()
    pool = _CampaignPool({
        "id": campaign_id,
        "status": "draft",
        "sequence_id": uuid4(),
        "step_number": 1,
        "recipient_email": None,
        "seq_recipient": "owner@example.com",
        "subject": "Renewal pressure",
        "body": "<p>The market is shifting and teams should pay attention.</p>",
        "channel": "email_followup",
        "metadata": {
            "reasoning_anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "a customer hit a $200k/year renewal decision in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    },
                ],
            },
            "reasoning_witness_highlights": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "a customer hit a $200k/year renewal decision in Q2",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Freshdesk",
                    "pain_category": "pricing",
                },
            ],
            "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
        },
        "company_context": {},
    })

    async def _not_suppressed(pool, email):
        return None

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod.settings.b2b_campaign, "revalidate_before_queue_send", True)
    monkeypatch.setattr("atlas_brain.autonomous.tasks.campaign_suppression.is_suppressed", _not_suppressed)
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())
    monkeypatch.setattr(mod, "record_attempt", AsyncMock())
    monkeypatch.setattr(mod, "emit_event", AsyncMock())

    with pytest.raises(mod.HTTPException) as exc:
        await mod.queue_campaign_for_send(str(campaign_id), body=mod.ApproveQueueBody(recipient_email="owner@example.com"))

    assert exc.value.status_code == 409
    assert "timing or numeric anchor" in str(exc.value.detail) or "witness-backed anchor" in str(exc.value.detail)
    assert any("SET metadata = $1::jsonb" in call[0] for call in pool.execute_calls)


@pytest.mark.asyncio
async def test_approve_campaign_uses_synthesis_fallback_and_records_success(monkeypatch):
    campaign_id = uuid4()
    pool = _CampaignPool({
        "id": campaign_id,
        "vendor_name": "Slack",
        "company_name": "Slack",
        "status": "draft",
        "sequence_id": None,
        "step_number": 1,
        "recipient_email": "owner@example.com",
        "subject": "Renewal pressure",
        "body": "<p>The Q2 renewal now carries a $200k/year pricing issue and Freshdesk is showing up.</p>",
        "channel": "email_followup",
        "metadata": {
            "generation_audit": {"status": "succeeded"},
        },
        "company_context": None,
    })

    async def _fake_reasoning_view(pool, vendor_name):
        assert vendor_name == "Slack"
        return _FallbackReasoningView()

    record_attempt = AsyncMock()

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod.settings.b2b_campaign, "revalidate_before_manual_approval", True)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.load_best_reasoning_view",
        _fake_reasoning_view,
    )
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())
    monkeypatch.setattr(mod, "record_attempt", record_attempt)
    monkeypatch.setattr(mod, "emit_event", AsyncMock())

    result = await mod.approve_campaign(str(campaign_id))

    assert result["ok"] is True
    assert any("SET metadata = $1::jsonb" in call[0] for call in pool.execute_calls)
    metadata_update = next(call for call in pool.execute_calls if "SET metadata = $1::jsonb" in call[0])
    assert "reasoning_anchor_examples" in metadata_update[1][0]
    assert "campaign_proof_terms" in metadata_update[1][0]
    assert "latest_specificity_audit" in metadata_update[1][0]
    record_attempt.assert_awaited_once()
    kwargs = record_attempt.await_args.kwargs
    assert kwargs["stage"] == "manual_approval"
    assert kwargs["status"] == "succeeded"


@pytest.mark.asyncio
async def test_approve_campaign_blocks_report_tier_language(monkeypatch):
    campaign_id = uuid4()
    pool = _CampaignPool({
        "id": campaign_id,
        "status": "draft",
        "sequence_id": None,
        "step_number": 1,
        "recipient_email": "owner@example.com",
        "subject": "Renewal pressure",
        "body": "<p>Get the dashboard before renewal.</p>",
        "cta": "Open the dashboard",
        "channel": "email_cold",
        "target_mode": "vendor_retention",
        "metadata": {
            "tier": "report",
            "generation_audit": {"status": "succeeded"},
        },
        "company_context": None,
    })

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod.settings.b2b_campaign, "revalidate_before_manual_approval", True)
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())
    monkeypatch.setattr(mod, "record_attempt", AsyncMock())
    monkeypatch.setattr(mod, "emit_event", AsyncMock())

    with pytest.raises(mod.HTTPException) as exc:
        await mod.approve_campaign(str(campaign_id))

    assert exc.value.status_code == 409
    assert "report_tier_language:dashboard" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_list_campaigns_surfaces_quality_summary_from_metadata(monkeypatch):
    created_at = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)

    class Pool:
        async def fetch(self, *_args):
            return [{
                "id": uuid4(),
                "company_name": "Acme Co",
                "vendor_name": "Slack",
                "product_category": "Communication",
                "opportunity_score": 81,
                "urgency_score": 7.4,
                "channel": "email_followup",
                "subject": "Renewal pressure",
                "body": "<p>Body</p>",
                "cta": "Book time",
                "status": "draft",
                "batch_id": "batch_1",
                "llm_model": "anthropic/claude-sonnet-4-5",
                "created_at": created_at,
                "approved_at": None,
                "sent_at": None,
                "partner_id": None,
                "industry": "Nonprofit",
                "recipient_email": None,
                "metadata": {
                    "latest_specificity_audit": {
                        "status": "fail",
                        "blocking_issues": ["content does not reference any witness-backed anchor despite anchors being available"],
                        "warnings": ["timing anchor exists but content does not mention the live trigger window"],
                    },
                },
                "company_context": {
                    "target_persona": "revops",
                    "company": "Acme Co",
                    "role_type": "revops",
                    "buying_stage": "evaluation",
                    "urgency": "high",
                    "contract_end": "2026-06-30",
                    "decision_timeline": "60 days",
                    "pain_categories": ["pricing"],
                    "competitors_considering": ["HubSpot"],
                    "key_quotes": ["We need better renewal controls."],
                    "reasoning_scope_summary": {
                        "selection_strategy": "vendor_facet_packet_v1",
                        "witnesses_in_scope": 8,
                        "witness_mix": {"timing": 2},
                    },
                    "reasoning_atom_context": {
                        "top_theses": [
                            {
                                "wedge": "pricing",
                                "summary": "Pricing pressure is driving re-evaluation.",
                                "why_now": "Budget owners are active before renewal.",
                                "confidence": "high",
                            },
                        ],
                        "timing_windows": [
                            {
                                "window_type": "renewal",
                                "anchor": "Q2 renewal",
                                "urgency": "high",
                                "recommended_action": "Engage before procurement freeze.",
                            },
                        ],
                        "account_signals": [
                            {
                                "company": "Acme Co",
                                "buying_stage": "evaluation",
                                "competitor_context": "HubSpot",
                                "primary_pain": "pricing",
                            },
                        ],
                        "coverage_limits": ["thin_account_signals"],
                    },
                    "reasoning_delta_summary": {
                        "changed": True,
                        "new_timing_windows": ["Q2 renewal"],
                        "new_account_signals": ["Acme Co"],
                    },
                },
            }]

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.list_campaigns(
        status=None,
        company=None,
        vendor=None,
        channel=None,
        batch_id=None,
        limit=50,
        user=None,
    )

    assert result["campaigns"][0]["quality_status"] == "fail"
    assert result["campaigns"][0]["blocker_count"] == 1
    assert result["campaigns"][0]["warning_count"] == 1
    assert result["campaigns"][0]["latest_error_summary"].startswith("content does not reference")
    assert result["campaigns"][0]["failure_explanation"]["primary_blocker"].startswith("content does not reference")
    assert result["campaigns"][0]["failure_explanation"]["boundary"] is None
    assert result["campaigns"][0]["reasoning_scope_summary"]["selection_strategy"] == "vendor_facet_packet_v1"
    assert result["campaigns"][0]["reasoning_atom_context"]["top_theses"][0]["summary"] == "Pricing pressure is driving re-evaluation."
    assert result["campaigns"][0]["reasoning_atom_context"]["account_signals"][0]["contract_end"] == "2026-06-30"
    assert result["campaigns"][0]["reasoning_delta_summary"]["new_account_signals"] == ["Acme Co"]


@pytest.mark.asyncio
async def test_review_queue_surfaces_quality_summary_from_metadata(monkeypatch):
    created_at = datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc)

    class Pool:
        async def fetch(self, *_args):
            return [{
                "id": uuid4(),
                "company_name": "Acme Co",
                "vendor_name": "Slack",
                "channel": "email_followup",
                "subject": "Renewal pressure",
                "body": "<p>Body</p>",
                "cta": "Book time",
                "status": "draft",
                "step_number": 1,
                "recipient_email": "owner@example.com",
                "partner_id": None,
                "created_at": created_at,
                "metadata": {
                    "latest_specificity_audit": {
                        "status": "fail",
                        "blocking_issues": ["content does not reference any witness-backed anchor despite anchors being available"],
                        "warnings": ["timing anchor exists but content does not mention the live trigger window"],
                    },
                },
                "seq_recipient": "owner@example.com",
                "open_count": 0,
                "click_count": 0,
                "seq_status": "active",
                "current_step": 1,
                "max_steps": 3,
                "company_context": {
                    "target_persona": "revops",
                    "company": "Acme Co",
                    "role_type": "revops",
                    "buying_stage": "evaluation",
                    "urgency": "high",
                    "contract_end": "2026-06-30",
                    "decision_timeline": "60 days",
                    "pain_categories": ["pricing"],
                    "competitors_considering": ["HubSpot"],
                    "key_quotes": ["We need better renewal controls."],
                    "reasoning_scope_summary": {
                        "selection_strategy": "vendor_facet_packet_v1",
                        "witnesses_in_scope": 8,
                    },
                    "reasoning_atom_context": {
                        "top_theses": [
                            {
                                "wedge": "pricing",
                                "summary": "Pricing pressure is driving re-evaluation.",
                                "why_now": "Budget owners are active before renewal.",
                                "confidence": "high",
                            },
                        ],
                        "timing_windows": [
                            {
                                "window_type": "renewal",
                                "anchor": "Q2 renewal",
                                "urgency": "high",
                                "recommended_action": "Engage before procurement freeze.",
                            },
                        ],
                        "account_signals": [
                            {
                                "company": "Acme Co",
                                "buying_stage": "evaluation",
                                "competitor_context": "HubSpot",
                                "primary_pain": "pricing",
                            },
                        ],
                        "coverage_limits": ["thin_account_signals"],
                    },
                    "reasoning_delta_summary": {
                        "changed": True,
                        "new_timing_windows": ["Q2 renewal"],
                        "new_account_signals": ["Acme Co"],
                    },
                },
                "partner_name": None,
                "product_name": None,
                "is_suppressed": 0,
                "prospect_first_name": "Alex",
                "prospect_last_name": "Kim",
                "prospect_title": "RevOps Lead",
                "prospect_seniority": "manager",
                "prospect_email_status": "valid",
            }]

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.review_queue(
        limit=50,
        offset=0,
        status="draft",
        include_prospects=True,
    )

    assert result["drafts"][0]["quality_status"] == "fail"
    assert result["drafts"][0]["blocker_count"] == 1
    assert result["drafts"][0]["warning_count"] == 1
    assert result["drafts"][0]["latest_error_summary"].startswith("content does not reference")
    assert result["drafts"][0]["failure_explanation"]["primary_blocker"].startswith("content does not reference")
    assert result["drafts"][0]["failure_explanation"]["blocking_issues"][0].startswith("content does not reference")
    assert result["drafts"][0]["reasoning_scope_summary"]["selection_strategy"] == "vendor_facet_packet_v1"
    assert result["drafts"][0]["reasoning_atom_context"]["timing_windows"][0]["anchor"] == "Q2 renewal"
    assert result["drafts"][0]["reasoning_atom_context"]["account_signals"][0]["quote"] == "We need better renewal controls."
    assert result["drafts"][0]["reasoning_delta_summary"]["changed"] is True


@pytest.mark.asyncio
async def test_campaign_stats_returns_quality_rollup(monkeypatch):
    class Pool:
        async def fetch(self, query, *_args):
            if "GROUP BY bc.status" in query:
                return [{"status": "draft", "cnt": 2}, {"status": "sent", "cnt": 1}]
            if "GROUP BY bc.channel" in query:
                return [{"channel": "email_followup", "cnt": 2}]
            if "GROUP BY bc.vendor_name" in query:
                return [{"vendor_name": "Slack", "cnt": 2}]
            if "GROUP BY boundary" in query:
                return [{"boundary": "manual_approval", "cnt": 2}]
            if "GROUP BY blocker.reason" in query:
                return [{"reason": "content does not reference any witness-backed anchor despite anchors being available", "cnt": 2}]
            return []

        async def fetchrow(self, query, *_args):
            if "quality_pass" in query:
                return {
                    "quality_pass": 1,
                    "quality_fail": 2,
                    "quality_missing": 0,
                    "blocker_total": 3,
                    "warning_total": 4,
                }
            return None

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.campaign_stats(user=None)

    assert result["total"] == 3
    assert result["quality"]["pass"] == 1
    assert result["quality"]["fail"] == 2
    assert result["quality"]["missing"] == 0
    assert result["quality"]["blocker_total"] == 3
    assert result["quality"]["warning_total"] == 4
    assert result["quality"]["by_boundary"]["manual_approval"] == 2
    assert result["quality"]["top_blockers"][0]["reason"].startswith("content does not reference")


@pytest.mark.asyncio
async def test_campaign_quality_trends_returns_series(monkeypatch):
    class Pool:
        async def fetch(self, query, *_args):
            if "WITH blocker_rows AS" in query:
                return [
                    {"day": "2026-03-28", "reason": "missing_exact_proof_term", "cnt": 2},
                    {"day": "2026-03-29", "reason": "missing_exact_proof_term", "cnt": 1},
                    {"day": "2026-03-29", "reason": "report_tier_language:dashboard", "cnt": 1},
                ]
            if "SUM(" in query and "blocker_total" in query:
                return [
                    {"day": "2026-03-28", "blocker_total": 2},
                    {"day": "2026-03-29", "blocker_total": 2},
                ]
            return []

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.campaign_quality_trends(days=7, top_n=3, user=None)

    assert result["days"] == 7
    assert result["top_n"] == 3
    assert result["top_blockers"][0] == {"reason": "missing_exact_proof_term", "count": 3}
    assert result["series"][0]["day"] == "2026-03-28"
    assert result["totals_by_day"][1]["blocker_total"] == 2


@pytest.mark.asyncio
async def test_campaign_quality_diagnostics_returns_grouped_failure_explanations(monkeypatch):
    class Pool:
        async def fetch(self, query, *_args):
            if "COALESCE(bc.metadata->'latest_specificity_audit'->>'boundary', 'missing') AS boundary" in query:
                return [{"boundary": "manual_approval", "cnt": 2}]
            if "failure_explanation'->>'cause_type'" in query:
                return [{"cause_type": "content_ignored_available_evidence", "cnt": 2}]
            if "failure_explanation'->>'primary_blocker'" in query:
                return [{"reason": "missing_exact_proof_term", "cnt": 2}]
            if "jsonb_array_elements_text" in query:
                return [{"input": "reasoning_reference_ids", "cnt": 1}]
            if "COALESCE(bc.target_mode, bc.metadata->>'target_mode', 'unknown') AS target_mode" in query:
                return [{"target_mode": "vendor_retention", "cnt": 2}]
            if "COALESCE(bc.vendor_name, 'unknown') AS vendor_name" in query:
                return [{"vendor_name": "Slack", "cnt": 2}]
            return []

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.campaign_quality_diagnostics(days=7, top_n=5, user=None)

    assert result["days"] == 7
    assert result["top_n"] == 5
    assert result["by_cause_type"][0]["cause_type"] == "content_ignored_available_evidence"
    assert result["top_primary_blockers"][0]["reason"] == "missing_exact_proof_term"
    assert result["top_missing_inputs"][0]["input"] == "reasoning_reference_ids"
    assert result["by_target_mode"][0]["target_mode"] == "vendor_retention"
    assert result["top_vendors"][0]["vendor_name"] == "Slack"


@pytest.mark.asyncio
async def test_review_queue_summary_returns_quality_rollup(monkeypatch):
    class Pool:
        async def fetchrow(self, query, *_args):
            if "pending_review" in query:
                return {
                    "pending_review": 3,
                    "pending_recipient": 1,
                    "ready_to_send": 2,
                    "suppressed": 0,
                    "oldest_draft_age_hours": 4.2,
                }
            if "quality_pass" in query:
                return {
                    "quality_pass": 1,
                    "quality_fail": 2,
                    "quality_missing": 0,
                    "blocker_total": 3,
                }
            return None

        async def fetch(self, query, *_args):
            if "GROUP BY ap.name" in query:
                return [{"name": "Partner A", "count": 2}]
            if "GROUP BY boundary" in query:
                return [{"boundary": "manual_approval", "count": 2}]
            if "GROUP BY blocker.reason" in query:
                return [{"reason": "content omits a concrete timing or numeric anchor even though one is available", "count": 1}]
            return []

    monkeypatch.setattr(mod, "_pool_or_503", lambda: Pool())

    result = await mod.review_queue_summary()

    assert result["pending_review"] == 3
    assert result["quality_pass"] == 1
    assert result["quality_fail"] == 2
    assert result["quality_missing"] == 0
    assert result["blocker_total"] == 3
    assert result["by_boundary"][0]["boundary"] == "manual_approval"
    assert result["top_blockers"][0]["reason"].startswith("content omits")
