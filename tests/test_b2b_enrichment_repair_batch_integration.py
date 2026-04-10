import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from atlas_brain.autonomous.tasks import b2b_enrichment_repair as repair_mod
from atlas_brain.autonomous.tasks.b2b_scrape_intake import _INSERT_SQL
from atlas_brain.config import settings


def _configure_repair_batch_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "anthropic_batch_enabled", True, raising=False)
    monkeypatch.setattr(
        settings.b2b_churn,
        "enrichment_repair_anthropic_batch_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        settings.b2b_churn,
        "enrichment_repair_model",
        "anthropic/claude-haiku-4-5",
        raising=False,
    )
    monkeypatch.setattr(settings.b2b_churn, "enrichment_repair_max_tokens", 512, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "review_truncate_length", 3000, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "enrichment_low_fidelity_enabled", False, raising=False)
    monkeypatch.setattr(settings.llm, "anthropic_api_key", "test-anthropic-key", raising=False)


def _baseline_enrichment() -> dict:
    return {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "contract_renewal_mentioned": True,
        },
        "urgency_score": 7,
        "would_recommend": False,
        "pain_category": "overall_dissatisfaction",
        "buyer_authority": {"role_type": "unknown", "buying_stage": "unknown"},
        "timeline": {"decision_timeline": "unknown"},
        "contract_context": {"contract_value_signal": "unknown"},
        "reviewer_context": {"role_level": "unknown", "decision_maker": False},
        "sentiment_trajectory": {"direction": "declining"},
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "feature_gaps": [],
        "event_mentions": [],
        "salience_flags": [],
        "evidence_spans": [],
        "replacement_mode": "none",
        "operating_model_shift": "none",
        "productivity_delta_claim": "unknown",
        "org_pressure_type": "none",
        "evidence_map_hash": "integration-test-hash",
        "enrichment_schema_version": 3,
    }


def _review_payload(*, review_id, import_batch_id) -> dict:
    return {
        "id": review_id,
        "import_batch_id": import_batch_id,
        "dedup_key": f"repair-test-{review_id}",
        "source": "reddit",
        "source_url": f"https://example.com/review/{review_id}",
        "source_review_id": str(review_id),
        "vendor_name": "Acme Analytics",
        "product_name": "Acme Analytics",
        "product_category": "analytics",
        "rating": 2,
        "rating_max": 5,
        "summary": "Renewal pricing triggered an evaluation",
        "review_text": (
            "Our renewal price jumped sharply, support was not helpful, and our operations "
            "leader started evaluating HubSpot before the contract renews next quarter."
        ),
        "pros": "",
        "cons": "",
        "reviewer_name": "Taylor Reviewer",
        "reviewer_title": "Revenue Operations Manager",
        "reviewer_company": "Northwind",
        "reviewer_company_norm": "northwind",
        "company_size_raw": "51-200",
        "reviewer_industry": "software",
        "reviewed_at": datetime.now(timezone.utc),
        "raw_metadata": {
            "source_weight": 0.9,
            "relevance_score": 0.8,
            "author_churn_score": 0.6,
        },
        "parser_version": "integration-test",
        "content_type": "review",
    }


async def _insert_review(db_pool, tracker: dict, payload: dict) -> None:
    await db_pool.execute(
        _INSERT_SQL,
        payload["dedup_key"],
        payload["source"],
        payload["source_url"],
        payload["source_review_id"],
        payload["vendor_name"],
        payload["product_name"],
        payload["product_category"],
        payload["rating"],
        payload["rating_max"],
        payload["summary"],
        payload["review_text"],
        payload["pros"],
        payload["cons"],
        payload["reviewer_name"],
        payload["reviewer_title"],
        payload["reviewer_company"],
        payload["reviewer_company_norm"],
        payload["company_size_raw"],
        payload["reviewer_industry"],
        payload["reviewed_at"],
        str(payload["import_batch_id"]),
        json.dumps(payload["raw_metadata"]),
        payload["parser_version"],
        payload["content_type"],
        None,
        0,
        payload["raw_metadata"]["relevance_score"],
        payload["raw_metadata"]["author_churn_score"],
        payload["raw_metadata"]["source_weight"],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "pending",
        payload["id"],
    )
    tracker["review_ids"].append(payload["id"])


async def _promote_review_to_repair_candidate(db_pool, *, review_id, enrichment: dict) -> None:
    await db_pool.execute(
        """
        UPDATE b2b_reviews
        SET enrichment = $2::jsonb,
            enrichment_status = 'enriched',
            enriched_at = NOW(),
            enrichment_model = 'anthropic/claude-haiku-4-5',
            enrichment_repair_status = 'repairing'
        WHERE id = $1
        """,
        review_id,
        json.dumps(enrichment),
    )


async def _fetch_repair_row(db_pool, review_id):
    return await db_pool.fetchrow(
        """
        SELECT id, vendor_name, product_name, product_category,
               source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment,
               enrichment_repair_attempts
        FROM b2b_reviews
        WHERE id = $1
        """,
        review_id,
    )


async def _insert_batch_job(
    db_pool,
    tracker: dict,
    *,
    batch_id,
    status: str,
    provider_batch_id: str,
) -> None:
    await db_pool.execute(
        """
        INSERT INTO anthropic_message_batches (
            id, stage_id, task_name, run_id, status, total_items,
            submitted_items, cache_prefiltered_items, fallback_single_call_items,
            completed_items, failed_items, metadata, provider_batch_id
        ) VALUES (
            $1::uuid, 'b2b_enrichment_repair.extraction', 'b2b_enrichment_repair', NULL, $2, 1,
            1, 0, 0,
            0, 0, '{}'::jsonb, $3
        )
        """,
        batch_id,
        status,
        provider_batch_id,
    )
    tracker["batch_ids"].append(batch_id)


async def _insert_batch_item(
    db_pool,
    *,
    batch_id,
    review_id,
    vendor_name: str,
    status: str,
    response_text: str | None = None,
) -> None:
    await db_pool.execute(
        """
        INSERT INTO anthropic_message_batch_items (
            id, batch_id, custom_id, stage_id, artifact_type, artifact_id,
            vendor_name, status, cache_prefiltered, fallback_single_call,
            response_text, input_tokens, billable_input_tokens, cached_tokens,
            cache_write_tokens, output_tokens, cost_usd, provider_request_id,
            error_text, request_metadata, completed_at
        ) VALUES (
            gen_random_uuid(), $1::uuid, $2, 'b2b_enrichment_repair.extraction', 'review_enrichment_repair', $3,
            $4, $5, FALSE, FALSE,
            $6, 0, 0, 0,
            0, 0, 0, NULL,
            NULL, $7::jsonb,
            CASE WHEN $5 = 'pending' THEN NULL ELSE NOW() END
        )
        """,
        batch_id,
        repair_mod._repair_batch_custom_id(review_id),
        str(review_id),
        vendor_name,
        status,
        response_text,
        json.dumps({"review_id": str(review_id)}),
    )


@pytest_asyncio.fixture
async def seeded_repair_artifacts(db_pool):
    tracker = {"review_ids": [], "batch_ids": []}
    try:
        yield tracker
    finally:
        if tracker["batch_ids"]:
            await db_pool.execute(
                "DELETE FROM anthropic_message_batches WHERE id = ANY($1::uuid[])",
                tracker["batch_ids"],
            )
        if tracker["review_ids"]:
            await db_pool.execute(
                "DELETE FROM b2b_reviews WHERE id = ANY($1::uuid[])",
                tracker["review_ids"],
            )


@pytest.mark.asyncio
async def test_repair_rows_resets_repair_status_when_reused_batch_item_is_pending(
    db_pool,
    seeded_repair_artifacts,
    monkeypatch,
):
    _configure_repair_batch_settings(monkeypatch)

    import_batch_id = uuid4()
    review_id = uuid4()
    await _insert_review(
        db_pool,
        seeded_repair_artifacts,
        _review_payload(review_id=review_id, import_batch_id=import_batch_id),
    )
    await _promote_review_to_repair_candidate(
        db_pool,
        review_id=review_id,
        enrichment=_baseline_enrichment(),
    )
    row = await _fetch_repair_row(db_pool, review_id)

    batch_id = uuid4()
    await _insert_batch_job(
        db_pool,
        seeded_repair_artifacts,
        batch_id=batch_id,
        status="submitted",
        provider_batch_id=f"msgbatch_{uuid4().hex[:12]}",
    )
    await _insert_batch_item(
        db_pool,
        batch_id=batch_id,
        review_id=review_id,
        vendor_name=row["vendor_name"],
        status="pending",
    )

    result = await repair_mod._repair_rows(
        [row],
        settings.b2b_churn,
        db_pool,
        task=SimpleNamespace(metadata={}),
    )
    stored = await db_pool.fetchrow(
        """
        SELECT enrichment_repair_status, enrichment_repair_attempts
        FROM b2b_reviews
        WHERE id = $1
        """,
        review_id,
    )

    assert result["anthropic_batch_reused_pending_items"] == 1
    assert result["anthropic_batch_rows_deferred"] == 1
    assert stored["enrichment_repair_status"] is None
    assert stored["enrichment_repair_attempts"] == 0


@pytest.mark.asyncio
async def test_repair_rows_reused_completed_cache_hit_preserves_usage_metrics(
    db_pool,
    seeded_repair_artifacts,
    monkeypatch,
):
    _configure_repair_batch_settings(monkeypatch)

    import_batch_id = uuid4()
    review_id = uuid4()
    await _insert_review(
        db_pool,
        seeded_repair_artifacts,
        _review_payload(review_id=review_id, import_batch_id=import_batch_id),
    )
    await _promote_review_to_repair_candidate(
        db_pool,
        review_id=review_id,
        enrichment=_baseline_enrichment(),
    )
    row = await _fetch_repair_row(db_pool, review_id)

    batch_id = uuid4()
    await _insert_batch_job(
        db_pool,
        seeded_repair_artifacts,
        batch_id=batch_id,
        status="ended",
        provider_batch_id=f"msgbatch_{uuid4().hex[:12]}",
    )
    await _insert_batch_item(
        db_pool,
        batch_id=batch_id,
        review_id=review_id,
        vendor_name=row["vendor_name"],
        status="cache_hit",
        response_text=json.dumps({}),
    )

    result = await repair_mod._repair_rows(
        [row],
        settings.b2b_churn,
        db_pool,
        task=SimpleNamespace(metadata={}),
    )
    stored = await db_pool.fetchrow(
        """
        SELECT enrichment_repair_status, enrichment_repair_attempts
        FROM b2b_reviews
        WHERE id = $1
        """,
        review_id,
    )

    assert result["anthropic_batch_reused_completed_items"] == 1
    assert result["exact_cache_hits"] == 1
    assert result["generated"] == 0
    assert stored["enrichment_repair_status"] == "shadowed"
    assert stored["enrichment_repair_attempts"] == 1
