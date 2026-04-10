import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.autonomous.tasks._b2b_batch_utils import resolve_anthropic_batch_llm
from atlas_brain.autonomous.tasks.b2b_scrape_intake import _INSERT_SQL
from atlas_brain.config import settings
from atlas_brain.services.b2b.cache_runner import (
    prepare_b2b_exact_stage_request,
    store_b2b_exact_stage_text,
)
from atlas_brain.services.b2b.llm_exact_cache import compute_cache_key
from atlas_brain.skills import get_skill_registry


def _configure_batch_settings(monkeypatch, *, tier2_model: str | None = None) -> None:
    monkeypatch.setattr(settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "anthropic_batch_enabled", True, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "enrichment_anthropic_batch_enabled", True, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "llm_exact_cache_enabled", True, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "enrichment_local_only", False, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "openrouter_api_key", "test-openrouter-key", raising=False)
    monkeypatch.setattr(
        settings.b2b_churn,
        "enrichment_openrouter_model",
        "anthropic/claude-haiku-4-5",
        raising=False,
    )
    monkeypatch.setattr(
        settings.b2b_churn,
        "enrichment_tier2_openrouter_model",
        tier2_model if tier2_model is not None else "anthropic/claude-haiku-4-5",
        raising=False,
    )
    monkeypatch.setattr(settings.b2b_churn, "enrichment_tier1_max_tokens", 1024, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "enrichment_tier2_max_tokens", 1536, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "review_truncate_length", 3000, raising=False)
    monkeypatch.setattr(settings.b2b_churn, "enrichment_tier2_strict_sources", "", raising=False)
    monkeypatch.setattr(settings.llm, "anthropic_api_key", "test-anthropic-key", raising=False)


def _review_payload(
    *,
    review_id,
    import_batch_id,
    vendor_name: str = "Acme Analytics",
    product_name: str = "Acme Analytics",
    product_category: str = "analytics",
    rating: int = 5,
    summary: str = "Useful product overall",
    review_text: str = (
        "The platform has been easy to roll out across our revenue operations team, "
        "and it has been reliable enough that we are not considering a replacement."
    ),
    source: str = "g2",
) -> dict:
    return {
        "id": review_id,
        "import_batch_id": import_batch_id,
        "dedup_key": f"test-{review_id}",
        "source": source,
        "source_url": f"https://example.com/reviews/{review_id}",
        "source_review_id": str(review_id),
        "vendor_name": vendor_name,
        "product_name": product_name,
        "product_category": product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": summary,
        "review_text": review_text,
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


async def _insert_review(db_pool, tracker: dict, payload: dict) -> dict:
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
    return {
        "id": payload["id"],
        "vendor_name": payload["vendor_name"],
        "product_name": payload["product_name"],
        "product_category": payload["product_category"],
        "source": payload["source"],
        "raw_metadata": payload["raw_metadata"],
        "rating": payload["rating"],
        "rating_max": payload["rating_max"],
        "summary": payload["summary"],
        "review_text": payload["review_text"],
        "pros": payload["pros"],
        "cons": payload["cons"],
        "reviewer_title": payload["reviewer_title"],
        "reviewer_company": payload["reviewer_company"],
        "company_size_raw": payload["company_size_raw"],
        "reviewer_industry": payload["reviewer_industry"],
        "enrichment_attempts": 0,
        "content_type": payload["content_type"],
    }


async def _insert_batch_job(
    db_pool,
    tracker: dict,
    *,
    batch_id,
    stage_id: str,
    status: str,
    provider_batch_id: str,
    total_items: int = 1,
) -> None:
    await db_pool.execute(
        """
        INSERT INTO anthropic_message_batches (
            id, stage_id, task_name, run_id, status, total_items,
            submitted_items, cache_prefiltered_items, fallback_single_call_items,
            completed_items, failed_items, metadata, provider_batch_id
        ) VALUES (
            $1::uuid, $2, 'b2b_enrichment', NULL, $3, $4,
            $4, 0, 0,
            0, 0, '{}'::jsonb, $5
        )
        """,
        batch_id,
        stage_id,
        status,
        total_items,
        provider_batch_id,
    )
    tracker["batch_ids"].append(batch_id)


async def _insert_batch_item(
    db_pool,
    *,
    batch_id,
    custom_id: str,
    stage_id: str,
    artifact_type: str,
    artifact_id: str,
    vendor_name: str,
    status: str,
    response_text: str | None = None,
    request_metadata: dict | None = None,
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
            gen_random_uuid(), $1::uuid, $2, $3, $4, $5,
            $6, $7, FALSE, FALSE,
            $8, 0, 0, 0,
            0, 0, 0, NULL,
            NULL, $9::jsonb,
            CASE WHEN $7 = 'pending' THEN NULL ELSE NOW() END
        )
        """,
        batch_id,
        custom_id,
        stage_id,
        artifact_type,
        artifact_id,
        vendor_name,
        status,
        response_text,
        json.dumps(request_metadata or {}),
    )


async def _seed_tier2_exact_cache(
    db_pool,
    tracker: dict,
    *,
    row: dict,
    tier1: dict,
    tier2: dict,
) -> None:
    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    assert skill is not None

    payload = b2b_enrichment._build_classify_payload(
        row,
        settings.b2b_churn.review_truncate_length,
    )
    payload["tier1_specific_complaints"] = tier1.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1.get("quotable_phrases", [])
    system_prompt = b2b_enrichment._tier2_system_prompt_for_content_type(
        skill.content,
        payload.get("content_type"),
    )
    request = prepare_b2b_exact_stage_request(
        "b2b_enrichment.tier2",
        provider="openrouter",
        model=str(settings.b2b_churn.enrichment_tier2_openrouter_model),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        max_tokens=settings.b2b_churn.enrichment_tier2_max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    stored = await store_b2b_exact_stage_text(
        request,
        response_text=json.dumps(tier2),
        metadata={"tier": 2, "backend": "integration_test"},
        pool=db_pool,
    )
    assert stored is True
    tracker["cache_keys"].append(compute_cache_key(request.namespace, request.request_envelope))


@pytest_asyncio.fixture
async def seeded_artifacts(db_pool):
    tracker = {
        "review_ids": [],
        "batch_ids": [],
        "cache_keys": [],
    }
    try:
        yield tracker
    finally:
        if tracker["cache_keys"]:
            await db_pool.execute(
                "DELETE FROM b2b_llm_exact_cache WHERE cache_key = ANY($1::text[])",
                tracker["cache_keys"],
            )
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
async def test_enrich_batch_resets_row_to_pending_when_tier1_batch_item_is_pending(
    db_pool,
    seeded_artifacts,
    monkeypatch,
):
    _configure_batch_settings(monkeypatch)

    import_batch_id = uuid4()
    review_id = uuid4()
    row = await _insert_review(
        db_pool,
        seeded_artifacts,
        _review_payload(review_id=review_id, import_batch_id=import_batch_id),
    )
    batch_id = uuid4()
    await _insert_batch_job(
        db_pool,
        seeded_artifacts,
        batch_id=batch_id,
        stage_id="b2b_enrichment.tier1",
        status="submitted",
        provider_batch_id=f"msgbatch_{uuid4().hex[:12]}",
    )
    await _insert_batch_item(
        db_pool,
        batch_id=batch_id,
        custom_id=b2b_enrichment._enrichment_batch_custom_id("tier1", row["id"]),
        stage_id="b2b_enrichment.tier1",
        artifact_type="review_enrichment_tier1",
        artifact_id=str(row["id"]),
        vendor_name=row["vendor_name"],
        status="pending",
    )

    result = await b2b_enrichment.enrich_batch(str(import_batch_id))
    stored = await db_pool.fetchrow(
        """
        SELECT enrichment_status, enrichment_attempts
        FROM b2b_reviews
        WHERE id = $1
        """,
        row["id"],
    )

    assert result["total"] == 1
    assert result["enriched"] == 0
    assert result["failed"] == 0
    assert result["anthropic_batch_reused_pending_items"] == 1
    assert result["anthropic_batch_rows_deferred"] == 1
    assert stored["enrichment_status"] == "pending"
    assert stored["enrichment_attempts"] == 0


@pytest.mark.asyncio
async def test_enrich_batch_resets_row_to_pending_when_tier2_batch_item_is_pending(
    db_pool,
    seeded_artifacts,
    monkeypatch,
):
    _configure_batch_settings(monkeypatch)

    import_batch_id = uuid4()
    review_id = uuid4()
    row = await _insert_review(
        db_pool,
        seeded_artifacts,
        _review_payload(
            review_id=review_id,
            import_batch_id=import_batch_id,
            rating=2,
            summary="Renewal risk is growing",
            review_text=(
                "The renewal price jumped sharply, support has been slow to answer, "
                "and we are actively evaluating alternatives before the contract renews."
            ),
        ),
    )
    tier1 = {
        "churn_signals": {"intent_to_leave": True},
        "specific_complaints": ["The renewal price jumped sharply and support could not justify it."],
        "quotable_phrases": ["We are actively evaluating alternatives before renewal."],
        "competitors_mentioned": [{"name": "HubSpot"}],
        "pricing_phrases": ["renewal price jumped sharply"],
        "recommendation_language": ["actively evaluating alternatives"],
        "feature_gaps": [],
        "event_mentions": [],
    }

    tier1_batch_id = uuid4()
    await _insert_batch_job(
        db_pool,
        seeded_artifacts,
        batch_id=tier1_batch_id,
        stage_id="b2b_enrichment.tier1",
        status="ended",
        provider_batch_id=f"msgbatch_{uuid4().hex[:12]}",
    )
    await _insert_batch_item(
        db_pool,
        batch_id=tier1_batch_id,
        custom_id=b2b_enrichment._enrichment_batch_custom_id("tier1", row["id"]),
        stage_id="b2b_enrichment.tier1",
        artifact_type="review_enrichment_tier1",
        artifact_id=str(row["id"]),
        vendor_name=row["vendor_name"],
        status="batch_succeeded",
        response_text=json.dumps(tier1),
    )

    tier2_batch_id = uuid4()
    await _insert_batch_job(
        db_pool,
        seeded_artifacts,
        batch_id=tier2_batch_id,
        stage_id="b2b_enrichment.tier2",
        status="submitted",
        provider_batch_id=f"msgbatch_{uuid4().hex[:12]}",
    )
    await _insert_batch_item(
        db_pool,
        batch_id=tier2_batch_id,
        custom_id=b2b_enrichment._enrichment_batch_custom_id("tier2", row["id"]),
        stage_id="b2b_enrichment.tier2",
        artifact_type="review_enrichment_tier2",
        artifact_id=str(row["id"]),
        vendor_name=row["vendor_name"],
        status="pending",
    )

    result = await b2b_enrichment.enrich_batch(str(import_batch_id))
    stored = await db_pool.fetchrow(
        """
        SELECT enrichment_status, enrichment_attempts
        FROM b2b_reviews
        WHERE id = $1
        """,
        row["id"],
    )

    assert result["total"] == 1
    assert result["anthropic_batch_reused_completed_items"] == 1
    assert result["anthropic_batch_reused_pending_items"] == 1
    assert result["anthropic_batch_rows_deferred"] == 1
    assert result["generated"] == 1
    assert result["tier1_generated_calls"] == 1
    assert stored["enrichment_status"] == "pending"
    assert stored["enrichment_attempts"] == 0


@pytest.mark.asyncio
async def test_enrich_batch_reused_tier1_success_persists_terminal_status_and_usage(
    db_pool,
    seeded_artifacts,
    monkeypatch,
):
    _configure_batch_settings(monkeypatch)

    import_batch_id = uuid4()
    review_id = uuid4()
    row = await _insert_review(
        db_pool,
        seeded_artifacts,
        _review_payload(review_id=review_id, import_batch_id=import_batch_id),
    )
    tier1 = {
        "churn_signals": {},
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "feature_gaps": [],
        "event_mentions": [],
    }

    batch_id = uuid4()
    await _insert_batch_job(
        db_pool,
        seeded_artifacts,
        batch_id=batch_id,
        stage_id="b2b_enrichment.tier1",
        status="ended",
        provider_batch_id=f"msgbatch_{uuid4().hex[:12]}",
    )
    await _insert_batch_item(
        db_pool,
        batch_id=batch_id,
        custom_id=b2b_enrichment._enrichment_batch_custom_id("tier1", row["id"]),
        stage_id="b2b_enrichment.tier1",
        artifact_type="review_enrichment_tier1",
        artifact_id=str(row["id"]),
        vendor_name=row["vendor_name"],
        status="cache_hit",
        response_text=json.dumps(tier1),
    )

    result = await b2b_enrichment.enrich_batch(str(import_batch_id))
    stored = await db_pool.fetchrow(
        """
        SELECT enrichment_status, enrichment_attempts, enrichment_model, enrichment
        FROM b2b_reviews
        WHERE id = $1
        """,
        row["id"],
    )

    assert result["total"] == 1
    assert result["anthropic_batch_reused_completed_items"] == 1
    assert result["exact_cache_hits"] == 1
    assert result["tier1_exact_cache_hits"] == 1
    assert result["generated"] == 0
    assert stored["enrichment_status"] == "no_signal"
    assert stored["enrichment_attempts"] == 1
    assert stored["enrichment_model"] == "anthropic/claude-haiku-4-5"
    assert stored["enrichment"] is not None


@pytest.mark.asyncio
async def test_enrich_batch_uses_single_call_tier2_exact_cache_when_batch_model_is_unavailable(
    db_pool,
    seeded_artifacts,
    monkeypatch,
):
    _configure_batch_settings(monkeypatch, tier2_model="openai/gpt-4o-mini")
    tier1_batch_llm = resolve_anthropic_batch_llm(
        current_llm=SimpleNamespace(
            name="openrouter",
            model=str(settings.b2b_churn.enrichment_openrouter_model),
        ),
        target_model_candidates=(settings.b2b_churn.enrichment_openrouter_model,),
    )
    tier2_batch_llm = resolve_anthropic_batch_llm(
        current_llm=SimpleNamespace(
            name="openrouter",
            model=str(settings.b2b_churn.enrichment_tier2_openrouter_model),
        ),
        target_model_candidates=(settings.b2b_churn.enrichment_tier2_openrouter_model,),
    )
    if tier1_batch_llm is None:
        pytest.skip("Tier 1 Anthropic batch LLM is unavailable in this runtime")
    if tier2_batch_llm is not None:
        pytest.skip(
            "Tier 2 Anthropic batch fallback branch is not reachable in this runtime without mocking"
        )

    import_batch_id = uuid4()
    review_id = uuid4()
    row = await _insert_review(
        db_pool,
        seeded_artifacts,
        _review_payload(
            review_id=review_id,
            import_batch_id=import_batch_id,
            rating=2,
            summary="Pricing push triggered an eval",
            review_text=(
                "The renewal quote increased sharply, the billing terms are harder to justify, "
                "and our operations leader has started evaluating HubSpot before renewal."
            ),
        ),
    )
    tier1 = {
        "churn_signals": {"intent_to_leave": True},
        "specific_complaints": ["The renewal quote increased sharply and billing terms are harder to justify."],
        "quotable_phrases": ["Our operations leader has started evaluating HubSpot before renewal."],
        "competitors_mentioned": [{"name": "HubSpot"}],
        "pricing_phrases": ["renewal quote increased sharply"],
        "recommendation_language": ["started evaluating HubSpot before renewal"],
        "feature_gaps": [],
        "event_mentions": [],
    }
    tier2 = {
        "pain_categories": [{"category": "pricing", "severity": "primary"}],
        "sentiment_trajectory": {"direction": "unknown"},
        "buyer_authority": {
            "role_type": "economic_buyer",
            "buying_stage": "evaluation",
            "executive_sponsor_mentioned": True,
        },
        "timeline": {"decision_timeline": "within_quarter"},
        "contract_context": {"contract_value_signal": "mid_market"},
        "positive_aspects": [],
        "feature_gaps": ["billing transparency"],
        "recommendation_language": ["started evaluating HubSpot before renewal"],
        "pricing_phrases": ["renewal quote increased sharply"],
        "event_mentions": [{"event": "renewal evaluation", "timeframe": "within quarter"}],
        "urgency_indicators": {"renewal_pressure": True},
        "competitors_mentioned": [
            {
                "name": "HubSpot",
                "evidence_type": "active_evaluation",
                "displacement_confidence": "high",
                "reason_category": "pricing",
            }
        ],
    }

    tier1_batch_id = uuid4()
    await _insert_batch_job(
        db_pool,
        seeded_artifacts,
        batch_id=tier1_batch_id,
        stage_id="b2b_enrichment.tier1",
        status="ended",
        provider_batch_id=f"msgbatch_{uuid4().hex[:12]}",
    )
    await _insert_batch_item(
        db_pool,
        batch_id=tier1_batch_id,
        custom_id=b2b_enrichment._enrichment_batch_custom_id("tier1", row["id"]),
        stage_id="b2b_enrichment.tier1",
        artifact_type="review_enrichment_tier1",
        artifact_id=str(row["id"]),
        vendor_name=row["vendor_name"],
        status="batch_succeeded",
        response_text=json.dumps(tier1),
    )
    await _seed_tier2_exact_cache(
        db_pool,
        seeded_artifacts,
        row=row,
        tier1=tier1,
        tier2=tier2,
    )

    result = await b2b_enrichment.enrich_batch(str(import_batch_id))
    stored = await db_pool.fetchrow(
        """
        SELECT enrichment_status, enrichment_model, enrichment
        FROM b2b_reviews
        WHERE id = $1
        """,
        row["id"],
    )

    assert result["total"] == 1
    assert result["anthropic_batch_tier2_single_fallback_rows"] == 1
    assert result["generated"] == 1
    assert result["tier1_generated_calls"] == 1
    assert result["exact_cache_hits"] == 1
    assert result["tier2_exact_cache_hits"] == 1
    assert stored["enrichment_status"] == "enriched"
    assert stored["enrichment_model"] == "hybrid:anthropic/claude-haiku-4-5+openai/gpt-4o-mini"
    assert stored["enrichment"] is not None
