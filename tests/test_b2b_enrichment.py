import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.autonomous.tasks._b2b_batch_utils import exact_stage_request_fingerprint, reconcile_existing_batch_artifacts
from atlas_brain.reasoning import evidence_engine
from atlas_brain.services.b2b.enrichment_stage_controller import (
    apply_stage_decision,
    apply_review_stage_transition,
    defer_review_transition,
    persist_review_transition,
    StageExecutionDecision,
    finalize_stage_batch,
    prepare_stage_execution,
    submit_stage_batch,
)
from atlas_brain.services.b2b.enrichment_persistence import (
    EnrichmentFinalizationDeps,
    EnrichmentPersistenceDeps,
    finalize_enrichment_for_persist as service_finalize_enrichment_for_persist,
    persist_enrichment_result as service_persist_enrichment_result,
)
from atlas_brain.services.b2b.enrichment_outcome_policy import (
    EnrichmentOutcomePolicyDeps,
    is_no_signal_result as service_is_no_signal_result,
    trusted_reviewer_company_name as service_trusted_reviewer_company_name,
    witness_metrics as service_witness_metrics,
)
from atlas_brain.services.b2b.enrichment_validation import (
    EnrichmentValidationDeps,
    validate_enrichment as service_validate_enrichment,
)
from atlas_brain.services.b2b.enrichment_derivation import (
    EnrichmentDerivationDeps,
    compute_derived_fields as service_compute_derived_fields,
)
from atlas_brain.services.b2b.enrichment_phrase_metadata import (
    EnrichmentPhraseMetadataDeps,
    apply_phrase_metadata_contract as service_apply_phrase_metadata_contract,
    coerce_legacy_phrase_arrays as service_coerce_legacy_phrase_arrays,
    normalize_tag_value as service_normalize_tag_value,
)
from atlas_brain.services.b2b.enrichment_repair import (
    EnrichmentRepairDeps,
    apply_structural_repair as service_apply_structural_repair,
    repair_target_fields as service_repair_target_fields,
)
from atlas_brain.services.b2b.enrichment_buyer_authority import (
    EnrichmentBuyerAuthorityDeps,
    derive_buyer_authority_fields as service_derive_buyer_authority_fields,
    infer_role_level_from_text as service_infer_role_level_from_text,
)
from atlas_brain.services.b2b.enrichment_timeline import (
    EnrichmentTimelineDeps,
    derive_concrete_timeline_fields as service_derive_concrete_timeline_fields,
    derive_decision_timeline as service_derive_decision_timeline,
)
from atlas_brain.services.b2b.enrichment_budget import (
    EnrichmentBudgetDeps,
    derive_budget_signals as service_derive_budget_signals,
    derive_contract_value_signal as service_derive_contract_value_signal,
)
from atlas_brain.services.b2b.enrichment_pain_competition import (
    EnrichmentPainCompetitionDeps,
    compute_pain_confidence as service_compute_pain_confidence,
    demote_primary_pain as service_demote_primary_pain,
    derive_competitor_annotations as service_derive_competitor_annotations,
    derive_pain_categories as service_derive_pain_categories,
    recover_competitor_mentions as service_recover_competitor_mentions,
    subject_vendor_phrase_texts as service_subject_vendor_phrase_texts,
)
from atlas_brain.services.b2b.enrichment_urgency import (
    EnrichmentUrgencyDeps,
    derive_urgency_indicators as service_derive_urgency_indicators,
)
from atlas_brain.services.b2b.enrichment_support import (
    coerce_bool as service_coerce_bool,
    coerce_json_dict as service_coerce_json_dict,
    combined_source_text as service_combined_source_text,
    contains_any as service_contains_any,
    has_technical_context as service_has_technical_context,
    normalize_compare_text as service_normalize_compare_text,
    normalized_low_fidelity_noisy_sources as service_normalized_low_fidelity_noisy_sources,
    normalized_name_tokens as service_normalized_name_tokens,
    text_mentions_name as service_text_mentions_name,
)
from atlas_brain.services.b2b.enrichment_stage_planner import (
    build_tier1_stage_plan,
    build_tier2_stage_plan,
    stage_backend_name,
)
from atlas_brain.services.b2b.enrichment_stage_runs import StageRunResolution, resolve_stage_run
from atlas_brain.storage.models import ScheduledTask


def _task() -> ScheduledTask:
    return ScheduledTask(
        id=uuid4(),
        name="b2b_enrichment",
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=300,
        enabled=True,
        metadata={"builtin_handler": "b2b_enrichment"},
    )


class _Pool:
    def __init__(self, batches):
        self.is_initialized = True
        self.fetch = AsyncMock(side_effect=batches)
        self.fetchval = AsyncMock(return_value=0)
        self.execute = AsyncMock(return_value="UPDATE 0")


def test_normalize_pain_category_maps_legacy_other_to_overall_dissatisfaction():
    assert b2b_enrichment._normalize_pain_category("other") == "overall_dissatisfaction"
    assert b2b_enrichment._normalize_pain_category("general_dissatisfaction") == "overall_dissatisfaction"


def test_normalize_pain_category_accepts_new_specific_buckets():
    assert b2b_enrichment._normalize_pain_category("admin_burden") == "admin_burden"
    assert b2b_enrichment._normalize_pain_category("integration_debt") == "integration_debt"


def test_enrichment_batch_custom_id_is_anthropic_safe():
    assert b2b_enrichment._enrichment_batch_custom_id("tier1", "1234-5678") == "tier1_1234-5678"


def test_stage_backend_name_maps_batch_and_provider():
    assert stage_backend_name(batch_enabled=True, provider="openrouter") == "anthropic_batch"
    assert stage_backend_name(batch_enabled=False, provider="openrouter") == "direct_openrouter"
    assert stage_backend_name(batch_enabled=False, provider="vllm") == "direct_vllm"


def test_build_tier1_stage_plan_captures_request_identity():
    def _prepare_stage_request(stage_id, **kwargs):
        return (
            SimpleNamespace(stage_id=stage_id, kwargs=kwargs),
            "request-fingerprint",
            "work-fingerprint",
        )

    plan = build_tier1_stage_plan(
        row={"id": "review-1"},
        payload_json='{"vendor_name":"Zendesk"}',
        system_prompt="tier1 prompt",
        model="anthropic/claude-haiku-4-5",
        provider="openrouter",
        batch_enabled=True,
        run_id="run-1",
        prepare_stage_request=_prepare_stage_request,
        max_tokens=4096,
        guided_json=None,
    )

    assert plan.stage_id == "b2b_enrichment.tier1"
    assert plan.backend == "anthropic_batch"
    assert plan.request_fingerprint == "request-fingerprint"
    assert plan.work_fingerprint == "work-fingerprint"
    assert plan.messages[0]["content"] == "tier1 prompt"


def test_build_tier2_stage_plan_includes_tier1_fields_and_prompt_filter():
    def _prepare_stage_request(stage_id, **kwargs):
        return (
            SimpleNamespace(stage_id=stage_id, kwargs=kwargs),
            "request-fingerprint",
            "work-fingerprint",
        )

    plan = build_tier2_stage_plan(
        row={"id": "review-1"},
        base_payload={"content_type": "review", "vendor_name": "Zendesk"},
        tier1_result={
            "specific_complaints": ["pricing pressure"],
            "quotable_phrases": ["renewal discussions are tense"],
        },
        system_prompt="tier2 prompt",
        model="anthropic/claude-haiku-4-5",
        provider="openrouter",
        batch_enabled=False,
        run_id="run-1",
        prepare_stage_request=_prepare_stage_request,
        prompt_for_content_type=lambda prompt, _content_type: f"{prompt}::filtered",
        max_tokens=512,
        workload="direct",
    )

    payload = json.loads(plan.payload_json)
    assert payload["tier1_specific_complaints"] == ["pricing pressure"]
    assert payload["tier1_quotable_phrases"] == ["renewal discussions are tense"]
    assert plan.messages[0]["content"] == "tier2 prompt::filtered"
    assert plan.backend == "direct_openrouter"


def test_derive_competitor_annotations_prunes_generic_provider_labels():
    row = {
        "vendor_name": "Amazon Web Services",
        "summary": "AWS Failed Me",
        "review_text": "We may need a competing provider after this outage.",
        "pros": "",
        "cons": "",
    }
    result = {
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": False},
        "competitors_mentioned": [{"name": "competing provider"}],
    }

    derived = b2b_enrichment._derive_competitor_annotations(result, row)

    assert derived == []


def test_derive_competitor_annotations_prunes_weak_neutral_mentions_without_named_context():
    row = {
        "vendor_name": "ActiveCampaign",
        "summary": "Pricing and support issues",
        "review_text": "Their pricing is outrageous and support is nonexistent. We need a better answer fast.",
        "pros": "",
        "cons": "",
    }
    result = {
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": False},
        "competitors_mentioned": [{"name": "HubSpot"}],
    }

    derived = b2b_enrichment._derive_competitor_annotations(result, row)

    assert derived == []


def test_service_subject_vendor_phrase_texts_filters_v2_non_subject_phrases():
    result = {
        "specific_complaints": ["pricing doubled", "our migration team was overwhelmed"],
        "phrase_metadata": [
            {"field": "specific_complaints", "index": 0, "subject": "subject_vendor", "polarity": "negative"},
            {"field": "specific_complaints", "index": 1, "subject": "self", "polarity": "negative"},
        ],
        "enrichment_schema_version": 4,
    }

    texts = service_subject_vendor_phrase_texts(
        result,
        "specific_complaints",
        deps=_pain_competition_test_deps(),
    )

    assert texts == ["pricing doubled"]


def test_service_derive_pain_categories_and_confidence_use_v2_filters():
    result = {
        "specific_complaints": ["pricing doubled", "our IT team was learning the system"],
        "pricing_phrases": ["renewal bill jumped 40%"],
        "feature_gaps": [],
        "quotable_phrases": [],
        "phrase_metadata": [
            {"field": "specific_complaints", "index": 0, "subject": "subject_vendor", "polarity": "negative"},
            {"field": "specific_complaints", "index": 1, "subject": "self", "polarity": "negative"},
            {"field": "pricing_phrases", "index": 0, "subject": "subject_vendor", "polarity": "mixed"},
        ],
        "enrichment_schema_version": 4,
        "churn_signals": {"intent_to_leave": True},
        "would_recommend": False,
        "sentiment_trajectory": {"direction": "declining"},
    }

    categories = service_derive_pain_categories(result, deps=_pain_competition_test_deps())
    confidence = service_compute_pain_confidence(result, "pricing", deps=_pain_competition_test_deps())

    assert categories[0] == {"category": "pricing", "severity": "primary"}
    assert confidence == "strong"


def test_service_demote_primary_pain_preserves_demoted_context():
    result = {
        "pain_categories": [
            {"category": "pricing", "severity": "primary"},
            {"category": "support", "severity": "secondary"},
        ]
    }

    service_demote_primary_pain(result, "pricing")

    assert result["pain_categories"][0] == {"category": "overall_dissatisfaction", "severity": "primary"}
    assert {"category": "pricing", "severity": "secondary"} in result["pain_categories"]


def test_service_recover_competitor_mentions_and_annotations():
    row = {
        "vendor_name": "Zendesk",
        "summary": "We switched after the renewal",
        "review_text": "We switched to Intercom after Zendesk raised prices at renewal.",
        "pros": "",
        "cons": "",
    }
    recovered = service_recover_competitor_mentions(
        {"competitors_mentioned": [], "quotable_phrases": []},
        row,
        deps=_pain_competition_test_deps(),
    )
    annotated = service_derive_competitor_annotations(
        {
            "competitors_mentioned": recovered,
            "churn_signals": {"migration_in_progress": True, "renewal_timing": "this quarter"},
        },
        row,
        deps=_pain_competition_test_deps(),
    )

    assert recovered == [{"name": "Intercom"}]
    assert annotated[0]["evidence_type"] == "explicit_switch"
    assert annotated[0]["displacement_confidence"] == "high"


def test_service_derive_urgency_indicators_tracks_price_and_decision_signals():
    row = {
        "summary": "Renewal decision is next quarter",
        "review_text": "We are considering switching and I decided we need another tool before renewal.",
        "pros": "",
        "cons": "",
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "renewal_timing": "next quarter",
        },
        "budget_signals": {"annual_spend_estimate": "$60k/year", "price_per_seat": None},
        "timeline": {"contract_end": "next quarter", "evaluation_deadline": None},
        "competitors_mentioned": [{"name": "Intercom", "reason_detail": "better pricing"}],
        "specific_complaints": ["pricing backlash"],
        "pricing_phrases": ["price increase at renewal"],
        "recommendation_language": ["I decided we should switch"],
        "reviewer_context": {"decision_maker": True},
    }

    indicators = service_derive_urgency_indicators(
        result,
        row,
        deps=_urgency_test_deps(),
    )

    assert indicators["intent_to_leave_signal"] is True
    assert indicators["actively_evaluating_signal"] is True
    assert indicators["named_alternative_with_reason"] is True
    assert indicators["price_pressure_language"] is True
    assert indicators["timeline_mentioned"] is True
    assert indicators["decision_maker_language"] is True


def test_repair_target_fields_skips_roundup_style_competitor_repair_without_named_displacement():
    row = {
        "source": "reddit",
        "enrichment_status": "enriched",
        "summary": "I went through hundreds of user reviews of project management tools, here's what actually matters.",
        "review_text": (
            "Asana is strong for structured teams and workflows. Monday.com has great UI. "
            "Notion is loved for docs plus project hybrid use. Trello falls short for growing teams."
        ),
        "pros": "",
        "cons": "",
    }
    result = {
        "pain_category": "overall_dissatisfaction",
        "salience_flags": [],
        "competitors_mentioned": [],
        "specific_complaints": ["limited customization compared to others"],
        "pricing_phrases": [],
        "recommendation_language": [],
        "feature_gaps": ["limited customization compared to others"],
        "event_mentions": [],
        "timeline": {"decision_timeline": "unknown"},
    }

    targets = b2b_enrichment._repair_target_fields(result, row)

    assert "competitors_mentioned" not in targets


def test_effective_enrichment_skip_sources_includes_deprecated_sources(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_skip_sources",
        "stackoverflow,github",
        raising=False,
    )
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "deprecated_review_sources",
        "software_advice,trustpilot,trustradius",
        raising=False,
    )

    assert b2b_enrichment._effective_enrichment_skip_sources() == {
        "stackoverflow",
        "github",
        "software_advice",
        "trustpilot",
        "trustradius",
    }


def test_trusted_repair_sources_filters_deprecated_sources(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_priority_sources",
        "g2,trustradius,capterra,software_advice,gartner,trustpilot,peerspot",
        raising=False,
    )
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "deprecated_review_sources",
        "software_advice,trustpilot,trustradius",
        raising=False,
    )

    assert b2b_enrichment._trusted_repair_sources() == {
        "g2",
        "gartner",
        "peerspot",
        "capterra",
        "software_advice",
        "trustradius",
    }


def test_effective_min_review_text_length_relaxes_capterra_to_scrape_floor(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_scrape,
        "capterra_min_enrichable_text_len",
        40,
        raising=False,
    )

    assert b2b_enrichment._effective_min_review_text_length({"source": "capterra"}) == 40
    assert b2b_enrichment._effective_min_review_text_length({"source": "reddit"}) == 80


def test_combined_review_text_length_counts_pros_and_cons():
    row = {
        "review_text": "short",
        "pros": "works well",
        "cons": "pricey",
    }

    assert b2b_enrichment._combined_review_text_length(row) == len("short") + len("works well") + len("pricey")


def test_tier2_system_prompt_trims_insider_section_for_non_insider_content():
    prompt = (
        "## Intro\n"
        "Keep output grounded.\n\n"
        "### insider_signals -- CLASSIFY + EXTRACT (only for insider_account)\n"
        "Insider-only instructions.\n\n"
        "## Output\n"
        "Return JSON."
    )

    trimmed = b2b_enrichment._tier2_system_prompt_for_content_type(prompt, "review")

    assert "Insider-only instructions." not in trimmed
    assert "### insider_signals -- CLASSIFY + EXTRACT (only for insider_account)" not in trimmed
    assert trimmed == "## Intro\nKeep output grounded.\n\n## Output\nReturn JSON."


def test_tier2_system_prompt_keeps_insider_section_for_insider_accounts():
    prompt = (
        "## Intro\n"
        "Keep output grounded.\n\n"
        "### insider_signals -- CLASSIFY + EXTRACT (only for insider_account)\n"
        "Insider-only instructions.\n\n"
        "## Output\n"
        "Return JSON."
    )

    preserved = b2b_enrichment._tier2_system_prompt_for_content_type(prompt, "insider_account")

    assert preserved == prompt


def test_build_classify_payload_omits_empty_optional_fields():
    payload = b2b_enrichment._build_classify_payload(
        {
            "vendor_name": "Zendesk",
            "product_name": "",
            "product_category": "",
            "source": "g2",
            "raw_metadata": {},
            "content_type": "review",
            "rating": None,
            "rating_max": 5,
            "summary": "",
            "review_text": "",
            "pros": "",
            "cons": "",
            "reviewer_title": " ",
            "reviewer_company": None,
            "company_size_raw": "",
            "reviewer_industry": "",
        }
    )

    assert payload["vendor_name"] == "Zendesk"
    assert payload["content_type"] == "review"
    assert payload["rating_max"] == 5
    assert "product_name" not in payload
    assert "product_category" not in payload
    assert "rating" not in payload
    assert "summary" not in payload
    assert "review_text" not in payload
    assert "pros" not in payload
    assert "cons" not in payload
    assert "reviewer_title" not in payload
    assert "reviewer_company" not in payload
    assert "company_size_raw" not in payload
    assert "reviewer_industry" not in payload


def test_tier1_has_extraction_gaps_keeps_default_behavior_for_non_strict_sources(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_tier2_strict_sources",
        "gartner,peerspot",
        raising=False,
    )
    tier1 = {
        "specific_complaints": [],
        "quotable_phrases": ["Users keep mentioning admin overhead."],
        "competitors_mentioned": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "churn_signals": {},
    }

    assert b2b_enrichment._tier1_has_extraction_gaps(tier1, source="g2") is True


def test_tier1_has_extraction_gaps_is_stricter_for_gartner_and_peerspot(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_tier2_strict_sources",
        "gartner,peerspot",
        raising=False,
    )
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_tier2_strict_min_complaints",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_tier2_strict_min_quotes",
        2,
        raising=False,
    )
    weak_tier1 = {
        "specific_complaints": [],
        "quotable_phrases": ["One generic quote"],
        "competitors_mentioned": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "churn_signals": {},
    }
    strong_tier1 = {
        "specific_complaints": ["Billing was confusing", "Renewal price jumped unexpectedly"],
        "quotable_phrases": ["Two useful quotes", "Another useful quote"],
        "competitors_mentioned": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "churn_signals": {},
    }

    assert b2b_enrichment._tier1_has_extraction_gaps(weak_tier1, source="gartner") is False
    assert b2b_enrichment._tier1_has_extraction_gaps(weak_tier1, source="peerspot") is False
    assert b2b_enrichment._tier1_has_extraction_gaps(strong_tier1, source="gartner") is True


def test_tier1_has_extraction_gaps_still_allows_strict_sources_with_churn_or_competitor(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_tier2_strict_sources",
        "gartner,peerspot",
        raising=False,
    )
    churn_tier1 = {
        "specific_complaints": [],
        "quotable_phrases": [],
        "competitors_mentioned": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "churn_signals": {"intent_to_leave": True},
    }
    competitor_tier1 = {
        "specific_complaints": [],
        "quotable_phrases": [],
        "competitors_mentioned": [{"name": "HubSpot"}],
        "pricing_phrases": [],
        "recommendation_language": [],
        "churn_signals": {},
    }

    assert b2b_enrichment._tier1_has_extraction_gaps(churn_tier1, source="gartner") is True
    assert b2b_enrichment._tier1_has_extraction_gaps(competitor_tier1, source="peerspot") is True


def test_is_no_signal_result_accepts_empty_community_discussion_without_rating():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "support_escalation": False,
            "contract_renewal_mentioned": False,
        },
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "event_mentions": [],
        "feature_gaps": [],
    }
    row = {
        "content_type": "community_discussion",
        "rating": None,
    }

    assert b2b_enrichment._is_no_signal_result(result, row) is True


def test_service_is_no_signal_result_accepts_empty_community_discussion_without_rating():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "support_escalation": False,
            "contract_renewal_mentioned": False,
        },
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "event_mentions": [],
        "feature_gaps": [],
    }
    row = {
        "content_type": "community_discussion",
        "rating": None,
    }

    assert service_is_no_signal_result(result, row) is True


def test_service_trusted_reviewer_company_name_filters_vendor_match():
    deps = EnrichmentOutcomePolicyDeps(
        normalized_low_fidelity_noisy_sources=lambda: set(),
        normalize_compare_text=lambda value: str(value or ""),
        text_mentions_name=lambda haystack, needle: False,
        normalized_name_tokens=lambda value: [],
        has_commercial_context=lambda text: False,
        has_strong_commercial_context=lambda text: False,
        has_technical_context=lambda summary_text, combined_text: False,
        has_consumer_context=lambda text: False,
        dedupe_reason_codes=lambda codes: codes,
        normalize_company_name=lambda value: str(value or "").strip().lower().replace(" ", ""),
    )
    row = {
        "reviewer_company": "HubSpot",
        "vendor_name": "Hub Spot",
    }

    assert service_trusted_reviewer_company_name(row, deps=deps) is None


def test_service_witness_metrics_counts_grounded_spans():
    result = {
        "evidence_spans": [
            {"text": "pricing doubled"},
            {"text": " support declined "},
            {"text": ""},
            {"other": "ignored"},
        ]
    }

    assert service_witness_metrics(result) == (1, 2)


def _validation_test_deps() -> EnrichmentValidationDeps:
    return EnrichmentValidationDeps(
        coerce_bool=b2b_enrichment._coerce_bool,
        normalize_pain_category=b2b_enrichment._normalize_pain_category,
        normalize_budget_value_text=b2b_enrichment._normalize_budget_value_text,
        normalize_budget_detail_text=b2b_enrichment._normalize_budget_detail_text,
        canonical_role_type=b2b_enrichment._canonical_role_type,
        canonical_role_level=b2b_enrichment._canonical_role_level,
        infer_role_level_from_text=b2b_enrichment._infer_role_level_from_text,
        infer_decision_maker=b2b_enrichment._infer_decision_maker,
        infer_buyer_role_type=b2b_enrichment._infer_buyer_role_type,
        coerce_json_dict=b2b_enrichment._coerce_json_dict,
        schema_version=b2b_enrichment._schema_version,
        missing_witness_primitives=b2b_enrichment._missing_witness_primitives,
        compute_derived_fields=b2b_enrichment._compute_derived_fields,
        trusted_reviewer_company_name=b2b_enrichment._trusted_reviewer_company_name,
        churn_signal_bool_fields=b2b_enrichment._CHURN_SIGNAL_BOOL_FIELDS,
        known_severity_levels=b2b_enrichment._KNOWN_SEVERITY_LEVELS,
        known_lock_in_levels=b2b_enrichment._KNOWN_LOCK_IN_LEVELS,
        known_sentiment_directions=b2b_enrichment._KNOWN_SENTIMENT_DIRECTIONS,
        known_buying_stages=b2b_enrichment._KNOWN_BUYING_STAGES,
        known_decision_timelines=b2b_enrichment._KNOWN_DECISION_TIMELINES,
        known_contract_value_signals=b2b_enrichment._KNOWN_CONTRACT_VALUE_SIGNALS,
        known_replacement_modes=b2b_enrichment._KNOWN_REPLACEMENT_MODES,
        known_operating_model_shifts=b2b_enrichment._KNOWN_OPERATING_MODEL_SHIFTS,
        known_productivity_delta_claims=b2b_enrichment._KNOWN_PRODUCTIVITY_DELTA_CLAIMS,
        known_org_pressure_types=b2b_enrichment._KNOWN_ORG_PRESSURE_TYPES,
        known_content_types=b2b_enrichment._KNOWN_CONTENT_TYPES,
        known_org_health_levels=b2b_enrichment._KNOWN_ORG_HEALTH_LEVELS,
        known_leadership_qualities=b2b_enrichment._KNOWN_LEADERSHIP_QUALITIES,
        known_innovation_climates=b2b_enrichment._KNOWN_INNOVATION_CLIMATES,
        known_morale_levels=b2b_enrichment._KNOWN_MORALE_LEVELS,
        known_departure_types=b2b_enrichment._KNOWN_DEPARTURE_TYPES,
        known_pain_categories=b2b_enrichment._KNOWN_PAIN_CATEGORIES,
    )


def _derivation_test_deps() -> EnrichmentDerivationDeps:
    from atlas_brain.reasoning import evidence_engine

    return EnrichmentDerivationDeps(
        get_evidence_engine=evidence_engine.get_evidence_engine,
        coerce_legacy_phrase_arrays=b2b_enrichment._coerce_legacy_phrase_arrays,
        apply_phrase_metadata_contract=b2b_enrichment._apply_phrase_metadata_contract,
        derive_pain_categories=b2b_enrichment._derive_pain_categories,
        recover_competitor_mentions=b2b_enrichment._recover_competitor_mentions,
        derive_competitor_annotations=b2b_enrichment._derive_competitor_annotations,
        derive_budget_signals=b2b_enrichment._derive_budget_signals,
        derive_buyer_authority_fields=b2b_enrichment._derive_buyer_authority_fields,
        derive_concrete_timeline_fields=b2b_enrichment._derive_concrete_timeline_fields,
        derive_decision_timeline=b2b_enrichment._derive_decision_timeline,
        derive_contract_value_signal=b2b_enrichment._derive_contract_value_signal,
        derive_urgency_indicators=b2b_enrichment._derive_urgency_indicators,
        normalize_pain_category=b2b_enrichment._normalize_pain_category,
        subject_vendor_phrase_texts=b2b_enrichment._subject_vendor_phrase_texts,
        compute_pain_confidence=b2b_enrichment._compute_pain_confidence,
        demote_primary_pain=b2b_enrichment._demote_primary_pain,
        derive_replacement_mode=b2b_enrichment.derive_replacement_mode,
        derive_operating_model_shift=b2b_enrichment.derive_operating_model_shift,
        derive_productivity_delta_claim=b2b_enrichment.derive_productivity_delta_claim,
        derive_org_pressure_type=b2b_enrichment.derive_org_pressure_type,
        derive_salience_flags=b2b_enrichment.derive_salience_flags,
        derive_evidence_spans=b2b_enrichment.derive_evidence_spans,
    )


def _budget_test_deps() -> EnrichmentBudgetDeps:
    return EnrichmentBudgetDeps(
        contains_any=b2b_enrichment._contains_any,
        coerce_bool=b2b_enrichment._coerce_bool,
        normalize_compare_text=b2b_enrichment._normalize_compare_text,
        normalize_text_list=b2b_enrichment._normalize_text_list,
        combined_source_text=b2b_enrichment._combined_source_text,
        normalized_low_fidelity_noisy_sources=b2b_enrichment._normalized_low_fidelity_noisy_sources,
        text_mentions_name=b2b_enrichment._text_mentions_name,
        has_commercial_context=b2b_enrichment._has_commercial_context,
        has_strong_commercial_context=b2b_enrichment._has_strong_commercial_context,
        has_technical_context=b2b_enrichment._has_technical_context,
        has_consumer_context=b2b_enrichment._has_consumer_context,
        timeline_ambiguous_vendor_tokens=b2b_enrichment._TIMELINE_AMBIGUOUS_VENDOR_TOKENS,
        timeline_ambiguous_vendor_product_context_patterns=(
            b2b_enrichment._TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS
        ),
        budget_any_amount_token_re=b2b_enrichment._BUDGET_ANY_AMOUNT_TOKEN_RE,
        budget_price_per_seat_re=b2b_enrichment._BUDGET_PRICE_PER_SEAT_RE,
        budget_annual_amount_re=b2b_enrichment._BUDGET_ANNUAL_AMOUNT_RE,
        budget_currency_token_re=b2b_enrichment._BUDGET_CURRENCY_TOKEN_RE,
        budget_seat_count_re=b2b_enrichment._BUDGET_SEAT_COUNT_RE,
        budget_price_increase_re=b2b_enrichment._BUDGET_PRICE_INCREASE_RE,
        budget_price_increase_detail_re=b2b_enrichment._BUDGET_PRICE_INCREASE_DETAIL_RE,
        budget_annual_period_patterns=b2b_enrichment._BUDGET_ANNUAL_PERIOD_PATTERNS,
        budget_monthly_period_patterns=b2b_enrichment._BUDGET_MONTHLY_PERIOD_PATTERNS,
        budget_noise_patterns=b2b_enrichment._BUDGET_NOISE_PATTERNS,
        budget_per_unit_patterns=b2b_enrichment._BUDGET_PER_UNIT_PATTERNS,
        budget_annual_context_patterns=b2b_enrichment._BUDGET_ANNUAL_CONTEXT_PATTERNS,
        budget_commercial_context_patterns=b2b_enrichment._BUDGET_COMMERCIAL_CONTEXT_PATTERNS,
    )


def _pain_competition_test_deps() -> EnrichmentPainCompetitionDeps:
    return EnrichmentPainCompetitionDeps(
        normalize_text_list=b2b_enrichment._normalize_text_list,
        normalize_pain_category=b2b_enrichment._normalize_pain_category,
        normalize_company_name=b2b_enrichment.normalize_company_name,
        pain_patterns=b2b_enrichment._PAIN_PATTERNS,
        pain_derivation_fields=b2b_enrichment._PAIN_DERIVATION_FIELDS,
        competitor_recovery_patterns=b2b_enrichment._COMPETITOR_RECOVERY_PATTERNS,
        competitor_recovery_blocklist=b2b_enrichment._COMPETITOR_RECOVERY_BLOCKLIST,
        generic_competitor_tokens=b2b_enrichment._GENERIC_COMPETITOR_TOKENS,
        competitor_context_patterns=b2b_enrichment._COMPETITOR_CONTEXT_PATTERNS,
    )


def _urgency_test_deps() -> EnrichmentUrgencyDeps:
    return EnrichmentUrgencyDeps(
        contains_any=b2b_enrichment._contains_any,
        normalize_text_list=b2b_enrichment._normalize_text_list,
    )


def _phrase_metadata_test_deps() -> EnrichmentPhraseMetadataDeps:
    return EnrichmentPhraseMetadataDeps(
        check_phrase_grounded=lambda phrase, summary=None, review_text=None: True,
    )


def _repair_test_deps() -> EnrichmentRepairDeps:
    return EnrichmentRepairDeps(
        normalize_text_list=b2b_enrichment._normalize_text_list,
        normalize_pain_category=b2b_enrichment._normalize_pain_category,
        contains_any=b2b_enrichment._contains_any,
        coerce_json_dict=b2b_enrichment._coerce_json_dict,
        is_unknownish=b2b_enrichment._is_unknownish,
        trusted_repair_sources=b2b_enrichment._trusted_repair_sources,
        normalize_company_name=b2b_enrichment.normalize_company_name,
        repair_negative_patterns=b2b_enrichment._REPAIR_NEGATIVE_PATTERNS,
        repair_competitor_patterns=b2b_enrichment._REPAIR_COMPETITOR_PATTERNS,
        repair_pricing_patterns=b2b_enrichment._REPAIR_PRICING_PATTERNS,
        repair_recommend_patterns=b2b_enrichment._REPAIR_RECOMMEND_PATTERNS,
        repair_feature_gap_patterns=b2b_enrichment._REPAIR_FEATURE_GAP_PATTERNS,
        repair_timeline_patterns=b2b_enrichment._REPAIR_TIMELINE_PATTERNS,
        repair_category_shift_patterns=b2b_enrichment._REPAIR_CATEGORY_SHIFT_PATTERNS,
        repair_currency_re=b2b_enrichment._REPAIR_CURRENCY_RE,
    )


def _buyer_authority_test_deps() -> EnrichmentBuyerAuthorityDeps:
    return EnrichmentBuyerAuthorityDeps(
        sanitize_reviewer_title=b2b_enrichment.sanitize_reviewer_title,
        coerce_bool=b2b_enrichment._coerce_bool,
        coerce_json_dict=b2b_enrichment._coerce_json_dict,
        contains_any=b2b_enrichment._contains_any,
        role_type_aliases=b2b_enrichment._ROLE_TYPE_ALIASES,
        role_level_aliases=b2b_enrichment._ROLE_LEVEL_ALIASES,
        champion_reviewer_title_pattern=b2b_enrichment._CHAMPION_REVIEWER_TITLE_PATTERN,
        evaluator_reviewer_title_pattern=b2b_enrichment._EVALUATOR_REVIEWER_TITLE_PATTERN,
        exec_role_text_pattern=b2b_enrichment._EXEC_ROLE_TEXT_PATTERN,
        director_role_text_pattern=b2b_enrichment._DIRECTOR_ROLE_TEXT_PATTERN,
        manager_role_text_pattern=b2b_enrichment._MANAGER_ROLE_TEXT_PATTERN,
        ic_role_text_pattern=b2b_enrichment._IC_ROLE_TEXT_PATTERN,
        commercial_decision_text_pattern=b2b_enrichment._COMMERCIAL_DECISION_TEXT_PATTERN,
        exec_reviewer_title_pattern=b2b_enrichment._EXEC_REVIEWER_TITLE_PATTERN,
        manager_decision_title_pattern=b2b_enrichment._MANAGER_DECISION_TITLE_PATTERN,
        economic_buyer_text_patterns=b2b_enrichment._ECONOMIC_BUYER_TEXT_PATTERNS,
        champion_text_patterns=b2b_enrichment._CHAMPION_TEXT_PATTERNS,
        evaluator_text_patterns=b2b_enrichment._EVALUATOR_TEXT_PATTERNS,
        end_user_text_patterns=b2b_enrichment._END_USER_TEXT_PATTERNS,
        post_purchase_review_sources=set(b2b_enrichment._POST_PURCHASE_REVIEW_SOURCES),
        post_purchase_usage_patterns=b2b_enrichment._POST_PURCHASE_USAGE_PATTERNS,
    )


def _timeline_test_deps() -> EnrichmentTimelineDeps:
    return EnrichmentTimelineDeps(
        contains_any=b2b_enrichment._contains_any,
        normalize_compare_text=b2b_enrichment._normalize_compare_text,
        has_commercial_context=b2b_enrichment._has_commercial_context,
        has_strong_commercial_context=b2b_enrichment._has_strong_commercial_context,
        has_technical_context=b2b_enrichment._has_technical_context,
        has_consumer_context=b2b_enrichment._has_consumer_context,
        normalized_low_fidelity_noisy_sources=b2b_enrichment._normalized_low_fidelity_noisy_sources,
        text_mentions_name=b2b_enrichment._text_mentions_name,
        timeline_month_day_re=b2b_enrichment._TIMELINE_MONTH_DAY_RE,
        timeline_slash_date_re=b2b_enrichment._TIMELINE_SLASH_DATE_RE,
        timeline_iso_date_re=b2b_enrichment._TIMELINE_ISO_DATE_RE,
        timeline_explicit_anchor_phrases=b2b_enrichment._TIMELINE_EXPLICIT_ANCHOR_PHRASES,
        timeline_relative_anchor_re=b2b_enrichment._TIMELINE_RELATIVE_ANCHOR_RE,
        timeline_contract_event_patterns=b2b_enrichment._TIMELINE_CONTRACT_EVENT_PATTERNS,
        timeline_decision_deadline_patterns=b2b_enrichment._TIMELINE_DECISION_DEADLINE_PATTERNS,
        timeline_contract_end_patterns=b2b_enrichment._TIMELINE_CONTRACT_END_PATTERNS,
        timeline_immediate_patterns=b2b_enrichment._TIMELINE_IMMEDIATE_PATTERNS,
        timeline_quarter_patterns=b2b_enrichment._TIMELINE_QUARTER_PATTERNS,
        timeline_year_patterns=b2b_enrichment._TIMELINE_YEAR_PATTERNS,
        timeline_decision_patterns=b2b_enrichment._TIMELINE_DECISION_PATTERNS,
        timeline_ambiguous_vendor_tokens=b2b_enrichment._TIMELINE_AMBIGUOUS_VENDOR_TOKENS,
        timeline_ambiguous_vendor_product_context_patterns=b2b_enrichment._TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS,
    )


def test_service_validate_enrichment_rejects_out_of_range_urgency():
    result = {
        "churn_signals": {},
        "urgency_score": 11,
    }

    assert service_validate_enrichment(result, None, deps=_validation_test_deps()) is False


def test_service_validate_enrichment_coerces_unknown_decision_timeline():
    result = {
        "churn_signals": {},
        "urgency_score": 4,
        "timeline": {"decision_timeline": "someday"},
    }

    assert service_validate_enrichment(result, None, deps=_validation_test_deps()) is True
    assert result["timeline"]["decision_timeline"] == "unknown"


def test_service_compute_derived_fields_promotes_event_timeframe_into_contract_end(monkeypatch):
    from atlas_brain.reasoning import evidence_engine

    class _Engine:
        map_hash = "test-hash"

        def derive_price_complaint(self, result):
            return False

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 7.1

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())
    row, result = _witness_ready_row_and_result()

    derived = service_compute_derived_fields(result, row, deps=_derivation_test_deps())

    assert derived["timeline"]["contract_end"] == "next quarter"
    assert derived["timeline"]["decision_timeline"] == "within_quarter"


def test_service_compute_derived_fields_sets_evidence_map_hash(monkeypatch):
    from atlas_brain.reasoning import evidence_engine

    class _Engine:
        map_hash = "service-derivation-hash"

        def derive_price_complaint(self, result):
            return False

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 5.0

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return None

        def derive_budget_authority(self, result):
            return True

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())
    row, result = _witness_ready_row_and_result()

    derived = service_compute_derived_fields(result, row, deps=_derivation_test_deps())

    assert derived["evidence_map_hash"] == "service-derivation-hash"


def test_service_coerce_legacy_phrase_arrays_extracts_text_from_dict_entries():
    result = {
        "specific_complaints": [
            {"text": "Support was slow"},
            "Billing was confusing",
            {"ignored": True},
        ]
    }

    service_coerce_legacy_phrase_arrays(result)

    assert result["specific_complaints"] == [
        "Support was slow",
        "Billing was confusing",
    ]


def test_service_normalize_tag_value_flattens_unknown_values():
    normalized, was_coerced = service_normalize_tag_value("WEIRD", ("allowed", "unclear"))

    assert normalized == "unclear"
    assert was_coerced is True


def test_service_apply_phrase_metadata_contract_sets_schema_v4_when_rows_are_usable():
    review_id = uuid4()
    result = {
        "specific_complaints": ["Support was slow"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "phrase_metadata": [
            {
                "field": "specific_complaints",
                "index": 0,
                "text": "Support was slow",
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            }
        ],
    }
    row = {
        "id": review_id,
        "summary": "Support was slow",
        "review_text": "Support was slow and unhelpful.",
    }

    service_apply_phrase_metadata_contract(result, row, deps=_phrase_metadata_test_deps())

    assert result["enrichment_schema_version"] == 4
    assert result["phrase_metadata"][0]["verbatim"] is True


def test_service_repair_target_fields_skips_competitor_repair_for_multi_vendor_summary():
    row = {
        "source": "g2",
        "summary": "Asana vs Monday vs Notion vs Trello",
        "review_text": (
            "Asana is strong for structured teams and workflows. Monday.com has great UI. "
            "Notion is loved for docs plus project hybrid use. Trello falls short for growing teams."
        ),
        "pros": "",
        "cons": "",
    }
    result = {
        "pain_category": "overall_dissatisfaction",
        "salience_flags": [],
        "competitors_mentioned": [],
        "specific_complaints": ["limited customization compared to others"],
        "pricing_phrases": [],
        "recommendation_language": [],
        "feature_gaps": ["limited customization compared to others"],
        "event_mentions": [],
        "timeline": {"decision_timeline": "unknown"},
    }

    targets = service_repair_target_fields(result, row, deps=_repair_test_deps())

    assert "competitors_mentioned" not in targets


def test_service_apply_structural_repair_backfills_only_unknown_fields():
    baseline = {
        "urgency_score": 8,
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": True},
        "buyer_authority": {"role_type": "unknown", "buying_stage": "unknown"},
        "timeline": {"decision_timeline": "unknown"},
        "contract_context": {"contract_value_signal": "unknown", "usage_duration": None},
    }
    repair = {
        "urgency_score": 2,
        "churn_signals": {"intent_to_leave": False, "actively_evaluating": False},
        "buyer_authority": {"role_type": "economic_buyer", "buying_stage": "renewal_decision"},
        "timeline": {"decision_timeline": "within_quarter"},
        "contract_context": {"contract_value_signal": "enterprise_mid", "usage_duration": "2_years"},
    }

    merged, applied = service_apply_structural_repair(baseline, repair, deps=_repair_test_deps())

    assert merged["urgency_score"] == 8
    assert merged["churn_signals"]["intent_to_leave"] is True
    assert merged["buyer_authority"]["role_type"] == "economic_buyer"
    assert merged["timeline"]["decision_timeline"] == "within_quarter"
    assert merged["contract_context"]["contract_value_signal"] == "enterprise_mid"
    assert "buyer_authority.role_type" in applied
    assert "timeline.decision_timeline" in applied
    assert "contract_context.contract_value_signal" in applied


def test_service_infer_role_level_from_text_aliases():
    row = {"summary": "", "review_text": "", "pros": "", "cons": ""}

    assert service_infer_role_level_from_text("PMO", row, deps=_buyer_authority_test_deps()) == "manager"
    assert service_infer_role_level_from_text("Product", row, deps=_buyer_authority_test_deps()) == "ic"
    assert service_infer_role_level_from_text("Owner/Managing Member", row, deps=_buyer_authority_test_deps()) == "executive"


def test_service_derive_buyer_authority_fields_defaults_structured_reviews_to_post_purchase():
    result = {
        "reviewer_context": {"role_level": "manager", "decision_maker": False},
        "churn_signals": {
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
    }
    row = {
        "source": "g2",
        "summary": "Reliable tool after rollout",
        "review_text": "We use this product every day across the team.",
        "pros": "",
        "cons": "",
    }

    _, _, buying_stage = service_derive_buyer_authority_fields(result, row, deps=_buyer_authority_test_deps())

    assert buying_stage == "post_purchase"


def test_service_derive_decision_timeline_uses_raw_text_when_commercial_context_exists():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": ["pricing pressure"],
        "specific_complaints": ["We need to decide next quarter"],
        "competitors_mentioned": [],
    }
    row = {
        "summary": "Renewal decision coming soon",
        "review_text": "Pricing pressure means we need to decide next quarter whether to switch.",
        "pros": "",
        "cons": "",
    }

    assert service_derive_decision_timeline(result, row, deps=_timeline_test_deps()) == "within_quarter"


def test_service_derive_concrete_timeline_fields_promotes_contract_notice_into_evaluation_deadline():
    row = {
        "summary": "Cancel before renewal",
        "review_text": (
            "We need to give 30 days notice before renewal or they auto renew the contract. "
            "Support refused to help us cancel."
        ),
        "pros": "",
        "cons": "",
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "contract_renewal_mentioned": True,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "event_mentions": [],
        "timeline": {},
        "pricing_phrases": [],
        "specific_complaints": ["Support refused to help us cancel."],
        "competitors_mentioned": [],
    }

    contract_end, evaluation_deadline = service_derive_concrete_timeline_fields(result, row, deps=_timeline_test_deps())

    assert contract_end == "renewal"
    assert evaluation_deadline == "30 days"


@pytest.mark.asyncio
async def test_run_limits_rounds_and_reports_orphan_recovery(monkeypatch):
    rows = [{"id": uuid4(), "enrichment_attempts": 0}]
    pool = _Pool([rows, rows])

    monkeypatch.setattr(b2b_enrichment, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        b2b_enrichment,
        "_recover_orphaned_enriching",
        AsyncMock(return_value=5),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_queue_version_upgrades",
        AsyncMock(return_value=2),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_enrich_rows",
        AsyncMock(return_value={"enriched": 3, "failed": 1, "no_signal": 2}),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_fetch_review_funnel_audit",
        AsyncMock(return_value={"found": 0, "enriched": 0}),
    )

    cfg = b2b_enrichment.settings.b2b_churn
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "enrichment_max_per_batch", 10)
    monkeypatch.setattr(cfg, "enrichment_max_attempts", 3)
    monkeypatch.setattr(cfg, "enrichment_max_rounds_per_run", 1)
    monkeypatch.setattr(cfg, "enrichment_priority_sources", "stackoverflow,github")

    result = await b2b_enrichment.run(_task())

    assert result["rounds"] == 1
    assert result["orphaned_requeued"] == 5
    assert result["version_upgrade_requeued"] == 2
    assert result["enriched"] == 3
    assert pool.fetch.await_count == 1
    fetch_query, max_attempts, max_batch, priority_sources = pool.fetch.await_args.args
    assert "source = ANY($3::text[])" in fetch_query
    assert "imported_at DESC" in fetch_query
    assert max_attempts == 3
    assert max_batch == 10
    assert priority_sources == ["stackoverflow", "github"]


@pytest.mark.asyncio
async def test_run_applies_manual_metadata_overrides(monkeypatch):
    rows = [{"id": uuid4(), "enrichment_attempts": 0}]
    pool = _Pool([rows, []])
    task = _task()
    task.metadata = {
        "builtin_handler": "b2b_enrichment",
        "enrichment_max_per_batch": 25,
        "enrichment_max_rounds_per_run": 3,
        "enrichment_concurrency": 17,
    }

    monkeypatch.setattr(b2b_enrichment, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        b2b_enrichment,
        "_recover_orphaned_enriching",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_queue_version_upgrades",
        AsyncMock(return_value=0),
    )
    enrich_rows = AsyncMock(return_value={"enriched": 1, "failed": 0, "no_signal": 0})
    monkeypatch.setattr(b2b_enrichment, "_enrich_rows", enrich_rows)
    monkeypatch.setattr(
        b2b_enrichment,
        "_fetch_review_funnel_audit",
        AsyncMock(return_value={"found": 0, "enriched": 0}),
    )

    cfg = b2b_enrichment.settings.b2b_churn
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "enrichment_max_per_batch", 10)
    monkeypatch.setattr(cfg, "enrichment_max_attempts", 3)
    monkeypatch.setattr(cfg, "enrichment_max_rounds_per_run", 1)
    monkeypatch.setattr(cfg, "enrichment_concurrency", 10)
    monkeypatch.setattr(cfg, "enrichment_priority_sources", "stackoverflow,github")

    result = await b2b_enrichment.run(task)

    assert result["rounds"] == 1
    fetch_query, max_attempts, max_batch, priority_sources = pool.fetch.await_args_list[0].args
    assert "source = ANY($3::text[])" in fetch_query
    assert max_attempts == 3
    assert max_batch == 25
    assert priority_sources == ["stackoverflow", "github"]
    assert enrich_rows.await_args.kwargs["concurrency_override"] == 17
    assert enrich_rows.await_args.kwargs["task"] is task


@pytest.mark.asyncio
async def test_recover_orphaned_enriching_parses_update_count():
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 7"))

    count = await b2b_enrichment._recover_orphaned_enriching(pool, 3)

    assert count == 7
    query = pool.execute.await_args.args[0]
    assert "WHERE enrichment_status = 'enriching'" in query


@pytest.mark.asyncio
async def test_enrich_rows_uses_configured_concurrency(monkeypatch):
    active = 0
    max_seen = 0

    async def _fake_enrich_single(pool, row, max_attempts, local_only, max_tokens, truncate_length):
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.01)
        active -= 1
        return True

    monkeypatch.setattr(b2b_enrichment, "_enrich_single", _fake_enrich_single)

    rows = [{"id": uuid4(), "enrichment_attempts": 0} for _ in range(5)]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
    )
    pool = SimpleNamespace(
        fetchval=AsyncMock(return_value=0),
        fetch=AsyncMock(return_value=[{"enrichment_status": "enriched", "ct": 5}]),
    )

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool)

    assert result["enriched"] == 5
    assert max_seen == 2


@pytest.mark.asyncio
async def test_enrich_rows_uses_anthropic_batch_when_enabled(monkeypatch):
    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    row_id = uuid4()
    rows = [{
        "id": row_id,
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Help Desk",
        "source": "reddit",
        "summary": "Pricing is getting rough",
        "review_text": (
            "Pricing is getting rough and support is slower now. "
            "Renewal discussions are already getting tense and the team is actively reviewing options."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "rating": None,
        "rating_max": 5,
        "enrichment_attempts": 0,
    }]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        openrouter_api_key="test-key",
        enrichment_openrouter_model="anthropic/claude-haiku-4-5",
        enrichment_tier1_max_tokens=512,
        enrichment_tier2_max_tokens=512,
        enrichment_tier2_openrouter_model="",
        anthropic_batch_enabled=True,
        enrichment_anthropic_batch_enabled=True,
    )
    task = _task()
    task.metadata["anthropic_batch_enabled"] = True
    task.metadata["enrichment_anthropic_batch_enabled"] = True
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[{"enrichment_status": "enriched", "ct": 1}]))
    persist = AsyncMock(return_value=True)

    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: FakeAnthropicLLM(),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        lambda *args, **_kwargs: SimpleNamespace(namespace="ns", request_envelope={}, provider="openrouter", model="anthropic/claude-haiku-4-5"),
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        AsyncMock(
            return_value=SimpleNamespace(
                local_batch_id="batch-1",
                provider_batch_id="provider-batch-1",
                submitted_items=1,
                cache_prefiltered_items=0,
                fallback_single_call_items=0,
                completed_items=1,
                failed_items=0,
                results_by_custom_id={
                    b2b_enrichment._enrichment_batch_custom_id("tier1", row_id): SimpleNamespace(
                        response_text=json.dumps({
                            "specific_complaints": ["pricing pressure"],
                            "quotable_phrases": [],
                        }),
                        cached=False,
                        usage={},
                        error_text=None,
                    )
                },
            )
        ),
    )
    monkeypatch.setattr(b2b_enrichment, "_persist_enrichment_result", persist)
    monkeypatch.setattr(b2b_enrichment, "_tier1_has_extraction_gaps", lambda *_args, **_kwargs: False)

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool, run_id="run-1", task=task)

    assert result["anthropic_batch_jobs"] == 1
    assert result["anthropic_batch_items_submitted"] == 1
    assert result["enriched"] == 1
    assert persist.await_count == 1


@pytest.mark.asyncio
async def test_enrich_rows_reuses_existing_completed_tier1_batch_result(monkeypatch):
    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    row_id = uuid4()
    rows = [{
        "id": row_id,
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Help Desk",
        "source": "reddit",
        "summary": "Pricing is getting rough",
        "review_text": (
            "Pricing is getting rough and support is slower now. "
            "Renewal discussions are already getting tense and the team is actively reviewing options."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "rating": None,
        "rating_max": 5,
        "enrichment_attempts": 0,
    }]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        openrouter_api_key="test-key",
        enrichment_openrouter_model="anthropic/claude-haiku-4-5",
        enrichment_tier1_max_tokens=512,
        enrichment_tier2_max_tokens=512,
        enrichment_tier2_openrouter_model="",
        anthropic_batch_enabled=True,
        enrichment_anthropic_batch_enabled=True,
    )
    task = _task()
    task.metadata["anthropic_batch_enabled"] = True
    task.metadata["enrichment_anthropic_batch_enabled"] = True
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[{"enrichment_status": "enriched", "ct": 1}]))
    persist = AsyncMock(return_value=True)

    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: FakeAnthropicLLM(),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        lambda *args, **_kwargs: SimpleNamespace(namespace="ns", request_envelope={}, provider="openrouter", model="anthropic/claude-haiku-4-5"),
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        b2b_enrichment,
        "prepare_stage_execution",
        AsyncMock(
            return_value=StageExecutionDecision(
                "reuse_batch_result",
                None,
                {
                    "specific_complaints": ["pricing pressure"],
                    "quotable_phrases": [],
                },
                {
                    "state": "succeeded",
                    "cached": False,
                    "response_text": json.dumps({
                        "specific_complaints": ["pricing pressure"],
                        "quotable_phrases": [],
                    }),
                    "custom_id": b2b_enrichment._enrichment_batch_custom_id("tier1", row_id),
                },
            )
        ),
    )
    run_batch = AsyncMock()
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        run_batch,
    )
    monkeypatch.setattr(b2b_enrichment, "_persist_enrichment_result", persist)
    monkeypatch.setattr(b2b_enrichment, "_tier1_has_extraction_gaps", lambda *_args, **_kwargs: False)

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool, run_id="run-1", task=task)

    assert result["anthropic_batch_jobs"] == 0
    assert result["anthropic_batch_reused_completed_items"] == 1
    assert result["enriched"] == 1
    assert persist.await_count == 1
    run_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_enrich_rows_passes_tier1_request_fingerprints_to_batch_reconcile(monkeypatch):
    row_id = uuid4()
    reconcile_mock = AsyncMock(return_value={})
    row = {
        "id": row_id,
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Help Desk",
        "source": "g2",
        "summary": "Pricing is getting rough",
        "review_text": (
            "Pricing pressure and support issues are pushing us to review options. "
            "Renewal discussions are tense, the team is comparing alternatives, "
            "and we need a better answer before the next contract cycle."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "rating": None,
        "rating_max": 5,
        "enrichment_attempts": 0,
    }
    cfg = SimpleNamespace(
        review_truncate_length=3000,
        enrichment_tier1_max_tokens=512,
    )

    def _prepare_request(*args, **kwargs):
        return SimpleNamespace(
            namespace="ns",
            provider="openrouter",
            model="anthropic/claude-haiku-4-5",
            request_envelope={
                "messages": kwargs["messages"],
                "max_tokens": kwargs["max_tokens"],
                "temperature": kwargs["temperature"],
                "response_format": kwargs["response_format"],
            },
        )

    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        _prepare_request,
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.reconcile_existing_batch_artifacts",
        reconcile_mock,
    )

    expected_request = _prepare_request(
        "b2b_enrichment.tier1",
        provider="openrouter",
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "system", "content": "digest/b2b_churn_extraction_tier1 prompt"},
            {"role": "user", "content": json.dumps(b2b_enrichment._build_classify_payload(row, cfg.review_truncate_length))},
        ],
        max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    expected_fingerprint = exact_stage_request_fingerprint(expected_request)

    decision = await prepare_stage_execution(
        pool=SimpleNamespace(),
        llm=object(),
        task_name="b2b_enrichment",
        artifact_type="review_enrichment_tier1",
        artifact_id=str(row_id),
        review_id=row_id,
        stage_id="b2b_enrichment.tier1",
        work_fingerprint="tier1-work",
        request_fingerprint=expected_fingerprint,
        parse_response_text=b2b_enrichment._parse_stage_row_result,
        defer_on_submitted=False,
        reconcile_batch=True,
    )

    assert decision.action == "execute"
    assert reconcile_mock.await_args.kwargs["expected_request_fingerprints"] == {
        str(row_id): expected_fingerprint,
    }


@pytest.mark.asyncio
async def test_enrich_rows_batch_item_metadata_includes_request_fingerprint(monkeypatch):
    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    row_id = uuid4()
    rows = [{
        "id": row_id,
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Help Desk",
        "source": "g2",
        "summary": "Pricing is getting rough",
        "review_text": (
            "Pricing pressure and support issues are pushing us to review options. "
            "Renewal discussions are tense, the team is comparing alternatives, "
            "and we need a better answer before the next contract cycle."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "rating": None,
        "rating_max": 5,
        "enrichment_attempts": 0,
    }]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        openrouter_api_key="test-key",
        enrichment_openrouter_model="anthropic/claude-haiku-4-5",
        enrichment_tier1_max_tokens=512,
        enrichment_tier2_max_tokens=512,
        enrichment_tier2_openrouter_model="",
        anthropic_batch_enabled=True,
        enrichment_anthropic_batch_enabled=True,
    )
    task = _task()
    task.metadata["anthropic_batch_enabled"] = True
    task.metadata["enrichment_anthropic_batch_enabled"] = True
    pool = SimpleNamespace(
        fetch=AsyncMock(return_value=[{"enrichment_status": "enriched", "ct": 1}]),
        execute=AsyncMock(return_value="OK"),
    )
    persist = AsyncMock(return_value=True)
    captured_items = []

    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: FakeAnthropicLLM(),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )

    def _prepare_request(*args, **kwargs):
        return SimpleNamespace(
            namespace="ns",
            provider="openrouter",
            model="anthropic/claude-haiku-4-5",
            request_envelope={
                "messages": kwargs["messages"],
                "max_tokens": kwargs["max_tokens"],
                "temperature": kwargs["temperature"],
                "response_format": kwargs["response_format"],
            },
        )

    async def _run_batch(**kwargs):
        captured_items.extend(kwargs["items"])
        return SimpleNamespace(
            local_batch_id="batch-1",
            provider_batch_id="provider-batch-1",
            submitted_items=1,
            cache_prefiltered_items=0,
            fallback_single_call_items=0,
            completed_items=1,
            failed_items=0,
            results_by_custom_id={
                b2b_enrichment._enrichment_batch_custom_id("tier1", row_id): SimpleNamespace(
                    response_text=json.dumps({
                        "specific_complaints": ["pricing pressure"],
                        "quotable_phrases": [],
                    }),
                    cached=False,
                    usage={},
                    error_text=None,
                )
            },
        )

    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        _prepare_request,
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        _run_batch,
    )
    monkeypatch.setattr(b2b_enrichment, "_persist_enrichment_result", persist)
    monkeypatch.setattr(b2b_enrichment, "_tier1_has_extraction_gaps", lambda *_args, **_kwargs: False)

    await b2b_enrichment._enrich_rows(rows, cfg, pool, run_id="run-1", task=task)

    expected_request = _prepare_request(
        "b2b_enrichment.tier1",
        provider="openrouter",
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "system", "content": "digest/b2b_churn_extraction_tier1 prompt"},
            {"role": "user", "content": json.dumps(b2b_enrichment._build_classify_payload(rows[0], cfg.review_truncate_length))},
        ],
        max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    expected_fingerprint = exact_stage_request_fingerprint(expected_request)

    assert captured_items
    assert captured_items[0].request_metadata["request_fingerprint"] == expected_fingerprint


@pytest.mark.asyncio
async def test_reconcile_existing_batch_artifacts_requires_matching_request_fingerprint():
    request = SimpleNamespace(
        namespace="b2b_enrichment.tier1",
        provider="openrouter",
        model="anthropic/claude-haiku-4-5",
        request_envelope={
            "messages": [
                {"role": "system", "content": "tier1 prompt"},
                {"role": "user", "content": "payload"},
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
    )
    fingerprint = exact_stage_request_fingerprint(request)
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "artifact_id": "review-1",
                    "batch_id": "batch-1",
                    "status": "batch_succeeded",
                    "response_text": json.dumps({"specific_complaints": ["pricing"]}),
                    "error_text": None,
                    "custom_id": "tier1_review-1",
                    "request_metadata": {"request_fingerprint": fingerprint},
                    "batch_status": "completed",
                    "provider_batch_id": "provider-1",
                }
            ]
        ),
    )

    result = await reconcile_existing_batch_artifacts(
        pool=pool,
        llm=None,
        task_name="b2b_enrichment",
        artifact_type="review_enrichment_tier1",
        artifact_ids=["review-1"],
        expected_request_fingerprints={"review-1": fingerprint},
    )

    assert result["review-1"]["state"] == "succeeded"
    assert result["review-1"]["custom_id"] == "tier1_review-1"


@pytest.mark.asyncio
async def test_reconcile_existing_batch_artifacts_skips_mismatched_request_fingerprint():
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "artifact_id": "review-1",
                    "batch_id": "batch-1",
                    "status": "batch_succeeded",
                    "response_text": json.dumps({"specific_complaints": ["pricing"]}),
                    "error_text": None,
                    "custom_id": "tier1_review-1",
                    "request_metadata": {"request_fingerprint": "stale-request"},
                    "batch_status": "completed",
                    "provider_batch_id": "provider-1",
                }
            ]
        ),
    )

    result = await reconcile_existing_batch_artifacts(
        pool=pool,
        llm=None,
        task_name="b2b_enrichment",
        artifact_type="review_enrichment_tier1",
        artifact_ids=["review-1"],
        expected_request_fingerprints={"review-1": "current-request"},
    )

    assert result == {}


def test_prepare_stage_request_keeps_work_fingerprint_backend_invariant(monkeypatch):
    def _prepare_request(stage_id, **kwargs):
        return SimpleNamespace(
            namespace=stage_id,
            provider=kwargs["provider"],
            model=kwargs["model"],
            request_envelope={
                "messages": kwargs["messages"],
                "max_tokens": kwargs["max_tokens"],
                "temperature": kwargs["temperature"],
                "response_format": kwargs.get("response_format"),
                "guided_json": kwargs.get("guided_json"),
            },
        )

    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        _prepare_request,
    )

    request_a, request_fingerprint_a, work_fingerprint_a = b2b_enrichment._prepare_stage_request(
        "b2b_enrichment.tier1",
        provider="openrouter",
        model="anthropic/claude-haiku-4-5",
        system_prompt="tier1 prompt",
        user_content='{"vendor":"Zendesk"}',
        max_tokens=4096,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    request_b, request_fingerprint_b, work_fingerprint_b = b2b_enrichment._prepare_stage_request(
        "b2b_enrichment.tier1",
        provider="vllm",
        model="local-tier1",
        system_prompt="tier1 prompt",
        user_content='{"vendor":"Zendesk"}',
        max_tokens=4096,
        temperature=0.0,
        guided_json={"type": "object"},
    )

    assert request_a.provider == "openrouter"
    assert request_b.provider == "vllm"
    assert request_fingerprint_a != request_fingerprint_b
    assert work_fingerprint_a == work_fingerprint_b


def test_get_base_enrichment_llm_uses_vllm_first(monkeypatch):
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    with patch("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm):
        llm = b2b_enrichment._get_base_enrichment_llm(local_only=False)

    assert llm is not None
    assert calls == [{
        "workload": "vllm",
        "try_openrouter": False,
        "auto_activate_ollama": False,
    }]


def test_get_base_enrichment_llm_respects_local_only(monkeypatch):
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return None

    with patch("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm):
        llm = b2b_enrichment._get_base_enrichment_llm(local_only=True)

    assert llm is None
    assert calls == [{
        "workload": "vllm",
        "try_openrouter": False,
        "auto_activate_ollama": False,
    }]


@pytest.mark.asyncio
async def test_call_vllm_tier1_uses_exact_cache_hit(monkeypatch):
    class _Registry:
        def get(self, name):
            if name == "digest/b2b_churn_extraction_tier1":
                return SimpleNamespace(content="tier1")
            return None

    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _Registry())
    monkeypatch.setattr(
        b2b_enrichment,
        "_lookup_cached_json_response",
        AsyncMock(
            return_value=(
                {"specific_complaints": ["support delays"], "churn_signals": {"actively_evaluating": True}},
                {"messages": [{"role": "user", "content": "cached"}]},
            )
        ),
    )
    client = SimpleNamespace(
        post=AsyncMock(side_effect=AssertionError("tier1 HTTP should not run on exact-cache hit"))
    )
    cfg = SimpleNamespace(
        enrichment_tier1_model="qwen3-30b",
        enrichment_tier1_max_tokens=512,
    )

    parsed, model = await b2b_enrichment._call_vllm_tier1(
        json.dumps({"vendor_name": "Zendesk"}),
        cfg,
        client,
    )

    assert parsed["specific_complaints"] == ["support delays"]
    assert parsed["churn_signals"]["actively_evaluating"] is True
    assert model == "qwen3-30b"
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_openrouter_tier2_uses_exact_cache_hit(monkeypatch):
    prompt = (
        "## Intro\n"
        "Base instructions.\n\n"
        "### insider_signals -- CLASSIFY + EXTRACT (only for insider_account)\n"
        "Insider-only instructions.\n\n"
        "## Output\n"
        "Return JSON."
    )

    class _Registry:
        def get(self, name):
            if name == "digest/b2b_churn_extraction_tier2":
                return SimpleNamespace(content=prompt)
            return None

    cache_lookup = AsyncMock(
        return_value=(
            {"pain_categories": [{"category": "pricing", "severity": "primary"}]},
            {"messages": [{"role": "user", "content": "cached"}]},
        )
    )
    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _Registry())
    monkeypatch.setattr(b2b_enrichment, "_lookup_cached_json_response", cache_lookup)
    cfg = SimpleNamespace(
        openrouter_api_key="test-key",
        enrichment_tier2_openrouter_model="anthropic/claude-haiku-4-5",
        enrichment_openrouter_model="openai/gpt-oss-120b",
        enrichment_tier2_max_tokens=512,
        enrichment_tier2_timeout_seconds=30.0,
    )
    row = {
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Helpdesk",
        "source": "g2",
        "raw_metadata": {},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
        "summary": "Pricing issues",
        "review_text": "We are evaluating alternatives because pricing keeps rising.",
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Support",
        "reviewer_company": "Acme",
        "company_size_raw": "201-500",
        "reviewer_industry": "SaaS",
    }
    tier1_result = {
        "specific_complaints": ["pricing keeps rising"],
        "quotable_phrases": ["evaluating alternatives"],
    }

    parsed, model = await b2b_enrichment._call_openrouter_tier2(
        tier1_result,
        row,
        cfg,
        truncate_length=3000,
    )

    assert parsed["pain_categories"] == [{"category": "pricing", "severity": "primary"}]
    assert model == "anthropic/claude-haiku-4-5"
    assert cache_lookup.await_count == 1
    assert (
        cache_lookup.await_args.kwargs["system_prompt"]
        == "## Intro\nBase instructions.\n\n## Output\nReturn JSON."
    )


@pytest.mark.asyncio
async def test_call_openrouter_tier1_defaults_to_claude_haiku_when_model_unset(monkeypatch):
    prompt = "Return JSON."

    class _Registry:
        def get(self, name):
            if name == "digest/b2b_churn_extraction_tier1":
                return SimpleNamespace(content=prompt)
            return None

    cache_lookup = AsyncMock(
        return_value=(
            {"specific_complaints": ["support delays"], "churn_signals": {"actively_evaluating": True}},
            {"messages": [{"role": "user", "content": "cached"}]},
        )
    )
    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _Registry())
    monkeypatch.setattr(b2b_enrichment, "_lookup_cached_json_response", cache_lookup)
    cfg = SimpleNamespace(
        openrouter_api_key="test-key",
        enrichment_openrouter_model="",
        enrichment_tier1_max_tokens=512,
    )

    parsed, model = await b2b_enrichment._call_openrouter_tier1(
        json.dumps({"vendor_name": "Zendesk"}),
        cfg,
    )

    assert parsed["specific_complaints"] == ["support delays"]
    assert model == "anthropic/claude-haiku-4-5"
    assert cache_lookup.await_count == 1
    assert cache_lookup.await_args.kwargs["model"] == "anthropic/claude-haiku-4-5"


@pytest.mark.asyncio
async def test_enrich_single_uses_single_pass_tier1_only(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "source": "g2",
        "enrichment_attempts": 0,
        "vendor_name": "Example",
        "product_name": "Example Product",
        "product_category": "CRM",
        "raw_metadata": {},
        "rating_max": 5,
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Sales",
        "reviewer_company": "Acme",
        "company_size_raw": "1001-5000",
        "reviewer_industry": "Technology",
        "content_type": "review",
        "summary": "Switching evaluation",
        "review_text": "We are actively evaluating alternatives after support issues." * 4,
        "rating": 2.0
    }
    tier1_result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {"role_level": "director", "decision_maker": True},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["support issues"],
        "quotable_phrases": ["actively evaluating alternatives"],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "urgency_indicators": {},
        "sentiment_trajectory": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "insider_signals": None,
    }

    tier1_call = AsyncMock(return_value=(tier1_result, "vllm-model"))
    monkeypatch.setattr(
        b2b_enrichment,
        "_get_tier1_client",
        lambda cfg: object(),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_call_vllm_tier1",
        tier1_call,
    )
    monkeypatch.setattr(b2b_enrichment, "_validate_enrichment", lambda result, source_row=None: True)
    monkeypatch.setattr(b2b_enrichment, "_notify_high_urgency", AsyncMock(return_value=None))

    ok = await b2b_enrichment._enrich_single(
        pool,
        row,
        max_attempts=3,
        local_only=True,
        max_tokens=512,
    )

    assert ok is True
    tier1_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_enrich_single_marks_stage_runs_with_stage_local_usage(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "source": "g2",
        "enrichment_attempts": 0,
        "vendor_name": "Example",
        "product_name": "Example Product",
        "product_category": "CRM",
        "raw_metadata": {},
        "rating_max": 5,
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Sales",
        "reviewer_company": "Acme",
        "company_size_raw": "1001-5000",
        "reviewer_industry": "Technology",
        "content_type": "review",
        "summary": "Switching evaluation",
        "review_text": "We are actively evaluating alternatives after support issues." * 4,
        "rating": 2.0,
    }
    tier1_result = {
        "specific_complaints": ["support issues"],
        "quotable_phrases": ["actively evaluating alternatives"],
    }
    tier2_result = {
        "pain_category": "support_quality",
        "pricing_phrases": [],
    }
    ensure_mock = AsyncMock(return_value=None)
    mark_mock = AsyncMock(return_value=None)
    persist_mock = AsyncMock(return_value=True)

    monkeypatch.setattr(b2b_enrichment, "ensure_stage_run", ensure_mock)
    monkeypatch.setattr(b2b_enrichment, "mark_stage_run", mark_mock)
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_prepare_stage_request",
        lambda stage_id, **_kwargs: (
            SimpleNamespace(stage_id=stage_id),
            f"{stage_id}-request-fp",
            f"{stage_id}-work-fp",
        ),
    )
    monkeypatch.setattr(b2b_enrichment, "_get_tier1_client", lambda _cfg: object())
    monkeypatch.setattr(b2b_enrichment, "_get_tier2_client", lambda _cfg: object())
    monkeypatch.setattr(b2b_enrichment, "_resolve_tier_routing", lambda *_args, **_kwargs: (False, False))
    monkeypatch.setattr(b2b_enrichment, "_call_vllm_tier1", AsyncMock(return_value=(tier1_result, "vllm-tier1")))
    monkeypatch.setattr(b2b_enrichment, "_call_vllm_tier2", AsyncMock(return_value=(tier2_result, "vllm-tier2")))
    monkeypatch.setattr(b2b_enrichment, "_tier1_has_extraction_gaps", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(b2b_enrichment, "_validate_enrichment", lambda result, source_row=None: True)
    monkeypatch.setattr(b2b_enrichment, "_notify_high_urgency", AsyncMock(return_value=None))
    monkeypatch.setattr(b2b_enrichment, "_persist_enrichment_result", persist_mock)

    ok = await b2b_enrichment._enrich_single(
        pool,
        row,
        max_attempts=3,
        local_only=True,
        max_tokens=512,
    )

    assert ok is True
    assert mark_mock.await_count == 2
    tier1_mark = mark_mock.await_args_list[0].kwargs
    tier2_mark = mark_mock.await_args_list[1].kwargs
    assert tier1_mark["stage_id"] == "b2b_enrichment.tier1"
    assert tier1_mark["usage"]["tier1_generated_calls"] == 1
    assert tier1_mark["usage"]["generated"] == 1
    assert tier1_mark["usage"]["tier2_generated_calls"] == 0
    assert tier1_mark["work_fingerprint"] == "b2b_enrichment.tier1-work-fp"
    assert json.loads(tier1_mark["response_text"]) == tier1_result
    assert tier2_mark["stage_id"] == "b2b_enrichment.tier2"
    assert tier2_mark["usage"]["tier2_generated_calls"] == 1
    assert tier2_mark["usage"]["generated"] == 1
    assert tier2_mark["usage"]["tier1_generated_calls"] == 0
    assert tier2_mark["work_fingerprint"] == "b2b_enrichment.tier2-work-fp"
    assert json.loads(tier2_mark["response_text"]) == tier2_result


@pytest.mark.asyncio
async def test_enrich_single_reuses_succeeded_stage_run_before_provider_call(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "source": "g2",
        "enrichment_attempts": 0,
        "vendor_name": "Example",
        "product_name": "Example Product",
        "product_category": "CRM",
        "raw_metadata": {},
        "rating_max": 5,
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Sales",
        "reviewer_company": "Acme",
        "company_size_raw": "1001-5000",
        "reviewer_industry": "Technology",
        "content_type": "review",
        "summary": "Switching evaluation",
        "review_text": "We are actively evaluating alternatives after support issues." * 4,
        "rating": 2.0,
    }
    tier1_result = {
        "specific_complaints": ["support issues"],
        "quotable_phrases": ["actively evaluating alternatives"],
    }
    tier1_call = AsyncMock()
    persist_mock = AsyncMock(return_value=True)

    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )
    monkeypatch.setattr(b2b_enrichment, "_resolve_tier_routing", lambda *_args, **_kwargs: (False, False))
    monkeypatch.setattr(
        b2b_enrichment,
        "_prepare_stage_request",
        lambda stage_id, **_kwargs: (
            SimpleNamespace(stage_id=stage_id),
            f"{stage_id}-request-fp",
            f"{stage_id}-work-fp",
        ),
    )
    monkeypatch.setattr(b2b_enrichment, "_get_tier1_client", lambda _cfg: object())
    monkeypatch.setattr(b2b_enrichment, "_call_vllm_tier1", tier1_call)
    monkeypatch.setattr(
        b2b_enrichment,
        "prepare_stage_execution",
        AsyncMock(
            return_value=StageExecutionDecision(
                "reuse_stage",
                {
                    "state": "succeeded",
                    "model": "vllm-tier1",
                    "result_source": "generated",
                    "response_text": json.dumps(tier1_result),
                    "usage_json": {"tier1_generated_calls": 1, "generated": 1},
                },
                tier1_result,
                None,
            )
        ),
    )
    monkeypatch.setattr(b2b_enrichment, "_tier1_has_extraction_gaps", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(b2b_enrichment, "_validate_enrichment", lambda result, source_row=None: True)
    monkeypatch.setattr(b2b_enrichment, "_notify_high_urgency", AsyncMock(return_value=None))
    monkeypatch.setattr(b2b_enrichment, "_persist_enrichment_result", persist_mock)

    ok = await b2b_enrichment._enrich_single(
        pool,
        row,
        max_attempts=3,
        local_only=True,
        max_tokens=512,
    )

    assert ok is True
    tier1_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_enrich_single_defers_when_stage_run_is_submitted(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "source": "g2",
        "enrichment_attempts": 0,
        "vendor_name": "Example",
        "product_name": "Example Product",
        "product_category": "CRM",
        "raw_metadata": {},
        "rating_max": 5,
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Sales",
        "reviewer_company": "Acme",
        "company_size_raw": "1001-5000",
        "reviewer_industry": "Technology",
        "content_type": "review",
        "summary": "Switching evaluation",
        "review_text": "We are actively evaluating alternatives after support issues." * 4,
        "rating": 2.0,
    }
    tier1_call = AsyncMock()

    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )
    monkeypatch.setattr(b2b_enrichment, "_resolve_tier_routing", lambda *_args, **_kwargs: (False, False))
    monkeypatch.setattr(
        b2b_enrichment,
        "_prepare_stage_request",
        lambda stage_id, **_kwargs: (
            SimpleNamespace(stage_id=stage_id),
            f"{stage_id}-request-fp",
            f"{stage_id}-work-fp",
        ),
    )
    monkeypatch.setattr(b2b_enrichment, "_get_tier1_client", lambda _cfg: object())
    monkeypatch.setattr(b2b_enrichment, "_call_vllm_tier1", tier1_call)
    monkeypatch.setattr(
        b2b_enrichment,
        "prepare_stage_execution",
        AsyncMock(
            return_value=StageExecutionDecision(
                "defer_submitted_stage",
                {
                    "state": "submitted",
                    "batch_custom_id": "tier1_pending",
                },
                None,
                None,
            )
        ),
    )

    status = await b2b_enrichment._enrich_single(
        pool,
        row,
        max_attempts=3,
        local_only=True,
        max_tokens=512,
    )

    assert status == "deferred"
    tier1_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_stage_run_defers_submitted_state():
    pool = SimpleNamespace(
        fetchrow=AsyncMock(
            return_value={
                "state": "submitted",
                "batch_custom_id": "tier2_pending",
                "response_text": "",
            }
        )
    )

    resolution = await resolve_stage_run(
        pool,
        review_id="review-1",
        stage_id="b2b_enrichment.tier2",
        work_fingerprint="tier2-work",
        parse_response_text=b2b_enrichment._parse_stage_row_result,
        defer_on_submitted=True,
    )

    assert resolution.action == "defer"
    assert resolution.stage_row["batch_custom_id"] == "tier2_pending"
    assert resolution.parsed_result is None


@pytest.mark.asyncio
async def test_prepare_stage_execution_reuses_pending_batch_result(monkeypatch):
    stage_row = {
        "state": "submitted",
        "batch_custom_id": "tier2_pending",
    }
    batch_result = {
        "state": "pending",
        "custom_id": "tier2_pending",
        "batch_id": "batch-1",
    }
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.get_stage_run",
        AsyncMock(return_value=stage_row),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.reconcile_existing_batch_artifacts",
        AsyncMock(return_value={"review-1": batch_result}),
    )

    decision = await prepare_stage_execution(
        pool=SimpleNamespace(),
        llm=object(),
        task_name="b2b_enrichment",
        artifact_type="review_enrichment_tier2",
        artifact_id="review-1",
        review_id="review-1",
        stage_id="b2b_enrichment.tier2",
        work_fingerprint="tier2-work",
        request_fingerprint="request-fp",
        parse_response_text=b2b_enrichment._parse_stage_row_result,
        defer_on_submitted=False,
        reconcile_batch=True,
    )

    assert decision.action == "defer_pending_batch"
    assert decision.stage_row == stage_row
    assert decision.batch_result == batch_result


@pytest.mark.asyncio
async def test_prepare_stage_execution_reuses_completed_batch_result(monkeypatch):
    batch_result = {
        "state": "succeeded",
        "cached": False,
        "response_text": json.dumps({"specific_complaints": ["pricing"]}),
        "custom_id": "tier1_review-1",
        "batch_id": "batch-1",
    }
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.get_stage_run",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.reconcile_existing_batch_artifacts",
        AsyncMock(return_value={"review-1": batch_result}),
    )

    decision = await prepare_stage_execution(
        pool=SimpleNamespace(),
        llm=object(),
        task_name="b2b_enrichment",
        artifact_type="review_enrichment_tier1",
        artifact_id="review-1",
        review_id="review-1",
        stage_id="b2b_enrichment.tier1",
        work_fingerprint="tier1-work",
        request_fingerprint="request-fp",
        parse_response_text=b2b_enrichment._parse_stage_row_result,
        defer_on_submitted=False,
        reconcile_batch=True,
    )

    assert decision.action == "reuse_batch_result"
    assert decision.parsed_result == {"specific_complaints": ["pricing"]}
    assert decision.batch_result == batch_result


@pytest.mark.asyncio
async def test_apply_stage_decision_marks_reused_batch_result(monkeypatch):
    mark_stage_run_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.mark_stage_run",
        mark_stage_run_mock,
    )

    applied = await apply_stage_decision(
        pool=SimpleNamespace(),
        decision=StageExecutionDecision(
            "reuse_batch_result",
            None,
            {"specific_complaints": ["pricing"]},
            {
                "batch_id": "batch-1",
                "custom_id": "tier1_review-1",
                "cached": False,
                "response_text": json.dumps({"specific_complaints": ["pricing"]}),
            },
        ),
        review_id="review-1",
        stage_id="b2b_enrichment.tier1",
        work_fingerprint="work-1",
        tier=1,
        usage_from_stage_row=b2b_enrichment._stage_usage_from_row,
        pending_metadata={"tier": 1, "workload": "anthropic_batch_pending"},
        success_metadata={"tier": 1, "workload": "anthropic_batch_reuse"},
        stage_usage_snapshot=b2b_enrichment._stage_usage_snapshot,
    )

    assert applied is not None
    assert applied.cache_hit is False
    assert applied.usage["tier1_generated_calls"] == 1
    assert applied.parsed_result == {"specific_complaints": ["pricing"]}
    assert mark_stage_run_mock.await_args.kwargs["state"] == "succeeded"
    assert mark_stage_run_mock.await_args.kwargs["result_source"] == "batch_reuse"


@pytest.mark.asyncio
async def test_apply_stage_decision_materializes_reuse_stage_without_write(monkeypatch):
    mark_stage_run_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.mark_stage_run",
        mark_stage_run_mock,
    )

    applied = await apply_stage_decision(
        pool=SimpleNamespace(),
        decision=StageExecutionDecision(
            "reuse_stage",
            {
                "model": "anthropic/claude-haiku-4-5",
                "result_source": "exact_cache",
                "response_text": json.dumps({"specific_complaints": ["pricing"]}),
                "usage_json": {"tier1_exact_cache_hits": 1, "exact_cache_hits": 1},
            },
            {"specific_complaints": ["pricing"]},
            None,
        ),
        review_id="review-1",
        stage_id="b2b_enrichment.tier1",
        work_fingerprint="work-1",
        tier=1,
        usage_from_stage_row=b2b_enrichment._stage_usage_from_row,
        pending_metadata={"tier": 1, "workload": "direct"},
        success_metadata={"tier": 1, "workload": "direct"},
        stage_usage_snapshot=b2b_enrichment._stage_usage_snapshot,
    )

    assert applied is not None
    assert applied.action == "reuse_stage"
    assert applied.cache_hit is True
    assert applied.model == "anthropic/claude-haiku-4-5"
    assert applied.usage["tier1_exact_cache_hits"] == 1
    mark_stage_run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_stage_decision_materializes_submitted_defer_without_write(monkeypatch):
    mark_stage_run_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.mark_stage_run",
        mark_stage_run_mock,
    )

    applied = await apply_stage_decision(
        pool=SimpleNamespace(),
        decision=StageExecutionDecision(
            "defer_submitted_stage",
            {
                "batch_custom_id": "tier2_pending",
                "batch_id": "batch-1",
            },
            None,
            None,
        ),
        review_id="review-1",
        stage_id="b2b_enrichment.tier2",
        work_fingerprint="work-2",
        tier=2,
        usage_from_stage_row=b2b_enrichment._stage_usage_from_row,
        pending_metadata={"tier": 2, "workload": "direct"},
        success_metadata={"tier": 2, "workload": "direct"},
        stage_usage_snapshot=b2b_enrichment._stage_usage_snapshot,
    )

    assert applied is not None
    assert applied.action == "defer_submitted_stage"
    assert applied.custom_id == "tier2_pending"
    assert applied.batch_id == "batch-1"
    mark_stage_run_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_review_transition_merges_and_persists():
    persist_review = AsyncMock(return_value={"status": "enriched"})

    result = await persist_review_transition(
        row={"id": "review-1"},
        tier1_result={"specific_complaints": ["pricing"]},
        tier2_result={"pain_categories": ["budget"]},
        model_id="hybrid:tier1+tier2",
        usage={"generated": 2},
        merge_results=lambda tier1, tier2: {"tier1": tier1, "tier2": tier2},
        persist_review=persist_review,
    )

    assert result == {"status": "enriched"}
    assert persist_review.await_args.args[1] == {
        "tier1": {"specific_complaints": ["pricing"]},
        "tier2": {"pain_categories": ["budget"]},
    }


@pytest.mark.asyncio
async def test_apply_review_stage_transition_defers_review():
    defer_review = AsyncMock(return_value={"status": "deferred"})

    result = await apply_review_stage_transition(
        applied=SimpleNamespace(
            action="defer_pending_batch",
            parsed_result=None,
            usage=None,
            custom_id="tier2_pending",
        ),
        row={"id": "review-1"},
        tier="tier2",
        usage={"generated": 1},
        tier1_result={"specific_complaints": ["pricing"]},
        model_id="hybrid:tier1+tier2",
        accumulate_usage=lambda *_args, **_kwargs: None,
        merge_results=lambda tier1, tier2: {"tier1": tier1, "tier2": tier2},
        persist_review=AsyncMock(),
        defer_review=defer_review,
    )

    assert result is not None
    assert result.action == "defer_pending_batch"
    assert result.row_result == {"status": "deferred"}
    assert defer_review.await_args.kwargs["tier"] == "tier2"
    assert defer_review.await_args.kwargs["custom_id"] == "tier2_pending"


@pytest.mark.asyncio
async def test_submit_stage_batch_marks_entries_submitted(monkeypatch):
    execution = SimpleNamespace(
        status="running",
        local_batch_id="batch-local-1",
        provider_batch_id="provider-batch-1",
        submitted_items=2,
        cache_prefiltered_items=0,
        fallback_single_call_items=0,
        completed_items=0,
        failed_items=0,
    )
    mark_stage_run_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.mark_stage_run",
        mark_stage_run_mock,
    )

    batch = await submit_stage_batch(
        run_batch=AsyncMock(return_value=execution),
        llm=object(),
        stage_id="b2b_enrichment.tier1",
        task_name="b2b_enrichment",
        items=[object(), object()],
        run_id="run-1",
        min_batch_size=2,
        batch_metadata={"stage": "tier1"},
        pool=SimpleNamespace(),
        entries=[
            {"row": {"id": "review-1"}, "work_fingerprint": "work-1"},
            {"row": {"id": "review-2"}, "work_fingerprint": "work-2"},
        ],
        custom_id_for_entry=lambda entry: f"tier1_{entry['row']['id']}",
        pending_metadata={"tier": 1, "workload": "anthropic_batch_pending"},
    )

    assert batch.execution is execution
    assert batch.metrics["jobs"] == 1
    assert mark_stage_run_mock.await_count == 2
    first_call = mark_stage_run_mock.await_args_list[0].kwargs
    assert first_call["state"] == "submitted"
    assert first_call["batch_id"] == "batch-local-1"
    assert first_call["batch_custom_id"] == "tier1_review-1"


@pytest.mark.asyncio
async def test_finalize_stage_batch_treats_missing_result_as_failed(monkeypatch):
    mark_stage_run_mock = AsyncMock(return_value=None)
    record_batch_fallback_mock = AsyncMock(return_value=None)
    store_cached_response_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.enrichment_stage_controller.mark_stage_run",
        mark_stage_run_mock,
    )

    outcomes = await finalize_stage_batch(
        pool=SimpleNamespace(),
        execution=SimpleNamespace(local_batch_id="batch-local-1", results_by_custom_id={}),
        entries=[
            {
                "row": {"id": "review-1"},
                "request": SimpleNamespace(),
                "work_fingerprint": "work-1",
            }
        ],
        stage_id="b2b_enrichment.tier1",
        custom_id_for_entry=lambda entry: f"tier1_{entry['row']['id']}",
        parse_response_text=lambda _text: None,
        normalize_response_text=lambda text: text,
        store_cached_response=store_cached_response_mock,
        stage_usage_snapshot=lambda **_kwargs: {"generated": 1},
        record_batch_fallback=record_batch_fallback_mock,
        success_metadata={"tier": 1, "workload": "anthropic_batch"},
        failure_metadata={"tier": 1, "workload": "anthropic_batch"},
        failure_error_code="tier1_batch_parse_failed",
    )

    assert len(outcomes) == 1
    assert outcomes[0].success is False
    assert outcomes[0].error_text == "tier1_batch_parse_failed"
    assert mark_stage_run_mock.await_args.kwargs["state"] == "failed"
    store_cached_response_mock.assert_not_awaited()
    record_batch_fallback_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_enrich_rows_reuses_stage_run_before_batch_submission(monkeypatch):
    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    row_id = uuid4()
    rows = [{
        "id": row_id,
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Help Desk",
        "source": "g2",
        "summary": "Pricing is getting rough",
        "review_text": (
            "Pricing pressure and support issues are pushing us to review options. "
            "Renewal discussions are tense, the team is comparing alternatives, "
            "and we need a better answer before the next contract cycle."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "rating": None,
        "rating_max": 5,
        "enrichment_attempts": 0,
    }]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        openrouter_api_key="test-key",
        enrichment_openrouter_model="anthropic/claude-haiku-4-5",
        enrichment_tier1_max_tokens=512,
        enrichment_tier2_max_tokens=512,
        enrichment_tier2_openrouter_model="",
        anthropic_batch_enabled=True,
        enrichment_anthropic_batch_enabled=True,
    )
    task = _task()
    task.metadata["anthropic_batch_enabled"] = True
    task.metadata["enrichment_anthropic_batch_enabled"] = True
    pool = SimpleNamespace(
        fetch=AsyncMock(return_value=[{"enrichment_status": "enriched", "ct": 1}]),
        fetchrow=AsyncMock(
            return_value={
                "state": "succeeded",
                "model": "anthropic/claude-haiku-4-5",
                "result_source": "generated",
                "response_text": json.dumps({
                    "specific_complaints": ["pricing pressure"],
                    "quotable_phrases": [],
                }),
                "usage_json": {"tier1_generated_calls": 1, "generated": 1},
            }
        ),
        execute=AsyncMock(return_value="OK"),
    )
    persist = AsyncMock(return_value=True)
    run_batch = AsyncMock()

    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: FakeAnthropicLLM(),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        run_batch,
    )
    monkeypatch.setattr(b2b_enrichment, "_persist_enrichment_result", persist)
    monkeypatch.setattr(b2b_enrichment, "_tier1_has_extraction_gaps", lambda *_args, **_kwargs: False)

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool, run_id="run-1", task=task)

    assert result["enriched"] == 1
    run_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_enrich_rows_single_call_fallback_defers_submitted_stage_run(monkeypatch):
    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    row_id = uuid4()
    rows = [{
        "id": row_id,
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Help Desk",
        "source": "g2",
        "summary": "Pricing is getting rough",
        "review_text": "Pricing pressure and support issues are pushing us to review options." * 4,
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "rating": None,
        "rating_max": 5,
        "enrichment_attempts": 0,
    }]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        openrouter_api_key="test-key",
        enrichment_openrouter_model="anthropic/claude-haiku-4-5",
        enrichment_tier1_max_tokens=512,
        enrichment_tier2_max_tokens=512,
        enrichment_tier2_openrouter_model="",
        anthropic_batch_enabled=True,
        enrichment_anthropic_batch_enabled=True,
    )
    task = _task()
    task.metadata["anthropic_batch_enabled"] = True
    task.metadata["enrichment_anthropic_batch_enabled"] = True
    pool = SimpleNamespace(
        fetch=AsyncMock(return_value=[{"enrichment_status": "pending", "ct": 1}]),
        fetchrow=AsyncMock(return_value=None),
        execute=AsyncMock(return_value="OK"),
    )
    persist = AsyncMock(return_value=True)
    run_batch = AsyncMock()
    tier1_result = {
        "specific_complaints": ["pricing pressure"],
        "quotable_phrases": ["review options"],
    }
    resolution_calls = []

    async def _prepare_stage_execution(*_args, **kwargs):
        resolution_calls.append((kwargs["stage_id"], kwargs.get("defer_on_submitted", False)))
        if kwargs["stage_id"] == "b2b_enrichment.tier1":
            return StageExecutionDecision(
                "reuse_stage",
                {
                    "state": "succeeded",
                    "model": "anthropic/claude-haiku-4-5",
                    "result_source": "generated",
                    "response_text": json.dumps(tier1_result),
                    "usage_json": {"tier1_generated_calls": 1, "generated": 1},
                },
                tier1_result,
                None,
            )
        return StageExecutionDecision(
            "defer_submitted_stage",
            {
                "state": "submitted",
                "batch_custom_id": "tier2_pending",
            },
            None,
            None,
        )

    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    batch_llm_resolutions = iter([FakeAnthropicLLM(), None])
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: next(batch_llm_resolutions),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(
            get=lambda name: SimpleNamespace(content=f"{name} prompt")
            if name in {"digest/b2b_churn_extraction_tier1", "digest/b2b_churn_extraction_tier2"}
            else None
        ),
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        run_batch,
    )
    monkeypatch.setattr(b2b_enrichment, "_persist_enrichment_result", persist)
    monkeypatch.setattr(b2b_enrichment, "_tier1_has_extraction_gaps", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(b2b_enrichment, "prepare_stage_execution", AsyncMock(side_effect=_prepare_stage_execution))

    await b2b_enrichment._enrich_rows(rows, cfg, pool, run_id="run-1", task=task)

    assert persist.await_count == 0
    run_batch.assert_not_awaited()
    assert resolution_calls == [
        ("b2b_enrichment.tier1", False),
        ("b2b_enrichment.tier2", True),
    ]




def test_detect_low_fidelity_reasons_flags_vendor_absent_noisy_source(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter",
        raising=False,
    )
    row = {
        "source": "hackernews",
        "vendor_name": "Shopify",
        "product_name": "Shopify",
        "summary": "I switched from Spotify back to Apple Music",
        "review_text": "Spotify playback is better. Apple Music failed in the car.",
        "pros": "",
        "cons": "",
    }
    result = {
        "competitors_mentioned": [{"name": "Spotify"}],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "vendor_absent_noisy_source" in reasons
    assert "competitor_only_context" in reasons


def test_normalized_low_fidelity_noisy_sources_merges_required_defaults(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter,github,stackoverflow",
        raising=False,
    )

    values = b2b_enrichment._normalized_low_fidelity_noisy_sources()

    assert "reddit" in values
    assert "software_advice" in values
    assert "hackernews" in values


def test_service_normalized_low_fidelity_noisy_sources_merges_defaults():
    values = service_normalized_low_fidelity_noisy_sources(
        "hackernews,quora,twitter,github,stackoverflow",
        b2b_enrichment.B2BChurnConfig.model_fields["enrichment_low_fidelity_noisy_sources"].default,
    )

    assert "reddit" in values
    assert "software_advice" in values
    assert "hackernews" in values


def test_detect_low_fidelity_reasons_keeps_vendor_present_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter",
        raising=False,
    )
    row = {
        "source": "hackernews",
        "vendor_name": "Intercom",
        "product_name": "Intercom",
        "summary": "We use Intercom for support and need secure file uploads",
        "review_text": "Intercom file limits are a long-term issue for our support workflow.",
        "pros": "",
        "cons": "",
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, {"competitors_mentioned": []})

    assert reasons == []


def test_detect_low_fidelity_reasons_flags_thin_reddit_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "reddit,hackernews,quora,twitter",
        raising=False,
    )
    row = {
        "source": "reddit",
        "vendor_name": "HubSpot",
        "product_name": "HubSpot",
        "summary": "Anyone switch?",
        "review_text": "Thinking about options.",
        "pros": "",
        "cons": "",
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, {"urgency_score": 2, "competitors_mentioned": []})

    assert "thin_social_context" in reasons


def test_detect_low_fidelity_reasons_flags_thin_software_advice_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "software_advice",
        raising=False,
    )
    row = {
        "source": "software_advice",
        "vendor_name": "HubSpot",
        "product_name": "HubSpot",
        "summary": "Bad",
        "review_text": "Too pricey.",
        "pros": "",
        "cons": "",
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, {"urgency_score": 3, "competitors_mentioned": []})

    assert "thin_review_platform_context" in reasons


def test_detect_low_fidelity_reasons_flags_technical_stackoverflow_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "stackoverflow,github",
        raising=False,
    )
    row = {
        "source": "stackoverflow",
        "vendor_name": "Intercom",
        "product_name": "Intercom",
        "summary": "How can I integrate Intercom in Xamarin.Forms?",
        "review_text": "I am trying to add Intercom in Xamarin.Forms for iOS and things are not going well.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 2,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "technical_question_context" in reasons


def test_detect_low_fidelity_reasons_keeps_commercial_stackoverflow_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "stackoverflow,github",
        raising=False,
    )
    row = {
        "source": "stackoverflow",
        "vendor_name": "Jira",
        "product_name": "Jira",
        "summary": "Looking for Jira alternatives before renewal",
        "review_text": "We are evaluating alternatives to Jira because renewal pricing doubled for our team.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 7,
        "competitors_mentioned": [{"name": "Linear"}],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "technical_question_context" not in reasons


def test_detect_low_fidelity_reasons_flags_consumer_trustpilot_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter,github,stackoverflow",
        raising=False,
    )
    row = {
        "source": "trustpilot",
        "vendor_name": "Microsoft Teams",
        "product_name": "Microsoft Teams",
        "summary": "Emailed copilot via app ghosting Email address on Google play",
        "review_text": "This app downloaded Microsoft 365 was a free version. Contacted app support via app and got a ghosting email from Google Play.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 3,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "consumer_support_context" in reasons


def test_detect_low_fidelity_reasons_keeps_commercial_trustpilot_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter,github,stackoverflow",
        raising=False,
    )
    row = {
        "source": "trustpilot",
        "vendor_name": "HubSpot",
        "product_name": "HubSpot",
        "summary": "Great product, but every wiggle cost money",
        "review_text": "HubSpot has powerful tools but the costs climb steeply and support wants to charge more for every change.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 4,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "consumer_support_context" not in reasons


def test_detect_low_fidelity_reasons_keeps_trustpilot_vendor_absent_false_positive(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter,github,stackoverflow",
        raising=False,
    )
    row = {
        "source": "trustpilot",
        "vendor_name": "RingCentral",
        "product_name": "RingCentral",
        "summary": "Ring Central's servicing is very poor",
        "review_text": "Ring Central's servicing is very poor. They commit to having a supervisor call to assist with unclear contract terms with no follow through.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 4,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "vendor_absent_noisy_source" not in reasons
    assert "consumer_support_context" not in reasons


def test_text_mentions_name_matches_compact_vendor_with_spaced_text():
    haystack = b2b_enrichment._normalize_compare_text(
        "Ring Central's servicing is very poor and support did not clarify the contract."
    )

    assert b2b_enrichment._text_mentions_name(haystack, "RingCentral") is True


def test_service_text_and_coercion_support_helpers_preserve_behavior():
    haystack = service_normalize_compare_text(
        "Ring Central's servicing is very poor and support did not clarify the contract."
    )

    assert service_text_mentions_name(
        haystack,
        "RingCentral",
        low_fidelity_token_stopwords=b2b_enrichment._LOW_FIDELITY_TOKEN_STOPWORDS,
    ) is True
    assert service_normalized_name_tokens(
        "The Ring Central platform",
        low_fidelity_token_stopwords=b2b_enrichment._LOW_FIDELITY_TOKEN_STOPWORDS,
    ) == ["ring", "central"]
    assert service_contains_any("pricing doubled at renewal", ("renewal", "cancel")) is True
    assert service_has_technical_context(
        "How do I configure this?",
        "API integration issue",
        low_fidelity_technical_patterns=b2b_enrichment._LOW_FIDELITY_TECHNICAL_PATTERNS,
    ) is True
    assert service_combined_source_text(
        {"summary": "One", "review_text": "Two", "pros": "", "cons": "Three"}
    ) == "One\nTwo\nThree"
    assert service_coerce_bool("true") is True
    assert service_coerce_bool("null") is False
    assert service_coerce_json_dict('{"ok": 1}') == {"ok": 1}
    assert service_coerce_json_dict("[]") == {}


def test_apply_structural_repair_promotes_only_unknown_fields():
    baseline = {
        "urgency_score": 8,
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": True},
        "buyer_authority": {"role_type": "unknown", "buying_stage": "unknown"},
        "timeline": {"decision_timeline": "unknown"},
        "contract_context": {"contract_value_signal": "unknown", "usage_duration": None},
    }
    repair = {
        "urgency_score": 2,
        "churn_signals": {"intent_to_leave": False, "actively_evaluating": False},
        "buyer_authority": {"role_type": "economic_buyer", "buying_stage": "renewal_decision"},
        "timeline": {"decision_timeline": "within_quarter"},
        "contract_context": {"contract_value_signal": "enterprise_mid", "usage_duration": "2_years"},
    }

    merged, applied = b2b_enrichment._apply_structural_repair(baseline, repair)

    assert merged["urgency_score"] == 8
    assert merged["churn_signals"]["intent_to_leave"] is True
    assert merged["buyer_authority"]["role_type"] == "economic_buyer"
    assert merged["timeline"]["decision_timeline"] == "within_quarter"
    assert merged["contract_context"]["contract_value_signal"] == "enterprise_mid"
    assert "buyer_authority.role_type" in applied
    assert "timeline.decision_timeline" in applied
    assert "contract_context.contract_value_signal" in applied


def test_derive_decision_timeline_uses_raw_text_when_commercial_context_exists():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": ["pricing pressure"],
        "specific_complaints": ["We need to decide next quarter"],
        "competitors_mentioned": [],
    }
    row = {
        "summary": "Renewal decision coming soon",
        "review_text": "Pricing pressure means we need to decide next quarter whether to switch.",
        "pros": "",
        "cons": "",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "within_quarter"


def test_derive_decision_timeline_ignores_generic_deadline_without_commercial_context():
    result = {
        "churn_signals": {},
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": [],
        "specific_complaints": [],
        "competitors_mentioned": [],
    }
    row = {
        "summary": "Helpful for daily work",
        "review_text": "This helps me track tasks and deadlines every week for my team.",
        "pros": "",
        "cons": "",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "unknown"


def test_derive_decision_timeline_ignores_soft_deadline_complaint_without_strong_commercial_signal():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": [],
        "specific_complaints": ["The reminders can get annoying."],
        "competitors_mentioned": [],
    }
    row = {
        "summary": "Helpful but noisy reminders",
        "review_text": "It reminds us about deadlines and action items all the time, which can get annoying.",
        "pros": "",
        "cons": "",
        "source": "software_advice",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "unknown"


def test_derive_decision_timeline_ignores_ambiguous_noisy_source_without_vendor_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "reddit,quora,twitter",
    )
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": [],
        "specific_complaints": ["The zipper design is weak."],
        "competitors_mentioned": [],
    }
    row = {
        "vendor_name": "Copper",
        "product_name": "Copper",
        "summary": "Thinking about replacing my Copper Spur tent next quarter",
        "review_text": "I am considering other tent options next quarter because the zipper design is weak.",
        "pros": "",
        "cons": "",
        "source": "reddit",
        "content_type": "community_discussion",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "unknown"


def test_derive_buyer_authority_fields_keeps_discussion_stage_unknown_without_explicit_buying_motion():
    result = {
        "reviewer_context": {"role_level": "manager", "decision_maker": False},
        "churn_signals": {
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
    }
    row = {
        "source": "reddit",
        "summary": "Helpful but frustrating",
        "review_text": "Support is slow and setup is annoying, but I did not mention any buying process.",
        "pros": "",
        "cons": "",
    }

    role_type, executive_sponsor_mentioned, buying_stage = b2b_enrichment._derive_buyer_authority_fields(result, row)

    assert role_type == "champion"
    assert executive_sponsor_mentioned is False
    assert buying_stage == "unknown"


def test_derive_buyer_authority_fields_defaults_structured_reviews_to_post_purchase():
    result = {
        "reviewer_context": {"role_level": "manager", "decision_maker": False},
        "churn_signals": {
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
    }
    row = {
        "source": "g2",
        "summary": "Reliable tool after rollout",
        "review_text": "We use this product every day across the team.",
        "pros": "",
        "cons": "",
    }

    _, _, buying_stage = b2b_enrichment._derive_buyer_authority_fields(result, row)

    assert buying_stage == "post_purchase"


def test_compute_derived_fields_promotes_contract_notice_into_evaluation_deadline(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 7.5

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return True

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "summary": "Cancel before renewal",
        "review_text": (
            "We need to give 30 days notice before renewal or they auto renew the contract. "
            "Support refused to help us cancel."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "Operations Manager",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.8},
        "content_type": "review",
        "rating": 1.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "contract_renewal_mentioned": True,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "manager",
            "decision_maker": False,
            "company_name": "Acme Corp",
        },
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["Support refused to help us cancel."],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["timeline"]["contract_end"] == "renewal"
    assert derived["timeline"]["evaluation_deadline"] == "30 days"
    assert derived["timeline"]["decision_timeline"] == "within_quarter"
    assert any(span["time_anchor"] == "30 days" for span in derived["evidence_spans"])


def test_compute_derived_fields_promotes_event_timeframe_into_contract_end(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 7.1

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "summary": "Renewal planning",
        "review_text": "We are evaluating alternatives ahead of renewal.",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "raw_metadata": {"source_weight": 0.8},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "contract_renewal_mentioned": True,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [{"event": "renewal", "timeframe": "next quarter"}],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["timeline"]["contract_end"] == "next quarter"
    assert derived["timeline"]["decision_timeline"] == "within_quarter"


def test_finalize_enrichment_for_persist_backfills_company_name_from_reviewer_company(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 4.1

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "vendor_name": "HubSpot",
        "summary": "Solid tool",
        "review_text": "We use it daily for sales and support workflows.",
        "pros": "",
        "cons": "",
        "reviewer_title": "Operations Manager",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.8},
        "content_type": "review",
        "rating": 4.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "support_escalation": False,
            "renewal_timing": None,
        },
        "reviewer_context": {"role_level": "manager", "decision_maker": False},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    finalized, error = b2b_enrichment._finalize_enrichment_for_persist(result, row)

    assert error is None
    assert finalized is not None
    assert finalized["reviewer_context"]["company_name"] == "Acme Corp"


def test_finalize_enrichment_for_persist_does_not_copy_vendor_name_into_company_name(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 4.1

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "vendor_name": "HubSpot",
        "summary": "Solid tool",
        "review_text": "We use it daily for sales and support workflows.",
        "pros": "",
        "cons": "",
        "reviewer_title": "Operations Manager",
        "reviewer_company": "HubSpot",
        "raw_metadata": {"source_weight": 0.8},
        "content_type": "review",
        "rating": 4.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "support_escalation": False,
            "renewal_timing": None,
        },
        "reviewer_context": {"role_level": "manager", "decision_maker": False},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    finalized, error = b2b_enrichment._finalize_enrichment_for_persist(result, row)

    assert error is None
    assert finalized is not None
    assert finalized["reviewer_context"].get("company_name") in (None, "")


def test_infer_role_level_from_generic_function_title_aliases():
    row = {"summary": "", "review_text": "", "pros": "", "cons": ""}

    assert b2b_enrichment._infer_role_level_from_text("PMO", row) == "manager"
    assert b2b_enrichment._infer_role_level_from_text("Product", row) == "ic"
    assert b2b_enrichment._infer_role_level_from_text("Owner/Managing Member", row) == "executive"


def test_finalize_enrichment_for_persist_promotes_manager_with_commercial_context_to_decision_maker(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 7.0

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return True

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "vendor_name": "Zendesk",
        "summary": "Renewal decision forced budget review",
        "review_text": (
            "As operations manager, I owned the renewal review after they quoted us $48k/year "
            "before the Q2 contract end."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "Operations Manager",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.9},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "contract_renewal_mentioned": True,
            "support_escalation": False,
            "renewal_timing": None,
        },
        "reviewer_context": {"role_level": "manager", "decision_maker": False},
        "budget_signals": {"annual_spend_estimate": "$48k/year"},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": ["quoted us $48k/year"],
        "event_mentions": [],
        "timeline": {"contract_end": "Q2 2026"},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    finalized, error = b2b_enrichment._finalize_enrichment_for_persist(result, row)

    assert error is None
    assert finalized is not None
    assert finalized["reviewer_context"]["role_level"] == "manager"
    assert finalized["reviewer_context"]["decision_maker"] is True
    assert finalized["buyer_authority"]["role_type"] == "economic_buyer"


def test_finalize_enrichment_for_persist_keeps_generic_manager_non_decision_maker_without_commercial_context(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 3.0

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "vendor_name": "Klaviyo",
        "summary": "Good ESP",
        "review_text": "As marketing manager I use it every day for campaigns and segmentation.",
        "pros": "",
        "cons": "",
        "reviewer_title": "Marketing Manager",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.9},
        "content_type": "review",
        "rating": 4.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "support_escalation": False,
            "renewal_timing": None,
        },
        "reviewer_context": {"role_level": "manager", "decision_maker": False},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    finalized, error = b2b_enrichment._finalize_enrichment_for_persist(result, row)

    assert error is None
    assert finalized is not None
    assert finalized["reviewer_context"]["decision_maker"] is False
    assert finalized["buyer_authority"]["role_type"] == "champion"


def test_compute_derived_fields_promotes_explicit_budget_anchors(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 8.2

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return True

        def derive_price_complaint(self, result):
            return True

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "summary": "Renewal quote forced a decision",
        "review_text": (
            "At renewal they quoted us $200k/year for 200 users, or $85/user/month. "
            "It was a 30% price increase and we started evaluating alternatives."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Operations",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.9},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "contract_renewal_mentioned": True,
            "renewal_timing": "next quarter",
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "director",
            "decision_maker": True,
            "company_name": "Acme Corp",
        },
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["The renewal quote forced us to reevaluate the tool."],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["budget_signals"]["annual_spend_estimate"] == "$200k/year"
    assert derived["budget_signals"]["price_per_seat"] == "$85/user/month"
    assert derived["budget_signals"]["seat_count"] == 200
    assert derived["budget_signals"]["price_increase_mentioned"] is True
    assert derived["budget_signals"]["price_increase_detail"] == "30% price increase"
    assert derived["contract_context"]["contract_value_signal"] == "enterprise_high"


def test_compute_derived_fields_ignores_salary_noise_for_budget_signals(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 1.5

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return None

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "vendor_name": "Copper",
        "product_name": "Copper",
        "summary": "My intern salary this year",
        "review_text": "I got a $140k salary offer as an intern this year and took the job.",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "raw_metadata": {"source_weight": 0.2},
        "content_type": "community_discussion",
        "rating": None,
        "rating_max": 5,
        "source": "reddit",
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {"role_level": "unknown", "decision_maker": False},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "community_discussion",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["budget_signals"] == {}


def test_compute_derived_fields_ignores_ambiguous_noisy_source_budget_noise(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 2.0

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return None

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "reddit,quora,twitter",
    )

    row = {
        "id": uuid4(),
        "vendor_name": "Close",
        "product_name": "Close",
        "summary": "Can't remove the charge? Well, I'll just use it then",
        "review_text": (
            "The apartment complex charged an upcharge of $25 a month for assigned parking. "
            "The lease breaking fees and rent increase would have cost over a thousand dollars."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "raw_metadata": {"source_weight": 0.2},
        "content_type": "community_discussion",
        "rating": None,
        "rating_max": 5,
        "source": "reddit",
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {"role_level": "unknown", "decision_maker": False},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "community_discussion",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": ["upcharge of $25 a month", "lease breaking fees", "over a thousand dollars"],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["budget_signals"] == {}


def test_compute_derived_fields_derives_annual_spend_from_monthly_unit_price_and_seat_count(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 1.5

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return None

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "vendor_name": "HubSpot",
        "product_name": "HubSpot Sales Hub",
        "summary": "Renewal pricing pushed us to reevaluate",
        "review_text": "Our renewal quote came in at $85/user/month for 200 users.",
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Operations",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.9},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "contract_renewal_mentioned": True,
            "renewal_timing": "next quarter",
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "director",
            "decision_maker": True,
            "company_name": "Acme Corp",
        },
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["The renewal quote forced us to reevaluate the tool."],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["budget_signals"]["price_per_seat"] == "$85/user/month"
    assert derived["budget_signals"]["seat_count"] == 200
    assert derived["budget_signals"]["annual_spend_estimate"] == "$204k/year"
    assert derived["contract_context"]["contract_value_signal"] == "enterprise_high"


def test_compute_derived_fields_skips_ambiguous_range_unit_price_for_annual_spend(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 1.5

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return None

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "vendor_name": "SentinelOne",
        "product_name": "SentinelOne Singularity Complete",
        "summary": "Pricing depends on volume",
        "review_text": "The quote was $7 to $10 per agent per month for 6000 agents.",
        "pros": "",
        "cons": "",
        "reviewer_title": "Director of Security",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.9},
        "content_type": "review",
        "rating": 3.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "contract_renewal_mentioned": True,
            "renewal_timing": "this quarter",
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "director",
            "decision_maker": True,
            "company_name": "Acme Corp",
        },
        "budget_signals": {
            "price_per_seat": "$7 to $10 per agent per month",
            "seat_count": 6000,
        },
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["The quote depends heavily on agent volume."],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["budget_signals"]["price_per_seat"] == "$7 to $10 per agent per month"
    assert derived["budget_signals"]["seat_count"] == 6000
    assert derived["budget_signals"].get("annual_spend_estimate") is None


def test_extract_numeric_amount_supports_suffixes():
    assert b2b_enrichment._extract_numeric_amount("$200k/year") == 200000
    assert b2b_enrichment._extract_numeric_amount("USD 1.5m annual contract") == 1500000


def test_normalize_budget_value_text_preserves_readable_time_markers():
    assert b2b_enrichment._normalize_budget_value_text("$8,000 a year") == "$8,000 a year"
    assert b2b_enrichment._normalize_budget_value_text("$8,000ayear") == "$8,000 a year"
    assert b2b_enrichment._normalize_budget_value_text("$10k / year") == "$10k/year"
    assert b2b_enrichment._normalize_budget_value_text("$4/ user/ month") == "$4/user/month"


def test_service_derive_budget_signals_extracts_budget_fields():
    row = {
        "source": "g2",
        "vendor_name": "Zendesk",
        "product_name": "Zendesk Support",
        "summary": "Renewal costs keep climbing",
        "review_text": (
            "Zendesk raised us to $15 per seat per month for 300 seats. "
            "That puts us around $54k/year and the renewal increase was hard to justify."
        ),
        "pros": "",
        "cons": "",
    }
    result = {
        "pricing_phrases": ["$15 per seat per month", "$54k/year", "renewal increase"],
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": True,
        },
        "budget_signals": {},
    }

    budget = service_derive_budget_signals(result, row, deps=_budget_test_deps())

    assert budget["price_per_seat"] == "$15 per seat per month"
    assert budget["annual_spend_estimate"] == "$54k/year"
    assert budget["seat_count"] == 300
    assert budget["price_increase_mentioned"] is True


def test_service_derive_budget_signals_derives_annual_spend_from_unit_price():
    row = {
        "source": "g2",
        "vendor_name": "Intercom",
        "product_name": "Intercom",
        "summary": "Seat pricing added up fast",
        "review_text": "We now pay $20 per seat per month for 250 seats.",
        "pros": "",
        "cons": "",
    }
    result = {
        "pricing_phrases": ["$20 per seat per month", "250 seats"],
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
        },
        "budget_signals": {},
    }

    budget = service_derive_budget_signals(result, row, deps=_budget_test_deps())

    assert budget["price_per_seat"] == "$20 per seat per month"
    assert budget["seat_count"] == 250
    assert budget["annual_spend_estimate"] == "$60k/year"


def test_service_derive_contract_value_signal_uses_budget_and_segment():
    assert service_derive_contract_value_signal(
        {
            "budget_signals": {"annual_spend_estimate": "$120k/year", "seat_count": 75},
            "reviewer_context": {"company_size_segment": "mid_market"},
        }
    ) == "enterprise_high"
    assert service_derive_contract_value_signal(
        {
            "budget_signals": {"annual_spend_estimate": None, "seat_count": 40},
            "reviewer_context": {"company_size_segment": "startup"},
        }
    ) == "mid_market"


@pytest.mark.asyncio
async def test_enrich_rows_counts_quarantined(monkeypatch):
    async def _fake_enrich_single(pool, row, max_attempts, local_only, max_tokens, truncate_length):
        return "quarantined"

    monkeypatch.setattr(b2b_enrichment, "_enrich_single", _fake_enrich_single)

    rows = [{"id": uuid4(), "enrichment_attempts": 0} for _ in range(3)]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
    )
    pool = SimpleNamespace(
        fetchval=AsyncMock(return_value=0),
        fetch=AsyncMock(return_value=[{"enrichment_status": "quarantined", "ct": 3}]),
    )

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool)

    assert result["enriched"] == 0
    assert result["quarantined"] == 3


def test_compute_derived_fields_adds_witness_primitives(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 8.4

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return True

        def derive_price_complaint(self, result):
            return True

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "summary": "Renewal pricing pushed us toward async docs",
        "review_text": (
            "Slack wanted $200k/year at renewal. "
            "We became more productive using docs and async updates instead."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "CTO",
        "reviewer_company": "Hack Club",
        "raw_metadata": {"source_weight": 0.9},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "contract_renewal_mentioned": True,
            "renewal_timing": "next quarter",
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "executive",
            "decision_maker": True,
            "company_name": "Hack Club",
        },
        "budget_signals": {
            "annual_spend_estimate": 200000,
            "price_per_seat": None,
            "seat_count": None,
            "price_increase_mentioned": True,
        },
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["Slack wanted $200k/year at renewal"],
        "quotable_phrases": ["We became more productive using docs and async updates instead"],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": ["I would not recommend this"],
        "pricing_phrases": ["$200k/year at renewal"],
        "event_mentions": [{"event": "renewal", "timeframe": "next quarter"}],
        "timeline": {"contract_end": "next quarter", "evaluation_deadline": None},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["replacement_mode"] == "workflow_substitution"
    assert derived["operating_model_shift"] == "sync_to_async"
    assert derived["productivity_delta_claim"] == "more_productive"
    assert derived["org_pressure_type"] == "none"
    assert {"explicit_dollar", "named_account", "decision_maker"}.issubset(set(derived["salience_flags"]))
    assert derived["evidence_spans"]
    assert any(span["signal_type"] == "pricing_backlash" for span in derived["evidence_spans"])
    assert any(span["productivity_delta_claim"] == "more_productive" for span in derived["evidence_spans"])


def test_compute_derived_fields_positive_pricing_context_does_not_emit_pricing_backlash():
    row = {
        "id": uuid4(),
        "summary": "Pricing fit",
        "review_text": (
            "Trello met all our expectations for sprint management. "
            "I find the pricing reasonable."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": None,
        "reviewer_company": "Infohob",
        "raw_metadata": {"source_weight": 0.8},
        "content_type": "review",
        "rating": 4.5,
        "rating_max": 5,
        "source": "peerspot",
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "executive",
            "decision_maker": True,
            "company_name": "Infohob",
        },
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": ["I find the pricing reasonable."],
        "positive_aspects": ["Trello met all our expectations for sprint management."],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": ["I find the pricing reasonable."],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["contract_context"]["price_complaint"] is False
    assert derived["contract_context"]["price_context"] == "I find the pricing reasonable."
    assert derived["urgency_indicators"]["price_pressure_language"] is False
    assert not any(span["signal_type"] == "pricing_backlash" for span in derived["evidence_spans"])


def test_repair_target_fields_flags_semantic_pricing_and_timeline_gaps():
    row = {
        "summary": "Renewal confusion",
        "review_text": "We got a $50k renewal quote and need to decide next quarter.",
        "pros": "",
        "cons": "",
        "source": "reddit",
        "enrichment_status": "enriched",
    }
    result = {
        "pain_category": "ux",
        "specific_complaints": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "feature_gaps": [],
        "event_mentions": [],
        "competitors_mentioned": [],
        "salience_flags": ["explicit_dollar"],
        "timeline": {"decision_timeline": "unknown"},
    }

    targets = b2b_enrichment._repair_target_fields(result, row)

    assert "specific_complaints" in targets
    assert "pricing_phrases" in targets
    assert "event_mentions" in targets


def _witness_ready_row_and_result():
    row = {
        "id": "review-1",
        "vendor_name": "Slack",
        "source": "g2",
        "content_type": "review",
        "summary": "Renewal pushed us away from Slack",
        "review_text": (
            "Slack wanted $200k/year at renewal. We became more productive using docs "
            "and async updates instead."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Operations",
        "reviewer_company": "Hack Club",
        "rating": 2,
        "rating_max": 5,
        "raw_metadata": {"source_weight": 0.9},
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "support_escalation": False,
            "contract_renewal_mentioned": True,
        },
        "reviewer_context": {
            "role_level": "executive",
            "decision_maker": True,
            "company_name": "Hack Club",
        },
        "budget_signals": {
            "annual_spend_estimate": 200000,
            "price_per_seat": None,
            "seat_count": None,
            "price_increase_mentioned": True,
        },
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["Slack wanted $200k/year at renewal"],
        "quotable_phrases": ["We became more productive using docs and async updates instead"],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": ["I would not recommend this"],
        "pricing_phrases": ["$200k/year at renewal"],
        "event_mentions": [{"event": "renewal", "timeframe": "next quarter"}],
        "timeline": {"contract_end": "next quarter", "evaluation_deadline": None},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }
    return row, result


def test_finalize_enrichment_for_persist_populates_witness_primitives():
    row, result = _witness_ready_row_and_result()

    finalized, error = b2b_enrichment._finalize_enrichment_for_persist(result, row)

    assert error is None
    assert finalized is not None
    assert finalized["replacement_mode"] == "workflow_substitution"
    assert finalized["operating_model_shift"] == "sync_to_async"
    assert finalized["productivity_delta_claim"] == "more_productive"
    assert finalized["org_pressure_type"] == "none"
    assert finalized["salience_flags"]
    assert finalized["evidence_spans"]
    assert finalized["evidence_map_hash"]


def test_service_finalize_enrichment_for_persist_rejects_invalid_payload():
    finalized, error = service_finalize_enrichment_for_persist(
        "not-a-dict",
        {"id": "review-1"},
        deps=EnrichmentFinalizationDeps(
            compute_derived_fields=lambda payload, _row: payload,
            validate_enrichment=lambda payload, _row: True,
        ),
    )

    assert finalized is None
    assert error == "invalid_payload"


def test_service_finalize_enrichment_for_persist_returns_compute_failed():
    def _raise_compute(_payload, _row):
        raise RuntimeError("boom")

    finalized, error = service_finalize_enrichment_for_persist(
        {"churn_signals": {}, "urgency_score": 1},
        {"id": "review-1"},
        deps=EnrichmentFinalizationDeps(
            compute_derived_fields=_raise_compute,
            validate_enrichment=lambda payload, _row: True,
        ),
    )

    assert finalized is None
    assert error == "compute_failed"


def test_service_finalize_enrichment_for_persist_returns_validation_failed():
    finalized, error = service_finalize_enrichment_for_persist(
        {"churn_signals": {}, "urgency_score": 1},
        {"id": "review-1"},
        deps=EnrichmentFinalizationDeps(
            compute_derived_fields=lambda payload, _row: payload,
            validate_enrichment=lambda _payload, _row: False,
        ),
    )

    assert finalized is None
    assert error == "validation_failed"


@pytest.mark.asyncio
async def test_service_persist_enrichment_result_increments_attempts_on_empty_result():
    pool = SimpleNamespace(execute=AsyncMock(return_value="OK"))
    increment_attempts = AsyncMock(return_value=None)

    status = await service_persist_enrichment_result(
        pool,
        {"id": uuid4(), "enrichment_attempts": 1},
        None,
        model_id="anthropic/claude-haiku-4-5",
        max_attempts=3,
        run_id="run-1",
        cache_usage={"secondary_write_hits": 0, "witness_rows": 0, "witness_count": 0},
        deps=EnrichmentPersistenceDeps(
            finalize_enrichment_for_persist=lambda result, row: (result, None),
            witness_metrics=lambda _result: (0, 0),
            detect_low_fidelity_reasons=lambda _row, _result: [],
            is_no_signal_result=lambda _result, _row: False,
            notify_high_urgency=AsyncMock(return_value=None),
            increment_attempts=increment_attempts,
            normalize_company_name=lambda value: value,
        ),
    )

    assert status is False
    increment_attempts.assert_awaited_once()


def test_validate_enrichment_schema_v3_recomputes_missing_witness_primitives():
    row, result = _witness_ready_row_and_result()
    derived = b2b_enrichment._compute_derived_fields(result, row)
    for key in (
        "replacement_mode",
        "operating_model_shift",
        "productivity_delta_claim",
        "org_pressure_type",
        "salience_flags",
        "evidence_spans",
        "evidence_map_hash",
    ):
        derived.pop(key, None)

    assert b2b_enrichment._validate_enrichment(derived, row)
    assert derived["replacement_mode"] == "workflow_substitution"
    assert derived["evidence_spans"]
    assert derived["evidence_map_hash"]


def test_validate_enrichment_schema_v3_rejects_missing_witness_primitives_without_source_row():
    row, result = _witness_ready_row_and_result()
    derived = b2b_enrichment._compute_derived_fields(result, row)
    derived.pop("evidence_spans", None)
    derived.pop("evidence_map_hash", None)

    assert not b2b_enrichment._validate_enrichment(derived)
