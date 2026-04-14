"""Tests for the B2B account resolution layer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks import b2b_account_resolution as task_mod
from atlas_brain.services.b2b import account_resolver as resolver_mod
from atlas_brain.services.b2b.account_resolver import (
    ExcludedCandidate,
    ResolutionResult,
    ResolutionSignal,
    _apply_guardrails,
    _clean_extracted_name,
    _compute_confidence,
    _domain_to_company_candidate,
    _extract_from_bio_regex,
    _extract_from_enrichment,
    _extract_from_hackernews_metadata,
    _extract_from_producthunt_metadata,
    _extract_from_quora_metadata,
    _extract_from_reddit_metadata,
    _extract_from_review_text,
    _extract_from_reviewer_company,
    _extract_from_title_bio,
    extract_from_github_profile,
    extract_from_hn_profile,
    fetch_reddit_profile,
    extract_from_reddit_profile,
    resolve_review,
)


# -- Bio Regex Extraction ---------------------------------------------------


class TestBioRegex:

    def test_cto_at_company(self):
        sig = _extract_from_bio_regex("CTO at Acme Corp", "bio")
        assert sig is not None
        assert sig.value == "Acme Corp"
        assert sig.signal_type == "title_at_company"

    def test_engineer_at_stripe(self):
        sig = _extract_from_bio_regex("Senior Engineer at Stripe", "bio")
        assert sig is not None
        assert sig.value == "Stripe"

    def test_at_symbol(self):
        sig = _extract_from_bio_regex("Engineer @ Datadog", "bio")
        assert sig is not None
        assert sig.value == "Datadog"
        # "Engineer" matches the title_at_company pattern first
        assert sig.signal_type in ("title_at_company", "handle_at_company")

    def test_founder_comma(self):
        sig = _extract_from_bio_regex("Founder, Acme Inc", "bio")
        assert sig is not None
        assert sig.value == "Acme Inc"
        assert sig.signal_type == "founder_of_company"

    def test_founder_sentence_fragment_rejected(self):
        sig = _extract_from_bio_regex("Founder, and like many of you", "bio")
        assert sig is None

    def test_company_pipe_title(self):
        sig = _extract_from_bio_regex("Acme Corp | Senior Engineer", "bio")
        assert sig is not None
        assert sig.value == "Acme Corp"
        assert sig.signal_type == "company_pipe_title"

    def test_company_pipe_title_rejects_role_like_prefix(self):
        sig = _extract_from_bio_regex("Chief Growth Generator | RevOps", "bio")
        assert sig is None

    def test_i_work_at(self):
        sig = _extract_from_bio_regex("I work at Google, on the Cloud team", "bio")
        assert sig is not None
        assert sig.value == "Google"
        assert sig.signal_type == "work_at_company"

    def test_i_work_for(self):
        sig = _extract_from_bio_regex("I work for Stripe", "bio")
        assert sig is not None
        assert sig.value == "Stripe"
        assert sig.signal_type == "work_at_company"

    def test_work_at_sentence_fragment_rejected(self):
        sig = _extract_from_bio_regex("I worked at earlier in my career", "bio")
        assert sig is None

    def test_work_at_company_with_context_trims_before_guardrails(self):
        result = resolve_review(
            {
                "source": "reddit",
                "review_text": "I work at Workday in the partner team and help support integrations.",
                "raw_metadata": {"employment_claim": True},
                "reviewer_company": None,
                "reviewer_title": None,
                "enrichment": None,
            },
            vendor_name="Workday",
        )
        assert result.resolved_company_name is None
        assert any(exc.reason == "incumbent_vendor" for exc in result.excluded_candidates)

    def test_use_at_company(self):
        # "we use X at Company" is valid
        sig = _extract_from_bio_regex("we use this at Datadog", "bio")
        assert sig is not None
        assert sig.value == "Datadog"
        assert sig.signal_type == "work_at_company"

    def test_use_for_not_extracted(self):
        # "we use X for task management" - should NOT extract "task management"
        sig = _extract_from_bio_regex("we use Workday for task management", "bio")
        assert sig is None or sig.value != "task management"

    def test_use_for_project_management_not_extracted(self):
        # false positive guard: "we use X for project management"
        sig = _extract_from_bio_regex("we use Jira for project management", "bio")
        assert sig is None or sig.value != "project management"

    def test_are_for_not_extracted(self):
        # "we are for open source" - should NOT extract "open source"
        sig = _extract_from_bio_regex("we are for open source projects", "bio")
        assert sig is None or sig.value != "open source projects"

    def test_article_a_not_extracted(self):
        # "I work at a large tech company" - should NOT extract "a large tech company"
        sig = _extract_from_bio_regex("I work at a large tech company", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("a "))

    def test_article_an_not_extracted(self):
        # "I work for an IT company" - should NOT extract "an IT company"
        sig = _extract_from_bio_regex("I work for an IT company", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("an "))

    def test_article_the_not_extracted(self):
        # "I work at the business development team" - should NOT extract "the business..."
        sig = _extract_from_bio_regex("I work at the business development team", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("the "))

    def test_possessive_our_not_extracted(self):
        # "we are at our main office" - should NOT extract "our main office"
        sig = _extract_from_bio_regex("we are at our main office", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("our "))

    def test_company_name_with_a_prefix_still_works(self):
        # "Acme" starts with 'A' but is NOT "a " (article) - must still resolve
        sig = _extract_from_bio_regex("I work at Acme Corp", "bio")
        assert sig is not None
        assert sig.value == "Acme Corp"

    def test_email_provider_rejected(self):
        # "handle@gmail" - should NOT extract "gmail" as a company name
        sig = _extract_from_bio_regex("johndoe @ gmail", "bio")
        assert sig is None or (sig.value and sig.value.lower() != "gmail")

    def test_generic_word_rejected(self):
        # "I work at support" - "support" is a generic term, not a company
        sig = _extract_from_bio_regex("I work at support", "bio")
        assert sig is None or (sig.value and sig.value.lower() != "support")

    def test_entrepreneur_rejected_from_founder(self):
        # "Founder, entrepreneur" - "entrepreneur" is not a company name
        sig = _extract_from_bio_regex("Founder, entrepreneur", "bio")
        assert sig is None or (sig.value and sig.value.lower() != "entrepreneur")

    def test_task_management_rejected(self):
        # Hardcoded in reject list even without article
        sig = _extract_from_bio_regex("I work at task management", "bio")
        assert sig is None or (sig.value and "management" not in sig.value.lower())

    def test_former_at(self):
        sig = _extract_from_bio_regex("Former engineer at Meta", "bio")
        assert sig is not None
        assert sig.value == "Meta"
        # "engineer" matches title_at_company; former_at_company also valid
        assert sig.signal_type in ("title_at_company", "former_at_company")

    def test_no_match(self):
        sig = _extract_from_bio_regex("Just a regular person", "bio")
        assert sig is None

    def test_empty_string(self):
        sig = _extract_from_bio_regex("", "bio")
        assert sig is None

    def test_role_only(self):
        sig = _extract_from_bio_regex("Sysadmin", "bio")
        assert sig is None


# -- Clean Extracted Name ---------------------------------------------------


class TestCleanName:

    def test_trailing_role_stripped(self):
        assert _clean_extracted_name("Acme Corp Engineer") == "Acme Corp"

    def test_trailing_punctuation(self):
        assert _clean_extracted_name("Acme Corp,") == "Acme Corp"

    def test_clean_passthrough(self):
        assert _clean_extracted_name("Stripe") == "Stripe"


# -- Direct Field Extraction ------------------------------------------------


class TestDirectFields:

    def test_reviewer_company_present(self):
        review = {"reviewer_company": "Acme Corp"}
        sig = _extract_from_reviewer_company(review)
        assert sig is not None
        assert sig.value == "Acme Corp"
        assert sig.confidence == 0.9

    def test_reviewer_company_empty(self):
        review = {"reviewer_company": ""}
        assert _extract_from_reviewer_company(review) is None

    def test_reviewer_company_none(self):
        review = {"reviewer_company": None}
        assert _extract_from_reviewer_company(review) is None

    def test_enrichment_company(self):
        review = {
            "enrichment": {
                "reviewer_context": {"company_name": "Beta LLC"}
            }
        }
        sig = _extract_from_enrichment(review)
        assert sig is not None
        assert sig.value == "Beta LLC"
        # Low confidence -- LLM-derived, needs corroboration for medium
        assert sig.confidence == 0.35

    def test_enrichment_empty(self):
        review = {"enrichment": {"reviewer_context": {"company_name": ""}}}
        assert _extract_from_enrichment(review) is None

    def test_enrichment_missing(self):
        review = {}
        assert _extract_from_enrichment(review) is None


# -- Source-Specific Extractors ---------------------------------------------


class TestRedditExtractor:

    def test_flair_company_pipe(self):
        review = {
            "source": "reddit",
            "raw_metadata": {"author_flair_text": "Cloudflare | SRE"},
        }
        signals = _extract_from_reddit_metadata(review)
        company_signals = [s for s in signals if "flair" in s.signal_type]
        assert len(company_signals) >= 1
        assert company_signals[0].value == "Cloudflare"

    def test_flair_role_only(self):
        review = {
            "source": "reddit",
            "raw_metadata": {"author_flair_text": "Sysadmin"},
        }
        signals = _extract_from_reddit_metadata(review)
        flair_sigs = [s for s in signals if s.signal_type == "reddit_flair_company"]
        assert len(flair_sigs) == 0

    def test_flair_pipe_role_like_prefix_not_treated_as_company(self):
        review = {
            "source": "reddit",
            "raw_metadata": {"author_flair_text": "Data Analyst | RevOps"},
        }
        signals = _extract_from_reddit_metadata(review)
        flair_sigs = [s for s in signals if s.signal_type == "reddit_flair_company"]
        assert len(flair_sigs) == 0

    def test_flair_pipe_academic_title_not_treated_as_company(self):
        review = {
            "source": "reddit",
            "raw_metadata": {"author_flair_text": "PhD | NLP"},
        }
        signals = _extract_from_reddit_metadata(review)
        flair_sigs = [s for s in signals if s.signal_type == "reddit_flair_company"]
        assert len(flair_sigs) == 0

    def test_flair_pipe_loyalty_status_not_treated_as_company(self):
        review = {
            "source": "reddit",
            "raw_metadata": {
                "author_flair_text": "MileagePlus 1K | Quality Contributor",
            },
        }
        signals = _extract_from_reddit_metadata(review)
        flair_sigs = [s for s in signals if s.signal_type == "reddit_flair_company"]
        assert len(flair_sigs) == 0

    def test_flair_pipe_trade_counts_not_treated_as_company(self):
        review = {
            "source": "reddit",
            "raw_metadata": {"author_flair_text": "S: 378 | B: 8"},
        }
        signals = _extract_from_reddit_metadata(review)
        flair_sigs = [s for s in signals if s.signal_type == "reddit_flair_company"]
        assert len(flair_sigs) == 0

    def test_flair_pipe_sale_buy_counts_not_treated_as_company(self):
        review = {
            "source": "reddit",
            "raw_metadata": {"author_flair_text": "0 Sale | 1 Buy"},
        }
        signals = _extract_from_reddit_metadata(review)
        flair_sigs = [s for s in signals if s.signal_type == "reddit_flair_company"]
        assert len(flair_sigs) == 0

    def test_flair_pipe_role_and_status_not_treated_as_company(self):
        review = {
            "source": "reddit",
            "raw_metadata": {"author_flair_text": "Principal AD | Moderator"},
        }
        signals = _extract_from_reddit_metadata(review)
        flair_sigs = [s for s in signals if s.signal_type == "reddit_flair_company"]
        assert len(flair_sigs) == 0

    def test_employment_claim_with_text(self):
        review = {
            "source": "reddit",
            "review_text": "I work at Datadog and we switched from PagerDuty last year",
            "raw_metadata": {"employment_claim": True},
        }
        signals = _extract_from_reddit_metadata(review)
        emp_sigs = [s for s in signals if "employment" in s.signal_type or "work_at" in s.signal_type]
        assert len(emp_sigs) >= 1
        assert emp_sigs[0].value == "Datadog"

    def test_past_employment_claim_is_not_medium_confidence(self):
        review = {
            "source": "reddit",
            "review_text": "Hey, former employee of Carbon Black here.",
            "raw_metadata": {
                "employment_claim": True,
                "employment_tense": "past",
            },
        }
        signals = _extract_from_reddit_metadata(review)
        emp_sigs = [s for s in signals if s.signal_type == "reddit_employment_claim"]
        assert len(emp_sigs) == 1
        assert emp_sigs[0].value == "Carbon Black"
        assert emp_sigs[0].confidence < 0.5

    def test_no_metadata(self):
        review = {"source": "reddit", "raw_metadata": {}}
        signals = _extract_from_reddit_metadata(review)
        assert len(signals) == 0


class TestQuoraExtractor:

    def test_credentials(self):
        review = {
            "source": "quora",
            "reviewer_title": "Software Engineer at Google (2015-present)",
        }
        signals = _extract_from_quora_metadata(review)
        assert len(signals) == 1
        assert signals[0].value == "Google"
        assert signals[0].signal_type == "quora_credentials"

    def test_wrong_source(self):
        review = {
            "source": "reddit",
            "reviewer_title": "Engineer at Google",
        }
        signals = _extract_from_quora_metadata(review)
        assert len(signals) == 0


class TestProductHuntExtractor:

    def test_headline(self):
        review = {
            "source": "producthunt",
            "reviewer_title": "CEO @ Startup Inc",
        }
        signals = _extract_from_producthunt_metadata(review)
        assert len(signals) == 1
        assert signals[0].value == "Startup Inc"


class TestTitleBio:

    def test_any_source_title(self):
        review = {"reviewer_title": "VP at Oracle", "source": "g2"}
        signals = _extract_from_title_bio(review)
        assert len(signals) == 1
        assert signals[0].value == "Oracle"


# -- Guardrails -------------------------------------------------------------


class TestGuardrails:

    def test_vendor_excluded(self):
        exc = _apply_guardrails("HubSpot", "HubSpot")
        assert exc is not None
        assert exc.reason == "incumbent_vendor"

    def test_vendor_excluded_case_insensitive(self):
        exc = _apply_guardrails("hubspot", "HubSpot")
        assert exc is not None
        assert exc.reason == "incumbent_vendor"

    def test_blocked_name(self):
        exc = _apply_guardrails("Competitor Inc", "HubSpot", blocked_names={"competitor"})
        assert exc is not None
        assert exc.reason == "blocked_name"

    def test_generic_descriptor(self):
        exc = _apply_guardrails("startup", "HubSpot")
        assert exc is not None
        assert exc.reason == "generic_descriptor"

    def test_empty_name(self):
        exc = _apply_guardrails("", "HubSpot")
        assert exc is not None
        assert exc.reason == "empty"

    def test_domain_like(self):
        exc = _apply_guardrails("acme.com", "HubSpot")
        assert exc is not None
        assert exc.reason == "domain_like"

    def test_domain_like_bypassed_for_direct_signals(self):
        # kore.ai / Purplle.com are real brand names declared by the reviewer
        exc = _apply_guardrails("kore.ai", "HubSpot", trust_direct=True)
        assert exc is None

    def test_domain_like_bypassed_for_dotcom(self):
        exc = _apply_guardrails("Purplle.com", "HubSpot", trust_direct=True)
        assert exc is None

    def test_vendor_still_blocked_even_with_trust_direct(self):
        # trust_direct only skips domain_like - vendor check still applies
        exc = _apply_guardrails("kore.ai", "kore.ai", trust_direct=True)
        assert exc is not None
        assert exc.reason == "incumbent_vendor"

    def test_valid_passes(self):
        exc = _apply_guardrails("Acme Corp", "HubSpot")
        assert exc is None

    def test_domain_company_from_reviewer_field_resolves(self):
        # reviewer_company_field signals get trust_direct=True in resolve_review
        result = resolve_review(
            {"reviewer_company": "kore.ai", "source": "trustradius",
             "review_text": "great product", "reviewer_title": None,
             "raw_metadata": {}, "enrichment": None},
            vendor_name="Zendesk",
        )
        assert result.resolved_company_name == "kore.ai"
        assert result.confidence_label == "high"

    def test_domain_company_from_github_profile_resolves(self):
        # github_profile_company signals also get trust_direct=True
        result = resolve_review(
            {"reviewer_company": None, "source": "github",
             "review_text": "switched away", "reviewer_title": None,
             "raw_metadata": {}, "enrichment": None,
             "_gh_profile": {"company": "sandeza.ai"}},
            vendor_name="Jira",
        )
        assert result.resolved_company_name == "sandeza.ai"
        assert result.confidence_label in ("medium", "high")

    def test_domain_like_default_still_blocked(self):
        # Without trust_direct, domain_like check applies (default behaviour unchanged)
        exc = _apply_guardrails("kore.ai", "Zendesk", trust_direct=False)
        assert exc is not None
        assert exc.reason == "domain_like"


# -- Confidence Model -------------------------------------------------------


class TestConfidence:

    def test_high_confidence_single_strong(self):
        signals = [ResolutionSignal("reviewer_company_field", "Acme", 0.9, "reviewer_company")]
        score, label = _compute_confidence(signals)
        assert label == "high"
        assert score >= 0.8

    def test_medium_confidence(self):
        signals = [ResolutionSignal("title_at_company", "Acme", 0.6, "reviewer_title")]
        score, label = _compute_confidence(signals)
        assert label == "medium"

    def test_boost_from_agreement(self):
        signals = [
            ResolutionSignal("title_at_company", "Acme Corp", 0.6, "reviewer_title"),
            ResolutionSignal("reddit_flair_company", "Acme Corp", 0.65, "flair"),
        ]
        score, label = _compute_confidence(signals)
        assert score > 0.65

    def test_empty_signals(self):
        score, label = _compute_confidence([])
        assert label == "unresolved"
        assert score == 0.0


# -- Full Pipeline ----------------------------------------------------------


class TestResolveReview:

    def test_reviewer_company_high_confidence(self):
        review = {
            "reviewer_company": "Acme Corp",
            "vendor_name": "HubSpot",
            "source": "g2",
        }
        result = resolve_review(review, vendor_name="HubSpot")
        assert result.resolved_company_name == "Acme Corp"
        assert result.confidence_label == "high"
        assert result.resolution_method == "reviewer_company_field"

    def test_reddit_flair_medium_confidence(self):
        review = {
            "reviewer_company": None,
            "reviewer_title": None,
            "source": "reddit",
            "review_text": "This tool is terrible",
            "vendor_name": "Jira",
            "raw_metadata": {"author_flair_text": "Stripe | DevOps"},
            "enrichment": None,
        }
        result = resolve_review(review, vendor_name="Jira")
        assert result.resolved_company_name == "Stripe"
        assert result.confidence_label in ("medium", "high")

    def test_vendor_name_excluded(self):
        review = {
            "reviewer_company": "HubSpot",
            "source": "g2",
            "vendor_name": "HubSpot",
        }
        result = resolve_review(review, vendor_name="HubSpot")
        assert result.resolved_company_name is None
        assert len(result.excluded_candidates) >= 1
        assert result.excluded_candidates[0].reason == "incumbent_vendor"

    def test_no_signals_unresolved(self):
        review = {
            "reviewer_company": None,
            "reviewer_title": None,
            "source": "reddit",
            "review_text": "not great software",
            "raw_metadata": {},
            "enrichment": None,
        }
        result = resolve_review(review, vendor_name="Jira")
        assert result.resolved_company_name is None
        assert result.confidence_label == "unresolved"

    def test_enrichment_fallback_low_confidence(self):
        review = {
            "reviewer_company": None,
            "reviewer_title": None,
            "source": "hackernews",
            "review_text": "switched away",
            "raw_metadata": {},
            "enrichment": {
                "reviewer_context": {"company_name": "Widgets Inc"}
            },
        }
        result = resolve_review(review, vendor_name="Jira")
        assert result.resolved_company_name == "Widgets Inc"
        # Enrichment-only = low confidence (0.35), needs corroboration for medium
        assert result.confidence_label == "low"

    def test_former_employer_only_stays_low_confidence(self):
        review = {
            "reviewer_company": None,
            "reviewer_title": None,
            "source": "reddit",
            "review_text": "Hey, former employee of Carbon Black here.",
            "raw_metadata": {
                "employment_claim": True,
                "employment_tense": "past",
            },
            "enrichment": None,
        }
        result = resolve_review(review, vendor_name="SentinelOne")
        assert result.resolved_company_name == "Carbon Black"
        assert result.confidence_label == "low"


# -- Evidence Serialization -------------------------------------------------


class TestEvidence:

    def test_to_evidence_json(self):
        result = ResolutionResult(
            resolved_company_name="Acme",
            signals=[ResolutionSignal("reviewer_company_field", "Acme", 0.9, "reviewer_company")],
            excluded_candidates=[ExcludedCandidate("HubSpot", "incumbent_vendor")],
        )
        evidence = result.to_evidence_json()
        assert len(evidence["signals"]) == 1
        assert evidence["signals"][0]["type"] == "reviewer_company_field"
        assert len(evidence["excluded_candidates"]) == 1
        assert evidence["excluded_candidates"][0]["reason"] == "incumbent_vendor"


# -- HackerNews Extractor --------------------------------------------------


class TestHackerNewsExtractor:

    def test_hn_text_with_company(self):
        review = {
            "source": "hackernews",
            "review_text": "I work at Cloudflare and we've been evaluating alternatives to Datadog",
        }
        signals = _extract_from_hackernews_metadata(review)
        assert len(signals) >= 1
        assert signals[0].value == "Cloudflare"
        assert signals[0].signal_type == "hackernews_text_company"

    def test_hn_no_company(self):
        review = {
            "source": "hackernews",
            "review_text": "this product is overpriced garbage",
        }
        signals = _extract_from_hackernews_metadata(review)
        assert len(signals) == 0

    def test_hn_wrong_source(self):
        review = {
            "source": "reddit",
            "review_text": "I work at Acme",
        }
        signals = _extract_from_hackernews_metadata(review)
        assert len(signals) == 0


def test_unique_usernames_dedupes_before_limit():
    rows = [
        {"reviewer_name": "repeat"},
        {"reviewer_name": "repeat"},
        {"reviewer_name": "unique-one"},
        {"reviewer_name": "unique-two"},
    ]

    assert task_mod._unique_usernames(rows, 2) == ["repeat", "unique-one"]


@pytest.mark.asyncio
async def test_account_resolution_task_claims_safe_enrichment_statuses(monkeypatch):
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[]),
        execute=AsyncMock(return_value="INSERT 0 0"),
    )
    cfg = SimpleNamespace(
        account_resolution_batch_size=250,
        account_resolution_backfill_min_confidence="medium",
        account_resolution_eligible_statuses=["enriched", "no_signal", "quarantined"],
        account_resolution_source_priority=["g2", "gartner", "capterra"],
        account_resolution_excluded_sources=["software_advice", "trustpilot"],
        account_resolution_retry_unresolved_sources=["reddit", "hackernews"],
        account_resolution_unresolved_retry_interval_hours=48,
    )
    monkeypatch.setattr(task_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(task_mod, "settings", SimpleNamespace(b2b_churn=cfg))

    result = await task_mod.run(SimpleNamespace())

    assert result == {
        "_skip_synthesis": "No reviews pending resolution",
        "excluded_sources": 0,
    }
    (
        query,
        statuses,
        excluded_sources,
        text_pattern,
        source_priority,
        retry_unresolved_sources,
        retry_interval_hours,
        batch_size,
    ) = pool.fetch.await_args.args
    assert "LEFT JOIN b2b_review_vendor_mentions primary_vm" in query
    assert "COALESCE(primary_vm.vendor_name, r.vendor_name) AS vendor_name" in query
    assert "r.enrichment_status = ANY($1::text[])" in query
    assert "r.enrichment IS NOT NULL" in query
    assert "NOT (r.source = ANY($2::text[]))" in query
    assert "ar.resolution_status = 'excluded'" in query
    assert "ar.resolution_method = 'unsupported_source'" in query
    assert "ar.resolution_status = 'unresolved'" in query
    assert "r.source = ANY($5::text[])" in query
    assert "make_interval(hours => $6::int)" in query
    assert "array_position($4::text[], r.source)" in query
    assert statuses == ["enriched", "no_signal", "quarantined"]
    assert excluded_sources == ["trustpilot"]
    assert "i work at" in text_pattern
    assert source_priority == ["g2", "gartner", "capterra"]
    assert retry_unresolved_sources == ["reddit", "hackernews"]
    assert retry_interval_hours == 48
    assert batch_size == 250


@pytest.mark.asyncio
async def test_account_resolution_task_defaults_unresolved_retry_sources(monkeypatch):
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=[]),
        execute=AsyncMock(return_value="INSERT 0 0"),
    )
    cfg = SimpleNamespace(
        account_resolution_batch_size=10,
        account_resolution_backfill_min_confidence="medium",
        account_resolution_eligible_statuses=["enriched"],
        account_resolution_source_priority=["reddit"],
        account_resolution_excluded_sources=["trustpilot"],
    )
    monkeypatch.setattr(task_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(task_mod, "settings", SimpleNamespace(b2b_churn=cfg))

    await task_mod.run(SimpleNamespace())

    _, _, _, _, _, retry_unresolved_sources, retry_interval_hours, _ = pool.fetch.await_args.args
    assert retry_unresolved_sources == ["reddit", "hackernews", "github"]
    assert retry_interval_hours == 24


@pytest.mark.asyncio
async def test_account_resolution_task_prefetches_unique_reddit_profiles_before_cap(monkeypatch):
    rows = [
        {
            "id": "r1",
            "source": "reddit",
            "source_url": "https://example.com/1",
            "reviewer_name": "repeat",
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "review_text": "text",
            "enrichment": {},
            "raw_metadata": {},
            "vendor_name": "Zendesk",
            "product_category": "Support",
            "rating": None,
            "rating_max": None,
        },
        {
            "id": "r2",
            "source": "reddit",
            "source_url": "https://example.com/2",
            "reviewer_name": "repeat",
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "review_text": "text",
            "enrichment": {},
            "raw_metadata": {},
            "vendor_name": "Zendesk",
            "product_category": "Support",
            "rating": None,
            "rating_max": None,
        },
        {
            "id": "r3",
            "source": "reddit",
            "source_url": "https://example.com/3",
            "reviewer_name": "unique-one",
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "review_text": "text",
            "enrichment": {},
            "raw_metadata": {},
            "vendor_name": "Zendesk",
            "product_category": "Support",
            "rating": None,
            "rating_max": None,
        },
        {
            "id": "r4",
            "source": "reddit",
            "source_url": "https://example.com/4",
            "reviewer_name": "unique-two",
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "review_text": "text",
            "enrichment": {},
            "raw_metadata": {},
            "vendor_name": "Zendesk",
            "product_category": "Support",
            "rating": None,
            "rating_max": None,
        },
    ]
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(return_value=rows),
        execute=AsyncMock(return_value="INSERT 0 1"),
    )
    cfg = SimpleNamespace(
        account_resolution_batch_size=100,
        account_resolution_backfill_min_confidence="medium",
        account_resolution_eligible_statuses=["enriched"],
        account_resolution_source_priority=["reddit"],
        account_resolution_excluded_sources=["trustpilot"],
        account_resolution_max_profile_fetches=1,
        account_resolution_reddit_max_profile_fetches=2,
        account_resolution_profile_fetch_concurrency=5,
        account_resolution_profile_fetch_timeout=1.0,
        account_resolution_reddit_profile_fetch_concurrency=2,
        account_resolution_reddit_profile_fetch_delay_seconds=0.5,
        account_resolution_reddit_profile_fetch_max_retries=2,
        account_resolution_reddit_profile_fetch_retry_after_cap_seconds=3.0,
    )
    fetch_reddit = AsyncMock(side_effect=[{}, {}])
    monkeypatch.setattr(task_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(task_mod, "settings", SimpleNamespace(b2b_churn=cfg))
    monkeypatch.setattr(task_mod, "_exclude_unsupported_sources", AsyncMock(return_value=0))
    monkeypatch.setattr(task_mod, "_propagate_user_resolutions", AsyncMock(return_value=0))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.account_resolver.fetch_reddit_profile",
        fetch_reddit,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.account_resolver.resolve_review",
        lambda *args, **kwargs: ResolutionResult(),
    )

    result = await task_mod.run(SimpleNamespace())

    assert result["reviews_processed"] == 4
    assert fetch_reddit.await_count == 2
    assert [call.args[0] for call in fetch_reddit.await_args_list] == ["repeat", "unique-one"]
    for call in fetch_reddit.await_args_list:
        assert call.kwargs["request_delay_seconds"] == 0.5
        assert call.kwargs["max_retries"] == 2
        assert call.kwargs["retry_after_cap_seconds"] == 3.0


@pytest.mark.asyncio
async def test_fetch_reddit_profile_retries_rate_limited_response(monkeypatch):
    class _Response:
        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, responses):
            self._responses = list(responses)

        async def get(self, url, headers=None):
            assert "/user/example/about.json" in url
            assert headers and "User-Agent" in headers
            return self._responses.pop(0)

    sleep_calls = []

    async def _fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        "atlas_brain.services.b2b.account_resolver.asyncio.sleep",
        _fake_sleep,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.account_resolver._get_reddit_credentials",
        lambda: ("", ""),
    )
    profile = await fetch_reddit_profile(
        "example",
        _Client([
            _Response(429, headers={"Retry-After": "7"}),
            _Response(
                200,
                payload={"data": {"subreddit": {"public_description": "Engineer at Stripe https://stripe.com"}}},
            ),
        ]),
        request_delay_seconds=0.25,
        max_retries=1,
        retry_after_cap_seconds=2.0,
    )

    assert profile["bio"] == "Engineer at Stripe https://stripe.com"
    assert profile["profile_urls"] == ["https://stripe.com"]
    assert sleep_calls == [0.25, 2.0, 0.25]


@pytest.mark.asyncio
async def test_fetch_reddit_profile_uses_oauth_when_credentials_available(monkeypatch):
    class _Response:
        def __init__(self, status_code, payload=None, headers=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = headers or {}

        def json(self):
            return self._payload

    class _Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, headers=None):
            self.calls.append((url, headers))
            return _Response(
                200,
                payload={"data": {"subreddit": {"public_description": "Works at Cloudflare"}}},
            )

    monkeypatch.setattr(
        "atlas_brain.services.b2b.account_resolver._get_reddit_credentials",
        lambda: ("client-id", "client-secret"),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.account_resolver._get_reddit_oauth_token",
        AsyncMock(return_value="oauth-token"),
    )
    client = _Client()

    profile = await fetch_reddit_profile("example", client, request_delay_seconds=0.0)

    assert profile["bio"] == "Works at Cloudflare"
    assert client.calls
    url, headers = client.calls[0]
    assert url == "https://oauth.reddit.com/user/example/about"
    assert headers["Authorization"] == "Bearer oauth-token"
    assert headers["User-Agent"]


@pytest.mark.asyncio
async def test_get_reddit_oauth_token_uses_single_request_under_concurrency(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"access_token": "shared-token", "expires_in": 3600}

    class _Client:
        def __init__(self):
            self.post_calls = 0

        async def post(self, url, auth=None, data=None, headers=None):
            self.post_calls += 1
            assert url == "https://www.reddit.com/api/v1/access_token"
            return _Response()

    monkeypatch.setattr(
        "atlas_brain.services.b2b.account_resolver._get_reddit_credentials",
        lambda: ("client-id", "client-secret"),
    )
    resolver_mod._REDDIT_OAUTH_TOKEN = None
    resolver_mod._REDDIT_OAUTH_TOKEN_EXPIRES_AT = 0.0
    client = _Client()

    first, second = await asyncio.gather(
        resolver_mod._get_reddit_oauth_token(client),
        resolver_mod._get_reddit_oauth_token(client),
    )

    assert first == "shared-token"
    assert second == "shared-token"
    assert client.post_calls == 1


@pytest.mark.asyncio
async def test_backfill_review_company_context_updates_enrichment_company_name():
    class _Pool:
        def __init__(self):
            self.calls = []

        async def execute(self, query, *args):
            self.calls.append((query, args))
            return "UPDATE 1"

    pool = _Pool()

    changed = await task_mod._backfill_review_company_context(
        pool,
        review_id="review-1",
        company_name="Acme Corp",
        company_norm="acme corp",
    )

    assert changed is True
    query, args = pool.calls[0]
    assert "jsonb_set" in query
    assert "{reviewer_context,company_name}" in query
    assert args == ("review-1", "Acme Corp", "acme corp")


# -- Review Text Extraction -------------------------------------------------


class TestReviewTextExtraction:

    def test_summary_company(self):
        review = {
            "summary": "As CTO at Acme, I found this tool lacking",
            "review_text": "",
        }
        signals = _extract_from_review_text(review)
        assert len(signals) >= 1
        company_sigs = [s for s in signals if s.value == "Acme"]
        assert len(company_sigs) >= 1

    def test_review_text_company(self):
        review = {
            "summary": "",
            "review_text": "We use this at Stripe and it's been terrible for our team",
        }
        signals = _extract_from_review_text(review)
        assert len(signals) >= 1
        assert any(s.value == "Stripe" for s in signals)

    def test_review_text_does_not_extract_founder_sentence_fragment(self):
        review = {
            "summary": "",
            "review_text": (
                "I'm a first time founder, and like many of you, I live in project management tools "
                "all day."
            ),
        }
        signals = _extract_from_review_text(review)
        assert len(signals) == 0

    def test_review_text_does_not_extract_handle_style_fragment(self):
        review = {
            "summary": "",
            "review_text": "Every startup stack post I found was clearly sponsored and dev @ all the things.",
        }
        signals = _extract_from_review_text(review)
        assert len(signals) == 0

    def test_review_text_does_not_extract_usage_clause_as_company(self):
        review = {
            "summary": "",
            "review_text": (
                "The company I work for is using Slack with Trello integration for daily coordination."
            ),
        }
        signals = _extract_from_review_text(review)
        assert len(signals) == 0

    def test_review_text_does_not_extract_specializes_clause_as_company(self):
        review = {
            "summary": "",
            "review_text": (
                "The company I work at specializes in Magento but I avoided it for months."
            ),
        }
        signals = _extract_from_review_text(review)
        assert len(signals) == 0

    def test_title_bio_does_not_extract_generic_commerce(self):
        review = {
            "reviewer_title": "Sr. Community Manager @ Commerce",
            "source": "reddit",
        }
        signals = _extract_from_title_bio(review)
        assert len(signals) == 0

    def test_review_text_use_for_not_a_company(self):
        # "we use X for task management" - should not extract "task management" as a company
        review = {
            "summary": "",
            "review_text": "We use Workday for task management and it has been slow",
        }
        signals = _extract_from_review_text(review)
        assert not any(s.value == "task management" for s in signals)

    def test_review_text_use_for_project_management_not_extracted(self):
        review = {
            "summary": "Switched from Jira",
            "review_text": "We use Linear for project management but the API is lacking",
        }
        signals = _extract_from_review_text(review)
        assert not any(s.value == "project management" for s in signals)

    def test_no_company_in_text(self):
        review = {
            "summary": "bad product",
            "review_text": "would not recommend to anyone",
        }
        signals = _extract_from_review_text(review)
        assert len(signals) == 0


# -- GitHub Profile Extraction ----------------------------------------------


class TestGitHubProfile:

    def test_company_field(self):
        profile = {"company": "Stripe"}
        signals = extract_from_github_profile(profile)
        assert len(signals) >= 1
        company_sigs = [s for s in signals if s.signal_type == "github_profile_company"]
        assert len(company_sigs) == 1
        assert company_sigs[0].value == "Stripe"
        assert company_sigs[0].confidence == 0.8

    def test_company_with_at_prefix(self):
        profile = {"company": "@google"}
        signals = extract_from_github_profile(profile)
        company_sigs = [s for s in signals if s.signal_type == "github_profile_company"]
        assert len(company_sigs) == 1
        # @ prefix stripped by the fetcher, not the extractor
        assert "google" in company_sigs[0].value.lower()

    def test_bio_only(self):
        profile = {"bio": "Senior Engineer at Datadog"}
        signals = extract_from_github_profile(profile)
        assert len(signals) >= 1
        assert signals[0].signal_type == "github_profile_bio"

    def test_empty_profile(self):
        profile = {}
        signals = extract_from_github_profile(profile)
        assert len(signals) == 0

    def test_resolve_with_github_profile(self):
        review = {
            "reviewer_company": None,
            "reviewer_title": None,
            "source": "github",
            "review_text": "this library has issues",
            "raw_metadata": {},
            "enrichment": None,
            "_gh_profile": {"company": "Netflix"},
        }
        result = resolve_review(review, vendor_name="Jira")
        assert result.resolved_company_name == "Netflix"
        assert result.confidence_label == "high"


# -- HN Profile Extraction -------------------------------------------------


class TestHNProfile:

    def test_about_with_company(self):
        profile = {"about": "CTO at Cloudflare. I build things.", "company_from_about": "Cloudflare"}
        sig = extract_from_hn_profile(profile)
        assert sig is not None
        assert sig.value == "Cloudflare"
        assert sig.signal_type == "hn_profile_about"

    def test_about_regex_fallback(self):
        profile = {"about": "I work at Google, building infrastructure"}
        sig = extract_from_hn_profile(profile)
        assert sig is not None
        assert sig.value == "Google"

    def test_empty_profile(self):
        profile = {}
        sig = extract_from_hn_profile(profile)
        assert sig is None

    def test_resolve_with_hn_profile(self):
        review = {
            "reviewer_company": None,
            "reviewer_title": None,
            "source": "hackernews",
            "review_text": "overpriced tool",
            "raw_metadata": {},
            "enrichment": None,
            "_hn_profile": {"about": "Staff Engineer at Shopify", "company_from_about": "Shopify"},
        }
        result = resolve_review(review, vendor_name="Jira")
        assert result.resolved_company_name == "Shopify"
        assert result.confidence_label in ("medium", "high")

    def test_profile_url_domain_fallback(self):
        # No bio text, but profile has a company URL
        profile = {"about": "https://stripe.com", "profile_urls": ["https://stripe.com"]}
        sig = extract_from_hn_profile(profile)
        assert sig is not None
        assert sig.value == "Stripe"
        assert sig.signal_type == "hn_profile_url_domain"
        assert sig.confidence == 0.45

    def test_profile_url_skips_platforms(self):
        # Twitter/GitHub URLs should not produce company signals
        profile = {
            "about": "https://twitter.com/foo https://github.com/foo",
            "profile_urls": ["https://twitter.com/foo", "https://github.com/foo"],
        }
        sig = extract_from_hn_profile(profile)
        assert sig is None

    def test_profile_url_prefers_bio_over_url(self):
        # bio regex match should win over URL domain fallback
        profile = {
            "about": "Engineer at Stripe",
            "company_from_about": "Stripe",
            "profile_urls": ["https://someotherdomain.com"],
        }
        sig = extract_from_hn_profile(profile)
        assert sig is not None
        assert sig.value == "Stripe"
        assert sig.signal_type == "hn_profile_about"


# -- Domain to Company Candidate -------------------------------------------


class TestDomainToCompanyCandidate:

    def test_simple_domain(self):
        assert _domain_to_company_candidate("https://stripe.com") == "Stripe"

    def test_strips_www(self):
        assert _domain_to_company_candidate("https://www.webiphany.com") == "Webiphany"

    def test_platform_returns_none(self):
        assert _domain_to_company_candidate("https://github.com/user") is None
        assert _domain_to_company_candidate("https://twitter.com/user") is None
        assert _domain_to_company_candidate("https://linkedin.com/in/user") is None

    def test_too_short_returns_none(self):
        assert _domain_to_company_candidate("https://ai.com") is None  # "ai" < 3 chars

    def test_digits_only_returns_none(self):
        assert _domain_to_company_candidate("https://123.com") is None

    def test_capitalises(self):
        result = _domain_to_company_candidate("https://acmecorp.io")
        assert result == "Acmecorp"


# -- Reddit Profile Extraction -----------------------------------------------


class TestRedditProfile:

    def test_bio_with_company(self):
        profile = {"bio": "Senior Engineer at Cloudflare. I like distributed systems."}
        sig = extract_from_reddit_profile(profile)
        assert sig is not None
        assert sig.value == "Cloudflare"
        assert sig.signal_type == "reddit_profile_bio"
        assert sig.confidence == 0.55

    def test_bio_work_for(self):
        profile = {"bio": "I work for Stripe, building payment infrastructure"}
        sig = extract_from_reddit_profile(profile)
        assert sig is not None
        assert sig.value == "Stripe"

    def test_bio_url_domain_fallback(self):
        # No bio text pattern match, but has company URL
        profile = {"bio": "Check out my work", "profile_urls": ["https://datadog.com"]}
        sig = extract_from_reddit_profile(profile)
        # bio "Check out my work" won't match -> falls back to URL domain
        assert sig is not None
        assert sig.value == "Datadog"
        assert sig.signal_type == "reddit_profile_url_domain"
        assert sig.confidence == 0.4

    def test_empty_profile(self):
        sig = extract_from_reddit_profile({})
        assert sig is None

    def test_platform_url_skipped(self):
        profile = {"bio": "just a regular dev", "profile_urls": ["https://github.com/user"]}
        sig = extract_from_reddit_profile(profile)
        assert sig is None

    def test_resolve_with_reddit_profile(self):
        review = {
            "reviewer_company": None,
            "reviewer_title": None,
            "source": "reddit",
            "review_text": "This tool is too expensive",
            "raw_metadata": {},
            "enrichment": None,
            "_reddit_profile": {"bio": "DevOps Engineer at HashiCorp"},
        }
        result = resolve_review(review, vendor_name="Terraform Cloud")
        assert result.resolved_company_name == "HashiCorp"
        assert result.confidence_label in ("medium", "high")


# -- Founder Pattern False Positive Guards ------------------------------------


class TestFounderPatternGuards:

    def test_founder_real_company(self):
        # "Founder, Acme Inc" - valid company name
        sig = _extract_from_bio_regex("Founder, Acme Inc", "bio")
        assert sig is not None
        assert sig.value == "Acme Inc"
        assert sig.signal_type == "founder_of_company"

    def test_founder_and_conjunction_rejected(self):
        # "Founder, and like many of you" - sentence continuation, not a company
        sig = _extract_from_bio_regex("Founder, and like many of you I use this", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("and "))

    def test_founder_i_am_rejected(self):
        # "Founder, I've been using this product" - pronoun start
        sig = _extract_from_bio_regex("Founder, I've been using this product for years", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("i'"))

    def test_founder_trying_to_improve_rejected(self):
        # "Founder, and i'm trying to improve that" - false positive from previous run
        sig = _extract_from_bio_regex(
            "Co-founder, and I'm trying to improve that metric", "bio"
        )
        assert sig is None or (sig.value and "trying" not in sig.value.lower())

    def test_owner_we_rejected(self):
        # "Owner, we switched from Slack" - "we" is a pronoun, not a company
        sig = _extract_from_bio_regex("Owner, we switched from Slack last year", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("we "))

    def test_founder_the_rejected(self):
        # "Founder, the company was acquired" - article start
        sig = _extract_from_bio_regex("Founder, the company was acquired", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("the "))

    def test_founder_many_rejected(self):
        # "Founder, many of you know me" - "many" is in reject lookahead
        sig = _extract_from_bio_regex("Founder, many of you know me", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("many"))


# -- Word Count Cap -----------------------------------------------------------


class TestWordCountCap:

    def test_long_phrase_rejected(self):
        # 7-word phrase should be empty after clean
        assert _clean_extracted_name("and like many of you out there") == ""

    def test_six_word_limit_exact(self):
        # Exactly 6 words - allowed
        result = _clean_extracted_name("Very Long Company Name Inc Ltd")
        # Should NOT be empty (6 words)
        assert result != ""

    def test_seven_words_rejected(self):
        # 7 words - rejected
        result = _clean_extracted_name("Very Long Company Name Inc Ltd Corp")
        assert result == ""

    def test_short_name_passes(self):
        assert _clean_extracted_name("Stripe") == "Stripe"

    def test_three_word_company_passes(self):
        assert _clean_extracted_name("International Business Machines") == "International Business Machines"


# -- Handle-at-Company Sentence Fragment Guards --------------------------------


class TestHandleAtCompanyGuards:

    def test_handle_at_real_company(self):
        # "dev @ Stripe" - valid
        sig = _extract_from_bio_regex("dev @ Stripe", "bio")
        assert sig is not None
        assert "Stripe" in sig.value

    def test_handle_at_conjunction_rejected(self):
        # "working @ all the things" - "all" is in reject lookahead
        sig = _extract_from_bio_regex("working @ all the things I do", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("all "))

    def test_handle_at_mentioned_rejected(self):
        # "freelancer @ mentioned on a card" - past-tense verb start, not a company
        sig = _extract_from_bio_regex("freelancer @ mentioned on a card project", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("mentioned"))

    def test_handle_at_many_rejected(self):
        # "freelancer @ many companies" - "many" in reject lookahead; "freelancer" is
        # not a title_at_company keyword so handle_at_company fires.
        sig = _extract_from_bio_regex("freelancer @ many companies and startups", "bio")
        assert sig is None or (sig.value and not sig.value.lower().startswith("many"))
