"""Tests for the B2B account resolution layer."""

from __future__ import annotations

import pytest

from atlas_brain.services.b2b.account_resolver import (
    ExcludedCandidate,
    ResolutionResult,
    ResolutionSignal,
    _apply_guardrails,
    _clean_extracted_name,
    _compute_confidence,
    _extract_from_bio_regex,
    _extract_from_enrichment,
    _extract_from_hackernews_metadata,
    _extract_from_producthunt_metadata,
    _extract_from_quora_metadata,
    _extract_from_reddit_metadata,
    _extract_from_review_text,
    _extract_from_reviewer_company,
    _extract_from_title_bio,
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

    def test_company_pipe_title(self):
        sig = _extract_from_bio_regex("Acme Corp | Senior Engineer", "bio")
        assert sig is not None
        assert sig.value == "Acme Corp"
        assert sig.signal_type == "company_pipe_title"

    def test_i_work_at(self):
        sig = _extract_from_bio_regex("I work at Google, on the Cloud team", "bio")
        assert sig is not None
        assert sig.value == "Google"
        assert sig.signal_type == "work_at_company"

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

    def test_valid_passes(self):
        exc = _apply_guardrails("Acme Corp", "HubSpot")
        assert exc is None


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

    def test_no_company_in_text(self):
        review = {
            "summary": "bad product",
            "review_text": "would not recommend to anyone",
        }
        signals = _extract_from_review_text(review)
        assert len(signals) == 0
