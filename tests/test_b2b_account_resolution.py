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

    def test_domain_like_bypassed_for_direct_signals(self):
        # kore.ai / Purplle.com are real brand names declared by the reviewer
        exc = _apply_guardrails("kore.ai", "HubSpot", trust_direct=True)
        assert exc is None

    def test_domain_like_bypassed_for_dotcom(self):
        exc = _apply_guardrails("Purplle.com", "HubSpot", trust_direct=True)
        assert exc is None

    def test_vendor_still_blocked_even_with_trust_direct(self):
        # trust_direct only skips domain_like — vendor check still applies
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
