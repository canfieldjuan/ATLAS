"""Tests for Reddit parser retrieval, ranking, and relevance scoring.

Covers:
  - Alias matching (_build_vendor_aliases, _build_alias_pattern)
  - Candidate scoring (_score_candidate)
  - Insider evidence extraction (_extract_insider_evidence)
  - Vendor-list and job-noise rejection
  - Public/authenticated query profile preservation
  - Subreddit exact vendor query behavior
  - Relevance scoring with Reddit metadata
"""

from __future__ import annotations

import re
import pytest
from unittest.mock import MagicMock

from atlas_brain.services.scraping.parsers.reddit import (
    _apply_author_batch_signals,
    _build_vendor_aliases,
    _build_alias_pattern,
    _extract_insider_evidence,
    _score_candidate,
    _CANDIDATE_SCORE_MIN,
    _SUBREDDIT_WEIGHT,
    RedditParser,
)
from atlas_brain.services.scraping.parsers import ScrapeTarget
from atlas_brain.services.scraping.relevance import score_relevance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_target(vendor_name: str = "HubSpot", **kwargs) -> ScrapeTarget:
    defaults = dict(
        id="test-001",
        source="reddit",
        vendor_name=vendor_name,
        product_name=f"{vendor_name} CRM",
        product_slug=vendor_name.lower().replace(" ", "-"),
        product_category="CRM",
        max_pages=5,
        metadata={"search_profile": "churn"},
    )
    defaults.update(kwargs)
    return ScrapeTarget(**defaults)


def _make_post(
    post_id: str = "abc123",
    title: str = "Test post",
    selftext: str = "A" * 200,
    subreddit: str = "CRM",
    score: int = 50,
    num_comments: int = 10,
    author: str = "test_user",
    **extra,
) -> dict:
    """Build a Reddit API-style post wrapper."""
    data = {
        "id": post_id,
        "name": f"t3_{post_id}",
        "title": title,
        "selftext": selftext,
        "subreddit": subreddit,
        "score": score,
        "num_comments": num_comments,
        "upvote_ratio": 0.85,
        "author": author,
        "created_utc": 1710000000,
        "permalink": f"/r/{subreddit}/comments/{post_id}/test/",
        "link_flair_text": None,
        "all_awardings": [],
        "crosspost_parent_list": None,
        "edited": False,
        "author_flair_text": None,
    }
    data.update(extra)
    return {"kind": "t3", "data": data}


# ---------------------------------------------------------------------------
# Alias matching
# ---------------------------------------------------------------------------

class TestAliasMatching:
    def test_basic_alias(self):
        aliases = _build_vendor_aliases("HubSpot")
        assert "hubspot" in aliases

    def test_dotcom_kept_but_common_word_not_stripped(self):
        aliases = _build_vendor_aliases("Monday.com")
        assert "monday.com" in aliases
        # "monday" is a common English word -- must NOT be a standalone alias
        assert "monday" not in aliases

    def test_dotcom_stripped_when_not_common(self):
        aliases = _build_vendor_aliases("HubSpot.com")
        assert "hubspot.com" in aliases
        assert "hubspot" in aliases  # not a common word, safe to strip

    def test_multi_word_splits_first(self):
        aliases = _build_vendor_aliases("Zoho CRM")
        assert "zoho crm" in aliases
        assert "zoho" in aliases

    def test_multi_word_does_not_split_common_first(self):
        aliases = _build_vendor_aliases("Close CRM")
        assert "close crm" in aliases
        # "close" is a common word -- must NOT be a standalone alias
        assert "close" not in aliases

    def test_extra_aliases(self):
        aliases = _build_vendor_aliases("HubSpot", extra_aliases=["HS", "hubspot.com"])
        assert "hs" in aliases
        assert "hubspot.com" in aliases

    def test_product_name_added_as_alias(self):
        aliases = _build_vendor_aliases("Close", product_name="Close CRM")
        assert "close" in aliases
        assert "close crm" in aliases

    def test_pattern_matches_possessive(self):
        aliases = _build_vendor_aliases("HubSpot")
        pattern = _build_alias_pattern(aliases)
        assert pattern.search("HubSpot's pricing is insane")
        assert pattern.search("I hate hubspot so much")

    def test_pattern_rejects_url(self):
        aliases = _build_vendor_aliases("HubSpot")
        pattern = _build_alias_pattern(aliases)
        assert not pattern.search("cdn2.hubspot.net/images/logo.png")
        assert not pattern.search("tracking.hubspot.com/pixel")

    def test_pattern_rejects_compound(self):
        aliases = _build_vendor_aliases("HubSpot")
        pattern = _build_alias_pattern(aliases)
        assert not pattern.search("foohubspot integration")

    def test_pattern_matches_standalone(self):
        aliases = _build_vendor_aliases("HubSpot")
        pattern = _build_alias_pattern(aliases)
        assert pattern.search("We switched from HubSpot to Salesforce")
        assert pattern.search("hubspot vs salesforce")


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

class TestCandidateScoring:
    def _aliases(self, vendor: str = "HubSpot"):
        return _build_alias_pattern(_build_vendor_aliases(vendor))

    def test_title_hit_boosts(self):
        pattern = self._aliases()
        score, reasons = _score_candidate(
            "HubSpot pricing is terrible",
            "We are paying too much for HubSpot. " * 10,
            {"score": 50, "num_comments": 10},
            pattern,
            "CRM",
        )
        assert score >= 5.0
        assert any("title_hit" in r for r in reasons)

    def test_no_mention_scores_low(self):
        pattern = self._aliases()
        score_no_mention, _ = _score_candidate(
            "Best CRM for small business",
            "I need a CRM system. Any recommendations? " * 5,
            {"score": 50, "num_comments": 10},
            pattern,
            "CRM",
        )
        score_with_mention, _ = _score_candidate(
            "HubSpot pricing is terrible",
            "We are paying too much for HubSpot. " * 10,
            {"score": 50, "num_comments": 10},
            pattern,
            "CRM",
        )
        # No vendor mention should score much lower than with mention
        assert score_no_mention < score_with_mention - 3.0

    def test_job_noise_penalty(self):
        pattern = self._aliases()
        score_clean, _ = _score_candidate(
            "HubSpot vs Salesforce for CRM",
            "Which is better for our team? " * 10,
            {"score": 50, "num_comments": 10},
            pattern,
            "CRM",
        )
        score_noisy, reasons = _score_candidate(
            "HubSpot interview experience and salary",
            "I interviewed at HubSpot. The recruiter was great. Leetcode onsite. " * 5,
            {"score": 50, "num_comments": 10},
            pattern,
            "cscareerquestions",
        )
        assert score_noisy < score_clean
        assert any("job_noise" in r for r in reasons)

    def test_vendor_list_penalty(self):
        pattern = self._aliases()
        _, reasons = _score_candidate(
            "HubSpot, Salesforce, Zoho, Pipedrive, Monday compared",
            "Just a list of CRMs I've tried. " * 10,
            {"score": 50, "num_comments": 10},
            pattern,
            "startups",
        )
        assert any("vendor_list" in r for r in reasons)

    def test_pain_language_boosts(self):
        pattern = self._aliases()
        score, reasons = _score_candidate(
            "HubSpot is terrible and overpriced",
            "Frustrated with HubSpot pricing. It's broken. " * 10,
            {"score": 100, "num_comments": 20},
            pattern,
            "SaaS",
        )
        assert any("pain" in r for r in reasons)
        assert score >= 6.0

    def test_comparison_language_boosts(self):
        pattern = self._aliases()
        score, reasons = _score_candidate(
            "HubSpot vs Salesforce - which is better?",
            "Comparing HubSpot to Salesforce for our company. " * 10,
            {"score": 30, "num_comments": 5},
            pattern,
            "CRM",
        )
        assert any("comparison" in r for r in reasons)

    def test_subreddit_prior_applied(self):
        pattern = self._aliases()
        score_high, _ = _score_candidate(
            "HubSpot issues",
            "HubSpot keeps breaking. " * 10,
            {"score": 50, "num_comments": 10},
            pattern,
            "CRM",  # high prior
        )
        score_low, _ = _score_candidate(
            "HubSpot issues",
            "HubSpot keeps breaking. " * 10,
            {"score": 50, "num_comments": 10},
            pattern,
            "funny",  # low prior (default)
        )
        assert score_high > score_low


# ---------------------------------------------------------------------------
# Insider claim extraction
# ---------------------------------------------------------------------------

class TestInsiderExtraction:
    def test_current_employment(self):
        result = _extract_insider_evidence("I work at HubSpot as a senior PM")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "current"

    def test_current_employment_multi_word_role(self):
        """'I'm a senior product manager at X' must detect employment."""
        result = _extract_insider_evidence("I'm a senior product manager at HubSpot")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "current"

    def test_current_employment_long_role(self):
        result = _extract_insider_evidence("I am a VP of customer success at HubSpot")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "current"

    def test_past_employment(self):
        result = _extract_insider_evidence("I left HubSpot last year after 3 years")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "past"

    def test_former_employee(self):
        result = _extract_insider_evidence("As a former employee of HubSpot, the culture was toxic")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "past"

    def test_ex_employee(self):
        result = _extract_insider_evidence("Ex-engineer here. The product quality is declining.")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "past"

    def test_no_employment(self):
        result = _extract_insider_evidence("HubSpot's pricing is too expensive for us")
        assert result["employment_claim"] is False
        assert result["employment_tense"] is None

    def test_org_signals_layoff(self):
        result = _extract_insider_evidence("There were massive layoffs at HubSpot last quarter")
        assert "layoff" in result["org_signal_types"]

    def test_org_signals_culture(self):
        result = _extract_insider_evidence("The toxic culture at HubSpot. Everyone is leaving.")
        assert "culture" in result["org_signal_types"]
        assert "talent_exodus" in result["org_signal_types"]

    def test_org_signals_morale(self):
        result = _extract_insider_evidence("Morale is at an all-time low. Burnout everywhere.")
        assert "morale" in result["org_signal_types"]

    def test_role_extraction(self):
        result = _extract_insider_evidence("I'm a senior product manager at HubSpot")
        assert result["extracted_role"] is not None
        assert "product manager" in result["extracted_role"].lower()

    def test_no_role_without_pattern(self):
        result = _extract_insider_evidence("HubSpot sucks and I hate it")
        assert result["extracted_role"] is None

    def test_generic_company_language_is_not_vendor_employment(self):
        aliases = _build_alias_pattern(_build_vendor_aliases("Power BI", product_name="Microsoft Power BI"))
        result = _extract_insider_evidence(
            "My company uses Power BI every day and I am a lowly office worker looking for career advice.",
            aliases,
        )
        assert result["employment_claim"] is False
        assert result["employment_tense"] is None

    def test_multiple_org_signals(self):
        result = _extract_insider_evidence(
            "After the reorg, morale tanked. Product quality is declining. Brain drain is real."
        )
        assert len(result["org_signal_types"]) >= 3


# ---------------------------------------------------------------------------
# _parse_post integration (list-post rejection, alias matching)
# ---------------------------------------------------------------------------

class TestParsePost:
    def setup_method(self):
        self.parser = RedditParser()
        self.target = _make_target()

    def test_accepts_on_topic_post(self):
        post = _make_post(
            title="I built an open-source CRM after getting frustrated with HubSpot's pricing",
            selftext="After hitting HubSpot's paywall one too many times. " * 10,
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is not None
        assert result["vendor_name"] == "HubSpot"

    def test_rejects_no_vendor_mention(self):
        post = _make_post(
            title="Best CRM for startups",
            selftext="Looking for a good CRM. Any recommendations? " * 10,
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is None

    def test_rejects_url_only_mention(self):
        post = _make_post(
            title="Check out my new landing page",
            selftext="Visit cdn2.hubspot.net/images/hero.png for the template. " * 5,
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is None

    def test_rejects_short_post(self):
        post = _make_post(
            title="HubSpot issues",
            selftext="Short.",
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is None

    def test_rejects_deleted(self):
        post = _make_post(
            title="HubSpot is terrible",
            selftext="[removed]",
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is None

    def test_deduplicates_by_id(self):
        post = _make_post(
            title="HubSpot pricing frustration",
            selftext="We pay too much for HubSpot. " * 10,
        )
        seen = set()
        r1 = self.parser._parse_post(post, self.target, seen)
        r2 = self.parser._parse_post(post, self.target, seen)
        assert r1 is not None
        assert r2 is None

    def test_metadata_contains_candidate_score(self):
        post = _make_post(
            title="Switching from HubSpot to Salesforce",
            selftext="We decided to switch from HubSpot because of pricing. " * 10,
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is not None
        meta = result["raw_metadata"]
        assert "candidate_score" in meta
        assert meta["candidate_score"] >= _CANDIDATE_SCORE_MIN
        assert "candidate_reason" in meta

    def test_metadata_contains_insider_evidence(self):
        post = _make_post(
            title="I left HubSpot after 3 years - AMA",
            selftext="I left HubSpot last year after being a senior PM. The culture was toxic. Morale was low. " * 5,
            subreddit="ExperiencedDevs",
        )
        result = self.parser._parse_post(
            post, self.target, set(), profile="insider",
        )
        assert result is not None
        meta = result["raw_metadata"]
        assert meta["employment_claim"] is True
        assert meta["employment_tense"] == "past"
        assert len(meta["org_signal_types"]) >= 1

    def test_metadata_contains_subreddit_weight(self):
        post = _make_post(
            title="HubSpot CRM review",
            selftext="Detailed review of HubSpot CRM for our team. " * 10,
            subreddit="CRM",
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is not None
        meta = result["raw_metadata"]
        assert "subreddit_weight" in meta
        assert meta["subreddit_weight"] == _SUBREDDIT_WEIGHT.get("CRM", 0.5)

    def test_vendor_in_title_flag(self):
        post = _make_post(
            title="HubSpot pricing frustration",
            selftext="We pay too much for HubSpot. " * 10,
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is not None
        assert result["raw_metadata"]["vendor_in_title"] is True

    def test_alias_matching_dotcom(self):
        """Monday.com should match 'Monday.com' but NOT bare 'Monday'."""
        target = _make_target("Monday.com")
        # Post using the full "Monday.com" name -- should match
        post_good = _make_post(
            title="Monday.com vs Asana for project management",
            selftext="We are evaluating Monday.com and Asana. Monday.com is better. " * 10,
        )
        result = self.parser._parse_post(post_good, target, set())
        assert result is not None

    def test_common_word_alias_rejected(self):
        """Bare 'Monday' should NOT match Monday.com vendor -- it's a common word."""
        target = _make_target("Monday.com")
        post_noise = _make_post(
            title="Monday was terrible for our rollout",
            selftext="Monday morning the deployment failed. Monday afternoon we rolled back. " * 10,
        )
        result = self.parser._parse_post(post_noise, target, set())
        assert result is None

    def test_reviewer_title_from_role_extraction(self):
        post = _make_post(
            title="HubSpot's new AI features are bad",
            selftext="I'm a senior product manager at a tech company. HubSpot's AI tools are useless. " * 5,
        )
        result = self.parser._parse_post(post, self.target, set())
        assert result is not None
        assert result["reviewer_title"] is not None
        assert "product manager" in result["reviewer_title"].lower()

    def test_author_batch_signals_do_not_inject_fake_reviewer_title(self):
        review = {"reviewer_title": None, "raw_metadata": {}}
        author_posts = [
            {"title": "Switching from HubSpot", "score": 25},
            {"title": "Alternative to HubSpot", "score": 18},
        ]

        _apply_author_batch_signals(review, author_posts, "high")

        assert review["reviewer_title"] is None
        assert review["raw_metadata"]["author_churn_score"] >= 7
        assert review["raw_metadata"]["author_high_churn_signal"] is True
        assert review["raw_metadata"]["trending_score"] == "high"

    def test_insider_profile_without_employment_is_community(self):
        """Bug 1: insider profile posts without employment claim should NOT
        be classified as insider_account."""
        post = _make_post(
            title="HubSpot layoffs discussion",
            selftext="I heard HubSpot had layoffs. Anyone know details? HubSpot seems to be struggling. " * 5,
            subreddit="technology",
        )
        result = self.parser._parse_post(
            post, self.target, set(), profile="insider",
        )
        assert result is not None
        # No employment claim -> should be community_discussion, not insider_account
        assert result["content_type"] == "community_discussion"
        assert result["raw_metadata"]["employment_claim"] is False

    def test_insider_profile_with_employment_is_insider(self):
        """Posts with actual employment claims should get insider_account."""
        post = _make_post(
            title="I left HubSpot last month",
            selftext="I worked at HubSpot for 3 years as a backend engineer. The culture got worse. " * 5,
            subreddit="ExperiencedDevs",
        )
        result = self.parser._parse_post(
            post, self.target, set(), profile="insider",
        )
        assert result is not None
        assert result["content_type"] == "insider_account"
        assert result["raw_metadata"]["employment_claim"] is True

    def test_insider_profile_with_org_signals_is_insider(self):
        """Posts with 2+ org signals (no employment claim) should get insider_account."""
        post = _make_post(
            title="HubSpot toxic culture and morale problems",
            selftext="The toxic culture at HubSpot is well-known. Morale is at an all-time low. Burnout everywhere. " * 5,
            subreddit="cscareerquestions",
        )
        result = self.parser._parse_post(
            post, self.target, set(), profile="insider",
        )
        assert result is not None
        assert result["content_type"] == "insider_account"

    def test_common_word_vendor_rejects_retail_close_story(self):
        target = _make_target("Close", product_name="Close CRM", product_category="CRM")
        post = _make_post(
            title="I did this exact thing to someone with a full basket",
            selftext=(
                "I had to close the register, close the till, and close the lane. "
                "The customer got frustrated and kept switching lines while my company was slammed. "
            ) * 5,
            subreddit="retail",
        )
        result = self.parser._parse_post(post, target, set())
        assert result is None

    def test_common_word_vendor_accepts_explicit_product_context(self):
        target = _make_target("Close", product_name="Close CRM", product_category="CRM")
        post = _make_post(
            title="Close CRM pricing is getting out of hand",
            selftext=(
                "We are replacing Close CRM because the CRM automation is too limited for our sales team. "
            ) * 5,
            subreddit="sales",
        )
        result = self.parser._parse_post(post, target, set())
        assert result is not None
        assert result["vendor_name"] == "Close"

    def test_power_bi_tool_user_is_not_insider(self):
        target = _make_target("Power BI", product_name="Microsoft Power BI", product_category="Data & Analytics")
        post = _make_post(
            title="Career advice for dashboard work",
            selftext=(
                "My company uses Power BI every day and I am a lowly office worker building dashboards. "
                "I want career advice because I am frustrated with the role and thinking about switching jobs. "
            ) * 5,
            subreddit="careeradvice",
        )
        result = self.parser._parse_post(post, target, set(), profile="insider")
        assert result is not None
        assert result["content_type"] == "community_discussion"
        assert result["raw_metadata"]["employment_claim"] is False

    def test_rejects_off_target_subreddit_without_product_context(self):
        target = _make_target(
            "Shopify",
            product_name="Shopify",
            product_category="E-commerce",
            metadata={"search_profile": "churn", "subreddits": "shopify,ecommerce"},
        )
        post = _make_post(
            title="Shopify Rebellion vs Cloud9 White | Post-Match Thread",
            selftext=(
                "Shopify Rebellion looked sharp in the lower final and the roster adapted well on map 3. "
                "This post-match thread covers the main event and the final bracket update. "
            ) * 5,
            subreddit="ValorantCompetitive",
        )
        result = self.parser._parse_post(post, target, set())
        assert result is None

    def test_accepts_off_target_subreddit_with_explicit_product_context(self):
        target = _make_target(
            "Shopify",
            product_name="Shopify",
            product_category="E-commerce",
            metadata={"search_profile": "churn", "subreddits": "shopify,ecommerce"},
        )
        post = _make_post(
            title="Shopify pricing is getting harder to justify for our storefront",
            selftext=(
                "We use Shopify for our ecommerce storefront, but the subscription pricing and app costs keep climbing. "
                "The platform works, yet our team is evaluating whether the software still fits our margin goals. "
            ) * 5,
            subreddit="technology",
        )
        result = self.parser._parse_post(post, target, set())
        assert result is not None
        assert result["raw_metadata"]["subreddit_expected"] is False
        assert result["raw_metadata"]["subreddit_context_reason"] == "explicit_product_context"

    def test_rejects_user_profile_subreddit_even_with_product_context(self):
        target = _make_target(
            "Shopify",
            product_name="Shopify",
            product_category="E-commerce",
            metadata={"search_profile": "churn", "subreddits": "shopify,ecommerce"},
        )
        post = _make_post(
            title="How to make a migration from Shopify to WooCommerce",
            selftext=(
                "We use Shopify for our ecommerce storefront and are comparing migration paths. "
                "This post walks through moving catalog, customers, and order history to another platform. "
            ) * 4,
            subreddit="u_Cart-to-Cart",
        )
        result = self.parser._parse_post(post, target, set())
        assert result is None

    def test_rejects_vendor_specific_career_subreddit(self):
        target = _make_target(
            "Salesforce",
            product_name="Salesforce",
            product_category="CRM",
            metadata={"search_profile": "churn", "subreddits": "salesforce,crm"},
        )
        post = _make_post(
            title="Looking for referral to a solution engineer role at Salesforce",
            selftext=(
                "I want to work at Salesforce and would appreciate career advice on interviewing, "
                "compensation, and getting a referral into the company."
            ) * 3,
            subreddit="SalesforceCareers",
        )
        result = self.parser._parse_post(post, target, set())
        assert result is None

    def test_rejects_vendor_specific_certification_subreddit(self):
        target = _make_target(
            "Azure",
            product_name="Azure",
            product_category="Cloud Infrastructure",
            metadata={"search_profile": "churn", "subreddits": "azure,devops"},
        )
        post = _make_post(
            title="Is AZ-104 still worth it in 2026?",
            selftext=(
                "I want Azure certification advice and career guidance on whether AZ-104 "
                "still matters for cloud roles."
            ) * 3,
            subreddit="AzureCertification",
        )
        result = self.parser._parse_post(post, target, set())
        assert result is None


# ---------------------------------------------------------------------------
# Employment tense detection
# ---------------------------------------------------------------------------

class TestEmploymentTense:
    def test_worked_at_is_past(self):
        """Bug 2: 'I worked at X' must be past, not current."""
        result = _extract_insider_evidence("I worked at HubSpot for 3 years")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "past"

    def test_work_at_is_current(self):
        result = _extract_insider_evidence("I work at HubSpot as an engineer")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "current"

    def test_left_is_past(self):
        result = _extract_insider_evidence("I left HubSpot last year")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "past"

    def test_iam_at_is_current(self):
        result = _extract_insider_evidence("I'm a PM at HubSpot right now")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "current"

    def test_used_to_work_is_past(self):
        result = _extract_insider_evidence("I used to work at HubSpot before the reorg")
        assert result["employment_claim"] is True
        assert result["employment_tense"] == "past"


# ---------------------------------------------------------------------------
# Comment insider classification
# ---------------------------------------------------------------------------

class TestCommentInsiderClassification:
    def setup_method(self):
        self.parser = RedditParser()
        self.target = _make_target()

    def _make_parent_post(self):
        return {
            "source_review_id": "parent123",
            "thread_id": "t3_parent123",
            "source_url": "https://www.reddit.com/r/CRM/comments/parent123/test/",
            "raw_metadata": {"search_profile": "insider"},
        }

    def _make_comment(self, body: str, score: int = 10) -> dict:
        return {
            "kind": "t1",
            "data": {
                "id": "comment456",
                "body": body,
                "score": score,
                "author": "insider_user",
                "created_utc": 1710000000,
                "subreddit": "ExperiencedDevs",
                "permalink": "/r/ExperiencedDevs/comments/parent123/test/comment456/",
                "author_flair_text": None,
            },
        }

    def test_insider_comment_gets_insider_type(self):
        """Bug 3: comments with employment claims should get insider_account type."""
        comment = self._make_comment(
            "I worked at HubSpot for 2 years. The toxic culture drove me out. "
            "Morale was terrible. Management was awful. " * 3,
        )
        result = self.parser._parse_comment(
            comment, self.target, self._make_parent_post(), set(),
            depth=0, min_score=2,
        )
        assert result is not None
        assert result["content_type"] == "insider_account"

    def test_normal_comment_stays_comment_type(self):
        """Comments without insider evidence should remain as 'comment' type."""
        comment = self._make_comment(
            "I think HubSpot is overpriced. We switched to Pipedrive last month. "
            "Much better for small teams. " * 3,
        )
        result = self.parser._parse_comment(
            comment, self.target, self._make_parent_post(), set(),
            depth=0, min_score=2,
        )
        assert result is not None
        assert result["content_type"] == "comment"

    def test_insider_comment_relevance_gets_boosts(self):
        """Insider comments should get insider boosts in relevance scoring,
        not noise penalties."""
        comment_review = {
            "source": "reddit",
            "summary": None,
            "review_text": (
                "I worked at HubSpot. Massive layoffs. Toxic culture. "
                "Leadership churn was constant. Morale was terrible. " * 5
            ),
            "content_type": "insider_account",
            "raw_metadata": {
                "score": 25,
                "upvote_ratio": 0.90,
                "subreddit_weight": 0.7,
                "vendor_in_title": None,
                "employment_claim": True,
                "employment_tense": "past",
                "org_signal_types": ["layoff", "culture", "morale"],
            },
        }
        score, reason = score_relevance(comment_review, "HubSpot")
        assert "insider_signals" in reason or "employment_claim" in reason
        assert "noise_language" not in reason
        assert score >= 0.65


# ---------------------------------------------------------------------------
# Query profile preservation
# ---------------------------------------------------------------------------

class TestQueryProfiles:
    def setup_method(self):
        self.parser = RedditParser()

    def test_churn_profile_uses_churn_queries(self):
        queries = self.parser._build_global_queries("HubSpot", "churn")
        assert any("issues" in q for q in queries)
        assert any("switching from" in q for q in queries)
        # Should NOT include insider templates
        assert not any("worked at" in q for q in queries)

    def test_deep_profile_extends_churn(self):
        churn = self.parser._build_global_queries("HubSpot", "churn")
        deep = self.parser._build_global_queries("HubSpot", "deep")
        assert len(deep) > len(churn)
        assert any("pricing increase" in q for q in deep)
        assert any("support nightmare" in q for q in deep)

    def test_insider_profile_uses_insider_queries(self):
        queries = self.parser._build_global_queries("HubSpot", "insider")
        assert any("worked at" in q for q in queries)
        assert any("layoffs" in q for q in queries)
        # Should NOT include churn templates
        assert not any("issues" in q.lower().replace("hubspot", "").strip('"') for q in queries
                       if "issues" not in q.lower().replace('"hubspot"', ""))


# ---------------------------------------------------------------------------
# Relevance scoring with Reddit metadata
# ---------------------------------------------------------------------------

class TestRelevanceRedditMetadata:
    def _make_review(self, **meta_overrides) -> dict:
        meta = {
            "score": 50,
            "upvote_ratio": 0.85,
            "subreddit": "CRM",
            "subreddit_weight": 0.9,
            "vendor_in_title": True,
            "vendor_mention_count": 3,
            "candidate_score": 6.0,
        }
        meta.update(meta_overrides)
        return {
            "source": "reddit",
            "summary": "HubSpot pricing frustration",
            "review_text": "We switched from HubSpot because it was too expensive. " * 10,
            "content_type": "community_discussion",
            "raw_metadata": meta,
        }

    def test_high_candidate_score_boosts(self):
        review = self._make_review(candidate_score=8.0)
        score, reason = score_relevance(review, "HubSpot")
        assert score >= 0.6
        assert "high_candidate_score" in reason

    def test_good_candidate_score_boosts(self):
        review = self._make_review(candidate_score=5.5)
        score, reason = score_relevance(review, "HubSpot")
        assert "good_candidate_score" in reason

    def test_high_signal_subreddit_boosts(self):
        review = self._make_review(subreddit_weight=0.9)
        score, reason = score_relevance(review, "HubSpot")
        assert "high_signal_subreddit" in reason

    def test_vendor_in_title_from_metadata(self):
        review = self._make_review(vendor_in_title=True)
        score, reason = score_relevance(review, "HubSpot")
        assert "vendor_in_title" in reason

    def test_vendor_mentions_from_metadata(self):
        review = self._make_review(vendor_mention_count=5, vendor_in_title=False)
        score, reason = score_relevance(review, "HubSpot")
        assert "vendor_mentions" in reason

    def test_employment_claim_in_non_insider(self):
        review = self._make_review(employment_claim=True)
        score, reason = score_relevance(review, "HubSpot")
        assert "employment_claim" in reason

    def test_job_noise_penalty(self):
        review = self._make_review()
        review["review_text"] = (
            "I interviewed at HubSpot. The recruiter set up an onsite. "
            "Salary was competitive. Leetcode questions were hard. " * 5
        )
        review["raw_metadata"]["org_signal_types"] = []
        score, reason = score_relevance(review, "HubSpot")
        assert "reddit_job_noise" in reason

    def test_no_job_noise_when_org_signals(self):
        review = self._make_review()
        review["review_text"] = (
            "After the layoffs at HubSpot, I interviewed elsewhere. "
            "The recruiter said morale was low. " * 5
        )
        review["raw_metadata"]["org_signal_types"] = ["layoff", "morale"]
        _, reason = score_relevance(review, "HubSpot")
        # Should NOT apply job noise penalty when org signals present
        assert "reddit_job_noise" not in reason

    def test_match_thread_noise_penalty(self):
        review = self._make_review(
            subreddit="ValorantCompetitive",
            subreddit_weight=0.5,
            candidate_score=7.5,
            vendor_mention_count=1,
        )
        review["summary"] = "Shopify Rebellion vs Cloud9 White | Post-Match Thread"
        review["review_text"] = (
            "Shopify Rebellion won the lower final in the main event. "
            "This post-match thread covers map 1, map 2, and the roster changes before playoffs. "
        ) * 4
        score, reason = score_relevance(review, "Shopify")
        assert score < 0.55
        assert "reddit_match_thread_noise" in reason

    def test_match_discussion_noise_penalty(self):
        review = self._make_review(
            subreddit="leagueoflegends",
            subreddit_weight=0.5,
            candidate_score=7.5,
            vendor_mention_count=2,
        )
        review["summary"] = "Shopify Rebellion vs 100 Thieves / LCS 2024 Spring - Week 6 / Post-Match Discussion"
        review["review_text"] = (
            "Official page links to lolesports, Leaguepedia, and Liquipedia for the LCS spring season. "
            "This post-match discussion covers the lower bracket result and roster storylines. "
        ) * 3
        score, reason = score_relevance(review, "Shopify")
        assert score < 0.55
        assert "reddit_match_thread_noise" in reason

    def test_reddit_aggregator_noise_penalty(self):
        review = self._make_review(
            subreddit="autotldr",
            subreddit_weight=0.5,
            candidate_score=7.0,
            vendor_mention_count=4,
        )
        review["summary"] = "Massive internet outage is sweeping the East Coast"
        review["review_text"] = (
            "This is an automatic summary, [original](http://example.com/story) reduced by 37%.\n"
            "Amazon Web Services outage explained in coverage includes multiple external articles."
        )
        score, reason = score_relevance(review, "Amazon Web Services")
        assert score < 0.55
        assert "reddit_aggregator_noise" in reason

    def test_reddit_investor_news_penalty(self):
        review = self._make_review(
            subreddit="stocks",
            subreddit_weight=0.5,
            candidate_score=7.0,
            vendor_mention_count=4,
        )
        review["summary"] = "Amazon Web Services CEO Adam Selipsky to step down"
        review["review_text"] = (
            "Amazon Web Services CEO Adam Selipsky will step down next month, the company announced. "
            "The investment community expects the stock response to follow the leadership change."
        )
        score, reason = score_relevance(review, "Amazon Web Services")
        assert score < 0.55
        assert "reddit_investor_news" in reason

    def test_reddit_low_signal_subreddit_penalty(self):
        review = self._make_review(
            subreddit="newworldgame",
            subreddit_weight=0.5,
            candidate_score=7.0,
            vendor_mention_count=4,
        )
        review["summary"] = "New World Servers: A technical explanation"
        review["review_text"] = (
            "Amazon Web Services powers parts of the game infrastructure, "
            "but this thread is really about regional game server behavior."
        ) * 4
        score, reason = score_relevance(review, "Amazon Web Services")
        assert score < 0.55
        assert "reddit_low_signal_subreddit" in reason

    def test_reddit_investor_noise_penalty_applies_to_insider_tagged_rows(self):
        review = self._make_review(
            subreddit="wallstreetbets",
            subreddit_weight=0.5,
            candidate_score=6.1,
            vendor_mention_count=3,
            employment_claim=True,
            employment_tense="past",
        )
        review["content_type"] = "insider_account"
        review["summary"] = "Amazon Investing an Additional $4 Billion in AI Firm Anthropic"
        review["review_text"] = (
            "Amazon.com Inc. is boosting its stake in Anthropic. "
            "The market is watching the investment and leadership posture closely."
        ) * 3
        score, reason = score_relevance(review, "Amazon Web Services")
        assert score < 0.55
        assert "reddit_investor_news" in reason

    def test_reddit_user_profile_subreddit_penalty(self):
        review = self._make_review(
            subreddit="u_Cart-to-Cart",
            subreddit_weight=0.5,
            candidate_score=7.2,
            vendor_mention_count=6,
        )
        review["summary"] = "How to make a migration from Shopify to WooCommerce"
        review["review_text"] = (
            "We use Shopify for our ecommerce storefront and are planning a migration path. "
            "This post explains how to move products, customers, and order history."
        ) * 4
        score, reason = score_relevance(review, "Shopify")
        assert score < 0.55
        assert "reddit_user_profile_subreddit" in reason

    def test_reddit_career_subreddit_penalty_for_employer_discussion(self):
        review = self._make_review(
            subreddit="cscareerquestions",
            subreddit_weight=0.6,
            candidate_score=7.2,
            vendor_mention_count=4,
        )
        review["summary"] = "Would Workday or HomeAway be a better place to work as a Software Engineering Intern?"
        review["review_text"] = (
            "I am deciding between internship offers and want opinions on engineering reputation, "
            "compensation, and long term career value."
        ) * 3
        score, reason = score_relevance(review, "Workday")
        assert score < 0.55
        assert "reddit_career_subreddit_noise" in reason

    def test_reddit_career_subreddit_keeps_strong_product_evaluation(self):
        review = self._make_review(
            subreddit="ITCareerQuestions",
            subreddit_weight=0.6,
            candidate_score=7.1,
            vendor_mention_count=3,
        )
        review["summary"] = "How I finally solved my project management chaos as a DevOps engineer"
        review["review_text"] = (
            "I switched from Todoist to ClickUp because I needed stronger ticketing, roadmap, and sprint management. "
            "The product comparison is about actual day to day usage, not a career change."
        ) * 3
        score, reason = score_relevance(review, "ClickUp")
        assert score >= 0.55
        assert "reddit_career_subreddit_noise" not in reason

    def test_reddit_builder_self_promo_penalty(self):
        review = self._make_review(
            subreddit="SideProject",
            subreddit_weight=0.6,
            candidate_score=7.2,
            vendor_mention_count=4,
        )
        review["summary"] = "I built an open-source CRM after getting frustrated with HubSpot's pricing"
        review["review_text"] = (
            "After hitting HubSpot's paywall one too many times, I built an open-source CRM. "
            "Looking for feedback from early users before I launch it more broadly."
        ) * 3
        score, reason = score_relevance(review, "HubSpot")
        assert score < 0.55
        assert "reddit_builder_self_promo" in reason

    def test_reddit_builder_self_promo_penalty_for_created_feedback_pitch(self):
        review = self._make_review(
            subreddit="SideProject",
            subreddit_weight=0.6,
            candidate_score=7.0,
            vendor_mention_count=3,
        )
        review["summary"] = "I just created a simple business platform as an alternative to HubSpot"
        review["review_text"] = (
            "I am developing an open-source alternative and would love feedback from founders. "
            "Full disclosure: I am the builder and I am looking for honest outside opinions."
        ) * 3
        score, reason = score_relevance(review, "HubSpot")
        assert score < 0.55
        assert "reddit_builder_self_promo" in reason

    def test_reddit_builder_self_promo_penalty_for_we_are_building_pitch(self):
        review = self._make_review(
            subreddit="SideProject",
            subreddit_weight=0.6,
            candidate_score=7.1,
            vendor_mention_count=3,
        )
        review["summary"] = "Would you use a Slack-based AI agent that connects to your tools?"
        review["review_text"] = (
            "We are building a Slack agent for engineering teams and would love honest outside opinions. "
            "The goal is to replace manual workflow steps across Jira, Slack, and Google Calendar."
        ) * 3
        score, reason = score_relevance(review, "Slack")
        assert score < 0.55
        assert "reddit_builder_self_promo" in reason

    def test_reddit_builder_self_promo_penalty_applies_to_insider_tagged_row(self):
        review = self._make_review(
            subreddit="SideProject",
            subreddit_weight=0.6,
            candidate_score=6.8,
            vendor_mention_count=3,
            employment_claim=True,
            employment_tense="current",
        )
        review["content_type"] = "insider_account"
        review["summary"] = "I built a macOS app that rewrites your Slack messages before you send them"
        review["review_text"] = (
            "I built a Slack app for teams and I am the founder. "
            "The product rewrites messages before send and I want feedback from early users."
        ) * 3
        score, reason = score_relevance(review, "Slack")
        assert score < 0.55
        assert "reddit_builder_self_promo" in reason

    def test_reddit_builder_growth_marketing_penalty(self):
        review = self._make_review(
            subreddit="SideProject",
            subreddit_weight=0.6,
            candidate_score=6.5,
            vendor_mention_count=3,
        )
        review["summary"] = "How I Got My First 200 Users by Gaming AI Recommendations"
        review["review_text"] = (
            "Launched my side project 6 months ago. SEO takes forever and paid ads burned through my budget. "
            "I got my first 200 users by getting mentioned in recommendation flows."
        ) * 2
        score, reason = score_relevance(review, "Notion")
        assert score < 0.55
        assert "reddit_builder_self_promo" in reason

    def test_reddit_builder_founder_research_penalty(self):
        review = self._make_review(
            subreddit="SideProject",
            subreddit_weight=0.6,
            candidate_score=6.2,
            vendor_mention_count=4,
        )
        review["summary"] = "Shopify app developers: merchant comprehension gap"
        review["review_text"] = (
            "I am looking into a potential bottleneck in app support. "
            "I have a hypothesis and I am trying to understand whether developers see the same issue."
        ) * 3
        score, reason = score_relevance(review, "Shopify")
        assert score < 0.55
        assert "reddit_builder_self_promo" in reason

    def test_reddit_builder_subreddit_keeps_plain_comparison(self):
        review = self._make_review(
            subreddit="SideProject",
            subreddit_weight=0.6,
            candidate_score=7.1,
            vendor_mention_count=3,
        )
        review["summary"] = "GetResponse and ActiveCampaign: which is better for beginners?"
        review["review_text"] = (
            "I am comparing GetResponse and ActiveCampaign for email marketing automation. "
            "I want to understand pricing, templates, and ease of use before I choose."
        ) * 3
        score, reason = score_relevance(review, "ActiveCampaign")
        assert score >= 0.55
        assert "reddit_builder_self_promo" not in reason


class TestRelevanceInsiderMetadata:
    def _make_insider_review(self, **meta_overrides) -> dict:
        meta = {
            "score": 100,
            "upvote_ratio": 0.90,
            "subreddit": "ExperiencedDevs",
            "subreddit_weight": 0.7,
            "vendor_in_title": True,
            "vendor_mention_count": 4,
            "candidate_score": 7.0,
            "employment_claim": True,
            "employment_tense": "past",
            "org_signal_types": ["culture", "morale"],
        }
        meta.update(meta_overrides)
        return {
            "source": "reddit",
            "summary": "I left HubSpot after 3 years",
            "review_text": "I worked at HubSpot. The toxic culture drove everyone away. Morale was terrible. " * 10,
            "content_type": "insider_account",
            "raw_metadata": meta,
        }

    def test_insider_employment_past_boost(self):
        review = self._make_insider_review(employment_tense="past")
        score, reason = score_relevance(review, "HubSpot")
        assert "employment_claim(past)" in reason
        assert score >= 0.7

    def test_insider_employment_current_boost(self):
        review = self._make_insider_review(employment_tense="current")
        score, reason = score_relevance(review, "HubSpot")
        assert "employment_claim(current)" in reason

    def test_insider_org_signals_boost(self):
        review = self._make_insider_review(org_signal_types=["layoff", "culture", "morale"])
        _, reason = score_relevance(review, "HubSpot")
        assert "org_signals" in reason

    def test_insider_author_score_boost(self):
        review = self._make_insider_review(insider_score=8.5)
        score, reason = score_relevance(review, "HubSpot")
        assert "insider_author_score" in reason
        assert score >= 0.8

    def test_insider_no_noise_penalty(self):
        """Insider accounts should not get noise penalties for layoff/HR language."""
        review = self._make_insider_review()
        review["review_text"] = (
            "After the massive layoffs, the CEO replaced the CTO. "
            "Revenue declined. Headcount reduction everywhere. " * 5
        )
        score, reason = score_relevance(review, "HubSpot")
        # Should get insider boosts, not noise penalties
        assert "noise_language" not in reason
        assert score >= 0.6


# ---------------------------------------------------------------------------
# Fallback: non-Reddit sources still use text-based vendor matching
# ---------------------------------------------------------------------------

class TestRelevanceFallback:
    def test_non_reddit_uses_text_matching(self):
        review = {
            "source": "hackernews",
            "summary": "HubSpot pricing rant",
            "review_text": "HubSpot is way too expensive. We switched to Salesforce. " * 10,
            "raw_metadata": {"score": 50},
        }
        score, reason = score_relevance(review, "HubSpot")
        assert "vendor_in_title" in reason
        assert score >= 0.5
