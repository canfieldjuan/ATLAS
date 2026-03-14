"""
Reddit parser for B2B review scraping.

Uses Reddit's official OAuth2 API (oauth.reddit.com) for authenticated access.
Falls back to public JSON endpoints if credentials are not configured.

Authenticated API benefits:
  - 600 requests/10 min (vs 10/min unauthenticated)
  - Access to all subreddits including quarantined
  - Larger result sets (up to 100 per page vs 25)
  - Reliable -- no 403/429 blocks

Search profiles (set via target.metadata["search_profile"]):
  - "churn"   (default) -- vendor name + churn qualifiers, posts only
  - "deep"    -- churn + pain/frustration + comparison queries, top comments
  - "insider" -- employee/org queries, insider subreddits, top comments

Enhancements (v3):
  - Precision-first query builder: exact/churn/evaluation variants per subreddit
  - Dynamic source_weight: flair boost (+0.2), award boost (+0.02 each), crosspost boost (+0.15)
  - Author churn scoring: per-batch score (0-10) flagging repeat migration authors
  - Edit history tracking: is_edited + edit_timestamp from post.edited
  - Cross-post detection: is_crosspost + crosspost_subreddits, source_weight +0.15
  - Time-based trending score: low/medium/high from 7-day vs 30-day daily mention rate
  - Semaphore parallelism: max 3 concurrent subreddit requests
  - Rate-limit guard: auto-pause 120s after 480 requests (~80% of cap)
  - Comment harvesting for deep/insider profiles (separate DB rows with threading metadata)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import httpx

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.reddit")

_DOMAIN = "reddit.com"
_MIN_SELFTEXT_LEN = 100       # Skip short posts
_MIN_COMMENT_LEN = 80         # Skip short comments
_MIN_COMMENT_SCORE = 2        # Skip low-signal comments
_MAX_SEARCH_REQUESTS = 60     # Cap total search requests per scrape to stay within rate limits
_MAX_COMMENT_FETCHES = 30     # Cap comment fetches per scrape
_COMMENT_TRIGGER_KEYWORDS = {
    "migration", "migrating", "evaluating", "replacing", "budget",
    "cto", "switch", "switching", "alternative", "leaving", "moved",
    "procurement", "pilot", "rfq",
}

# ---------------------------------------------------------------------------
# Subreddit lists
# ---------------------------------------------------------------------------

# Default B2B discussion subreddits (churn / deep profiles)
_DEFAULT_SUBREDDITS = [
    "sysadmin", "salesforce", "aws", "ITManagers",
    "devops", "msp", "networking", "cybersecurity",
    "CRM", "projectmanagement", "SaaS", "startups",
    "smallbusiness", "marketing", "CustomerSuccess",
    "EnterpriseIT", "business", "softwarearchitecture",
]

# Subreddits focused on employee/org insider accounts
_INSIDER_SUBREDDITS = [
    "cscareerquestions", "ExperiencedDevs", "ITCareerQuestions",
    "sysadmin", "devops", "antiwork", "jobs", "technology",
    "cscareeradvice", "softwaregore",
]

# ---------------------------------------------------------------------------
# Query templates per profile
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global search queries -- these search ALL of Reddit (primary strategy)
# ---------------------------------------------------------------------------

# Churn intent -- people actively switching or evaluating
_CHURN_QUERIES = [
    '"{vendor}" issues',
    '"{vendor}" problems',
    '"switching from {vendor}"',
    '"alternative to {vendor}"',
    '"replacing {vendor}"',
    '"migrating from {vendor}"',
    '"moved away from {vendor}"',
    '"leaving {vendor}"',
    '"{vendor}" frustrated',
    '"{vendor}" vs',
    '"{vendor}" worth it',
    '"{vendor}" too expensive',
    '"{vendor}" terrible',
    '"{vendor}" hate',
]

# Deep profile -- pain, pricing, support failures
_DEEP_QUERY_TEMPLATES = [
    '"{vendor}" pricing increase',
    '"{vendor}" support nightmare',
    '"{vendor}" downgrade',
    '"{vendor}" cancelling',
    '"{vendor}" regret',
    '"{vendor}" broken',
    '"{vendor}" outage',
    '"{vendor}" worst experience',
    '"{vendor}" looking for replacement',
    '"{vendor}" contract locked',
]

# Insider profile -- employee/org intelligence
_INSIDER_QUERY_TEMPLATES = [
    '"worked at {vendor}"',
    '"left {vendor}"',
    '"{vendor}" culture toxic',
    '"{vendor}" layoffs',
    '"{vendor}" morale',
    '"{vendor}" product quality declining',
    '"{vendor}" engineering culture',
    '"{vendor}" management terrible',
    '"{vendor}" glassdoor',
    '"inside {vendor}"',
    '"{vendor}" employees',
]

# ---------------------------------------------------------------------------
# Weight constants
# ---------------------------------------------------------------------------

# Flair → source_weight boost
_FLAIR_WEIGHT_BOOST: dict[str, float] = {
    "rant": 0.2,
    "help": 0.15,
    "discussion": 0.1,
    "question": 0.1,
    "advice": 0.1,
    "complaint": 0.2,
}

# Author churn score weights
_AUTHOR_CHURN_MIGRATION_WEIGHT = 3
_AUTHOR_CHURN_UPVOTE_WEIGHT = 0.1
_AUTHOR_HIGH_SCORE_THRESHOLD = 7

# ---------------------------------------------------------------------------
# Subreddit priors -- weight multiplier reflecting signal density per sub
# ---------------------------------------------------------------------------

_SUBREDDIT_WEIGHT: dict[str, float] = {
    # High-signal B2B discussion subs
    "sysadmin": 0.8,
    "ITManagers": 0.8,
    "devops": 0.7,
    "msp": 0.8,
    "CRM": 0.9,
    "CustomerSuccess": 0.9,
    "SaaS": 0.8,
    "EnterpriseIT": 0.8,
    "projectmanagement": 0.7,
    # Vendor-specific subs tend to be high-signal when matched
    "salesforce": 0.85,
    "hubspot": 0.85,
    "Zoho": 0.85,
    # Career / insider subs
    "cscareerquestions": 0.6,
    "ExperiencedDevs": 0.7,
    "antiwork": 0.5,
    # General subs -- lower prior, higher noise
    "startups": 0.6,
    "smallbusiness": 0.6,
    "marketing": 0.5,
    "sales": 0.5,
    "technology": 0.4,
    "business": 0.4,
}

_DEFAULT_SUBREDDIT_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Insider evidence extraction patterns
# ---------------------------------------------------------------------------

_EMPLOYMENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Past employment (checked first -- more specific matches win)
    (re.compile(r"\b(?:I|we)\s+worked\s+(?:at|for)\s+", re.I), "past"),
    (re.compile(r"\b(?:I|we)\s+(?:left|quit|resigned\s+from|was\s+laid\s+off\s+from)\s+", re.I), "past"),
    (re.compile(r"\bformer(?:ly)?\s+(?:at|employee|engineer|dev)", re.I), "past"),
    (re.compile(r"\bused\s+to\s+work\s+(?:at|for)\s+", re.I), "past"),
    (re.compile(r"\bex[\s-](?:employee|engineer|dev|PM|manager)", re.I), "past"),
    # Current employment
    (re.compile(r"\b(?:I|we)\s+work\s+(?:at|for)\s+", re.I), "current"),
    (re.compile(r"\b(?:I'm|I am)\s+(?:a|an)\s+\w+\s+at\s+", re.I), "current"),
    (re.compile(r"\bmy\s+(?:company|employer|team|org)\b", re.I), "current"),
]

_ORG_SIGNAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\blayoffs?\b|\blet\s+go\b|\bheadcount\s+(?:cut|reduction)", re.I), "layoff"),
    (re.compile(r"\breorg(?:aniz)?(?:ed|ation|ing)?\b", re.I), "reorg"),
    (re.compile(r"\b(?:toxic|terrible|awful)\s+(?:culture|management|leadership)", re.I), "culture"),
    (re.compile(r"\bmorale\b|\bburnout\b|\bturnover\b", re.I), "morale"),
    (re.compile(r"\bproduct\s+(?:quality|direction)\s+(?:is\s+)?(?:declining|tanking|bad)", re.I), "quality_decline"),
    (re.compile(r"\b(?:leadership|exec|C-suite)\s+(?:churn|turnover|exodus)", re.I), "leadership_churn"),
    (re.compile(r"\bbrain\s+drain\b|\beveryone(?:'s| is)\s+leaving\b", re.I), "talent_exodus"),
]

# Role extraction from self-identification text
_ROLE_PATTERN = re.compile(
    r"\bI(?:'m| am)\s+(?:a|an)\s+([\w\s/&-]{3,30}?)\s+(?:at|for|working)\b",
    re.I,
)

# Candidate score thresholds
_CANDIDATE_SCORE_MIN = 2.0   # Reject posts below this score
_MAX_AUTHOR_FETCHES = 10     # Cap selective author history lookups

# ---------------------------------------------------------------------------
# Churn / comparison / pain language for candidate scoring
# ---------------------------------------------------------------------------

_COMPARISON_WORDS = frozenset({
    "vs", "versus", "compared", "comparison", "alternative", "alternatives",
    "better", "worse", "competitor",
})

_PAIN_WORDS = frozenset({
    "frustrated", "frustrating", "hate", "terrible", "awful", "broken",
    "nightmare", "expensive", "overpriced", "buggy", "worst", "regret",
    "sucks", "horrible", "unusable", "unreliable",
})

# Job-hunt / interview noise words (penalized in candidate scoring)
_JOB_NOISE_WORDS = frozenset({
    "interview", "interviewing", "interviewed", "hiring", "recruiter",
    "resume", "salary", "leetcode", "onsite", "offer", "rejected",
    "job hunt", "job hunting", "job search",
})


# ---------------------------------------------------------------------------
# Alias builder
# ---------------------------------------------------------------------------

# Common English words that should NOT be used as standalone vendor aliases.
# These arise from .com stripping (Monday.com -> "monday") or multi-word
# splitting and would match non-product text.
_COMMON_WORD_BLOCKLIST = frozenset({
    "monday", "friday", "sunday", "saturday",
    "click", "base", "smart", "team", "fresh",
    "notion", "slack", "zoom", "power", "close",
    "copper", "ripple", "gusto", "brevo", "wrike",
    "help", "work", "hub", "pipe", "drive",
    "big", "get", "go", "look", "open",
})


def _build_vendor_aliases(vendor_name: str, extra_aliases: list[str] | None = None) -> list[str]:
    """Derive normalized alias variants from a vendor name.

    Returns a list of lowercase aliases including the original name.
    Handles cases like "Monday.com" -> ["monday.com"] (NOT "monday"),
    "HubSpot" -> ["hubspot"], "Zoho CRM" -> ["zoho crm", "zoho"].

    Single common-word derivatives are suppressed to avoid matching
    plain English (e.g. "Monday was terrible" != Monday.com).
    """
    base = vendor_name.lower().strip()
    aliases = {base}

    # Strip trailing ".com", ".io", etc. -- but only add if not a common word
    stripped = re.sub(r"\.\w{2,4}$", "", base)
    if stripped and stripped != base and stripped not in _COMMON_WORD_BLOCKLIST:
        aliases.add(stripped)

    # Split multi-word: "Zoho CRM" -> also match "Zoho" (unless common word)
    parts = base.split()
    if len(parts) >= 2 and parts[0] not in _COMMON_WORD_BLOCKLIST:
        aliases.add(parts[0])

    # Add explicit aliases from target metadata
    for alias in (extra_aliases or []):
        a = alias.lower().strip()
        if a:
            aliases.add(a)

    return sorted(aliases, key=len, reverse=True)


def _build_alias_pattern(aliases: list[str]) -> re.Pattern[str]:
    """Build a compiled regex that matches any vendor alias with word boundaries."""
    escaped = [re.escape(a) for a in aliases]
    joined = "|".join(escaped)
    return re.compile(
        r"(?<![./\w])(?:" + joined + r")(?![.\w])",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Candidate scorer
# ---------------------------------------------------------------------------

def _score_candidate(
    title: str,
    selftext: str,
    post: dict,
    alias_pattern: re.Pattern[str],
    subreddit: str,
) -> tuple[float, list[str]]:
    """Score a Reddit post candidate before full parsing.

    Returns (score, list_of_reasons). Higher = more relevant.
    Score components:
      - Title alias hit: +3.0
      - Early body hit (<200 chars): +2.0
      - Total mention count: +0.5 per mention (capped at +2.0)
      - Comparison/churn language: +1.5
      - Pain language: +1.0
      - Subreddit prior: +0.0 to +1.0
      - Engagement: log-scaled from score + comments
      - Job-hunt noise: -2.0
      - Vendor-list pattern: -1.5
    """
    score = 0.0
    reasons: list[str] = []
    title_lower = title.lower()
    body_preview = selftext[:2000].lower()
    combined = f"{title_lower} {body_preview}"

    # -- Vendor mention analysis --
    title_hits = len(alias_pattern.findall(title))
    body_hits = len(alias_pattern.findall(selftext[:2000]))
    total_mentions = title_hits + body_hits

    if title_hits > 0:
        score += 3.0
        reasons.append("title_hit=+3.0")

    early_hit = alias_pattern.search(selftext[:200])
    if early_hit and title_hits == 0:
        score += 2.0
        reasons.append("early_body=+2.0")

    mention_bonus = min(total_mentions * 0.5, 2.0)
    if mention_bonus > 0:
        score += mention_bonus
        reasons.append(f"mentions({total_mentions})=+{mention_bonus:.1f}")

    # -- Churn / comparison language --
    churn_hit = any(w in combined for w in _COMPARISON_WORDS)
    if churn_hit:
        score += 1.5
        reasons.append("comparison=+1.5")

    # -- Pain language --
    pain_hit = any(w in combined for w in _PAIN_WORDS)
    if pain_hit:
        score += 1.0
        reasons.append("pain=+1.0")

    # -- Subreddit prior --
    sub_weight = _SUBREDDIT_WEIGHT.get(subreddit, _DEFAULT_SUBREDDIT_WEIGHT)
    sub_bonus = sub_weight * 1.0  # scale to 0.0-1.0 range
    score += sub_bonus
    reasons.append(f"sub_prior({subreddit})=+{sub_bonus:.1f}")

    # -- Engagement --
    reddit_score = post.get("score", 0)
    num_comments = post.get("num_comments", 0)
    import math
    engagement = math.log1p(max(reddit_score, 0) + max(num_comments, 0) * 2) * 0.3
    engagement = min(engagement, 1.5)
    score += engagement
    reasons.append(f"engagement=+{engagement:.1f}")

    # -- Penalties --

    # Job-hunt / interview noise
    job_noise = sum(1 for w in _JOB_NOISE_WORDS if w in combined)
    if job_noise >= 2:
        score -= 2.0
        reasons.append(f"job_noise({job_noise})=-2.0")

    # Vendor-list detection: if the text contains 3+ other vendor/company names
    # in a list-like pattern near the vendor mention, it's likely noise
    # Heuristic: count commas within 100 chars of a vendor mention in title
    if title_hits > 0 and title_lower.count(",") >= 3:
        score -= 1.5
        reasons.append("vendor_list=-1.5")

    return round(score, 2), reasons


# ---------------------------------------------------------------------------
# Insider evidence extraction
# ---------------------------------------------------------------------------

def _extract_insider_evidence(text: str) -> dict:
    """Extract employment claims and org signals from post text.

    Returns a dict with:
      - employment_claim: bool
      - employment_tense: "current" | "past" | None
      - org_signal_types: list[str]  (e.g. ["layoff", "culture"])
      - extracted_role: str | None  (e.g. "senior PM")
    """
    employment_claim = False
    employment_tense = None
    org_signals: list[str] = []
    extracted_role = None

    for pat, tense in _EMPLOYMENT_PATTERNS:
        if pat.search(text):
            employment_claim = True
            employment_tense = tense
            break

    for pat, signal_type in _ORG_SIGNAL_PATTERNS:
        if pat.search(text):
            org_signals.append(signal_type)

    role_match = _ROLE_PATTERN.search(text)
    if role_match:
        extracted_role = role_match.group(1).strip()

    return {
        "employment_claim": employment_claim,
        "employment_tense": employment_tense,
        "org_signal_types": org_signals,
        "extracted_role": extracted_role,
    }


def _get_reddit_credentials() -> tuple[str, str]:
    """Load Reddit API credentials from config."""
    try:
        from ....config import settings
        return settings.b2b_scrape.reddit_client_id, settings.b2b_scrape.reddit_client_secret
    except Exception:
        pass
    import os
    return (
        os.environ.get("ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID", ""),
        os.environ.get("ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET", ""),
    )


def _compute_author_churn_score(author_posts: list[dict]) -> float:
    """
    Score an author's churn signal based on their post history within this batch.

    Weights:
      - Each migration-related post: +3
      - Each upvote on any post: +0.1
      - Use of churn qualifiers in titles: +2 each (capped at 6)
    """
    migration_keywords = {"migration", "migrating", "switching", "replacing", "leaving", "alternative"}
    qualifier_keywords = {"switching from", "alternative to", "replacing",
                          "migrating from", "moved away from", "leaving"}

    migration_count = sum(
        1 for p in author_posts
        if any(kw in p.get("title", "").lower() for kw in migration_keywords)
    )
    upvote_total = sum(p.get("score", 0) for p in author_posts)
    qualifier_count = min(
        sum(
            1 for p in author_posts
            if any(q in p.get("title", "").lower() for q in qualifier_keywords)
        ),
        3,  # cap at 3 occurrences → max 6 points
    )

    raw = (
        migration_count * _AUTHOR_CHURN_MIGRATION_WEIGHT
        + upvote_total * _AUTHOR_CHURN_UPVOTE_WEIGHT
        + qualifier_count * 2
    )
    return min(round(raw, 2), 10.0)


def _compute_batch_trending_score(vendor_posts: list[dict]) -> str:
    """
    Classify trending level for the current scrape batch by comparing the
    7-day daily mention rate to the 30-day daily mention rate.

    Spike ≥2× monthly daily rate → high; ≥1.25× → medium; else low.
    vendor_posts: all posts collected for this vendor in the current batch.

    Returns a single score applied uniformly to all posts in the batch —
    this is a batch-level signal, not a per-post metric.
    """
    if not vendor_posts:
        return "low"

    now = datetime.now(tz=timezone.utc)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    recent = 0
    monthly = 0
    for p in vendor_posts:
        ts = p.get("reviewed_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if dt >= cutoff_7d:
            recent += 1
        if dt >= cutoff_30d:
            monthly += 1

    if monthly == 0:
        return "low"

    daily_recent = recent / 7
    daily_monthly = monthly / 30

    if daily_monthly == 0:
        return "low"

    ratio = daily_recent / daily_monthly
    if ratio >= 2.0:
        return "high"
    if ratio >= 1.25:
        return "medium"
    return "low"


class RedditParser:
    """Parse Reddit posts (and optionally comments) as B2B review proxies."""

    source_name = "reddit"
    prefer_residential = False  # No proxy needed
    version = "reddit:3"

    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expires: float = 0

    # ------------------------------------------------------------------
    # OAuth2
    # ------------------------------------------------------------------

    async def _get_oauth_token(self, client_id: str, client_secret: str) -> str | None:
        """Obtain Reddit OAuth2 application-only token."""
        import time
        if self._token and time.monotonic() < self._token_expires:
            return self._token

        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=(client_id, client_secret),
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": "Atlas/2.0 B2B Intelligence"},
                )
                resp.raise_for_status()
                data = resp.json()
                self._token = data["access_token"]
                # Token valid for ~1 hour, refresh at 50 min
                self._token_expires = time.monotonic() + 3000
                logger.info("Reddit OAuth2 token acquired (expires in %ds)", data.get("expires_in", 3600))
                return self._token
        except Exception as exc:
            logger.warning("Reddit OAuth2 token request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Reddit for posts (and optionally comments) mentioning the vendor."""
        client_id, client_secret = _get_reddit_credentials()
        has_auth = bool(client_id and client_secret)

        if has_auth:
            token = await self._get_oauth_token(client_id, client_secret)
            if token:
                return await self._scrape_authenticated(target, token)
            logger.warning("OAuth2 token failed, falling back to public endpoints")

        return await self._scrape_public(target, client)

    # ------------------------------------------------------------------
    # Authenticated scrape
    # ------------------------------------------------------------------

    async def _scrape_authenticated(self, target: ScrapeTarget, token: str) -> ScrapeResult:
        """Scrape via Reddit OAuth2 API. Dispatches by search_profile."""
        profile = (target.metadata.get("search_profile") or "churn").lower()

        if profile == "insider":
            subreddits = target.metadata.get("subreddits") or _INSIDER_SUBREDDITS
        else:
            subreddits = target.metadata.get("subreddits") or _DEFAULT_SUBREDDITS

        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(",")]

        # Build alias pattern once for the entire scrape
        aliases = _build_vendor_aliases(
            target.vendor_name,
            target.metadata.get("vendor_aliases"),
        )
        alias_pattern = _build_alias_pattern(aliases)

        reviews: list[dict] = []
        errors: list[str] = []
        seen_ids: set[str] = set()
        # author -> list of their parsed post dicts (for scoring)
        author_index: dict[str, list[dict]] = {}

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Atlas/2.0 B2B Intelligence",
        }

        # Semaphore: max 3 concurrent requests
        sem = asyncio.Semaphore(3)
        pages_scraped_ref = [0]  # mutable int via list

        async def _get(http: httpx.AsyncClient, url: str) -> httpx.Response | None:
            async with sem:
                try:
                    resp = await http.get(url)
                    pages_scraped_ref[0] += 1
                    if resp.status_code == 429:
                        logger.debug("Reddit rate limited, pausing 2s")
                        await asyncio.sleep(2)
                        errors.append(f"GET {url}: 429 rate limited")
                        return None
                    return resp
                except Exception as exc:
                    errors.append(f"GET {url}: {exc}")
                    return None

        async with httpx.AsyncClient(timeout=30, headers=headers) as http:

            async def _maybe_pause() -> None:
                if pages_scraped_ref[0] > 0 and pages_scraped_ref[0] % 480 == 0:
                    logger.info("Reddit: approaching rate limit, pausing 120s")
                    await asyncio.sleep(120)

            # ---- Strategy 1 (PRIMARY): global intent search across ALL of Reddit ----
            # This is where the signal lives -- real people in general subreddits
            # complaining, evaluating alternatives, sharing insider info.
            for query in self._build_global_queries(target.vendor_name, profile):
                url = (
                    f"https://oauth.reddit.com/search"
                    f"?q={quote_plus(query)}&sort=relevance&limit=100&t=all"
                )
                resp = await _get(http, url)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    for post_wrapper in data.get("data", {}).get("children", []):
                        review = self._parse_post(post_wrapper, target, seen_ids, profile=profile, alias_pattern=alias_pattern)
                        if review:
                            reviews.append(review)
                            author = review.get("reviewer_name", "")
                            if author:
                                author_index.setdefault(author, []).append({
                                    "title": review.get("summary", ""),
                                    "score": review.get("raw_metadata", {}).get("score", 0),
                                    "reviewed_at": review.get("reviewed_at"),
                                })
                await _maybe_pause()
                await asyncio.sleep(0.3)

            # Also search by "new" to catch recent posts that aren't top-ranked yet
            for query in self._build_global_queries(target.vendor_name, profile)[:5]:
                url = (
                    f"https://oauth.reddit.com/search"
                    f"?q={quote_plus(query)}&sort=new&limit=100&t=year"
                )
                resp = await _get(http, url)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    for post_wrapper in data.get("data", {}).get("children", []):
                        review = self._parse_post(post_wrapper, target, seen_ids, profile=profile, alias_pattern=alias_pattern)
                        if review:
                            reviews.append(review)
                            author = review.get("reviewer_name", "")
                            if author:
                                author_index.setdefault(author, []).append({
                                    "title": review.get("summary", ""),
                                    "score": review.get("raw_metadata", {}).get("score", 0),
                                    "reviewed_at": review.get("reviewed_at"),
                                })
                await asyncio.sleep(0.3)

            # ---- Comment harvesting for deep / insider profiles ----
            if profile in ("deep", "insider"):
                comment_limit = 15 if profile == "insider" else 10
                comment_fetches = 0

                for post_dict in list(reviews):  # iterate snapshot
                    if comment_fetches >= _MAX_COMMENT_FETCHES:
                        logger.info(
                            "Reddit comment fetch cap (%d) reached for %s [%s]",
                            _MAX_COMMENT_FETCHES, target.vendor_name, profile,
                        )
                        break

                    post_id = post_dict.get("source_review_id", "")
                    num_comments = (post_dict.get("raw_metadata") or {}).get("num_comments", 0)
                    if not post_id or num_comments < 3:
                        continue

                    comments, cpages, cerrs = await self._fetch_comments_authenticated(
                        http, post_id, target, post_dict, comment_limit, _MIN_COMMENT_SCORE, seen_ids,
                    )
                    reviews.extend(comments)
                    pages_scraped_ref[0] += cpages
                    errors.extend(cerrs)
                    comment_fetches += 1

            # ---- Selective insider author expansion ----
            # For insider profile: fetch recent history for top insider candidates
            # to find repeated vendor mentions and org-level signals.
            if profile == "insider":
                insider_candidates = [
                    r for r in reviews
                    if r.get("comment_depth", 0) == 0  # posts only
                    and (r.get("raw_metadata") or {}).get("employment_claim")
                    and (r.get("raw_metadata") or {}).get("candidate_score", 0) >= 5.0
                ]
                # Sort by candidate score descending, cap fetches
                insider_candidates.sort(
                    key=lambda r: (r.get("raw_metadata") or {}).get("candidate_score", 0),
                    reverse=True,
                )
                author_fetches = 0
                expanded_authors: set[str] = set()
                for ic in insider_candidates:
                    if author_fetches >= _MAX_AUTHOR_FETCHES:
                        break
                    author_name = ic.get("reviewer_name", "")
                    if not author_name or author_name in expanded_authors or author_name == "[deleted]":
                        continue
                    expanded_authors.add(author_name)

                    # Fetch recent submissions by this author
                    author_url = (
                        f"https://oauth.reddit.com/user/{quote_plus(author_name)}"
                        f"/submitted?sort=new&limit=25&t=year"
                    )
                    resp = await _get(http, author_url)
                    author_fetches += 1
                    if not resp or resp.status_code != 200:
                        continue

                    author_data = resp.json()
                    vendor_mentions_in_history = 0
                    org_terms_in_history = 0
                    for child in author_data.get("data", {}).get("children", []):
                        apost = child.get("data", {})
                        atext = f"{apost.get('title', '')} {apost.get('selftext', '')[:500]}".lower()
                        if alias_pattern.search(atext):
                            vendor_mentions_in_history += 1
                        for pat, _ in _ORG_SIGNAL_PATTERNS:
                            if pat.search(atext):
                                org_terms_in_history += 1
                                break

                    # Enrich the original review with author history signal
                    ic_meta = ic.get("raw_metadata") or {}
                    ic_meta["author_vendor_history_mentions"] = vendor_mentions_in_history
                    ic_meta["author_org_history_signals"] = org_terms_in_history
                    if vendor_mentions_in_history >= 2:
                        ic_meta["insider_score"] = min(
                            10.0,
                            ic_meta.get("candidate_score", 0)
                            + vendor_mentions_in_history * 1.0
                            + org_terms_in_history * 0.5,
                        )
                    ic["raw_metadata"] = ic_meta
                    await asyncio.sleep(0.3)

                if author_fetches > 0:
                    logger.info(
                        "Reddit insider author expansion: %d authors checked for %s",
                        author_fetches, target.vendor_name,
                    )

            # ---- Enrich posts with author scores + trending ----
            trending_baseline = [
                {"reviewed_at": r.get("reviewed_at")} for r in reviews if r.get("reviewed_at")
            ]
            batch_trending = _compute_batch_trending_score(trending_baseline)

            for review in reviews:
                author = review.get("reviewer_name", "")
                author_posts = author_index.get(author, [])
                churn_score = _compute_author_churn_score(author_posts)

                meta = review.get("raw_metadata") or {}
                meta["author_churn_score"] = churn_score
                meta["author_post_count_in_batch"] = len(author_posts)

                if churn_score >= _AUTHOR_HIGH_SCORE_THRESHOLD:
                    # Don't overwrite extracted_role with churn label
                    if not review.get("reviewer_title"):
                        review["reviewer_title"] = f"Repeat Churn Signal (Score: {churn_score})"

                meta["trending_score"] = batch_trending
                review["raw_metadata"] = meta

        pages_scraped = pages_scraped_ref[0]
        logger.info(
            "Reddit authenticated scrape [%s] for %s: %d reviews, %d pages",
            profile, target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    async def _fetch_comments_authenticated(
        self,
        http: httpx.AsyncClient,
        post_id: str,
        target: ScrapeTarget,
        parent_post: dict,
        limit: int,
        min_score: int,
        seen_ids: set[str],
    ) -> tuple[list[dict], int, list[str]]:
        """Fetch top-level comments for a post. Returns (comments, pages_count, errors)."""
        url = (
            f"https://oauth.reddit.com/comments/{post_id}"
            f"?sort=best&limit={limit}&depth=2"
        )
        comments: list[dict] = []
        errors: list[str] = []

        try:
            resp = await http.get(url)
            await asyncio.sleep(0.5)

            if resp.status_code == 429:
                await asyncio.sleep(2)
                return comments, 1, errors
            if resp.status_code != 200:
                errors.append(f"comments/{post_id}: HTTP {resp.status_code}")
                return comments, 1, errors

            data = resp.json()
            # Response is [post_listing, comment_listing]
            if not isinstance(data, list) or len(data) < 2:
                return comments, 1, errors

            comment_listing = data[1]
            for child in comment_listing.get("data", {}).get("children", []):
                comment = self._parse_comment(
                    child, target, parent_post, seen_ids, depth=0, min_score=min_score,
                )
                if comment:
                    comments.append(comment)

        except Exception as exc:
            errors.append(f"comments/{post_id}: {exc}")
            logger.warning("Reddit comment fetch failed for %s: %s", post_id, exc)

        return comments, 1, errors

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    def _build_global_queries(self, vendor_name: str, profile: str = "churn") -> list[str]:
        """
        Build global (all-Reddit) search queries for a vendor.

        These search ALL of Reddit -- no subreddit restriction.
        This is the primary search strategy: find real people in general
        subreddits complaining, evaluating alternatives, sharing insider info.
        """
        if profile == "insider":
            return [t.format(vendor=vendor_name) for t in _INSIDER_QUERY_TEMPLATES]

        # Base churn intent queries -- always run
        queries = [t.format(vendor=vendor_name) for t in _CHURN_QUERIES]

        # Deep profile adds pain/pricing/support failure queries
        if profile == "deep":
            queries += [t.format(vendor=vendor_name) for t in _DEEP_QUERY_TEMPLATES]

        return queries

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_post(
        self,
        post_wrapper: dict,
        target: ScrapeTarget,
        seen_ids: set[str],
        *,
        profile: str = "churn",
        alias_pattern: re.Pattern[str] | None = None,
    ) -> dict | None:
        """Parse a single Reddit post into a review dict. Returns None if skipped."""
        post = post_wrapper.get("data", {})
        post_id = post.get("id", "")
        selftext = post.get("selftext", "")

        if not post_id:
            return None
        if post_id in seen_ids:
            return None
        if selftext in ("[removed]", "[deleted]"):
            return None
        if len(selftext) < _MIN_SELFTEXT_LEN:
            return None

        title = post.get("title", "")
        subreddit = post.get("subreddit", "")

        # Build alias pattern if not provided (public fallback path)
        if alias_pattern is None:
            aliases = _build_vendor_aliases(
                target.vendor_name,
                target.metadata.get("vendor_aliases"),
            )
            alias_pattern = _build_alias_pattern(aliases)

        # ---- Candidate scoring gate ----
        candidate_score, candidate_reasons = _score_candidate(
            title, selftext, post, alias_pattern, subreddit,
        )

        # Hard reject: no alias match at all in title or body
        title_hits = len(alias_pattern.findall(title))
        body_hits = len(alias_pattern.findall(selftext[:2000]))
        if title_hits == 0 and body_hits == 0:
            return None

        # Reject low-score candidates
        if candidate_score < _CANDIDATE_SCORE_MIN:
            return None

        seen_ids.add(post_id)

        created_utc = post.get("created_utc", 0)
        reviewed_at = (
            datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
            if created_utc
            else None
        )

        # Base source weight depends on search profile; content_type is
        # determined after insider evidence extraction below.
        if profile == "insider":
            base_source_weight = 0.6
        else:
            base_source_weight = 0.5

        # ---- Dynamic source_weight ----
        source_weight = base_source_weight

        # Subreddit prior boost
        sub_weight = _SUBREDDIT_WEIGHT.get(subreddit, _DEFAULT_SUBREDDIT_WEIGHT)
        source_weight += (sub_weight - 0.5) * 0.2  # range: -0.02 to +0.08

        # Flair boost
        flair = (post.get("link_flair_text") or "").lower()
        source_weight += _FLAIR_WEIGHT_BOOST.get(flair, 0.0)

        # Award boost
        awards = post.get("all_awardings", []) or []
        source_weight += min(len(awards) * 0.02, 0.1)

        # Cross-post boost
        crossposts = post.get("crosspost_parent_list") or []
        is_crosspost = bool(crossposts)
        crosspost_subreddits: list[str] = []
        if is_crosspost:
            source_weight += 0.15
            crosspost_subreddits = [
                cp.get("subreddit", "") for cp in crossposts if cp.get("subreddit")
            ]

        source_weight = round(min(source_weight, 1.0), 3)

        # ---- Insider evidence extraction ----
        combined_text = f"{title}\n{selftext[:3000]}"
        insider_evidence = _extract_insider_evidence(combined_text)

        # Content type: only classify as insider_account when evidence supports it.
        # Employment claim or 2+ org signals = insider. Otherwise community_discussion.
        if insider_evidence["employment_claim"] or len(insider_evidence["org_signal_types"]) >= 2:
            content_type = "insider_account"
        else:
            content_type = "community_discussion"

        # Promote reviewer_title if we extracted a role or employment claim
        reviewer_title = None
        if insider_evidence["extracted_role"]:
            reviewer_title = insider_evidence["extracted_role"]

        # ---- Edit history ----
        edited = post.get("edited")
        is_edited = bool(edited and edited is not False)
        edit_timestamp: str | None = None
        if is_edited and isinstance(edited, (int, float)):
            edit_timestamp = datetime.fromtimestamp(edited, tz=timezone.utc).isoformat()

        # Reddit fullname used as thread_id (e.g. "t3_abc123")
        fullname = post.get("name", f"t3_{post_id}")

        # Author flair (often contains role/company info in B2B subs)
        author_flair = post.get("author_flair_text") or ""

        return {
            "source": "reddit",
            "source_url": f"https://www.reddit.com{post.get('permalink', '')}",
            "source_review_id": post_id,
            "vendor_name": target.vendor_name,
            "product_name": target.product_name,
            "product_category": target.product_category,
            "rating": None,
            "rating_max": 5,
            "summary": post.get("title", "")[:500],
            "review_text": selftext[:10000],
            "pros": None,
            "cons": None,
            "reviewer_name": post.get("author", ""),
            "reviewer_title": reviewer_title,
            "reviewer_company": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "reviewed_at": reviewed_at,
            # Threading fields
            "content_type": content_type,
            "parent_review_id": None,
            "thread_id": fullname,
            "comment_depth": 0,
            "raw_metadata": {
                "extraction_method": "api_json",
                "source_weight": source_weight,
                "source_type": content_type,
                "search_profile": profile,
                "subreddit": subreddit,
                "subreddit_weight": sub_weight,
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
                # Vendor match metadata
                "vendor_in_title": title_hits > 0,
                "vendor_mention_count": title_hits + body_hits,
                "candidate_score": candidate_score,
                "candidate_reason": "; ".join(candidate_reasons),
                # Insider evidence
                "employment_claim": insider_evidence["employment_claim"],
                "employment_tense": insider_evidence["employment_tense"],
                "org_signal_types": insider_evidence["org_signal_types"],
                # Flair
                "post_flair": post.get("link_flair_text") or "",
                "author_flair_text": author_flair,
                # Awards
                "award_count": len(awards),
                # Edit tracking
                "is_edited": is_edited,
                "edit_timestamp": edit_timestamp,
                # Cross-post
                "is_crosspost": is_crosspost,
                "crosspost_subreddits": crosspost_subreddits,
                # Enriched in post-processing:
                "author_churn_score": 0.0,
                "author_post_count_in_batch": 0,
                "trending_score": "low",
            },
        }

    def _parse_comment(
        self,
        child: dict,
        target: ScrapeTarget,
        parent_post: dict,
        seen_ids: set[str],
        *,
        depth: int,
        min_score: int,
    ) -> dict | None:
        """Parse a Reddit comment into a review dict. Returns None if skipped."""
        if child.get("kind") != "t1":
            return None

        data = child.get("data", {})
        comment_id = data.get("id", "")
        body = data.get("body", "")

        if not comment_id:
            return None
        if comment_id in seen_ids:
            return None
        if body in ("[removed]", "[deleted]", ""):
            return None
        if len(body) < _MIN_COMMENT_LEN:
            return None

        score = data.get("score", 0)
        if isinstance(score, (int, float)) and score < min_score:
            return None

        seen_ids.add(comment_id)

        created_utc = data.get("created_utc", 0)
        reviewed_at = (
            datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
            if created_utc
            else None
        )

        # Store the parent post's source_review_id so b2b_scrape_intake can resolve
        # it to a UUID after the parent post is inserted.
        parent_source_review_id = parent_post.get("source_review_id")
        thread_id = parent_post.get("thread_id")

        # Insider evidence from comment body
        insider_evidence = _extract_insider_evidence(body[:3000])
        reviewer_title = insider_evidence["extracted_role"]

        # Comments with employment claims or strong org signals get insider_account
        # so relevance scoring applies insider boosts instead of noise penalties.
        if insider_evidence["employment_claim"] or len(insider_evidence["org_signal_types"]) >= 2:
            comment_content_type = "insider_account"
        else:
            comment_content_type = "comment"

        subreddit = data.get("subreddit", "")
        sub_weight = _SUBREDDIT_WEIGHT.get(subreddit, _DEFAULT_SUBREDDIT_WEIGHT)
        author_flair = data.get("author_flair_text") or ""

        # Comment source_weight: base 0.4, with subreddit prior adjustment
        comment_source_weight = round(0.4 + (sub_weight - 0.5) * 0.1, 3)

        return {
            "source": "reddit",
            "source_url": (
                f"https://www.reddit.com{data.get('permalink', '')}"
                if data.get("permalink")
                else parent_post.get("source_url", "")
            ),
            "source_review_id": f"t1_{comment_id}",
            "vendor_name": target.vendor_name,
            "product_name": target.product_name,
            "product_category": target.product_category,
            "rating": None,
            "rating_max": 5,
            "summary": None,
            "review_text": body[:5000],
            "pros": None,
            "cons": None,
            "reviewer_name": data.get("author", ""),
            "reviewer_title": reviewer_title,
            "reviewer_company": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "reviewed_at": reviewed_at,
            # Threading fields
            "content_type": comment_content_type,
            "parent_review_id": None,  # resolved post-insert by b2b_scrape_intake
            "thread_id": thread_id,
            "comment_depth": depth + 1,
            "raw_metadata": {
                "extraction_method": "api_json",
                "source_weight": comment_source_weight,
                "source_type": "comment",
                "search_profile": parent_post.get("raw_metadata", {}).get("search_profile", "churn"),
                "subreddit": subreddit,
                "subreddit_weight": sub_weight,
                "score": score,
                "upvote_ratio": None,
                "num_comments": None,
                # Insider evidence
                "employment_claim": insider_evidence["employment_claim"],
                "employment_tense": insider_evidence["employment_tense"],
                "org_signal_types": insider_evidence["org_signal_types"],
                "author_flair_text": author_flair,
                # For parent_review_id resolution after insert
                "parent_source_review_id": parent_source_review_id,
                # Enriched in post-processing
                "author_churn_score": 0.0,
                "author_post_count_in_batch": 0,
                "trending_score": "low",
            },
        }

    # ------------------------------------------------------------------
    # Public (unauthenticated) fallback
    # ------------------------------------------------------------------

    async def _scrape_public(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Fallback: scrape via public JSON endpoints (no auth). Posts only."""
        profile = (target.metadata.get("search_profile") or "churn").lower()

        if profile == "insider":
            subreddits = target.metadata.get("subreddits") or _INSIDER_SUBREDDITS
        else:
            subreddits = target.metadata.get("subreddits") or _DEFAULT_SUBREDDITS

        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(",")]

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        # Public mode: exact quoted vendor name per subreddit
        # to keep request volume tight (10 req/min limit).
        vendor_encoded = quote_plus(f'"{target.vendor_name}"')

        for sub in subreddits[:target.max_pages]:
            sub_encoded = quote_plus(sub)
            url = (
                f"https://www.reddit.com/search.json"
                f"?q={vendor_encoded}+subreddit:{sub_encoded}&sort=new&limit=25&t=year"
            )

            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=f"https://www.reddit.com/r/{sub}/",
                    sticky_session=False,
                    prefer_residential=False,
                )
                pages_scraped += 1

                if resp.status_code in (403, 429):
                    await asyncio.sleep(3)
                    fallback_url = (
                        f"https://old.reddit.com/r/{sub}/search.json"
                        f"?q={vendor_encoded}&sort=new&limit=25&t=year&restrict_sr=on"
                    )
                    resp = await client.get(
                        fallback_url,
                        domain="old.reddit.com",
                        referer=f"https://old.reddit.com/r/{sub}/",
                        sticky_session=False,
                        prefer_residential=False,
                    )

                if resp.status_code != 200:
                    errors.append(f"r/{sub}: HTTP {resp.status_code}")
                    continue

                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    errors.append(f"r/{sub}: non-JSON response ({ct[:40]})")
                    continue

                try:
                    data = resp.json()
                except (ValueError, TypeError):
                    errors.append(f"r/{sub}: non-parseable JSON body")
                    continue

                for post_wrapper in data.get("data", {}).get("children", []):
                    post_dict = self._parse_post(post_wrapper, target, seen_ids, profile=profile)
                    if post_dict:
                        reviews.append(post_dict)

            except Exception as exc:
                errors.append(f"r/{sub}: {exc}")
                logger.warning("Reddit scrape failed for r/%s: %s", sub, exc)

        # Global intent searches in public mode — top 2 only to stay under rate limit
        for query in self._build_global_queries(target.vendor_name, profile)[:2]:
            url = (
                f"https://www.reddit.com/search.json"
                f"?q={quote_plus(query)}&sort=new&limit=25&t=year"
            )
            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer="https://www.reddit.com/search",
                    sticky_session=False,
                    prefer_residential=False,
                )
                pages_scraped += 1
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except (ValueError, TypeError):
                        continue
                    for post_wrapper in data.get("data", {}).get("children", []):
                        post_dict = self._parse_post(post_wrapper, target, seen_ids, profile=profile)
                        if post_dict:
                            reviews.append(post_dict)
                await asyncio.sleep(1.5)
            except Exception as exc:
                errors.append(f"public global q={query}: {exc}")

        logger.info(
            "Reddit public scrape [%s] for %s: %d reviews from %d subreddits",
            profile, target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(RedditParser())
