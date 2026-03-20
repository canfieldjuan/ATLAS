"""
Rule-based relevance scorer for B2B review content.

Filters noise from social media sources (Reddit, HN, GitHub, RSS) that
keyword-match vendor names but aren't actual product reviews or churn signals.
Pure regex + heuristics, no LLM call, runs in <1ms per review.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

_CHURN_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    # Strong churn / evaluation signals (+0.15)
    (re.compile(r"(?:we|I|our\s+(?:team|company))\s+switch(?:ed|ing)\s+(?:from|to)", re.I), 0.15),
    (re.compile(r"switch(?:ed|ing)\s+(?:from|to)", re.I), 0.05),
    (re.compile(r"migrat(?:ed|ing)\s+away", re.I), 0.15),
    (re.compile(r"looking\s+for\s+alternative", re.I), 0.15),
    (re.compile(r"not\s+renewing", re.I), 0.15),
    (re.compile(r"cancel(?:ed|led|ling)", re.I), 0.10),
    (re.compile(r"ditched|dumped|dropped", re.I), 0.10),
    # Comparative language (+0.10)
    (re.compile(r"compared\s+to", re.I), 0.10),
    (re.compile(r"better\s+than|worse\s+than", re.I), 0.10),
    (re.compile(r"\bvs\b", re.I), 0.05),
    # Experience language (+0.10)
    (re.compile(r"my\s+experience\s+with", re.I), 0.10),
    (re.compile(r"after\s+\d+\s+years?\s+of\s+using", re.I), 0.10),
    (re.compile(r"(?:we|I)\s+(?:use|used)\b", re.I), 0.05),
    # Review language (+0.05)
    (re.compile(r"pros?\s+and\s+cons?", re.I), 0.05),
    (re.compile(r"what\s+I\s+(?:like|hate|dislike)", re.I), 0.05),
    (re.compile(r"frustrated\s+with", re.I), 0.05),
    (re.compile(r"too\s+expensive", re.I), 0.05),
    (re.compile(r"poor\s+support", re.I), 0.05),
]

_NOISE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    # Finance / corporate news (-0.15)
    (re.compile(r"stock\s+price|earnings|market\s+cap|quarterly\s+results?", re.I), -0.15),
    (re.compile(r"\bIPO\b"), -0.15),
    (re.compile(r"acquisition|merger\b", re.I), -0.10),
    # HR / jobs (-0.10)
    (re.compile(r"laid\s+off|layoffs|headcount", re.I), -0.10),
    (re.compile(r"hiring|job\s+opening|salary|we(?:'re| are)\s+looking\s+for", re.I), -0.10),
    # Dev tooling (-0.10)
    (re.compile(r"npm\s+install|pip\s+install", re.I), -0.10),
    (re.compile(r"API\s+documentation|SDK\b", re.I), -0.05),
    # Tutorial / how-to (-0.05)
    (re.compile(r"tutorial|how\s+to\s+(?:set\s+up|install|configure)", re.I), -0.05),
    # Press / announcements (-0.10)
    (re.compile(r"press\s+release|according\s+to|spokesperson", re.I), -0.10),
    (re.compile(r"announced|partnership\b", re.I), -0.05),
    # Career / jobs (not product evaluation)
    (re.compile(r"career|job\s+(?:market|safe|opening|hunt)|interview\s+(?:prep|question)", re.I), -0.10),
    (re.compile(r"resume|linkedin|recruiter|are\s+\w+\s+jobs?\s+safe", re.I), -0.10),
    # Executive / corporate news (not customer experience)
    (re.compile(r"\bCEO\b|\bCFO\b|\bCTO\b|executive|leadership\s+churn", re.I), -0.10),
    (re.compile(r"revenue|billion|workforce|lays?\s+off|headcount\s+reduction", re.I), -0.10),
    # Bug report / issue template (developer noise)
    (re.compile(r"steps?\s+to\s+reproduce|expected\s+(?:result|behavior)|actual\s+(?:result|behavior)", re.I), -0.15),
    (re.compile(r"stack\s+trace|traceback|exception|error\s+code|status\s+code", re.I), -0.10),
    (re.compile(r"\bCLI\b|command\s+line|terraform|provider\s+version", re.I), -0.10),
    # IT admin / infrastructure (not product evaluation)
    (re.compile(r"\bSSO\b|\bSAML\b|\bMFA\b|oauth|single\s+sign[- ]on", re.I), -0.10),
    (re.compile(r"\bDMARC\b|\bSPF\b|\bDKIM\b|dns\s+record", re.I), -0.10),
]

_CHURN_CAP = 0.35
_NOISE_CAP = -0.45

# Signal patterns that are *noise* for normal reviews but *signal* for insider accounts.
# These override the noise penalties when content_type == "insider_account".
_INSIDER_BOOST_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    # Talent drain / org health signals
    (re.compile(r"laid\s+off|layoffs|headcount\s+reduction", re.I), 0.10),
    (re.compile(r"\bCEO\b|\bCFO\b|\bCTO\b|executive|leadership\s+churn", re.I), 0.05),
    (re.compile(r"revenue\s+decline|billion\s+(?:in\s+)?loss|workforce\s+cut", re.I), 0.05),
    # Culture / morale indicators
    (re.compile(r"\b(?:toxic|terrible|awful)\s+culture", re.I), 0.15),
    (re.compile(r"\bmorale\b", re.I), 0.10),
    (re.compile(r"micromanagement|bureaucracy|red\s+tape", re.I), 0.10),
    (re.compile(r"dead[\s-]end|no\s+growth|going\s+nowhere", re.I), 0.10),
    (re.compile(r"brain\s+drain|talent\s+exodus|everyone\s+(?:is\s+)?leaving", re.I), 0.15),
    (re.compile(r"product\s+(?:quality\s+)?(?:is\s+)?(?:declining|getting\s+worse)", re.I), 0.10),
    (re.compile(r"engineering\s+(?:culture|morale|quality)", re.I), 0.10),
]

_REDDIT_MATCH_THREAD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"post[- ]match\s+(?:thread|discussion)|match\s+(?:thread|discussion)", re.I),
    re.compile(r"grand\s+final|lower\s+final|upper\s+final|lower\s+bracket|upper\s+bracket", re.I),
    re.compile(r"main\s+event|group\s+stage|swiss\s+stage|playoffs?|best\s+of\s+[1357]|\bmap\s+[1-5]\b", re.I),
    re.compile(r"esports|roster|valorant|league\s+of\s+legends|\bLCS\b|\bLTA\b|leaguepedia|liquipedia", re.I),
    re.compile(r"lolesports|week\s+\d+|split\s+\d+|season\s+(?:spring|summer|winter)", re.I),
]
_REDDIT_MATCH_THREAD_MIN_HITS = 2
_REDDIT_MATCH_THREAD_PENALTY = -0.25
_REDDIT_AGGREGATOR_SUBREDDITS = frozenset({"autotldr", "buzzfeedbot", "pwnhub", "symbynews"})
_REDDIT_INVESTOR_SUBREDDITS = frozenset({"amzn", "investing", "stocks", "wallstreetbets"})
_REDDIT_LOW_SIGNAL_SUBREDDITS = frozenset({
    "addons4kodi",
    "amazonprime",
    "anticonsumption",
    "conspiracy",
    "newworldgame",
})
_REDDIT_CAREER_SUBREDDITS = frozenset({
    "careerguidance",
    "cscareerquestions",
    "csmajors",
    "developersindia",
    "experienceddevs",
    "itcareerquestions",
    "jobs",
})
_REDDIT_BUILDER_SUBREDDITS = frozenset({
    "alphaandbetausers",
    "buildinpublic",
    "sideproject",
})
_REDDIT_USER_PROFILE_SUBREDDIT_RE = re.compile(r"^u[_/][a-z0-9][a-z0-9_-]{1,}$", re.I)
_REDDIT_BUILDER_PROMO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:i|we|my\s+team)(?:\W{0,3}(?:have|ve))?\s+(?:built|made|created|launched|released)\b", re.I),
    re.compile(r"\b(?:i|we)\W{0,3}(?:am|m|are|re)?\s*(?:working on|building|developing)\b", re.I),
    re.compile(r"\bthinking\s+of\s+building\b", re.I),
    re.compile(r"\blooking\s+for\s+(?:feedback|early\s+users?|testers?|honest\s+outside\s+opinions)\b", re.I),
    re.compile(r"\b(?:got|get|getting)\s+\d+(?:,\d{3})*\s+(?:unique\s+)?(?:users?|visitors?|signups?|demo\s+requests?)\b", re.I),
    re.compile(r"\bfirst\s+\d+(?:,\d{3})*\s+users?\b", re.I),
    re.compile(r"\b(?:alpha|beta)\s+users?\b", re.I),
    re.compile(r"\b(?:beta\s+testing|out\s+of\s+beta)\b", re.I),
    re.compile(r"\broast\s+my\s+landing\s+page\b", re.I),
    re.compile(r"\b(?:would|wouldn't|i'?d)\s+love\s+(?:honest\s+|some\s+|your\s+|outside\s+)?(?:feedback|opinions)\b", re.I),
    re.compile(r"\b(?:i'?m|i am)\s+(?:looking\s+into|trying\s+to\s+understand)\b", re.I),
    re.compile(r"\bi\s+have\s+a\s+hypothesis\b", re.I),
    re.compile(r"\b(?:week|month)\s+of\s+building\s+in\s+public\b", re.I),
    re.compile(r"\bbuilding\s+in\s+public\b", re.I),
    re.compile(r"\bSEO\s+takes\s+forever\b", re.I),
    re.compile(r"\bpaid\s+ads?\s+burned\s+through\s+my\s+budget\b", re.I),
    re.compile(r"\bmessage\s+me\b", re.I),
    re.compile(r"\bfeedback\s+welcome\b", re.I),
    re.compile(r"\bfull\s+disclosure\b", re.I),
    re.compile(r"\b(?:just\s+launched|launching|lifetime\s+access)\b", re.I),
]
_REDDIT_AGGREGATOR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"this is an automatic summary|this is the best tl;dr i could make", re.I),
    re.compile(r"coverage includes:|first of \d+ articles?|first of \d+ article", re.I),
    re.compile(r"\[original\]\(|reduced by \d+%|\(i'?m a bot\)", re.I),
]
_REDDIT_INVESTOR_NEWS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"will step down|the company announced|boosting its stake|signed a deal", re.I),
    re.compile(r"earnings|stock|investing|investment|billion|quarterly|ceo|cfo|cto", re.I),
    re.compile(r"works on issue|outage explained|transparency report|client message", re.I),
]
_REDDIT_AGGREGATOR_PENALTY = -0.35
_REDDIT_INVESTOR_NEWS_PENALTY = -0.30
_REDDIT_LOW_SIGNAL_SUBREDDIT_PENALTY = -0.30
_REDDIT_CAREER_SUBREDDIT_PENALTY = -0.25
_REDDIT_BUILDER_SELF_PROMO_PENALTY = -0.50
_REDDIT_USER_PROFILE_SUBREDDIT_PENALTY = -0.40
_REDDIT_CAREER_KEEP_CHURN_MIN = 0.15

# Structured review sources that bypass the filter entirely
from .sources import STRUCTURED_SOURCES


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

def score_relevance(
    review: dict[str, Any],
    vendor_name: str,
    content_type: str | None = None,
) -> tuple[float, str]:
    """Score a review's relevance for B2B churn intelligence.

    content_type: pass explicitly to override review["content_type"] lookup.
    Returns (score 0.0-1.0, human-readable reason string).
    """
    score = 0.5
    reasons: list[str] = []

    title = (review.get("summary") or review.get("title") or "").strip()
    body = (review.get("review_text") or "").strip()
    text = f"{title}\n{body}"
    meta = review.get("raw_metadata") or {}

    # Resolve content type (explicit arg wins, then field, then default)
    ctype = content_type or review.get("content_type") or "review"
    is_insider = ctype == "insider_account"

    # --- Signal 1: Churn / review language ---
    churn_boost = 0.0
    for pat, weight in _CHURN_PATTERNS:
        if churn_boost >= _CHURN_CAP:
            break
        if pat.search(text):
            churn_boost += weight
    churn_boost = min(churn_boost, _CHURN_CAP)
    if churn_boost > 0:
        score += churn_boost
        reasons.append(f"churn_language=+{churn_boost:.2f}")

    # --- Signal 2a: Noise patterns (skipped / inverted for insider accounts) ---
    if is_insider:
        # For insider accounts the noise patterns are irrelevant or counter-
        # productive (e.g. "layoffs" is signal, not noise). Apply insider boosts
        # instead and skip standard noise scoring.
        insider_boost = 0.0
        for pat, weight in _INSIDER_BOOST_PATTERNS:
            if pat.search(text):
                insider_boost += weight
        insider_boost = min(insider_boost, 0.40)
        if insider_boost > 0:
            score += insider_boost
            reasons.append(f"insider_signals=+{insider_boost:.2f}")

        # Reddit-specific insider metadata boosts (from parser evidence extraction)
        if meta.get("employment_claim"):
            emp_tense = meta.get("employment_tense", "")
            emp_boost = 0.10 if emp_tense == "past" else 0.05
            score += emp_boost
            reasons.append(f"employment_claim({emp_tense})=+{emp_boost:.2f}")

        org_signals = meta.get("org_signal_types") or []
        if org_signals:
            org_boost = min(len(org_signals) * 0.05, 0.15)
            score += org_boost
            reasons.append(f"org_signals({','.join(org_signals)})=+{org_boost:.2f}")

        # Author expansion insider_score from author history fetch
        insider_score_val = meta.get("insider_score")
        if isinstance(insider_score_val, (int, float)) and insider_score_val >= 6.0:
            score += 0.10
            reasons.append(f"insider_author_score({insider_score_val:.1f})=+0.10")
    else:
        # --- Signal 2b: Standard noise patterns ---
        noise_penalty = 0.0
        for pat, weight in _NOISE_PATTERNS:
            if noise_penalty <= _NOISE_CAP:
                break
            if pat.search(text):
                noise_penalty += weight
        noise_penalty = max(noise_penalty, _NOISE_CAP)
        if noise_penalty < 0:
            score += noise_penalty
            reasons.append(f"noise_language={noise_penalty:.2f}")

    # --- Signal 3: Vendor name prominence ---
    # Use pre-computed metadata from Reddit parser when available
    vendor_in_title = meta.get("vendor_in_title")
    vendor_mention_count = meta.get("vendor_mention_count")

    if vendor_in_title is not None:
        # Use parser-provided metadata (alias-aware)
        if vendor_in_title:
            score += 0.05
            reasons.append("vendor_in_title=+0.05")
        mention_count = vendor_mention_count or 0
        if mention_count >= 3:
            score += 0.05
            reasons.append(f"vendor_mentions={mention_count}=+0.05")
        elif mention_count <= 1 and len(body) > 500:
            score -= 0.15
            reasons.append("tangential_mention=-0.15")
    else:
        # Fallback: compute from text (non-Reddit sources)
        vendor_lower = vendor_name.lower()
        if vendor_lower in title.lower():
            score += 0.05
            reasons.append("vendor_in_title=+0.05")

        body_mentions = len(re.findall(re.escape(vendor_lower), body.lower()))
        if body_mentions >= 3:
            score += 0.05
            reasons.append(f"vendor_mentions={body_mentions}=+0.05")
        elif body_mentions <= 1 and len(body) > 500:
            score -= 0.15
            reasons.append("tangential_mention=-0.15")

    # --- Signal 4: Source-specific engagement quality ---
    source = (review.get("source") or "").lower()

    if source == "reddit":
        subreddit_lower = str(meta.get("subreddit") or "").strip().lower()
        reddit_score = meta.get("score", meta.get("ups", 0))
        upvote_ratio = meta.get("upvote_ratio", 1.0)
        if (isinstance(reddit_score, (int, float)) and reddit_score < 5) or \
           (isinstance(upvote_ratio, (int, float)) and upvote_ratio < 0.55):
            score -= 0.10
            reasons.append("low_reddit_engagement=-0.10")

        # --- Signal 4a: Reddit-specific metadata (candidate score, subreddit, insider) ---
        candidate_score = meta.get("candidate_score")
        if candidate_score is not None:
            # High candidate score from parser = confirmed relevance
            if candidate_score >= 7.0:
                score += 0.10
                reasons.append(f"high_candidate_score({candidate_score})=+0.10")
            elif candidate_score >= 5.0:
                score += 0.05
                reasons.append(f"good_candidate_score({candidate_score})=+0.05")

        # Subreddit weight from parser
        sub_weight = meta.get("subreddit_weight")
        if isinstance(sub_weight, (int, float)) and sub_weight >= 0.8:
            score += 0.05
            reasons.append(f"high_signal_subreddit=+0.05")

        # Employment claim boost (insider path handles separately below)
        if not is_insider and meta.get("employment_claim"):
            score += 0.05
            reasons.append("employment_claim=+0.05")

        # Job-hunt / interview noise penalty for Reddit
        org_signals = meta.get("org_signal_types") or []
        if not is_insider and not org_signals:
            # Check for job-seeker noise patterns specific to Reddit
            _job_noise_re = re.compile(
                r"\b(?:interview(?:ed|ing)?|job\s+hunt|salary|recruiter|leetcode|onsite)\b",
                re.I,
            )
            job_noise_hits = len(_job_noise_re.findall(text[:1000]))
            if job_noise_hits >= 2:
                score -= 0.10
                reasons.append(f"reddit_job_noise({job_noise_hits})=-0.10")

        match_thread_hits = sum(
            1 for pat in _REDDIT_MATCH_THREAD_PATTERNS if pat.search(text[:1500])
        )
        if match_thread_hits >= _REDDIT_MATCH_THREAD_MIN_HITS:
            score += _REDDIT_MATCH_THREAD_PENALTY
            reasons.append(
                f"reddit_match_thread_noise({match_thread_hits})={_REDDIT_MATCH_THREAD_PENALTY:.2f}"
            )

        aggregator_hits = sum(
            1 for pat in _REDDIT_AGGREGATOR_PATTERNS if pat.search(text[:1500])
        )
        if churn_boost < 0.10 and (
            subreddit_lower in _REDDIT_AGGREGATOR_SUBREDDITS or aggregator_hits > 0
        ):
            score += _REDDIT_AGGREGATOR_PENALTY
            reasons.append(
                f"reddit_aggregator_noise({aggregator_hits or 1})={_REDDIT_AGGREGATOR_PENALTY:.2f}"
            )

        investor_hits = sum(
            1 for pat in _REDDIT_INVESTOR_NEWS_PATTERNS if pat.search(text[:1500])
        )
        if churn_boost < 0.10 and (
            subreddit_lower in _REDDIT_INVESTOR_SUBREDDITS or investor_hits >= 2
        ):
            score += _REDDIT_INVESTOR_NEWS_PENALTY
            reasons.append(
                f"reddit_investor_news({investor_hits or 1})={_REDDIT_INVESTOR_NEWS_PENALTY:.2f}"
            )

        if subreddit_lower in _REDDIT_LOW_SIGNAL_SUBREDDITS:
            score += _REDDIT_LOW_SIGNAL_SUBREDDIT_PENALTY
            reasons.append(
                f"reddit_low_signal_subreddit({subreddit_lower})={_REDDIT_LOW_SIGNAL_SUBREDDIT_PENALTY:.2f}"
            )

        if _REDDIT_USER_PROFILE_SUBREDDIT_RE.match(subreddit_lower):
            score += _REDDIT_USER_PROFILE_SUBREDDIT_PENALTY
            reasons.append(
                f"reddit_user_profile_subreddit({subreddit_lower})={_REDDIT_USER_PROFILE_SUBREDDIT_PENALTY:.2f}"
            )

        if (
            subreddit_lower in _REDDIT_CAREER_SUBREDDITS
            and not org_signals
            and not (churn_boost >= _REDDIT_CAREER_KEEP_CHURN_MIN and not meta.get("employment_claim"))
        ):
            score += _REDDIT_CAREER_SUBREDDIT_PENALTY
            reasons.append(
                f"reddit_career_subreddit_noise({subreddit_lower})={_REDDIT_CAREER_SUBREDDIT_PENALTY:.2f}"
            )

        builder_hits = sum(
            1 for pat in _REDDIT_BUILDER_PROMO_PATTERNS if pat.search(text[:1500])
        )
        if (
            not org_signals
            and subreddit_lower in _REDDIT_BUILDER_SUBREDDITS
            and builder_hits > 0
        ):
            score += _REDDIT_BUILDER_SELF_PROMO_PENALTY
            reasons.append(
                f"reddit_builder_self_promo({subreddit_lower},{builder_hits})={_REDDIT_BUILDER_SELF_PROMO_PENALTY:.2f}"
            )

    elif source == "hackernews":
        points = meta.get("points", meta.get("score", 0))
        if isinstance(points, (int, float)) and points < 8:
            score -= 0.10
            reasons.append("low_hn_points=-0.10")

    elif source == "github":
        # Bug report template detection
        template_markers = ("steps to reproduce", "expected result", "expected behavior",
                            "actual result", "actual behavior", "affected resource",
                            "terraform core version", "provider version")
        text_lower = text.lower()
        if any(marker in text_lower for marker in template_markers):
            score -= 0.20
            reasons.append("github_issue_template=-0.20")
        # Distinguish issues vs repos
        stars = meta.get("stars", meta.get("stargazers_count"))
        reactions = meta.get("reactions", meta.get("reaction_count"))
        if stars is not None:
            if isinstance(stars, (int, float)) and stars < 25:
                score -= 0.10
                reasons.append("low_github_stars=-0.10")
        elif reactions is not None:
            if isinstance(reactions, (int, float)) and reactions < 3:
                score -= 0.05
                reasons.append("low_github_reactions=-0.05")

    elif source == "rss":
        if len(body) < 100:
            score -= 0.25
            reasons.append("short_rss_content=-0.25")
        elif len(body) < 200:
            score -= 0.15
            reasons.append("short_rss_content=-0.15")
        elif len(body) < 400:
            score -= 0.05
            reasons.append("short_rss_content=-0.05")

    # Clamp
    score = max(0.0, min(1.0, score))

    reason_str = "; ".join(reasons) if reasons else "baseline"
    return round(score, 3), reason_str


# ---------------------------------------------------------------------------
# Batch filter
# ---------------------------------------------------------------------------

def filter_reviews(
    reviews: list[dict[str, Any]],
    vendor_name: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    """Filter a list of reviews by relevance score.

    Returns (kept_reviews, filtered_count).
    Injects ``relevance_score`` and ``relevance_reason`` into each kept
    review's ``raw_metadata``.
    """
    kept: list[dict[str, Any]] = []
    filtered = 0

    for review in reviews:
        # Pass content_type so insider accounts use the appropriate scoring path
        rel_score, reason = score_relevance(review, vendor_name, review.get("content_type"))

        if rel_score >= threshold:
            meta = review.get("raw_metadata") or {}
            meta["relevance_score"] = rel_score
            meta["relevance_reason"] = reason
            review["raw_metadata"] = meta
            kept.append(review)
        else:
            filtered += 1

    return kept, filtered
