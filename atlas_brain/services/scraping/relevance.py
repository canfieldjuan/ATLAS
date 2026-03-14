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
