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
    (re.compile(r"switch(?:ed|ing)\s+(?:from|to)", re.I), 0.15),
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
]

_CHURN_CAP = 0.35
_NOISE_CAP = -0.35

# Structured review sources that bypass the filter entirely
STRUCTURED_SOURCES = frozenset({"g2", "capterra", "trustradius"})


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

def score_relevance(review: dict[str, Any], vendor_name: str) -> tuple[float, str]:
    """Score a review's relevance for B2B churn intelligence.

    Returns (score 0.0-1.0, human-readable reason string).
    """
    score = 0.5
    reasons: list[str] = []

    title = (review.get("summary") or review.get("title") or "").strip()
    body = (review.get("review_text") or "").strip()
    text = f"{title}\n{body}"
    meta = review.get("raw_metadata") or {}

    # --- Signal 1: Churn / review language ---
    churn_boost = 0.0
    for pat, weight in _CHURN_PATTERNS:
        if pat.search(text):
            churn_boost += weight
    churn_boost = min(churn_boost, _CHURN_CAP)
    if churn_boost > 0:
        score += churn_boost
        reasons.append(f"churn_language=+{churn_boost:.2f}")

    # --- Signal 2: Noise patterns ---
    noise_penalty = 0.0
    for pat, weight in _NOISE_PATTERNS:
        if pat.search(text):
            noise_penalty += weight
    noise_penalty = max(noise_penalty, _NOISE_CAP)
    if noise_penalty < 0:
        score += noise_penalty
        reasons.append(f"noise_language={noise_penalty:.2f}")

    # --- Signal 3: Vendor name prominence ---
    vendor_lower = vendor_name.lower()
    if vendor_lower in title.lower():
        score += 0.10
        reasons.append("vendor_in_title=+0.10")

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

    elif source == "hackernews":
        points = meta.get("points", meta.get("score", 0))
        if isinstance(points, (int, float)) and points < 8:
            score -= 0.10
            reasons.append("low_hn_points=-0.10")

    elif source == "github":
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
        if len(body) < 400:
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
        rel_score, reason = score_relevance(review, vendor_name)

        if rel_score >= threshold:
            meta = review.get("raw_metadata") or {}
            meta["relevance_score"] = rel_score
            meta["relevance_reason"] = reason
            review["raw_metadata"] = meta
            kept.append(review)
        else:
            filtered += 1

    return kept, filtered
