from __future__ import annotations

import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "cleanup_b2b_noise",
    Path("scripts/cleanup_b2b_noise.py"),
)
cleanup_b2b_noise = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(cleanup_b2b_noise)


def test_review_dict_backfills_reddit_subreddit():
    row = {
        "source": "reddit",
        "summary": "AWS outage roundup",
        "review_text": "A summary of multiple service disruptions.",
        "content_type": "community_discussion",
        "reddit_subreddit": "pwnhub",
    }

    review = cleanup_b2b_noise._review_dict_from_row(row, {})

    assert review["raw_metadata"]["subreddit"] == "pwnhub"


def test_review_dict_preserves_existing_reddit_subreddit():
    row = {
        "source": "reddit",
        "summary": "AWS outage roundup",
        "review_text": "A summary of multiple service disruptions.",
        "content_type": "community_discussion",
        "reddit_subreddit": "pwnhub",
    }

    review = cleanup_b2b_noise._review_dict_from_row(row, {"subreddit": "wallstreetbets"})

    assert review["raw_metadata"]["subreddit"] == "wallstreetbets"


def test_should_honor_protection_blocks_hard_reddit_noise():
    assert not cleanup_b2b_noise._should_honor_protection(
        "reddit",
        "reddit_aggregator_noise(2)=-0.35",
    )
    assert not cleanup_b2b_noise._should_honor_protection(
        "reddit",
        "reddit_low_signal_subreddit(newworldgame)=-0.30",
    )
    assert not cleanup_b2b_noise._should_honor_protection(
        "reddit",
        "reddit_user_profile_subreddit(u_cart-to-cart)=-0.40",
    )
    assert not cleanup_b2b_noise._should_honor_protection(
        "reddit",
        "reddit_career_subreddit_noise(cscareerquestions)=-0.25",
    )
    assert not cleanup_b2b_noise._should_honor_protection(
        "reddit",
        "reddit_builder_self_promo(sideproject,2)=-0.25",
    )


def test_should_honor_protection_keeps_non_noise_paths():
    assert cleanup_b2b_noise._should_honor_protection(
        "reddit",
        "vendor_in_title=+0.05; good_candidate_score(6.0)=+0.05",
    )
    assert cleanup_b2b_noise._should_honor_protection(
        "github",
        "github_issue_template=-0.20",
    )
