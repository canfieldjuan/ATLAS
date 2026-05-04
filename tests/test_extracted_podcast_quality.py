from __future__ import annotations

from extracted_content_pipeline.services.podcast_quality import (
    podcast_quality_revalidation,
)


def _make(body: str, format_type: str, *, title: str = "t", metadata=None, idea=None):
    return podcast_quality_revalidation(
        draft={"title": title, "body": body, "metadata": metadata or {}},
        format_type=format_type,
        idea=idea,
    )


def test_newsletter_word_count_band_passes_in_band() -> None:
    body = "word " * 800
    result = _make(body, "newsletter")
    assert result["audit"]["status"] == "pass"


def test_newsletter_blocks_when_far_under_band() -> None:
    body = "word " * 50  # well under 500 lower bound
    result = _make(body, "newsletter")
    assert "length_under_band" in result["audit"]["blocking_issues"]


def test_blog_requires_h1() -> None:
    body = "Body without an h1 heading. " * 300  # in word band but no #
    result = _make(body, "blog")
    assert "missing_h1" in result["audit"]["blocking_issues"]


def test_blog_with_h1_passes() -> None:
    body = "# Title\n\n" + ("Paragraph text. " * 600)
    result = _make(body, "blog", metadata={"meta_description": "short"})
    assert result["audit"]["status"] == "pass"


def test_linkedin_first_line_too_long_blocks() -> None:
    long_line = "x" * 200
    body = long_line + "\n\n" + ("body " * 100)
    result = _make(body, "linkedin")
    assert "linkedin_hook_too_long" in result["audit"]["blocking_issues"]


def test_x_thread_tweet_count_out_of_band_blocks() -> None:
    tweets = ["short tweet"] * 12  # 12 > max of 10
    body = "\n\n---\n\n".join(tweets)
    result = _make(body, "x_thread")
    assert "x_thread_tweet_count_out_of_band" in result["audit"]["blocking_issues"]


def test_x_thread_tweet_too_long_blocks() -> None:
    tweets = ["x" * 290] + ["ok"] * 4  # first tweet over 280
    body = "\n\n---\n\n".join(tweets)
    result = _make(body, "x_thread")
    assert "x_thread_tweet_too_long" in result["audit"]["blocking_issues"]


def test_x_thread_in_band_passes() -> None:
    tweets = [f"Tweet {i}/6 with content." for i in range(1, 7)]
    body = "\n\n---\n\n".join(tweets)
    result = _make(body, "x_thread", metadata={"tweet_count": 6})
    assert result["audit"]["status"] == "pass"


def test_shorts_missing_label_blocks() -> None:
    body = "HOOK: hook only.\n" + ("word " * 100)
    result = _make(body, "shorts")
    assert "shorts_missing_label" in result["audit"]["blocking_issues"]


def test_shorts_spoiler_too_early_blocks() -> None:
    """If teaching_moments[-1] appears at the start of body, block."""

    spoiler = "the surprise twist ending"
    body = (
        "HOOK: opener.\n"
        f"BODY: {spoiler}. " + ("filler word " * 80) + "\n"
        "CTA: subscribe"
    )
    idea = {"teaching_moments": ["the surprise twist ending"]}
    result = _make(body, "shorts", idea=idea)
    assert "spoiler_too_early" in result["audit"]["blocking_issues"]


def test_shorts_spoiler_in_final_tail_passes() -> None:
    """Spoiler in final 10% does not block."""

    spoiler = "the surprise twist ending"
    body = (
        "HOOK: opener. "
        "BODY: " + ("filler word " * 100) + spoiler + ".\n"
        "CTA: subscribe."
    )
    idea = {"teaching_moments": [spoiler]}
    result = _make(body, "shorts", idea=idea)
    assert "spoiler_too_early" not in result["audit"]["blocking_issues"]


def test_placeholder_token_blocks_all_formats() -> None:
    body = "Body content [Name] with placeholder. " + ("word " * 700)
    for fmt in ("newsletter", "blog", "linkedin", "x_thread", "shorts"):
        result = _make(body, fmt)
        assert "placeholder_token" in result["audit"]["blocking_issues"], fmt


def test_quote_drift_warning_fires_for_near_match() -> None:
    body = "He said the quick brown fox jumped over the fence yesterday. " + ("word " * 700)
    idea = {"key_quotes": ["The quick brown fox jumped over the fence yesterday"]}
    result = _make(body, "newsletter", idea=idea)
    assert "quote_drift" in result["audit"]["warnings"]


def test_banned_phrase_blocks() -> None:
    body = "We will delve into the topic at length. " + ("word " * 700)
    result = podcast_quality_revalidation(
        draft={"title": "t", "body": body, "metadata": {}},
        format_type="newsletter",
        voice_anchors={"banned_phrases": ["delve into"]},
    )
    assert "banned_phrase" in result["audit"]["blocking_issues"]
