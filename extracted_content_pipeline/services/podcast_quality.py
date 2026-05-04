"""Per-format deterministic quality validators for podcast repurposing."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .campaign_quality import _PLACEHOLDER_RE


# Word-count bands per format. (lo, hi).
_FORMAT_BANDS: dict[str, tuple[int, int]] = {
    "newsletter": (500, 1500),
    "blog": (1500, 3000),
    "linkedin": (100, 300),
    "x_thread": (50, 2800),  # not used directly; tweet-level checks below
    "shorts": (100, 200),
}

_X_THREAD_TWEET_LIMIT = 280
_X_THREAD_TWEET_COUNT_RANGE = (5, 10)
_LINKEDIN_FIRST_LINE_LIMIT = 120
_BLOG_META_DESC_LIMIT = 160
_SHORTS_LABELS = ("HOOK:", "BODY:", "CTA:")
_SHORTS_SPOILER_TAIL_FRACTION = 0.10  # final 10% of body chars
_LENGTH_BAND_HARD_FAIL_FRACTION = 0.25  # >25% out of band -> blocking


def podcast_quality_revalidation(
    *,
    draft: Mapping[str, Any],
    format_type: str,
    idea: Mapping[str, Any] | None = None,
    voice_anchors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a single per-format draft.

    Returns ``{audit, metadata, format_type}`` mirroring the campaign quality
    envelope. ``audit['status']`` is ``"pass"`` or ``"fail"``.
    """

    body = str(draft.get("body") or "")
    title = str(draft.get("title") or "")
    metadata = dict(draft.get("metadata") or {})
    fmt = str(format_type or "").strip().lower()

    blocking_issues: list[str] = []
    warnings: list[str] = []

    # Common: placeholder tokens block every format.
    if _PLACEHOLDER_RE.search(title) or _PLACEHOLDER_RE.search(body):
        blocking_issues.append("placeholder_token")

    # Common: banned phrases from voice_anchors.
    banned = _string_list((voice_anchors or {}).get("banned_phrases"))
    for phrase in banned:
        if phrase and phrase.lower() in body.lower():
            blocking_issues.append("banned_phrase")
            break

    # Common: quote fidelity.
    if idea is not None:
        quote_status = _quote_fidelity(body, _string_list(idea.get("key_quotes")))
        if quote_status == "drift":
            warnings.append("quote_drift")

    # Format-specific checks.
    if fmt == "newsletter":
        _check_word_count(body, fmt, blocking_issues, warnings)
    elif fmt == "blog":
        _check_word_count(body, fmt, blocking_issues, warnings)
        if not re.search(r"^#\s+.+", body, re.MULTILINE):
            blocking_issues.append("missing_h1")
        meta_desc = str(metadata.get("meta_description") or "")
        if len(meta_desc) > _BLOG_META_DESC_LIMIT:
            warnings.append("meta_description_too_long")
    elif fmt == "linkedin":
        _check_word_count(body, fmt, blocking_issues, warnings)
        first_line = body.splitlines()[0] if body else ""
        if len(first_line) > _LINKEDIN_FIRST_LINE_LIMIT:
            blocking_issues.append("linkedin_hook_too_long")
    elif fmt == "x_thread":
        _check_x_thread(body, metadata, blocking_issues, warnings)
    elif fmt == "shorts":
        _check_word_count(body, fmt, blocking_issues, warnings)
        for label in _SHORTS_LABELS:
            if label not in body:
                blocking_issues.append("shorts_missing_label")
                break
        if idea is not None and "shorts_missing_label" not in blocking_issues:
            spoiler = _shorts_spoiler_check(body, idea)
            if spoiler:
                blocking_issues.append(spoiler)
    else:
        warnings.append("unknown_format")

    # De-duplicate.
    blocking_issues = list(dict.fromkeys(blocking_issues))
    warnings = list(dict.fromkeys(warnings))

    audit = {
        "boundary": "podcast_format_repurpose",
        "format_type": fmt,
        "status": "fail" if blocking_issues else "pass",
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "primary_blocker": blocking_issues[0] if blocking_issues else None,
        "word_count": _word_count(body),
    }
    return {
        "audit": audit,
        "metadata": metadata,
        "format_type": fmt,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _check_word_count(
    body: str,
    fmt: str,
    blocking: list[str],
    warnings: list[str],
) -> None:
    band = _FORMAT_BANDS.get(fmt)
    if band is None:
        return
    lo, hi = band
    count = _word_count(body)
    if count < lo:
        gap_fraction = (lo - count) / max(1, lo)
        if gap_fraction > _LENGTH_BAND_HARD_FAIL_FRACTION:
            blocking.append("length_under_band")
        else:
            warnings.append("length_under_band")
    elif count > hi:
        gap_fraction = (count - hi) / max(1, hi)
        if gap_fraction > _LENGTH_BAND_HARD_FAIL_FRACTION:
            blocking.append("length_over_band")
        else:
            warnings.append("length_over_band")


def _check_x_thread(
    body: str,
    metadata: Mapping[str, Any],
    blocking: list[str],
    warnings: list[str],
) -> None:
    tweets = [chunk.strip() for chunk in body.split("\n\n---\n\n") if chunk.strip()]
    lo, hi = _X_THREAD_TWEET_COUNT_RANGE
    if len(tweets) < lo or len(tweets) > hi:
        blocking.append("x_thread_tweet_count_out_of_band")
    for tweet in tweets:
        if len(tweet) > _X_THREAD_TWEET_LIMIT:
            blocking.append("x_thread_tweet_too_long")
            break
    declared = metadata.get("tweet_count")
    if declared is not None:
        try:
            declared_int = int(declared)
        except (TypeError, ValueError):
            declared_int = -1
        if declared_int != len(tweets):
            warnings.append("x_thread_tweet_count_mismatch")


def _shorts_spoiler_check(body: str, idea: Mapping[str, Any]) -> str | None:
    spoilers = _string_list(idea.get("teaching_moments"))
    if spoilers:
        spoiler_text = spoilers[-1]
    else:
        args = _string_list(idea.get("arguments"))
        spoiler_text = args[-1] if args else ""
    spoiler_text = spoiler_text.strip()
    if not spoiler_text:
        return None
    body_lower = body.lower()
    spoiler_lower = spoiler_text.lower()
    if spoiler_lower not in body_lower:
        return None
    pos = body_lower.find(spoiler_lower)
    tail_start = int(len(body) * (1 - _SHORTS_SPOILER_TAIL_FRACTION))
    if pos < tail_start:
        return "spoiler_too_early"
    return None


def _quote_fidelity(body: str, quotes: list[str]) -> str | None:
    """Return ``"drift"`` if any quote near-matches but is not exact, else None."""

    if not quotes:
        return None
    normalized_body = _normalize_for_match(body)
    for quote in quotes:
        text = quote.strip()
        # Match the extraction skill spec: ignore quotes shorter than 8 words
        # rather than 8 characters. Short quotes are too generic to drift-check.
        if not text or _word_count(text) < 8:
            continue
        if text in body:
            continue
        # Near-match: collapse whitespace + drop common punctuation.
        normalized_quote = _normalize_for_match(text)
        if not normalized_quote:
            continue
        if normalized_quote in normalized_body:
            return "drift"
    return None


def _normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s]+", " ", text).strip()
    text = re.sub(r"[\.,;:!\?\"'\-]", "", text)
    return text


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value or "").strip()
    return [text] if text else []


__all__ = ["podcast_quality_revalidation"]
