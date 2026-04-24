"""Normalized grounding check for phrase verbatim validation (Phase 1b).

Used by:
- v2 write-time validation: decides if `verbatim=True` in phrase_metadata is
  retained or coerced to False.
- v1 read-time compatibility: API layer checks stored witness excerpts
  against review_text even when the LLM was never asked to self-report.
- Batch-populate backfill: populates the `grounding_status` column on the
  `b2b_vendor_witnesses` table.

Design notes:
- `_normalize_for_grounding()` produces a canonical form used SOLELY for
  substring-match comparison. The normalized string must not be displayed
  or persisted; the original text is the only source of truth for UI
  rendering.
- Normalization handles three real-world noise sources observed in scraped
  review text: messy whitespace (tabs/newlines/non-breaking spaces),
  unicode punctuation (curly quotes, em/en dashes, ellipsis) and markdown
  artifacts (backslash escapes, bold/italic emphasis wrappers).
- `check_phrase_grounded()` concatenates `summary + ' ' + review_text` to
  match the source blob that `_excerpt_text()` uses when deriving witness
  excerpts. A phrase that is verbatim in the summary alone still counts.
"""

from __future__ import annotations

import re
from typing import Any

# Unicode punctuation normalization. Keys use \uXXXX escapes so this
# source file stays ASCII-clean per repo convention.
_UNICODE_PUNCTUATION_MAP: dict[str, str] = {
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote / typographic apostrophe
    "\u201A": "'",    # single low-9 quote
    "\u201C": '"',  # left double quote
    "\u201D": '"',  # right double quote
    "\u201E": '"',  # double low-9 quote
    "\u2013": "-",    # en dash
    "\u2014": "--",   # em dash
    "\u2026": "...",  # horizontal ellipsis
    "\u00A0": " ",    # non-breaking space
    "\u2009": " ",    # thin space
    "\u200A": " ",    # hair space
    "\u200B": "",     # zero-width space
    "\u200C": "",     # zero-width non-joiner
    "\u200D": "",     # zero-width joiner
    "\uFEFF": "",     # byte-order mark
}

_BACKSLASH_ESCAPE_RE = re.compile(r"\\([.,!?;:'\"()\[\]\-])")
_MARKDOWN_BOLD_AST_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MARKDOWN_BOLD_UND_RE = re.compile(r"__(.+?)__", re.DOTALL)
_MARKDOWN_ITAL_AST_RE = re.compile(r"(?<!\w)\*(.+?)\*(?!\w)", re.DOTALL)
_MARKDOWN_ITAL_UND_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_grounding(text: Any) -> str:
    """Canonicalize text for substring comparison in grounding checks.

    Steps (order matters):
      1. Coerce None / non-str inputs to empty string.
      2. Apply unicode punctuation map (curly quotes, em/en dashes, etc.).
      3. Strip backslash escape characters before punctuation.
      4. Strip markdown emphasis wrappers (keep inner content).
      5. Collapse all whitespace runs to a single space.
      6. Strip leading/trailing whitespace.
      7. Casefold for case-insensitive comparison.

    The returned string is for substring-equality comparison only. Never
    display it; never persist it. The original text remains the source of
    truth for UI rendering and storage.
    """
    if not isinstance(text, str) or not text:
        return ""

    for source_char, replacement in _UNICODE_PUNCTUATION_MAP.items():
        if source_char in text:
            text = text.replace(source_char, replacement)

    text = _BACKSLASH_ESCAPE_RE.sub(r"\1", text)
    text = _MARKDOWN_BOLD_AST_RE.sub(r"\1", text)
    text = _MARKDOWN_BOLD_UND_RE.sub(r"\1", text)
    text = _MARKDOWN_ITAL_AST_RE.sub(r"\1", text)
    text = _MARKDOWN_ITAL_UND_RE.sub(r"\1", text)

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.casefold()


def check_phrase_grounded(
    phrase_text: Any,
    *,
    summary: Any = None,
    review_text: Any = None,
) -> bool:
    """Return True iff the normalized phrase appears as a substring of EITHER
    the normalized summary OR the normalized review_text, checked
    independently.

    Concatenating summary + review_text before checking would let a phrase
    "spanning" the artificial join be flagged as grounded even when it
    exists in neither field as a real quote. Quote-grade verbatim cannot
    tolerate that. Each candidate source must contain the phrase on its own.

    A phrase grounded in just the summary still counts -- that's a real
    quote, just from the title rather than the body. Empty phrase or both
    sources empty returns False.
    """
    phrase_norm = _normalize_for_grounding(phrase_text)
    if not phrase_norm:
        return False

    if isinstance(summary, str) and summary.strip():
        if phrase_norm in _normalize_for_grounding(summary):
            return True
    if isinstance(review_text, str) and review_text.strip():
        if phrase_norm in _normalize_for_grounding(review_text):
            return True
    return False
