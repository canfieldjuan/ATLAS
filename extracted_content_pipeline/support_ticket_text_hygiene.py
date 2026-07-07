"""Shared support-ticket text and comment-boundary hygiene."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html import unescape
from html.parser import HTMLParser
from typing import Any

from .support_ticket_clustering import support_ticket_plain_text


_AUTO_REPLY_INLINE_RE = re.compile(
    r"^\s*(?:auto(?:mated)?[-\s]?reply|automatic reply|out of office)\s*:\s*"
    r"(?:we\b.*\b(?:received|got|will|respond|reply)|"
    r"(?:we'?ll|will)\b.*\b(?:respond|reply)|"
    r"(?:thanks?|thank you)\b|"
    r"your (?:ticket|request)\b|"
    r"i (?:am|'m|will be)\b|"
    r"i'll\b|"
    r"this (?:mailbox|message)\b|"
    r"away from\b|"
    r"currently out\b)",
    re.IGNORECASE,
)
_AUTO_REPLY_HEADER_RE = re.compile(
    r"^\s*(?:auto(?:mated)?[-\s]?reply|automatic reply|out of office)\s*:?\s*$",
    re.IGNORECASE,
)
_AUTO_REPLY_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"we\b.*\b(?:received|got|will|respond|reply)|"
    r"(?:we'?ll|will)\b.*\b(?:respond|reply)|"
    r"(?:thanks?|thank you)\b.*\b(?:contacting|reaching out)|"
    r"your (?:ticket|request)\b.*\b(?:received|created)|"
    r"(?:i am|i'm|i will be|i'll)\b.*\b(?:out|away|respond)|"
    r"this (?:mailbox|message)\b|"
    r"away from\b|"
    r"currently out\b"
    r")",
    re.IGNORECASE,
)
_QUOTED_REPLY_HEADER_RE = re.compile(
    r"^\s*on\s+"
    r"(?:(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?\b|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|"
    r"\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?\b|"
    r".{0,120}\b\d{1,2}:\d{2}\b|"
    r".{0,120}<[^>]+@[^>]+>|"
    r".{0,120}\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b)"
    r".{0,160}\s+wrote:\s*$",
    re.IGNORECASE,
)
_SIGNATURE_BOUNDARY_RE = re.compile(r"^\s*(?:--+|__+)\s*$")
_MOBILE_SIGNATURE_RE = re.compile(
    r"^\s*sent from my (?:iphone|ipad|android|mobile device)\.?\s*$",
    re.IGNORECASE,
)
_HTML_SIGNAL_RE = re.compile(
    r"(?is)<\s*/?\s*"
    r"(?:a|blockquote|body|br|div|em|html|li|ol|p|script|span|strong|style|td|tr|ul)\b"
)
_HTML_DROP_BLOCK_RE = re.compile(
    r"(?is)<\s*(script|style|blockquote)\b[^>]*>.*?<\s*/\s*\1\s*>"
)
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")
_NUMERIC_ZERO_RE = re.compile(r"^[+-]?0+(?:\.0+)?$")
_NUMERIC_ONE_RE = re.compile(r"^\+?1(?:\.0+)?$")
_HTML_SKIP_TAGS = frozenset({"script", "style", "blockquote"})
_HTML_LINE_TAGS = frozenset({
    "blockquote",
    "br",
    "div",
    "li",
    "ol",
    "p",
    "tr",
    "ul",
})
_COMMENT_PUBLIC_KEYS = ("public", "is_public")
_COMMENT_PRIVATE_KEYS = (
    "private",
    "private_note",
    "private_comment",
    "is_private",
    "is_private_note",
    "is_private_comment",
    "internal",
    "internal_note",
    "internal_comment",
    "is_internal",
    "is_internal_note",
    "is_internal_comment",
)
_COMMENT_VISIBILITY_KEYS = (
    "visibility",
    "visibility_type",
    "comment_type",
    "type",
    "kind",
)
_COMMENT_PRIVATE_LABELS = frozenset({
    "agentnote",
    "internal",
    "internalcomment",
    "internalnote",
    "private",
    "privatecomment",
    "privatenote",
    "staffnote",
})
_COMMENT_PUBLIC_LABELS = frozenset({
    "customer",
    "customercomment",
    "external",
    "public",
    "publiccomment",
})
_TRUEISH_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSEISH_VALUES = frozenset({"0", "false", "no", "n", "off"})


def support_ticket_text_component(value: Any) -> str:
    """Return ticket text after input hygiene, before clustering."""

    return support_ticket_plain_text(_strip_ticket_text_junk(value))


def support_ticket_history_text(value: Any) -> str:
    """Return transcript text while preserving later messages after signatures."""

    return support_ticket_plain_text(
        _strip_ticket_text_junk(value, stop_at_boundary=False)
    )


def support_ticket_comment_is_private(item: Mapping[str, Any]) -> bool:
    """Return whether a comment should be excluded from customer/report text."""

    for key in _COMMENT_PRIVATE_KEYS:
        value = _first_value(item, (key,))
        if value is None:
            continue
        marker = _boolish(value)
        if marker is not False:
            return True
    for key in _COMMENT_PUBLIC_KEYS:
        value = _first_value(item, (key,))
        if value is None:
            continue
        marker = _boolish(value)
        if marker is True:
            continue
        if marker is None and _key(value) in _COMMENT_PUBLIC_LABELS:
            continue
        if marker is not True:
            return True
    for key in _COMMENT_VISIBILITY_KEYS:
        label = _key(_first_value(item, (key,)))
        if label in _COMMENT_PRIVATE_LABELS:
            return True
    return False


def _strip_ticket_text_junk(value: Any, *, stop_at_boundary: bool = True) -> str:
    text = _line_preserving_text(value)
    if not text.strip():
        return ""
    lines: list[str] = []
    skip_auto_block = False
    skip_signature_block = False
    skip_quote_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            skip_auto_block = False
            skip_signature_block = False
            skip_quote_block = False
            lines.append(raw_line)
            continue
        if skip_auto_block and _AUTO_REPLY_BOILERPLATE_RE.match(line):
            continue
        if skip_auto_block:
            skip_auto_block = False
        if skip_signature_block:
            continue
        if skip_quote_block and line.startswith(">"):
            continue
        if skip_quote_block:
            skip_quote_block = False
        if _AUTO_REPLY_INLINE_RE.match(line):
            continue
        if _AUTO_REPLY_HEADER_RE.match(line):
            skip_auto_block = True
            continue
        if _SIGNATURE_BOUNDARY_RE.match(line) or _MOBILE_SIGNATURE_RE.match(line):
            if stop_at_boundary:
                break
            skip_signature_block = True
            continue
        if _QUOTED_REPLY_HEADER_RE.match(line):
            if stop_at_boundary:
                break
            skip_quote_block = True
            continue
        if line.startswith(">") and not stop_at_boundary:
            continue
        if line.startswith(">"):
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def _line_preserving_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    if not text.strip():
        return ""
    unescaped = unescape(text)
    if not _HTML_SIGNAL_RE.search(text):
        if _HTML_SIGNAL_RE.search(unescaped):
            return _html_to_line_preserving_text(unescaped)
        return unescaped
    return _html_to_line_preserving_text(text)


def _html_to_line_preserving_text(text: str) -> str:
    parser = _LinePreservingHTMLExtractor()
    try:
        parser.feed(text)
        parser.close()
        return "".join(parser.parts)
    except Exception:
        text = _HTML_DROP_BLOCK_RE.sub("\n", text)
        text = _HTML_TAG_RE.sub(" ", text)
        return unescape(text)


class _LinePreservingHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in _HTML_SKIP_TAGS:
            self._skip_depth += 1
            self.parts.append("\n")
            return
        if lowered in _HTML_LINE_TAGS:
            self.parts.append("\n")
            return
        self.parts.append(" ")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in _HTML_SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            self.parts.append("\n")
            return
        if lowered in _HTML_LINE_TAGS:
            self.parts.append("\n")
            return
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value or "").strip()
    if _NUMERIC_ONE_RE.fullmatch(text):
        return True
    if _NUMERIC_ZERO_RE.fullmatch(text):
        return False
    key = _key(text)
    if key in _TRUEISH_VALUES:
        return True
    if key in _FALSEISH_VALUES:
        return False
    return None


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        normalized_key = _key(key)
        for raw_key, value in row.items():
            if _key(raw_key) == normalized_key and _has_value(value):
                return value
    return None


def _has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value not in (None, "", [], {})


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


__all__ = [
    "support_ticket_comment_is_private",
    "support_ticket_history_text",
    "support_ticket_text_component",
]
