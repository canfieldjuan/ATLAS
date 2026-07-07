"""Shared support-ticket text and comment-boundary hygiene."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html import unescape
from typing import Any

from .support_ticket_clustering import support_ticket_plain_text


_AUTO_REPLY_LINE_RE = re.compile(
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
_QUOTED_REPLY_HEADER_RE = re.compile(
    r"^\s*on\s+.{1,160}\s+wrote:\s*$",
    re.IGNORECASE,
)
_SIGNATURE_BOUNDARY_RE = re.compile(r"^\s*(?:--+|__+)\s*$")
_MOBILE_SIGNATURE_RE = re.compile(
    r"^\s*sent from my (?:iphone|ipad|android|mobile device)\.?\s*$",
    re.IGNORECASE,
)
_HTML_SIGNAL_RE = re.compile(
    r"(?is)<\s*/?\s*"
    r"(?:a|blockquote|body|br|div|em|html|li|ol|p|span|strong|td|tr|ul)\b"
)
_HTML_LINE_BREAK_RE = re.compile(
    r"(?is)<\s*(?:br|/blockquote|/div|/li|/p|/tr)\b[^>]*>"
)
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")
_NUMERIC_ZERO_RE = re.compile(r"^[+-]?0+(?:\.0+)?$")
_NUMERIC_ONE_RE = re.compile(r"^\+?1(?:\.0+)?$")
_COMMENT_PUBLIC_KEYS = ("public", "is_public")
_COMMENT_PRIVATE_KEYS = (
    "private",
    "is_private",
    "internal",
    "is_internal",
    "internal_note",
    "is_internal_note",
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
_TRUEISH_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSEISH_VALUES = frozenset({"0", "false", "no", "n", "off"})


def support_ticket_text_component(value: Any) -> str:
    """Return ticket text after input hygiene, before clustering."""

    return support_ticket_plain_text(_strip_ticket_text_junk(value))


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
        if marker is not True:
            return True
    for key in _COMMENT_VISIBILITY_KEYS:
        label = _key(_first_value(item, (key,)))
        if label in _COMMENT_PRIVATE_LABELS:
            return True
    return False


def _strip_ticket_text_junk(value: Any) -> str:
    text = _line_preserving_text(value)
    if not text.strip():
        return ""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(raw_line)
            continue
        if _AUTO_REPLY_LINE_RE.match(line):
            continue
        if (
            _SIGNATURE_BOUNDARY_RE.match(line)
            or _MOBILE_SIGNATURE_RE.match(line)
            or _QUOTED_REPLY_HEADER_RE.match(line)
        ):
            break
        if line.startswith(">"):
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def _line_preserving_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    if not text.strip() or not _HTML_SIGNAL_RE.search(text):
        return text
    text = _HTML_LINE_BREAK_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return unescape(text)


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
    "support_ticket_text_component",
]
