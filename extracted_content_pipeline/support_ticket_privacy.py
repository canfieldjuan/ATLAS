"""Support-ticket private/internal admission helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any


_COMPACT_RE = re.compile(r"[^a-z0-9]+")

_PUBLIC_BOOL_KEYS = frozenset({"public", "ispublic"})
_PRIVATE_BOOL_KEYS = frozenset({
    "private",
    "isprivate",
    "internal",
    "isinternal",
    "hidden",
    "nonpublic",
})
_COMMENT_PRIVATE_ALIAS_KEYS = frozenset({
    "privatecomment",
    "privatecomments",
    "privatenote",
    "privatenotes",
    "internalcomment",
    "internalcomments",
    "internalnote",
    "internalnotes",
})
_PRIVACY_LABEL_KEYS = frozenset({
    "visibility",
    "commentvisibility",
    "privacy",
    "privacysetting",
    "access",
    "audience",
})
_KIND_LABEL_KEYS = frozenset({
    "type",
    "kind",
    "commenttype",
    "messagetype",
    "notetype",
})

_TRUTHY_TEXT = frozenset({"true", "t", "yes", "y", "on"})
_FALSEY_TEXT = frozenset({"false", "f", "no", "n", "off"})
_PUBLIC_LABELS = frozenset({
    "public",
    "publiccomment",
    "publicreply",
    "customer",
    "customerreply",
    "external",
    "visible",
    "published",
})
_PRIVATE_LABELS = frozenset({
    "private",
    "privatecomment",
    "privatenote",
    "internal",
    "internalcomment",
    "internalnote",
    "agentonly",
    "staffonly",
    "nonpublic",
    "hidden",
})


def support_ticket_comment_is_private(comment: Mapping[str, Any]) -> bool:
    """Return true when a support-ticket comment should be treated as private."""

    return _mapping_is_private(comment, include_comment_alias_keys=True)


def support_ticket_row_is_private(row: Mapping[str, Any]) -> bool:
    """Return true when a whole support-ticket source row is marked private."""

    return _mapping_is_private(row, include_comment_alias_keys=False)


def _mapping_is_private(
    value: Mapping[str, Any],
    *,
    include_comment_alias_keys: bool,
) -> bool:
    markers = {_key(key): marker_value for key, marker_value in value.items()}
    for key in _PUBLIC_BOOL_KEYS:
        marker = _marker(markers.get(key))
        if marker is None:
            continue
        if _boolish(marker) is True or marker in _PUBLIC_LABELS:
            continue
        return True
    private_keys = set(_PRIVATE_BOOL_KEYS)
    if include_comment_alias_keys:
        private_keys.update(_COMMENT_PRIVATE_ALIAS_KEYS)
    for key in private_keys:
        marker = _marker(markers.get(key))
        if marker is None:
            continue
        boolish = _boolish(marker)
        if boolish is False or marker in _PUBLIC_LABELS:
            continue
        return True
    for key in _PRIVACY_LABEL_KEYS:
        marker = _marker(markers.get(key))
        if marker is None:
            continue
        if marker in _PUBLIC_LABELS:
            continue
        return True
    for key in _KIND_LABEL_KEYS:
        marker = _marker(markers.get(key))
        if marker in _PRIVATE_LABELS:
            return True
    return False


def _marker(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _numeric_marker(str(value))
    text = str(value).strip().lower()
    if not text:
        return None
    numeric = _numeric_marker(text)
    if numeric is not None:
        return numeric
    return _key(text)


def _numeric_marker(value: str) -> str | None:
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    if number == 0:
        return "false"
    if number == 1:
        return "true"
    return None


def _boolish(marker: str) -> bool | None:
    if marker in _TRUTHY_TEXT:
        return True
    if marker in _FALSEY_TEXT:
        return False
    return None


def _key(value: Any) -> str:
    return _COMPACT_RE.sub("", str(value or "").lower())


__all__ = [
    "support_ticket_comment_is_private",
    "support_ticket_row_is_private",
]
