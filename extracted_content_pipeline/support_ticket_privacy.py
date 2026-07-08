"""Support-ticket private/internal admission helpers.

Design: fail-closed pattern-family classification, not marker enumeration.

The privacy-marker vocabulary (key aliases x value spellings x container
shapes) is producer-defined and open, so exact-match enumeration cannot
converge -- every unlisted alias admits private text. The closed rule here:

- A key is privacy-relevant iff its compacted token remainder (raw, and after
  dropping structural stopwords and ``is/has/was`` prefixes and
  note/comment/reply/flag suffixes) EXACTLY equals a privacy stem. Exact
  equality of the semantic remainder -- never substring containment -- so data
  columns such as ``internal_id``, ``private_ip``, ``publication_date``,
  ``hidden_count``, and ``has_access`` are not markers.
- For a privacy-relevant key, only values that affirmatively resolve public
  admit the row/comment. Unknown text, conflicting or empty object markers,
  and malformed numerics all classify private. Kind keys (``type``/``kind``)
  keep fail-open categorical semantics; ``access``/``audience`` keep
  value-label semantics.
- The public predicates are total functions: any unexpected classification
  error returns private instead of raising, so a malformed marker can never
  crash package or Zendesk ingestion.
- Row-mode note-content carveout: a note/comment-suffixed private key whose
  value is free text is a content column, not a flag, and does not reject the
  whole row; flag-valued forms (``is_private_note: true``) do.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal
from functools import lru_cache
from typing import Any

_COMPACT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMERIC_MARKER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?\Z")

# Key stems: the CLOSED set of privacy meanings a key can carry. A key
# classifies iff its semantic remainder equals one of these exactly.
_PRIVATE_KEY_STEMS = frozenset({
    "private",
    "internal",
    "hidden",
    "confidential",
    "nonpublic",
    "restricted",
    "agentonly",
    "agentsonly",
    "staffonly",
})
_PUBLIC_KEY_STEMS = frozenset({"public", "visible", "external"})
_STRICT_LABEL_KEY_STEMS = frozenset({"visibility", "privacy"})
_VALUE_LABEL_KEY_STEMS = frozenset({"access", "audience"})
_KIND_KEY_STEMS = frozenset({"type", "kind"})

# Structural tokens that carry no privacy meaning; removing them exposes the
# semantic remainder (``visible_to_customer`` -> ``visible``).
_KEY_STOP_TOKENS = frozenset({
    "is", "has", "was", "are", "to", "for", "of", "the",
    "flag", "flags", "marker", "markers",
    "note", "notes", "comment", "comments", "reply", "replies",
    "message", "messages", "msg",
    "row", "rows", "ticket", "tickets", "entry", "entries",
    "field", "fields", "setting", "settings",
    "only", "use", "uses",
    "customer", "customers", "agent", "agents", "staff",
    "admin", "admins", "team", "teams",
    "user", "users", "viewer", "viewers",
})
# Prefix/suffix strips for keys that arrive pre-compacted (``isprivatenote``).
_STRIP_PREFIXES = ("is", "has", "was")
_STRIP_SUFFIXES = (
    "notes", "note", "comments", "comment", "replies", "reply",
    "messages", "message", "msg", "flags", "flag", "only",
)
_NOTE_HINTS = ("note", "comment", "reply", "message", "msg")
# Audience tokens are droppable structure, but a private audience flips a
# public-visibility key private: ``visible_to_agents`` is the inverse of
# ``visible_to_customer``.
_PRIVATE_AUDIENCE_TOKENS = frozenset({
    "agent", "agents", "staff", "admin", "admins", "team", "teams",
})
# Label-style suffixes are structural ONLY on label-class stems:
# ``visibility_status`` means visibility, but ``internal_status`` is a data
# column, so these strip only when the remainder is a label stem.
_LABELISH_TOKENS = frozenset({
    "label", "labels", "status", "state", "states",
    "value", "values", "level", "levels", "mode", "modes",
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
    "internalreply",
    "privatereply",
    "agentonly",
    "agentsonly",
    "staffonly",
    "restricted",
    "restrictedaccess",
    "nonpublic",
    "hidden",
})
_OBJECT_MARKER_KEYS = frozenset({
    "name",
    "label",
    "value",
    "status",
    "visibility",
    "privacy",
    "type",
    "kind",
})
_AMBIGUOUS_MARKER = "__ambiguous__"

_KIND_PRIVATE = "private"
_KIND_PUBLIC = "public"
_KIND_STRICT_LABEL = "strict_label"
_KIND_VALUE_LABEL = "value_label"
_KIND_KIND = "kind"


def support_ticket_comment_is_private(comment: Mapping[str, Any]) -> bool:
    """Return true when a support-ticket comment should be treated as private.

    Total function: classification errors fail closed (private), never raise.
    """

    try:
        return _mapping_is_private(comment, row_mode=False)
    except Exception:
        return True


def support_ticket_row_is_private(row: Mapping[str, Any]) -> bool:
    """Return true when a whole support-ticket source row is marked private.

    Total function: classification errors fail closed (private), never raise.
    """

    try:
        return _mapping_is_private(row, row_mode=True)
    except Exception:
        return True


def _mapping_is_private(value: Mapping[str, Any], *, row_mode: bool) -> bool:
    for key, raw_value in value.items():
        classified = _classify_key(str(key))
        if classified is None:
            continue
        kind, note_alias = classified
        marker = _marker(raw_value)
        if marker is None:
            continue
        if _marker_is_private(kind, marker, note_alias=note_alias, row_mode=row_mode):
            return True
    return False


def _marker_is_private(
    kind: str,
    marker: str,
    *,
    note_alias: bool,
    row_mode: bool,
) -> bool:
    if kind == _KIND_PRIVATE:
        boolish = _boolish(marker)
        if boolish is False or marker in _PUBLIC_LABELS:
            return False
        if (
            boolish is True
            or marker == _AMBIGUOUS_MARKER
            or marker in _PRIVATE_LABELS
        ):
            return True
        # Unresolved free text under a note/comment-suffixed key is a content
        # column in row mode, not a privacy flag; comment mode fails closed.
        return not (row_mode and note_alias)
    if kind == _KIND_PUBLIC:
        boolish = _boolish(marker)
        if boolish is True or marker in _PUBLIC_LABELS:
            return False
        if boolish is False or marker == _AMBIGUOUS_MARKER or marker in _PRIVATE_LABELS:
            return True
        # Unresolved free text under a note/comment-suffixed public key is a
        # content column (public_comment / public_comments hold customer
        # text in _PUBLIC_COMMENT_KEYS ingestion), not a privacy flag.
        return not note_alias
    if kind == _KIND_STRICT_LABEL:
        return marker not in _PUBLIC_LABELS
    if kind == _KIND_VALUE_LABEL:
        return marker == _AMBIGUOUS_MARKER or marker in _PRIVATE_LABELS
    # _KIND_KIND: categorical, fail-open on unknown kinds by design.
    return marker == _AMBIGUOUS_MARKER or marker in _PRIVATE_LABELS


@lru_cache(maxsize=4096)
def _classify_key(key: str) -> tuple[str, bool] | None:
    tokens = _key_tokens(key)
    if not tokens:
        return None
    compact_all = "".join(tokens)
    note_alias = any(hint in compact_all for hint in _NOTE_HINTS)
    private_audience = any(token in _PRIVATE_AUDIENCE_TOKENS for token in tokens)
    candidates = [compact_all]
    filtered = [token for token in tokens if token not in _KEY_STOP_TOKENS]
    if filtered and len(filtered) < len(tokens):
        candidates.append("".join(filtered))
    for candidate in candidates:
        for stem in _stem_forms(candidate):
            kind = _stem_kind(stem)
            if kind is not None:
                if kind == _KIND_PUBLIC and private_audience:
                    # visible_to_agents / public_to_staff assert privacy.
                    kind = _KIND_PRIVATE
                return (kind, note_alias)
    # Label-style suffixes are structural only on label-class stems
    # (visibility_status / privacy_label / access_label), never on
    # private stems (internal_status stays a data column).
    delabeled = [token for token in filtered or tokens if token not in _LABELISH_TOKENS]
    if delabeled and len(delabeled) < len(filtered or tokens):
        for stem in _stem_forms("".join(delabeled)):
            kind = _stem_kind(stem)
            if kind in (_KIND_STRICT_LABEL, _KIND_VALUE_LABEL):
                return (kind, note_alias)
    return None


def _key_tokens(key: str) -> tuple[str, ...]:
    spaced = _CAMEL_RE.sub(" ", key)
    return tuple(_TOKEN_RE.findall(spaced.lower()))


def _stem_forms(compact: str) -> tuple[str, ...]:
    """All prefix/suffix-stripped variants of a compacted key, incl itself."""

    seen: set[str] = set()
    queue = [compact]
    while queue:
        current = queue.pop()
        if not current or current in seen:
            continue
        seen.add(current)
        for prefix in _STRIP_PREFIXES:
            if current.startswith(prefix) and len(current) > len(prefix):
                queue.append(current[len(prefix):])
        for suffix in _STRIP_SUFFIXES:
            if current.endswith(suffix) and len(current) > len(suffix):
                queue.append(current[: -len(suffix)])
    return tuple(seen)


def _stem_kind(stem: str) -> str | None:
    if stem in _PRIVATE_KEY_STEMS:
        return _KIND_PRIVATE
    if stem in _PUBLIC_KEY_STEMS:
        return _KIND_PUBLIC
    if stem in _STRICT_LABEL_KEY_STEMS:
        return _KIND_STRICT_LABEL
    if stem in _VALUE_LABEL_KEY_STEMS:
        return _KIND_VALUE_LABEL
    if stem in _KIND_KEY_STEMS:
        return _KIND_KIND
    return None


def _marker(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        # A present-but-empty structured marker is unresolvable, not absent:
        # {"privacy": {}} fails closed like any other unreadable marker.
        if not value:
            return _AMBIGUOUS_MARKER
        return _object_marker(value)
    if isinstance(value, (int, float)):
        return _numeric_marker(str(value).strip().lower()) or _key(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        numeric = _numeric_marker(text)
        if numeric is not None:
            return numeric
        return _key(text)
    if isinstance(value, (list, tuple, set, frozenset)) and not value:
        return None
    # Present but shaped in a way this boundary cannot resolve: fail closed.
    return _AMBIGUOUS_MARKER


def _object_marker(value: Mapping[str, Any]) -> str:
    resolved: list[str] = []
    for key, marker_value in value.items():
        if _key(key) not in _OBJECT_MARKER_KEYS:
            continue
        marker = _marker(marker_value)
        if marker is not None:
            resolved.append(marker)
    if not resolved:
        return _AMBIGUOUS_MARKER
    first = resolved[0]
    if all(marker == first for marker in resolved):
        return first
    return _AMBIGUOUS_MARKER


def _numeric_marker(value: str) -> str | None:
    if not _NUMERIC_MARKER_RE.fullmatch(value):
        return None
    try:
        number = Decimal(value)
    except (ArithmeticError, ValueError):
        # Malformed/oversized numerics are unresolvable, never a crash.
        return _AMBIGUOUS_MARKER
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
