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
    "notpublic",
    "restricted",
    "agentonly",
    "agentsonly",
    "staffonly",
})
_PUBLIC_KEY_STEMS = frozenset({"public", "visible", "external"})
# Publication/facing flags assert privacy ONLY through boolean values:
# ``published: false`` is private, but ``published: <date>`` is a data column.
_PUBLIC_FLAG_KEY_STEMS = frozenset({
    "published", "customerfacing", "clientfacing", "userfacing", "publicfacing",
})
_STRICT_LABEL_KEY_STEMS = frozenset({"visibility", "privacy", "confidentiality"})
_PUBLIC_FAMILY_TOKENS = frozenset({"public", "visible", "visibility", "external"})
_PUBLIC_AUDIENCE_TOKENS = frozenset({
    "customer", "customers", "user", "users", "enduser", "endusers",
    "requester", "requesters", "client", "clients", "viewer", "viewers",
    "everyone",
})
# Structural tokens inside VALUE phrases ("restricted to agents",
# "private_response", "for internal use only").
_VALUE_STRUCTURAL_TOKENS = frozenset({
    "is", "to", "for", "of", "the", "from", "only", "use", "uses",
    "access", "team", "teams", "note", "notes", "comment", "comments",
    "reply", "replies", "message", "messages", "msg",
    "response", "responses", "answer", "answers", "facing",
})
_VALUE_LABEL_KEY_STEMS = frozenset({"access", "audience"})
_KIND_KEY_STEMS = frozenset({"type", "kind"})

# Structural tokens that carry no privacy meaning; removing them exposes the
# semantic remainder (``visible_to_customer`` -> ``visible``).
_KEY_STOP_TOKENS = frozenset({
    "is", "has", "was", "are", "to", "for", "of", "the", "from",
    "flag", "flags", "marker", "markers",
    "note", "notes", "comment", "comments", "reply", "replies",
    "message", "messages", "msg",
    "row", "rows", "ticket", "tickets", "entry", "entries",
    "field", "fields", "setting", "settings",
    "only", "use", "uses",
    "customer", "customers", "agent", "agents", "staff",
    "admin", "admins", "team", "teams",
    "user", "users", "viewer", "viewers",
    "end", "ends", "requester", "requesters", "client", "clients",
    "support", "supports", "facing",
})
# Prefix/suffix strips for keys that arrive pre-compacted (``isprivatenote``).
_STRIP_PREFIXES = ("is", "has", "was", "access")
_STRIP_SUFFIXES = (
    "notes", "note", "comments", "comment", "replies", "reply",
    "messages", "message", "msg", "flags", "flag", "only", "access",
)
_NOTE_HINTS = ("note", "comment", "reply", "message", "msg")
# Audience tokens are droppable structure, but a private audience flips a
# public-visibility key private: ``visible_to_agents`` is the inverse of
# ``visible_to_customer``.
_PRIVATE_AUDIENCE_TOKENS = frozenset({
    "agent", "agents", "staff", "admin", "admins", "team", "teams",
    "support", "supports",
    "internal", "private", "hidden", "confidential", "restricted",
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
    "everyone",
    "anyone",
    "all",
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
    "public",
    "ispublic",
    "private",
    "isprivate",
    "internal",
    "hidden",
})
_AMBIGUOUS_MARKER = "__ambiguous__"

_KIND_PRIVATE = "private"
_KIND_PUBLIC = "public"
_KIND_PUBLIC_FLAG = "public_flag"
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
        kind, note_alias, flag_shaped = classified
        marker = _marker(raw_value, negative_polarity=kind == _KIND_PRIVATE)
        if marker is None:
            continue
        content_column = note_alias and not flag_shaped
        if _marker_is_private(
            kind, marker, content_column=content_column, row_mode=row_mode
        ):
            return True
    return False


def _marker_is_private(
    kind: str,
    marker: str,
    *,
    content_column: bool,
    row_mode: bool,
) -> bool:
    if kind == _KIND_PRIVATE:
        boolish = _boolish(marker)
        if boolish is False or marker in _PUBLIC_LABELS:
            return False
        if boolish is True:
            return True
        if marker != _AMBIGUOUS_MARKER and _label_is_private(marker):
            return True
        # Remaining markers are unresolved text or container shapes. Under a
        # non-flag note/comment-suffixed key in row mode these are content
        # columns, not privacy flags -- but digit-only values are flag
        # values, not content; everywhere else they fail closed.
        if row_mode and content_column and not marker.isdigit():
            return False
        return True
    if kind == _KIND_PUBLIC:
        boolish = _boolish(marker)
        if boolish is True or marker in _PUBLIC_LABELS:
            return False
        if boolish is False:
            return True
        if marker != _AMBIGUOUS_MARKER and _label_is_private(marker):
            return True
        # Remaining markers are unresolved text or container shapes. Under a
        # non-flag note/comment-suffixed public key these are content columns
        # (public_comment / public_comments hold customer text or comment
        # lists in _PUBLIC_COMMENT_KEYS ingestion), not privacy flags --
        # but digit-only values are flag values, not content; everywhere
        # else (incl. is_/has_ flag aliases) they fail closed.
        return not (content_column and not marker.isdigit())
    if kind == _KIND_PUBLIC_FLAG:
        # Publication/facing flags assert privacy only via booleans:
        # published: false is private; published: <date> is a data column.
        return _boolish(marker) is False
    if kind == _KIND_STRICT_LABEL:
        return marker not in _PUBLIC_LABELS
    if kind == _KIND_VALUE_LABEL:
        return marker == _AMBIGUOUS_MARKER or _label_is_private(marker)
    # _KIND_KIND: categorical, fail-open on unknown kinds by design.
    return marker == _AMBIGUOUS_MARKER or _label_is_private(marker)


@lru_cache(maxsize=4096)
def _classify_key(key: str) -> tuple[str, bool, bool] | None:
    tokens = _key_tokens(key)
    if not tokens:
        return None
    compact_all = "".join(tokens)
    note_alias = any(hint in compact_all for hint in _NOTE_HINTS)
    # Flag-shaped aliases (is_/has_/was_ prefixed or *_flag) are assertions,
    # never content columns, so the content carveouts must not apply to them.
    flag_shaped = (
        compact_all.startswith(_STRIP_PREFIXES)
        or "flag" in tokens
        or "flags" in tokens
    )
    private_audience = any(token in _PRIVATE_AUDIENCE_TOKENS for token in tokens)
    candidates = [compact_all]
    filtered = [token for token in tokens if token not in _KEY_STOP_TOKENS]
    if filtered and len(filtered) < len(tokens):
        candidates.append("".join(filtered))
    # Negated assertions flip polarity: not_visible_to_customer /
    # not_publicly_visible / not_customer_facing assert the private side,
    # not_private asserts the public side.
    if tokens[0] in ("not", "non") and len(tokens) > 1:
        rest = _classify_key(" ".join(tokens[1:]))
        if rest is not None:
            rest_kind, rest_note, rest_flag = rest
            if rest_kind in (_KIND_PUBLIC, _KIND_PUBLIC_FLAG):
                return (_KIND_PRIVATE, note_alias or rest_note, flag_shaped or rest_flag)
            if rest_kind == _KIND_PRIVATE:
                return (_KIND_PUBLIC, note_alias or rest_note, flag_shaped or rest_flag)
        # Unresolvable negations stay unclassified rather than guessing.
    # Audience-only compounds (admin_only / team_only / support_team_only):
    # every token before "only" is a private audience.
    if (
        len(tokens) >= 2
        and tokens[-1] == "only"
        and all(token in _PRIVATE_AUDIENCE_TOKENS for token in tokens[:-1])
    ):
        return (_KIND_PRIVATE, note_alias, flag_shaped)
    # Audience-note keys (staff_note / agent_reply / support_note): every
    # token is a private audience or note structure, with both present.
    note_structure = frozenset({
        "note", "notes", "comment", "comments", "reply", "replies",
        "message", "messages", "msg", "response", "responses",
    })
    if (
        len(tokens) >= 2
        and any(token in _PRIVATE_AUDIENCE_TOKENS for token in tokens)
        and any(token in note_structure for token in tokens)
        and all(
            token in _PRIVATE_AUDIENCE_TOKENS or token in note_structure
            for token in tokens
        )
    ):
        return (_KIND_PRIVATE, True, flag_shaped)
    # All-public-family compounds (publicly_visible / public_visibility):
    # every meaningful token asserts the public side.
    meaningful = [token for token in tokens if token not in _KEY_STOP_TOKENS]
    if (
        meaningful
        and len(meaningful) >= 2
        and all(token in _PUBLIC_FAMILY_TOKENS for token in meaningful)
    ):
        return (_KIND_PUBLIC, note_alias, flag_shaped)
    for candidate in candidates:
        # Bare audience-only compounds (admin_only / team_only /
        # support_only) are the staff-only side, same as agent_only.
        if candidate.endswith("only") and candidate[:-4] in _PRIVATE_AUDIENCE_TOKENS:
            return (_KIND_PRIVATE, note_alias, flag_shaped)
        for stem in _stem_forms(candidate):
            kind = _stem_kind(stem)
            if kind is not None:
                if kind == _KIND_PUBLIC and private_audience:
                    # visible_to_agents / public_to_staff assert privacy.
                    kind = _KIND_PRIVATE
                return (kind, note_alias, flag_shaped)
    # Audience-flip pass: private-audience tokens are droppable structure on
    # a public-visibility stem (visible_to_internal / public_to_internal).
    if private_audience:
        depersoned = [
            token
            for token in tokens
            if token not in _KEY_STOP_TOKENS
            and token not in _PRIVATE_AUDIENCE_TOKENS
        ]
        if depersoned and len(depersoned) < len(tokens):
            for stem in _stem_forms("".join(depersoned)):
                if _stem_kind(stem) in (_KIND_PUBLIC, _KIND_PUBLIC_FLAG):
                    return (_KIND_PRIVATE, note_alias, flag_shaped)
    # Label-style suffixes are structural only on label-class stems
    # (visibility_status / privacy_label / access_label), never on
    # private stems (internal_status stays a data column).
    delabeled = [token for token in filtered or tokens if token not in _LABELISH_TOKENS]
    if delabeled and len(delabeled) < len(filtered or tokens):
        for stem in _stem_forms("".join(delabeled)):
            kind = _stem_kind(stem)
            if kind in (_KIND_STRICT_LABEL, _KIND_VALUE_LABEL):
                return (kind, note_alias, flag_shaped)
    return None


_ADVERB_TOKEN_MAP = {
    "publicly": "public",
    "privately": "private",
    "internally": "internal",
    "externally": "external",
    "visibly": "visible",
}


def _key_tokens(key: str) -> tuple[str, ...]:
    spaced = _CAMEL_RE.sub(" ", key)
    return tuple(
        _ADVERB_TOKEN_MAP.get(token, token)
        for token in _TOKEN_RE.findall(spaced.lower())
    )


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
    if stem in _PUBLIC_FLAG_KEY_STEMS:
        return _KIND_PUBLIC_FLAG
    if stem in _STRICT_LABEL_KEY_STEMS:
        return _KIND_STRICT_LABEL
    if stem in _VALUE_LABEL_KEY_STEMS:
        return _KIND_VALUE_LABEL
    if stem in _KIND_KEY_STEMS:
        return _KIND_KIND
    return None


def _marker(value: Any, *, negative_polarity: bool = False) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        # A present-but-empty structured marker is unresolvable, not absent:
        # {"privacy": {}} fails closed like any other unreadable marker.
        if not value:
            return _AMBIGUOUS_MARKER
        return _object_marker(value, negative_polarity=negative_polarity)
    if isinstance(value, (int, float)):
        return _numeric_marker(str(value).strip().lower()) or _key(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        numeric = _numeric_marker(text)
        if numeric is not None:
            return numeric
        return _value_marker(text)
    if isinstance(value, (list, tuple, set, frozenset)) and not value:
        # Present-but-empty sequence markers fail closed like empty objects.
        return _AMBIGUOUS_MARKER
    # Present but shaped in a way this boundary cannot resolve: fail closed.
    return _AMBIGUOUS_MARKER


def _object_marker(value: Mapping[str, Any], *, negative_polarity: bool) -> str:
    """Resolve a structured marker to a verdict in the OUTER key's frame.

    Neutral boolean subfields ({"value": true}) take the enclosing key's
    polarity; alias subfields (publicly_visible, hidden_from_customer) take
    their own classified polarity; label subfields resolve by value. Any
    conflict between verdicts fails closed.
    """

    verdicts: set[str] = set()
    for key, marker_value in value.items():
        compact_key = _key(key)
        classified = _classify_key(str(key))
        subfield_kind = classified[0] if classified is not None else None
        polarity_subfield = subfield_kind in (
            _KIND_PRIVATE, _KIND_PUBLIC, _KIND_PUBLIC_FLAG,
        )
        if not polarity_subfield and compact_key not in _OBJECT_MARKER_KEYS:
            if subfield_kind is None:
                continue
        marker = _marker(marker_value)
        if marker is None:
            continue
        boolish = _boolish(marker)
        if boolish is not None:
            # Assertion subfields carry their OWN polarity ({"private": true},
            # {"publicly_visible": false}); neutral subfields ({"value": true})
            # take the enclosing key's polarity.
            if subfield_kind == _KIND_PRIVATE:
                verdicts.add("private" if boolish else "public")
            elif subfield_kind in (_KIND_PUBLIC, _KIND_PUBLIC_FLAG):
                verdicts.add("public" if boolish else "private")
            elif compact_key in _OBJECT_MARKER_KEYS:
                if negative_polarity:
                    verdicts.add("private" if boolish else "public")
                else:
                    verdicts.add("public" if boolish else "private")
            else:
                # Label-class alias subfield with a bare boolean: unresolvable.
                verdicts.add("unknown")
            continue
        verdicts.add(_marker_verdict(marker))
    if not verdicts or "unknown" in verdicts or len(verdicts) > 1:
        return _AMBIGUOUS_MARKER
    return next(iter(verdicts))


def _marker_verdict(marker: str) -> str:
    if _boolish(marker) is True or marker in _PUBLIC_LABELS:
        return "public"
    if _boolish(marker) is False or (
        marker != _AMBIGUOUS_MARKER and _label_is_private(marker)
    ):
        return "private"
    return "unknown"


def _value_marker(text: str) -> str:
    """Resolve a text value to a marker, token-classifying label phrases.

    Multi-token values get the same closed token rule as keys, so
    "restricted to agents", "support staff only", "private_response", and
    "visible to customer" resolve by their semantic tokens instead of
    per-spelling compact enumeration. Single tokens and already-known
    compacts keep the pinned membership behavior.
    """

    compact = _key(text)
    if (
        compact in _PUBLIC_LABELS
        or compact in _PRIVATE_LABELS
        or compact in _TRUTHY_TEXT
        or compact in _FALSEY_TEXT
    ):
        return compact
    tokens = _key_tokens(text)
    if len(tokens) >= 2:
        private_side = _PRIVATE_KEY_STEMS | _PRIVATE_AUDIENCE_TOKENS
        public_side = (
            _PUBLIC_FAMILY_TOKENS | _PUBLIC_AUDIENCE_TOKENS | _PUBLIC_LABELS
        )
        has_private = any(token in private_side for token in tokens)
        has_public = any(token in public_side for token in tokens)
        structural = _VALUE_STRUCTURAL_TOKENS
        if has_private and all(
            token in private_side or token in structural or token in public_side
            for token in tokens
        ):
            # Any private signal in a privacy phrase fails closed, even
            # alongside public words ("visible to staff").
            return "private"
        if (
            has_public
            and not has_private
            and all(token in public_side or token in structural for token in tokens)
        ):
            return "public"
    return compact


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


def _label_is_private(marker: str) -> bool:
    """Private-label check with a closed value-stem rule.

    Strips structural suffixes (only/use/access/team) so labels like
    ``internal only``, ``private-use-only``, and ``agents only`` resolve to a
    private stem or private audience without enumerating each spelling.
    """

    if marker == _AMBIGUOUS_MARKER:
        return True
    if marker in _PRIVATE_LABELS:
        return True
    if marker in _PUBLIC_LABELS or marker in _TRUTHY_TEXT or marker in _FALSEY_TEXT:
        return False
    current = marker
    changed = True
    while changed:
        changed = False
        for suffix in (
            "only", "use", "uses", "access", "team", "teams",
            "note", "notes", "comment", "comments",
            "reply", "replies", "message", "messages",
        ):
            if current.endswith(suffix) and len(current) > len(suffix):
                current = current[: -len(suffix)]
                changed = True
        for prefix in ("for", "is"):
            if current.startswith(prefix) and len(current) > len(prefix):
                current = current[len(prefix):]
                changed = True
    if current in _PRIVATE_KEY_STEMS or current in _PRIVATE_AUDIENCE_TOKENS:
        return True
    return current in _PRIVATE_LABELS


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
