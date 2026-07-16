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
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from typing import Any

from .support_ticket_clustering import support_ticket_plain_text

_COMPACT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMERIC_MARKER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?\Z")
_NONFINITE_NUMERIC_RE = re.compile(r"[+-]?(?:inf(?:inity)?|[sq]?nan\d*)\Z")
_ISO_DATE_DATA_RE = re.compile(
    r"(?P<year>\d{4})-"
    r"(?P<month>0[1-9]|1[0-2])-"
    r"(?P<day>0[1-9]|[12]\d|3[01])"
    r"(?:(?:[t ](?:[01]\d|2[0-3]):[0-5]\d"
    r"(?::[0-5]\d(?:\.\d+)?)?"
    r"(?:z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?)?"
    r"|(?:z|[+-](?:[01]\d|2[0-3]):?[0-5]\d))?\Z"
)
_ISO_DATE_CANDIDATE_RE = re.compile(r"\d{4}-\d")

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
# Publication/facing flags admit recognized public values plus explicit
# date/finite-number data shapes; every other present value fails closed.
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
    "and", "or", "end",
    "access", "team", "teams", "note", "notes", "comment", "comments",
    "reply", "replies", "message", "messages", "msg",
    "response", "responses", "answer", "answers", "facing",
})
_VALUE_LABEL_KEY_STEMS = frozenset({"access", "audience"})
_KIND_KEY_STEMS = frozenset({"type", "kind"})
# Exact producer data columns that otherwise resemble a prefix-stripped marker.
# Normalizing first keeps snake/camel/space/hyphen spellings equivalent.
_NON_MARKER_KEY_COMPACTS = frozenset({"hasaccess"})

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
_STRICT_PUBLIC_LABELS = _PUBLIC_LABELS | _PUBLIC_AUDIENCE_TOKENS
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
_COMMENT_CONTENT_KEY_TOKENS = frozenset({
    "body", "message", "text", "content", "description",
    "plainbody", "htmlbody",
})
_AMBIGUOUS_MARKER = "__ambiguous__"
_PUBLIC_PHRASE_MARKER = "__public_phrase__"

class _MarkerFamily(Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    PUBLIC_FLAG = "public_flag"
    STRICT_LABEL = "strict_label"
    VALUE_LABEL = "value_label"
    KIND = "kind"


_KIND_PRIVATE = _MarkerFamily.PRIVATE
_KIND_PUBLIC = _MarkerFamily.PUBLIC
_KIND_PUBLIC_FLAG = _MarkerFamily.PUBLIC_FLAG
_KIND_STRICT_LABEL = _MarkerFamily.STRICT_LABEL
_KIND_VALUE_LABEL = _MarkerFamily.VALUE_LABEL
_KIND_KIND = _MarkerFamily.KIND


class _PrivacyVerdict(Enum):
    ABSENT = "absent"
    PUBLIC = "public"
    PRIVATE = "private"
    NEUTRAL_UNKNOWN = "neutral_unknown"
    AMBIGUOUS = "ambiguous"


class _PublicationDataKind(Enum):
    NOT_DATA = "not_data"
    VALID = "valid"
    INVALID_DATE = "invalid_date"


@dataclass(frozen=True)
class _MarkerDecision:
    verdict: _PrivacyVerdict
    token: str = ""
    boolish: bool | None = None
    boolean_assertion: bool = False
    malformed_numeric: bool = False
    malformed_boolean_numeric: bool = False
    nonfinite_numeric: bool = False
    recognized_failing: bool = False
    publication_data: bool = False
    privacy_structure: bool = False
    strict_public_label: bool = False
    public_phrase: bool = False
    content_container: bool = False


@dataclass(frozen=True)
class _KeyDecision:
    family: _MarkerFamily
    note_alias: bool = False
    flag_shaped: bool = False


@dataclass(frozen=True)
class _FamilyPolicy:
    boolean_admits: frozenset[bool]
    verdicts_admit: frozenset[_PrivacyVerdict]
    admit_strict_public_label: bool = False
    admit_public_phrase: bool = False
    admit_publication_data: bool = False
    admit_neutral: bool = False
    admit_content: bool = False
    malformed_fails: bool = False
    malformed_boolean_fails: bool = False
    nonfinite_fails: bool = False
    recognized_failing_fails: bool = False


_FAMILY_POLICIES = {
    _KIND_PRIVATE: _FamilyPolicy(
        boolean_admits=frozenset({False}),
        verdicts_admit=frozenset({_PrivacyVerdict.PUBLIC}),
        admit_content=True,
        malformed_fails=True,
        malformed_boolean_fails=True,
        recognized_failing_fails=True,
    ),
    _KIND_PUBLIC: _FamilyPolicy(
        boolean_admits=frozenset({True}),
        verdicts_admit=frozenset({_PrivacyVerdict.PUBLIC}),
        admit_public_phrase=True,
        admit_content=True,
        malformed_fails=True,
        recognized_failing_fails=True,
    ),
    _KIND_PUBLIC_FLAG: _FamilyPolicy(
        boolean_admits=frozenset({True}),
        verdicts_admit=frozenset({_PrivacyVerdict.PUBLIC}),
        admit_strict_public_label=True,
        admit_public_phrase=True,
        admit_publication_data=True,
        nonfinite_fails=True,
        recognized_failing_fails=True,
    ),
    _KIND_STRICT_LABEL: _FamilyPolicy(
        boolean_admits=frozenset(),
        verdicts_admit=frozenset({_PrivacyVerdict.PUBLIC}),
        admit_strict_public_label=True,
        admit_public_phrase=True,
        malformed_fails=True,
        malformed_boolean_fails=True,
        recognized_failing_fails=True,
    ),
    _KIND_VALUE_LABEL: _FamilyPolicy(
        boolean_admits=frozenset({False, True}),
        verdicts_admit=frozenset({
            _PrivacyVerdict.PUBLIC,
            _PrivacyVerdict.NEUTRAL_UNKNOWN,
        }),
        admit_neutral=True,
        malformed_fails=True,
        malformed_boolean_fails=True,
    ),
    _KIND_KIND: _FamilyPolicy(
        boolean_admits=frozenset({False, True}),
        verdicts_admit=frozenset({
            _PrivacyVerdict.PUBLIC,
            _PrivacyVerdict.NEUTRAL_UNKNOWN,
        }),
        admit_neutral=True,
    ),
}


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
        kind = classified.family
        content_column = classified.note_alias and not classified.flag_shaped
        allow_content = content_column and (
            row_mode or kind == _KIND_PUBLIC
        )
        marker = _marker(
            raw_value,
            negative_polarity=kind == _KIND_PRIVATE,
            outer_kind=kind,
            allow_content_sequence=allow_content,
        )
        if marker.verdict is _PrivacyVerdict.ABSENT:
            continue
        if _decision_is_private(
            kind, marker, content_column=content_column, row_mode=row_mode
        ):
            return True
    return False


def _decision_is_private(
    kind: _MarkerFamily,
    marker: _MarkerDecision,
    *,
    content_column: bool,
    row_mode: bool,
) -> bool:
    policy = _FAMILY_POLICIES[kind]
    if marker.nonfinite_numeric and policy.nonfinite_fails:
        return True
    if (
        marker.malformed_numeric
        and policy.malformed_fails
        and (
            not marker.malformed_boolean_numeric
            or policy.malformed_boolean_fails
        )
    ):
        return True
    if (
        marker.recognized_failing
        and marker.privacy_structure
        and policy.recognized_failing_fails
    ):
        return True
    if (
        policy.admit_content
        and content_column
        and marker.content_container
        and marker.boolish is None
        and (
            marker.verdict is _PrivacyVerdict.NEUTRAL_UNKNOWN
            or (
                marker.verdict is _PrivacyVerdict.AMBIGUOUS
                and not marker.privacy_structure
            )
        )
        and (row_mode or kind is _KIND_PUBLIC)
    ):
        return False
    if marker.recognized_failing and policy.recognized_failing_fails:
        return True
    if marker.boolish is not None:
        return marker.boolish not in policy.boolean_admits
    if marker.verdict in policy.verdicts_admit:
        return False
    if (
        policy.admit_strict_public_label
        and marker.verdict is _PrivacyVerdict.NEUTRAL_UNKNOWN
        and marker.strict_public_label
    ):
        return False
    if (
        policy.admit_public_phrase
        and marker.verdict is _PrivacyVerdict.NEUTRAL_UNKNOWN
        and marker.public_phrase
    ):
        return False
    if policy.admit_publication_data and marker.publication_data:
        return False
    if policy.admit_neutral and marker.verdict is _PrivacyVerdict.NEUTRAL_UNKNOWN:
        return False
    return True


def _boolean_verdict(
    family: _MarkerFamily | None,
    value: bool,
    *,
    neutral_wrapper: bool,
) -> _PrivacyVerdict:
    if family is None:
        return _PrivacyVerdict.AMBIGUOUS
    if family in (_KIND_VALUE_LABEL, _KIND_KIND):
        return _PrivacyVerdict.NEUTRAL_UNKNOWN
    if family is _KIND_STRICT_LABEL:
        return (
            _PrivacyVerdict.NEUTRAL_UNKNOWN
            if neutral_wrapper
            else _PrivacyVerdict.AMBIGUOUS
        )
    return (
        _PrivacyVerdict.PUBLIC
        if value in _FAMILY_POLICIES[family].boolean_admits
        else _PrivacyVerdict.PRIVATE
    )


@lru_cache(maxsize=4096)
def _classify_key(key: str) -> _KeyDecision | None:
    tokens = _key_tokens(key)
    if not tokens:
        return None
    compact_all = "".join(tokens)
    if compact_all in _NON_MARKER_KEY_COMPACTS:
        return None
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
            if rest.family in (_KIND_PUBLIC, _KIND_PUBLIC_FLAG):
                return _KeyDecision(
                    _KIND_PRIVATE,
                    note_alias or rest.note_alias,
                    flag_shaped or rest.flag_shaped,
                )
            if rest.family == _KIND_PRIVATE:
                return _KeyDecision(
                    _KIND_PUBLIC,
                    note_alias or rest.note_alias,
                    flag_shaped or rest.flag_shaped,
                )
        # Unresolvable negations stay unclassified rather than guessing.
    # Compact ``un<public-stem>`` assertions (unpublished / unpublic) carry the
    # private polarity. Restrict this to an exact public/public-flag remainder;
    # arbitrary ``un...`` producer columns remain unclassified.
    for candidate in candidates:
        if _is_compact_negated_public_stem(candidate):
            return _KeyDecision(_KIND_PRIVATE, note_alias, True)
    # Audience-only compounds (admin_only / team_only / support_team_only):
    # every token before "only" is a private audience.
    if (
        len(tokens) >= 2
        and tokens[-1] == "only"
        and all(token in _PRIVATE_AUDIENCE_TOKENS for token in tokens[:-1])
    ):
        return _KeyDecision(_KIND_PRIVATE, note_alias, flag_shaped)
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
        return _KeyDecision(_KIND_PRIVATE, True, flag_shaped)
    # All-public-family compounds (publicly_visible / public_visibility):
    # every meaningful token asserts the public side.
    meaningful = [token for token in tokens if token not in _KEY_STOP_TOKENS]
    if (
        meaningful
        and len(meaningful) >= 2
        and all(token in _PUBLIC_FAMILY_TOKENS for token in meaningful)
    ):
        return _KeyDecision(_KIND_PUBLIC, note_alias, flag_shaped)
    for candidate in candidates:
        # Bare audience-only compounds (admin_only / team_only /
        # support_only) are the staff-only side, same as agent_only.
        if candidate.endswith("only") and candidate[:-4] in _PRIVATE_AUDIENCE_TOKENS:
            return _KeyDecision(_KIND_PRIVATE, note_alias, flag_shaped)
        for stem in _stem_forms(candidate):
            kind = _stem_kind(stem)
            if kind is not None:
                if kind == _KIND_PUBLIC and private_audience:
                    # visible_to_agents / public_to_staff assert privacy.
                    kind = _KIND_PRIVATE
                return _KeyDecision(kind, note_alias, flag_shaped)
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
                    return _KeyDecision(_KIND_PRIVATE, note_alias, flag_shaped)
    # Label-style suffixes are structural only on label-class stems
    # (visibility_status / privacy_label / access_label), never on
    # private stems (internal_status stays a data column).
    delabeled = [token for token in filtered or tokens if token not in _LABELISH_TOKENS]
    if delabeled and len(delabeled) < len(filtered or tokens):
        for stem in _stem_forms("".join(delabeled)):
            if _is_compact_negated_public_stem(stem):
                return _KeyDecision(_KIND_PRIVATE, note_alias, True)
            kind = _stem_kind(stem)
            if kind in (
                _KIND_PUBLIC_FLAG, _KIND_STRICT_LABEL, _KIND_VALUE_LABEL,
            ):
                return _KeyDecision(kind, note_alias, flag_shaped)
    # Pre-compacted producer keys (customerFacingStatus ->
    # customerfacingstatus) cannot use the token-level delabel pass because the
    # public-facing words are also structural stop tokens. Strip the label/status
    # suffix only when the exact remainder is a public-flag stem.
    for candidate in candidates:
        for suffix in _LABELISH_TOKENS:
            if candidate.endswith(suffix) and len(candidate) > len(suffix):
                remainder = candidate[: -len(suffix)]
                if _is_compact_negated_public_stem(remainder):
                    return _KeyDecision(_KIND_PRIVATE, note_alias, True)
                kind = _stem_kind(remainder)
                if kind in (
                    _KIND_PUBLIC_FLAG,
                    _KIND_STRICT_LABEL,
                    _KIND_VALUE_LABEL,
                ):
                    return _KeyDecision(
                        kind,
                        note_alias,
                        flag_shaped,
                    )
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


_VALUE_SEGMENT_TOKENS = frozenset({
    *_PRIVATE_KEY_STEMS,
    *_PRIVATE_AUDIENCE_TOKENS,
    *_PUBLIC_FAMILY_TOKENS,
    *_PUBLIC_AUDIENCE_TOKENS,
    *_PUBLIC_LABELS,
    *_VALUE_STRUCTURAL_TOKENS,
    *_TRUTHY_TEXT,
    *_FALSEY_TEXT,
    *_ADVERB_TOKEN_MAP,
    "kept",
    "longer",
    "made",
    "never",
    "no",
    "non",
    "not",
    "open",
    "shared",
    "un",
    "withheld",
    "without",
})
_VALUE_SEGMENT_TOKENS_LONGEST = tuple(
    sorted(_VALUE_SEGMENT_TOKENS, key=lambda token: (-len(token), token))
)


@lru_cache(maxsize=4096)
def _value_tokens(value: str) -> tuple[str, ...]:
    tokens = _key_tokens(value)
    if len(tokens) != 1:
        return tokens
    compact = tokens[0]

    @lru_cache(maxsize=None)
    def split(remainder: str) -> tuple[str, ...] | None:
        if not remainder:
            return ()
        for token in _VALUE_SEGMENT_TOKENS_LONGEST:
            if not remainder.startswith(token):
                continue
            tail = split(remainder[len(token):])
            if tail is not None:
                return (token, *tail)
        return None

    segmented = split(compact)
    if segmented is None or len(segmented) < 2:
        return tokens
    return tuple(_ADVERB_TOKEN_MAP.get(token, token) for token in segmented)


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


def _stem_kind(stem: str) -> _MarkerFamily | None:
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


def _is_compact_negated_public_stem(compact: str) -> bool:
    for prefix in ("un", "not", "non"):
        if compact.startswith(prefix) and len(compact) > len(prefix):
            if _stem_kind(compact[len(prefix):]) in (
                _KIND_PUBLIC,
                _KIND_PUBLIC_FLAG,
            ):
                return True
    return False


def _marker(
    value: Any,
    *,
    negative_polarity: bool = False,
    outer_kind: _MarkerFamily | None = None,
    allow_content_sequence: bool = False,
    neutral_boolean_wrapper: bool = False,
) -> _MarkerDecision:
    if value is None:
        return _MarkerDecision(_PrivacyVerdict.ABSENT)
    if isinstance(value, bool):
        return _scalar_marker_decision("true" if value else "false")
    if isinstance(value, Mapping):
        # A present-but-empty structured marker is unresolvable, not absent:
        # {"privacy": {}} fails closed like any other unreadable marker.
        if not value:
            return _MarkerDecision(
                _PrivacyVerdict.AMBIGUOUS,
                token=_AMBIGUOUS_MARKER,
                privacy_structure=True,
            )
        return _object_marker(
            value,
            negative_polarity=negative_polarity,
            outer_kind=outer_kind,
            allow_content=allow_content_sequence,
        )
    if isinstance(value, (int, float)):
        text = str(value).strip().lower()
        marker = _numeric_marker(text)
        if marker is not None:
            return _scalar_marker_decision(marker)
        publication_kind = _publication_data_kind(text)
        return _scalar_marker_decision(
            _key(value),
            # Type provenance is authoritative here: JSON exponent overflow is
            # decoded as a non-finite float whose rendered token is ``inf``.
            malformed_numeric=True,
            nonfinite_numeric=_text_is_nonfinite_numeric(text),
            recognized_failing=_text_is_nonfinite_numeric(text),
            publication_data=publication_kind is _PublicationDataKind.VALID,
        )
    if isinstance(value, str):
        raw_text = value.strip()
        text = raw_text.lower()
        if not text:
            return _MarkerDecision(_PrivacyVerdict.ABSENT)
        numeric = _numeric_marker(text)
        if numeric is not None:
            return _scalar_marker_decision(
                numeric,
                malformed_numeric="e" in text,
                malformed_boolean_numeric="e" in text,
            )
        resolved_marker = _value_marker(raw_text)
        nonfinite_numeric = _text_is_nonfinite_numeric(text)
        publication_kind = _publication_data_kind(text)
        decision = _scalar_marker_decision(
            resolved_marker,
            malformed_numeric=(
                _NUMERIC_MARKER_RE.fullmatch(text) is not None
                or nonfinite_numeric
            ),
            nonfinite_numeric=nonfinite_numeric,
            recognized_failing=(
                resolved_marker == _AMBIGUOUS_MARKER
                or _label_is_private(resolved_marker)
                or nonfinite_numeric
                or publication_kind is _PublicationDataKind.INVALID_DATE
                or _is_compact_negated_public_stem(_key(text))
                or (
                    resolved_marker not in _STRICT_PUBLIC_LABELS
                    and resolved_marker != _PUBLIC_PHRASE_MARKER
                    and _text_has_privacy_vocabulary(raw_text)
                )
            ),
            publication_data=publication_kind is _PublicationDataKind.VALID,
        )
        if allow_content_sequence:
            return replace(decision, content_container=True)
        return decision
    if isinstance(value, (list, tuple, set, frozenset)):
        if not value:
            # Empty sequences remain unresolved; row content-column policy may
            # still recognize them as an empty comments container.
            return _MarkerDecision(
                _PrivacyVerdict.AMBIGUOUS,
                token=_AMBIGUOUS_MARKER,
                content_container=allow_content_sequence,
            )
        return _sequence_marker(
            value,
            negative_polarity=negative_polarity,
            outer_kind=outer_kind,
            allow_content=allow_content_sequence,
            neutral_boolean_wrapper=neutral_boolean_wrapper,
        )
    # Present but shaped in a way this boundary cannot resolve: fail closed.
    return _MarkerDecision(
        _PrivacyVerdict.AMBIGUOUS,
        token=_AMBIGUOUS_MARKER,
    )


def _rendered_comment_text_has_content(value: str) -> bool:
    return any(character.isalnum() for character in support_ticket_plain_text(value))


def _is_absent_placeholder(
    value: Any,
    *,
    rendered_content: bool = False,
) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        if rendered_content:
            return not _rendered_comment_text_has_content(value)
        return not value.strip()
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(
            _is_absent_placeholder(item, rendered_content=rendered_content)
            for item in value
        )
    if isinstance(value, Mapping):
        allowed_keys = _OBJECT_MARKER_KEYS
        if rendered_content:
            allowed_keys = allowed_keys | _COMMENT_CONTENT_KEY_TOKENS
        return bool(value) and all(
            _key(key) in allowed_keys
            and _is_absent_placeholder(
                item,
                rendered_content=rendered_content,
            )
            for key, item in value.items()
        )
    return False


def _sequence_marker(
    value: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    *,
    negative_polarity: bool,
    outer_kind: _MarkerFamily | None,
    allow_content: bool,
    neutral_boolean_wrapper: bool,
) -> _MarkerDecision:
    """Combine marker-shaped sequence elements without consuming content lists."""

    substantive_items = tuple(
        item
        for item in value
        if not _is_absent_placeholder(item, rendered_content=allow_content)
    )
    if not substantive_items:
        return _MarkerDecision(_PrivacyVerdict.ABSENT)
    if len(substantive_items) == 1:
        return _marker(
            next(iter(substantive_items)),
            negative_polarity=negative_polarity,
            outer_kind=outer_kind,
            allow_content_sequence=allow_content,
            neutral_boolean_wrapper=neutral_boolean_wrapper,
        )

    decisions: list[_MarkerDecision] = []
    unresolved_element = False
    content_element = False
    for item in substantive_items:
        if allow_content and isinstance(item, str):
            content_marker = _marker(
                item,
                negative_polarity=negative_polarity,
                outer_kind=outer_kind,
                allow_content_sequence=True,
                neutral_boolean_wrapper=neutral_boolean_wrapper,
            )
            if content_marker.verdict is _PrivacyVerdict.ABSENT:
                continue
            if outer_kind is None or _decision_is_private(
                outer_kind,
                content_marker,
                content_column=True,
                row_mode=True,
            ):
                decisions.append(_MarkerDecision(
                    _PrivacyVerdict.PRIVATE,
                    boolean_assertion=content_marker.boolean_assertion,
                    malformed_numeric=content_marker.malformed_numeric,
                    malformed_boolean_numeric=(
                        content_marker.malformed_boolean_numeric
                    ),
                    nonfinite_numeric=content_marker.nonfinite_numeric,
                    recognized_failing=content_marker.recognized_failing,
                    privacy_structure=True,
                    strict_public_label=content_marker.strict_public_label,
                    public_phrase=content_marker.public_phrase,
                ))
            else:
                content_element = True
            continue
        if (
            allow_content
            and isinstance(item, Mapping)
            and _mapping_has_comment_content(item)
        ):
            content_marker = _object_marker(
                item,
                negative_polarity=negative_polarity,
                outer_kind=outer_kind,
                allow_content=True,
            )
            if outer_kind is None or _decision_is_private(
                outer_kind,
                content_marker,
                content_column=True,
                row_mode=True,
            ):
                decisions.append(_MarkerDecision(
                    _PrivacyVerdict.PRIVATE,
                    privacy_structure=True,
                ))
            else:
                content_element = True
            continue
        marker = _marker(
            item,
            negative_polarity=negative_polarity,
            outer_kind=outer_kind,
            neutral_boolean_wrapper=neutral_boolean_wrapper,
        )
        if marker.verdict is _PrivacyVerdict.ABSENT:
            continue
        if (
            allow_content
            and marker.boolish is not None
            and not marker.privacy_structure
        ):
            unresolved_element = True
            continue
        marker_shaped = (
            marker.privacy_structure
            or marker.malformed_numeric
            or marker.recognized_failing
            or marker.publication_data
            or marker.boolish is not None
            or marker.verdict in (_PrivacyVerdict.PUBLIC, _PrivacyVerdict.PRIVATE)
            or marker.strict_public_label
        )
        if not marker_shaped:
            # Preserve unresolved provenance. The caller's row-mode content
            # carveout, not this recursive parser, decides whether an entirely
            # content-shaped sequence is admissible.
            unresolved_element = True
            continue
        verdict = marker.verdict
        if marker.boolish is not None:
            verdict = _boolean_verdict(
                outer_kind,
                marker.boolish,
                neutral_wrapper=neutral_boolean_wrapper,
            )
        decisions.append(_MarkerDecision(
            verdict,
            token=marker.token,
            boolean_assertion=marker.boolean_assertion,
            malformed_numeric=marker.malformed_numeric,
            malformed_boolean_numeric=marker.malformed_boolean_numeric,
            nonfinite_numeric=marker.nonfinite_numeric,
            recognized_failing=marker.recognized_failing,
            publication_data=marker.publication_data,
            privacy_structure=True,
            strict_public_label=marker.strict_public_label,
            public_phrase=marker.public_phrase,
        ))
    if unresolved_element and (decisions or content_element):
        decisions.append(_MarkerDecision(
            _PrivacyVerdict.AMBIGUOUS,
            token=_AMBIGUOUS_MARKER,
            privacy_structure=True,
        ))
    if not decisions and not unresolved_element and not content_element:
        return _MarkerDecision(_PrivacyVerdict.ABSENT)
    combined = _combine_marker_decisions(decisions)
    if (
        len(decisions) > 1
        and not any(
            decision.verdict is _PrivacyVerdict.PUBLIC
            for decision in decisions
        )
        and all(decision.publication_data for decision in decisions)
    ):
        combined = replace(combined, publication_data=False)
    if content_element and not decisions:
        return replace(combined, content_container=True)
    return combined


def _object_marker(
    value: Mapping[str, Any],
    *,
    negative_polarity: bool,
    outer_kind: _MarkerFamily | None,
    allow_content: bool,
) -> _MarkerDecision:
    """Resolve a structured marker to a verdict in the OUTER key's frame.

    Neutral boolean subfields ({"value": true}) take the enclosing key's
    polarity; alias subfields (publicly_visible, hidden_from_customer) take
    their own classified polarity; label subfields resolve by value. Any
    conflict between verdicts fails closed.
    """

    decisions: list[_MarkerDecision] = []
    for key, marker_value in value.items():
        compact_key = _key(key)
        classified = _classify_key(str(key))
        subfield_kind = classified.family if classified is not None else None
        polarity_subfield = subfield_kind in (
            _KIND_PRIVATE, _KIND_PUBLIC, _KIND_PUBLIC_FLAG,
        )
        if not polarity_subfield and compact_key not in _OBJECT_MARKER_KEYS:
            if subfield_kind is None:
                continue
        if subfield_kind == _KIND_PRIVATE:
            nested_negative_polarity = True
        elif subfield_kind in (_KIND_PUBLIC, _KIND_PUBLIC_FLAG):
            nested_negative_polarity = False
        else:
            nested_negative_polarity = negative_polarity
        marker = _marker(
            marker_value,
            negative_polarity=nested_negative_polarity,
            outer_kind=subfield_kind or outer_kind,
            neutral_boolean_wrapper=(
                subfield_kind is None and compact_key in _OBJECT_MARKER_KEYS
            ),
        )
        if marker.verdict is _PrivacyVerdict.ABSENT:
            continue
        if marker.boolish is not None:
            # Assertion subfields carry their OWN polarity ({"private": true},
            # {"publicly_visible": false}); neutral subfields ({"value": true})
            # take the enclosing key's polarity.
            if subfield_kind is not None:
                verdict = _boolean_verdict(
                    subfield_kind,
                    marker.boolish,
                    neutral_wrapper=False,
                )
            elif compact_key in _OBJECT_MARKER_KEYS:
                verdict = _boolean_verdict(
                    outer_kind,
                    marker.boolish,
                    neutral_wrapper=True,
                )
            else:
                # Label-class alias subfield with a bare boolean: unresolvable.
                verdict = _PrivacyVerdict.AMBIGUOUS
            decisions.append(_MarkerDecision(
                verdict,
                token=marker.token,
                boolean_assertion=True,
                malformed_numeric=marker.malformed_numeric,
                malformed_boolean_numeric=marker.malformed_boolean_numeric,
                nonfinite_numeric=marker.nonfinite_numeric,
                recognized_failing=marker.recognized_failing,
                publication_data=marker.publication_data,
                privacy_structure=True,
                strict_public_label=marker.strict_public_label,
                public_phrase=marker.public_phrase,
            ))
            continue
        if subfield_kind is not None:
            policy = _FAMILY_POLICIES[subfield_kind]
            locally_affirmative = (
                marker.verdict is _PrivacyVerdict.PUBLIC
                or (
                    marker.verdict is _PrivacyVerdict.NEUTRAL_UNKNOWN
                    and (
                        (
                            policy.admit_strict_public_label
                            and marker.strict_public_label
                        )
                        or (
                            policy.admit_public_phrase
                            and marker.public_phrase
                        )
                    )
                )
            )
            if marker.verdict is _PrivacyVerdict.AMBIGUOUS:
                verdict = _PrivacyVerdict.AMBIGUOUS
            elif _decision_is_private(
                subfield_kind,
                marker,
                content_column=False,
                row_mode=False,
            ):
                verdict = _PrivacyVerdict.PRIVATE
            elif locally_affirmative:
                verdict = _PrivacyVerdict.PUBLIC
            elif marker.verdict is _PrivacyVerdict.NEUTRAL_UNKNOWN:
                verdict = _PrivacyVerdict.NEUTRAL_UNKNOWN
            else:
                verdict = _PrivacyVerdict.PUBLIC
            preserve_rejected_provenance = verdict in (
                _PrivacyVerdict.PRIVATE,
                _PrivacyVerdict.AMBIGUOUS,
            )
            preserve_affirmative_provenance = verdict is _PrivacyVerdict.PUBLIC
            decisions.append(_MarkerDecision(
                verdict,
                token=marker.token,
                boolean_assertion=marker.boolean_assertion,
                malformed_numeric=(
                    marker.malformed_numeric
                    if preserve_rejected_provenance
                    else False
                ),
                malformed_boolean_numeric=(
                    marker.malformed_boolean_numeric
                    if preserve_rejected_provenance
                    else False
                ),
                nonfinite_numeric=(
                    marker.nonfinite_numeric
                    if preserve_rejected_provenance
                    else False
                ),
                recognized_failing=(
                    marker.recognized_failing
                    if preserve_rejected_provenance
                    else False
                ),
                publication_data=(
                    marker.publication_data
                    if preserve_affirmative_provenance
                    else False
                ),
                privacy_structure=True,
                strict_public_label=(
                    marker.strict_public_label
                    if preserve_affirmative_provenance
                    else False
                ),
                public_phrase=(
                    marker.public_phrase
                    if preserve_affirmative_provenance
                    else False
                ),
            ))
            continue
        decisions.append(_MarkerDecision(
            marker.verdict,
            token=marker.token,
            boolish=marker.boolish,
            boolean_assertion=marker.boolean_assertion,
            malformed_numeric=marker.malformed_numeric,
            malformed_boolean_numeric=marker.malformed_boolean_numeric,
            nonfinite_numeric=marker.nonfinite_numeric,
            recognized_failing=marker.recognized_failing,
            publication_data=marker.publication_data,
            privacy_structure=True,
            strict_public_label=marker.strict_public_label,
            public_phrase=marker.public_phrase,
        ))
    content_container = allow_content and _mapping_has_comment_content(value)
    if not decisions:
        if content_container:
            return _MarkerDecision(
                _PrivacyVerdict.NEUTRAL_UNKNOWN,
                content_container=True,
            )
        return _MarkerDecision(
            _PrivacyVerdict.AMBIGUOUS,
            token=_AMBIGUOUS_MARKER,
            privacy_structure=True,
        )
    return replace(
        _combine_marker_decisions(decisions),
        content_container=content_container,
    )


def _combine_marker_decisions(
    decisions: list[_MarkerDecision],
) -> _MarkerDecision:
    verdicts = {decision.verdict for decision in decisions}
    assertive_verdicts = verdicts - {_PrivacyVerdict.NEUTRAL_UNKNOWN}
    recognized_failing = any(
        decision.recognized_failing for decision in decisions
    )
    if (
        not verdicts
        or _PrivacyVerdict.AMBIGUOUS in verdicts
        or len(assertive_verdicts) > 1
    ):
        return _MarkerDecision(
            _PrivacyVerdict.AMBIGUOUS,
            token=_AMBIGUOUS_MARKER,
            boolean_assertion=any(
                decision.boolean_assertion for decision in decisions
            ),
            malformed_numeric=any(
                decision.malformed_numeric for decision in decisions
            ),
            malformed_boolean_numeric=any(
                decision.malformed_boolean_numeric for decision in decisions
            ),
            nonfinite_numeric=any(
                decision.nonfinite_numeric for decision in decisions
            ),
            recognized_failing=recognized_failing,
            publication_data=False,
            privacy_structure=bool(decisions),
            strict_public_label=any(
                decision.strict_public_label for decision in decisions
            ),
            public_phrase=any(decision.public_phrase for decision in decisions),
        )
    return _MarkerDecision(
        next(iter(assertive_verdicts), _PrivacyVerdict.NEUTRAL_UNKNOWN),
        token=next(
            (
                decision.token
                for decision in decisions
                if decision.malformed_numeric
            ),
            "",
        ),
        boolean_assertion=any(decision.boolean_assertion for decision in decisions),
        malformed_numeric=any(
            decision.malformed_numeric for decision in decisions
        ),
        malformed_boolean_numeric=any(
            decision.malformed_boolean_numeric for decision in decisions
        ),
        nonfinite_numeric=any(
            decision.nonfinite_numeric for decision in decisions
        ),
        recognized_failing=recognized_failing,
        publication_data=(
            not assertive_verdicts
            and bool(decisions)
            and all(decision.publication_data for decision in decisions)
        ),
        privacy_structure=True,
        strict_public_label=any(
            decision.strict_public_label for decision in decisions
        ),
        public_phrase=any(decision.public_phrase for decision in decisions),
    )


def _scalar_marker_decision(
    marker: str,
    *,
    malformed_numeric: bool = False,
    malformed_boolean_numeric: bool = False,
    nonfinite_numeric: bool = False,
    recognized_failing: bool = False,
    publication_data: bool = False,
) -> _MarkerDecision:
    boolish = _boolish(marker)
    if boolish is not None:
        # Scalar booleans keep their raw polarity for the outer key family.
        return _MarkerDecision(
            _PrivacyVerdict.NEUTRAL_UNKNOWN,
            token=marker,
            boolish=boolish,
            boolean_assertion=True,
            malformed_numeric=malformed_numeric,
            malformed_boolean_numeric=malformed_boolean_numeric,
            nonfinite_numeric=nonfinite_numeric,
            recognized_failing=recognized_failing,
            publication_data=publication_data,
        )
    if marker in _PUBLIC_LABELS:
        verdict = _PrivacyVerdict.PUBLIC
    elif marker == _AMBIGUOUS_MARKER:
        verdict = _PrivacyVerdict.AMBIGUOUS
    elif _label_is_private(marker):
        verdict = _PrivacyVerdict.PRIVATE
    else:
        verdict = _PrivacyVerdict.NEUTRAL_UNKNOWN
    return _MarkerDecision(
        verdict,
        token=marker,
        malformed_numeric=malformed_numeric,
        malformed_boolean_numeric=malformed_boolean_numeric,
        nonfinite_numeric=nonfinite_numeric,
        recognized_failing=recognized_failing,
        publication_data=publication_data,
        strict_public_label=(
            marker in _STRICT_PUBLIC_LABELS
            or marker == _PUBLIC_PHRASE_MARKER
        ),
        public_phrase=marker == _PUBLIC_PHRASE_MARKER,
    )


def _value_marker(text: str) -> str:
    """Resolve a text value to a marker, token-classifying label phrases.

    Multi-token values get the same closed token rule as keys, so
    "restricted to agents", "support staff only", "private_response", and
    "visible to customer" resolve by their semantic tokens instead of
    per-spelling compact enumeration. Single tokens and already-known
    compacts keep the pinned membership behavior. Positive public-audience
    phrases become strict-public evidence rather than an affirmative public
    verdict, so the family policy decides whether that evidence admits.
    Unrecognized prose remains neutral.
    """

    compact = _key(text)
    if (
        compact in _PUBLIC_LABELS
        or compact in _PRIVATE_LABELS
        or compact in _TRUTHY_TEXT
        or compact in _FALSEY_TEXT
    ):
        return compact
    tokens = _value_tokens(text)
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
            return _PUBLIC_PHRASE_MARKER
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


def _text_is_nonfinite_numeric(value: str) -> bool:
    return _NONFINITE_NUMERIC_RE.fullmatch(value) is not None


def _publication_data_kind(value: str) -> _PublicationDataKind:
    if _NUMERIC_MARKER_RE.fullmatch(value) is not None:
        return _PublicationDataKind.VALID
    if _ISO_DATE_CANDIDATE_RE.match(value) is None:
        return _PublicationDataKind.NOT_DATA
    match = _ISO_DATE_DATA_RE.fullmatch(value)
    if match is None:
        return _PublicationDataKind.INVALID_DATE
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    if month == 2:
        leap_year = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        max_day = 29 if leap_year else 28
    elif month in {4, 6, 9, 11}:
        max_day = 30
    else:
        max_day = 31
    return (
        _PublicationDataKind.VALID
        if day <= max_day
        else _PublicationDataKind.INVALID_DATE
    )


def _text_has_privacy_vocabulary(value: str) -> bool:
    private_side = _PRIVATE_KEY_STEMS | _PRIVATE_AUDIENCE_TOKENS
    public_side = (
        _PUBLIC_FAMILY_TOKENS
        | _PUBLIC_AUDIENCE_TOKENS
        | _PUBLIC_LABELS
        | _PUBLIC_FLAG_KEY_STEMS
    )
    return any(
        token in private_side or token in public_side
        for token in _value_tokens(value)
    )


def _content_value_has_private_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        if _mapping_is_private(value, row_mode=False):
            return True
        marker = _marker(value, outer_kind=_KIND_PUBLIC)
        if marker.privacy_structure and marker.recognized_failing:
            return True
        return any(
            _key(key) in _COMMENT_CONTENT_KEY_TOKENS
            and _content_value_has_private_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_content_value_has_private_marker(item) for item in value)
    return False


def _contains_nonblank_comment_text(value: Any) -> bool:
    if isinstance(value, str):
        return _rendered_comment_text_has_content(value)
    if isinstance(value, Mapping):
        return any(
            _key(key) in _COMMENT_CONTENT_KEY_TOKENS
            and _contains_nonblank_comment_text(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_nonblank_comment_text(item) for item in value)
    return False


def _mapping_has_comment_content(value: Mapping[str, Any]) -> bool:
    if _content_value_has_private_marker(value):
        return False
    return any(
        _key(key) in _COMMENT_CONTENT_KEY_TOKENS
        and _contains_nonblank_comment_text(item)
        for key, item in value.items()
    )


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
